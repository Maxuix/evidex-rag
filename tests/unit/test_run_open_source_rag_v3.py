from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
from uuid import UUID

from tools import run_open_source_rag_v3 as runner


def _runtime_identity() -> dict[str, str]:
    return {
        "workspace_id": str(UUID(int=1)),
        "knowledge_base_id": str(UUID(int=2)),
        "index_revision_id": str(UUID(int=3)),
        "graph_build_id": str(UUID(int=4)),
        "embedding_profile_revision_id": str(UUID(int=5)),
        "graph_chat_profile_revision_id": str(UUID(int=6)),
        "schema_profile_key": "software_knowledge_v1",
        "schema_profile_digest": "6cae93809f060d21f0c85ba04cde955abdc5259fd93e1b1757fb7445d51eaf38",
        "extractor_version": "graphiti_v4",
        "index_configuration_sha256": "a" * 64,
        "serving_document_set_sha256": "b" * 64,
    }


def _extraction() -> dict:
    return {
        "extracted_relation_count": 1,
        "exact_matched_extracted_relation_count": 1,
        "exact_gold_relation_ids": ["OSR001"],
        "topology_gold_relation_ids": ["OSR001"],
        "self_loop_count": 0,
    }


class RunOpenSourceRagV3Tests(unittest.TestCase):
    def test_preflight_does_not_require_external_provider_confirmation(self) -> None:
        with (
            patch("sys.argv", ["run_open_source_rag_v3.py", "--preflight-only"]),
            patch.object(
                runner,
                "_run",
                new=AsyncMock(return_value={"status": "preflight_ok"}),
            ) as run,
            patch("builtins.print"),
        ):
            self.assertEqual(runner.main(), 0)

        run.assert_awaited_once()

    def test_relation_locator_maps_public_evidence_without_relation_id_leakage(self) -> None:
        manifest = {
            "documents": [
                {"document_id": "doc-1", "filename": "one.md"},
                {"document_id": "doc-2", "filename": "two.md"},
            ]
        }
        relations = [
            {
                "relation_id": "R1",
                "document_id": "doc-1",
                "evidence_text": "PSF 运营 Python Package Index。",
                "source_locator": {"section_title": "包索引"},
            },
            {
                "relation_id": "R2",
                "document_id": "doc-2",
                "evidence_text": "项目使用 Apache-2.0。",
                "source_locator": {"section_title": "许可证"},
            },
        ]
        rows = [
            {
                "index_chunk_id": str(UUID(int=11)),
                "original_filename": "one.md",
                "content": "## 包索引\nPSF 运营 Python Package Index。",
                "source_location": {},
                "hierarchy": {"section": "包索引"},
                "source_metadata": {},
            },
            {
                "index_chunk_id": str(UUID(int=12)),
                "original_filename": "two.md",
                "content": "项目使用 Apache-2.0。",
                "source_location": {},
                "hierarchy": {"section": "许可证"},
                "source_metadata": {},
            },
        ]

        result = runner.relation_chunk_map(
            manifest=manifest, relations=relations, serving_rows=rows
        )

        self.assertEqual(result["R1"], str(UUID(int=11)))
        self.assertEqual(result["R2"], str(UUID(int=12)))

    def test_relation_locator_ignores_markdown_inline_code_markers(self) -> None:
        manifest = {"documents": [{"document_id": "doc", "filename": "one.md"}]}
        relation = {
            "relation_id": "R1",
            "document_id": "doc",
            "evidence_text": "公开仓库是 `numpy/numpy`。",
            "source_locator": {"section_title": "仓库身份"},
        }
        row = {
            "index_chunk_id": str(UUID(int=11)),
            "original_filename": "one.md",
            "content": "公开仓库是 numpy/numpy。",
            "source_location": {},
            "hierarchy": {"section": "仓库身份"},
            "source_metadata": {},
        }

        result = runner.relation_chunk_map(
            manifest=manifest, relations=[relation], serving_rows=[row]
        )

        self.assertEqual(result["R1"], str(UUID(int=11)))

    def test_relation_locator_rejects_ambiguous_or_missing_chunks(self) -> None:
        manifest = {"documents": [{"document_id": "doc", "filename": "one.md"}]}
        relation = {
            "relation_id": "R1",
            "document_id": "doc",
            "evidence_text": "same fact",
            "source_locator": {"section_title": "section"},
        }
        row = {
            "index_chunk_id": str(UUID(int=11)),
            "original_filename": "one.md",
            "content": "same fact",
            "source_location": {},
            "hierarchy": {},
            "source_metadata": {},
        }
        duplicate = {**row, "index_chunk_id": str(UUID(int=12))}

        with self.assertRaisesRegex(runner.V3RunnerError, "not_unique"):
            runner.relation_chunk_map(
                manifest=manifest,
                relations=[relation],
                serving_rows=[row, duplicate],
            )
        with self.assertRaisesRegex(runner.V3RunnerError, "not_unique"):
            runner.relation_chunk_map(
                manifest=manifest, relations=[relation], serving_rows=[]
            )

    def test_checkpoint_is_owner_only_resumable_and_identity_bound(self) -> None:
        expected = runner._checkpoint_payload(
            runtime_identity=_runtime_identity(), extraction=_extraction()
        )
        completed = copy.deepcopy(expected["case_observations"][0])
        completed.update(
            {
                "simple_relation_ids": [],
                "graph_full_relation_ids_by_layer": {
                    layer: [] for layer in runner.LAYERS
                },
                "graph_incremental_packed_relation_ids": [],
                "auto": {
                    "attempted": False,
                    "admitted": False,
                    "new_source_backed_evidence_count": 0,
                },
                "actual_outcome": "refused",
                "forbidden_claim_hit": False,
            }
        )
        with tempfile.TemporaryDirectory(prefix="rag-v3-runner-") as directory:
            path = Path(directory) / "checkpoint.json"
            first = runner._load_checkpoint(path, expected)
            first["case_observations"][0] = completed
            runner._write_checkpoint(path, first)
            resumed = runner._load_checkpoint(path, expected)
            self.assertEqual(resumed["case_observations"][0], completed)
            self.assertEqual(path.stat().st_mode & 0o077, 0)

            changed = copy.deepcopy(expected)
            changed["runtime_identity"]["index_configuration_sha256"] = "c" * 64
            with self.assertRaisesRegex(runner.V3RunnerError, "identity_changed"):
                runner._load_checkpoint(path, changed)

    def test_checkpoint_rejects_completion_after_an_incomplete_case(self) -> None:
        expected = runner._checkpoint_payload(
            runtime_identity=_runtime_identity(), extraction=_extraction()
        )
        value = copy.deepcopy(expected)
        value["case_observations"][1]["simple_relation_ids"] = []
        with tempfile.TemporaryDirectory(prefix="rag-v3-order-") as directory:
            path = Path(directory) / "checkpoint.json"
            runner._write_checkpoint(path, value)
            with self.assertRaisesRegex(runner.V3RunnerError, "completion_order"):
                runner._load_checkpoint(path, expected)

    def test_checkpoint_rejects_content_fields_outside_closed_schema(self) -> None:
        expected = runner._checkpoint_payload(
            runtime_identity=_runtime_identity(), extraction=_extraction()
        )
        value = copy.deepcopy(expected)
        value["case_observations"][0]["content"] = "must not persist"
        with tempfile.TemporaryDirectory(prefix="rag-v3-safe-") as directory:
            path = Path(directory) / "checkpoint.json"
            runner._write_checkpoint(path, value)
            with self.assertRaisesRegex(runner.V3RunnerError, "observation_invalid"):
                runner._load_checkpoint(path, expected)

    def test_graph_extraction_observation_preserves_alignment_dimensions(self) -> None:
        result = runner._graph_extraction_observation(
            {
                "observed_edge_count": 3,
                "self_loop_edge_count": 0,
                "matched_observed_edge_precision": {"numerator": 2},
                "all_relations": {
                    "complete_relation_ids": ["OSR001", "OSR002"],
                    "directed_relation_ids": ["OSR001", "OSR002", "OSR003"],
                },
            }
        )

        self.assertEqual(result["extracted_relation_count"], 3)
        self.assertEqual(result["exact_matched_extracted_relation_count"], 2)
        self.assertEqual(result["topology_gold_relation_ids"], ["OSR001", "OSR002", "OSR003"])

    def test_forbidden_claim_check_does_not_persist_answer_content(self) -> None:
        self.assertTrue(
            runner._forbidden_claim_hit(
                "结论：Kubernetes 使用 BSD-3-Clause。",
                ["Kubernetes 使用 BSD-3-Clause"],
            )
        )
        self.assertFalse(
            runner._forbidden_claim_hit(
                "资料标注为 Apache-2.0。",
                ["Kubernetes 使用 BSD-3-Clause"],
            )
        )


if __name__ == "__main__":
    unittest.main()

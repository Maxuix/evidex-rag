from __future__ import annotations

import unittest
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID

from rag_kb.domain import GRAPH_EXTRACTOR_VERSION, GRAPH_RETRIEVAL_PROFILE_VERSION
from tools.evaluation_runtime import AdaptiveGraphIdentity
from tools.provision_routing_rag_eval import (
    EXPECTED_DOCUMENT_COUNT,
    MUSIQUE_MINI_SPEC,
    ProvisioningError,
    V3_SPEC,
    _bind_runtime,
    _corpus_digest,
    _corpus_paths,
    _paged_items,
    _host_source_identity,
    _require_current_runtime,
    _wait_for_indexing,
    _wait_for_graph,
)


class RoutingRagProvisioningTests(unittest.TestCase):
    def test_host_source_identity_matches_upload_scope_and_content(self) -> None:
        identity = AdaptiveGraphIdentity(
            **{
                name: UUID(f"01900000-0000-7000-8000-{index:012d}")
                for index, name in enumerate(
                    (
                        "workspace_id",
                        "knowledge_base_id",
                        "index_revision_id",
                        "graph_build_id",
                        "answer_profile_revision_id",
                        "judge_profile_revision_id",
                    ),
                    start=1,
                )
            }
        )
        runtime = SimpleNamespace(
            owner="0123456789abcdef0123456789abcdef",
            adaptive_graph=identity,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.txt"
            path.write_text("frozen source", encoding="utf-8")
            first = _host_source_identity(
                runtime,  # type: ignore[arg-type]
                "01900000-0000-7000-8000-000000000007",
                path,
                V3_SPEC,
            )
            second = _host_source_identity(
                runtime,  # type: ignore[arg-type]
                "01900000-0000-7000-8000-000000000007",
                path,
                V3_SPEC,
            )

        self.assertEqual(first, second)
        self.assertEqual(first.workspace_id, identity.workspace_id)
        self.assertEqual(len(first.key), 64)

    def test_runtime_binding_rewrites_the_validated_canonical_manifest(self) -> None:
        manifest = Path("/primary/.runtime/evaluations/rag-eval/runtime.json")
        identity = AdaptiveGraphIdentity(
            **{
                name: UUID(f"01900000-0000-7000-8000-{index:012d}")
                for index, name in enumerate(
                    (
                        "workspace_id",
                        "knowledge_base_id",
                        "index_revision_id",
                        "graph_build_id",
                        "answer_profile_revision_id",
                        "judge_profile_revision_id",
                    ),
                    start=1,
                )
            }
        )
        runtime = SimpleNamespace(
            manifest=manifest,
            owner="0123456789abcdef0123456789abcdef",
            build_revision="a" * 40,
            adaptive_graph=identity,
        )

        with patch("tools.provision_routing_rag_eval._write_private") as write:
            _bind_runtime(
                runtime,  # type: ignore[arg-type]
                knowledge_base_id=UUID("01900000-0000-7000-8000-000000000007"),
                index_revision_id=UUID("01900000-0000-7000-8000-000000000008"),
                graph_build_id=UUID("01900000-0000-7000-8000-000000000009"),
            )

        self.assertEqual(write.call_args.args[0], manifest)

    def test_frozen_corpus_is_exact_and_digest_is_stable(self) -> None:
        paths = _corpus_paths()

        self.assertEqual(len(paths), EXPECTED_DOCUMENT_COUNT)
        self.assertEqual(_corpus_digest(paths), _corpus_digest(paths))
        self.assertEqual({path.suffix for path in paths}, {".md", ".txt"})

    def test_v3_frozen_corpus_is_exact_and_supports_csv(self) -> None:
        paths = _corpus_paths(V3_SPEC)

        self.assertEqual(len(paths), 14)
        self.assertEqual(
            {path.suffix for path in paths}, {".md", ".txt", ".csv"}
        )
        self.assertEqual(_corpus_digest(paths), _corpus_digest(paths))

    def test_musique_mini_corpus_is_exact_and_markdown_only(self) -> None:
        paths = _corpus_paths(MUSIQUE_MINI_SPEC)

        self.assertEqual(len(paths), 80)
        self.assertEqual({path.suffix for path in paths}, {".md"})
        self.assertEqual(_corpus_digest(paths), _corpus_digest(paths))

    def test_dataset_specs_freeze_their_graph_schema_profiles(self) -> None:
        self.assertEqual(V3_SPEC.schema_profile_key, "software_knowledge_v1")
        self.assertEqual(
            MUSIQUE_MINI_SPEC.schema_profile_key,
            "generic_open_domain_v1",
        )
        self.assertNotEqual(
            V3_SPEC.schema_profile_digest,
            MUSIQUE_MINI_SPEC.schema_profile_digest,
        )

    def test_pagination_rejects_a_non_string_cursor(self) -> None:
        with patch(
            "tools.provision_routing_rag_eval._request",
            return_value={"items": [], "next_cursor": 3},
        ):
            with self.assertRaises(ProvisioningError):
                _paged_items("http://127.0.0.1:28000/api/v1", "knowledge-bases")

    def test_current_runtime_uses_public_modes_contract(self) -> None:
        runtime = SimpleNamespace(api_base_url="http://127.0.0.1:28000/api/v1")
        response = {
            "default_mode": "vector",
            "modes": [
                {
                    "mode": "graph",
                    "profile_version": GRAPH_RETRIEVAL_PROFILE_VERSION,
                    "enabled": True,
                }
            ],
        }

        with patch(
            "tools.provision_routing_rag_eval._request",
            return_value=response,
        ):
            _require_current_runtime(runtime)  # type: ignore[arg-type]

    def test_current_runtime_rejects_wrong_capability_shape(self) -> None:
        runtime = SimpleNamespace(api_base_url="http://127.0.0.1:28000/api/v1")
        response = {
            "capabilities": [
                {
                    "mode": "graph",
                    "profile_version": GRAPH_RETRIEVAL_PROFILE_VERSION,
                    "enabled": True,
                }
            ]
        }

        with patch(
            "tools.provision_routing_rag_eval._request",
            return_value=response,
        ):
            with self.assertRaises(ProvisioningError):
                _require_current_runtime(runtime)  # type: ignore[arg-type]

    def test_ready_graph_requires_a_positive_eligible_chunk_count(self) -> None:
        config = {
            "status": "ready",
            "extractor_version": GRAPH_EXTRACTOR_VERSION,
            "processed_chunk_count": 0,
            "eligible_chunk_count": 0,
            "build_id": "01900000-0000-7000-8000-000000000001",
        }
        runtime = SimpleNamespace(api_base_url="http://127.0.0.1:28000/api/v1")

        with patch(
            "tools.provision_routing_rag_eval._request",
            return_value=config,
        ):
            with self.assertRaises(ProvisioningError):
                _wait_for_graph(
                    runtime,  # type: ignore[arg-type]
                    "01900000-0000-7000-8000-000000000002",
                    answer_profile_revision_id=UUID(
                        "01900000-0000-7000-8000-000000000003"
                    ),
                    timeout_seconds=60.0,
                    retry_failed=False,
                    force_rebuild_failed=False,
                )

    def test_host_indexing_retries_failed_job_before_accepting_completion(self) -> None:
        job_id = "01900000-0000-7000-8000-000000000011"
        revision_id = "01900000-0000-7000-8000-000000000012"
        failed = {
            "job_id": job_id,
            "status": "failed",
            "can_retry": True,
            "updated_at": "2026-08-23T00:00:00Z",
        }
        completed = {
            "job_id": job_id,
            "status": "completed",
            "index_revision_id": revision_id,
        }
        runtime = SimpleNamespace(api_base_url="http://127.0.0.1:28000/api/v1")

        with (
            patch(
                "tools.provision_routing_rag_eval._paged_items",
                side_effect=((failed,), (completed,)),
            ),
            patch("tools.provision_routing_rag_eval._request") as request,
            patch("tools.provision_routing_rag_eval.time.sleep"),
        ):
            result = _wait_for_indexing(
                runtime,  # type: ignore[arg-type]
                "01900000-0000-7000-8000-000000000013",
                timeout_seconds=60.0,
                expected_document_count=1,
                host_retry_spec=V3_SPEC,
            )

        self.assertEqual(result, revision_id)
        self.assertIn(f"/indexing-jobs/{job_id}/retry", request.call_args.args[0])

    def test_host_retry_accepts_only_a_proven_concurrent_state_change(self) -> None:
        job_id = "01900000-0000-7000-8000-000000000011"
        revision_id = "01900000-0000-7000-8000-000000000012"
        failed = {
            "job_id": job_id,
            "status": "failed",
            "can_retry": True,
            "updated_at": "2026-08-23T00:00:00Z",
        }
        completed = {
            "job_id": job_id,
            "status": "completed",
            "index_revision_id": revision_id,
        }
        runtime = SimpleNamespace(api_base_url="http://127.0.0.1:28000/api/v1")

        with (
            patch(
                "tools.provision_routing_rag_eval._paged_items",
                side_effect=((failed,), (completed,)),
            ),
            patch(
                "tools.provision_routing_rag_eval._request",
                side_effect=(
                    ProvisioningError("retry raced"),
                    {"status": "queued"},
                ),
            ),
            patch("tools.provision_routing_rag_eval.time.sleep"),
        ):
            result = _wait_for_indexing(
                runtime,  # type: ignore[arg-type]
                "01900000-0000-7000-8000-000000000013",
                timeout_seconds=60.0,
                expected_document_count=1,
                host_retry_spec=V3_SPEC,
            )

        self.assertEqual(result, revision_id)

    def test_graph_failure_after_enable_gets_one_authorized_resume(self) -> None:
        disabled = {"status": "disabled"}
        processing = {"status": "processing"}
        failed = {
            "status": "failed",
            "extractor_version": V3_SPEC.extractor_version,
            "schema_profile_key": V3_SPEC.schema_profile_key,
            "schema_profile_digest": V3_SPEC.schema_profile_digest,
        }
        ready = {
            "status": "ready",
            "extractor_version": GRAPH_EXTRACTOR_VERSION,
            "schema_profile_key": V3_SPEC.schema_profile_key,
            "schema_profile_digest": V3_SPEC.schema_profile_digest,
            "active_build_schema_profile_key": V3_SPEC.schema_profile_key,
            "active_build_schema_profile_digest": V3_SPEC.schema_profile_digest,
            "processed_chunk_count": 2,
            "eligible_chunk_count": 2,
            "build_id": "01900000-0000-7000-8000-000000000001",
        }
        runtime = SimpleNamespace(api_base_url="http://127.0.0.1:28000/api/v1")

        with (
            patch(
                "tools.provision_routing_rag_eval._request",
                side_effect=(disabled, processing, failed, ready),
            ) as request,
            patch("tools.provision_routing_rag_eval.time.sleep"),
        ):
            result = _wait_for_graph(
                runtime,  # type: ignore[arg-type]
                "01900000-0000-7000-8000-000000000002",
                answer_profile_revision_id=UUID(
                    "01900000-0000-7000-8000-000000000003"
                ),
                timeout_seconds=60.0,
                retry_failed=True,
                force_rebuild_failed=False,
            )

        self.assertEqual(result, ready)
        self.assertEqual(
            request.call_args_list[-1].kwargs["payload"],
            {"enabled": True, "retry": True},
        )

    def test_host_graph_retry_uses_current_source_mutation_not_api_put(self) -> None:
        failed = {
            "status": "failed",
            "extractor_version": V3_SPEC.extractor_version,
            "schema_profile_key": V3_SPEC.schema_profile_key,
            "schema_profile_digest": V3_SPEC.schema_profile_digest,
        }
        ready = {
            "status": "ready",
            "extractor_version": GRAPH_EXTRACTOR_VERSION,
            "schema_profile_key": V3_SPEC.schema_profile_key,
            "schema_profile_digest": V3_SPEC.schema_profile_digest,
            "active_build_schema_profile_key": V3_SPEC.schema_profile_key,
            "active_build_schema_profile_digest": V3_SPEC.schema_profile_digest,
            "processed_chunk_count": 2,
            "eligible_chunk_count": 2,
            "build_id": "01900000-0000-7000-8000-000000000001",
        }
        runtime = SimpleNamespace(api_base_url="http://127.0.0.1:28000/api/v1")

        with (
            patch(
                "tools.provision_routing_rag_eval._request",
                return_value=failed,
            ) as request,
            patch(
                "tools.provision_routing_rag_eval._host_graph_mutation",
                return_value=ready,
            ) as mutation,
        ):
            result = _wait_for_graph(
                runtime,  # type: ignore[arg-type]
                "01900000-0000-7000-8000-000000000002",
                answer_profile_revision_id=UUID(
                    "01900000-0000-7000-8000-000000000003"
                ),
                timeout_seconds=60.0,
                retry_failed=True,
                force_rebuild_failed=False,
                host_runtime=runtime,  # type: ignore[arg-type]
            )

        self.assertEqual(result, ready)
        mutation.assert_called_once()
        self.assertEqual(request.call_count, 1)

    def test_failed_graph_retry_resumes_without_force_rebuild(self) -> None:
        failed = {
            "status": "failed",
            "extractor_version": V3_SPEC.extractor_version,
            "schema_profile_key": V3_SPEC.schema_profile_key,
            "schema_profile_digest": V3_SPEC.schema_profile_digest,
        }
        ready = {
            "status": "ready",
            "extractor_version": GRAPH_EXTRACTOR_VERSION,
            "schema_profile_key": V3_SPEC.schema_profile_key,
            "schema_profile_digest": V3_SPEC.schema_profile_digest,
            "active_build_schema_profile_key": V3_SPEC.schema_profile_key,
            "active_build_schema_profile_digest": V3_SPEC.schema_profile_digest,
            "processed_chunk_count": 2,
            "eligible_chunk_count": 2,
            "build_id": "01900000-0000-7000-8000-000000000001",
        }
        runtime = SimpleNamespace(api_base_url="http://127.0.0.1:28000/api/v1")

        with patch(
            "tools.provision_routing_rag_eval._request",
            side_effect=(failed, ready),
        ) as request:
            result = _wait_for_graph(
                runtime,  # type: ignore[arg-type]
                "01900000-0000-7000-8000-000000000002",
                answer_profile_revision_id=UUID(
                    "01900000-0000-7000-8000-000000000003"
                ),
                timeout_seconds=60.0,
                retry_failed=True,
                force_rebuild_failed=False,
            )

        self.assertEqual(result, ready)
        self.assertEqual(
            request.call_args_list[1].kwargs["payload"],
            {"enabled": True, "retry": True},
        )

    def test_legacy_failed_graph_force_rebuild_is_explicit(self) -> None:
        failed = {
            "status": "failed",
            "extractor_version": V3_SPEC.extractor_version,
            "schema_profile_key": V3_SPEC.schema_profile_key,
            "schema_profile_digest": V3_SPEC.schema_profile_digest,
        }
        ready = {
            "status": "ready",
            "extractor_version": GRAPH_EXTRACTOR_VERSION,
            "schema_profile_key": V3_SPEC.schema_profile_key,
            "schema_profile_digest": V3_SPEC.schema_profile_digest,
            "active_build_schema_profile_key": V3_SPEC.schema_profile_key,
            "active_build_schema_profile_digest": V3_SPEC.schema_profile_digest,
            "processed_chunk_count": 2,
            "eligible_chunk_count": 2,
            "build_id": "01900000-0000-7000-8000-000000000001",
        }
        runtime = SimpleNamespace(api_base_url="http://127.0.0.1:28000/api/v1")

        with patch(
            "tools.provision_routing_rag_eval._request",
            side_effect=(failed, ready),
        ) as request:
            _wait_for_graph(
                runtime,  # type: ignore[arg-type]
                "01900000-0000-7000-8000-000000000002",
                answer_profile_revision_id=UUID(
                    "01900000-0000-7000-8000-000000000003"
                ),
                timeout_seconds=60.0,
                retry_failed=False,
                force_rebuild_failed=True,
            )

        self.assertEqual(
            request.call_args_list[1].kwargs["payload"],
            {"enabled": True, "retry": True, "force_rebuild": True},
        )


if __name__ == "__main__":
    unittest.main()

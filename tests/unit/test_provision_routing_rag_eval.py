from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID

from rag_kb.domain import GRAPH_EXTRACTOR_VERSION, GRAPH_RETRIEVAL_PROFILE_VERSION
from tools.provision_routing_rag_eval import (
    EXPECTED_DOCUMENT_COUNT,
    ProvisioningError,
    V3_SPEC,
    _corpus_digest,
    _corpus_paths,
    _paged_items,
    _require_current_runtime,
    _wait_for_graph,
)


class RoutingRagProvisioningTests(unittest.TestCase):
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

    def test_failed_graph_retry_resumes_without_force_rebuild(self) -> None:
        failed = {"status": "failed"}
        ready = {
            "status": "ready",
            "extractor_version": GRAPH_EXTRACTOR_VERSION,
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
        failed = {"status": "failed"}
        ready = {
            "status": "ready",
            "extractor_version": GRAPH_EXTRACTOR_VERSION,
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

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from uuid import UUID

from rag_kb.domain import GRAPH_EXTRACTOR_VERSION
from tools.evaluation_resilience import RESILIENCE_POLICY_SHA256, RetryPolicy
from tools.provision_large_evaluation_host import (
    ProvisioningSpec,
    _wait_for_graph,
)


class LargeEvaluationGraphResilienceTests(unittest.TestCase):
    def test_graph_resume_budget_survives_process_restart_and_resets_on_progress(
        self,
    ) -> None:
        spec = ProvisioningSpec(
            dataset_id="frozen-v1",
            corpus_root=Path("/private/tmp/frozen-v1"),
            expected_document_count=1,
            knowledge_base_name="frozen-v1",
            confirmation="CONFIRM",
            graph_schema_key="generic-v1",
            graph_schema_digest="a" * 64,
        )
        failed = {
            "status": "failed",
            "eligible_chunk_count": 100,
            "processed_chunk_count": 20,
            "extracted_chunk_count": 20,
            "last_error_code": "GRAPH_PROVIDER_UNAVAILABLE",
        }
        ready = {
            "status": "ready",
            "eligible_chunk_count": 100,
            "processed_chunk_count": 100,
            "extracted_chunk_count": 100,
            "last_error_code": None,
            "schema_profile_key": spec.graph_schema_key,
            "schema_profile_digest": spec.graph_schema_digest,
            "extractor_version": GRAPH_EXTRACTOR_VERSION,
            "active_build_schema_profile_key": spec.graph_schema_key,
            "active_build_schema_profile_digest": spec.graph_schema_digest,
            "build_id": str(UUID(int=2)),
        }
        checkpoint = {
            "graph_resume_count": 20,
            "graph_consecutive_failure_count": 4,
            "graph": {"processed_chunk_count": 20},
            "events": [],
        }
        zero_delay = RetryPolicy(
            max_attempts=64,
            backoff_seconds=(0.0,),
            maximum_backoff_seconds=0.0,
        )
        with TemporaryDirectory() as temporary:
            checkpoint_path = Path(temporary) / "checkpoint.json"
            with (
                patch(
                    "tools.provision_large_evaluation_host._graph_config",
                    side_effect=[failed, ready],
                ),
                patch(
                    "tools.provision_large_evaluation_host.GRAPH_BUILD_RETRY_POLICY",
                    zero_delay,
                ),
            ):
                result = _wait_for_graph(
                    SimpleNamespace(api_base_url="http://127.0.0.1:1/api/v1"),
                    kb_id=str(UUID(int=1)),
                    spec=spec,
                    chat_profile_revision_id=UUID(int=3),
                    timeout_seconds=10.0,
                    checkpoint=checkpoint,
                    checkpoint_path=checkpoint_path,
                )

        self.assertEqual(result["status"], "ready")
        self.assertEqual(checkpoint["graph_resume_count"], 21)
        self.assertEqual(checkpoint["graph_consecutive_failure_count"], 0)
        self.assertEqual(
            checkpoint["resilience_policy_sha256"], RESILIENCE_POLICY_SHA256
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from rag_kb.domain import (
    ENTERPRISE_GRAPH_SCHEMA_PROFILE_DIGEST,
    ENTERPRISE_GRAPH_SCHEMA_PROFILE_KEY,
)
from tools import run_enterprise_extraction_pilot as module


class EnterpriseExtractionPilotTests(unittest.TestCase):
    def test_arm_produces_an_isolated_enterprise_provisioning_spec(self) -> None:
        dataset_key, spec = module._spec("baseline")

        self.assertEqual(dataset_key, "enterprise_pilot_baseline")
        self.assertEqual(
            spec.dataset_id,
            "enterprise-profile-quality-pilot-baseline-v1",
        )
        self.assertEqual(spec.expected_document_count, 138)
        self.assertEqual(spec.graph_schema_key, ENTERPRISE_GRAPH_SCHEMA_PROFILE_KEY)
        self.assertEqual(
            spec.graph_schema_digest,
            ENTERPRISE_GRAPH_SCHEMA_PROFILE_DIGEST,
        )

    def test_arm_rejects_unsafe_dataset_identifiers(self) -> None:
        for arm in ("", "UPPER", "../outside", "a" * 33):
            with self.subTest(arm=arm):
                with self.assertRaisesRegex(
                    ValueError,
                    "enterprise_pilot_arm_invalid",
                ):
                    module._spec(arm)

    def test_stress_corpus_produces_a_small_isolated_spec(self) -> None:
        dataset_key, spec = module._spec("baseline", "stress")

        self.assertEqual(dataset_key, "enterprise_stress_pilot_baseline")
        self.assertEqual(
            spec.dataset_id,
            "enterprise-profile-stress-pilot-baseline-v1",
        )
        self.assertEqual(spec.expected_document_count, 12)
        self.assertEqual(
            spec.corpus_root,
            module.STRESS_CORPUS_ROOT / "documents",
        )

    def test_unknown_corpus_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "enterprise_pilot_corpus_invalid"):
            module._spec("baseline", "unknown")


if __name__ == "__main__":
    unittest.main()

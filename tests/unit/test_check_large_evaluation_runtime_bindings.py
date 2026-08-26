from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tools.check_large_evaluation_runtime import (
    LargeEvaluationRuntimeError,
    _assert_all_suite_bindings,
)
from tools.evaluation_campaign_state import digest, write_private_json
from tools.provision_large_evaluation_host import BINDINGS_SCHEMA, SPECS


class LargeEvaluationBindingsTests(unittest.TestCase):
    def _plan(self) -> dict[str, object]:
        return {
            "plan_binding": {
                "corpora": [
                    {"dataset_id": spec.dataset_id} for spec in SPECS.values()
                ]
            }
        }

    def _bindings(self) -> dict[str, object]:
        suites: dict[str, object] = {}
        for index, spec in enumerate(SPECS.values(), start=1):
            suites[spec.dataset_id] = {
                "corpus_sha256": "a" * 64,
                "knowledge_base_id": f"kb-{index}",
                "index_revision_id": f"index-{index}",
                "graph_build_id": f"build-{index}" if spec.graph_enabled else None,
            }
        return {
            "schema_version": BINDINGS_SCHEMA,
            "suites": suites,
            "binding_sha256": digest({"suites": suites}),
        }

    def test_requires_all_planned_suite_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_private_json(root / "large-evaluation-bindings.json", self._bindings())
            _assert_all_suite_bindings(root, self._plan())

    def test_rejects_missing_suite_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bindings = self._bindings()
            suites = bindings["suites"]
            assert isinstance(suites, dict)
            suites.pop("graph-rag-v1")
            bindings["binding_sha256"] = digest({"suites": suites})
            write_private_json(root / "large-evaluation-bindings.json", bindings)
            with self.assertRaisesRegex(
                LargeEvaluationRuntimeError, "large_evaluation_bindings_invalid"
            ):
                _assert_all_suite_bindings(root, self._plan())

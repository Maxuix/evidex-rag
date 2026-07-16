from __future__ import annotations

import copy
import json
import unittest

from tools.check_frontend_api_contract import (
    DEFAULT_CLIENT,
    DEFAULT_OPENAPI,
    check_frontend_contract,
    main,
)


class FrontendApiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.openapi = json.loads(DEFAULT_OPENAPI.read_text(encoding="utf-8"))
        cls.client = DEFAULT_CLIENT.read_text(encoding="utf-8")

    def test_reviewed_snapshot_and_frontend_client_pass(self) -> None:
        self.assertEqual(check_frontend_contract(self.openapi, self.client), [])
        self.assertEqual(main(DEFAULT_OPENAPI, DEFAULT_CLIENT), 0)

    def test_operation_status_upload_and_response_drift_are_detected(self) -> None:
        cases: list[tuple[str, dict[str, object], str]] = []

        missing_operation = copy.deepcopy(self.openapi)
        del missing_operation["paths"]["/api/v1/chat/runs"]["post"]
        cases.append(("operation", missing_operation, "missing frontend operation"))

        missing_status = copy.deepcopy(self.openapi)
        del missing_status["paths"]["/api/v1/retrieval/query"]["post"][
            "responses"
        ]["503"]
        cases.append(("status", missing_status, "response statuses drifted"))

        upload_header = copy.deepcopy(self.openapi)
        parameters = upload_header["paths"][
            "/api/v1/knowledge-bases/{kb_id}/documents"
        ]["post"]["parameters"]
        next(
            item
            for item in parameters
            if item.get("name") == "X-Document-Filename"
        )["required"] = False
        cases.append(("upload", upload_header, "upload headers drifted"))

        response_field = copy.deepcopy(self.openapi)
        run_schema = response_field["components"]["schemas"]["ChatRunResponse"]
        del run_schema["properties"]["events_url"]
        run_schema["required"].remove("events_url")
        cases.append(("field", response_field, "response fields drifted"))

        narrowed_enum = copy.deepcopy(self.openapi)
        narrowed_enum["components"]["schemas"]["IndexingJobResponse"][
            "properties"
        ]["status"]["enum"].remove("cancelled")
        cases.append(("enum", narrowed_enum, "enum drifted"))

        for name, document, expected in cases:
            with self.subTest(name=name):
                errors = check_frontend_contract(document, self.client)
                self.assertTrue(any(expected in error for error in errors), errors)

    def test_retrieval_debug_and_client_route_drift_are_detected(self) -> None:
        weakened_debug = self.client.replace("include_debug: true", "include_debug: false")
        errors = check_frontend_contract(self.openapi, weakened_debug)
        self.assertTrue(
            any("retrieval debug body drifted" in error for error in errors), errors
        )

        private_route = self.client.replace(
            'this.request("/retrieval/query"',
            'this.request("/retrieval/private"',
        )
        errors = check_frontend_contract(self.openapi, private_route)
        self.assertTrue(any("client route drifted" in error for error in errors), errors)
        self.assertTrue(
            any("public route literals drifted" in error for error in errors), errors
        )

        identity_override = self.client.replace(
            "Accept: \"application/json\"",
            "Accept: \"application/json\", \"X-Workspace-Id\": \"other\"",
        )
        errors = check_frontend_contract(self.openapi, identity_override)
        self.assertTrue(
            any("forbidden identity header" in error for error in errors), errors
        )

    def test_cli_is_read_only(self) -> None:
        before = {
            path: (path.stat().st_mtime_ns, path.read_bytes())
            for path in (DEFAULT_OPENAPI, DEFAULT_CLIENT)
        }
        self.assertEqual(main(DEFAULT_OPENAPI, DEFAULT_CLIENT), 0)
        after = {
            path: (path.stat().st_mtime_ns, path.read_bytes())
            for path in (DEFAULT_OPENAPI, DEFAULT_CLIENT)
        }
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import copy
import unittest

from tools.check_openapi_compatibility import find_breaking_changes


BASELINE = {
    "paths": {
        "/api/v1/items": {
            "get": {
                "responses": {
                    "200": {"description": "ok"},
                    "400": {"description": "bad request"},
                }
            }
        }
    },
    "components": {
        "schemas": {
            "Item": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "status": {"type": "string", "enum": ["ready", "failed"]},
                },
                "required": ["id"],
            }
        }
    },
}


class OpenApiCompatibilityTests(unittest.TestCase):
    def test_additive_optional_changes_are_compatible(self) -> None:
        current = copy.deepcopy(BASELINE)
        current["paths"]["/api/v1/other"] = {"get": {"responses": {"200": {}}}}
        current["components"]["schemas"]["Item"]["properties"]["detail"] = {
            "type": "string"
        }
        self.assertEqual(find_breaking_changes(BASELINE, current), [])

    def test_removals_enum_narrowing_and_new_required_fields_break(self) -> None:
        current = copy.deepcopy(BASELINE)
        del current["paths"]["/api/v1/items"]["get"]["responses"]["400"]
        del current["components"]["schemas"]["Item"]["properties"]["id"]
        current["components"]["schemas"]["Item"]["properties"]["status"][
            "enum"
        ] = ["ready"]
        current["components"]["schemas"]["Item"]["required"].append("status")

        changes = find_breaking_changes(BASELINE, current)
        self.assertIn("removed response: GET /api/v1/items 400", changes)
        self.assertIn("removed schema property: Item.id", changes)
        self.assertIn("removed enum value: Item.status", changes)
        self.assertIn("added required property: Item.status", changes)


if __name__ == "__main__":
    unittest.main()

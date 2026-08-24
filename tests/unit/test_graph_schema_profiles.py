from __future__ import annotations

from dataclasses import replace
import unittest

from rag_kb.graph.schema_profiles import (
    GENERIC_GRAPH_SCHEMA_PROFILE_KEY,
    GraphSchemaProfileMismatch,
    SOFTWARE_GRAPH_SCHEMA_PROFILE_KEY,
    get_graph_schema_registry,
)
from rag_kb.domain import (
    GENERIC_GRAPH_SCHEMA_PROFILE_DIGEST,
    SOFTWARE_GRAPH_SCHEMA_PROFILE_DIGEST,
)


class GraphSchemaProfileTests(unittest.TestCase):
    def test_generic_is_the_only_default(self) -> None:
        registry = get_graph_schema_registry()
        profiles = registry.list()

        self.assertEqual(
            [profile.key for profile in profiles if profile.is_default],
            [GENERIC_GRAPH_SCHEMA_PROFILE_KEY],
        )
        self.assertEqual(
            registry.resolve(GENERIC_GRAPH_SCHEMA_PROFILE_KEY).digest,
            GENERIC_GRAPH_SCHEMA_PROFILE_DIGEST,
        )
        self.assertEqual(
            registry.resolve(SOFTWARE_GRAPH_SCHEMA_PROFILE_KEY).digest,
            SOFTWARE_GRAPH_SCHEMA_PROFILE_DIGEST,
        )
        self.assertEqual(
            {profile.key for profile in profiles},
            {GENERIC_GRAPH_SCHEMA_PROFILE_KEY, SOFTWARE_GRAPH_SCHEMA_PROFILE_KEY},
        )

    def test_digest_is_stable_and_covers_manifest_policy_and_instructions(self) -> None:
        registry = get_graph_schema_registry()
        software = registry.resolve(SOFTWARE_GRAPH_SCHEMA_PROFILE_KEY)
        reordered = replace(
            software,
            entity_manifest=tuple(reversed(software.entity_manifest)),
            edge_manifest=tuple(reversed(software.edge_manifest)),
            edge_type_map_manifest=tuple(reversed(software.edge_type_map_manifest)),
        )
        self.assertEqual(software.digest, reordered.digest)
        self.assertNotEqual(
            software.digest,
            replace(software, extraction_instructions="changed").digest,
        )
        self.assertNotEqual(
            software.digest,
            replace(
                software,
                validation_policy=replace(
                    software.validation_policy,
                    standalone_alias_orphan_check=False,
                ),
            ).digest,
        )
        self.assertNotEqual(
            software.digest,
            replace(
                software,
                edge_type_map_manifest=tuple(
                    item
                    for item in software.edge_type_map_manifest
                    if item[0] != "Project" or item[1] != "Project"
                ),
            ).digest,
        )

    def test_generic_compiles_to_native_graphiti_arguments(self) -> None:
        profile = get_graph_schema_registry().compile(GENERIC_GRAPH_SCHEMA_PROFILE_KEY)

        self.assertIsNone(profile.entity_types)
        self.assertIsNone(profile.edge_types)
        self.assertIsNone(profile.edge_type_map)
        self.assertNotIn("software", profile.extraction_instructions.lower())
        self.assertNotIn("alias_surface", profile.extraction_instructions.lower())
        self.assertFalse(profile.validation_policy.standalone_alias_orphan_check)

    def test_software_compiles_the_locked_typed_contract(self) -> None:
        profile = get_graph_schema_registry().compile(SOFTWARE_GRAPH_SCHEMA_PROFILE_KEY)

        assert profile.entity_types is not None
        assert profile.edge_types is not None
        assert profile.edge_type_map is not None
        self.assertEqual(
            set(profile.entity_types),
            {
                "Organization",
                "Project",
                "Repository",
                "Service",
                "License",
                "LicenseExpression",
                "AliasSurface",
            },
        )
        self.assertEqual(
            set(profile.edge_types),
            {
                "Stewards", "Hosts", "Maintains", "Operates", "HasRepository",
                "DistributedUnder", "Lists", "HasLicenseExpression",
                "DocumentationHostedAt", "UsesImportNamespace", "Requires", "BuiltOn",
                "FoundationFor", "OriginatedFrom", "DevelopedOn", "Sponsors",
                "GraduatedProjectOf", "PartOf", "HasShortName", "AliasOf",
            },
        )
        self.assertEqual(
            profile.entity_types["Organization"].model_fields["short_names"].default_factory,
            list,
        )
        self.assertTrue(profile.validation_policy.standalone_alias_orphan_check)

    def test_unknown_and_digest_mismatch_fail_closed(self) -> None:
        registry = get_graph_schema_registry()

        with self.assertRaisesRegex(GraphSchemaProfileMismatch, "unknown"):
            registry.resolve("missing_profile")
        with self.assertRaisesRegex(GraphSchemaProfileMismatch, "mismatch"):
            registry.resolve(GENERIC_GRAPH_SCHEMA_PROFILE_KEY, digest="0" * 64)
        with self.assertRaisesRegex(GraphSchemaProfileMismatch, "extractor"):
            registry.compile(GENERIC_GRAPH_SCHEMA_PROFILE_KEY, extractor_version="graphiti_v3")


if __name__ == "__main__":
    unittest.main()

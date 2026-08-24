"""Code-owned Graphiti schema profiles with deterministic compilation.

The profile manifests intentionally contain only serializable facts.  Graphiti
and Pydantic objects are produced by :meth:`GraphSchemaProfile.compile`, so a
build identity can be checked without depending on generated schema ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel, Field, create_model


GENERIC_GRAPH_SCHEMA_PROFILE_KEY = "generic_open_domain_v1"
SOFTWARE_GRAPH_SCHEMA_PROFILE_KEY = "software_knowledge_v1"
_PROFILE_KEY_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")


class GraphSchemaProfileError(ValueError):
    """A built-in profile manifest or identity is invalid."""


class GraphSchemaProfileMismatch(GraphSchemaProfileError):
    """The requested profile identity is not the installed identity."""


@dataclass(frozen=True, slots=True)
class GraphSchemaValidationPolicy:
    """Topology checks that are specific to one schema profile."""

    standalone_alias_orphan_check: bool = False
    reject_entity_self_loops: bool = True
    reject_empty_relation_names: bool = True

    def as_manifest(self) -> dict[str, bool]:
        return {
            "reject_empty_relation_names": self.reject_empty_relation_names,
            "reject_entity_self_loops": self.reject_entity_self_loops,
            "standalone_alias_orphan_check": self.standalone_alias_orphan_check,
        }


@dataclass(frozen=True, slots=True)
class GraphSchemaField:
    """A deterministic Pydantic attribute description."""

    name: str
    kind: Literal["optional_string", "string_list"]
    description: str = ""

    def as_manifest(self) -> dict[str, str]:
        return {
            "description": self.description,
            "kind": self.kind,
            "name": self.name,
        }


@dataclass(frozen=True, slots=True)
class GraphSchemaEntity:
    name: str
    description: str
    attributes: tuple[GraphSchemaField, ...] = ()

    def as_manifest(self) -> dict[str, Any]:
        return {
            "attributes": [
                field.as_manifest()
                for field in sorted(self.attributes, key=lambda item: item.name)
            ],
            "description": self.description,
            "name": self.name,
        }


@dataclass(frozen=True, slots=True)
class GraphSchemaEdge:
    name: str
    description: str
    attributes: tuple[GraphSchemaField, ...] = ()

    def as_manifest(self) -> dict[str, Any]:
        return {
            "attributes": [
                field.as_manifest()
                for field in sorted(self.attributes, key=lambda item: item.name)
            ],
            "description": self.description,
            "name": self.name,
        }


@dataclass(frozen=True, slots=True)
class CompiledGraphSchema:
    """Arguments passed to ``Graphiti.add_episode`` for one profile."""

    entity_types: dict[str, type[BaseModel]] | None
    edge_types: dict[str, type[BaseModel]] | None
    edge_type_map: dict[tuple[str, str], list[str]] | None
    extraction_instructions: str
    validation_policy: GraphSchemaValidationPolicy


@dataclass(frozen=True, slots=True)
class GraphSchemaProfile:
    key: str
    display_name: str
    description: str
    entity_manifest: tuple[GraphSchemaEntity, ...]
    edge_manifest: tuple[GraphSchemaEdge, ...]
    edge_type_map_manifest: tuple[tuple[str, str, tuple[str, ...]], ...]
    extraction_instructions: str
    alias_policy: str
    validation_policy: GraphSchemaValidationPolicy
    is_default: bool = False
    compatible_extractor_versions: tuple[str, ...] = ("graphiti_v4",)

    def __post_init__(self) -> None:
        if not _PROFILE_KEY_RE.fullmatch(self.key):
            raise GraphSchemaProfileError("graph_schema_profile_key_invalid")
        if not self.display_name.strip() or not self.description.strip():
            raise GraphSchemaProfileError("graph_schema_profile_metadata_invalid")
        if not self.extraction_instructions.strip():
            raise GraphSchemaProfileError("graph_schema_profile_instructions_empty")
        if not self.alias_policy.strip():
            raise GraphSchemaProfileError("graph_schema_profile_alias_policy_empty")
        _validate_manifest(self)

    @property
    def digest(self) -> str:
        payload = _canonical_manifest(self)
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
                "utf-8"
            )
        ).hexdigest()

    def manifest(self) -> dict[str, Any]:
        """Return the canonical, JSON-safe manifest used by ``digest``."""

        return _canonical_manifest(self)

    def compile(self, *, extractor_version: str = "graphiti_v4") -> CompiledGraphSchema:
        if extractor_version not in self.compatible_extractor_versions:
            raise GraphSchemaProfileMismatch("graph_schema_profile_extractor_mismatch")
        if self.key == GENERIC_GRAPH_SCHEMA_PROFILE_KEY:
            return CompiledGraphSchema(
                entity_types=None,
                edge_types=None,
                edge_type_map=None,
                extraction_instructions=self.extraction_instructions,
                validation_policy=self.validation_policy,
            )
        entity_types = {
            entity.name: _compile_model(entity.name, entity.description, entity.attributes)
            for entity in self.entity_manifest
        }
        edge_types = {
            edge.name: _compile_model(edge.name, edge.description, edge.attributes)
            for edge in self.edge_manifest
        }
        edge_type_map = {
            (source, target): list(edge_names)
            for source, target, edge_names in self.edge_type_map_manifest
        }
        return CompiledGraphSchema(
            entity_types=entity_types,
            edge_types=edge_types,
            edge_type_map=edge_type_map,
            extraction_instructions=self.extraction_instructions,
            validation_policy=self.validation_policy,
        )


class GraphSchemaRegistry:
    """Fail-closed resolver for the installed built-in profiles."""

    def __init__(self, profiles: tuple[GraphSchemaProfile, ...]) -> None:
        by_key: dict[str, GraphSchemaProfile] = {}
        for profile in profiles:
            if profile.key in by_key:
                raise GraphSchemaProfileError("graph_schema_profile_duplicate_key")
            by_key[profile.key] = profile
        defaults = [profile for profile in profiles if profile.is_default]
        if len(defaults) != 1 or defaults[0].key != GENERIC_GRAPH_SCHEMA_PROFILE_KEY:
            raise GraphSchemaProfileError("graph_schema_profile_default_invalid")
        self._profiles = MappingProxyType(dict(sorted(by_key.items())))

    def list(self) -> tuple[GraphSchemaProfile, ...]:
        return tuple(self._profiles.values())

    def resolve(
        self,
        key: str,
        *,
        digest: str | None = None,
        extractor_version: str | None = None,
    ) -> GraphSchemaProfile:
        profile = self._profiles.get(key)
        if profile is None:
            raise GraphSchemaProfileMismatch("graph_schema_profile_unknown")
        if digest is not None and digest != profile.digest:
            raise GraphSchemaProfileMismatch("graph_schema_profile_mismatch")
        if extractor_version is not None and extractor_version not in profile.compatible_extractor_versions:
            raise GraphSchemaProfileMismatch("graph_schema_profile_extractor_mismatch")
        return profile

    def compile(
        self,
        key: str,
        *,
        digest: str | None = None,
        extractor_version: str | None = None,
    ) -> CompiledGraphSchema:
        profile = self.resolve(
            key,
            digest=digest,
            extractor_version=extractor_version,
        )
        return profile.compile(extractor_version=extractor_version or "graphiti_v4")


def _compile_model(
    name: str,
    description: str,
    fields: tuple[GraphSchemaField, ...],
) -> type[BaseModel]:
    model_fields: dict[str, tuple[Any, Any]] = {}
    for field in fields:
        if field.kind == "string_list":
            model_fields[field.name] = (
                list[str],
                Field(default_factory=list, description=field.description),
            )
        else:
            model_fields[field.name] = (
                str | None,
                Field(default=None, description=field.description),
            )
    class_name = f"{name}Entity" if name in _ENTITY_NAMES else "TypedRelation"
    return create_model(
        class_name,
        __base__=BaseModel,
        __module__="rag_kb.graph.schema_profiles.registry",
        __doc__=description,
        **model_fields,
    )


_GRAPHITI_BASE_FIELDS = frozenset(
    {
        "uuid",
        "name",
        "group_id",
        "labels",
        "source",
        "source_description",
        "content",
        "created_at",
        "valid_at",
        "invalid_at",
        "summary",
        "attributes",
        "fact",
        "episodes",
    }
)
_ENTITY_NAMES = frozenset(
    {
        "Organization",
        "Project",
        "Repository",
        "Service",
        "License",
        "LicenseExpression",
        "AliasSurface",
    }
)


def _canonical_manifest(profile: GraphSchemaProfile) -> dict[str, Any]:
    """Normalize every order-bearing manifest component before hashing."""

    return {
        "alias_policy": profile.alias_policy,
        "compatible_extractor_versions": sorted(profile.compatible_extractor_versions),
        "description": profile.description,
        "display_name": profile.display_name,
        "edge_manifest": [
            edge.as_manifest()
            for edge in sorted(profile.edge_manifest, key=lambda item: item.name)
        ],
        "edge_type_map_manifest": [
            {
                "edge_names": sorted(edge_names),
                "source": source,
                "target": target,
            }
            for source, target, edge_names in sorted(
                profile.edge_type_map_manifest,
                key=lambda item: (item[0], item[1]),
            )
        ],
        "entity_manifest": [
            entity.as_manifest()
            for entity in sorted(profile.entity_manifest, key=lambda item: item.name)
        ],
        "extraction_instructions": profile.extraction_instructions,
        "is_default": profile.is_default,
        "key": profile.key,
        "validation_policy": profile.validation_policy.as_manifest(),
    }


def _validate_manifest(profile: GraphSchemaProfile) -> None:
    entity_names = [entity.name for entity in profile.entity_manifest]
    edge_names = [edge.name for edge in profile.edge_manifest]
    if len(entity_names) != len(set(entity_names)):
        raise GraphSchemaProfileError("graph_schema_profile_duplicate_entity")
    if len(edge_names) != len(set(edge_names)):
        raise GraphSchemaProfileError("graph_schema_profile_duplicate_edge")
    if any(not name.strip() for name in (*entity_names, *edge_names)):
        raise GraphSchemaProfileError("graph_schema_profile_empty_type")
    for entity in profile.entity_manifest:
        field_names = [field.name for field in entity.attributes]
        if len(field_names) != len(set(field_names)):
            raise GraphSchemaProfileError("graph_schema_profile_duplicate_field")
        if _GRAPHITI_BASE_FIELDS.intersection(field_names):
            raise GraphSchemaProfileError("graph_schema_profile_base_field_collision")
    known_entities = set(entity_names) | {"Entity"}
    known_edges = set(edge_names)
    seen_pairs: set[tuple[str, str]] = set()
    for source, target, pair_edges in profile.edge_type_map_manifest:
        pair = (source, target)
        if pair in seen_pairs:
            raise GraphSchemaProfileError("graph_schema_profile_duplicate_edge_map")
        seen_pairs.add(pair)
        if source not in known_entities or target not in known_entities:
            raise GraphSchemaProfileError("graph_schema_profile_edge_map_type_unknown")
        if not pair_edges or any(edge not in known_edges for edge in pair_edges):
            raise GraphSchemaProfileError("graph_schema_profile_edge_map_edge_unknown")


_SAFE_EXTRACTION_INSTRUCTIONS = """
Extract only facts explicitly stated by a sentence in the current Episode.
Preserve direction and do not infer missing entities, dates, relations, or
world knowledge. Ignore headings, filenames, evaluator IDs, and authoring
instructions. Keep source provenance and distinct relation participants.
""".strip()

_SOFTWARE_EXTRACTION_INSTRUCTIONS = """
Extract only factual relations explicitly stated in the episode. Use the custom
entity and edge types whenever they apply. Preserve the stated direction: the
grammatical subject/source is the source node and the object/target is the target
node. Keep projects, organizations, repositories, services, licenses, and
license expressions as distinct concepts. A repository path is a Repository,
not a Project. A compound SPDX expression is one LicenseExpression and must not
be silently reduced to one of its licenses.

Preserve codes, short names, and aliases exactly. When an alias or short name is
explicitly stated, create an AliasSurface node for that literal surface and link
it to the canonical entity with HasShortName or AliasOf. Do not merge an alias
edge into a self-loop.

Do not extract document structure or authoring instructions as knowledge-graph
facts. In particular, ignore section labels, relation IDs, evidence-unit IDs,
filenames, headings that only organize the document, and generic descriptions
of how evidence or graph extraction should work. Do not infer missing entities,
relations, directions, dates, or world knowledge. Every extracted edge must be
supported by a specific sentence in this episode and must connect two distinct
participants named or unambiguously referenced in that sentence.
""".strip()


def _field(name: str, kind: Literal["optional_string", "string_list"], description: str = "") -> GraphSchemaField:
    return GraphSchemaField(name=name, kind=kind, description=description)


_SOFTWARE_ENTITIES = (
    GraphSchemaEntity(
        "Organization",
        "A foundation, company, standards body, or other organization.",
        (_field("short_names", "string_list"),),
    ),
    GraphSchemaEntity(
        "Project",
        "A software, research, or community project.",
        (_field("aliases", "string_list"),),
    ),
    GraphSchemaEntity(
        "Repository",
        "A source-code repository or explicit repository path.",
        (_field("repository_path", "optional_string"),),
    ),
    GraphSchemaEntity(
        "Service",
        "A hosted website, documentation site, registry, or network service.",
        (_field("service_kind", "optional_string"),),
    ),
    GraphSchemaEntity(
        "License",
        "A named software or content license, preferably with its SPDX identifier.",
        (_field("spdx_identifier", "optional_string"),),
    ),
    GraphSchemaEntity(
        "LicenseExpression",
        "A complete SPDX-style license expression, including AND/OR operators.",
        (_field("expression", "optional_string"),),
    ),
    GraphSchemaEntity(
        "AliasSurface",
        "An explicit alias surface that remains distinct from its canonical node.",
        (_field("canonical_surface", "optional_string"),),
    ),
)

_SOFTWARE_EDGES = tuple(
    GraphSchemaEdge(
        name,
        "An explicitly stated, directed relation between two typed entities.",
        (_field("qualifier", "optional_string"),),
    )
    for name in (
        "Stewards", "Hosts", "Maintains", "Operates", "HasRepository",
        "DistributedUnder", "Lists", "HasLicenseExpression",
        "DocumentationHostedAt", "UsesImportNamespace", "Requires", "BuiltOn",
        "FoundationFor", "OriginatedFrom", "DevelopedOn", "Sponsors",
        "GraduatedProjectOf", "PartOf", "HasShortName", "AliasOf",
    )
)

_SOFTWARE_EDGE_MAP = (
    ("Organization", "Project", ("Stewards", "Hosts", "Sponsors", "FoundationFor")),
    ("Organization", "Service", ("Operates", "Hosts")),
    ("Organization", "Repository", ("Maintains", "Hosts")),
    ("Project", "Repository", ("HasRepository", "Maintains", "DevelopedOn")),
    ("Project", "Service", ("DocumentationHostedAt", "Hosts")),
    ("Project", "License", ("DistributedUnder", "Lists")),
    ("Project", "LicenseExpression", ("HasLicenseExpression", "Lists")),
    ("Project", "Project", ("Requires", "BuiltOn", "OriginatedFrom", "PartOf")),
    ("Project", "AliasSurface", ("HasShortName", "AliasOf")),
    ("Organization", "AliasSurface", ("HasShortName", "AliasOf")),
    ("Entity", "Entity", tuple(edge.name for edge in _SOFTWARE_EDGES)),
)

GENERIC_GRAPH_SCHEMA_PROFILE = GraphSchemaProfile(
    key=GENERIC_GRAPH_SCHEMA_PROFILE_KEY,
    display_name="Generic open-domain knowledge",
    description="Graphiti native extraction for general personal knowledge.",
    entity_manifest=(),
    edge_manifest=(),
    edge_type_map_manifest=(),
    extraction_instructions=_SAFE_EXTRACTION_INSTRUCTIONS,
    alias_policy="native_dedupe",
    validation_policy=GraphSchemaValidationPolicy(),
    is_default=True,
    compatible_extractor_versions=("graphiti_v4",),
)

SOFTWARE_GRAPH_SCHEMA_PROFILE = GraphSchemaProfile(
    key=SOFTWARE_GRAPH_SCHEMA_PROFILE_KEY,
    display_name="Software and project knowledge",
    description="Typed entities and relations for software/project corpora.",
    entity_manifest=_SOFTWARE_ENTITIES,
    edge_manifest=_SOFTWARE_EDGES,
    edge_type_map_manifest=_SOFTWARE_EDGE_MAP,
    extraction_instructions=_SOFTWARE_EXTRACTION_INSTRUCTIONS,
    alias_policy="standalone_typed_alias",
    validation_policy=GraphSchemaValidationPolicy(standalone_alias_orphan_check=True),
    is_default=False,
    compatible_extractor_versions=("graphiti_v3", "graphiti_v4"),
)

_REGISTRY = GraphSchemaRegistry((GENERIC_GRAPH_SCHEMA_PROFILE, SOFTWARE_GRAPH_SCHEMA_PROFILE))


def get_graph_schema_registry() -> GraphSchemaRegistry:
    return _REGISTRY


__all__ = [
    "CompiledGraphSchema",
    "GENERIC_GRAPH_SCHEMA_PROFILE",
    "GENERIC_GRAPH_SCHEMA_PROFILE_KEY",
    "GraphSchemaEdge",
    "GraphSchemaEntity",
    "GraphSchemaField",
    "GraphSchemaProfile",
    "GraphSchemaProfileError",
    "GraphSchemaProfileMismatch",
    "GraphSchemaRegistry",
    "GraphSchemaValidationPolicy",
    "SOFTWARE_GRAPH_SCHEMA_PROFILE",
    "SOFTWARE_GRAPH_SCHEMA_PROFILE_KEY",
    "get_graph_schema_registry",
]

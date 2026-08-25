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
ENTERPRISE_GRAPH_SCHEMA_PROFILE_KEY = "enterprise_knowledge_v1"
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
            entity.name: _compile_model(
                entity.name,
                entity.description,
                entity.attributes,
                model_kind="entity",
            )
            for entity in self.entity_manifest
        }
        edge_types = {
            edge.name: _compile_model(
                edge.name,
                edge.description,
                edge.attributes,
                model_kind="edge",
            )
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
        self._compiled_cache: dict[tuple[str, str, str], CompiledGraphSchema] = {}

    def list(self) -> tuple[GraphSchemaProfile, ...]:
        return tuple(
            sorted(
                self._profiles.values(),
                key=lambda profile: (not profile.is_default, profile.key),
            )
        )

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
        resolved_extractor_version = extractor_version or "graphiti_v4"
        cache_key = (profile.key, profile.digest, resolved_extractor_version)
        compiled = self._compiled_cache.get(cache_key)
        if compiled is None:
            compiled = profile.compile(extractor_version=resolved_extractor_version)
            self._compiled_cache[cache_key] = compiled
        return compiled


def _compile_model(
    name: str,
    description: str,
    fields: tuple[GraphSchemaField, ...],
    *,
    model_kind: Literal["entity", "edge"],
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
    class_name = f"{name}Entity" if model_kind == "entity" else "TypedRelation"
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
    for schema_type in (*profile.entity_manifest, *profile.edge_manifest):
        field_names = [field.name for field in schema_type.attributes]
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

_ENTERPRISE_EXTRACTION_INSTRUCTIONS = """
Extract only enterprise facts explicitly stated in the current episode. Use the
custom entity and edge types whenever they apply, and normalize active or
passive wording into the semantic direction defined below. Do not infer an
organization chart, responsibility, approval, process dependency, policy scope,
or current status from convention or outside knowledge.

Keep named individuals (Person), positions or responsibility hats (Role), and
teams or departments (OrganizationalUnit) distinct. Keep a recurring workflow
(Process), business application (BusinessSystem), customer or internal offering
(Product), normative rule (Policy), time-bounded initiative (Project),
operational site (Facility),
geographic place (Location), and concrete non-policy artifact (Document)
distinct. Organization may represent the enterprise itself or an explicitly
named external company, authority, fund, university, research institute,
supplier, or partner. Preserve official names, acronyms, identifiers,
versions, status words, and effective periods exactly; use native entity
resolution for explicit aliases instead of creating separate alias entities.

Use these directions consistently: a child PartOf its parent; a Person MemberOf
an organization or unit and ReportsTo a manager; a Person HoldsRole a role and
ServesAs that stated position at an organization or unit; an actor Owns, Leads,
Sponsors, Approves, or has a RACI relation toward the governed object; a parent
Controls or Establishes an organization; an acquirer Acquires its target; an
investor InvestsIn its target; a merging source MergesInto its destination; a
supplier SuppliesTo its recipient; a developer Develops a product or system; a
builder Builds a project or facility; a governed object GovernedBy a Policy, while a Policy
AppliesTo its scope; a dependent DependsOn its dependency; a Document Documents
its subject; a Document or Policy Defines a term; a Process Produces or Consumes
an artifact or resource; a Project Delivers its outcome; an item LocatedAt a
place or DeployedAt a project, facility, or location; and a newer item
Supersedes the older item. PartnersWith and IndependentOf are symmetric facts
even though one stored direction is retained.

Create ResponsibleFor, AccountableFor, ConsultedOn, and InformedAbout only when
that exact RACI meaning or an unambiguous equivalent is stated. Membership does
not imply reporting, ownership does not imply approval, authorship does not
imply accountability, and a policy mention does not imply that the policy
applies. Do not turn a minority investment into Controls, a contract into
PartnersWith, or a supplied item into ownership. Do not turn confidentiality
labels, access-control notices, headings, filenames, table column labels,
templates, or authoring instructions into graph facts. Every edge must be
supported by a specific sentence or table row in this episode and connect two
distinct, explicitly named or unambiguously referenced participants.
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

_ENTERPRISE_ENTITIES = (
    GraphSchemaEntity(
        "Organization",
        "The enterprise or a named company, authority, fund, university, "
        "institute, supplier, or partner.",
        (
            _field("organization_kind", "optional_string"),
            _field("aliases", "string_list"),
        ),
    ),
    GraphSchemaEntity(
        "OrganizationalUnit",
        "A department, division, business unit, team, committee, or other organization unit.",
        (
            _field("unit_kind", "optional_string"),
            _field("aliases", "string_list"),
        ),
    ),
    GraphSchemaEntity(
        "Person",
        "A specifically named individual, not a job title or generic actor.",
        (_field("job_title", "optional_string"),),
    ),
    GraphSchemaEntity(
        "Role",
        "A job title, process role, governance role, or responsibility hat "
        "independent of a person.",
        (_field("role_scope", "optional_string"),),
    ),
    GraphSchemaEntity(
        "Policy",
        "A named policy, standard, rule, control obligation, or governance requirement.",
        (
            _field("policy_identifier", "optional_string"),
            _field("status", "optional_string"),
        ),
    ),
    GraphSchemaEntity(
        "Process",
        "A recurring business process, procedure, workflow, or operational activity.",
        (
            _field("process_identifier", "optional_string"),
            _field("aliases", "string_list"),
        ),
    ),
    GraphSchemaEntity(
        "BusinessSystem",
        "A named business application, platform, data system, or operational technology system.",
        (
            _field("system_identifier", "optional_string"),
            _field("aliases", "string_list"),
        ),
    ),
    GraphSchemaEntity(
        "Product",
        "A named customer-facing or internal product, service offering, or managed capability.",
        (
            _field("product_kind", "optional_string"),
            _field("aliases", "string_list"),
        ),
    ),
    GraphSchemaEntity(
        "Project",
        "A time-bounded initiative, program, transformation, or delivery project.",
        (
            _field("project_status", "optional_string"),
            _field("aliases", "string_list"),
        ),
    ),
    GraphSchemaEntity(
        "Document",
        "A named non-policy document or business artifact such as a procedure, "
        "contract, report, or form.",
        (
            _field("document_identifier", "optional_string"),
            _field("document_kind", "optional_string"),
            _field("version", "optional_string"),
        ),
    ),
    GraphSchemaEntity(
        "Location",
        "A named city, region, jurisdiction, address, or other geographic place.",
        (_field("location_kind", "optional_string"),),
    ),
    GraphSchemaEntity(
        "Facility",
        "A named office, plant, warehouse, data center, laboratory, park, station, "
        "or operational site.",
        (_field("facility_kind", "optional_string"),),
    ),
    GraphSchemaEntity(
        "BusinessTerm",
        "A defined enterprise term, acronym, classification, or glossary concept.",
        (_field("abbreviations", "string_list"),),
    ),
)

_ENTERPRISE_EDGE_ATTRIBUTES = (
    _field("qualifier", "optional_string", "An explicitly stated scope or condition."),
    _field(
        "effective_period",
        "optional_string",
        "The exact stated effective or validity period, without inference.",
    ),
)


def _enterprise_edge(name: str, description: str) -> GraphSchemaEdge:
    return GraphSchemaEdge(name, description, _ENTERPRISE_EDGE_ATTRIBUTES)


_ENTERPRISE_EDGES = (
    _enterprise_edge(
        "PartOf",
        "The source child is structurally part of the target parent.",
    ),
    _enterprise_edge(
        "MemberOf",
        "The source person belongs to the target unit or organization.",
    ),
    _enterprise_edge(
        "ReportsTo",
        "The source person or unit formally reports to the target manager or unit.",
    ),
    _enterprise_edge("HoldsRole", "The source person explicitly holds the target role."),
    _enterprise_edge(
        "ServesAs",
        "The source person explicitly serves in the stated position at the "
        "target organization or unit.",
    ),
    _enterprise_edge("Leads", "The source person, role, or unit explicitly leads the target."),
    _enterprise_edge("Owns", "The source actor has stated business ownership of the target."),
    _enterprise_edge(
        "Controls",
        "The source organization explicitly controls the target organization.",
    ),
    _enterprise_edge(
        "Establishes",
        "The source actor explicitly establishes the target organization.",
    ),
    _enterprise_edge(
        "Acquires",
        "The source organization explicitly acquires the target organization.",
    ),
    _enterprise_edge(
        "InvestsIn",
        "The source investor explicitly invests in the target organization or project.",
    ),
    _enterprise_edge(
        "MergesInto",
        "The source organization explicitly merges into the target organization.",
    ),
    _enterprise_edge(
        "ResponsibleFor",
        "The source actor has stated execution responsibility for the target.",
    ),
    _enterprise_edge(
        "AccountableFor",
        "The source actor has stated ultimate accountability for the target.",
    ),
    _enterprise_edge("ConsultedOn", "The source actor is explicitly consulted about the target."),
    _enterprise_edge(
        "InformedAbout",
        "The source actor must explicitly be informed about the target.",
    ),
    _enterprise_edge("Approves", "The source actor explicitly approves the target."),
    _enterprise_edge(
        "Sponsors",
        "The source actor explicitly sponsors the target initiative or product.",
    ),
    _enterprise_edge(
        "Appoints",
        "The source organization or unit explicitly appoints the target person.",
    ),
    _enterprise_edge(
        "Operates",
        "The source actor explicitly operates the target process, system, "
        "product, project, or facility.",
    ),
    _enterprise_edge("Uses", "The source actor or process explicitly uses the target resource."),
    _enterprise_edge(
        "Provides",
        "The source organization or unit explicitly provides the target system or product.",
    ),
    _enterprise_edge(
        "Develops",
        "The source actor explicitly develops the target system or product.",
    ),
    _enterprise_edge(
        "Builds",
        "The source actor explicitly builds or implements the target project or facility.",
    ),
    _enterprise_edge(
        "SuppliesTo",
        "The source supplier explicitly supplies goods, equipment, data, or "
        "interfaces to the target.",
    ),
    _enterprise_edge(
        "ContractsWith",
        "The source party explicitly has the stated contract with the target party or project.",
    ),
    _enterprise_edge(
        "PartnersWith",
        "The source and target explicitly collaborate, partner, or jointly build something.",
    ),
    _enterprise_edge(
        "IndependentOf",
        "The source and target are explicitly stated to be independent or unaffiliated.",
    ),
    _enterprise_edge(
        "DependsOn",
        "The source dependent explicitly depends on the target dependency.",
    ),
    _enterprise_edge(
        "GovernedBy",
        "The source object or activity is explicitly governed by the target policy.",
    ),
    _enterprise_edge("AppliesTo", "The source policy explicitly applies to the target scope."),
    _enterprise_edge("Documents", "The source document explicitly documents the target subject."),
    _enterprise_edge(
        "Defines",
        "The source document or policy explicitly defines the target term.",
    ),
    _enterprise_edge(
        "Produces",
        "The source process explicitly produces the target artifact or product.",
    ),
    _enterprise_edge(
        "Consumes",
        "The source process explicitly consumes the target artifact or system.",
    ),
    _enterprise_edge(
        "Supports",
        "The source actor, process, system, product, project, or facility "
        "explicitly supports the target.",
    ),
    _enterprise_edge("LocatedAt", "The source entity is explicitly located at the target place."),
    _enterprise_edge(
        "DeployedAt",
        "The source system or product is explicitly deployed at the target "
        "project, facility, or location.",
    ),
    _enterprise_edge(
        "CertifiedBy",
        "The source organization, system, or product is explicitly certified "
        "by the target organization.",
    ),
    _enterprise_edge(
        "LicensedBy",
        "The source organization, system, or product explicitly receives a "
        "license from the target organization.",
    ),
    _enterprise_edge(
        "Supersedes",
        "The source newer item explicitly supersedes the target older item.",
    ),
    _enterprise_edge("Delivers", "The source project explicitly delivers the target outcome."),
)

_ENTERPRISE_RACI_EDGES = (
    "ResponsibleFor",
    "AccountableFor",
    "ConsultedOn",
    "InformedAbout",
)
_ENTERPRISE_STEWARDSHIP_EDGES = (
    "Owns",
    *_ENTERPRISE_RACI_EDGES,
)
_ENTERPRISE_APPROVAL_EDGES = (
    *_ENTERPRISE_STEWARDSHIP_EDGES,
    "Approves",
)

_ENTERPRISE_EDGE_MAP = (
    (
        "Organization",
        "Organization",
        (
            "PartOf",
            "Controls",
            "Establishes",
            "Acquires",
            "InvestsIn",
            "MergesInto",
            "Operates",
            "SuppliesTo",
            "ContractsWith",
            "PartnersWith",
            "Supports",
            "CertifiedBy",
            "LicensedBy",
            "IndependentOf",
        ),
    ),
    ("Organization", "Person", ("Appoints",)),
    ("Organization", "Policy", ("GovernedBy",)),
    (
        "Organization",
        "BusinessSystem",
        ("Owns", "Operates", "Provides", "Develops", "Supports"),
    ),
    (
        "Organization",
        "Product",
        ("Owns", "Operates", "Provides", "Develops", "Supports"),
    ),
    (
        "Organization",
        "Project",
        (
            "Owns",
            "InvestsIn",
            "Sponsors",
            "Operates",
            "Builds",
            "SuppliesTo",
            "ContractsWith",
            "Supports",
        ),
    ),
    (
        "Organization",
        "Facility",
        ("Owns", "Operates", "Builds", "SuppliesTo", "ContractsWith", "Supports"),
    ),
    ("Organization", "Document", ("Owns", "Approves")),
    ("Organization", "Location", ("LocatedAt",)),
    ("OrganizationalUnit", "Organization", ("PartOf",)),
    (
        "OrganizationalUnit",
        "OrganizationalUnit",
        ("PartOf", "ReportsTo", "PartnersWith", "Supports"),
    ),
    ("OrganizationalUnit", "Person", ("Appoints",)),
    (
        "OrganizationalUnit",
        "Policy",
        (*_ENTERPRISE_APPROVAL_EDGES, "GovernedBy"),
    ),
    (
        "OrganizationalUnit",
        "Process",
        ("Leads", *_ENTERPRISE_APPROVAL_EDGES, "Operates", "Supports"),
    ),
    (
        "OrganizationalUnit",
        "BusinessSystem",
        (
            *_ENTERPRISE_STEWARDSHIP_EDGES,
            "Operates",
            "Uses",
            "Provides",
            "Develops",
            "Supports",
        ),
    ),
    (
        "OrganizationalUnit",
        "Product",
        ("Leads", *_ENTERPRISE_STEWARDSHIP_EDGES, "Provides", "Develops", "Supports"),
    ),
    (
        "OrganizationalUnit",
        "Project",
        (
            "Leads",
            *_ENTERPRISE_APPROVAL_EDGES,
            "InvestsIn",
            "Sponsors",
            "Operates",
            "Builds",
            "ContractsWith",
            "Supports",
        ),
    ),
    ("OrganizationalUnit", "Document", _ENTERPRISE_APPROVAL_EDGES),
    (
        "OrganizationalUnit",
        "Facility",
        (*_ENTERPRISE_STEWARDSHIP_EDGES, "Operates", "Uses", "Builds", "Supports"),
    ),
    ("OrganizationalUnit", "Location", ("LocatedAt",)),
    ("Person", "Organization", ("MemberOf", "ServesAs")),
    ("Person", "OrganizationalUnit", ("MemberOf", "ServesAs", "Leads")),
    ("Person", "Person", ("ReportsTo",)),
    ("Person", "Role", ("HoldsRole",)),
    ("Person", "Policy", _ENTERPRISE_APPROVAL_EDGES),
    ("Person", "Process", ("Leads", *_ENTERPRISE_APPROVAL_EDGES)),
    ("Person", "BusinessSystem", (*_ENTERPRISE_STEWARDSHIP_EDGES, "Uses")),
    ("Person", "Product", ("Leads", *_ENTERPRISE_STEWARDSHIP_EDGES)),
    ("Person", "Project", ("Leads", *_ENTERPRISE_APPROVAL_EDGES, "Sponsors")),
    ("Person", "Document", _ENTERPRISE_APPROVAL_EDGES),
    ("Person", "Facility", _ENTERPRISE_STEWARDSHIP_EDGES),
    ("Person", "Location", ("LocatedAt",)),
    ("Role", "OrganizationalUnit", ("PartOf", "Leads")),
    ("Role", "Policy", _ENTERPRISE_APPROVAL_EDGES),
    ("Role", "Process", ("Leads", *_ENTERPRISE_APPROVAL_EDGES)),
    ("Role", "BusinessSystem", (*_ENTERPRISE_STEWARDSHIP_EDGES, "Uses")),
    ("Role", "Product", ("Leads", *_ENTERPRISE_STEWARDSHIP_EDGES)),
    ("Role", "Project", ("Leads", *_ENTERPRISE_APPROVAL_EDGES, "Sponsors")),
    ("Role", "Document", _ENTERPRISE_APPROVAL_EDGES),
    ("Role", "Facility", _ENTERPRISE_STEWARDSHIP_EDGES),
    ("Policy", "Organization", ("AppliesTo",)),
    ("Policy", "OrganizationalUnit", ("AppliesTo",)),
    ("Policy", "Person", ("AppliesTo",)),
    ("Policy", "Role", ("AppliesTo",)),
    ("Policy", "Process", ("AppliesTo",)),
    ("Policy", "BusinessSystem", ("AppliesTo",)),
    ("Policy", "Product", ("AppliesTo",)),
    ("Policy", "Project", ("AppliesTo",)),
    ("Policy", "Location", ("AppliesTo",)),
    ("Policy", "Facility", ("AppliesTo",)),
    ("Policy", "BusinessTerm", ("Defines",)),
    ("Policy", "Policy", ("Supersedes", "DependsOn")),
    ("Process", "Policy", ("GovernedBy",)),
    ("Process", "Process", ("DependsOn", "Supersedes", "Supports")),
    ("Process", "BusinessSystem", ("Uses", "DependsOn")),
    ("Process", "Product", ("Produces", "Supports")),
    ("Process", "Document", ("Produces", "Consumes")),
    ("BusinessSystem", "Policy", ("GovernedBy",)),
    ("BusinessSystem", "Organization", ("CertifiedBy", "LicensedBy")),
    ("BusinessSystem", "BusinessSystem", ("DependsOn", "Supersedes")),
    ("BusinessSystem", "Process", ("Supports",)),
    ("BusinessSystem", "Product", ("Supports",)),
    ("BusinessSystem", "Project", ("DeployedAt", "Supports")),
    ("BusinessSystem", "Facility", ("DeployedAt", "Supports")),
    ("BusinessSystem", "Location", ("LocatedAt",)),
    ("Product", "Policy", ("GovernedBy",)),
    ("Product", "Organization", ("CertifiedBy", "LicensedBy")),
    ("Product", "BusinessSystem", ("DependsOn",)),
    ("Product", "Product", ("DependsOn", "Supersedes")),
    ("Product", "Project", ("DeployedAt", "Supports")),
    ("Product", "Facility", ("DeployedAt", "Supports")),
    ("Product", "Location", ("DeployedAt", "LocatedAt")),
    ("Project", "Policy", ("GovernedBy",)),
    ("Project", "Process", ("Delivers", "DependsOn")),
    ("Project", "BusinessSystem", ("Delivers", "DependsOn")),
    ("Project", "Product", ("Delivers",)),
    ("Project", "Document", ("Delivers",)),
    ("Project", "Project", ("PartOf", "DependsOn", "Supersedes", "Supports")),
    ("Project", "Facility", ("Delivers", "Builds", "Supports")),
    ("Project", "Location", ("LocatedAt",)),
    ("Facility", "Policy", ("GovernedBy",)),
    ("Facility", "BusinessSystem", ("Uses", "DependsOn")),
    ("Facility", "Facility", ("PartOf", "DependsOn", "Supersedes")),
    ("Facility", "Location", ("LocatedAt",)),
    ("Document", "OrganizationalUnit", ("Documents",)),
    ("Document", "Organization", ("Documents",)),
    ("Document", "Role", ("Documents",)),
    ("Document", "Policy", ("Documents",)),
    ("Document", "Process", ("Documents",)),
    ("Document", "BusinessSystem", ("Documents",)),
    ("Document", "Product", ("Documents",)),
    ("Document", "Project", ("Documents",)),
    ("Document", "Facility", ("Documents",)),
    ("Document", "BusinessTerm", ("Defines",)),
    ("Document", "Document", ("Supersedes", "DependsOn")),
    ("Location", "Location", ("PartOf",)),
    ("Entity", "Entity", tuple(edge.name for edge in _ENTERPRISE_EDGES)),
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

ENTERPRISE_GRAPH_SCHEMA_PROFILE = GraphSchemaProfile(
    key=ENTERPRISE_GRAPH_SCHEMA_PROFILE_KEY,
    display_name="Enterprise organization and operations",
    description=(
        "Typed organization, people, responsibility, policy, process, system, "
        "project, product, facility, document, glossary, and location knowledge."
    ),
    entity_manifest=_ENTERPRISE_ENTITIES,
    edge_manifest=_ENTERPRISE_EDGES,
    edge_type_map_manifest=_ENTERPRISE_EDGE_MAP,
    extraction_instructions=_ENTERPRISE_EXTRACTION_INSTRUCTIONS,
    alias_policy="native_dedupe",
    validation_policy=GraphSchemaValidationPolicy(),
    is_default=False,
    compatible_extractor_versions=("graphiti_v4",),
)

_REGISTRY = GraphSchemaRegistry(
    (
        GENERIC_GRAPH_SCHEMA_PROFILE,
        SOFTWARE_GRAPH_SCHEMA_PROFILE,
        ENTERPRISE_GRAPH_SCHEMA_PROFILE,
    )
)


def get_graph_schema_registry() -> GraphSchemaRegistry:
    return _REGISTRY


__all__ = [
    "CompiledGraphSchema",
    "ENTERPRISE_GRAPH_SCHEMA_PROFILE",
    "ENTERPRISE_GRAPH_SCHEMA_PROFILE_KEY",
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

#!/usr/bin/env python3
"""Build the small stable-ID stress corpus for ``enterprise_knowledge_v1``."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from rag_kb.domain import (
    ENTERPRISE_GRAPH_SCHEMA_PROFILE_DIGEST,
    ENTERPRISE_GRAPH_SCHEMA_PROFILE_KEY,
    GRAPH_EXTRACTOR_VERSION,
)
from rag_kb.graph.schema_profiles.registry import ENTERPRISE_GRAPH_SCHEMA_PROFILE


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "evaluation/enterprise-profile-qualification-v2"
SCHEMA = "enterprise_profile_stress_v1"
DATASET_ID = "enterprise-profile-qualification-v2"


class CorpusError(RuntimeError):
    """Raised when the code-owned Enterprise stress corpus is invalid."""


def _entity(
    entity_id: str,
    name: str,
    entity_type: str,
    *aliases: str,
) -> dict[str, Any]:
    return {
        "entity_id": entity_id,
        "name": name,
        "entity_type": entity_type,
        "aliases": list(aliases),
    }


_ENTITIES = (
    _entity("org-meridian", "Meridian Systems International", "Organization", "Meridian", "MSI"),
    _entity("person-alice", "Alice Chen", "Person", "Alice", "A. Chen"),
    _entity("role-cto", "Chief Technology Officer", "Role", "CTO"),
    _entity("unit-platform", "Platform Engineering Division", "OrganizationalUnit", "Platform Engineering"),
    _entity("unit-risk", "Enterprise Risk Committee", "OrganizationalUnit", "Risk Committee"),
    _entity("unit-security", "Security Architecture Office", "OrganizationalUnit", "Security Office"),
    _entity("unit-operations", "Operations Governance Board", "OrganizationalUnit", "Operations Board"),
    _entity("system-orion", "Orion Platform", "BusinessSystem", "Orion", "Orion system"),
    _entity("org-harbor", "Harbor Systems", "Organization", "Harbor", "HS"),
    _entity("system-atlas", "Atlas Platform", "BusinessSystem", "Atlas", "Atlas platform"),
    _entity("org-northwind", "Northwind Logistics", "Organization", "Northwind"),
    _entity("project-delta", "Delta Renewal Project", "Project", "Delta Renewal", "Delta project"),
    _entity("product-nova", "Nova Analytics", "Product", "Nova", "Nova product"),
    _entity("process-release", "Release Management Process", "Process", "Release Management"),
    _entity("doc-runbook", "Operations Runbook", "Document", "runbook"),
    _entity("facility-east-dock", "East Dock Data Center", "Facility", "East Dock"),
    _entity("location-singapore", "Singapore", "Location", "Singapore location"),
    _entity("policy-access", "Access Control Standard", "Policy", "Access Standard"),
    _entity("doc-control", "Control Manual", "Document", "control handbook"),
    _entity("term-privileged", "Privileged Access", "BusinessTerm", "PA classification"),
    _entity("policy-legacy", "Legacy Access Standard", "Policy", "Legacy Standard"),
    _entity("policy-revised", "Revised Access Standard", "Policy", "Revised Standard"),
    _entity("org-meridian-integration", "Meridian System Integration", "Organization", "Meridian Integration", "MII"),
)


def _relation(
    relation_id: str,
    source_entity_id: str,
    edge_type: str,
    target_entity_id: str,
    evidence_text: str,
) -> dict[str, str]:
    return {
        "relation_id": relation_id,
        "source_entity_id": source_entity_id,
        "edge_type": edge_type,
        "target_entity_id": target_entity_id,
        "evidence_text": evidence_text,
    }


def _control(
    control_id: str,
    source_entity_id: str,
    asserted_edge: str,
    target_entity_id: str,
    forbidden_edge: str,
    evidence_text: str,
) -> dict[str, str]:
    return {
        "control_id": control_id,
        "source_entity_id": source_entity_id,
        "asserted_edge": asserted_edge,
        "target_entity_id": target_entity_id,
        "forbidden_edge": forbidden_edge,
        "evidence_text": evidence_text,
    }


_EPISODES: tuple[dict[str, Any], ...] = (
    {
        "episode_id": "stress-role-raci",
        "family": "semantic_overlap",
        "filename": "01-role-raci.md",
        "text": """# Role and RACI record

At Meridian Systems International, Alice Chen serves as Chief Technology Officer. Alice Chen holds the Chief Technology Officer role and leads Platform Engineering Division. Platform Engineering Division is responsible for Orion Platform. Enterprise Risk Committee is accountable for Orion Platform. Security Architecture Office is consulted on Orion Platform. Operations Governance Board must be informed about Orion Platform. Serving at Meridian does not mean that Alice holds Meridian itself as a role.
""",
        "relations": (
            _relation("stress-role-01", "person-alice", "ServesAs", "org-meridian", "Alice Chen serves as Chief Technology Officer"),
            _relation("stress-role-02", "person-alice", "HoldsRole", "role-cto", "Alice Chen holds the Chief Technology Officer role"),
            _relation("stress-role-03", "person-alice", "Leads", "unit-platform", "leads Platform Engineering Division"),
            _relation("stress-role-04", "unit-platform", "ResponsibleFor", "system-orion", "Platform Engineering Division is responsible for Orion Platform"),
            _relation("stress-role-05", "unit-risk", "AccountableFor", "system-orion", "Enterprise Risk Committee is accountable for Orion Platform"),
            _relation("stress-role-06", "unit-security", "ConsultedOn", "system-orion", "Security Architecture Office is consulted on Orion Platform"),
            _relation("stress-role-07", "unit-operations", "InformedAbout", "system-orion", "Operations Governance Board must be informed about Orion Platform"),
        ),
        "controls": (
            _control("stress-control-01", "person-alice", "ServesAs", "org-meridian", "HoldsRole", "Serving at Meridian does not mean that Alice holds Meridian itself as a role"),
        ),
    },
    {
        "episode_id": "stress-delivery-family",
        "family": "semantic_overlap",
        "filename": "02-delivery-family.md",
        "text": """# Delivery and supply record

Harbor Systems provides Atlas Platform. Harbor Systems supplies equipment to Northwind Logistics. Delta Renewal Project delivers Nova Analytics. Release Management Process produces Nova Analytics and consumes Operations Runbook. Harbor Systems supports Delta Renewal Project. Providing Atlas is not a statement that Harbor delivers Atlas, and Delta delivering Nova is not a statement that Delta provides Nova.
""",
        "relations": (
            _relation("stress-delivery-01", "org-harbor", "Provides", "system-atlas", "Harbor Systems provides Atlas Platform"),
            _relation("stress-delivery-02", "org-harbor", "SuppliesTo", "org-northwind", "Harbor Systems supplies equipment to Northwind Logistics"),
            _relation("stress-delivery-03", "project-delta", "Delivers", "product-nova", "Delta Renewal Project delivers Nova Analytics"),
            _relation("stress-delivery-04", "process-release", "Produces", "product-nova", "Release Management Process produces Nova Analytics"),
            _relation("stress-delivery-05", "process-release", "Consumes", "doc-runbook", "consumes Operations Runbook"),
            _relation("stress-delivery-06", "org-harbor", "Supports", "project-delta", "Harbor Systems supports Delta Renewal Project"),
        ),
        "controls": (
            _control("stress-control-02", "org-harbor", "Provides", "system-atlas", "Delivers", "Providing Atlas is not a statement that Harbor delivers Atlas"),
            _control("stress-control-03", "project-delta", "Delivers", "product-nova", "Provides", "Delta delivering Nova is not a statement that Delta provides Nova"),
        ),
    },
    {
        "episode_id": "stress-location-deployment",
        "family": "semantic_overlap",
        "filename": "03-location-deployment.md",
        "text": """# Location and deployment record

Nova Analytics is deployed at East Dock Data Center. Nova Analytics is physically located in Singapore. Atlas Platform is deployed at East Dock Data Center and is located in Singapore. East Dock Data Center is located in Singapore. Delta Renewal Project is located in Singapore. The physical location of Nova in Singapore does not state that Nova is deployed at Singapore itself.
""",
        "relations": (
            _relation("stress-location-01", "product-nova", "DeployedAt", "facility-east-dock", "Nova Analytics is deployed at East Dock Data Center"),
            _relation("stress-location-02", "product-nova", "LocatedAt", "location-singapore", "Nova Analytics is physically located in Singapore"),
            _relation("stress-location-03", "system-atlas", "DeployedAt", "facility-east-dock", "Atlas Platform is deployed at East Dock Data Center"),
            _relation("stress-location-04", "system-atlas", "LocatedAt", "location-singapore", "is located in Singapore"),
            _relation("stress-location-05", "facility-east-dock", "LocatedAt", "location-singapore", "East Dock Data Center is located in Singapore"),
            _relation("stress-location-06", "project-delta", "LocatedAt", "location-singapore", "Delta Renewal Project is located in Singapore"),
        ),
        "controls": (
            _control("stress-control-04", "product-nova", "LocatedAt", "location-singapore", "DeployedAt", "does not state that Nova is deployed at Singapore itself"),
        ),
    },
    {
        "episode_id": "stress-governance",
        "family": "semantic_overlap",
        "filename": "04-governance.md",
        "text": """# Governance record

Orion Platform is governed by Access Control Standard. Access Control Standard applies to Orion Platform. Control Manual documents Orion Platform. Access Control Standard defines Privileged Access. Revised Access Standard supersedes Legacy Access Standard and depends on Access Control Standard. Applying Access Control Standard to Orion does not mean that the standard is governed by Orion.
""",
        "relations": (
            _relation("stress-governance-01", "system-orion", "GovernedBy", "policy-access", "Orion Platform is governed by Access Control Standard"),
            _relation("stress-governance-02", "policy-access", "AppliesTo", "system-orion", "Access Control Standard applies to Orion Platform"),
            _relation("stress-governance-03", "doc-control", "Documents", "system-orion", "Control Manual documents Orion Platform"),
            _relation("stress-governance-04", "policy-access", "Defines", "term-privileged", "Access Control Standard defines Privileged Access"),
            _relation("stress-governance-05", "policy-revised", "Supersedes", "policy-legacy", "Revised Access Standard supersedes Legacy Access Standard"),
            _relation("stress-governance-06", "policy-revised", "DependsOn", "policy-access", "depends on Access Control Standard"),
        ),
        "controls": (
            _control("stress-control-05", "policy-access", "AppliesTo", "system-orion", "GovernedBy", "does not mean that the standard is governed by Orion"),
        ),
    },
    {
        "episode_id": "stress-identity-canonical",
        "family": "cross_episode_identity",
        "filename": "05-identity-canonical.md",
        "text": "# Identity record A\n\nMeridian Systems International develops Orion Platform.\n",
        "relations": (
            _relation("stress-identity-01", "org-meridian", "Develops", "system-orion", "Meridian Systems International develops Orion Platform"),
        ),
        "controls": (),
    },
    {
        "episode_id": "stress-identity-acronym",
        "family": "cross_episode_identity",
        "filename": "06-identity-acronym.md",
        "text": "# Identity record B\n\nMSI operates Orion.\n",
        "relations": (
            _relation("stress-identity-02", "org-meridian", "Operates", "system-orion", "MSI operates Orion"),
        ),
        "controls": (),
    },
    {
        "episode_id": "stress-identity-short-name",
        "family": "cross_episode_identity",
        "filename": "07-identity-short-name.md",
        "text": "# Identity record C\n\nMeridian supports the Orion system.\n",
        "relations": (
            _relation("stress-identity-03", "org-meridian", "Supports", "system-orion", "Meridian supports the Orion system"),
        ),
        "controls": (),
    },
    {
        "episode_id": "stress-identity-near-name",
        "family": "cross_episode_identity",
        "filename": "08-identity-near-name.md",
        "text": "# Distinct near-name entities\n\nMeridian System Integration contracts with Meridian Systems International. They are separate legal entities and must not be merged.\n",
        "relations": (
            _relation("stress-identity-04", "org-meridian-integration", "ContractsWith", "org-meridian", "Meridian System Integration contracts with Meridian Systems International"),
        ),
        "controls": (),
    },
    {
        "episode_id": "stress-repeat-canonical",
        "family": "repeated_fact",
        "filename": "09-repeat-canonical.md",
        "text": "# Repeated fact A\n\nHarbor Systems provides Atlas Platform.\n",
        "relations": (
            _relation("stress-repeat-01", "org-harbor", "Provides", "system-atlas", "Harbor Systems provides Atlas Platform"),
        ),
        "controls": (),
    },
    {
        "episode_id": "stress-repeat-alias",
        "family": "repeated_fact",
        "filename": "10-repeat-alias.md",
        "text": "# Repeated fact B\n\nHS provides Atlas.\n",
        "relations": (
            _relation("stress-repeat-02", "org-harbor", "Provides", "system-atlas", "HS provides Atlas"),
        ),
        "controls": (),
    },
    {
        "episode_id": "stress-repeat-paraphrase",
        "family": "repeated_fact",
        "filename": "11-repeat-paraphrase.md",
        "text": "# Repeated fact C\n\nAtlas Platform is provided by Harbor Systems.\n",
        "relations": (
            _relation("stress-repeat-03", "org-harbor", "Provides", "system-atlas", "Atlas Platform is provided by Harbor Systems"),
        ),
        "controls": (),
    },
    {
        "episode_id": "stress-repeat-distinct",
        "family": "repeated_fact",
        "filename": "12-repeat-distinct.md",
        "text": "# Distinct facts on familiar endpoints\n\nHarbor Systems supports Atlas Platform during migration. Harbor Systems supplies interfaces to Northwind Logistics. Supporting Atlas does not establish that Harbor owns Atlas.\n",
        "relations": (
            _relation("stress-repeat-04", "org-harbor", "Supports", "system-atlas", "Harbor Systems supports Atlas Platform during migration"),
            _relation("stress-repeat-05", "org-harbor", "SuppliesTo", "org-northwind", "Harbor Systems supplies interfaces to Northwind Logistics"),
        ),
        "controls": (
            _control("stress-control-06", "org-harbor", "Supports", "system-atlas", "Owns", "Supporting Atlas does not establish that Harbor owns Atlas"),
        ),
    },
)


_IDENTITY_CONTROLS = (
    {
        "control_id": "identity-distinct-01",
        "left_entity_id": "org-meridian",
        "right_entity_id": "org-meridian-integration",
        "expectation": "distinct",
        "reason": "Near-name organizations are explicitly separate legal entities.",
    },
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _specific_signatures() -> set[tuple[str, str, str]]:
    return {
        (source, edge, target)
        for source, target, edges in ENTERPRISE_GRAPH_SCHEMA_PROFILE.edge_type_map_manifest
        if source != "Entity" and target != "Entity"
        for edge in edges
    }


def _readme() -> str:
    return """# Enterprise profile qualification corpus v2

This is a small stress extension to `../enterprise-profile-qualification-v1/`.
Keep v1 as the complete 42-relation atomic calibration baseline. This corpus
adds only 12 documents that concentrate the failure modes hidden by one-edge
documents: overlapping relation families, aliases across episodes, a near-name
non-merge control, and repeated/paraphrased facts.

Only `documents/` is ingested. Gold artifacts use stable `entity_id` values;
names and aliases are observed surfaces, not identity keys. Relation rows may
repeat one semantic triple across episodes. Score unique triples for recall and
report assertion support, entity fragmentation, forbidden entity merges,
fact-level duplicate edges, and endpoint/relation instance excess separately.

This corpus is deliberately small and synthetic. It is a diagnostic stress
slice, not a substitute for the v1 coverage baseline or a real-document audit.
"""


def build(output: Path) -> None:
    if output.exists():
        raise CorpusError(f"output already exists: {output}")
    output.mkdir(parents=True)
    documents = output / "documents"
    documents.mkdir()
    entities_by_id = {row["entity_id"]: dict(row) for row in _ENTITIES}
    relations: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    for episode in _EPISODES:
        path = documents / str(episode["filename"])
        path.write_text(str(episode["text"]), encoding="utf-8")
        episode_relations: list[str] = []
        episode_controls: list[str] = []
        for relation in episode["relations"]:
            row = dict(relation)
            row["episode_id"] = episode["episode_id"]
            row["document_filename"] = episode["filename"]
            row["source_entity"] = entities_by_id[row["source_entity_id"]]["name"]
            row["target_entity"] = entities_by_id[row["target_entity_id"]]["name"]
            relations.append(row)
            episode_relations.append(row["relation_id"])
        for control in episode["controls"]:
            row = dict(control)
            row["episode_id"] = episode["episode_id"]
            row["document_filename"] = episode["filename"]
            row["source_entity"] = entities_by_id[row["source_entity_id"]]["name"]
            row["target_entity"] = entities_by_id[row["target_entity_id"]]["name"]
            controls.append(row)
            episode_controls.append(row["control_id"])
        episodes.append(
            {
                "episode_id": episode["episode_id"],
                "family": episode["family"],
                "document_filename": episode["filename"],
                "artifact_sha256": _sha256(path),
                "relation_ids": episode_relations,
                "control_ids": episode_controls,
            }
        )
    (output / "README.md").write_text(_readme(), encoding="utf-8")
    _write_jsonl(output / "entities.jsonl", _ENTITIES)
    _write_jsonl(output / "relations.jsonl", relations)
    _write_jsonl(output / "negative_controls.jsonl", controls)
    _write_jsonl(output / "identity_controls.jsonl", _IDENTITY_CONTROLS)
    _write_jsonl(output / "episodes.jsonl", episodes)
    unique_relations = {
        (row["source_entity_id"], row["edge_type"], row["target_entity_id"])
        for row in relations
    }
    artifacts = (
        "README.md",
        "entities.jsonl",
        "relations.jsonl",
        "negative_controls.jsonl",
        "identity_controls.jsonl",
        "episodes.jsonl",
    )
    manifest = {
        "schema": SCHEMA,
        "dataset_id": DATASET_ID,
        "synthetic": True,
        "extends_dataset_id": "enterprise-profile-qualification-v1",
        "document_count": len(episodes),
        "entity_count": len(_ENTITIES),
        "relation_assertion_count": len(relations),
        "unique_gold_relation_count": len(unique_relations),
        "relation_type_count": len({row["edge_type"] for row in relations}),
        "negative_control_count": len(controls),
        "identity_control_count": len(_IDENTITY_CONTROLS),
        "family_counts": dict(sorted(Counter(row["family"] for row in episodes).items())),
        "recommended_schema_profile": {
            "key": ENTERPRISE_GRAPH_SCHEMA_PROFILE_KEY,
            "digest": ENTERPRISE_GRAPH_SCHEMA_PROFILE_DIGEST,
            "extractor_version": GRAPH_EXTRACTOR_VERSION,
        },
        "artifacts": {name: _sha256(output / name) for name in artifacts},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate(output: Path) -> dict[str, int]:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("dataset_id") != DATASET_ID:
        raise CorpusError("enterprise stress identity is invalid")
    profile = manifest.get("recommended_schema_profile", {})
    if profile != {
        "key": ENTERPRISE_GRAPH_SCHEMA_PROFILE_KEY,
        "digest": ENTERPRISE_GRAPH_SCHEMA_PROFILE_DIGEST,
        "extractor_version": GRAPH_EXTRACTOR_VERSION,
    }:
        raise CorpusError("enterprise stress profile identity is invalid")
    entities = _rows(output / "entities.jsonl")
    relations = _rows(output / "relations.jsonl")
    controls = _rows(output / "negative_controls.jsonl")
    identity_controls = _rows(output / "identity_controls.jsonl")
    episodes = _rows(output / "episodes.jsonl")
    entities_by_id = {str(row.get("entity_id")): row for row in entities}
    if len(entities_by_id) != len(entities) or any(
        re.fullmatch(r"[a-z][a-z0-9-]{2,63}", entity_id) is None
        for entity_id in entities_by_id
    ):
        raise CorpusError("enterprise stress entity IDs are invalid")
    surfaces: dict[str, str] = {}
    for entity_id, entity in entities_by_id.items():
        aliases = entity.get("aliases")
        if not entity.get("name") or not isinstance(aliases, list) or not aliases:
            raise CorpusError("enterprise stress entity surfaces are invalid")
        for value in (entity["name"], *aliases):
            surface = " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))
            previous = surfaces.setdefault(surface, entity_id)
            if previous != entity_id:
                raise CorpusError("enterprise stress entity surface is ambiguous")
    signatures = _specific_signatures()
    edge_names = {edge.name for edge in ENTERPRISE_GRAPH_SCHEMA_PROFILE.edge_manifest}
    relation_ids: set[str] = set()
    used_entities: set[str] = set()
    episode_by_id = {str(row.get("episode_id")): row for row in episodes}
    if len(episode_by_id) != len(episodes):
        raise CorpusError("enterprise stress episode IDs are invalid")
    for relation in relations:
        relation_id = str(relation.get("relation_id", ""))
        source_id = str(relation.get("source_entity_id", ""))
        target_id = str(relation.get("target_entity_id", ""))
        if not relation_id or relation_id in relation_ids:
            raise CorpusError("enterprise stress relation IDs are invalid")
        relation_ids.add(relation_id)
        if source_id not in entities_by_id or target_id not in entities_by_id:
            raise CorpusError(f"enterprise stress relation endpoint is invalid: {relation_id}")
        used_entities.update((source_id, target_id))
        signature = (
            str(entities_by_id[source_id]["entity_type"]),
            str(relation.get("edge_type")),
            str(entities_by_id[target_id]["entity_type"]),
        )
        if signature not in signatures:
            raise CorpusError(f"enterprise stress relation signature is invalid: {relation_id}")
        episode = episode_by_id.get(str(relation.get("episode_id")))
        if episode is None or relation_id not in episode.get("relation_ids", []):
            raise CorpusError(f"enterprise stress relation episode is invalid: {relation_id}")
        text = (output / "documents" / str(relation["document_filename"])).read_text(
            encoding="utf-8"
        )
        if str(relation.get("evidence_text", "")) not in text:
            raise CorpusError(f"enterprise stress relation evidence is invalid: {relation_id}")
    control_ids: set[str] = set()
    for control in controls:
        control_id = str(control.get("control_id", ""))
        source_id = str(control.get("source_entity_id", ""))
        target_id = str(control.get("target_entity_id", ""))
        if not control_id or control_id in control_ids:
            raise CorpusError("enterprise stress control IDs are invalid")
        control_ids.add(control_id)
        if source_id not in entities_by_id or target_id not in entities_by_id:
            raise CorpusError(f"enterprise stress control endpoint is invalid: {control_id}")
        asserted_signature = (
            str(entities_by_id[source_id]["entity_type"]),
            str(control.get("asserted_edge")),
            str(entities_by_id[target_id]["entity_type"]),
        )
        if asserted_signature not in signatures or control.get("forbidden_edge") not in edge_names:
            raise CorpusError(f"enterprise stress control relation is invalid: {control_id}")
        episode = episode_by_id.get(str(control.get("episode_id")))
        if episode is None or control_id not in episode.get("control_ids", []):
            raise CorpusError(f"enterprise stress control episode is invalid: {control_id}")
        text = (output / "documents" / str(control["document_filename"])).read_text(
            encoding="utf-8"
        )
        if str(control.get("evidence_text", "")) not in text:
            raise CorpusError(f"enterprise stress control evidence is invalid: {control_id}")
    for control in identity_controls:
        left = str(control.get("left_entity_id", ""))
        right = str(control.get("right_entity_id", ""))
        if (
            control.get("expectation") != "distinct"
            or left == right
            or left not in entities_by_id
            or right not in entities_by_id
        ):
            raise CorpusError("enterprise stress identity control is invalid")
    if used_entities != set(entities_by_id):
        raise CorpusError("enterprise stress contains unused gold entities")
    for episode in episodes:
        path = output / "documents" / str(episode.get("document_filename"))
        if not path.is_file() or _sha256(path) != episode.get("artifact_sha256"):
            raise CorpusError(f"enterprise stress episode artifact is invalid: {episode.get('episode_id')}")
    for name, digest in manifest.get("artifacts", {}).items():
        if _sha256(output / name) != digest:
            raise CorpusError(f"enterprise stress artifact digest is invalid: {name}")
    unique_relations = {
        (row["source_entity_id"], row["edge_type"], row["target_entity_id"])
        for row in relations
    }
    expected_counts = {
        "document_count": len(episodes),
        "entity_count": len(entities),
        "relation_assertion_count": len(relations),
        "unique_gold_relation_count": len(unique_relations),
        "negative_control_count": len(controls),
        "identity_control_count": len(identity_controls),
    }
    if any(manifest.get(key) != value for key, value in expected_counts.items()):
        raise CorpusError("enterprise stress manifest counts are invalid")
    return {
        "documents": len(episodes),
        "relation_assertions": len(relations),
        "unique_relations": len(unique_relations),
        "controls": len(controls),
        "identity_controls": len(identity_controls),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    if arguments.command == "build":
        build(arguments.output)
        print(f"built {arguments.output}")
    else:
        print(json.dumps(validate(arguments.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

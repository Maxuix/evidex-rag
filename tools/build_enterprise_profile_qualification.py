#!/usr/bin/env python3
"""Build a complete gold calibration corpus for ``enterprise_knowledge_v1``."""

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
DEFAULT_OUTPUT = ROOT / "evaluation/enterprise-profile-qualification-v1"
SCHEMA = "enterprise_profile_qualification_v1"
DATASET_ID = "enterprise-profile-qualification-v1"
VARIANTS_PER_EDGE = 3

_NAME_MODIFIERS = (
    "Amber", "Blue", "Cedar", "Delta", "Elm", "Falcon", "Golden", "Harbor",
    "Indigo", "Juniper", "Keystone", "Lunar", "Maple", "Nimbus", "Orchid",
    "Pioneer", "Quartz", "Redwood", "Silver", "Tidal", "Union", "Violet",
    "Willow", "Zenith",
)
_NAME_STEMS = (
    "Atlas", "Beacon", "Canyon", "Drift", "Evergreen", "Forge", "Grove",
    "Horizon", "Isle", "Junction", "Kite", "Lantern", "Meadow", "Northstar",
    "Oasis", "Prairie", "Quarry", "Ridge", "Summit", "Terrace", "Vale",
    "Watershed", "Yard", "Zephyr",
)
_ENTITY_SUFFIXES = {
    "Organization": "Group",
    "OrganizationalUnit": "Division",
    "Role": "Officer",
    "Policy": "Standard",
    "Process": "Workflow",
    "BusinessSystem": "Platform",
    "Product": "Suite",
    "Project": "Program",
    "Document": "Handbook",
    "Location": "District",
    "Facility": "Center",
    "BusinessTerm": "Classification",
}

WORDS = {
    "PartOf": "is part of",
    "MemberOf": "is a member of",
    "ReportsTo": "reports to",
    "HoldsRole": "holds the role of",
    "ServesAs": "serves as a position at",
    "Leads": "leads",
    "Owns": "owns",
    "Controls": "controls",
    "Establishes": "established",
    "Acquires": "acquired",
    "InvestsIn": "invested in",
    "MergesInto": "merged into",
    "ResponsibleFor": "is responsible for",
    "AccountableFor": "is accountable for",
    "ConsultedOn": "is consulted on",
    "InformedAbout": "must be informed about",
    "Approves": "approves",
    "Sponsors": "sponsors",
    "Appoints": "appointed",
    "Operates": "operates",
    "Uses": "uses",
    "Provides": "provides",
    "Develops": "develops",
    "Builds": "builds",
    "SuppliesTo": "supplies equipment to",
    "ContractsWith": "has a contract with",
    "PartnersWith": "partners with",
    "IndependentOf": "is independent of",
    "DependsOn": "depends on",
    "GovernedBy": "is governed by",
    "AppliesTo": "applies to",
    "Documents": "documents",
    "Defines": "defines",
    "Produces": "produces",
    "Consumes": "consumes",
    "Supports": "supports",
    "LocatedAt": "is located at",
    "DeployedAt": "is deployed at",
    "CertifiedBy": "is certified by",
    "LicensedBy": "is licensed by",
    "Supersedes": "supersedes",
    "Delivers": "delivers",
}

CONTROL_PAIRS = (
    ("InvestsIn", "Controls"),
    ("InvestsIn", "Owns"),
    ("ContractsWith", "PartnersWith"),
    ("MemberOf", "ReportsTo"),
    ("Documents", "AccountableFor"),
    ("SuppliesTo", "Owns"),
    ("Approves", "ResponsibleFor"),
    ("Uses", "Owns"),
    ("CertifiedBy", "LicensedBy"),
    ("PartOf", "Controls"),
    ("ServesAs", "HoldsRole"),
    ("Supports", "Delivers"),
)


class CorpusError(RuntimeError):
    """Raised when this code-owned calibration corpus is invalid."""


class _EntityNameAllocator:
    """Allocate distinct, human-readable entities without numeric suffixes."""

    def __init__(self) -> None:
        self._index = 0

    def allocate(self, entity_type: str) -> dict[str, Any]:
        capacity = len(_NAME_MODIFIERS) * len(_NAME_STEMS)
        if self._index >= capacity:
            raise CorpusError("enterprise calibration entity-name capacity exhausted")
        modifier = _NAME_MODIFIERS[self._index // len(_NAME_STEMS)]
        stem = _NAME_STEMS[self._index % len(_NAME_STEMS)]
        self._index += 1
        alias = f"{modifier} {stem}"
        if entity_type == "Person":
            canonical = alias
            alias = f"{modifier[0]}. {stem}"
        else:
            suffix = _ENTITY_SUFFIXES.get(entity_type)
            if suffix is None:
                raise CorpusError(f"unknown enterprise entity type: {entity_type}")
            canonical = f"{alias} {suffix}"
        return {"name": canonical, "entity_type": entity_type, "aliases": [alias]}


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


def _typed_pairs() -> dict[str, tuple[str, str]]:
    pairs: dict[str, tuple[str, str]] = {}
    for source, target, edge_names in ENTERPRISE_GRAPH_SCHEMA_PROFILE.edge_type_map_manifest:
        if source == "Entity" or target == "Entity":
            continue
        for edge in edge_names:
            pairs.setdefault(edge, (source, target))
    expected = {edge.name for edge in ENTERPRISE_GRAPH_SCHEMA_PROFILE.edge_manifest}
    if set(pairs) != expected or set(WORDS) != expected:
        raise CorpusError("enterprise edge vocabulary changed; update qualification corpus")
    return pairs


def _relation_text(
    edge: str,
    source_type: str,
    target_type: str,
    variant: int,
    allocator: _EntityNameAllocator,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    subject = allocator.allocate(source_type)
    target = allocator.allocate(target_type)
    qualifier = (
        " effective from 2026-01-01 under the recorded operating scope"
        if variant == 3
        else ""
    )
    subject_surface = str(subject["name"])
    target_surface = str(target["name"])
    if variant == 2:
        subject_surface += f" (also known as {subject['aliases'][0]})"
        target_surface += f" (also known as {target['aliases'][0]})"
    text = (
        f"The {source_type} named {subject_surface} {WORDS[edge]} the "
        f"{target_type} named {target_surface}{qualifier}."
    )
    return subject, target, text


def _surface_ngrams(value: str) -> set[str]:
    normalized = " ".join(re.findall(r"[a-z0-9]+", value.casefold()))
    if len(normalized) < 3:
        return {normalized} if normalized else set()
    return {normalized[index : index + 3] for index in range(len(normalized) - 2)}


def _surface_similarity(left: str, right: str) -> float:
    left_grams = _surface_ngrams(left)
    right_grams = _surface_ngrams(right)
    union = left_grams | right_grams
    return len(left_grams & right_grams) / len(union) if union else 0.0


def _readme() -> str:
    return """# Enterprise profile qualification corpus v1

This controlled calibration corpus gives every one of the 42 relations in
`enterprise_knowledge_v1` three explicit, typed positive examples (126 total)
and 12 explicit non-entailment controls for known semantic confusions.

It is intentionally separate from the external-realism slice in
`../public-rag-benchmark-suite-v1/`: this corpus measures typed extraction
coverage and false-positive boundaries; the external slice measures messy
enterprise retrieval, conflict disclosure, and closed-world refusal.  Neither
score may be reported as a substitute for the other.

Only `documents/` is ingested. `relations.jsonl`, `entities.jsonl`,
`cases.jsonl`, and `negative_controls.jsonl` are evaluator-only gold files.
Every entity has one explicit alias. Distinct entities are deliberately named
below Graphiti's fuzzy duplicate threshold, while each alias remains an
evaluator-declared equivalent surface for the same canonical entity.

## Required real qualification

Run exactly the configured OpenCode Go `mimo-v2.5` graph extraction over this
corpus using `enterprise_knowledge_v1`, then compare normalized output edges
with `relations.jsonl` by `(source entity, edge type, target entity)`. Report
micro/macro precision, recall, F1, every relation's support, canonical endpoint
resolution, conservative fact-level duplicates, and every `forbidden_edge`
hit. Do not calculate a relation with zero attempted gold examples as 100%.

The 42 direct answer cases test evidence-backed final answering after
extraction.  Use `../graph-rag-v1/` for 1–3-hop final-answer retrieval, and
the public suite for real enterprise conflicts and absence controls.
"""


def build(output: Path) -> None:
    if output.exists():
        raise CorpusError(f"output already exists: {output}")
    pairs = _typed_pairs()
    output.mkdir(parents=True)
    document_dir = output / "documents"
    document_dir.mkdir()
    allocator = _EntityNameAllocator()
    relations: list[dict[str, Any]] = []
    entities: dict[tuple[str, str], dict[str, Any]] = {}
    cases: list[dict[str, Any]] = []
    for edge_ordinal, edge in enumerate(sorted(pairs), start=1):
        source_type, target_type = pairs[edge]
        first_relation_id = ""
        first_subject = ""
        first_target = ""
        for variant in range(1, VARIANTS_PER_EDGE + 1):
            relation_id = f"edge-{edge_ordinal:02d}-{variant}"
            subject, target, text = _relation_text(
                edge, source_type, target_type, variant, allocator
            )
            filename = f"{relation_id}-{edge.casefold()}.md"
            path = document_dir / filename
            path.write_text(f"# Enterprise record {relation_id}\n\n{text}\n", encoding="utf-8")
            relations.append(
                {
                    "relation_id": relation_id,
                    "source_entity": subject["name"],
                    "source_type": source_type,
                    "edge_type": edge,
                    "target_entity": target["name"],
                    "target_type": target_type,
                    "document_filename": filename,
                    "explicit_text": text,
                    "artifact_sha256": _sha256(path),
                }
            )
            entities[(str(subject["name"]), source_type)] = subject
            entities[(str(target["name"]), target_type)] = target
            if variant == 1:
                first_relation_id = relation_id
                first_subject = str(subject["name"])
                first_target = str(target["name"])
        cases.append(
            {
                "case_id": f"answer-{edge.casefold()}",
                "question": (
                    f"Which named target is explicitly linked from {first_subject} "
                    f"by the {edge} relation?"
                ),
                "expected_answer": first_target,
                "required_relation_ids": [first_relation_id],
                "required_hops": 1,
                "edge_type": edge,
            }
        )
    controls: list[dict[str, Any]] = []
    for ordinal, (asserted, forbidden) in enumerate(CONTROL_PAIRS, start=1):
        source_type, target_type = pairs[asserted]
        subject, target, text = _relation_text(
            asserted, source_type, target_type, 100 + ordinal, allocator
        )
        text += (
            f" This statement does not establish that {subject['name']} "
            f"{WORDS[forbidden]} {target['name']}."
        )
        filename = f"control-{ordinal:02d}-{asserted.casefold()}-not-{forbidden.casefold()}.md"
        path = document_dir / filename
        path.write_text(f"# Enterprise boundary control {ordinal}\n\n{text}\n", encoding="utf-8")
        controls.append(
            {
                "control_id": f"control-{ordinal:02d}",
                "source_entity": subject["name"],
                "source_type": source_type,
                "asserted_edge": asserted,
                "target_entity": target["name"],
                "target_type": target_type,
                "forbidden_edge": forbidden,
                "document_filename": filename,
                "explicit_text": text,
                "artifact_sha256": _sha256(path),
            }
        )
        entities[(str(subject["name"]), source_type)] = subject
        entities[(str(target["name"]), target_type)] = target
    _write_jsonl(output / "relations.jsonl", relations)
    _write_jsonl(output / "entities.jsonl", sorted(entities.values(), key=lambda item: (item["entity_type"], item["name"])))
    _write_jsonl(output / "cases.jsonl", cases)
    _write_jsonl(output / "negative_controls.jsonl", controls)
    (output / "README.md").write_text(_readme(), encoding="utf-8")
    manifest = {
        "schema": SCHEMA,
        "dataset_id": DATASET_ID,
        "language": "en",
        "synthetic": True,
        "recommended_schema_profile": {
            "key": ENTERPRISE_GRAPH_SCHEMA_PROFILE_KEY,
            "digest": ENTERPRISE_GRAPH_SCHEMA_PROFILE_DIGEST,
            "extractor_version": GRAPH_EXTRACTOR_VERSION,
        },
        "relation_type_count": len(pairs),
        "positive_relation_count": len(relations),
        "positive_examples_per_relation": VARIANTS_PER_EDGE,
        "negative_control_count": len(controls),
        "gold_entity_count": len(entities),
        "answer_case_count": len(cases),
        "artifacts": {},
    }
    manifest["artifacts"] = {
        name: _sha256(output / name)
        for name in ("README.md", "relations.jsonl", "entities.jsonl", "cases.jsonl", "negative_controls.jsonl")
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validate(output)


def validate(output: Path) -> dict[str, int]:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("dataset_id") != DATASET_ID:
        raise CorpusError("manifest identity is invalid")
    identity = {
        "key": ENTERPRISE_GRAPH_SCHEMA_PROFILE_KEY,
        "digest": ENTERPRISE_GRAPH_SCHEMA_PROFILE_DIGEST,
        "extractor_version": GRAPH_EXTRACTOR_VERSION,
    }
    if manifest.get("recommended_schema_profile") != identity:
        raise CorpusError("profile identity is invalid")
    pairs = _typed_pairs()
    relations = _rows(output / "relations.jsonl")
    controls = _rows(output / "negative_controls.jsonl")
    cases = _rows(output / "cases.jsonl")
    entities = _rows(output / "entities.jsonl")
    counts = Counter(row.get("edge_type") for row in relations)
    if set(counts) != set(pairs) or any(count != VARIANTS_PER_EDGE for count in counts.values()):
        raise CorpusError("relation type coverage is incomplete")
    if len(relations) != len(pairs) * VARIANTS_PER_EDGE or len(cases) != len(pairs):
        raise CorpusError("positive count is invalid")
    if len(controls) != len(CONTROL_PAIRS):
        raise CorpusError("negative control count is invalid")
    canonical_names = {str(row.get("name", "")) for row in entities}
    expected_names = {
        str(row[field])
        for row in (*relations, *controls)
        for field in ("source_entity", "target_entity")
    }
    if canonical_names != expected_names or manifest.get("gold_entity_count") != len(entities):
        raise CorpusError("gold entity coverage is incomplete")
    surfaces: list[tuple[str, str]] = []
    for row in entities:
        name = str(row.get("name", ""))
        aliases = row.get("aliases")
        if not name or not isinstance(aliases, list) or len(aliases) != 1:
            raise CorpusError("gold entity aliases are invalid")
        if re.search(r"\d+$", name):
            raise CorpusError("gold entity name uses a numeric discriminator")
        surfaces.append((name, name))
        surfaces.append((name, str(aliases[0])))
    normalized_surfaces: dict[str, str] = {}
    for canonical, surface in surfaces:
        normalized = " ".join(re.findall(r"[a-z0-9]+", surface.casefold()))
        previous = normalized_surfaces.setdefault(normalized, canonical)
        if previous != canonical:
            raise CorpusError("gold entity surface is ambiguous")
    for index, (left_canonical, left_surface) in enumerate(surfaces):
        for right_canonical, right_surface in surfaces[index + 1 :]:
            if left_canonical == right_canonical:
                continue
            if _surface_similarity(left_surface, right_surface) >= 0.9:
                raise CorpusError("distinct gold entities exceed fuzzy duplicate threshold")
    for row in relations:
        source_type, target_type = pairs[str(row["edge_type"])]
        if (row.get("source_type"), row.get("target_type")) != (source_type, target_type):
            raise CorpusError(f"typed pair invalid: {row.get('relation_id')}")
        path = output / "documents" / str(row.get("document_filename"))
        if not path.is_file() or _sha256(path) != row.get("artifact_sha256"):
            raise CorpusError(f"relation document invalid: {row.get('relation_id')}")
    for row in controls:
        if row.get("asserted_edge") not in pairs or row.get("forbidden_edge") not in pairs:
            raise CorpusError(f"control edge invalid: {row.get('control_id')}")
        path = output / "documents" / str(row.get("document_filename"))
        if not path.is_file() or _sha256(path) != row.get("artifact_sha256"):
            raise CorpusError(f"control document invalid: {row.get('control_id')}")
    for name, digest in manifest.get("artifacts", {}).items():
        if _sha256(output / name) != digest:
            raise CorpusError(f"artifact digest invalid: {name}")
    return {"relation_types": len(pairs), "positive_relations": len(relations), "controls": len(controls)}


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

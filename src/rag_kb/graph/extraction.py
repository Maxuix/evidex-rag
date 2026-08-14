"""Fixed, locatable entity/relation extraction protocol."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from rag_kb.domain import (
    GRAPH_MAX_ENTITIES,
    GRAPH_MAX_RELATIONS,
    GRAPH_RELATION_GROUNDING_CODES,
    GraphAdmissionStats,
    GraphChunkExtraction,
    GraphChunkResultStatus,
    GraphEntityMention,
    GraphEntityType,
    GraphProtocolError,
    GraphRelationAssertion,
    GraphResourceLimitError,
    entity_key,
    first_grounded_span,
    graph_extraction_hash,
    normalize_entity_surface,
    normalize_predicate,
)


GRAPH_MAX_CHUNK_CHARS = 32_000
GRAPH_MAX_RESPONSE_BYTES = 64 * 1024
_ID = Annotated[str, Field(min_length=1, max_length=64)]
_ROOT_KEYS = frozenset({"entities", "relations"})
_REPAIRABLE_ROOT_CODES = frozenset(
    {
        "json_invalid",
        "response_not_text",
        "root_not_object",
        "schema_unexpected_key",
        "schema_missing",
        "schema_type",
    }
)
_PROTOCOL_ERROR_FAMILIES = {
    "schema_unexpected_key": "schema_invalid",
    "schema_missing": "schema_invalid",
    "schema_type": "schema_invalid",
    "schema_entity_enum": "schema_invalid",
    "schema_id_format": "schema_invalid",
    "schema_length": "schema_invalid",
    "schema_null_contract": "schema_invalid",
    "entity_surface_not_locatable": "support_text_not_locatable",
    "disambiguator_support_not_locatable": "support_text_not_locatable",
    "relation_support_not_locatable": "support_text_not_locatable",
    "relation_support_missing_subject": "relation_support_missing_endpoint",
    "relation_support_missing_object": "relation_support_missing_endpoint",
    "relation_support_missing_both": "relation_support_missing_endpoint",
}


class _GraphSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GraphExtractionEntity(_GraphSchema):
    id: _ID
    type: GraphEntityType
    surface: Annotated[str, Field(min_length=1, max_length=512)]
    disambiguator: Annotated[str, Field(min_length=1, max_length=512)] | None
    disambiguator_support: Annotated[str, Field(min_length=1, max_length=1024)] | None

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not value.replace("_", "").replace("-", "").isalnum():
            raise ValueError("entity id contains unsupported characters")
        return value


class GraphExtractionRelation(_GraphSchema):
    subject: _ID
    predicate: Annotated[str, Field(min_length=1, max_length=256)]
    object: _ID
    support: Annotated[str, Field(min_length=1, max_length=2048)]


class GraphExtractionPayload(_GraphSchema):
    entities: list[GraphExtractionEntity] = Field(max_length=GRAPH_MAX_ENTITIES)
    relations: list[GraphExtractionRelation] = Field(max_length=GRAPH_MAX_RELATIONS)


def graph_protocol_error_family(code: str) -> str:
    """Map a content-safe detail code to its stable historical family."""

    return _PROTOCOL_ERROR_FAMILIES.get(code, code)


def graph_protocol_error_is_repairable(code: str) -> bool:
    """Return whether a protocol error is a root-level failure worth one model repair."""

    return code in _REPAIRABLE_ROOT_CODES


def _schema_error_code(error: ValidationError) -> str:
    """Classify Pydantic errors without reading values, messages, or contexts."""

    details = error.errors(include_url=False, include_context=False, include_input=False)
    kinds = {str(detail.get("type", "")) for detail in details}
    locations = {tuple(detail.get("loc", ())) for detail in details}
    if "extra_forbidden" in kinds:
        return "schema_unexpected_key"
    if "missing" in kinds:
        return "schema_missing"
    length_kinds = {"too_short", "too_long", "string_too_short", "string_too_long"}
    if any(kind in length_kinds for kind in kinds):
        return "schema_length"
    if "value_error" in kinds and any(location[-1:] == ("id",) for location in locations):
        return "schema_id_format"
    if "enum" in kinds and any(location[-1:] == ("type",) for location in locations):
        return "schema_entity_enum"
    return "schema_type"


def parse_graph_extraction(
    payload: str | bytes | bytearray | Mapping[str, Any],
    text: str,
) -> GraphChunkExtraction:
    """Parse and ground one response, keeping only independently valid items."""

    if not isinstance(text, str) or not text.strip():
        raise GraphProtocolError("chunk_text_empty")
    if len(text) > GRAPH_MAX_CHUNK_CHARS:
        raise GraphResourceLimitError("input_chars")

    raw: Any
    if isinstance(payload, Mapping):
        raw = dict(payload)
    else:
        try:
            raw_bytes = bytes(payload, "utf-8") if isinstance(payload, str) else bytes(payload)
        except (TypeError, ValueError) as error:
            raise GraphProtocolError("response_not_text") from error
        if len(raw_bytes) > GRAPH_MAX_RESPONSE_BYTES:
            raise GraphResourceLimitError("output_bytes")
        try:
            raw = json.loads(raw_bytes)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise GraphProtocolError("json_invalid") from error

    if not isinstance(raw, dict):
        raise GraphProtocolError("root_not_object")
    entities_raw, relations_raw = _root_arrays(raw)
    attempted = bool(entities_raw or relations_raw)
    last_rejection: str | None = None
    entity_drops: Counter[str] = Counter()
    relation_drops: Counter[str] = Counter()
    mentions: list[GraphEntityMention] = []
    by_id: dict[str, GraphEntityMention] = {}
    for ordinal, raw_entity in enumerate(entities_raw):
        try:
            item = GraphExtractionEntity.model_validate(raw_entity)
        except ValidationError as error:
            last_rejection = _schema_error_code(error)
            entity_drops[last_rejection] += 1
            continue
        if item.id in by_id:
            last_rejection = "duplicate_entity_id"
            entity_drops[last_rejection] += 1
            continue
        try:
            mention = _materialize_entity(item, text, ordinal)
        except GraphProtocolError as error:
            last_rejection = error.code
            entity_drops[last_rejection] += 1
            continue
        mentions.append(mention)
        by_id[item.id] = mention

    relation_by_identity: dict[tuple[str, str, str], GraphRelationAssertion] = {}
    for ordinal, raw_relation in enumerate(relations_raw):
        try:
            item = GraphExtractionRelation.model_validate(raw_relation)
        except ValidationError as error:
            last_rejection = _schema_error_code(error)
            relation_drops[last_rejection] += 1
            continue
        try:
            relation = _materialize_relation(item, text, by_id, ordinal)
        except GraphProtocolError as error:
            last_rejection = error.code
            relation_drops[last_rejection] += 1
            continue
        identity = (
            relation.subject_entity_key,
            relation.object_entity_key,
            relation.normalized_predicate,
        )
        previous = relation_by_identity.get(identity)
        if previous is None or (
            relation.support_start,
            relation.ordinal,
        ) < (previous.support_start, previous.ordinal):
            relation_by_identity[identity] = relation

    relations = sorted(
        relation_by_identity.values(), key=lambda item: (item.support_start, item.ordinal)
    )
    admission = _admission_stats(entity_drops, relation_drops)
    if mentions:
        mention_tuple = tuple(mentions)
        relation_tuple = tuple(relations)
        return GraphChunkExtraction(
            result_status=GraphChunkResultStatus.EXTRACTED,
            mentions=mention_tuple,
            relations=relation_tuple,
            result_hash=graph_extraction_hash(mention_tuple, relation_tuple),
            admission=admission,
        )
    if not attempted:
        return GraphChunkExtraction(
            result_status=GraphChunkResultStatus.EMPTY,
            admission=admission,
        )
    raise GraphProtocolError(
        last_rejection or "no_admitted_items",
        admission=admission,
    )


def _admission_stats(
    entity_drops: Counter[str],
    relation_drops: Counter[str],
) -> GraphAdmissionStats:
    return GraphAdmissionStats(
        dropped_entity_count=sum(entity_drops.values()),
        dropped_relation_count=sum(relation_drops.values()),
        dropped_relation_grounding_count=sum(
            relation_drops[code] for code in GRAPH_RELATION_GROUNDING_CODES
        ),
        entity_drop_codes=tuple(sorted(entity_drops.items())),
        relation_drop_codes=tuple(sorted(relation_drops.items())),
    )


def _root_arrays(raw: dict[str, Any]) -> tuple[list[Any], list[Any]]:
    if set(raw) - _ROOT_KEYS:
        raise GraphProtocolError("schema_unexpected_key")
    if "entities" not in raw or "relations" not in raw:
        raise GraphProtocolError("schema_missing")
    entities = raw["entities"]
    relations = raw["relations"]
    if not isinstance(entities, list) or not isinstance(relations, list):
        raise GraphProtocolError("schema_type")
    if len(entities) > GRAPH_MAX_ENTITIES:
        raise GraphResourceLimitError("entities_limit")
    if len(relations) > GRAPH_MAX_RELATIONS:
        raise GraphResourceLimitError("relations_limit")
    return entities, relations


def _materialize_entity(
    item: GraphExtractionEntity,
    text: str,
    ordinal: int,
) -> GraphEntityMention:
    try:
        try:
            surface_start, surface_end = first_grounded_span(text, item.surface)
        except GraphProtocolError as error:
            raise GraphProtocolError("entity_surface_not_locatable") from error
        materialized_surface = text[surface_start:surface_end]
        normalized_surface = normalize_entity_surface(materialized_surface)
        if normalized_surface != normalize_entity_surface(item.surface):
            raise GraphProtocolError("entity_surface_identity_mismatch")
        if item.disambiguator is None:
            if item.disambiguator_support is not None:
                raise GraphProtocolError("schema_null_contract")
            support_start = support_end = None
        else:
            if item.disambiguator_support is None:
                raise GraphProtocolError("schema_null_contract")
            try:
                support_start, support_end = first_grounded_span(
                    text, item.disambiguator_support
                )
            except GraphProtocolError as error:
                raise GraphProtocolError("disambiguator_support_not_locatable") from error
            if normalize_entity_surface(item.disambiguator) not in normalize_entity_surface(
                item.disambiguator_support
            ):
                raise GraphProtocolError("disambiguator_not_supported")
        return GraphEntityMention(
            mention_id=item.id,
            ordinal=ordinal,
            entity_type=item.type,
            surface=materialized_surface,
            normalized_surface=normalized_surface,
            disambiguator=item.disambiguator,
            disambiguator_support_start=support_start,
            disambiguator_support_end=support_end,
            surface_start=surface_start,
            surface_end=surface_end,
            entity_key=entity_key(item.type, materialized_surface, item.disambiguator),
        )
    except GraphProtocolError:
        raise
    except (TypeError, ValueError) as error:
        raise GraphProtocolError("entity_grounding_invalid") from error


def _materialize_relation(
    item: GraphExtractionRelation,
    text: str,
    by_id: Mapping[str, GraphEntityMention],
    ordinal: int,
) -> GraphRelationAssertion:
    subject = by_id.get(item.subject)
    object_ = by_id.get(item.object)
    if subject is None or object_ is None:
        raise GraphProtocolError("relation_endpoint_unknown")
    if subject.entity_key == object_.entity_key:
        raise GraphProtocolError("self_relation")
    try:
        normalized_predicate = normalize_predicate(item.predicate)
        try:
            support_start, support_end = first_grounded_span(text, item.support)
        except GraphProtocolError as error:
            raise GraphProtocolError("relation_support_not_locatable") from error
    except GraphProtocolError:
        raise
    except (TypeError, ValueError) as error:
        raise GraphProtocolError("relation_grounding_invalid") from error
    materialized_support = text[support_start:support_end]
    subject_missing = object_missing = False
    try:
        first_grounded_span(materialized_support, subject.surface)
    except GraphProtocolError:
        subject_missing = True
    try:
        first_grounded_span(materialized_support, object_.surface)
    except GraphProtocolError:
        object_missing = True
    if subject_missing or object_missing:
        suffix = "both" if subject_missing and object_missing else (
            "subject" if subject_missing else "object"
        )
        raise GraphProtocolError(f"relation_support_missing_{suffix}")
    return GraphRelationAssertion(
        relation_id=f"r-{ordinal}",
        ordinal=ordinal,
        subject_mention_id=subject.mention_id,
        object_mention_id=object_.mention_id,
        subject_entity_key=subject.entity_key,
        object_entity_key=object_.entity_key,
        predicate=item.predicate,
        normalized_predicate=normalized_predicate,
        support_start=support_start,
        support_end=support_end,
    )

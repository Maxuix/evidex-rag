"""Validated, content-limited observations of a single chat attempt.

These values describe execution; they never control it or contain model reasoning.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import re
from typing import Any, Literal
from uuid import UUID


CHAT_ACTIVITY_VERSION = "chat_activity_v1"
CHAT_ACTIVITY_ARTIFACT = "chat_activity"
MAX_ACTIVITY_STEPS = 1024
MAX_ACTIVITY_BYTES = 1_048_576
ACTIVITY_TOOLS = frozenset({
    "semantic_search", "keyword_search", "read_chunk_context", "list_documents",
    "search_graph_relations", "calculate", "unknown",
})
ActivityStatus = Literal["pending", "running", "processing", "succeeded", "failed", "rejected", "cancelled"]
ACTIVE_STATUSES = frozenset({"pending", "running", "processing"})


def _integer(value: object, minimum: int = 0) -> None:
    if type(value) is not int or not minimum <= value <= 2**53 - 1:
        raise ValueError("invalid activity integer")


def _text(value: object, maximum: int) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError("invalid activity text")


def _shape(value: object, cls: type) -> dict[str, Any]:
    if isinstance(value, dict) and cls is ActivityStep:
        value = {"scope_results": (), **value}
    if isinstance(value, dict) and cls is ActivitySource:
        value = {"knowledge_base_id": None, "knowledge_base_name": None, "index_revision_id": None, **value}
    if not isinstance(value, dict) or set(value) != {f.name for f in fields(cls)}:
        raise ValueError("invalid activity fields")
    return dict(value)


@dataclass(frozen=True, slots=True)
class ActivitySource:
    document_id: str
    document_version_id: str
    title: str
    index_chunk_id: str | None = None
    ref: str | None = None
    location: str | None = None
    knowledge_base_id: str | None = None
    knowledge_base_name: str | None = None
    index_revision_id: str | None = None

    def __post_init__(self) -> None:
        for value in (self.document_id, self.document_version_id):
            _text(value, 36)
            UUID(value)
        for value in (self.knowledge_base_id, self.index_revision_id):
            if value is not None:
                _text(value, 36)
                UUID(value)
        if self.knowledge_base_name is not None:
            _text(self.knowledge_base_name, 256)
        if self.index_chunk_id is not None:
            _text(self.index_chunk_id, 36)
            UUID(self.index_chunk_id)
        _text(self.title, 512)
        if self.ref is not None and not re.fullmatch(r"ev_[1-9][0-9]*", self.ref):
            raise ValueError("invalid activity evidence ref")
        if self.location is not None:
            _text(self.location, 256)


@dataclass(frozen=True, slots=True)
class ActivityScope:
    knowledge_base_id: str
    name: str
    status: str
    query: str | None = None
    retrieved_count: int | None = None
    admitted_count: int | None = None
    displayed_count: int | None = None
    omitted_count: int | None = None

    def __post_init__(self) -> None:
        _text(self.knowledge_base_id, 36)
        UUID(self.knowledge_base_id)
        _text(self.name, 255)
        _text(self.status, 80)
        if not re.fullmatch(r"[A-Za-z0-9_]+", self.status):
            raise ValueError("invalid activity scope status")
        if self.query is not None:
            _text(self.query, 2048)
        for value in (self.retrieved_count, self.admitted_count, self.displayed_count, self.omitted_count):
            if value is not None:
                _integer(value)


@dataclass(frozen=True, slots=True)
class ActivityStep:
    step_id: str
    ordinal: int
    seq: int
    kind: Literal["model", "tool", "system"]
    name: str
    status: ActivityStatus
    round: int | None = None
    started_offset_ms: int | None = None
    ended_offset_ms: int | None = None
    queries: tuple[str, ...] = ()
    refs: tuple[str, ...] = ()
    expression: str | None = None
    include_outline: bool | None = None
    top_k: int | None = None
    returned_count: int | None = None
    new_evidence_count: int | None = None
    document_count: int | None = None
    citation_count: int | None = None
    image_count: int | None = None
    path_count: int | None = None
    hop1_count: int | None = None
    hop2_count: int | None = None
    hop3_count: int | None = None
    result_value: str | None = None
    result_code: str | None = None
    sources: tuple[ActivitySource, ...] = ()
    scope_results: tuple[ActivityScope, ...] = ()
    details_truncated: bool = False

    def __post_init__(self) -> None:
        _integer(self.ordinal, 1)
        _integer(self.seq, 1)
        if self.step_id != f"step_{self.ordinal}":
            raise ValueError("invalid activity step identity")
        allowed = {
            "model": {"model_round"},
            "tool": ACTIVITY_TOOLS,
            "system": {"load_context", "prepare_visuals", "resolve_citations", "persist_result", "close_search", "token_wrap_up"},
        }
        if self.kind not in allowed or self.name not in allowed[self.kind]:
            raise ValueError("invalid activity name")
        if self.status not in ACTIVE_STATUSES | {"succeeded", "failed", "rejected", "cancelled"}:
            raise ValueError("invalid activity status")
        for key in ("round", "started_offset_ms", "ended_offset_ms", "top_k", "returned_count", "new_evidence_count", "document_count", "citation_count", "image_count", "path_count", "hop1_count", "hop2_count", "hop3_count"):
            value = getattr(self, key)
            if value is not None:
                _integer(value, 1 if key in {"round", "top_k"} else 0)
        if self.ended_offset_ms is not None and (self.started_offset_ms is None or self.ended_offset_ms < self.started_offset_ms):
            raise ValueError("invalid activity duration")
        for values, maximum in ((self.queries, 2048), (self.refs, 128)):
            if not isinstance(values, tuple) or len(values) > 3:
                raise ValueError("invalid activity inputs")
            for value in values:
                _text(value, maximum)
        for key, maximum in (("expression", 512), ("result_value", 1024), ("result_code", 80)):
            value = getattr(self, key)
            if value is not None:
                _text(value, maximum)
        if self.result_code is not None and not re.fullmatch(r"[A-Za-z0-9_]+", self.result_code):
            raise ValueError("invalid activity result code")
        if self.include_outline is not None and type(self.include_outline) is not bool:
            raise ValueError("invalid outline flag")
        if type(self.details_truncated) is not bool or not isinstance(self.sources, tuple) or len(self.sources) > 100:
            raise ValueError("invalid activity details")
        if not isinstance(self.scope_results, tuple) or len(self.scope_results) > 100 or any(not isinstance(scope, ActivityScope) for scope in self.scope_results):
            raise ValueError("invalid activity scopes")
        if any(not isinstance(source, ActivitySource) for source in self.sources):
            raise ValueError("invalid activity source")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> ActivityStep:
        data = _shape(value, cls)
        for key in ("queries", "refs", "sources"):
            if not isinstance(data[key], (list, tuple)):
                raise ValueError("invalid activity list")
        if not isinstance(data["scope_results"], (list, tuple)):
            raise ValueError("invalid activity scope list")
        data["scope_results"] = tuple(ActivityScope(**_shape(scope, ActivityScope)) for scope in data["scope_results"])
        data["queries"] = tuple(data["queries"])
        data["refs"] = tuple(data["refs"])
        data["sources"] = tuple(ActivitySource(**_shape(source, ActivitySource)) for source in data["sources"])
        return cls(**data)


@dataclass(frozen=True, slots=True)
class ChatActivitySnapshot:
    attempt: int
    steps: tuple[ActivityStep, ...]
    total_steps: int
    omitted_step_count: int
    status: Literal["running", "completed", "failed", "cancelled"]
    elapsed_ms: int
    version: Literal["chat_activity_v1"] = CHAT_ACTIVITY_VERSION

    def __post_init__(self) -> None:
        _integer(self.attempt, 1)
        _integer(self.total_steps)
        _integer(self.omitted_step_count)
        _integer(self.elapsed_ms)
        if self.version != CHAT_ACTIVITY_VERSION or self.status not in {"running", "completed", "failed", "cancelled"}:
            raise ValueError("invalid activity snapshot")
        if len(self.steps) > MAX_ACTIVITY_STEPS or self.total_steps != len(self.steps) + self.omitted_step_count:
            raise ValueError("invalid activity coverage")
        ordinals = [step.ordinal for step in self.steps]
        if ordinals != sorted(set(ordinals)) or (ordinals and ordinals[-1] > self.total_steps):
            raise ValueError("invalid activity order")
        if len(activity_json(self.as_dict()).encode("utf-8")) > MAX_ACTIVITY_BYTES:
            raise ValueError("activity snapshot exceeds byte limit")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> ChatActivitySnapshot:
        data = _shape(value, cls)
        if not isinstance(data["steps"], (list, tuple)):
            raise ValueError("invalid activity steps")
        data["steps"] = tuple(ActivityStep.from_dict(item) for item in data["steps"])
        return cls(**data)


@dataclass(frozen=True, slots=True)
class ChatActivityEvent:
    run_id: UUID
    attempt: int
    seq: int
    step: ActivityStep
    elapsed_ms: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, UUID):
            raise ValueError("invalid activity run")
        _integer(self.attempt, 1)
        _integer(self.seq, 1)
        _integer(self.elapsed_ms)
        if self.seq != self.step.seq:
            raise ValueError("invalid activity sequence")


def activity_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

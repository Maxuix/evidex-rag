"""Frozen per-run knowledge-base identities, independent of execution policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any
from uuid import UUID


def normalize_knowledge_base_ids(
    knowledge_base_ids: Sequence[UUID] | None = None,
    knowledge_base_id: UUID | None = None,
) -> tuple[UUID, ...]:
    if knowledge_base_ids is not None and knowledge_base_id is not None:
        raise ValueError("provide knowledge_base_ids or knowledge_base_id, not both")
    values = tuple(knowledge_base_ids) if knowledge_base_ids is not None else (
        (knowledge_base_id,) if knowledge_base_id is not None else ()
    )
    if not values or any(not isinstance(value, UUID) for value in values):
        raise ValueError("select at least one knowledge base")
    if len(values) != len(set(values)):
        raise ValueError("knowledge base selection must be unique")
    return tuple(sorted(values, key=str))


@dataclass(frozen=True, slots=True)
class ChatKnowledgeBaseSnapshot:
    knowledge_base_id: UUID
    name: str
    index_revision_id: UUID | None
    retrieval_strategy: Mapping[str, Any]
    description: str = ""
    graph_build_id: UUID | None = None
    status: str = "ready"

    def __post_init__(self) -> None:
        if self.status not in {"ready", "index_unavailable", "deleted"}:
            raise ValueError("invalid knowledge base snapshot status")
        if self.status == "ready" and self.index_revision_id is None:
            raise ValueError("ready scope requires an index revision")
        if not self.name.strip():
            raise ValueError("knowledge base snapshot requires a name")
        object.__setattr__(self, "retrieval_strategy", MappingProxyType(dict(self.retrieval_strategy)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "knowledge_base_id": str(self.knowledge_base_id),
            "name": self.name,
            "description": self.description,
            "index_revision_id": str(self.index_revision_id) if self.index_revision_id else None,
            "graph_build_id": str(self.graph_build_id) if self.graph_build_id else None,
            "retrieval_strategy": dict(self.retrieval_strategy),
            "status": self.status,
        }


def resolve_scope(
    snapshots: tuple[ChatKnowledgeBaseSnapshot, ...], target: object,
) -> tuple[ChatKnowledgeBaseSnapshot, ...]:
    if target == "all_selected":
        return snapshots
    if not isinstance(target, str):
        raise ValueError("knowledge_base_id_required")
    try:
        identifier = UUID(target)
    except ValueError as error:
        raise ValueError("invalid_knowledge_base_id") from error
    for snapshot in snapshots:
        if snapshot.knowledge_base_id == identifier:
            return (snapshot,)
    raise ValueError("knowledge_base_out_of_scope")

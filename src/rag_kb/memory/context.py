"""Deterministic selection and strict serialization for Session context."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import tiktoken

from rag_kb.domain.memory import (
    SESSION_CONTEXT_STRATEGY,
    SESSION_CONTEXT_VERSION,
    ConversationContextSnapshot,
    ConversationTurn,
)


class ConversationContextSelector:
    """Preload the fixed tokenizer, then select turns without external I/O."""

    def __init__(
        self,
        *,
        max_turns: int = 6,
        token_budget: int = 4000,
        tokenizer: str = "cl100k_base",
    ) -> None:
        if max_turns != 6 or token_budget != 4000 or tokenizer != "cl100k_base":
            raise ValueError("unsupported Session context policy")
        self._max_turns = max_turns
        self._token_budget = token_budget
        self._encoding = tiktoken.get_encoding(tokenizer)

    def select(
        self, newest_first: tuple[ConversationTurn, ...]
    ) -> ConversationContextSnapshot:
        return _select_with_encoding(
            newest_first,
            max_turns=self._max_turns,
            token_budget=self._token_budget,
            encoding=self._encoding,
        )


def empty_conversation_context(
    *, token_budget: int = 4000,
) -> ConversationContextSnapshot:
    return _snapshot(
        turns=(),
        token_budget=token_budget,
        token_count=0,
        candidate_turn_count=0,
        truncated=False,
    )


def select_conversation_context(
    newest_first: tuple[ConversationTurn, ...],
    *,
    max_turns: int = 6,
    token_budget: int = 4000,
    tokenizer: str = "cl100k_base",
) -> ConversationContextSnapshot:
    if max_turns != 6 or token_budget != 4000 or tokenizer != "cl100k_base":
        raise ValueError("unsupported Session context policy")
    encoding = tiktoken.get_encoding(tokenizer)
    return _select_with_encoding(
        newest_first,
        max_turns=max_turns,
        token_budget=token_budget,
        encoding=encoding,
    )


def _select_with_encoding(
    newest_first: tuple[ConversationTurn, ...],
    *,
    max_turns: int,
    token_budget: int,
    encoding: tiktoken.Encoding,
) -> ConversationContextSnapshot:
    selected: list[ConversationTurn] = []
    token_count = 0
    for turn in newest_first[:max_turns]:
        turn_tokens = len(encoding.encode(_canonical_json(_turn_payload(turn))))
        if token_count + turn_tokens > token_budget:
            break
        selected.append(turn)
        token_count += turn_tokens
    chronological = tuple(reversed(selected))
    truncated = len(chronological) < len(newest_first)
    return _snapshot(
        turns=chronological,
        token_budget=token_budget,
        token_count=token_count,
        candidate_turn_count=len(newest_first),
        truncated=truncated,
    )


def serialize_conversation_context(
    snapshot: ConversationContextSnapshot,
) -> dict[str, Any]:
    payload = _snapshot_payload(snapshot)
    if _hash(payload) != snapshot.content_hash:
        raise ValueError("conversation context hash does not match its content")
    return {**payload, "content_hash": snapshot.content_hash}


def hydrate_conversation_context(value: object) -> ConversationContextSnapshot:
    if not isinstance(value, dict):
        raise ValueError("conversation context must be an object")
    expected = {
        "version",
        "strategy",
        "turns",
        "token_budget",
        "token_count",
        "candidate_turn_count",
        "truncated",
        "content_hash",
    }
    if set(value) != expected:
        raise ValueError("conversation context fields are invalid")
    turns_value = value["turns"]
    if not isinstance(turns_value, list):
        raise ValueError("conversation context turns must be an array")
    turns = tuple(_hydrate_turn(item) for item in turns_value)
    if any(isinstance(value[name], bool) or not isinstance(value[name], int) for name in (
        "token_budget", "token_count", "candidate_turn_count"
    )):
        raise ValueError("conversation context counters are invalid")
    if not isinstance(value["truncated"], bool) or not isinstance(value["content_hash"], str):
        raise ValueError("conversation context metadata is invalid")
    snapshot = ConversationContextSnapshot(
        version=value["version"],
        strategy=value["strategy"],
        turns=turns,
        token_budget=value["token_budget"],
        token_count=value["token_count"],
        candidate_turn_count=value["candidate_turn_count"],
        truncated=value["truncated"],
        content_hash=value["content_hash"],
    )
    encoding = tiktoken.get_encoding("cl100k_base")
    actual_tokens = sum(
        len(encoding.encode(_canonical_json(_turn_payload(turn))))
        for turn in snapshot.turns
    )
    if actual_tokens != snapshot.token_count:
        raise ValueError("conversation context token count is invalid")
    if serialize_conversation_context(snapshot) != value:
        raise ValueError("conversation context is not canonical")
    return snapshot


def _snapshot(
    *,
    turns: tuple[ConversationTurn, ...],
    token_budget: int,
    token_count: int,
    candidate_turn_count: int,
    truncated: bool,
) -> ConversationContextSnapshot:
    partial = ConversationContextSnapshot(
        version=SESSION_CONTEXT_VERSION,
        strategy=SESSION_CONTEXT_STRATEGY,
        turns=turns,
        token_budget=token_budget,
        token_count=token_count,
        candidate_turn_count=candidate_turn_count,
        truncated=truncated,
        content_hash="sha256:" + "0" * 64,
    )
    return ConversationContextSnapshot(
        version=partial.version,
        strategy=partial.strategy,
        turns=partial.turns,
        token_budget=partial.token_budget,
        token_count=partial.token_count,
        candidate_turn_count=partial.candidate_turn_count,
        truncated=partial.truncated,
        content_hash=_hash(_snapshot_payload(partial)),
    )


def _snapshot_payload(snapshot: ConversationContextSnapshot) -> dict[str, Any]:
    return {
        "version": snapshot.version,
        "strategy": snapshot.strategy,
        "turns": [_turn_payload(turn) for turn in snapshot.turns],
        "token_budget": snapshot.token_budget,
        "token_count": snapshot.token_count,
        "candidate_turn_count": snapshot.candidate_turn_count,
        "truncated": snapshot.truncated,
    }


def _turn_payload(turn: ConversationTurn) -> dict[str, Any]:
    return {
        "user": {
            "role": "user",
            "message_id": str(turn.user_message_id),
            "content": turn.user_content,
        },
        "assistant": {
            "role": "assistant",
            "message_id": str(turn.assistant_message_id),
            "content": turn.assistant_content,
        },
    }


def _hydrate_turn(value: object) -> ConversationTurn:
    if not isinstance(value, dict) or set(value) != {"user", "assistant"}:
        raise ValueError("conversation turn shape is invalid")
    messages: dict[str, dict[str, Any]] = {}
    for role in ("user", "assistant"):
        message = value[role]
        if not isinstance(message, dict) or set(message) != {"role", "message_id", "content"}:
            raise ValueError("conversation message shape is invalid")
        if (
            message["role"] != role
            or not isinstance(message["message_id"], str)
            or not isinstance(message["content"], str)
        ):
            raise ValueError("conversation message role or content is invalid")
        messages[role] = message
    from uuid import UUID

    return ConversationTurn(
        user_message_id=UUID(messages["user"]["message_id"]),
        user_content=messages["user"]["content"],
        assistant_message_id=UUID(messages["assistant"]["message_id"]),
        assistant_content=messages["assistant"]["content"],
    )


def _hash(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

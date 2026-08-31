from __future__ import annotations

import json
import unittest
from uuid import uuid4

from rag_kb.domain import (
    ConversationTurn,
)
from rag_kb.memory import (
    hydrate_conversation_context,
    select_conversation_context,
    serialize_conversation_context,
)


def _turn(number: int, *, content: str | None = None) -> ConversationTurn:
    text = content or f"turn {number}"
    return ConversationTurn(
        user_message_id=uuid4(),
        user_content=f"user {text}",
        assistant_message_id=uuid4(),
        assistant_content=f"assistant {text}",
    )


class ConversationContextTests(unittest.TestCase):
    def test_zero_one_and_six_turn_windows_are_not_truncated(self) -> None:
        for count in (0, 1, 6):
            chronological = tuple(_turn(number) for number in range(count))
            with self.subTest(count=count):
                snapshot = select_conversation_context(tuple(reversed(chronological)))
                self.assertEqual(snapshot.turns, chronological)
                self.assertFalse(snapshot.truncated)

    def test_recent_window_is_complete_bounded_and_chronological(self) -> None:
        chronological = tuple(_turn(number) for number in range(7))
        snapshot = select_conversation_context(tuple(reversed(chronological)))
        self.assertEqual(snapshot.turns, chronological[1:])
        self.assertTrue(snapshot.truncated)

    def test_oversized_latest_turn_stops_without_skipping_older_turns(self) -> None:
        snapshot = select_conversation_context((_turn(2, content="token " * 5000), _turn(1)))
        self.assertEqual(snapshot.turns, ())
        self.assertTrue(snapshot.truncated)

    def test_snapshot_hash_detects_tampering(self) -> None:
        snapshot = select_conversation_context((_turn(2), _turn(1)))
        serialized = serialize_conversation_context(snapshot)
        self.assertEqual(hydrate_conversation_context(serialized), snapshot)
        tampered = json.loads(json.dumps(serialized))
        tampered["turns"][0]["user"]["content"] = "changed"
        with self.assertRaises(ValueError):
            hydrate_conversation_context(tampered)


if __name__ == "__main__":
    unittest.main()

"""Transaction modes used by the concrete database boundary."""

from __future__ import annotations

from enum import StrEnum


class TransactionMode(StrEnum):
    READ_WRITE = "read_write"
    REPEATABLE_READ_ONLY = "repeatable_read_only"

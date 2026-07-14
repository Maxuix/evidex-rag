"""One-shot Linux parser-isolation self-check used by delivery validation."""

from __future__ import annotations

import asyncio
import sys

from rag_kb.adapters import IsolatedPlainTextProcessor
from rag_kb.services import (
    ErrorCode,
    ParserExecutionError,
    ParserLimits,
    ParserSource,
)


def _memory_probe(connection, source, limits, maximum, overlap) -> None:
    del source, maximum, overlap
    import resource

    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (limits.memory_bytes, hard))
    try:
        bytearray(limits.memory_bytes * 4)
    except MemoryError:
        connection.send(
            (
                "error",
                {
                    "code": ErrorCode.PARSER_RESOURCE_LIMIT.value,
                    "diagnostic": {"limit_name": "memory_bytes"},
                },
            )
        )
    finally:
        connection.close()


async def check_parser() -> None:
    processor = IsolatedPlainTextProcessor(ParserLimits())
    result = await processor.process(
        ParserSource("isolation-check.txt", "text/plain", b"isolated parser ready")
    )
    if len(result.chunks) != 1 or result.chunks[0].text != "isolated parser ready":
        raise RuntimeError("isolated parser self-check returned an invalid result")
    if sys.platform.startswith("linux"):
        memory_probe = IsolatedPlainTextProcessor(
            ParserLimits(
                max_chunks=1,
                wall_seconds=5,
                cpu_seconds=2,
                memory_bytes=128 * 1024 * 1024,
            ),
            child_target=_memory_probe,
        )
        try:
            await memory_probe.process(
                ParserSource("memory-check.txt", "text/plain", b"memory")
            )
        except ParserExecutionError as error:
            if error.code is ErrorCode.PARSER_RESOURCE_LIMIT:
                return
            raise
        raise RuntimeError("parser memory limit was not enforced")


def main() -> int:
    asyncio.run(check_parser())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Spawn-isolated P1A text parser with bounded resources and no network."""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import signal
import sys
from collections.abc import Callable
from multiprocessing.connection import Connection
from typing import Any

from rag_kb.domain import (
    ErrorCode,
    ParserExecutionError,
    ParserLimits,
    ParserSource,
    ProcessedDocument,
)
from rag_kb.adapters.parser.plain_text import process_plain_text


ChildTarget = Callable[[Connection, ParserSource, ParserLimits, int, int], None]


class IsolatedPlainTextProcessor:
    def __init__(
        self,
        limits: ParserLimits,
        *,
        max_characters: int = 2_000,
        overlap_characters: int = 200,
        child_target: ChildTarget | None = None,
    ) -> None:
        if overlap_characters >= max_characters:
            raise ValueError("overlap_characters must be less than max_characters")
        self._limits = limits
        self._max_characters = max_characters
        self._overlap_characters = overlap_characters
        self._child_target = child_target or _parser_child

    async def process(self, source: ParserSource) -> ProcessedDocument:
        return await asyncio.to_thread(self._process, source)

    def _process(self, source: ParserSource) -> ProcessedDocument:
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=False)
        process = context.Process(
            target=self._child_target,
            args=(child, source, self._limits, self._max_characters, self._overlap_characters),
            name="rag-kb-parser",
        )
        process.start()
        child.close()
        try:
            if not parent.poll(self._limits.wall_seconds):
                _stop(process)
                raise ParserExecutionError(
                    ErrorCode.PARSER_TIMEOUT,
                    diagnostic={"limit_name": "wall_seconds", "limit": self._limits.wall_seconds},
                )
            try:
                message = parent.recv()
            except EOFError:
                message = None
            process.join(timeout=1.0)
            if message is None:
                raise _exit_error(process.exitcode)
            kind, payload = message
            if kind == "ok" and isinstance(payload, ProcessedDocument):
                return payload
            if kind == "error" and isinstance(payload, dict):
                try:
                    code = ErrorCode(payload["code"])
                except (KeyError, ValueError):
                    code = ErrorCode.PARSER_CRASHED
                diagnostic = payload.get("diagnostic")
                raise ParserExecutionError(
                    code,
                    diagnostic=diagnostic if isinstance(diagnostic, dict) else {},
                )
            raise ParserExecutionError(ErrorCode.PARSER_CRASHED)
        finally:
            parent.close()
            if process.is_alive():
                _stop(process)
            process.close()


def _parser_child(
    connection: Connection,
    source: ParserSource,
    limits: ParserLimits,
    max_characters: int,
    overlap_characters: int,
) -> None:
    try:
        _apply_resource_limits(limits)
        os.environ.clear()
        _deny_network()
        _verify_isolation()
        result = process_plain_text(
            source,
            max_characters=max_characters,
            overlap_characters=overlap_characters,
            max_chunks=limits.max_chunks,
        )
        connection.send(("ok", result))
    except ParserExecutionError as error:
        connection.send(("error", {"code": error.code.value, "diagnostic": error.diagnostic}))
    except MemoryError:
        connection.send(
            (
                "error",
                {
                    "code": ErrorCode.PARSER_RESOURCE_LIMIT.value,
                    "diagnostic": {"limit_name": "memory_bytes", "limit": limits.memory_bytes},
                },
            )
        )
    except BaseException:
        try:
            connection.send(("error", {"code": ErrorCode.PARSER_CRASHED.value, "diagnostic": {}}))
        except BaseException:
            pass
    finally:
        connection.close()


def _apply_resource_limits(limits: ParserLimits) -> None:
    try:
        import resource
    except ImportError as error:
        raise ParserExecutionError(
            ErrorCode.PARSER_ISOLATION_FAILED,
            diagnostic={"check": "resource_module"},
        ) from error
    try:
        _, cpu_hard = resource.getrlimit(resource.RLIMIT_CPU)
        resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, cpu_hard))
    except (OSError, ValueError) as error:
        raise ParserExecutionError(
            ErrorCode.PARSER_ISOLATION_FAILED,
            diagnostic={"check": "cpu_limit"},
        ) from error
    if sys.platform == "darwin":
        # Darwin rejects lowering the Python process address/data limits even
        # when the reported hard limit is infinite. The shipped runtime is the
        # Linux Compose worker, where RLIMIT_AS is mandatory and exercised.
        return
    try:
        memory_resource = resource.RLIMIT_AS
        _, memory_hard = resource.getrlimit(memory_resource)
        resource.setrlimit(memory_resource, (limits.memory_bytes, memory_hard))
    except (OSError, ValueError) as error:
        raise ParserExecutionError(
            ErrorCode.PARSER_ISOLATION_FAILED,
            diagnostic={"check": "memory_limit"},
        ) from error


def _deny_network() -> None:
    def audit(event: str, args: tuple[Any, ...]) -> None:
        del args
        if event.startswith("socket."):
            raise PermissionError("network disabled")

    sys.addaudithook(audit)


def _verify_isolation() -> None:
    if os.environ:
        raise ParserExecutionError(
            ErrorCode.PARSER_ISOLATION_FAILED,
            diagnostic={"check": "environment"},
        )
    import socket

    try:
        probe = socket.socket()
    except PermissionError:
        return
    probe.close()
    raise ParserExecutionError(
        ErrorCode.PARSER_ISOLATION_FAILED,
        diagnostic={"check": "network"},
    )


def _stop(process: multiprocessing.Process) -> None:
    process.terminate()
    process.join(timeout=1.0)
    if process.is_alive():
        process.kill()
        process.join(timeout=1.0)


def _exit_error(exitcode: int | None) -> ParserExecutionError:
    resource_signals = {getattr(signal, "SIGXCPU", -1), signal.SIGKILL}
    if exitcode is not None and exitcode < 0 and -exitcode in resource_signals:
        return ParserExecutionError(
            ErrorCode.PARSER_RESOURCE_LIMIT,
            diagnostic={"exit_kind": "resource_limit"},
        )
    return ParserExecutionError(
        ErrorCode.PARSER_CRASHED,
        diagnostic={"exit_kind": "abnormal_exit"},
    )

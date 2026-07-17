"""Spawn-isolated local Unstructured processor with bounded resources and no network."""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import signal
import sys
import tempfile
from collections.abc import Callable
from multiprocessing.connection import Connection
from typing import Any

from rag_kb.adapters.parser.langchain_unstructured import process_with_unstructured
from rag_kb.domain import (
    ErrorCode,
    ParserExecutionError,
    ParserLimits,
    ParserSource,
    ProcessedDocument,
)


ChildTarget = Callable[[Connection, ParserSource, ParserLimits], None]


class IsolatedUnstructuredProcessor:
    def __init__(
        self,
        limits: ParserLimits,
        *,
        child_target: ChildTarget | None = None,
    ) -> None:
        self._limits = limits
        self._child_target = child_target or _parser_child

    async def process(self, source: ParserSource) -> ProcessedDocument:
        return await asyncio.to_thread(self._process, source)

    def _process(self, source: ParserSource) -> ProcessedDocument:
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=False)
        process = context.Process(
            target=self._child_target,
            args=(child, source, self._limits),
            name="rag-kb-parser",
        )
        process.start()
        child.close()
        try:
            if not parent.poll(self._limits.wall_seconds):
                _stop(process)
                raise ParserExecutionError(
                    ErrorCode.PARSER_TIMEOUT,
                    diagnostic={
                        "limit_name": "wall_seconds",
                        "limit": self._limits.wall_seconds,
                    },
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
) -> None:
    try:
        _apply_resource_limits(limits)
        with tempfile.TemporaryDirectory(prefix="rag-kb-parser-") as scratch:
            _configure_environment(scratch)
            _deny_network()
            _verify_isolation(scratch)
            connection.send(("ok", process_with_unstructured(source, limits)))
    except ParserExecutionError as error:
        connection.send(
            (
                "error",
                {"code": error.code.value, "diagnostic": error.diagnostic},
            )
        )
    except MemoryError:
        connection.send(
            (
                "error",
                {
                    "code": ErrorCode.PARSER_RESOURCE_LIMIT.value,
                    "diagnostic": {
                        "limit_name": "memory_bytes",
                        "limit": limits.memory_bytes,
                    },
                },
            )
        )
    except BaseException:
        try:
            connection.send(
                (
                    "error",
                    {
                        "code": ErrorCode.PARSER_CRASHED.value,
                        "diagnostic": {},
                    },
                )
            )
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
    import socket

    network_families = {socket.AF_INET, socket.AF_INET6}

    def audit(event: str, args: tuple[Any, ...]) -> None:
        if (
            event == "socket.__new__"
            and len(args) >= 2
            and args[1] in network_families
        ) or event in {
            "socket.connect",
            "socket.connect_ex",
            "socket.getaddrinfo",
            "socket.gethostbyaddr",
            "socket.gethostbyname",
        }:
            raise PermissionError("network disabled")

    sys.addaudithook(audit)


def _configure_environment(scratch: str) -> None:
    tiktoken_cache_dir = os.environ.get("TIKTOKEN_CACHE_DIR")
    isolated_environment = {
        "HOME": scratch,
        "HF_HOME": scratch,
        "HF_HUB_OFFLINE": "1",
        "MKL_NUM_THREADS": "1",
        "MPLCONFIGDIR": scratch,
        "NUMBA_CACHE_DIR": scratch,
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
        "XDG_CACHE_HOME": scratch,
    }
    if tiktoken_cache_dir is not None:
        isolated_environment["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
    os.environ.clear()
    os.environ.update(isolated_environment)


def _verify_isolation(scratch: str) -> None:
    expected_keys = {
        "HOME",
        "HF_HOME",
        "HF_HUB_OFFLINE",
        "MKL_NUM_THREADS",
        "MPLCONFIGDIR",
        "NUMBA_CACHE_DIR",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "TOKENIZERS_PARALLELISM",
        "TRANSFORMERS_OFFLINE",
        "XDG_CACHE_HOME",
    }
    tiktoken_cache_dir = os.environ.get("TIKTOKEN_CACHE_DIR")
    if tiktoken_cache_dir is not None:
        expected_keys.add("TIKTOKEN_CACHE_DIR")
        if (
            not os.path.isabs(tiktoken_cache_dir)
            or not os.path.isdir(tiktoken_cache_dir)
        ):
            raise ParserExecutionError(
                ErrorCode.PARSER_ISOLATION_FAILED,
                diagnostic={"check": "tokenizer_cache"},
            )
    if set(os.environ) != expected_keys or any(
        key != "TIKTOKEN_CACHE_DIR"
        and key.endswith(("HOME", "CACHE_HOME", "CONFIGDIR", "CACHE_DIR"))
        and value != scratch
        for key, value in os.environ.items()
    ):
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

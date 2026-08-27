#!/usr/bin/env python3
"""Run one child command and durably publish its exit status.

The persistent evaluation supervisor uses this small wrapper so a supervisor
restart can distinguish an active child from a child that finished with a
non-zero status.  The exit record contains no command output or provider data.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any

from tools.evaluation_campaign_state import write_private_json


SCHEMA = "supervised_command_exit_v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exit-file", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _write_exit(path: Path, *, started_at: str, exit_code: int, status: str) -> None:
    write_private_json(
        path,
        {
            "schema_version": SCHEMA,
            "status": status,
            "exit_code": int(exit_code),
            "started_at": started_at,
            "completed_at": _timestamp(),
        },
    )


def main() -> int:
    arguments = _parser().parse_args()
    command = list(arguments.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("a supervised command is required")

    started_at = _timestamp()
    child: subprocess.Popen[bytes] | None = None
    interrupted_by: int | None = None

    def forward_signal(signum: int, _frame: Any) -> None:
        nonlocal interrupted_by
        interrupted_by = signum
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, forward_signal)
    signal.signal(signal.SIGINT, forward_signal)
    try:
        child = subprocess.Popen(command, start_new_session=True)
        exit_code = child.wait()
        if interrupted_by is not None and exit_code == 0:
            exit_code = 128 + interrupted_by
        _write_exit(
            arguments.exit_file,
            started_at=started_at,
            exit_code=exit_code,
            status="completed" if exit_code == 0 else "failed",
        )
        return exit_code
    except BaseException:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            _write_exit(
                arguments.exit_file,
                started_at=started_at,
                exit_code=70,
                status="failed",
            )
        except BaseException:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())

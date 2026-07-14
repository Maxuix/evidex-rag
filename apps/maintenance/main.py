"""Run one bounded maintenance pass and exit."""

from __future__ import annotations

import argparse
import asyncio
import json

from apps.maintenance.dependencies import build_maintenance_dependencies


async def cleanup() -> dict[str, int]:
    dependencies = build_maintenance_dependencies()
    try:
        result = await dependencies.cleanup.run_once(
            dependencies.auth_provider.get_context()
        )
        return {
            "pending_activated": result.files.pending_activated,
            "missing_compensated": result.files.missing_compensated,
            "file_cleanup_completed": result.files.cleanup_completed,
            "file_cleanup_failed": result.files.cleanup_failed,
            "orphan_files_removed": result.files.orphans_removed,
            "retired_targets_cleaned": result.index.retired_targets_cleaned,
            "vectors_deleted": result.index.vectors_deleted,
            "chunks_deleted": result.index.chunks_deleted,
            "jobs_deleted": result.index.jobs_deleted,
            "file_cleanup_records_deleted": (
                result.index.file_cleanup_records_deleted
            ),
        }
    finally:
        await dependencies.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("cleanup",))
    arguments = parser.parse_args()
    if arguments.command == "cleanup":
        print(json.dumps(asyncio.run(cleanup()), separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

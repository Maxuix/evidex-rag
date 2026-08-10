"""Run one bounded maintenance pass and exit."""

from __future__ import annotations

import argparse
import asyncio
import json

from apps.maintenance.dependencies import build_maintenance_dependencies
from rag_kb.observability import configure_logging, get_logger, log_event, log_exception


LOGGER = get_logger("rag_kb.maintenance.runtime")


async def cleanup() -> dict[str, int]:
    dependencies = build_maintenance_dependencies()
    configure_logging(
        level=dependencies.settings.observability.log_level,
        process="maintenance",
        log_directory=dependencies.settings.observability.log_directory,
    )
    try:
        result = await dependencies.cleanup.run_once(
            dependencies.auth_provider.get_context()
        )
        summary = {
            "pending_activated": result.files.pending_activated,
            "missing_compensated": result.files.missing_compensated,
            "file_cleanup_completed": result.files.cleanup_completed,
            "file_cleanup_failed": result.files.cleanup_failed,
            "orphan_files_removed": result.files.orphans_removed,
            "retired_targets_cleaned": result.index.retired_targets_cleaned,
            "vectors_deleted": result.index.vectors_deleted,
            "chunks_deleted": result.index.chunks_deleted,
            "plans_deleted": result.index.plans_deleted,
            "manifests_deleted": result.index.manifests_deleted,
            "assets_deleted": result.index.assets_deleted,
            "jobs_deleted": result.index.jobs_deleted,
            "file_cleanup_records_deleted": (
                result.index.file_cleanup_records_deleted
            ),
        }
        log_event(LOGGER, "maintenance_cleanup_completed", **summary)
        return summary
    finally:
        await dependencies.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("cleanup",))
    arguments = parser.parse_args()
    configure_logging(level="INFO", process="maintenance")
    try:
        if arguments.command == "cleanup":
            print(json.dumps(asyncio.run(cleanup()), separators=(",", ":")))
    except Exception as error:
        log_exception(LOGGER, "process_failed", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

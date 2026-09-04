#!/usr/bin/env python3
"""Recompute existing lexical rows from source only; no embeddings or QA generation."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from uuid import UUID

from sqlalchemy import text

from rag_kb.config import load_settings
from rag_kb.db import DatabaseProcess, create_database_resources
from rag_kb.document_processing.lexical import (
    LEXICAL_ANALYZER_VERSION, analyze_document, lexical_manifest_hash,
)


async def rebuild(session, *, workspace_id, kb_id, revision_id, apply=False):
    """Caller owns one transaction: derived rows and manifests change atomically."""
    scope = {"workspace": workspace_id, "kb": kb_id, "revision": revision_id,
             "analyzer": LEXICAL_ANALYZER_VERSION}
    targets = (await session.execute(text(
        "SELECT id FROM indexed_document_version WHERE workspace_id=:workspace "
        "AND kb_id=:kb AND index_revision_id=:revision "
        "AND build_status='ready' ORDER BY id" + (" FOR UPDATE" if apply else "")
    ), scope)).scalars().all()
    if not targets:
        raise RuntimeError("No ready targets in the exact requested scope")
    changed = 0
    for target in targets:
        values = {**scope, "target": target}
        rows = (await session.execute(text(
            "SELECT l.index_chunk_id, l.lexical_text_hash, c.content, c.embedding_text "
            "FROM index_chunk_lexical l JOIN index_chunk c ON c.id=l.index_chunk_id "
            "AND c.indexed_document_version_id=l.indexed_document_version_id "
            "AND c.workspace_id=l.workspace_id AND c.kb_id=l.kb_id "
            "WHERE l.workspace_id=:workspace AND l.kb_id=:kb "
            "AND l.indexed_document_version_id=:target AND l.analyzer_version=:analyzer "
            "ORDER BY l.index_chunk_id" + (" FOR UPDATE OF l" if apply else "")
        ), values)).mappings().all()
        manifest = (await session.execute(text(
            "SELECT lexical_chunk_count, lexical_manifest_hash FROM index_lexical_manifest "
            "WHERE workspace_id=:workspace AND kb_id=:kb "
            "AND indexed_document_version_id=:target AND analyzer_version=:analyzer"
        ), values)).mappings().one()
        previous_hash = lexical_manifest_hash(
            LEXICAL_ANALYZER_VERSION,
            ((row["index_chunk_id"], row["lexical_text_hash"]) for row in rows),
        )
        if manifest["lexical_chunk_count"] != len(rows) or manifest["lexical_manifest_hash"] != previous_hash:
            raise RuntimeError("Existing lexical manifest is incomplete; no repair applied")
        hashes = []
        for row in rows:
            analyzed = analyze_document(row["embedding_text"] or row["content"])
            if analyzed is None:
                raise RuntimeError("Source has no lexical representation; explicit rebuild required")
            hashes.append((row["index_chunk_id"], analyzed.lexical_text_hash))
            if analyzed.lexical_text_hash != row["lexical_text_hash"]:
                changed += 1
                if apply:
                    await session.execute(text(
                        "UPDATE index_chunk_lexical SET lexical_text=:lexical, lexical_text_hash=:hash "
                        "WHERE workspace_id=:workspace AND kb_id=:kb "
                        "AND indexed_document_version_id=:target AND index_chunk_id=:chunk "
                        "AND analyzer_version=:analyzer"
                    ), {**values, "chunk": row["index_chunk_id"], "lexical": analyzed.lexical_text,
                        "hash": analyzed.lexical_text_hash})
        if apply:
            await session.execute(text(
                "UPDATE index_lexical_manifest SET lexical_manifest_hash=:hash "
                "WHERE workspace_id=:workspace AND kb_id=:kb "
                "AND indexed_document_version_id=:target AND analyzer_version=:analyzer"
            ), {**values, "hash": lexical_manifest_hash(LEXICAL_ANALYZER_VERSION, hashes)})
    return {"targets": len(targets), "changed_rows": changed, "applied": apply,
            "qa_generation_calls": 0, "embedding_calls": 0}


async def run(args):
    settings = load_settings(env_file=args.env_file)
    async with create_database_resources(
        settings.database.runtime_dsn.get_secret_value(), pool_size=1, max_overflow=0,
        process=DatabaseProcess.MAINTENANCE,
    ) as database:
        async with database.sessions() as session, session.begin():
            if not args.apply:
                await session.execute(text("SET TRANSACTION READ ONLY"))
            result = await rebuild(session, workspace_id=settings.identity.workspace_id,
                                   kb_id=args.kb_id, revision_id=args.revision_id, apply=args.apply)
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--kb-id", type=UUID, required=True)
    parser.add_argument("--revision-id", type=UUID, required=True)
    parser.add_argument("--apply", action="store_true", help="Commit; default is read-only preview")
    asyncio.run(run(parser.parse_args()))

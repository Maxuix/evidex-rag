"""Graph configuration and one-work-item extraction services."""

from __future__ import annotations

from datetime import UTC, datetime
from dataclasses import dataclass
from time import perf_counter
from typing import Any
from uuid import UUID, uuid4

from rag_kb.auth import AccessPolicy, AuthContext
from rag_kb.domain import (
    GRAPH_EXTRACTOR_VERSION,
    GraphChunkExtraction,
    GraphChunkResultStatus,
    GraphConfigSnapshot,
    GraphProtocolError,
    GraphResourceLimitError,
    GraphWorkItem,
    GraphWorkKind,
    ChatModelExecutionError,
    ChatModelMessage,
    ChatModelRequest,
    ErrorCode,
    ResourceNotFoundError,
    ResourceStateConflictError,
)
from rag_kb.graph.extraction import graph_protocol_error_family, parse_graph_extraction
from rag_kb.observability import get_logger, log_event
from rag_kb.ports.model_api import ChatModelAdapter
from rag_kb.uow import UnitOfWork, UnitOfWorkFactory, UnitOfWorkPurpose, execute_in_transaction


GRAPH_RESPONSE_FORMAT: dict[str, Any] = {"type": "json_object"}
_LOGGER = get_logger(__name__)
_TRUNCATED_RESPONSE_MARKER = '{"_response_truncated":true}'
_TRUNCATED_FINISH_REASONS = frozenset(
    {"length", "max_tokens", "max_output_tokens", "content_filter_length"}
)
_REPAIR_RULES = {
    "support_text_not_locatable": (
        "Copy every surface and support from the chunk; delete any item that cannot be located."
    ),
    "relation_support_missing_endpoint": (
        "Every relation support must explicitly contain both referenced entity surfaces; "
        "delete the relation if it does not."
    ),
    "relation_endpoint_unknown": (
        "Every relation endpoint must reference an entity id defined in this response; "
        "delete the relation if it cannot be corrected."
    ),
    "json_invalid": "Return one JSON object parseable by json.loads, with no fence or explanation.",
    "schema_invalid": (
        "Use only the specified keys, entity type enum, JSON value types, and null rules."
    ),
    "self_relation": (
        "Delete every relation whose subject and object resolve to the same entity."
    ),
}
_JSON_SKELETON = (
    '{"entities":[{"id":"entity_1","type":"organization","surface":"exact slice",'
    '"disambiguator":null,"disambiguator_support":null}],'
    '"relations":[{"subject":"entity_1","predicate":"exact relation",'
    '"object":"entity_2","support":"exact slice containing both surfaces"}]}'
)
_GRAPH_SYSTEM_PROMPT = (
    "You extract only explicitly grounded entities and relations from one text chunk. "
    "Return one JSON object with exactly two arrays: entities and relations. "
    "Each entity has id, type, surface, disambiguator, disambiguator_support. "
    "Entity id uses only letters, numbers, underscore, or hyphen, is 1 through 64 "
    "characters, and is unique within the response. "
    "For ordinary named entities set disambiguator and disambiguator_support to null. "
    "Use a disambiguator only when the text explicitly distinguishes two same-surface "
    "entities; then the exact disambiguator string must appear inside "
    "disambiguator_support. Never use an inferred category, role, or adjacent word "
    "as a disambiguator; when uncertain, use null for both fields. "
    "Types are person, organization, location, product, system, document, event, concept. "
    "Each relation has subject, predicate, object, support. "
    "Relation subject and object must exactly reference ids defined in this response. "
    "Never return a self relation. "
    "Relation support must explicitly contain both referenced entity surfaces; "
    "delete a relation when this cannot be satisfied. "
    "Every surface and support is one continuous exact substring from the chunk. "
    "A pronoun, alias, canonical name, paraphrase, or evidence assembled across spans cannot "
    "replace either endpoint surface. Do not infer from outside knowledge. "
    "Return at most 16 entities and 8 relations. Prefer relations with explicit two-endpoint "
    "support and never infer items to reach a count. "
    "Do not treat standalone numbers, dates, or short table headers as entities. "
    "Return empty arrays when no grounded fact exists. Do not add keys. "
    "Follow this shape (example strings are placeholders): "
    + _JSON_SKELETON
)
_PREFLIGHT_TEXT = "Atlas Labs released Orion in 2024."
@dataclass(frozen=True, slots=True)
class GraphConfigView:
    snapshot: GraphConfigSnapshot
    profile_name: str | None = None
    provider_name: str | None = None
    model: str | None = None


class GraphConfigurationService:
    def __init__(self, unit_of_work: UnitOfWorkFactory, access_policy: AccessPolicy) -> None:
        self._unit_of_work = unit_of_work
        self._access_policy = access_policy

    async def get(self, context: AuthContext, kb_id: UUID) -> GraphConfigSnapshot:
        self._access_policy.require_workspace(context, context.workspace_id)

        async def load(uow: UnitOfWork) -> GraphConfigSnapshot:
            _require_scope(uow, context)
            return await uow.graph.ensure_config(kb_id)

        return await execute_in_transaction(
            self._unit_of_work, load, purpose=UnitOfWorkPurpose.REQUEST
        )

    async def get_view(self, context: AuthContext, kb_id: UUID) -> GraphConfigView:
        self._access_policy.require_workspace(context, context.workspace_id)

        async def load(uow: UnitOfWork) -> GraphConfigView:
            _require_scope(uow, context)
            snapshot = await uow.graph.ensure_config(kb_id)
            return _config_view(snapshot, await _profile_bundle(uow, snapshot))

        return await execute_in_transaction(
            self._unit_of_work, load, purpose=UnitOfWorkPurpose.REQUEST
        )

    async def configure(
        self,
        context: AuthContext,
        kb_id: UUID,
        *,
        enabled: bool,
        chat_profile_revision_id: UUID | None,
        extractor_version: str = GRAPH_EXTRACTOR_VERSION,
        force_rebuild: bool = False,
    ) -> GraphConfigSnapshot:
        self._access_policy.require_workspace(context, context.workspace_id)

        async def persist(uow: UnitOfWork) -> GraphConfigSnapshot:
            _require_scope(uow, context)
            return await uow.graph.configure(
                kb_id,
                chat_profile_revision_id=chat_profile_revision_id,
                enabled=enabled,
                extractor_version=extractor_version,
                force_rebuild=force_rebuild,
            )

        return await execute_in_transaction(self._unit_of_work, persist)

    async def retry(
        self,
        context: AuthContext,
        kb_id: UUID,
        *,
        force_rebuild: bool = False,
    ) -> GraphConfigSnapshot:
        self._access_policy.require_workspace(context, context.workspace_id)

        async def persist(uow: UnitOfWork) -> GraphConfigSnapshot:
            _require_scope(uow, context)
            return await uow.graph.retry(
                kb_id,
                extractor_version=GRAPH_EXTRACTOR_VERSION,
                force_rebuild=force_rebuild,
            )

        return await execute_in_transaction(self._unit_of_work, persist)


class GraphExtractionWorker:
    """Process exactly one preflight, chunk, or finalize work item."""

    def __init__(
        self,
        unit_of_work: UnitOfWorkFactory,
        chat_model: ChatModelAdapter,
        *,
        max_output_tokens: int = 8192,
    ) -> None:
        if not 256 <= max_output_tokens <= 8192:
            raise ValueError("Graph extractor output limit is invalid")
        self._unit_of_work = unit_of_work
        self._chat_model = chat_model
        self._max_output_tokens = max_output_tokens

    async def process_next_work_item(self) -> bool:
        work = await execute_in_transaction(
            self._unit_of_work,
            lambda uow: uow.graph.next_work_item(),
            purpose=UnitOfWorkPurpose.CLAIM,
        )
        if work is None:
            return False
        await self.process_work_item(work)
        return True

    async def process_work_item(self, work: GraphWorkItem) -> None:
        """Execute a scanner result without claiming a second item."""

        if work.kind is GraphWorkKind.PREFLIGHT:
            await self._preflight(work)
        elif work.kind is GraphWorkKind.CHUNK:
            assert work.chunk is not None
            await self._extract_chunk(work)
        else:
            await self._finalize(work)

    async def _preflight(self, work: GraphWorkItem) -> None:
        revision_id = _require_revision(work)
        try:
            response = await self._complete(
                revision_id,
                _PREFLIGHT_TEXT,
                purpose="preflight",
                knowledge_base_id=work.config.knowledge_base_id,
            )
            _raise_if_truncated(response)
            extraction = parse_graph_extraction(response.content, _PREFLIGHT_TEXT)
            if (
                extraction.result_status is not GraphChunkResultStatus.EXTRACTED
                or len(extraction.mentions) < 2
                or not extraction.relations
            ):
                raise GraphProtocolError("preflight_requires_grounded_relation")
        except GraphResourceLimitError:
            await self._mark_failed(work, "preflight_resource_limit")
            return
        except GraphProtocolError as error:
            log_event(
                _LOGGER,
                "graph_extraction_protocol",
                phase="preflight",
                outcome="failed",
                error_code=error.code,
                knowledge_base_id=work.config.knowledge_base_id,
            )
            await self._mark_failed(work, "preflight_protocol_invalid")
            return
        except ChatModelExecutionError:
            await self._mark_failed(work, ErrorCode.GRAPH_PROVIDER_UNAVAILABLE.value)
            return
        except Exception:
            await self._mark_failed(work, ErrorCode.GRAPH_BUILD_FAILED.value)
            return

        await execute_in_transaction(
            self._unit_of_work,
            lambda uow: uow.graph.save_preflight_success(
                work.config.knowledge_base_id,
                build_id=work.config.build_id,
                extractor_version=work.config.extractor_version,
            ),
            purpose=UnitOfWorkPurpose.INDEXING,
        )

    async def _extract_chunk(self, work: GraphWorkItem) -> None:
        chunk = work.chunk
        assert chunk is not None
        if len(chunk.content) > 32_000:
            extraction = GraphChunkExtraction(
                result_status=GraphChunkResultStatus.SKIPPED_RESOURCE,
                error_code="input_chars",
            )
            await self._save_extraction(work, extraction)
            return
        revision_id = _require_revision(work)
        trace_id = str(uuid4())
        try:
            response = await self._complete(
                revision_id,
                chunk.content,
                purpose="chunk",
                trace_id=trace_id,
                knowledge_base_id=work.config.knowledge_base_id,
            )
            try:
                _raise_if_truncated(response)
                extraction = parse_graph_extraction(response.content, chunk.content)
            except GraphProtocolError as initial_error:
                log_event(
                    _LOGGER,
                    "graph_extraction_protocol",
                    phase="initial",
                    outcome="failed",
                    error_code=initial_error.code,
                    trace_id=trace_id,
                    knowledge_base_id=work.config.knowledge_base_id,
                )
                repair = await self._complete(
                    revision_id,
                    chunk.content,
                    purpose="repair",
                    repair_error_code=initial_error.code,
                    trace_id=trace_id,
                    knowledge_base_id=work.config.knowledge_base_id,
                )
                _raise_if_truncated(repair)
                try:
                    extraction = parse_graph_extraction(repair.content, chunk.content)
                except GraphProtocolError as repair_error:
                    log_event(
                        _LOGGER,
                        "graph_extraction_protocol",
                        phase="repair",
                        outcome="failed",
                        error_code=repair_error.code,
                        trace_id=trace_id,
                        knowledge_base_id=work.config.knowledge_base_id,
                    )
                    raise
                if extraction.result_status is GraphChunkResultStatus.EMPTY:
                    raise GraphProtocolError("repair_empty_after_protocol_error")
        except GraphResourceLimitError as error:
            extraction = GraphChunkExtraction(
                result_status=GraphChunkResultStatus.SKIPPED_RESOURCE,
                error_code=error.code,
            )
        except GraphProtocolError as error:
            extraction = GraphChunkExtraction(
                result_status=GraphChunkResultStatus.SKIPPED_PROTOCOL,
                error_code=error.code,
            )
        except ChatModelExecutionError:
            await self._mark_failed(work, ErrorCode.GRAPH_PROVIDER_UNAVAILABLE.value)
            return
        except Exception:
            await self._mark_failed(work, ErrorCode.GRAPH_BUILD_FAILED.value)
            return
        log_event(
            _LOGGER,
            "graph_extraction_final",
            phase="final",
            outcome=extraction.result_status.value,
            error_code=extraction.error_code,
            trace_id=trace_id,
            knowledge_base_id=work.config.knowledge_base_id,
        )
        await self._save_extraction(work, extraction)

    async def _save_extraction(
        self, work: GraphWorkItem, extraction: GraphChunkExtraction
    ) -> None:
        assert work.chunk is not None
        await execute_in_transaction(
            self._unit_of_work,
            lambda uow: uow.graph.save_chunk_extraction(
                kb_id=work.config.knowledge_base_id,
                build_id=work.config.build_id,
                index_chunk_id=work.chunk.index_chunk_id,
                content_hash=work.chunk.content_hash,
                extractor_version=work.config.extractor_version,
                extraction=extraction,
            ),
            purpose=UnitOfWorkPurpose.INDEXING,
        )

    async def _finalize(self, work: GraphWorkItem) -> None:
        await execute_in_transaction(
            self._unit_of_work,
            lambda uow: uow.graph.finalize_if_complete(
                work.config.knowledge_base_id,
                build_id=work.config.build_id,
                observed_at=datetime.now(UTC),
            ),
            purpose=UnitOfWorkPurpose.INDEXING,
        )

    async def _mark_failed(self, work: GraphWorkItem, error_code: str) -> None:
        await execute_in_transaction(
            self._unit_of_work,
            lambda uow: uow.graph.mark_failed(
                work.config.knowledge_base_id,
                build_id=work.config.build_id,
                error_code=error_code,
            ),
            purpose=UnitOfWorkPurpose.INDEXING,
        )

    async def _complete(
        self,
        revision_id: UUID,
        chunk_text: str,
        *,
        purpose: str,
        repair_error_code: str | None = None,
        trace_id: str | None = None,
        knowledge_base_id: UUID | None = None,
    ):
        user_content = chunk_text
        if repair_error_code is not None:
            repair_family = graph_protocol_error_family(repair_error_code)
            repair_rule = _REPAIR_RULES.get(
                repair_family,
                "Regenerate one response that follows the fixed extraction protocol.",
            )
            user_content = (
                "Return a corrected JSON object only. Protocol error code: "
                + repair_error_code
                + ". Rule: "
                + repair_rule
                + " Preserve every other item that already satisfies the protocol. "
                + "Follow this JSON shape: "
                + _JSON_SKELETON
                + "\nChunk:\n"
                + chunk_text
            )
        started_at = perf_counter()
        response = await self._chat_model.complete(
            ChatModelRequest(
                messages=(
                    ChatModelMessage(role="system", content=_GRAPH_SYSTEM_PROMPT),
                    ChatModelMessage(
                        role="user",
                        content=(
                            f"Extraction operation: {purpose}.\n"
                            "Use only the following chunk:\n"
                            + user_content
                        ),
                    ),
                ),
                max_output_tokens=self._max_output_tokens,
                model_profile_revision_id=revision_id,
                thinking_enabled=None,
                response_format=GRAPH_RESPONSE_FORMAT,
            )
        )
        fields: dict[str, object] = {
            "phase": "repair" if repair_error_code is not None else "initial",
            "outcome": "completed",
            "duration_ms": max(0, int((perf_counter() - started_at) * 1000)),
        }
        if trace_id is not None:
            fields["trace_id"] = trace_id
        if knowledge_base_id is not None:
            fields["knowledge_base_id"] = knowledge_base_id
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if key in response.usage:
                fields[key] = response.usage[key]
        log_event(_LOGGER, "graph_extraction_model_call", **fields)
        return response


def _raise_if_truncated(response) -> None:
    finish_reason = (response.finish_reason or "").strip().casefold()
    if (
        finish_reason in _TRUNCATED_FINISH_REASONS
        or response.content.strip() == _TRUNCATED_RESPONSE_MARKER
    ):
        raise GraphResourceLimitError("output_truncated")


def _require_revision(work: GraphWorkItem) -> UUID:
    if work.config.chat_profile_revision_id is None:
        raise ResourceStateConflictError("Graph Chat Profile Revision is missing")
    return work.config.chat_profile_revision_id


def _require_scope(uow: UnitOfWork, context: AuthContext) -> None:
    if uow.workspace_id != context.workspace_id:
        raise ResourceNotFoundError("resource was not found")


async def _profile_bundle(uow: UnitOfWork, snapshot: GraphConfigSnapshot):
    if snapshot.chat_profile_revision_id is None:
        return None
    return await uow.model_settings.get_profile_revision(
        snapshot.chat_profile_revision_id
    )


def _config_view(snapshot: GraphConfigSnapshot, bundle) -> GraphConfigView:
    if bundle is None:
        return GraphConfigView(snapshot)
    return GraphConfigView(
        snapshot=snapshot,
        profile_name=bundle.profile.name,
        provider_name=bundle.provider.name,
        model=bundle.current_revision.model,
    )

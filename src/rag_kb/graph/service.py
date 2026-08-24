"""Graph configuration and Graphiti build work-item services."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
from uuid import UUID

from rag_kb.auth import AccessPolicy, AuthContext
from rag_kb.domain import (
    GRAPH_EXTRACTOR_VERSION,
    GraphConfigSnapshot,
    GraphitiBuildSnapshot,
    GRAPH_WORK_HEARTBEAT_SECONDS,
    GraphWorkItem,
    GraphWorkKind,
    ResourceNotFoundError,
)
from rag_kb.observability import get_logger, log_event, log_exception
from rag_kb.ports.graphiti import GraphitiGraph
from rag_kb.graph.schema_profiles import get_graph_schema_registry
from rag_kb.uow import (
    UnitOfWork,
    UnitOfWorkFactory,
    UnitOfWorkPurpose,
    execute_in_transaction,
)


LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class GraphConfigView:
    snapshot: GraphConfigSnapshot
    profile_name: str | None = None
    provider_name: str | None = None
    model: str | None = None
    schema_profile_name: str = "Generic open-domain knowledge"


class GraphConfigurationService:
    def __init__(
        self,
        unit_of_work: UnitOfWorkFactory,
        access_policy: AccessPolicy,
        graphiti_graph: GraphitiGraph,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._access_policy = access_policy
        self._graphiti_graph = graphiti_graph

    async def get(self, context: AuthContext, kb_id: UUID) -> GraphConfigSnapshot:
        self._access_policy.require_workspace(context, context.workspace_id)

        async def load(uow: UnitOfWork) -> GraphConfigSnapshot:
            _require_scope(uow, context)
            return await uow.graph.ensure_config(kb_id)

        return await execute_in_transaction(
            self._unit_of_work,
            load,
            purpose=UnitOfWorkPurpose.REQUEST,
        )

    async def get_view(self, context: AuthContext, kb_id: UUID) -> GraphConfigView:
        self._access_policy.require_workspace(context, context.workspace_id)

        async def load(uow: UnitOfWork) -> GraphConfigView:
            _require_scope(uow, context)
            snapshot = await uow.graph.ensure_config(kb_id)
            return _config_view(snapshot, await _profile_bundle(uow, snapshot))

        return await execute_in_transaction(
            self._unit_of_work,
            load,
            purpose=UnitOfWorkPurpose.REQUEST,
        )

    async def configure(
        self,
        context: AuthContext,
        kb_id: UUID,
        *,
        enabled: bool,
        chat_profile_revision_id: UUID | None,
        schema_profile_key: str | None = None,
        extractor_version: str = GRAPH_EXTRACTOR_VERSION,
        force_rebuild: bool = False,
    ) -> GraphConfigSnapshot:
        self._access_policy.require_workspace(context, context.workspace_id)

        async def persist(
            uow: UnitOfWork,
        ) -> tuple[GraphConfigSnapshot, tuple[GraphitiBuildSnapshot, ...]]:
            _require_scope(uow, context)
            snapshot = await uow.graph.configure(
                kb_id,
                chat_profile_revision_id=chat_profile_revision_id,
                enabled=enabled,
                schema_profile_key=schema_profile_key,
                extractor_version=extractor_version,
                force_rebuild=force_rebuild,
            )
            return snapshot, uow.graph.take_retired_graphiti_builds()

        snapshot, retired = await execute_in_transaction(
            self._unit_of_work, persist
        )
        await _recycle_graphiti_graphs(self._graphiti_graph, retired)
        return snapshot

    async def retry(
        self,
        context: AuthContext,
        kb_id: UUID,
        *,
        force_rebuild: bool = False,
    ) -> GraphConfigSnapshot:
        self._access_policy.require_workspace(context, context.workspace_id)

        async def persist(
            uow: UnitOfWork,
        ) -> tuple[GraphConfigSnapshot, tuple[GraphitiBuildSnapshot, ...]]:
            _require_scope(uow, context)
            snapshot = await uow.graph.retry(
                kb_id,
                extractor_version=GRAPH_EXTRACTOR_VERSION,
                force_rebuild=force_rebuild,
            )
            return snapshot, uow.graph.take_retired_graphiti_builds()

        snapshot, retired = await execute_in_transaction(
            self._unit_of_work, persist
        )
        await _recycle_graphiti_graphs(self._graphiti_graph, retired)
        return snapshot


class GraphExtractionWorker:
    """Process exactly one Graphiti preflight, episode, or finalize item."""

    def __init__(
        self,
        unit_of_work: UnitOfWorkFactory,
        graphiti_graph: GraphitiGraph,
        *,
        worker_id: str | None = None,
        heartbeat_interval_seconds: float = GRAPH_WORK_HEARTBEAT_SECONDS,
    ) -> None:
        if heartbeat_interval_seconds <= 0:
            raise ValueError("Graph work heartbeat interval must be positive")
        self._unit_of_work = unit_of_work
        self._graphiti_graph = graphiti_graph
        self._worker_id = worker_id
        self._heartbeat_interval_seconds = heartbeat_interval_seconds

    async def recycle_retired(
        self, builds: tuple[GraphitiBuildSnapshot, ...]
    ) -> None:
        await _recycle_graphiti_graphs(self._graphiti_graph, builds)

    async def process_next_work_item(self) -> bool:
        async def claim(
            uow: UnitOfWork,
        ) -> tuple[GraphWorkItem | None, tuple[GraphitiBuildSnapshot, ...]]:
            if self._worker_id is None:
                work = await uow.graph.next_work_item()
            else:
                work = await uow.graph.next_work_item(worker_id=self._worker_id)
            return work, uow.graph.take_retired_graphiti_builds()

        work, retired = await execute_in_transaction(
            self._unit_of_work,
            claim,
            purpose=UnitOfWorkPurpose.CLAIM,
        )
        await _recycle_graphiti_graphs(self._graphiti_graph, retired)
        if work is None:
            return False
        await self.process_work_item(work)
        return True

    async def process_work_item(self, work: GraphWorkItem) -> None:
        stop_heartbeat = asyncio.Event()
        heartbeat_task = (
            asyncio.create_task(self._heartbeat(work, stop_heartbeat))
            if work.lease_token is not None
            else None
        )
        try:
            build = await execute_in_transaction(
                self._unit_of_work,
                lambda uow: uow.graph.get_graphiti_build(
                    work.config.knowledge_base_id,
                    build_id=work.config.build_id,
                ),
                purpose=UnitOfWorkPurpose.REQUEST,
            )
            if build is None:
                await self._mark_failed(
                    work,
                    f"{_graph_build_phase_code(work.kind)}:build_missing",
                )
                return
            if work.kind is GraphWorkKind.PREFLIGHT:
                if not await self._graphiti_graph.probe(build):
                    raise RuntimeError("Graphiti preflight probe failed")
                log_event(
                    LOGGER,
                    "graphiti_probe",
                    build_id=build.build_id,
                    knowledge_base_id=work.config.knowledge_base_id,
                    operation="preflight",
                    outcome="ok",
                )
                await execute_in_transaction(
                    self._unit_of_work,
                    lambda uow: uow.graph.save_preflight_success(
                        kb_id=work.config.knowledge_base_id,
                        build_id=build.build_id,
                        extractor_version=build.extractor_version,
                        **_lease_kwargs(work),
                    ),
                    purpose=UnitOfWorkPurpose.INDEXING,
                )
                return

            if work.kind is GraphWorkKind.CHUNK:
                chunk = work.chunk
                assert chunk is not None
                episode_uuid = await self._graphiti_graph.add_episode(build, chunk)
                saved = await execute_in_transaction(
                    self._unit_of_work,
                    lambda uow: uow.graph.save_graphiti_episode(
                        kb_id=work.config.knowledge_base_id,
                        build_id=build.build_id,
                        index_chunk_id=chunk.index_chunk_id,
                        content_hash=chunk.content_hash,
                        episode_uuid=episode_uuid,
                        **_lease_kwargs(work),
                    ),
                    purpose=UnitOfWorkPurpose.INDEXING,
                )
                if not saved:
                    log_event(
                        LOGGER,
                        "graphiti_episode_write",
                        build_id=build.build_id,
                        knowledge_base_id=work.config.knowledge_base_id,
                        operation="chunk",
                        outcome="skipped",
                    )
                    return
                log_event(
                    LOGGER,
                    "graphiti_episode_write",
                    build_id=build.build_id,
                    knowledge_base_id=work.config.knowledge_base_id,
                    operation="chunk",
                    outcome="ok",
                )
                return

            episode_uuid = await execute_in_transaction(
                self._unit_of_work,
                lambda uow: uow.graph.first_graphiti_episode_uuid(
                    work.config.knowledge_base_id,
                    build_id=build.build_id,
                ),
                purpose=UnitOfWorkPurpose.REQUEST,
            )
            if build.expected_episode_count and episode_uuid is None:
                raise RuntimeError("Graphiti ready probe has no episode mapping")
            if not await self._graphiti_graph.probe(
                build,
                episode_uuid=episode_uuid,
                require_complete=True,
            ):
                raise RuntimeError("Graphiti ready probe failed")
            log_event(
                LOGGER,
                "graphiti_probe",
                build_id=build.build_id,
                knowledge_base_id=work.config.knowledge_base_id,
                operation="finalize",
                outcome="ok",
            )

            async def finalize(
                uow: UnitOfWork,
            ) -> tuple[GraphConfigSnapshot | None, tuple[GraphitiBuildSnapshot, ...]]:
                snapshot = await uow.graph.finalize_graphiti_if_complete(
                    work.config.knowledge_base_id,
                    build_id=build.build_id,
                    **_lease_kwargs(work),
                )
                return snapshot, uow.graph.take_retired_graphiti_builds()

            _, retired = await execute_in_transaction(
                self._unit_of_work,
                finalize,
                purpose=UnitOfWorkPurpose.INDEXING,
            )
            await _recycle_graphiti_graphs(self._graphiti_graph, retired)
        except Exception as error:
            error_code = _graph_build_error_code(work.kind, error)
            log_exception(
                LOGGER,
                "graphiti_build_failed",
                error,
                build_id=work.config.build_id,
                knowledge_base_id=work.config.knowledge_base_id,
                operation=work.kind.value,
                error_code=error_code,
            )
            await self._mark_failed(work, error_code)
        finally:
            if heartbeat_task is not None:
                stop_heartbeat.set()
                heartbeat_task.cancel()
                await asyncio.gather(heartbeat_task, return_exceptions=True)
            await self._release(work)

    async def _mark_failed(self, work: GraphWorkItem, error_code: str) -> None:
        async def persist(uow: UnitOfWork) -> tuple[GraphitiBuildSnapshot, ...]:
            await uow.graph.mark_failed(
                work.config.knowledge_base_id,
                build_id=work.config.build_id,
                error_code=error_code,
                **_lease_kwargs(work),
            )
            return uow.graph.take_retired_graphiti_builds()

        retired = await execute_in_transaction(
            self._unit_of_work,
            persist,
            purpose=UnitOfWorkPurpose.INDEXING,
        )
        await _recycle_graphiti_graphs(self._graphiti_graph, retired)

    async def _heartbeat(
        self,
        work: GraphWorkItem,
        stopped: asyncio.Event,
    ) -> None:
        while not stopped.is_set():
            try:
                await asyncio.wait_for(
                    stopped.wait(), timeout=self._heartbeat_interval_seconds
                )
                return
            except TimeoutError:
                pass
            try:
                owned = await execute_in_transaction(
                    self._unit_of_work,
                    lambda uow: uow.graph.heartbeat_graph_work(
                        work,
                        observed_at=datetime.now(UTC),
                    ),
                    purpose=UnitOfWorkPurpose.HEARTBEAT,
                )
            except Exception as error:
                log_exception(
                    LOGGER,
                    "graphiti_work_lease_heartbeat_failed",
                    error,
                    build_id=work.config.build_id,
                    knowledge_base_id=work.config.knowledge_base_id,
                    operation=work.kind.value,
                )
                continue
            if not owned:
                log_event(
                    LOGGER,
                    "graphiti_work_lease_lost",
                    build_id=work.config.build_id,
                    knowledge_base_id=work.config.knowledge_base_id,
                    operation=work.kind.value,
                )
                return

    async def _release(self, work: GraphWorkItem) -> None:
        if work.lease_token is None:
            return
        try:
            await execute_in_transaction(
                self._unit_of_work,
                lambda uow: uow.graph.release_graph_work(work),
                purpose=UnitOfWorkPurpose.RECONCILIATION,
            )
        except Exception as error:
            log_exception(
                LOGGER,
                "graphiti_work_lease_release_failed",
                error,
                build_id=work.config.build_id,
                knowledge_base_id=work.config.knowledge_base_id,
                operation=work.kind.value,
            )


async def _recycle_graphiti_graphs(
    graphiti_graph: GraphitiGraph,
    builds: tuple[GraphitiBuildSnapshot, ...],
) -> None:
    for build in builds:
        try:
            await graphiti_graph.delete_graph(build)
        except Exception as error:
            log_exception(
                LOGGER,
                "graphiti_build_failed",
                error,
                build_id=build.build_id,
                knowledge_base_id=build.knowledge_base_id,
                operation="recycle",
                error_code="graph_recycle_failed",
            )


def _require_scope(uow: UnitOfWork, context: AuthContext) -> None:
    if uow.workspace_id != context.workspace_id:
        raise ResourceNotFoundError("resource was not found")


def _graph_build_phase_code(kind: GraphWorkKind) -> str:
    return {
        GraphWorkKind.PREFLIGHT: "graphiti_preflight_failed",
        GraphWorkKind.CHUNK: "graphiti_episode_extraction_failed",
        GraphWorkKind.FINALIZE: "graphiti_finalize_failed",
    }[kind]


def _graph_build_error_code(kind: GraphWorkKind, error: BaseException) -> str:
    """Persist a phase plus a content-free, stable exception-class fingerprint."""

    chain: list[str] = []
    current: BaseException | None = error
    observed: set[int] = set()
    while current is not None and id(current) not in observed and len(chain) < 4:
        observed.add(id(current))
        error_type = type(current)
        chain.append(f"{error_type.__module__}.{error_type.__qualname__}")
        current = current.__cause__ or current.__context__
    fingerprint = hashlib.sha256("|".join(chain).encode("utf-8")).hexdigest()[:16]
    return f"{_graph_build_phase_code(kind)}:{fingerprint}"


def _lease_kwargs(work: GraphWorkItem) -> dict[str, UUID]:
    if work.lease_token is None:
        return {}
    return {"lease_token": work.lease_token}


async def _profile_bundle(uow: UnitOfWork, snapshot: GraphConfigSnapshot):
    if snapshot.chat_profile_revision_id is None:
        return None
    return await uow.model_settings.get_profile_revision(
        snapshot.chat_profile_revision_id
    )


def _config_view(snapshot: GraphConfigSnapshot, bundle) -> GraphConfigView:
    schema_profile = get_graph_schema_registry().resolve(
        snapshot.schema_profile_key,
        digest=snapshot.schema_profile_digest,
        extractor_version=snapshot.extractor_version,
    )
    if bundle is None:
        return GraphConfigView(snapshot, schema_profile_name=schema_profile.display_name)
    return GraphConfigView(
        snapshot=snapshot,
        profile_name=bundle.profile.name,
        provider_name=bundle.provider.name,
        model=bundle.current_revision.model,
        schema_profile_name=schema_profile.display_name,
    )

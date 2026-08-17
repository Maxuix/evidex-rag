"""Graph configuration and Graphiti build work-item services."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from rag_kb.auth import AccessPolicy, AuthContext
from rag_kb.domain import (
    GRAPH_EXTRACTOR_VERSION,
    ErrorCode,
    GraphConfigSnapshot,
    GraphitiBuildSnapshot,
    GraphWorkItem,
    GraphWorkKind,
    ResourceNotFoundError,
)
from rag_kb.observability import get_logger, log_event, log_exception
from rag_kb.ports.graphiti import GraphitiGraph
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
    ) -> None:
        self._unit_of_work = unit_of_work
        self._graphiti_graph = graphiti_graph

    async def recycle_retired(
        self, builds: tuple[GraphitiBuildSnapshot, ...]
    ) -> None:
        await _recycle_graphiti_graphs(self._graphiti_graph, builds)

    async def process_next_work_item(self) -> bool:
        async def claim(
            uow: UnitOfWork,
        ) -> tuple[GraphWorkItem | None, tuple[GraphitiBuildSnapshot, ...]]:
            work = await uow.graph.next_work_item()
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
        build = await execute_in_transaction(
            self._unit_of_work,
            lambda uow: uow.graph.get_graphiti_build(
                work.config.knowledge_base_id,
                build_id=work.config.build_id,
            ),
            purpose=UnitOfWorkPurpose.REQUEST,
        )
        if build is None:
            await self._mark_failed(work, ErrorCode.GRAPH_BUILD_FAILED.value)
            return
        try:
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
                        work.config.knowledge_base_id,
                        build_id=build.build_id,
                        extractor_version=build.extractor_version,
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
                )
                return snapshot, uow.graph.take_retired_graphiti_builds()

            _, retired = await execute_in_transaction(
                self._unit_of_work,
                finalize,
                purpose=UnitOfWorkPurpose.INDEXING,
            )
            await _recycle_graphiti_graphs(self._graphiti_graph, retired)
        except Exception as error:
            log_exception(
                LOGGER,
                "graphiti_build_failed",
                error,
                build_id=work.config.build_id,
                knowledge_base_id=work.config.knowledge_base_id,
                operation=work.kind.value,
                error_code=ErrorCode.GRAPH_BUILD_FAILED.value,
            )
            await self._mark_failed(work, ErrorCode.GRAPH_BUILD_FAILED.value)

    async def _mark_failed(self, work: GraphWorkItem, error_code: str) -> None:
        async def persist(uow: UnitOfWork) -> tuple[GraphitiBuildSnapshot, ...]:
            await uow.graph.mark_failed(
                work.config.knowledge_base_id,
                build_id=work.config.build_id,
                error_code=error_code,
            )
            return uow.graph.take_retired_graphiti_builds()

        retired = await execute_in_transaction(
            self._unit_of_work,
            persist,
            purpose=UnitOfWorkPurpose.INDEXING,
        )
        await _recycle_graphiti_graphs(self._graphiti_graph, retired)


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

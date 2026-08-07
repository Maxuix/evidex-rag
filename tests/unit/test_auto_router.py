from __future__ import annotations

import json
import unittest
from dataclasses import replace

from rag_kb.domain import (
    ChatModelExecutionError,
    ChatPipelineExecutionError,
    ChatPipelinePhase,
    ChatResolvedMode,
    ChatRouteReason,
    ChatRouteStatus,
    ChatWorkflowMode,
    ContextualizedQuery,
    ErrorCode,
    QueryContextStatus,
    QueryRewriteSource,
    initial_chat_workflow,
)
from rag_kb.retrieval.router import (
    AutoWorkflowRouter,
    _repair_route_request,
    _route_request,
)
from tests.unit.test_answering import _Model, _context, _pack, _response


class _Retriever:
    def __init__(self, pack, *, error=None) -> None:
        self.pack = pack
        self.error = error
        self.calls = 0

    async def retrieve_query(self, context, query, *, top_k_override=None):
        del context, query
        self.calls += 1
        if self.error is not None:
            raise self.error
        self.last_top_k = top_k_override
        return self.pack


class _Store:
    def __init__(self) -> None:
        self.values = []

    async def persist_resolution(self, context, value):
        del context
        self.values.append(value)
        return value


class _UnavailableModel:
    async def complete(self, request):
        del request
        raise ChatModelExecutionError(
            ErrorCode.CHAT_PROVIDER_UNAVAILABLE,
            diagnostic={"check": "transport"},
        )


def _workflow_context(mode: ChatWorkflowMode):
    context = _context()
    configuration, state = initial_chat_workflow(mode)
    return replace(
        context,
        workflow_configuration=configuration.as_dict(),
        workflow_state=state.as_dict(),
    )


def _query_context(context):
    return ContextualizedQuery(
        version="contextual_query_v2",
        status=QueryContextStatus.ORIGINAL,
        original_query=context.query,
        standalone_query=context.query,
        context_hash=context.conversation_context.content_hash,
        rewrite_source=QueryRewriteSource.ORIGINAL,
    )


def _route(mode: str, reasons) -> str:
    return json.dumps(
        {
            "version": "auto_route_v1",
            "mode": mode,
            "reason_codes": list(reasons),
        }
    )


class AutoWorkflowRouterTests(unittest.IsolatedAsyncioTestCase):
    async def test_json_mode_requests_explicitly_require_json(self) -> None:
        context = _workflow_context(ChatWorkflowMode.AUTO)
        request = _route_request(
            context,
            _query_context(context),
            _pack(context, "probe evidence"),
        )
        repair = _repair_route_request(request, "{}")

        for candidate in (request, repair):
            with self.subTest(message_count=len(candidate.messages)):
                self.assertIn(
                    "json",
                    "\n".join(
                        message.content for message in candidate.messages
                    ).lower(),
                )
        for field in ("version", "mode", "reason_codes"):
            self.assertIn(field, request.messages[0].content)

    async def test_only_auto_probes_and_routes(self) -> None:
        for mode in (ChatWorkflowMode.SIMPLE, ChatWorkflowMode.AGENT):
            with self.subTest(mode=mode):
                context = _workflow_context(mode)
                retriever = _Retriever(_pack(context, "probe"))
                store = _Store()
                router = AutoWorkflowRouter(_Model(), retriever, store)

                state, calls = await router.resolve(
                    context, _query_context(context)
                )

                self.assertEqual(state.resolved_mode.value, mode.value)
                self.assertEqual(calls, ())
                self.assertEqual(retriever.calls, 0)
                self.assertEqual(store.values, [])

    async def test_auto_agent_resolution_is_persisted(self) -> None:
        context = _workflow_context(ChatWorkflowMode.AUTO)
        retriever = _Retriever(_pack(context, "probe evidence"))
        store = _Store()
        model = _Model(
            _response(
                _route("agent", ("multi_hop_required",)),
                request_id="route",
            )
        )

        state, calls = await AutoWorkflowRouter(
            model, retriever, store
        ).resolve(context, _query_context(context))

        self.assertEqual(state.resolved_mode, ChatResolvedMode.AGENT)
        self.assertEqual(state.route_status, ChatRouteStatus.RESOLVED)
        self.assertEqual(len(calls), 1)
        self.assertEqual(retriever.calls, 1)
        self.assertEqual(store.values, [state])

    async def test_invalid_route_repairs_once_then_falls_back_visibly(self) -> None:
        context = _workflow_context(ChatWorkflowMode.AUTO)
        retriever = _Retriever(_pack(context, "probe"))
        store = _Store()
        model = _Model(
            _response("not-json", request_id="invalid"),
            _response('{"still":"invalid"}', request_id="repair"),
        )

        state, calls = await AutoWorkflowRouter(
            model, retriever, store
        ).resolve(context, _query_context(context))

        self.assertEqual(state.resolved_mode, ChatResolvedMode.SIMPLE)
        self.assertEqual(state.route_status, ChatRouteStatus.FALLBACK)
        self.assertEqual(state.route_reason_codes[0].value, "router_invalid")
        self.assertEqual(len(calls), 2)

    async def test_provider_failure_falls_back_but_probe_consistency_does_not(self) -> None:
        context = _workflow_context(ChatWorkflowMode.AUTO)
        retriever = _Retriever(_pack(context, "probe"))
        store = _Store()
        state, calls = await AutoWorkflowRouter(
            _UnavailableModel(), retriever, store
        ).resolve(context, _query_context(context))
        self.assertEqual(state.route_status, ChatRouteStatus.FALLBACK)
        self.assertEqual(state.route_reason_codes[0].value, "router_unavailable")
        self.assertEqual(calls, ())

        consistency = ChatPipelineExecutionError(
            ErrorCode.CHAT_REVISION_MISMATCH,
            phase=ChatPipelinePhase.RETRIEVE_EVIDENCE,
            diagnostic={"check": "frozen_revision"},
        )
        failing = _Retriever(_pack(context), error=consistency)
        with self.assertRaises(ChatPipelineExecutionError) as raised:
            await AutoWorkflowRouter(_Model(), failing, _Store()).resolve(
                context, _query_context(context)
            )
        self.assertIs(raised.exception, consistency)

    async def test_retry_reuses_the_persisted_resolution(self) -> None:
        context = _workflow_context(ChatWorkflowMode.AUTO)
        _, pending = initial_chat_workflow(ChatWorkflowMode.AUTO)
        resolved = replace(
            pending,
            resolved_mode=ChatResolvedMode.SIMPLE,
            route_status=ChatRouteStatus.FALLBACK,
            route_reason_codes=(ChatRouteReason.ROUTER_UNAVAILABLE,),
        )
        context = replace(context, workflow_state=resolved.as_dict())
        retriever = _Retriever(_pack(context, "unused"))

        state, calls = await AutoWorkflowRouter(
            _Model(), retriever, _Store()
        ).resolve(context, _query_context(context))

        self.assertEqual(state, resolved)
        self.assertEqual(calls, ())
        self.assertEqual(retriever.calls, 0)

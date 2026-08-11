"""Deterministic evidence admission and grounded generation pipeline steps."""

from __future__ import annotations

import asyncio
import json

from rag_kb.answering.model_execution import (
    complete_model,
    complete_model_streaming,
    model_call_record,
    require_frozen_model,
)
from rag_kb.answering.preview import (
    NoOpChatPreviewSink,
    PartialAnswerPreviewProjector,
    emit_preview_delta_safely,
    emit_preview_reset_safely,
)
from rag_kb.answering.prompt_builder import (
    build_evidence_envelope,
    build_generation_request,
    serialize_final_llm_context,
)
from rag_kb.domain import (
    AnswerControlReason,
    AnswerDraftCandidate,
    AnswerDraftSource,
    AnswerOutcome,
    AnswerStyle,
    ChatAnsweringState,
    ChatExecutionContext,
    ChatModelCallRecord,
    ChatModelOperation,
    ChatPipelineExecutionError,
    ChatPipelinePhase,
    ChatPipelineState,
    ChatPreviewResetReason,
    EvidenceAssessment,
    EvidenceCoverage,
    EvidenceEnvelope,
    EvidencePack,
    EvidenceScoreKind,
    ErrorCode,
    InsufficiencyPolicy,
    ChatWorkflowState,
    ResearchStatus,
)
from rag_kb.ports.chat_preview import ChatPreviewSink
from rag_kb.ports.model_api import ChatModelAdapter
from rag_kb.retrieval.calculator import (
    CALCULATION_FACTS_ARTIFACT,
    DecimalCalculationFact,
)
from rag_kb.retrieval.eligibility import EvidenceEligibilityPolicy


WORKFLOW_STATE_ARTIFACT = "chat_workflow_state"
WORKFLOW_MODEL_CALLS_ARTIFACT = "chat_workflow_model_calls"


class CosineEvidenceAssessmentStep:
    """Admit evidence using rerank quality with a semantic-similarity floor."""

    def __init__(
        self,
        min_cosine_similarity: float,
        min_rerank_score: float = 0.45,
        cross_modal_min_cosine_similarity: float = 0.25,
    ) -> None:
        if not -1.0 <= min_cosine_similarity <= 1.0:
            raise ValueError("min_cosine_similarity must be between -1 and 1")
        if not 0.0 <= min_rerank_score <= 1.0:
            raise ValueError("min_rerank_score must be between 0 and 1")
        if not -1.0 <= cross_modal_min_cosine_similarity <= 1.0:
            raise ValueError(
                "cross_modal_min_cosine_similarity must be between -1 and 1"
            )
        self._policy = EvidenceEligibilityPolicy(
            float(min_cosine_similarity),
            float(min_rerank_score),
            float(cross_modal_min_cosine_similarity),
        )
        self._min_cosine_similarity = float(min_cosine_similarity)
        self._min_rerank_score = float(min_rerank_score)
        self._cross_modal_min_cosine_similarity = float(
            cross_modal_min_cosine_similarity
        )

    async def run(self, state: ChatPipelineState) -> ChatPipelineState:
        context, pack = _require_inputs(state, ChatPipelinePhase.ASSESS_EVIDENCE)
        evidence = build_evidence_envelope(pack)
        usable_citation_ids = tuple(
            prompt_item.citation_id
            for item, prompt_item in zip(pack.evidence, evidence.items, strict=True)
            if self._policy.usable(item)
        )
        if not usable_citation_ids:
            assessment = EvidenceAssessment(
                coverage=EvidenceCoverage.NONE,
                usable_citation_ids=(),
                supported_aspects=(),
                missing_aspects=(),
            )
        else:
            # Numeric admission only makes evidence eligible for the answer
            # model. The model must still return ``refused`` when no requested
            # fact is directly supported; this is not semantic sufficiency.
            assessment = EvidenceAssessment(
                coverage=EvidenceCoverage.SUFFICIENT,
                usable_citation_ids=usable_citation_ids,
                supported_aspects=("question",),
                missing_aspects=(),
            )
        answering = ChatAnsweringState(
            evidence=evidence,
            assessment=assessment,
            model_calls=(
                state.query_context.model_calls_for_attempt(context.attempt)
                if state.query_context is not None
                else ()
            )
            + _workflow_model_calls(state),
        )
        return ChatPipelineState(
            context=context,
            evidence_pack=pack,
            answering=answering,
            query_context=state.query_context,
            artifacts=state.artifacts,
        )


class AdaptiveEvidenceAssessmentStep(CosineEvidenceAssessmentStep):
    """Combine numeric admission with an independently verified Research Result."""

    async def run(self, state: ChatPipelineState) -> ChatPipelineState:
        context, pack = _require_inputs(state, ChatPipelinePhase.ASSESS_EVIDENCE)
        workflow_state = state.artifacts.get(WORKFLOW_STATE_ARTIFACT)
        if (
            not isinstance(workflow_state, ChatWorkflowState)
            or workflow_state.research_result is None
        ):
            raise _context_error(
                ChatPipelinePhase.ASSESS_EVIDENCE, "research_result_state"
            )
        result = workflow_state.research_result
        evidence = build_evidence_envelope(pack)
        pack_by_chunk_id = {
            item.index_chunk_id: item for item in pack.evidence
        }
        selected = set(result.selected_evidence_keys)
        admitted: dict[str, str] = {}
        for item, prompt in zip(pack.evidence, evidence.items, strict=True):
            key = _evidence_key(item)
            if item.score_kind is not EvidenceScoreKind.ADJACENCY:
                if self._policy.usable(item):
                    admitted[key] = prompt.citation_id
                continue
            anchor = pack_by_chunk_id.get(
                item.adjacency_anchor_index_chunk_id
            )
            if (
                key in selected
                and anchor is not None
                and anchor.score_kind is not EvidenceScoreKind.ADJACENCY
                and item.index_revision_id == anchor.index_revision_id
                and item.indexed_document_version_id
                == anchor.indexed_document_version_id
                and item.document_id == anchor.document_id
                and item.document_version_id == anchor.document_version_id
                and item.ordinal - anchor.ordinal == item.adjacency_offset
                and self._policy.usable(anchor)
            ):
                admitted[key] = prompt.citation_id
        usable_citation_ids = tuple(
            admitted[key]
            for key in result.selected_evidence_keys
            if key in admitted
        )
        supported = result.covered_aspects or (
            ("question",) if usable_citation_ids else ()
        )
        if not usable_citation_ids or result.status in {
            ResearchStatus.NO_EVIDENCE,
            ResearchStatus.PREMISE_UNSUPPORTED,
        }:
            assessment = EvidenceAssessment(
                coverage=EvidenceCoverage.NONE,
                usable_citation_ids=(),
                supported_aspects=(),
                missing_aspects=result.missing_aspects,
            )
        elif result.status is ResearchStatus.SUFFICIENT:
            assessment = EvidenceAssessment(
                coverage=EvidenceCoverage.SUFFICIENT,
                usable_citation_ids=usable_citation_ids,
                supported_aspects=supported,
                missing_aspects=(),
            )
        elif result.status is ResearchStatus.CONFLICT:
            assessment = EvidenceAssessment(
                coverage=EvidenceCoverage.CONFLICT,
                usable_citation_ids=usable_citation_ids,
                supported_aspects=supported,
                missing_aspects=result.conflicts or ("conflict",),
            )
        else:
            assessment = EvidenceAssessment(
                coverage=EvidenceCoverage.PARTIAL,
                usable_citation_ids=usable_citation_ids,
                supported_aspects=supported,
                missing_aspects=result.missing_aspects or ("coverage",),
            )
        answering = ChatAnsweringState(
            evidence=evidence,
            assessment=assessment,
            model_calls=(
                state.query_context.model_calls_for_attempt(context.attempt)
                if state.query_context is not None
                else ()
            )
            + _workflow_model_calls(state),
        )
        return ChatPipelineState(
            context=context,
            evidence_pack=pack,
            answering=answering,
            query_context=state.query_context,
            artifacts=state.artifacts,
        )


class AnswerGenerationStep:
    def __init__(
        self,
        model: ChatModelAdapter,
        *,
        preview_sink: ChatPreviewSink | None = None,
        preview_max_visible_bytes: int = 64 * 1024,
    ) -> None:
        self._model = model
        self._preview_sink = preview_sink or NoOpChatPreviewSink()
        self._preview_max_visible_bytes = preview_max_visible_bytes

    async def run(self, state: ChatPipelineState) -> ChatPipelineState:
        context, pack = _require_inputs(
            state, ChatPipelinePhase.GENERATE_OR_REFUSE
        )
        answering = state.answering
        if answering is None or answering.draft is not None:
            raise _context_error(
                ChatPipelinePhase.GENERATE_OR_REFUSE, "assessment_state"
            )
        _, insufficiency = _require_policy(context)
        route = _route(answering.assessment.coverage, insufficiency)
        if isinstance(route, AnswerControlReason):
            draft = _deterministic_refusal(route)
            calls = answering.model_calls
        else:
            request = build_generation_request(
                context,
                answering.evidence,
                answering.assessment,
                query_context=state.query_context,
                expected_outcome=route,
                visual_content=answering.visual_content,
                calculation_facts=_calculation_facts(state),
            )
            if self._preview_sink.enabled:
                projector = PartialAnswerPreviewProjector(
                    max_visible_bytes=self._preview_max_visible_bytes
                )
                accumulated = ""
                preview_active = True

                async def on_content_delta(delta: str) -> None:
                    nonlocal accumulated, preview_active
                    if not preview_active:
                        return
                    accumulated += delta
                    projection = projector.feed(accumulated)
                    if projection.delta is not None:
                        await emit_preview_delta_safely(
                            self._preview_sink,
                            run_id=context.run_id,
                            attempt=context.attempt,
                            delta=projection.delta,
                        )
                    elif projection.invalidated:
                        await emit_preview_reset_safely(
                            self._preview_sink,
                            run_id=context.run_id,
                            attempt=context.attempt,
                            reason=ChatPreviewResetReason.PREVIEW_INVALID,
                        )
                        preview_active = False
                        accumulated = ""

                try:
                    response = await complete_model_streaming(
                        self._model,
                        request,
                        phase=ChatPipelinePhase.GENERATE_OR_REFUSE,
                        on_content_delta=on_content_delta,
                    )
                except asyncio.CancelledError:
                    await emit_preview_reset_safely(
                        self._preview_sink,
                        run_id=context.run_id,
                        attempt=context.attempt,
                        reason=ChatPreviewResetReason.GENERATION_FAILED,
                    )
                    raise
                except ChatPipelineExecutionError:
                    await emit_preview_reset_safely(
                        self._preview_sink,
                        run_id=context.run_id,
                        attempt=context.attempt,
                        reason=ChatPreviewResetReason.GENERATION_FAILED,
                    )
                    raise
            else:
                response = await complete_model(
                    self._model,
                    request,
                    phase=ChatPipelinePhase.GENERATE_OR_REFUSE,
                )
            call = model_call_record(ChatModelOperation.GENERATE_ANSWER, response)
            try:
                require_frozen_model(
                    context, response, phase=ChatPipelinePhase.GENERATE_OR_REFUSE
                )
            except ChatPipelineExecutionError as error:
                await emit_preview_reset_safely(
                    self._preview_sink,
                    run_id=context.run_id,
                    attempt=context.attempt,
                    reason=ChatPreviewResetReason.GENERATION_FAILED,
                )
                raise error.retain_model_calls(answering.model_calls + (call,))
            draft = AnswerDraftCandidate(
                raw_json=response.content,
                expected_outcome=route,
                source=AnswerDraftSource.PROVIDER,
            )
            calls = answering.model_calls + (
                call,
            )
            artifacts = {
                **state.artifacts,
                "final_llm_context": serialize_final_llm_context(
                    request,
                    operation=ChatModelOperation.GENERATE_ANSWER,
                    evidence=answering.evidence,
                ),
            }
        if isinstance(route, AnswerControlReason):
            artifacts = state.artifacts
        return ChatPipelineState(
            context=context,
            evidence_pack=pack,
            answering=ChatAnsweringState(
                evidence=answering.evidence,
                assessment=answering.assessment,
                draft=draft,
                model_calls=calls,
                visual_content=answering.visual_content,
                visual_decisions=answering.visual_decisions,
                visual_total_bytes=answering.visual_total_bytes,
            ),
            query_context=state.query_context,
            artifacts=artifacts,
        )


def _route(
    coverage: EvidenceCoverage, insufficiency: InsufficiencyPolicy
) -> AnswerOutcome | AnswerControlReason:
    if coverage is EvidenceCoverage.SUFFICIENT:
        return AnswerOutcome.ANSWERED
    if coverage is EvidenceCoverage.PARTIAL:
        if insufficiency is InsufficiencyPolicy.PARTIAL_ANSWER:
            return AnswerOutcome.PARTIAL
        return AnswerControlReason.INSUFFICIENT_EVIDENCE
    if coverage is EvidenceCoverage.AMBIGUOUS:
        return AnswerControlReason.AMBIGUOUS_QUESTION
    if coverage is EvidenceCoverage.CONFLICT:
        return AnswerControlReason.CONFLICT_UNRESOLVED
    return AnswerControlReason.NO_USABLE_EVIDENCE


def _workflow_model_calls(
    state: ChatPipelineState,
) -> tuple[ChatModelCallRecord, ...]:
    value = state.artifacts.get(WORKFLOW_MODEL_CALLS_ARTIFACT, ())
    if (
        not isinstance(value, tuple)
        or any(not isinstance(item, ChatModelCallRecord) for item in value)
    ):
        raise _context_error(
            ChatPipelinePhase.ASSESS_EVIDENCE, "workflow_model_calls"
        )
    return value


def _calculation_facts(
    state: ChatPipelineState,
) -> tuple[DecimalCalculationFact, ...]:
    value = state.artifacts.get(CALCULATION_FACTS_ARTIFACT, ())
    if (
        not isinstance(value, tuple)
        or len(value) > 4
        or any(not isinstance(item, DecimalCalculationFact) for item in value)
    ):
        raise _context_error(
            ChatPipelinePhase.GENERATE_OR_REFUSE, "calculation_facts"
        )
    return value


def _evidence_key(value) -> str:
    return f"chunk:{value.index_chunk_id}"


def _deterministic_refusal(reason: AnswerControlReason) -> AnswerDraftCandidate:
    return AnswerDraftCandidate(
        raw_json=json.dumps(
            {"outcome": "refused", "claims": [], "missing_aspects": []},
            separators=(",", ":"),
        ),
        expected_outcome=AnswerOutcome.REFUSED,
        source=AnswerDraftSource.DETERMINISTIC,
        control_reason=reason,
    )


def _require_policy(
    context: ChatExecutionContext,
) -> tuple[AnswerStyle, InsufficiencyPolicy]:
    policy = context.effective_policy
    try:
        if (
            policy["grounding_policy"] != "evidence_only"
            or policy["citation_required"] is not True
            or policy["citation_granularity"] != "claim_level"
            or policy["answer_task"] != "answer"
            or policy["policy_version"] != "p1"
        ):
            raise ValueError
        return (
            AnswerStyle(policy["answer_style"]),
            InsufficiencyPolicy(policy["insufficiency_policy"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise _context_error(
            ChatPipelinePhase.GENERATE_OR_REFUSE, "effective_policy"
        ) from error


def _require_inputs(
    state: ChatPipelineState, phase: ChatPipelinePhase
) -> tuple[ChatExecutionContext, EvidencePack]:
    if state.context is None or state.evidence_pack is None:
        raise _context_error(phase, "pipeline_inputs")
    return state.context, state.evidence_pack


def _context_error(
    phase: ChatPipelinePhase, check: str
) -> ChatPipelineExecutionError:
    return ChatPipelineExecutionError(
        ErrorCode.CHAT_CONTEXT_INVALID,
        phase=phase,
        diagnostic={"check": check},
    )

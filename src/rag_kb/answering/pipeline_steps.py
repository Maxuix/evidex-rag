"""Deterministic evidence admission and grounded generation pipeline steps."""

from __future__ import annotations

import json

from rag_kb.adapters.model_api import ChatModelAdapter
from rag_kb.answering.model_execution import (
    complete_model,
    model_call_record,
    require_frozen_model,
)
from rag_kb.answering.prompt_builder import (
    build_evidence_envelope,
    build_generation_request,
)
from rag_kb.domain import (
    AnswerControlReason,
    AnswerDraftCandidate,
    AnswerDraftSource,
    AnswerOutcome,
    AnswerStyle,
    ChatAnsweringState,
    ChatExecutionContext,
    ChatModelOperation,
    ChatPipelineExecutionError,
    ChatPipelinePhase,
    ChatPipelineState,
    EvidenceAssessment,
    EvidenceCoverage,
    EvidenceEnvelope,
    EvidencePack,
    EvidenceScoreKind,
    InsufficiencyPolicy,
)


class CosineEvidenceAssessmentStep:
    """Admit evidence using rerank quality with a semantic-similarity floor."""

    def __init__(
        self,
        min_cosine_similarity: float,
        min_rerank_score: float = 0.45,
    ) -> None:
        if not -1.0 <= min_cosine_similarity <= 1.0:
            raise ValueError("min_cosine_similarity must be between -1 and 1")
        if not 0.0 <= min_rerank_score <= 1.0:
            raise ValueError("min_rerank_score must be between 0 and 1")
        self._min_cosine_similarity = float(min_cosine_similarity)
        self._min_rerank_score = float(min_rerank_score)

    async def run(self, state: ChatPipelineState) -> ChatPipelineState:
        context, pack = _require_inputs(state, ChatPipelinePhase.ASSESS_EVIDENCE)
        evidence = build_evidence_envelope(pack)
        usable_citation_ids = tuple(
            prompt_item.citation_id
            for item, prompt_item in zip(pack.evidence, evidence.items, strict=True)
            if self._usable(item)
        )
        if not usable_citation_ids:
            assessment = EvidenceAssessment(
                coverage=EvidenceCoverage.NONE,
                usable_citation_ids=(),
                supported_aspects=(),
                missing_aspects=(),
            )
        else:
            assessment = EvidenceAssessment(
                coverage=EvidenceCoverage.SUFFICIENT,
                usable_citation_ids=usable_citation_ids,
                supported_aspects=("question",),
                missing_aspects=(),
            )
        answering = ChatAnsweringState(evidence=evidence, assessment=assessment)
        return ChatPipelineState(
            context=context,
            evidence_pack=pack,
            answering=answering,
            artifacts=state.artifacts,
        )

    def _usable(self, item) -> bool:
        if item.score is None:
            return False
        if item.score_kind is not EvidenceScoreKind.HYBRID_RERANK:
            return item.score >= self._min_cosine_similarity
        vector_similarity = item.vector_similarity
        return (
            vector_similarity is not None
            and vector_similarity >= self._min_cosine_similarity
            and item.score >= self._min_rerank_score
            and (
                item.lexical_coverage > 0.0
                or item.score >= max(self._min_rerank_score + 0.10, 0.55)
            )
        )


class AnswerGenerationStep:
    def __init__(self, model: ChatModelAdapter) -> None:
        self._model = model

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
            response = await complete_model(
                self._model,
                build_generation_request(
                    context,
                    answering.evidence,
                    answering.assessment,
                    expected_outcome=route,
                ),
                phase=ChatPipelinePhase.GENERATE_OR_REFUSE,
            )
            call = model_call_record(ChatModelOperation.GENERATE_ANSWER, response)
            try:
                require_frozen_model(
                    context, response, phase=ChatPipelinePhase.GENERATE_OR_REFUSE
                )
            except ChatPipelineExecutionError as error:
                raise error.retain_model_calls(answering.model_calls + (call,))
            draft = AnswerDraftCandidate(
                raw_json=response.content,
                expected_outcome=route,
                source=AnswerDraftSource.PROVIDER,
            )
            calls = answering.model_calls + (
                call,
            )
        return ChatPipelineState(
            context=context,
            evidence_pack=pack,
            answering=ChatAnsweringState(
                evidence=answering.evidence,
                assessment=answering.assessment,
                draft=draft,
                model_calls=calls,
            ),
            artifacts=state.artifacts,
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
    return AnswerControlReason.NO_USABLE_EVIDENCE


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

"""Concrete W04 evidence-assessment and generation pipeline steps."""

from __future__ import annotations

import json
from typing import Any

from rag_kb.adapters.model_api import ChatModelAdapter
from rag_kb.answering.model_execution import (
    complete_model,
    model_call_record,
    require_frozen_model,
)
from rag_kb.answering.prompt_builder import (
    build_assessment_request,
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
    ErrorCode,
    EvidenceAssessment,
    EvidenceCoverage,
    EvidenceEnvelope,
    EvidencePack,
    InsufficiencyPolicy,
)


class EvidenceAssessmentStep:
    def __init__(self, model: ChatModelAdapter) -> None:
        self._model = model

    async def run(self, state: ChatPipelineState) -> ChatPipelineState:
        context, pack = _require_inputs(state, ChatPipelinePhase.ASSESS_EVIDENCE)
        evidence = build_evidence_envelope(pack)
        if not evidence.items:
            assessment = EvidenceAssessment(
                coverage=EvidenceCoverage.NONE,
                usable_citation_ids=(),
                supported_aspects=(),
                missing_aspects=(),
            )
            answering = ChatAnsweringState(evidence=evidence, assessment=assessment)
        else:
            response = await complete_model(
                self._model,
                build_assessment_request(context, evidence),
                phase=ChatPipelinePhase.ASSESS_EVIDENCE,
            )
            call = model_call_record(ChatModelOperation.ASSESS_EVIDENCE, response)
            try:
                require_frozen_model(
                    context, response, phase=ChatPipelinePhase.ASSESS_EVIDENCE
                )
                assessment = _parse_assessment(response.content, evidence)
            except ChatPipelineExecutionError as error:
                raise error.retain_model_calls((call,))
            answering = ChatAnsweringState(
                evidence=evidence,
                assessment=assessment,
                model_calls=(
                    call,
                ),
            )
        return ChatPipelineState(
            context=context,
            evidence_pack=pack,
            answering=answering,
            artifacts=state.artifacts,
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


def _parse_assessment(raw_json: str, evidence: EvidenceEnvelope) -> EvidenceAssessment:
    try:
        value: Any = json.loads(raw_json)
        if not isinstance(value, dict) or set(value) != {
            "coverage",
            "usable_citation_ids",
            "supported_aspects",
            "missing_aspects",
        }:
            raise TypeError
        lists = (
            value["usable_citation_ids"],
            value["supported_aspects"],
            value["missing_aspects"],
        )
        if any(
            not isinstance(items, list)
            or any(not isinstance(item, str) for item in items)
            for items in lists
        ):
            raise TypeError
        assessment = EvidenceAssessment(
            coverage=EvidenceCoverage(value["coverage"]),
            usable_citation_ids=tuple(value["usable_citation_ids"]),
            supported_aspects=tuple(value["supported_aspects"]),
            missing_aspects=tuple(value["missing_aspects"]),
        )
        if not set(assessment.usable_citation_ids) <= evidence.citation_ids:
            raise ValueError
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ChatPipelineExecutionError(
            ErrorCode.CHAT_ASSESSMENT_INVALID,
            phase=ChatPipelinePhase.ASSESS_EVIDENCE,
            diagnostic={"check": "assessment_structure"},
        ) from error
    return assessment


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

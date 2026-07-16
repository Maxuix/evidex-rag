"""Strict answer validation, one bounded repair, and safe rendering."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from rag_kb.adapters.model_api import ChatModelAdapter
from rag_kb.answering.model_execution import (
    complete_model,
    model_call_record,
    require_frozen_model,
)
from rag_kb.answering.prompt_builder import build_repair_request
from rag_kb.answering.wire_schemas import WireAnswer
from rag_kb.domain import (
    AnswerClaim,
    AnswerControlReason,
    AnswerDraftCandidate,
    AnswerDraftSource,
    AnswerOutcome,
    AnswerValidationIssue,
    AnswerValidationRecord,
    ChatAnsweringState,
    ChatModelOperation,
    ChatPipelineExecutionError,
    ChatPipelinePhase,
    ChatPipelineState,
    ErrorCode,
    EvidenceAssessment,
    EvidenceEnvelope,
    RenderedAnswer,
    RenderedCitation,
    ValidatedAnswer,
)


class AnswerStructureValidationStep:
    """Allow only validated structure to cross the user-visible boundary."""

    def __init__(self, model: ChatModelAdapter) -> None:
        self._model = model

    async def run(self, state: ChatPipelineState) -> ChatPipelineState:
        context = state.context
        answering = state.answering
        if (
            context is None
            or state.evidence_pack is None
            or answering is None
            or answering.draft is None
            or answering.validated is not None
        ):
            raise _context_error("answer_draft")

        validated, rendered, initial_issues = _validate_and_render(
            answering.draft,
            answering.evidence,
            answering.assessment,
        )
        calls = answering.model_calls
        if validated is not None and rendered is not None:
            record = AnswerValidationRecord(initial_issues=())
        elif answering.draft.source is AnswerDraftSource.DETERMINISTIC:
            validated, rendered = _safe_validation_refusal(answering.evidence)
            record = AnswerValidationRecord(
                initial_issues=initial_issues,
                safe_fallback=True,
            )
        else:
            response = await complete_model(
                self._model,
                build_repair_request(
                    context,
                    answering.evidence,
                    answering.assessment,
                    expected_outcome=answering.draft.expected_outcome,
                    raw_draft=answering.draft.raw_json,
                    issues=initial_issues,
                ),
                phase=ChatPipelinePhase.VALIDATE_STRUCTURE,
            )
            call = model_call_record(ChatModelOperation.REPAIR_ANSWER, response)
            try:
                require_frozen_model(
                    context,
                    response,
                    phase=ChatPipelinePhase.VALIDATE_STRUCTURE,
                )
            except ChatPipelineExecutionError as error:
                raise error.retain_model_calls(calls + (call,))
            calls += (call,)
            repaired = AnswerDraftCandidate(
                raw_json=response.content,
                expected_outcome=answering.draft.expected_outcome,
                source=AnswerDraftSource.PROVIDER,
            )
            validated, rendered, repair_issues = _validate_and_render(
                repaired,
                answering.evidence,
                answering.assessment,
            )
            if validated is not None and rendered is not None:
                record = AnswerValidationRecord(
                    initial_issues=initial_issues,
                    repair_attempted=True,
                    repair_succeeded=True,
                )
            else:
                validated, rendered = _safe_validation_refusal(answering.evidence)
                record = AnswerValidationRecord(
                    initial_issues=initial_issues,
                    repair_issues=repair_issues,
                    repair_attempted=True,
                    safe_fallback=True,
                )

        return ChatPipelineState(
            context=context,
            evidence_pack=state.evidence_pack,
            answering=ChatAnsweringState(
                evidence=answering.evidence,
                assessment=answering.assessment,
                draft=answering.draft,
                model_calls=calls,
                validated=validated,
                rendered=rendered,
                validation=record,
            ),
            artifacts=state.artifacts,
        )


def _validate_and_render(
    draft: AnswerDraftCandidate,
    evidence: EvidenceEnvelope,
    assessment: EvidenceAssessment,
) -> tuple[
    ValidatedAnswer | None,
    RenderedAnswer | None,
    tuple[AnswerValidationIssue, ...],
]:
    parsed, parse_issue = _parse_wire_answer(draft.raw_json)
    if parsed is None:
        return None, None, (parse_issue,)

    issues: list[AnswerValidationIssue] = []

    def add(issue: AnswerValidationIssue) -> None:
        if issue not in issues:
            issues.append(issue)

    outcome = AnswerOutcome(parsed.outcome)
    if outcome is not draft.expected_outcome:
        add(AnswerValidationIssue.OUTCOME_MISMATCH)

    normalized_missing = tuple(value.strip() for value in parsed.missing_aspects)
    if any(
        not value
        or len(value) > 1000
        or value != original
        for value, original in zip(normalized_missing, parsed.missing_aspects, strict=True)
    ) or len(normalized_missing) != len(set(normalized_missing)):
        add(AnswerValidationIssue.SCHEMA_INVALID)

    if outcome is AnswerOutcome.ANSWERED:
        if not parsed.claims:
            add(AnswerValidationIssue.CLAIMS_REQUIRED)
        if parsed.missing_aspects:
            add(AnswerValidationIssue.MISSING_ASPECTS_FORBIDDEN)
    elif outcome is AnswerOutcome.PARTIAL:
        if not parsed.claims:
            add(AnswerValidationIssue.CLAIMS_REQUIRED)
        if not parsed.missing_aspects:
            add(AnswerValidationIssue.MISSING_ASPECTS_REQUIRED)
        elif set(normalized_missing) != set(assessment.missing_aspects):
            add(AnswerValidationIssue.MISSING_ASPECTS_MISMATCH)
    else:
        if parsed.claims:
            add(AnswerValidationIssue.CLAIMS_FORBIDDEN)
        if parsed.missing_aspects:
            add(AnswerValidationIssue.MISSING_ASPECTS_FORBIDDEN)

    allowed = evidence.citation_ids & frozenset(assessment.usable_citation_ids)
    claims: list[AnswerClaim] = []
    for wire_claim in parsed.claims:
        text = wire_claim.text.strip()
        if not text or text != wire_claim.text:
            add(AnswerValidationIssue.SCHEMA_INVALID)
        if not wire_claim.citation_ids:
            add(AnswerValidationIssue.CITATIONS_REQUIRED)
        if len(wire_claim.citation_ids) != len(set(wire_claim.citation_ids)):
            add(AnswerValidationIssue.CITATION_DUPLICATE)
        if any(citation_id not in allowed for citation_id in wire_claim.citation_ids):
            add(AnswerValidationIssue.CITATION_NOT_ALLOWED)
        if text and wire_claim.citation_ids:
            try:
                claims.append(
                    AnswerClaim(text=text, citation_ids=tuple(wire_claim.citation_ids))
                )
            except ValueError:
                add(AnswerValidationIssue.SCHEMA_INVALID)

    if issues:
        return None, None, tuple(issues)

    source = draft.source
    control_reason = draft.control_reason
    missing_aspects = (
        assessment.missing_aspects if outcome is AnswerOutcome.PARTIAL else ()
    )
    try:
        validated = ValidatedAnswer(
            outcome=outcome,
            claims=tuple(claims),
            missing_aspects=missing_aspects,
            source=source,
            control_reason=control_reason,
        )
        rendered = render_validated_answer(validated, evidence)
    except ValueError:
        return None, None, (AnswerValidationIssue.RENDER_LIMIT_EXCEEDED,)
    return validated, rendered, ()


def _parse_wire_answer(
    raw_json: str,
) -> tuple[WireAnswer | None, AnswerValidationIssue]:
    try:
        value: Any = json.loads(raw_json)
    except json.JSONDecodeError:
        return None, AnswerValidationIssue.JSON_INVALID
    try:
        return (
            WireAnswer.model_validate(value, strict=True),
            AnswerValidationIssue.SCHEMA_INVALID,
        )
    except ValidationError:
        return None, AnswerValidationIssue.SCHEMA_INVALID


def render_validated_answer(
    answer: ValidatedAnswer, evidence: EvidenceEnvelope
) -> RenderedAnswer:
    if answer.outcome is AnswerOutcome.REFUSED:
        reason = answer.control_reason
        messages = {
            AnswerControlReason.INSUFFICIENT_EVIDENCE: (
                "The available evidence is insufficient to answer reliably."
            ),
            AnswerControlReason.NO_USABLE_EVIDENCE: (
                "No usable evidence is available to answer this question."
            ),
            AnswerControlReason.AMBIGUOUS_QUESTION: (
                "The question is ambiguous. Please clarify it and try again."
            ),
            AnswerControlReason.STRUCTURE_VALIDATION_FAILED: (
                "A structurally valid evidence-grounded answer could not be produced."
            ),
        }
        if reason not in messages:
            raise ValueError("refusal control reason is not renderable")
        return RenderedAnswer(
            outcome=answer.outcome,
            content=messages[reason],
            citations=(),
        )

    lookup = {item.citation_id: item for item in evidence.items}
    ordinals: dict[str, int] = {}
    citations: list[RenderedCitation] = []
    paragraphs: list[str] = []
    for claim in answer.claims:
        markers: list[str] = []
        for citation_id in claim.citation_ids:
            item = lookup.get(citation_id)
            if item is None:
                raise ValueError("validated citation is not locatable")
            if citation_id not in ordinals:
                ordinal = len(citations)
                ordinals[citation_id] = ordinal
                citations.append(
                    RenderedCitation(
                        ordinal=ordinal,
                        citation_id=citation_id,
                        index_chunk_id=item.index_chunk_id,
                        document_id=item.document_id,
                        document_version_id=item.document_version_id,
                        quoted_text=item.excerpt,
                        source_location=item.source_location,
                        score=item.score,
                    )
                )
            markers.append(f"[{ordinals[citation_id] + 1}]")
        paragraphs.append(f"{claim.text} {''.join(markers)}")
    if answer.outcome is AnswerOutcome.PARTIAL:
        paragraphs.append(
            "Missing information: " + "; ".join(answer.missing_aspects)
        )
    return RenderedAnswer(
        outcome=answer.outcome,
        content="\n\n".join(paragraphs),
        citations=tuple(citations),
    )


def _safe_validation_refusal(
    evidence: EvidenceEnvelope,
) -> tuple[ValidatedAnswer, RenderedAnswer]:
    validated = ValidatedAnswer(
        outcome=AnswerOutcome.REFUSED,
        claims=(),
        missing_aspects=(),
        source=AnswerDraftSource.DETERMINISTIC,
        control_reason=AnswerControlReason.STRUCTURE_VALIDATION_FAILED,
    )
    return validated, render_validated_answer(validated, evidence)


def _context_error(check: str) -> ChatPipelineExecutionError:
    return ChatPipelineExecutionError(
        ErrorCode.CHAT_CONTEXT_INVALID,
        phase=ChatPipelinePhase.VALIDATE_STRUCTURE,
        diagnostic={"check": check},
    )

"""Strict answer validation, one bounded repair, and safe rendering."""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from pydantic import ValidationError

from rag_kb.answering.model_execution import (
    complete_model,
    model_call_record,
    require_frozen_model,
)
from rag_kb.answering.preview import (
    NoOpChatPreviewSink,
    emit_preview_reset_safely,
)
from rag_kb.answering.prompt_builder import (
    allowed_answer_outcomes,
    build_repair_request,
    required_answer_document_ids,
    serialize_final_llm_context,
)
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
    ChatPreviewResetReason,
    ErrorCode,
    EvidenceAssessment,
    EvidenceCoverage,
    EvidenceEnvelope,
    InsufficiencyPolicy,
    RenderedAnswer,
    RenderedCitation,
    ValidatedAnswer,
)
from rag_kb.ports.chat_preview import ChatPreviewSink
from rag_kb.ports.model_api import ChatModelAdapter
from rag_kb.retrieval.calculator import (
    CALCULATION_FACTS_ARTIFACT,
    DecimalCalculationFact,
)


class AnswerStructureValidationStep:
    """Allow only validated structure to cross the user-visible boundary."""

    def __init__(
        self,
        model: ChatModelAdapter,
        *,
        preview_sink: ChatPreviewSink | None = None,
    ) -> None:
        self._model = model
        self._preview_sink = preview_sink or NoOpChatPreviewSink()

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
            current_query=context.query,
            insufficiency=InsufficiencyPolicy(
                context.effective_policy["insufficiency_policy"]
            ),
            required_document_ids=required_answer_document_ids(context),
        )
        calls = answering.model_calls
        artifacts = state.artifacts
        if validated is not None and rendered is not None:
            record = AnswerValidationRecord(initial_issues=())
        elif (
            answering.draft.source is AnswerDraftSource.PROVIDER
            and _is_uncited_substantive_draft(answering.draft.raw_json)
        ):
            await emit_preview_reset_safely(
                self._preview_sink,
                run_id=context.run_id,
                attempt=context.attempt,
                reason=ChatPreviewResetReason.VALIDATION_REPAIR,
            )
            validated, rendered = _safe_validation_refusal(
                answering.evidence,
                reason=AnswerControlReason.INSUFFICIENT_EVIDENCE,
                current_query=context.query,
            )
            record = AnswerValidationRecord(
                initial_issues=initial_issues,
                safe_fallback=True,
            )
        elif answering.draft.source is AnswerDraftSource.DETERMINISTIC:
            validated, rendered = _safe_validation_refusal(
                answering.evidence,
                current_query=context.query,
            )
            record = AnswerValidationRecord(
                initial_issues=initial_issues,
                safe_fallback=True,
            )
        else:
            await emit_preview_reset_safely(
                self._preview_sink,
                run_id=context.run_id,
                attempt=context.attempt,
                reason=ChatPreviewResetReason.VALIDATION_REPAIR,
            )
            request = build_repair_request(
                context,
                answering.evidence,
                answering.assessment,
                expected_outcome=answering.draft.expected_outcome,
                raw_draft=answering.draft.raw_json,
                issues=initial_issues,
                query_context=state.query_context,
                visual_content=answering.visual_content,
                calculation_facts=_calculation_facts(state),
            )
            response = await complete_model(
                self._model,
                request,
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
                current_query=context.query,
                insufficiency=InsufficiencyPolicy(
                    context.effective_policy["insufficiency_policy"]
                ),
                required_document_ids=required_answer_document_ids(context),
            )
            if validated is not None and rendered is not None:
                record = AnswerValidationRecord(
                    initial_issues=initial_issues,
                    repair_attempted=True,
                    repair_succeeded=True,
                )
            else:
                validated, rendered = _safe_validation_refusal(
                    answering.evidence,
                    current_query=context.query,
                )
                record = AnswerValidationRecord(
                    initial_issues=initial_issues,
                    repair_issues=repair_issues,
                    repair_attempted=True,
                    safe_fallback=True,
                )

            artifacts = {
                **state.artifacts,
                "final_llm_context": serialize_final_llm_context(
                    request,
                    operation=ChatModelOperation.REPAIR_ANSWER,
                    evidence=answering.evidence,
                ),
            }

        return ChatPipelineState(
            context=context,
            evidence_pack=state.evidence_pack,
            answering=ChatAnsweringState(
                evidence=answering.evidence,
                assessment=answering.assessment,
                draft=answering.draft,
                model_calls=calls,
                visual_content=answering.visual_content,
                visual_decisions=answering.visual_decisions,
                visual_total_bytes=answering.visual_total_bytes,
                validated=validated,
                rendered=rendered,
                validation=record,
            ),
            query_context=state.query_context,
            artifacts=artifacts,
        )


def _validate_and_render(
    draft: AnswerDraftCandidate,
    evidence: EvidenceEnvelope,
    assessment: EvidenceAssessment,
    *,
    current_query: str,
    insufficiency: InsufficiencyPolicy,
    required_document_ids: tuple[UUID, ...],
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
    if outcome not in allowed_answer_outcomes(draft.expected_outcome, insufficiency):
        add(AnswerValidationIssue.OUTCOME_MISMATCH)

    normalized_missing = tuple(value.strip() for value in parsed.missing_aspects)
    if any(
        not value
        or not _display_aspect(value)
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
        elif (
            assessment.coverage is EvidenceCoverage.PARTIAL
            and set(normalized_missing) != set(assessment.missing_aspects)
        ):
            add(AnswerValidationIssue.MISSING_ASPECTS_MISMATCH)
    else:
        if parsed.claims:
            add(AnswerValidationIssue.CLAIMS_FORBIDDEN)
        if parsed.missing_aspects:
            add(AnswerValidationIssue.MISSING_ASPECTS_FORBIDDEN)

    allowed = evidence.citation_ids & frozenset(assessment.usable_citation_ids)
    evidence_documents = {
        item.citation_id: item.document_id for item in evidence.items
    }
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

    if (
        outcome is AnswerOutcome.ANSWERED
        and assessment.coverage is EvidenceCoverage.SUFFICIENT
        and len(required_document_ids) > 1
    ):
        used_documents = {
            evidence_documents[citation_id]
            for wire_claim in parsed.claims
            for citation_id in wire_claim.citation_ids
            if citation_id in allowed
        }
        if not set(required_document_ids) <= used_documents:
            add(AnswerValidationIssue.REQUIRED_DOCUMENT_CITATIONS_MISSING)

    if issues:
        return None, None, tuple(issues)

    source = draft.source
    control_reason = draft.control_reason
    if outcome is AnswerOutcome.REFUSED and source is AnswerDraftSource.PROVIDER:
        control_reason = AnswerControlReason.INSUFFICIENT_EVIDENCE
    missing_aspects = normalized_missing if outcome is AnswerOutcome.PARTIAL else ()
    try:
        validated = ValidatedAnswer(
            outcome=outcome,
            claims=tuple(claims),
            missing_aspects=missing_aspects,
            source=source,
            control_reason=control_reason,
        )
        rendered = render_validated_answer(
            validated,
            evidence,
            current_query=current_query,
        )
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
    answer: ValidatedAnswer,
    evidence: EvidenceEnvelope,
    *,
    current_query: str | None = None,
) -> RenderedAnswer:
    if answer.outcome is AnswerOutcome.REFUSED:
        reason = answer.control_reason
        if reason is None:
            raise ValueError("refusal control reason is not renderable")
        content = _refusal_messages(current_query).get(reason)
        if content is None:
            raise ValueError("refusal control reason is not renderable")
        return RenderedAnswer(
            outcome=answer.outcome,
            content=content,
            citations=(),
            control_reason=reason,
        )

    if answer.outcome is AnswerOutcome.ACKNOWLEDGED:
        if current_query is None:
            raise ValueError("acknowledgement rendering requires the current query")
        content = (
            "好的，明白了。"
            if _contains_cjk(current_query)
            else "Understood."
        )
        return RenderedAnswer(
            outcome=answer.outcome,
            content=content,
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
                        modality=item.modality,
                        asset_snapshot=item.asset_snapshot,
                        matched_representations=item.matched_representations,
                        document_display_name=item.document_display_name,
                        document_original_filename=item.document_original_filename,
                    )
                )
            markers.append(f"[{ordinals[citation_id] + 1}]")
        paragraphs.append(f"{claim.text} {''.join(markers)}")
    if answer.outcome is AnswerOutcome.PARTIAL:
        if current_query is None:
            raise ValueError("partial rendering requires the current query")
        missing = tuple(_display_aspect(value) for value in answer.missing_aspects)
        if _contains_cjk(current_query):
            topics = "、".join(f"“{value}”" for value in missing)
            paragraphs.append(f"另外，关于{topics}，我目前无法给出可靠回答。")
        else:
            topics = "; ".join(missing)
            paragraphs.append(f"I can’t reliably answer these parts yet: {topics}.")
    return RenderedAnswer(
        outcome=answer.outcome,
        content="\n\n".join(paragraphs),
        citations=tuple(citations),
    )


def _refusal_messages(
    current_query: str | None,
) -> dict[AnswerControlReason, str]:
    if current_query is not None and _contains_cjk(current_query):
        return {
            AnswerControlReason.INSUFFICIENT_EVIDENCE: (
                "当前知识库没有足够证据回答这个问题。"
            ),
            AnswerControlReason.NO_USABLE_EVIDENCE: (
                "当前知识库没有可用证据回答这个问题。"
            ),
            AnswerControlReason.AMBIGUOUS_QUESTION: (
                "问题不够明确，请补充说明后再试。"
            ),
            AnswerControlReason.CONFLICT_UNRESOLVED: (
                "现有证据相互冲突，暂时无法给出可靠回答。"
            ),
            AnswerControlReason.STRUCTURE_VALIDATION_FAILED: (
                "暂时无法生成可靠回答，请重试。"
            ),
        }
    return {
        AnswerControlReason.INSUFFICIENT_EVIDENCE: (
            "The available evidence is insufficient to answer reliably."
        ),
        AnswerControlReason.NO_USABLE_EVIDENCE: (
            "No usable evidence is available to answer this question."
        ),
        AnswerControlReason.AMBIGUOUS_QUESTION: (
            "The question is ambiguous. Please clarify it and try again."
        ),
        AnswerControlReason.CONFLICT_UNRESOLVED: (
            "The available evidence conflicts, so a reliable answer cannot be given."
        ),
        AnswerControlReason.STRUCTURE_VALIDATION_FAILED: (
            "A reliable answer could not be generated. Please try again."
        ),
    }


def _contains_cjk(value: str) -> bool:
    return any("\u4e00" <= character <= "\u9fff" for character in value)


def _display_aspect(value: str) -> str:
    return value.rstrip("。.!！?？;；")


def _safe_validation_refusal(
    evidence: EvidenceEnvelope,
    *,
    reason: AnswerControlReason = AnswerControlReason.STRUCTURE_VALIDATION_FAILED,
    current_query: str | None = None,
) -> tuple[ValidatedAnswer, RenderedAnswer]:
    validated = ValidatedAnswer(
        outcome=AnswerOutcome.REFUSED,
        claims=(),
        missing_aspects=(),
        source=AnswerDraftSource.DETERMINISTIC,
        control_reason=reason,
    )
    return validated, render_validated_answer(
        validated,
        evidence,
        current_query=current_query,
    )


def _is_uncited_substantive_draft(raw_json: str) -> bool:
    """Convert a citation-free substantive draft into a safe evidence refusal.

    A provider that has no citation for any claim has not produced an admissible
    factual answer. Mixed cited/uncited claims remain a normal validation/repair
    failure so supported content is never silently discarded.
    """

    try:
        parsed = WireAnswer.model_validate(json.loads(raw_json), strict=True)
    except (json.JSONDecodeError, ValidationError):
        return False
    return (
        bool(parsed.claims)
        and parsed.outcome in {"answered", "partial"}
        and all(not claim.citation_ids for claim in parsed.claims)
    )


def _context_error(check: str) -> ChatPipelineExecutionError:
    return ChatPipelineExecutionError(
        ErrorCode.CHAT_CONTEXT_INVALID,
        phase=ChatPipelinePhase.VALIDATE_STRUCTURE,
        diagnostic={"check": check},
    )


def _calculation_facts(
    state: ChatPipelineState,
) -> tuple[DecimalCalculationFact, ...]:
    value = state.artifacts.get(CALCULATION_FACTS_ARTIFACT, ())
    if (
        not isinstance(value, tuple)
        or len(value) > 4
        or any(not isinstance(item, DecimalCalculationFact) for item in value)
    ):
        raise _context_error("calculation_facts")
    return value

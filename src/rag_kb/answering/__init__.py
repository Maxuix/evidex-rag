"""Evidence-grounded assessment, generation, and validation boundary."""

from rag_kb.answering.pipeline_steps import (
    AnswerGenerationStep,
    CosineEvidenceAssessmentStep,
)
from rag_kb.answering.prompt_builder import (
    build_evidence_envelope,
    build_generation_request,
    build_repair_request,
)
from rag_kb.answering.structure_validator import (
    AnswerStructureValidationStep,
    render_validated_answer,
)
from rag_kb.answering.wire_schemas import WireAnswer

__all__ = [
    "AnswerGenerationStep",
    "AnswerStructureValidationStep",
    "CosineEvidenceAssessmentStep",
    "WireAnswer",
    "build_evidence_envelope",
    "build_generation_request",
    "build_repair_request",
    "render_validated_answer",
]

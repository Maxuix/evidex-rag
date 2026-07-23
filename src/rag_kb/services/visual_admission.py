"""Deterministic descriptor-only ranking for visual evidence candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from rag_kb.domain import (
    ChunkAssetRelationType,
    Evidence,
    EvidenceAsset,
    EvidencePack,
    VisualEvidenceDecision,
    VisualEvidenceReason,
)


@dataclass(frozen=True, slots=True)
class VisualEvidenceCandidate:
    visual_unit_id: UUID
    asset: EvidenceAsset
    parent_evidence: Evidence
    parent_citation_id: str
    evidence_group_key: str
    relation_type: ChunkAssetRelationType | None
    figure_label: str | None
    modality: str
    source_location: dict[str, Any]
    reason_code: VisualEvidenceReason
    text_rank: int | None
    cross_modal_rank: int | None
    priority_micros: int

    def decision(
        self, reason: VisualEvidenceReason | None = None
    ) -> VisualEvidenceDecision:
        return VisualEvidenceDecision(
            visual_unit_id=self.visual_unit_id,
            asset_id=self.asset.id,
            reason_code=reason or self.reason_code,
            parent_text_citation_ids=(self.parent_citation_id,),
            relation_type=self.relation_type,
            text_rank=self.text_rank,
            cross_modal_rank=self.cross_modal_rank,
            priority_micros=self.priority_micros,
        )


class VisualEvidenceAdmissionPolicy:
    """Rank visual descriptors without reading assets or calling a model."""

    PROFILE = "visual_evidence_admission_v2"
    RRF_K = 60
    HARD_MAX_IMAGES = 4

    def decide(
        self,
        pack: EvidencePack,
        usable_citation_ids: tuple[str, ...],
        *,
        max_images: int = 2,
    ) -> tuple[VisualEvidenceDecision, ...]:
        if not 1 <= max_images <= self.HARD_MAX_IMAGES:
            raise ValueError("visual image limit must be between one and four")
        usable = set(usable_citation_ids)
        ranked = self.rank_candidates(pack, usable_citation_ids)
        ranked_by_asset = {item.asset.id: item for item in ranked}
        selected_assets = {item.asset.id for item in ranked[:max_images]}
        decisions: list[VisualEvidenceDecision] = []
        observed_assets = set()
        for evidence in pack.evidence:
            citation_id = f"cite_{evidence.rank}"
            for related in evidence.related_visuals:
                relation_type = ChunkAssetRelationType(related.relation_type)
                candidate = ranked_by_asset.get(related.asset.id)
                if citation_id not in usable:
                    reason = VisualEvidenceReason.REJECTED_PARENT_NOT_ADMITTED
                elif not relation_type.is_strong:
                    reason = VisualEvidenceReason.REJECTED_WEAK_RELATION
                elif candidate is None or related.asset.id in observed_assets:
                    reason = VisualEvidenceReason.REJECTED_DUPLICATE
                elif related.asset.id not in selected_assets:
                    reason = VisualEvidenceReason.REJECTED_VISUAL_BUDGET
                else:
                    decisions.append(candidate.decision())
                    observed_assets.add(related.asset.id)
                    continue
                decisions.append(
                    VisualEvidenceDecision(
                        visual_unit_id=related.visual_unit_id,
                        asset_id=related.asset.id,
                        reason_code=reason,
                        parent_text_citation_ids=(citation_id,),
                        relation_type=relation_type,
                        text_rank=related.text_space_rank or evidence.text_space_rank,
                        cross_modal_rank=related.cross_modal_rank,
                        priority_micros=(
                            candidate.priority_micros if candidate is not None else 0
                        ),
                    )
                )
                observed_assets.add(related.asset.id)
            if evidence.modality not in {"image", "table"} or evidence.asset is None:
                continue
            candidate = ranked_by_asset.get(evidence.asset.id)
            if citation_id not in usable:
                reason = VisualEvidenceReason.REJECTED_LOW_SIMILARITY
            elif candidate is None or evidence.asset.id in observed_assets:
                reason = VisualEvidenceReason.REJECTED_DUPLICATE
            elif evidence.asset.id not in selected_assets:
                reason = VisualEvidenceReason.REJECTED_VISUAL_BUDGET
            else:
                decisions.append(candidate.decision())
                observed_assets.add(evidence.asset.id)
                continue
            decisions.append(
                VisualEvidenceDecision(
                    visual_unit_id=evidence.index_chunk_id,
                    asset_id=evidence.asset.id,
                    reason_code=reason,
                    parent_text_citation_ids=(citation_id,),
                    text_rank=evidence.text_space_rank,
                    cross_modal_rank=evidence.cross_modal_rank,
                    priority_micros=(
                        candidate.priority_micros if candidate is not None else 0
                    ),
                )
            )
            observed_assets.add(evidence.asset.id)
        return tuple(decisions)

    def rank_candidates(
        self,
        pack: EvidencePack,
        usable_citation_ids: tuple[str, ...],
    ) -> tuple[VisualEvidenceCandidate, ...]:
        usable = set(usable_citation_ids)
        candidates: list[VisualEvidenceCandidate] = []
        for evidence in pack.evidence:
            citation_id = f"cite_{evidence.rank}"
            parent_admitted = citation_id in usable
            for related in evidence.related_visuals:
                relation_type = ChunkAssetRelationType(related.relation_type)
                if not parent_admitted or not relation_type.is_strong:
                    continue
                if (
                    relation_type
                    is not ChunkAssetRelationType.EXPLICIT_FIGURE_REFERENCE
                    and related.cross_modal_rank is None
                    and evidence.rank != 1
                ):
                    continue
                if relation_type is ChunkAssetRelationType.EXPLICIT_FIGURE_REFERENCE:
                    reason = VisualEvidenceReason.SELECTED_EXPLICIT_REFERENCE
                    relation_weight = 4_000_000
                elif related.cross_modal_rank is not None:
                    reason = VisualEvidenceReason.SELECTED_DUAL_LANE
                    relation_weight = 3_250_000
                else:
                    reason = VisualEvidenceReason.SELECTED_STRONG_RELATION
                    relation_weight = 3_000_000
                candidates.append(
                    VisualEvidenceCandidate(
                        visual_unit_id=related.visual_unit_id,
                        asset=related.asset,
                        parent_evidence=evidence,
                        parent_citation_id=citation_id,
                        evidence_group_key=related.evidence_group_key,
                        relation_type=relation_type,
                        figure_label=related.figure_label,
                        modality=related.modality,
                        source_location=dict(related.source_location or {}),
                        reason_code=reason,
                        text_rank=related.text_space_rank or evidence.text_space_rank,
                        cross_modal_rank=related.cross_modal_rank,
                        priority_micros=_priority(
                            relation_weight,
                            related.text_space_rank or evidence.text_space_rank,
                            related.cross_modal_rank,
                        ),
                    )
                )
            if (
                parent_admitted
                and evidence.modality in {"image", "table"}
                and evidence.asset is not None
            ):
                dual_lane = (
                    evidence.text_space_rank is not None
                    and evidence.cross_modal_rank is not None
                )
                reason = (
                    VisualEvidenceReason.SELECTED_DUAL_LANE
                    if dual_lane
                    else VisualEvidenceReason.SELECTED_IMAGE_ONLY
                )
                candidates.append(
                    VisualEvidenceCandidate(
                        visual_unit_id=evidence.index_chunk_id,
                        asset=evidence.asset,
                        parent_evidence=evidence,
                        parent_citation_id=citation_id,
                        evidence_group_key=(
                            evidence.evidence_group_key or str(evidence.index_chunk_id)
                        ),
                        relation_type=None,
                        figure_label=None,
                        modality=evidence.modality,
                        source_location=dict(evidence.source_location),
                        reason_code=reason,
                        text_rank=evidence.text_space_rank,
                        cross_modal_rank=evidence.cross_modal_rank,
                        priority_micros=_priority(
                            2_000_000 if dual_lane else 1_000_000,
                            evidence.text_space_rank,
                            evidence.cross_modal_rank,
                        ),
                    )
                )

        ordered = sorted(
            candidates,
            key=lambda item: (
                -item.priority_micros,
                item.text_rank or 2**31,
                item.cross_modal_rank or 2**31,
                item.visual_unit_id.int,
                item.asset.id.int,
            ),
        )
        unique: list[VisualEvidenceCandidate] = []
        asset_ids = set()
        checksums = set()
        groups = set()
        for candidate in ordered:
            if (
                candidate.asset.id in asset_ids
                or candidate.asset.checksum_sha256 in checksums
                or candidate.evidence_group_key in groups
            ):
                continue
            asset_ids.add(candidate.asset.id)
            checksums.add(candidate.asset.checksum_sha256)
            groups.add(candidate.evidence_group_key)
            unique.append(candidate)
        return tuple(unique[: self.HARD_MAX_IMAGES])


def _priority(
    relation_weight: int,
    text_rank: int | None,
    cross_modal_rank: int | None,
) -> int:
    text_component = (
        1_000_000 // (VisualEvidenceAdmissionPolicy.RRF_K + text_rank)
        if text_rank
        else 0
    )
    cross_component = (
        1_000_000 // (VisualEvidenceAdmissionPolicy.RRF_K + cross_modal_rank)
        if cross_modal_rank
        else 0
    )
    dual_bonus = 250_000 if text_rank is not None and cross_modal_rank is not None else 0
    return relation_weight + text_component + cross_component + dual_bonus

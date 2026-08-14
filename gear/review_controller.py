"""Point-level evidence controller for StructuredReview."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from .config import GearConfig, load_config
from .contracts import (
    ClaimStrength,
    ClaimType,
    EvidenceSpan,
    FailureRecord,
    PaperClaim,
    PaperIR,
    RelationLabel,
    RetrievalBudget,
)
from .review_contracts import (
    NoveltyJudgment,
    PointValidationStatus,
    ReviewPointState,
    ReviewPoint,
    ReviewState,
)
from .prior_art import PriorArtService, RelationClassifier
from .trace import EvidenceStore

ANTECEDENT_LABELS = {
    RelationLabel.DIRECT_ANTECEDENT,
    RelationLabel.PARTIAL_ANTECEDENT,
}
LIMITING_LABELS = {*ANTECEDENT_LABELS, RelationLabel.EXTENSION}


class ReviewController:
    """Validate points; graph affects effort and tension flags, never semantics."""

    def __init__(self, config: Optional[GearConfig] = None) -> None:
        self.config = config or load_config()

    def run(
        self,
        state: ReviewState,
        paper_ir: PaperIR,
        evidence_store: EvidenceStore,
        *,
        prior_art: Optional[PriorArtService] = None,
        relation_classifier: Optional[RelationClassifier] = None,
    ) -> ReviewState:
        roles = _point_roles(state)
        for point in state.draft_review.all_points():
            point_state = state.point_states[point.point_id]
            if not self._validate_internal(point, paper_ir, evidence_store):
                point_state.status = PointValidationStatus.REJECTED
                point_state.retained = False
                point_state.validation_notes.append("paper_evidence_invalid")
                continue
            role = roles[point.point_id]
            needs_external = point.external_verification_required or role.startswith(
                "novelty_"
            )
            if not needs_external:
                point_state.status = PointValidationStatus.VALIDATED
                continue
            self._verify_external(
                point,
                role,
                point_state,
                state,
                paper_ir,
                evidence_store,
                prior_art,
                relation_classifier,
            )
        self._record_graph_tension(state)
        state.finalized = True
        return state

    @staticmethod
    def _validate_internal(
        point: ReviewPoint,
        paper_ir: PaperIR,
        evidence_store: EvidenceStore,
    ) -> bool:
        if not point.evidence_keys:
            return False
        span_keys = {f"P:{span.span_id}" for span in paper_ir.spans}
        return all(
            key in span_keys and evidence_store.has(key)
            for key in point.evidence_keys
            if key.startswith("P:")
        ) and any(key.startswith("P:") for key in point.evidence_keys)

    def _verify_external(
        self,
        point: ReviewPoint,
        role: str,
        point_state: ReviewPointState,
        state: ReviewState,
        paper_ir: PaperIR,
        evidence_store: EvidenceStore,
        prior_art: Optional[PriorArtService],
        relation_classifier: Optional[RelationClassifier],
    ) -> None:
        if prior_art is None or relation_classifier is None:
            self._mark_unresolved(
                point.point_id,
                point_state,
                state,
                "external_verification_service_unavailable",
            )
            return
        claim, span = _claim_and_span(point, paper_ir)
        if claim is None or span is None:
            self._mark_unresolved(
                point.point_id,
                point_state,
                state,
                "novelty_point_has_no_claim_span",
            )
            return
        budget = RetrievalBudget(
            normal_max=self.config.retrieval.normal_max,
            contrastive_max=(
                self.config.retrieval.contrastive_max if _graph_extreme(state) else 0
            ),
            citation_expansion_max=self.config.retrieval.citation_expansion_max,
            fulltext_max=self.config.retrieval.fulltext_max,
        )
        works = prior_art.retrieve(claim, state.cutoff_date, budget)
        if prior_art.last_failures:
            for failure in prior_art.last_failures:
                state.failure_ledger.append(
                    FailureRecord(
                        stage="prior_art",
                        reason=f"{point.point_id}:{failure}",
                        recoverable=True,
                    )
                )
        relations = []
        for work in works:
            work_key = f"W:{work.target_claim_id}:{work.work_id}"
            evidence_store.add_evidence(work_key, "retrieved_work", work)
            if work_key not in state.retrieved_work_evidence_keys:
                state.retrieved_work_evidence_keys.append(work_key)
            card = relation_classifier.classify(
                span,
                work,
                target_claim_id=claim.claim_id,
                cutoff=state.cutoff_date,
            )
            relation_key = f"R:{card.relation_id}"
            evidence_store.add_evidence(relation_key, "prior_relation", card)
            if relation_key not in state.relation_evidence_keys:
                state.relation_evidence_keys.append(relation_key)
            if relation_key not in point_state.relation_evidence_keys:
                point_state.relation_evidence_keys.append(relation_key)
            relations.append(card)
        labels = {card.relation_label for card in relations if card.temporal_valid}
        if role == "novelty_support" and labels & ANTECEDENT_LABELS:
            point_state.status = PointValidationStatus.REJECTED
            point_state.retained = False
            point_state.validation_notes.append(
                "validated_antecedent_conflicts_with_support"
            )
        elif role == "novelty_limit" and labels & LIMITING_LABELS:
            point_state.status = PointValidationStatus.EXTERNALLY_VALIDATED
        elif relations and all(
            card.relation_label not in {RelationLabel.UNRESOLVED} for card in relations
        ):
            point_state.status = PointValidationStatus.EXTERNALLY_VALIDATED
        else:
            self._mark_unresolved(
                point.point_id,
                point_state,
                state,
                "bounded_prior_art_inconclusive",
            )

    @staticmethod
    def _mark_unresolved(
        point_id: str,
        point_state: ReviewPointState,
        state: ReviewState,
        reason: str,
    ) -> None:
        point_state.status = PointValidationStatus.UNRESOLVED
        point_state.validation_notes.append(reason)
        state.failure_ledger.append(
            FailureRecord(
                stage="external_verification",
                reason=f"{point_id}:{reason}",
                recoverable=True,
            )
        )

    @staticmethod
    def _record_graph_tension(state: ReviewState) -> None:
        judgment = state.draft_review.novelty.judgment
        percentile = state.graph_context.d5_percentile
        if percentile is None:
            return
        tension = (
            percentile >= 90.0
            and judgment in {NoveltyJudgment.NEGATIVE, NoveltyJudgment.MIXED}
        ) or (percentile <= 10.0 and judgment == NoveltyJudgment.POSITIVE)
        if tension:
            state.graph_text_tension_point_ids = [
                point.point_id
                for point in [
                    *state.draft_review.novelty.supporting_points,
                    *state.draft_review.novelty.limiting_points,
                ]
            ]


def _point_roles(state: ReviewState) -> Dict[str, str]:
    roles: Dict[str, str] = {}
    for point in state.draft_review.novelty.supporting_points:
        roles[point.point_id] = "novelty_support"
    for point in state.draft_review.novelty.limiting_points:
        roles[point.point_id] = "novelty_limit"
    for section in ("strengths", "weaknesses", "questions"):
        for point in getattr(state.draft_review, section):
            roles[point.point_id] = section
    return roles


def _claim_and_span(
    point: ReviewPoint,
    paper_ir: PaperIR,
) -> Tuple[Optional[PaperClaim], Optional[EvidenceSpan]]:
    span_ids = [
        key.removeprefix("P:") for key in point.evidence_keys if key.startswith("P:")
    ]
    if not span_ids:
        return None, None
    span_map = paper_ir.span_map()
    span = span_map.get(span_ids[0])
    if span is None:
        return None, None
    claim = next(
        (item for item in paper_ir.claims if item.span_id in span_ids),
        None,
    )
    if claim is None:
        claim = PaperClaim(
            claim_id=f"C-{point.point_id}",
            claim_type=ClaimType.NOVELTY,
            span_id=span.span_id,
            text=span.text,
            strength=ClaimStrength.MODERATE,
        )
    return claim, span


def _graph_extreme(state: ReviewState) -> bool:
    percentile = state.graph_context.d5_percentile
    return percentile is not None and (percentile <= 10.0 or percentile >= 90.0)


__all__ = ["ReviewController"]

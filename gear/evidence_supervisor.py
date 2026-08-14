"""Finite evidence-state controller connecting normal, counterfactual, and citation search."""

from __future__ import annotations

import hashlib
import time

from .config import GearConfig, load_config
from .contracts import (
    ActionRecord,
    ClaimStrength,
    ClaimType,
    EvidenceSpan,
    FailureRecord,
    PaperClaim,
    PaperIR,
    RelationLabel,
    RetrievalBudget,
    RetrievedWork,
)
from .evidence_policy import is_high_risk, next_evidence_action
from .prior_art import PriorArtService, RelationClassifier
from .review_contracts import (
    CanonicalReviewPoint,
    EvidenceAction,
    PointSeverity,
    PointValidationStatus,
    ReviewPhase,
    ReviewStateV2,
)
from .trace import EvidenceStore, sha256_value

ANTECEDENTS = {RelationLabel.DIRECT_ANTECEDENT, RelationLabel.PARTIAL_ANTECEDENT}
LIMITING = {*ANTECEDENTS, RelationLabel.EXTENSION}


class EvidenceSupervisor:
    def __init__(self, config: GearConfig | None = None) -> None:
        self.config = config or load_config()
        self._works: dict[str, list[RetrievedWork]] = {}
        self._budgets: dict[str, RetrievalBudget] = {}

    def resolve(
        self,
        state: ReviewStateV2,
        paper_ir: PaperIR,
        evidence_store: EvidenceStore,
        *,
        prior_art: PriorArtService | None = None,
        relation_classifier: RelationClassifier | None = None,
    ) -> ReviewStateV2:
        state.phase = ReviewPhase.EVIDENCE_GATHERING
        while not state.finalized:
            if (
                state.action_budget.actions_used
                >= state.action_budget.total_actions_max
            ):
                self._exhaust(state)
                break
            action, target_id, reason = next_evidence_action(state)
            before = state.action_budget.actions_used
            started = time.monotonic()
            failure: str | None = None
            try:
                self._execute(
                    action,
                    target_id,
                    state,
                    paper_ir,
                    evidence_store,
                    prior_art,
                    relation_classifier,
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                failure = f"{type(exc).__name__}:{exc}"
                state.failures.append(
                    FailureRecord(
                        stage=f"evidence_supervisor:{action.value}",
                        reason=failure,
                        claim_id=target_id,
                    )
                )
                if target_id is not None and target_id in state.canonical_points:
                    point = state.canonical_points[target_id]
                    point.validation_status = PointValidationStatus.UNRESOLVED
                    point.validation_notes.append(failure)
            state.action_budget.actions_used = before + 1
            self._trace_action(
                evidence_store,
                action,
                target_id,
                reason,
                before,
                state.action_budget.actions_used,
                state,
                started,
                failure,
            )
        self._update_process_features(state)
        return state

    def _execute(
        self,
        action: EvidenceAction,
        target_id: str | None,
        state: ReviewStateV2,
        paper_ir: PaperIR,
        store: EvidenceStore,
        prior_art: PriorArtService | None,
        classifier: RelationClassifier | None,
    ) -> None:
        if action == EvidenceAction.FINALIZE:
            self._finalize(state)
            return
        if target_id is None:
            raise ValueError(f"{action.value} requires a target")
        point = state.canonical_points[target_id]
        if action == EvidenceAction.VERIFY_POINT:
            self._verify_internal(point, paper_ir, store)
        elif action in {
            EvidenceAction.SEARCH_PRIOR_ART,
            EvidenceAction.COUNTERFACTUAL_SEARCH,
        }:
            self._search(action, point, state, paper_ir, store, prior_art, classifier)
        elif action == EvidenceAction.CITATION_EXPAND:
            self._citation(point, state, paper_ir, store, prior_art, classifier)
        elif action == EvidenceAction.STABILITY_TEST:
            self._stability(point, state)

    @staticmethod
    def _verify_internal(
        point: CanonicalReviewPoint,
        paper_ir: PaperIR,
        store: EvidenceStore,
    ) -> None:
        valid = {f"P:{span.span_id}" for span in paper_ir.spans}
        point.semantic_verified = bool(point.paper_evidence_keys) and all(
            key in valid and store.has(key) for key in point.paper_evidence_keys
        )
        if not point.semantic_verified:
            point.retained = False
            point.validation_status = PointValidationStatus.REJECTED
            point.validation_notes.append("paper_evidence_invalid")
            return
        qwen_only = not point.agent_support
        if point.validation_status == PointValidationStatus.EXTERNALLY_VALIDATED:
            point.validation_status = PointValidationStatus.VALIDATED
            return
        if qwen_only and point.severity == PointSeverity.MAJOR:
            point.retained = False
            point.validation_status = PointValidationStatus.REJECTED
            point.validation_notes.append("qwen_only_major_requires_semantic_verifier")
        elif not point.requires_external_evidence:
            point.validation_status = PointValidationStatus.VALIDATED

    def _search(
        self,
        action: EvidenceAction,
        point: CanonicalReviewPoint,
        state: ReviewStateV2,
        paper_ir: PaperIR,
        store: EvidenceStore,
        prior_art: PriorArtService | None,
        classifier: RelationClassifier | None,
    ) -> None:
        if prior_art is None or classifier is None:
            point.validation_status = PointValidationStatus.UNRESOLVED
            point.validation_notes.append("prior_art_service_unavailable")
            if action == EvidenceAction.SEARCH_PRIOR_ART:
                point.normal_search_done = True
            else:
                point.counterfactual_search_done = True
            return
        claim, span = _claim_and_span(point, paper_ir)
        budget = self._budget(point.point_id, state)
        family = (
            "contrastive"
            if action == EvidenceAction.COUNTERFACTUAL_SEARCH
            else "normal"
        )
        works = prior_art.retrieve(claim, state.cutoff_date, budget, family=family)
        self._works.setdefault(point.point_id, []).extend(works)
        self._classify(point, state, span, claim, works, store, classifier)
        if action == EvidenceAction.SEARCH_PRIOR_ART:
            point.normal_search_done = True
        else:
            point.counterfactual_search_done = True

    def _citation(
        self,
        point: CanonicalReviewPoint,
        state: ReviewStateV2,
        paper_ir: PaperIR,
        store: EvidenceStore,
        prior_art: PriorArtService | None,
        classifier: RelationClassifier | None,
    ) -> None:
        point.citation_expanded = True
        seeds = self._works.get(point.point_id, [])
        if not seeds or prior_art is None or classifier is None:
            point.validation_notes.append("citation_expansion_has_no_seed")
            return
        claim, span = _claim_and_span(point, paper_ir)
        works = prior_art.expand_neighbors(
            seeds[0], claim, state.cutoff_date, self._budget(point.point_id, state)
        )
        self._works[point.point_id].extend(works)
        self._classify(point, state, span, claim, works, store, classifier)

    @staticmethod
    def _classify(
        point: CanonicalReviewPoint,
        state: ReviewStateV2,
        span: EvidenceSpan,
        claim: PaperClaim,
        works: list[RetrievedWork],
        store: EvidenceStore,
        classifier: RelationClassifier,
    ) -> None:
        labels = set()
        for work in works:
            if (
                len(state.relation_evidence_keys)
                >= state.action_budget.relation_cards_max
            ):
                point.validation_notes.append("relation_card_budget_exhausted")
                break
            work_key = f"W:{claim.claim_id}:{work.work_id}"
            store.add_evidence(work_key, "retrieved_work", work)
            if work_key not in state.retrieved_work_evidence_keys:
                state.retrieved_work_evidence_keys.append(work_key)
            card = classifier.classify(
                span, work, target_claim_id=claim.claim_id, cutoff=state.cutoff_date
            )
            relation_key = f"R:{card.relation_id}"
            store.add_evidence(relation_key, "prior_relation", card)
            if relation_key not in state.relation_evidence_keys:
                state.relation_evidence_keys.append(relation_key)
            if relation_key not in point.relation_evidence_keys:
                point.relation_evidence_keys.append(relation_key)
            if card.temporal_valid:
                labels.add(card.relation_label)
        if point.section == "novelty_support" and labels & ANTECEDENTS:
            point.retained = False
            point.validation_status = PointValidationStatus.REJECTED
            point.validation_notes.append(
                "direct_antecedent_overrides_graph_and_support"
            )
        elif point.section == "novelty_limit" and labels & LIMITING:
            point.validation_status = PointValidationStatus.EXTERNALLY_VALIDATED
        elif point.section == "novelty_support":
            point.validation_status = PointValidationStatus.UNRESOLVED
        elif labels and RelationLabel.UNRESOLVED not in labels:
            point.validation_status = PointValidationStatus.EXTERNALLY_VALIDATED
        else:
            point.validation_status = PointValidationStatus.UNRESOLVED

    @staticmethod
    def _stability(point: CanonicalReviewPoint, state: ReviewStateV2) -> None:
        # Graph-removal invariant holds because Graph never contributes relation labels.
        relation_count = len(point.relation_evidence_keys)
        stable = not point.requires_external_evidence or relation_count >= (
            2 if point.section == "novelty_support" and is_high_risk(point) else 1
        )
        point.stability_status = "stable" if stable else "unstable"
        if not stable:
            point.validation_notes.append("evidence_stability_insufficient")
            if point.severity == PointSeverity.MAJOR:
                point.retained = False
                point.validation_status = PointValidationStatus.REJECTED
        elif point.validation_status in {
            PointValidationStatus.PENDING,
            PointValidationStatus.EXTERNALLY_VALIDATED,
        }:
            point.validation_status = PointValidationStatus.VALIDATED

    def _budget(self, point_id: str, state: ReviewStateV2) -> RetrievalBudget:
        if point_id not in self._budgets:
            self._budgets[point_id] = RetrievalBudget(
                normal_max=state.action_budget.normal_per_claim_max,
                contrastive_max=state.action_budget.counterfactual_per_claim_max,
                citation_expansion_max=state.action_budget.citation_per_claim_max,
                fulltext_max=min(
                    self.config.retrieval.fulltext_max,
                    state.action_budget.relation_cards_max,
                ),
            )
        return self._budgets[point_id]

    @staticmethod
    def _finalize(state: ReviewStateV2) -> None:
        for point in state.canonical_points.values():
            if point.validation_status == PointValidationStatus.UNRESOLVED:
                point.validation_notes.append("finalized_unresolved")
                if point.severity == PointSeverity.MAJOR:
                    point.retained = False
            if point.validation_status == PointValidationStatus.EXTERNALLY_VALIDATED:
                point.validation_status = PointValidationStatus.VALIDATED
        state.unresolved_target_ids = [
            point.point_id
            for point in state.canonical_points.values()
            if point.retained
            and point.validation_status == PointValidationStatus.UNRESOLVED
        ]
        state.phase = ReviewPhase.EVIDENCE_FINALIZED
        state.finalized = True

    @staticmethod
    def _exhaust(state: ReviewStateV2) -> None:
        for point in state.canonical_points.values():
            if point.validation_status not in {
                PointValidationStatus.VALIDATED,
                PointValidationStatus.REJECTED,
            }:
                point.validation_status = PointValidationStatus.UNRESOLVED
                point.validation_notes.append("evidence_action_budget_exhausted")
                if point.severity == PointSeverity.MAJOR:
                    point.retained = False
        state.failures.append(
            FailureRecord(stage="evidence_supervisor", reason="action_budget_exhausted")
        )
        EvidenceSupervisor._finalize(state)

    @staticmethod
    def _update_process_features(state: ReviewStateV2) -> None:
        external = [
            point
            for point in state.canonical_points.values()
            if point.requires_external_evidence
        ]
        completed = sum(point.normal_search_done for point in external)
        state.process_features.retrieval_coverage = completed / max(len(external), 1)
        state.process_features.independent_prior_count = len(
            state.retrieved_work_evidence_keys
        )
        state.process_features.relation_conflict = any(
            point.qwen_conflict for point in state.canonical_points.values()
        )
        state.process_features.counterfactual_completed = all(
            point.counterfactual_search_done
            for point in external
            if is_high_risk(point)
        )
        tested = [
            point
            for point in state.canonical_points.values()
            if point.stability_status != "not_required"
        ]
        state.process_features.stability_passed = all(
            point.stability_status == "stable" for point in tested
        )
        state.process_features.failure_count = len(state.failures)

    @staticmethod
    def _trace_action(
        store: EvidenceStore,
        action: EvidenceAction,
        target_id: str | None,
        reason: str,
        before: int,
        after: int,
        state: ReviewStateV2,
        started: float,
        failure: str | None,
    ) -> None:
        input_hash = sha256_value(
            {"action": action.value, "target_id": target_id, "budget": before}
        )
        output_hash = sha256_value(state)
        identity = f"{action.value}|{target_id}|{input_hash}|{output_hash}"
        store.append_action(
            ActionRecord(
                action_id="ACT2-"
                + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:18],
                stage="evidence_supervisor",
                input_sha256=input_hash,
                output_sha256=output_hash,
                duration_ms=int((time.monotonic() - started) * 1000),
                failure=failure,
                reason_code=reason,
                target_id=target_id,
                budget_before=before,
                budget_after=after,
            )
        )


def _claim_and_span(
    point: CanonicalReviewPoint, paper_ir: PaperIR
) -> tuple[PaperClaim, EvidenceSpan]:
    """Build an evidence-anchored retrieval claim from the fused review point.

    The point's proposition is the reviewer-normalized scientific question.  It
    is only used after ``_verify_internal`` has confirmed that its linked paper
    spans exist in the EvidenceStore.  The span remains the auditable source;
    using its compiler-extracted claim text for retrieval can instead select an
    unrelated local sentence.
    """
    span_map = paper_ir.span_map()
    span_id = next(
        (
            key.removeprefix("P:")
            for key in point.paper_evidence_keys
            if key.startswith("P:") and key.removeprefix("P:") in span_map
        ),
        None,
    )
    if span_id is None:
        raise ValueError("canonical point has no valid target span")
    span = span_map[span_id]
    claim = PaperClaim(
        claim_id=f"C-{point.point_id}",
        claim_type=ClaimType.NOVELTY,
        span_id=span_id,
        text=point.proposition.strip() or span.text,
        strength=(
            ClaimStrength.STRONG if is_high_risk(point) else ClaimStrength.MODERATE
        ),
    )
    point.novelty_claim_id = claim.claim_id
    return claim, span


__all__ = ["EvidenceSupervisor"]

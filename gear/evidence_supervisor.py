"""Finite evidence controller for normal, counterfactual, and citation search."""

from __future__ import annotations

import hashlib
import re
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
        self._classified_work_ids: dict[str, set[str]] = {}
        self._valid_prior_work_ids: set[str] = set()

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
            self._stability(point, state, store)

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
            point.novelty_resolution = "search_failed"
            point.validation_notes.append("prior_art_service_unavailable")
            state.failures.append(
                FailureRecord(
                    stage="prior_art",
                    reason="prior_art_service_unavailable",
                    claim_id=point.point_id,
                )
            )
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
        works = prior_art.retrieve(
            claim,
            state.cutoff_date,
            budget,
            family=family,
            target_span=span,
            paper_ir=paper_ir,
        )
        self._store_retrieval_audit(claim, prior_art, store)
        self._record_retrieval_outcome(point, state, prior_art)
        self._works.setdefault(point.point_id, []).extend(works)
        self._classify(point, state, span, claim, works, store, classifier)
        if action == EvidenceAction.SEARCH_PRIOR_ART:
            point.normal_search_done = True
        else:
            point.counterfactual_search_done = True
        self._store_coverage(point, state, claim, prior_art, store)

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
        self._store_retrieval_audit(claim, prior_art, store)
        self._record_retrieval_outcome(point, state, prior_art)
        self._classify(point, state, span, claim, works, store, classifier)
        self._store_coverage(point, state, claim, prior_art, store)

    def _classify(
        self,
        point: CanonicalReviewPoint,
        state: ReviewStateV2,
        span: EvidenceSpan,
        claim: PaperClaim,
        works: list[RetrievedWork],
        store: EvidenceStore,
        classifier: RelationClassifier,
    ) -> None:
        labels = set()
        classified = self._classified_work_ids.setdefault(point.point_id, set())
        for work in works:
            classification_id = "|".join(
                [work.work_id, *sorted(set(work.source_query_ids))]
            )
            if classification_id in classified:
                continue
            if (
                len(state.relation_evidence_keys)
                >= state.action_budget.relation_cards_max
            ):
                point.validation_notes.append("relation_card_budget_exhausted")
                break
            work_key = f"W:{work.work_id}"
            canonical_work = work.model_dump(
                mode="json",
                exclude={
                    "target_claim_id",
                    "retrieval_query_id",
                    "source_query_ids",
                    "spans",
                },
            )
            store.add_evidence(work_key, "retrieved_work", canonical_work)
            if work_key not in state.retrieved_work_evidence_keys:
                state.retrieved_work_evidence_keys.append(work_key)
            for prior_span in work.spans:
                store.add_evidence(
                    f"PS:{prior_span.span_id}",
                    "prior_work_span",
                    prior_span,
                )
            card = classifier.classify(
                span, work, target_claim_id=claim.claim_id, cutoff=state.cutoff_date
            )
            classified.add(classification_id)
            if card.relation_label == RelationLabel.DISTANT:
                store.add_evidence(
                    f"D:{card.relation_id}",
                    "candidate_relation_rejection",
                    card,
                )
                point.validation_notes.append(
                    "distant_candidate_rejected_before_relation_store"
                )
                continue
            relation_key = f"R:{card.relation_id}"
            store.add_evidence(relation_key, "prior_relation", card)
            if relation_key not in state.relation_evidence_keys:
                state.relation_evidence_keys.append(relation_key)
            if relation_key not in point.relation_evidence_keys:
                point.relation_evidence_keys.append(relation_key)
            if card.temporal_valid:
                labels.add(card.relation_label)
                if card.relation_label != RelationLabel.UNRESOLVED:
                    self._valid_prior_work_ids.add(card.prior_work_id)
        # Retrieval is append-only.  A later citation or contrastive batch that
        # contains only distant/unresolved works must not erase a stronger
        # relation established by an earlier batch for the same review point.
        for relation_key in point.relation_evidence_keys:
            relation_record = store.get(relation_key)
            payload = relation_record.payload if relation_record is not None else {}
            if payload.get("temporal_valid") is not True:
                continue
            try:
                labels.add(RelationLabel(str(payload.get("relation_label"))))
            except ValueError:
                continue
        if point.section == "novelty_support" and labels & ANTECEDENTS:
            point.section = "novelty_limit"
            point.novelty_resolution = "antecedent_found"
            point.validation_status = PointValidationStatus.EXTERNALLY_VALIDATED
            point.resolved_proposition = (
                "Prior literature contains a direct or partial antecedent to this "
                "novelty claim; the defensible contribution is the technical "
                f"difference stated in the paired evidence for: {point.proposition}"
            )
            point.validation_notes.append(
                "direct_antecedent_reclassified_as_novelty_limit"
            )
        elif point.section == "novelty_limit" and labels & LIMITING:
            point.novelty_resolution = "antecedent_found"
            point.validation_status = PointValidationStatus.EXTERNALLY_VALIDATED
        elif labels & {
            RelationLabel.EXTENSION,
            RelationLabel.PARALLEL,
            RelationLabel.SUPPORT,
        }:
            point.novelty_resolution = "incremental_or_parallel"
            point.validation_status = PointValidationStatus.EXTERNALLY_VALIDATED
            point.resolved_proposition = (
                f"Related prior work establishes an incremental, parallel, or "
                f"supporting context for this bounded claim: {point.proposition}"
            )
        elif point.section == "novelty_support":
            point.validation_status = PointValidationStatus.UNRESOLVED
        elif labels and RelationLabel.UNRESOLVED not in labels:
            point.validation_status = PointValidationStatus.EXTERNALLY_VALIDATED
        else:
            point.validation_status = PointValidationStatus.UNRESOLVED

    @staticmethod
    def _stability(
        point: CanonicalReviewPoint,
        state: ReviewStateV2,
        store: EvidenceStore,
    ) -> None:
        # Graph-removal invariant holds because Graph never contributes relation labels.
        valid_work_ids = set()
        antecedent_query_roles: set[str] = set()
        antecedent_work_ids: set[str] = set()
        for key in point.relation_evidence_keys:
            record = store.get(key)
            payload = record.payload if record is not None else {}
            if (
                payload.get("temporal_valid") is True
                and payload.get("relation_label") not in {"DISTANT", "UNRESOLVED"}
                and payload.get("prior_work_id")
            ):
                valid_work_ids.add(str(payload["prior_work_id"]))
                if payload.get("relation_label") in {
                    "DIRECT_ANTECEDENT",
                    "PARTIAL_ANTECEDENT",
                }:
                    antecedent_work_ids.add(str(payload["prior_work_id"]))
                    for query_id in payload.get("source_query_ids") or []:
                        query_record = store.get(f"Q:{query_id}")
                        if query_record and query_record.payload.get("query_role"):
                            antecedent_query_roles.add(
                                str(query_record.payload["query_role"])
                            )
        coverage = EvidenceSupervisor._latest_coverage(point, store)
        if point.novelty_resolution == "antecedent_found":
            stable = len(antecedent_work_ids) >= 2 or len(antecedent_query_roles) >= 2
            if not stable and coverage and coverage.get("service_failed") is False:
                point.section = "questions"
                point.novelty_resolution = "inconclusive"
                point.resolved_proposition = (
                    "One retrieved source suggests a possible antecedent, but the "
                    "audited search did not independently confirm that relationship. "
                    "The authors should compare the claimed contribution directly "
                    "with that nearest work and state the technical difference."
                )
                point.validation_status = PointValidationStatus.VALIDATED
                point.stability_status = "not_required"
                point.validation_notes.append(
                    "single_antecedent_downgraded_to_question"
                )
                return
        elif coverage and coverage.get("service_failed") is True:
            point.novelty_resolution = "search_failed"
            stable = False
        elif coverage and coverage.get("coverage_sufficient") is True:
            point.novelty_resolution = "bounded_no_antecedent"
            point.resolved_proposition = (
                f"As of {state.cutoff_date.isoformat()}, no direct antecedent was "
                "found within the audited search scope for this manuscript-grounded "
                f"claim: {_without_absolute_priority(point.proposition)} "
                "This is not proof of global priority."
            )
            point.validation_status = PointValidationStatus.EXTERNALLY_VALIDATED
            stable = True
        elif point.requires_external_evidence and coverage:
            point.section = "questions"
            point.novelty_resolution = "inconclusive"
            point.resolved_proposition = (
                "The audited search did not provide enough evidence-bearing prior "
                "work for a reliable novelty conclusion. The authors should compare "
                "the contribution more explicitly with its nearest prior work: "
                f"{_without_absolute_priority(point.proposition)}"
            )
            point.validation_status = PointValidationStatus.VALIDATED
            point.stability_status = "not_required"
            point.validation_notes.append(
                "insufficient_coverage_downgraded_to_question"
            )
            return
        else:
            stable = not point.requires_external_evidence or len(valid_work_ids) >= 1
        point.stability_status = "stable" if stable else "unstable"
        if not stable:
            point.validation_notes.append("evidence_stability_insufficient")
            if point.novelty_resolution != "search_failed":
                point.validation_status = PointValidationStatus.UNRESOLVED
        elif point.validation_status in {
            PointValidationStatus.PENDING,
            PointValidationStatus.EXTERNALLY_VALIDATED,
        }:
            point.validation_status = PointValidationStatus.VALIDATED

    @staticmethod
    def _latest_coverage(
        point: CanonicalReviewPoint,
        store: EvidenceStore,
    ) -> dict:
        if not point.coverage_evidence_keys:
            return {}
        record = store.get(point.coverage_evidence_keys[-1])
        return dict(record.payload) if record is not None else {}

    @staticmethod
    def _record_retrieval_outcome(
        point: CanonicalReviewPoint,
        state: ReviewStateV2,
        prior_art: PriorArtService,
    ) -> None:
        if getattr(prior_art, "last_service_failed", False):
            point.novelty_resolution = "search_failed"
            point.validation_notes.append("prior_art_service_failed")
            for reason in getattr(prior_art, "last_failures", []):
                state.failures.append(
                    FailureRecord(
                        stage="prior_art",
                        reason=str(reason),
                        claim_id=point.point_id,
                    )
                )
        else:
            point.validation_notes.extend(
                str(note) for note in getattr(prior_art, "last_advisories", [])
            )

    @staticmethod
    def _store_coverage(
        point: CanonicalReviewPoint,
        state: ReviewStateV2,
        claim: PaperClaim,
        prior_art: PriorArtService,
        store: EvidenceStore,
    ) -> None:
        build = getattr(prior_art, "coverage_card", None)
        if not callable(build):
            return
        direct_found = False
        for key in point.relation_evidence_keys:
            record = store.get(key)
            if record and record.payload.get("relation_label") in {
                "DIRECT_ANTECEDENT",
                "PARTIAL_ANTECEDENT",
            }:
                direct_found = True
                break
        card = build(
            claim.claim_id,
            state.cutoff_date,
            require_contrastive=is_high_risk(point),
            direct_or_partial_found=direct_found,
        )
        key = f"COV:{card.coverage_id}"
        store.add_evidence(key, "retrieval_coverage", card)
        point.coverage_evidence_keys = [key]

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

    def _update_process_features(self, state: ReviewStateV2) -> None:
        external = [
            point
            for point in state.canonical_points.values()
            if point.requires_external_evidence
        ]
        completed = sum(point.normal_search_done for point in external)
        state.process_features.retrieval_coverage = completed / max(len(external), 1)
        state.process_features.independent_prior_count = len(self._valid_prior_work_ids)
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
    def _store_retrieval_audit(
        claim: PaperClaim,
        prior_art: PriorArtService,
        store: EvidenceStore,
    ) -> None:
        if prior_art.last_frame is not None:
            store.add_evidence(
                f"QF:{claim.claim_id}",
                "scientific_search_frame",
                prior_art.last_frame,
            )
        for query in prior_art.last_query_specs:
            store.add_evidence(f"Q:{query.query_id}", "retrieval_query", query)
        for hit in prior_art.last_hits:
            store.add_evidence(f"H:{hit.hit_id}", "retrieval_hit", hit)

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


def _without_absolute_priority(text: str) -> str:
    cleaned = re.sub(
        r"\b(?:first|first-ever|unprecedented|unique|world-first)\b|"
        r"首次|首个|前所未有|唯一",
        "claimed",
        str(text),
        flags=re.I,
    )
    return re.sub(r"\s+", " ", cleaned).strip()


__all__ = ["EvidenceSupervisor"]

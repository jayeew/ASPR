"""Finite evidence controller for normal, counterfactual, and citation search."""

from __future__ import annotations

import hashlib
import inspect
import re
import time
from collections.abc import Mapping
from typing import Any, Literal

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
from .graph_guidance import is_absolute_priority_claim, is_prior_art_direction_claim
from .graph_prior_contracts import GraphClaimGuidanceV1
from .prior_art import PriorArtService, RelationClassifier
from .review_contracts import (
    CanonicalReviewPoint,
    EvidenceAction,
    NoveltyJudgment,
    PointSeverity,
    PointValidationStatus,
    ReviewAspect,
    ReviewCorrectionEventV1,
    ReviewPhase,
    ReviewStateV3,
)
from .trace import EvidenceStore, sha256_value

DIRECT_ANTECEDENTS = {RelationLabel.DIRECT_ANTECEDENT}
CONTEXTUAL_RELATIONS = {
    RelationLabel.BACKGROUND,
    RelationLabel.BUILDING_BLOCK,
    RelationLabel.PARTIAL_ANTECEDENT,
    RelationLabel.EXTENSION,
    RelationLabel.PARALLEL,
    RelationLabel.SUPPORT,
}
RESIDUAL_RELATIONS = {
    RelationLabel.PARTIAL_ANTECEDENT,
    RelationLabel.EXTENSION,
    RelationLabel.PARALLEL,
    RelationLabel.SUPPORT,
}
LIMITING = {
    *DIRECT_ANTECEDENTS,
    RelationLabel.PARTIAL_ANTECEDENT,
    RelationLabel.BUILDING_BLOCK,
}

MIN_CLAIM_RELEVANT_FACET_COVERAGE = 0.4


def relation_payload_is_claim_relevant(payload: Mapping[str, Any]) -> bool:
    """Return whether a classified relation covers a material claim facet.

    Relation classification and scientific relevance are separate decisions. A
    method-only parallel from an unrelated phenotype may be a valid audit record,
    but it must not drive novelty correction or the claim-relevant Graph KPI.
    """

    try:
        label = RelationLabel(str(payload.get("relation_label")))
        coverage = float(payload.get("essential_facet_coverage") or 0.0)
    except (TypeError, ValueError):
        return False
    return (
        payload.get("temporal_valid") is True
        and label not in {RelationLabel.DISTANT, RelationLabel.UNRESOLVED}
        and bool(payload.get("prior_work_id"))
        and bool(payload.get("common_dimensions"))
        and bool(payload.get("difference_dimensions"))
        and coverage >= MIN_CLAIM_RELEVANT_FACET_COVERAGE
    )


def _claim_guidance(state: ReviewStateV3, point_id: str) -> GraphClaimGuidanceV1 | None:
    plan = getattr(state, "graph_guidance_plan", None)
    if plan is None:
        return None
    return next(
        (
            guidance
            for guidance in plan.claim_guidance
            if guidance.review_point_id == point_id
        ),
        None,
    )


class EvidenceSupervisor:
    def __init__(self, config: GearConfig | None = None) -> None:
        self.config = config or load_config()
        self._works: dict[str, list[RetrievedWork]] = {}
        self._budgets: dict[str, RetrievalBudget] = {}
        self._classified_work_ids: dict[str, set[str]] = {}
        self._valid_prior_work_ids: set[str] = set()
        self._topology_anchors: dict[str, list[RetrievedWork]] = {}

    def resolve(
        self,
        state: ReviewStateV3,
        paper_ir: PaperIR,
        evidence_store: EvidenceStore,
        *,
        prior_art: PriorArtService | None = None,
        relation_classifier: RelationClassifier | None = None,
    ) -> ReviewStateV3:
        state.phase = ReviewPhase.EVIDENCE_GATHERING
        while not state.finalized:
            action, target_id, reason = next_evidence_action(state)
            zero_cost = action in {
                EvidenceAction.VERIFY_POINT,
                EvidenceAction.STABILITY_TEST,
                EvidenceAction.FINALIZE,
            }
            if (
                not zero_cost
                and state.action_budget.actions_used
                >= state.action_budget.total_actions_max
            ):
                self._exhaust(state)
                break
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
                if target_id is not None and target_id in state.canonical_points:
                    point = state.canonical_points[target_id]
                    point.validation_notes.append(failure)
                    if action in {
                        EvidenceAction.SEARCH_PRIOR_ART,
                        EvidenceAction.COUNTERFACTUAL_SEARCH,
                        EvidenceAction.CITATION_EXPAND,
                    }:
                        self._downgrade_external_gap(point, action)
                    else:
                        state.failures.append(
                            FailureRecord(
                                stage=f"evidence_supervisor:{action.value}",
                                reason=failure,
                                claim_id=target_id,
                            )
                        )
                        point.validation_status = PointValidationStatus.UNRESOLVED
            state.action_budget.actions_used = before + (0 if zero_cost else 1)
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
        self._update_aspr_assessments(state)
        return state

    def _execute(
        self,
        action: EvidenceAction,
        target_id: str | None,
        state: ReviewStateV3,
        paper_ir: PaperIR,
        store: EvidenceStore,
        prior_art: PriorArtService | None,
        classifier: RelationClassifier | None,
    ) -> None:
        if action == EvidenceAction.FINALIZE:
            self._apply_cross_point_limiting_consensus(state, store)
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
        state: ReviewStateV3,
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
                point.counterfactual_search_count += 1
                point.counterfactual_search_done = (
                    point.counterfactual_search_count >= 1
                )
            return
        claim, span = build_retrieval_claim(point, paper_ir)
        budget = self._budget(point.point_id, state)
        family = (
            "contrastive"
            if action == EvidenceAction.COUNTERFACTUAL_SEARCH
            else "normal"
        )
        retrieve_kwargs: dict[str, Any] = {
            "family": family,
            "target_span": span,
            "paper_ir": paper_ir,
        }
        parameters = inspect.signature(prior_art.retrieve).parameters
        guidance = (
            None
            if self.config.graph_guidance.shadow
            else _claim_guidance(state, point.point_id)
        )
        graph_seed_work_ids = (
            [
                work_id
                for mission in guidance.missions
                if mission.origin == "topology"
                for work_id in mission.seed_work_ids
            ]
            if guidance is not None
            else []
        )
        if "graph_seed_work_ids" in parameters:
            retrieve_kwargs["graph_seed_work_ids"] = graph_seed_work_ids
        if "graph_seed_searches" in parameters and state.graph_result is not None:
            seed_titles = {
                seed.work_id: seed.title
                for seed in state.graph_result.topology_seeds
                if seed.title
            }
            retrieve_kwargs["graph_seed_searches"] = [
                (work_id, seed_titles[work_id])
                for work_id in graph_seed_work_ids
                if work_id in seed_titles
            ]
        if "graph_neighbor_slots" in parameters:
            retrieve_kwargs["graph_neighbor_slots"] = (
                sum(
                    1
                    for mission in guidance.missions
                    if mission.origin == "topology"
                    and mission.seed_work_ids
                    and mission.traversal != "none"
                )
                if guidance is not None
                else 0
            )
        if "allowed_query_roles" in parameters and guidance is not None:
            retrieve_kwargs["allowed_query_roles"] = self._allowed_query_roles(
                guidance,
                prefer_profile=any(
                    mission.origin == "topology" for mission in guidance.missions
                ),
            )
        if "resource_ledger" in parameters:
            retrieve_kwargs["resource_ledger"] = state.resource_ledger
        works = prior_art.retrieve(claim, state.cutoff_date, budget, **retrieve_kwargs)
        anchors = list(getattr(prior_art, "last_graph_seed_works", []))
        if anchors:
            existing = {
                work.work_id
                for work in self._topology_anchors.setdefault(point.point_id, [])
            }
            self._topology_anchors[point.point_id].extend(
                work for work in anchors if work.work_id not in existing
            )
        self._store_retrieval_audit(claim, prior_art, store)
        self._record_retrieval_outcome(point, state, prior_art)
        self._works.setdefault(point.point_id, []).extend(works)
        self._classify(point, state, span, claim, works, store, classifier)
        if action == EvidenceAction.SEARCH_PRIOR_ART:
            point.normal_search_done = True
        else:
            point.counterfactual_search_count += 1
            point.counterfactual_search_done = point.counterfactual_search_count >= 1
        self._store_coverage(point, state, claim, prior_art, store)

    def _citation(
        self,
        point: CanonicalReviewPoint,
        state: ReviewStateV3,
        paper_ir: PaperIR,
        store: EvidenceStore,
        prior_art: PriorArtService | None,
        classifier: RelationClassifier | None,
    ) -> None:
        point.citation_expanded = True
        guidance = (
            None
            if self.config.graph_guidance.shadow
            else _claim_guidance(state, point.point_id)
        )
        topology = [
            mission
            for mission in (guidance.missions if guidance is not None else [])
            if mission.origin == "topology"
        ]
        seeds = (
            self._topology_anchors.get(point.point_id, [])
            if topology
            else self._works.get(point.point_id, [])
        )
        if prior_art is None or classifier is None:
            point.validation_notes.append("citation_expansion_service_unavailable")
            return
        claim, span = build_retrieval_claim(point, paper_ir)
        if topology and (not seeds or not self._topology_seed_has_value(point, store)):
            self._topology_provider_fallback(
                point,
                state,
                paper_ir,
                store,
                prior_art,
                classifier,
                claim,
                span,
                guidance,
            )
            return
        if not seeds:
            point.validation_notes.append("citation_expansion_has_no_seed")
            return
        requested_ids = {
            work_id for mission in topology for work_id in mission.seed_work_ids
        }
        selected_seeds = [
            seed for seed in seeds if not requested_ids or seed.work_id in requested_ids
        ][:2]
        direction = topology[0].traversal if topology else "references"
        all_works: list[RetrievedWork] = []
        for seed in selected_seeds:
            works = prior_art.expand_neighbors(
                seed,
                claim,
                state.cutoff_date,
                self._budget(point.point_id, state),
                direction=(
                    direction
                    if direction in {"references", "citations"}
                    else "references"
                ),
                resource_ledger=state.resource_ledger,
            )
            all_works.extend(works)
            self._store_retrieval_audit(claim, prior_art, store)
            self._record_retrieval_outcome(point, state, prior_art)
        self._works[point.point_id].extend(all_works)
        self._classify(
            point,
            state,
            span,
            claim,
            all_works,
            store,
            classifier,
            reserve_topology=False,
        )
        self._store_coverage(point, state, claim, prior_art, store)

    def _classify(
        self,
        point: CanonicalReviewPoint,
        state: ReviewStateV3,
        span: EvidenceSpan,
        claim: PaperClaim,
        works: list[RetrievedWork],
        store: EvidenceStore,
        classifier: RelationClassifier,
        *,
        reserve_topology: bool = True,
    ) -> None:
        before_section = point.section
        before_text = point.resolved_proposition or point.proposition
        before_confidence = point.novelty_confidence
        labels: set[RelationLabel] = set()
        classified = self._classified_work_ids.setdefault(point.point_id, set())
        ordered_works = [
            work
            for _, work in sorted(
                enumerate(works),
                key=lambda item: (
                    not self._work_has_query_role(item[1], store, "graph_seed"),
                    item[0],
                ),
            )
        ]
        for work in ordered_works:
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
            if state.resource_ledger is not None:
                classification_cap = state.resource_ledger.caps.relation_classifications
                graph_seed = self._work_has_query_role(work, store, "graph_seed")
                if (
                    reserve_topology
                    and not graph_seed
                    and self._pending_topology_expansion(state)
                ):
                    classification_cap = max(
                        0,
                        classification_cap
                        - min(
                            2,
                            state.resource_ledger.caps.neighbor_expansions,
                        ),
                    )
                if (
                    state.resource_ledger.relation_classification_calls
                    >= classification_cap
                ):
                    point.validation_notes.append(
                        "relation_classification_cap_exhausted"
                    )
                    break
                state.resource_ledger.relation_classification_calls += 1
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
            if card.relation_label == RelationLabel.UNRESOLVED:
                store.add_evidence(
                    f"U:{card.relation_id}",
                    "candidate_relation_unresolved",
                    card,
                )
                point.validation_notes.append(
                    "unresolved_candidate_excluded_from_relation_store"
                )
                continue
            relation_key = f"R:{card.relation_id}"
            store.add_evidence(relation_key, "prior_relation", card)
            if relation_key not in state.relation_evidence_keys:
                state.relation_evidence_keys.append(relation_key)
            claim_relevant = relation_payload_is_claim_relevant(
                card.model_dump(mode="json")
            )
            if claim_relevant and relation_key not in point.relation_evidence_keys:
                point.relation_evidence_keys.append(relation_key)
            elif not claim_relevant:
                point.validation_notes.append(
                    "low_facet_relation_retained_for_audit_only"
                )
            if claim_relevant:
                labels.add(card.relation_label)
                if card.relation_label != RelationLabel.UNRESOLVED:
                    self._valid_prior_work_ids.add(card.prior_work_id)
        # Retrieval is append-only.  A later citation or contrastive batch that
        # contains only distant/unresolved works must not erase a stronger
        # relation established by an earlier batch for the same review point.
        for relation_key in point.relation_evidence_keys:
            relation_record = store.get(relation_key)
            payload = relation_record.payload if relation_record is not None else {}
            if not relation_payload_is_claim_relevant(payload):
                continue
            try:
                labels.add(RelationLabel(str(payload.get("relation_label"))))
            except ValueError:
                continue
        if point.section == "novelty_support" and self._complete_direct_antecedent(
            point, store
        ):
            point.section = "novelty_limit"
            point.novelty_resolution = "antecedent_found"
            point.validation_status = PointValidationStatus.EXTERNALLY_VALIDATED
            point.resolved_proposition = (
                "Prior literature contains a direct antecedent covering the "
                "essential facets of this novelty claim; the defensible contribution is the technical "
                f"difference stated in the paired evidence for: {point.proposition}"
            )
            point.validation_notes.append(
                "complete_direct_antecedent_reclassified_as_novelty_limit"
            )
        elif point.section == "novelty_support" and labels & DIRECT_ANTECEDENTS:
            point.novelty_resolution = "inconclusive"
            point.validation_status = PointValidationStatus.VALIDATED
            point.novelty_confidence = min(point.novelty_confidence or 1.0, 0.45)
            point.resolved_proposition = (
                "A retrieved work may be a direct antecedent, but essential-facet "
                "coverage and independent verification are incomplete. The authors "
                f"should clarify the residual delta: {point.proposition}"
            )
            point.validation_notes.append("direct_antecedent_not_fully_verified")
        elif point.section == "novelty_support" and labels & RESIDUAL_RELATIONS:
            point.novelty_resolution = "incremental_or_parallel"
            point.validation_status = PointValidationStatus.EXTERNALLY_VALIDATED
            if not point.resolved_proposition:
                point.resolved_proposition = (
                    "The contribution retains a bounded residual delta beyond its "
                    f"shared prior-art base: {point.proposition}"
                )
            point.validation_notes.append(
                "residual_delta_retained_after_partial_relation"
            )
        elif (
            point.section == "questions"
            and point.aspect
            in {ReviewAspect.NOVELTY_PRIOR_ART, ReviewAspect.CONTRIBUTION}
            and labels & RESIDUAL_RELATIONS
        ):
            point.section = "novelty_support"
            point.novelty_resolution = "incremental_or_parallel"
            point.validation_status = PointValidationStatus.EXTERNALLY_VALIDATED
            point.resolved_proposition = (
                "Paired prior-art evidence identifies a bounded residual delta: "
                f"{point.proposition}"
            )
            point.validation_notes.append("verified_relation_resolved_novelty_question")
        elif point.section == "novelty_limit" and self._limiting_relation_consensus(
            point, store
        ):
            point.novelty_resolution = "antecedent_found"
            point.validation_status = PointValidationStatus.EXTERNALLY_VALIDATED
            point.resolved_proposition = (
                "Independent paired prior-art evidence materially bounds this "
                f"novelty claim while leaving its stated residual delta auditable: {point.proposition}"
            )
        elif point.section == "novelty_limit" and labels & LIMITING:
            point.novelty_resolution = "inconclusive"
            point.validation_status = PointValidationStatus.EXTERNALLY_VALIDATED
            point.novelty_confidence = min(point.novelty_confidence or 1.0, 0.55)
            point.resolved_proposition = (
                "One paired comparison identifies partial prior-art overlap, but "
                "independent limiting evidence is insufficient to change the "
                f"paper-level novelty direction: {point.proposition}"
            )
        elif point.section == "novelty_limit" and self._residual_relation_consensus(
            point, store
        ):
            point.section = "novelty_support"
            point.novelty_resolution = "incremental_or_parallel"
            point.validation_status = PointValidationStatus.EXTERNALLY_VALIDATED
            point.resolved_proposition = (
                "Independent paired comparisons confirm a shared prior-art base "
                "while consistently isolating a bounded residual contribution: "
                f"{point.proposition}"
            )
            point.validation_notes.append(
                "independent_residual_consensus_resolved_novelty_limit"
            )
        elif labels & CONTEXTUAL_RELATIONS:
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
        self._record_correction(
            point,
            state,
            store,
            before_section=before_section,
            before_text=before_text,
            before_confidence=before_confidence,
        )

    @staticmethod
    def _allowed_query_roles(
        guidance: GraphClaimGuidanceV1,
        *,
        prefer_profile: bool,
    ) -> list[str]:
        del prefer_profile
        local_roles = ["author_terminology", "object_problem"]
        remote_roles = ["mechanism_outcome", "purpose_semantic"]
        allowed = {*local_roles, *remote_roles}
        score_roles = [
            role
            for mission in guidance.missions
            if mission.origin in {"score", "rescue"}
            for role in mission.query_roles
            if role in allowed
        ]
        if not score_roles:
            score_roles = (
                remote_roles + local_roles
                if guidance.allocated_remote_query_slots
                > guidance.allocated_local_query_slots
                else local_roles + remote_roles
            )
        profile_roles = [
            role
            for mission in guidance.missions
            if mission.origin == "profile"
            for role in mission.query_roles
            if role in allowed
        ]
        # Topology may change the content of an allocated query, but it must not
        # silently change the Score/Profile query-family allocation.  Keeping
        # this order invariant makes Full versus Score+Profile a true content-
        # guidance comparison under the same logical resource geometry.
        ordered = [score_roles[0], *profile_roles, *score_roles[1:]]
        return list(dict.fromkeys(ordered))

    @staticmethod
    def _work_has_query_role(
        work: RetrievedWork,
        store: EvidenceStore,
        role: str,
    ) -> bool:
        return any(
            (query_record := store.get(f"Q:{query_id}")) is not None
            and (
                query_record.payload.get("query_role") == role
                or (
                    role == "graph_seed"
                    and str(query_record.payload.get("transformation", "")).startswith(
                        "graph_claim_aligned_topology_search:"
                    )
                )
            )
            for query_id in work.source_query_ids
        )

    @staticmethod
    def _topology_seed_has_value(
        point: CanonicalReviewPoint,
        store: EvidenceStore,
    ) -> bool:
        for key in point.relation_evidence_keys:
            record = store.get(key)
            payload = record.payload if record is not None else {}
            if not relation_payload_is_claim_relevant(payload):
                continue
            graph_sourced = any(
                (query_record := store.get(f"Q:{query_id}")) is not None
                and (
                    query_record.payload.get("query_role") == "graph_seed"
                    or str(
                        query_record.payload.get("transformation", "")
                    ).startswith("graph_claim_aligned_topology_search:")
                )
                for query_id in payload.get("source_query_ids") or []
            )
            if not graph_sourced:
                continue
            try:
                label = RelationLabel(str(payload.get("relation_label")))
            except ValueError:
                continue
            coverage = float(payload.get("essential_facet_coverage") or 0.0)
            # A generic method-level overlap is useful as a direct candidate,
            # but it is too weak an anchor for citation traversal.  In that
            # case spend the already-reserved slot on a claim-specific provider
            # fallback instead of following an unrelated citation neighborhood.
            if coverage < 0.5:
                continue
            if point.section == "novelty_limit":
                return label in LIMITING and is_prior_art_direction_claim(point)
            if label not in {RelationLabel.UNRESOLVED, RelationLabel.DISTANT}:
                return bool(
                    payload.get("common_dimensions")
                    and payload.get("difference_dimensions")
                )
        return False

    def _topology_provider_fallback(
        self,
        point: CanonicalReviewPoint,
        state: ReviewStateV3,
        paper_ir: PaperIR,
        store: EvidenceStore,
        prior_art: PriorArtService,
        classifier: RelationClassifier,
        claim: PaperClaim,
        span: EvidenceSpan,
        guidance: GraphClaimGuidanceV1 | None,
    ) -> None:
        kwargs: dict[str, Any] = {
            "family": "normal",
            "target_span": span,
            "paper_ir": paper_ir,
        }
        parameters = inspect.signature(prior_art.retrieve).parameters
        if guidance is not None and "allowed_query_roles" in parameters:
            kwargs["allowed_query_roles"] = self._allowed_query_roles(
                guidance,
                prefer_profile=True,
            )
        if "max_provider_queries" in parameters:
            kwargs["max_provider_queries"] = 1
        if "resource_ledger" in parameters:
            kwargs["resource_ledger"] = state.resource_ledger
        works = prior_art.retrieve(
            claim,
            state.cutoff_date,
            self._budget(point.point_id, state),
            **kwargs,
        )
        self._store_retrieval_audit(claim, prior_art, store)
        self._record_retrieval_outcome(point, state, prior_art)
        self._works.setdefault(point.point_id, []).extend(works)
        self._classify(point, state, span, claim, works, store, classifier)
        self._store_coverage(point, state, claim, prior_art, store)
        point.validation_notes.append("topology_low_yield_provider_fallback")

    @staticmethod
    def _pending_topology_expansion(state: ReviewStateV3) -> bool:
        plan = state.graph_guidance_plan
        if plan is None:
            return False
        point_map = state.canonical_points
        return any(
            any(
                mission.origin == "topology" and mission.traversal != "none"
                for mission in guidance.missions
            )
            and guidance.review_point_id in point_map
            and not point_map[guidance.review_point_id].citation_expanded
            for guidance in plan.claim_guidance
        )

    @staticmethod
    def _residual_relation_consensus(
        point: CanonicalReviewPoint, store: EvidenceStore
    ) -> bool:
        work_ids: set[str] = set()
        for key in point.relation_evidence_keys:
            record = store.get(key)
            payload = record.payload if record is not None else {}
            try:
                label = RelationLabel(str(payload.get("relation_label")))
            except ValueError:
                continue
            if not relation_payload_is_claim_relevant(payload):
                continue
            if label in LIMITING or label == RelationLabel.CONFLICT:
                return False
            if (
                label in RESIDUAL_RELATIONS
                and payload.get("common_dimensions")
                and payload.get("difference_dimensions")
                and payload.get("prior_work_id")
            ):
                work_ids.add(str(payload["prior_work_id"]))
        return len(work_ids) >= 2

    @staticmethod
    def _limiting_relation_consensus(
        point: CanonicalReviewPoint, store: EvidenceStore
    ) -> bool:
        if not is_prior_art_direction_claim(point):
            return False
        partial_rows: list[dict[str, Any]] = []
        for key in point.relation_evidence_keys:
            record = store.get(key)
            payload = record.payload if record is not None else {}
            if not relation_payload_is_claim_relevant(payload):
                continue
            label = str(payload.get("relation_label"))
            if (
                label == "DIRECT_ANTECEDENT"
                and payload.get("independent_verification_passed") is True
                and float(payload.get("essential_facet_coverage", 0.0)) >= 0.9
            ):
                return True
            if (
                label in {"PARTIAL_ANTECEDENT", "BUILDING_BLOCK"}
                and payload.get("common_dimensions")
                and payload.get("difference_dimensions")
                and payload.get("prior_work_id")
            ):
                partial_rows.append(dict(payload))
        if is_absolute_priority_claim(point) and any(
            float(row.get("essential_facet_coverage", 0.0)) >= 0.5
            for row in partial_rows
        ):
            return True
        for index, left in enumerate(partial_rows):
            for right in partial_rows[index + 1 :]:
                if left["prior_work_id"] == right["prior_work_id"]:
                    continue
                overlap = EvidenceSupervisor._relation_dimension_tokens(
                    left
                ) & EvidenceSupervisor._relation_dimension_tokens(right)
                if len(overlap) >= 3:
                    return True
        return False

    @staticmethod
    def _limiting_relation_rows(
        point: CanonicalReviewPoint,
        store: EvidenceStore,
    ) -> list[tuple[str, dict[str, Any]]]:
        rows: list[tuple[str, dict[str, Any]]] = []
        for key in point.relation_evidence_keys:
            record = store.get(key)
            payload = dict(record.payload) if record is not None else {}
            if not relation_payload_is_claim_relevant(payload):
                continue
            if payload.get("relation_label") not in {
                "DIRECT_ANTECEDENT",
                "PARTIAL_ANTECEDENT",
                "BUILDING_BLOCK",
            }:
                continue
            if not (
                payload.get("prior_work_id")
                and payload.get("common_dimensions")
                and payload.get("difference_dimensions")
            ):
                continue
            rows.append((key, payload))
        return rows

    @staticmethod
    def _relation_dimension_tokens(payload: dict[str, Any]) -> set[str]:
        generic = {
            "analysis",
            "approach",
            "evidence",
            "associated",
            "association",
            "binding",
            "binds",
            "method",
            "interaction",
            "interactions",
            "involves",
            "involving",
            "paper",
            "result",
            "study",
            "system",
            "work",
        }
        text = " ".join(str(item) for item in payload.get("common_dimensions") or [])
        text = text.replace("-", " ")
        return {
            token
            for token in re.findall(r"[a-z][a-z0-9-]+", text.casefold())
            if len(token) > 2 and token not in generic
        }

    @staticmethod
    def _apply_cross_point_limiting_consensus(
        state: ReviewStateV3,
        store: EvidenceStore,
    ) -> None:
        if state.novelty_direction != NoveltyJudgment.POSITIVE:
            return
        points = list(state.canonical_points.values())
        limits = [
            point
            for point in points
            if point.retained
            and (point.initial_section or point.section) == "novelty_limit"
            and is_prior_art_direction_claim(point)
        ]
        for limit in limits:
            primary_rows = EvidenceSupervisor._limiting_relation_rows(limit, store)
            for other in points:
                if other.point_id == limit.point_id or not other.retained:
                    continue
                other_rows = EvidenceSupervisor._limiting_relation_rows(other, store)
                for primary_key, primary in primary_rows:
                    for other_key, secondary in other_rows:
                        if primary["prior_work_id"] == secondary["prior_work_id"]:
                            continue
                        overlap = EvidenceSupervisor._relation_dimension_tokens(
                            primary
                        ) & EvidenceSupervisor._relation_dimension_tokens(secondary)
                        if len(overlap) < 3:
                            continue
                        before_text = limit.resolved_proposition or limit.proposition
                        before_confidence = limit.novelty_confidence
                        limit.novelty_resolution = "antecedent_found"
                        limit.validation_status = (
                            PointValidationStatus.EXTERNALLY_VALIDATED
                        )
                        limit.resolved_proposition = (
                            "Independent paired comparisons across aligned "
                            "contribution facets materially bound this novelty "
                            f"claim while retaining its residual delta: {limit.proposition}"
                        )
                        limit.novelty_confidence = min(
                            limit.novelty_confidence or 0.60,
                            0.60,
                        )
                        guidance = [
                            item
                            for item in (
                                _claim_guidance(state, limit.point_id),
                                _claim_guidance(state, other.point_id),
                            )
                            if item is not None
                        ]
                        event = ReviewCorrectionEventV1(
                            point_id=limit.point_id,
                            before_text=before_text,
                            after_text=limit.resolved_proposition,
                            before_section=limit.section,
                            after_section=limit.section,
                            before_direction=NoveltyJudgment.POSITIVE,
                            after_direction=NoveltyJudgment.MIXED,
                            trigger_relation_ids=[
                                primary_key.removeprefix("R:"),
                                other_key.removeprefix("R:"),
                            ],
                            trigger_mission_ids=list(
                                dict.fromkeys(
                                    mission.mission_id
                                    for item in guidance
                                    for mission in item.missions
                                )
                            ),
                            correction_type="partial_antecedent_refinement",
                            confidence_change=(limit.novelty_confidence or 0.0)
                            - (before_confidence or 0.0),
                        )
                        event_id = (
                            "RC:"
                            + hashlib.sha256(
                                event.model_dump_json().encode("utf-8")
                            ).hexdigest()[:18]
                        )
                        store.add_evidence(
                            event_id,
                            "review_correction_event",
                            event,
                        )
                        if event_id not in state.correction_event_evidence_keys:
                            state.correction_event_evidence_keys.append(event_id)
                        state.novelty_direction = NoveltyJudgment.MIXED
                        return

    @staticmethod
    def _record_correction(
        point: CanonicalReviewPoint,
        state: ReviewStateV3,
        store: EvidenceStore,
        *,
        before_section: str,
        before_text: str,
        before_confidence: float | None,
    ) -> None:
        after_text = point.resolved_proposition or point.proposition
        if before_section == point.section and before_text == after_text:
            return
        verified_relations = EvidenceSupervisor._correction_relation_ids(point, store)
        if not verified_relations:
            return
        guidance = _claim_guidance(state, point.point_id)
        correction_type: Literal[
            "direct_antecedent_challenge",
            "partial_antecedent_refinement",
            "residual_novelty_refinement",
            "attribution_scope_refinement",
            "confidence_downgrade",
            "confidence_upgrade",
            "prior_work_added_only",
        ] = "prior_work_added_only"
        if before_section == "novelty_support" and point.section == "novelty_limit":
            correction_type = "direct_antecedent_challenge"
        elif (
            before_section == "novelty_limit"
            and point.novelty_resolution == "antecedent_found"
        ):
            correction_type = "partial_antecedent_refinement"
        elif point.novelty_resolution == "incremental_or_parallel":
            correction_type = "residual_novelty_refinement"
        elif point.novelty_resolution == "inconclusive":
            correction_type = "partial_antecedent_refinement"
        after_confidence = point.novelty_confidence
        before_direction = getattr(state, "novelty_direction", None)
        after_direction = EvidenceSupervisor._direction_after_correction(
            state,
            point,
            before_section=before_section,
            before_direction=before_direction,
        )
        if hasattr(state, "novelty_direction"):
            state.novelty_direction = after_direction
        event = ReviewCorrectionEventV1(
            point_id=point.point_id,
            before_text=before_text,
            after_text=after_text,
            before_section=before_section,
            after_section=point.section,
            before_direction=before_direction,
            after_direction=after_direction,
            trigger_relation_ids=verified_relations,
            trigger_mission_ids=(
                [mission.mission_id for mission in guidance.missions]
                if guidance is not None
                else []
            ),
            correction_type=correction_type,
            confidence_change=(after_confidence or 0.0) - (before_confidence or 0.0),
        )
        event_id = (
            "RC:"
            + hashlib.sha256(event.model_dump_json().encode("utf-8")).hexdigest()[:18]
        )
        store.add_evidence(event_id, "review_correction_event", event)
        correction_keys = getattr(state, "correction_event_evidence_keys", None)
        if correction_keys is not None and event_id not in correction_keys:
            correction_keys.append(event_id)

    @staticmethod
    def _correction_relation_ids(
        point: CanonicalReviewPoint,
        store: EvidenceStore,
    ) -> list[str]:
        selected: list[str] = []
        for key in point.relation_evidence_keys:
            record = store.get(key)
            payload = record.payload if record is not None else {}
            if not relation_payload_is_claim_relevant(payload):
                continue
            try:
                label = RelationLabel(str(payload.get("relation_label")))
            except ValueError:
                continue
            if label in {RelationLabel.DISTANT, RelationLabel.UNRESOLVED}:
                continue
            if point.novelty_resolution == "antecedent_found" and label not in LIMITING:
                continue
            if (
                point.novelty_resolution == "incremental_or_parallel"
                and label not in CONTEXTUAL_RELATIONS
            ):
                continue
            if point.novelty_resolution == "inconclusive" and label not in LIMITING:
                continue
            selected.append(key.removeprefix("R:"))
        return selected

    @staticmethod
    def _direction_after_correction(
        state: ReviewStateV3,
        point: CanonicalReviewPoint,
        *,
        before_section: str,
        before_direction: NoveltyJudgment | None,
    ) -> NoveltyJudgment | None:
        """Apply only direction changes justified by a verified point correction."""
        if before_section == "novelty_support" and (point.section == "novelty_limit"):
            remaining_support = any(
                candidate.retained
                and candidate.point_id != point.point_id
                and candidate.section == "novelty_support"
                for candidate in getattr(state, "canonical_points", {}).values()
            )
            return (
                NoveltyJudgment.MIXED if remaining_support else NoveltyJudgment.NEGATIVE
            )
        if before_section == "novelty_limit" and point.section == "novelty_support":
            remaining_limit = any(
                candidate.retained
                and candidate.point_id != point.point_id
                and candidate.section == "novelty_limit"
                for candidate in getattr(state, "canonical_points", {}).values()
            )
            return (
                NoveltyJudgment.MIXED if remaining_limit else NoveltyJudgment.POSITIVE
            )
        if (
            before_section == "novelty_limit"
            and point.section == "novelty_limit"
            and point.novelty_resolution == "antecedent_found"
            and before_direction == NoveltyJudgment.POSITIVE
        ):
            return NoveltyJudgment.MIXED
        if (
            before_direction
            in {
                NoveltyJudgment.UNCERTAIN,
                NoveltyJudgment.NOT_DISCUSSED,
            }
            and point.section == "novelty_support"
        ):
            has_limit = any(
                candidate.retained and candidate.section == "novelty_limit"
                for candidate in getattr(state, "canonical_points", {}).values()
            )
            return NoveltyJudgment.MIXED if has_limit else NoveltyJudgment.POSITIVE
        return before_direction

    @staticmethod
    def _stability(
        point: CanonicalReviewPoint,
        state: ReviewStateV3,
        store: EvidenceStore,
    ) -> None:
        # Graph-removal invariant holds because Graph never contributes relation labels.
        valid_work_ids = set()
        antecedent_query_roles: set[str] = set()
        antecedent_work_ids: set[str] = set()
        for key in point.relation_evidence_keys:
            record = store.get(key)
            payload = record.payload if record is not None else {}
            if relation_payload_is_claim_relevant(payload):
                valid_work_ids.add(str(payload["prior_work_id"]))
                if payload.get("relation_label") == "DIRECT_ANTECEDENT":
                    antecedent_work_ids.add(str(payload["prior_work_id"]))
                    for query_id in payload.get("source_query_ids") or []:
                        query_record = store.get(f"Q:{query_id}")
                        if query_record and query_record.payload.get("query_role"):
                            antecedent_query_roles.add(
                                str(query_record.payload["query_role"])
                            )
        coverage = EvidenceSupervisor._latest_coverage(point, store)
        if point.novelty_resolution == "antecedent_found":
            independently_verified = any(
                (
                    (record := store.get(key)) is not None
                    and record.payload.get("relation_label") == "DIRECT_ANTECEDENT"
                    and record.payload.get("independent_verification_passed") is True
                    and float(record.payload.get("essential_facet_coverage", 0.0))
                    >= 0.9
                )
                for key in point.relation_evidence_keys
            )
            stable = (
                independently_verified
                or len(antecedent_work_ids) >= 2
                or len(antecedent_query_roles) >= 2
            )
            if not stable and coverage and coverage.get("service_failed") is False:
                point.novelty_resolution = "inconclusive"
                point.resolved_proposition = (
                    "One retrieved source suggests a possible antecedent, but the "
                    "audited search did not independently confirm that relationship. "
                    "The authors should compare the claimed contribution directly "
                    "with that nearest work and state the technical difference."
                )
                point.validation_status = PointValidationStatus.VALIDATED
                point.novelty_confidence = min(point.novelty_confidence or 1.0, 0.45)
                point.stability_status = "not_required"
                point.validation_notes.append(
                    "single_antecedent_downgraded_to_question"
                )
                return
        elif coverage and coverage.get("service_failed") is True:
            EvidenceSupervisor._downgrade_external_gap(point, None)
            return
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
            point.novelty_resolution = "inconclusive"
            point.resolved_proposition = (
                "The audited search did not provide enough evidence-bearing prior "
                "work for a reliable novelty conclusion. The authors should compare "
                "the contribution more explicitly with its nearest prior work: "
                f"{_without_absolute_priority(point.proposition)}"
            )
            point.validation_status = PointValidationStatus.VALIDATED
            point.novelty_confidence = min(point.novelty_confidence or 1.0, 0.35)
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
    def _complete_direct_antecedent(
        point: CanonicalReviewPoint,
        store: EvidenceStore,
    ) -> bool:
        """Return true only for independently confirmed full-facet antecedence."""
        for key in point.relation_evidence_keys:
            record = store.get(key)
            payload = record.payload if record is not None else {}
            if (
                payload.get("temporal_valid") is True
                and payload.get("relation_label") == "DIRECT_ANTECEDENT"
                and float(payload.get("essential_facet_coverage", 0.0)) >= 1.0
                and payload.get("independent_verification_passed") is True
            ):
                return True
        return False

    @staticmethod
    def _record_retrieval_outcome(
        point: CanonicalReviewPoint,
        state: ReviewStateV3,
        prior_art: PriorArtService,
    ) -> None:
        if getattr(prior_art, "last_service_failed", False):
            point.validation_notes.append("prior_art_service_failed_downgraded")
            if not any(
                failure.reason == "retrieval_unavailable"
                and failure.claim_id == point.point_id
                for failure in state.failures
            ):
                state.failures.append(
                    FailureRecord(
                        stage="prior_art",
                        reason="retrieval_unavailable",
                        claim_id=point.point_id,
                    )
                )
            for reason in getattr(prior_art, "last_failures", []):
                point.validation_notes.append(f"prior_art_advisory:{reason}")
        else:
            point.validation_notes.extend(
                str(note) for note in getattr(prior_art, "last_advisories", [])
            )

    @staticmethod
    def _downgrade_external_gap(
        point: CanonicalReviewPoint, action: EvidenceAction | None
    ) -> None:
        """Keep an auditable manuscript-grounded question when search is unavailable."""
        if action == EvidenceAction.SEARCH_PRIOR_ART:
            point.normal_search_done = True
        elif action == EvidenceAction.COUNTERFACTUAL_SEARCH:
            point.counterfactual_search_count += 1
            point.counterfactual_search_done = point.counterfactual_search_count >= 1
        elif action == EvidenceAction.CITATION_EXPAND:
            point.citation_expanded = True
        if point.requires_external_evidence:
            point.novelty_resolution = "inconclusive"
            point.novelty_confidence = min(point.novelty_confidence or 1.0, 0.30)
            point.resolved_proposition = (
                "The manuscript-grounded contribution requires clearer comparison "
                "with the nearest prior work; external retrieval was incomplete, "
                f"so no priority conclusion is asserted: {_without_absolute_priority(point.proposition)}"
            )
        point.validation_status = PointValidationStatus.VALIDATED
        point.stability_status = "not_required"
        point.validation_notes.append("external_retrieval_gap_downgraded_to_question")

    @staticmethod
    def _store_coverage(
        point: CanonicalReviewPoint,
        state: ReviewStateV3,
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
            if record and record.payload.get("relation_label") == "DIRECT_ANTECEDENT":
                direct_found = True
                break
        card = build(
            claim.claim_id,
            state.cutoff_date,
            require_contrastive=is_high_risk(point),
            direct_or_partial_found=direct_found,
        )
        base_key = f"COV:{card.coverage_id}"
        existing = store.get(base_key)
        key = base_key
        if existing is not None and sha256_value(existing.payload) != sha256_value(
            card
        ):
            key = f"{base_key}:{sha256_value(card).removeprefix('sha256:')[:12]}"
        if not store.has(key):
            store.add_evidence(key, "retrieval_coverage", card)
        point.coverage_evidence_keys = [key]

    def _budget(self, point_id: str, state: ReviewStateV3) -> RetrievalBudget:
        if point_id not in self._budgets:
            guidance = (
                None
                if self.config.graph_guidance.shadow
                else _claim_guidance(state, point_id)
            )
            planned_normal_max = (
                guidance.allocated_local_query_slots
                + guidance.allocated_remote_query_slots
                if guidance is not None
                else state.action_budget.normal_per_claim_max
            )
            if guidance is not None and any(
                "legacy_contrastive" in mission.query_roles
                for mission in guidance.missions
            ):
                planned_normal_max = max(0, planned_normal_max - 1)
            self._budgets[point_id] = RetrievalBudget(
                normal_max=planned_normal_max,
                contrastive_max=(state.action_budget.counterfactual_per_claim_max),
                citation_expansion_max=state.action_budget.citation_per_claim_max,
                fulltext_max=min(
                    self.config.retrieval.fulltext_max,
                    state.action_budget.relation_cards_max,
                ),
            )
        return self._budgets[point_id]

    @staticmethod
    def _finalize(state: ReviewStateV3) -> None:
        for point in state.canonical_points.values():
            if point.validation_status == PointValidationStatus.UNRESOLVED:
                point.validation_notes.append("finalized_unresolved")
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
    def _exhaust(state: ReviewStateV3) -> None:
        for point in state.canonical_points.values():
            if point.validation_status not in {
                PointValidationStatus.VALIDATED,
                PointValidationStatus.REJECTED,
            }:
                point.validation_notes.append("evidence_action_budget_exhausted")
                if (point.initial_section or point.section).startswith("novelty_"):
                    point.validation_status = PointValidationStatus.VALIDATED
                    point.novelty_resolution = "inconclusive"
                    point.stability_status = "not_required"
                    point.novelty_confidence = min(
                        point.novelty_confidence or 1.0, 0.30
                    )
                    point.resolved_proposition = (
                        "The graph-blind review direction remains provisional because "
                        "the evidence-action budget did not permit a complete prior-art "
                        f"check: {_without_absolute_priority(point.proposition)}"
                    )
                else:
                    point.validation_status = PointValidationStatus.UNRESOLVED
                if (
                    point.severity == PointSeverity.MAJOR
                    and point.validation_status == PointValidationStatus.UNRESOLVED
                ):
                    point.retained = False
        EvidenceSupervisor._finalize(state)

    def _update_process_features(self, state: ReviewStateV3) -> None:
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
    def _update_aspr_assessments(state: ReviewStateV3) -> None:
        assessments = {}
        for point in state.canonical_points.values():
            if not point.section.startswith("novelty_"):
                continue
            claim_id = point.novelty_claim_id or point.point_id
            assessments[claim_id] = {
                "antecedent_found": "CHALLENGED",
                "incremental_or_parallel": "REFINED",
                "bounded_no_antecedent": "NOT_CHALLENGED",
                "inconclusive": "INCONCLUSIVE",
                "search_failed": "INCONCLUSIVE",
                "not_applicable": "NOT_APPLICABLE",
            }[point.novelty_resolution]
        state.aspr_evidence_assessments = assessments

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
        state: ReviewStateV3,
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


def build_retrieval_claim(
    point: CanonicalReviewPoint, paper_ir: PaperIR
) -> tuple[PaperClaim, EvidenceSpan]:
    """Build an evidence-anchored retrieval claim from the fused review point.

    The point's proposition is the reviewer-normalized scientific question.
    Prefer its linked scientific spans, but never use a bibliography entry as
    the target of a paper-to-prior-work relation when an extracted paper claim
    is available.  Any deterministic re-anchoring is added to the point trace.
    """
    span_map = paper_ir.span_map()
    linked = [
        span_map[key.removeprefix("P:")]
        for key in point.paper_evidence_keys
        if key.startswith("P:") and key.removeprefix("P:") in span_map
    ]
    if not linked:
        raise ValueError("canonical point has no valid target span")
    reference_span_ids = {reference.source_span_id for reference in paper_ir.references}
    candidates = [span for span in linked if span.span_id not in reference_span_ids]
    if not candidates:
        claim_span_ids = list(
            dict.fromkeys(
                claim.span_id
                for claim in paper_ir.claims
                if claim.span_id in span_map and claim.span_id not in reference_span_ids
            )
        )
        candidates = [span_map[span_id] for span_id in claim_span_ids]
    if not candidates:
        candidates = linked
    proposition_tokens = _retrieval_tokens(point.proposition)
    span = max(
        candidates,
        key=lambda candidate: (
            len(proposition_tokens & _retrieval_tokens(candidate.text)),
            min(len(candidate.text), 2_000),
            -candidate.char_start,
        ),
    )
    span_id = span.span_id
    target_key = f"P:{span_id}"
    if target_key not in point.paper_evidence_keys:
        point.paper_evidence_keys.append(target_key)
        point.validation_notes.append("retrieval_target_span_reanchored")
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


def _retrieval_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z][a-z0-9-]+", text.casefold())
        if len(token) > 2
        and token
        not in {
            "claim",
            "contribution",
            "evidence",
            "external",
            "literature",
            "manuscript",
            "paper",
            "prior",
            "study",
            "work",
        }
    }


def _without_absolute_priority(text: str) -> str:
    cleaned = re.sub(
        r"\b(?:first|first-ever|unprecedented|unique|world-first)\b|"
        r"首次|首个|前所未有|唯一",
        "claimed",
        str(text),
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", cleaned).strip()


__all__ = ["EvidenceSupervisor"]

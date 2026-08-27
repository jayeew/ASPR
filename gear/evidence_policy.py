"""Rule-first bounded action policy for post-fusion evidence gaps."""

from __future__ import annotations

import re

from .review_contracts import (
    CanonicalReviewPoint,
    EvidenceAction,
    PointSeverity,
    PointValidationStatus,
    ReviewState,
)

STRONG_CLAIM_RE = re.compile(
    r"\b(?:first|first-ever|unprecedented|breakthrough|paradigm shift)\b|首次|首个|突破",
    re.IGNORECASE,
)


def is_high_risk(point: CanonicalReviewPoint) -> bool:
    return bool(
        STRONG_CLAIM_RE.search(point.proposition)
        or point.qwen_conflict
        or (
            point.section.startswith("novelty_")
            and point.severity == PointSeverity.MAJOR
        )
    )


def _has_topology_mission(state: ReviewState, point_id: str) -> bool:
    plan = state.graph_guidance_plan
    if plan is None:
        return False
    return any(
        guidance.review_point_id == point_id
        and any(
            mission.origin == "topology" and mission.traversal != "none"
            for mission in guidance.missions
        )
        for guidance in plan.claim_guidance
    )


def _has_counterfactual_mission(state: ReviewState, point_id: str) -> bool:
    plan = state.graph_guidance_plan
    if plan is None:
        return True
    return any(
        guidance.review_point_id == point_id
        and any(
            "legacy_contrastive" in mission.query_roles for mission in guidance.missions
        )
        for guidance in plan.claim_guidance
    )


def highest_risk_unresolved_target(
    state: ReviewState,
) -> CanonicalReviewPoint | None:
    candidates = [
        point
        for point in state.canonical_points.values()
        if point.retained
        and not (
            point.validation_status == PointValidationStatus.UNRESOLVED
            and point.citation_expanded
            and point.stability_status in {"stable", "unstable"}
        )
        and point.validation_status
        not in {PointValidationStatus.VALIDATED, PointValidationStatus.REJECTED}
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda point: (
            point.graph_tension_score,
            point.graph_focus_weight,
            is_high_risk(point),
            point.severity == PointSeverity.MAJOR,
            point.requires_external_evidence,
            point.point_id,
        ),
    )


def next_evidence_action(
    state: ReviewState,
) -> tuple[EvidenceAction, str | None, str]:
    candidates = [
        point
        for point in state.canonical_points.values()
        if point.retained
        and point.validation_status
        not in {PointValidationStatus.VALIDATED, PointValidationStatus.REJECTED}
    ]

    def choose(items: list[CanonicalReviewPoint]) -> CanonicalReviewPoint | None:
        if not items:
            return None
        return max(
            items,
            key=lambda point: (
                point.graph_tension_score,
                point.graph_focus_weight,
                is_high_risk(point),
                point.severity == PointSeverity.MAJOR,
                point.requires_external_evidence,
                point.point_id,
            ),
        )

    target = choose([point for point in candidates if not point.semantic_verified])
    if target is not None:
        return (
            EvidenceAction.VERIFY_POINT,
            target.point_id,
            "paper_span_verification_required",
        )
    target = choose(
        [
            point
            for point in candidates
            if point.requires_external_evidence and not point.normal_search_done
        ]
    )
    if target is not None:
        return EvidenceAction.SEARCH_PRIOR_ART, target.point_id, "external_evidence_gap"
    target = choose(
        [
            point
            for point in candidates
            if point.requires_external_evidence
            and _has_topology_mission(state, point.point_id)
            and point.normal_search_done
            and not point.citation_expanded
        ]
    )
    if target is not None:
        return (
            EvidenceAction.CITATION_EXPAND,
            target.point_id,
            "graph_topology_traversal",
        )
    target = choose(
        [
            point
            for point in candidates
            if point.requires_external_evidence
            and (
                _has_counterfactual_mission(state, point.point_id)
                if state.graph_guidance_plan is not None
                else is_high_risk(point)
            )
            and point.counterfactual_search_count < 1
        ]
    )
    if target is not None:
        return (
            EvidenceAction.COUNTERFACTUAL_SEARCH,
            target.point_id,
            "high_risk_novelty_claim",
        )
    target = choose(
        [
            point
            for point in candidates
            if point.requires_external_evidence
            and state.graph_guidance_plan is None
            and point.validation_status == PointValidationStatus.UNRESOLVED
            and point.normal_search_done
            and (point.counterfactual_search_done or not is_high_risk(point))
            and not point.citation_expanded
        ]
    )
    if target is not None:
        return (
            EvidenceAction.CITATION_EXPAND,
            target.point_id,
            "prior_search_unresolved",
        )
    target = choose(
        [point for point in candidates if point.stability_status == "pending"]
    )
    if target is not None:
        return (
            EvidenceAction.STABILITY_TEST,
            target.point_id,
            "high_risk_stability_required",
        )
    return EvidenceAction.FINALIZE, None, "no_unresolved_target"


__all__ = [
    "STRONG_CLAIM_RE",
    "highest_risk_unresolved_target",
    "is_high_risk",
    "next_evidence_action",
]

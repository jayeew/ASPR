"""Rule-first bounded action policy for post-fusion evidence gaps."""

from __future__ import annotations

import re
from typing import Optional, Tuple

from .review_contracts import (
    CanonicalReviewPoint,
    EvidenceAction,
    PointSeverity,
    PointValidationStatus,
    ReviewStateV2,
)

STRONG_CLAIM_RE = re.compile(
    r"\b(?:first|first-ever|unprecedented|breakthrough|paradigm shift)\b|首次|首个|突破",
    re.I,
)


def is_high_risk(point: CanonicalReviewPoint) -> bool:
    return bool(
        STRONG_CLAIM_RE.search(point.proposition)
        or point.qwen_conflict
        or point.graph_tension
        or (
            point.section.startswith("novelty_")
            and point.severity == PointSeverity.MAJOR
        )
    )


def highest_risk_unresolved_target(
    state: ReviewStateV2,
) -> Optional[CanonicalReviewPoint]:
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
    return sorted(
        candidates,
        key=lambda point: (
            is_high_risk(point),
            point.severity == PointSeverity.MAJOR,
            point.requires_external_evidence,
            point.point_id,
        ),
        reverse=True,
    )[0]


def next_evidence_action(
    state: ReviewStateV2,
) -> Tuple[EvidenceAction, Optional[str], str]:
    candidates = [
        point
        for point in state.canonical_points.values()
        if point.retained
        and point.validation_status
        not in {PointValidationStatus.VALIDATED, PointValidationStatus.REJECTED}
    ]

    def choose(items: list[CanonicalReviewPoint]) -> Optional[CanonicalReviewPoint]:
        if not items:
            return None
        return sorted(
            items,
            key=lambda point: (
                is_high_risk(point),
                point.severity == PointSeverity.MAJOR,
                point.requires_external_evidence,
                point.point_id,
            ),
            reverse=True,
        )[0]

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
            and is_high_risk(point)
            and not point.counterfactual_search_done
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
    pending_stability = choose(
        [
            point
            for point in state.canonical_points.values()
            if point.retained and point.stability_status == "pending"
        ]
    )
    if pending_stability is not None:
        return (
            EvidenceAction.STABILITY_TEST,
            pending_stability.point_id,
            "stability_pending",
        )
    return EvidenceAction.FINALIZE, None, "no_unresolved_target"


__all__ = [
    "STRONG_CLAIM_RE",
    "highest_risk_unresolved_target",
    "is_high_risk",
    "next_evidence_action",
]

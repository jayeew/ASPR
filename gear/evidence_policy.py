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
    target = highest_risk_unresolved_target(state)
    if target is None:
        pending_stability = next(
            (
                point
                for point in state.canonical_points.values()
                if point.retained and point.stability_status == "pending"
            ),
            None,
        )
        if pending_stability is not None:
            return (
                EvidenceAction.STABILITY_TEST,
                pending_stability.point_id,
                "stability_pending",
            )
        return EvidenceAction.FINALIZE, None, "no_unresolved_target"
    if not target.semantic_verified:
        return (
            EvidenceAction.VERIFY_POINT,
            target.point_id,
            "paper_span_verification_required",
        )
    if target.requires_external_evidence and not target.normal_search_done:
        return EvidenceAction.SEARCH_PRIOR_ART, target.point_id, "external_evidence_gap"
    if (
        target.requires_external_evidence
        and is_high_risk(target)
        and not target.counterfactual_search_done
    ):
        return (
            EvidenceAction.COUNTERFACTUAL_SEARCH,
            target.point_id,
            "high_risk_novelty_claim",
        )
    if (
        target.requires_external_evidence
        and target.validation_status == PointValidationStatus.UNRESOLVED
        and target.normal_search_done
        and (target.counterfactual_search_done or not is_high_risk(target))
        and not target.citation_expanded
    ):
        return (
            EvidenceAction.CITATION_EXPAND,
            target.point_id,
            "prior_search_unresolved",
        )
    if target.stability_status == "pending":
        return (
            EvidenceAction.STABILITY_TEST,
            target.point_id,
            "high_risk_stability_required",
        )
    return EvidenceAction.VERIFY_POINT, target.point_id, "final_point_resolution"


__all__ = [
    "STRONG_CLAIM_RE",
    "highest_risk_unresolved_target",
    "is_high_risk",
    "next_evidence_action",
]

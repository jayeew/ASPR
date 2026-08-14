"""Initialize compact current state using references instead of raw evidence."""

from __future__ import annotations

import hashlib
from datetime import date

from .contracts import PaperIR
from .graph_prior_contracts import GraphPriorResult
from .review_contracts import (
    EvidenceBudget,
    GraphReviewContext,
    PaperSpecificRubric,
    ProcessFeatures,
    ReviewPhase,
    ReviewPointState,
    ReviewState,
    ReviewStateV2,
    StructuredReview,
)


def initialize_review_state(
    paper_ir: PaperIR,
    graph_context: GraphReviewContext,
    draft_review: StructuredReview,
    cutoff_date: date,
) -> ReviewState:
    if draft_review.paper_id != paper_ir.paper_id:
        raise ValueError("draft review paper_id differs from PaperIR")
    identity = (
        f"{paper_ir.paper_sha256}|{cutoff_date.isoformat()}|"
        f"{draft_review.model_dump_json()}"
    )
    return ReviewState(
        state_id="STATE-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:18],
        paper_id=paper_ir.paper_id,
        paper_sha256=paper_ir.paper_sha256,
        cutoff_date=cutoff_date,
        draft_review=draft_review,
        graph_context=graph_context,
        point_states={
            point.point_id: ReviewPointState(
                point_id=point.point_id,
                evidence_keys=list(point.evidence_keys),
            )
            for point in draft_review.all_points()
        },
    )


def initialize_review_state_v2(
    paper_ir: PaperIR,
    rubric: PaperSpecificRubric,
    graph_prior: GraphPriorResult,
    cutoff_date: date,
    *,
    graph_prior_evidence_key: str = "G:PRIOR",
    action_budget: EvidenceBudget | None = None,
) -> ReviewStateV2:
    if (
        rubric.paper_id != paper_ir.paper_id
        or graph_prior.paper_id != paper_ir.paper_id
    ):
        raise ValueError("V2 state inputs must refer to the same paper")
    identity = (
        f"{paper_ir.paper_sha256}|{cutoff_date.isoformat()}|"
        f"{rubric.model_dump_json()}|{graph_prior.model_dump_json()}"
    )
    return ReviewStateV2(
        state_id="STATE2-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:18],
        phase=ReviewPhase.INITIALIZED,
        paper_id=paper_ir.paper_id,
        paper_sha256=paper_ir.paper_sha256,
        cutoff_date=cutoff_date,
        rubric=rubric,
        branch_reviews={},
        graph_prior_evidence_key=graph_prior_evidence_key,
        graph_prior=graph_prior,
        action_budget=action_budget or EvidenceBudget(),
        process_features=ProcessFeatures(
            graph_score_available=graph_prior.score_0_100 is not None
        ),
    )


__all__ = ["initialize_review_state", "initialize_review_state_v2"]

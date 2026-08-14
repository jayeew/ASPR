"""Initialize compact current state using references instead of raw evidence."""

from __future__ import annotations

import hashlib
from datetime import date

from .contracts import PaperIR
from .review_contracts import (
    GraphReviewContext,
    ReviewPointState,
    ReviewState,
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


__all__ = ["initialize_review_state"]

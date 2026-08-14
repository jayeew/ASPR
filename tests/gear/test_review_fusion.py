from __future__ import annotations

from gear.graph_prior_contracts import GraphPriorResult
from gear.paper_extraction import PaperRubricBuilder
from gear.review_contracts import (
    BranchReview,
    NoveltyAssessment,
    NoveltyJudgment,
    ReviewAspect,
    ReviewPoint,
    ReviewSource,
    ReviewSummary,
)
from gear.review_fusion import ReviewFusion
from gear.review_state import initialize_review_state_v2


def _branch(paper_ir, source: ReviewSource) -> BranchReview:
    return BranchReview(
        paper_id=paper_ir.paper_id,
        source=source,
        model_id=source.value,
        prompt_sha256="sha256:prompt",
        input_sha256="sha256:input",
        summary=ReviewSummary(
            text="The paper presents an evidence controller.",
            evidence_keys=[f"P:{paper_ir.spans[0].span_id}"],
        ),
        novelty=NoveltyAssessment(
            judgment=NoveltyJudgment.NOT_DISCUSSED,
            supporting_points=[],
            limiting_points=[],
        ),
    )


def test_graph_cannot_create_review_point(paper_ir, paper_request) -> None:
    graph = GraphPriorResult(
        paper_id=paper_ir.paper_id,
        status="exact_lookup",
        score_0_100=99.0,
        feature_coverage=1.0,
    )
    state = initialize_review_state_v2(
        paper_ir,
        PaperRubricBuilder().build(paper_ir),
        graph,
        paper_request.evidence_date,
    )
    fused, report = ReviewFusion().fuse(state, _branch(paper_ir, ReviewSource.AGENT))
    assert fused.canonical_points == {}
    assert report.canonical_point_ids == []


def test_fusion_forces_external_evidence_for_misplaced_novelty_point(
    paper_ir, paper_request
) -> None:
    graph = GraphPriorResult(
        paper_id=paper_ir.paper_id,
        status="exact_lookup",
        score_0_100=50.0,
        feature_coverage=1.0,
    )
    state = initialize_review_state_v2(
        paper_ir,
        PaperRubricBuilder().build(paper_ir),
        graph,
        paper_request.evidence_date,
    )
    branch = _branch(paper_ir, ReviewSource.AGENT)
    branch.weaknesses = [
        ReviewPoint(
            point_id="misplaced-novelty",
            aspect=ReviewAspect.NOVELTY_PRIOR_ART,
            text="The nearest prior work requires external verification.",
            evidence_keys=[f"P:{paper_ir.spans[0].span_id}"],
            external_verification_required=False,
        )
    ]
    fused, _ = ReviewFusion().fuse(state, branch)
    point = next(iter(fused.canonical_points.values()))
    assert point.requires_external_evidence is True

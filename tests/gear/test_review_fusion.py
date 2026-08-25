from __future__ import annotations

from gear.evidence_supervisor import EvidenceSupervisor
from gear.graph_prior_contracts import GraphResultV3
from gear.paper_extraction import PaperRubricBuilder
from gear.review_compiler import ReviewCompiler
from gear.review_contracts import (
    BranchReview,
    NoveltyAssessment,
    NoveltyJudgment,
    NoveltyVerificationStatus,
    ReviewAspect,
    ReviewPoint,
    ReviewSource,
    ReviewSummary,
)
from gear.review_fusion import ReviewFusion
from gear.review_state import initialize_review_state_v3


def _graph(paper_id: str, score: float, coverage: float = 1.0) -> GraphResultV3:
    return GraphResultV3(
        paper_id=paper_id,
        score_0_100=score,
        p_uptake=0.8,
        conditional_diffusion=0.6,
        feature_coverage=coverage,
    )


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
    graph = _graph(paper_ir.paper_id, 99.0)
    state = initialize_review_state_v3(
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
    graph = _graph(paper_ir.paper_id, 50.0)
    state = initialize_review_state_v3(
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


def test_fusion_is_graph_blind_and_has_no_calibration_side_effects(
    paper_ir, paper_request
) -> None:
    span_key = f"P:{paper_ir.spans[0].span_id}"
    branch = _branch(paper_ir, ReviewSource.AGENT)
    branch.novelty = NoveltyAssessment(
        judgment=NoveltyJudgment.NEGATIVE,
        supporting_points=[],
        limiting_points=[
            ReviewPoint(
                point_id="novelty-limit",
                aspect=ReviewAspect.NOVELTY_PRIOR_ART,
                text="The contribution appears incremental.",
                evidence_keys=[span_key],
            )
        ],
    )
    branch.weaknesses = [
        ReviewPoint(
            point_id="method-point",
            aspect=ReviewAspect.METHOD,
            text="The method needs a clearer applicability boundary.",
            evidence_keys=[span_key],
        )
    ]
    state = initialize_review_state_v3(
        paper_ir,
        PaperRubricBuilder().build(paper_ir),
        _graph(paper_ir.paper_id, 95.0),
        paper_request.evidence_date,
    )

    fused, report = ReviewFusion().fuse(state, branch)

    novelty = next(
        point
        for point in fused.canonical_points.values()
        if point.section == "novelty_limit"
    )
    method = next(
        point
        for point in fused.canonical_points.values()
        if point.aspect == ReviewAspect.METHOD
    )
    assert novelty.graph_tension_score == 0.0
    assert novelty.graph_extra_counterfactual_actions == 0
    assert novelty.graph_focus_weight == 0.0
    assert method.graph_focus_weight == 0.0
    assert report.graph_tension_scores == {}
    assert report.graph_triggered_actions == {}


def test_legacy_coverage_does_not_modify_fused_points(paper_ir, paper_request) -> None:
    span_key = f"P:{paper_ir.spans[0].span_id}"
    branch = _branch(paper_ir, ReviewSource.AGENT)
    branch.novelty = NoveltyAssessment(
        judgment=NoveltyJudgment.NEGATIVE,
        supporting_points=[],
        limiting_points=[
            ReviewPoint(
                point_id="novelty-limit",
                aspect=ReviewAspect.NOVELTY_PRIOR_ART,
                text="The contribution appears incremental.",
                evidence_keys=[span_key],
            )
        ],
    )
    state = initialize_review_state_v3(
        paper_ir,
        PaperRubricBuilder().build(paper_ir),
        _graph(paper_ir.paper_id, 95.0, coverage=0.5),
        paper_request.evidence_date,
    )

    fused, _ = ReviewFusion().fuse(state, branch)

    point = next(iter(fused.canonical_points.values()))
    assert fused.process_features.graph_score_available is True
    assert point.graph_tension_score == 0.0
    assert point.graph_extra_counterfactual_actions == 0


def test_evidence_exhaustion_does_not_collapse_novelty_direction(
    paper_ir, paper_request
) -> None:
    span_key = f"P:{paper_ir.spans[0].span_id}"
    branch = _branch(paper_ir, ReviewSource.AGENT)
    branch.novelty = NoveltyAssessment(
        judgment=NoveltyJudgment.MIXED,
        supporting_points=[
            ReviewPoint(
                point_id="support",
                aspect=ReviewAspect.NOVELTY_PRIOR_ART,
                text="The method makes a bounded technical contribution.",
                evidence_keys=[span_key],
            )
        ],
        limiting_points=[
            ReviewPoint(
                point_id="limit",
                aspect=ReviewAspect.NOVELTY_PRIOR_ART,
                text="Its distinction from the nearest method needs checking.",
                evidence_keys=[span_key],
            )
        ],
    )
    state = initialize_review_state_v3(
        paper_ir,
        PaperRubricBuilder().build(paper_ir),
        _graph(paper_ir.paper_id, 80.0),
        paper_request.evidence_date,
    )
    fused, _ = ReviewFusion().fuse(state, branch)

    EvidenceSupervisor._exhaust(fused)
    review = ReviewCompiler().compile_v3(fused)

    assert review.novelty.judgment == NoveltyJudgment.MIXED
    assert (
        review.novelty.verification_status
        == NoveltyVerificationStatus.INSUFFICIENT_COVERAGE
    )
    assert review.novelty.confidence == 0.3
    assert len(review.novelty.supporting_points) == 1
    assert len(review.novelty.limiting_points) == 1
    assert all(
        "provisional" in point.text
        for point in [
            *review.novelty.supporting_points,
            *review.novelty.limiting_points,
        ]
    )

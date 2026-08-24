from __future__ import annotations

from gear.evidence_supervisor import EvidenceSupervisor
from gear.graph_prior_contracts import GraphResultV3
from gear.paper_extraction import PaperRubricBuilder
from gear.review_contracts import (
    BranchReview,
    NoveltyAssessment,
    NoveltyJudgment,
    PointSeverity,
    ReviewAspect,
    ReviewPoint,
    ReviewSource,
    ReviewSummary,
)
from gear.review_fusion import ReviewFusion
from gear.review_state import initialize_review_state_v3
from gear.trace import EvidenceStore
from tests.gear.fakes import EmptyPriorArt, UnusedRelationClassifier


def test_strong_first_claim_triggers_counterfactual_search(
    tmp_path, gear_config, paper_ir, paper_request
) -> None:
    span = paper_ir.spans[0]
    point = ReviewPoint(
        point_id="first-point",
        aspect=ReviewAspect.NOVELTY_PRIOR_ART,
        text="This is the first evidence-state reviewer.",
        severity=PointSeverity.MAJOR,
        evidence_keys=[f"P:{span.span_id}"],
        external_verification_required=True,
    )
    branch = BranchReview(
        paper_id=paper_ir.paper_id,
        source=ReviewSource.AGENT,
        model_id="agent",
        prompt_sha256="sha256:prompt",
        input_sha256="sha256:input",
        summary=ReviewSummary(
            text="The manuscript presents a reviewer.",
            evidence_keys=[f"P:{span.span_id}"],
        ),
        novelty=NoveltyAssessment(
            judgment=NoveltyJudgment.POSITIVE,
            supporting_points=[point],
            limiting_points=[],
        ),
    )
    state = initialize_review_state_v3(
        paper_ir,
        PaperRubricBuilder().build(paper_ir),
        None,
        paper_request.evidence_date,
    )
    state, _ = ReviewFusion().fuse(state, branch)
    store = EvidenceStore(tmp_path / "supervisor")
    for paper_span in paper_ir.spans:
        store.add_evidence(f"P:{paper_span.span_id}", "paper_span", paper_span)
    prior_art = EmptyPriorArt()
    state = EvidenceSupervisor(gear_config).resolve(
        state,
        paper_ir,
        store,
        prior_art=prior_art,
        relation_classifier=UnusedRelationClassifier(),
    )
    canonical = next(iter(state.canonical_points.values()))
    assert "normal" in prior_art.families
    assert "contrastive" in prior_art.families
    assert prior_art.claim_texts
    assert all(text == point.text for text in prior_art.claim_texts)
    assert point.text != paper_ir.claims[0].text
    assert canonical.counterfactual_search_done is True
    assert state.action_budget.actions_used <= state.action_budget.total_actions_max


def test_continuous_graph_tension_adds_counterfactual_searches(
    tmp_path, gear_config, paper_ir, paper_request
) -> None:
    span = paper_ir.spans[0]
    point = ReviewPoint(
        point_id="first-point",
        aspect=ReviewAspect.NOVELTY_PRIOR_ART,
        text="This is the first evidence-state reviewer.",
        severity=PointSeverity.MAJOR,
        evidence_keys=[f"P:{span.span_id}"],
        external_verification_required=True,
    )
    branch = BranchReview(
        paper_id=paper_ir.paper_id,
        source=ReviewSource.AGENT,
        model_id="agent",
        prompt_sha256="sha256:prompt",
        input_sha256="sha256:input",
        summary=ReviewSummary(
            text="The manuscript presents a reviewer.",
            evidence_keys=[f"P:{span.span_id}"],
        ),
        novelty=NoveltyAssessment(
            judgment=NoveltyJudgment.POSITIVE,
            supporting_points=[point],
            limiting_points=[],
        ),
    )
    graph = GraphResultV3(
        paper_id=paper_ir.paper_id,
        score_0_100=5.0,
        p_uptake=0.8,
        conditional_diffusion=0.6,
        feature_coverage=1.0,
    )
    state = initialize_review_state_v3(
        paper_ir,
        PaperRubricBuilder().build(paper_ir),
        graph,
        paper_request.evidence_date,
    )
    state, _ = ReviewFusion().fuse(state, branch)
    store = EvidenceStore(tmp_path / "supervisor")
    for paper_span in paper_ir.spans:
        store.add_evidence(f"P:{paper_span.span_id}", "paper_span", paper_span)
    prior_art = EmptyPriorArt()

    state = EvidenceSupervisor(gear_config).resolve(
        state,
        paper_ir,
        store,
        prior_art=prior_art,
        relation_classifier=UnusedRelationClassifier(),
    )

    canonical = next(iter(state.canonical_points.values()))
    assert canonical.graph_tension_score == 0.855
    assert canonical.graph_extra_counterfactual_actions == 0
    assert canonical.counterfactual_search_count == 1
    assert prior_art.families.count("contrastive") == 1

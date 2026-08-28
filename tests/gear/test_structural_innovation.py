from __future__ import annotations

from datetime import date

import pytest

from experiments.gear.evaluation.graph_ablation import (
    assert_score_shuffle_changes_signal,
)
from gear.contracts import RetrievalCoverageCard
from gear.graph_action_policy import GraphActionPolicy
from gear.graph_guidance import GraphGuidancePlanner
from gear.graph_prior_contracts import (
    ClaimGraphPrior,
    GraphActionDecision,
    GraphResourceCaps,
    GraphRuntimePacket,
    GraphSignalBundle,
    InfluenceForecast,
)
from gear.paper_extraction import PaperRubricBuilder
from gear.review_contracts import (
    CanonicalReviewPoint,
    PointSeverity,
    PointValidationStatus,
    ReviewAspect,
)
from gear.review_state import initialize_review_state
from gear.structural_innovation import (
    attribute_graph_signal,
    build_claim_inventory,
    fuse_structural_innovation,
)
from gear.trace import EvidenceStore


def _forecast(score: float, expected: float) -> InfluenceForecast:
    return InfluenceForecast(
        status="available",
        prospective_5y_diffusion_percentile=score,
        uptake_probability=0.5,
        conditional_diffusion=0.4,
        expected_diffusion=expected,
        field_year_base=0.1,
        feature_coverage=1.0,
        release_id="release",
        model_sha256="sha256:model",
        percentile_reference_sha256="sha256:reference",
    )


def _packet(paper_id: str, cutoff: date, score: float, expected: float):
    return GraphRuntimePacket(
        paper_id=paper_id,
        cutoff_date=cutoff,
        forecast=_forecast(score, expected),
    )


def test_planner_behavior_monotonicity_and_equal_caps(paper_ir) -> None:
    cutoff = date(2010, 1, 2)
    rubric = PaperRubricBuilder().build(paper_ir)
    low_packet = _packet(paper_ir.paper_id, cutoff, 0.0, 0.1)
    state = initialize_review_state(paper_ir, rubric, low_packet, cutoff)
    point = CanonicalReviewPoint(
        point_id="CP",
        section="novelty_support",
        aspect=ReviewAspect.NOVELTY_PRIOR_ART,
        severity=PointSeverity.MINOR,
        proposition="The manuscript claims a distinct evidence controller.",
        paper_evidence_keys=[f"P:{paper_ir.claims[0].span_id}"],
        agent_support=True,
    )
    state.canonical_points = {point.point_id: point}
    planner = GraphGuidancePlanner(resource_caps=GraphResourceCaps(provider_searches=8))
    low = planner.plan(
        state,
        enable_score_routing=True,
        enable_topology=False,
        calibration_variant="scalar_score",
    )
    high_state = state.model_copy(
        update={"graph_result": _packet(paper_ir.paper_id, cutoff, 100.0, 0.9)}
    )
    high = planner.plan(
        high_state,
        enable_score_routing=True,
        enable_topology=False,
        calibration_variant="scalar_score",
    )

    assert low.controller_state["local_slots"] > high.controller_state["local_slots"]
    assert low.controller_state["remote_slots"] < high.controller_state["remote_slots"]
    assert low.resource_caps == high.resource_caps


def test_promoted_action_policy_abstention_disables_graph_missions(paper_ir) -> None:
    cutoff = date(2010, 1, 2)
    rubric = PaperRubricBuilder().build(paper_ir)
    state = initialize_review_state(
        paper_ir, rubric, _packet(paper_ir.paper_id, cutoff, 90.0, 0.8), cutoff
    )
    point = CanonicalReviewPoint(
        point_id="CP-abstain",
        section="novelty_support",
        aspect=ReviewAspect.NOVELTY_PRIOR_ART,
        severity=PointSeverity.MINOR,
        proposition="The manuscript claims a distinct evidence controller.",
        paper_evidence_keys=[f"P:{paper_ir.claims[0].span_id}"],
        agent_support=True,
    )
    state.canonical_points = {point.point_id: point}
    state.graph_action_decision = GraphActionDecision(
        action="abstain",
        predicted_uplift=0.0,
        uplift_lcb=0.0,
        selected=False,
        reason="uplift_lcb_nonpositive",
    )

    plan = GraphGuidancePlanner().plan(state, enable_score_routing=True)

    assert plan.claim_guidance[0].allocated_local_query_slots == 0
    assert plan.claim_guidance[0].allocated_remote_query_slots == 0
    assert [mission.mission_type for mission in plan.claim_guidance[0].missions] == [
        "abstain"
    ]


def test_randomized_baseline_keeps_neutral_gear_missions(paper_ir) -> None:
    cutoff = date(2010, 1, 2)
    state = initialize_review_state(
        paper_ir,
        PaperRubricBuilder().build(paper_ir),
        _packet(paper_ir.paper_id, cutoff, 50.0, 0.5),
        cutoff,
    )
    point = CanonicalReviewPoint(
        point_id="CP-baseline",
        section="novelty_support",
        aspect=ReviewAspect.NOVELTY_PRIOR_ART,
        severity=PointSeverity.MINOR,
        proposition="The manuscript claims a distinct evidence controller.",
        paper_evidence_keys=[f"P:{paper_ir.claims[0].span_id}"],
        agent_support=True,
    )
    state.canonical_points = {point.point_id: point}
    state.graph_action_decision = GraphActionDecision(
        action="baseline",
        predicted_uplift=0.0,
        uplift_lcb=0.0,
        selected=False,
        reason="preassigned_randomized_action",
    )

    plan = GraphGuidancePlanner().plan(state, enable_score_routing=False)

    guidance = plan.claim_guidance[0]
    assert guidance.allocated_local_query_slots > 0
    assert guidance.allocated_remote_query_slots > 0
    assert all(mission.mission_type != "abstain" for mission in guidance.missions)


def test_randomized_topology_starts_with_retrievable_seed_query(paper_ir) -> None:
    cutoff = date(2010, 1, 2)
    state = initialize_review_state(
        paper_ir,
        PaperRubricBuilder().build(paper_ir),
        _packet(paper_ir.paper_id, cutoff, 50.0, 0.5),
        cutoff,
    )
    point = CanonicalReviewPoint(
        point_id="CP-topology",
        section="novelty_support",
        aspect=ReviewAspect.NOVELTY_PRIOR_ART,
        severity=PointSeverity.MINOR,
        proposition="The manuscript claims a distinct evidence controller.",
        paper_evidence_keys=[f"P:{paper_ir.claims[0].span_id}"],
        agent_support=True,
    )
    state.canonical_points = {point.point_id: point}
    state.graph_action_decision = GraphActionDecision(
        action="topology_expansion",
        predicted_uplift=0.0,
        uplift_lcb=0.0,
        selected=True,
        reason="preassigned_randomized_action",
    )

    plan = GraphGuidancePlanner().plan(state, enable_score_routing=False)

    mission = plan.claim_guidance[0].missions[0]
    assert mission.mission_type == "topology_expansion"
    assert mission.query_roles == ["author_terminology", "object_problem"]
    assert mission.traversal == "references"


def test_claim_attribution_conservation(paper_ir) -> None:
    inventory = build_claim_inventory(paper_ir)
    bundle = GraphSignalBundle(
        paper_id=paper_ir.paper_id,
        expected_diffusion=0.6,
        field_year_base=0.1,
        reliability=1.0,
        shrunk_diffusion=0.6,
    )
    packet = _packet(paper_ir.paper_id, date(2010, 1, 2), 70.0, 0.6)
    priors = attribute_graph_signal(inventory, bundle, packet)

    assert sum(prior.attribution_weight for prior in priors) == pytest.approx(1.0)
    assert sum(prior.diffusion_prior for prior in priors) == pytest.approx(0.6)


def test_structural_score_monotone_and_direct_antecedent_noncompensatory(
    paper_ir, tmp_path
) -> None:
    store = EvidenceStore(tmp_path / "structural")
    for span in paper_ir.spans:
        store.add_evidence(f"P:{span.span_id}", "paper_span", span)
    inventory = build_claim_inventory(paper_ir)[:1]
    claim = inventory[0]
    coverage_key = "CV:test"
    store.add_evidence(
        coverage_key,
        "retrieval_coverage",
        RetrievalCoverageCard(
            coverage_id="test",
            target_claim_id=claim.claim_id,
            cutoff_date=date(2010, 1, 2),
            required_query_roles=["object_problem"],
            completed_query_roles=["object_problem"],
            coverage_sufficient=True,
        ),
    )
    point = CanonicalReviewPoint(
        point_id="CP",
        section="novelty_support",
        aspect=ReviewAspect.NOVELTY_PRIOR_ART,
        severity=PointSeverity.MINOR,
        proposition=claim.text,
        paper_evidence_keys=claim.manuscript_evidence_keys,
        coverage_evidence_keys=[coverage_key],
        novelty_resolution="bounded_no_antecedent",
        validation_status=PointValidationStatus.VALIDATED,
        agent_support=True,
    )
    rubric = PaperRubricBuilder().build(paper_ir)
    packet = _packet(paper_ir.paper_id, date(2010, 1, 2), 50.0, 0.5)
    state = initialize_review_state(paper_ir, rubric, packet, date(2010, 1, 2))
    state.claim_inventory = inventory
    state.canonical_points = {point.point_id: point}
    state.claim_graph_priors = [
        ClaimGraphPrior(
            claim_id=claim.claim_id,
            attribution_weight=1.0,
            diffusion_prior=0.2,
            confidence=1.0,
        )
    ]

    def score(diffusion: float) -> float:
        candidate = state.model_copy(
            update={
                "graph_signal_bundle": GraphSignalBundle(
                    paper_id=paper_ir.paper_id,
                    expected_diffusion=diffusion,
                    field_year_base=0.0,
                    reliability=1.0,
                    shrunk_diffusion=diffusion,
                ),
                "claim_graph_priors": [
                    ClaimGraphPrior(
                        claim_id=claim.claim_id,
                        attribution_weight=1.0,
                        diffusion_prior=diffusion,
                        confidence=1.0,
                    )
                ],
            }
        )
        fused = fuse_structural_innovation(candidate, store)
        return fused.structural_innovation_cards[0].structural_innovation_score

    assert score(0.8) > score(0.2)

    relation_key = "R:direct"
    store.add_evidence(
        relation_key,
        "relation_card",
        {
            "relation_label": "DIRECT_ANTECEDENT",
            "temporal_valid": True,
            "independent_verification_passed": True,
            "essential_facet_coverage": 1.0,
        },
    )
    point.relation_evidence_keys = [relation_key]
    point.novelty_resolution = "antecedent_found"
    assert score(1.0) == 0.0


def test_policy_abstains_when_uplift_lcb_nonpositive() -> None:
    decision = GraphActionPolicy().decide(
        {"cross_field_pathway": 0.2},
        {"cross_field_pathway": 0.0},
        guardrails_pass=True,
    )
    assert decision.action == "abstain"
    assert decision.selected is False


def test_score_shuffle_really_changes_signal() -> None:
    assert_score_shuffle_changes_signal({"p1": 0.1, "p2": 0.9}, {"p1": 0.9, "p2": 0.1})
    with pytest.raises(ValueError, match="did not change"):
        assert_score_shuffle_changes_signal(
            {"p1": 0.1, "p2": 0.9}, {"p1": 0.1, "p2": 0.9}
        )

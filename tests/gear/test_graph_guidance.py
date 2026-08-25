from __future__ import annotations

import pytest
from pydantic import ValidationError

from gear.graph_guidance import GraphGuidancePlanner, score_controller
from gear.graph_prior import graph_runtime_packet
from gear.graph_prior_contracts import (
    FULLTEXT16_FEATURE_IDS,
    GraphRuntimePacketV1,
    GraphTopologySeedV1,
    ResourceLedgerV1,
)
from gear.paper_extraction import PaperRubricBuilder
from gear.review_contracts import CanonicalReviewPoint, PointSeverity, ReviewAspect
from gear.review_state import initialize_review_state_v3


def _packet(
    paper_id: str, score: float, *, missing: list[str] | None = None
) -> GraphRuntimePacketV1:
    missing_ids = list(dict.fromkeys([*(missing or []), "EF0197"]))
    return GraphRuntimePacketV1(
        paper_id=paper_id,
        score_0_100=score,
        raw_expected_diffusion=0.3,
        p_uptake=0.8,
        conditional_diffusion=0.4,
        feature_values={"EF0017": 1.0, "EF0197": None},
        historical_bands={
            "EF0017": "high_extreme",
            "EF0309": "high_extreme",
            "EF0052": "low_extreme",
        },
        missing_feature_ids=missing_ids,
        topology_seeds=[
            GraphTopologySeedV1(
                work_id="W1",
                title="evidence state controller for scientific review",
                publication_year=2019,
                shared_reference_count=3,
            )
        ],
    )


def _state(paper_ir, paper_request, packet: GraphRuntimePacketV1):
    state = initialize_review_state_v3(
        paper_ir,
        PaperRubricBuilder().build(paper_ir),
        packet,
        paper_request.evidence_date,
    )
    point = CanonicalReviewPoint(
        point_id="CP-1",
        section="novelty_support",
        initial_section="novelty_support",
        aspect=ReviewAspect.NOVELTY_PRIOR_ART,
        severity=PointSeverity.MINOR,
        proposition="The evidence state controller is a scientific review contribution.",
        novelty_claim_id="C-1",
        agent_support=True,
        retained=True,
    )
    state.canonical_points = {point.point_id: point}
    return state


def test_score_controller_is_monotonic_and_preserves_both_geometries() -> None:
    allocations = [
        score_controller(_packet("p", score)) for score in (0, 25, 50, 75, 100)
    ]
    assert [remote for _, remote, _ in allocations] == sorted(
        remote for _, remote, _ in allocations
    )
    assert all(local >= 1 and remote >= 1 for local, remote, _ in allocations)
    degraded = _packet(
        "p", 100, missing=sorted(FULLTEXT16_FEATURE_IDS - {"EF0017", "EF0197"})[:8]
    )
    assert score_controller(degraded)[1] < score_controller(_packet("p", 100))[1]


def test_planner_groups_profile_features_and_assigns_claim_seed(
    paper_ir, paper_request
) -> None:
    state = _state(paper_ir, paper_request, _packet(paper_ir.paper_id, 95))
    plan = GraphGuidancePlanner().plan(state)
    guidance = plan.claim_guidance[0]
    profile_types = [
        mission.mission_type
        for mission in guidance.missions
        if mission.origin == "profile"
    ]
    assert profile_types.count("reference_structure_diversity") == 1
    assert len(profile_types) == 1
    topology = [
        mission for mission in guidance.missions if mission.origin == "topology"
    ]
    assert topology[0].seed_work_ids == ["W1"]
    assert topology[0].orientation == "rescue"
    assert topology[0].traversal == "none"
    assert guidance.claim_relevance < 1.0
    assert plan.controller_state["remote_slots"] > plan.controller_state["local_slots"]
    assert plan.controller_state["executable_query_slots"] == 3
    assert (
        guidance.allocated_local_query_slots + guidance.allocated_remote_query_slots
        == 3
    )

    low_state = _state(paper_ir, paper_request, _packet(paper_ir.paper_id, 5))
    low_topology = [
        mission
        for mission in GraphGuidancePlanner().plan(low_state).claim_guidance[0].missions
        if mission.origin == "topology"
    ]
    assert low_topology[0].traversal == "none"


def test_planner_rejects_seed_with_only_generic_scientific_overlap(
    paper_ir, paper_request
) -> None:
    packet = _packet(paper_ir.paper_id, 95).model_copy(
        update={
            "topology_seeds": [
                GraphTopologySeedV1(
                    work_id="off-topic",
                    title="A new model and mechanism for scientific analysis",
                    publication_year=2019,
                    shared_reference_count=9,
                )
            ]
        }
    )
    state = _state(paper_ir, paper_request, packet)

    guidance = GraphGuidancePlanner().plan(state).claim_guidance[0]

    assert guidance.claim_relevance == 0.0
    assert all(mission.origin != "topology" for mission in guidance.missions)


def test_planner_guides_novelty_questions(paper_ir, paper_request) -> None:
    state = _state(paper_ir, paper_request, _packet(paper_ir.paper_id, 80))
    point = next(iter(state.canonical_points.values()))
    point.section = "questions"
    point.initial_section = "questions"
    point.requires_external_evidence = True

    plan = GraphGuidancePlanner().plan(state)

    assert [row.review_point_id for row in plan.claim_guidance] == [point.point_id]


def test_planner_apportions_only_executable_slots_across_claims(
    paper_ir, paper_request
) -> None:
    state = _state(paper_ir, paper_request, _packet(paper_ir.paper_id, 95))
    second = next(iter(state.canonical_points.values())).model_copy(
        update={"point_id": "CP-2", "novelty_claim_id": "C-2"}
    )
    state.canonical_points[second.point_id] = second

    plan = GraphGuidancePlanner().plan(state)

    assert plan.controller_state["executable_query_slots"] == 6
    assert (
        sum(
            row.allocated_local_query_slots + row.allocated_remote_query_slots
            for row in plan.claim_guidance
        )
        == 6
    )
    assert all(
        row.allocated_local_query_slots + row.allocated_remote_query_slots == 3
        for row in plan.claim_guidance
    )
    assert all(
        "legacy_contrastive" not in mission.query_roles
        for row in plan.claim_guidance
        for mission in row.missions
    )


def test_planner_caps_topology_to_two_high_priority_claims(
    paper_ir, paper_request
) -> None:
    packet = _packet(paper_ir.paper_id, 95).model_copy(
        update={
            "topology_seeds": [
                GraphTopologySeedV1(
                    work_id=f"W{index}",
                    title="evidence state controller for scientific review",
                    publication_year=2019,
                    shared_reference_count=10 - index,
                )
                for index in range(1, 5)
            ]
        }
    )
    state = _state(paper_ir, paper_request, packet)
    base = next(iter(state.canonical_points.values()))
    state.canonical_points = {
        f"CP-{index}": base.model_copy(
            update={"point_id": f"CP-{index}", "novelty_claim_id": f"C-{index}"}
        )
        for index in range(1, 5)
    }

    plan = GraphGuidancePlanner().plan(state)

    guided = [
        row.review_point_id
        for row in plan.claim_guidance
        if any(mission.origin == "topology" for mission in row.missions)
    ]
    assert guided == ["CP-1", "CP-2"]


def test_prior_art_limit_gets_topology_without_starving_normal_query(
    paper_ir, paper_request
) -> None:
    state = _state(paper_ir, paper_request, _packet(paper_ir.paper_id, 95))
    point = next(iter(state.canonical_points.values()))
    point.section = "novelty_limit"
    point.initial_section = "novelty_limit"
    point.proposition = (
        "Earlier work may already contain the evidence state controller."
    )

    guidance = GraphGuidancePlanner().plan(state).claim_guidance[0]

    topology = [
        mission for mission in guidance.missions if mission.origin == "topology"
    ]
    assert len(topology) == 1
    assert topology[0].traversal == "none"
    assert all(
        "legacy_contrastive" not in mission.query_roles for mission in guidance.missions
    )
    assert any(
        mission.origin in {"score", "profile"}
        and set(mission.query_roles)
        & {
            "author_terminology",
            "object_problem",
            "mechanism_outcome",
            "purpose_semantic",
        }
        for mission in guidance.missions
    )


def test_prior_art_limit_recognizes_plural_antecedent_language(
    paper_ir, paper_request
) -> None:
    state = _state(paper_ir, paper_request, _packet(paper_ir.paper_id, 70))
    point = next(iter(state.canonical_points.values()))
    point.section = "novelty_limit"
    point.initial_section = "novelty_limit"
    point.proposition = (
        "The manuscript does not delimit the closest antecedents from the "
        "claimed evidence state controller for scientific review."
    )

    guidance = GraphGuidancePlanner().plan(state).claim_guidance[0]

    assert any(mission.origin == "topology" for mission in guidance.missions)


def test_scope_limit_does_not_receive_prior_art_topology(
    paper_ir, paper_request
) -> None:
    state = _state(paper_ir, paper_request, _packet(paper_ir.paper_id, 95))
    point = next(iter(state.canonical_points.values()))
    point.section = "novelty_limit"
    point.initial_section = "novelty_limit"
    point.proposition = (
        "The contribution is currently demonstrated only on one bounded dataset."
    )

    guidance = GraphGuidancePlanner().plan(state).claim_guidance[0]

    assert all(mission.origin != "topology" for mission in guidance.missions)


def test_prior_art_limit_receives_more_slots_than_generic_question(
    paper_ir, paper_request
) -> None:
    state = _state(paper_ir, paper_request, _packet(paper_ir.paper_id, 80))
    base = next(iter(state.canonical_points.values()))
    prior_limit = base.model_copy(
        update={
            "point_id": "CP-prior",
            "novelty_claim_id": "C-prior",
            "section": "novelty_limit",
            "initial_section": "novelty_limit",
            "proposition": "Prior work may already implement the same controller.",
        }
    )
    generic_question = base.model_copy(
        update={
            "point_id": "CP-question",
            "novelty_claim_id": "C-question",
            "section": "questions",
            "initial_section": "questions",
            "requires_external_evidence": True,
        }
    )
    state.canonical_points = {
        prior_limit.point_id: prior_limit,
        base.point_id: base,
        generic_question.point_id: generic_question,
        "CP-question-2": generic_question.model_copy(
            update={"point_id": "CP-question-2", "novelty_claim_id": "C-question-2"}
        ),
    }

    plan = GraphGuidancePlanner().plan(state)
    slots = {
        row.review_point_id: (
            row.allocated_local_query_slots + row.allocated_remote_query_slots
        )
        for row in plan.claim_guidance
    }

    assert slots[prior_limit.point_id] == 3
    assert slots[generic_question.point_id] <= 2
    assert sum(slots.values()) == 8


def test_v4_migration_drops_search_terms_from_execution() -> None:
    packet = graph_runtime_packet(
        {
            "contract": "aspr_graph_result_v4",
            "paper_id": "p",
            "score_0_100": 80,
            "p_uptake": 0.8,
            "conditional_diffusion": 0.4,
            "feature_coverage": 0.5,
            "seed_work_ids": ["W1"],
            "search_terms": ["10.1234 noisy doi"],
        }
    )
    assert [seed.work_id for seed in packet.topology_seeds] == ["W1"]
    assert "search_terms" not in packet.model_dump()


def test_runtime_contracts_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ResourceLedgerV1(paper_id="p", unknown_counter=1)

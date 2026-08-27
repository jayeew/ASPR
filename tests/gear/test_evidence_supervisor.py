from __future__ import annotations

from types import SimpleNamespace

from gear.contracts import RetrievedWork
from gear.evidence_supervisor import EvidenceSupervisor, build_retrieval_claim
from gear.graph_prior_contracts import (
    ClaimGuidance,
    GraphRuntimePacket,
    InfluenceForecast,
    RetrievalMission,
)
from gear.paper_extraction import PaperRubricBuilder
from gear.review_contracts import (
    BranchReview,
    CanonicalReviewPoint,
    NoveltyAssessment,
    NoveltyJudgment,
    PointSeverity,
    ReviewAspect,
    ReviewPoint,
    ReviewSource,
    ReviewSummary,
)
from gear.review_fusion import ReviewFusion
from gear.review_state import initialize_review_state
from gear.trace import EvidenceStore
from tests.gear.fakes import EmptyPriorArt, UnusedRelationClassifier


def test_low_coverage_graph_seed_uses_claim_fallback(tmp_path) -> None:
    point = CanonicalReviewPoint(
        point_id="CP-graph",
        section="novelty_support",
        initial_section="novelty_support",
        aspect=ReviewAspect.NOVELTY_PRIOR_ART,
        severity=PointSeverity.MINOR,
        proposition="A bounded scientific contribution.",
        agent_support=True,
        retained=True,
        relation_evidence_keys=["R:low"],
    )
    store = EvidenceStore(tmp_path / "topology-value")
    store.add_evidence(
        "Q:graph",
        "retrieval_query",
        {"query_role": "graph_seed"},
    )
    store.add_evidence(
        "R:low",
        "relation_card",
        {
            "temporal_valid": True,
            "relation_label": "PARALLEL",
            "prior_work_id": "W-low",
            "essential_facet_coverage": 0.2,
            "source_query_ids": ["graph"],
            "common_dimensions": ["generic method"],
            "difference_dimensions": ["different scientific target"],
        },
    )

    assert EvidenceSupervisor._topology_seed_has_value(point, store) is False

    store.add_evidence(
        "R:high",
        "relation_card",
        {
            "temporal_valid": True,
            "relation_label": "PARALLEL",
            "prior_work_id": "W-high",
            "essential_facet_coverage": 0.5,
            "source_query_ids": ["graph"],
            "common_dimensions": ["shared mechanism"],
            "difference_dimensions": ["different implementation"],
        },
    )
    point.relation_evidence_keys = ["R:high"]
    assert EvidenceSupervisor._topology_seed_has_value(point, store) is True


def test_graph_seed_candidate_is_recognized_for_classification_priority(
    tmp_path,
) -> None:
    store = EvidenceStore(tmp_path / "graph-candidate-priority")
    store.add_evidence(
        "Q:graph",
        "retrieval_query",
        {"query_role": "graph_seed"},
    )
    work = RetrievedWork(
        work_id="W-graph",
        target_claim_id="C-1",
        title="Relevant graph anchor",
        retrieval_query_id="graph",
        source_query_ids=["graph"],
        retrieval_source="test",
    )

    assert EvidenceSupervisor._work_has_query_role(work, store, "graph_seed")
    assert not EvidenceSupervisor._work_has_query_role(work, store, "citation_neighbor")


def test_query_roles_follow_the_equal_budget_guidance_missions() -> None:
    guidance = ClaimGuidance(
        review_point_id="CP-order",
        claim_id="C-order",
        claim_relevance=1.0,
        allocated_local_query_slots=1,
        allocated_remote_query_slots=2,
        missions=[
            RetrievalMission(
                mission_id="GM-order",
                mission_type="remote_mechanism_analogue",
                origin="score",
                target_claim_id="C-order",
                orientation="neutral",
                query_roles=[
                    "author_citation",
                    "graph_seed",
                    "purpose_semantic",
                    "mechanism_outcome",
                ],
                stop_rule="test",
            )
        ],
    )
    topology_roles = EvidenceSupervisor._allowed_query_roles(guidance)
    profile_roles = EvidenceSupervisor._allowed_query_roles(guidance)

    assert (
        topology_roles
        == profile_roles
        == [
            "author_citation",
            "graph_seed",
            "purpose_semantic",
            "mechanism_outcome",
        ]
    )


def test_semantically_admitted_author_citation_is_verified_first(
    tmp_path, gear_config
) -> None:
    point = CanonicalReviewPoint(
        point_id="CP-citation-rank1",
        section="novelty_limit",
        initial_section="novelty_limit",
        aspect=ReviewAspect.NOVELTY_PRIOR_ART,
        severity=PointSeverity.MINOR,
        proposition="The prior-work boundary requires verification.",
        agent_support=True,
        retained=True,
    )
    guidance = ClaimGuidance(
        review_point_id=point.point_id,
        claim_id="C-citation-rank1",
        claim_relevance=1.0,
        allocated_local_query_slots=1,
        allocated_remote_query_slots=1,
        missions=[
            RetrievalMission(
                mission_id="GM-citation-rank1",
                mission_type="topology_seed",
                origin="topology",
                target_claim_id="C-citation-rank1",
                orientation="neutral",
                query_roles=["author_citation"],
                stop_rule="test",
            )
        ],
    )
    store = EvidenceStore(tmp_path / "citation-rank1")
    for query_id, role in (
        ("q-local", "object_problem"),
        ("q-citation", "author_citation"),
        ("q-remote", "mechanism_outcome"),
    ):
        store.add_evidence(f"Q:{query_id}", "retrieval_query", {"query_role": role})
    works = [
        RetrievedWork(
            work_id=f"W-{name}",
            target_claim_id="C-citation-rank1",
            title=name,
            retrieval_query_id=query_id,
            source_query_ids=[query_id],
            retrieval_source="test",
        )
        for name, query_id in (
            ("local", "q-local"),
            ("citation", "q-citation"),
            ("remote", "q-remote"),
        )
    ]
    state = SimpleNamespace(
        paper_id="paper",
        graph_guidance_plan=SimpleNamespace(claim_guidance=[guidance]),
        graph_result=None,
        resource_ledger=None,
        branch_reviews={},
        retrieval_routing_plans={},
    )

    routed = EvidenceSupervisor(gear_config)._route_works(point, state, works, store)

    assert [work.work_id for work in routed][0] == "W-citation"
    plan = state.retrieval_routing_plans[point.point_id]
    citation = next(row for row in plan.candidates if row.candidate_id == "W-citation")
    assert citation.final_rank == 0
    assert citation.reason == "claim_linked_citation_rank1_after_semantic_gate"


def test_retrieval_claim_avoids_reference_target_span(paper_ir) -> None:
    reference_span_ids = {reference.source_span_id for reference in paper_ir.references}
    assert reference_span_ids
    scientific_claim = next(
        claim for claim in paper_ir.claims if claim.span_id not in reference_span_ids
    )
    reference_span_id = next(iter(reference_span_ids))
    point = CanonicalReviewPoint(
        point_id="CP-reanchor",
        section="novelty_limit",
        initial_section="novelty_limit",
        aspect=ReviewAspect.NOVELTY_PRIOR_ART,
        severity=PointSeverity.MINOR,
        proposition="The closest antecedents need to be delimited.",
        agent_support=True,
        retained=True,
        paper_evidence_keys=[f"P:{reference_span_id}"],
    )

    claim, span = build_retrieval_claim(point, paper_ir)

    assert span.span_id not in reference_span_ids
    assert claim.span_id == span.span_id
    assert f"P:{span.span_id}" in point.paper_evidence_keys
    assert "retrieval_target_span_reanchored" in point.validation_notes
    assert scientific_claim.span_id in paper_ir.span_map()


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
    state = initialize_review_state(
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
    graph = GraphRuntimePacket(
        paper_id=paper_ir.paper_id,
        cutoff_date=paper_request.evidence_date,
        forecast=InfluenceForecast(
            status="available",
            prospective_5y_diffusion_percentile=5.0,
            uptake_probability=0.8,
            conditional_diffusion=0.6,
            expected_diffusion=0.48,
            feature_coverage=1.0,
            release_id="test",
            model_sha256="sha256:test",
            percentile_reference_sha256="sha256:test",
        ),
    )
    state = initialize_review_state(
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
    assert canonical.graph_tension_score == 0.0
    assert canonical.graph_extra_counterfactual_actions == 0
    assert canonical.counterfactual_search_count == 1
    assert prior_art.families.count("contrastive") == 1

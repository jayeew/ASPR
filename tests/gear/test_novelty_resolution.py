from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from gear.contracts import QuerySpec, RetrievalCoverageCard
from gear.evidence_supervisor import EvidenceSupervisor
from gear.review_contracts import (
    CanonicalReviewPoint,
    NoveltyJudgment,
    PointSeverity,
    PointValidationStatus,
    ReviewAspect,
    ReviewPoint,
)
from gear.trace import EvidenceStore


def _point() -> CanonicalReviewPoint:
    return CanonicalReviewPoint(
        point_id="novelty-point",
        section="novelty_support",
        aspect=ReviewAspect.NOVELTY_PRIOR_ART,
        severity=PointSeverity.MAJOR,
        proposition="This is the first bounded evidence controller.",
        paper_evidence_keys=["P:S-one"],
        agent_support=True,
        requires_external_evidence=True,
        validation_status=PointValidationStatus.UNRESOLVED,
        stability_status="pending",
        normal_search_done=True,
        counterfactual_search_done=True,
    )


def _coverage(*, sufficient: bool, failed: bool = False) -> RetrievalCoverageCard:
    return RetrievalCoverageCard(
        coverage_id="COV-test",
        target_claim_id="C-novelty-point",
        cutoff_date=date(2020, 1, 1),
        required_query_roles=[
            "author_terminology",
            "object_problem",
            "mechanism_outcome",
            "purpose_semantic",
            "legacy_contrastive",
        ],
        completed_query_roles=[
            "author_terminology",
            "object_problem",
            "mechanism_outcome",
            "purpose_semantic",
            "legacy_contrastive",
        ],
        unique_eligible_count=20,
        compared_work_ids=[f"W-{index}" for index in range(10)],
        whole_paper_ranking_completed=True,
        purpose_ranking_completed=True,
        ranker="test",
        service_failed=failed,
        coverage_sufficient=sufficient,
    )


def test_sufficient_search_without_antecedent_is_bounded_not_rejected(tmp_path):
    store = EvidenceStore(tmp_path)
    point = _point()
    card = _coverage(sufficient=True)
    key = f"COV:{card.coverage_id}"
    store.add_evidence(key, "retrieval_coverage", card)
    point.coverage_evidence_keys = [key]

    class State:
        cutoff_date = date(2020, 1, 1)

    EvidenceSupervisor._stability(point, State(), store)

    assert point.validation_status == PointValidationStatus.VALIDATED
    assert point.stability_status == "stable"
    assert point.novelty_resolution == "bounded_no_antecedent"
    assert "not proof of global priority" in (point.resolved_proposition or "")
    assert "first" not in (point.resolved_proposition or "").casefold()


def test_partial_search_softens_without_erasing_direction(tmp_path):
    store = EvidenceStore(tmp_path)
    point = _point()
    card = _coverage(sufficient=False)
    key = f"COV:{card.coverage_id}"
    store.add_evidence(key, "retrieval_coverage", card)
    point.coverage_evidence_keys = [key]

    class State:
        cutoff_date = date(2020, 1, 1)

    EvidenceSupervisor._stability(point, State(), store)

    assert point.section == "novelty_support"
    assert point.validation_status == PointValidationStatus.VALIDATED
    assert point.stability_status == "not_required"
    assert point.novelty_resolution == "inconclusive"
    assert point.novelty_confidence == 0.35


def test_failed_search_is_downgraded_to_a_manuscript_grounded_question(tmp_path):
    store = EvidenceStore(tmp_path)
    point = _point()
    card = _coverage(sufficient=False, failed=True)
    key = f"COV:{card.coverage_id}"
    store.add_evidence(key, "retrieval_coverage", card)
    point.coverage_evidence_keys = [key]

    class State:
        cutoff_date = date(2020, 1, 1)

    EvidenceSupervisor._stability(point, State(), store)

    assert point.validation_status == PointValidationStatus.VALIDATED
    assert point.section == "novelty_support"
    assert point.stability_status == "not_required"
    assert point.novelty_resolution == "inconclusive"
    assert point.novelty_confidence == 0.3


def test_review_point_accepts_and_deduplicates_coverage_keys():
    point = ReviewPoint(
        point_id="coverage-point",
        aspect=ReviewAspect.NOVELTY_PRIOR_ART,
        text="The bounded search result is reported with its audited scope.",
        evidence_keys=["P:S-one", "COV:COV-one", "COV:COV-one"],
    )
    assert point.evidence_keys == ["P:S-one", "COV:COV-one"]


def test_evolving_coverage_is_stored_as_append_only_snapshots(tmp_path, paper_ir):
    store = EvidenceStore(tmp_path)
    point = _point()
    cards = [_coverage(sufficient=False), _coverage(sufficient=True)]

    class PriorArt:
        def coverage_card(self, *args, **kwargs):
            return cards.pop(0)

    state = SimpleNamespace(cutoff_date=date(2020, 1, 1))
    claim = paper_ir.claims[0]

    EvidenceSupervisor._store_coverage(point, state, claim, PriorArt(), store)
    first_key = point.coverage_evidence_keys[0]
    EvidenceSupervisor._store_coverage(point, state, claim, PriorArt(), store)
    second_key = point.coverage_evidence_keys[0]

    assert first_key == "COV:COV-test"
    assert second_key.startswith("COV:COV-test:")
    assert store.has(first_key)
    assert store.has(second_key)


def test_single_antecedent_is_downgraded_instead_of_blocking(tmp_path):
    store = EvidenceStore(tmp_path)
    point = _point()
    point.novelty_resolution = "antecedent_found"
    card = _coverage(sufficient=True)
    coverage_key = f"COV:{card.coverage_id}"
    store.add_evidence(coverage_key, "retrieval_coverage", card)
    point.coverage_evidence_keys = [coverage_key]
    query = QuerySpec(
        query_id="Q-one",
        claim_id="C-novelty-point",
        family="lexical",
        query_role="object_problem",
        query="bounded evidence",
    )
    store.add_evidence("Q:Q-one", "retrieval_query", query)
    store.add_evidence(
        "R:R-one",
        "prior_relation",
        {
            "prior_work_id": "W-one",
            "relation_label": "DIRECT_ANTECEDENT",
            "temporal_valid": True,
            "source_query_ids": ["Q-one"],
        },
    )
    point.relation_evidence_keys = ["R:R-one"]

    class State:
        cutoff_date = date(2020, 1, 1)

    EvidenceSupervisor._stability(point, State(), store)

    assert point.section == "novelty_support"
    assert point.validation_status == PointValidationStatus.VALIDATED
    assert point.stability_status == "not_required"
    assert "single_antecedent_downgraded_to_question" in point.validation_notes
    assert point.novelty_confidence == 0.45


def test_unverified_direct_relation_does_not_erase_novelty_support(tmp_path, paper_ir):
    store = EvidenceStore(tmp_path)
    point = _point()
    store.add_evidence(
        "R:R-existing",
        "prior_relation",
        {
            "prior_work_id": "W-existing",
            "relation_label": "DIRECT_ANTECEDENT",
            "temporal_valid": True,
            "source_query_ids": ["Q-existing"],
        },
    )
    point.relation_evidence_keys = ["R:R-existing"]

    class State:
        def __init__(self) -> None:
            self.relation_evidence_keys = ["R:R-existing"]

    EvidenceSupervisor()._classify(
        point,
        State(),
        paper_ir.spans[0],
        paper_ir.claims[0],
        [],
        store,
        None,
    )

    assert point.section == "novelty_support"
    assert point.novelty_resolution == "inconclusive"
    assert point.validation_status == PointValidationStatus.VALIDATED
    assert point.novelty_confidence == 0.45


def test_complete_verified_direct_antecedent_reclassifies_support(tmp_path, paper_ir):
    store = EvidenceStore(tmp_path)
    point = _point()
    store.add_evidence(
        "R:R-complete",
        "prior_relation",
        {
            "prior_work_id": "W-complete",
            "relation_label": "DIRECT_ANTECEDENT",
            "temporal_valid": True,
            "essential_facet_coverage": 1.0,
            "independent_verification_passed": True,
        },
    )
    point.relation_evidence_keys = ["R:R-complete"]

    class State:
        def __init__(self) -> None:
            self.relation_evidence_keys = ["R:R-complete"]

    EvidenceSupervisor()._classify(
        point, State(), paper_ir.spans[0], paper_ir.claims[0], [], store, None
    )

    assert point.section == "novelty_limit"
    assert point.novelty_resolution == "antecedent_found"


def test_partial_antecedent_retains_residual_delta_support(tmp_path, paper_ir):
    store = EvidenceStore(tmp_path)
    point = _point()
    store.add_evidence(
        "R:R-partial",
        "prior_relation",
        {
            "prior_work_id": "W-partial",
            "relation_label": "PARTIAL_ANTECEDENT",
            "temporal_valid": True,
        },
    )
    point.relation_evidence_keys = ["R:R-partial"]

    class State:
        def __init__(self) -> None:
            self.relation_evidence_keys = ["R:R-partial"]

    EvidenceSupervisor()._classify(
        point, State(), paper_ir.spans[0], paper_ir.claims[0], [], store, None
    )

    assert point.section == "novelty_support"
    assert point.novelty_resolution == "incremental_or_parallel"
    assert "residual delta" in (point.resolved_proposition or "")


def test_verified_residual_relation_can_resolve_directional_uncertainty(
    tmp_path, paper_ir
):
    store = EvidenceStore(tmp_path)
    point = _point()
    point.section = "questions"
    point.initial_section = "questions"
    store.add_evidence(
        "R:R-parallel",
        "prior_relation",
        {
            "prior_work_id": "W-parallel",
            "relation_label": "PARALLEL",
            "temporal_valid": True,
            "difference_dimensions": ["mechanism"],
        },
    )
    point.relation_evidence_keys = ["R:R-parallel"]
    state = SimpleNamespace(
        relation_evidence_keys=["R:R-parallel"],
        correction_event_evidence_keys=[],
        canonical_points={point.point_id: point},
        novelty_direction=NoveltyJudgment.UNCERTAIN,
        graph_guidance_plan=None,
    )

    EvidenceSupervisor()._classify(
        point, state, paper_ir.spans[0], paper_ir.claims[0], [], store, None
    )

    assert point.section == "novelty_support"
    assert state.novelty_direction == NoveltyJudgment.POSITIVE
    event = store.get(state.correction_event_evidence_keys[0])
    assert event is not None
    assert event.payload["before_direction"] == "uncertain"
    assert event.payload["after_direction"] == "positive"


def test_independent_residual_consensus_can_resolve_novelty_limit(tmp_path, paper_ir):
    store = EvidenceStore(tmp_path)
    point = _point()
    point.section = "novelty_limit"
    point.initial_section = "novelty_limit"
    relation_keys = []
    for index in (1, 2):
        key = f"R:R-parallel-{index}"
        store.add_evidence(
            key,
            "prior_relation",
            {
                "prior_work_id": f"W-parallel-{index}",
                "relation_label": "PARALLEL",
                "temporal_valid": True,
                "common_dimensions": ["shared bounded controller task"],
                "difference_dimensions": [f"target delta {index}"],
            },
        )
        relation_keys.append(key)
    point.relation_evidence_keys = relation_keys
    state = SimpleNamespace(
        relation_evidence_keys=relation_keys,
        correction_event_evidence_keys=[],
        canonical_points={point.point_id: point},
        novelty_direction=NoveltyJudgment.MIXED,
        graph_guidance_plan=None,
    )

    EvidenceSupervisor()._classify(
        point, state, paper_ir.spans[0], paper_ir.claims[0], [], store, None
    )

    assert point.section == "novelty_support"
    assert state.novelty_direction == NoveltyJudgment.POSITIVE
    event = store.get(state.correction_event_evidence_keys[0])
    assert event is not None
    assert event.payload["trigger_relation_ids"] == ["R-parallel-1", "R-parallel-2"]


def test_limiting_relation_consensus_can_bound_positive_direction(tmp_path, paper_ir):
    store = EvidenceStore(tmp_path)
    point = _point()
    point.section = "novelty_limit"
    point.initial_section = "novelty_limit"
    relation_keys = []
    for index in (1, 2):
        key = f"R:R-partial-limit-{index}"
        store.add_evidence(
            key,
            "prior_relation",
            {
                "prior_work_id": f"W-partial-limit-{index}",
                "relation_label": "PARTIAL_ANTECEDENT",
                "temporal_valid": True,
                "common_dimensions": ["shared bounded controller task"],
                "difference_dimensions": [f"bounded residual {index}"],
            },
        )
        relation_keys.append(key)
    point.relation_evidence_keys = relation_keys
    state = SimpleNamespace(
        relation_evidence_keys=relation_keys,
        correction_event_evidence_keys=[],
        canonical_points={point.point_id: point},
        novelty_direction=NoveltyJudgment.POSITIVE,
        graph_guidance_plan=None,
    )

    EvidenceSupervisor()._classify(
        point, state, paper_ir.spans[0], paper_ir.claims[0], [], store, None
    )

    assert point.section == "novelty_limit"
    assert state.novelty_direction == NoveltyJudgment.MIXED
    event = store.get(state.correction_event_evidence_keys[0])
    assert event is not None
    assert event.payload["before_direction"] == "positive"
    assert event.payload["after_direction"] == "mixed"


def test_single_partial_relation_does_not_change_positive_direction(tmp_path, paper_ir):
    store = EvidenceStore(tmp_path)
    point = _point()
    point.section = "novelty_limit"
    point.initial_section = "novelty_limit"
    store.add_evidence(
        "R:R-single-partial",
        "prior_relation",
        {
            "prior_work_id": "W-single-partial",
            "relation_label": "PARTIAL_ANTECEDENT",
            "temporal_valid": True,
            "common_dimensions": ["shared task"],
            "difference_dimensions": ["bounded residual"],
        },
    )
    point.relation_evidence_keys = ["R:R-single-partial"]
    state = SimpleNamespace(
        relation_evidence_keys=["R:R-single-partial"],
        correction_event_evidence_keys=[],
        canonical_points={point.point_id: point},
        novelty_direction=NoveltyJudgment.POSITIVE,
        graph_guidance_plan=None,
    )

    EvidenceSupervisor()._classify(
        point, state, paper_ir.spans[0], paper_ir.claims[0], [], store, None
    )

    assert point.novelty_resolution == "inconclusive"
    assert state.novelty_direction == NoveltyJudgment.POSITIVE


def test_absolute_priority_claim_can_be_bounded_by_one_covered_counterexample(
    tmp_path, paper_ir
) -> None:
    store = EvidenceStore(tmp_path)
    point = _point()
    point.section = "novelty_limit"
    point.initial_section = "novelty_limit"
    point.proposition = "The manuscript claims the first comprehensive analysis."
    store.add_evidence(
        "R:R-absolute",
        "prior_relation",
        {
            "prior_work_id": "W-absolute",
            "relation_label": "PARTIAL_ANTECEDENT",
            "temporal_valid": True,
            "common_dimensions": ["same population and genome-wide analysis"],
            "difference_dimensions": ["target adds a prediction workflow"],
            "essential_facet_coverage": 0.5,
        },
    )
    point.relation_evidence_keys = ["R:R-absolute"]
    state = SimpleNamespace(
        relation_evidence_keys=["R:R-absolute"],
        correction_event_evidence_keys=[],
        canonical_points={point.point_id: point},
        novelty_direction=NoveltyJudgment.POSITIVE,
        graph_guidance_plan=None,
    )

    EvidenceSupervisor()._classify(
        point, state, paper_ir.spans[0], paper_ir.claims[0], [], store, None
    )

    assert point.novelty_resolution == "antecedent_found"
    assert state.novelty_direction == NoveltyJudgment.MIXED


def test_redundant_partial_relations_are_not_independent_consensus(
    tmp_path, paper_ir
) -> None:
    store = EvidenceStore(tmp_path)
    point = _point()
    point.section = "novelty_limit"
    point.initial_section = "novelty_limit"
    point.proposition = "Earlier work discusses a GroEL background mechanism."
    relation_keys = []
    for index, common in enumerate(
        (
            "GroEL binds a client through a folding interaction",
            "GroEL interaction involves an associated client",
        ),
        1,
    ):
        key = f"R:R-redundant-{index}"
        store.add_evidence(
            key,
            "prior_relation",
            {
                "prior_work_id": f"W-redundant-{index}",
                "relation_label": "PARTIAL_ANTECEDENT",
                "temporal_valid": True,
                "common_dimensions": [common],
                "difference_dimensions": [f"distinct residual {index}"],
                "essential_facet_coverage": 0.4,
            },
        )
        relation_keys.append(key)
    point.relation_evidence_keys = relation_keys
    state = SimpleNamespace(
        relation_evidence_keys=relation_keys,
        correction_event_evidence_keys=[],
        canonical_points={point.point_id: point},
        novelty_direction=NoveltyJudgment.POSITIVE,
        graph_guidance_plan=None,
    )

    EvidenceSupervisor()._classify(
        point, state, paper_ir.spans[0], paper_ir.claims[0], [], store, None
    )

    assert point.novelty_resolution == "inconclusive"
    assert state.novelty_direction == NoveltyJudgment.POSITIVE


def test_scope_limitation_cannot_change_direction_via_prior_art_consensus(
    tmp_path, paper_ir
) -> None:
    store = EvidenceStore(tmp_path)
    point = _point()
    point.section = "novelty_limit"
    point.initial_section = "novelty_limit"
    point.proposition = "The experiments cover only one bounded dataset."
    relation_keys = []
    for index in (1, 2):
        key = f"R:R-scope-{index}"
        store.add_evidence(
            key,
            "prior_relation",
            {
                "prior_work_id": f"W-scope-{index}",
                "relation_label": "PARTIAL_ANTECEDENT",
                "temporal_valid": True,
                "common_dimensions": ["bounded dataset evaluation"],
                "difference_dimensions": [f"scope delta {index}"],
            },
        )
        relation_keys.append(key)
    point.relation_evidence_keys = relation_keys
    state = SimpleNamespace(
        relation_evidence_keys=relation_keys,
        correction_event_evidence_keys=[],
        canonical_points={point.point_id: point},
        novelty_direction=NoveltyJudgment.POSITIVE,
        graph_guidance_plan=None,
    )

    EvidenceSupervisor()._classify(
        point, state, paper_ir.spans[0], paper_ir.claims[0], [], store, None
    )

    assert point.novelty_resolution == "inconclusive"
    assert state.novelty_direction == NoveltyJudgment.POSITIVE


def test_aligned_cross_point_antecedents_can_bound_positive_direction(tmp_path) -> None:
    store = EvidenceStore(tmp_path)
    limit = _point()
    limit.section = "novelty_limit"
    limit.initial_section = "novelty_limit"
    limit.proposition = "Prior work may already contain the bounded controller."
    other = _point().model_copy(
        update={
            "point_id": "second-point",
            "proposition": "The controller applies evidence state transitions.",
        }
    )
    rows = [
        (limit, "R:R-cross-1", "W-cross-1"),
        (other, "R:R-cross-2", "W-cross-2"),
    ]
    for point, key, work_id in rows:
        store.add_evidence(
            key,
            "prior_relation",
            {
                "prior_work_id": work_id,
                "relation_label": "PARTIAL_ANTECEDENT",
                "temporal_valid": True,
                "common_dimensions": ["bounded evidence controller state transition"],
                "difference_dimensions": [f"residual delta for {work_id}"],
            },
        )
        point.relation_evidence_keys = [key]
    state = SimpleNamespace(
        correction_event_evidence_keys=[],
        canonical_points={limit.point_id: limit, other.point_id: other},
        novelty_direction=NoveltyJudgment.POSITIVE,
        graph_guidance_plan=None,
    )

    EvidenceSupervisor._apply_cross_point_limiting_consensus(state, store)

    assert state.novelty_direction == NoveltyJudgment.MIXED
    assert len(state.correction_event_evidence_keys) == 1
    event = store.get(state.correction_event_evidence_keys[0])
    assert event is not None
    assert set(event.payload["trigger_relation_ids"]) == {"R-cross-1", "R-cross-2"}


def test_unaligned_cross_point_relations_do_not_change_direction(tmp_path) -> None:
    store = EvidenceStore(tmp_path)
    limit = _point()
    limit.section = "novelty_limit"
    limit.initial_section = "novelty_limit"
    limit.proposition = "Prior work may already contain the bounded controller."
    other = _point().model_copy(update={"point_id": "second-point"})
    payloads = [
        (limit, "R:R-cross-a", "W-cross-a", "protein folding sequence"),
        (other, "R:R-cross-b", "W-cross-b", "climate rainfall forecast"),
    ]
    for point, key, work_id, common in payloads:
        store.add_evidence(
            key,
            "prior_relation",
            {
                "prior_work_id": work_id,
                "relation_label": "PARTIAL_ANTECEDENT",
                "temporal_valid": True,
                "common_dimensions": [common],
                "difference_dimensions": ["bounded delta"],
            },
        )
        point.relation_evidence_keys = [key]
    state = SimpleNamespace(
        correction_event_evidence_keys=[],
        canonical_points={limit.point_id: limit, other.point_id: other},
        novelty_direction=NoveltyJudgment.POSITIVE,
        graph_guidance_plan=None,
    )

    EvidenceSupervisor._apply_cross_point_limiting_consensus(state, store)

    assert state.novelty_direction == NoveltyJudgment.POSITIVE
    assert state.correction_event_evidence_keys == []

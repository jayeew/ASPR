from __future__ import annotations

from datetime import date

from gear.contracts import QuerySpec, RetrievalCoverageCard
from gear.evidence_supervisor import EvidenceSupervisor
from gear.review_contracts import (
    CanonicalReviewPoint,
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

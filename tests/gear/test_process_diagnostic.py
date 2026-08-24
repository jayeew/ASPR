from __future__ import annotations

from gear.contracts import EvidenceReadiness
from gear.graph_prior_contracts import GraphResultV3
from gear.paper_extraction import PaperRubricBuilder
from gear.process_diagnostic import diagnose_process
from gear.review_state import initialize_review_state_v3


def _graph(paper_id: str) -> GraphResultV3:
    return GraphResultV3(
        paper_id=paper_id,
        score_0_100=50.0,
        p_uptake=0.5,
        conditional_diffusion=0.4,
        feature_coverage=1.0,
    )


def test_graph_unavailable_is_limited_not_low(paper_ir, paper_request) -> None:
    state = initialize_review_state_v3(
        paper_ir,
        PaperRubricBuilder().build(paper_ir),
        None,
        paper_request.evidence_date,
    )
    state.process_features.agent_review_available = True
    diagnostic = diagnose_process(state)
    assert diagnostic.status == "limited"
    assert "graph_unavailable" in diagnostic.reasons


def test_semantic_extraction_degraded_is_advisory(paper_ir, paper_request) -> None:
    state = initialize_review_state_v3(
        paper_ir,
        PaperRubricBuilder().build(paper_ir),
        _graph(paper_ir.paper_id),
        paper_request.evidence_date,
    )
    state.process_features.agent_review_available = True
    state.process_features.graph_score_available = True
    state.process_features.semantic_verifier_passed = True
    state.process_features.stability_passed = True
    paper_ir.quality_report.evidence_readiness = EvidenceReadiness.LIMITED
    paper_ir.quality_report.semantic_extraction_ready = False
    paper_ir.quality_report.blocking_reasons = ["semantic_extraction_degraded"]

    diagnostic = diagnose_process(state, paper_ir)

    assert diagnostic.status == "sufficient"
    assert "semantic_extraction_degraded" not in diagnostic.reasons
    assert "semantic_extraction_degraded" in diagnostic.advisories

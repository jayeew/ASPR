from __future__ import annotations

from datetime import date

from gear.contracts import EvidenceReadiness
from gear.graph_prior_contracts import GraphRuntimePacket, InfluenceForecast
from gear.paper_extraction import PaperRubricBuilder
from gear.process_diagnostic import diagnose_process
from gear.review_state import initialize_review_state


def _graph(paper_id: str) -> GraphRuntimePacket:
    return GraphRuntimePacket(
        paper_id=paper_id,
        cutoff_date=date(2010, 1, 2),
        forecast=InfluenceForecast(
            status="available",
            prospective_5y_diffusion_percentile=50.0,
            uptake_probability=0.5,
            conditional_diffusion=0.4,
            expected_diffusion=0.2,
            feature_coverage=1.0,
            release_id="test",
            model_sha256="sha256:test",
            percentile_reference_sha256="sha256:test",
        ),
    )


def test_graph_unavailable_is_limited_not_low(paper_ir, paper_request) -> None:
    state = initialize_review_state(
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
    state = initialize_review_state(
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

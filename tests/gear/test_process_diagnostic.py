from __future__ import annotations

from gear.graph_prior_contracts import GraphPriorResult
from gear.paper_extraction import PaperRubricBuilder
from gear.process_diagnostic import diagnose_process
from gear.review_state import initialize_review_state_v2


def test_graph_unavailable_is_limited_not_low(paper_ir, paper_request) -> None:
    state = initialize_review_state_v2(
        paper_ir,
        PaperRubricBuilder().build(paper_ir),
        GraphPriorResult(paper_id=paper_ir.paper_id, status="unavailable"),
        paper_request.evidence_date,
    )
    state.process_features.agent_review_available = True
    diagnostic = diagnose_process(state)
    assert diagnostic.status == "limited"
    assert "graph_prior_unavailable" in diagnostic.reasons

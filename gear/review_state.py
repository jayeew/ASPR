"""Initialize the sole compact review state using evidence references."""

from __future__ import annotations

import hashlib
from datetime import date

from .contracts import PaperIR
from .graph_prior_contracts import GraphRuntimePacket, ResourceLedger
from .review_contracts import (
    EvidenceBudget,
    PaperSpecificRubric,
    ProcessFeatures,
    ReviewPhase,
    ReviewState,
)


def initialize_review_state(
    paper_ir: PaperIR,
    rubric: PaperSpecificRubric,
    graph_result: GraphRuntimePacket | None,
    cutoff_date: date,
    *,
    graph_result_evidence_key: str = "G:RESULT",
    action_budget: EvidenceBudget | None = None,
) -> ReviewState:
    if rubric.paper_id != paper_ir.paper_id:
        raise ValueError("state inputs must refer to the same paper")
    if graph_result is not None:
        graph_result = GraphRuntimePacket.model_validate(graph_result)
        if graph_result.paper_id != paper_ir.paper_id:
            raise ValueError("Graph result paper_id mismatch")
        if graph_result.cutoff_date != cutoff_date:
            raise ValueError("Graph result cutoff_date mismatch")
    graph_payload = graph_result.model_dump_json() if graph_result is not None else ""
    identity = (
        f"{paper_ir.paper_sha256}|{cutoff_date.isoformat()}|"
        f"{rubric.model_dump_json()}|{graph_payload}"
    )
    return ReviewState(
        state_id="STATE-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:18],
        phase=ReviewPhase.INITIALIZED,
        paper_id=paper_ir.paper_id,
        paper_sha256=paper_ir.paper_sha256,
        cutoff_date=cutoff_date,
        rubric=rubric,
        branch_reviews={},
        graph_result_evidence_key=(
            graph_result_evidence_key if graph_result is not None else None
        ),
        graph_result=graph_result,
        resource_ledger=ResourceLedger(paper_id=paper_ir.paper_id),
        action_budget=action_budget or EvidenceBudget(),
        process_features=ProcessFeatures(
            graph_score_available=(
                graph_result is not None and graph_result.forecast.status == "available"
            )
        ),
    )


__all__ = ["initialize_review_state"]

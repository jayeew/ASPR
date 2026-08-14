"""Read compatibility for legacy GEAR state artifacts."""

from __future__ import annotations

from .contracts import PaperIR
from .graph_prior_contracts import GraphPriorResult
from .paper_extraction import PaperRubricBuilder
from .review_contracts import (
    BranchReview,
    ProcessFeatures,
    ReviewPhase,
    ReviewSource,
    ReviewState,
    ReviewStateV2,
)


def migrate_review_state_v1(
    state: ReviewState,
    paper_ir: PaperIR,
) -> ReviewStateV2:
    """Convert a V1 state without pretending legacy Graph-exposed input was blind."""
    graph = state.graph_context
    graph_prior = GraphPriorResult(
        paper_id=state.paper_id,
        status=("unavailable" if graph.d5_percentile is None else "exact_lookup"),
        score_0_100=graph.d5_percentile,
        model_id=(
            "legacy_calibration_adapter" if graph.d5_percentile is not None else None
        ),
        feature_coverage=graph.feature_coverage,
        drift_flags=list(graph.drift_flags),
        quality_flags=["migrated_from_graph_exposed_v1"],
    )
    legacy_branch = BranchReview.from_structured(
        state.draft_review,
        source=ReviewSource.AGENT,
        model_id="legacy_v1_reviewer",
        prompt_sha256="sha256:legacy_graph_exposed_prompt",
        input_sha256="sha256:legacy_graph_exposed_input",
        failures=["legacy_branch_independence_not_verifiable"],
    )
    rubric = PaperRubricBuilder().build(paper_ir)
    return ReviewStateV2(
        state_id=state.state_id.replace("STATE-", "STATE2-", 1),
        phase=ReviewPhase.SOURCES_READY,
        paper_id=state.paper_id,
        paper_sha256=state.paper_sha256,
        cutoff_date=state.cutoff_date,
        rubric=rubric,
        branch_reviews={ReviewSource.AGENT: legacy_branch},
        graph_prior_evidence_key="G:CTX",
        graph_prior=graph_prior,
        retrieved_work_evidence_keys=list(state.retrieved_work_evidence_keys),
        relation_evidence_keys=list(state.relation_evidence_keys),
        unresolved_target_ids=[
            point_id
            for point_id, point_state in state.point_states.items()
            if point_state.retained
        ],
        process_features=ProcessFeatures(
            agent_review_available=not legacy_branch.failures,
            graph_score_available=graph_prior.score_0_100 is not None,
            failure_count=len(state.failure_ledger) + 1,
        ),
        failures=list(state.failure_ledger),
    )


__all__ = ["migrate_review_state_v1"]

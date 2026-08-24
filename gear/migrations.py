"""Read compatibility for legacy GEAR state artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .contracts import PaperIR
from .graph_prior_contracts import GraphPriorResult, GraphResultV4
from .paper_extraction import PaperRubricBuilder
from .review_contracts import (
    BranchReview,
    CriticRunMetadata,
    FusionReport,
    ProcessFeatures,
    ReviewBundle,
    ReviewPhase,
    ReviewSource,
    ReviewState,
    ReviewStateV2,
    ReviewStateV3,
    StructuredReview,
    VerificationReport,
)


def graph_result_from_v2(
    graph_prior: GraphPriorResult,
    calibration: Mapping[str, Any] | None = None,
) -> GraphResultV4 | None:
    """Migrate only complete V2 data; never invent missing Graph components."""
    forecast = dict((calibration or {}).get("forecast") or {})
    values = {
        "score_0_100": graph_prior.score_0_100,
        "p_uptake": forecast.get("p_uptake"),
        "conditional_diffusion": forecast.get("conditional_diffusion"),
    }
    if any(value is None for value in values.values()):
        return None
    return GraphResultV4(
        paper_id=graph_prior.paper_id,
        score_0_100=float(values["score_0_100"]),
        p_uptake=float(values["p_uptake"]),
        conditional_diffusion=float(values["conditional_diffusion"]),
        feature_coverage=graph_prior.feature_coverage,
    )


def migrate_review_state_v2(
    state: ReviewStateV2,
    calibration: Mapping[str, Any] | None = None,
) -> ReviewStateV3:
    graph_result = graph_result_from_v2(state.graph_prior, calibration)
    return ReviewStateV3(
        state_id=state.state_id.replace("STATE2-", "STATE3-", 1),
        phase=state.phase,
        paper_id=state.paper_id,
        paper_sha256=state.paper_sha256,
        cutoff_date=state.cutoff_date,
        rubric=state.rubric,
        branch_reviews=state.branch_reviews,
        graph_result_evidence_key="G:RESULT" if graph_result is not None else None,
        graph_result=graph_result,
        canonical_points=state.canonical_points,
        retrieved_work_evidence_keys=state.retrieved_work_evidence_keys,
        relation_evidence_keys=state.relation_evidence_keys,
        unresolved_target_ids=state.unresolved_target_ids,
        action_budget=state.action_budget,
        process_features=state.process_features.model_copy(
            update={"graph_score_available": graph_result is not None}
        ),
        failures=state.failures,
        finalized=state.finalized,
    )


def migrate_review_bundle_v2(payload: Mapping[str, Any]) -> ReviewBundle:
    """Read a persisted V2 bundle and expose it through the V3 runtime model."""
    state_v2 = ReviewStateV2.model_validate(payload["state_v2"])
    calibration = payload.get("calibration")
    state_v3 = migrate_review_state_v2(
        state_v2,
        calibration if isinstance(calibration, Mapping) else None,
    )
    fusion_payload = payload.get("fusion_report")
    fusion_report = None
    if isinstance(fusion_payload, Mapping):
        fusion_report = FusionReport(
            paper_id=str(fusion_payload.get("paper_id") or state_v3.paper_id),
            matches=list(fusion_payload.get("matches") or []),
            canonical_point_ids=list(
                fusion_payload.get("canonical_point_ids") or []
            ),
            failures=list(fusion_payload.get("failures") or []),
        )
    return ReviewBundle(
        status=payload["status"],
        paper_ir=PaperIR.model_validate(payload["paper_ir"]),
        critic=CriticRunMetadata.model_validate(payload["critic"]),
        structured_review=StructuredReview.model_validate(
            payload["structured_review"]
        ),
        review_markdown=str(payload.get("review_markdown") or ""),
        verification=VerificationReport.model_validate(payload["verification"]),
        output_files=dict(payload.get("output_files") or {}),
        agent_review=(
            BranchReview.model_validate(payload["agent_review"])
            if payload.get("agent_review") is not None
            else None
        ),
        qwen_review=(
            BranchReview.model_validate(payload["qwen_review"])
            if payload.get("qwen_review") is not None
            else None
        ),
        graph_result=state_v3.graph_result,
        fusion_report=fusion_report,
        state_v3=state_v3,
        process_diagnostic=(
            dict(payload["process_diagnostic"])
            if isinstance(payload.get("process_diagnostic"), Mapping)
            else None
        ),
        grounding_report=(
            dict(payload["grounding_report"])
            if isinstance(payload.get("grounding_report"), Mapping)
            else None
        ),
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


__all__ = [
    "graph_result_from_v2",
    "migrate_review_bundle_v2",
    "migrate_review_state_v1",
    "migrate_review_state_v2",
]

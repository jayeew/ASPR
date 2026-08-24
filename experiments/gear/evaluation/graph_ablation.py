"""Legal GraphResultV4 variants and isolation checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from gear.graph_prior import graph_result_v4
from gear.graph_prior_contracts import GraphResultV4

from .contracts import GraphAblationVariant


def graph_variants(result: GraphResultV4) -> list[GraphAblationVariant]:
    """Build the four within-paper variants; shuffled is dataset-level."""
    result = graph_result_v4(result)
    return [
        GraphAblationVariant(name="full", result=result),
        GraphAblationVariant(
            name="neutral",
            result=GraphResultV4(
                paper_id=result.paper_id,
                score_0_100=50.0,
                p_uptake=0.0,
                conditional_diffusion=0.0,
                feature_coverage=0.0,
            ),
        ),
        GraphAblationVariant(
            name="score_only",
            result=GraphResultV4(
                paper_id=result.paper_id,
                score_0_100=result.score_0_100,
                p_uptake=result.p_uptake,
                conditional_diffusion=result.conditional_diffusion,
                feature_coverage=result.feature_coverage,
            ),
        ),
        GraphAblationVariant(
            name="guidance_only",
            result=GraphResultV4(
                paper_id=result.paper_id,
                score_0_100=50.0,
                p_uptake=0.0,
                conditional_diffusion=0.0,
                feature_coverage=0.0,
                seed_work_ids=result.seed_work_ids,
                search_terms=result.search_terms,
            ),
        ),
    ]


def shuffled_graph_results(
    results: Sequence[GraphResultV4],
) -> dict[str, GraphResultV4]:
    """Cycle values across papers while retaining each target paper ID."""
    if len(results) < 2:
        raise ValueError("shuffled Graph ablation requires at least two papers")
    migrated = [graph_result_v4(result) for result in results]
    shuffled: dict[str, GraphResultV4] = {}
    for index, target in enumerate(migrated):
        source = migrated[(index + 1) % len(migrated)]
        shuffled[target.paper_id] = GraphResultV4(
            paper_id=target.paper_id,
            score_0_100=source.score_0_100,
            p_uptake=source.p_uptake,
            conditional_diffusion=source.conditional_diffusion,
            feature_coverage=source.feature_coverage,
            seed_work_ids=source.seed_work_ids,
            search_terms=source.search_terms,
        )
    return shuffled


def assert_branch_isolation(
    run_payloads: Mapping[str, Mapping[str, object]],
) -> None:
    """Require identical pre-Fusion Agent artifacts across all variants."""
    required = ("agent_input_sha256", "agent_branch_sha256", "candidate_set_sha256")
    for field in required:
        values = {str(payload.get(field, "")) for payload in run_payloads.values()}
        if "" in values or len(values) != 1:
            raise ValueError(f"Graph ablation branch isolation failed for {field}")


def graph_tension(score_0_100: float, coverage: float, agent_axis: int) -> float:
    direction = (score_0_100 - 50.0) / 50.0
    return coverage * abs(direction) * abs(direction - agent_axis) / 2.0


def tension_band(value: float) -> str:
    if value < 0.25:
        return "low"
    if value < 0.50:
        return "medium"
    return "high"


__all__ = [
    "assert_branch_isolation",
    "graph_tension",
    "graph_variants",
    "shuffled_graph_results",
    "tension_band",
]

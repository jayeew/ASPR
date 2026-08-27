"""Four equal-budget Graph variants and isolation checks."""

from __future__ import annotations

from collections.abc import Mapping

from gear.graph_prior_contracts import GraphRuntimePacket

from .contracts import GraphAblationVariant


def graph_variants(
    target: GraphRuntimePacket,
    *,
    placebo: GraphRuntimePacket,
) -> list[GraphAblationVariant]:
    if target.paper_id == placebo.paper_id:
        raise ValueError("placebo Graph packet must come from a different paper")
    if target.cutoff_date != placebo.cutoff_date:
        raise ValueError("placebo packet must be pre-matched to the target cutoff")
    no_topology = target.model_copy(update={"topology_seeds": []})
    placebo_packet = target.model_copy(
        update={
            "forecast": placebo.forecast,
            "topology_seeds": placebo.topology_seeds,
            "diagnostics": [*target.diagnostics, "evaluation_placebo_graph"],
        }
    )
    return [
        GraphAblationVariant(name="neutral", result=no_topology),
        GraphAblationVariant(name="score", result=no_topology),
        GraphAblationVariant(name="score_topology", result=target),
        GraphAblationVariant(name="placebo_graph", result=placebo_packet),
    ]


def assert_branch_isolation(
    run_payloads: Mapping[str, Mapping[str, object]],
) -> None:
    if set(run_payloads) != {
        "neutral",
        "score",
        "score_topology",
        "placebo_graph",
    }:
        raise ValueError("Graph ablation requires exactly four variants")
    for field in ("draft_sha256", "resource_caps_sha256"):
        values = {str(payload.get(field, "")) for payload in run_payloads.values()}
        if "" in values or len(values) != 1:
            raise ValueError(f"Graph branch isolation failed for {field}")
    for field in ("local_candidate_pool_sha256", "remote_candidate_pool_sha256"):
        values = {
            str(run_payloads[name].get(field, "")) for name in ("neutral", "score")
        }
        if "" in values or len(values) != 1:
            raise ValueError(f"score changed the candidate pool for {field}")


__all__ = ["assert_branch_isolation", "graph_variants"]

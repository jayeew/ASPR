"""Matched-resource Graph packet variants and isolation checks."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

from gear.graph_prior import graph_runtime_packet
from gear.graph_prior_contracts import GraphRuntimePacketV1, GraphTopologySeedV1

from .contracts import GraphAblationVariant


def graph_variants(result: GraphRuntimePacketV1) -> list[GraphAblationVariant]:
    result = graph_runtime_packet(result)
    return [
        GraphAblationVariant(
            name="neutral", result=_packet(result, score=50.0, profile=False)
        ),
        GraphAblationVariant(name="score_only", result=_packet(result, profile=False)),
        GraphAblationVariant(
            name="score_profile", result=_packet(result, profile=True)
        ),
        GraphAblationVariant(name="full", result=result),
    ]


def shuffled_graph_components(
    results: Sequence[GraphRuntimePacketV1],
) -> dict[str, dict[str, GraphRuntimePacketV1]]:
    if len(results) < 2:
        raise ValueError("shuffled Graph ablation requires at least two papers")
    packets = sorted(
        (graph_runtime_packet(result) for result in results),
        key=lambda row: row.paper_id,
    )
    controls: dict[str, dict[str, GraphRuntimePacketV1]] = {}
    for index, target in enumerate(packets):
        donor = packets[(index + 1) % len(packets)]
        controls[target.paper_id] = {
            "shuffled_score": target.model_copy(
                update={
                    "score_0_100": donor.score_0_100,
                    "raw_expected_diffusion": donor.raw_expected_diffusion,
                    "p_uptake": donor.p_uptake,
                    "conditional_diffusion": donor.conditional_diffusion,
                }
            ),
            "shuffled_profile": target.model_copy(
                update={
                    "feature_values": donor.feature_values,
                    "historical_bands": donor.historical_bands,
                    "missing_feature_ids": donor.missing_feature_ids,
                    "diagnostic_flags": donor.diagnostic_flags,
                }
            ),
            "random_matched_topology": target.model_copy(
                update={
                    "topology_seeds": _matched_topology(target, packets),
                    "diagnostic_flags": list(
                        dict.fromkeys(
                            [
                                *target.diagnostic_flags,
                                "evaluation_identity_rewired_topology",
                            ]
                        )
                    ),
                }
            ),
        }
    return controls


def _matched_topology(
    target: GraphRuntimePacketV1, packets: Sequence[GraphRuntimePacketV1]
) -> list[GraphTopologySeedV1]:
    pool = [
        seed
        for packet in packets
        if packet.paper_id != target.paper_id
        for seed in packet.topology_seeds
        if seed.work_id not in {item.work_id for item in target.topology_seeds}
    ]
    selected: list[GraphTopologySeedV1] = []
    used: set[str] = set()
    for anchor in target.topology_seeds:
        anchor_fields = set(anchor.anchor_field_ids)
        ranked = sorted(
            (seed for seed in pool if seed.work_id not in used),
            key=lambda seed: (
                abs((seed.publication_year or 0) - (anchor.publication_year or 0)),
                -len(anchor_fields & set(seed.anchor_field_ids)),
                abs(seed.shared_reference_count - anchor.shared_reference_count),
                _stable_control_order(target.paper_id, anchor.work_id, seed.work_id),
            ),
        )
        if not ranked:
            break
        donor = ranked[0]
        # Runtime guidance now searches by seed title instead of direct ID.
        # Replacing only work_id would therefore execute the real Full query
        # and make the random control semantically identical.  Use the whole
        # matched donor seed; the ResourceLedger separately enforces equal
        # logical request caps even when claim alignment differs.
        selected.append(donor)
        used.add(donor.work_id)
    return selected


def _stable_control_order(paper_id: str, anchor_id: str, donor_id: str) -> str:
    return hashlib.sha256(f"{paper_id}|{anchor_id}|{donor_id}".encode()).hexdigest()


def shuffled_graph_results(
    results: Sequence[GraphRuntimePacketV1],
) -> dict[str, GraphRuntimePacketV1]:
    """Compatibility alias for isolated shuffled-score controls."""
    controls = shuffled_graph_components(results)
    return {paper_id: rows["shuffled_score"] for paper_id, rows in controls.items()}


def _packet(
    source: GraphRuntimePacketV1,
    *,
    score: float | None = None,
    profile: bool,
) -> GraphRuntimePacketV1:
    neutral = score is not None
    return GraphRuntimePacketV1(
        paper_id=source.paper_id,
        score_0_100=source.score_0_100 if score is None else score,
        raw_expected_diffusion=0.0 if neutral else source.raw_expected_diffusion,
        p_uptake=0.0 if neutral else source.p_uptake,
        conditional_diffusion=0.0 if neutral else source.conditional_diffusion,
        feature_values=source.feature_values if profile else {},
        historical_bands=source.historical_bands if profile else {},
        missing_feature_ids=source.missing_feature_ids if profile else [],
        diagnostic_flags=source.diagnostic_flags if profile else [],
        topology_seeds=[],
    )


def assert_branch_isolation(
    run_payloads: Mapping[str, Mapping[str, object]],
) -> None:
    required = ("agent_input_sha256", "agent_branch_sha256", "candidate_set_sha256")
    for field in required:
        values = {str(payload.get(field, "")) for payload in run_payloads.values()}
        if "" in values or len(values) != 1:
            raise ValueError(f"Graph ablation branch isolation failed for {field}")


def graph_tension(score_0_100: float, coverage: float, agent_axis: int) -> float:
    """Read-only legacy metric; current guidance policy never consumes it."""
    direction = (score_0_100 - 50.0) / 50.0
    return coverage * abs(direction) * abs(direction - agent_axis) / 2.0


def tension_band(value: float) -> str:
    """Read-only compatibility bucketing for historical result files."""
    return "low" if value < 0.25 else "medium" if value < 0.50 else "high"


__all__ = [
    "assert_branch_isolation",
    "graph_tension",
    "graph_variants",
    "shuffled_graph_components",
    "shuffled_graph_results",
    "tension_band",
]

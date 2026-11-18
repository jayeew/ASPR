"""Source-bound detailed profile for panel c, separate from the main paper."""

from __future__ import annotations

import math
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

from figure_pipeline.fig1_reference.data import connect, local_metrics, paths, rao

from .data import ROOT, file_source, write


def prepare_profile(data: dict[str, Any], out: Path) -> None:
    """Recompute the displayed X descriptors without altering saved study results."""
    case = next(c for c in data["contrasts"] if c["alias"] == "X")
    fact, role = case["fact"], case["claim"]["claim_type"]
    local = local_metrics(fact["neighbors"], fact["neighbor_edges"], role)
    probabilities = local["community_probabilities"]
    community_ids = sorted(probabilities)
    pair_table = pd.read_parquet(
        ROOT / "data/claim_graph/community_pair_history.parquet"
    )
    lookup = {
        (int(r["community_a"]), int(r["community_b"])): r
        for r in pair_table.to_dict("records")
    }
    pair_rows, surprise = [], []
    for pair in combinations(community_ids, 2):
        row = lookup.get(pair)
        pair_rows.append(
            {"communities": pair, "observed": row is not None, "record": row}
        )
        if row and all(
            row[k] > 0
            for k in [
                "community_a_claim_count",
                "community_b_claim_count",
                "historical_claim_count",
            ]
        ):
            commonness = (
                (row["pair_connector_count"] + 0.5)
                * row["historical_claim_count"]
                / (row["community_a_claim_count"] * row["community_b_claim_count"])
            )
            surprise.append(-math.log(commonness))
    with connect("paper_graph_index.sqlite") as db:
        witnesses = {
            n["claim_id"]: paths(db, case["paper"], n) for n in fact["neighbors"]
        }
    values = {
        name: local[name]
        for name in [
            "nearest_prior_similarity",
            "mean_top5_similarity",
            "effective_community_count",
            "neighbor_induced_density",
            "component_merge_count",
            "newly_connected_neighbor_pair_count",
        ]
    }
    values.update(
        {
            "community_rao_stirling": rao(probabilities),
            "first_observed_recent_nature_pair_share": sum(
                not r["observed"] for r in pair_rows
            )
            / len(pair_rows),
            "community_pair_mean_surprisal": sum(surprise) / len(surprise)
            if len(surprise) == len(pair_rows)
            else None,
            "cross_boundary_weight_share": 1 - max(probabilities.values()),
            "cross_type_neighbor_count": sum(
                n["claim_type"] != role for n in fact["neighbors"]
            ),
        }
    )
    for name, kind in [
        ("direct_citation_neighbor_count", "direct"),
        ("two_hop_neighbor_count", "two_hop"),
        ("co_citation_neighbor_count", "shared_reference"),
    ]:
        values[name] = (
            sum(
                any(w["kind"] == kind for w in r["witnesses"])
                for r in witnesses.values()
            )
            if all(r["counts"] is not None for r in witnesses.values())
            else None
        )
    if any(v is None or not math.isfinite(v) for v in values.values()):
        raise ValueError(
            "Selected X example lacks complete finite metrics; select another real example."
        )
    # Date objects in the source parquet are made explicit JSON date strings.
    for row in pair_rows:
        if row["record"]:
            row["record"]["first_observed_date"] = str(
                row["record"]["first_observed_date"]
            )
    sources = [
        file_source(ROOT / "data/claim_graph" / name)
        for name in [
            "claim_communities.parquet",
            "community_pair_history.parquet",
            "community_centroid_index.parquet",
            "community_centroid_matrix.npy",
        ]
    ]
    profile = {
        "alias": case["alias"],
        "claim_id": case["claim"]["claim_id"],
        "paper_id": case["paper"]["paper_id"],
        "metrics": values,
        "saved_metrics": case["metrics"],
        "community_probabilities": probabilities,
        "community_pairs": pair_rows,
        "path_witnesses": witnesses,
        "sources": sources,
        "scope": "Figure-derived X profile; does not replace C1 facts, joint analysis, or saved study outputs.",
    }
    data["detailed_profile"] = profile
    data["sources"] = list({r["path"]: r for r in data["sources"] + sources}.values())
    write(out / "data/detailed_profile_X.json", profile)
    write(out / "data/path_witnesses_X.json", witnesses)

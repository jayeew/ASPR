"""Pre-locked structural validation subset and RGPM-S graph deltas."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.metrics import adjusted_rand_score

from gear.corpus import normalize_openalex_id

from .cohorts import select_structural_subset
from .graph_snapshots import PriorGraph, SnapshotRepository, sample_reference_pairs
from .targets import build_structural_targets


STRUCTURAL_DEFINITION_VERSION = "nature-multihorizon-structural-v1"


def _reference_list(value: Any) -> List[str]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = []
    elif isinstance(value, (list, tuple, np.ndarray)):
        parsed = list(value)
    else:
        parsed = []
    return sorted(
        set(
            normalized
            for item in parsed
            if (normalized := normalize_openalex_id(item))
        )
    )


def annotate_future_reference_coverage(
    membership: pd.DataFrame,
    future_citers: pd.DataFrame,
) -> pd.DataFrame:
    """Add future-citer bibliography coverage for deterministic subset locking."""
    output = membership.copy()
    citers = future_citers.copy()
    if "referenced_works" not in citers.columns:
        output["future_citer_reference_coverage"] = 0.0
        return output
    citers["has_reference_metadata"] = citers["referenced_works"].map(
        lambda value: int(bool(_reference_list(value)))
    )
    coverage = (
        citers.groupby(["paper_id", "horizon"], observed=True)["has_reference_metadata"]
        .mean()
        .rename("future_citer_reference_coverage")
        .reset_index()
    )
    output = output.drop(columns=["future_citer_reference_coverage"], errors="ignore")
    return output.merge(coverage, on=["paper_id", "horizon"], how="left", validate="one_to_one")


def annotate_future_reference_coverage_from_parquet(
    membership: pd.DataFrame,
    future_citers_path: Path,
    *,
    batch_size: int = 100_000,
) -> pd.DataFrame:
    """Stream future-citer bibliography coverage for cohort keys only."""

    required = {"paper_id", "horizon"}
    missing = sorted(required - set(membership.columns))
    if missing:
        raise ValueError(f"membership is missing columns: {missing}")
    wanted = membership[["paper_id", "horizon"]].drop_duplicates().copy()
    wanted["paper_id"] = wanted["paper_id"].astype(str)
    wanted["horizon"] = pd.to_numeric(
        wanted["horizon"], errors="raise"
    ).astype(int)
    totals: Counter[Tuple[str, int]] = Counter()
    valid: Counter[Tuple[str, int]] = Counter()
    parquet = pq.ParquetFile(Path(future_citers_path))
    schema_names = set(parquet.schema_arrow.names)
    if not {"paper_id", "horizon", "referenced_works"}.issubset(schema_names):
        output = membership.copy()
        output["future_citer_reference_coverage"] = 0.0
        return output
    for batch in parquet.iter_batches(
        batch_size=int(batch_size),
        columns=["paper_id", "horizon", "referenced_works"],
    ):
        keys = batch.select(["paper_id", "horizon"]).to_pandas()
        reference_index = batch.schema.get_field_index("referenced_works")
        lengths = pc.fill_null(
            pc.list_value_length(batch.column(reference_index)), 0
        ).to_numpy(zero_copy_only=False)
        keys["has_reference_metadata"] = (lengths > 0).astype(int)
        selected = keys.merge(
            wanted,
            on=["paper_id", "horizon"],
            how="inner",
            validate="many_to_one",
        )
        if selected.empty:
            continue
        grouped = selected.groupby(
            ["paper_id", "horizon"], observed=True
        )["has_reference_metadata"].agg(["size", "sum"])
        for key, row in grouped.iterrows():
            normalized = (str(key[0]), int(key[1]))
            totals[normalized] += int(row["size"])
            valid[normalized] += int(row["sum"])
    coverage = pd.DataFrame(
        [
            {
                "paper_id": paper_id,
                "horizon": horizon,
                "future_citer_reference_coverage": (
                    float(valid[(paper_id, horizon)] / total)
                    if total
                    else 0.0
                ),
            }
            for (paper_id, horizon), total in totals.items()
        ]
    )
    output = membership.drop(
        columns=["future_citer_reference_coverage"], errors="ignore"
    )
    if coverage.empty:
        output = output.copy()
        output["future_citer_reference_coverage"] = 0.0
        return output
    return output.merge(
        coverage,
        on=["paper_id", "horizon"],
        how="left",
        validate="one_to_one",
    )


def read_future_citers_for_subset(
    future_citers_path: Path,
    subset: pd.DataFrame,
    *,
    batch_size: int = 100_000,
) -> pd.DataFrame:
    """Read only pre-locked paper/horizon rows from a large Parquet table."""

    keys = subset[["paper_id", "horizon"]].drop_duplicates().copy()
    keys["paper_id"] = keys["paper_id"].astype(str)
    keys["horizon"] = pd.to_numeric(keys["horizon"], errors="raise").astype(int)
    if keys.empty:
        return pd.DataFrame()
    parquet = pq.ParquetFile(Path(future_citers_path))
    required = [
        "paper_id",
        "horizon",
        "citer_id",
        "citer_year",
        "referenced_works",
    ]
    missing = sorted(set(required) - set(parquet.schema_arrow.names))
    if missing:
        raise ValueError(f"future_citers is missing structural columns: {missing}")
    parts: List[pd.DataFrame] = []
    for batch in parquet.iter_batches(
        batch_size=int(batch_size), columns=required
    ):
        frame = batch.to_pandas()
        selected = frame.merge(
            keys,
            on=["paper_id", "horizon"],
            how="inner",
            validate="many_to_one",
        )
        if not selected.empty:
            parts.append(selected)
    if not parts:
        return pd.DataFrame(columns=required)
    return pd.concat(parts, ignore_index=True)


def lock_structural_subset(
    membership: pd.DataFrame,
    future_citers: Union[pd.DataFrame, Path],
    *,
    max_papers: int = 5_000,
    min_future_reference_coverage: float = 0.80,
    seed: int = 20260710,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Select a score-independent subset after bibliography coverage is known."""
    if isinstance(future_citers, (str, Path)):
        annotated = annotate_future_reference_coverage_from_parquet(
            membership, Path(future_citers)
        )
    else:
        annotated = annotate_future_reference_coverage(membership, future_citers)
    selected = select_structural_subset(
        annotated,
        max_papers=max_papers,
        min_future_reference_coverage=min_future_reference_coverage,
        seed=seed,
    )
    by_horizon = {
        str(int(horizon)): {
            "n_selected": int(len(group)),
            "n_domains": int(group["domain12"].nunique()),
            "confirmatory": bool(
                len(group) >= 2_000 and group["domain12"].nunique() >= 6
            ),
        }
        for horizon, group in selected.groupby("horizon", sort=True)
    }
    audit = {
        "definition_version": STRUCTURAL_DEFINITION_VERSION,
        "seed": int(seed),
        "n_selected": int(len(selected)),
        "n_domains": int(selected["domain12"].nunique()) if "domain12" in selected else 0,
        "n_horizons": int(selected["horizon"].nunique()) if "horizon" in selected else 0,
        "min_future_reference_coverage": float(min_future_reference_coverage),
        "maximum_per_horizon": int(max_papers),
        "by_horizon": by_horizon,
        "exploratory": not bool(
            by_horizon.get("5", {}).get("confirmatory", False)
        ),
        "selection_uses_prediction_or_target": False,
    }
    return selected, audit


def _cross_community_share(edges: Iterable[Tuple[str, str]], graph: PriorGraph) -> float:
    values = list(edges)
    if not values:
        return 0.0
    return float(
        np.mean(
            [
                graph.community.get(left, -1) != graph.community.get(right, -1)
                for left, right in values
            ]
        )
    )


def _partition_labels(graph: nx.Graph, nodes: Sequence[str]) -> List[int]:
    if not nodes:
        return []
    if graph.number_of_edges() == 0:
        return list(range(len(nodes)))
    groups = list(nx.algorithms.community.asyn_lpa_communities(graph, seed=20260710))
    lookup = {
        str(node): index
        for index, group in enumerate(groups)
        for node in group
    }
    return [int(lookup.get(node, -1)) for node in nodes]


def _focal_efficiency(graph: nx.Graph, focal_nodes: Sequence[str]) -> float:
    nodes = [node for node in focal_nodes if node in graph]
    if len(nodes) < 2:
        return 0.0
    values: List[float] = []
    for index, source in enumerate(nodes):
        lengths = nx.single_source_shortest_path_length(graph, source)
        for target in nodes[index + 1 :]:
            distance = lengths.get(target)
            values.append(0.0 if distance is None or distance <= 0 else 1.0 / distance)
    return float(np.mean(values)) if values else 0.0


def _local_structural_delta(
    paper_id: str,
    horizon: int,
    focal_references: Sequence[str],
    citer_rows: pd.DataFrame,
    prior: PriorGraph,
    *,
    max_nodes: int,
    max_pairs: int,
    seed: int,
) -> Dict[str, Any]:
    focal = sorted(set(focal_references))
    counts: Counter[str] = Counter()
    valid_citer_bibliographies = 0
    for value in citer_rows.get("referenced_works", pd.Series(dtype=object)):
        references = _reference_list(value)
        if references:
            valid_citer_bibliographies += 1
            counts.update(references)
    adopted = [item for item, _ in counts.most_common(max(0, max_nodes - len(focal)))]
    nodes = sorted(set(focal + adopted))[:max_nodes]
    base = nx.Graph()
    base.add_nodes_from(nodes)
    base_pairs, base_rate = sample_reference_pairs(
        nodes,
        max_pairs=max_pairs,
        seed_key=f"{seed}:{paper_id}:{horizon}:base",
    )
    base_edges = [pair for pair in base_pairs if pair in prior.edges]
    base.add_edges_from(base_edges)
    future = base.copy()
    added_edges: set[Tuple[str, str]] = set()
    node_set = set(nodes)
    for citer in citer_rows.to_dict("records"):
        bibliography = [item for item in _reference_list(citer.get("referenced_works")) if item in node_set]
        pairs, _ = sample_reference_pairs(
            bibliography,
            max_pairs=max_pairs,
            seed_key=f"{seed}:{paper_id}:{horizon}:{citer.get('citer_id', '')}",
        )
        added_edges.update(pairs)
    future.add_edges_from(added_edges)

    focal_for_partition = [node for node in focal if node in nodes]
    before_labels = _partition_labels(base, focal_for_partition)
    after_labels = _partition_labels(future, focal_for_partition)
    partition_change = (
        float(1.0 - adjusted_rand_score(before_labels, after_labels))
        if len(focal_for_partition) >= 2
        else 0.0
    )
    base_cross = _cross_community_share(base.edges(), prior)
    future_cross = _cross_community_share(future.edges(), prior)
    return {
        "paper_id": paper_id,
        "horizon": int(horizon),
        "modularity_shock": float(
            prior.modularity()
            - prior.modularity_after_sampled_pairs(
                sorted(added_edges),
                sampling_rate=1.0,
            )
        ),
        "boundary_mixing_change": float(future_cross - base_cross),
        "partition_change": partition_change,
        "path_shortening": float(
            _focal_efficiency(future, focal_for_partition)
            - _focal_efficiency(base, focal_for_partition)
        ),
        "n_local_nodes": int(len(nodes)),
        "n_base_edges": int(base.number_of_edges()),
        "n_future_edges_added": int(len(added_edges - set(base.edges()))),
        "n_future_citers_with_references": int(valid_citer_bibliographies),
        "base_pair_sampling_rate": float(base_rate),
        "definition_version": STRUCTURAL_DEFINITION_VERSION,
        "source_max_year": int(prior.source_max_year),
    }


def build_structural_validation(
    subset: pd.DataFrame,
    papers: pd.DataFrame,
    paper_references: pd.DataFrame,
    future_citers: pd.DataFrame,
    snapshot_catalog: Union[pd.DataFrame, Path],
    *,
    max_nodes: int = 250,
    max_pairs: int = 10_000,
    seed: int = 20260710,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Compute four future-structure deltas and aggregate RGPM-S3/S5/S8."""
    repository = SnapshotRepository(snapshot_catalog)
    paper_year = papers.set_index("paper_id")["publication_year"].to_dict()
    reference_map = (
        paper_references.groupby("paper_id", observed=True)["reference_id"]
        .agg(list)
        .to_dict()
    )
    citer_groups = {
        (str(paper_id), int(horizon)): group
        for (paper_id, horizon), group in future_citers.groupby(["paper_id", "horizon"], observed=True)
    }
    rows: List[Dict[str, Any]] = []
    for record in subset.to_dict("records"):
        paper_id = str(record["paper_id"])
        horizon = int(record["horizon"])
        year = int(paper_year[paper_id])
        prior = repository.for_year(year)
        delta = _local_structural_delta(
            paper_id,
            horizon,
            reference_map.get(paper_id, []),
            citer_groups.get((paper_id, horizon), pd.DataFrame()),
            prior,
            max_nodes=max_nodes,
            max_pairs=max_pairs,
            seed=seed,
        )
        delta["domain12"] = record.get("domain12")
        delta["publication_year"] = year
        rows.append(delta)
    deltas = pd.DataFrame(rows)
    targets = build_structural_targets(deltas) if not deltas.empty else pd.DataFrame()
    by_horizon = {
        str(int(horizon)): {
            "n_deltas": int(len(group)),
            "n_domains": int(group["domain12"].nunique()),
            "confirmatory": bool(
                len(group) >= 2_000 and group["domain12"].nunique() >= 6
            ),
        }
        for horizon, group in deltas.groupby("horizon", sort=True)
    } if not deltas.empty else {}
    audit = {
        "definition_version": STRUCTURAL_DEFINITION_VERSION,
        "n_subset": int(len(subset)),
        "n_deltas": int(len(deltas)),
        "n_targets": int(len(targets)),
        "n_domains": int(deltas["domain12"].nunique()) if not deltas.empty else 0,
        "by_horizon": by_horizon,
        "go_for_confirmatory": bool(
            by_horizon.get("5", {}).get("confirmatory", False)
        ),
        "go_for_confirmatory_tau5": bool(
            by_horizon.get("5", {}).get("confirmatory", False)
        ),
    }
    return deltas, targets, audit

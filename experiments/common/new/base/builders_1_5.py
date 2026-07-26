"""Data builders for Fig.1–Fig.5 of the Nature-style ASPR figure suite."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import MDS
from sklearn.metrics import ndcg_score

from experiments.common.new.base.common import (
    ANGLE_LABELS,
    ANGLE_ORDER,
    FEATURE_LABELS,
    MODEL_LABELS,
    FigureBundle,
    SuitePaths,
    grouped_percentile,
    load_json,
    numeric,
    percentile_rank,
    safe_spearman,
    stable_seed,
)


PRIMARY_FEATURES: Tuple[str, ...] = tuple(FEATURE_LABELS)


def _read_table(path: Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Read a local CSV or Parquet table."""
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, columns=list(columns) if columns else None)
    frame = pd.read_csv(path, usecols=list(columns) if columns else None)
    return frame


def _v61_paths(paths: SuitePaths) -> Dict[str, Path]:
    """Resolve frequently used v6.1 artifacts."""
    analysis = paths["v6_1_analysis"]
    dataset = paths["v6_1_dataset"]
    baseline = paths["v6_1_figure_baseline"]
    return {
        "papers": dataset / "papers_primary_articles.parquet",
        "features": dataset / "innovation_candidate_features.parquet",
        "controls": dataset / "control_features_v6_1.parquet",
        "targets": dataset / "targets_zero_inclusive.parquet",
        "oof": analysis / "oof_d361264b867c/oof_predictions.parquet",
        "oof_metrics": analysis / "oof_d361264b867c/oof_metrics.csv",
        "oof_folds": analysis / "oof_d361264b867c/oof_fold_metrics.csv",
        "folds": analysis / "oof_d361264b867c/temporal_folds.csv",
        "pure_oof": (
            analysis
            / "supplement_innovation_only_3b387272d53d"
            / "innovation_only_oof_predictions.parquet"
        ),
        "screening": analysis / "screening_ceec00f0809b",
        "model_points": baseline / "experiment_04/data/model_points.csv",
        "paired_gains": baseline / "experiment_04/data/paired_gains.csv",
        "prediction_deciles": baseline / "experiment_04/data/prediction_deciles.csv",
        "angle_summary": baseline / "experiment_09/data/angle_summary.csv",
    }


# ============================================================================
# Fig.1 — landmark graph perturbation
# ============================================================================


def _normalize_positions(nodes: pd.DataFrame) -> pd.DataFrame:
    """Normalize supplied graph coordinates without changing their geometry."""
    output = nodes.copy()
    for column in ("x", "y"):
        values = numeric(output[column])
        span = float(values.max() - values.min())
        output[column] = 2.0 * (values - values.min()) / max(span, 1e-12) - 1.0
    return output


def _community_year_counts(works: pd.DataFrame) -> pd.DataFrame:
    """Count visible papers by community and year."""
    clean = works.dropna(subset=["community", "year"]).copy()
    clean["community"] = clean["community"].astype(int)
    clean["year"] = numeric(clean["year"]).astype(int)
    return (
        clean.groupby(["community", "year"], as_index=False)
        .size()
        .rename(columns={"size": "paper_count"})
    )


def _select_snapshot_communities(
    topic_nodes: pd.DataFrame,
    works: pd.DataFrame,
    top_n: int,
) -> List[int]:
    """Keep high-volume topics while guaranteeing inclusion of landmark topics."""
    top = (
        topic_nodes.sort_values(["n_papers", "cited_by_count"], ascending=False)
        .head(int(top_n))["community"]
        .astype(int)
        .tolist()
    )
    anchors = (
        works.loc[works["anchor_label"].notna(), "community"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    output = list(dict.fromkeys(anchors + top))
    return output[: max(int(top_n), len(anchors))]


def _aggregate_snapshot_edges(
    works: pd.DataFrame,
    edges: pd.DataFrame,
    communities: Sequence[int],
    cutoff: int,
) -> pd.DataFrame:
    """Aggregate paper-level links into one time-bounded community graph."""
    work_map = works.set_index("id")[["year", "community"]].copy()
    work_map["year"] = numeric(work_map["year"])
    edge = edges.merge(
        work_map.rename(columns={"year": "source_year", "community": "source_community"}),
        left_on="source",
        right_index=True,
        how="inner",
    )
    edge = edge.merge(
        work_map.rename(columns={"year": "target_year", "community": "target_community"}),
        left_on="target",
        right_index=True,
        how="inner",
    )
    edge = edge.loc[
        edge[["source_year", "target_year"]].max(axis=1).le(int(cutoff))
        & edge["source_community"].isin(communities)
        & edge["target_community"].isin(communities)
        & edge["source_community"].ne(edge["target_community"])
    ].copy()
    if edge.empty:
        return pd.DataFrame(
            columns=["source_community", "target_community", "weight", "edge_count"]
        )
    left = edge[["source_community", "target_community"]].min(axis=1).astype(int)
    right = edge[["source_community", "target_community"]].max(axis=1).astype(int)
    edge["left"] = left
    edge["right"] = right
    return (
        edge.groupby(["left", "right"], as_index=False)
        .agg(weight=("weight", "sum"), edge_count=("weight", "size"))
        .rename(columns={"left": "source_community", "right": "target_community"})
        .sort_values("weight", ascending=False)
    )


def _snapshot_tables(
    domain: Mapping[str, Any],
    graph_root: Path,
    top_n: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build three fixed-layout topic snapshots and one CRISPR-ready diff table."""
    root = graph_root / str(domain["slug"])
    nodes = _normalize_positions(pd.read_csv(root / "topic_nodes.csv"))
    works = pd.read_csv(root / "works_selected.csv")
    paper_edges = pd.read_csv(root / "paper_edges.csv")
    communities = _select_snapshot_communities(nodes, works, top_n)
    landmark_year = int(domain["landmark_year"])
    cutoffs = {
        "pre": landmark_year - 1,
        "landmark_window": landmark_year + 2,
        "post": min(landmark_year + 8, int(numeric(works["year"]).max())),
    }
    counts = _community_year_counts(works)
    node_rows: List[pd.DataFrame] = []
    edge_rows: List[pd.DataFrame] = []
    stage_edges: Dict[str, pd.DataFrame] = {}
    selected_nodes = nodes.loc[nodes["community"].astype(int).isin(communities)].copy()
    anchor_communities = set(
        works.loc[works["anchor_label"].notna(), "community"].dropna().astype(int)
    )
    for stage, cutoff in cutoffs.items():
        stage_nodes = selected_nodes.copy()
        stage_counts = (
            counts.loc[counts["year"].le(cutoff)]
            .groupby("community")["paper_count"]
            .sum()
        )
        stage_nodes["paper_count_visible"] = (
            stage_nodes["community"].astype(int).map(stage_counts).fillna(0).astype(int)
        )
        stage_nodes = stage_nodes.loc[stage_nodes["paper_count_visible"].gt(0)].copy()
        stage_nodes["domain"] = str(domain["slug"])
        stage_nodes["domain_label"] = str(domain["label"])
        stage_nodes["stage"] = stage
        stage_nodes["cutoff_year"] = cutoff
        stage_nodes["is_landmark_community"] = (
            stage_nodes["community"].astype(int).isin(anchor_communities).astype(int)
        )
        node_rows.append(stage_nodes)
        stage_edge = _aggregate_snapshot_edges(
            works,
            paper_edges,
            stage_nodes["community"].astype(int).tolist(),
            cutoff,
        )
        stage_edge["domain"] = str(domain["slug"])
        stage_edge["domain_label"] = str(domain["label"])
        stage_edge["stage"] = stage
        stage_edge["cutoff_year"] = cutoff
        stage_edges[stage] = stage_edge.copy()
        edge_rows.append(stage_edge)
    pre = stage_edges["pre"].rename(columns={"weight": "pre_weight"})
    post = stage_edges["post"].rename(columns={"weight": "post_weight"})
    diff = post.merge(
        pre[["source_community", "target_community", "pre_weight"]],
        on=["source_community", "target_community"],
        how="left",
    )
    diff["pre_weight"] = numeric(diff["pre_weight"]).fillna(0.0)
    diff["delta_weight"] = numeric(diff["post_weight"]) - diff["pre_weight"]
    diff["is_new_bridge"] = diff["pre_weight"].eq(0).astype(int)
    diff["domain"] = str(domain["slug"])
    return (
        pd.concat(node_rows, ignore_index=True),
        pd.concat(edge_rows, ignore_index=True),
        diff.sort_values("delta_weight", ascending=False),
        works,
    )


def _paper_graph_modularity(edge: pd.DataFrame) -> float:
    """Calculate modularity of the paper graph under frozen topic labels."""
    if edge.empty:
        return float("nan")
    graph = nx.Graph()
    for row in edge.itertuples(index=False):
        graph.add_edge(
            str(row.source),
            str(row.target),
            weight=float(row.weight),
        )
    groups: Dict[int, set[str]] = defaultdict(set)
    for row in edge.itertuples(index=False):
        if str(row.source) in graph:
            groups[int(row.sc)].add(str(row.source))
        if str(row.target) in graph:
            groups[int(row.tc)].add(str(row.target))
    partition = [members for members in groups.values() if members]
    if len(partition) < 2 or not graph.number_of_edges():
        return float("nan")
    return float(nx.algorithms.community.modularity(graph, partition, weight="weight"))


def _graph_metrics_by_cutoff(
    works: pd.DataFrame,
    paper_edges: pd.DataFrame,
    years: Iterable[int],
) -> pd.DataFrame:
    """Compute transparent cumulative graph diagnostics for selected cutoffs."""
    work_map = works.set_index("id")[["year", "community"]].copy()
    work_map["year"] = numeric(work_map["year"])
    joined = paper_edges.merge(
        work_map.rename(columns={"year": "sy", "community": "sc"}),
        left_on="source",
        right_index=True,
        how="inner",
    ).merge(
        work_map.rename(columns={"year": "ty", "community": "tc"}),
        left_on="target",
        right_index=True,
        how="inner",
    )
    joined["appearance_year"] = joined[["sy", "ty"]].max(axis=1)
    joined = joined.dropna(subset=["sc", "tc", "appearance_year"])
    joined["sc"] = joined["sc"].astype(int)
    joined["tc"] = joined["tc"].astype(int)
    rows: List[Dict[str, Any]] = []
    for cutoff in sorted(set(int(year) for year in years)):
        visible_works = works.loc[numeric(works["year"]).le(cutoff)].copy()
        visible_edges = joined.loc[joined["appearance_year"].le(cutoff)].copy()
        topic_counts = visible_works["community"].dropna().astype(int).value_counts()
        probabilities = topic_counts / max(topic_counts.sum(), 1)
        if visible_edges.empty:
            rows.append(
                {
                    "cutoff_year": cutoff,
                    "paper_count": int(len(visible_works)),
                    "community_coverage": int(topic_counts.size),
                    "bridge_share": 0.0,
                    "modularity": 0.0,
                    "diffusion_reach": 0,
                    "diffusion_evenness": float(
                        entropy(probabilities) / math.log(len(probabilities))
                        if len(probabilities) > 1
                        else 0.0
                    ),
                }
            )
            continue
        visible_edges["left"] = visible_edges[["sc", "tc"]].min(axis=1)
        visible_edges["right"] = visible_edges[["sc", "tc"]].max(axis=1)
        aggregate = (
            visible_edges.groupby(["left", "right"], as_index=False)["weight"]
            .sum()
            .rename(
                columns={
                    "left": "source_community",
                    "right": "target_community",
                }
            )
        )
        inter = aggregate["source_community"].ne(aggregate["target_community"])
        weights = numeric(aggregate["weight"]).fillna(0)
        rows.append(
            {
                "cutoff_year": cutoff,
                "paper_count": int(len(visible_works)),
                "community_coverage": int(topic_counts.size),
                "bridge_share": float(weights.loc[inter].sum() / max(weights.sum(), 1e-12)),
                "modularity": _paper_graph_modularity(visible_edges),
                "diffusion_reach": int(
                    aggregate.loc[inter, ["source_community", "target_community"]]
                    .stack()
                    .nunique()
                ),
                "diffusion_evenness": float(
                    entropy(probabilities) / math.log(len(probabilities))
                    if len(probabilities) > 1
                    else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def _aligned_event_metrics(
    domains: Sequence[Mapping[str, Any]],
    graph_root: Path,
    bin_width: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Align five transparent graph diagnostics around four landmark years."""
    raw_rows: List[pd.DataFrame] = []
    yearly_cache: Dict[str, pd.DataFrame] = {}
    for domain in domains:
        root = graph_root / str(domain["slug"])
        works = pd.read_csv(root / "works_selected.csv")
        edges = pd.read_csv(root / "paper_edges.csv")
        landmark = int(domain["landmark_year"])
        offsets = list(range(-2, 3))
        cutoffs = [landmark + offset * int(bin_width) + int(bin_width) - 1 for offset in offsets]
        metrics = _graph_metrics_by_cutoff(works, edges, cutoffs)
        metrics["relative_period"] = offsets
        metrics["domain"] = str(domain["slug"])
        metrics["domain_label"] = str(domain["label"])
        raw_rows.append(metrics)
        min_year = max(int(numeric(works["year"]).min()) + 2, landmark - 12)
        max_year = min(int(numeric(works["year"]).max()) - 2, landmark + 12)
        yearly_cache[str(domain["slug"])] = _graph_metrics_by_cutoff(
            works,
            edges,
            range(min_year, max_year + 1),
        )
    raw = pd.concat(raw_rows, ignore_index=True)
    metric_columns = [
        "community_coverage",
        "bridge_share",
        "modularity",
        "diffusion_reach",
        "diffusion_evenness",
    ]
    long = raw.melt(
        id_vars=["domain", "domain_label", "relative_period", "cutoff_year"],
        value_vars=metric_columns,
        var_name="metric",
        value_name="value",
    )
    long["z_value"] = (
        long.groupby(["domain", "metric"])["value"]
        .transform(lambda s: (s - s.mean()) / max(float(s.std(ddof=0)), 1e-12))
    )
    yearly = pd.concat(
        [frame.assign(domain=domain) for domain, frame in yearly_cache.items()],
        ignore_index=True,
    )
    return long, yearly


def _event_shock(
    metrics: pd.DataFrame,
    event_year: int,
) -> float:
    """Combine standardized pre/post changes without fitting an outcome."""
    before = metrics.loc[metrics["cutoff_year"].eq(int(event_year) - 2)]
    after = metrics.loc[metrics["cutoff_year"].eq(int(event_year) + 2)]
    if before.empty or after.empty:
        return float("nan")
    signed = {
        "community_coverage": 1.0,
        "bridge_share": 1.0,
        "modularity": -1.0,
        "diffusion_reach": 1.0,
        "diffusion_evenness": 1.0,
    }
    changes = []
    for column, direction in signed.items():
        scale = max(float(numeric(metrics[column]).std(ddof=0)), 1e-12)
        change = (
            direction
            * (float(after.iloc[0][column]) - float(before.iloc[0][column]))
            / scale
        )
        if np.isfinite(change):
            changes.append(change)
    return float(np.mean(changes)) if len(changes) >= 3 else float("nan")


def _pseudo_event_estimates(
    domains: Sequence[Mapping[str, Any]],
    graph_root: Path,
    draws: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Match landmark events to similarly sized within-domain pseudo-events."""
    candidates: List[Dict[str, Any]] = []
    real_rows: List[Dict[str, Any]] = []
    for domain in domains:
        root = graph_root / str(domain["slug"])
        works = pd.read_csv(root / "works_selected.csv")
        edges = pd.read_csv(root / "paper_edges.csv")
        landmark = int(domain["landmark_year"])
        min_year = int(numeric(works["year"]).min()) + 2
        max_year = int(numeric(works["year"]).max()) - 2
        metric_start = min(min_year, landmark - 2)
        metrics = _graph_metrics_by_cutoff(
            works,
            edges,
            range(metric_start, max_year + 1),
        )
        event_count = int(numeric(works["year"]).between(landmark - 1, landmark + 1).sum())
        domain_candidates: List[Dict[str, Any]] = []
        for year in range(min_year + 2, max_year - 1):
            if abs(year - landmark) <= 4:
                continue
            count = int(numeric(works["year"]).between(year - 1, year + 1).sum())
            size_ratio = count / max(event_count, 1)
            domain_candidates.append(
                {
                    "domain": str(domain["slug"]),
                    "event_year": year,
                    "paper_count": count,
                    "size_ratio": size_ratio,
                    "shock": _event_shock(metrics, year),
                    "event_type": "pseudo",
                    "match_distance": abs(math.log(max(size_ratio, 1e-12))),
                }
            )
        domain_candidates = [
            row
            for row in domain_candidates
            if np.isfinite(float(row["shock"]))
        ]
        domain_candidates = sorted(
            domain_candidates,
            key=lambda row: (float(row["match_distance"]), int(row["event_year"])),
        )[:12]
        candidates.extend(domain_candidates)
        real_rows.append(
            {
                "domain": str(domain["slug"]),
                "event_year": landmark,
                "paper_count": event_count,
                "size_ratio": 1.0,
                "shock": _event_shock(metrics, landmark),
                "event_type": "landmark",
                "matched_pool_size": len(domain_candidates),
                "match_distance": 0.0,
            }
        )
    columns = [
        "domain",
        "event_year",
        "paper_count",
        "size_ratio",
        "shock",
        "event_type",
        "match_distance",
    ]
    candidate_frame = pd.DataFrame(candidates, columns=columns).dropna(subset=["shock"])
    real_frame = pd.DataFrame(real_rows).dropna(subset=["shock"])
    rng = np.random.default_rng(int(seed))
    draw_rows: List[Dict[str, Any]] = []
    for draw in range(1, int(draws) + 1):
        selected = []
        for domain in real_frame["domain"]:
            pool = candidate_frame.loc[candidate_frame["domain"].eq(domain)]
            if pool.empty:
                continue
            selected.append(float(pool.iloc[int(rng.integers(0, len(pool)))]["shock"]))
        if not selected:
            continue
        pseudo_mean = float(np.mean(selected))
        real_mean = float(real_frame["shock"].mean())
        draw_rows.append(
            {
                "draw": draw,
                "landmark_mean": real_mean,
                "pseudo_mean": pseudo_mean,
                "difference": real_mean - pseudo_mean,
            }
        )
    return pd.concat([real_frame, candidate_frame], ignore_index=True), pd.DataFrame(draw_rows)


def build_fig1(
    config: Mapping[str, Any],
    paths: SuitePaths,
) -> FigureBundle:
    """Build Fig.1 graph snapshots, event alignment and pseudo-event controls."""
    graph_root = paths["graph_view_root"]
    top_n = int(config["fig1"]["top_communities"])
    node_rows: List[pd.DataFrame] = []
    edge_rows: List[pd.DataFrame] = []
    diff_rows: List[pd.DataFrame] = []
    source_paths: List[Path] = []
    for domain in config["graph_domains"]:
        nodes, edges, diff, _ = _snapshot_tables(domain, graph_root, top_n)
        node_rows.append(nodes)
        edge_rows.append(edges)
        diff_rows.append(diff)
        source_paths.extend(
            [
                graph_root / str(domain["slug"]) / "topic_nodes.csv",
                graph_root / str(domain["slug"]) / "works_selected.csv",
                graph_root / str(domain["slug"]) / "paper_edges.csv",
            ]
        )
    aligned, yearly = _aligned_event_metrics(
        config["graph_domains"],
        graph_root,
        int(config["fig1"]["event_bin_width_years"]),
    )
    event_pool, draws = _pseudo_event_estimates(
        config["graph_domains"],
        graph_root,
        int(config["fig1"]["pseudo_event_draws"]),
        int(config["fig1"]["seed"]),
    )
    tables = {
        "snapshot_nodes": pd.concat(node_rows, ignore_index=True),
        "snapshot_edges": pd.concat(edge_rows, ignore_index=True),
        "structural_differences": pd.concat(diff_rows, ignore_index=True),
        "event_aligned_metrics": aligned,
        "yearly_graph_metrics": yearly,
        "pseudo_event_pool": event_pool,
        "pseudo_event_draws": draws,
    }
    real_mean = float(draws["landmark_mean"].iloc[0]) if not draws.empty else float("nan")
    difference = float(draws["difference"].mean()) if not draws.empty else float("nan")
    interval = (
        draws["difference"].quantile([0.025, 0.975]).tolist()
        if not draws.empty
        else [float("nan"), float("nan")]
    )
    panel_text = {
        "a": "Fixed-layout topic graphs before, during and after each landmark window.",
        "b": "CRISPR post-minus-pre links; amber links are newly visible cross-topic bridges.",
        "c": "Five graph diagnostics aligned to landmark-relative two-year periods and standardized within domain and diagnostic.",
        "d": {
            "landmark_mean_shock": real_mean,
            "mean_difference_vs_matched_pseudo": difference,
            "difference_interval": interval,
            "warning": config["claim_boundaries"]["fig1"],
        },
    }
    contract = {
        "figure_id": 1,
        "panels": {
            "a": {"mark": "fixed-layout network small multiples", "data": ["snapshot_nodes", "snapshot_edges"]},
            "b": {"mark": "difference network", "data": ["structural_differences"]},
            "c": {"mark": "ridgeline/raincloud", "data": ["event_aligned_metrics"]},
            "d": {"mark": "Gardner–Altman estimation", "data": ["pseudo_event_pool", "pseudo_event_draws"]},
        },
        "numeric_rendering": "python_only",
        "causal_claim": False,
    }
    return FigureBundle(
        figure_id=1,
        title="Landmark papers coincide with knowledge-graph reconfiguration",
        status="complete_descriptive_control",
        tables=tables,
        panel_text=panel_text,
        chart_contract=contract,
        source_paths=source_paths,
        notes=[config["claim_boundaries"]["fig1"]],
    )


# ============================================================================
# Fig.2 — indicator governance and construct checks
# ============================================================================


def _registry_primary_map(registry: Mapping[str, Any]) -> pd.DataFrame:
    """Flatten the frozen eight primary indicator records."""
    rows: List[Dict[str, Any]] = []
    for candidate_id, candidate in registry["candidates"].items():
        if candidate.get("final_role") != "primary":
            continue
        screen = candidate.get("empirical_screen", {})
        rows.append(
            {
                "candidate_id": candidate_id,
                "code_name": candidate["code_name"],
                "feature_label": FEATURE_LABELS.get(candidate["code_name"], candidate["code_name"]),
                "angle_id": candidate["angle_id"],
                "angle_label": ANGLE_LABELS[candidate["angle_id"]],
                "formula": candidate.get("formula"),
                "original_source_count": len(candidate.get("original_source_ids", [])),
                "application_source_count": len(candidate.get("paper_application_source_ids", [])),
                "validation_source_count": len(candidate.get("validation_source_ids", [])),
                "source_ids": "|".join(
                    dict.fromkeys(
                        candidate.get("original_source_ids", [])
                        + candidate.get("paper_application_source_ids", [])
                        + candidate.get("validation_source_ids", [])
                    )
                ),
                "overall_coverage": screen.get("overall_coverage"),
                "minimum_domain_coverage": screen.get("minimum_domain_coverage"),
                "stability_spearman": screen.get("stability_spearman"),
                "stability_median_relative_error": screen.get(
                    "stability_median_relative_error"
                ),
                "approximation_spearman": screen.get("approximation_spearman"),
                "toy_test_pass": screen.get("toy_test_pass"),
                "temporal_test_pass": screen.get("temporal_test_pass"),
                "nondegenerate_test_pass": screen.get("nondegenerate_test_pass"),
            }
        )
    return pd.DataFrame(rows).sort_values(["angle_id", "candidate_id"])


def _first_failure_reason(row: pd.Series) -> str:
    """Assign one mutually exclusive, non-outcome gate destination."""
    if row.get("proposed_final_role") == "excluded":
        reason = str(row.get("proposed_decision_reason", "")).lower()
        if "external" in reason or "not available" in reason or "without" in reason:
            return "External/unavailable data"
        if "future" in reason or "publication-time" in reason:
            return "Future leakage"
        if not bool(row.get("coverage_pass", 0)):
            return "Coverage gate"
        if not bool(row.get("stability_pass", 0)):
            return "Stability gate"
        if not bool(row.get("approximation_pass", 0)):
            return "Formula fidelity"
        return "Evidence/construct rule"
    if not bool(row.get("coverage_pass", 0)):
        return "Coverage gate"
    if not bool(row.get("stability_pass", 0)):
        return "Stability gate"
    if not bool(row.get("approximation_pass", 0)):
        return "Formula fidelity"
    if not bool(row.get("eligible_all_runtime_gates", 0)):
        return "Evidence/construct rule"
    return "All runtime gates"


def _screening_sankey(decisions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate candidate flow through angle, computability, gates and role."""
    frame = decisions.copy()
    frame["discovery"] = "Literature candidates"
    frame["angle"] = frame["angle_id"].map(ANGLE_LABELS)
    frame["computability"] = np.where(
        numeric(frame["raw_overall_coverage"]).fillna(0).gt(0),
        "Local computation available",
        "No local implementation",
    )
    frame["runtime_gate"] = frame.apply(_first_failure_reason, axis=1)
    frame["role"] = frame["proposed_final_role"].str.title()
    stages = ["discovery", "angle", "computability", "runtime_gate", "role"]
    rows: List[Dict[str, Any]] = []
    for order, (source, target) in enumerate(zip(stages[:-1], stages[1:]), start=1):
        counts = (
            frame.groupby([source, target], dropna=False)
            .size()
            .reset_index(name="candidate_count")
        )
        counts["stage_order"] = order
        counts["source_stage"] = source
        counts["target_stage"] = target
        counts = counts.rename(columns={source: "source", target: "target"})
        rows.extend(counts.to_dict("records"))
    return pd.DataFrame(rows)


def _indicator_distributions(
    features: pd.DataFrame,
    papers: pd.DataFrame,
    sample_size: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create field-year normalized distribution and correlation contracts."""
    frame = features[["paper_id", "publication_year", "domain12", *PRIMARY_FEATURES]].copy()
    for feature in PRIMARY_FEATURES:
        frame[f"{feature}__percentile"] = grouped_percentile(
            frame,
            feature,
            ["domain12", "publication_year"],
            id_column="paper_id",
        )
    rng = np.random.default_rng(int(seed))
    distributions: List[pd.DataFrame] = []
    for feature in PRIMARY_FEATURES:
        valid = frame.loc[
            frame[f"{feature}__percentile"].notna(),
            [
                "paper_id",
                "publication_year",
                "domain12",
                feature,
                f"{feature}__percentile",
            ],
        ].copy()
        raw = numeric(valid[feature])
        median = float(raw.median())
        iqr = float(raw.quantile(0.75) - raw.quantile(0.25))
        valid["robust_value"] = ((raw - median) / max(iqr, 1e-12)).clip(-3, 3)
        if len(valid) > int(sample_size):
            valid = valid.iloc[
                np.sort(rng.choice(len(valid), size=int(sample_size), replace=False))
            ]
        valid = valid.rename(
            columns={
                feature: "raw_value",
                f"{feature}__percentile": "percentile",
            }
        )
        valid["code_name"] = feature
        valid["feature_label"] = FEATURE_LABELS[feature]
        distributions.append(valid)
    correlation = frame[list(PRIMARY_FEATURES)].corr(method="spearman")
    corr_long = (
        correlation.rename_axis("feature_x")
        .reset_index()
        .melt(id_vars="feature_x", var_name="feature_y", value_name="spearman")
    )
    corr_long["label_x"] = corr_long["feature_x"].map(FEATURE_LABELS)
    corr_long["label_y"] = corr_long["feature_y"].map(FEATURE_LABELS)
    pair_rows: List[pd.DataFrame] = []
    pairs = [
        ("field_variety", "rao_stirling_integration"),
        ("field_gini_balance", "field_disparity_cosine_mean"),
        ("reference_overlap_novelty_t0", "first_time_source_pair_share"),
    ]
    for left, right in pairs:
        pair = frame[["paper_id", left, right]].dropna()
        if len(pair) > 700:
            pair = pair.sample(
                700,
                random_state=stable_seed(f"{left}:{right}", seed),
            )
        pair = pair.rename(columns={left: "x", right: "y"})
        pair["feature_x"] = left
        pair["feature_y"] = right
        pair_rows.append(pair)
    del papers
    return pd.concat(distributions, ignore_index=True), corr_long, pd.concat(pair_rows)


def _known_group_effects(
    features: pd.DataFrame,
    oof: pd.DataFrame,
    iterations: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compare top-decile future diffusion papers with matched controls."""
    truth = oof.loc[
        oof["model_id"].eq("final_innovation_plus_k1"),
        ["paper_id", "publication_year", "domain12", "realized_diffusion_target"],
    ].drop_duplicates("paper_id")
    frame = truth.merge(
        features[
            [
                "paper_id",
                "valid_reference_count",
                *PRIMARY_FEATURES,
            ]
        ],
        on="paper_id",
        how="inner",
    )
    frame["target_percentile"] = percentile_rank(
        frame["realized_diffusion_target"],
        frame["paper_id"],
    )
    frame["reference_bin"] = pd.qcut(
        numeric(frame["valid_reference_count"]).rank(method="first"),
        5,
        labels=False,
        duplicates="drop",
    )
    frame["known_group"] = frame["target_percentile"].ge(0.90)
    high = frame.loc[frame["known_group"]].copy()
    controls = frame.loc[~frame["known_group"]].copy()
    matched_rows: List[Dict[str, Any]] = []
    for row in high.itertuples(index=False):
        pool = controls.loc[
            controls["domain12"].eq(row.domain12)
            & controls["publication_year"].eq(row.publication_year)
            & controls["reference_bin"].eq(row.reference_bin)
        ]
        if pool.empty:
            continue
        chosen = pool.iloc[
            stable_seed(str(row.paper_id), seed) % len(pool)
        ]
        matched_rows.extend(
            [
                {"pair_id": row.paper_id, "paper_id": row.paper_id, "group": "High future diffusion"},
                {"pair_id": row.paper_id, "paper_id": chosen["paper_id"], "group": "Matched control"},
            ]
        )
    membership = pd.DataFrame(matched_rows)
    matched = membership.merge(
        frame[["paper_id", "domain12", "publication_year", *PRIMARY_FEATURES]],
        on="paper_id",
        how="left",
    )
    long_rows: List[pd.DataFrame] = []
    for feature in PRIMARY_FEATURES:
        values = frame[["paper_id", "domain12", "publication_year", feature]].copy()
        values["percentile"] = grouped_percentile(
            values,
            feature,
            ["domain12", "publication_year"],
            id_column="paper_id",
        )
        subset = matched[["pair_id", "paper_id", "group"]].merge(
            values[["paper_id", "percentile"]],
            on="paper_id",
            how="left",
        )
        subset["code_name"] = feature
        long_rows.append(subset)
    long = pd.concat(long_rows, ignore_index=True)
    pivot = (
        long.pivot_table(
            index=["pair_id", "code_name"],
            columns="group",
            values="percentile",
            aggfunc="first",
        )
        .dropna()
        .reset_index()
    )
    rng = np.random.default_rng(int(seed))
    effects: List[Dict[str, Any]] = []
    for feature, group in pivot.groupby("code_name"):
        delta = (
            group["High future diffusion"].to_numpy(float)
            - group["Matched control"].to_numpy(float)
        )
        estimates = np.empty(int(iterations), dtype=float)
        for index in range(int(iterations)):
            estimates[index] = rng.choice(delta, size=len(delta), replace=True).mean()
        effects.append(
            {
                "code_name": feature,
                "feature_label": FEATURE_LABELS[feature],
                "n_pairs": len(delta),
                "mean_percentile_difference": float(delta.mean()),
                "ci_low": float(np.quantile(estimates, 0.025)),
                "ci_high": float(np.quantile(estimates, 0.975)),
            }
        )
    return membership, pd.DataFrame(effects)


def build_fig2(
    config: Mapping[str, Any],
    paths: SuitePaths,
) -> FigureBundle:
    """Build Fig.2 candidate flow, evidence map and construct diagnostics."""
    resolved = _v61_paths(paths)
    registry = load_json(paths["candidate_registry"])
    decisions_path = resolved["screening"] / "candidate_decisions.csv"
    decisions = pd.read_csv(decisions_path)
    primary_map = _registry_primary_map(registry)
    papers = pd.read_parquet(
        resolved["papers"],
        columns=["paper_id", "publication_year", "domain12"],
    )
    features = pd.read_parquet(
        resolved["features"],
        columns=["paper_id", "publication_year", "domain12", "valid_reference_count", *PRIMARY_FEATURES],
    )
    distributions, correlations, pair_sample = _indicator_distributions(
        features,
        papers,
        int(config["fig2"]["distribution_sample_per_indicator"]),
        int(config["fig2"]["seed"]),
    )
    oof = pd.read_parquet(
        resolved["oof"],
        columns=[
            "paper_id",
            "publication_year",
            "domain12",
            "model_id",
            "realized_diffusion_target",
        ],
    )
    known_membership, known_effects = _known_group_effects(
        features,
        oof,
        int(config["fig2"]["bootstrap_iterations"]),
        int(config["fig2"]["seed"]),
    )
    sources = pd.DataFrame(registry["sources"].values())
    angles = pd.DataFrame(registry["observation_angles"].values())
    gate_columns = [
        "candidate_id",
        "code_name",
        "angle_id",
        "overall_coverage",
        "minimum_domain_coverage",
        "stability_spearman",
        "stability_median_relative_error",
        "approximation_spearman",
        "approximation_median_relative_error",
        "coverage_pass",
        "stability_pass",
        "approximation_pass",
        "toy_test_pass",
        "temporal_test_pass",
        "nondegenerate_test_pass",
        "proposed_final_role",
    ]
    quality = decisions.loc[
        decisions["proposed_final_role"].eq("primary"),
        gate_columns,
    ].copy()
    tables = {
        "candidate_decisions": decisions,
        "candidate_flow": _screening_sankey(decisions),
        "observation_angles": angles,
        "primary_indicator_map": primary_map,
        "source_map": sources,
        "indicator_distributions": distributions,
        "indicator_correlations": correlations,
        "correlation_pair_sample": pair_sample,
        "primary_quality_gates": quality,
        "known_group_membership": known_membership,
        "known_group_effects": known_effects,
    }
    role_counts = decisions["proposed_final_role"].value_counts().to_dict()
    panel_text = {
        "a": {
            "candidate_count": int(len(decisions)),
            "role_counts": {str(k): int(v) for k, v in role_counts.items()},
            "selection_statement": registry["selection_principle"],
        },
        "b": {
            "centre": "Publication-time reference evidence",
            "angles": {
                angle_id: registry["observation_angles"][angle_id]
                for angle_id in ANGLE_ORDER
            },
        },
        "c": "Robust-scaled raw values expose distribution shape; raw-value Spearman exposes redundancy.",
        "d": "Primary gates: overall coverage ≥0.70, weakest-domain coverage ≥0.50, 80% resampling Spearman ≥0.90.",
        "e": {
            "definition": "Top 10% realized D5 diffusion versus same-domain, same-year, reference-volume matched controls.",
            "warning": config["claim_boundaries"]["fig2"],
        },
    }
    contract = {
        "figure_id": 2,
        "panels": {
            "a": {"mark": "Sankey", "data": ["candidate_flow"]},
            "b": {"mark": "evidence wheel", "data": ["observation_angles", "primary_indicator_map"]},
            "c": {"mark": "violin + sole triangular correlation matrix", "data": ["indicator_distributions", "indicator_correlations"]},
            "d": {"mark": "two-dimensional threshold scatter", "data": ["primary_quality_gates"]},
            "e": {"mark": "paired effect estimates", "data": ["known_group_effects"]},
        },
        "traditional_heatmap_count": 1,
        "outcome_used_for_indicator_selection": False,
    }
    return FigureBundle(
        figure_id=2,
        title="Evidence governance yields five angles and eight primary indicators",
        status="complete_registered_evidence_map",
        tables=tables,
        panel_text=panel_text,
        chart_contract=contract,
        source_paths=[
            paths["candidate_registry"],
            decisions_path,
            resolved["features"],
            resolved["oof"],
        ],
        notes=[config["claim_boundaries"]["fig2"]],
    )


# ============================================================================
# Fig.3 — temporal OOF predictive validity
# ============================================================================


def _model_ladder(
    model_points: pd.DataFrame,
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Attach descriptive cross-fold ranges to global OOF estimates."""
    folds = (
        fold_metrics.loc[fold_metrics["horizon"].eq(5)]
        .groupby("model_id")["spearman_expected"]
        .agg(fold_min="min", fold_max="max", fold_median="median")
        .reset_index()
    )
    output = model_points.loc[model_points["horizon"].eq(5)].merge(
        folds,
        on="model_id",
        how="left",
    )
    output["model_label_en"] = output["model_id"].map(MODEL_LABELS)
    return output.sort_values("model_order")


def _final_oof_sample(
    oof: pd.DataFrame,
    sample_per_decile: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build an exact OOF joint table and a bounded raincloud sample."""
    frame = oof.loc[
        oof["model_id"].eq("final_innovation_plus_k1")
        & oof["horizon"].eq(5),
        [
            "paper_id",
            "publication_year",
            "domain12",
            "outer_fold_id",
            "expected_diffusion_score",
            "realized_diffusion_target",
        ],
    ].dropna(subset=["expected_diffusion_score", "realized_diffusion_target"])
    frame["prediction_decile"] = pd.qcut(
        frame["expected_diffusion_score"].rank(method="first"),
        10,
        labels=False,
    ).astype(int) + 1
    rng = np.random.default_rng(int(seed))
    samples = []
    for _, group in frame.groupby("prediction_decile"):
        if len(group) > int(sample_per_decile):
            group = group.iloc[
                np.sort(rng.choice(len(group), int(sample_per_decile), replace=False))
            ]
        samples.append(group)
    sampled = pd.concat(samples, ignore_index=True)
    top_definition = percentile_rank(
        frame["realized_diffusion_target"],
        frame["paper_id"],
    ).ge(0.90)
    frame["realized_top_decile"] = top_definition.astype(int)
    enrichment = (
        frame.groupby("prediction_decile")
        .agg(
            n=("paper_id", "size"),
            observed_top_rate=("realized_top_decile", "mean"),
            target_mean=("realized_diffusion_target", "mean"),
            target_median=("realized_diffusion_target", "median"),
        )
        .reset_index()
    )
    enrichment["enrichment_over_base"] = (
        enrichment["observed_top_rate"] / max(float(frame["realized_top_decile"].mean()), 1e-12)
    )
    sampled = sampled.merge(
        enrichment[["prediction_decile", "enrichment_over_base"]],
        on="prediction_decile",
        how="left",
    )
    return frame, sampled


def build_fig3(
    config: Mapping[str, Any],
    paths: SuitePaths,
) -> FigureBundle:
    """Build Fig.3 temporal OOF model and prediction diagnostics."""
    resolved = _v61_paths(paths)
    model_points = pd.read_csv(resolved["model_points"])
    fold_metrics = pd.read_csv(resolved["oof_folds"])
    paired = pd.read_csv(resolved["paired_gains"])
    folds = pd.read_csv(resolved["folds"])
    if "horizon" in folds:
        folds = folds.loc[folds["horizon"].eq(5)].copy()
    angles = pd.read_csv(resolved["angle_summary"])
    oof = pd.read_parquet(
        resolved["oof"],
        columns=[
            "paper_id",
            "publication_year",
            "domain12",
            "horizon",
            "model_id",
            "outer_fold_id",
            "expected_diffusion_score",
            "realized_diffusion_target",
        ],
    )
    joint, sampled = _final_oof_sample(
        oof,
        int(config["fig3"]["raincloud_sample_per_decile"]),
        int(config["fig3"]["seed"]),
    )
    main_rho = safe_spearman(
        joint["expected_diffusion_score"],
        joint["realized_diffusion_target"],
    )
    tables = {
        "temporal_folds": folds,
        "model_ladder": _model_ladder(model_points, fold_metrics),
        "paired_model_gains": paired,
        "oof_joint_density": joint,
        "prediction_decile_sample": sampled,
        "angle_add_delete": angles,
    }
    top_enrichment = (
        sampled.loc[sampled["prediction_decile"].eq(10), "enrichment_over_base"].iloc[0]
        if not sampled.empty
        else float("nan")
    )
    panel_text = {
        "a": {
            "model": "Two-part medium model",
            "part_1": "P(future uptake > 0)",
            "part_2": "Diffusion intensity | future uptake",
            "output": "Expected diffusion = uptake probability × conditional intensity",
            "fold_count": int(len(folds)),
        },
        "b": {
            "main_oof_spearman": main_rho,
            "k1_spearman": float(
                model_points.loc[model_points["model_id"].eq("k1_controls"), "spearman_expected"].iloc[0]
            ),
            "innovation_only_spearman": float(
                model_points.loc[model_points["model_id"].eq("innovation_only"), "spearman_expected"].iloc[0]
            ),
            "interval_note": "Horizontal whiskers are the range across six time folds; paired gain confidence intervals are shown separately.",
        },
        "c": {"n": int(len(joint)), "spearman": main_rho},
        "d": {"highest_decile_enrichment": float(top_enrichment)},
        "e": {
            "interpretation": "Single-angle additions and leave-one-angle-out losses are post-hoc diagnostics, never selection criteria.",
            "warning": config["claim_boundaries"]["fig3"],
        },
    }
    contract = {
        "figure_id": 3,
        "panels": {
            "a": {"mark": "deterministic model schematic", "data": ["temporal_folds"]},
            "b": {"mark": "estimate ladder", "data": ["model_ladder", "paired_model_gains"]},
            "c": {"mark": "hexbin with marginal densities", "data": ["oof_joint_density"]},
            "d": {"mark": "decile rainclouds", "data": ["prediction_decile_sample"]},
            "e": {"mark": "polar add-delete dot plot", "data": ["angle_add_delete"]},
        },
        "primary_outcome": "D5 realized diffusion",
        "primary_metric": "global temporal-OOF Spearman",
    }
    return FigureBundle(
        figure_id=3,
        title="Publication-time evidence predicts five-year diffusion out of time",
        status="complete_temporal_oof",
        tables=tables,
        panel_text=panel_text,
        chart_contract=contract,
        source_paths=[
            resolved["oof"],
            resolved["oof_metrics"],
            resolved["oof_folds"],
            resolved["folds"],
            resolved["model_points"],
            resolved["paired_gains"],
            resolved["angle_summary"],
        ],
        notes=[config["claim_boundaries"]["fig3"]],
    )


# ============================================================================
# Fig.4 — blinded peer review alignment
# ============================================================================


def _label_completion(fig4_root: Path) -> Tuple[pd.DataFrame, int, int]:
    """Read the immutable blinded-label completion audit."""
    audit_path = fig4_root / "fig4_blinded_labeling_completion_audit.csv"
    audit = pd.read_csv(audit_path)
    completed = 0
    required_labels = [
        "label_novelty_1_5",
        "label_significance_1_5",
        "label_prior_art_1_5",
    ]
    for labeler_path in sorted(
        fig4_root.glob("fig4_completed_blinded_labels_labeler_*.csv")
    ):
        labels = pd.read_csv(labeler_path)
        if all(column in labels for column in required_labels):
            completed += int(labels[required_labels].notna().all(axis=1).sum())
    required = 90
    return audit, min(completed, required), required


def _fig4_sample_coverage(fig4_root: Path) -> pd.DataFrame:
    """Create the range-coverage panel from the locked blinded answer key."""
    answer = pd.read_csv(fig4_root / "fig4_blinded_labeling_answer_key.csv")
    score_candidates = [
        column
        for column in answer.columns
        if "score" in column.lower() or "prediction" in column.lower()
    ]
    score_column = score_candidates[0] if score_candidates else None
    output = answer.copy()
    output["validation_score"] = (
        numeric(output[score_column]) if score_column else np.nan
    )
    return output


def build_fig4(
    config: Mapping[str, Any],
    paths: SuitePaths,
) -> FigureBundle:
    """Build Fig.4 while hard-blocking unfinished blinded validity claims."""
    root = paths["fig4_root"]
    audit, completed, required = _label_completion(root)
    coverage = _fig4_sample_coverage(root)
    tables: Dict[str, pd.DataFrame] = {
        "blinded_completion_audit": audit,
        "validation_sample_coverage": coverage,
        "human_model_connections": pd.DataFrame(),
        "agreement_estimates": pd.DataFrame(),
        "score_by_human_rating": pd.DataFrame(),
        "quote_grounded_cases": pd.DataFrame(),
    }
    status = "complete_external_construct_validity" if completed >= required else "draft_labels_incomplete"
    panel_text = {
        "a": {
            "sampling": "Locked low/middle/high score strata, 10 primary cases per stratum.",
            "required_judgements": required,
            "completed_judgements": completed,
        },
        "b": "Blocked until blinded labels are returned.",
        "c": "Blocked until blinded labels are returned.",
        "d": "Blocked until blinded labels are returned.",
        "e": "Blocked until blinded labels are returned.",
        "warning": config["claim_boundaries"]["fig4"],
    }
    contract = {
        "figure_id": 4,
        "status_gate": {
            "required_rows": required,
            "completed_rows": completed,
            "passed": completed >= required,
        },
        "panels": {
            "a": {"mark": "density + rug", "data": ["validation_sample_coverage"]},
            "b": {"mark": "parallel connection plot", "blocked": completed < required},
            "c": {"mark": "agreement estimate plot", "blocked": completed < required},
            "d": {"mark": "rating rainclouds", "blocked": completed < required},
            "e": {"mark": "quote-grounded cards", "blocked": completed < required},
        },
    }
    return FigureBundle(
        figure_id=4,
        title="Blinded peer-review construct validity",
        status=status,
        tables=tables,
        panel_text=panel_text,
        chart_contract=contract,
        source_paths=[
            root / "fig4_blinded_labeling_answer_key.csv",
            root / "fig4_blinded_labeling_completion_audit.csv",
            root / "fig4_blinded_labeling_protocol.json",
        ],
        notes=[config["claim_boundaries"]["fig4"]],
    )


# ============================================================================
# Fig.5 — strict historical frontier backtests
# ============================================================================


def _topic_scores_for_cutoff(
    frame: pd.DataFrame,
    cutoff: int,
    seed_window: int,
    validation_window: int,
    minimum_seed: int,
    minimum_validation: int,
) -> pd.DataFrame:
    """Aggregate fold-valid paper scores into one historical topic backtest."""
    seed_start = int(cutoff) - int(seed_window) + 1
    validation_end = int(cutoff) + int(validation_window)
    seed = frame.loc[
        frame["publication_year"].between(seed_start, cutoff)
        & frame["expected_diffusion_score"].notna()
    ].copy()
    validation = frame.loc[
        frame["publication_year"].between(cutoff + 1, validation_end)
    ].copy()
    prior = frame.loc[
        frame["publication_year"].between(seed_start - seed_window, seed_start - 1)
    ].copy()
    seed_group = (
        seed.groupby("display_topic_label")
        .agg(
            seed_n=("paper_id", "size"),
            prediction_score=("expected_diffusion_score", "mean"),
            k1_score=("k1_expected_diffusion_score", "mean"),
            prior_popularity=("log_prior_reference_popularity_median", "mean"),
            seed_target_mean=("realized_diffusion_target", "mean"),
        )
        .reset_index()
    )
    validation_group = (
        validation.groupby("display_topic_label")
        .agg(
            validation_n=("paper_id", "size"),
            validation_target_mean=("realized_diffusion_target", "mean"),
            validation_high_share=("realized_high_diffusion", "mean"),
        )
        .reset_index()
    )
    prior_group = (
        prior.groupby("display_topic_label").size().rename("prior_n").reset_index()
    )
    output = seed_group.merge(validation_group, on="display_topic_label", how="left")
    output = output.merge(prior_group, on="display_topic_label", how="left")
    output["validation_n"] = numeric(output["validation_n"]).fillna(0).astype(int)
    output["prior_n"] = numeric(output["prior_n"]).fillna(0).astype(int)
    output = output.loc[
        output["seed_n"].ge(int(minimum_seed))
        & output["validation_n"].ge(int(minimum_validation))
    ].copy()
    output["historical_growth"] = np.log1p(output["seed_n"]) - np.log1p(output["prior_n"])
    output["realized_frontier_score"] = (
        percentile_rank(output["validation_n"], output["display_topic_label"])
        + percentile_rank(
            output["validation_target_mean"],
            output["display_topic_label"],
        )
        + percentile_rank(
            output["validation_high_share"],
            output["display_topic_label"],
        )
    ) / 3.0
    for column in [
        "prediction_score",
        "k1_score",
        "historical_growth",
        "prior_popularity",
        "realized_frontier_score",
    ]:
        values = numeric(output[column])
        if values.notna().any():
            fill_value = float(values.min()) - max(float(values.std(ddof=0)), 1e-6)
        else:
            fill_value = 0.0
        output[column] = values.fillna(fill_value)
        output[f"{column}_rank"] = (
            output[column].rank(method="first", ascending=False).astype(int)
        )
    output["cutoff"] = int(cutoff)
    output["seed_window"] = f"{seed_start}–{cutoff}"
    output["validation_window"] = f"{cutoff + 1}–{validation_end}"
    return output


def _frontier_metrics(
    topics: pd.DataFrame,
    top_n: int,
    seed: int,
) -> pd.DataFrame:
    """Evaluate topic rankings against one realized future-frontier ranking."""
    rows: List[Dict[str, Any]] = []
    score_columns = {
        "ASPR temporal OOF": "prediction_score",
        "K1 control-only": "k1_score",
        "Historical growth": "historical_growth",
        "Publication-prior popularity": "prior_popularity",
    }
    for cutoff, group in topics.groupby("cutoff"):
        truth_order = group.nsmallest(int(top_n), "realized_frontier_score_rank")
        truth_set = set(truth_order["display_topic_label"])
        relevance = group["realized_frontier_score"].to_numpy(float)
        for method, column in score_columns.items():
            prediction = group[column].to_numpy(float)
            predicted = group.nlargest(int(top_n), column)
            hit_count = len(set(predicted["display_topic_label"]) & truth_set)
            rows.append(
                {
                    "cutoff": int(cutoff),
                    "method": method,
                    "precision_at_10": hit_count / max(int(top_n), 1),
                    "ndcg_at_10": float(
                        ndcg_score(
                            relevance.reshape(1, -1),
                            prediction.reshape(1, -1),
                            k=int(top_n),
                        )
                    ),
                    "frontier_coverage": hit_count / max(len(truth_set), 1),
                    "topic_count": len(group),
                }
            )
        rng = np.random.default_rng(stable_seed(str(cutoff), seed))
        random_values = rng.random(len(group))
        predicted_index = np.argsort(-random_values)[: int(top_n)]
        hit_count = len(
            set(group.iloc[predicted_index]["display_topic_label"]) & truth_set
        )
        rows.append(
            {
                "cutoff": int(cutoff),
                "method": "Random",
                "precision_at_10": hit_count / max(int(top_n), 1),
                "ndcg_at_10": float(
                    ndcg_score(
                        relevance.reshape(1, -1),
                        random_values.reshape(1, -1),
                        k=int(top_n),
                    )
                ),
                "frontier_coverage": hit_count / max(len(truth_set), 1),
                "topic_count": len(group),
            }
        )
    return pd.DataFrame(rows)


def _topic_landscape(topics: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    """Create deterministic two-dimensional topic-label coordinates."""
    frame = topics.loc[topics["cutoff"].eq(int(cutoff))].copy()
    if len(frame) < 3:
        frame["x"] = np.arange(len(frame), dtype=float)
        frame["y"] = 0.0
        return frame
    vectors = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform(
        frame["display_topic_label"].fillna("unknown").astype(str)
    )
    similarity = (vectors @ vectors.T).toarray()
    distance = np.clip(1.0 - similarity, 0.0, 1.0)
    coordinates = MDS(
        n_components=2,
        metric="precomputed",
        init="random",
        random_state=20260725,
        n_init=4,
        max_iter=400,
        normalized_stress="auto",
    ).fit_transform(distance)
    frame["x"] = coordinates[:, 0]
    frame["y"] = coordinates[:, 1]
    predicted = set(frame.nsmallest(10, "prediction_score_rank")["display_topic_label"])
    realized = set(frame.nsmallest(10, "realized_frontier_score_rank")["display_topic_label"])
    frame["classification"] = [
        "hit" if label in predicted and label in realized
        else "false_positive" if label in predicted
        else "miss" if label in realized
        else "background"
        for label in frame["display_topic_label"]
    ]
    return frame


def _frontier_seed_cards(
    paper_frame: pd.DataFrame,
    topics: pd.DataFrame,
    cutoff: int,
    top_count: int = 4,
) -> pd.DataFrame:
    """Select locked-by-score seed papers and attach five-angle percentiles."""
    top_topics = topics.loc[topics["cutoff"].eq(int(cutoff))].nsmallest(
        int(top_count),
        "prediction_score_rank",
    )
    candidates = paper_frame.loc[
        paper_frame["publication_year"].between(cutoff - 2, cutoff)
        & paper_frame["display_topic_label"].isin(top_topics["display_topic_label"])
    ].copy()
    candidates = candidates.sort_values(
        ["expected_diffusion_score", "paper_id"],
        ascending=[False, True],
    )
    selected = candidates.groupby("display_topic_label", as_index=False).head(1).head(top_count)
    angle_features = {
        "A1_COMBINATION_RARITY": ["reference_overlap_novelty_t0"],
        "A2_ATYPICALITY_CONVENTIONALITY": ["hypergeom_conventionality_median_t0"],
        "A3_FIRST_TIME_COMBINATION": ["first_time_source_pair_share"],
        "A4_KNOWLEDGE_BREADTH_BALANCE": [
            "field_gini_balance",
            "reference_other_field_share",
            "field_variety",
        ],
        "A5_COGNITIVE_DISTANCE_INTEGRATION": [
            "field_disparity_cosine_mean",
            "rao_stirling_integration",
        ],
    }
    for feature in PRIMARY_FEATURES:
        paper_frame[f"{feature}__pct"] = grouped_percentile(
            paper_frame,
            feature,
            ["domain12", "publication_year"],
            id_column="paper_id",
        )
    selected = selected.drop(
        columns=[column for column in selected if column.endswith("__pct")],
        errors="ignore",
    ).merge(
        paper_frame[["paper_id", *[f"{f}__pct" for f in PRIMARY_FEATURES]]],
        on="paper_id",
        how="left",
    )
    for angle, features in angle_features.items():
        selected[angle] = selected[[f"{feature}__pct" for feature in features]].mean(axis=1)
    keep = [
        "paper_id",
        "title",
        "publication_year",
        "display_topic_label",
        "expected_diffusion_score",
        *ANGLE_ORDER,
    ]
    return selected[keep]


def build_fig5(
    config: Mapping[str, Any],
    paths: SuitePaths,
) -> FigureBundle:
    """Build strict historical topic-frontier backtests from temporal OOF scores."""
    resolved = _v61_paths(paths)
    oof = pd.read_parquet(
        resolved["oof"],
        columns=[
            "paper_id",
            "publication_year",
            "domain12",
            "horizon",
            "model_id",
            "expected_diffusion_score",
            "realized_diffusion_target",
        ],
    )
    final = oof.loc[
        oof["horizon"].eq(5)
        & oof["model_id"].eq("final_innovation_plus_k1")
    ].drop_duplicates("paper_id")
    k1 = oof.loc[
        oof["horizon"].eq(5) & oof["model_id"].eq("k1_controls"),
        ["paper_id", "expected_diffusion_score"],
    ].rename(columns={"expected_diffusion_score": "k1_expected_diffusion_score"})
    papers = pd.read_parquet(
        resolved["papers"],
        columns=[
            "paper_id",
            "publication_year",
            "domain12",
            "display_topic_label",
            "venue_family",
        ],
    )
    controls = pd.read_parquet(
        resolved["controls"],
        columns=["paper_id", "log_prior_reference_popularity_median"],
    )
    features = pd.read_parquet(
        resolved["features"],
        columns=["paper_id", *PRIMARY_FEATURES],
    )
    titles = pd.read_csv(
        paths["target_works"],
        usecols=["id", "title"],
    ).rename(columns={"id": "paper_id"})
    frame = (
        final.merge(k1, on="paper_id", how="left")
        .merge(
            papers.drop(columns=["publication_year", "domain12"]),
            on="paper_id",
            how="left",
        )
        .merge(controls, on="paper_id", how="left")
        .merge(features, on="paper_id", how="left")
        .merge(titles, on="paper_id", how="left")
    )
    frame["realized_high_diffusion"] = percentile_rank(
        frame["realized_diffusion_target"],
        frame["paper_id"],
    ).ge(0.90).astype(int)
    topic_rows = []
    for cutoff in config["fig5"]["cutoffs"]:
        topic_rows.append(
            _topic_scores_for_cutoff(
                frame,
                int(cutoff),
                int(config["fig5"]["seed_window_years"]),
                int(config["fig5"]["validation_window_years"]),
                int(config["fig5"]["minimum_topic_seed_papers"]),
                int(config["fig5"]["minimum_topic_validation_papers"]),
            )
        )
    topics = pd.concat(topic_rows, ignore_index=True)
    metrics = _frontier_metrics(
        topics,
        int(config["fig5"]["top_n"]),
        int(config["fig5"]["seed"]),
    )
    main_cutoff = max(int(value) for value in config["fig5"]["cutoffs"])
    landscape = _topic_landscape(topics, main_cutoff)
    bump = topics.loc[
        topics["prediction_score_rank"].le(int(config["fig5"]["top_n"]))
        | topics["realized_frontier_score_rank"].le(int(config["fig5"]["top_n"]))
    ].copy()
    cards = _frontier_seed_cards(frame, topics, main_cutoff)
    windows = pd.DataFrame(
        [
            {
                "cutoff": int(cutoff),
                "training_end": int(cutoff) - 1,
                "prediction_start": int(cutoff) - int(config["fig5"]["seed_window_years"]) + 1,
                "prediction_end": int(cutoff),
                "validation_start": int(cutoff) + 1,
                "validation_end": int(cutoff) + int(config["fig5"]["validation_window_years"]),
            }
            for cutoff in config["fig5"]["cutoffs"]
        ]
    )
    tables = {
        "historical_windows": windows,
        "topic_backtest_scores": topics,
        "topic_landscape": landscape,
        "rank_bump_topics": bump,
        "seed_cards": cards,
        "backtest_metrics": metrics,
    }
    aspr = metrics.loc[metrics["method"].eq("ASPR temporal OOF")]
    panel_text = {
        "a": "Each paper score is generated by a temporal fold whose training window ends before the paper.",
        "b": {
            "cutoff": main_cutoff,
            "topic_count": int(len(landscape)),
            "classification": "Hit, false-positive and miss are defined only from top-10 topic ranks.",
        },
        "c": "Predicted and realized top-10 ranks are connected for topics entering either list.",
        "d": "Four seed cards are selected by the locked historical prediction, never by later realized outcome.",
        "e": {
            "cutoff_count": int(aspr["cutoff"].nunique()),
            "mean_precision_at_10": float(aspr["precision_at_10"].mean()),
            "mean_ndcg_at_10": float(aspr["ndcg_at_10"].mean()),
            "warning": config["claim_boundaries"]["fig5"],
        },
    }
    contract = {
        "figure_id": 5,
        "panels": {
            "a": {"mark": "historical time bands", "data": ["historical_windows"]},
            "b": {"mark": "topic landscape", "data": ["topic_landscape"]},
            "c": {"mark": "bump chart", "data": ["rank_bump_topics"]},
            "d": {"mark": "seed cards", "data": ["seed_cards"]},
            "e": {"mark": "paired cutoff estimates", "data": ["backtest_metrics"]},
        },
        "future_trained_scores_allowed": False,
        "current_2024_2026_frontier_is_validation": False,
    }
    return FigureBundle(
        figure_id=5,
        title="Strict historical backtests test early frontier identification",
        status="complete_historical_backtest",
        tables=tables,
        panel_text=panel_text,
        chart_contract=contract,
        source_paths=[
            resolved["oof"],
            resolved["papers"],
            resolved["controls"],
            resolved["features"],
            paths["target_works"],
        ],
        notes=[config["claim_boundaries"]["fig5"]],
    )

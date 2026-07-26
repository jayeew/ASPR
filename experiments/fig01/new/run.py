"""Render the canonical Fig.1 with current v6.1 indicators.

Only historical drawing primitives for the four graph snapshots are reused.
No historical perturbation metric is computed, loaded, renamed, or plotted.
The right-hand panels are built from the frozen v6.1 primary feature table and
the frozen five-angle/eight-indicator contract.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from experiments.fig01.old import (
    fig1_knowledge_perturbation as snapshot_renderer,
)
from experiments.common.new.adapters.contracts import (
    ANGLE_FEATURES,
    FEATURE_DIRECTION,
    PRIMARY_FEATURES,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(__file__).with_name("config.json")
FORMULA_SOURCE_PATHS: Tuple[str, ...] = (
    "aspr/nature_multihorizon/features_v6.py",
    "aspr/nature_multihorizon/features_v6_1.py",
    "aspr/nature_multihorizon/materialize_v6_1.py",
    "experiments/common/new/adapters/contracts.py",
)
FEATURE_LABELS: Dict[str, str] = {
    "reference_overlap_novelty_t0": "Reference-overlap novelty",
    "hypergeom_conventionality_median_t0": "Low conventionality",
    "first_time_source_pair_share": "First-time source pairs",
    "field_gini_balance": "Field balance (1−Gini)",
    "reference_other_field_share": "Out-of-field references",
    "field_variety": "Field variety",
    "field_disparity_cosine_mean": "Mean cognitive distance",
    "rao_stirling_integration": "Rao–Stirling integration",
}
FEATURE_COLORS: Dict[str, str] = {
    "reference_overlap_novelty_t0": "#2563EB",
    "hypergeom_conventionality_median_t0": "#7C3AED",
    "first_time_source_pair_share": "#D97706",
    "field_gini_balance": "#059669",
    "reference_other_field_share": "#0F766E",
    "field_variety": "#DB2777",
    "field_disparity_cosine_mean": "#DC2626",
    "rao_stirling_integration": "#475569",
}
FEATURE_MARKERS: Dict[str, str] = {
    "reference_overlap_novelty_t0": "o",
    "hypergeom_conventionality_median_t0": "s",
    "first_time_source_pair_share": "^",
    "field_gini_balance": "D",
    "reference_other_field_share": "v",
    "field_variety": "P",
    "field_disparity_cosine_mean": "X",
    "rao_stirling_integration": "h",
}


@dataclass
class FigureDomain:
    """Graph and time-window data needed for one visual row."""

    cfg: Dict[str, Any]
    graph: nx.Graph
    display_comm_map: Dict[str, int]
    display_labels: Dict[int, str]
    positions: Dict[int, np.ndarray]
    colors: Dict[int, Any]
    rolling_windows: List[Tuple[int, int]]
    cumulative_windows: List[Tuple[int, int]]
    feature_domain12: str
    feature_subfields: Tuple[str, ...]
    selected_communities: Tuple[int, ...]


def _resolve(path_value: str) -> Path:
    """Resolve a repository-relative or absolute path."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write deterministic, human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _optional_int(value: Any) -> int | None:
    """Convert a scalar to int while preserving missing values."""
    if pd.isna(value):
        return None
    return int(float(value))


def _optional_text(value: Any) -> str:
    """Convert a scalar to text while preserving CSV missing values."""
    return "" if pd.isna(value) else str(value)


def _node_attributes(row: Any) -> Dict[str, Any]:
    """Build the graph-node schema required by the snapshot renderer."""
    anchor_label = _optional_text(getattr(row, "anchor_label", ""))
    year = _optional_int(getattr(row, "year", None))
    return {
        "title": _optional_text(getattr(row, "title", "")),
        "year": year,
        "cited_by_count": int(float(getattr(row, "cited_by_count", 0) or 0)),
        "primary_topic": _optional_text(getattr(row, "primary_topic", "")),
        "topics": [],
        "anchor_label": anchor_label,
        "anchor_year": year if anchor_label else None,
        "anchor_citer": bool(
            int(float(getattr(row, "anchor_citer", 0) or 0))
        ),
        "reference_stub": bool(
            int(float(getattr(row, "reference_stub", 0) or 0))
        ),
    }


def _build_graph(works: pd.DataFrame, edges: pd.DataFrame) -> nx.Graph:
    """Reconstruct a selected paper graph without recomputing its edges."""
    graph = nx.Graph()
    for row in works.itertuples(index=False):
        graph.add_node(str(row.id), **_node_attributes(row))
    for row in edges.itertuples(index=False):
        source, target = str(row.source), str(row.target)
        if source not in graph or target not in graph:
            continue
        graph.add_edge(
            source,
            target,
            weight=float(row.weight),
            direct=int(row.direct),
            bibliographic=int(row.bibliographic),
            cocitation=int(row.cocitation),
        )
    return graph


def _sorted_graph(graph: nx.Graph) -> nx.Graph:
    """Copy a graph with stable node and edge insertion order."""
    stable = nx.Graph()
    stable.graph.update(graph.graph)
    for node in sorted(graph.nodes(), key=str):
        stable.add_node(node, **dict(graph.nodes[node]))
    for source, target, attributes in sorted(
        graph.edges(data=True),
        key=lambda item: tuple(sorted((str(item[0]), str(item[1])))),
    ):
        stable.add_edge(source, target, **dict(attributes))
    return stable


def _community_maps(
    works: pd.DataFrame,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Read the frozen display communities used by the current graph corpus."""
    display_map: Dict[str, int] = {}
    display_labels: Dict[int, str] = {}
    for row in works.itertuples(index=False):
        community = _optional_int(row.display_community)
        if community is None:
            continue
        display_map[str(row.id)] = community
        display_labels.setdefault(
            community,
            _optional_text(getattr(row, "display_label", "")),
        )
    return display_map, display_labels


def _apply_domain_windows(
    cfg: Dict[str, Any],
    domain_spec: Mapping[str, Any],
) -> Dict[str, Any]:
    """Apply the current, feature-horizon-aligned graph windows."""
    raw_windows = domain_spec.get("graph_windows")
    if not raw_windows:
        return cfg
    windows = [
        [int(start), int(end)]
        for start, end in raw_windows
    ]
    overrides: Dict[str, Any] = {
        "start_year": windows[0][0],
        "end_year": windows[-1][1],
        "custom_windows": windows,
        "snapshot_years": [end for _, end in windows],
    }
    captions = domain_spec.get("panel_captions")
    if captions:
        overrides["plot"] = {"panel_captions": list(captions)}
    updated = snapshot_renderer.deep_update(cfg, overrides)
    snapshot_renderer.validate_time_windows(updated)
    return updated


def _select_display_communities(
    works: pd.DataFrame,
    windows: Sequence[Tuple[int, int]],
    maximum_topics: int,
) -> Tuple[int, ...]:
    """Select a small, temporally representative set of graph communities."""
    frame = works.copy()
    frame["_year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame["_community"] = frame["display_community"].map(_optional_int)
    frame = frame[
        frame["_year"].notna()
        & frame["_community"].notna()
        & frame["_year"].le(int(windows[-1][1]))
    ]
    if frame.empty:
        return ()

    selected: List[int] = []

    def add(communities: Iterable[Any]) -> None:
        for value in communities:
            community = int(value)
            if community not in selected:
                selected.append(community)

    anchors = frame[frame["anchor_label"].notna()]
    add(
        anchors.sort_values(
            ["_year", "_community"],
            kind="mergesort",
        )["_community"]
    )
    for start, end in windows:
        period = frame[frame["_year"].between(int(start), int(end))]
        counts = (
            period.groupby("_community", observed=True)
            .size()
            .rename("n")
            .reset_index()
            .sort_values(
                ["n", "_community"],
                ascending=[False, True],
                kind="mergesort",
            )
        )
        add(counts["_community"].head(1))
    overall = (
        frame.groupby("_community", observed=True)
        .size()
        .rename("n")
        .reset_index()
        .sort_values(
            ["n", "_community"],
            ascending=[False, True],
            kind="mergesort",
        )
    )
    add(overall["_community"])
    keep = max(int(maximum_topics), len(set(anchors["_community"])))
    return tuple(selected[:keep])


def _source_paths(frozen_root: Path, slug: str) -> Dict[str, Path]:
    """Return graph files used for one visual row."""
    graph_dir = frozen_root / slug
    return {
        "works": graph_dir / "works_selected.csv",
        "edges": graph_dir / "paper_edges.csv",
        "topic_nodes": graph_dir / "topic_nodes.csv",
        "topic_edges": graph_dir / "topic_edges.csv",
    }


def _load_domain(
    domain_spec: Mapping[str, Any],
    frozen_root: Path,
    maximum_topics: int,
) -> Tuple[FigureDomain, Dict[str, Any]]:
    """Load one graph row without invoking any historical metric formula."""
    slug = str(domain_spec["slug"])
    cfg = snapshot_renderer.load_config(_resolve(str(domain_spec["config"])))
    cfg = _apply_domain_windows(cfg, domain_spec)
    paths = _source_paths(frozen_root, slug)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing graph sources for {slug}: {missing}")

    works = pd.read_csv(paths["works"], low_memory=False)
    edges = pd.read_csv(paths["edges"], low_memory=False)
    graph = _build_graph(works, edges)
    all_display_map, all_display_labels = _community_maps(works)
    rolling = snapshot_renderer.make_rolling_windows_from_config(cfg)
    selected_communities = _select_display_communities(
        works,
        rolling,
        maximum_topics,
    )
    selected_set = set(selected_communities)
    display_map = {
        paper_id: community
        for paper_id, community in all_display_map.items()
        if community in selected_set
    }
    display_labels = {
        community: all_display_labels[community]
        for community in selected_communities
        if community in all_display_labels
    }
    topic_graph = _sorted_graph(
        snapshot_renderer.make_topic_graph(
            graph,
            display_map,
            display_labels,
            set(graph.nodes()),
        )
    )
    positions = snapshot_renderer.layout_topic_graph(topic_graph, cfg)
    colors = snapshot_renderer.community_color_map(list(topic_graph.nodes()))
    cumulative = snapshot_renderer.make_cumulative_windows_from_config(
        cfg, rolling
    )
    result = FigureDomain(
        cfg=cfg,
        graph=graph,
        display_comm_map=display_map,
        display_labels=display_labels,
        positions=positions,
        colors=colors,
        rolling_windows=rolling,
        cumulative_windows=cumulative,
        feature_domain12=str(domain_spec["feature_domain12"]),
        feature_subfields=tuple(domain_spec["feature_subfields"]),
        selected_communities=selected_communities,
    )
    endpoints = set(edges["source"].astype(str)) | set(
        edges["target"].astype(str)
    )
    node_ids = set(works["id"].astype(str))
    report = {
        "slug": slug,
        "papers": int(len(works)),
        "paper_edges": int(len(edges)),
        "rolling_windows": [list(item) for item in rolling],
        "cumulative_windows": [list(item) for item in cumulative],
        "edge_endpoints_missing_from_nodes": int(len(endpoints - node_ids)),
        "display_communities_missing_positions": int(
            len(set(display_map.values()) - set(positions))
        ),
        "selected_display_communities": list(selected_communities),
        "selected_display_topic_count": len(selected_communities),
        "snapshot_display_topic_counts": [
            int(
                snapshot_renderer.make_topic_graph(
                    graph,
                    display_map,
                    display_labels,
                    _node_set_until_year(graph, end),
                ).number_of_nodes()
            )
            for _, end in cumulative
        ],
        "feature_domain12": result.feature_domain12,
        "feature_subfields": list(result.feature_subfields),
        "source_files": {
            key: {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for key, path in paths.items()
        },
    }
    return result, report


def _load_registry(
    registry_path: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load and validate the frozen primary-indicator evidence records."""
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    for feature in PRIMARY_FEATURES:
        matches = [
            candidate
            for candidate in registry["candidates"].values()
            if candidate.get("code_name") == feature
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one registry row for {feature}, found {len(matches)}"
            )
        candidate = matches[0]
        if candidate.get("final_role") != "primary":
            raise ValueError(f"{feature} is not frozen as a primary indicator")
        rows.append(
            {
                "feature": feature,
                "angle": candidate["angle_id"],
                "formula": candidate["formula"],
                "registry_direction": candidate["direction"],
                "visual_direction": FEATURE_DIRECTION[feature],
                "candidate_id": candidate["candidate_id"],
                "final_role": candidate["final_role"],
                "maximum_information_time": candidate[
                    "maximum_information_time"
                ],
            }
        )
    return pd.DataFrame(rows), registry


def _load_current_features(
    paper_path: Path,
    feature_path: Path,
) -> pd.DataFrame:
    """Load the already-materialized v6.1 primary indicators."""
    paper_columns = [
        "paper_id",
        "publication_year",
        "domain12",
        "openalex_primary_subfield",
    ]
    feature_columns = [
        "paper_id",
        "source_max_year",
        "definition_version",
        *PRIMARY_FEATURES,
    ]
    papers = pd.read_parquet(paper_path, columns=paper_columns)
    features = pd.read_parquet(feature_path, columns=feature_columns)
    merged = papers.merge(
        features, on="paper_id", how="inner", validate="one_to_one"
    )
    if tuple(PRIMARY_FEATURES) != tuple(ANGLE_FEATURES_TO_FLAT()):
        raise ValueError("Five-angle mapping no longer matches PRIMARY_FEATURES")
    source_max = pd.to_numeric(merged["source_max_year"], errors="coerce")
    years = pd.to_numeric(merged["publication_year"], errors="coerce")
    if source_max.notna().any() and source_max[source_max.notna()].ge(
        years[source_max.notna()]
    ).any():
        raise ValueError("Current indicator table contains time leakage")
    return merged


def ANGLE_FEATURES_TO_FLAT() -> Tuple[str, ...]:
    """Flatten the frozen angle mapping in presentation order."""
    return tuple(
        feature
        for features in ANGLE_FEATURES.values()
        for feature in features
    )


def _add_oriented_percentiles(features: pd.DataFrame) -> pd.DataFrame:
    """Add visualization-only within-domain12/year oriented percentiles."""
    output = features.copy()
    groups = [output["domain12"], output["publication_year"]]
    for feature in PRIMARY_FEATURES:
        ascending = FEATURE_DIRECTION[feature] > 0
        output[f"{feature}__oriented_pct"] = output.groupby(
            groups,
            observed=True,
        )[feature].rank(
            method="average",
            pct=True,
            ascending=ascending,
        )
    return output


def _matched_feature_rows(
    domains: Sequence[FigureDomain],
    all_features: pd.DataFrame,
) -> pd.DataFrame:
    """Select broad-domain/subfield feature scopes for the four visual rows."""
    frames: List[pd.DataFrame] = []
    for result in domains:
        mask = all_features["domain12"].eq(result.feature_domain12)
        if result.feature_subfields:
            mask &= all_features["openalex_primary_subfield"].isin(
                result.feature_subfields
            )
        selected = all_features[mask].copy()
        selected.insert(0, "visual_domain", result.cfg["slug"])
        frames.append(selected)
    return pd.concat(frames, ignore_index=True)


def _window_feature_summary(
    matched: pd.DataFrame,
    domains: Sequence[FigureDomain],
    horizon_end: int,
) -> pd.DataFrame:
    """Summarize every current indicator in the configured plot windows."""
    rows: List[Dict[str, Any]] = []
    for result in domains:
        domain_rows = matched[
            matched["visual_domain"].eq(result.cfg["slug"])
        ]
        for window_index, (start, end) in enumerate(
            result.rolling_windows, start=1
        ):
            observed_end = min(int(end), int(horizon_end))
            selected = domain_rows[
                domain_rows["publication_year"].between(start, observed_end)
            ] if int(start) <= observed_end else domain_rows.iloc[0:0]
            for feature in PRIMARY_FEATURES:
                values = pd.to_numeric(selected[feature], errors="coerce")
                percentiles = pd.to_numeric(
                    selected[f"{feature}__oriented_pct"],
                    errors="coerce",
                )
                rows.append(
                    {
                        "domain": result.cfg["slug"],
                        "window_index": window_index,
                        "requested_start": int(start),
                        "requested_end": int(end),
                        "observed_end": observed_end
                        if int(start) <= observed_end
                        else np.nan,
                        "right_truncated": bool(int(end) > int(horizon_end)),
                        "feature": feature,
                        "angle": next(
                            angle
                            for angle, members in ANGLE_FEATURES.items()
                            if feature in members
                        ),
                        "n_scope_papers": int(len(selected)),
                        "n_valid": int(values.notna().sum()),
                        "coverage": float(values.notna().mean())
                        if len(selected)
                        else np.nan,
                        "raw_median": float(values.median())
                        if values.notna().any()
                        else np.nan,
                        "oriented_percentile_median": float(
                            percentiles.median()
                        )
                        if percentiles.notna().any()
                        else np.nan,
                        "oriented_percentile_q25": float(
                            percentiles.quantile(0.25)
                        )
                        if percentiles.notna().any()
                        else np.nan,
                        "oriented_percentile_q75": float(
                            percentiles.quantile(0.75)
                        )
                        if percentiles.notna().any()
                        else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _select_indicator_trajectories(
    indicator_summary: pd.DataFrame,
    minimum_sample: int,
    minimum_selected: int,
    maximum_selected: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Select indicators using only their across-window median movement."""
    selection_rows: List[Dict[str, Any]] = []
    for domain, domain_rows in indicator_summary.groupby(
        "domain",
        sort=False,
    ):
        candidates: List[Dict[str, Any]] = []
        for feature, feature_rows in domain_rows.groupby(
            "feature",
            sort=False,
        ):
            ordered = feature_rows.sort_values("window_index")
            medians = ordered["oriented_percentile_median"].to_numpy(
                dtype=float
            )
            counts = ordered["n_valid"].to_numpy(dtype=int)
            complete = (
                len(medians) == 4
                and np.isfinite(medians).all()
                and (counts >= int(minimum_sample)).all()
            )
            candidates.append(
                {
                    "domain": str(domain),
                    "feature": str(feature),
                    "feature_label": FEATURE_LABELS[str(feature)],
                    "eligible": bool(complete),
                    "minimum_n_valid": int(counts.min())
                    if len(counts)
                    else 0,
                    "change_range": float(np.ptp(medians))
                    if np.isfinite(medians).all()
                    else np.nan,
                    "maximum_adjacent_change": float(
                        np.max(np.abs(np.diff(medians)))
                    )
                    if np.isfinite(medians).all() and len(medians) > 1
                    else np.nan,
                }
            )
        ranked = sorted(
            (row for row in candidates if row["eligible"]),
            key=lambda row: (
                -float(row["change_range"]),
                -float(row["maximum_adjacent_change"]),
                str(row["feature"]),
            ),
        )
        if len(ranked) < int(minimum_selected):
            raise ValueError(
                f"{domain} has only {len(ranked)} complete indicators; "
                f"{minimum_selected} required"
            )
        lower = min(int(minimum_selected), len(ranked))
        upper = min(int(maximum_selected), len(ranked))
        if lower == upper or len(ranked) <= upper:
            selected_count = upper
        else:
            gaps = {
                count: float(ranked[count - 1]["change_range"])
                - float(ranked[count]["change_range"])
                for count in range(lower, upper + 1)
            }
            selected_count = max(
                gaps,
                key=lambda count: (gaps[count], -count),
            )
        rank_by_feature = {
            str(row["feature"]): rank
            for rank, row in enumerate(ranked, start=1)
        }
        selected_features = {
            str(row["feature"]) for row in ranked[:selected_count]
        }
        for row in candidates:
            feature = str(row["feature"])
            row["rank_by_change"] = rank_by_feature.get(feature)
            row["selected"] = feature in selected_features
            row["selected_count_for_domain"] = selected_count
            row["selection_rule"] = (
                "complete_in_all_four_windows_and_largest_range_elbow_4_to_5"
            )
            selection_rows.append(row)
    selection = pd.DataFrame(selection_rows)
    trajectories = indicator_summary.merge(
        selection[
            [
                "domain",
                "feature",
                "feature_label",
                "eligible",
                "selected",
                "rank_by_change",
                "change_range",
                "maximum_adjacent_change",
                "minimum_n_valid",
            ]
        ],
        on=["domain", "feature"],
        how="left",
        validate="many_to_one",
    )
    return trajectories, selection


def _snapshot_draw_config(
    cfg: Mapping[str, Any],
    compact: bool,
) -> Dict[str, Any]:
    """Apply sparse display overrides while preserving the visual grammar."""
    plot = cfg.get("plot", {})
    max_labels = 3 if compact else 5
    max_edges = 7 if compact else 9
    extra_edges = 1 if compact else 2
    max_papers = 4 if compact else 5
    node_size = 36 if compact else 48
    return snapshot_renderer.deep_update(
        dict(cfg),
        {
            "plot": {
                "show_internal_cluster_edges": False,
                "max_representative_papers": min(
                    int(plot.get("max_representative_papers", 6)),
                    max_papers,
                ),
                "max_labels_per_panel": min(
                    int(plot.get("max_labels_per_panel", 8)),
                    max_labels,
                ),
                "display_max_backbone_edges": min(
                    int(plot.get("display_max_backbone_edges", 15)),
                    max_edges,
                ),
                "display_extra_edges": min(
                    int(plot.get("display_extra_edges", 6)),
                    extra_edges,
                ),
                "cluster_radius_min": min(
                    float(plot.get("cluster_radius_min", 0.13)),
                    0.095 if compact else 0.11,
                ),
                "cluster_radius_max": min(
                    float(plot.get("cluster_radius_max", 0.24)),
                    0.16 if compact else 0.19,
                ),
                "node_size_min": min(
                    float(plot.get("node_size_min", 64)),
                    node_size,
                ),
                "context_topic_alpha": 0.25 if compact else 0.34,
                "context_edge_alpha": 0.14 if compact else 0.18,
                "foreground_edge_alpha": 0.78,
                "landmark_edge_alpha": 0.65,
            }
        },
    )


def _window_label(row: pd.Series, horizon_end: int) -> str:
    """Format a requested window and disclose right truncation."""
    start, end = int(row["requested_start"]), int(row["requested_end"])
    if start > horizon_end:
        return f"{start}–{end}\n(out of scope)"
    if end > horizon_end:
        return f"{start}–{horizon_end}†"
    return f"{start}–{end}"


def _draw_indicator_panel(
    axis: plt.Axes,
    result: FigureDomain,
    trajectories: pd.DataFrame,
    horizon_end: int,
    compact: bool,
) -> None:
    """Draw the locally selected high-change indicator trajectories."""
    subset = trajectories[
        trajectories["domain"].eq(result.cfg["slug"])
        & trajectories["selected"].fillna(False)
    ].copy()
    selected_features = (
        subset[["feature", "rank_by_change"]]
        .drop_duplicates()
        .sort_values("rank_by_change")["feature"]
        .tolist()
    )
    window_rows = (
        subset.sort_values("window_index")
        .drop_duplicates("window_index")
        .set_index("window_index")
    )
    x_values = np.arange(1, len(result.rolling_windows) + 1)
    landmark_year = int(result.cfg["plot"]["landmark_focus_year"])
    event_index = next(
        (
            index
            for index, (start, end) in enumerate(
                result.rolling_windows, start=1
            )
            if int(start) <= landmark_year <= int(end)
        ),
        None,
    )
    if event_index is not None:
        axis.axvspan(
            event_index - 0.42,
            event_index + 0.42,
            color="#FEF3C7",
            alpha=0.52,
            zorder=0,
        )
        axis.text(
            event_index,
            0.985,
            f"landmark {landmark_year}",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6.2,
            color="#92400E",
        )

    for feature in selected_features:
        feature_rows = (
            subset[subset["feature"].eq(feature)]
            .sort_values("window_index")
            .set_index("window_index")
            .reindex(range(1, len(result.rolling_windows) + 1))
        )
        medians = feature_rows[
            "oriented_percentile_median"
        ].to_numpy(dtype=float)
        valid = np.isfinite(medians)
        if valid.any():
            axis.plot(
                x_values[valid],
                medians[valid],
                color=FEATURE_COLORS[feature],
                marker=FEATURE_MARKERS[feature],
                markersize=4.2 if compact else 5.4,
                linewidth=1.45 if compact else 1.8,
                label=FEATURE_LABELS[feature],
                zorder=3,
            )

    minimum_valid_by_window = (
        subset.groupby("window_index", sort=True)["n_valid"]
        .min()
        .reindex(range(1, len(result.rolling_windows) + 1), fill_value=0)
    )
    labels = [
        _window_label(window_rows.loc[index], horizon_end)
        for index in range(1, len(result.rolling_windows) + 1)
    ]
    axis.set_xticks(x_values)
    axis.set_xticklabels(labels, fontsize=6.4)
    axis.set_xlim(0.65, len(x_values) + 0.35)
    selected_values = pd.to_numeric(
        subset["oriented_percentile_median"],
        errors="coerce",
    ).dropna()
    low = min(0.25, float(selected_values.min()) - 0.03)
    high = max(0.75, float(selected_values.max()) + 0.03)
    axis.set_ylim(max(0.0, low), min(1.0, high))
    axis.tick_params(axis="y", labelsize=6.3 if compact else 7.2)
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_ylabel(
        "Median innovation-oriented percentile\nwithin domain12 × year",
        fontsize=6.8 if compact else 8.2,
    )
    axis.set_title(
        f"Selected indicators: {len(selected_features)} largest shifts",
        fontsize=8.2 if compact else 10.0,
        fontweight="bold",
        pad=14 if compact else 12,
    )
    for x_value, count in zip(x_values, minimum_valid_by_window):
        axis.text(
            x_value,
            axis.get_ylim()[0] + 0.012 * np.ptp(axis.get_ylim()),
            f"n≥{int(count):,}",
            ha="center",
            va="bottom",
            fontsize=5.7 if compact else 6.8,
            color="#6B7280",
        )
    axis.legend(
        loc="upper right",
        ncol=2,
        fontsize=5.7 if compact else 7.2,
        frameon=True,
        framealpha=0.88,
        edgecolor="none",
        handlelength=1.7,
        columnspacing=1.0,
    )
    axis.text(
        0.0,
        1.015,
        result.feature_domain12,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.5 if compact else 6.5,
        color="#6B7280",
    )
    axis.text(
        1.0,
        1.015,
        "selection: largest four-window median range",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.4 if compact else 6.5,
        color="#6B7280",
    )


def _draw_domain_row(
    figure: plt.Figure,
    grid: GridSpec,
    row_index: int,
    result: FigureDomain,
    max_snapshots: int,
    trajectories: pd.DataFrame,
    horizon_end: int,
) -> None:
    """Draw four sparse snapshots and one current-indicator panel."""
    cfg_draw = _snapshot_draw_config(result.cfg, compact=True)
    for column, (_, end_year) in enumerate(result.cumulative_windows):
        if column:
            previous = result.cumulative_windows[column - 1][1]
        elif int(result.cfg["plot"]["landmark_focus_year"]) <= int(
            end_year
        ):
            previous = int(result.cfg["start_year"]) - 1
        else:
            previous = None
        axis = figure.add_subplot(grid[row_index, column])
        snapshot_renderer.draw_snapshot(
            axis,
            result.graph,
            result.display_comm_map,
            result.display_labels,
            result.positions,
            result.colors,
            cfg_draw,
            end_year=end_year,
            prev_end_year=previous,
            panel_label="",
            show_ylabel=column == 0,
        )
    for column in range(len(result.cumulative_windows), max_snapshots):
        figure.add_subplot(grid[row_index, column]).axis("off")
    indicator_axis = figure.add_subplot(grid[row_index, max_snapshots])
    _draw_indicator_panel(
        indicator_axis,
        result,
        trajectories,
        horizon_end,
        compact=True,
    )


def _draw_figure(
    results: Sequence[FigureDomain],
    trajectories: pd.DataFrame,
    config: Mapping[str, Any],
    output_dir: Path,
) -> Dict[str, Path]:
    """Render the four-domain group figure."""
    max_snapshots = max(len(result.cumulative_windows) for result in results)
    horizon_end = int(config["feature_horizon_end"])
    plt.rcParams["svg.hashsalt"] = "aspr-fig1-current-indicators-v3"
    figure = plt.figure(
        figsize=(
            float(config.get("figure_width", 28.0)),
            float(config.get("row_height", 4.35)) * len(results),
        ),
        dpi=int(config.get("dpi", 300)),
    )
    grid = GridSpec(
        len(results),
        max_snapshots + 1,
        figure=figure,
        width_ratios=[1.0] * max_snapshots + [1.75],
        hspace=0.38,
        wspace=0.16,
        left=0.035,
        right=0.995,
        top=0.94,
        bottom=0.055,
    )
    figure.suptitle(
        str(config["title"]), y=0.995, fontsize=16, fontweight="bold"
    )
    figure.text(
        0.5,
        0.982,
        str(config.get("subtitle", "")),
        ha="center",
        va="top",
        fontsize=8.3,
        color="#4B5563",
    )
    for row_index, result in enumerate(results):
        _draw_domain_row(
            figure,
            grid,
            row_index,
            result,
            max_snapshots,
            trajectories,
            horizon_end,
        )
    figure.text(
        0.995,
        0.004,
        "Left: seven-topic frozen graph view. Right: selected v6.1 indicators; "
        "all windows end by 2017. Descriptive, not causal.",
        ha="right",
        va="bottom",
        fontsize=6.3,
        color="#6B7280",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: Dict[str, Path] = {}
    for extension in config.get("formats", ["png", "svg", "pdf"]):
        path = output_dir / f"figure_full.{extension}"
        if extension == "pdf":
            fixed_time = dt.datetime(2026, 7, 26, tzinfo=dt.timezone.utc)
            metadata: Dict[str, Any] = {
                "Creator": "ASPR Fig.1 current formulas",
                "CreationDate": fixed_time,
                "ModDate": fixed_time,
            }
        elif extension == "svg":
            metadata = {
                "Creator": "ASPR Fig.1 current formulas",
                "Date": "2026-07-26",
            }
        else:
            metadata = {"Software": "ASPR Fig.1 current formulas"}
        figure.savefig(path, bbox_inches="tight", metadata=metadata)
        rendered[str(extension)] = path
    plt.close(figure)
    return rendered


def _draw_domain_figures(
    results: Sequence[FigureDomain],
    trajectories: pd.DataFrame,
    config: Mapping[str, Any],
    output_dir: Path,
) -> Dict[str, Dict[str, Path]]:
    """Render one readable detail figure for each landmark field."""
    rendered: Dict[str, Dict[str, Path]] = {}
    horizon_end = int(config["feature_horizon_end"])
    for result in results:
        slug = str(result.cfg["slug"])
        snapshot_count = len(result.cumulative_windows)
        figure = plt.figure(
            figsize=(
                float(config.get("domain_figure_width", 20.0)),
                float(config.get("domain_figure_height", 10.2)),
            ),
            dpi=int(config.get("dpi", 300)),
        )
        grid = GridSpec(
            3,
            snapshot_count,
            figure=figure,
            height_ratios=[0.38, 2.55, 1.45],
            hspace=0.32,
            wspace=0.14,
            left=0.055,
            right=0.985,
            top=0.91,
            bottom=0.075,
        )
        figure.suptitle(
            f"{result.cfg['domain_name']}: graph reconfiguration and indicators",
            y=0.985,
            fontsize=15,
            fontweight="bold",
        )
        figure.text(
            0.5,
            0.955,
            "Seven temporally representative topics; indicators selected only "
            "by their four-window median change.",
            ha="center",
            va="top",
            fontsize=9,
            color="#4B5563",
        )
        timeline = figure.add_subplot(grid[0, :])
        snapshot_renderer.draw_top_time_axis(timeline, result)
        cfg_draw = _snapshot_draw_config(result.cfg, compact=False)
        for column, (_, end_year) in enumerate(
            result.cumulative_windows
        ):
            if column:
                previous = result.cumulative_windows[column - 1][1]
            elif int(result.cfg["plot"]["landmark_focus_year"]) <= int(
                end_year
            ):
                previous = int(result.cfg["start_year"]) - 1
            else:
                previous = None
            axis = figure.add_subplot(grid[1, column])
            snapshot_renderer.draw_snapshot(
                axis,
                result.graph,
                result.display_comm_map,
                result.display_labels,
                result.positions,
                result.colors,
                cfg_draw,
                end_year=end_year,
                prev_end_year=previous,
                panel_label="",
                show_ylabel=column == 0,
            )
        indicator_axis = figure.add_subplot(grid[2, :])
        _draw_indicator_panel(
            indicator_axis,
            result,
            trajectories,
            horizon_end,
            compact=False,
        )
        figure.text(
            0.99,
            0.012,
            "Publication-time indicators only; descriptive alignment, not a "
            "causal event-study estimate.",
            ha="right",
            va="bottom",
            fontsize=7,
            color="#6B7280",
        )
        domain_dir = output_dir / "domains" / slug
        domain_dir.mkdir(parents=True, exist_ok=True)
        rendered[slug] = {}
        for extension in config.get("formats", ["png", "svg", "pdf"]):
            path = domain_dir / f"figure_{slug}.{extension}"
            if extension == "pdf":
                fixed_time = dt.datetime(
                    2026,
                    7,
                    26,
                    tzinfo=dt.timezone.utc,
                )
                metadata: Dict[str, Any] = {
                    "Creator": "ASPR Fig.1 current formulas",
                    "CreationDate": fixed_time,
                    "ModDate": fixed_time,
                }
            elif extension == "svg":
                metadata = {
                    "Creator": "ASPR Fig.1 current formulas",
                    "Date": "2026-07-26",
                }
            else:
                metadata = {"Software": "ASPR Fig.1 current formulas"}
            figure.savefig(path, bbox_inches="tight", metadata=metadata)
            rendered[slug][str(extension)] = path
        plt.close(figure)
    return rendered


def _node_set_until_year(graph: nx.Graph, cutoff: int) -> set[str]:
    """Select graph nodes visible by a cumulative cutoff."""
    return {
        str(node)
        for node, attributes in graph.nodes(data=True)
        if attributes.get("year") is not None
        and int(attributes["year"]) <= int(cutoff)
    }


def _snapshot_summary(results: Iterable[FigureDomain]) -> pd.DataFrame:
    """Build a compact audit table for cumulative snapshots."""
    rows: List[Dict[str, Any]] = []
    for result in results:
        previous_nodes: set[str] = set()
        for stage, (_, cutoff) in enumerate(
            result.cumulative_windows, start=1
        ):
            active = _node_set_until_year(result.graph, cutoff)
            topic_graph = snapshot_renderer.make_topic_graph(
                result.graph,
                result.display_comm_map,
                result.display_labels,
                active,
            )
            rows.append(
                {
                    "domain": result.cfg["slug"],
                    "stage": stage,
                    "cutoff_year": cutoff,
                    "paper_count": len(active),
                    "new_papers": len(active - previous_nodes),
                    "displayed_topic_count": int(
                        topic_graph.number_of_nodes()
                    ),
                    "selected_topic_cap": len(
                        result.selected_communities
                    ),
                }
            )
            previous_nodes = active
    return pd.DataFrame(rows)


def _write_contract(
    output_dir: Path,
    config: Mapping[str, Any],
    indicator_map: pd.DataFrame,
    selection: pd.DataFrame,
) -> None:
    """Write the chart contract and two-lens claim boundary."""
    _write_json(
        output_dir / "chart_contract.json",
        {
            "analytical_question": (
                "Which graph transitions and publication-time innovation "
                "indicator shifts are visible across four landmark fields?"
            ),
            "layout": (
                "one four-domain overview plus four domain detail figures; "
                "each field has four sparse cumulative graph snapshots and "
                "one selected-indicator trajectory panel"
            ),
            "left_numeric_source": (
                "v2_publication_v6a_locked_candidate/views/fig1"
            ),
            "right_numeric_source": (
                "nature_multihorizon_v6_1_local/"
                "innovation_candidate_features.parquet"
            ),
            "right_feature_scope": (
                "predeclared domain12 and primary-subfield matches in config"
            ),
            "right_aggregation": (
                "each raw registered indicator is converted only for display "
                "to an innovation-oriented within-domain12/year percentile; "
                "the plotted point is the window median"
            ),
            "indicator_selection": (
                "an indicator must have at least the configured valid sample "
                "in all four windows; candidates are ranked by max-minus-min "
                "window median; a deterministic largest-gap rule retains four "
                "or five indicators. Prediction outcomes are never consulted."
            ),
            "primary_features": indicator_map["feature"].tolist(),
            "selected_features_by_domain": {
                domain: group.sort_values("rank_by_change")[
                    "feature"
                ].tolist()
                for domain, group in selection[
                    selection["selected"]
                ].groupby("domain", sort=False)
            },
            "claim_boundary": (
                "the left and right panels are two aligned descriptive lenses, "
                "not the same paper sample and not a causal event study"
            ),
            "title": config["title"],
        },
    )


def _write_panel_text(
    output_dir: Path,
    domains: Sequence[FigureDomain],
    selection: pd.DataFrame,
) -> None:
    """Write exact reader-facing labels outside the figure image."""
    _write_json(
        output_dir / "panel_text.json",
        {
            "title": "Knowledge-graph reconfiguration and publication-time innovation signals",
            "rows": {
                result.cfg["slug"]: {
                    "graph_scope": result.cfg["domain_name"],
                    "indicator_scope_domain12": result.feature_domain12,
                    "indicator_scope_subfields": list(
                        result.feature_subfields
                    )
                    if result.feature_subfields
                    else ["all subfields in domain12"],
                    "selected_indicators": selection[
                        selection["domain"].eq(result.cfg["slug"])
                        & selection["selected"]
                    ]
                    .sort_values("rank_by_change")["feature_label"]
                    .tolist(),
                }
                for result in domains
            },
            "indicator_labels": FEATURE_LABELS,
            "horizon_note": "Current v6.1 feature materialization ends in 2017.",
        },
    )


def _write_manifest(
    output_dir: Path,
    config_path: Path,
    config: Mapping[str, Any],
    audits: Sequence[Mapping[str, Any]],
    indicator_map: pd.DataFrame,
    matched: pd.DataFrame,
    trajectories: pd.DataFrame,
    selection: pd.DataFrame,
    rendered: Mapping[str, Path],
    domain_rendered: Mapping[str, Mapping[str, Path]],
    source_paths: Mapping[str, Path],
    stage: str,
) -> Dict[str, Any]:
    """Write the run manifest and formula-provenance hashes."""
    formula_sources = {
        str(_resolve(path)): _sha256(_resolve(path))
        for path in FORMULA_SOURCE_PATHS
    }
    primary_exact = tuple(indicator_map["feature"]) == tuple(
        PRIMARY_FEATURES
    )
    registry_primary = indicator_map["final_role"].eq("primary").all()
    no_post_publication = (
        pd.to_numeric(matched["source_max_year"], errors="coerce")
        < pd.to_numeric(matched["publication_year"], errors="coerce")
    ).fillna(True).all()
    selected_rows = trajectories[
        trajectories["selected"].fillna(False)
    ]
    selected_counts = (
        selection[selection["selected"]]
        .groupby("domain", sort=False)
        .size()
    )
    selected_sets = {
        tuple(sorted(group["feature"].astype(str)))
        for _, group in selection[selection["selected"]].groupby(
            "domain",
            sort=False,
        )
    }
    rendered_domain_paths = [
        path
        for artifacts in domain_rendered.values()
        for path in artifacts.values()
    ]
    checks = {
        "four_domains_loaded": len(audits) == 4,
        "all_edge_endpoints_resolved": all(
            int(item["edge_endpoints_missing_from_nodes"]) == 0
            for item in audits
        ),
        "all_display_positions_resolved": all(
            int(item["display_communities_missing_positions"]) == 0
            for item in audits
        ),
        "all_graph_snapshots_nonempty": all(
            all(
                int(count) > 0
                for count in item["snapshot_display_topic_counts"]
            )
            for item in audits
        ),
        "display_topic_cap_respected": all(
            int(item["selected_display_topic_count"])
            <= int(config["maximum_display_topics"])
            for item in audits
        ),
        "primary_feature_set_exact": bool(primary_exact),
        "all_eight_registry_roles_primary": bool(registry_primary),
        "publication_time_boundary_pass": bool(no_post_publication),
        "matched_feature_rows_nonempty": len(matched) > 0,
        "trajectory_rows_complete": len(trajectories)
        == len(audits) * 4 * len(PRIMARY_FEATURES),
        "all_selected_indicator_cells_complete": bool(
            selected_rows["oriented_percentile_median"].notna().all()
            and selected_rows["n_valid"]
            .ge(int(config["minimum_indicator_sample"]))
            .all()
        ),
        "selected_indicator_count_in_range": bool(
            len(selected_counts) == len(audits)
            and selected_counts.between(
                int(config["minimum_selected_indicators"]),
                int(config["maximum_selected_indicators"]),
            ).all()
        ),
        "domain_indicator_sets_not_identical": len(selected_sets) > 1,
        "all_windows_within_feature_horizon": bool(
            not trajectories["right_truncated"].any()
        ),
        "no_historical_metric_output_files": not any(
            (output_dir / "panel_data" / name).exists()
            for name in (
                "perturbation_metrics.csv",
                "dominant_parameter_trajectories.csv",
                "angle_window_trajectories.csv",
            )
        ),
        "all_formats_rendered": all(
            path.exists() and path.stat().st_size > 0
            for path in rendered.values()
        ),
        "all_four_domain_figures_rendered": (
            len(domain_rendered) == len(audits)
            and all(
                path.exists() and path.stat().st_size > 0
                for path in rendered_domain_paths
            )
        ),
    }
    manifest = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "reproduction": (
            "python3 -m experiments.fig01.new.run "
            "--config experiments/fig01/new/config.json --stage all"
        ),
        "stage": stage,
        "stage_execution": "integrated_all",
        "config": {
            "path": str(config_path),
            "sha256": _sha256(config_path),
        },
        "source_policy": "local_frozen_only",
        "network_used": False,
        "visual_reuse_only": {
            "source": str(
                _resolve(str(config["snapshot_renderer_source"]))
            ),
            "allowed_operations": [
                "configuration/window parsing",
                "frozen topic-graph layout",
                "snapshot drawing",
            ],
            "historical_metric_formulas_used": False,
        },
        "current_measurement": {
            "primary_features": list(PRIMARY_FEATURES),
            "angle_features": {
                key: list(value) for key, value in ANGLE_FEATURES.items()
            },
            "plotted_unit": "registered_indicators_not_angle_aggregates",
            "selection_rule": (
                "complete_in_all_four_windows_then_largest_median_range_"
                "with_deterministic_elbow_between_4_and_5"
            ),
            "selected_features_by_domain": {
                domain: group.sort_values("rank_by_change")[
                    "feature"
                ].tolist()
                for domain, group in selection[
                    selection["selected"]
                ].groupby("domain", sort=False)
            },
            "materialization_versions": sorted(
                matched["definition_version"].dropna().astype(str).unique()
            ),
            "formula_source_sha256": formula_sources,
        },
        "sources": {
            key: {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for key, path in source_paths.items()
        },
        "domains": list(audits),
        "rendered": {
            key: {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for key, path in rendered.items()
        },
        "domain_rendered": {
            domain: {
                key: {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for key, path in artifacts.items()
            }
            for domain, artifacts in domain_rendered.items()
        },
        "case_replacements": list(config.get("case_replacements", [])),
        "checks": checks,
        "passed": all(checks.values()),
        "status": "DESCRIPTIVE_ONLY",
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    _write_json(
        output_dir / "audit_report.json",
        {
            "checks": checks,
            "passed": all(checks.values()),
            "status": "DESCRIPTIVE_ONLY",
            "nonfailure_disclosures": [
                "The graph and indicator panels are aligned domain lenses, not identical paper samples.",
                "Every displayed graph and indicator window ends by the frozen 2017 feature horizon.",
                "Exoplanets were replaced because the frozen 1990–2004 astronomy slice has insufficient calculable reference indicators.",
                "Indicator subsets are selected by within-case trajectory movement, never by D5 prediction performance.",
            ],
        },
    )
    return manifest


def run_figure1(
    config_path: Path,
    stage: str = "all",
) -> Dict[str, Any]:
    """Build the canonical Fig.1 using only current indicator values.

    Args:
        config_path: Path to the frozen Fig.1 configuration.
        stage: Shared-suite stage name. Fig.1 executes its integrated,
            deterministic pipeline for every accepted stage.

    Returns:
        The complete run manifest.
    """
    if stage not in {"prepare", "run", "plot", "audit", "all"}:
        raise ValueError(f"Unsupported stage: {stage}")
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = _resolve(str(config["output_dir"]))
    panel_data = output_dir / "panel_data"
    panel_data.mkdir(parents=True, exist_ok=True)
    for obsolete_name in (
        "perturbation_metrics.csv",
        "dominant_parameter_trajectories.csv",
        "angle_window_trajectories.csv",
    ):
        (panel_data / obsolete_name).unlink(missing_ok=True)

    frozen_root = _resolve(str(config["frozen_graph_root"]))
    domains: List[FigureDomain] = []
    audits: List[Dict[str, Any]] = []
    for domain_spec in config["domains"]:
        domain, audit = _load_domain(
            domain_spec,
            frozen_root,
            int(config["maximum_display_topics"]),
        )
        domains.append(domain)
        audits.append(audit)

    registry_path = _resolve(str(config["candidate_registry"]))
    indicator_map, _ = _load_registry(registry_path)
    paper_path = _resolve(str(config["current_paper_dataset"]))
    feature_path = _resolve(str(config["current_feature_dataset"]))
    all_features = _add_oriented_percentiles(
        _load_current_features(paper_path, feature_path)
    )
    matched = _matched_feature_rows(domains, all_features)
    horizon_end = int(config["feature_horizon_end"])
    indicator_summary = _window_feature_summary(
        matched, domains, horizon_end
    )
    trajectories, selection = _select_indicator_trajectories(
        indicator_summary,
        int(config["minimum_indicator_sample"]),
        int(config["minimum_selected_indicators"]),
        int(config["maximum_selected_indicators"]),
    )

    snapshot_summary = _snapshot_summary(domains)
    snapshot_summary.to_csv(
        panel_data / "snapshot_summary.csv", index=False
    )
    indicator_map.to_csv(
        panel_data / "primary_indicator_map.csv", index=False
    )
    indicator_summary.to_csv(
        panel_data / "indicator_window_summary.csv", index=False
    )
    trajectories.to_csv(
        panel_data / "indicator_trajectories.csv", index=False
    )
    selection.to_csv(
        panel_data / "indicator_selection.csv", index=False
    )
    matched.to_parquet(
        panel_data / "matched_paper_indicator_features.parquet", index=False
    )

    rendered = _draw_figure(domains, trajectories, config, output_dir)
    domain_rendered = _draw_domain_figures(
        domains,
        trajectories,
        config,
        output_dir,
    )
    _write_contract(output_dir, config, indicator_map, selection)
    _write_panel_text(output_dir, domains, selection)
    source_paths = {
        "paper_dataset": paper_path,
        "feature_dataset": feature_path,
        "candidate_registry": registry_path,
    }
    return _write_manifest(
        output_dir,
        config_path,
        config,
        audits,
        indicator_map,
        matched,
        trajectories,
        selection,
        rendered,
        domain_rendered,
        source_paths,
        stage,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Render the canonical Fig.1 with current v6.1 indicators."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--stage",
        choices=["prepare", "run", "plot", "audit", "all"],
        default="all",
        help=(
            "Accepted for the shared runner interface. Fig.1 rebuilds its "
            "integrated deterministic artifact for every stage."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    if os.environ.get("PYTHONHASHSEED") != "0":
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = "0"
        os.execvpe(
            sys.executable,
            [
                sys.executable,
                "-m",
                "experiments.fig01.new.run",
                *sys.argv[1:],
            ],
            environment,
        )
    args = parse_args(argv)
    manifest = run_figure1(args.config, stage=args.stage)
    print(
        json.dumps(
            {
                "passed": manifest["passed"],
                "rendered": manifest["rendered"],
            },
            indent=2,
        )
    )
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

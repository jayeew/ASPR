"""Build the data-governed, high-density Nature redesign of Fig. 1.

The four domains and their field-specific indicator lists are frozen before
annual trajectories and bootstrap intervals are computed. Graphs use four
cumulative knowledge states beginning at t−6 with three-year additions.
Indicators use publication-time
features only, oriented and ranked within publication year across the ten
candidate domains. No citation outcome, D5 label, OOF prediction, or future
impact variable is read by this module.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

from experiments.common.new.adapters.contracts import (
    ANGLE_FEATURES,
    FEATURE_DIRECTION,
    PRIMARY_FEATURES,
)

from .descriptive_contract import (
    DOMAIN_LABELS,
    FEATURE_LABELS,
    STAGE_KEYS,
    STAGE_LABELS,
)
from .event_data import canonical_hash, sha256_file, stable_seed, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[3]
GraphEdges = Dict[Tuple[str, str], float]
GraphNodes = Dict[str, Dict[str, Any]]


# ============================================================================
# Frozen contracts
# ============================================================================


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_frozen_selection(
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    path = _resolve_project_path(str(config["frozen_selection_file"]))
    observed = sha256_file(path)
    expected = str(config["frozen_selection_sha256"])
    if observed != expected:
        raise ValueError(
            "Frozen Fig.1 selection hash changed: "
            f"expected {expected}, observed {observed}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    if len(cases) != 4:
        raise ValueError("Frozen Fig.1 selection must contain four cases")
    return payload


def _load_topic_labels(
    config: Mapping[str, Any],
) -> Mapping[str, Mapping[str, str]]:
    path = _resolve_project_path(str(config["topic_short_labels_file"]))
    observed = sha256_file(path)
    expected = str(config["topic_short_labels_sha256"])
    if observed != expected:
        raise ValueError(
            "Frozen topic-label hash changed: "
            f"expected {expected}, observed {observed}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["labels"]


def _frozen_case_lookup(
    frozen: Mapping[str, Any],
) -> Mapping[str, Mapping[str, Any]]:
    return {
        str(case["episode_id"]): case
        for case in frozen["cases"]
    }


def _selection_fingerprint(frozen: Mapping[str, Any]) -> str:
    compact = [
        {
            "selection_rank": int(case["selection_rank"]),
            "episode_id": str(case["episode_id"]),
            "domain": str(case["domain"]),
            "features": [str(value) for value in case["features"]],
        }
        for case in frozen["cases"]
    ]
    return canonical_hash(compact)


# ============================================================================
# Publication-window topic graphs
# ============================================================================


def _reference_list(value: Any) -> List[str]:
    """Normalize a serialized or native reference array."""
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = []
        return [str(item) for item in parsed if str(item)]
    return []


def _weighted_jaccard_distance(
    left: Mapping[Tuple[str, str], float],
    right: Mapping[Tuple[str, str], float],
) -> float:
    """Return one minus weighted Jaccard similarity."""
    keys = set(left) | set(right)
    if not keys:
        return float("nan")
    shared = sum(min(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    union = sum(max(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    return float(1.0 - shared / union) if union > 0 else float("nan")


def _effective_topic_count(values: Iterable[str]) -> float:
    counts = np.asarray(
        list(Counter(value for value in values if value).values()),
        dtype=float,
    )
    if counts.size == 0:
        return float("nan")
    probabilities = counts / counts.sum()
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    return float(np.exp(entropy))


def _topic_graph(
    papers: pd.DataFrame,
    *,
    minimum_shared_references: int,
) -> Tuple[GraphEdges, GraphNodes, Mapping[str, int]]:
    """Aggregate cosine-normalized paper coupling to primary-topic edges."""
    rows = [
        row
        for row in papers.itertuples(index=False)
        if str(row.primary_topic_id)
        and len(_reference_list(row.referenced_works)) >= 2
    ]
    if len(rows) < 2:
        return {}, {}, {
            "coupling_edges": 0,
            "reference_count": 0,
            "threshold_used": int(minimum_shared_references),
        }
    reference_ids = sorted(
        {
            reference_id
            for row in rows
            for reference_id in _reference_list(row.referenced_works)
        }
    )
    reference_index = {
        value: index for index, value in enumerate(reference_ids)
    }
    incidence_rows: List[int] = []
    incidence_columns: List[int] = []
    for row_index, row in enumerate(rows):
        for reference_id in set(_reference_list(row.referenced_works)):
            incidence_rows.append(row_index)
            incidence_columns.append(reference_index[reference_id])
    incidence = sparse.csr_matrix(
        (
            np.ones(len(incidence_rows), dtype=np.float64),
            (incidence_rows, incidence_columns),
        ),
        shape=(len(rows), len(reference_ids)),
    )
    reference_counts = np.asarray(incidence.sum(axis=1)).ravel()
    shared = sparse.triu(incidence @ incidence.T, k=1).tocoo()
    threshold = int(minimum_shared_references)
    keep = shared.data >= threshold
    if not np.any(keep):
        threshold = 1
        keep = shared.data >= threshold
    topic_edges: MutableMapping[Tuple[str, str], float] = defaultdict(float)
    coupling_edges = 0
    for left, right, overlap in zip(
        shared.row[keep],
        shared.col[keep],
        shared.data[keep],
    ):
        denominator = math.sqrt(
            reference_counts[left] * reference_counts[right]
        )
        if denominator <= 0:
            continue
        left_topic = str(rows[int(left)].primary_topic_id)
        right_topic = str(rows[int(right)].primary_topic_id)
        if not left_topic or not right_topic or left_topic == right_topic:
            continue
        edge = tuple(sorted((left_topic, right_topic)))
        topic_edges[edge] += float(overlap / denominator)
        coupling_edges += 1
    total_weight = float(sum(topic_edges.values()))
    normalized = {
        edge: weight / total_weight
        for edge, weight in topic_edges.items()
        if total_weight > 0
    }
    node_counts = Counter(str(row.primary_topic_id) for row in rows)
    node_names: Dict[str, str] = {}
    node_subfields: Dict[str, str] = {}
    for row in rows:
        topic_id = str(row.primary_topic_id)
        node_names.setdefault(topic_id, str(row.primary_topic_name))
        node_subfields.setdefault(topic_id, str(row.primary_subfield_name))
    nodes = {
        topic_id: {
            "topic_name": node_names.get(
                topic_id,
                topic_id.rsplit("/", 1)[-1],
            ),
            "subfield": node_subfields.get(topic_id, ""),
            "paper_count": int(count),
        }
        for topic_id, count in node_counts.items()
    }
    return normalized, nodes, {
        "coupling_edges": int(coupling_edges),
        "reference_count": int(len(reference_ids)),
        "threshold_used": int(threshold),
    }


def _stage_windows(
    episode: Mapping[str, Any],
    config: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Create four equal-duration windows anchored on the landmark start."""
    start_year = int(episode["start_year"])
    window_years = int(config["windows"]["years_per_stage"])
    offsets = config["windows"]["start_offsets"]
    rows: List[Dict[str, Any]] = []
    for index, stage in enumerate(STAGE_KEYS):
        requested_start = start_year + int(offsets[stage])
        rows.append(
            {
                "stage_index": index,
                "stage": stage,
                "stage_label": STAGE_LABELS[stage],
                "start_year": requested_start,
                "end_year": requested_start + window_years - 1,
            }
        )
    return rows


def _graph_snapshot_windows(
    episode: Mapping[str, Any],
    config: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Create cumulative graph snapshots while retaining three-year additions."""
    if not bool(config["graph"]["cumulative_snapshots"]):
        return _stage_windows(episode, config)
    event_start = int(episode["start_year"])
    history_start = event_start + int(
        config["graph"]["history_start_offset"]
    )
    increments = _stage_windows(episode, config)
    rows: List[Dict[str, Any]] = []
    for index, increment in enumerate(increments):
        end_year = int(increment["end_year"])
        rows.append(
            {
                **increment,
                "start_year": history_start,
                "end_year": end_year,
                "increment_start_year": (
                    history_start
                    if index == 0
                    else int(increments[index]["start_year"])
                ),
                "increment_end_year": end_year,
                "cumulative_snapshot": True,
                "stage_label": (
                    f"t{int(config['graph']['history_start_offset']):+d} "
                    "to t−1"
                    if index == 0
                    else f"through t+{end_year - event_start}"
                ).replace("t-6", "t−6"),
            }
        )
    return rows


def _episode_graphs(
    episode: Mapping[str, Any],
    domain_papers: pd.DataFrame,
    config: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Build four cumulative knowledge-state graphs from frozen local papers."""
    graphs: List[Dict[str, Any]] = []
    for window in _graph_snapshot_windows(episode, config):
        selected = domain_papers[
            domain_papers["publication_year"].between(
                int(window["start_year"]),
                int(window["end_year"]),
                inclusive="both",
            )
        ].copy()
        edges, nodes, graph_audit = _topic_graph(
            selected,
            minimum_shared_references=int(
                config["graph"]["minimum_shared_references"]
            ),
        )
        graphs.append(
            {
                **window,
                "papers": selected,
                "edges": edges,
                "nodes": nodes,
                "audit": graph_audit,
            }
        )
    return graphs


def _topic_turnover(left: GraphNodes, right: GraphNodes) -> float:
    union = set(left) | set(right)
    if not union:
        return float("nan")
    return float(1.0 - len(set(left) & set(right)) / len(union))


def _episode_score_row(
    episode: Mapping[str, Any],
    graphs: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Reproduce the registered graph-change screen for disclosure."""
    pre, late = graphs[0], graphs[-1]
    successive = [
        _weighted_jaccard_distance(
            graphs[index - 1]["edges"],
            graphs[index]["edges"],
        )
        for index in range(1, len(graphs))
    ]
    effective_topics = [
        _effective_topic_count(
            graph["papers"]["primary_topic_id"].fillna("").astype(str)
        )
        for graph in graphs
    ]
    edge_log_change = abs(
        math.log1p(len(late["edges"])) - math.log1p(len(pre["edges"]))
    )
    topic_log_change = abs(
        math.log1p(effective_topics[-1])
        - math.log1p(effective_topics[0])
    )
    minimums = config["selection"]
    paper_counts = [len(graph["papers"]) for graph in graphs]
    node_counts = [len(graph["nodes"]) for graph in graphs]
    edge_counts = [len(graph["edges"]) for graph in graphs]
    reasons: List[str] = []
    if min(paper_counts) < int(minimums["minimum_papers_per_window"]):
        reasons.append("insufficient_papers")
    if min(node_counts) < int(minimums["minimum_graph_nodes"]):
        reasons.append("insufficient_topic_nodes")
    if min(edge_counts) < int(minimums["minimum_graph_edges"]):
        reasons.append("empty_or_sparse_topic_graph")
    domain = str(episode["domain"])
    return {
        "episode_id": str(episode["episode_id"]),
        "domain": domain,
        "domain_label": DOMAIN_LABELS.get(domain, domain),
        "episode_label": str(episode["label"]),
        "landmark_start_year": int(episode["start_year"]),
        "landmark_end_year": int(episode["end_year"]),
        "pre_late_edge_turnover": _weighted_jaccard_distance(
            pre["edges"],
            late["edges"],
        ),
        "mean_successive_edge_turnover": float(np.nanmean(successive)),
        "pre_late_topic_turnover": _topic_turnover(
            pre["nodes"],
            late["nodes"],
        ),
        "absolute_log_edge_count_change": float(edge_log_change),
        "absolute_log_effective_topic_change": float(topic_log_change),
        "minimum_window_papers": int(min(paper_counts)),
        "minimum_window_topic_nodes": int(min(node_counts)),
        "minimum_window_topic_edges": int(min(edge_counts)),
        "eligible": not reasons,
        "exclusion_reason": "|".join(reasons),
    }


def _rank_and_select_episodes(
    candidates: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Reproduce the original registered screen before freezing cases."""
    weights = config["selection"]["domain_score_weights"]
    components = {
        "pre_late_edge_turnover": "edge_turnover_rank",
        "mean_successive_edge_turnover": "successive_turnover_rank",
        "pre_late_topic_turnover": "topic_turnover_rank",
        "absolute_log_edge_count_change": "edge_count_change_rank",
        "absolute_log_effective_topic_change": "effective_topic_change_rank",
    }
    frame = candidates.copy()
    for source, target in components.items():
        frame[target] = frame[source].rank(
            method="average",
            pct=True,
            na_option="bottom",
        )
    frame["graph_change_score"] = sum(
        frame[target] * float(weights[source])
        for source, target in components.items()
    )
    frame["selected"] = False
    frame["selection_rank"] = pd.Series(
        pd.NA,
        index=frame.index,
        dtype="Int64",
    )
    eligible = frame.loc[frame["eligible"]].sort_values(
        ["graph_change_score", "episode_id"],
        ascending=[False, True],
        kind="stable",
    )
    unique_rows = eligible.drop_duplicates("domain", keep="first").head(
        int(config["selection"]["display_domain_count"])
    )
    for rank, index in enumerate(unique_rows.index, start=1):
        frame.loc[index, "selected"] = True
        frame.loc[index, "selection_rank"] = rank
    frame["selection_rule"] = (
        "registered_graph_change_screen_before_display_freeze"
    )
    return frame


def _apply_frozen_domain_selection(
    screened: pd.DataFrame,
    frozen: Mapping[str, Any],
) -> pd.DataFrame:
    """Replace dynamic display flags with the frozen four-case contract."""
    frame = screened.rename(
        columns={
            "selected": "screen_selected",
            "selection_rank": "screen_selection_rank",
        }
    ).copy()
    frame["selected"] = False
    frame["selection_rank"] = pd.Series(
        pd.NA,
        index=frame.index,
        dtype="Int64",
    )
    for case in frozen["cases"]:
        episode_id = str(case["episode_id"])
        domain = str(case["domain"])
        mask = frame["episode_id"].eq(episode_id) & frame["domain"].eq(domain)
        if int(mask.sum()) != 1:
            raise ValueError(
                f"Frozen episode/domain pair not found exactly once: {episode_id}"
            )
        if not bool(frame.loc[mask, "eligible"].iloc[0]):
            raise ValueError(f"Frozen episode no longer passes gates: {episode_id}")
        frame.loc[mask, "selected"] = True
        frame.loc[mask, "selection_rank"] = int(case["selection_rank"])
    frame["selection_rule"] = (
        "frozen_registered_high_change_cases_no_annual_or_effect_feedback"
    )
    return frame.sort_values(
        ["selected", "selection_rank", "graph_change_score", "episode_id"],
        ascending=[False, True, False, True],
        kind="stable",
    ).reset_index(drop=True)


# ============================================================================
# Deterministic display graph tables
# ============================================================================


def _landmark_topics(
    papers: pd.DataFrame,
    episode: Mapping[str, Any],
) -> set[str]:
    selected = papers[
        papers["publication_year"].between(
            int(episode["start_year"]),
            int(episode["end_year"]),
            inclusive="both",
        )
        & papers["is_landmark"].eq(1)
    ]
    return set(selected["primary_topic_id"].dropna().astype(str))


def _display_graphs_with_landmark_onset(
    graphs: Sequence[Mapping[str, Any]],
    landmark_topics: set[str],
) -> List[Dict[str, Any]]:
    """Hide landmark-bearing topics before t0 and retain them thereafter."""
    displayed: List[Dict[str, Any]] = []
    for graph in graphs:
        item = dict(graph)
        if int(graph["stage_index"]) == 0 and landmark_topics:
            item["nodes"] = {
                node_id: attrs
                for node_id, attrs in graph["nodes"].items()
                if node_id not in landmark_topics
            }
            item["edges"] = {
                edge: weight
                for edge, weight in graph["edges"].items()
                if not set(edge) & landmark_topics
            }
            item["papers"] = graph["papers"][
                ~graph["papers"]["primary_topic_id"]
                .fillna("")
                .astype(str)
                .isin(landmark_topics)
            ].copy()
        displayed.append(item)
    return displayed


def _node_scores(
    graphs: Sequence[Mapping[str, Any]],
) -> Mapping[str, float]:
    scores: MutableMapping[str, float] = defaultdict(float)
    for graph in graphs:
        for node_id, attrs in graph["nodes"].items():
            scores[node_id] += float(attrs["paper_count"])
        for (left, right), weight in graph["edges"].items():
            scores[left] += 5000.0 * float(weight)
            scores[right] += 5000.0 * float(weight)
    return scores


def _display_node_ids(
    graphs: Sequence[Mapping[str, Any]],
    landmark_topics: set[str],
    maximum_nodes: int,
) -> List[str]:
    """Choose the deterministic detail-node union used by all four windows."""
    selected: List[str] = sorted(landmark_topics)
    ranked_edges = [
        sorted(
            graph["edges"].items(),
            key=lambda item: (-float(item[1]), item[0]),
        )
        for graph in graphs
    ]
    for edge_rank in range(4):
        for stage_edges in ranked_edges:
            if edge_rank >= len(stage_edges):
                continue
            additions = [
                node_id
                for node_id in stage_edges[edge_rank][0]
                if node_id not in selected
            ]
            if len(selected) + len(additions) <= maximum_nodes:
                selected.extend(additions)
    scores = _node_scores(graphs)
    for node_id in sorted(scores, key=lambda value: (-scores[value], value)):
        if node_id not in selected:
            selected.append(node_id)
        if len(selected) >= maximum_nodes:
            break
    return selected[:maximum_nodes]


def _best_main_subset(
    detail_ids: Sequence[str],
    graphs: Sequence[Mapping[str, Any]],
    landmark_topics: set[str],
    maximum_nodes: int,
) -> List[str]:
    """Find the eight-topic subset with the strongest minimum-stage support."""
    if len(detail_ids) <= maximum_nodes:
        return list(detail_ids)
    required = set(detail_ids) & landmark_topics
    best_subset: Tuple[str, ...] | None = None
    best_score: Tuple[float, ...] | None = None
    node_scores = _node_scores(graphs)
    for subset in itertools.combinations(detail_ids, maximum_nodes):
        subset_set = set(subset)
        if not required.issubset(subset_set):
            continue
        counts: List[int] = []
        weights: List[float] = []
        for graph in graphs:
            stage_edges = [
                float(weight)
                for edge, weight in graph["edges"].items()
                if set(edge).issubset(subset_set)
            ]
            counts.append(len(stage_edges))
            weights.append(sum(stage_edges))
        score = (
            float(min(counts)),
            float(sum(min(value, 16) for value in counts)),
            float(sum(weights)),
            float(sum(node_scores.get(value, 0.0) for value in subset)),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_subset = subset
    if best_subset is None:
        raise ValueError("Unable to construct the frozen main topic subset")
    selected = set(best_subset)
    return [node_id for node_id in detail_ids if node_id in selected]


def _collision_aware_layout(
    layout: Mapping[str, np.ndarray],
    node_ids: Sequence[str],
    graphs: Sequence[Mapping[str, Any]],
    scope: str,
    config: Mapping[str, Any],
) -> Mapping[str, np.ndarray]:
    """Repel topic centres using their cumulative paper volume."""
    if len(layout) <= 1:
        return layout
    keys = [node_id for node_id in node_ids if node_id in layout]
    points = np.asarray([layout[node_id] for node_id in keys], dtype=float)
    points -= points.mean(axis=0)
    initial_scale = max(float(np.max(np.abs(points))), 1e-12)
    points = points / initial_scale * 0.78
    counts = np.asarray(
        [
            max(
                float(graph["nodes"].get(node_id, {}).get("paper_count", 0))
                for graph in graphs
            )
            for node_id in keys
        ],
        dtype=float,
    )
    logged = np.log1p(counts)
    if float(np.ptp(logged)) > 1e-12:
        size = (logged - logged.min()) / np.ptp(logged)
    else:
        size = np.full(len(keys), 0.5, dtype=float)
    base = float(
        config["graph"][
            "main_layout_minimum_separation"
            if scope == "main"
            else "detail_layout_minimum_separation"
        ]
    )
    iterations = int(config["graph"]["layout_collision_iterations"])
    for _ in range(iterations):
        displacement = np.zeros_like(points)
        for left in range(len(keys)):
            for right in range(left + 1, len(keys)):
                delta = points[left] - points[right]
                distance = float(np.linalg.norm(delta))
                if distance < 1e-9:
                    digest = hashlib.sha256(
                        f"{keys[left]}|{keys[right]}".encode("utf-8")
                    ).digest()
                    angle = 2.0 * math.pi * digest[0] / 255.0
                    delta = np.asarray(
                        [math.cos(angle), math.sin(angle)],
                        dtype=float,
                    )
                    distance = 1e-9
                required = base * (
                    0.84 + 0.20 * (size[left] + size[right])
                )
                if distance >= required:
                    continue
                unit = delta / distance
                push = 0.11 * (required - distance) * unit
                displacement[left] += push
                displacement[right] -= push
        displacement -= 0.0012 * points
        points += displacement
        points = np.clip(points, -0.94, 0.94)
    points -= points.mean(axis=0)
    x_extent = float(
        config["graph"][
            "main_layout_target_x_extent"
            if scope == "main"
            else "detail_layout_target_x_extent"
        ]
    )
    y_extent = float(
        config["graph"][
            "main_layout_target_y_extent"
            if scope == "main"
            else "detail_layout_target_y_extent"
        ]
    )
    observed_x = max(float(np.max(np.abs(points[:, 0]))), 1e-12)
    observed_y = max(float(np.max(np.abs(points[:, 1]))), 1e-12)
    points[:, 0] *= x_extent / observed_x
    points[:, 1] *= y_extent / observed_y
    return {
        node_id: np.asarray(points[index], dtype=float)
        for index, node_id in enumerate(keys)
    }


def _stable_layout(
    node_ids: Sequence[str],
    graphs: Sequence[Mapping[str, Any]],
    domain: str,
    scope: str,
    config: Mapping[str, Any],
) -> Mapping[str, np.ndarray]:
    union = nx.Graph()
    union.add_nodes_from(node_ids)
    for graph in graphs:
        for (left, right), weight in graph["edges"].items():
            if left not in union or right not in union:
                continue
            if union.has_edge(left, right):
                union[left][right]["weight"] += float(weight)
            else:
                union.add_edge(left, right, weight=float(weight))
    if len(union) == 1:
        return {next(iter(union)): np.asarray([0.0, 0.0])}
    seed = stable_seed(
        "fig1-nature-dense-layout",
        config["graph"]["layout_seed"],
        domain,
        scope,
    )
    layout = nx.spring_layout(
        union,
        seed=seed,
        weight="weight",
        iterations=int(config["graph"]["layout_iterations"]),
        k=float(config["graph"]["layout_spring_k_scale"])
        / math.sqrt(max(2, len(union))),
    )
    points = np.asarray(list(layout.values()), dtype=float)
    center = points.mean(axis=0)
    scale = np.max(np.abs(points - center))
    if scale <= 0:
        scale = 1.0
    normalized = {
        node_id: (np.asarray(point, dtype=float) - center) / scale
        for node_id, point in layout.items()
    }
    return _collision_aware_layout(
        normalized,
        node_ids,
        graphs,
        scope,
        config,
    )


def _top_stage_edges(
    graph: Mapping[str, Any],
    node_ids: Sequence[str],
    cap: int,
    persistence: Mapping[Tuple[str, str], int],
) -> List[Tuple[Tuple[str, str], float]]:
    node_set = set(node_ids)
    candidates = [
        (edge, float(weight))
        for edge, weight in graph["edges"].items()
        if set(edge).issubset(node_set)
    ]
    return sorted(
        candidates,
        key=lambda item: (
            -float(item[1]),
            -int(persistence.get(item[0], 0)),
            item[0],
        ),
    )[:cap]


def _transition_rows(
    *,
    episode: Mapping[str, Any],
    graphs: Sequence[Mapping[str, Any]],
    node_ids: Sequence[str],
    view: str,
    cap: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    persistence = Counter(
        edge for graph in graphs for edge in graph["edges"]
    )
    selected = [
        _top_stage_edges(graph, node_ids, cap, persistence)
        for graph in graphs
    ]
    edge_sets = [{edge for edge, _ in rows} for rows in selected]
    weight_maps = [dict(rows) for rows in selected]
    skeleton = sorted(set().union(*edge_sets))
    union_weights = {
        edge: max(weights.get(edge, 0.0) for weights in weight_maps)
        for edge in skeleton
    }
    transition_rows: List[Dict[str, Any]] = []
    active_rows: List[Dict[str, Any]] = []
    domain = str(episode["domain"])
    for stage_index, graph in enumerate(graphs):
        current = edge_sets[stage_index]
        previous = edge_sets[stage_index - 1] if stage_index > 0 else set()
        for edge in skeleton:
            if edge in current:
                status = (
                    "baseline"
                    if stage_index == 0
                    else "retained"
                    if edge in previous
                    else "gained"
                )
            elif stage_index > 0 and edge in previous:
                status = "lost"
            else:
                status = "background"
            current_weight = float(weight_maps[stage_index].get(edge, 0.0))
            previous_weight = (
                float(weight_maps[stage_index - 1].get(edge, 0.0))
                if stage_index > 0
                else 0.0
            )
            row = {
                "episode_id": str(episode["episode_id"]),
                "domain": domain,
                "display_scope": view,
                "stage_index": int(stage_index),
                "stage": str(graph["stage"]),
                "start_year": int(graph["start_year"]),
                "end_year": int(graph["end_year"]),
                "source": edge[0],
                "target": edge[1],
                "status": status,
                "current_weight": current_weight,
                "previous_weight": previous_weight,
                "display_weight": (
                    current_weight
                    if status in {"baseline", "retained", "gained"}
                    else previous_weight
                    if status == "lost"
                    else float(union_weights[edge])
                ),
                "union_max_weight": float(union_weights[edge]),
            }
            transition_rows.append(row)
            if edge in current:
                active_rows.append(
                    {
                        **row,
                        "weight": current_weight,
                        "new_edge": status == "gained",
                    }
                )
    return transition_rows, active_rows


def _representative_papers(
    *,
    episode: Mapping[str, Any],
    graph: Mapping[str, Any],
    detail_ids: Sequence[str],
    main_ids: Sequence[str],
    maximum: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    papers = graph["papers"].copy()
    papers["reference_count"] = papers["referenced_works"].map(
        lambda value: len(_reference_list(value))
    )
    papers["stable_order"] = papers["work_id"].map(
        lambda value: hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    )
    for topic_id in detail_ids:
        candidates = papers[
            papers["primary_topic_id"].fillna("").astype(str).eq(topic_id)
        ].copy()
        candidates = candidates.sort_values(
            ["is_landmark", "reference_count", "stable_order", "work_id"],
            ascending=[False, False, True, True],
            kind="stable",
        ).head(maximum)
        for bead_index, row in enumerate(
            candidates.itertuples(index=False),
            start=1,
        ):
            rows.append(
                {
                    "episode_id": str(episode["episode_id"]),
                    "domain": str(episode["domain"]),
                    "stage_index": int(graph["stage_index"]),
                    "stage": str(graph["stage"]),
                    "start_year": int(graph["start_year"]),
                    "end_year": int(graph["end_year"]),
                    "topic_id": topic_id,
                    "paper_id": str(row.work_id),
                    "title": str(row.title),
                    "publication_year": int(row.publication_year),
                    "reference_count": int(row.reference_count),
                    "is_landmark": bool(row.is_landmark),
                    "bead_index": int(bead_index),
                    "display_main": topic_id in set(main_ids),
                    "display_detail": True,
                    "selection_rule": (
                        "landmark_first_then_reference_count_then_stable_work_id"
                    ),
                }
            )
    return rows


def _display_tables(
    episode: Mapping[str, Any],
    graphs: Sequence[Mapping[str, Any]],
    domain_papers: pd.DataFrame,
    config: Mapping[str, Any],
    topic_labels: Mapping[str, Mapping[str, str]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create nodes, active edges, transitions, summaries, and paper beads."""
    domain = str(episode["domain"])
    landmarks = _landmark_topics(domain_papers, episode)
    display_graphs = _display_graphs_with_landmark_onset(
        graphs,
        landmarks,
    )
    detail_ids = _display_node_ids(
        display_graphs,
        landmarks,
        int(config["graph"]["detail_maximum_display_nodes"]),
    )
    main_ids = _best_main_subset(
        detail_ids,
        display_graphs,
        landmarks,
        int(config["graph"]["main_maximum_display_nodes"]),
    )
    main_layout = _stable_layout(
        main_ids,
        display_graphs,
        domain,
        "main",
        config,
    )
    detail_layout = _stable_layout(
        detail_ids,
        display_graphs,
        domain,
        "detail",
        config,
    )
    main_transitions, main_active_edges = _transition_rows(
        episode=episode,
        graphs=display_graphs,
        node_ids=main_ids,
        view="main",
        cap=int(
            config["graph"]["main_maximum_active_edges_per_snapshot"]
        ),
    )
    detail_transitions, detail_active_edges = _transition_rows(
        episode=episode,
        graphs=display_graphs,
        node_ids=detail_ids,
        view="detail",
        cap=int(
            config["graph"]["detail_maximum_active_edges_per_snapshot"]
        ),
    )
    transition_frame = pd.DataFrame(main_transitions + detail_transitions)
    active_frame = pd.DataFrame(main_active_edges + detail_active_edges)
    node_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    representative_rows: List[Dict[str, Any]] = []
    previous_nodes: set[str] = set()
    for graph, display_graph in zip(graphs, display_graphs):
        active_nodes = set(display_graph["nodes"]) & set(detail_ids)
        landmark_coupling: MutableMapping[str, float] = defaultdict(float)
        for (left, right), weight in display_graph["edges"].items():
            if left in landmarks and right not in landmarks:
                landmark_coupling[right] += float(weight)
            if right in landmarks and left not in landmarks:
                landmark_coupling[left] += float(weight)
        for node_id in detail_ids:
            attrs = display_graph["nodes"].get(node_id, {})
            raw_name = str(attrs.get("topic_name") or "")
            registered = topic_labels.get(node_id)
            if registered is None:
                raise ValueError(
                    f"Displayed topic lacks frozen short label: {node_id}"
                )
            if raw_name and str(registered["raw"]) != raw_name:
                raise ValueError(
                    "OpenAlex topic name changed for frozen label: "
                    f"{node_id}: {raw_name!r}"
                )
            node_rows.append(
                {
                    "episode_id": str(episode["episode_id"]),
                    "domain": domain,
                    "stage_index": int(graph["stage_index"]),
                    "stage": str(graph["stage"]),
                    "stage_label": str(graph["stage_label"]),
                    "start_year": int(graph["start_year"]),
                    "end_year": int(graph["end_year"]),
                    "node_id": node_id,
                    "topic_name_raw": str(registered["raw"]),
                    "topic_label": str(registered["short"]),
                    "subfield": str(attrs.get("subfield") or ""),
                    "paper_count": int(attrs.get("paper_count") or 0),
                    "active": node_id in active_nodes,
                    "suppressed_pre_landmark": bool(
                        int(graph["stage_index"]) == 0
                        and node_id in landmarks
                    ),
                    "new_node": (
                        int(graph["stage_index"]) > 0
                        and node_id in active_nodes
                        and node_id not in previous_nodes
                    ),
                    "landmark_topic": node_id in landmarks,
                    "landmark_coupling_weight": float(
                        landmark_coupling.get(node_id, 0.0)
                    ),
                    "community_relation": (
                        "landmark_bearing_topic"
                        if node_id in landmarks
                        else "direct_landmark_neighbor"
                        if landmark_coupling.get(node_id, 0.0) > 0.0
                        else "pre_landmark_context"
                        if int(graph["stage_index"]) == 0
                        else "field_backbone_context"
                    ),
                    "display_main": node_id in set(main_ids),
                    "display_detail": True,
                    "main_x": (
                        float(main_layout[node_id][0])
                        if node_id in main_layout
                        else np.nan
                    ),
                    "main_y": (
                        float(main_layout[node_id][1])
                        if node_id in main_layout
                        else np.nan
                    ),
                    "detail_x": float(detail_layout[node_id][0]),
                    "detail_y": float(detail_layout[node_id][1]),
                }
            )
        representative_rows.extend(
            _representative_papers(
                episode=episode,
                graph=display_graph,
                detail_ids=detail_ids,
                main_ids=main_ids,
                maximum=int(
                    config["graph"][
                        "maximum_representative_papers_per_topic"
                    ]
                ),
            )
        )
        summary: Dict[str, Any] = {
            "episode_id": str(episode["episode_id"]),
            "domain": domain,
            "stage_index": int(graph["stage_index"]),
            "stage": str(graph["stage"]),
            "stage_label": str(graph["stage_label"]),
            "start_year": int(graph["start_year"]),
            "end_year": int(graph["end_year"]),
            "increment_start_year": int(graph["increment_start_year"]),
            "increment_end_year": int(graph["increment_end_year"]),
            "cumulative_snapshot": bool(graph["cumulative_snapshot"]),
            "paper_count": int(len(graph["papers"])),
            "increment_paper_count": int(
                graph["papers"]["publication_year"]
                .between(
                    int(graph["increment_start_year"]),
                    int(graph["increment_end_year"]),
                    inclusive="both",
                )
                .sum()
            ),
            "valid_reference_papers": int(
                graph["papers"]["referenced_works"]
                .map(lambda value: len(_reference_list(value)) >= 2)
                .sum()
            ),
            "full_topic_count": int(len(graph["nodes"])),
            "full_topic_edge_count": int(len(graph["edges"])),
        }
        for view, ids in (("main", main_ids), ("detail", detail_ids)):
            transitions = transition_frame[
                transition_frame["display_scope"].eq(view)
                & transition_frame["stage_index"].eq(
                    int(graph["stage_index"])
                )
            ]
            active_statuses = {"baseline", "retained", "gained"}
            summary[f"{view}_active_topic_count"] = int(
                len(set(ids) & active_nodes)
            )
            summary[f"{view}_active_edge_count"] = int(
                transitions["status"].isin(active_statuses).sum()
            )
            summary[f"{view}_retained_edge_count"] = int(
                transitions["status"].eq("retained").sum()
            )
            summary[f"{view}_gained_edge_count"] = int(
                transitions["status"].eq("gained").sum()
            )
            summary[f"{view}_lost_edge_count"] = int(
                transitions["status"].eq("lost").sum()
            )
        summary_rows.append(summary)
        previous_nodes = active_nodes
    return (
        pd.DataFrame(node_rows),
        active_frame,
        transition_frame,
        pd.DataFrame(summary_rows),
        pd.DataFrame(representative_rows),
    )


# ============================================================================
# Frozen v6.1 indicator summaries
# ============================================================================


def _angle_lookup() -> Mapping[str, str]:
    return {
        feature: angle
        for angle, features in ANGLE_FEATURES.items()
        for feature in features
    }


def _oriented_year_percentiles(features: pd.DataFrame) -> pd.DataFrame:
    """Orient and rank each indicator by year across ten candidate domains."""
    frame = features.copy()
    for feature in PRIMARY_FEATURES:
        values = pd.to_numeric(frame[feature], errors="coerce")
        oriented = values * int(FEATURE_DIRECTION[feature])
        frame[f"{feature}__oriented_percentile"] = oriented.groupby(
            frame["publication_year"]
        ).rank(method="average", pct=True)
    return frame


def _indicator_window_rows(
    episode: Mapping[str, Any],
    features: pd.DataFrame,
    config: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Summarize all eight indicators in the four graph windows."""
    rows: List[Dict[str, Any]] = []
    angle_lookup = _angle_lookup()
    domain = str(episode["domain"])
    for window in _stage_windows(episode, config):
        in_window = features[
            features["domain"].eq(domain)
            & features["publication_year"].between(
                int(window["start_year"]),
                int(window["end_year"]),
                inclusive="both",
            )
        ]
        for feature in PRIMARY_FEATURES:
            values = pd.to_numeric(
                in_window[f"{feature}__oriented_percentile"],
                errors="coerce",
            ).dropna()
            rows.append(
                {
                    "episode_id": str(episode["episode_id"]),
                    "domain": domain,
                    **window,
                    "feature": feature,
                    "feature_label": FEATURE_LABELS[feature],
                    "angle": angle_lookup[feature],
                    "n_scope_papers": int(len(in_window)),
                    "n_valid": int(len(values)),
                    "coverage": float(len(values) / max(len(in_window), 1)),
                    "oriented_percentile_q25": (
                        float(values.quantile(0.25))
                        if len(values)
                        else np.nan
                    ),
                    "oriented_percentile_median": (
                        float(values.median()) if len(values) else np.nan
                    ),
                    "oriented_percentile_q75": (
                        float(values.quantile(0.75))
                        if len(values)
                        else np.nan
                    ),
                }
            )
    return rows


def _selection_score_rows(
    summaries: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    settings = config["indicators"]
    rows: List[Dict[str, Any]] = []
    for (episode_id, domain, feature), group in summaries.groupby(
        ["episode_id", "domain", "feature"],
        sort=True,
    ):
        ordered = group.sort_values("stage_index", kind="stable")
        medians = ordered["oriented_percentile_median"].to_numpy(dtype=float)
        eligible = bool(
            len(ordered) == len(STAGE_KEYS)
            and np.isfinite(medians).all()
            and ordered["n_valid"].min()
            >= int(settings["minimum_valid_per_window"])
            and ordered["coverage"].min()
            >= float(settings["minimum_coverage_per_window"])
        )
        change_range = (
            float(np.max(medians) - np.min(medians))
            if eligible
            else float("nan")
        )
        adjacent = (
            float(np.max(np.abs(np.diff(medians))))
            if eligible
            else float("nan")
        )
        score = (
            float(settings["range_weight"]) * change_range
            + float(settings["adjacent_change_weight"]) * adjacent
            if eligible
            else float("nan")
        )
        rows.append(
            {
                "episode_id": str(episode_id),
                "domain": str(domain),
                "feature": str(feature),
                "feature_label": FEATURE_LABELS[str(feature)],
                "eligible": eligible,
                "minimum_n_valid": int(ordered["n_valid"].min()),
                "minimum_coverage": float(ordered["coverage"].min()),
                "change_range": change_range,
                "maximum_adjacent_change": adjacent,
                "change_score": score,
            }
        )
    return pd.DataFrame(rows)


def _select_indicators(
    summaries: pd.DataFrame,
    config: Mapping[str, Any],
    frozen: Mapping[str, Any] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the frozen 4/5/5/4 display lists to registered score rows."""
    frozen = frozen or _load_frozen_selection(config)
    selection = _selection_score_rows(summaries, config)
    selection["rank_by_change"] = pd.Series(
        pd.NA,
        index=selection.index,
        dtype="Int64",
    )
    selection["display_rank"] = pd.Series(
        pd.NA,
        index=selection.index,
        dtype="Int64",
    )
    selection["selected"] = False
    selection["selected_count_for_domain"] = 0
    for episode_id, group in selection.groupby("episode_id", sort=True):
        eligible = group.loc[group["eligible"]].sort_values(
            ["change_score", "feature"],
            ascending=[False, True],
            kind="stable",
        )
        for rank, index in enumerate(eligible.index, start=1):
            selection.loc[index, "rank_by_change"] = rank
    for case in frozen["cases"]:
        episode_id = str(case["episode_id"])
        features = [str(value) for value in case["features"]]
        domain_mask = selection["episode_id"].eq(episode_id)
        for display_rank, feature in enumerate(features, start=1):
            mask = domain_mask & selection["feature"].eq(feature)
            if int(mask.sum()) != 1:
                raise ValueError(
                    f"Frozen indicator not found: {episode_id}/{feature}"
                )
            if not bool(selection.loc[mask, "eligible"].iloc[0]):
                raise ValueError(
                    f"Frozen indicator no longer passes gates: "
                    f"{episode_id}/{feature}"
                )
            selection.loc[mask, "selected"] = True
            selection.loc[mask, "display_rank"] = display_rank
        selection.loc[
            domain_mask,
            "selected_count_for_domain",
        ] = len(features)
    selection["selection_rule"] = (
        "frozen_field_specific_change_ranking_no_annual_or_effect_feedback"
    )
    trajectories = summaries.merge(
        selection[
            [
                "episode_id",
                "domain",
                "feature",
                "eligible",
                "selected",
                "rank_by_change",
                "display_rank",
                "change_range",
                "maximum_adjacent_change",
                "change_score",
            ]
        ],
        on=["episode_id", "domain", "feature"],
        how="left",
        validate="many_to_one",
    )
    return selection, trajectories


def _annual_indicator_rows(
    *,
    episode: Mapping[str, Any],
    selected_features: Sequence[str],
    features: pd.DataFrame,
    config: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    settings = config["indicators"]
    domain = str(episode["domain"])
    event_start = int(episode["start_year"])
    minimum_n = int(settings["minimum_annual_valid"])
    start_offset = int(settings["annual_start_offset"])
    end_offset = int(settings["annual_end_offset"])
    angle_lookup = _angle_lookup()
    rows: List[Dict[str, Any]] = []
    for display_rank, feature in enumerate(selected_features, start=1):
        feature_rows: List[Dict[str, Any]] = []
        for event_time in range(start_offset, end_offset + 1):
            publication_year = event_start + event_time
            scope = features[
                features["domain"].eq(domain)
                & features["publication_year"].eq(publication_year)
            ]
            values = pd.to_numeric(
                scope[f"{feature}__oriented_percentile"],
                errors="coerce",
            ).dropna()
            n_scope = int(len(scope))
            n_valid = int(len(values))
            if n_scope == 0:
                missing_reason = "no_scope_papers"
            elif n_valid < minimum_n:
                missing_reason = f"n_valid_below_{minimum_n}"
            else:
                missing_reason = ""
            eligible = not missing_reason
            raw_q25 = float(values.quantile(0.25)) if n_valid else np.nan
            raw_median = float(values.median()) if n_valid else np.nan
            raw_q75 = float(values.quantile(0.75)) if n_valid else np.nan
            feature_rows.append(
                {
                    "episode_id": str(episode["episode_id"]),
                    "domain": domain,
                    "event_time": int(event_time),
                    "publication_year": int(publication_year),
                    "feature": feature,
                    "feature_label": FEATURE_LABELS[feature],
                    "angle": angle_lookup[feature],
                    "display_rank": int(display_rank),
                    "n_scope_papers": n_scope,
                    "n_valid": n_valid,
                    "coverage": float(n_valid / max(n_scope, 1)),
                    "eligible": eligible,
                    "missing_reason": missing_reason,
                    "oriented_percentile_q25_raw": raw_q25,
                    "oriented_percentile_median_raw": raw_median,
                    "oriented_percentile_q75_raw": raw_q75,
                }
            )
        pre_medians = [
            row["oriented_percentile_median_raw"]
            for row in feature_rows
            if row["event_time"] in {-3, -2, -1} and row["eligible"]
        ]
        baseline = (
            float(np.median(pre_medians))
            if len(pre_medians) == 3
            else float("nan")
        )
        for row in feature_rows:
            row["pre_yearly_median_baseline"] = baseline
            for quantile in ("q25", "median", "q75"):
                raw_value = row[f"oriented_percentile_{quantile}_raw"]
                row[f"delta_{quantile}"] = (
                    float(raw_value - baseline)
                    if row["eligible"]
                    and np.isfinite(raw_value)
                    and np.isfinite(baseline)
                    else np.nan
                )
            rows.append(row)
    return rows


def _trajectory_display_scales(
    annual: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Freeze honest feature-specific display limits for annual trajectories.

    The plotted values are unchanged. Each limit is symmetric around zero,
    covers the largest annual median with configured headroom, and also covers
    the configured absolute IQR quantile. Limits are rounded upward and
    bounded by explicit minimum and maximum values.
    """
    settings = config["indicators"]["trajectory_display_scale"]
    minimum = float(settings["minimum_limit"])
    maximum = float(settings["maximum_limit"])
    step = float(settings["round_to"])
    iqr_quantile = float(settings["iqr_absolute_quantile"])
    median_headroom = float(settings["median_headroom"])
    if not 0.0 < minimum <= maximum:
        raise ValueError("Invalid trajectory display-scale bounds")
    if step <= 0.0 or not 0.0 <= iqr_quantile <= 1.0:
        raise ValueError("Invalid trajectory display-scale rule")
    rows: List[Dict[str, Any]] = []
    groups = annual.groupby(
        ["episode_id", "domain", "feature", "display_rank"],
        sort=True,
        dropna=False,
    )
    for keys, group in groups:
        eligible = group[group["eligible"].astype(bool)]
        medians = (
            pd.to_numeric(eligible["delta_median"], errors="coerce")
            .abs()
            .dropna()
            .to_numpy(dtype=float)
        )
        iqr_values = pd.concat(
            [
                pd.to_numeric(eligible["delta_q25"], errors="coerce").abs(),
                pd.to_numeric(eligible["delta_q75"], errors="coerce").abs(),
            ],
            ignore_index=True,
        ).dropna().to_numpy(dtype=float)
        median_extent = float(medians.max()) if len(medians) else 0.0
        iqr_extent = (
            float(np.quantile(iqr_values, iqr_quantile))
            if len(iqr_values)
            else 0.0
        )
        raw_limit = max(
            median_extent + median_headroom,
            iqr_extent,
            minimum,
        )
        rounded = math.ceil((raw_limit - 1e-12) / step) * step
        display_limit = float(np.clip(rounded, minimum, maximum))
        q25 = pd.to_numeric(eligible["delta_q25"], errors="coerce")
        q75 = pd.to_numeric(eligible["delta_q75"], errors="coerce")
        median = pd.to_numeric(eligible["delta_median"], errors="coerce")
        rows.append(
            {
                "episode_id": str(keys[0]),
                "domain": str(keys[1]),
                "feature": str(keys[2]),
                "display_rank": int(keys[3]),
                "scale_mode": str(settings["mode"]),
                "display_limit": display_limit,
                "display_label": f"±{int(round(display_limit * 100))} pp",
                "minimum_limit": minimum,
                "maximum_limit": maximum,
                "round_to": step,
                "iqr_absolute_quantile": iqr_quantile,
                "median_headroom": median_headroom,
                "median_absolute_extent": median_extent,
                "iqr_absolute_quantile_extent": iqr_extent,
                "median_clipped_count": int(
                    median.abs().gt(display_limit).sum()
                ),
                "iqr_bound_clipped_count": int(
                    q25.abs().gt(display_limit).sum()
                    + q75.abs().gt(display_limit).sum()
                ),
                "eligible_annual_slots": int(len(eligible)),
                "unit": str(settings["unit"]),
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        ["domain", "display_rank"],
        kind="stable",
    )
    frame["domain_shared_display_limit"] = frame.groupby(
        ["episode_id", "domain"],
        sort=False,
    )["display_limit"].transform("max")
    frame["domain_shared_display_label"] = frame[
        "domain_shared_display_limit"
    ].map(lambda value: f"±{int(round(float(value) * 100))} pp")
    return frame


def _bootstrap_year_medians(
    values: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    indices = rng.integers(
        0,
        len(values),
        size=(draws, len(values)),
        endpoint=False,
    )
    return np.median(values[indices], axis=1)


def _indicator_effect_rows(
    *,
    episode: Mapping[str, Any],
    selected_features: Sequence[str],
    features: pd.DataFrame,
    config: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    settings = config["bootstrap"]
    minimum_n = int(config["indicators"]["minimum_annual_valid"])
    draws = int(settings["draws"])
    confidence = float(settings["confidence_level"])
    alpha = (1.0 - confidence) / 2.0
    domain = str(episode["domain"])
    event_start = int(episode["start_year"])
    pre_times = (-3, -2, -1)
    late_times = (6, 7, 8)
    effect_rows: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []
    for display_rank, feature in enumerate(selected_features, start=1):
        values_by_time: Dict[int, np.ndarray] = {}
        for event_time in (*pre_times, *late_times):
            year = event_start + event_time
            values = pd.to_numeric(
                features.loc[
                    features["domain"].eq(domain)
                    & features["publication_year"].eq(year),
                    f"{feature}__oriented_percentile",
                ],
                errors="coerce",
            ).dropna().to_numpy(dtype=float)
            values_by_time[event_time] = values
        eligible = all(
            len(values_by_time[event_time]) >= minimum_n
            for event_time in (*pre_times, *late_times)
        )
        if not eligible:
            effect_rows.append(
                {
                    "episode_id": str(episode["episode_id"]),
                    "domain": domain,
                    "feature": feature,
                    "feature_label": FEATURE_LABELS[feature],
                    "display_rank": int(display_rank),
                    "effect": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "n_pre": int(
                        sum(len(values_by_time[t]) for t in pre_times)
                    ),
                    "n_late": int(
                        sum(len(values_by_time[t]) for t in late_times)
                    ),
                    "eligible": False,
                    "missing_reason": "one_or_more_years_below_minimum_n",
                    "bootstrap_draws": 0,
                    "bootstrap_seed": pd.NA,
                    "confidence_level": confidence,
                }
            )
            continue
        pre_point = np.mean(
            [np.median(values_by_time[t]) for t in pre_times]
        )
        late_point = np.mean(
            [np.median(values_by_time[t]) for t in late_times]
        )
        point_effect = float(late_point - pre_point)
        seed = stable_seed(
            "fig1-year-stratified-bootstrap",
            settings["seed"],
            episode["episode_id"],
            feature,
        )
        rng = np.random.default_rng(seed)
        pre_draws = np.mean(
            [
                _bootstrap_year_medians(values_by_time[t], draws, rng)
                for t in pre_times
            ],
            axis=0,
        )
        late_draws = np.mean(
            [
                _bootstrap_year_medians(values_by_time[t], draws, rng)
                for t in late_times
            ],
            axis=0,
        )
        effects = late_draws - pre_draws
        ci_low, ci_high = np.quantile(effects, [alpha, 1.0 - alpha])
        effect_rows.append(
            {
                "episode_id": str(episode["episode_id"]),
                "domain": domain,
                "feature": feature,
                "feature_label": FEATURE_LABELS[feature],
                "display_rank": int(display_rank),
                "pre_equal_year_median": float(pre_point),
                "late_equal_year_median": float(late_point),
                "effect": point_effect,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "n_pre": int(
                    sum(len(values_by_time[t]) for t in pre_times)
                ),
                "n_late": int(
                    sum(len(values_by_time[t]) for t in late_times)
                ),
                "pre_years": "|".join(
                    str(event_start + value) for value in pre_times
                ),
                "late_years": "|".join(
                    str(event_start + value) for value in late_times
                ),
                "eligible": True,
                "missing_reason": "",
                "bootstrap_draws": draws,
                "bootstrap_seed": int(seed),
                "confidence_level": confidence,
                "bootstrap_method": (
                    "resample_papers_within_publication_year_then_equal_weight_"
                    "mean_of_three_year_specific_medians"
                ),
                "interval_role": (
                    "post_selection_descriptive_not_confirmatory"
                ),
            }
        )
        sample_rows.extend(
            {
                "episode_id": str(episode["episode_id"]),
                "domain": domain,
                "feature": feature,
                "display_rank": int(display_rank),
                "bootstrap_index": int(index),
                "effect": float(value),
                "bootstrap_seed": int(seed),
            }
            for index, value in enumerate(effects)
        )
    return effect_rows, sample_rows


def _candidate_case_refresh_audit(
    *,
    episodes: Sequence[Mapping[str, Any]],
    features: pd.DataFrame,
    graph_screen: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Audit all registered episodes using point effects without bootstrapping."""
    minimum_n = int(config["indicators"]["minimum_annual_valid"])
    threshold = float(
        config["indicator_display"]["minimum_absolute_late_pre_effect"]
    )
    graph_lookup = graph_screen.set_index("episode_id").to_dict("index")
    rows: List[Dict[str, Any]] = []
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        domain = str(episode["domain"])
        summaries = pd.DataFrame(
            _indicator_window_rows(episode, features, config)
        )
        scores = _selection_score_rows(summaries, config).copy()
        scores["rank_by_change"] = pd.Series(
            pd.NA,
            index=scores.index,
            dtype="Int64",
        )
        eligible_scores = scores[scores["eligible"].astype(bool)].sort_values(
            ["change_score", "feature"],
            ascending=[False, True],
            kind="stable",
        )
        for rank, index in enumerate(eligible_scores.index, start=1):
            scores.loc[index, "rank_by_change"] = rank
        scores = scores.sort_values(
            ["eligible", "rank_by_change", "feature"],
            ascending=[False, True, True],
            kind="stable",
        )
        event_start = int(episode["start_year"])
        for score in scores.itertuples(index=False):
            feature = str(score.feature)
            values_by_time: Dict[int, np.ndarray] = {}
            for event_time in (-3, -2, -1, 6, 7, 8):
                values_by_time[event_time] = pd.to_numeric(
                    features.loc[
                        features["domain"].eq(domain)
                        & features["publication_year"].eq(
                            event_start + event_time
                        ),
                        f"{feature}__oriented_percentile",
                    ],
                    errors="coerce",
                ).dropna().to_numpy(dtype=float)
            effect_eligible = all(
                len(values_by_time[value]) >= minimum_n
                for value in values_by_time
            )
            effect = (
                float(
                    np.mean(
                        [
                            np.median(values_by_time[value])
                            for value in (6, 7, 8)
                        ]
                    )
                    - np.mean(
                        [
                            np.median(values_by_time[value])
                            for value in (-3, -2, -1)
                        ]
                    )
                )
                if effect_eligible
                else float("nan")
            )
            graph_row = graph_lookup[episode_id]
            rank = (
                int(score.rank_by_change)
                if not pd.isna(score.rank_by_change)
                else None
            )
            within_top_four = bool(
                bool(score.eligible)
                and rank is not None
                and rank <= 4
            )
            rows.append(
                {
                    "episode_id": episode_id,
                    "episode_label": str(episode["label"]),
                    "domain": domain,
                    "graph_eligible": bool(graph_row["eligible"]),
                    "graph_change_score": float(
                        graph_row["graph_change_score"]
                    ),
                    "feature": feature,
                    "feature_label": FEATURE_LABELS[feature],
                    "rank_by_four_window_change": rank,
                    "four_window_change_score": (
                        float(score.change_score)
                        if np.isfinite(score.change_score)
                        else np.nan
                    ),
                    "late_pre_point_effect": effect,
                    "effect_eligible": effect_eligible,
                    "passes_display_effect_threshold": bool(
                        effect_eligible and abs(effect) >= threshold
                    ),
                    "within_top_four_change_rank": within_top_four,
                    "used_for_descriptive_refresh": episode_id
                    == str(config["selection"]["refresh_episode_id"]),
                }
            )
    frame = pd.DataFrame(rows)
    top_four = frame["within_top_four_change_rank"].astype(bool)
    counts = (
        frame[top_four]
        .groupby("episode_id")["passes_display_effect_threshold"]
        .sum()
        .to_dict()
    )
    frame["top_four_passing_effect_count"] = frame["episode_id"].map(
        counts
    ).fillna(0).astype(int)
    return frame.sort_values(
        ["episode_id", "rank_by_four_window_change"],
        kind="stable",
    ).reset_index(drop=True)


def _indicator_display_filter(
    annual: pd.DataFrame,
    effects: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Choose visibly changing indicators for the paired panels b/c only."""
    settings = config["indicator_display"]
    effect_threshold = float(
        settings["minimum_absolute_late_pre_effect"]
    )
    range_threshold = float(settings["minimum_annual_peak_to_peak"])
    minimum = int(settings["minimum_per_domain"])
    maximum = int(settings["maximum_per_domain"])
    require_effect = bool(
        settings["paired_panels_require_late_pre_effect"]
    )
    if effect_threshold <= 0 or range_threshold <= 0:
        raise ValueError("Indicator display thresholds must be positive")
    if not 1 <= minimum <= maximum:
        raise ValueError("Invalid per-domain indicator display bounds")
    rows: List[Dict[str, Any]] = []
    for keys, group in annual.groupby(
        ["episode_id", "domain", "feature", "display_rank"],
        sort=True,
    ):
        medians = pd.to_numeric(
            group.loc[
                group["eligible"].astype(bool),
                "delta_median",
            ],
            errors="coerce",
        ).dropna()
        annual_range = (
            float(medians.max() - medians.min())
            if len(medians)
            else float("nan")
        )
        effect_row = effects[
            effects["episode_id"].eq(str(keys[0]))
            & effects["feature"].eq(str(keys[2]))
        ]
        if len(effect_row) != 1:
            raise ValueError(
                "Expected exactly one effect row for display filtering: "
                f"{keys[0]} / {keys[2]}"
            )
        effect = float(effect_row.iloc[0]["effect"])
        passes_effect = bool(abs(effect) >= effect_threshold)
        passes_range = bool(
            np.isfinite(annual_range) and annual_range >= range_threshold
        )
        effect_ratio = abs(effect) / effect_threshold
        range_ratio = (
            annual_range / range_threshold
            if np.isfinite(annual_range)
            else 0.0
        )
        rows.append(
            {
                "episode_id": str(keys[0]),
                "domain": str(keys[1]),
                "feature": str(keys[2]),
                "display_rank": int(keys[3]),
                "absolute_late_pre_effect": abs(effect),
                "annual_peak_to_peak": annual_range,
                "minimum_absolute_late_pre_effect": effect_threshold,
                "minimum_annual_peak_to_peak": range_threshold,
                "passes_effect_threshold": passes_effect,
                "passes_trajectory_threshold": passes_range,
                "primary_display_pass": (
                    passes_effect
                    if require_effect
                    else passes_effect or passes_range
                ),
                "display_salience_score": max(effect_ratio, range_ratio),
                "display": False,
                "display_reason": "below_display_threshold",
                "display_only_not_feature_selection": True,
            }
        )
    frame = pd.DataFrame(rows)
    for _, group in frame.groupby(["episode_id", "domain"], sort=True):
        ranked = group.sort_values(
            ["display_salience_score", "display_rank", "feature"],
            ascending=[False, True, True],
            kind="stable",
        )
        primary = ranked[ranked["primary_display_pass"].astype(bool)]
        chosen = list(primary.index[:maximum])
        if len(chosen) < minimum:
            additions = [
                index
                for index in ranked.index
                if index not in chosen
            ][: minimum - len(chosen)]
            chosen.extend(additions)
        frame.loc[chosen, "display"] = True
        for index in chosen:
            reasons: List[str] = []
            if bool(frame.loc[index, "passes_effect_threshold"]):
                reasons.append("absolute_late_pre_effect")
            if bool(frame.loc[index, "passes_trajectory_threshold"]):
                reasons.append("annual_peak_to_peak")
            if not reasons:
                reasons.append("minimum_context_fill")
            frame.loc[index, "display_reason"] = "|".join(reasons)
    frame["display_order"] = pd.Series(
        pd.NA,
        index=frame.index,
        dtype="Int64",
    )
    for _, group in frame[frame["display"].astype(bool)].groupby(
        ["episode_id", "domain"],
        sort=True,
    ):
        ordered = group.sort_values("display_rank", kind="stable")
        for order, index in enumerate(ordered.index, start=1):
            frame.loc[index, "display_order"] = order
    return frame.sort_values(
        ["domain", "display_rank"],
        kind="stable",
    ).reset_index(drop=True)


# ============================================================================
# Pipeline
# ============================================================================


def run_descriptive_analysis(
    config: Mapping[str, Any],
    data_dir: Path,
    output_dir: Path,
) -> Mapping[str, Any]:
    """Build graph transitions, annual trajectories, and effect intervals."""
    frozen = _load_frozen_selection(config)
    frozen_lookup = _frozen_case_lookup(frozen)
    topic_labels = _load_topic_labels(config)
    focal_path = data_dir / str(config["data"]["focal_works_file"])
    features_path = data_dir / str(config["data"]["indicator_features_file"])
    focal = pd.read_parquet(focal_path)
    features = pd.read_parquet(features_path)
    missing = sorted(set(PRIMARY_FEATURES) - set(features.columns))
    if missing:
        raise ValueError(f"Frozen indicator columns are missing: {missing}")
    main_domains = set(str(value) for value in config["main_domains"])
    focal = focal[focal["domain"].isin(main_domains)].copy()
    features = features[features["domain"].isin(main_domains)].copy()
    features = _oriented_year_percentiles(features)
    episode_lookup = {
        str(episode["episode_id"]): episode for episode in config["episodes"]
    }
    graph_cache: Dict[str, List[Dict[str, Any]]] = {}
    candidate_rows: List[Dict[str, Any]] = []
    for episode in config["episodes"]:
        domain = str(episode["domain"])
        graphs = _episode_graphs(
            episode,
            focal[focal["domain"].eq(domain)],
            config,
        )
        graph_cache[str(episode["episode_id"])] = graphs
        candidate_rows.append(_episode_score_row(episode, graphs, config))
    screened = _rank_and_select_episodes(
        pd.DataFrame(candidate_rows),
        config,
    )
    case_refresh_audit = _candidate_case_refresh_audit(
        episodes=config["episodes"],
        features=features,
        graph_screen=screened,
        config=config,
    )
    domain_selection = _apply_frozen_domain_selection(screened, frozen)
    selected_ids = [
        str(case["episode_id"])
        for case in sorted(
            frozen["cases"],
            key=lambda value: int(value["selection_rank"]),
        )
    ]
    node_frames: List[pd.DataFrame] = []
    edge_frames: List[pd.DataFrame] = []
    transition_frames: List[pd.DataFrame] = []
    summary_frames: List[pd.DataFrame] = []
    representative_frames: List[pd.DataFrame] = []
    indicator_rows: List[Dict[str, Any]] = []
    annual_rows: List[Dict[str, Any]] = []
    effect_rows: List[Dict[str, Any]] = []
    bootstrap_rows: List[Dict[str, Any]] = []
    landmark_rows: List[Dict[str, Any]] = []
    for episode_id in selected_ids:
        episode = episode_lookup[episode_id]
        case = frozen_lookup[episode_id]
        domain = str(episode["domain"])
        domain_papers = focal[focal["domain"].eq(domain)]
        nodes, edges, transitions, summaries, representatives = (
            _display_tables(
                episode,
                graph_cache[episode_id],
                domain_papers,
                config,
                topic_labels,
            )
        )
        node_frames.append(nodes)
        edge_frames.append(edges)
        transition_frames.append(transitions)
        summary_frames.append(summaries)
        representative_frames.append(representatives)
        indicator_rows.extend(
            _indicator_window_rows(episode, features, config)
        )
        annual_rows.extend(
            _annual_indicator_rows(
                episode=episode,
                selected_features=case["features"],
                features=features,
                config=config,
            )
        )
        episode_effects, episode_bootstrap = _indicator_effect_rows(
            episode=episode,
            selected_features=case["features"],
            features=features,
            config=config,
        )
        effect_rows.extend(episode_effects)
        bootstrap_rows.extend(episode_bootstrap)
        landmarks = domain_papers[
            domain_papers["is_landmark"].eq(1)
            & domain_papers["publication_year"].between(
                int(episode["start_year"]),
                int(episode["end_year"]),
                inclusive="both",
            )
        ]
        landmark_rows.extend(
            {
                "episode_id": episode_id,
                "domain": domain,
                "paper_id": str(row.work_id),
                "publication_year": int(row.publication_year),
                "title": str(row.title),
                "topic_id": str(row.primary_topic_id),
                "topic_name": str(row.primary_topic_name),
            }
            for row in landmarks.itertuples(index=False)
        )
    nodes = pd.concat(node_frames, ignore_index=True)
    edges = pd.concat(edge_frames, ignore_index=True)
    transitions = pd.concat(transition_frames, ignore_index=True)
    snapshots = pd.concat(summary_frames, ignore_index=True)
    representatives = pd.concat(representative_frames, ignore_index=True)
    indicator_summary = pd.DataFrame(indicator_rows)
    indicator_selection, window_trajectories = _select_indicators(
        indicator_summary,
        config,
        frozen,
    )
    annual = pd.DataFrame(annual_rows)
    trajectory_scales = _trajectory_display_scales(annual, config)
    effects = pd.DataFrame(effect_rows)
    indicator_display = _indicator_display_filter(
        annual,
        effects,
        config,
    )
    bootstrap = pd.DataFrame(bootstrap_rows)
    panel_data = output_dir / "panel_data"
    panel_data.mkdir(parents=True, exist_ok=True)
    domain_selection.to_csv(
        panel_data / "domain_selection.csv",
        index=False,
    )
    case_refresh_audit.to_csv(
        panel_data / "case_refresh_audit.csv",
        index=False,
    )
    snapshots.to_csv(panel_data / "snapshot_summary.csv", index=False)
    nodes.to_parquet(panel_data / "snapshot_nodes.parquet", index=False)
    edges.to_parquet(panel_data / "snapshot_edges.parquet", index=False)
    transitions.to_parquet(
        panel_data / "transition_edges.parquet",
        index=False,
    )
    representatives.to_parquet(
        panel_data / "representative_papers.parquet",
        index=False,
    )
    pd.DataFrame(landmark_rows).to_csv(
        panel_data / "landmark_papers.csv",
        index=False,
    )
    indicator_summary.to_csv(
        panel_data / "indicator_window_summary.csv",
        index=False,
    )
    indicator_selection.to_csv(
        panel_data / "indicator_selection.csv",
        index=False,
    )
    window_trajectories.to_csv(
        panel_data / "indicator_trajectories.csv",
        index=False,
    )
    annual.to_csv(
        panel_data / "annual_indicator_trajectories.csv",
        index=False,
    )
    trajectory_scales.to_csv(
        panel_data / "trajectory_display_scales.csv",
        index=False,
    )
    indicator_display.to_csv(
        panel_data / "indicator_display_filter.csv",
        index=False,
    )
    effects.to_csv(panel_data / "indicator_effects.csv", index=False)
    bootstrap.to_parquet(
        panel_data / "indicator_effect_bootstrap.parquet",
        index=False,
    )
    topic_audit = (
        nodes[
            ["domain", "node_id", "topic_name_raw", "topic_label"]
        ]
        .drop_duplicates()
        .sort_values(["domain", "node_id"], kind="stable")
    )
    topic_audit.to_csv(
        panel_data / "topic_label_audit.csv",
        index=False,
    )
    community_label_audit = nodes[
        [
            "episode_id",
            "domain",
            "stage_index",
            "node_id",
            "topic_name_raw",
            "topic_label",
            "active",
            "landmark_topic",
            "landmark_coupling_weight",
            "community_relation",
            "paper_count",
            "display_main",
            "display_detail",
        ]
    ].sort_values(
        [
            "episode_id",
            "stage_index",
            "landmark_topic",
            "landmark_coupling_weight",
            "paper_count",
            "node_id",
        ],
        ascending=[True, True, False, False, False, True],
        kind="stable",
    )
    community_label_audit.to_csv(
        panel_data / "community_label_selection.csv",
        index=False,
    )
    selection_snapshot = {
        "selection_version": frozen["selection_version"],
        "selection_file_sha256": str(config["frozen_selection_sha256"]),
        "selection_fingerprint": _selection_fingerprint(frozen),
        "cases": frozen["cases"],
    }
    write_json(
        panel_data / "frozen_selection_snapshot.json",
        selection_snapshot,
    )
    selected_domains = domain_selection.loc[
        domain_selection["selected"],
        ["selection_rank", "episode_id", "domain", "graph_change_score"],
    ].sort_values("selection_rank")
    manifest = {
        "artifact_kind": "fig1_cumulative_transition_analysis",
        "design_version": config["design_version"],
        "status": "DESCRIPTIVE_SELECTED_CASES",
        "claim_boundary": (
            "The cases and indicators were deliberately selected to make "
            "publication-time graph and indicator changes visible. The "
            "fourth case was refreshed after an eleven-episode descriptive "
            "display audit. The figure is illustrative, not representative, "
            "and does not show that landmark papers caused the changes."
        ),
        "selection_disclosure": {
            "domain_selection_uses_graph_change": True,
            "indicator_selection_uses_indicator_change": True,
            "display_refresh_used_annual_and_bootstrap_contrasts": True,
            "selection_frozen_before_current_rerender": True,
            "future_impact_outcome_used": False,
            "citation_count_used": False,
            "d5_used": False,
            "oof_prediction_used": False,
        },
        "selection_fingerprint": _selection_fingerprint(frozen),
        "selected_domains": selected_domains.to_dict("records"),
        "source_artifacts": {
            "focal_works": {
                "path": str(focal_path.resolve()),
                "sha256": sha256_file(focal_path),
            },
            "indicator_features": {
                "path": str(features_path.resolve()),
                "sha256": sha256_file(features_path),
            },
            "frozen_selection": {
                "path": str(
                    _resolve_project_path(
                        str(config["frozen_selection_file"])
                    ).resolve()
                ),
                "sha256": str(config["frozen_selection_sha256"]),
            },
            "topic_short_labels": {
                "path": str(
                    _resolve_project_path(
                        str(config["topic_short_labels_file"])
                    ).resolve()
                ),
                "sha256": str(config["topic_short_labels_sha256"]),
            },
        },
        "methods": {
            "annual_range": "t-6 through t+8",
            "annual_baseline": (
                "median of the three annual medians at t-3,t-2,t-1"
            ),
            "annual_minimum_valid_n": int(
                config["indicators"]["minimum_annual_valid"]
            ),
            "trajectory_display_scale": {
                "mode": str(
                    config["indicators"]["trajectory_display_scale"]["mode"]
                ),
                "symmetric_about_zero": True,
                "formula": (
                    "ceil_to_step(max(abs annual median + headroom, "
                    "configured quantile of absolute annual IQR bounds, "
                    "minimum)); clipped at configured maximum"
                ),
                "shared_effect_limit": float(
                    config["indicators"]["trajectory_display_scale"][
                        "shared_effect_limit"
                    ]
                ),
            },
            "indicator_display_filter": {
                "role": str(config["indicator_display"]["role"]),
                "rule": (
                    "paired panels require the absolute late-pre effect "
                    "threshold; enforce the configured per-domain minimum "
                    "and maximum by deterministic salience ranking"
                ),
                "minimum_absolute_late_pre_effect": float(
                    config["indicator_display"][
                        "minimum_absolute_late_pre_effect"
                    ]
                ),
                "minimum_annual_peak_to_peak": float(
                    config["indicator_display"][
                        "minimum_annual_peak_to_peak"
                    ]
                ),
                "minimum_per_domain": int(
                    config["indicator_display"]["minimum_per_domain"]
                ),
                "maximum_per_domain": int(
                    config["indicator_display"]["maximum_per_domain"]
                ),
            },
            "community_label_priority": (
                "landmark-bearing topic first; direct bibliographic-coupling "
                "neighbors ranked by coupling weight next; active field "
                "backbone context ranked by paper count last"
            ),
            "effect": "late t+6:t+8 minus pre t-3:t-1",
            "bootstrap_draws": int(config["bootstrap"]["draws"]),
            "bootstrap_method": str(
                config["bootstrap"]["window_year_aggregation"]
            ),
        },
        "row_counts": {
            "focal_papers_main_domains": int(len(focal)),
            "indicator_feature_rows": int(len(features)),
            "candidate_episodes": int(len(domain_selection)),
            "case_refresh_audit_rows": int(len(case_refresh_audit)),
            "selected_episodes": int(domain_selection["selected"].sum()),
            "snapshot_nodes": int(len(nodes)),
            "snapshot_active_edges": int(len(edges)),
            "transition_edges": int(len(transitions)),
            "representative_papers": int(len(representatives)),
            "window_indicator_rows": int(len(window_trajectories)),
            "annual_indicator_rows": int(len(annual)),
            "trajectory_display_scale_rows": int(len(trajectory_scales)),
            "indicator_display_filter_rows": int(len(indicator_display)),
            "displayed_indicator_rows": int(
                indicator_display["display"].astype(bool).sum()
            ),
            "effect_rows": int(len(effects)),
            "bootstrap_rows": int(len(bootstrap)),
        },
    }
    manifest["artifact_id"] = canonical_hash(manifest)
    write_json(output_dir / "analysis_manifest.json", manifest)
    return manifest


__all__ = [
    "_rank_and_select_episodes",
    "_graph_snapshot_windows",
    "_indicator_display_filter",
    "_select_indicators",
    "_stage_windows",
    "_trajectory_display_scales",
    "_weighted_jaccard_distance",
    "run_descriptive_analysis",
]

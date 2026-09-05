#!/usr/bin/env python3
"""Phase 8：构建历史 Claim 伪插入画像、社区组合历史和运行时百分位标尺。"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PROJECT_ROOT / "data" / "claim_graph"
CLAIM_TYPES = ("METHOD", "FINDING", "MECHANISM", "RESOURCE", "THEORY")
CORE_METRICS = (
    "nearest_prior_similarity",
    "mean_top5_similarity",
    "effective_community_count",
    "community_rao_stirling",
    "first_observed_recent_nature_pair_share",
    "community_pair_mean_surprisal",
    "neighbor_induced_density",
    "component_merge_count",
    "newly_connected_neighbor_pair_count",
    "cross_boundary_weight_share",
)
METRIC_DIRECTIONS = {
    "nearest_prior_similarity": "lower_more_structurally_distant",
    "mean_top5_similarity": "lower_more_structurally_distant",
    "effective_community_count": "higher_more_cross_community",
    "community_rao_stirling": "higher_more_cross_community",
    "first_observed_recent_nature_pair_share": "higher_more_atypical",
    "community_pair_mean_surprisal": "higher_more_atypical",
    "neighbor_induced_density": "lower_more_brokerage",
    "component_merge_count": "higher_more_brokerage",
    "newly_connected_neighbor_pair_count": "higher_more_brokerage",
    "cross_boundary_weight_share": "higher_more_cross_boundary",
}

PROFILE_SCHEMA = pa.schema(
    [
        pa.field("claim_id", pa.string(), False),
        pa.field("parent_paper_id", pa.string(), False),
        pa.field("claim_type", pa.string(), False),
        pa.field("publication_date", pa.date32(), False),
        pa.field("neighbor_count", pa.int32(), False),
        pa.field("cross_type_neighbor_count", pa.int32(), False),
        pa.field("cross_type_neighbor_share", pa.float32()),
        pa.field("effective_neighbor_claim_type_count", pa.float32()),
        pa.field("neighbor_claim_type_distribution_json", pa.string(), False),
        pa.field("nearest_prior_claim_id", pa.string()),
        pa.field("nearest_prior_similarity", pa.float32()),
        pa.field("mean_top5_similarity", pa.float32()),
        pa.field("community_count", pa.int32(), False),
        pa.field("dominant_community_id", pa.int32()),
        pa.field("dominant_community_share", pa.float32()),
        pa.field("effective_community_count", pa.float32()),
        pa.field("community_rao_stirling", pa.float32()),
        pa.field("community_weight_distribution_json", pa.string(), False),
        pa.field("community_pair_count", pa.int32(), False),
        pa.field("first_observed_recent_nature_pair_share", pa.float32()),
        pa.field("community_pair_mean_surprisal", pa.float32()),
        pa.field("community_pair_surprisal_defined_count", pa.int32(), False),
        pa.field("neighbor_induced_density", pa.float32()),
        pa.field("components_before", pa.int32(), False),
        pa.field("components_after", pa.int32(), False),
        pa.field("component_merge_count", pa.int32(), False),
        pa.field("newly_connected_neighbor_pair_count", pa.int32(), False),
        pa.field("cross_boundary_weight_share", pa.float32()),
        pa.field("paper_path_supported_neighbor_count", pa.int32(), False),
        pa.field("paper_path_supported_neighbor_share", pa.float32()),
        pa.field("semantic_and_paper_path_agreement_count", pa.int32(), False),
        pa.field("paper_direct_citation_neighbor_count", pa.int32(), False),
        pa.field("paper_two_hop_neighbor_count", pa.int32(), False),
        pa.field("paper_shared_reference_neighbor_count", pa.int32(), False),
    ]
)

PAIR_SCHEMA = pa.schema(
    [
        pa.field("community_a", pa.int32(), False),
        pa.field("community_b", pa.int32(), False),
        pa.field("pair_connector_count", pa.int32(), False),
        pa.field("community_a_claim_count", pa.int32(), False),
        pa.field("community_b_claim_count", pa.int32(), False),
        pa.field("historical_claim_count", pa.int32(), False),
        pa.field("first_observed_date", pa.date32(), False),
    ]
)

_CLAIMS: list[dict[str, Any]] = []
_INCOMING: list[list[tuple[int, float, bool, bool, int, int]]] = []
_COMMUNITIES: list[int | None] = []
_EDGE_KEYS: set[int] = set()
_CENTROIDS: np.ndarray | None = None
_COMMUNITY_ROWS: dict[int, int] = {}
_CLAIM_TYPE_LABELS: list[str] = []
_CLAIM_COUNT = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--claims", type=Path, default=DEFAULT_ROOT / "claim_nodes.parquet"
    )
    parser.add_argument(
        "--edges", type=Path, default=DEFAULT_ROOT / "claim_edges.parquet"
    )
    parser.add_argument(
        "--communities", type=Path, default=DEFAULT_ROOT / "claim_communities.parquet"
    )
    parser.add_argument(
        "--embeddings", type=Path, default=DEFAULT_ROOT / "claim_embedding_matrix.npy"
    )
    parser.add_argument(
        "--embedding-index",
        type=Path,
        default=DEFAULT_ROOT / "claim_embedding_index.parquet",
    )
    parser.add_argument(
        "--graph-index", type=Path, default=DEFAULT_ROOT / "claim_graph_index.sqlite"
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--workers", type=int, default=0, help="0 表示使用全部可用 CPU 核心"
    )
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def build_logger(path: Path, verbose: bool) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("claim_graph.phase8")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(path, encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        logger.addHandler(handler)
    return logger


def require_files(paths: Iterable[Path]) -> None:
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"输入文件不存在：{path}")


def atomic_write_table(
    rows: list[dict[str, Any]], schema: pa.Schema, path: Path
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema),
        temporary,
        compression="zstd",
        use_dictionary=True,
    )
    temporary.replace(path)


def atomic_write_numpy(values: np.ndarray, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, values)
    temporary.replace(path)


def load_claims(path: Path, graph_index: Path) -> list[dict[str, Any]]:
    metadata = {
        str(row["claim_id"]): (str(row["parent_paper_id"]), str(row["claim_type"]))
        for row in pq.read_table(
            path, columns=["claim_id", "parent_paper_id", "claim_type"]
        ).to_pylist()
    }
    connection = sqlite3.connect(f"file:{graph_index}?mode=ro", uri=True)
    try:
        graph_rows = connection.execute(
            "SELECT claim_row, claim_id, parent_paper_id, claim_type, publication_date "
            "FROM claim_nodes ORDER BY claim_row"
        ).fetchall()
    finally:
        connection.close()
    claims = []
    for claim_row, claim_id, paper_id, claim_type, publication_date in graph_rows:
        claim_id = str(claim_id)
        if claim_id not in metadata:
            raise ValueError(f"图索引中的 Claim 不存在于节点表：{claim_id}")
        claims.append(
            {
                "claim_row": int(claim_row),
                "claim_id": claim_id,
                "paper_id": str(paper_id),
                "claim_type": str(claim_type),
                "date": np.datetime64(str(publication_date), "D").astype(object),
            }
        )
    if [row["claim_row"] for row in claims] != list(range(len(claims))):
        raise ValueError("claim_graph_index.sqlite 的 claim_row 必须连续且从 0 开始")
    if len(claims) != len(metadata):
        raise ValueError("Claim 节点表与 Phase 7 图索引行数不一致")
    return claims


def load_communities(path: Path, claims: list[dict[str, Any]]) -> list[int | None]:
    values = {
        str(row["claim_id"]): (
            None if row["community_id"] is None else int(row["community_id"])
        )
        for row in pq.read_table(path, columns=["claim_id", "community_id"]).to_pylist()
    }
    if set(values) != {row["claim_id"] for row in claims}:
        raise ValueError("社区表与 Claim 节点集合不一致")
    return [values[row["claim_id"]] for row in claims]


def load_edges(
    path: Path, claims: list[dict[str, Any]]
) -> tuple[list[list[tuple[int, float, bool, bool, int, int]]], set[int]]:
    positions = {row["claim_id"]: index for index, row in enumerate(claims)}
    incoming: list[list[tuple[int, float, bool, bool, int, int]]] = [[] for _ in claims]
    edge_keys: set[int] = set()
    columns = [
        "earlier_claim_id",
        "later_claim_id",
        "cosine_similarity",
        "from_semantic",
        "from_paper_path",
        "paper_direct_citation",
        "paper_directed_two_hop_count",
        "paper_shared_reference_count",
    ]
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=100_000, columns=columns):
        for row in batch.to_pylist():
            earlier = positions[str(row["earlier_claim_id"])]
            later = positions[str(row["later_claim_id"])]
            if not bool(row["from_semantic"]):
                raise ValueError("Phase 8 不接受 paper_path_only Claim 边")
            incoming[later].append(
                (
                    earlier,
                    float(row["cosine_similarity"]),
                    bool(row["from_paper_path"]),
                    bool(row["paper_direct_citation"]),
                    int(row["paper_directed_two_hop_count"]),
                    int(row["paper_shared_reference_count"]),
                )
            )
            left, right = sorted((earlier, later))
            edge_keys.add(left * len(claims) + right)
    for values in incoming:
        values.sort(key=lambda item: (-item[1], item[0]))
    return incoming, edge_keys


def load_vectors(
    matrix_path: Path, index_path: Path, claims: list[dict[str, Any]]
) -> np.ndarray:
    matrix = np.load(matrix_path, mmap_mode="r")
    rows = pq.read_table(index_path, columns=["claim_id", "claim_row"]).to_pylist()
    positions = {str(row["claim_id"]): int(row["claim_row"]) for row in rows}
    order = np.asarray([positions[row["claim_id"]] for row in claims], dtype=np.int64)
    if matrix.ndim != 2 or len(matrix) != len(rows):
        raise ValueError("Claim embedding 矩阵与索引无法对齐")
    return np.asarray(matrix[order], dtype=np.float32)


def build_centroids(
    vectors: np.ndarray, communities: list[int | None]
) -> tuple[np.ndarray, list[dict[str, Any]], dict[int, int]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, community_id in enumerate(communities):
        if community_id is not None:
            grouped[community_id].append(index)
    records: list[dict[str, Any]] = []
    centroids = []
    community_rows = {}
    for row_index, community_id in enumerate(sorted(grouped)):
        centroid = np.asarray(
            vectors[grouped[community_id]].mean(axis=0), dtype=np.float32
        )
        norm = float(np.linalg.norm(centroid))
        if norm == 0.0:
            raise ValueError(f"社区 {community_id} 的中心向量为零")
        centroid /= norm
        community_rows[community_id] = row_index
        centroids.append(centroid)
        records.append(
            {
                "community_id": community_id,
                "centroid_row": row_index,
                "claim_count": len(grouped[community_id]),
            }
        )
    return np.asarray(centroids, dtype=np.float32), records, community_rows


def weighted_distribution(
    neighbors: list[tuple[int, float, bool, bool, int, int]], labels: list[Any]
) -> tuple[dict[Any, float], float]:
    weights: dict[Any, float] = defaultdict(float)
    for neighbor, cosine, *_ in neighbors:
        label = labels[neighbor]
        if label is not None:
            weights[label] += max(float(cosine), 0.0)
    total = sum(weights.values())
    if total <= 0.0:
        return {}, 0.0
    return {label: weight / total for label, weight in weights.items()}, total


def connected_components(neighbors: list[int]) -> list[list[int]]:
    parent = {node: node for node in neighbors}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left_index, left in enumerate(neighbors):
        for right in neighbors[left_index + 1 :]:
            a, b = sorted((left, right))
            if a * _CLAIM_COUNT + b in _EDGE_KEYS:
                union(left, right)
    groups: dict[int, list[int]] = defaultdict(list)
    for node in neighbors:
        groups[find(node)].append(node)
    return list(groups.values())


def community_rao(probabilities: dict[int, float]) -> float | None:
    if len(probabilities) < 2 or _CENTROIDS is None:
        return None
    result = 0.0
    ordered = sorted(probabilities)
    for index, left in enumerate(ordered):
        left_vector = _CENTROIDS[_COMMUNITY_ROWS[left]]
        for right in ordered[index + 1 :]:
            similarity = float(np.dot(left_vector, _CENTROIDS[_COMMUNITY_ROWS[right]]))
            distance = float(np.clip(1.0 - similarity, 0.0, 1.0))
            result += 2.0 * probabilities[left] * probabilities[right] * distance
    return result


def static_profile(index: int) -> tuple[dict[str, Any], list[int]]:
    claim = _CLAIMS[index]
    neighbors = _INCOMING[index]
    neighbor_rows = [item[0] for item in neighbors]
    count = len(neighbors)
    type_counts = Counter(_CLAIMS[row]["claim_type"] for row in neighbor_rows)
    type_probabilities, _ = weighted_distribution(neighbors, _CLAIM_TYPE_LABELS)
    community_probabilities, _ = weighted_distribution(neighbors, _COMMUNITIES)
    community_ids = sorted(community_probabilities)
    components = connected_components(neighbor_rows) if neighbor_rows else []
    possible_edges = count * (count - 1) // 2
    internal_edges = sum(
        1
        for left_index, left in enumerate(neighbor_rows)
        for right in neighbor_rows[left_index + 1 :]
        if min(left, right) * _CLAIM_COUNT + max(left, right) in _EDGE_KEYS
    )
    disconnected_pairs = possible_edges - sum(
        len(component) * (len(component) - 1) // 2 for component in components
    )
    cross_type = sum(
        _CLAIMS[row]["claim_type"] != claim["claim_type"] for row in neighbor_rows
    )
    paper_path_count = sum(item[2] for item in neighbors)
    similarities = [item[1] for item in neighbors]
    dominant = (
        max(community_probabilities, key=community_probabilities.get)
        if community_probabilities
        else None
    )
    profile = {
        "claim_id": claim["claim_id"],
        "parent_paper_id": claim["paper_id"],
        "claim_type": claim["claim_type"],
        "publication_date": claim["date"],
        "neighbor_count": count,
        "cross_type_neighbor_count": cross_type,
        "cross_type_neighbor_share": cross_type / count if count else None,
        "effective_neighbor_claim_type_count": (
            1.0 / sum(value * value for value in type_probabilities.values())
            if type_probabilities
            else None
        ),
        "neighbor_claim_type_distribution_json": json.dumps(
            dict(sorted(type_counts.items())), ensure_ascii=False
        ),
        "nearest_prior_claim_id": (
            _CLAIMS[neighbors[0][0]]["claim_id"] if neighbors else None
        ),
        "nearest_prior_similarity": similarities[0] if similarities else None,
        "mean_top5_similarity": (
            float(np.mean(similarities[:5])) if similarities else None
        ),
        "community_count": len(community_ids),
        "dominant_community_id": dominant,
        "dominant_community_share": (
            community_probabilities.get(dominant) if dominant is not None else None
        ),
        "effective_community_count": (
            1.0 / sum(value * value for value in community_probabilities.values())
            if community_probabilities
            else None
        ),
        "community_rao_stirling": community_rao(community_probabilities),
        "community_weight_distribution_json": json.dumps(
            {str(key): value for key, value in sorted(community_probabilities.items())},
            ensure_ascii=False,
        ),
        "community_pair_count": len(community_ids) * (len(community_ids) - 1) // 2,
        "first_observed_recent_nature_pair_share": None,
        "community_pair_mean_surprisal": None,
        "community_pair_surprisal_defined_count": 0,
        "neighbor_induced_density": (
            internal_edges / possible_edges if possible_edges else None
        ),
        "components_before": len(components),
        "components_after": 1 if count else 0,
        "component_merge_count": max(len(components) - 1, 0),
        "newly_connected_neighbor_pair_count": disconnected_pairs,
        "cross_boundary_weight_share": (
            1.0 - max(community_probabilities.values())
            if community_probabilities
            else None
        ),
        "paper_path_supported_neighbor_count": paper_path_count,
        "paper_path_supported_neighbor_share": (
            paper_path_count / count if count else None
        ),
        "semantic_and_paper_path_agreement_count": paper_path_count,
        "paper_direct_citation_neighbor_count": sum(item[3] for item in neighbors),
        "paper_two_hop_neighbor_count": sum(item[4] > 0 for item in neighbors),
        "paper_shared_reference_neighbor_count": sum(item[5] > 0 for item in neighbors),
    }
    return profile, community_ids


def profile_chunk(indices: list[int]) -> list[tuple[int, dict[str, Any], list[int]]]:
    return [(index, *static_profile(index)) for index in indices]


def build_static_profiles(
    workers: int, chunk_size: int, logger: logging.Logger
) -> tuple[list[dict[str, Any]], list[list[int]]]:
    count = len(_CLAIMS)
    profiles: list[dict[str, Any] | None] = [None] * count
    community_sets: list[list[int] | None] = [None] * count
    chunks = [
        list(range(start, min(start + chunk_size, count)))
        for start in range(0, count, chunk_size)
    ]
    completed = 0
    if workers == 1:
        results = map(profile_chunk, chunks)
        for result in results:
            for index, profile, communities in result:
                profiles[index], community_sets[index] = profile, communities
            completed += len(result)
            logger.info("[步骤 3/7] 静态画像进度=%d/%d", completed, count)
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=get_context("fork")
        ) as executor:
            futures = [executor.submit(profile_chunk, chunk) for chunk in chunks]
            for future in as_completed(futures):
                result = future.result()
                for index, profile, communities in result:
                    profiles[index], community_sets[index] = profile, communities
                completed += len(result)
                logger.info("[步骤 3/7] 静态画像进度=%d/%d", completed, count)
    return (
        [profile for profile in profiles if profile is not None],
        [communities for communities in community_sets if communities is not None],
    )


def community_pairs(community_ids: list[int]) -> list[tuple[int, int]]:
    return [
        (left, right)
        for left_index, left in enumerate(community_ids)
        for right in community_ids[left_index + 1 :]
    ]


def add_temporal_pair_metrics(
    profiles: list[dict[str, Any]],
    community_sets: list[list[int]],
    logger: logging.Logger,
) -> tuple[dict[tuple[int, int], int], dict[int, int], dict[tuple[int, int], Any], int]:
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    community_counts: dict[int, int] = defaultdict(int)
    first_dates: dict[tuple[int, int], Any] = {}
    historical_claim_count = 0
    order = sorted(
        range(len(_CLAIMS)),
        key=lambda index: (_CLAIMS[index]["date"], _CLAIMS[index]["claim_id"]),
    )
    start = 0
    while start < len(order):
        end = start + 1
        date = _CLAIMS[order[start]]["date"]
        while end < len(order) and _CLAIMS[order[end]]["date"] == date:
            end += 1
        for index in order[start:end]:
            pairs = community_pairs(community_sets[index])
            if not pairs:
                continue
            profiles[index]["first_observed_recent_nature_pair_share"] = sum(
                pair_counts[pair] == 0 for pair in pairs
            ) / len(pairs)
            surprises = []
            for left, right in pairs:
                if (
                    historical_claim_count == 0
                    or community_counts[left] == 0
                    or community_counts[right] == 0
                ):
                    continue
                commonness = (
                    (pair_counts[(left, right)] + 0.5)
                    * historical_claim_count
                    / (community_counts[left] * community_counts[right])
                )
                surprises.append(-math.log(commonness))
            profiles[index]["community_pair_surprisal_defined_count"] = len(surprises)
            profiles[index]["community_pair_mean_surprisal"] = (
                float(np.mean(surprises)) if surprises else None
            )
        for index in order[start:end]:
            communities = community_sets[index]
            if not communities:
                continue
            historical_claim_count += 1
            for community_id in communities:
                community_counts[community_id] += 1
            for pair in community_pairs(communities):
                pair_counts[pair] += 1
                first_dates.setdefault(pair, date)
        logger.info(
            "[步骤 4/7] 时间基线进度=%d/%d，日期=%s，已观察社区对=%d",
            end,
            len(order),
            date,
            len(pair_counts),
        )
        start = end
    return pair_counts, community_counts, first_dates, historical_claim_count


def pair_rows(
    pair_counts: dict[tuple[int, int], int],
    community_counts: dict[int, int],
    first_dates: dict[tuple[int, int], Any],
    historical_claim_count: int,
) -> list[dict[str, Any]]:
    return [
        {
            "community_a": left,
            "community_b": right,
            "pair_connector_count": pair_counts[(left, right)],
            "community_a_claim_count": community_counts[left],
            "community_b_claim_count": community_counts[right],
            "historical_claim_count": historical_claim_count,
            "first_observed_date": first_dates[(left, right)],
        }
        for left, right in sorted(pair_counts)
    ]


def percentile_assets(
    profiles: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    percentile_rows = []
    summary_rows = []
    groups: list[tuple[str, str | None, list[dict[str, Any]]]] = [
        ("ALL", None, profiles)
    ]
    groups.extend(
        (
            "CLAIM_TYPE",
            claim_type,
            [row for row in profiles if row["claim_type"] == claim_type],
        )
        for claim_type in CLAIM_TYPES
    )
    for scope, claim_type, group in groups:
        for metric in CORE_METRICS:
            values = np.asarray(
                [
                    float(row[metric])
                    for row in group
                    if row[metric] is not None and math.isfinite(float(row[metric]))
                ],
                dtype=np.float64,
            )
            if not len(values):
                continue
            quantiles = np.percentile(values, np.arange(101), method="linear")
            percentile_rows.extend(
                {
                    "reference_scope": scope,
                    "claim_type": claim_type,
                    "metric_name": metric,
                    "raw_percentile": percentile,
                    "metric_value": float(value),
                    "metric_direction": METRIC_DIRECTIONS[metric],
                    "reference_count": len(values),
                }
                for percentile, value in enumerate(quantiles)
            )
            summary_rows.append(
                {
                    "reference_scope": scope,
                    "claim_type": claim_type,
                    "metric_name": metric,
                    "metric_direction": METRIC_DIRECTIONS[metric],
                    "reference_count": len(values),
                    "minimum": float(values.min()),
                    "mean": float(values.mean()),
                    "standard_deviation": float(values.std()),
                    "median": float(np.median(values)),
                    "maximum": float(values.max()),
                }
            )
    return percentile_rows, summary_rows


def write_runtime_sqlite(
    path: Path,
    pairs: list[dict[str, Any]],
    community_counts: dict[int, int],
    percentiles: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    historical_claim_count: int,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript("""
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE community_pair_history (
                community_a INTEGER NOT NULL,
                community_b INTEGER NOT NULL,
                pair_connector_count INTEGER NOT NULL,
                community_a_claim_count INTEGER NOT NULL,
                community_b_claim_count INTEGER NOT NULL,
                historical_claim_count INTEGER NOT NULL,
                first_observed_date TEXT NOT NULL,
                PRIMARY KEY (community_a, community_b)
            ) WITHOUT ROWID;
            CREATE TABLE community_history_counts (
                community_id INTEGER PRIMARY KEY,
                connector_claim_count INTEGER NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE metric_percentiles (
                reference_scope TEXT NOT NULL,
                claim_type TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                raw_percentile INTEGER NOT NULL,
                metric_value REAL NOT NULL,
                metric_direction TEXT NOT NULL,
                reference_count INTEGER NOT NULL,
                PRIMARY KEY (reference_scope, claim_type, metric_name, raw_percentile)
            ) WITHOUT ROWID;
            CREATE TABLE metric_summary (
                reference_scope TEXT NOT NULL,
                claim_type TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_direction TEXT NOT NULL,
                reference_count INTEGER NOT NULL,
                minimum REAL NOT NULL,
                mean REAL NOT NULL,
                standard_deviation REAL NOT NULL,
                median REAL NOT NULL,
                maximum REAL NOT NULL,
                PRIMARY KEY (reference_scope, claim_type, metric_name)
            ) WITHOUT ROWID;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
            """)
        connection.executemany(
            "INSERT INTO community_pair_history VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    row["community_a"],
                    row["community_b"],
                    row["pair_connector_count"],
                    row["community_a_claim_count"],
                    row["community_b_claim_count"],
                    row["historical_claim_count"],
                    str(row["first_observed_date"]),
                )
                for row in pairs
            ],
        )
        connection.executemany(
            "INSERT INTO community_history_counts VALUES (?, ?)",
            sorted(community_counts.items()),
        )
        connection.executemany(
            "INSERT INTO metric_percentiles VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    row["reference_scope"],
                    row["claim_type"] or "",
                    row["metric_name"],
                    row["raw_percentile"],
                    row["metric_value"],
                    row["metric_direction"],
                    row["reference_count"],
                )
                for row in percentiles
            ],
        )
        connection.executemany(
            "INSERT INTO metric_summary VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    row["reference_scope"],
                    row["claim_type"] or "",
                    row["metric_name"],
                    row["metric_direction"],
                    row["reference_count"],
                    row["minimum"],
                    row["mean"],
                    row["standard_deviation"],
                    row["median"],
                    row["maximum"],
                )
                for row in summaries
            ],
        )
        connection.execute(
            "INSERT INTO metadata VALUES ('historical_claim_count', ?)",
            (str(historical_claim_count),),
        )
        connection.execute("ANALYZE")
        connection.commit()
    finally:
        connection.close()
    temporary.replace(path)


def output_schemas() -> tuple[pa.Schema, pa.Schema, pa.Schema, pa.Schema]:
    centroid_schema = pa.schema(
        [
            pa.field("community_id", pa.int32(), False),
            pa.field("centroid_row", pa.int32(), False),
            pa.field("claim_count", pa.int32(), False),
        ]
    )
    community_count_schema = pa.schema(
        [
            pa.field("community_id", pa.int32(), False),
            pa.field("connector_claim_count", pa.int32(), False),
            pa.field("historical_claim_count", pa.int32(), False),
        ]
    )
    percentile_schema = pa.schema(
        [
            pa.field("reference_scope", pa.string(), False),
            pa.field("claim_type", pa.string()),
            pa.field("metric_name", pa.string(), False),
            pa.field("raw_percentile", pa.int16(), False),
            pa.field("metric_value", pa.float64(), False),
            pa.field("metric_direction", pa.string(), False),
            pa.field("reference_count", pa.int32(), False),
        ]
    )
    summary_schema = pa.schema(
        [
            pa.field("reference_scope", pa.string(), False),
            pa.field("claim_type", pa.string()),
            pa.field("metric_name", pa.string(), False),
            pa.field("metric_direction", pa.string(), False),
            pa.field("reference_count", pa.int32(), False),
            pa.field("minimum", pa.float64(), False),
            pa.field("mean", pa.float64(), False),
            pa.field("standard_deviation", pa.float64(), False),
            pa.field("median", pa.float64(), False),
            pa.field("maximum", pa.float64(), False),
        ]
    )
    return centroid_schema, community_count_schema, percentile_schema, summary_schema


def configure_globals(
    claims: list[dict[str, Any]],
    incoming: list[list[tuple[int, float, bool, bool, int, int]]],
    communities: list[int | None],
    edge_keys: set[int],
    centroids: np.ndarray,
    community_rows: dict[int, int],
) -> None:
    global _CLAIMS, _INCOMING, _COMMUNITIES, _EDGE_KEYS
    global _CENTROIDS, _COMMUNITY_ROWS, _CLAIM_TYPE_LABELS, _CLAIM_COUNT
    _CLAIMS = claims
    _INCOMING = incoming
    _COMMUNITIES = communities
    _EDGE_KEYS = edge_keys
    _CENTROIDS = centroids
    _COMMUNITY_ROWS = community_rows
    _CLAIM_TYPE_LABELS = [str(row["claim_type"]) for row in claims]
    _CLAIM_COUNT = len(claims)


def main() -> int:
    args = parse_args()
    if args.workers < 0 or args.chunk_size <= 0:
        raise ValueError("--workers 不能为负数，--chunk-size 必须为正数")
    workers = args.workers or max(1, os.cpu_count() or 1)
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    logger = build_logger(
        root / "logs" / "phase8_build_claim_graph_statistics.log", args.verbose
    )
    require_files(
        [
            args.claims,
            args.edges,
            args.communities,
            args.embeddings,
            args.embedding_index,
            args.graph_index,
        ]
    )
    started = time.monotonic()
    logger.info("[步骤 1/7] 读取 Phase 7 Claim、正式边、社区和向量资产")
    claims = load_claims(args.claims, args.graph_index)
    communities = load_communities(args.communities, claims)
    incoming, edge_keys = load_edges(args.edges, claims)
    vectors = load_vectors(args.embeddings, args.embedding_index, claims)
    logger.info(
        "[步骤 1/7] Claim=%d，正式边=%d，向量维度=%d",
        len(claims),
        len(edge_keys),
        vectors.shape[1],
    )

    logger.info("[步骤 2/7] 计算社区中心向量；孤立 Claim 不参与社区中心")
    centroids, centroid_rows, community_row_map = build_centroids(vectors, communities)
    configure_globals(
        claims, incoming, communities, edge_keys, centroids, community_row_map
    )
    logger.info(
        "[步骤 2/7] 社区=%d，中心向量维度=%d", len(centroid_rows), centroids.shape[1]
    )

    logger.info("[步骤 3/7] 并行计算历史 Claim 静态伪插入画像；workers=%d", workers)
    profiles, community_sets = build_static_profiles(workers, args.chunk_size, logger)

    logger.info("[步骤 4/7] 按发表日期计算严格历史社区组合和 surprisal")
    pair_counts, community_counts, first_dates, historical_count = (
        add_temporal_pair_metrics(profiles, community_sets, logger)
    )
    pairs = pair_rows(pair_counts, community_counts, first_dates, historical_count)

    logger.info("[步骤 5/7] 建立全体 Claim 与同类型辅助百分位标尺")
    percentiles, summaries = percentile_assets(profiles)
    centroid_schema, community_count_schema, percentile_schema, summary_schema = (
        output_schemas()
    )

    logger.info("[步骤 6/7] 覆盖写入约定的 Phase 8 中间产物")
    atomic_write_table(
        profiles, PROFILE_SCHEMA, root / "historical_insertion_profiles.parquet"
    )
    atomic_write_table(pairs, PAIR_SCHEMA, root / "community_pair_history.parquet")
    atomic_write_numpy(centroids, root / "community_centroid_matrix.npy")
    atomic_write_table(
        centroid_rows, centroid_schema, root / "community_centroid_index.parquet"
    )
    community_count_rows = [
        {
            "community_id": community_id,
            "connector_claim_count": count,
            "historical_claim_count": historical_count,
        }
        for community_id, count in sorted(community_counts.items())
    ]
    atomic_write_table(
        community_count_rows,
        community_count_schema,
        root / "community_history_counts.parquet",
    )
    atomic_write_table(
        percentiles, percentile_schema, root / "historical_metric_percentiles.parquet"
    )
    atomic_write_table(
        summaries, summary_schema, root / "historical_metric_summary.parquet"
    )
    write_runtime_sqlite(
        root / "claim_graph_runtime_statistics.sqlite",
        pairs,
        community_counts,
        percentiles,
        summaries,
        historical_count,
    )

    logger.info("[步骤 7/7] 写入统计摘要")
    stats = {
        "claim_count": len(claims),
        "claim_with_prior_neighbor_count": sum(
            row["neighbor_count"] > 0 for row in profiles
        ),
        "claim_with_community_pair_count": sum(
            row["community_pair_count"] > 0 for row in profiles
        ),
        "community_count": len(centroid_rows),
        "isolated_claim_count": sum(value is None for value in communities),
        "observed_community_pair_count": len(pairs),
        "historical_connector_claim_count": historical_count,
        "percentile_row_count": len(percentiles),
        "workers": workers,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "note": "固定全图社区仅用于 2025 年后运行态；严格回顾测评需按 cutoff 重建社区快照。",
    }
    (root / "phase8_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "Phase 8 完成：画像=%d，社区=%d，历史社区对=%d，耗时=%.1fs",
        len(profiles),
        len(centroid_rows),
        len(pairs),
        time.monotonic() - started,
    )
    logger.info("历史画像：%s", root / "historical_insertion_profiles.parquet")
    logger.info("运行时统计索引：%s", root / "claim_graph_runtime_statistics.sqlite")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileNotFoundError,
        KeyError,
        RuntimeError,
        ValueError,
        OSError,
        sqlite3.Error,
        pa.ArrowException,
    ) as error:
        print(f"Phase 8 构建失败：{type(error).__name__} | {error}", file=sys.stderr)
        raise SystemExit(1) from error

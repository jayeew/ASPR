"""Source-defined and explicitly adapted publication-time v6 metrics.

The functions in this module are intentionally small and hand-calculable.
Large-table materialization is kept separate so mathematical unit tests do not
depend on storage layout or graph snapshot machinery.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from typing import Hashable, Mapping, Sequence, Tuple

import networkx as nx
import numpy as np


Label = Hashable
Pair = Tuple[Label, Label]


def canonical_pair(left: Label, right: Label) -> Pair:
    """Return a stable undirected pair key."""
    return tuple(sorted((left, right), key=str))  # type: ignore[return-value]


def field_variety(labels: Sequence[Label]) -> float:
    """Count represented non-missing categories (Stirling variety)."""
    return float(len({label for label in labels if label is not None and str(label)}))


def field_pielou_evenness(labels: Sequence[Label]) -> float:
    """Compute Pielou evenness; return NaN when fewer than two fields exist."""
    counts = Counter(label for label in labels if label is not None and str(label))
    if len(counts) < 2:
        return float("nan")
    values = np.asarray(list(counts.values()), dtype=float)
    probabilities = values / values.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return float(entropy / math.log(len(counts)))


def field_disparity_mean(
    labels: Sequence[Label], distances: Mapping[Pair, float]
) -> float:
    """Mean distance among distinct occupied fields (Stirling disparity)."""
    occupied = sorted(
        {label for label in labels if label is not None and str(label)}, key=str
    )
    if len(occupied) < 2:
        return float("nan")
    values = [
        float(distances[canonical_pair(left, right)])
        for left, right in itertools.combinations(occupied, 2)
    ]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("field distances must be finite and in [0, 1]")
    return float(np.mean(values))


def rao_stirling_integration(
    labels: Sequence[Label], distances: Mapping[Pair, float]
) -> float:
    """Compute frequency-weighted Rao-Stirling integration."""
    counts = Counter(label for label in labels if label is not None and str(label))
    total = sum(counts.values())
    if total == 0:
        return float("nan")
    probabilities = {label: count / total for label, count in counts.items()}
    result = 0.0
    for left, right in itertools.combinations(sorted(counts, key=str), 2):
        distance = float(distances[canonical_pair(left, right)])
        if not math.isfinite(distance) or not 0.0 <= distance <= 1.0:
            raise ValueError("field distances must be finite and in [0, 1]")
        result += 2.0 * probabilities[left] * probabilities[right] * distance
    return float(result)


def cosine_distance_profiles(
    profiles: Mapping[Label, Sequence[float]],
) -> Mapping[Pair, float]:
    """Build cosine distances between frozen historical field profiles."""
    output = {}
    ordered = sorted(profiles, key=str)
    for left, right in itertools.combinations(ordered, 2):
        left_values = np.asarray(profiles[left], dtype=float)
        right_values = np.asarray(profiles[right], dtype=float)
        if left_values.shape != right_values.shape or left_values.ndim != 1:
            raise ValueError("field profiles must be aligned one-dimensional vectors")
        denominator = float(np.linalg.norm(left_values) * np.linalg.norm(right_values))
        similarity = (
            float(np.dot(left_values, right_values) / denominator)
            if denominator > 0.0
            else 0.0
        )
        output[canonical_pair(left, right)] = float(
            np.clip(1.0 - similarity, 0.0, 1.0)
        )
    return output


def _source_pairs(source_ids: Sequence[Label]) -> Sequence[Pair]:
    values = sorted(
        {
            value
            for value in source_ids
            if value is not None and str(value)
        },
        key=str,
    )
    return [canonical_pair(left, right) for left, right in itertools.combinations(values, 2)]


def novelty_u(
    source_ids: Sequence[Label],
    pair_counts: Mapping[Pair, int],
    source_counts: Mapping[Label, int],
    n_historical_papers: int,
    *,
    quantile: float = 0.10,
    zero_pair_smoothing: float = 0.5,
) -> float:
    """Compute the Lee-Walsh-Wang U family on frozen historical source pairs.

    The v6 registry labels the strictly-prior source implementation as a
    registered adaptation because the published construction is not a
    publication-time historical-only estimator.
    """
    if n_historical_papers <= 0:
        return float("nan")
    if not 0.0 < quantile < 1.0 or zero_pair_smoothing <= 0.0:
        raise ValueError("quantile and smoothing must be positive and valid")
    commonness = []
    for pair in _source_pairs(source_ids):
        left, right = pair
        left_count = int(source_counts.get(left, 0))
        right_count = int(source_counts.get(right, 0))
        if left_count <= 0 or right_count <= 0:
            continue
        observed = float(pair_counts.get(pair, 0))
        observed = observed if observed > 0.0 else float(zero_pair_smoothing)
        commonness.append(
            observed * float(n_historical_papers) / float(left_count * right_count)
        )
    if not commonness:
        return float("nan")
    lower_tail = float(np.quantile(commonness, quantile))
    return float(-math.log(max(lower_tail, np.finfo(float).tiny)))


def first_time_source_pair_share(
    source_ids: Sequence[Label], pair_counts: Mapping[Pair, int]
) -> float:
    """Share of focal source pairs absent from the strictly prior history."""
    pairs = _source_pairs(source_ids)
    if not pairs:
        return float("nan")
    return float(np.mean([int(pair_counts.get(pair, 0)) == 0 for pair in pairs]))


def first_time_source_pair_distance_mean(
    source_ids: Sequence[Label],
    pair_counts: Mapping[Pair, int],
    distances: Mapping[Pair, float],
) -> float:
    """Mean cognitive distance among historically unseen source pairs."""
    novel_pairs = [
        pair for pair in _source_pairs(source_ids) if int(pair_counts.get(pair, 0)) == 0
    ]
    if not novel_pairs:
        return float("nan")
    values = [float(distances[pair]) for pair in novel_pairs if pair in distances]
    if not values:
        return float("nan")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("source distances must be finite and in [0, 1]")
    return float(np.mean(values))


def uzzi_atypicality_p10(z_scores: Sequence[float]) -> float:
    """Return negative tenth percentile of source-pair null-model z scores."""
    values = np.asarray(z_scores, dtype=float)
    values = values[np.isfinite(values)]
    return float(-np.quantile(values, 0.10)) if len(values) else float("nan")


def uzzi_conventionality_median(z_scores: Sequence[float]) -> float:
    """Return median source-pair null-model z score."""
    values = np.asarray(z_scores, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else float("nan")


def marginal_pair_z_scores(
    source_ids: Sequence[Label],
    pair_counts: Mapping[Pair, int],
    source_counts: Mapping[Label, int],
    n_historical_papers: int,
) -> Sequence[float]:
    """Approximate pair z scores under fixed marginal paper frequencies.

    For each distinct source pair, overlap follows a hypergeometric null with
    fixed ``N_i`` and ``N_j``. This preserves source marginals but not the full
    paper degree sequence, so v6 registers it as an approximation requiring
    comparison with bipartite edge-swap randomization.
    """
    if n_historical_papers <= 1:
        return []
    unique_sources = sorted(
        {value for value in source_ids if value is not None and str(value)},
        key=str,
    )
    scores = []
    for left, right in itertools.combinations(unique_sources, 2):
        left_count = int(source_counts.get(left, 0))
        right_count = int(source_counts.get(right, 0))
        if left_count <= 0 or right_count <= 0:
            continue
        population = float(n_historical_papers)
        expected = left_count * right_count / population
        variance = (
            right_count
            * (left_count / population)
            * (1.0 - left_count / population)
            * ((population - right_count) / (population - 1.0))
        )
        if variance <= 0.0:
            continue
        observed = float(pair_counts.get(canonical_pair(left, right), 0))
        scores.append(float((observed - expected) / math.sqrt(variance)))
    return scores


def uzzi_atypicality_from_marginals(
    source_ids: Sequence[Label],
    pair_counts: Mapping[Pair, int],
    source_counts: Mapping[Label, int],
    n_historical_papers: int,
) -> float:
    """Compute v6's conditional T0 atypicality approximation."""
    return uzzi_atypicality_p10(
        marginal_pair_z_scores(
            source_ids, pair_counts, source_counts, n_historical_papers
        )
    )


def uzzi_conventionality_from_marginals(
    source_ids: Sequence[Label],
    pair_counts: Mapping[Pair, int],
    source_counts: Mapping[Label, int],
    n_historical_papers: int,
) -> float:
    """Compute v6's conditional T0 conventionality approximation."""
    return uzzi_conventionality_median(
        marginal_pair_z_scores(
            source_ids, pair_counts, source_counts, n_historical_papers
        )
    )


def sva_modularity_change_rate(
    baseline_graph: nx.Graph,
    augmented_graph: nx.Graph,
    communities: Sequence[set[Label]],
    *,
    weight: str = "weight",
) -> float:
    """Compute Chen's relative modularity change using a fixed partition."""
    baseline_q = float(nx.community.modularity(baseline_graph, communities, weight=weight))
    if math.isclose(baseline_q, 0.0, abs_tol=1e-15):
        return float("nan")
    augmented_q = float(
        nx.community.modularity(augmented_graph, communities, weight=weight)
    )
    return float((baseline_q - augmented_q) / baseline_q)


def _community_lookup(communities: Sequence[set[Label]]) -> Mapping[Label, int]:
    lookup = {}
    for index, community in enumerate(communities):
        for node in community:
            if node in lookup:
                raise ValueError("communities must be a disjoint partition")
            lookup[node] = index
    return lookup


def sva_linkage_score(
    graph: nx.Graph,
    communities: Sequence[set[Label]],
    *,
    scaling_factor: float = 1.0,
) -> float:
    """Compute between-cluster Linkage(G,C) with a frozen scaling factor."""
    if scaling_factor <= 0.0:
        raise ValueError("scaling_factor must be positive")
    membership = _community_lookup(communities)
    if set(graph.nodes) - set(membership):
        raise ValueError("community partition does not cover every graph node")
    between = sum(
        1.0
        for left, right in graph.edges
        if membership[left] != membership[right]
    )
    return float(between / scaling_factor)


def sva_cluster_linkage(
    baseline_graph: nx.Graph,
    augmented_graph: nx.Graph,
    communities: Sequence[set[Label]],
    contributing_references: int,
    total_references: int,
    *,
    scaling_factor: float = 1.0,
) -> float:
    """Compute Chen/Sebastian cluster linkage with the CR/NR weight."""
    if total_references <= 0 or not 0 <= contributing_references <= total_references:
        return float("nan")
    baseline = sva_linkage_score(
        baseline_graph, communities, scaling_factor=scaling_factor
    )
    if math.isclose(baseline, 0.0, abs_tol=1e-15):
        return float("nan")
    augmented = sva_linkage_score(
        augmented_graph, communities, scaling_factor=scaling_factor
    )
    return float(
        ((augmented - baseline) / baseline)
        * 100.0
        * (contributing_references / total_references)
    )


def sva_centrality_divergence(
    baseline_graph: nx.Graph,
    augmented_graph: nx.Graph,
    *,
    weight: str | None = None,
    epsilon: float = 1e-12,
) -> float:
    """Compute smoothed KL divergence of betweenness distributions.

    Normalization and epsilon smoothing are explicitly registered adaptations
    needed to make the source equation a defined probability divergence when
    centralities contain zeros.
    """
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if set(baseline_graph.nodes) != set(augmented_graph.nodes):
        raise ValueError("baseline and augmented graphs must share the node set")
    nodes = sorted(baseline_graph.nodes, key=str)
    baseline = nx.betweenness_centrality(
        baseline_graph, normalized=True, weight=weight
    )
    augmented = nx.betweenness_centrality(
        augmented_graph, normalized=True, weight=weight
    )
    p_values = np.asarray([baseline[node] for node in nodes], dtype=float) + epsilon
    q_values = np.asarray([augmented[node] for node in nodes], dtype=float) + epsilon
    p_values /= p_values.sum()
    q_values /= q_values.sum()
    return float(np.sum(p_values * np.log(p_values / q_values)))


__all__ = [
    "canonical_pair",
    "cosine_distance_profiles",
    "field_disparity_mean",
    "field_pielou_evenness",
    "field_variety",
    "first_time_source_pair_distance_mean",
    "first_time_source_pair_share",
    "marginal_pair_z_scores",
    "novelty_u",
    "rao_stirling_integration",
    "sva_centrality_divergence",
    "sva_cluster_linkage",
    "sva_linkage_score",
    "sva_modularity_change_rate",
    "uzzi_atypicality_p10",
    "uzzi_atypicality_from_marginals",
    "uzzi_conventionality_median",
    "uzzi_conventionality_from_marginals",
]

"""Hand-calculable candidate indicators for the ASPR v6.1 evidence map.

The module contains mathematical candidates, not model-selection logic.
Whether a candidate becomes a primary input is decided exclusively by the
frozen evidence registry and outcome-blind screening report.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from typing import Hashable, Mapping, Sequence, Tuple

import numpy as np

from .features_v6 import canonical_pair


Label = Hashable
Pair = Tuple[Label, Label]


def _counts(labels: Sequence[Label]) -> Counter[Label]:
    """Return counts for non-missing categorical labels."""
    return Counter(
        label for label in labels if label is not None and str(label).strip()
    )


def field_relative_variety(
    labels: Sequence[Label], *, total_categories: int
) -> float:
    """Return occupied categories divided by the frozen taxonomy size."""
    if int(total_categories) <= 0:
        raise ValueError("total_categories must be positive")
    return float(len(_counts(labels)) / int(total_categories))


def field_other_field_share(
    labels: Sequence[Label], *, focal_field: Label | None
) -> float:
    """Return the reference share outside the focal paper's primary field."""
    values = [
        label
        for label in labels
        if label is not None and str(label).strip()
    ]
    if not values or focal_field is None or not str(focal_field).strip():
        return float("nan")
    return float(np.mean([label != focal_field for label in values]))


def field_gini_balance(labels: Sequence[Label]) -> float:
    """Return one minus Gini across occupied reference-field shares.

    The occupied-category convention follows paper-level scientometric uses
    of ``1-Gini``. A one-category list is perfectly balanced within its
    observed support and therefore has value one.
    """
    counts = _counts(labels)
    if not counts:
        return float("nan")
    values = np.asarray(list(counts.values()), dtype=float)
    mean = float(values.mean())
    if mean <= 0.0:
        return float("nan")
    gini = float(
        np.abs(values[:, None] - values[None, :]).sum()
        / (2.0 * len(values) ** 2 * mean)
    )
    return float(1.0 - gini)


def field_shannon_entropy(labels: Sequence[Label]) -> float:
    """Return Shannon entropy of the reference-field distribution."""
    counts = _counts(labels)
    if not counts:
        return float("nan")
    values = np.asarray(list(counts.values()), dtype=float)
    probabilities = values / values.sum()
    return float(-np.sum(probabilities * np.log(probabilities)))


def field_gini_simpson(labels: Sequence[Label]) -> float:
    """Return the probability that two draws have different fields."""
    counts = _counts(labels)
    if not counts:
        return float("nan")
    values = np.asarray(list(counts.values()), dtype=float)
    probabilities = values / values.sum()
    return float(1.0 - np.sum(probabilities**2))


def field_hhi(labels: Sequence[Label]) -> float:
    """Return Herfindahl-Hirschman concentration of reference fields."""
    diversity = field_gini_simpson(labels)
    return float(1.0 - diversity) if math.isfinite(diversity) else diversity


def field_hill_number(labels: Sequence[Label], *, order: int) -> float:
    """Return Hill effective category count for orders zero, one, or two."""
    counts = _counts(labels)
    if not counts:
        return float("nan")
    if int(order) == 0:
        return float(len(counts))
    if int(order) == 1:
        return float(math.exp(field_shannon_entropy(labels)))
    if int(order) == 2:
        values = np.asarray(list(counts.values()), dtype=float)
        probabilities = values / values.sum()
        return float(1.0 / np.sum(probabilities**2))
    raise ValueError("only Hill orders 0, 1, and 2 are registered")


def occupied_field_distances(
    labels: Sequence[Label],
    distances: Mapping[Pair, float],
) -> np.ndarray:
    """Return validated distances among distinct occupied field pairs."""
    occupied = sorted(_counts(labels), key=str)
    if len(occupied) < 2:
        return np.asarray([], dtype=float)
    values = np.asarray(
        [
            float(distances[canonical_pair(left, right)])
            for left, right in itertools.combinations(occupied, 2)
        ],
        dtype=float,
    )
    if (
        not np.isfinite(values).all()
        or (values < 0.0).any()
        or (values > 1.0).any()
    ):
        raise ValueError("field distances must be finite and in [0, 1]")
    return values


def field_distance_quantile(
    labels: Sequence[Label],
    distances: Mapping[Pair, float],
    *,
    quantile: float,
) -> float:
    """Return a frozen quantile of occupied-pair cognitive distances."""
    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    values = occupied_field_distances(labels, distances)
    return (
        float(np.quantile(values, float(quantile)))
        if len(values)
        else float("nan")
    )


def div_index(
    labels: Sequence[Label],
    distances: Mapping[Pair, float],
    *,
    total_categories: int,
) -> float:
    """Return DIV = relative variety × 1-Gini × mean disparity."""
    variety = field_relative_variety(
        labels, total_categories=int(total_categories)
    )
    balance = field_gini_balance(labels)
    disparity_values = occupied_field_distances(labels, distances)
    disparity = (
        float(np.mean(disparity_values))
        if len(disparity_values)
        else float("nan")
    )
    if not all(math.isfinite(value) for value in (variety, balance, disparity)):
        return float("nan")
    return float(variety * balance * disparity)


def true_diversity_from_rao(rao_stirling: float) -> float:
    """Return the order-two effective diversity transform 1/(1-RS)."""
    value = float(rao_stirling)
    if not math.isfinite(value) or not 0.0 <= value < 1.0:
        return float("nan")
    return float(1.0 / (1.0 - value))


def first_time_pair_any(
    source_ids: Sequence[Label],
    pair_counts: Mapping[Pair, int],
) -> float:
    """Return one when at least one source pair is absent from history."""
    pairs = _source_pairs(source_ids)
    if not pairs:
        return float("nan")
    return float(any(int(pair_counts.get(pair, 0)) == 0 for pair in pairs))


def first_time_pair_count(
    source_ids: Sequence[Label],
    pair_counts: Mapping[Pair, int],
) -> float:
    """Count source pairs absent from strictly prior history."""
    pairs = _source_pairs(source_ids)
    if not pairs:
        return float("nan")
    return float(sum(int(pair_counts.get(pair, 0)) == 0 for pair in pairs))


def first_time_pair_distance_sum(
    source_ids: Sequence[Label],
    pair_counts: Mapping[Pair, int],
    distances: Mapping[Pair, float],
) -> float:
    """Sum cognitive distance across first-time source pairs."""
    pairs = [
        pair
        for pair in _source_pairs(source_ids)
        if int(pair_counts.get(pair, 0)) == 0 and pair in distances
    ]
    if not pairs:
        return float("nan")
    values = np.asarray([float(distances[pair]) for pair in pairs], dtype=float)
    if (
        not np.isfinite(values).all()
        or (values < 0.0).any()
        or (values > 1.0).any()
    ):
        raise ValueError("source distances must be finite and in [0, 1]")
    return float(values.sum())


def source_pair_mean_surprisal(
    source_ids: Sequence[Label],
    pair_counts: Mapping[Pair, int],
    source_counts: Mapping[Label, int],
    n_historical_papers: int,
    *,
    zero_pair_smoothing: float = 0.5,
) -> float:
    """Return mean negative log observed/expected source-pair commonness."""
    if int(n_historical_papers) <= 0 or float(zero_pair_smoothing) <= 0.0:
        return float("nan")
    values = []
    for left, right in _source_pairs(source_ids):
        left_count = int(source_counts.get(left, 0))
        right_count = int(source_counts.get(right, 0))
        if left_count <= 0 or right_count <= 0:
            continue
        observed = float(pair_counts.get(canonical_pair(left, right), 0))
        observed = observed if observed > 0.0 else float(zero_pair_smoothing)
        commonness = (
            observed
            * float(n_historical_papers)
            / float(left_count * right_count)
        )
        values.append(-math.log(max(commonness, np.finfo(float).tiny)))
    return float(np.mean(values)) if values else float("nan")


def hypergeometric_pair_z_scores(
    source_ids: Sequence[Label],
    pair_counts: Mapping[Pair, int],
    source_counts: Mapping[Label, int],
    n_historical_papers: int,
) -> Tuple[float, ...]:
    """Return exact fixed-marginal hypergeometric co-occurrence z scores.

    This is an analytically defined null model. It is deliberately named
    separately from the Uzzi edge-swap Monte Carlo null, whose paper-level
    P10 and median aggregation it reuses as a source-adapted sensitivity.
    """
    population = int(n_historical_papers)
    if population <= 1:
        return ()
    values = []
    for left, right in _source_pairs(source_ids):
        left_count = int(source_counts.get(left, 0))
        right_count = int(source_counts.get(right, 0))
        if left_count <= 0 or right_count <= 0:
            continue
        mean = left_count * right_count / float(population)
        variance = (
            right_count
            * (left_count / float(population))
            * (1.0 - left_count / float(population))
            * ((population - right_count) / (population - 1.0))
        )
        if variance <= 0.0:
            continue
        observed = float(pair_counts.get(canonical_pair(left, right), 0))
        values.append(float((observed - mean) / math.sqrt(variance)))
    return tuple(values)


def low_frequency_pair_share(
    source_ids: Sequence[Label],
    pair_counts: Mapping[Pair, int],
    *,
    maximum_prior_count: int,
) -> float:
    """Return share of pairs at or below a frozen prior-count threshold."""
    if int(maximum_prior_count) < 0:
        raise ValueError("maximum_prior_count must be non-negative")
    pairs = _source_pairs(source_ids)
    if not pairs:
        return float("nan")
    return float(
        np.mean(
            [
                int(pair_counts.get(pair, 0)) <= int(maximum_prior_count)
                for pair in pairs
            ]
        )
    )


def reference_overlap_novelty(
    focal_references: Sequence[Label],
    comparison_references: Sequence[Sequence[Label]],
) -> float:
    """Return one minus mean Jaccard overlap with same-domain prior papers.

    This implements the paper-level formula validated by Matsumoto et al.
    (2021). Selection of same-domain papers and citation windows is performed
    by the caller.
    """
    focal = {
        value
        for value in focal_references
        if value is not None and str(value).strip()
    }
    if not focal or not comparison_references:
        return float("nan")
    similarities = []
    for references in comparison_references:
        prior = {
            value
            for value in references
            if value is not None and str(value).strip()
        }
        union = focal | prior
        if union:
            similarities.append(len(focal & prior) / len(union))
    return (
        float(1.0 - np.mean(similarities))
        if similarities
        else float("nan")
    )


def _source_pairs(source_ids: Sequence[Label]) -> Tuple[Pair, ...]:
    values = sorted(
        {
            value
            for value in source_ids
            if value is not None and str(value).strip()
        },
        key=str,
    )
    return tuple(
        canonical_pair(left, right)
        for left, right in itertools.combinations(values, 2)
    )


__all__ = [
    "div_index",
    "field_distance_quantile",
    "field_gini_balance",
    "field_gini_simpson",
    "field_hhi",
    "field_hill_number",
    "field_other_field_share",
    "field_relative_variety",
    "field_shannon_entropy",
    "first_time_pair_any",
    "first_time_pair_count",
    "first_time_pair_distance_sum",
    "hypergeometric_pair_z_scores",
    "low_frequency_pair_share",
    "occupied_field_distances",
    "reference_overlap_novelty",
    "source_pair_mean_surprisal",
    "true_diversity_from_rao",
]

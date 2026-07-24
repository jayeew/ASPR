"""Outcome-blind candidate screening and freeze evidence for ASPR v6.1."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from aspr.corpus import normalize_openalex_id

from .candidate_registry_v6_1 import (
    CandidateRegistryV61,
    candidate_registry_sha256,
    load_candidate_registry_v6_1,
    verify_search_log,
)
from .feature_materializer_v6 import annual_field_distances
from .features_v6 import (
    canonical_pair,
    first_time_source_pair_distance_mean,
    first_time_source_pair_share,
    marginal_pair_z_scores,
    novelty_u,
    rao_stirling_integration,
    uzzi_atypicality_p10,
    uzzi_conventionality_median,
)
from .features_v6_1 import (
    div_index,
    field_distance_quantile,
    field_gini_balance,
    field_gini_simpson,
    field_hhi,
    field_hill_number,
    field_other_field_share,
    field_relative_variety,
    field_shannon_entropy,
    first_time_pair_any,
    first_time_pair_count,
    first_time_pair_distance_sum,
    hypergeometric_pair_z_scores,
    low_frequency_pair_share,
    occupied_field_distances,
    reference_overlap_novelty,
    source_pair_mean_surprisal,
    true_diversity_from_rao,
)
from .materialize_v6_1 import (
    FIELD_TAXONOMY_SIZE,
    REFERENCE_OVERLAP_COCITING_WINDOW_YEARS,
    REFERENCE_OVERLAP_REFERENCE_WINDOW_YEARS,
)
from .modeling_v6 import safe_spearman
from .source_audit_v6 import sha256_file


SCREENING_VERSION = "aspr-v6.1-outcome-blind-screening-7"
STABILITY_CODE_NAMES: Tuple[str, ...] = (
    "novelty_u_t0_source",
    "source_pair_mean_surprisal",
    "low_frequency_source_pair_share",
    "reference_overlap_novelty_t0",
    "uzzi_atypicality_p10_t0",
    "uzzi_conventionality_median_t0",
    "hypergeom_atypicality_p10_t0",
    "hypergeom_conventionality_median_t0",
    "first_time_source_pair_any",
    "first_time_source_pair_count",
    "first_time_source_pair_share",
    "first_time_source_pair_distance_sum",
    "first_time_source_pair_distance_mean",
    "field_variety",
    "field_relative_variety",
    "reference_other_field_share",
    "field_gini_balance",
    "field_shannon_entropy",
    "field_pielou_evenness",
    "field_gini_simpson",
    "field_hhi",
    "field_hill_q0",
    "field_hill_q1",
    "field_hill_q2",
    "field_disparity_cosine_mean",
    "field_disparity_cosine_max",
    "field_disparity_cosine_p90",
    "rao_stirling_integration",
    "field_div_index",
    "rao_true_diversity_q2",
)
PRIMARY_SELECTION_GROUPS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("A1_COMMONNESS_LOWER_TAIL", ("A1.NOVELTY_U",)),
    ("A1_REFERENCE_OVERLAP", ("A1.REFERENCE_OVERLAP",)),
    ("A2_LEFT_TAIL", ("A2.HYPERGEOM_P10",)),
    ("A2_CENTRE", ("A2.HYPERGEOM_MEDIAN",)),
    (
        "A3_FIRST_INCIDENCE",
        ("A3.FIRST_SHARE", "A3.FIRST_COUNT", "A3.FIRST_ANY"),
    ),
    (
        "A3_FIRST_DISTANCE",
        ("A3.FIRST_DISTANCE_MEAN", "A3.FIRST_DISTANCE_SUM"),
    ),
    (
        "A4_VARIETY",
        ("A4.VARIETY", "A4.RELATIVE_VARIETY", "A4.HILL_Q0"),
    ),
    ("A4_FOCAL_BREADTH", ("A4.OTHER_FIELD_SHARE",)),
    (
        "A4_BALANCE",
        (
            "A4.GINI_BALANCE",
            "A4.PIELOU",
            "A4.SHANNON",
            "A4.GINI_SIMPSON",
            "A4.HILL_Q1",
            "A4.HILL_Q2",
            "A4.HHI",
        ),
    ),
    (
        "A5_UNWEIGHTED_DISTANCE",
        ("A5.MEAN_DISTANCE", "A5.P90_DISTANCE", "A5.MAX_DISTANCE"),
    ),
    (
        "A5_COMPREHENSIVE_INDEX",
        (
            "A5.RAO_STIRLING",
            "A5.DIV",
            "A5.PORTER_INTEGRATION",
            "A5.DIV_STAR",
        ),
    ),
)


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _stable_seed(value: str, *, salt: str) -> int:
    digest = hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def _pair_set(values: Sequence[str]) -> Set[Tuple[Any, Any]]:
    unique = sorted({value for value in values if value})
    return {
        canonical_pair(left, right)
        for left, right in itertools.combinations(unique, 2)
    }


def _as_list(value: Any) -> List[str]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = []
    elif isinstance(value, (list, tuple, set, np.ndarray)):
        parsed = list(value)
    else:
        parsed = []
    return [str(item) for item in parsed if str(item).strip()]


def coverage_audit(
    registry: CandidateRegistryV61,
    features: pd.DataFrame,
    *,
    denominator_policy: str = "all_papers",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Measure raw and gate-denominator coverage for every candidate.

    ``eligible_by_metric_family`` is a versioned diagnostic revision for
    corpora that retain papers with structurally unavailable bibliographies.
    It never turns missing values into zero: source-pair candidates are
    assessed among papers with at least ten references and 60% source mapping;
    field candidates use the analogous 60% field-mapping requirement. Raw
    whole-cohort coverage remains in the output and must still be reported.
    """
    allowed_policies = {"all_papers", "eligible_by_metric_family"}
    if denominator_policy not in allowed_policies:
        raise ValueError(
            f"unknown coverage denominator policy: {denominator_policy}"
        )
    rows = []
    domain_rows = []
    for candidate in registry.candidates.values():
        code_name = candidate.code_name
        if denominator_policy == "eligible_by_metric_family":
            valid_bibliography = pd.to_numeric(
                features["valid_reference_count"], errors="coerce"
            ).ge(10)
            if code_name == "reference_overlap_novelty_t0":
                # The Matsumoto metric uses work-level reference IDs and a
                # paper field, not cited-source or cited-field mappings.
                eligible = valid_bibliography
            else:
                mapping_column = (
                    "source_mapping_coverage"
                    if candidate.angle_id
                    in {
                        "A1_COMBINATION_RARITY",
                        "A2_ATYPICALITY_CONVENTIONALITY",
                        "A3_FIRST_TIME_COMBINATION",
                    }
                    else "field_mapping_coverage"
                )
                eligible = (
                    valid_bibliography
                    & pd.to_numeric(
                        features[mapping_column], errors="coerce"
                    ).ge(0.6)
                )
        else:
            eligible = pd.Series(True, index=features.index)
        if code_name not in features:
            rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "code_name": code_name,
                    "n_rows": len(features),
                    "n_eligible": int(eligible.sum()),
                    "n_finite": 0,
                    "raw_n_finite": 0,
                    "raw_overall_coverage": 0.0,
                    "raw_minimum_domain_coverage": 0.0,
                    "overall_coverage": 0.0,
                    "minimum_domain_coverage": 0.0,
                    "n_unique_finite": 0,
                    "nondegenerate_test_pass": 0,
                    "implemented_in_candidate_view": 0,
                }
            )
            continue
        values = pd.to_numeric(features[code_name], errors="coerce")
        finite_values = values[eligible & np.isfinite(values)]
        n_unique_finite = int(finite_values.nunique(dropna=True))
        per_domain = []
        raw_per_domain = []
        for domain, group_index in features.groupby(
            "domain12", sort=True
        ).groups.items():
            group_values = values.loc[group_index]
            group_eligible = eligible.loc[group_index]
            selected = group_values[group_eligible]
            coverage = (
                float(selected.notna().mean()) if len(selected) else 0.0
            )
            raw_coverage = float(group_values.notna().mean())
            per_domain.append(coverage)
            raw_per_domain.append(raw_coverage)
            domain_rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "code_name": code_name,
                    "domain12": str(domain),
                    "n_rows": len(group_values),
                    "n_eligible": int(group_eligible.sum()),
                    "n_finite": int(selected.notna().sum()),
                    "raw_n_finite": int(group_values.notna().sum()),
                    "raw_coverage": raw_coverage,
                    "coverage": coverage,
                    "denominator_policy": denominator_policy,
                }
            )
        selected_values = values[eligible]
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "code_name": code_name,
                "n_rows": len(values),
                "n_eligible": int(eligible.sum()),
                "n_finite": int(selected_values.notna().sum()),
                "raw_n_finite": int(values.notna().sum()),
                "raw_overall_coverage": float(values.notna().mean()),
                "raw_minimum_domain_coverage": (
                    float(min(raw_per_domain)) if raw_per_domain else 0.0
                ),
                "overall_coverage": (
                    float(selected_values.notna().mean())
                    if len(selected_values)
                    else 0.0
                ),
                "minimum_domain_coverage": (
                    float(min(per_domain)) if per_domain else 0.0
                ),
                "n_unique_finite": n_unique_finite,
                "nondegenerate_test_pass": int(n_unique_finite >= 2),
                "implemented_in_candidate_view": 1,
                "denominator_policy": denominator_policy,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(domain_rows)


def select_stability_sample(
    papers: pd.DataFrame,
    features: pd.DataFrame,
    *,
    max_per_domain_era: int,
    salt: str,
) -> pd.DataFrame:
    """Select an outcome-blind domain-by-five-year sample."""
    columns = [
        "paper_id",
        "valid_reference_count",
        "source_mapping_coverage",
        "field_mapping_coverage",
    ]
    frame = papers[
        [
            "paper_id",
            "publication_year",
            "domain12",
            "openalex_primary_field",
        ]
    ].merge(
        features[columns],
        on="paper_id",
        how="inner",
        validate="one_to_one",
    )
    eligible = (
        frame["valid_reference_count"].ge(10)
        & frame["source_mapping_coverage"].ge(0.8)
        & frame["field_mapping_coverage"].ge(0.8)
    )
    frame = frame[eligible].copy()
    frame["publication_era_5y"] = (
        frame["publication_year"].astype(int) // 5 * 5
    )
    frame["selection_hash"] = frame["paper_id"].astype(str).map(
        lambda value: hashlib.sha256(
            f"{salt}:{value}".encode("utf-8")
        ).hexdigest()
    )
    return (
        frame.sort_values("selection_hash", kind="stable")
        .groupby(
            ["domain12", "publication_era_5y"],
            sort=True,
            group_keys=False,
        )
        .head(int(max_per_domain_era))
        .sort_values(["publication_year", "paper_id"], kind="stable")
        .reset_index(drop=True)
    )


def _sample_reference_rows(
    sample: pd.DataFrame,
    paper_references: pd.DataFrame,
    reference_metadata: pd.DataFrame,
) -> Dict[str, List[Dict[str, Any]]]:
    sample_year = sample.set_index("paper_id")["publication_year"].to_dict()
    metadata = reference_metadata[
        ["reference_id", "reference_year", "source_id", "field_id"]
    ].copy()
    metadata["reference_id"] = metadata["reference_id"].map(
        normalize_openalex_id
    )
    metadata["reference_year"] = pd.to_numeric(
        metadata["reference_year"], errors="coerce"
    )
    joined = paper_references[
        paper_references["paper_id"].astype(str).isin(sample_year)
    ].merge(
        metadata,
        on="reference_id",
        how="left",
        validate="many_to_one",
    )
    joined["publication_year"] = joined["paper_id"].map(sample_year)
    joined = joined[
        joined["reference_year"].notna()
        & joined["reference_year"].lt(joined["publication_year"])
    ]
    output: Dict[str, List[Dict[str, Any]]] = {}
    for paper_id, group in joined.groupby("paper_id", sort=False):
        output[str(paper_id)] = [
            {
                "reference_id": str(row.reference_id),
                "reference_year": int(row.reference_year),
                "source_id": str(row.source_id or ""),
                "field_id": str(row.field_id or ""),
            }
            for row in group.itertuples(index=False)
        ]
    return output


def _subsample_reference_items(
    items: Sequence[Mapping[str, Any]],
    *,
    fraction: float,
    repetition: int,
    paper_id: str,
    salt: str,
) -> List[Mapping[str, Any]]:
    """Return the deterministic focal-bibliography resample."""
    sample_size = max(2, int(np.floor(float(fraction) * len(items))))
    rng = np.random.default_rng(
        _stable_seed(f"{repetition}:{paper_id}", salt=salt)
    )
    positions = (
        rng.choice(
            len(items),
            size=min(sample_size, len(items)),
            replace=False,
        )
        if items
        else np.asarray([], dtype=int)
    )
    return [items[int(position)] for position in positions]


def _reference_overlap_subsample_values(
    sample: pd.DataFrame,
    references: Mapping[str, Sequence[Mapping[str, Any]]],
    historical_paper_references: pd.DataFrame,
    *,
    fraction: float,
    repetitions: int,
    salt: str,
    reference_window_years: int | None = (
        REFERENCE_OVERLAP_REFERENCE_WINDOW_YEARS
    ),
    cociting_window_years: int | None = (
        REFERENCE_OVERLAP_COCITING_WINDOW_YEARS
    ),
) -> Dict[Tuple[int, str], float]:
    """Recompute source-faithful overlap novelty for every focal resample.

    Comparison bibliographies remain fixed publication-time history. Only the
    focal paper's references are resampled, matching the declared stability
    intervention and avoiding any use of future papers.
    """
    history = historical_paper_references.copy()
    history["publication_year"] = pd.to_numeric(
        history["publication_year"], errors="coerce"
    )
    history = history[history["publication_year"].notna()].copy()
    history["publication_year"] = history["publication_year"].astype(int)
    output: Dict[Tuple[int, str], float] = {}
    for year, year_papers in sample.groupby(
        "publication_year", sort=True
    ):
        year = int(year)
        prior = history[history["publication_year"].lt(year)]
        if cociting_window_years is not None:
            prior = prior[
                prior["publication_year"].ge(
                    year - int(cociting_window_years)
                )
            ]
        comparison_sets: Dict[str, Set[str]] = {}
        comparison_fields: Dict[str, str] = {}
        postings: Dict[Tuple[str, str], Set[str]] = {}
        for item in prior.itertuples(index=False):
            reference_ids = _as_list(item.reference_ids)
            try:
                reference_years = list(item.reference_years)
            except TypeError:
                reference_years = []
            prior_references = {
                reference_id
                for reference_id, reference_year in zip(
                    reference_ids, reference_years
                )
                if pd.notna(reference_year)
                and int(reference_year) < year
                and (
                    reference_window_years is None
                    or int(reference_year)
                    >= year - int(reference_window_years)
                )
            }
            field = str(item.openalex_primary_field or "")
            if not field or not prior_references:
                continue
            work_id = str(item.work_id)
            comparison_sets[work_id] = prior_references
            comparison_fields[work_id] = field
            for reference_id in prior_references:
                postings.setdefault((field, reference_id), set()).add(
                    work_id
                )
        for paper in year_papers.itertuples(index=False):
            paper_id = str(paper.paper_id)
            field = str(paper.openalex_primary_field or "")
            items = references.get(paper_id, [])
            for repetition in range(int(repetitions)):
                selected = _subsample_reference_items(
                    items,
                    fraction=fraction,
                    repetition=repetition,
                    paper_id=paper_id,
                    salt=salt,
                )
                focal = {
                    str(item["reference_id"])
                    for item in selected
                    if int(item["reference_year"]) < year
                    and (
                        reference_window_years is None
                        or int(item["reference_year"])
                        >= year - int(reference_window_years)
                    )
                }
                comparison_ids: Set[str] = set()
                for reference_id in focal:
                    comparison_ids.update(
                        postings.get((field, reference_id), set())
                    )
                comparisons = [
                    comparison_sets[work_id]
                    for work_id in sorted(comparison_ids)
                    if comparison_fields[work_id] == field
                ]
                output[(repetition, paper_id)] = (
                    reference_overlap_novelty(focal, comparisons)
                )
    return output


def _history_context_by_year(
    sample: pd.DataFrame,
    references: Mapping[str, Sequence[Mapping[str, str]]],
    historical_paper_sources: pd.DataFrame,
) -> Dict[int, Dict[str, Any]]:
    required_sources_by_year: Dict[int, Set[str]] = {}
    required_pairs_by_year: Dict[int, Set[Tuple[Any, Any]]] = {}
    all_sources: Set[str] = set()
    all_pairs: Set[Tuple[Any, Any]] = set()
    for row in sample.itertuples(index=False):
        sources = [
            item["source_id"]
            for item in references.get(str(row.paper_id), [])
            if item["source_id"]
        ]
        year = int(row.publication_year)
        required_sources_by_year.setdefault(year, set()).update(sources)
        required_pairs_by_year.setdefault(year, set()).update(
            _pair_set(sources)
        )
        all_sources.update(sources)
        all_pairs.update(_pair_set(sources))
    historical = historical_paper_sources.copy()
    historical["publication_year"] = pd.to_numeric(
        historical["publication_year"], errors="coerce"
    )
    historical = historical[
        historical["publication_year"].notna()
    ].sort_values(["publication_year", "work_id"], kind="stable")
    records = historical.itertuples(index=False)
    record = next(records, None)
    source_counts: Counter[str] = Counter()
    pair_counts: Counter[Tuple[Any, Any]] = Counter()
    n_history = 0
    output: Dict[int, Dict[str, Any]] = {}
    for year in sorted(required_sources_by_year):
        while (
            record is not None
            and int(record.publication_year) < year
        ):
            values = set(_as_list(record.cited_source_ids))
            source_counts.update(values & all_sources)
            pair_counts.update(
                pair for pair in _pair_set(list(values)) if pair in all_pairs
            )
            n_history += 1
            record = next(records, None)
        output[year] = {
            "source_counts": {
                source: int(source_counts[source])
                for source in required_sources_by_year[year]
            },
            "pair_counts": {
                pair: int(pair_counts[pair])
                for pair in required_pairs_by_year[year]
            },
            "n_historical_papers": int(n_history),
        }
    return output


def _annual_source_profiles(
    events: pd.DataFrame,
    years: Sequence[int],
    *,
    window_years: int,
) -> Dict[int, Dict[str, np.ndarray]]:
    output = {}
    for year in sorted(set(int(value) for value in years)):
        selected = events[
            events["source_year"].between(
                year - int(window_years),
                year - 1,
                inclusive="both",
            )
        ]
        if selected.empty:
            output[year] = {}
            continue
        matrix = selected.pivot_table(
            index="source_id",
            columns="target_field_id",
            values="citation_count",
            aggfunc="sum",
            fill_value=0.0,
        ).astype(float)
        output[year] = {
            str(source): matrix.loc[source].to_numpy(dtype=float)
            for source in matrix.index
        }
    return output


def _pair_distances(
    sources: Sequence[str],
    pair_counts: Mapping[Tuple[Any, Any], int],
    profiles: Mapping[str, np.ndarray],
) -> Dict[Tuple[Any, Any], float]:
    output = {}
    for pair in _pair_set(sources):
        if int(pair_counts.get(pair, 0)) != 0:
            continue
        left = profiles.get(str(pair[0]))
        right = profiles.get(str(pair[1]))
        if left is None or right is None:
            continue
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator <= 0.0:
            continue
        output[pair] = float(
            np.clip(1.0 - np.dot(left, right) / denominator, 0.0, 1.0)
        )
    return output


def _calculate_candidates(
    *,
    sources: Sequence[str],
    fields: Sequence[str],
    focal_field: str,
    context: Mapping[str, Any],
    field_distances: Mapping[Tuple[Any, Any], float],
    source_profiles: Mapping[str, np.ndarray],
) -> Dict[str, float]:
    pair_counts = context["pair_counts"]
    source_counts = context["source_counts"]
    n_history = int(context["n_historical_papers"])
    z_scores = marginal_pair_z_scores(
        sources, pair_counts, source_counts, n_history
    )
    hypergeometric_z_scores = hypergeometric_pair_z_scores(
        sources, pair_counts, source_counts, n_history
    )
    source_distances = _pair_distances(
        sources, pair_counts, source_profiles
    )
    try:
        occupied = occupied_field_distances(fields, field_distances)
        mean_distance = (
            float(np.mean(occupied)) if len(occupied) else float("nan")
        )
        max_distance = field_distance_quantile(
            fields, field_distances, quantile=1.0
        )
        p90_distance = field_distance_quantile(
            fields, field_distances, quantile=0.9
        )
        rao = rao_stirling_integration(fields, field_distances)
        div = div_index(
            fields,
            field_distances,
            total_categories=FIELD_TAXONOMY_SIZE,
        )
    except KeyError:
        mean_distance = max_distance = p90_distance = rao = div = float(
            "nan"
        )
    variety = float(len(set(fields))) if fields else float("nan")
    shannon = field_shannon_entropy(fields)
    return {
        "novelty_u_t0_source": novelty_u(
            sources, pair_counts, source_counts, n_history
        ),
        "source_pair_mean_surprisal": source_pair_mean_surprisal(
            sources, pair_counts, source_counts, n_history
        ),
        "low_frequency_source_pair_share": low_frequency_pair_share(
            sources, pair_counts, maximum_prior_count=1
        ),
        "uzzi_atypicality_p10_t0": uzzi_atypicality_p10(z_scores),
        "uzzi_conventionality_median_t0": (
            uzzi_conventionality_median(z_scores)
        ),
        "hypergeom_atypicality_p10_t0": uzzi_atypicality_p10(
            hypergeometric_z_scores
        ),
        "hypergeom_conventionality_median_t0": (
            uzzi_conventionality_median(hypergeometric_z_scores)
        ),
        "first_time_source_pair_any": first_time_pair_any(
            sources, pair_counts
        ),
        "first_time_source_pair_count": first_time_pair_count(
            sources, pair_counts
        ),
        "first_time_source_pair_share": first_time_source_pair_share(
            sources, pair_counts
        ),
        "first_time_source_pair_distance_sum": (
            first_time_pair_distance_sum(
                sources, pair_counts, source_distances
            )
        ),
        "first_time_source_pair_distance_mean": (
            first_time_source_pair_distance_mean(
                sources, pair_counts, source_distances
            )
        ),
        "field_variety": variety,
        "field_relative_variety": (
            field_relative_variety(
                fields, total_categories=FIELD_TAXONOMY_SIZE
            )
            if fields
            else float("nan")
        ),
        "reference_other_field_share": field_other_field_share(
            fields, focal_field=focal_field
        ),
        "field_gini_balance": field_gini_balance(fields),
        "field_shannon_entropy": shannon,
        "field_pielou_evenness": (
            shannon / math.log(len(set(fields)))
            if len(set(fields)) >= 2
            else float("nan")
        ),
        "field_gini_simpson": field_gini_simpson(fields),
        "field_hhi": field_hhi(fields),
        "field_hill_q0": (
            field_hill_number(fields, order=0)
            if fields
            else float("nan")
        ),
        "field_hill_q1": (
            field_hill_number(fields, order=1)
            if fields
            else float("nan")
        ),
        "field_hill_q2": (
            field_hill_number(fields, order=2)
            if fields
            else float("nan")
        ),
        "field_disparity_cosine_mean": mean_distance,
        "field_disparity_cosine_max": max_distance,
        "field_disparity_cosine_p90": p90_distance,
        "rao_stirling_integration": rao,
        "field_div_index": div,
        "rao_true_diversity_q2": true_diversity_from_rao(rao),
    }


def reference_subsampling_stability(
    sample: pd.DataFrame,
    full_features: pd.DataFrame,
    paper_references: pd.DataFrame,
    reference_metadata: pd.DataFrame,
    field_events: pd.DataFrame,
    historical_paper_sources: pd.DataFrame,
    historical_paper_references: pd.DataFrame,
    source_field_events: pd.DataFrame,
    *,
    fraction: float,
    repetitions: int,
    salt: str,
    field_profile_window_years: int,
    relative_error_denominator_policy: str = "absolute_value_epsilon",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute all feasible families after 80% reference resampling."""
    allowed_error_policies = {
        "absolute_value_epsilon",
        "median_absolute_floor",
    }
    if relative_error_denominator_policy not in allowed_error_policies:
        raise ValueError(
            "unknown relative-error denominator policy: "
            f"{relative_error_denominator_policy}"
        )
    references = _sample_reference_rows(
        sample, paper_references, reference_metadata
    )
    history = _history_context_by_year(
        sample, references, historical_paper_sources
    )
    overlap_values = _reference_overlap_subsample_values(
        sample,
        references,
        historical_paper_references,
        fraction=fraction,
        repetitions=int(repetitions),
        salt=salt,
    )
    field_distance_by_year = annual_field_distances(
        field_events,
        sample["publication_year"].unique(),
        window_years=int(field_profile_window_years),
    )
    source_profiles_by_year = _annual_source_profiles(
        source_field_events,
        sample["publication_year"].unique(),
        window_years=int(field_profile_window_years),
    )
    full = full_features[
        ["paper_id", *[name for name in STABILITY_CODE_NAMES if name in full_features]]
    ].set_index("paper_id")
    sample_indexed = sample.set_index("paper_id")
    rows = []
    for repetition in range(int(repetitions)):
        recomputed_rows = []
        for paper_id, paper in sample_indexed.iterrows():
            items = references.get(str(paper_id), [])
            selected = _subsample_reference_items(
                items,
                fraction=fraction,
                repetition=repetition,
                paper_id=str(paper_id),
                salt=salt,
            )
            sources = [
                item["source_id"] for item in selected if item["source_id"]
            ]
            fields = [
                item["field_id"] for item in selected if item["field_id"]
            ]
            year = int(paper["publication_year"])
            values = _calculate_candidates(
                sources=sources,
                fields=fields,
                focal_field=str(paper["openalex_primary_field"] or ""),
                context=history[year],
                field_distances=field_distance_by_year.get(year, {}),
                source_profiles=source_profiles_by_year.get(year, {}),
            )
            values["reference_overlap_novelty_t0"] = overlap_values.get(
                (repetition, str(paper_id)), float("nan")
            )
            values["paper_id"] = str(paper_id)
            recomputed_rows.append(values)
        recomputed = pd.DataFrame(recomputed_rows).set_index("paper_id")
        for metric in STABILITY_CODE_NAMES:
            if metric not in full or metric not in recomputed:
                continue
            paired = pd.concat(
                [full[metric], recomputed[metric]],
                axis=1,
                keys=["full", "subsample"],
            ).dropna()
            full_values = paired["full"].to_numpy(dtype=float)
            absolute_full = np.abs(full_values)
            scale_floor = (
                float(np.median(absolute_full))
                if (
                    relative_error_denominator_policy
                    == "median_absolute_floor"
                    and len(absolute_full)
                )
                else 0.0
            )
            denominator = np.maximum.reduce(
                (
                    absolute_full,
                    np.full(len(absolute_full), scale_floor),
                    np.full(len(absolute_full), 1e-6),
                )
            )
            relative_error = (
                np.abs(
                    paired["subsample"].to_numpy(dtype=float)
                    - full_values
                )
                / denominator
            )
            rows.append(
                {
                    "repetition": repetition + 1,
                    "code_name": metric,
                    "n_paired": len(paired),
                    "spearman": safe_spearman(
                        paired["full"], paired["subsample"]
                    ),
                    "median_relative_error": (
                        float(np.median(relative_error))
                        if len(relative_error)
                        else np.nan
                    ),
                    "relative_error_scale_floor": scale_floor,
                    "relative_error_denominator_policy": (
                        relative_error_denominator_policy
                    ),
                }
            )
    repetitions_frame = pd.DataFrame(rows)
    summary_rows = []
    for code_name, group in repetitions_frame.groupby(
        "code_name", sort=True
    ):
        rho = pd.to_numeric(group["spearman"], errors="coerce")
        error = pd.to_numeric(
            group["median_relative_error"], errors="coerce"
        )
        complete = bool(
            len(group) == int(repetitions)
            and np.isfinite(rho).all()
            and np.isfinite(error).all()
        )
        summary_rows.append(
            {
                "code_name": str(code_name),
                "n_repetitions": len(group),
                "all_repetitions_valid": int(complete),
                "n_paired_min": int(group["n_paired"].min()),
                "stability_spearman": (
                    float(rho.min()) if complete else np.nan
                ),
                "stability_spearman_median": (
                    float(rho.median()) if complete else np.nan
                ),
                "stability_median_relative_error": (
                    float(error.max()) if complete else np.nan
                ),
                "stability_median_relative_error_median": (
                    float(error.median()) if complete else np.nan
                ),
                "relative_error_scale_floor_median": float(
                    pd.to_numeric(
                        group["relative_error_scale_floor"],
                        errors="coerce",
                    ).median()
                ),
                "relative_error_denominator_policy": (
                    relative_error_denominator_policy
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    return repetitions_frame, summary


def _exact_hypergeometric_z_scores(
    sources: Sequence[str],
    pair_counts: Mapping[Tuple[Any, Any], int],
    source_counts: Mapping[str, int],
    n_historical_papers: int,
) -> List[float]:
    values = []
    population = int(n_historical_papers)
    if population <= 1:
        return values
    for left, right in sorted(_pair_set(sources), key=str):
        left_count = int(source_counts.get(str(left), 0))
        right_count = int(source_counts.get(str(right), 0))
        if left_count <= 0 or right_count <= 0:
            continue
        distribution = hypergeom(
            M=population,
            n=left_count,
            N=right_count,
        )
        variance = float(distribution.var())
        if variance <= 0.0:
            continue
        observed = float(pair_counts.get(canonical_pair(left, right), 0))
        values.append(
            float((observed - float(distribution.mean())) / math.sqrt(variance))
        )
    return values


def approximation_fidelity(
    sample: pd.DataFrame,
    paper_references: pd.DataFrame,
    reference_metadata: pd.DataFrame,
    historical_paper_sources: pd.DataFrame,
) -> pd.DataFrame:
    """Compare optimized U/Uzzi code with explicit reference equations."""
    references = _sample_reference_rows(
        sample, paper_references, reference_metadata
    )
    history = _history_context_by_year(
        sample, references, historical_paper_sources
    )
    rows = []
    for paper in sample.itertuples(index=False):
        sources = [
            item["source_id"]
            for item in references.get(str(paper.paper_id), [])
            if item["source_id"]
        ]
        context = history[int(paper.publication_year)]
        pair_counts = context["pair_counts"]
        source_counts = context["source_counts"]
        n_history = int(context["n_historical_papers"])
        approximate_z = marginal_pair_z_scores(
            sources, pair_counts, source_counts, n_history
        )
        analytic_z = hypergeometric_pair_z_scores(
            sources, pair_counts, source_counts, n_history
        )
        exact_z = _exact_hypergeometric_z_scores(
            sources, pair_counts, source_counts, n_history
        )
        optimized_u = novelty_u(
            sources, pair_counts, source_counts, n_history
        )
        explicit_commonness = []
        for left, right in _pair_set(sources):
            left_count = int(source_counts.get(str(left), 0))
            right_count = int(source_counts.get(str(right), 0))
            if left_count <= 0 or right_count <= 0 or n_history <= 0:
                continue
            observed = float(
                pair_counts.get(canonical_pair(left, right), 0)
            )
            observed = observed if observed > 0.0 else 0.5
            explicit_commonness.append(
                observed * n_history / (left_count * right_count)
            )
        exact_u = (
            -math.log(max(float(np.quantile(explicit_commonness, 0.1)), np.finfo(float).tiny))
            if explicit_commonness
            else np.nan
        )
        rows.append(
            {
                "paper_id": str(paper.paper_id),
                "novelty_u_optimized": optimized_u,
                "novelty_u_reference": exact_u,
                "uzzi_p10_optimized": uzzi_atypicality_p10(approximate_z),
                "uzzi_p10_reference": uzzi_atypicality_p10(exact_z),
                "uzzi_median_optimized": (
                    uzzi_conventionality_median(approximate_z)
                ),
                "uzzi_median_reference": (
                    uzzi_conventionality_median(exact_z)
                ),
                "hypergeom_p10_analytic": uzzi_atypicality_p10(analytic_z),
                "hypergeom_p10_scipy_reference": (
                    uzzi_atypicality_p10(exact_z)
                ),
                "hypergeom_median_analytic": (
                    uzzi_conventionality_median(analytic_z)
                ),
                "hypergeom_median_scipy_reference": (
                    uzzi_conventionality_median(exact_z)
                ),
            }
        )
    frame = pd.DataFrame(rows)
    summaries = []
    mappings = {
        "novelty_u_t0_source": (
            "novelty_u_optimized",
            "novelty_u_reference",
        ),
        "hypergeom_atypicality_p10_t0": (
            "hypergeom_p10_analytic",
            "hypergeom_p10_scipy_reference",
        ),
        "hypergeom_conventionality_median_t0": (
            "hypergeom_median_analytic",
            "hypergeom_median_scipy_reference",
        ),
    }
    for code_name, (approximate, exact) in mappings.items():
        paired = frame[[approximate, exact]].dropna()
        denominator = np.maximum(
            np.abs(paired[exact].to_numpy(dtype=float)), 1e-6
        )
        error = (
            np.abs(
                paired[approximate].to_numpy(dtype=float)
                - paired[exact].to_numpy(dtype=float)
            )
            / denominator
        )
        summaries.append(
            {
                "code_name": code_name,
                "n_paired": len(paired),
                "approximation_spearman": safe_spearman(
                    paired[approximate], paired[exact]
                ),
                "approximation_median_relative_error": (
                    float(np.median(error)) if len(error) else np.nan
                ),
                "reference_scope": "independent SciPy hypergeometric equation",
            }
        )
    for code_name in (
        "uzzi_atypicality_p10_t0",
        "uzzi_conventionality_median_t0",
    ):
        summaries.append(
            {
                "code_name": code_name,
                "n_paired": 0,
                "approximation_spearman": np.nan,
                "approximation_median_relative_error": np.nan,
                "reference_scope": (
                    "not validated against the original degree-preserving "
                    "edge-swap Monte Carlo null; forced out of primary use"
                ),
            }
        )
    return pd.DataFrame(summaries)


def propose_candidate_decisions(
    registry: CandidateRegistryV61,
    coverage: pd.DataFrame,
    stability: pd.DataFrame,
    approximation: pd.DataFrame,
) -> pd.DataFrame:
    """Apply fixed evidence and measurement gates without outcomes."""
    coverage_by_code = coverage.set_index("code_name").to_dict("index")
    stability_by_code = stability.set_index("code_name").to_dict("index")
    approximation_by_code = approximation.set_index("code_name").to_dict(
        "index"
    )
    thresholds = registry.thresholds
    rows = []
    for candidate in registry.candidates.values():
        cov = coverage_by_code.get(candidate.code_name, {})
        stable = stability_by_code.get(candidate.code_name, {})
        fidelity = approximation_by_code.get(candidate.code_name, {})
        coverage_pass = (
            float(cov.get("overall_coverage", 0.0))
            >= thresholds.overall_coverage_min
            and float(cov.get("minimum_domain_coverage", 0.0))
            >= thresholds.each_domain_coverage_min
        )
        stability_pass = (
            float(stable.get("stability_spearman", -1.0))
            >= thresholds.stability_spearman_min
            and float(
                stable.get(
                    "stability_median_relative_error",
                    float("inf"),
                )
            )
            <= thresholds.stability_median_relative_error_max
        )
        approximation_required = candidate.candidate_id in {
            "A1.NOVELTY_U",
            "A2.UZZI_P10",
            "A2.UZZI_MEDIAN",
            "A2.HYPERGEOM_P10",
            "A2.HYPERGEOM_MEDIAN",
        }
        approximation_pass = (
            not approximation_required
            or (
                float(fidelity.get("approximation_spearman", -1.0))
                >= thresholds.approximation_spearman_min
                and float(
                    fidelity.get(
                        "approximation_median_relative_error",
                        float("inf"),
                    )
                )
                <= thresholds.approximation_median_relative_error_max
            )
        )
        eligible = (
            all(candidate.gate_checks[f"I{index}"] for index in range(1, 6))
            and coverage_pass
            and stability_pass
            and approximation_pass
            and candidate.empirical_screen.toy_test_pass
            and candidate.empirical_screen.temporal_test_pass
            and bool(cov.get("nondegenerate_test_pass", 0))
        )
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "code_name": candidate.code_name,
                "angle_id": candidate.angle_id,
                "mathematical_family": candidate.mathematical_family,
                "total_n": cov.get("n_rows", np.nan),
                "eligible_n": cov.get("n_eligible", np.nan),
                "coverage_denominator_policy": cov.get(
                    "denominator_policy", "all_papers"
                ),
                "raw_overall_coverage": cov.get(
                    "raw_overall_coverage",
                    cov.get("overall_coverage", np.nan),
                ),
                "raw_minimum_domain_coverage": cov.get(
                    "raw_minimum_domain_coverage",
                    cov.get("minimum_domain_coverage", np.nan),
                ),
                "overall_coverage": cov.get("overall_coverage", np.nan),
                "minimum_domain_coverage": cov.get(
                    "minimum_domain_coverage", np.nan
                ),
                "stability_spearman": stable.get(
                    "stability_spearman", np.nan
                ),
                "stability_median_relative_error": stable.get(
                    "stability_median_relative_error", np.nan
                ),
                "relative_error_denominator_policy": stable.get(
                    "relative_error_denominator_policy",
                    "absolute_value_epsilon",
                ),
                "relative_error_scale_floor": stable.get(
                    "relative_error_scale_floor_median", np.nan
                ),
                "approximation_spearman": fidelity.get(
                    "approximation_spearman", np.nan
                ),
                "approximation_median_relative_error": fidelity.get(
                    "approximation_median_relative_error", np.nan
                ),
                "coverage_pass": int(coverage_pass),
                "stability_pass": int(stability_pass),
                "approximation_pass": int(approximation_pass),
                "toy_test_pass": int(
                    candidate.empirical_screen.toy_test_pass
                ),
                "temporal_test_pass": int(
                    candidate.empirical_screen.temporal_test_pass
                ),
                "nondegenerate_test_pass": int(
                    cov.get("nondegenerate_test_pass", 0)
                ),
                "outcome_used": 0,
                "eligible_all_runtime_gates": int(eligible),
            }
        )
    decisions = pd.DataFrame(rows)
    eligible_ids = set(
        decisions.loc[
            decisions["eligible_all_runtime_gates"].eq(1),
            "candidate_id",
        ].astype(str)
    )
    winners: Dict[str, str] = {}
    priorities: Dict[str, Tuple[str, int]] = {}
    for group_name, candidate_ids in PRIMARY_SELECTION_GROUPS:
        for priority, candidate_id in enumerate(candidate_ids, start=1):
            priorities[candidate_id] = (group_name, priority)
        winner = next(
            (
                candidate_id
                for candidate_id in candidate_ids
                if candidate_id in eligible_ids
            ),
            None,
        )
        if winner is not None:
            winners[winner] = group_name
    roles = []
    selection_groups = []
    selection_priorities = []
    selection_reasons = []
    for row in decisions.itertuples(index=False):
        candidate = registry.candidates[str(row.candidate_id)]
        group_name, priority = priorities.get(
            candidate.candidate_id, ("not_primary_competition", 0)
        )
        is_winner = candidate.candidate_id in winners
        if is_winner:
            role = "primary"
            reason = (
                "First outcome-blind eligible representative in the frozen "
                f"{group_name} priority order."
            )
        elif (
            candidate.final_role not in {"excluded", "exploratory"}
            and candidate.implementation_name
        ):
            role = "sensitivity"
            reason = (
                "Implemented candidate retained for sensitivity; it either "
                "failed a primary gate or lost a frozen redundancy-family "
                "competition."
            )
        else:
            role = candidate.final_role
            reason = candidate.decision_reason
        roles.append(role)
        selection_groups.append(group_name)
        selection_priorities.append(priority)
        selection_reasons.append(reason)
    decisions["selection_group"] = selection_groups
    decisions["selection_priority"] = selection_priorities
    decisions["preference_selected"] = decisions["candidate_id"].isin(
        winners
    ).astype(int)
    decisions["proposed_final_role"] = roles
    decisions["proposed_decision_reason"] = selection_reasons
    return decisions


def run_candidate_screening(
    *,
    project_root: Path,
    registry_path: Path,
    dataset_dir: Path,
    output_root: Path,
    repetitions: int = 20,
    max_per_domain_era: int = 100,
    seed_salt: str = "aspr-v6.1-stability-20260724",
    coverage_denominator_policy: str = "all_papers",
    relative_error_denominator_policy: str = "absolute_value_epsilon",
) -> Tuple[Mapping[str, Any], Path]:
    """Run and persist the complete no-outcome candidate screen."""
    project_root = Path(project_root).resolve()
    registry_path = Path(registry_path).resolve()
    dataset_dir = Path(dataset_dir).resolve()
    registry = load_candidate_registry_v6_1(registry_path)
    search_log_path = verify_search_log(registry, project_root)
    features_path = dataset_dir / "innovation_candidate_features.parquet"
    historical_references_path = (
        dataset_dir / "historical_paper_references.parquet"
    )
    if not historical_references_path.is_file():
        raise FileNotFoundError(
            "source-faithful overlap context is required before screening: "
            f"{historical_references_path}"
        )
    lineage = {
        "screening_version": SCREENING_VERSION,
        "screening_implementation_sha256": sha256_file(
            Path(__file__).resolve()
        ),
        "feature_formula_implementation_sha256": sha256_file(
            project_root
            / "aspr"
            / "nature_multihorizon"
            / "features_v6_1.py"
        ),
        "legacy_formula_implementation_sha256": sha256_file(
            project_root
            / "aspr"
            / "nature_multihorizon"
            / "features_v6.py"
        ),
        "materialization_implementation_sha256": sha256_file(
            project_root
            / "aspr"
            / "nature_multihorizon"
            / "materialize_v6_1.py"
        ),
        "candidate_catalog_sha256": candidate_registry_sha256(registry),
        "search_log_sha256": registry.search_log_sha256,
        "candidate_features_sha256": sha256_file(features_path),
        "historical_paper_references_sha256": sha256_file(
            historical_references_path
        ),
        "future_influence_outcomes_used": False,
        "network_used": False,
        "repetitions": int(repetitions),
        "max_per_domain_era": int(max_per_domain_era),
        "seed_salt": seed_salt,
        "coverage_denominator_policy": coverage_denominator_policy,
        "relative_error_denominator_policy": (
            relative_error_denominator_policy
        ),
    }
    run_hash = _canonical_hash(lineage)
    output_dir = Path(output_root).resolve() / (
        f"screening_{run_hash.removeprefix('sha256:')[:12]}"
    )
    manifest_path = output_dir / "screening_manifest.json"
    if manifest_path.is_file():
        return (
            json.loads(manifest_path.read_text(encoding="utf-8")),
            output_dir,
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    features = pd.read_parquet(features_path)
    papers = pd.read_parquet(
        dataset_dir / "papers_primary_articles.parquet"
    )
    paper_references = pd.read_parquet(
        dataset_dir / "paper_references.parquet"
    )
    reference_metadata = pd.read_parquet(
        dataset_dir / "reference_metadata.parquet"
    )
    field_events = pd.read_parquet(
        dataset_dir / "field_citation_events_aggregated.parquet"
    )
    historical = pd.read_parquet(
        dataset_dir / "historical_paper_sources.parquet"
    )
    historical_references = pd.read_parquet(
        historical_references_path
    )
    source_field_events = pd.read_parquet(
        dataset_dir / "source_field_citation_events.parquet"
    )
    source_year = pd.to_numeric(
        features["source_max_year"], errors="coerce"
    )
    publication_year = pd.to_numeric(
        features["publication_year"], errors="coerce"
    )
    temporal_leakage_rows = int(
        (
            source_year.notna()
            & publication_year.notna()
            & source_year.ge(publication_year)
        ).sum()
    )
    if temporal_leakage_rows:
        raise ValueError(
            "candidate feature view contains publication-time leakage"
        )
    coverage, domain_coverage = coverage_audit(
        registry,
        features,
        denominator_policy=coverage_denominator_policy,
    )
    sample = select_stability_sample(
        papers,
        features,
        max_per_domain_era=int(max_per_domain_era),
        salt=seed_salt,
    )
    repetitions_frame, stability = reference_subsampling_stability(
        sample,
        features,
        paper_references,
        reference_metadata,
        field_events,
        historical,
        historical_references,
        source_field_events,
        fraction=0.8,
        repetitions=int(repetitions),
        salt=seed_salt,
        field_profile_window_years=5,
        relative_error_denominator_policy=(
            relative_error_denominator_policy
        ),
    )
    approximation = approximation_fidelity(
        sample,
        paper_references,
        reference_metadata,
        historical,
    )
    decisions = propose_candidate_decisions(
        registry, coverage, stability, approximation
    )
    summary = {
        "n_candidates": len(registry.candidates),
        "n_stability_sample": len(sample),
        "n_proposed_primary": int(
            decisions["proposed_final_role"].eq("primary").sum()
        ),
        "primary_candidate_ids": decisions.loc[
            decisions["proposed_final_role"].eq("primary"),
            "candidate_id",
        ].tolist(),
        "all_five_angles_have_primary": bool(
            set(
                decisions.loc[
                    decisions["proposed_final_role"].eq("primary"),
                    "angle_id",
                ]
            )
            == set(registry.observation_angles)
        ),
        "future_influence_outcomes_used": False,
        "temporal_leakage_rows": temporal_leakage_rows,
        "coverage_denominator_policy": coverage_denominator_policy,
        "relative_error_denominator_policy": (
            relative_error_denominator_policy
        ),
        "search_log_path": str(search_log_path),
        "selection_warning": (
            "The fixed preferred representative is not replaced because of "
            "OOF. If a preferred candidate fails, the registry must document "
            "an evidence/measurement revision before any OOF run."
        ),
    }
    paths = {
        "coverage": output_dir / "candidate_coverage.csv",
        "domain_coverage": output_dir / "candidate_domain_coverage.csv",
        "sample": output_dir / "stability_sample.parquet",
        "repetitions": output_dir
        / "reference_subsampling_repetitions.csv",
        "stability": output_dir / "reference_subsampling_summary.csv",
        "approximation": output_dir / "approximation_fidelity.csv",
        "decisions": output_dir / "candidate_decisions.csv",
        "summary": output_dir / "screening_summary.json",
    }
    coverage.to_csv(paths["coverage"], index=False)
    domain_coverage.to_csv(paths["domain_coverage"], index=False)
    sample.to_parquet(paths["sample"], index=False)
    repetitions_frame.to_csv(paths["repetitions"], index=False)
    stability.to_csv(paths["stability"], index=False)
    approximation.to_csv(paths["approximation"], index=False)
    decisions.to_csv(paths["decisions"], index=False)
    paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "artifact_kind": "aspr_v6_1_outcome_blind_candidate_screening",
        "lineage": lineage,
        "summary": summary,
        "outputs": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        },
    }
    manifest["artifact_id"] = _canonical_hash(manifest)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return manifest, output_dir


def _finite_or_none(value: Any) -> float | None:
    """Return a finite float for JSON, otherwise an explicit null."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _verified_manifest_output(
    manifest: Mapping[str, Any],
    name: str,
) -> Path:
    """Resolve one screening output and verify its recorded content hash."""
    item = manifest["outputs"][name]
    path = Path(item["path"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != item["sha256"]:
        raise ValueError(f"screening output changed after manifest: {name}")
    return path


def freeze_registry_from_screening(
    *,
    project_root: Path,
    catalog_path: Path,
    screening_manifest_path: Path,
    output_path: Path,
) -> CandidateRegistryV61:
    """Create the final registry from outcome-blind screening only.

    This function never opens target labels or OOF predictions. It verifies the
    catalog and every screening input hash before applying the predeclared
    coverage, stability, formula-fidelity, and redundancy rules.
    """
    project_root = Path(project_root).resolve()
    catalog_path = Path(catalog_path).resolve()
    output_path = Path(output_path).resolve()
    catalog = load_candidate_registry_v6_1(catalog_path)
    if catalog.registry_stage != "candidate_catalog":
        raise ValueError("registry-freeze input must be the candidate catalog")
    verify_search_log(catalog, project_root)
    manifest = json.loads(
        Path(screening_manifest_path).resolve().read_text(encoding="utf-8")
    )
    if (
        manifest.get("artifact_kind")
        != "aspr_v6_1_outcome_blind_candidate_screening"
    ):
        raise ValueError("unexpected screening artifact kind")
    lineage = manifest.get("lineage") or {}
    if lineage.get("candidate_catalog_sha256") != candidate_registry_sha256(
        catalog
    ):
        raise ValueError("screening was not produced from this catalog")
    if (
        bool(lineage.get("future_influence_outcomes_used"))
        or bool(lineage.get("network_used"))
        or bool(
            (manifest.get("summary") or {}).get(
                "future_influence_outcomes_used"
            )
        )
    ):
        raise ValueError("outcome or network use forbids registry freezing")
    decisions = pd.read_csv(
        _verified_manifest_output(manifest, "decisions")
    )
    if decisions["candidate_id"].duplicated().any():
        raise ValueError("screening decisions contain duplicate candidates")
    if set(decisions["candidate_id"].astype(str)) != set(catalog.candidates):
        raise ValueError("screening decisions do not cover the catalog exactly")
    if not decisions["outcome_used"].eq(0).all():
        raise ValueError("outcomes were used in candidate decisions")
    decision_by_id = decisions.set_index("candidate_id").to_dict("index")
    formula_candidates = {
        "A1.NOVELTY_U",
        "A2.UZZI_P10",
        "A2.UZZI_MEDIAN",
        "A2.HYPERGEOM_P10",
        "A2.HYPERGEOM_MEDIAN",
    }
    candidate_payloads: Dict[str, Dict[str, Any]] = {}
    artifact_id = str(manifest["artifact_id"])
    for candidate_id, candidate in catalog.candidates.items():
        row = decision_by_id[candidate_id]
        screen = {
            "total_n": (
                int(row["total_n"])
                if _finite_or_none(row.get("total_n")) is not None
                else None
            ),
            "eligible_n": (
                int(row["eligible_n"])
                if _finite_or_none(row.get("eligible_n")) is not None
                else None
            ),
            "coverage_denominator_policy": str(
                row.get("coverage_denominator_policy")
            ),
            "raw_overall_coverage": _finite_or_none(
                row.get("raw_overall_coverage")
            ),
            "raw_minimum_domain_coverage": _finite_or_none(
                row.get("raw_minimum_domain_coverage")
            ),
            "overall_coverage": _finite_or_none(
                row.get("overall_coverage")
            ),
            "minimum_domain_coverage": _finite_or_none(
                row.get("minimum_domain_coverage")
            ),
            "stability_spearman": _finite_or_none(
                row.get("stability_spearman")
            ),
            "stability_median_relative_error": _finite_or_none(
                row.get("stability_median_relative_error")
            ),
            "relative_error_denominator_policy": str(
                row.get("relative_error_denominator_policy")
            ),
            "relative_error_scale_floor": _finite_or_none(
                row.get("relative_error_scale_floor")
            ),
            "approximation_spearman": _finite_or_none(
                row.get("approximation_spearman")
            ),
            "approximation_median_relative_error": _finite_or_none(
                row.get("approximation_median_relative_error")
            ),
            "toy_test_pass": bool(row.get("toy_test_pass")),
            "temporal_test_pass": bool(row.get("temporal_test_pass")),
            "nondegenerate_test_pass": bool(
                row.get("nondegenerate_test_pass")
            ),
            "screening_artifact_id": artifact_id,
        }
        gates = dict(candidate.gate_checks)
        gates["I6"] = bool(row.get("coverage_pass"))
        gates["I7"] = bool(row.get("stability_pass"))
        if candidate_id in formula_candidates:
            gates["I8"] = bool(row.get("approximation_pass"))
        gates["I9"] = bool(
            gates["I9"]
            and screen["toy_test_pass"]
            and screen["temporal_test_pass"]
            and screen["nondegenerate_test_pass"]
        )
        payload = candidate.model_dump(mode="json")
        payload.update(
            {
                "gate_checks": gates,
                "empirical_screen": screen,
                "final_role": str(row["proposed_final_role"]),
                "decision_reason": str(row["proposed_decision_reason"]),
                "oof_used_for_selection": False,
            }
        )
        candidate_payloads[candidate_id] = payload
    registry_payload = catalog.model_dump(mode="json")
    registry_payload.update(
        {
            "registry_version": (
                "6.1-screened-frozen-before-oof-2026-07-24"
            ),
            "registry_stage": (
                "posthoc_versioned_extension_frozen_before_oof"
            ),
            "disclosure": (
                f"{catalog.disclosure} Final roles were generated from "
                f"outcome-blind screening artifact {artifact_id}; no target "
                "label or OOF prediction was read."
            ),
            "candidates": candidate_payloads,
        }
    )
    registry = CandidateRegistryV61.model_validate(registry_payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            registry.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return registry


__all__ = [
    "PRIMARY_SELECTION_GROUPS",
    "SCREENING_VERSION",
    "STABILITY_CODE_NAMES",
    "approximation_fidelity",
    "coverage_audit",
    "freeze_registry_from_screening",
    "propose_candidate_decisions",
    "reference_subsampling_stability",
    "run_candidate_screening",
    "select_stability_sample",
]

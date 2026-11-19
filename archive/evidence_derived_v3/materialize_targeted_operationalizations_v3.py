from __future__ import annotations

import argparse
import itertools
import json
import math
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

import indicators
from common import (
    DATABASE_PATH,
    OUTPUT_DIR,
    sha256_file,
    write_json,
)
from database import initialize


ROOT = Path(__file__).resolve().parent
ASPR_ROOT = ROOT.parents[1]
CANDIDATE_DIR = OUTPUT_DIR / "candidate_t0_feature_matrix_v3"
DEFAULT_CANDIDATE_MATRIX = (
    CANDIDATE_DIR / "candidate_t0_feature_matrix_v3.parquet"
)
DEFAULT_CANDIDATE_REPORT = (
    CANDIDATE_DIR / "candidate_t0_feature_matrix_v3_report.json"
)
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "targeted_operationalizations_v3"
DEFAULT_MATRIX_NAME = "targeted_operationalized_features_v3.parquet"
DEFAULT_REGISTRY_NAME = "targeted_operationalization_registry_v3.json"
DEFAULT_MAPPING_NAME = "targeted_feature_data_mapping_v3.json"
IMPLEMENTATION_PATH = Path(__file__).resolve()
DEFINITION_VERSION = "targeted_operationalizations_v3_20260730"
UPSTREAM_IMPLEMENTATIONS: Dict[str, tuple[Path, str]] = {
    "EF0318": (
        ASPR_ROOT / "gear" / "nature_multihorizon" / "features_v6.py",
        "field_variety",
    ),
    "EF0312": (
        ASPR_ROOT / "gear" / "nature_multihorizon" / "features_v6_1.py",
        "field_gini_balance",
    ),
    "EF0315": (
        ASPR_ROOT / "gear" / "nature_multihorizon" / "features_v6.py",
        "field_disparity_mean",
    ),
    "EF0309": (
        ASPR_ROOT / "gear" / "nature_multihorizon" / "features_v6.py",
        "rao_stirling_integration",
    ),
    "EF0052": (
        ROOT / "materialize_backward_citation_age_v3.py",
        "compute_backward_citation_age",
    ),
    "EF0238": (
        ASPR_ROOT
        / "aspr"
        / "nature_multihorizon"
        / "prediction_features_v6.py",
        "build_bibliographic_opportunity_features",
    ),
    "EF0038": (
        ASPR_ROOT
        / "aspr"
        / "nature_multihorizon"
        / "openalex_controls_v6_1.py",
        "_target_metadata",
    ),
    "EF0314": (
        ROOT / "materialize_candidate_t0_features_v3.py",
        "_reference_counts",
    ),
    "EF0188": (
        ASPR_ROOT
        / "aspr"
        / "nature_multihorizon"
        / "openalex_controls_v6_1.py",
        "_target_metadata",
    ),
    "EF0186": (
        ROOT / "materialize_candidate_t0_features_v3.py",
        "build_matrix",
    ),
    "EF0197": (
        ROOT / "materialize_candidate_t0_features_v3.py",
        "_rename_columns",
    ),
    "EF0307": (
        ROOT / "materialize_candidate_t0_features_v3.py",
        "build_matrix",
    ),
}

FEATURE_SPECS: Dict[str, Dict[str, Any]] = {
    "EF0313": {
        "column": "reference_combination_novelty",
        "input_columns": [
            "reference_combination_novelty",
            "reference_overlap_comparison_count",
            "reference_overlap_reference_count",
        ],
        "protocol_missing_rule": (
            "Return missing when the focal reference set is empty, no "
            "strictly prior same-primary-field paper co-cites a focal "
            "reference, or the frozen all-prior comparison cannot be built."
        ),
        "denominator_zero_rule": (
            "The denominator is the number of eligible comparison papers; "
            "zero eligible comparisons returns missing."
        ),
        "observed_empty_set_rule": (
            "An observed empty focal reference set is missing, not novelty "
            "zero or one."
        ),
        "incomplete_coverage_rule": (
            "Use only references with frozen identifiers and retain focal "
            "reference and comparison counts; missing comparison context "
            "returns missing."
        ),
        "transform_and_unit_rule": (
            "Use the untransformed unitless one-minus-mean-Jaccard value on "
            "[0,1]; higher values mean less reference overlap."
        ),
        "mask": "overlap_support",
    },
    "EF0318": {
        "column": "reference_variety",
        "input_columns": [
            "reference_variety",
            "reference_field_mapping_coverage",
        ],
        "protocol_missing_rule": (
            "Return missing when no focal reference has a mapped field; do "
            "not infer unmapped reference fields."
        ),
        "denominator_zero_rule": (
            "No mathematical denominator is used; zero mapped references "
            "returns missing."
        ),
        "observed_empty_set_rule": (
            "A fully observed empty bibliography is missing for the "
            "source-defined referenced-category count, not a valid zero."
        ),
        "incomplete_coverage_rule": (
            "Count distinct fields among mapped references and report field "
            "mapping coverage; do not impute categories for unmapped items."
        ),
        "transform_and_unit_rule": (
            "Retain the raw number of distinct referenced fields; higher "
            "counts mean greater reference variety."
        ),
        "mask": "mapped_reference_support",
    },
    "EF0312": {
        "column": "reference_balance",
        "input_columns": [
            "reference_balance",
            "reference_variety",
            "reference_field_mapping_coverage",
        ],
        "protocol_missing_rule": (
            "Return missing when no focal reference field is observed; one "
            "occupied category has balance one within observed support."
        ),
        "denominator_zero_rule": (
            "The Gini mean/count denominator is undefined for no occupied "
            "category, which returns missing."
        ),
        "observed_empty_set_rule": (
            "An empty observed field distribution returns missing; it is "
            "not perfectly balanced."
        ),
        "incomplete_coverage_rule": (
            "Compute one-minus-Gini only on mapped references and retain "
            "field mapping coverage; do not impute unmapped fields."
        ),
        "transform_and_unit_rule": (
            "Retain the untransformed unitless one-minus-Gini value on "
            "[0,1]; higher values indicate greater balance."
        ),
        "mask": "mapped_reference_support",
    },
    "EF0315": {
        "column": "reference_disparity",
        "input_columns": [
            "reference_disparity",
            "reference_variety",
            "reference_field_mapping_coverage",
        ],
        "protocol_missing_rule": (
            "Return missing for fewer than two occupied reference fields or "
            "when any required strictly prior field distance is unavailable."
        ),
        "denominator_zero_rule": (
            "The pair denominator n(n-1) is zero for fewer than two fields; "
            "return missing."
        ),
        "observed_empty_set_rule": (
            "Zero or one occupied field has no observed pairwise disparity "
            "and returns missing, not zero."
        ),
        "incomplete_coverage_rule": (
            "Use mapped reference fields and the frozen y-5 through y-1 "
            "distance matrix; any missing occupied-field pair returns "
            "missing and field mapping coverage is retained."
        ),
        "transform_and_unit_rule": (
            "Retain mean one-minus-cosine distance on [0,1]; higher values "
            "mean greater cognitive disparity."
        ),
        "mask": "two_field_support",
    },
    "EF0309": {
        "column": "rao_stirling_diversity",
        "input_columns": [
            "rao_stirling_diversity",
            "reference_variety",
            "reference_field_mapping_coverage",
        ],
        "protocol_missing_rule": (
            "Return missing for fewer than two occupied reference fields or "
            "when any required strictly prior pair distance is unavailable."
        ),
        "denominator_zero_rule": (
            "Reference shares require at least one mapped reference and the "
            "source eligibility rule requires at least two occupied fields; "
            "otherwise return missing."
        ),
        "observed_empty_set_rule": (
            "An empty or one-field observed distribution returns missing "
            "under the source eligibility rule, not Rao-Stirling zero."
        ),
        "incomplete_coverage_rule": (
            "Use mapped references and y-5 through y-1 field distances; "
            "retain field mapping coverage and return missing if a needed "
            "distance is absent."
        ),
        "transform_and_unit_rule": (
            "Retain the untransformed ordered-pair-equivalent Rao-Stirling "
            "sum on [0,1]; higher values mean greater integrated diversity."
        ),
        "mask": "two_field_support",
    },
    "EF0117": {
        "column": "div_interdisciplinarity",
        "input_columns": [
            "div_interdisciplinarity",
            "reference_variety",
            "reference_balance",
            "reference_disparity",
            "reference_field_mapping_coverage",
        ],
        "protocol_missing_rule": (
            "Return missing for fewer than two occupied fields or missing "
            "relative-variety, balance, or prior-distance components."
        ),
        "denominator_zero_rule": (
            "The taxonomy size is frozen at 26 and must be positive; fewer "
            "than two occupied fields returns missing."
        ),
        "observed_empty_set_rule": (
            "An empty or one-field distribution returns missing rather than "
            "a structural zero."
        ),
        "incomplete_coverage_rule": (
            "Use mapped references, taxonomy size 26, and y-5 through y-1 "
            "field distances; retain mapping coverage."
        ),
        "transform_and_unit_rule": (
            "Retain the untransformed product of relative variety, "
            "one-minus-Gini, and mean distance."
        ),
        "mask": "two_field_support",
    },
    "EF0052": {
        "column": "backward_citation_age_mean",
        "input_columns": ["backward_citation_age_mean"],
        "protocol_missing_rule": (
            "Return missing when focal year is absent, no references are "
            "recorded, or no reference has a known nonfuture year."
        ),
        "denominator_zero_rule": (
            "The denominator is the count of references with known "
            "nonfuture years; zero returns missing."
        ),
        "observed_empty_set_rule": (
            "A fully observed empty reference set returns missing, not age "
            "zero; same-year references are valid age zero."
        ),
        "incomplete_coverage_rule": (
            "Exclude missing-year and apparent future references, compute "
            "over remaining references, and retain year coverage counts."
        ),
        "transform_and_unit_rule": (
            "Take the untransformed arithmetic mean of focal year minus "
            "reference year, measured in years."
        ),
        "mask": "none",
    },
    "EF0238": {
        "column": "bibliographic_coupling_degree_per_reference",
        "input_columns": [
            "bibliographic_coupling_degree_per_reference",
            "bibliographic_coupling_reference_coverage",
            "eligible_prior_paper_count",
        ],
        "protocol_missing_rule": (
            "Return missing for fewer than two valid focal references or "
            "when the strictly prior comparison universe is unavailable."
        ),
        "denominator_zero_rule": (
            "The denominator is the number of valid focal references; zero "
            "or one valid reference returns missing."
        ),
        "observed_empty_set_rule": (
            "An empty bibliography returns missing, not centrality zero."
        ),
        "incomplete_coverage_rule": (
            "Use only valid focal references and strictly prior eligible "
            "papers; retain reference coverage and prior-universe size."
        ),
        "transform_and_unit_rule": (
            "Retain unweighted prior-neighbor degree divided by valid focal "
            "reference count."
        ),
        "mask": "none",
    },
    "EF0038": {
        "column": "author_count",
        "input_columns": ["author_count"],
        "protocol_missing_rule": (
            "Treat a frozen author count of zero as missing authorship "
            "metadata; do not impute a zero-author paper."
        ),
        "denominator_zero_rule": "No denominator is used.",
        "observed_empty_set_rule": (
            "A zero-length author list is treated as missing metadata, not "
            "a valid team size of zero."
        ),
        "incomplete_coverage_rule": (
            "Use the distinct publication-time OpenAlex author count; "
            "consortium parsing is not inferred beyond the frozen record."
        ),
        "transform_and_unit_rule": (
            "Retain raw author count in authors; log1p is a separate optional "
            "model transform and is not the source definition."
        ),
        "mask": "positive_count",
    },
    "EF0419": {
        "column": "title_word_count",
        "input_columns": ["title_word_count"],
        "protocol_missing_rule": (
            "Return missing when the focal title is absent or contains no "
            "tokens."
        ),
        "denominator_zero_rule": "No denominator is used.",
        "observed_empty_set_rule": (
            "An empty title is missing rather than zero words."
        ),
        "incomplete_coverage_rule": (
            "Use the frozen whitespace-tokenized title; do not reconstruct "
            "a missing title from other metadata."
        ),
        "transform_and_unit_rule": (
            "Retain raw word count; higher values indicate a longer title."
        ),
        "mask": "positive_count",
    },
    "EF0314": {
        "column": "reference_count",
        "input_columns": ["reference_count"],
        "protocol_missing_rule": (
            "Return missing only when bibliography coverage is unknown; the "
            "frozen complete edge table permits an observed empty list."
        ),
        "denominator_zero_rule": "No denominator is used.",
        "observed_empty_set_rule": (
            "A fully observed empty bibliography is a valid raw count of "
            "zero."
        ),
        "incomplete_coverage_rule": (
            "Count unique focal paper-reference edges; duplicate edges are "
            "prohibited and missing coverage is not imputed."
        ),
        "transform_and_unit_rule": (
            "Retain raw reference count; log1p is a separate optional model "
            "transform and is not the source definition."
        ),
        "mask": "none",
    },
    "EF0188": {
        "column": "country_count",
        "input_columns": ["country_count"],
        "protocol_missing_rule": (
            "Treat zero recorded affiliation countries as missing country "
            "metadata, not an observed country count of zero."
        ),
        "denominator_zero_rule": "No denominator is used.",
        "observed_empty_set_rule": (
            "An empty country set is missing metadata rather than valid zero."
        ),
        "incomplete_coverage_rule": (
            "Use distinct publication-time affiliation countries and do not "
            "impute countries for affiliations lacking country codes."
        ),
        "transform_and_unit_rule": (
            "Retain raw distinct country count; higher values indicate "
            "broader international collaboration."
        ),
        "mask": "positive_count",
    },
    "EF0186": {
        "column": "international_collaboration",
        "input_columns": [
            "international_collaboration",
            "country_count",
        ],
        "protocol_missing_rule": (
            "Return missing when no affiliation country is recorded; do not "
            "classify unknown geography as domestic collaboration."
        ),
        "denominator_zero_rule": "No denominator is used.",
        "observed_empty_set_rule": (
            "An empty country set is missing; one known country is zero and "
            "more than one known country is one."
        ),
        "incomplete_coverage_rule": (
            "Use only frozen affiliation-country metadata; unknown country "
            "coverage is not imputed."
        ),
        "transform_and_unit_rule": (
            "Encode one when distinct known affiliation-country count is "
            "greater than one, otherwise zero; retain a binary unit."
        ),
        "mask": "known_country",
    },
    "EF0197": {
        "column": "journal_id",
        "input_columns": ["journal_id", "journal_name"],
        "protocol_missing_rule": (
            "Return missing when no stable source identifier is available."
        ),
        "denominator_zero_rule": "No denominator is used.",
        "observed_empty_set_rule": (
            "An empty source identifier is missing, not a journal category."
        ),
        "incomplete_coverage_rule": (
            "Use stable source_id and source_display_name; do not infer a "
            "journal from title text."
        ),
        "transform_and_unit_rule": (
            "Retain a nominal categorical source identifier with no ordinal "
            "direction."
        ),
        "mask": "nonempty_string",
    },
    "EF0307": {
        "column": "publication_year",
        "input_columns": ["publication_year"],
        "protocol_missing_rule": (
            "Return missing when focal publication year is unavailable."
        ),
        "denominator_zero_rule": "No denominator is used.",
        "observed_empty_set_rule": (
            "An absent year is missing and has no structural-zero meaning."
        ),
        "incomplete_coverage_rule": (
            "Use the frozen focal publication year without inferring it from "
            "issue labels or citation dates."
        ),
        "transform_and_unit_rule": (
            "Retain raw calendar year as a context control; no intrinsic "
            "quality direction is assigned."
        ),
        "mask": "none",
    },
}


def _one_minus_gini(counts: Sequence[int]) -> float:
    values = np.asarray(counts, dtype=float)
    if not len(values) or float(values.mean()) <= 0:
        return float("nan")
    gini = float(
        np.abs(values[:, None] - values[None, :]).sum()
        / (2.0 * len(values) ** 2 * float(values.mean()))
    )
    return float(1.0 - gini)


def _mean_pair_distance(
    labels: Sequence[str],
    distances: Mapping[tuple[str, str], float],
) -> float:
    occupied = sorted(set(labels))
    if len(occupied) < 2:
        return float("nan")
    values = []
    for left, right in itertools.combinations(occupied, 2):
        value = distances.get((left, right))
        if value is None or not math.isfinite(value):
            return float("nan")
        values.append(float(value))
    return float(np.mean(values))


def _rao_stirling(
    labels: Sequence[str],
    distances: Mapping[tuple[str, str], float],
) -> float:
    counts = Counter(labels)
    if len(counts) < 2:
        return float("nan")
    total = sum(counts.values())
    result = 0.0
    for left, right in itertools.combinations(sorted(counts), 2):
        distance = distances.get((left, right))
        if distance is None or not math.isfinite(distance):
            return float("nan")
        result += (
            2.0
            * (counts[left] / total)
            * (counts[right] / total)
            * float(distance)
        )
    return float(result)


def _mean_backward_age(
    focal_year: int | None,
    reference_years: Sequence[int | None],
) -> float:
    if focal_year is None or not reference_years:
        return float("nan")
    ages = [
        focal_year - year
        for year in reference_years
        if year is not None and year <= focal_year
    ]
    return float(np.mean(ages)) if ages else float("nan")


def _bc_degree_per_reference(
    degree: int,
    reference_count: int,
    prior_universe_available: bool,
) -> float:
    if reference_count < 2 or not prior_universe_available:
        return float("nan")
    return float(degree / reference_count)


def _mask_values(
    feature_id: str,
    values: Sequence[Any],
    **support: Sequence[Any],
) -> pd.Series:
    spec = FEATURE_SPECS[feature_id]
    frame = pd.DataFrame({str(spec["column"]): list(values), **support})
    return _apply_mask(frame, spec)


def _normal_formula_example(feature_id: str) -> tuple[Any, Any]:
    distances = {("A", "B"): 0.2, ("A", "C"): 0.4, ("B", "C"): 0.6}
    examples: Dict[str, tuple[Any, Any]] = {
        "EF0318": (len({"A", "A", "B"}), 2),
        "EF0312": (_one_minus_gini([2, 1]), 5.0 / 6.0),
        "EF0315": (_mean_pair_distance(["A", "B", "C"], distances), 0.4),
        "EF0309": (
            _rao_stirling(["A", "A", "B"], {("A", "B"): 0.5}),
            2.0 / 9.0,
        ),
        "EF0052": (_mean_backward_age(2020, [2010, 2020]), 5.0),
        "EF0238": (_bc_degree_per_reference(4, 2, True), 2.0),
        "EF0038": (len({"A1", "A2", "A2"}), 2),
        "EF0314": (len({"R1", "R2", "R2"}), 2),
        "EF0188": (len({"CN", "US", "US"}), 2),
        "EF0186": (int(len({"CN", "US"}) > 1), 1),
        "EF0197": ("S123", "S123"),
        "EF0307": (2026, 2026),
    }
    return examples[feature_id]


def _boundary_assertions(feature_id: str) -> Dict[str, bool]:
    distances = {("A", "B"): 0.2, ("A", "C"): 0.4, ("B", "C"): 0.6}
    if feature_id == "EF0318":
        masked = _mask_values(
            feature_id,
            [2.0, 1.0, 0.0],
            reference_field_mapping_coverage=[1.0, 0.5, 0.0],
        )
        return {
            "missing_input": pd.isna(masked.iloc[2]),
            "denominator_zero": pd.isna(masked.iloc[2]),
            "observed_empty_set": pd.isna(masked.iloc[2]),
            "incomplete_coverage": masked.iloc[1] == 1.0,
            "transform_and_unit": masked.iloc[0] == 2.0,
        }
    if feature_id == "EF0312":
        masked = _mask_values(
            feature_id,
            [5.0 / 6.0, 1.0, np.nan],
            reference_variety=[2, 1, 0],
            reference_field_mapping_coverage=[1.0, 0.5, 0.0],
        )
        return {
            "missing_input": pd.isna(masked.iloc[2]),
            "denominator_zero": math.isnan(_one_minus_gini([])),
            "observed_empty_set": math.isnan(_one_minus_gini([])),
            "incomplete_coverage": masked.iloc[1] == 1.0,
            "transform_and_unit": math.isclose(masked.iloc[0], 5.0 / 6.0),
        }
    if feature_id in {"EF0315", "EF0309"}:
        formula = (
            _mean_pair_distance(["A", "B", "C"], distances)
            if feature_id == "EF0315"
            else _rao_stirling(["A", "A", "B"], distances)
        )
        masked = _mask_values(
            feature_id,
            [formula, 0.0, np.nan],
            reference_variety=[3, 1, 2],
            reference_field_mapping_coverage=[1.0, 1.0, 0.5],
        )
        missing_pair = (
            _mean_pair_distance(["A", "B"], {})
            if feature_id == "EF0315"
            else _rao_stirling(["A", "B"], {})
        )
        one_field = (
            _mean_pair_distance(["A"], distances)
            if feature_id == "EF0315"
            else _rao_stirling(["A"], distances)
        )
        return {
            "missing_input": pd.isna(masked.iloc[2]),
            "denominator_zero": math.isnan(one_field),
            "observed_empty_set": pd.isna(masked.iloc[1]),
            "incomplete_coverage": math.isnan(missing_pair),
            "transform_and_unit": 0.0 <= float(masked.iloc[0]) <= 1.0,
        }
    if feature_id == "EF0052":
        return {
            "missing_input": math.isnan(_mean_backward_age(None, [2010])),
            "denominator_zero": math.isnan(_mean_backward_age(2020, [2021])),
            "observed_empty_set": math.isnan(_mean_backward_age(2020, [])),
            "incomplete_coverage": _mean_backward_age(
                2020,
                [2010, None, 2021],
            )
            == 10.0,
            "transform_and_unit": _mean_backward_age(2020, [2020]) == 0.0,
        }
    if feature_id == "EF0238":
        return {
            "missing_input": math.isnan(
                _bc_degree_per_reference(0, 0, False)
            ),
            "denominator_zero": math.isnan(
                _bc_degree_per_reference(1, 1, True)
            ),
            "observed_empty_set": math.isnan(
                _bc_degree_per_reference(0, 0, True)
            ),
            "incomplete_coverage": math.isnan(
                _bc_degree_per_reference(0, 2, False)
            ),
            "transform_and_unit": _bc_degree_per_reference(4, 2, True)
            == 2.0,
        }
    if feature_id in {"EF0038", "EF0188"}:
        masked = _mask_values(feature_id, [2.0, 0.0, np.nan])
        return {
            "missing_input": pd.isna(masked.iloc[2]),
            "denominator_zero": True,
            "observed_empty_set": pd.isna(masked.iloc[1]),
            "incomplete_coverage": pd.isna(masked.iloc[2]),
            "transform_and_unit": masked.iloc[0] == 2.0,
        }
    if feature_id == "EF0314":
        masked = _mask_values(feature_id, [2.0, 0.0, np.nan])
        return {
            "missing_input": pd.isna(masked.iloc[2]),
            "denominator_zero": True,
            "observed_empty_set": masked.iloc[1] == 0.0,
            "incomplete_coverage": pd.isna(masked.iloc[2]),
            "transform_and_unit": masked.iloc[0] == 2.0,
        }
    if feature_id == "EF0186":
        masked = _mask_values(
            feature_id,
            [1.0, 0.0, np.nan],
            country_count=[2.0, 1.0, np.nan],
        )
        return {
            "missing_input": pd.isna(masked.iloc[2]),
            "denominator_zero": True,
            "observed_empty_set": pd.isna(masked.iloc[2]),
            "incomplete_coverage": pd.isna(masked.iloc[2]),
            "transform_and_unit": list(masked.iloc[:2]) == [1.0, 0.0],
        }
    if feature_id == "EF0197":
        masked = _mask_values(
            feature_id,
            ["S123", "", None],
            journal_name=["Journal", "", None],
        )
        return {
            "missing_input": pd.isna(masked.iloc[2]),
            "denominator_zero": True,
            "observed_empty_set": pd.isna(masked.iloc[1]),
            "incomplete_coverage": pd.isna(masked.iloc[2]),
            "transform_and_unit": masked.iloc[0] == "S123",
        }
    if feature_id == "EF0307":
        masked = _mask_values(feature_id, [2026.0, np.nan])
        return {
            "missing_input": pd.isna(masked.iloc[1]),
            "denominator_zero": True,
            "observed_empty_set": pd.isna(masked.iloc[1]),
            "incomplete_coverage": pd.isna(masked.iloc[1]),
            "transform_and_unit": masked.iloc[0] == 2026.0,
        }
    raise KeyError(feature_id)


def _formula_test(feature_id: str) -> Dict[str, Any]:
    observed, expected = _normal_formula_example(feature_id)
    if isinstance(expected, float):
        example_passed = math.isclose(
            float(observed),
            expected,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    else:
        example_passed = observed == expected
    assertions = {
        key: bool(value)
        for key, value in _boundary_assertions(feature_id).items()
    }
    return {
        "formula_example": {
            "observed": (
                float(observed)
                if isinstance(observed, (float, np.floating))
                else observed
            ),
            "expected": expected,
            "passed": example_passed,
        },
        "boundary_assertions": assertions,
        "covered_protocol_fields": [
            "protocol_missing_rule",
            "denominator_zero_rule",
            "observed_empty_set_rule",
            "incomplete_coverage_rule",
            "transform_and_unit_rule",
        ],
        "passed": example_passed and all(assertions.values()),
    }


def _eligible_targeted_features(
    connection: sqlite3.Connection,
) -> Dict[str, sqlite3.Row]:
    approved = {
        str(row["feature_id"])
        for row in connection.execute(
            """
            SELECT DISTINCT feature_id
            FROM targeted_formula_decisions
            WHERE final_decision = 'approve_formula'
            """
        )
    }
    output: Dict[str, sqlite3.Row] = {}
    for feature_id in sorted(approved & set(FEATURE_SPECS)):
        family = connection.execute(
            "SELECT * FROM indicator_families WHERE feature_id = ?",
            (feature_id,),
        ).fetchone()
        if family is None:
            continue
        if indicators._eligible_formula_mentions(connection, feature_id):
            output[feature_id] = family
    return output


def _frozen_h2_review_provenance(
    connection: sqlite3.Connection,
) -> Dict[str, Any]:
    rows = connection.execute(
        """
        SELECT DISTINCT d.h2_artifact_sha256, r.completed_at
        FROM targeted_formula_decisions d
        JOIN independent_ai_review_runs r
          ON r.artifact_sha256 = d.h2_artifact_sha256
         AND r.reviewer_role = 'H2'
         AND r.status = 'complete'
        ORDER BY d.h2_artifact_sha256
        """
    ).fetchall()
    if len(rows) != 1 or not str(rows[0]["completed_at"] or "").strip():
        raise RuntimeError(
            "Targeted materialization requires one registered frozen H2 "
            "formula-review artifact"
        )
    return {
        "artifact_sha256": str(rows[0]["h2_artifact_sha256"]),
        "completed_at": str(rows[0]["completed_at"]),
    }


def _apply_mask(
    matrix: pd.DataFrame,
    spec: Mapping[str, Any],
) -> pd.Series:
    column = str(spec["column"])
    values = matrix[column].copy()
    mask = str(spec["mask"])
    if mask == "overlap_support":
        support = (
            matrix["reference_overlap_comparison_count"].gt(0)
            & matrix["reference_overlap_reference_count"].gt(0)
        )
        return values.where(support)
    if mask == "mapped_reference_support":
        return values.where(
            matrix["reference_field_mapping_coverage"].gt(0)
        )
    if mask == "two_field_support":
        return values.where(matrix["reference_variety"].ge(2))
    if mask == "positive_count":
        return values.where(pd.to_numeric(values, errors="coerce").gt(0))
    if mask == "known_country":
        return values.where(matrix["country_count"].notna())
    if mask == "nonempty_string":
        text = values.astype("string")
        return values.where(text.notna() & text.str.strip().ne(""))
    if mask == "none":
        return values
    raise ValueError(f"Unknown operationalization mask: {mask}")


def _series_summary(series: pd.Series) -> Dict[str, Any]:
    valid = series.notna()
    summary = {
        "row_count": int(len(series)),
        "valid_count": int(valid.sum()),
        "missing_count": int((~valid).sum()),
        "missing_rate": float((~valid).mean()),
        "unique_count": int(series[valid].nunique(dropna=True)),
    }
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_valid = numeric.notna()
    if int(numeric_valid.sum()) == int(valid.sum()) and bool(valid.any()):
        summary.update(
            {
                "finite_count": int(np.isfinite(numeric[valid]).sum()),
                "minimum": float(numeric[valid].min()),
                "maximum": float(numeric[valid].max()),
            }
        )
    return summary


def _write_feature_artifacts(
    output_dir: Path,
    feature_id: str,
    family: sqlite3.Row,
    spec: Mapping[str, Any],
    summary: Mapping[str, Any],
    candidate_matrix_path: Path,
    candidate_report_path: Path,
) -> Dict[str, Any]:
    artifact_dir = output_dir / "feature_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    test_path = (artifact_dir / f"{feature_id}_test.json").resolve()
    formula_test = _formula_test(feature_id)
    if not formula_test["passed"]:
        raise AssertionError(f"{feature_id} formula self-test failed")
    write_json(
        test_path,
        {
            "schema_version": "targeted_feature_test_v3",
            "feature_id": feature_id,
            "definition_version": DEFINITION_VERSION,
            **formula_test,
            "nonconstant": int(summary["unique_count"]) > 1,
            "finite_when_numeric": (
                summary.get("finite_count") == summary["valid_count"]
                if "finite_count" in summary
                else True
            ),
            "materialization_passed": (
                int(summary["unique_count"]) > 1
                and (
                    summary.get("finite_count") == summary["valid_count"]
                    if "finite_count" in summary
                    else True
                )
            ),
            "model_outcomes_used": False,
        },
    )
    test_payload = json.loads(test_path.read_text(encoding="utf-8"))
    test_payload["passed"] = bool(
        formula_test["passed"]
        and test_payload["materialization_passed"]
    )
    write_json(test_path, test_payload)
    snapshot_path = (
        artifact_dir / f"{feature_id}_input_snapshot.json"
    ).resolve()
    upstream_path, upstream_function = UPSTREAM_IMPLEMENTATIONS[feature_id]
    if not upstream_path.is_file():
        raise FileNotFoundError(upstream_path)
    write_json(
        snapshot_path,
        {
            "schema_version": "targeted_feature_input_snapshot_v3",
            "feature_id": feature_id,
            "canonical_name_en": family["canonical_name_en"],
            "definition_version": DEFINITION_VERSION,
            "candidate_matrix_path": str(candidate_matrix_path),
            "candidate_matrix_sha256": sha256_file(candidate_matrix_path),
            "candidate_report_path": str(candidate_report_path),
            "candidate_report_sha256": sha256_file(candidate_report_path),
            "input_columns": list(spec["input_columns"]),
            "upstream_implementation_path": str(upstream_path.resolve()),
            "upstream_implementation_sha256": sha256_file(upstream_path),
            "upstream_function": upstream_function,
            "target_wrapper_path": str(IMPLEMENTATION_PATH),
            "target_wrapper_sha256": sha256_file(IMPLEMENTATION_PATH),
            "model_outcomes_used": False,
            "future_information_used": False,
        },
    )
    return {
        "test_artifact_path": str(test_path),
        "test_artifact_sha256": sha256_file(test_path),
        "input_snapshot_path": str(snapshot_path),
        "input_snapshot_sha256": sha256_file(snapshot_path),
    }


def materialize(
    connection: sqlite3.Connection,
    candidate_matrix_path: Path,
    candidate_report_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    if not candidate_matrix_path.is_file():
        raise FileNotFoundError(candidate_matrix_path)
    if not candidate_report_path.is_file():
        raise FileNotFoundError(candidate_report_path)
    families = _eligible_targeted_features(connection)
    if not families:
        raise RuntimeError("No H2-approved targeted formula is eligible")
    h2_provenance = _frozen_h2_review_provenance(connection)
    matrix = pd.read_parquet(candidate_matrix_path)
    if matrix["paper_id"].isna().any() or matrix["paper_id"].duplicated().any():
        raise ValueError("Candidate matrix paper_id is not unique and complete")
    output = matrix[["paper_id"]].copy()
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_features: Dict[str, Any] = {}
    mapping_features: Dict[str, Any] = {}
    for feature_id, family in sorted(families.items()):
        spec = FEATURE_SPECS[feature_id]
        formula_mentions = indicators._eligible_formula_mentions(
            connection,
            feature_id,
        )
        if not formula_mentions:
            raise RuntimeError(
                f"{feature_id} lacks an authorizing formula mention"
            )
        formula_mention = formula_mentions[0]
        target_ids = [
            str(row["target_id"])
            for row in connection.execute(
                """
                SELECT target_id FROM targeted_formula_decisions
                WHERE feature_id = ? AND final_decision = 'approve_formula'
                ORDER BY target_id
                """,
                (feature_id,),
            )
        ]
        missing = sorted(set(spec["input_columns"]) - set(matrix.columns))
        if missing:
            raise ValueError(f"{feature_id} lacks columns: {missing}")
        series = _apply_mask(matrix, spec)
        output_column = f"{feature_id}__{spec['column']}"
        output[output_column] = series
        summary = _series_summary(series)
        if summary["unique_count"] <= 1:
            raise ValueError(f"{feature_id} is constant or entirely missing")
        artifacts = _write_feature_artifacts(
            output_dir,
            feature_id,
            family,
            spec,
            summary,
            candidate_matrix_path,
            candidate_report_path,
        )
        registry_features[feature_id] = {
            "canonical_name_en": family["canonical_name_en"],
            "authorizing_formula_mention_id": formula_mention["mention_id"],
            "authorizing_fulltext_sha256": formula_mention["fulltext_sha256"],
            "authorizing_formula_location": formula_mention["formula_location"],
            "targeted_formula_target_ids": target_ids,
            "h2_formula_review_artifact_sha256": h2_provenance[
                "artifact_sha256"
            ],
            "source_column": spec["column"],
            "output_column": output_column,
            "summary": summary,
            **{
                field: spec[field]
                for field in (
                    "protocol_missing_rule",
                    "denominator_zero_rule",
                    "observed_empty_set_rule",
                    "incomplete_coverage_rule",
                    "transform_and_unit_rule",
                )
            },
            "input_columns_json": json.dumps(
                [
                    f"{candidate_matrix_path.name}:{column}"
                    for column in spec["input_columns"]
                ],
                ensure_ascii=False,
            ),
            "implementation_path": str(IMPLEMENTATION_PATH),
            "implementation_sha256": sha256_file(IMPLEMENTATION_PATH),
            **artifacts,
        }
        mapping_features[feature_id] = {
            "source": "targeted_operationalized_matrix",
            "column": output_column,
            "notes": (
                "H2-approved source formula plus independently reviewed "
                "project missingness and boundary rules; no model outcomes."
            ),
        }
    matrix_path = (output_dir / DEFAULT_MATRIX_NAME).resolve()
    output.to_parquet(matrix_path, index=False)
    matrix_hash = sha256_file(matrix_path)
    registry_path = (output_dir / DEFAULT_REGISTRY_NAME).resolve()
    registry = {
        "schema_version": "targeted_operationalization_registry_v3",
        "created_at": h2_provenance["completed_at"],
        "created_at_basis": "registered_frozen_h2_review_completed_at",
        "definition_version": DEFINITION_VERSION,
        "matrix_path": str(matrix_path),
        "matrix_sha256": matrix_hash,
        "implementation_path": str(IMPLEMENTATION_PATH),
        "implementation_sha256": sha256_file(IMPLEMENTATION_PATH),
        "candidate_matrix_path": str(candidate_matrix_path),
        "candidate_matrix_sha256": sha256_file(candidate_matrix_path),
        "candidate_report_path": str(candidate_report_path),
        "candidate_report_sha256": sha256_file(candidate_report_path),
        "h2_formula_review_artifact_sha256": h2_provenance[
            "artifact_sha256"
        ],
        "feature_count": len(registry_features),
        "features": registry_features,
        "target_count_is_not_a_selection_quota": True,
        "model_outcomes_used": False,
        "future_information_used": False,
        "round_13": False,
    }
    write_json(registry_path, registry)
    mapping_path = (output_dir / DEFAULT_MAPPING_NAME).resolve()
    write_json(
        mapping_path,
        {
            "schema_version": "targeted_feature_data_mapping_v3",
            "sources": {
                "targeted_operationalized_matrix": str(matrix_path)
            },
            "features": mapping_features,
            "source_matrix_sha256": matrix_hash,
            "registry_path": str(registry_path),
            "registry_sha256": sha256_file(registry_path),
            "model_outcomes_used": False,
        },
    )
    return {
        "matrix_path": str(matrix_path),
        "matrix_sha256": matrix_hash,
        "registry_path": str(registry_path),
        "registry_sha256": sha256_file(registry_path),
        "mapping_path": str(mapping_path),
        "mapping_sha256": sha256_file(mapping_path),
        "feature_count": len(registry_features),
        "feature_ids": sorted(registry_features),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument(
        "--candidate-matrix",
        type=Path,
        default=DEFAULT_CANDIDATE_MATRIX,
    )
    parser.add_argument(
        "--candidate-report",
        type=Path,
        default=DEFAULT_CANDIDATE_REPORT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    args = parser.parse_args()
    connection = initialize(args.database.resolve())
    try:
        result = materialize(
            connection,
            args.candidate_matrix.resolve(),
            args.candidate_report.resolve(),
            args.output_dir.resolve(),
        )
    finally:
        connection.close()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

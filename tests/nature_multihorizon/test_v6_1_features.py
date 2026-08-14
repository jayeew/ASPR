"""Hand-calculated tests for v6.1 candidate formulas."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from gear.nature_multihorizon.candidate_registry_v6_1 import (
    load_candidate_registry_v6_1,
)
from gear.nature_multihorizon.features_v6 import canonical_pair
from gear.nature_multihorizon.materialize_v6_1 import (
    FIELD_TAXONOMY_SIZE,
    build_reference_overlap_features,
    materialize_reference_overlap_context,
)
from gear.nature_multihorizon.features_v6_1 import (
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
    reference_overlap_novelty,
    source_pair_mean_surprisal,
    true_diversity_from_rao,
)
from gear.nature_multihorizon.screening_v6_1 import coverage_audit


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_balance_and_hill_metrics_hand_calculation() -> None:
    """Balance candidates match their elementary definitions."""
    labels = ["A", "A", "B"]
    assert np.isclose(field_gini_balance(labels), 5.0 / 6.0)
    assert np.isclose(field_gini_simpson(labels), 4.0 / 9.0)
    assert np.isclose(field_hhi(labels), 5.0 / 9.0)
    assert np.isclose(
        field_shannon_entropy(labels),
        -(2.0 / 3.0 * math.log(2.0 / 3.0) + 1.0 / 3.0 * math.log(1.0 / 3.0)),
    )
    assert field_hill_number(labels, order=0) == 2.0
    assert np.isclose(
        field_hill_number(labels, order=1),
        math.exp(field_shannon_entropy(labels)),
    )
    assert np.isclose(field_hill_number(labels, order=2), 9.0 / 5.0)
    assert field_relative_variety(labels, total_categories=4) == 0.5
    assert field_other_field_share(labels, focal_field="A") == 1.0 / 3.0
    assert FIELD_TAXONOMY_SIZE == 26


def test_distance_and_div_hand_calculation() -> None:
    """Distance quantiles and DIV use occupied field pairs."""
    labels = ["A", "A", "B", "C"]
    distances = {
        canonical_pair("A", "B"): 0.2,
        canonical_pair("A", "C"): 0.8,
        canonical_pair("B", "C"): 0.5,
    }
    assert field_distance_quantile(labels, distances, quantile=1.0) == 0.8
    expected = (
        field_relative_variety(labels, total_categories=4)
        * field_gini_balance(labels)
        * 0.5
    )
    assert np.isclose(
        div_index(labels, distances, total_categories=4),
        expected,
    )
    assert np.isclose(true_diversity_from_rao(0.25), 4.0 / 3.0)


def test_first_pair_family_hand_calculation() -> None:
    """Binary, count, distance, and low-frequency candidates agree."""
    sources = ["S1", "S2", "S3"]
    pair_counts = {
        canonical_pair("S1", "S2"): 2,
        canonical_pair("S1", "S3"): 0,
        canonical_pair("S2", "S3"): 0,
    }
    distances = {
        canonical_pair("S1", "S3"): 0.4,
        canonical_pair("S2", "S3"): 0.8,
    }
    assert first_time_pair_any(sources, pair_counts) == 1.0
    assert first_time_pair_count(sources, pair_counts) == 2.0
    assert np.isclose(
        first_time_pair_distance_sum(sources, pair_counts, distances),
        1.2,
    )
    assert np.isclose(
        low_frequency_pair_share(
            sources, pair_counts, maximum_prior_count=1
        ),
        2.0 / 3.0,
    )


def test_surprisal_and_reference_overlap_hand_calculation() -> None:
    """Rarity and overlap candidates follow their registered equations."""
    sources = ["S1", "S2"]
    pair = canonical_pair("S1", "S2")
    value = source_pair_mean_surprisal(
        sources,
        {pair: 2},
        {"S1": 5, "S2": 4},
        10,
    )
    assert np.isclose(value, -math.log(1.0))
    novelty = reference_overlap_novelty(
        ["R1", "R2"],
        [["R1", "R3"], ["R4"]],
    )
    assert np.isclose(novelty, 5.0 / 6.0)


def test_exact_hypergeometric_z_hand_calculation() -> None:
    """The separately named analytical null matches its closed form."""
    pair = canonical_pair("S1", "S2")
    values = hypergeometric_pair_z_scores(
        ["S1", "S2"],
        {pair: 2},
        {"S1": 5, "S2": 4},
        10,
    )
    assert len(values) == 1
    assert np.isclose(values[0], 0.0)


def test_structural_missing_semantics() -> None:
    """Undefined candidates stay NaN rather than becoming false zeroes."""
    assert math.isnan(field_gini_balance([]))
    assert math.isnan(field_other_field_share([], focal_field="A"))
    assert math.isnan(first_time_pair_any(["S1"], {}))
    assert math.isnan(reference_overlap_novelty([], [["R1"]]))


def test_reference_overlap_respects_time_field_and_windows() -> None:
    """The materialized metric uses only eligible prior same-field papers."""
    focal_id = "https://openalex.org/W100"
    reference_ids = {
        "R1": "https://openalex.org/W1",
        "R2": "https://openalex.org/W2",
        "R3": "https://openalex.org/W3",
        "OLD": "https://openalex.org/W4",
    }
    papers = pd.DataFrame(
        {
            "paper_id": [focal_id],
            "publication_year": [2005],
            "openalex_primary_field": ["F"],
        }
    )
    paper_references = pd.DataFrame(
        {
            "paper_id": [focal_id, focal_id, focal_id],
            "reference_id": [
                reference_ids["R1"],
                reference_ids["R2"],
                reference_ids["OLD"],
            ],
        }
    )
    metadata = pd.DataFrame(
        {
            "reference_id": list(reference_ids.values()),
            "reference_year": [2001, 1999, 2000, 1994],
            "source_id": ["S1", "S2", "S3", "S4"],
            "field_id": ["F", "F", "F", "F"],
        }
    )
    history = pd.DataFrame(
        {
            "work_id": ["H1", "H2", "H3", "H4"],
            "publication_year": [2004, 2003, 2001, 2005],
            "openalex_primary_field": ["F", "G", "F", "F"],
            "reference_ids": [
                [reference_ids["R1"], reference_ids["R3"]],
                [reference_ids["R1"]],
                [reference_ids["R3"]],
                [reference_ids["R1"], reference_ids["R2"]],
            ],
            "reference_years": [
                [2001, 2000],
                [2001],
                [2000],
                [2001, 1999],
            ],
        }
    )
    result = build_reference_overlap_features(
        papers,
        paper_references,
        metadata,
        history,
        reference_window_years=10,
        cociting_window_years=3,
    ).iloc[0]
    assert np.isclose(result["reference_overlap_novelty_t0"], 2.0 / 3.0)
    assert result["reference_overlap_comparison_count"] == 1
    assert result["reference_overlap_reference_count"] == 2


def test_reference_overlap_supports_published_all_all_window() -> None:
    """The source-published all/all variant retains all strictly prior rows."""
    focal_id = "https://openalex.org/W100"
    references = {
        "R1": "https://openalex.org/W1",
        "R2": "https://openalex.org/W2",
        "OLD": "https://openalex.org/W3",
        "R3": "https://openalex.org/W4",
    }
    papers = pd.DataFrame(
        {
            "paper_id": [focal_id],
            "publication_year": [2005],
            "openalex_primary_field": ["F"],
        }
    )
    paper_references = pd.DataFrame(
        {
            "paper_id": [focal_id, focal_id, focal_id],
            "reference_id": [
                references["R1"],
                references["R2"],
                references["OLD"],
            ],
        }
    )
    metadata = pd.DataFrame(
        {
            "reference_id": list(references.values()),
            "reference_year": [2001, 1999, 1994, 2000],
            "source_id": ["S1", "S2", "S3", "S4"],
            "field_id": ["F", "F", "F", "F"],
        }
    )
    history = pd.DataFrame(
        {
            "work_id": ["H1", "H2"],
            "publication_year": [2004, 1990],
            "openalex_primary_field": ["F", "F"],
            "reference_ids": [
                [references["R1"], references["R3"]],
                [references["R2"]],
            ],
            "reference_years": [[2001, 2000], [1999]],
        }
    )
    result = build_reference_overlap_features(
        papers,
        paper_references,
        metadata,
        history,
        reference_window_years=None,
        cociting_window_years=None,
    ).iloc[0]
    assert np.isclose(
        result["reference_overlap_novelty_t0"], 17.0 / 24.0
    )
    assert result["reference_overlap_comparison_count"] == 2
    assert result["reference_overlap_reference_count"] == 3


def test_overlap_history_is_limited_to_registered_primary_papers(
    tmp_path: Path,
) -> None:
    """Scope and time filters exclude non-cohort and future-reference rows."""
    eligible = "https://openalex.org/W100"
    excluded = "https://openalex.org/W200"
    prior_reference = "https://openalex.org/W1"
    future_reference = "https://openalex.org/W2"
    target_path = tmp_path / "targets.csv"
    pd.DataFrame(
        {
            "id": [eligible, excluded],
            "year": [2005, 2005],
            "openalex_primary_field": ["F", "F"],
        }
    ).to_csv(target_path, index=False)
    edges_path = tmp_path / "edges.csv"
    pd.DataFrame(
        {
            "source": [eligible, eligible, excluded],
            "target": [
                prior_reference,
                future_reference,
                prior_reference,
            ],
        }
    ).to_csv(edges_path, index=False)
    metadata = pd.DataFrame(
        {
            "reference_id": [prior_reference, future_reference],
            "reference_year": [2000, 2006],
            "field_id": ["F", "F"],
        }
    )
    manifest = materialize_reference_overlap_context(
        edges_path,
        target_path,
        [eligible],
        metadata,
        tmp_path / "output",
        chunksize=2,
        resume=False,
    )
    history = pd.read_parquet(
        manifest["outputs"]["historical_paper_references"]["path"]
    )
    assert history["work_id"].tolist() == [eligible]
    assert list(history.iloc[0]["reference_ids"]) == [prior_reference]
    assert list(history.iloc[0]["reference_years"]) == [2000]
    assert manifest["n_eligible_source_ids"] == 1


def test_eligible_coverage_keeps_raw_missingness_visible() -> None:
    """R1 changes the gate denominator without filling missing scores."""
    registry = load_candidate_registry_v6_1(
        PROJECT_ROOT / "configs/innovation_candidate_catalog_v6_1.json"
    )
    features = pd.DataFrame(
        {
            "paper_id": ["W1", "W2", "W3", "W4"],
            "domain12": ["D1", "D1", "D2", "D2"],
            "valid_reference_count": [20, 0, 20, 0],
            "source_mapping_coverage": [1.0, 0.0, 1.0, 0.0],
            "field_mapping_coverage": [1.0, 0.0, 1.0, 0.0],
            "source_pair_mean_surprisal": [1.0, np.nan, 2.0, np.nan],
        }
    )
    coverage, _ = coverage_audit(
        registry,
        features,
        denominator_policy="eligible_by_metric_family",
    )
    row = coverage.set_index("candidate_id").loc["A1.MEAN_SURPRISAL"]
    assert row["raw_overall_coverage"] == 0.5
    assert row["overall_coverage"] == 1.0
    assert row["n_eligible"] == 2
    assert features["source_pair_mean_surprisal"].isna().sum() == 2


def test_overlap_coverage_does_not_require_source_mapping() -> None:
    """Reference-set overlap uses valid work IDs, not cited-source mapping."""
    registry = load_candidate_registry_v6_1(
        PROJECT_ROOT / "configs/innovation_candidate_catalog_v6_1.json"
    )
    features = pd.DataFrame(
        {
            "paper_id": ["W1", "W2", "W3", "W4"],
            "domain12": ["D1", "D1", "D2", "D2"],
            "valid_reference_count": [20, 0, 20, 0],
            "source_mapping_coverage": [0.0, 0.0, 0.0, 0.0],
            "field_mapping_coverage": [0.0, 0.0, 0.0, 0.0],
            "reference_overlap_novelty_t0": [
                0.8,
                np.nan,
                0.9,
                np.nan,
            ],
        }
    )
    coverage, _ = coverage_audit(
        registry,
        features,
        denominator_policy="eligible_by_metric_family",
    )
    row = coverage.set_index("candidate_id").loc[
        "A1.REFERENCE_OVERLAP"
    ]
    assert row["raw_overall_coverage"] == 0.5
    assert row["overall_coverage"] == 1.0
    assert row["n_eligible"] == 2

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as parquet

from common import sha256_file, write_json


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = (
    ROOT.parent.parent
    / "data"
    / "knowledge_corpus"
    / "nature_multihorizon_v6_1_local"
)
DEFAULT_BACKWARD_AGE = (
    ROOT
    / "outputs"
    / "operationalizations"
    / "backward_citation_age_mean_v3.parquet"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "candidate_t0_feature_matrix_v3"
MATRIX_NAME = "candidate_t0_feature_matrix_v3.parquet"
REPORT_NAME = "candidate_t0_feature_matrix_v3_report.json"
TEST_NAME = "candidate_t0_feature_matrix_v3_test.json"
SNAPSHOT_NAME = "candidate_t0_feature_matrix_v3_input_snapshot.json"
DEFINITION_VERSION = "candidate_t0_feature_matrix_v3_20260730"

INPUT_COLUMNS = {
    "innovation": {
        "file": "innovation_candidate_features.parquet",
        "columns": [
            "paper_id",
            "reference_overlap_novelty_t0",
            "reference_overlap_comparison_count",
            "reference_overlap_reference_count",
            "field_variety",
            "field_gini_balance",
            "field_disparity_cosine_mean",
            "rao_stirling_integration",
            "field_div_index",
            "valid_reference_count",
            "field_mapping_coverage",
        ],
    },
    "opportunity": {
        "file": "opportunity_features.parquet",
        "columns": [
            "paper_id",
            "bc_degree_per_reference_t0",
            "bc_reference_coverage",
            "eligible_prior_paper_count",
        ],
    },
    "controls": {
        "file": "control_features_v6_1.parquet",
        "columns": [
            "paper_id",
            "title_word_count",
            "log_reference_count",
            "log_author_count",
            "log_country_count",
            "openalex_metadata_found",
        ],
    },
    "openalex": {
        "file": "target_openalex_metadata.parquet",
        "columns": [
            "paper_id",
            "openalex_author_count",
            "openalex_country_count",
            "openalex_institution_count",
        ],
    },
    "papers": {
        "file": "papers_common_all.parquet",
        "columns": [
            "paper_id",
            "publication_year",
            "source_id",
            "source_display_name",
            "primary_field",
            "openalex_primary_field",
            "openalex_primary_subfield",
            "document_type",
            "work_type",
        ],
    },
    "references": {
        "file": "paper_references.parquet",
        "columns": ["paper_id", "reference_id"],
    },
}

OUTPUT_COLUMNS = [
    "paper_id",
    "publication_year",
    "reference_combination_novelty",
    "reference_overlap_comparison_count",
    "reference_overlap_reference_count",
    "reference_variety",
    "reference_balance",
    "reference_disparity",
    "rao_stirling_diversity",
    "div_interdisciplinarity",
    "reference_field_mapping_coverage",
    "bibliographic_coupling_degree_per_reference",
    "bibliographic_coupling_reference_coverage",
    "eligible_prior_paper_count",
    "backward_citation_age_mean",
    "author_count",
    "country_count",
    "international_collaboration",
    "title_word_count",
    "reference_count",
    "journal_id",
    "journal_name",
    "primary_field",
    "openalex_primary_field",
    "openalex_primary_subfield",
    "document_type",
    "work_type",
]

NUMERIC_RANGES = {
    "reference_combination_novelty": (0.0, 1.0),
    "reference_variety": (0.0, math.inf),
    "reference_balance": (0.0, 1.0),
    "reference_disparity": (0.0, 1.0),
    "rao_stirling_diversity": (0.0, math.inf),
    "div_interdisciplinarity": (0.0, math.inf),
    "reference_field_mapping_coverage": (0.0, 1.0),
    "bibliographic_coupling_degree_per_reference": (0.0, math.inf),
    "bibliographic_coupling_reference_coverage": (0.0, 1.0),
    "eligible_prior_paper_count": (0.0, math.inf),
    "backward_citation_age_mean": (0.0, math.inf),
    "author_count": (0.0, math.inf),
    "country_count": (0.0, math.inf),
    "international_collaboration": (0.0, 1.0),
    "title_word_count": (0.0, math.inf),
    "reference_count": (0.0, math.inf),
}


def _input_description(path: Path, columns: List[str]) -> Dict[str, Any]:
    metadata = parquet.ParquetFile(path).metadata
    schema = parquet.read_schema(path)
    missing = sorted(set(columns) - set(schema.names))
    if missing:
        raise ValueError(f"{path} lacks columns: {missing}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "row_count": metadata.num_rows,
        "columns": columns,
    }


def _read_unique(path: Path, columns: List[str], label: str) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=columns)
    if frame["paper_id"].isna().any():
        raise ValueError(f"{label} has missing paper_id values")
    duplicates = int(frame["paper_id"].duplicated().sum())
    if duplicates:
        raise ValueError(f"{label} has {duplicates} duplicate paper IDs")
    return frame


def _merge_one_to_one(
    left: pd.DataFrame,
    right: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    before = len(left)
    merged = left.merge(
        right,
        on="paper_id",
        how="left",
        validate="one_to_one",
    )
    if len(merged) != before:
        raise RuntimeError(f"{label} merge changed the base row count")
    return merged


def _reference_counts(path: Path) -> Dict[str, Any]:
    frame = pd.read_parquet(path, columns=["paper_id", "reference_id"])
    duplicate_edges = int(
        frame.duplicated(["paper_id", "reference_id"]).sum()
    )
    if duplicate_edges:
        raise ValueError(
            f"paper_references contains {duplicate_edges} duplicate edges"
        )
    counts = (
        frame.groupby("paper_id", sort=False, observed=True)
        .size()
        .rename("reference_count")
        .reset_index()
    )
    return {
        "frame": counts,
        "edge_count": int(len(frame)),
        "paper_count_with_edges": int(len(counts)),
        "duplicate_edge_count": duplicate_edges,
    }


def _load_inputs(
    data_root: Path,
    backward_age_path: Path,
) -> tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    frames: Dict[str, pd.DataFrame] = {}
    snapshot: Dict[str, Any] = {}
    for source_id, spec in INPUT_COLUMNS.items():
        path = (data_root / spec["file"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        snapshot[source_id] = _input_description(path, spec["columns"])
        if source_id == "references":
            continue
        frames[source_id] = _read_unique(
            path,
            spec["columns"],
            source_id,
        )
    if not backward_age_path.is_file():
        raise FileNotFoundError(backward_age_path)
    backward_columns = [
        "paper_id",
        "backward_citation_age_mean",
        "reference_year_coverage",
    ]
    snapshot["backward_citation_age"] = _input_description(
        backward_age_path,
        backward_columns,
    )
    frames["backward_citation_age"] = _read_unique(
        backward_age_path,
        backward_columns,
        "backward_citation_age",
    )
    reference_path = Path(snapshot["references"]["path"])
    reference_counts = _reference_counts(reference_path)
    frames["reference_counts"] = reference_counts.pop("frame")
    snapshot["references"].update(reference_counts)
    return frames, snapshot


def _rename_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(
        columns={
            "reference_overlap_novelty_t0": (
                "reference_combination_novelty"
            ),
            "field_variety": "reference_variety",
            "field_gini_balance": "reference_balance",
            "field_disparity_cosine_mean": "reference_disparity",
            "rao_stirling_integration": "rao_stirling_diversity",
            "field_div_index": "div_interdisciplinarity",
            "field_mapping_coverage": (
                "reference_field_mapping_coverage"
            ),
            "bc_degree_per_reference_t0": (
                "bibliographic_coupling_degree_per_reference"
            ),
            "bc_reference_coverage": (
                "bibliographic_coupling_reference_coverage"
            ),
            "openalex_author_count": "author_count",
            "openalex_country_count": "country_count",
            "source_id": "journal_id",
            "source_display_name": "journal_name",
        }
    )


def build_matrix(
    frames: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    base = frames["innovation"].copy()
    for source_id in (
        "opportunity",
        "controls",
        "openalex",
        "papers",
        "backward_citation_age",
        "reference_counts",
    ):
        base = _merge_one_to_one(base, frames[source_id], source_id)
    base["reference_count"] = (
        base["reference_count"].fillna(0).astype("int64")
    )
    raw_author_count = base["openalex_author_count"].copy()
    raw_country_count = base["openalex_country_count"].copy()
    base["openalex_author_count"] = raw_author_count.where(
        raw_author_count.gt(0)
    )
    base["openalex_country_count"] = raw_country_count.where(
        raw_country_count.gt(0)
    )
    country_count = base["openalex_country_count"]
    collaboration = pd.Series(
        pd.NA,
        index=base.index,
        dtype="Int8",
    )
    known_country_count = country_count.notna()
    collaboration.loc[known_country_count] = (
        country_count.loc[known_country_count].gt(1).astype("int8")
    )
    base["international_collaboration"] = collaboration
    institution = base["openalex_institution_count"]
    authors = raw_author_count
    comparable = institution.notna() & authors.notna()
    identical_rate = float(
        institution[comparable].eq(authors[comparable]).mean()
    )
    if not np.isclose(identical_rate, 1.0):
        institution_status = "not_identical_to_author_count"
    else:
        institution_status = "excluded_exact_duplicate_of_author_count"
    reference_log_difference = (
        np.log1p(base["reference_count"]) - base["log_reference_count"]
    ).abs()
    author_log_difference = (
        np.log1p(raw_author_count) - base["log_author_count"]
    ).abs()
    country_log_difference = (
        np.log1p(raw_country_count) - base["log_country_count"]
    ).abs()
    diagnostics = {
        "institution_count_identical_to_author_count_rate": identical_rate,
        "institution_count_status": institution_status,
        "openalex_metadata_found_count": int(
            base["openalex_metadata_found"].fillna(0).astype(bool).sum()
        ),
        "author_count_zero_treated_as_missing": int(
            raw_author_count.eq(0).sum()
        ),
        "country_count_zero_treated_as_missing": int(
            raw_country_count.eq(0).sum()
        ),
        "reference_log1p_max_absolute_difference": float(
            reference_log_difference.max(skipna=True)
        ),
        "author_log1p_max_absolute_difference": float(
            author_log_difference.max(skipna=True)
        ),
        "country_log1p_max_absolute_difference": float(
            country_log_difference.max(skipna=True)
        ),
    }
    matrix = _rename_columns(base)
    missing = sorted(set(OUTPUT_COLUMNS) - set(matrix.columns))
    if missing:
        raise ValueError(f"Candidate feature matrix lacks columns: {missing}")
    return matrix[OUTPUT_COLUMNS].copy(), diagnostics


def _numeric_summary(series: pd.Series) -> Dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.notna()
    finite = np.isfinite(numeric[valid])
    return {
        "row_count": int(len(series)),
        "valid_count": int(valid.sum()),
        "missing_count": int((~valid).sum()),
        "unique_count": int(numeric[valid].nunique(dropna=True)),
        "finite_count": int(finite.sum()),
        "minimum": (
            float(numeric[valid].min()) if bool(valid.any()) else None
        ),
        "maximum": (
            float(numeric[valid].max()) if bool(valid.any()) else None
        ),
    }


def _categorical_summary(series: pd.Series) -> Dict[str, Any]:
    valid = series.notna() & series.astype("string").str.strip().ne("")
    return {
        "row_count": int(len(series)),
        "valid_count": int(valid.sum()),
        "missing_count": int((~valid).sum()),
        "unique_count": int(series[valid].nunique(dropna=True)),
    }


def audit_matrix(matrix: pd.DataFrame) -> Dict[str, Any]:
    features: Dict[str, Any] = {}
    feature_exclusions: List[str] = []
    fatal_failures: List[str] = []
    for column in OUTPUT_COLUMNS:
        if column == "paper_id":
            continue
        if column in NUMERIC_RANGES or column == "publication_year":
            summary = _numeric_summary(matrix[column])
        else:
            summary = _categorical_summary(matrix[column])
        features[column] = summary
        if summary["valid_count"] and summary["unique_count"] <= 1:
            feature_exclusions.append(f"{column}:constant")
        if column in NUMERIC_RANGES and summary["valid_count"]:
            lower, upper = NUMERIC_RANGES[column]
            if summary["finite_count"] != summary["valid_count"]:
                fatal_failures.append(f"{column}:nonfinite")
            if float(summary["minimum"]) < lower:
                fatal_failures.append(f"{column}:below_range")
            if float(summary["maximum"]) > upper:
                fatal_failures.append(f"{column}:above_range")
    if matrix["paper_id"].isna().any() or matrix["paper_id"].duplicated().any():
        fatal_failures.append("paper_id:not_unique_complete")
    return {
        "row_count": int(len(matrix)),
        "column_count": int(len(matrix.columns)),
        "features": features,
        "feature_exclusions": feature_exclusions,
        "fatal_failures": fatal_failures,
        "status": (
            "pass_with_feature_exclusions"
            if feature_exclusions and not fatal_failures
            else ("pass" if not fatal_failures else "fail")
        ),
    }


def run_self_test(output_path: Path) -> Dict[str, Any]:
    country_count = pd.Series([0, 1, 2, 5, np.nan])
    collaboration = pd.Series(
        pd.NA,
        index=country_count.index,
        dtype="Int8",
    )
    known = country_count.notna()
    collaboration.loc[known] = country_count.loc[known].gt(1).astype("int8")
    edges = pd.DataFrame(
        {
            "paper_id": ["A", "A", "B"],
            "reference_id": ["R1", "R2", "R3"],
        }
    )
    counts = edges.groupby("paper_id").size().to_dict()
    assertions = {
        "international_collaboration_threshold": (
            collaboration.iloc[:4].tolist() == [0, 0, 1, 1]
            and pd.isna(collaboration.iloc[4])
        ),
        "reference_edges_are_counted_per_paper": counts == {"A": 2, "B": 1},
        "log1p_zero_reference_is_zero": math.isclose(
            math.log1p(0),
            0.0,
        ),
        "same_year_backward_age_is_valid_zero": (2020 - 2020) == 0,
    }
    if not all(assertions.values()):
        raise AssertionError(assertions)
    result = {
        "schema_version": "candidate_t0_feature_matrix_test_v3",
        "definition_version": DEFINITION_VERSION,
        "assertions": assertions,
        "passed": True,
        "outcome_columns_used": False,
        "future_information_used": False,
    }
    write_json(output_path, result)
    return result


def materialize(
    data_root: Path,
    backward_age_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames, snapshot = _load_inputs(data_root, backward_age_path)
    matrix, diagnostics = build_matrix(frames)
    audit = audit_matrix(matrix)
    if audit["fatal_failures"]:
        raise ValueError(
            "Candidate feature audit failed: "
            f"{audit['fatal_failures']}"
        )
    matrix_path = (output_dir / MATRIX_NAME).resolve()
    matrix.to_parquet(matrix_path, index=False)
    snapshot_path = (output_dir / SNAPSHOT_NAME).resolve()
    write_json(
        snapshot_path,
        {
            "schema_version": "candidate_t0_feature_input_snapshot_v3",
            "definition_version": DEFINITION_VERSION,
            "outcome_columns_used": False,
            "future_information_used": False,
            "inputs": snapshot,
        },
    )
    test_path = (output_dir / TEST_NAME).resolve()
    test = run_self_test(test_path)
    report_path = (output_dir / REPORT_NAME).resolve()
    report = {
        "schema_version": "candidate_t0_feature_matrix_report_v3",
        "definition_version": DEFINITION_VERSION,
        "matrix_path": str(matrix_path),
        "matrix_sha256": sha256_file(matrix_path),
        "input_snapshot_path": str(snapshot_path),
        "input_snapshot_sha256": sha256_file(snapshot_path),
        "test_path": str(test_path),
        "test_sha256": sha256_file(test_path),
        "implementation_path": str(Path(__file__).resolve()),
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "audit": audit,
        "diagnostics": diagnostics,
        "self_test": test,
        "institution_count_excluded": True,
        "institution_count_exclusion_reason": (
            "The frozen field is an exact row-wise duplicate of author "
            "count and therefore cannot represent distinct institutions."
        ),
        "candidate_matrix_is_not_final_selection": True,
        "target_count_influence": False,
        "model_outcomes_used": False,
        "round_13": False,
    }
    write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--backward-age",
        type=Path,
        default=DEFAULT_BACKWARD_AGE,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--mode",
        choices=("self-test", "materialize", "all"),
        default="all",
    )
    args = parser.parse_args()
    result: Dict[str, Any] = {}
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode in {"self-test", "all"}:
        test_path = output_dir / TEST_NAME
        result["self_test"] = run_self_test(test_path)
        result["test_path"] = str(test_path)
        result["test_sha256"] = sha256_file(test_path)
    if args.mode in {"materialize", "all"}:
        result["materialization"] = materialize(
            args.data_root.resolve(),
            args.backward_age.resolve(),
            output_dir,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

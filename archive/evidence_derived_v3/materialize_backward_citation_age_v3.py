from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

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
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "operationalizations"
FEATURE_ID = "EF0052"
FEATURE_COLUMN = "backward_citation_age_mean"
DEFINITION_VERSION = "backward_citation_age_mean_v3_20260730"


def _require_unique(
    frame: pd.DataFrame,
    columns: List[str],
    label: str,
) -> None:
    """Reject ambiguous identifiers in a frozen input table."""
    duplicate_count = int(frame.duplicated(columns).sum())
    if duplicate_count:
        raise ValueError(
            f"{label} has {duplicate_count} duplicate rows for {columns}"
        )


def compute_backward_citation_age(
    papers: pd.DataFrame,
    references: pd.DataFrame,
    reference_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the source-defined mean age of each paper's references.

    The age of one backward citation is the focal publication year minus
    the referenced publication year. Same-year references are valid zeroes.
    References with unknown years or apparent future years are excluded
    from the mean and reported in separate coverage fields. A paper with no
    valid reference year receives a missing value rather than zero.

    Args:
        papers: One row per focal paper with publication year.
        references: One row per focal-paper/reference edge.
        reference_metadata: One row per referenced work with its year.

    Returns:
        One row per focal paper with the mean age and audit counts.
    """
    required = {
        "papers": (papers, {"paper_id", "publication_year"}),
        "references": (references, {"paper_id", "reference_id"}),
        "reference_metadata": (
            reference_metadata,
            {"reference_id", "reference_year"},
        ),
    }
    for label, (frame, columns) in required.items():
        missing = columns - set(frame.columns)
        if missing:
            raise ValueError(f"{label} lacks columns: {sorted(missing)}")
    _require_unique(papers, ["paper_id"], "papers")
    _require_unique(
        reference_metadata,
        ["reference_id"],
        "reference_metadata",
    )
    _require_unique(
        references,
        ["paper_id", "reference_id"],
        "references",
    )

    focal = papers[["paper_id", "publication_year"]].copy()
    edges = references[["paper_id", "reference_id"]].merge(
        focal,
        on="paper_id",
        how="left",
        validate="many_to_one",
    )
    if bool(edges["publication_year"].isna().any()):
        count = int(edges["publication_year"].isna().sum())
        raise ValueError(f"{count} reference edges lack a focal paper year")
    edges = edges.merge(
        reference_metadata[["reference_id", "reference_year"]],
        on="reference_id",
        how="left",
        validate="many_to_one",
    )
    edges["reference_age"] = (
        edges["publication_year"] - edges["reference_year"]
    )
    edges["valid_reference_year"] = (
        edges["reference_year"].notna()
        & edges["reference_age"].ge(0)
    )
    edges["missing_reference_year"] = edges["reference_year"].isna()
    edges["future_dated_reference"] = edges["reference_age"].lt(0)
    edges["valid_reference_age"] = edges["reference_age"].where(
        edges["valid_reference_year"]
    )

    grouped = edges.groupby("paper_id", sort=False, observed=True)
    audit = grouped.agg(
        reference_edge_count=("reference_id", "size"),
        reference_year_valid_count=("valid_reference_year", "sum"),
        reference_year_missing_count=("missing_reference_year", "sum"),
        future_dated_reference_count=("future_dated_reference", "sum"),
        backward_citation_age_mean=("valid_reference_age", "mean"),
    ).reset_index()
    output = focal.merge(
        audit,
        on="paper_id",
        how="left",
        validate="one_to_one",
    )
    count_columns = [
        "reference_edge_count",
        "reference_year_valid_count",
        "reference_year_missing_count",
        "future_dated_reference_count",
    ]
    output[count_columns] = output[count_columns].fillna(0).astype("int64")
    denominator = output["reference_edge_count"].where(
        output["reference_edge_count"].gt(0)
    )
    output["reference_year_coverage"] = (
        output["reference_year_valid_count"] / denominator
    )
    output["definition_version"] = DEFINITION_VERSION
    return output[
        [
            "paper_id",
            "publication_year",
            FEATURE_COLUMN,
            "reference_edge_count",
            "reference_year_valid_count",
            "reference_year_missing_count",
            "future_dated_reference_count",
            "reference_year_coverage",
            "definition_version",
        ]
    ].sort_values("paper_id", kind="stable")


def _input_description(path: Path) -> Dict[str, Any]:
    """Describe and hash one immutable Parquet input."""
    metadata = parquet.ParquetFile(path).metadata
    schema = parquet.read_schema(path)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "row_count": metadata.num_rows,
        "columns": [
            {"name": field.name, "type": str(field.type)}
            for field in schema
        ],
    }


def materialize(
    data_root: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Materialize the feature and its frozen input snapshot."""
    inputs = {
        "papers": data_root / "papers_common_all.parquet",
        "references": data_root / "paper_references.parquet",
        "reference_metadata": data_root / "reference_metadata.parquet",
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    papers = pd.read_parquet(
        inputs["papers"],
        columns=["paper_id", "publication_year"],
    )
    references = pd.read_parquet(
        inputs["references"],
        columns=["paper_id", "reference_id"],
    )
    reference_metadata = pd.read_parquet(
        inputs["reference_metadata"],
        columns=["reference_id", "reference_year"],
    )
    output = compute_backward_citation_age(
        papers,
        references,
        reference_metadata,
    )
    feature_path = (
        output_dir / "backward_citation_age_mean_v3.parquet"
    ).resolve()
    output.to_parquet(feature_path, index=False)
    input_snapshot_path = (
        output_dir / "backward_citation_age_mean_v3_input_snapshot.json"
    ).resolve()
    snapshot = {
        "schema_version": "feature_input_snapshot_v3",
        "feature_id": FEATURE_ID,
        "definition_version": DEFINITION_VERSION,
        "outcome_blind": True,
        "inputs": {
            key: _input_description(path)
            for key, path in inputs.items()
        },
    }
    write_json(input_snapshot_path, snapshot)
    valid = output[FEATURE_COLUMN].notna()
    summary = {
        "feature_id": FEATURE_ID,
        "feature_column": FEATURE_COLUMN,
        "definition_version": DEFINITION_VERSION,
        "row_count": int(len(output)),
        "valid_count": int(valid.sum()),
        "missing_count": int((~valid).sum()),
        "unique_count": int(output.loc[valid, FEATURE_COLUMN].nunique()),
        "minimum": float(output.loc[valid, FEATURE_COLUMN].min()),
        "maximum": float(output.loc[valid, FEATURE_COLUMN].max()),
        "zero_count": int(output[FEATURE_COLUMN].eq(0).sum()),
        "feature_path": str(feature_path),
        "feature_sha256": sha256_file(feature_path),
        "input_snapshot_path": str(input_snapshot_path),
        "input_snapshot_sha256": sha256_file(input_snapshot_path),
    }
    return summary


def run_self_test(output_path: Path) -> Dict[str, Any]:
    """Exercise empty, missing, zero, future-year, and normal cases."""
    papers = pd.DataFrame(
        {
            "paper_id": ["P_EMPTY", "P_MIXED", "P_INVALID", "P_NORMAL"],
            "publication_year": [2020, 2020, 2020, 2010],
        }
    )
    references = pd.DataFrame(
        {
            "paper_id": [
                "P_MIXED",
                "P_MIXED",
                "P_MIXED",
                "P_MIXED",
                "P_INVALID",
                "P_NORMAL",
                "P_NORMAL",
            ],
            "reference_id": ["R1", "R2", "R3", "R4", "R5", "R6", "R7"],
        }
    )
    reference_metadata = pd.DataFrame(
        {
            "reference_id": ["R1", "R2", "R3", "R4", "R5", "R6", "R7"],
            "reference_year": [2010, 2020, 2021, None, 2022, 2000, 2005],
        }
    )
    output = compute_backward_citation_age(
        papers,
        references,
        reference_metadata,
    ).set_index("paper_id")
    assertions = {
        "empty_set_is_missing": math.isnan(
            float(output.loc["P_EMPTY", FEATURE_COLUMN])
        ),
        "same_year_is_valid_zero": (
            float(output.loc["P_MIXED", FEATURE_COLUMN]) == 5.0
        ),
        "mixed_valid_count": (
            int(output.loc["P_MIXED", "reference_year_valid_count"]) == 2
        ),
        "mixed_coverage": (
            float(output.loc["P_MIXED", "reference_year_coverage"]) == 0.5
        ),
        "future_year_excluded": (
            int(output.loc["P_MIXED", "future_dated_reference_count"]) == 1
        ),
        "missing_year_reported": (
            int(output.loc["P_MIXED", "reference_year_missing_count"]) == 1
        ),
        "all_invalid_is_missing": math.isnan(
            float(output.loc["P_INVALID", FEATURE_COLUMN])
        ),
        "normal_mean": (
            float(output.loc["P_NORMAL", FEATURE_COLUMN]) == 7.5
        ),
    }
    if not all(assertions.values()):
        raise AssertionError(assertions)
    result = {
        "schema_version": "feature_operationalization_test_v3",
        "feature_id": FEATURE_ID,
        "definition_version": DEFINITION_VERSION,
        "assertions": assertions,
        "passed": True,
        "outcome_data_used": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, result)
    return result


def main() -> None:
    """Materialize the feature, run its tests, or both."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--mode",
        choices=["materialize", "self-test", "all"],
        default="all",
    )
    args = parser.parse_args()
    result: Dict[str, Any] = {}
    if args.mode in {"self-test", "all"}:
        test_path = (
            args.output_dir
            / "backward_citation_age_mean_v3_test.json"
        ).resolve()
        result["test"] = run_self_test(test_path)
        result["test_path"] = str(test_path)
        result["test_sha256"] = sha256_file(test_path)
    if args.mode in {"materialize", "all"}:
        result["materialization"] = materialize(
            args.data_root.resolve(),
            args.output_dir.resolve(),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

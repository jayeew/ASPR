from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fetch_nature_supplemental_horizons_v5 import (
    DEFAULT_COHORTS,
    DEFAULT_COMPLETE_END_YEAR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCE_DIR,
    parse_cohort_specs,
    read_cohort_targets,
)
from scripts.nature_portfolio_v5 import utc_now


def _row_counts(output_dir: Path) -> Dict[str, int]:
    paths = {
        "future_citers": output_dir / "future_citers.parquet",
        "future_fetch_status": output_dir / "future_fetch_status.parquet",
        "future_request_manifest": output_dir / "future_request_manifest.parquet",
        "future_graph_deltas": (
            output_dir / "future_graph_deltas_multihorizon.parquet"
        ),
    }
    return {
        name: int(pq.ParquetFile(path).metadata.num_rows)
        for name, path in paths.items()
    }


def _expected_counts(
    targets: pd.DataFrame,
    cohort_specs: Mapping[int, Sequence[int]],
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for year, horizons in cohort_specs.items():
        n_papers = int(targets["publication_year"].eq(year).sum())
        for horizon in horizons:
            counts[f"{year}+{int(horizon)}"] = n_papers
    return counts


def _observed_counts(delta: pd.DataFrame) -> Dict[str, int]:
    counts = (
        delta.groupby(["publication_year", "horizon"], observed=True)
        .size()
        .sort_index()
    )
    return {
        f"{int(year)}+{int(horizon)}": int(count)
        for (year, horizon), count in counts.items()
    }


def _frame_checks(
    targets: pd.DataFrame,
    cohort_specs: Mapping[int, Sequence[int]],
    output_dir: Path,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    delta = pd.read_parquet(
        output_dir / "future_graph_deltas_multihorizon.parquet"
    )
    status = pd.read_parquet(output_dir / "future_fetch_status.parquet")
    requests = pd.read_parquet(output_dir / "future_request_manifest.parquet")
    expected = _expected_counts(targets, cohort_specs)
    observed = _observed_counts(delta)
    requested_pairs = {
        (int(year), int(horizon))
        for year, horizons in cohort_specs.items()
        for horizon in horizons
    }
    actual_pairs = set(
        zip(
            delta["publication_year"].astype(int),
            delta["horizon"].astype(int),
        )
    )
    checks = {
        "cohort_counts_exact": observed == expected,
        "cohort_pairs_exact": actual_pairs == requested_pairs,
        "delta_primary_key_unique": not delta.duplicated(
            ["paper_id", "horizon"]
        ).any(),
        "status_primary_key_unique": not status.duplicated(
            ["paper_id", "requested_horizon"]
        ).any(),
        "request_primary_key_unique": not requests.duplicated(
            ["paper_id", "requested_horizon"]
        ).any(),
        "all_fetches_success": bool(status["fetch_status"].eq("success").all()),
        "all_delta_rows_valid": bool(
            delta["fetch_status"].eq("success").all()
            and delta["fetch_valid"].eq(1).all()
        ),
        "status_target_coverage_exact": set(status["paper_id"])
        == set(targets["paper_id"]),
        "request_target_coverage_exact": set(requests["paper_id"])
        == set(targets["paper_id"]),
        "no_incomplete_outcome_window": bool(
            (
                delta["publication_year"].astype(int)
                + delta["horizon"].astype(int)
                <= DEFAULT_COMPLETE_END_YEAR
            ).all()
        ),
        "nonnegative_future_counts": bool(
            delta["n_future_citers"].fillna(-1).ge(0).all()
        ),
    }
    return {
        "expected_cohort_rows": expected,
        "observed_cohort_rows": observed,
        "checks": checks,
        "n_failed_status_rows": int(status["fetch_status"].ne("success").sum()),
        "n_invalid_delta_rows": int(delta["fetch_valid"].ne(1).sum()),
    }, delta


def _citer_checks(
    targets: pd.DataFrame,
    cohort_specs: Mapping[int, Sequence[int]],
    output_dir: Path,
    delta: pd.DataFrame,
) -> Dict[str, Any]:
    citer_path = output_dir / "future_citers.parquet"
    citers = pd.read_parquet(
        citer_path,
        columns=["paper_id", "horizon", "citer_id", "citer_year"],
    )
    publication_years = targets.set_index("paper_id")[
        "publication_year"
    ].astype(int)
    citers["publication_year"] = citers["paper_id"].map(publication_years)
    missing_parent = citers["publication_year"].isna()
    valid_rows = citers.loc[~missing_parent].copy()
    valid_rows["publication_year"] = valid_rows["publication_year"].astype(int)
    invalid_start = valid_rows["citer_year"].astype(int).le(
        valid_rows["publication_year"]
    )
    invalid_end = valid_rows["citer_year"].astype(int).gt(
        valid_rows["publication_year"] + valid_rows["horizon"].astype(int)
    )
    invalid_complete_year = valid_rows["citer_year"].astype(int).gt(
        DEFAULT_COMPLETE_END_YEAR
    )
    actual_counts = (
        citers.groupby(["paper_id", "horizon"], observed=True)
        .size()
        .rename("observed_n")
        .reset_index()
    )
    expected_counts = delta[
        ["paper_id", "horizon", "n_future_citers"]
    ].copy()
    reconciled = expected_counts.merge(
        actual_counts,
        on=["paper_id", "horizon"],
        how="left",
        validate="one_to_one",
    )
    reconciled["observed_n"] = reconciled["observed_n"].fillna(0).astype(int)
    count_mismatches = reconciled["observed_n"].ne(
        reconciled["n_future_citers"].astype(int)
    )
    monotonic_failures = 0
    for year, horizons in cohort_specs.items():
        if len(horizons) < 2:
            continue
        subset = delta[delta["publication_year"].eq(year)].pivot(
            index="paper_id",
            columns="horizon",
            values="n_future_citers",
        )
        ordered = list(sorted(int(value) for value in horizons))
        for lower, upper in zip(ordered, ordered[1:]):
            monotonic_failures += int(subset[lower].gt(subset[upper]).sum())
    return {
        "n_rows": int(len(citers)),
        "n_primary_key_duplicates": int(
            citers.duplicated(["paper_id", "horizon", "citer_id"]).sum()
        ),
        "n_missing_target_parents": int(missing_parent.sum()),
        "n_citers_not_after_publication": int(invalid_start.sum()),
        "n_citers_after_horizon": int(invalid_end.sum()),
        "n_citers_after_complete_end_year": int(invalid_complete_year.sum()),
        "n_delta_count_mismatches": int(count_mismatches.sum()),
        "n_nested_window_monotonicity_failures": monotonic_failures,
        "checks": {
            "primary_key_unique": not citers.duplicated(
                ["paper_id", "horizon", "citer_id"]
            ).any(),
            "target_referential_integrity": not missing_parent.any(),
            "strictly_post_publication": not invalid_start.any(),
            "within_requested_horizon": not invalid_end.any(),
            "within_complete_end_year": not invalid_complete_year.any(),
            "delta_counts_reconcile": not count_mismatches.any(),
            "nested_window_counts_monotonic": monotonic_failures == 0,
        },
    }


def audit(
    target_works: Path,
    output_dir: Path,
    cohort_specs: Mapping[int, Sequence[int]],
) -> Dict[str, Any]:
    targets = read_cohort_targets(target_works, cohort_specs)
    frame_audit, delta = _frame_checks(
        targets, cohort_specs, output_dir
    )
    citer_audit = _citer_checks(
        targets, cohort_specs, output_dir, delta
    )
    row_counts = _row_counts(output_dir)
    manifest = json.loads(
        (output_dir / "future_supplemental_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    manifest_counts_match = {
        "future_citers": row_counts["future_citers"]
        == int(manifest["n_future_citer_rows"]),
        "future_fetch_status": row_counts["future_fetch_status"]
        == int(manifest["n_fetch_status_rows"]),
        "future_graph_deltas": row_counts["future_graph_deltas"]
        == int(manifest["n_future_delta_rows"]),
        "target_papers": len(targets) == int(manifest["n_target_papers"]),
    }
    all_checks = [
        *frame_audit["checks"].values(),
        *citer_audit["checks"].values(),
        *manifest_counts_match.values(),
    ]
    return {
        "artifact_kind": "nature_portfolio_v5_supplemental_independent_audit",
        "created_at": utc_now(),
        "dataset_grain": {
            "future_citers": ["paper_id", "horizon", "citer_id"],
            "future_graph_deltas": ["paper_id", "horizon"],
            "future_fetch_status": ["paper_id", "requested_horizon"],
        },
        "intended_use": (
            "Complete post-publication outcome labels through 2025; "
            "not publication-day model features."
        ),
        "physical_row_counts": row_counts,
        "frame_audit": frame_audit,
        "citer_audit": citer_audit,
        "manifest_counts_match": manifest_counts_match,
        "overall_pass": all(bool(value) for value in all_checks),
        "severity": "none" if all(bool(value) for value in all_checks) else "critical",
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently audit Nature Portfolio v5 supplemental horizons."
    )
    parser.add_argument(
        "--target-works",
        type=Path,
        default=DEFAULT_SOURCE_DIR / "nature_target_works.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cohort", action="append", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cohort_specs = parse_cohort_specs(
        args.cohort or DEFAULT_COHORTS,
        complete_end_year=DEFAULT_COMPLETE_END_YEAR,
    )
    result = audit(args.target_works, args.output_dir, cohort_specs)
    output_path = args.output_dir / "independent_data_quality_audit.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "overall_pass": result["overall_pass"],
                "output": str(output_path.resolve()),
                "physical_row_counts": result["physical_row_counts"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

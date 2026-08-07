"""Versioned data contract for the expanded Nature multi-horizon cohort."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

EXPANDED_DATASET_VERSION = "nature-multihorizon-expanded-v1"
HORIZON_PUBLICATION_YEAR_MAX: Mapping[int, int] = {
    3: 2022,
    5: 2020,
    8: 2017,
}
PRIMARY_KEY = ("paper_id", "horizon")
FEATURE_FILES: Sequence[str] = (
    "innovation_features.parquet",
    "control_features.parquet",
    "opportunity_features.parquet",
)


def sha256_file(path: Path) -> str:
    """Return a stable SHA-256 identifier for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write deterministic UTF-8 JSON."""
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_source(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    required = {"paper_id", "publication_year", "horizon"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")
    output = frame.copy()
    output["paper_id"] = output["paper_id"].astype(str)
    output["publication_year"] = pd.to_numeric(
        output["publication_year"], errors="raise"
    ).astype(int)
    output["horizon"] = pd.to_numeric(output["horizon"], errors="raise").astype(int)
    if output.duplicated(list(PRIMARY_KEY)).any():
        raise ValueError(f"{label} has duplicate paper/horizon rows")
    return output


def build_expanded_future_layer(
    *,
    historical_path: Path,
    supplemental_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Combine audited historical and supplemental future-label partitions.

    The output contains exactly the registered mature windows: D3 through
    2022, D5 through 2020, and D8 through 2017. Historical rows take
    precedence through 2017; supplemental rows supply later mature windows.
    """
    historical = _validate_source(
        pd.read_parquet(historical_path), label="historical future layer"
    )
    supplemental = _validate_source(
        pd.read_parquet(supplemental_path), label="supplemental future layer"
    )
    historical["partition"] = "historical_through_2017"
    supplemental["partition"] = "supplemental_2018_2024"
    combined = pd.concat([historical, supplemental], ignore_index=True, sort=False)
    allowed = combined["horizon"].map(HORIZON_PUBLICATION_YEAR_MAX)
    combined = combined[
        allowed.notna() & combined["publication_year"].le(allowed)
    ].copy()
    combined["partition_priority"] = combined["partition"].map(
        {"historical_through_2017": 0, "supplemental_2018_2024": 1}
    )
    combined = combined.sort_values(
        ["paper_id", "horizon", "partition_priority"], kind="stable"
    )
    combined = combined.drop_duplicates(list(PRIMARY_KEY), keep="last")
    combined = combined.drop(columns="partition_priority")
    combined = combined.sort_values(
        ["publication_year", "paper_id", "horizon"], kind="stable"
    ).reset_index(drop=True)
    if combined.duplicated(list(PRIMARY_KEY)).any():
        raise ValueError("expanded future layer has duplicate primary keys")
    observed_caps = {
        int(horizon): int(group["publication_year"].max())
        for horizon, group in combined.groupby("horizon", observed=True)
    }
    if observed_caps != dict(HORIZON_PUBLICATION_YEAR_MAX):
        raise ValueError(
            f"expanded future-layer caps are {observed_caps}, "
            f"expected {dict(HORIZON_PUBLICATION_YEAR_MAX)}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    deltas_path = output_dir / "future_graph_deltas_multihorizon.parquet"
    requests_path = output_dir / "future_request_manifest.parquet"
    status_path = output_dir / "future_fetch_status.parquet"
    combined.to_parquet(deltas_path, index=False)

    requests = (
        combined.groupby(["paper_id", "publication_year"], as_index=False)["horizon"]
        .max()
        .rename(columns={"horizon": "requested_horizon"})
    )
    requests["request_batch"] = EXPANDED_DATASET_VERSION
    requests.to_parquet(requests_path, index=False)

    status_columns = [
        column
        for column in (
            "paper_id",
            "publication_year",
            "horizon",
            "fetch_status",
            "fetch_valid",
            "cap_hit",
            "requested_horizon_cap_hit",
            "n_future_citers",
            "partition",
        )
        if column in combined.columns
    ]
    status = combined[status_columns].rename(columns={"horizon": "requested_horizon"})
    status.to_parquet(status_path, index=False)

    counts_by_horizon = {
        str(int(horizon)): {
            "rows": len(group),
            "unique_papers": int(group["paper_id"].nunique()),
            "publication_year_min": int(group["publication_year"].min()),
            "publication_year_max": int(group["publication_year"].max()),
        }
        for horizon, group in combined.groupby("horizon", observed=True)
    }
    manifest: dict[str, Any] = {
        "artifact_kind": "nature_expanded_future_label_layer",
        "dataset_version": EXPANDED_DATASET_VERSION,
        "grain": list(PRIMARY_KEY),
        "horizon_publication_year_max": {
            str(key): value for key, value in HORIZON_PUBLICATION_YEAR_MAX.items()
        },
        "counts_by_horizon": counts_by_horizon,
        "unique_papers": int(combined["paper_id"].nunique()),
        "future_information_role": "label_only",
        "publication_time_feature_use_forbidden": True,
        "inputs": {
            "historical": {
                "path": str(Path(historical_path).resolve()),
                "sha256": sha256_file(historical_path),
            },
            "supplemental": {
                "path": str(Path(supplemental_path).resolve()),
                "sha256": sha256_file(supplemental_path),
            },
        },
        "outputs": {
            "future_graph_deltas": {
                "path": str(deltas_path.resolve()),
                "sha256": sha256_file(deltas_path),
            },
            "future_request_manifest": {
                "path": str(requests_path.resolve()),
                "sha256": sha256_file(requests_path),
            },
            "future_fetch_status": {
                "path": str(status_path.resolve()),
                "sha256": sha256_file(status_path),
            },
        },
        "quality_checks": {
            "primary_key_unique": True,
            "registered_horizon_caps_exact": True,
            "future_labels_excluded_from_features": True,
        },
    }
    write_json(output_dir / "expanded_future_manifest.json", manifest)
    write_json(output_dir / "expanded_dataset_contract.json", manifest)
    write_json(
        output_dir / "data_quality_report.json",
        {
            "artifact_kind": "nature_expanded_future_label_quality",
            "dataset_version": EXPANDED_DATASET_VERSION,
            "overall_pass": True,
            "checks": manifest["quality_checks"],
            "counts_by_horizon": counts_by_horizon,
        },
    )
    return manifest


def audit_materialized_expanded_dataset(
    dataset_dir: Path,
    *,
    dataset_version: str | None = None,
    horizon_year_max: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    """Audit the expanded publication-time feature and training views."""
    root = Path(dataset_dir)
    contract_path = root / "expanded_dataset_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    resolved_version = str(
        dataset_version or contract.get("dataset_version") or EXPANDED_DATASET_VERSION
    )
    resolved_horizons = dict(
        horizon_year_max
        or {
            int(key): int(value)
            for key, value in contract.get(
                "horizon_publication_year_max", HORIZON_PUBLICATION_YEAR_MAX
            ).items()
        }
    )
    papers = pd.read_parquet(root / "papers_primary_articles.parquet")
    targets = pd.read_parquet(root / "targets_zero_inclusive.parquet")
    cohort = pd.read_parquet(root / "cohort_membership.parquet")
    paper_ids = set(papers["paper_id"].astype(str))
    label_columns = [
        "future_uptake",
        "future_field_reach",
        "future_subfield_reach",
        "future_topic_reach",
        "future_field_simpson",
        "future_topic_simpson",
    ]
    eligible_keys = cohort.loc[cohort["cohort_member"].eq(1), list(PRIMARY_KEY)]
    eligible_targets = eligible_keys.merge(
        targets, on=list(PRIMARY_KEY), how="left", validate="one_to_one"
    )
    checks: dict[str, bool] = {
        "paper_id_unique": not papers.duplicated("paper_id").any(),
        "target_primary_key_unique": not targets.duplicated(list(PRIMARY_KEY)).any(),
        "cohort_primary_key_unique": not cohort.duplicated(list(PRIMARY_KEY)).any(),
        "eligible_target_labels_complete": not eligible_targets[label_columns]
        .isna()
        .any()
        .any(),
    }
    feature_rows: dict[str, int] = {}
    forbidden_feature_columns: dict[str, list[str]] = {}
    for name in FEATURE_FILES:
        view = pd.read_parquet(root / name)
        feature_rows[name] = len(view)
        checks[f"{name}_paper_id_unique"] = not view.duplicated("paper_id").any()
        checks[f"{name}_paper_set_matches"] = (
            set(view["paper_id"].astype(str)) == paper_ids
        )
        forbidden = sorted(
            column
            for column in view.columns
            if column.startswith("future_")
            or column in {"horizon", "cap_hit", "cohort_member"}
        )
        forbidden_feature_columns[name] = forbidden
        checks[f"{name}_has_no_outcome_columns"] = not forbidden

    target_caps = {
        str(int(horizon)): int(group["publication_year"].max())
        for horizon, group in targets.groupby("horizon", observed=True)
    }
    checks["registered_horizon_caps_exact"] = target_caps == {
        str(key): value for key, value in resolved_horizons.items()
    }
    cohort_members = cohort[cohort["cohort_member"].eq(1)].copy()
    counts_by_horizon = {
        str(int(horizon)): {
            "target_rows": len(targets[targets["horizon"].eq(horizon)]),
            "cohort_rows": len(group),
            "publication_year_min": int(group["publication_year"].min()),
            "publication_year_max": int(group["publication_year"].max()),
            "domain_count": int(group["domain12"].nunique(dropna=False)),
            "minimum_domain_rows": int(group.groupby("domain12").size().min()),
        }
        for horizon, group in cohort_members.groupby("horizon", observed=True)
    }
    checks["all_horizons_cover_12_domains"] = all(
        item["domain_count"] == 12 for item in counts_by_horizon.values()
    )
    checks["overall_pass"] = all(checks.values())
    report: dict[str, Any] = {
        "artifact_kind": "nature_expanded_materialized_data_quality",
        "dataset_version": resolved_version,
        "dataset_dir": str(root.resolve()),
        "grain": {
            "paper_features": ["paper_id"],
            "targets_and_cohort": list(PRIMARY_KEY),
        },
        "paper_rows": len(papers),
        "feature_rows": feature_rows,
        "counts_by_horizon": counts_by_horizon,
        "horizon_publication_year_max": target_caps,
        "forbidden_feature_columns": forbidden_feature_columns,
        "checks": checks,
        "overall_pass": checks["overall_pass"],
    }
    write_json(root / "materialized_data_quality_report.json", report)
    contract["materialized_dataset"] = {
        "path": str(root.resolve()),
        "paper_rows": len(papers),
        "counts_by_horizon": counts_by_horizon,
        "quality_report": str(
            (root / "materialized_data_quality_report.json").resolve()
        ),
        "overall_pass": report["overall_pass"],
    }
    write_json(contract_path, contract)
    return report


__all__ = [
    "EXPANDED_DATASET_VERSION",
    "HORIZON_PUBLICATION_YEAR_MAX",
    "audit_materialized_expanded_dataset",
    "build_expanded_future_layer",
    "sha256_file",
    "write_json",
]

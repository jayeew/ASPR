"""Audit calibration, lift, worst-group, and interval coverage for HGB-P OOF predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TARGET = "perturbation_target_fold"
PREDICTION = "perturbation_head_p"


def audit_predictions(
    temporal_path: Path,
    domain_path: Path,
    output_path: Path,
    *,
    latest_protocol: str = "forward_temporal_latest_holdout",
    interval_level: float = 0.9,
) -> dict[str, Any]:
    """Audit existing leakage-safe predictions without refitting or changing targets."""
    temporal = _read_valid(temporal_path)
    domain = _read_valid(domain_path)
    latest = temporal[temporal["prediction_protocol"].eq(latest_protocol)].copy()
    calibration = temporal[~temporal.index.isin(latest.index)].copy()
    if latest.empty or calibration.empty:
        raise ValueError(
            "temporal audit requires prior calibration and latest holdout rows"
        )
    temporal_metrics = _metrics(latest)
    temporal_interval = _interval_coverage(calibration, latest, interval_level)
    domain_metrics = _metrics(domain)
    worst_group = _worst_group(domain)
    domain_intervals = _group_interval_coverage(domain, interval_level)
    checks = {
        "temporal_rank_positive": temporal_metrics["spearman"] > 0.0,
        "temporal_top_decile_lift_positive": temporal_metrics["top_decile_lift"] > 1.0,
        "domain_rank_positive": domain_metrics["spearman"] > 0.0,
        "domain_top_decile_lift_positive": domain_metrics["top_decile_lift"] > 1.0,
        "worst_domain_rank_positive": worst_group["metrics"]["spearman"] > 0.0,
        "temporal_interval_near_nominal": temporal_interval["coverage"]
        >= interval_level - 0.05,
        "worst_domain_interval_near_nominal": domain_intervals["worst_group_coverage"]
        >= interval_level - 0.05,
    }
    result = {
        "contract": "gear_hgb_p_prediction_audit_v1",
        "status": "supported" if all(checks.values()) else "partially_supported",
        "checks": checks,
        "uses_future_features": False,
        "predictions_refit": False,
        "interval_method": "absolute-residual split conformal diagnostic",
        "interval_level": interval_level,
        "temporal_latest": temporal_metrics,
        "temporal_interval": temporal_interval,
        "domain_overall": domain_metrics,
        "domain_worst_group": worst_group,
        "domain_leave_one_group_interval": domain_intervals,
        "temporal_prediction_sha256": _sha256(temporal_path),
        "domain_prediction_sha256": _sha256(domain_path),
        "ood_definition": "heldout domain; feature-space OOD distance unavailable",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _read_valid(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"paper_id", "domain12", TARGET, PREDICTION}
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"prediction columns are missing: {missing}")
    numeric = frame[[TARGET, PREDICTION]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("prediction audit refuses missing or non-finite values")
    return frame.assign(**{column: numeric[column] for column in numeric})


def _metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    target = frame[TARGET].to_numpy(float)
    prediction = frame[PREDICTION].to_numpy(float)
    slope, intercept = (
        np.polyfit(prediction, target, 1)
        if len(frame) >= 2 and float(np.std(prediction)) > 0.0
        else (float("nan"), float("nan"))
    )
    cutoff = float(np.quantile(prediction, 0.9))
    top = target[prediction >= cutoff]
    mean = float(np.mean(target))
    return {
        "n": len(frame),
        "spearman": float(frame[PREDICTION].corr(frame[TARGET], method="spearman")),
        "mae": float(np.mean(np.abs(target - prediction))),
        "rmse": float(np.sqrt(np.mean((target - prediction) ** 2))),
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
        "top_decile_n": len(top),
        "top_decile_target_mean": float(np.mean(top)),
        "overall_target_mean": mean,
        "top_decile_lift": float(np.mean(top) / mean) if mean > 0.0 else float("nan"),
    }


def _interval_coverage(
    calibration: pd.DataFrame, test: pd.DataFrame, level: float
) -> dict[str, float | int]:
    residual = np.abs(calibration[TARGET] - calibration[PREDICTION]).to_numpy(float)
    radius = float(np.quantile(residual, level, method="higher"))
    error = np.abs(test[TARGET] - test[PREDICTION]).to_numpy(float)
    return {
        "calibration_n": len(calibration),
        "test_n": len(test),
        "radius": radius,
        "coverage": float(np.mean(error <= radius)),
    }


def _worst_group(frame: pd.DataFrame) -> dict[str, Any]:
    metrics = {
        str(group): _metrics(rows)
        for group, rows in frame.groupby("domain12", observed=True)
        if len(rows) >= 5
    }
    if not metrics:
        raise ValueError("no domain has at least five OOF predictions")
    worst = min(metrics, key=lambda group: float(metrics[group]["spearman"]))
    return {"minimum_group_n": 5, "worst_domain": worst, "metrics": metrics[worst]}


def _group_interval_coverage(frame: pd.DataFrame, level: float) -> dict[str, Any]:
    groups: dict[str, dict[str, float | int]] = {}
    for group, test in frame.groupby("domain12", observed=True):
        calibration = frame[frame["domain12"].ne(group)]
        groups[str(group)] = _interval_coverage(calibration, test, level)
    return {
        "groups": groups,
        "worst_group_coverage": min(
            float(value["coverage"]) for value in groups.values()
        ),
    }


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal", type=Path, required=True)
    parser.add_argument("--domain", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_predictions(args.temporal, args.domain, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["audit_predictions"]

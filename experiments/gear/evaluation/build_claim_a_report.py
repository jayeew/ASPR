"""Assemble the bounded Claim-A result without hiding worst-group limitations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_claim_a_report(
    stage_a_path: Path,
    structural_validation_path: Path,
    prediction_audit_path: Path,
    coverage_audit_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Require supported OOF/temporal/domain claims and preserve diagnostics."""
    stage_a = _read(stage_a_path)
    structural = _read(structural_validation_path)
    diagnostic = _read(prediction_audit_path)
    coverage = _read(coverage_audit_path)
    _require(
        stage_a.get("contract") == "gear_stage_a_real_data_validation_v1",
        "Stage A contract",
    )
    graph = _mapping(stage_a.get("graph_predictive_validity"), "HGB-D metrics")
    _require(graph.get("status") == "supported", "HGB-D status")
    for key in ("spearman", "worst_domain_spearman", "worst_fold_spearman"):
        _require(_number(graph.get(key), key) > 0.0, f"positive HGB-D {key}")
    _require(
        _number(graph.get("top_decile_lift"), "top-decile lift") > 1.0, "HGB-D lift"
    )
    _require(
        structural.get("contract") == "gear_structural_head_validation_v1",
        "structural validation contract",
    )
    _require(structural.get("promotion_passed") is True, "structural promotion")
    gates = _mapping(structural.get("promotion_gates"), "structural gates")
    _require(
        bool(gates) and all(value is True for value in gates.values()),
        "structural gates",
    )
    metrics = _mapping(structural.get("metrics"), "structural metrics")
    for axis in ("forward_temporal_latest", "leave_one_domain_out"):
        heads = _mapping(metrics.get(axis), axis)
        for head in ("d_excess", "perturbation"):
            values = _mapping(heads.get(head), f"{axis}:{head}")
            _require(
                _number(values.get("spearman_ci95_low"), "CI lower bound") > 0.0,
                f"positive {axis}:{head} CI",
            )
            _require(
                _number(values.get("real_minus_permuted"), "permutation contrast")
                > 0.0,
                f"positive {axis}:{head} permutation contrast",
            )
    _require(
        diagnostic.get("contract") == "gear_hgb_p_prediction_audit_v1",
        "prediction diagnostic contract",
    )
    diagnostic_checks = _mapping(diagnostic.get("checks"), "diagnostic checks")
    _require(
        diagnostic_checks.get("temporal_interval_near_nominal") is True,
        "temporal interval coverage",
    )
    _require(
        coverage.get("contract") == "gear_structural_head_coverage_audit_v1",
        "coverage contract",
    )
    _require(coverage.get("passed") is True, "registered target coverage")
    for cohort in ("stage_b_241", "stage_c_150", "runtime_10"):
        _require(
            _mapping(coverage.get(cohort), cohort).get("passed") is True,
            f"{cohort} coverage",
        )
    worst_supported = bool(
        diagnostic_checks.get("worst_domain_rank_positive") is True
        and diagnostic_checks.get("worst_domain_interval_near_nominal") is True
    )
    result = {
        "contract": "gear_claim_a_bounded_validation_v1",
        "status": "supported" if worst_supported else "supported_with_limitations",
        "claim_allowed": True,
        "uses_future_features": False,
        "claim_boundaries": {
            "hgb_d_oof": "supported",
            "hgb_p_forward_temporal": "supported",
            "hgb_p_leave_one_domain_out": "supported",
            "registered_target_coverage": "supported",
            "worst_group_consistency": (
                "supported" if worst_supported else "not_claimed"
            ),
        },
        "hgb_d": graph,
        "structural_heads": metrics,
        "coverage": {
            key: coverage[key] for key in ("stage_b_241", "stage_c_150", "runtime_10")
        },
        "diagnostic_limitations": {
            "prediction_audit_status": diagnostic.get("status"),
            "failed_checks": sorted(
                key for key, value in diagnostic_checks.items() if value is not True
            ),
            "worst_domain": diagnostic.get("domain_worst_group"),
            "worst_domain_interval_coverage": (
                diagnostic.get("domain_leave_one_group_interval") or {}
            ).get("worst_group_coverage"),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(value, str(path))


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return value


def _number(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{label} must be numeric")
    return float(value)


def _require(condition: bool, label: str) -> None:
    if not condition:
        raise ValueError(f"Claim A requirement failed: {label}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-a", type=Path, required=True)
    parser.add_argument("--structural-validation", type=Path, required=True)
    parser.add_argument("--prediction-audit", type=Path, required=True)
    parser.add_argument("--coverage-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_claim_a_report(
        args.stage_a,
        args.structural_validation,
        args.prediction_audit,
        args.coverage_audit,
        args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_claim_a_report"]

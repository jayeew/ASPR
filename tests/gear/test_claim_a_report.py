import json
from pathlib import Path

from experiments.gear.evaluation.build_claim_a_report import build_claim_a_report


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_claim_a_report_keeps_worst_group_out_of_claim_boundary(tmp_path: Path) -> None:
    stage_a = _write(
        tmp_path / "stage_a.json",
        {
            "contract": "gear_stage_a_real_data_validation_v1",
            "graph_predictive_validity": {
                "status": "supported",
                "spearman": 0.7,
                "worst_domain_spearman": 0.4,
                "worst_fold_spearman": 0.3,
                "top_decile_lift": 1.5,
            },
        },
    )
    head_metric = {
        "spearman_ci95_low": 0.01,
        "real_minus_permuted": 0.02,
    }
    structural = _write(
        tmp_path / "structural.json",
        {
            "contract": "gear_structural_head_validation_v1",
            "promotion_passed": True,
            "promotion_gates": {"all": True},
            "metrics": {
                axis: {"d_excess": head_metric, "perturbation": head_metric}
                for axis in ("forward_temporal_latest", "leave_one_domain_out")
            },
        },
    )
    prediction = _write(
        tmp_path / "prediction.json",
        {
            "contract": "gear_hgb_p_prediction_audit_v1",
            "status": "partially_supported",
            "checks": {
                "temporal_interval_near_nominal": True,
                "worst_domain_rank_positive": False,
                "worst_domain_interval_near_nominal": False,
            },
            "domain_worst_group": {"worst_domain": "small-domain"},
            "domain_leave_one_group_interval": {"worst_group_coverage": 0.75},
        },
    )
    coverage = _write(
        tmp_path / "coverage.json",
        {
            "contract": "gear_structural_head_coverage_audit_v1",
            "passed": True,
            "stage_b_241": {"passed": True},
            "stage_c_150": {"passed": True},
            "runtime_10": {"passed": True},
        },
    )
    result = build_claim_a_report(
        stage_a,
        structural,
        prediction,
        coverage,
        tmp_path / "result.json",
    )
    assert result["claim_allowed"] is True
    assert result["status"] == "supported_with_limitations"
    assert result["claim_boundaries"]["worst_group_consistency"] == "not_claimed"

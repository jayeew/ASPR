from pathlib import Path

import pandas as pd
import pytest

from experiments.gear.evaluation.audit_hgb_p_predictions import audit_predictions


def test_hgb_p_audit_uses_prior_rows_for_latest_interval(tmp_path: Path) -> None:
    rows = []
    for index in range(20):
        rows.append(
            {
                "paper_id": f"p{index}",
                "domain12": "a" if index < 10 else "b",
                "perturbation_target_fold": index / 20,
                "perturbation_head_p": index / 20 + 0.02,
                "prediction_protocol": (
                    "forward_temporal_latest_holdout"
                    if index >= 15
                    else "domain_oof_development"
                ),
            }
        )
    temporal = tmp_path / "temporal.parquet"
    domain = tmp_path / "domain.parquet"
    pd.DataFrame(rows).to_parquet(temporal, index=False)
    pd.DataFrame(rows).to_parquet(domain, index=False)
    result = audit_predictions(temporal, domain, tmp_path / "audit.json")
    assert result["temporal_latest"]["n"] == 5
    assert result["temporal_interval"]["calibration_n"] == 15
    assert result["temporal_interval"]["coverage"] == pytest.approx(1.0)
    assert result["predictions_refit"] is False
    assert result["checks"]["temporal_rank_positive"] is True


def test_hgb_p_audit_rejects_nonfinite_predictions(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "paper_id": ["p1"],
            "domain12": ["a"],
            "perturbation_target_fold": [0.2],
            "perturbation_head_p": [float("nan")],
            "prediction_protocol": ["forward_temporal_latest_holdout"],
        }
    )
    path = tmp_path / "bad.parquet"
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="non-finite"):
        audit_predictions(path, path, tmp_path / "audit.json")

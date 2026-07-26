from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.fig05.old import build_fig5_forecast_score_table as scorer


def test_load_weights_handles_fig3_unnamed_metric_column(tmp_path: Path) -> None:
    path = tmp_path / "fig3_best_weights.csv"
    path.write_text(",weight\nB,0.0\nRS,0.0\nDeltaQ0,0.25\nRTD,0.75\n", encoding="utf-8")

    weights = scorer.load_weights(path)

    assert weights["DeltaQ0"] == 0.25
    assert weights["RTD"] == 0.75
    assert weights["B"] == 0.0


def test_build_score_table_marks_forecast_scores_not_oof() -> None:
    rows = []
    for idx in range(80):
        rows.append(
            {
                "paper_id": f"p{idx}",
                "title": f"Paper {idx}",
                "domain": "test_domain",
                "year": 2000 + (idx % 4),
                "primary_field": "field_a" if idx < 40 else "field_b",
                "display_community": idx % 5,
                "is_landmark": 0,
                "reference_count": 5 + idx % 3,
                "cited_by_count": float(idx),
                "B": np.nan,
                "RS": np.nan,
                "DeltaQ0": float(idx % 11),
                "Uzzi": np.nan,
                "RTD": float((idx % 7) / 7.0),
                "BurtIP": np.nan,
                "PDE": np.nan,
                "field_variety": 2,
            }
        )
    metrics = pd.DataFrame(rows)
    weights = pd.Series({key: 0.0 for key in scorer.METRIC_KEYS})
    weights["DeltaQ0"] = 0.25
    weights["RTD"] = 0.75

    table, feature_diag, active = scorer.build_score_table(metrics, weights)

    assert "S_w_oof" not in table.columns
    assert set(active) == {"DeltaQ0", "RTD"}
    assert table["S_w"].notna().all()
    assert table["score_is_oof"].eq(0).all()
    assert feature_diag.loc[feature_diag["metric"].eq("DeltaQ0"), "active_for_learning"].iloc[0] == 1

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import fig3_v5_rank_learning as mod
from scripts import performance_gate_v5_method as gate


def test_project_simplex_returns_nonnegative_unit_sum() -> None:
    weights = mod.project_simplex(np.array([0.8, -0.2, 2.0, 0.1]))

    assert np.all(weights >= -1e-12)
    assert abs(float(weights.sum()) - 1.0) < 1e-9


def test_outer_time_folds_train_only_on_past_folds() -> None:
    frame = pd.DataFrame({"fold_id": [-1, 1, 2, 3, 4], "value": range(5)})

    folds = mod.outer_time_folds(frame)

    assert [(fold.fold_id, fold.train_idx.tolist(), fold.test_idx.tolist()) for fold in folds] == [
        (1, [0], [1]),
        (2, [0, 1], [2]),
        (3, [0, 1, 2], [3]),
        (4, [0, 1, 2, 3], [4]),
    ]


def test_pairwise_ranker_learns_synthetic_monotone_signal() -> None:
    rng = np.random.default_rng(7)
    n_rows = 180
    x0 = rng.normal(size=n_rows)
    x1 = rng.normal(size=n_rows)
    target = 1.7 * x0 - 0.2 * x1 + rng.normal(scale=0.05, size=n_rows)
    frame = pd.DataFrame(
        {
            "paper_id": [f"p{i}" for i in range(n_rows)],
            "fold_id": np.repeat([-1, 1, 2, 3], repeats=45),
            "B_z": x0,
            "RS_z": x1,
            "DeltaQ0_z": 0.0,
            "Uzzi_z": 0.0,
            "RTD_z": 0.0,
            "BurtIP_z": 0.0,
            "PDE_z": 0.0,
            "RGPM": target,
            "S_equal": (x0 + x1) / 2.0,
        }
    )

    result = mod.run_nested_rank_learning(
        frame,
        feature_cols=mod.DEFAULT_FEATURE_COLS,
        target_col="RGPM",
        seed=11,
        max_pairs=4000,
        epochs=160,
        learning_rate=0.08,
        model_grid=["signed_pairwise"],
        l2_grid=[0.001],
    )

    assert result.summary["learned_oof_spearman"] > 0.85
    assert result.summary["learned_oof_spearman"] > result.summary["equal_weight_oof_spearman"]


def test_within_domain_percentile_transform_is_domain_local() -> None:
    frame = pd.DataFrame(
        {
            "domain": ["a", "a", "b", "b"],
            "RGPM": [10.0, 20.0, 100.0, 200.0],
        }
    )

    out = mod.apply_target_transform(frame, "RGPM", "within_domain_percentile")

    assert out[mod.TRANSFORMED_TARGET_COL].tolist() == [0.5, 1.0, 0.5, 1.0]


def test_v5_gate_requires_oof_latest_deltas_and_baseline_improvement(tmp_path: Path) -> None:
    v5_dir = tmp_path / "v5"
    baseline_dir = tmp_path / "baseline"
    v5_dir.mkdir()
    baseline_dir.mkdir()
    (v5_dir / "fig3_v5_diagnostics_summary.json").write_text(
        json.dumps(
            {
                "learned_oof_spearman": 0.44,
                "equal_weight_oof_spearman": 0.20,
                "learned_vs_equal_delta": 0.24,
                "latest_fold_test_spearman": 0.40,
                "n_contributing_graph_deltas": 5,
            }
        ),
        encoding="utf-8",
    )
    (v5_dir / "fig3_v5_effect_summary.json").write_text(
        json.dumps({"top_vs_bottom_score_decile_rgpm_top10_enrichment": 7.0}),
        encoding="utf-8",
    )
    (baseline_dir / "fig3_diagnostics_summary.json").write_text(
        json.dumps({"learned_oof_spearman": 0.30}),
        encoding="utf-8",
    )

    decision = gate.build_decision(
        v5_dir,
        baseline_dir,
        min_oof=0.45,
        min_latest_fold=0.35,
        min_learned_vs_equal=0.03,
        min_contributing_deltas=5,
        min_enrichment=5.0,
    )

    assert decision["checks"]["learned_oof_spearman"] is False
    assert decision["checks"]["latest_fold"] is True
    assert decision["checks"]["beats_baseline"] is True
    assert decision["final_pass"] is False

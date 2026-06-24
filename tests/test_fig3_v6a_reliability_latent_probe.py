from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pandas as pd

from scripts import fig3_v6a_reliability_latent_probe as mod


def test_reliability_cohorts_are_nested_and_measurement_based() -> None:
    frame = pd.DataFrame(
        {
            "n_future_citers": [0, 6, 12],
            "n_controls": [50, 80, 120],
            "control_tier": ["all_non_landmark", "field_year", "field_year_refbin"],
            "n_delta_z_clipped": [3, 1, 0],
            "n_delta_scale_floor_used": [7, 5, 4],
        }
    )

    out = mod.add_reliability_cohorts(frame)

    assert out["cohort_broad"].tolist() == [1, 1, 1]
    assert out["cohort_moderate"].tolist() == [0, 1, 1]
    assert out["cohort_strict"].tolist() == [0, 0, 1]


def test_latent_target_is_finite_and_oriented_to_future_delta_mass() -> None:
    frame = pd.DataFrame(
        {
            "community_reach": [1, 2, 3, 4, 5],
            "field_entropy": [0.1, 0.2, 0.4, 0.8, 1.6],
            "cross_community_adoption": [0.0, 0.1, 0.1, 0.4, 0.9],
            "path_shortening": [0, 0, 1, 1, 2],
            "modularity_shock": [0.0, 0.1, 0.2, 0.3, 0.4],
            "partition_change": [0.0, 0.2, 0.2, 0.5, 0.7],
            "boundary_mixing": [0.0, 0.1, 0.3, 0.6, 0.8],
            "hub_formation": [0, 1, 1, 1, 2],
        }
    )

    out = mod.add_latent_targets(frame)

    assert np.isfinite(out["RGPM_latent_future_factor"]).all()
    assert np.isfinite(out["RGPM_latent_future_percentile"]).all()
    assert out["RGPM_latent_future_percentile"].iloc[-1] > out["RGPM_latent_future_percentile"].iloc[0]


def test_v6a_gate_rejects_high_oof_when_sample_or_domain_floor_fails() -> None:
    args = argparse.Namespace(
        min_oof=0.45,
        min_latest_fold=0.35,
        min_learned_vs_equal=0.03,
        min_contributing_deltas=5,
        min_enrichment=5.0,
        min_rows=500,
        min_domains=8,
        min_rows_per_domain=20,
        max_domain_share=0.5,
    )
    row = {
        "learned_oof_spearman": 0.60,
        "latest_fold_test_spearman": 0.40,
        "learned_vs_equal_delta": 0.20,
        "n_contributing_graph_deltas": 5,
        "top_bottom_enrichment": 8.0,
        "n_rows": 200,
        "n_domains": 8,
        "min_rows_per_domain": 50,
        "max_domain_share": 0.4,
    }

    assert mod.pass_gate(row, args) is False


def test_enrichment_summary_caps_infinity_for_strict_json_gate() -> None:
    summary = mod.summarize_enrichment(
        {
            "top_vs_bottom_score_decile_rgpm_top10_enrichment": math.inf,
            "top_vs_bottom_score_decile_rgpm_top20_enrichment": 19.8,
        }
    )

    assert summary["top_bottom_enrichment"] == mod.INFINITE_ENRICHMENT_SENTINEL
    assert summary["top_bottom_enrichment_finite_max"] == 19.8
    assert summary["top_bottom_enrichment_had_infinite"] is True
    json.dumps(summary, allow_nan=False)


def test_decision_uses_strict_json_safe_values() -> None:
    args = argparse.Namespace(
        min_oof=0.45,
        min_latest_fold=0.35,
        min_learned_vs_equal=0.03,
        min_contributing_deltas=5,
        min_enrichment=5.0,
        min_rows=500,
        min_domains=8,
        min_rows_per_domain=20,
        max_domain_share=0.5,
    )
    matrix = pd.DataFrame(
        [
            {
                "run_name": "locked",
                "learned_oof_spearman": 0.6,
                "latest_fold_test_spearman": 0.5,
                "top_bottom_enrichment": math.inf,
                "v6a_gate_pass": 1,
            }
        ]
    )

    decision = mod.build_decision(matrix, args)

    assert decision["final_pass"] is True
    assert decision["next_step"] == "independent_recompute_and_materialization_gate"
    assert decision["best_run"]["top_bottom_enrichment"] is None
    json.dumps(decision, allow_nan=False)

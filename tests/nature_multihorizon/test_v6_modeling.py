from __future__ import annotations

import numpy as np
import pandas as pd

from aspr.nature_multihorizon.development_v6 import (
    audit_required_feature_sets,
    evaluate_development_gates,
    evaluate_directional_horizon_gates,
)
from aspr.nature_multihorizon.modeling_v6 import (
    evaluate_development_oof,
    evaluate_temporal_folds,
    fit_calibrated_final_model,
    make_expanding_year_folds,
    run_nested_development_oof,
    select_final_parameters,
)


def _synthetic_frame(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    index = 0
    for year in range(2000, 2012):
        for _ in range(80):
            signal = rng.normal()
            uptake = float(rng.random() < 1.0 / (1.0 + np.exp(-signal)))
            breadth = max(0.0, 3.0 + signal + rng.normal(scale=0.5))
            evenness = float(np.clip(0.5 + 0.1 * signal + rng.normal(scale=0.1), 0, 1))
            rows.append(
                {
                    "paper_id": f"W{index}",
                    "publication_year": year,
                    "domain12": "physics" if index % 2 else "chemistry",
                    "venue_family": "nature_a" if index % 3 else "nature_b",
                    "future_uptake": uptake,
                    "conditional_diffusion_member": int(
                        uptake == 1 and index % 7 != 0
                    ),
                    "future_field_reach": breadth,
                    "future_subfield_reach": breadth + 1,
                    "future_topic_reach": breadth + 2,
                    "future_field_simpson": evenness,
                    "future_topic_simpson": min(1.0, evenness + 0.05),
                    "publication_signal": signal,
                }
            )
            index += 1
    return pd.DataFrame(rows)


def test_expanding_folds_are_strictly_temporal() -> None:
    frame = _synthetic_frame()
    folds = make_expanding_year_folds(frame, n_splits=3)

    assert len(folds) == 3
    assert all(fold.train_year_max < fold.test_year_min for fold in folds)
    test_indices = np.concatenate([fold.test_index for fold in folds])
    assert len(test_indices) == len(np.unique(test_indices))


def test_nested_two_part_oof_is_complete_for_test_folds() -> None:
    frame = _synthetic_frame()
    compact = (
        {
            "parameter_id": "toy",
            "max_leaf_nodes": 7,
            "max_depth": 2,
            "min_samples_leaf": 20,
            "learning_rate": 0.1,
            "max_iter": 20,
            "l2_regularization": 1.0,
        },
    )
    predictions, ledger, folds = run_nested_development_oof(
        frame,
        feature_sets={
            "controls_only": (
                "publication_year",
                "domain12",
                "venue_family",
            ),
            "innovation_plus_controls": (
                "publication_year",
                "domain12",
                "venue_family",
                "publication_signal",
            ),
        },
        horizon=5,
        n_outer=2,
        n_inner=2,
        parameter_grid=compact,
        seed=11,
    )
    metrics, domains = evaluate_development_oof(
        predictions, bootstrap_iterations=20, min_domain_rows=20
    )

    assert len(folds) == 2
    assert not predictions.duplicated(["paper_id", "model_id"]).any()
    assert predictions["expected_diffusion_score"].between(0, 1).all()
    ineligible_positive = predictions[
        predictions["future_uptake"].eq(1)
        & predictions["conditional_diffusion_member"].eq(0)
    ]
    assert len(ineligible_positive) > 0
    assert ineligible_positive["realized_diffusion_target"].isna().all()
    assert ledger["selected"].all()
    assert set(metrics["model_id"]) == {
        "controls_only",
        "innovation_plus_controls",
    }
    assert len(domains) == 4
    temporal = evaluate_temporal_folds(predictions)
    assert temporal.groupby("model_id")["latest_development_fold"].sum().eq(1).all()
    parameters = select_final_parameters(
        ledger,
        model_id="innovation_plus_controls",
        parameter_grid=compact,
    )
    bundle = fit_calibrated_final_model(
        frame,
        predictions[
            predictions["model_id"].eq("innovation_plus_controls")
        ],
        feature_names=(
            "publication_year",
            "domain12",
            "venue_family",
            "publication_signal",
        ),
        parameters=parameters,
        horizon=5,
        seed=13,
    )
    final_predictions = bundle.predict(frame.head(10))
    assert final_predictions["expected_diffusion_score"].between(0, 1).all()


def test_feature_set_and_development_gate_audits_are_fail_closed() -> None:
    sets = {
        "controls_only": ("publication_year",),
        "innovation_plus_controls": ("publication_year", "signal"),
        "opportunity_only_plus_controls": ("publication_year", "opportunity"),
        "innovation_plus_opportunity_plus_controls": (
            "publication_year",
            "signal",
            "opportunity",
        ),
        "n1_recombination_plus_controls": ("publication_year", "n1"),
        "c1_knowledge_diversity_plus_controls": (
            "publication_year",
            "c1",
            "c2",
        ),
        "leave_out_n1_recombination": (
            "publication_year",
            "c1",
            "c2",
        ),
        "leave_out_c1_knowledge_diversity": (
            "publication_year",
            "n1",
        ),
    }
    assert audit_required_feature_sets(sets)["overall_pass"]
    incomplete = dict(sets)
    incomplete.pop("controls_only")
    assert not audit_required_feature_sets(incomplete)["overall_pass"]

    metrics = pd.DataFrame(
        [
            {
                "model_id": "innovation_plus_controls",
                "spearman_expected": 0.70,
                "spearman_ci_low": 0.65,
                "domain_macro_spearman": 0.55,
                "n_reportable_domains": 12,
                "gain_over_controls": 0.06,
                "gain_over_controls_ci_low": 0.04,
                "uptake_ece_10": 0.02,
                "uptake_brier_skill_score": 0.30,
                "realized_interval_coverage_90": 0.90,
                "realized_interval_mean_width": 0.50,
            }
        ]
    )
    temporal = pd.DataFrame(
        [
            {
                "model_id": "innovation_plus_controls",
                "latest_development_fold": 1,
                "spearman_expected": 0.50,
            }
        ]
    )
    config = {
        "outcome_protocol": {
            "primary_feature_model_id": "innovation_plus_controls"
        },
        "acceptance_gates": {
            "nested_oof_spearman_min": 0.45,
            "domain_macro_spearman_min": 0.25,
            "gain_over_strong_controls_min": 0.05,
            "temporal_holdout_spearman_min": 0.30,
            "uptake_ece_10_max": 0.05,
            "uptake_brier_skill_score_min": 0.0,
            "realized_interval_coverage_90_min": 0.87,
            "realized_interval_mean_width_max": 0.75,
        },
    }
    gates = evaluate_development_gates(
        metrics, temporal, config=config, horizon=5
    )
    assert gates["passed"].eq(1).all()
    metrics.loc[0, "gain_over_controls"] = 0.049
    failed = evaluate_development_gates(
        metrics, temporal, config=config, horizon=5
    )
    assert failed.loc[
        failed["gate_id"].eq("G5_GAIN_OVER_CONTROLS"), "passed"
    ].eq(0).all()
    directional = evaluate_directional_horizon_gates(
        metrics, temporal, config=config, horizon=3
    )
    assert directional["passed"].eq(1).all()

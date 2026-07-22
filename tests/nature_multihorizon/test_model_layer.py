from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from aspr.nature_multihorizon.contracts import (
    AUXILIARY_FEATURES,
    CORE_FEATURES,
    FeatureSpec,
    SplitSpec,
)
from aspr.nature_multihorizon.evaluation import (
    conditional_spearman,
    evaluate_oof_predictions,
    run_nested_oof,
)
from aspr.nature_multihorizon.models import (
    DomainYearCalibrator,
    FoldLocalFeatureTransformer,
    fit_candidate_model,
)
from aspr.nature_multihorizon.scoring import build_score_packets, packets_to_frame
from aspr.nature_multihorizon.splits import make_nested_folds


def synthetic_frame(n_rows: int = 240, seed: int = 17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {column: rng.normal(size=n_rows) for column in CORE_FEATURES + AUXILIARY_FEATURES}
    )
    frame["paper_id"] = [f"p{index:04d}" for index in range(n_rows)]
    frame["domain12"] = np.array(["life", "chemistry", "physics"])[np.arange(n_rows) % 3]
    frame["publication_year"] = 1980 + (np.arange(n_rows) % 41)
    frame["venue_family"] = np.array(["nature", "communications", "npj"])[np.arange(n_rows) % 3]
    frame["n_future_citers"] = rng.integers(10, 80, size=n_rows)
    frame["target"] = (
        0.8 * frame["delta_q0_shock"]
        + 0.3 * frame["rtd_simpson"]
        + rng.normal(scale=0.25, size=n_rows)
    )
    return frame


def test_nested_folds_are_deterministic_disjoint_and_complete() -> None:
    frame = synthetic_frame(250)

    left = make_nested_folds(frame, n_outer=5, n_inner=4, seed=23)
    right = make_nested_folds(frame, n_outer=5, n_inner=4, seed=23)

    assert left.assignments["outer_fold"].tolist() == right.assignments["outer_fold"].tolist()
    seen: list[int] = []
    for outer in left.outer_folds:
        assert set(outer.train_idx).isdisjoint(outer.test_idx)
        seen.extend(outer.test_idx.tolist())
        for inner in outer.inner_folds:
            assert set(inner.train_idx).isdisjoint(inner.test_idx)
            assert set(inner.train_idx).issubset(set(outer.train_idx))
            assert set(inner.test_idx).issubset(set(outer.train_idx))
    assert sorted(seen) == list(range(len(frame)))
    assert left.audit["all_rows_assigned_once"] is True


def test_fold_local_transformer_does_not_fit_on_test_extreme() -> None:
    train = synthetic_frame(80)
    test = synthetic_frame(3)
    test.loc[:, "delta_q0_shock"] = [1e9, 2e9, 3e9]

    transformer = FoldLocalFeatureTransformer(FeatureSpec()).fit(train)
    before = transformer.core_medians_["delta_q0_shock"]
    transformed = transformer.transform(test)

    assert transformer.core_medians_["delta_q0_shock"] == before
    assert transformed.core8[:, 0].tolist() == [1.0, 1.0, 1.0]
    assert transformed.full18.shape == (3, 18)
    assert transformed.mechanism5.shape == (3, 5)


def test_simplex_is_nonnegative_and_learns_rank_signal() -> None:
    frame = synthetic_frame(180)
    model = fit_candidate_model("mechanism5_simplex", frame, "target", seed=31)
    predictions = model.predict(frame)

    assert np.all(model.estimator.weights_ >= 0)
    assert abs(float(model.estimator.weights_.sum()) - 1.0) < 1e-10
    assert pd.Series(predictions).corr(frame["target"], method="spearman") > 0.75


def test_domain_year_calibration_is_train_only_and_handles_unknown_domain() -> None:
    frame = synthetic_frame(120)
    raw = frame["delta_q0_shock"].to_numpy()
    calibrator = DomainYearCalibrator().fit(frame, raw, frame["target"])
    unseen = frame.iloc[:3].copy()
    unseen["domain12"] = "unseen-domain"

    result = calibrator.predict(unseen, raw[:3])

    assert result.shape == (3,)
    assert np.isfinite(result).all()


def test_conditional_spearman_removes_domain_level_ordering() -> None:
    frame = pd.DataFrame(
        {
            "domain12": np.repeat(["a", "b"], 30),
            "publication_year": 2000,
            "score": np.repeat([0.0, 1.0], 30),
            "target": np.concatenate([np.arange(30), 100 + np.arange(30)]),
        }
    )

    rho, n_rows, n_cells = conditional_spearman(
        frame,
        "score",
        "target",
        min_cell_size=30,
    )

    assert np.isnan(rho)  # scores have no within-domain variation
    assert n_rows == 60
    assert n_cells == 2


def test_nested_oof_produces_normalized_tables_and_holdout() -> None:
    frame = synthetic_frame(240)
    split_spec = SplitSpec(
        outer_folds=3,
        inner_folds=2,
        min_conditional_cell_size=8,
        min_domain_oof_size=15,
        bootstrap_iterations=100,
        seed=41,
    )

    result = run_nested_oof(
        frame,
        horizon=5,
        target_col="target",
        split_spec=split_spec,
        candidate_ids=("mechanism5_simplex", "gam18", "hgb18", "rank_blend"),
        bootstrap_iterations=20,
    )

    assert not result.oof_predictions.duplicated(["paper_id", "model_id"]).any()
    assert result.oof_predictions["prediction_raw"].notna().all()
    assert result.oof_predictions.groupby("paper_id")["is_selected"].sum().eq(1).all()
    assert result.oof_predictions.filter(like="mechanism__").shape[1] == 5
    assert {
        "development_oof",
        "development_oof_all_models",
        "upgrade_gate",
        "sealed_temporal_holdout",
    }.issubset(set(result.evaluation_metrics["scope"]))
    metric = result.evaluation_metrics.loc[
        result.evaluation_metrics["scope"].eq("development_oof")
        & result.evaluation_metrics["metric"].eq("rho_global_uncalibrated"),
        "value",
    ].iloc[0]
    assert metric > 0.5
    assert len(result.model_ledger) > 0
    assert len(result.holdout_predictions) > 0


def test_score_packet_adapter_round_trips_dual_scores() -> None:
    table = pd.DataFrame(
        {
            "paper_id": ["p1"],
            "horizon": [5],
            "mechanism__boundary_perturbation": [0.8],
            "mechanism__community_diffusion": [0.6],
            "mechanism__interdisciplinarity": [0.5],
            "mechanism__knowledge_recombination": [0.4],
            "mechanism__knowledge_brokerage": [0.3],
            "score_mechanism": [0.7],
            "score_performance_raw": [0.4],
            "score_performance_calibrated": [0.5],
            "score_performance_percentile": [0.9],
            "quality_flags": ["low_pair_count;cap_hit"],
        }
    )

    packets = build_score_packets(table, model_version="mh-v1")
    restored = packets_to_frame(packets)

    assert packets[0].quality_flags == ("low_pair_count", "cap_hit")
    assert packets[0].score_performance_percentile == 0.9
    assert restored.loc[0, "mechanism__boundary_perturbation"] == 0.8


def test_metric_sensitivity_rows_have_distinct_keys() -> None:
    frame = synthetic_frame(180)
    result = run_nested_oof(
        frame,
        horizon=5,
        target_col="target",
        split_spec=SplitSpec(
            outer_folds=3,
            inner_folds=2,
            min_conditional_cell_size=8,
            min_domain_oof_size=15,
            bootstrap_iterations=100,
            seed=73,
        ),
        candidate_ids=("mechanism5_simplex", "gam18", "hgb18", "rank_blend"),
        bootstrap_iterations=10,
        run_holdout=False,
    )
    metrics = evaluate_oof_predictions(
        result.oof_predictions,
        bootstrap_iterations=10,
        seed=73,
    )
    key = ["horizon", "model_id", "scope", "metric", "sensitivity"]
    assert not metrics.duplicated(key).any()
    threshold_rows = metrics[
        metrics["scope"].eq("sensitivity_citation_threshold")
        & metrics["metric"].eq("rho_global_calibrated")
    ]
    assert set(threshold_rows["sensitivity"]) == {
        "future_citers_ge_10",
        "future_citers_ge_20",
        "future_citers_ge_50",
    }


class ModelLayerTests(unittest.TestCase):
    def test_nested_folds(self) -> None:
        test_nested_folds_are_deterministic_disjoint_and_complete()

    def test_fold_local_transform(self) -> None:
        test_fold_local_transformer_does_not_fit_on_test_extreme()

    def test_simplex(self) -> None:
        test_simplex_is_nonnegative_and_learns_rank_signal()

    def test_calibration(self) -> None:
        test_domain_year_calibration_is_train_only_and_handles_unknown_domain()

    def test_conditional_spearman(self) -> None:
        test_conditional_spearman_removes_domain_level_ordering()

    def test_nested_oof(self) -> None:
        test_nested_oof_produces_normalized_tables_and_holdout()

    def test_score_packet(self) -> None:
        test_score_packet_adapter_round_trips_dual_scores()

    def test_metric_sensitivity_keys(self) -> None:
        test_metric_sensitivity_rows_have_distinct_keys()


if __name__ == "__main__":
    unittest.main()

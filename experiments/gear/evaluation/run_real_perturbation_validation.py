"""Cross-fit HGB-P on real future-graph perturbation targets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from gear.nature_multihorizon.perturbation_targets import (
    build_perturbation_components,
)
from gear.nature_multihorizon.targets_v6 import (
    PERTURBATION_COMPONENTS,
    FoldLocalPerturbationTargetTransformer,
)

OOF_PATH = Path(
    "data/calibration/releases/gear-d5-primary16-current/oof_predictions.parquet"
)
FEATURE_PATH = Path(
    "data/calibration/releases/gear-d5-primary16-current/training_snapshot.parquet"
)
REGISTRY_PATH = Path(
    "data/calibration/releases/gear-d5-primary16-current/feature_registry.json"
)


def run_validation(
    raw_inputs_path: Path,
    output_dir: Path,
    *,
    seed: int = 20260828,
) -> dict[str, Any]:
    """Fit fold-local real and shuffled P heads using only frozen T0 features."""
    raw = pd.read_parquet(raw_inputs_path)
    components = build_perturbation_components(raw)
    frame = _join_features(raw, components)
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    feature_columns = registry["feature_names"]
    categorical_columns = registry["categorical_feature_names"]
    output, fold_reports = _cross_fit(
        frame,
        split_column="outer_fold_id",
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        seed=seed,
    )
    domain_output, domain_reports = _cross_fit(
        frame,
        split_column="domain12",
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        seed=seed + 100,
    )
    latest_temporal_split = _latest_temporal_split(frame)
    forward_temporal = output[
        output["heldout_outer_fold_id"].eq(latest_temporal_split)
    ].copy()
    temporal_block_cv_metrics = _metrics(output, seed)
    forward_temporal_metrics = _metrics(forward_temporal, seed)
    domain_metrics = _metrics(domain_output, seed + 100)
    temporal_hybrid = _temporal_hybrid(
        domain_output,
        forward_temporal,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "hgb_p_oof_predictions.parquet"
    domain_prediction_path = output_dir / "hgb_p_domain_oof_predictions.parquet"
    temporal_hybrid_path = (
        output_dir / "hgb_p_forward_temporal_hybrid_predictions.parquet"
    )
    output.to_parquet(prediction_path, index=False)
    domain_output.to_parquet(domain_prediction_path, index=False)
    temporal_hybrid.to_parquet(temporal_hybrid_path, index=False)
    supported = all(
        metrics["spearman_ci95_low"] > 0.0
        and metrics["spearman"] > metrics["shuffled_spearman"]
        for metrics in (forward_temporal_metrics, domain_metrics)
    )
    result = {
        "contract": "gear_real_hgb_p_validation_v2",
        "status": "supported" if supported else "not_supported",
        "claim_a_status": "supported" if supported else "not_supported",
        "claim_boundaries": {
            "temporal": "supported" if supported else "not_supported",
            "domain": "supported" if supported else "not_supported",
            "coverage": "not_claimed",
            "worst_group": "not_claimed",
        },
        "papers": len(output),
        "folds": len(fold_reports),
        "perturbation_components": list(PERTURBATION_COMPONENTS),
        "claim_adoption_head": "separate_future_context_label",
        "feature_columns": feature_columns,
        "uses_future_features": False,
        "forward_temporal_holdout_split": str(latest_temporal_split),
        "forward_temporal_holdout_metrics": forward_temporal_metrics,
        "blocked_temporal_cv_metrics_diagnostic_only": temporal_block_cv_metrics,
        "blocked_temporal_cv_uses_future_blocks_for_early_folds": True,
        "domain_holdout_metrics": domain_metrics,
        "temporal_fold_reports": fold_reports,
        "domain_fold_reports": domain_reports,
        "prediction_sha256": "sha256:"
        + hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
        "domain_prediction_sha256": "sha256:"
        + hashlib.sha256(domain_prediction_path.read_bytes()).hexdigest(),
        "forward_temporal_hybrid_prediction_sha256": "sha256:"
        + hashlib.sha256(temporal_hybrid_path.read_bytes()).hexdigest(),
        "forward_temporal_hybrid_training_rule": (
            "domain_oof_for_development_and_nonlatest_rows;"
            "prior-block-only_prediction_for_latest_temporal_rows"
        ),
    }
    (output_dir / "hgb_p_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def _latest_temporal_split(frame: pd.DataFrame) -> Any:
    means = frame.groupby("outer_fold_id", observed=True)["publication_year"].mean()
    if means.empty:
        raise ValueError("no temporal split is available")
    return means.idxmax()


def _temporal_hybrid(
    domain_output: pd.DataFrame,
    forward_temporal: pd.DataFrame,
) -> pd.DataFrame:
    replacement = forward_temporal.set_index("paper_id")
    output = domain_output.copy().set_index("paper_id")
    columns = [
        "domain12",
        "publication_year",
        "perturbation_target_fold",
        "perturbation_head_p",
        "shuffled_perturbation_head_p",
    ]
    output.loc[replacement.index, columns] = replacement[columns]
    output["prediction_protocol"] = "domain_oof_development"
    output.loc[replacement.index, "prediction_protocol"] = (
        "forward_temporal_latest_holdout"
    )
    return output.reset_index()


def _cross_fit(
    frame: pd.DataFrame,
    *,
    split_column: str,
    feature_columns: list[str],
    categorical_columns: list[str],
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    predictions: list[pd.DataFrame] = []
    reports: list[dict[str, Any]] = []
    for split in sorted(frame[split_column].dropna().unique(), key=str):
        train = frame[frame[split_column].ne(split)].copy()
        test = frame[frame[split_column].eq(split)].copy()
        predicted, report = _fit_fold(
            train, test, feature_columns, categorical_columns, seed
        )
        predicted[f"heldout_{split_column}"] = split
        predictions.append(predicted)
        reports.append({f"heldout_{split_column}": str(split), **report})
    return pd.concat(predictions, ignore_index=True), reports


def _join_features(raw: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    oof = pd.read_parquet(
        OOF_PATH,
        columns=["paper_id", "outer_fold_id", "domain12", "publication_year"],
    )
    features = pd.read_parquet(FEATURE_PATH)
    raw_targets = raw.drop(
        columns=["outer_fold_id", "domain12", "publication_year"], errors="ignore"
    )
    base = pd.concat(
        [raw_targets.reset_index(drop=True), components.reset_index(drop=True)], axis=1
    )
    base = base.loc[:, ~base.columns.duplicated()]
    required = [*PERTURBATION_COMPONENTS]
    base = base.dropna(subset=required)
    return (
        base.merge(oof, on="paper_id", how="inner", validate="one_to_one")
        .merge(
            features,
            on=["paper_id", "publication_year"],
            how="inner",
            validate="one_to_one",
        )
        .reset_index(drop=True)
    )


def _fit_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    transformer = FoldLocalPerturbationTargetTransformer().fit(train)
    train_target = transformer.transform(train)["perturbation_fold"]
    test_target = transformer.transform(test)["perturbation_fold"]
    train_x, test_x = _design(train, test, feature_columns, categorical_columns)
    valid = train_target.notna()
    if int(valid.sum()) < 20:
        raise ValueError("outer training fold has fewer than 20 perturbation targets")
    model = HistGradientBoostingRegressor(random_state=seed)
    model.fit(train_x.loc[valid], train_target.loc[valid])
    prediction = np.clip(model.predict(test_x), 0.0, 1.0)
    shuffled_target = _shuffle_target(train, train_target, seed)
    shuffled_model = HistGradientBoostingRegressor(random_state=seed + 1)
    shuffled_model.fit(train_x.loc[valid], shuffled_target.loc[valid])
    shuffled_prediction = np.clip(shuffled_model.predict(test_x), 0.0, 1.0)
    return (
        pd.DataFrame(
            {
                "paper_id": test["paper_id"].astype(str).to_numpy(),
                "domain12": test["domain12"].astype(str).to_numpy(),
                "publication_year": test["publication_year"].to_numpy(),
                "perturbation_target_fold": test_target.to_numpy(float),
                "perturbation_head_p": prediction,
                "shuffled_perturbation_head_p": shuffled_prediction,
            }
        ),
        {
            "train_rows": int(valid.sum()),
            "test_rows": int(test_target.notna().sum()),
            "target_fit_scope": "outer_training_fold_only",
        },
    )


def _design(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_columns = [
        column for column in feature_columns if column not in categorical_columns
    ]
    train_numeric = train[numeric_columns].apply(pd.to_numeric, errors="coerce")
    test_numeric = test[numeric_columns].apply(pd.to_numeric, errors="coerce")
    train_category = pd.get_dummies(train[categorical_columns].astype(str), dtype=float)
    test_category = pd.get_dummies(test[categorical_columns].astype(str), dtype=float)
    test_category = test_category.reindex(
        columns=train_category.columns, fill_value=0.0
    )
    train_x = pd.concat([train_numeric, train_category], axis=1)
    test_x = pd.concat([test_numeric, test_category], axis=1)
    train_x = train_x.replace([np.inf, -np.inf], np.nan)
    test_x = test_x.replace([np.inf, -np.inf], np.nan)
    medians = train_x.median().fillna(0.0)
    return train_x.fillna(medians), test_x.fillna(medians)


def _shuffle_target(train: pd.DataFrame, target: pd.Series, seed: int) -> pd.Series:
    generator = np.random.default_rng(seed)
    output = target.copy()
    year_bin = pd.to_numeric(train["publication_year"]) // 5 * 5
    groups = pd.DataFrame(
        {"domain12": train["domain12"].astype(str), "year_bin": year_bin},
        index=train.index,
    )
    for indexes in groups.groupby(["domain12", "year_bin"]).groups.values():
        indexes = list(indexes)
        output.loc[indexes] = generator.permutation(target.loc[indexes].to_numpy())
    if output.equals(target):
        output.loc[:] = generator.permutation(target.to_numpy())
    return output


def _metrics(frame: pd.DataFrame, seed: int) -> dict[str, float]:
    valid = frame.dropna(subset=["perturbation_target_fold"])
    rho = _spearman(valid, "perturbation_head_p")
    shuffled = _spearman(valid, "shuffled_perturbation_head_p")
    generator = np.random.default_rng(seed)
    bootstrap: list[float] = []
    for _ in range(1000):
        sampled = valid.iloc[generator.integers(0, len(valid), len(valid))]
        value = _spearman(sampled, "perturbation_head_p")
        if np.isfinite(value):
            bootstrap.append(value)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "spearman": rho,
        "shuffled_spearman": shuffled,
        "real_minus_shuffled": rho - shuffled,
        "spearman_ci95_low": float(low),
        "spearman_ci95_high": float(high),
    }


def _spearman(frame: pd.DataFrame, column: str) -> float:
    return float(
        frame[column].corr(frame["perturbation_target_fold"], method="spearman")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()
    result = run_validation(args.raw_inputs, args.output_dir, seed=args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_validation"]

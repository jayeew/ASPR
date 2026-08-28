"""Cross-fitted U/D/P/R head training using only explicit T0 features."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

FORBIDDEN_FEATURE_TOKENS = (
    "future",
    "citer",
    "target",
    "rgpm",
    "excess_diffusion",
    "perturbation",
    "outcome",
)
TARGET_COLUMNS = {
    "uptake": "future_uptake",
    "diffusion": "excess_diffusion_fold",
    "perturbation": "perturbation_fold",
}


def run_cross_fitted_structural_heads(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    fold_column: str = "outer_fold_id",
    paper_column: str = "paper_id",
    joint_lambda: float = 0.5,
    seed: int = 20260827,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate leakage-safe OOF predictions for the registered U/D/P/R heads."""
    _validate_inputs(frame, feature_columns, fold_column, paper_column, joint_lambda)
    output: list[pd.DataFrame] = []
    fold_reports: list[dict[str, Any]] = []
    for fold in sorted(frame[fold_column].dropna().unique(), key=str):
        train = frame[frame[fold_column].ne(fold)].copy()
        test = frame[frame[fold_column].eq(fold)].copy()
        predictions, report = _fit_fold(
            train,
            test,
            feature_columns=feature_columns,
            seed=seed,
            joint_lambda=joint_lambda,
        )
        predictions[fold_column] = fold
        output.append(predictions)
        fold_reports.append({"outer_fold_id": str(fold), **report})
    combined = pd.concat(output, ignore_index=True)
    manifest = {
        "contract": "gear_structural_heads_oof_v1",
        "status": "estimated",
        "rows": len(combined),
        "folds": len(fold_reports),
        "feature_columns": list(feature_columns),
        "feature_registry_sha256": _digest(list(feature_columns)),
        "joint_lambda": joint_lambda,
        "fold_reports": fold_reports,
    }
    return combined, manifest


def _fit_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    seed: int,
    joint_lambda: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    x_train = _numeric_matrix(train, feature_columns)
    x_test = _numeric_matrix(test, feature_columns)
    medians = x_train.median(axis=0).fillna(0.0)
    x_train = x_train.fillna(medians)
    x_test = x_test.fillna(medians)
    uptake = pd.to_numeric(train[TARGET_COLUMNS["uptake"]], errors="coerce")
    if uptake.dropna().nunique() < 2:
        raise ValueError("each outer-training fold requires both uptake classes")
    uptake_model = HistGradientBoostingClassifier(random_state=seed)
    uptake_model.fit(
        x_train.loc[uptake.notna()], uptake.loc[uptake.notna()].astype(int)
    )
    p_uptake = uptake_model.predict_proba(x_test)[:, 1]
    excess = _fit_regression_head(
        x_train, train[TARGET_COLUMNS["diffusion"]], x_test, seed=seed
    )
    perturbation = _fit_regression_head(
        x_train, train[TARGET_COLUMNS["perturbation"]], x_test, seed=seed + 1
    )
    coverage = test[list(feature_columns)].notna().mean(axis=1).to_numpy(float)
    support = _training_support(x_train, x_test)
    reliability = np.clip(coverage * support, 0.0, 1.0)
    joint = p_uptake * (joint_lambda * excess + (1.0 - joint_lambda) * perturbation)
    prediction = pd.DataFrame(
        {
            "paper_id": test["paper_id"].astype(str).to_numpy(),
            "uptake_probability_head_u": np.clip(p_uptake, 0.0, 1.0),
            "excess_diffusion_head_d": np.clip(excess, 0.0, 1.0),
            "perturbation_head_p": np.clip(perturbation, 0.0, 1.0),
            "reliability_head_r": reliability,
            "aspr_joint": np.clip(joint, 0.0, 1.0),
        }
    )
    return prediction, {
        "train_rows": len(train),
        "test_rows": len(test),
        "mean_reliability": float(reliability.mean()),
    }


def _fit_regression_head(
    x_train: pd.DataFrame,
    target: pd.Series,
    x_test: pd.DataFrame,
    *,
    seed: int,
) -> np.ndarray:
    values = pd.to_numeric(target, errors="coerce")
    valid = values.notna()
    if int(valid.sum()) < 10:
        raise ValueError("each regression head requires at least ten training rows")
    model = HistGradientBoostingRegressor(random_state=seed)
    model.fit(x_train.loc[valid], values.loc[valid])
    return model.predict(x_test)


def _training_support(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    lower = train.quantile(0.01)
    upper = train.quantile(0.99)
    within = test.ge(lower, axis=1) & test.le(upper, axis=1)
    return within.mean(axis=1).to_numpy(float)


def _numeric_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    return frame[list(columns)].apply(pd.to_numeric, errors="coerce")


def _validate_inputs(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    fold_column: str,
    paper_column: str,
    joint_lambda: float,
) -> None:
    if not 0.0 <= joint_lambda <= 1.0:
        raise ValueError("joint_lambda must be in [0, 1]")
    if not feature_columns:
        raise ValueError("at least one T0 feature is required")
    leaked = [
        column
        for column in feature_columns
        if any(token in column.casefold() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if leaked:
        raise ValueError(f"post-T0 target leakage in feature columns: {sorted(leaked)}")
    required = {
        fold_column,
        paper_column,
        *feature_columns,
        *TARGET_COLUMNS.values(),
    }
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"structural-head inputs are missing: {missing}")
    if frame[fold_column].nunique() < 2:
        raise ValueError("cross-fitting requires at least two outer folds")
    if frame[paper_column].duplicated().any():
        raise ValueError("structural-head input must be unique at paper grain")


def _digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


__all__ = ["run_cross_fitted_structural_heads"]

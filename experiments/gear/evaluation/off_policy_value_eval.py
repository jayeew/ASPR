"""Cross-fitted doubly robust value estimate for a selective action policy."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .graph_action_randomized_runner import ACTIONS


def doubly_robust_value(
    frame: pd.DataFrame, *, expected_n: int | None = None
) -> dict[str, Any]:
    scores = _doubly_robust_scores(frame)
    _require_expected_n(scores, expected_n)
    return _summary(scores)


def switch_doubly_robust_value(
    frame: pd.DataFrame,
    *,
    importance_threshold: float = 10.0,
    expected_n: int | None = None,
) -> dict[str, Any]:
    """SWITCH-DR clips unstable corrections by reverting to the outcome model."""
    if importance_threshold <= 0.0:
        raise ValueError("importance_threshold must be positive")
    scores, weight = _switch_scores(frame, importance_threshold=importance_threshold)
    _require_expected_n(scores, expected_n)
    result = _summary(scores)
    result.update(
        {
            "estimator": "switch_dr",
            "importance_threshold": importance_threshold,
            "switched_fraction": float((weight > importance_threshold).mean()),
        }
    )
    return result


def paired_switch_doubly_robust_contrast(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    importance_threshold: float = 10.0,
    expected_n: int | None = None,
) -> dict[str, Any]:
    """Estimate a paired SWITCH-DR sensitivity contrast on frozen rows."""
    if importance_threshold <= 0.0:
        raise ValueError("importance_threshold must be positive")
    candidate_aligned, reference_aligned = _align_policy_frames(candidate, reference)
    candidate_scores, _ = _switch_scores(
        candidate_aligned, importance_threshold=importance_threshold
    )
    reference_scores, _ = _switch_scores(
        reference_aligned, importance_threshold=importance_threshold
    )
    difference = candidate_scores - reference_scores
    _require_expected_n(difference, expected_n)
    result = _summary(difference)
    result.update(
        {
            "estimand": "candidate_minus_reference_paired_switch_dr_value",
            "candidate_value": float(candidate_scores.mean()),
            "reference_value": float(reference_scores.mean()),
            "importance_threshold": importance_threshold,
        }
    )
    return result


def paired_doubly_robust_contrast(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    expected_n: int | None = None,
) -> dict[str, Any]:
    """Estimate a paired DR contrast on the exact same randomized observations."""
    candidate_aligned, reference_aligned = _align_policy_frames(candidate, reference)
    candidate_scores = _doubly_robust_scores(candidate_aligned)
    reference_scores = _doubly_robust_scores(reference_aligned)
    difference = candidate_scores - reference_scores
    _require_expected_n(difference, expected_n)
    result = _summary(difference)
    result.update(
        {
            "estimand": "candidate_minus_reference_paired_doubly_robust_value",
            "candidate_value": float(candidate_scores.mean()),
            "reference_value": float(reference_scores.mean()),
            "target_action_disagreement_fraction": float(
                candidate_aligned["target_action"]
                .astype(str)
                .ne(reference_aligned["target_action"].astype(str))
                .mean()
            ),
        }
    )
    return result


def cross_fit_action_values(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    fold_column: str = "outer_fold_id",
    seed: int = 20260827,
) -> pd.DataFrame:
    """Estimate Q(x,a) out of fold for all A0-A5 logged actions."""
    required = {fold_column, "logged_action", "outcome", *feature_columns}
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"Q-model columns are missing: {missing}")
    output = frame.copy()
    for action in ACTIONS:
        output[f"q_{action}"] = np.nan
    for fold in sorted(frame[fold_column].dropna().unique(), key=str):
        train = frame[frame[fold_column].ne(fold)]
        test = frame[frame[fold_column].eq(fold)]
        x_train, x_test = _design(train, test, feature_columns)
        for action in ACTIONS:
            action_rows = train["logged_action"].eq(action)
            if int(action_rows.sum()) < 10:
                raise ValueError(f"Q-model action has fewer than ten rows: {action}")
            model = _q_model(seed)
            model.fit(
                x_train.loc[action_rows],
                pd.to_numeric(train.loc[action_rows, "outcome"], errors="coerce"),
            )
            output.loc[test.index, f"q_{action}"] = model.predict(x_test)
    output["q_logged"] = [
        output.at[index, f"q_{action}"]
        for index, action in output["logged_action"].items()
    ]
    return output


def fit_development_and_holdout_action_values(
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    *,
    feature_columns: list[str],
    fold_column: str = "outer_fold_id",
    seed: int = 20260827,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-fit development Q values and predict a never-fit confirmatory holdout."""
    scored_development = cross_fit_action_values(
        development,
        feature_columns=feature_columns,
        fold_column=fold_column,
        seed=seed,
    )
    required = {"logged_action", "outcome", *feature_columns}
    missing = sorted(required - set(holdout))
    if missing:
        raise ValueError(f"holdout Q-model columns are missing: {missing}")
    scored_holdout = holdout.copy()
    train_x, test_x = _design(development, holdout, feature_columns)
    for action in ACTIONS:
        action_rows = development["logged_action"].eq(action)
        if int(action_rows.sum()) < 10:
            raise ValueError(f"Q-model action has fewer than ten rows: {action}")
        model = _q_model(seed)
        model.fit(
            train_x.loc[action_rows],
            pd.to_numeric(development.loc[action_rows, "outcome"], errors="coerce"),
        )
        scored_holdout[f"q_{action}"] = model.predict(test_x)
    scored_holdout["q_logged"] = [
        scored_holdout.at[index, f"q_{action}"]
        for index, action in scored_holdout["logged_action"].items()
    ]
    return scored_development, scored_holdout


def _design(
    train: pd.DataFrame, test: pd.DataFrame, feature_columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_x = train[feature_columns].apply(pd.to_numeric, errors="coerce")
    test_x = test[feature_columns].apply(pd.to_numeric, errors="coerce")
    train_x = train_x.replace([np.inf, -np.inf], np.nan)
    test_x = test_x.replace([np.inf, -np.inf], np.nan)
    medians = train_x.median().fillna(0.0)
    return train_x.fillna(medians), test_x.fillna(medians)


def _q_model(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        random_state=seed,
        max_depth=2,
        min_samples_leaf=5,
        l2_regularization=1.0,
    )


def attach_target_policy_values(
    frame: pd.DataFrame, target_actions: pd.Series
) -> pd.DataFrame:
    """Attach target action and its cross-fitted Q estimate."""
    if not target_actions.index.equals(frame.index):
        raise ValueError("target action index must match the evaluation frame")
    if not target_actions.isin(ACTIONS).all():
        raise ValueError("target policy selected an unknown action")
    output = frame.copy()
    output["target_action"] = target_actions
    output["q_target"] = [
        output.at[index, f"q_{action}"] for index, action in target_actions.items()
    ]
    return output


def _off_policy_arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    required = {
        "outcome",
        "logged_action",
        "propensity",
        "target_action",
        "q_logged",
        "q_target",
    }
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"off-policy columns are missing: {missing}")
    values = {
        column: pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        for column in ("outcome", "propensity", "q_logged", "q_target")
    }
    if np.any(~np.isfinite(values["propensity"])) or np.any(
        values["propensity"] <= 0.0
    ):
        raise ValueError("propensities must be finite and positive")
    values["match"] = (
        frame["logged_action"].astype(str) == frame["target_action"].astype(str)
    ).to_numpy(float)
    if any(np.any(~np.isfinite(value)) for value in values.values()):
        raise ValueError("off-policy numeric columns must be finite")
    return values


def _doubly_robust_scores(frame: pd.DataFrame) -> np.ndarray:
    data = _off_policy_arrays(frame)
    return data["q_target"] + (
        data["match"] * (data["outcome"] - data["q_logged"]) / data["propensity"]
    )


def _switch_scores(
    frame: pd.DataFrame, *, importance_threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    data = _off_policy_arrays(frame)
    weight = data["match"] / data["propensity"]
    correction = np.where(
        weight <= importance_threshold,
        weight * (data["outcome"] - data["q_logged"]),
        0.0,
    )
    return data["q_target"] + correction, weight


def _align_policy_frames(
    candidate: pd.DataFrame, reference: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    identity = ["paper_id", "context_id"]
    missing = [
        f"{label}:{column}"
        for label, frame in (("candidate", candidate), ("reference", reference))
        for column in identity
        if column not in frame
    ]
    if missing:
        raise ValueError(f"paired policy identity columns are missing: {missing}")
    for label, frame in (("candidate", candidate), ("reference", reference)):
        if frame[identity].isna().any().any() or frame.duplicated(identity).any():
            raise ValueError(f"{label} policy identities must be non-null and unique")
    left = candidate.sort_values(identity).reset_index(drop=True)
    right = reference.sort_values(identity).reset_index(drop=True)
    if not left[identity].equals(right[identity]):
        raise ValueError("paired policies do not contain identical paper/context rows")
    for column in ("logged_action", "outcome", "propensity"):
        if column not in left or column not in right:
            raise ValueError(f"paired policy observed column is missing: {column}")
        if column == "logged_action":
            equal = left[column].astype(str).equals(right[column].astype(str))
        else:
            left_values = pd.to_numeric(left[column], errors="coerce").to_numpy(float)
            right_values = pd.to_numeric(right[column], errors="coerce").to_numpy(float)
            equal = bool(
                np.isfinite(left_values).all()
                and np.isfinite(right_values).all()
                and np.array_equal(left_values, right_values)
            )
        if not equal:
            raise ValueError(f"paired policy observed column changed: {column}")
    return left, right


def _require_expected_n(scores: np.ndarray, expected_n: int | None) -> None:
    if expected_n is not None and len(scores) != expected_n:
        raise ValueError(f"off-policy row count changed: {len(scores)} != {expected_n}")


def _summary(scores: np.ndarray) -> dict[str, Any]:
    if not len(scores) or np.any(~np.isfinite(scores)):
        raise ValueError("off-policy scores must be non-empty and finite")
    standard_error = (
        float(scores.std(ddof=1) / np.sqrt(len(scores))) if len(scores) > 1 else 0.0
    )
    value = float(scores.mean())
    return {
        "n": len(scores),
        "value": value,
        "standard_error": standard_error,
        "lcb_95": value - 1.96 * standard_error,
    }


__all__ = [
    "attach_target_policy_values",
    "cross_fit_action_values",
    "doubly_robust_value",
    "fit_development_and_holdout_action_values",
    "paired_doubly_robust_contrast",
    "paired_switch_doubly_robust_contrast",
    "switch_doubly_robust_value",
]

"""Cross-fitted claim-adoption head trained from aligned future citing contexts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .structural_head_training import FORBIDDEN_FEATURE_TOKENS


def run_claim_attribution_oof(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    fold_column: str = "outer_fold_id",
    seed: int = 20260827,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Predict claim adoption breadth, then normalize weights within each paper."""
    required = {
        "paper_id",
        "claim_id",
        "future_claim_adoption_breadth",
        fold_column,
        *feature_columns,
    }
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"claim-attribution training columns are missing: {missing}")
    leaked = [
        column
        for column in feature_columns
        if any(token in column.casefold() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if leaked:
        raise ValueError(f"post-T0 claim feature leakage: {sorted(leaked)}")
    rows: list[pd.DataFrame] = []
    for fold in sorted(frame[fold_column].dropna().unique(), key=str):
        train = frame[frame[fold_column].ne(fold)]
        test = frame[frame[fold_column].eq(fold)]
        x_train = train[list(feature_columns)].apply(pd.to_numeric, errors="coerce")
        x_test = test[list(feature_columns)].apply(pd.to_numeric, errors="coerce")
        medians = x_train.median().fillna(0.0)
        x_train = x_train.fillna(medians)
        x_test = x_test.fillna(medians)
        target = pd.to_numeric(train["future_claim_adoption_breadth"], errors="coerce")
        valid = target.notna()
        if int(valid.sum()) < 10:
            raise ValueError("claim-attribution fold has fewer than ten labels")
        model = HistGradientBoostingRegressor(random_state=seed)
        model.fit(x_train.loc[valid], target.loc[valid])
        scored = test[["paper_id", "claim_id", fold_column]].copy()
        scored["adoption_score"] = np.maximum(0.0, model.predict(x_test))
        rows.append(scored)
    output = pd.concat(rows, ignore_index=True)
    denominator = output.groupby("paper_id")["adoption_score"].transform("sum")
    claim_count = output.groupby("paper_id")["claim_id"].transform("size")
    output["attribution_weight"] = np.where(
        denominator > 0.0,
        output["adoption_score"] / denominator,
        1.0 / claim_count,
    )
    return output, {
        "contract": "gear_claim_attribution_oof_v1",
        "status": "estimated",
        "rows": len(output),
        "papers": int(output["paper_id"].nunique()),
        "folds": int(output[fold_column].nunique()),
        "future_contexts_used_at_inference": False,
    }


def fit_claim_attribution_development_holdout(
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    fold_column: str = "outer_fold_id",
    seed: int = 20260827,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Cross-fit development and score a holdout whose labels never enter fitting."""
    development_oof, report = run_claim_attribution_oof(
        development,
        feature_columns=feature_columns,
        fold_column=fold_column,
        seed=seed,
    )
    x_train = development[list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    x_holdout = holdout[list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    medians = x_train.median().fillna(0.0)
    target = pd.to_numeric(
        development["future_claim_adoption_breadth"], errors="coerce"
    )
    valid = target.notna()
    if int(valid.sum()) < 10:
        raise ValueError("claim-attribution development has fewer than ten labels")
    model = HistGradientBoostingRegressor(random_state=seed)
    model.fit(x_train.loc[valid].fillna(medians), target.loc[valid])
    scored = holdout[["paper_id", "claim_id"]].copy()
    scored["adoption_score"] = np.maximum(0.0, model.predict(x_holdout.fillna(medians)))
    denominator = scored.groupby("paper_id")["adoption_score"].transform("sum")
    claim_count = scored.groupby("paper_id")["claim_id"].transform("size")
    scored["attribution_weight"] = np.where(
        denominator > 0.0,
        scored["adoption_score"] / denominator,
        1.0 / claim_count,
    )
    report = {
        **report,
        "development_rows": len(development),
        "confirmatory_holdout_rows": len(holdout),
        "holdout_labels_used_for_training": False,
    }
    return development_oof, scored, report


__all__ = [
    "fit_claim_attribution_development_holdout",
    "run_claim_attribution_oof",
]

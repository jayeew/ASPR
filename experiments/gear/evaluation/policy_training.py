"""Freeze conservative action-promotion rules on development observations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .graph_action_randomized_runner import ACTIONS


def fit_action_promotion_rules(
    development: pd.DataFrame,
    *,
    z_value: float = 1.96,
) -> dict[str, dict[str, Any]]:
    """Estimate per-action uncertainty margins and observed safety guardrails."""
    required = {
        "logged_action",
        "outcome",
        "wrong_correction",
        "unsupported_claim",
        "realized_cost",
        *{f"q_{action}" for action in ACTIONS},
    }
    missing = sorted(required - set(development))
    if missing:
        raise ValueError(f"policy-development columns are missing: {missing}")
    baseline = development["logged_action"].eq("baseline")
    if int(baseline.sum()) < 10:
        raise ValueError("policy development requires at least ten baseline rows")
    baseline_residual = _residuals(development.loc[baseline], "baseline")
    baseline_outcome = pd.to_numeric(development.loc[baseline, "outcome"])
    rules: dict[str, dict[str, Any]] = {}
    for action in ACTIONS[1:]:
        selected = development["logged_action"].eq(action)
        if int(selected.sum()) < 10:
            raise ValueError(f"policy development has fewer than ten rows: {action}")
        action_residual = _residuals(development.loc[selected], action)
        action_outcome = pd.to_numeric(development.loc[selected, "outcome"])
        standard_error = float(
            np.sqrt(
                action_residual.var(ddof=1) / len(action_residual)
                + baseline_residual.var(ddof=1) / len(baseline_residual)
            )
        )
        guardrails = _action_guardrails(
            development, selected=selected, baseline=baseline
        )
        average_uplift = float(action_outcome.mean() - baseline_outcome.mean())
        average_uplift_se = float(
            np.sqrt(
                action_outcome.var(ddof=1) / len(action_outcome)
                + baseline_outcome.var(ddof=1) / len(baseline_outcome)
            )
        )
        average_uplift_lcb = average_uplift - z_value * average_uplift_se
        development_positive = average_uplift_lcb > 0.0
        rules[action] = {
            "uplift_margin": z_value * standard_error,
            **guardrails,
            "development_average_uplift": average_uplift,
            "development_average_uplift_lcb": average_uplift_lcb,
            "development_positive_uplift_pass": development_positive,
            "guardrails_pass": all(guardrails.values()) and development_positive,
            "development_rows": int(selected.sum()),
        }
    return rules


def apply_selective_policy(
    frame: pd.DataFrame,
    rules: dict[str, dict[str, Any]],
) -> pd.Series:
    """Select only actions whose predicted uplift clears the frozen margin."""
    required = {f"q_{action}" for action in ACTIONS}
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"policy application columns are missing: {missing}")
    unknown = sorted(set(rules) - set(ACTIONS[1:]))
    if unknown:
        raise ValueError(f"policy rules contain unknown actions: {unknown}")
    output: list[str] = []
    for _, row in frame.iterrows():
        candidates: list[tuple[float, float, str]] = []
        baseline = float(row["q_baseline"])
        for action, rule in rules.items():
            uplift = float(row[f"q_{action}"]) - baseline
            conservative_uplift = uplift - float(rule["uplift_margin"])
            if bool(rule["guardrails_pass"]) and conservative_uplift > 0.0:
                candidates.append((conservative_uplift, uplift, action))
        output.append(max(candidates)[2] if candidates else "baseline")
    return pd.Series(output, index=frame.index, name="target_action")


def _residuals(frame: pd.DataFrame, action: str) -> pd.Series:
    return pd.to_numeric(frame["outcome"]) - pd.to_numeric(frame[f"q_{action}"])


def _action_guardrails(
    frame: pd.DataFrame, *, selected: pd.Series, baseline: pd.Series
) -> dict[str, bool]:
    output: dict[str, bool] = {}
    names = {
        "wrong_correction": "wrong_correction_pass",
        "unsupported_claim": "unsupported_claim_pass",
        "realized_cost": "cost_pass",
    }
    for column, name in names.items():
        action_value = float(pd.to_numeric(frame.loc[selected, column]).mean())
        baseline_value = float(pd.to_numeric(frame.loc[baseline, column]).mean())
        output[name] = action_value <= baseline_value
    return output


__all__ = ["apply_selective_policy", "fit_action_promotion_rules"]

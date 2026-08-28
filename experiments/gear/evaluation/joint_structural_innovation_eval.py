"""Primary GEAR-only versus evidence-gated HGB integration estimand."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def evaluate_joint_value(
    frame: pd.DataFrame,
    *,
    outcome: str = "future_structural_outcome",
    evidence_only: str = "gear_evidence_score",
    joint: str = "joint_structural_score",
) -> dict[str, Any]:
    required = {outcome, evidence_only, joint}
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"joint evaluation columns are missing: {missing}")
    data = frame[list(required)].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < 3:
        raise ValueError("joint evaluation requires at least three complete rows")
    evidence_value = float(data[outcome].corr(data[evidence_only], method="spearman"))
    joint_value = float(data[outcome].corr(data[joint], method="spearman"))
    return {
        "n": len(data),
        "gear_only_spearman": evidence_value,
        "joint_spearman": joint_value,
        "integration_value": joint_value - evidence_value,
        "joint_top_decile_outcome": _top_fraction_mean(data, joint, outcome),
        "gear_top_decile_outcome": _top_fraction_mean(data, evidence_only, outcome),
    }


def _top_fraction_mean(frame: pd.DataFrame, score: str, outcome: str) -> float:
    count = max(1, int(np.ceil(len(frame) * 0.1)))
    return float(frame.nlargest(count, score)[outcome].mean())


__all__ = ["evaluate_joint_value"]

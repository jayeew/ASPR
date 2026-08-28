"""Audit whether an integration sample covers the Graph score range."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def audit_score_range(
    frame: pd.DataFrame, *, score_column: str = "graph_percentile"
) -> dict[str, Any]:
    if score_column not in frame:
        raise ValueError(f"missing score column: {score_column}")
    scores = pd.to_numeric(frame[score_column], errors="coerce").dropna()
    if scores.empty:
        raise ValueError("score range audit has no finite scores")
    deciles = np.clip(np.floor(scores.to_numpy(float) / 10.0), 0, 9).astype(int)
    covered = sorted(set(deciles.tolist()))
    return {
        "n": len(scores),
        "minimum": float(scores.min()),
        "maximum": float(scores.max()),
        "range": float(scores.max() - scores.min()),
        "covered_deciles": covered,
        "covered_decile_count": len(covered),
        "range_restricted": len(covered) < 8,
    }


__all__ = ["audit_score_range"]

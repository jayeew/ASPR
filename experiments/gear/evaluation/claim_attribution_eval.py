"""Claim-level Graph attribution diagnostics."""

from __future__ import annotations

from typing import Any

import pandas as pd


def evaluate_claim_attribution(frame: pd.DataFrame) -> dict[str, Any]:
    required = {"paper_id", "claim_id", "attribution_weight", "future_adoption"}
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"claim attribution columns are missing: {missing}")
    data = frame.copy()
    data["attribution_weight"] = pd.to_numeric(
        data["attribution_weight"], errors="coerce"
    )
    data["future_adoption"] = pd.to_numeric(data["future_adoption"], errors="coerce")
    sums = data.groupby("paper_id")["attribution_weight"].sum()
    adoption_range = data.groupby("paper_id")["future_adoption"].agg(
        lambda values: values.max() - values.min()
    )
    eligible_papers = set(adoption_range[adoption_range.gt(0.0)].index)
    ranking_data = data[data["paper_id"].isin(eligible_papers)]
    predicted = (
        ranking_data.sort_values(
            ["paper_id", "attribution_weight", "claim_id"],
            ascending=[True, False, True],
        )
        .groupby("paper_id")
        .first()
    )
    observed = (
        ranking_data.sort_values(
            ["paper_id", "future_adoption", "claim_id"],
            ascending=[True, False, True],
        )
        .groupby("paper_id")
        .first()
    )
    return {
        "papers": int(sums.size),
        "max_conservation_error": float((sums - 1.0).abs().max()),
        "top_claim_eligible_papers": len(eligible_papers),
        "top_claim_accuracy": (
            float((predicted["claim_id"] == observed["claim_id"]).mean())
            if eligible_papers
            else float("nan")
        ),
        "claim_spearman": float(
            data["attribution_weight"].corr(data["future_adoption"], method="spearman")
        ),
    }


__all__ = ["evaluate_claim_attribution"]

"""Frozen eight-dimension review-quality rubric aggregation."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

RUBRIC_DIMENSIONS = (
    "core_contribution_accuracy",
    "novelty_related_work",
    "methodological_critique",
    "results_interpretation",
    "evidence_based_critique",
    "completeness_coverage",
    "actionability",
    "false_or_contradictory_claims",
)


def aggregate_binary_rubric(
    rows: Iterable[Mapping[str, bool]],
) -> Dict[str, float | None]:
    materialized = list(rows)
    return {
        dimension: (
            sum(bool(row[dimension]) for row in materialized) / len(materialized)
            if materialized
            else None
        )
        for dimension in RUBRIC_DIMENSIONS
    }


__all__ = ["RUBRIC_DIMENSIONS", "aggregate_binary_rubric"]

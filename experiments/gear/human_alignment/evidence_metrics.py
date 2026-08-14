"""Point and evidence metrics with explicit empty-set handling."""

from __future__ import annotations

from typing import Dict, Sequence

from .contracts import PointMatch


def atomic_match_metrics(
    matches: Sequence[PointMatch],
    *,
    reference_count: int,
    candidate_count: int,
) -> Dict[str, object]:
    if reference_count == 0 and candidate_count == 0:
        return {
            "both_empty": True,
            "precision": None,
            "recall": None,
            "f1": None,
        }
    true_matches = sum(match.label == "SAME" for match in matches)
    precision = true_matches / candidate_count if candidate_count else 0.0
    recall = true_matches / reference_count if reference_count else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if recall is not None and precision + recall
        else 0.0 if recall is not None else None
    )
    return {
        "both_empty": False,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evidence_metrics(
    *,
    strong_first_claims: int,
    false_first_claims: int,
    expert_antecedents: int,
    retrieved_antecedents: int,
    cited_items: int,
    correct_citations: int,
    major_points: int,
    supported_major_points: int,
) -> Dict[str, object]:
    return {
        "false_first_rate": _ratio(false_first_claims, strong_first_claims),
        "antecedent_recall": _ratio(retrieved_antecedents, expert_antecedents),
        "citation_correctness": _ratio(correct_citations, cited_items),
        "evidence_supported_critique_rate": _ratio(
            supported_major_points, major_points
        ),
        "unsupported_major_rate": _ratio(
            major_points - supported_major_points, major_points
        ),
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


__all__ = ["atomic_match_metrics", "evidence_metrics"]

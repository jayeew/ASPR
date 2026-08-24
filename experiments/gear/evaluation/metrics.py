"""Deterministic metric calculations for unified GEAR evaluation."""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from statistics import mean
from typing import Any

from experiments.gear.review_reconstruction.evaluation import (
    MatchLabel,
    PointMatchDecision,
)
from gear.review_contracts import NoveltyJudgment, PointSeverity, StructuredReview

from .contracts import PointSupportDecisionV1, RevisionIssueLabel, RubricDecision

RUBRIC_TITLES = (
    "Core Contribution Accuracy",
    "Results Interpretation",
    "Comparative Analysis",
    "Evidence-Based Critique",
    "Critique Clarity",
    "Completeness Coverage",
    "Constructive Tone",
    "False or Contradictory Claims",
)
ANALYTICAL_TITLES = (
    "Core Contribution Accuracy",
    "Results Interpretation",
    "Comparative Analysis",
    "Evidence-Based Critique",
    "Completeness Coverage",
)


def rubric_metrics(scores: Sequence[RubricDecision]) -> dict[str, Any]:
    by_title = {row.title: row for row in scores}
    if set(by_title) != set(RUBRIC_TITLES):
        raise ValueError("rubric scores must contain the fixed eight dimensions")
    values: dict[str, Any] = {
        f"reviewbench_{_slug(title)}": by_title[title].score for title in RUBRIC_TITLES
    }
    values["reviewbench_total"] = sum(row.score for row in scores)
    values["analytical_quality"] = mean(
        by_title[title].score for title in ANALYTICAL_TITLES
    )
    values["unverifiable_rubric_count"] = sum(row.unverifiable for row in scores)
    return values


def evidence_support_metrics(
    review: StructuredReview,
    decisions: Sequence[PointSupportDecisionV1],
) -> dict[str, Any]:
    points = review.all_points()
    point_by_id = {point.point_id: point for point in points}
    decision_by_id = {row.point_id: row for row in decisions}
    if len(decision_by_id) != len(decisions) or set(decision_by_id) != set(point_by_id):
        raise ValueError("support decisions must cover every review point exactly")
    strict = sum(row.label == "SUPPORTED" for row in decisions)
    partial = sum(row.label == "PARTIALLY_SUPPORTED" for row in decisions)
    major_ids = {
        point.point_id
        for point in points
        if point.severity in {PointSeverity.MAJOR, PointSeverity.CRITICAL}
    }
    supported_major = sum(
        decision_by_id[point_id].label == "SUPPORTED" for point_id in major_ids
    )
    unsupported_major = sum(
        decision_by_id[point_id].label in {"UNSUPPORTED", "UNVERIFIABLE"}
        for point_id in major_ids
    )
    return {
        "point_count": len(points),
        "strict_support_precision": _ratio(strict, len(points)),
        "soft_support_precision": _ratio(strict + 0.5 * partial, len(points)),
        "major_point_count": len(major_ids),
        "major_support_precision": _ratio(supported_major, len(major_ids)),
        "unsupported_major_rate": _ratio(unsupported_major, len(major_ids)),
        "support_label_counts": dict(Counter(row.label for row in decisions)),
    }


def semantic_match_metrics(
    decisions: Sequence[PointMatchDecision],
    *,
    reference_count: int,
    candidate_count: int,
) -> dict[str, float | None]:
    strict = _greedy_match_count(decisions, {MatchLabel.SAME_POINT})
    soft = _greedy_match_count(
        decisions, {MatchLabel.SAME_POINT, MatchLabel.PARTIAL_POINT}
    )
    strict_p, strict_r, strict_f = _prf(strict, candidate_count, reference_count)
    soft_p, soft_r, soft_f = _prf(soft, candidate_count, reference_count)
    weighted = _greedy_match_weight(decisions)
    weighted_p, weighted_r, weighted_f = _weighted_prf(
        weighted, candidate_count, reference_count
    )
    return {
        "strict_precision": strict_p,
        "strict_recall": strict_r,
        "strict_f1": strict_f,
        "soft_precision": soft_p,
        "soft_recall": soft_r,
        "soft_f1": soft_f,
        "human_concern_coverage": soft_r,
        "weighted_alignment_precision": weighted_p,
        "weighted_alignment_recall": weighted_r,
        "weighted_alignment_f1": weighted_f,
    }


def concern_coverage_metrics(
    reference: StructuredReview,
    candidate: StructuredReview,
    decisions: Sequence[PointMatchDecision],
) -> dict[str, float | None]:
    """Report semantic coverage at issue-family and major-concern granularity."""
    references = _concern_context(reference)
    candidates = _concern_context(candidate)
    relevant_decisions = [
        row
        for row in decisions
        if row.reference_point_id in references and row.candidate_point_id in candidates
    ]
    matches = _greedy_matches(
        relevant_decisions, {MatchLabel.SAME_POINT, MatchLabel.PARTIAL_POINT}
    )
    matched_references = {row.reference_point_id for row in matches}
    families = {
        (section, point.aspect.value)
        for point_id, (section, point) in references.items()
    }
    matched_families = {
        (references[point_id][0], references[point_id][1].aspect.value)
        for point_id in matched_references
        if point_id in references
    }
    major_ids = {
        point_id
        for point_id, (_, point) in references.items()
        if point.severity in {PointSeverity.MAJOR, PointSeverity.CRITICAL}
    }
    weighted = _greedy_match_weight(relevant_decisions)
    weighted_p, weighted_r, weighted_f = _weighted_prf(
        weighted, len(candidates), len(references)
    )
    return {
        "human_concern_coverage": _ratio(len(matched_references), len(references)),
        "weighted_alignment_precision": weighted_p,
        "weighted_alignment_recall": weighted_r,
        "weighted_alignment_f1": weighted_f,
        "issue_family_coverage": _ratio(len(matched_families), len(families)),
        "major_human_concern_coverage": _ratio(
            len(major_ids & matched_references), len(major_ids)
        ),
    }


def novelty_direction_metrics(
    references: Sequence[NoveltyJudgment],
    candidates: Sequence[NoveltyJudgment],
) -> dict[str, Any]:
    if len(references) != len(candidates):
        raise ValueError("novelty direction arrays must be aligned")
    pairs = list(zip(references, candidates))
    accuracy = _ratio(sum(left == right for left, right in pairs), len(pairs))
    f1_by_label = {label.value: _label_f1(pairs, label) for label in NoveltyJudgment}
    observed = {left for left, _ in pairs}
    axis = {
        NoveltyJudgment.NEGATIVE: -1,
        NoveltyJudgment.MIXED: 0,
        NoveltyJudgment.POSITIVE: 1,
    }
    ordered = [
        (axis[left], axis[right])
        for left, right in pairs
        if left in axis and right in axis
    ]
    agreement = [
        1.0 if left == right else 0.5 if abs(left - right) == 1 else 0.0
        for left, right in ordered
    ]
    return {
        "judgment_accuracy": accuracy,
        "judgment_macro_f1": mean(f1_by_label.values()) if f1_by_label else None,
        "observed_label_novelty_macro_f1": (
            mean(f1_by_label[label.value] for label in observed) if observed else None
        ),
        "novelty_direction_agreement": mean(agreement) if agreement else None,
        "judgment_f1_by_label": f1_by_label,
        "judgment_confusion": dict(
            Counter(f"{left.value}->{right.value}" for left, right in pairs)
        ),
        "directional_coverage": _ratio(len(ordered), len(pairs)),
        "positive_shift_rate": _ratio(
            sum(right > left for left, right in ordered), len(ordered)
        ),
        "negative_shift_rate": _ratio(
            sum(right < left for left, right in ordered), len(ordered)
        ),
    }


def engagement_metrics(
    review: StructuredReview,
    relation_payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    novelty_points = [
        *review.novelty.supporting_points,
        *review.novelty.limiting_points,
        *review.novelty.uncertain_points,
    ]
    relation_keys = {
        key
        for point in novelty_points
        for key in point.evidence_keys
        if key.startswith("R:") and key in relation_payloads
    }
    work_ids = {
        str(relation_payloads[key].get("prior_work_id"))
        for key in relation_keys
        if relation_payloads[key].get("prior_work_id")
    }
    dimensions = {
        str(value)
        for key in relation_keys
        for value in relation_payloads[key].get("difference_dimensions", [])
        if str(value).strip()
    }
    return {
        "independent_prior_work_count": len(work_ids),
        "prior_work_engagement": (
            "none" if not work_ids else "limited" if len(work_ids) <= 2 else "extensive"
        ),
        "difference_dimension_count": len(dimensions),
        "analysis_depth": (
            "surface"
            if not dimensions
            else "moderate" if len(dimensions) <= 2 else "deep"
        ),
        "novelty_external_evidence_rate": _ratio(
            sum(
                any(key.startswith(("R:", "COV:")) for key in point.evidence_keys)
                for point in novelty_points
            ),
            len(novelty_points),
        ),
    }


def revision_metrics(
    labels: Sequence[RevisionIssueLabel],
    matched_issue_ids: set[str],
) -> dict[str, Any]:
    by_status: dict[str, list[RevisionIssueLabel]] = {
        status: [row for row in labels if row.status == status]
        for status in ("persists", "partially_resolved", "resolved", "unverifiable")
    }
    recalls = {
        status: _ratio(
            sum(row.issue_id in matched_issue_ids for row in rows), len(rows)
        )
        for status, rows in by_status.items()
    }
    major = [
        row
        for row in labels
        if row.status in {"persists", "partially_resolved"}
        and row.severity in {PointSeverity.MAJOR, PointSeverity.CRITICAL}
    ]
    measurable = [
        value
        for key, value in recalls.items()
        if key != "unverifiable" and value is not None
    ]
    return {
        "persistent_concern_recall": recalls["persists"],
        "partially_resolved_concern_recall": recalls["partially_resolved"],
        "resolved_issue_resurrection_rate": recalls["resolved"],
        "revision_status_macro_recall": mean(measurable) if measurable else None,
        "unresolved_major_recall": _ratio(
            sum(row.issue_id in matched_issue_ids for row in major), len(major)
        ),
        "unverifiable_issue_count": len(by_status["unverifiable"]),
    }


def retrieval_ranking_metrics(
    ranked_ids: Sequence[str], gold_ids: set[str], cutoffs: Sequence[int] = (5, 10, 20)
) -> dict[str, Any]:
    if not gold_ids:
        raise ValueError("prior-art ranking metrics require non-empty gold IDs")
    result: dict[str, Any] = {}
    for cutoff in cutoffs:
        observed = list(ranked_ids[:cutoff])
        result[f"recall_at_{cutoff}"] = len(set(observed) & gold_ids) / len(gold_ids)
    ranks = [index + 1 for index, value in enumerate(ranked_ids) if value in gold_ids]
    result["mrr"] = 1.0 / min(ranks) if ranks else 0.0
    for cutoff in (10, 20):
        gains = [1.0 if value in gold_ids else 0.0 for value in ranked_ids[:cutoff]]
        dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
        ideal = sum(
            1.0 / math.log2(index + 2) for index in range(min(len(gold_ids), cutoff))
        )
        result[f"ndcg_at_{cutoff}"] = dcg / ideal if ideal else None
    return result


def bootstrap_ci(
    values: Sequence[float | None], *, samples: int, seed: int
) -> tuple[float, float] | None:
    clean = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    if not clean:
        return None
    rng = random.Random(seed)
    estimates = sorted(mean(rng.choice(clean) for _ in clean) for _ in range(samples))
    return (
        estimates[int(0.025 * (len(estimates) - 1))],
        estimates[int(0.975 * (len(estimates) - 1))],
    )


def macro(values: Sequence[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return mean(clean) if clean else None


def _label_f1(
    pairs: Sequence[tuple[NoveltyJudgment, NoveltyJudgment]],
    label: NoveltyJudgment,
) -> float:
    tp = sum(left == label and right == label for left, right in pairs)
    predicted = sum(right == label for _, right in pairs)
    reference = sum(left == label for left, _ in pairs)
    precision = tp / predicted if predicted else 0.0
    recall = tp / reference if reference else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _ratio(value: float, total: int) -> float | None:
    return value / total if total else None


def _prf(
    matches: int, candidates: int, references: int
) -> tuple[float | None, float | None, float | None]:
    precision = _ratio(matches, candidates)
    recall = _ratio(matches, references)
    if precision is None or recall is None:
        return precision, recall, None
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _greedy_match_count(
    decisions: Sequence[PointMatchDecision], labels: set[MatchLabel]
) -> int:
    used_left: set[str] = set()
    used_right: set[str] = set()
    count = 0
    ranked = sorted(decisions, key=lambda row: row.confidence, reverse=True)
    for row in ranked:
        if row.label not in labels:
            continue
        if row.reference_point_id in used_left or row.candidate_point_id in used_right:
            continue
        used_left.add(row.reference_point_id)
        used_right.add(row.candidate_point_id)
        count += 1
    return count


def _greedy_matches(
    decisions: Sequence[PointMatchDecision], labels: set[MatchLabel]
) -> list[PointMatchDecision]:
    used_left: set[str] = set()
    used_right: set[str] = set()
    output: list[PointMatchDecision] = []
    for row in sorted(decisions, key=lambda value: value.confidence, reverse=True):
        if row.label not in labels:
            continue
        if row.reference_point_id in used_left or row.candidate_point_id in used_right:
            continue
        used_left.add(row.reference_point_id)
        used_right.add(row.candidate_point_id)
        output.append(row)
    return output


def _greedy_match_weight(decisions: Sequence[PointMatchDecision]) -> float:
    return sum(
        1.0 if row.label == MatchLabel.SAME_POINT else 0.5
        for row in _greedy_matches(
            decisions, {MatchLabel.SAME_POINT, MatchLabel.PARTIAL_POINT}
        )
    )


def _weighted_prf(
    weight: float, candidates: int, references: int
) -> tuple[float | None, float | None, float | None]:
    precision = _ratio(weight, candidates)
    recall = _ratio(weight, references)
    if precision is None or recall is None:
        return precision, recall, None
    return (
        precision,
        recall,
        (2 * precision * recall / (precision + recall) if precision + recall else 0.0),
    )


def _point_context(review: StructuredReview) -> dict[str, tuple[str, Any]]:
    groups = {
        "novelty": [
            *review.novelty.supporting_points,
            *review.novelty.limiting_points,
            *review.novelty.uncertain_points,
        ],
        "strengths": review.strengths,
        "weaknesses": review.weaknesses,
        "questions": review.questions,
    }
    return {
        point.point_id: (section, point)
        for section, points in groups.items()
        for point in points
    }


def _concern_context(review: StructuredReview) -> dict[str, tuple[str, Any]]:
    groups = {
        "novelty": [
            *review.novelty.limiting_points,
            *review.novelty.uncertain_points,
        ],
        "weaknesses": review.weaknesses,
        "questions": review.questions,
    }
    return {
        point.point_id: (section, point)
        for section, points in groups.items()
        for point in points
    }


def _slug(value: str) -> str:
    return value.casefold().replace(" ", "_").replace("/", "_")


__all__ = [
    "ANALYTICAL_TITLES",
    "RUBRIC_TITLES",
    "bootstrap_ci",
    "engagement_metrics",
    "evidence_support_metrics",
    "macro",
    "novelty_direction_metrics",
    "retrieval_ranking_metrics",
    "revision_metrics",
    "rubric_metrics",
    "semantic_match_metrics",
]

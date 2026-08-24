"""Direct StructuredReview consistency metrics without a composite score."""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections.abc import Sequence
from enum import Enum
from statistics import mean
from typing import Literal

from pydantic import Field, model_validator

from gear.review_contracts import (
    NoveltyJudgment,
    PointSeverity,
    ReviewPoint,
    StructuredReview,
    ReviewModel,
)
from gear.review_verifier import GRAPH_SEMANTIC_TERMS


class MatchLabel(str, Enum):
    SAME_POINT = "SAME_POINT"
    PARTIAL_POINT = "PARTIAL_POINT"
    CONTRADICTORY = "CONTRADICTORY"
    NO_MATCH = "NO_MATCH"


class PointMatchDecision(ReviewModel):
    contract: Literal["point_match_decision"] = "point_match_decision"
    paper_id: str
    reference_point_id: str
    candidate_point_id: str
    label: MatchLabel
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    rationale: str = ""


class MatchJudgePackage(ReviewModel):
    contract: Literal["blind_match_judge_package"] = "blind_match_judge_package"
    task_id: str
    paper_id_hash: str
    reference_points: list[ReviewPoint]
    candidate_points: list[ReviewPoint]
    candidate_pairs: list[tuple[str, str]]
    blinded: bool = True
    instructions: str


class MatchJudgeResponse(ReviewModel):
    contract: Literal["blind_match_judge_response"] = (
        "blind_match_judge_response"
    )
    task_id: str
    model_id: str
    conversation_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    decisions: list[PointMatchDecision]

    @model_validator(mode="after")
    def unique_pairs(self) -> MatchJudgeResponse:
        pairs = [
            (row.reference_point_id, row.candidate_point_id)
            for row in self.decisions
        ]
        if len(pairs) != len(set(pairs)):
            raise ValueError("blind match decisions must contain unique pairs")
        return self


class PairMetrics(ReviewModel):
    contract: Literal["structured_review_pair_metrics"] = (
        "structured_review_pair_metrics"
    )
    paper_id: str
    development_non_confirmatory: bool = False
    atomic_precision: float
    atomic_recall: float
    atomic_f1: float
    major_weakness_question_recall: float
    section_coverage: dict[str, float]
    section_f1: dict[str, float]
    novelty_judgment_correct: bool
    novelty_point_precision: float
    novelty_point_recall: float
    novelty_point_f1: float
    valid_evidence_key_ratio: float
    semantic_support_precision: float | None = None
    unsupported_major_rate: float
    graph_semantic_violation_count: int
    partial_count: int = 0
    contradictory_count: int = 0


class CorpusMetrics(ReviewModel):
    contract: Literal["structured_review_corpus_metrics"] = (
        "structured_review_corpus_metrics"
    )
    paper_count: int
    paper_macro: dict[str, float]
    novelty_judgment_accuracy: float
    novelty_judgment_macro_f1: float
    bootstrap_95_ci: dict[str, tuple[float, float]]
    development_non_confirmatory: bool
    composite_score: None = None

    @model_validator(mode="after")
    def no_composite(self) -> CorpusMetrics:
        if self.composite_score is not None:
            raise ValueError("GEAR does not define a composite score")
        return self


def build_blind_match_package(
    reference: StructuredReview,
    candidate: StructuredReview,
    *,
    top_k: int = 5,
) -> MatchJudgePackage:
    if reference.paper_id != candidate.paper_id:
        raise ValueError("blind match package requires the same paper")
    reference_points = reference.all_points()
    candidate_points = candidate.all_points()
    pairs: list[tuple[str, str]] = []
    for ref in reference_points:
        ranked = sorted(
            candidate_points,
            key=lambda item: _lexical_similarity(ref.text, item.text),
            reverse=True,
        )[:top_k]
        pairs.extend((ref.point_id, item.point_id) for item in ranked)
    identity = f"{reference.paper_id}|{pairs}"
    return MatchJudgePackage(
        task_id="MATCH2-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:18],
        paper_id_hash="sha256:"
        + hashlib.sha256(reference.paper_id.encode("utf-8")).hexdigest(),
        reference_points=reference_points,
        candidate_points=candidate_points,
        candidate_pairs=list(dict.fromkeys(pairs)),
        instructions=(
            "Blindly label each candidate pair SAME_POINT, PARTIAL_POINT, "
            "CONTRADICTORY, or NO_MATCH. SAME_POINT requires the same atomic "
            "scientific proposition and direction; style overlap is insufficient. "
            "PARTIAL_POINT means the same actionable scientific concern with "
            "different granularity, boundary, or evidence scope; identical wording "
            "is not required."
        ),
    )


def validate_match_judge_response(
    package: MatchJudgePackage, response: MatchJudgeResponse
) -> None:
    if response.task_id != package.task_id:
        raise ValueError("blind match response task_id mismatch")
    observed = {
        (row.reference_point_id, row.candidate_point_id)
        for row in response.decisions
    }
    if observed != set(package.candidate_pairs):
        raise ValueError("blind match response must cover every candidate pair exactly")
    if any(row.paper_id != package.paper_id_hash for row in response.decisions):
        raise ValueError("blind match response leaked or changed paper identity")


def evaluate_review_pair(
    reference: StructuredReview,
    candidate: StructuredReview,
    decisions: Sequence[PointMatchDecision],
    *,
    valid_evidence_keys: set[str] | None = None,
    semantically_supported_point_ids: set[str] | None = None,
    development_non_confirmatory: bool = False,
) -> PairMetrics:
    if reference.paper_id != candidate.paper_id:
        raise ValueError("pair metrics require matching paper IDs")
    matches, partial, contradictory = _one_to_one_matches(
        reference, candidate, decisions
    )
    ref_points = reference.all_points()
    cand_points = candidate.all_points()
    precision, recall, f1 = _prf(len(matches), len(cand_points), len(ref_points))
    major_ids = {
        point.point_id
        for point in [*reference.weaknesses, *reference.questions]
        if point.severity == PointSeverity.MAJOR
    }
    matched_ref_ids = {left for left, _ in matches}
    major_recall = (
        len(major_ids & matched_ref_ids) / len(major_ids) if major_ids else 1.0
    )
    section_coverage, section_f1 = _section_metrics(reference, candidate, matches)
    novelty_ref = {
        point.point_id
        for point in [
            *reference.novelty.supporting_points,
            *reference.novelty.limiting_points,
        ]
    }
    novelty_cand = {
        point.point_id
        for point in [
            *candidate.novelty.supporting_points,
            *candidate.novelty.limiting_points,
        ]
    }
    novelty_tp = sum(
        left in novelty_ref and right in novelty_cand for left, right in matches
    )
    novelty_precision, novelty_recall, novelty_f1 = _prf(
        novelty_tp, len(novelty_cand), len(novelty_ref)
    )
    keys = [
        *candidate.summary.evidence_keys,
        *(key for point in cand_points for key in point.evidence_keys),
    ]
    if valid_evidence_keys is None:
        valid_ratio = float("nan") if keys else 1.0
    else:
        valid_ratio = sum(key in valid_evidence_keys for key in keys) / max(
            len(keys), 1
        )
    support_precision: float | None = None
    if semantically_supported_point_ids is not None:
        support_precision = len(
            {point.point_id for point in cand_points} & semantically_supported_point_ids
        ) / max(len(cand_points), 1)
    major_candidate = [
        point for point in cand_points if point.severity == PointSeverity.MAJOR
    ]
    unsupported_major = [
        point
        for point in major_candidate
        if not point.evidence_keys
        or (
            valid_evidence_keys is not None
            and any(key not in valid_evidence_keys for key in point.evidence_keys)
        )
        or (
            semantically_supported_point_ids is not None
            and point.point_id not in semantically_supported_point_ids
        )
    ]
    visible_texts = [
        candidate.summary.text,
        *(f"{point.text} {point.suggested_action}" for point in cand_points),
    ]
    graph_violations = sum(
        bool(GRAPH_SEMANTIC_TERMS.search(text)) for text in visible_texts
    )
    return PairMetrics(
        paper_id=reference.paper_id,
        development_non_confirmatory=development_non_confirmatory,
        atomic_precision=precision,
        atomic_recall=recall,
        atomic_f1=f1,
        major_weakness_question_recall=major_recall,
        section_coverage=section_coverage,
        section_f1=section_f1,
        novelty_judgment_correct=(
            reference.novelty.judgment == candidate.novelty.judgment
        ),
        novelty_point_precision=novelty_precision,
        novelty_point_recall=novelty_recall,
        novelty_point_f1=novelty_f1,
        valid_evidence_key_ratio=valid_ratio,
        semantic_support_precision=support_precision,
        unsupported_major_rate=len(unsupported_major) / max(len(major_candidate), 1),
        graph_semantic_violation_count=graph_violations,
        partial_count=partial,
        contradictory_count=contradictory,
    )


def evaluate_corpus(
    pair_metrics: Sequence[PairMetrics],
    novelty_pairs: Sequence[tuple[NoveltyJudgment, NoveltyJudgment]],
    *,
    development_non_confirmatory: bool,
    bootstrap_samples: int = 2000,
    seed: int = 20260811,
) -> CorpusMetrics:
    if not pair_metrics:
        raise ValueError("corpus evaluation requires at least one paper")
    metric_names = (
        "atomic_precision",
        "atomic_recall",
        "atomic_f1",
        "major_weakness_question_recall",
        "novelty_point_f1",
        "valid_evidence_key_ratio",
        "unsupported_major_rate",
    )
    paper_macro = {
        name: _nanmean([float(getattr(row, name)) for row in pair_metrics])
        for name in metric_names
    }
    paper_macro.update(
        {
            f"section_{section}_f1": mean(
                row.section_f1[section] for row in pair_metrics
            )
            for section in (
                "summary",
                "novelty",
                "strengths",
                "weaknesses",
                "questions",
            )
        }
    )
    paper_macro.update(
        {
            f"section_{section}_coverage": mean(
                row.section_coverage[section] for row in pair_metrics
            )
            for section in (
                "summary",
                "novelty",
                "strengths",
                "weaknesses",
                "questions",
            )
        }
    )
    accuracy = sum(left == right for left, right in novelty_pairs) / max(
        len(novelty_pairs), 1
    )
    macro_f1 = _novelty_macro_f1(novelty_pairs)
    cis = {
        name: _bootstrap_ci(
            [float(getattr(row, name)) for row in pair_metrics],
            samples=bootstrap_samples,
            seed=seed + index,
        )
        for index, name in enumerate(metric_names)
    }
    return CorpusMetrics(
        paper_count=len(pair_metrics),
        paper_macro=paper_macro,
        novelty_judgment_accuracy=accuracy,
        novelty_judgment_macro_f1=macro_f1,
        bootstrap_95_ci=cis,
        development_non_confirmatory=development_non_confirmatory,
        composite_score=None,
    )


def wrong_paper_shuffle(
    references: Sequence[StructuredReview],
    candidates: Sequence[StructuredReview],
) -> list[tuple[StructuredReview, StructuredReview]]:
    if len(references) != len(candidates) or len(references) < 2:
        raise ValueError("wrong-paper shuffle needs aligned corpora of size >=2")
    return [
        (
            reference,
            candidates[(index + 1) % len(candidates)].model_copy(
                update={"paper_id": reference.paper_id}
            ),
        )
        for index, reference in enumerate(references)
    ]


def _one_to_one_matches(
    reference: StructuredReview,
    candidate: StructuredReview,
    decisions: Sequence[PointMatchDecision],
) -> tuple[set[tuple[str, str]], int, int]:
    ref_ids = {point.point_id for point in reference.all_points()}
    cand_ids = {point.point_id for point in candidate.all_points()}
    valid = [
        row
        for row in decisions
        if row.paper_id == reference.paper_id
        and row.reference_point_id in ref_ids
        and row.candidate_point_id in cand_ids
    ]
    same = sorted(
        (row for row in valid if row.label == MatchLabel.SAME_POINT),
        key=lambda row: (
            -row.confidence,
            row.reference_point_id,
            row.candidate_point_id,
        ),
    )
    matches: set[tuple[str, str]] = set()
    used_ref: set[str] = set()
    used_cand: set[str] = set()
    for row in same:
        if row.reference_point_id in used_ref or row.candidate_point_id in used_cand:
            continue
        matches.add((row.reference_point_id, row.candidate_point_id))
        used_ref.add(row.reference_point_id)
        used_cand.add(row.candidate_point_id)
    partial = sum(row.label == MatchLabel.PARTIAL_POINT for row in valid)
    contradictory = sum(row.label == MatchLabel.CONTRADICTORY for row in valid)
    return matches, partial, contradictory


def _section_metrics(
    reference: StructuredReview,
    candidate: StructuredReview,
    matches: set[tuple[str, str]],
) -> tuple[dict[str, float], dict[str, float]]:
    ref_sections = _section_point_ids(reference)
    cand_sections = _section_point_ids(candidate)
    coverage = {
        "summary": float(bool(candidate.summary.text.strip())),
        "novelty": 1.0,
        "strengths": float(bool(candidate.strengths)),
        "weaknesses": float(bool(candidate.weaknesses)),
        "questions": float(bool(candidate.questions)),
    }
    result = {"summary": _text_token_f1(reference.summary.text, candidate.summary.text)}
    for section in ("novelty", "strengths", "weaknesses", "questions"):
        tp = sum(
            left in ref_sections[section] and right in cand_sections[section]
            for left, right in matches
        )
        result[section] = _prf(
            tp, len(cand_sections[section]), len(ref_sections[section])
        )[2]
    return coverage, result


def _section_point_ids(review: StructuredReview) -> dict[str, set[str]]:
    return {
        "novelty": {
            point.point_id
            for point in [
                *review.novelty.supporting_points,
                *review.novelty.limiting_points,
            ]
        },
        "strengths": {point.point_id for point in review.strengths},
        "weaknesses": {point.point_id for point in review.weaknesses},
        "questions": {point.point_id for point in review.questions},
    }


def _prf(tp: int, predicted: int, reference: int) -> tuple[float, float, float]:
    precision = tp / predicted if predicted else (1.0 if reference == 0 else 0.0)
    recall = tp / reference if reference else (1.0 if predicted == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _novelty_macro_f1(
    pairs: Sequence[tuple[NoveltyJudgment, NoveltyJudgment]],
) -> float:
    values: list[float] = []
    for label in NoveltyJudgment:
        tp = sum(left == label and right == label for left, right in pairs)
        predicted = sum(right == label for _, right in pairs)
        reference = sum(left == label for left, _ in pairs)
        values.append(_prf(tp, predicted, reference)[2])
    return mean(values)


def _bootstrap_ci(
    values: Sequence[float], *, samples: int, seed: int
) -> tuple[float, float]:
    clean = [value for value in values if not math.isnan(value)]
    if not clean:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    estimates = sorted(
        mean(rng.choice(clean) for _ in clean) for _ in range(max(samples, 1))
    )
    lower = estimates[int(0.025 * (len(estimates) - 1))]
    upper = estimates[int(0.975 * (len(estimates) - 1))]
    return lower, upper


def _nanmean(values: Sequence[float]) -> float:
    clean = [value for value in values if not math.isnan(value)]
    return mean(clean) if clean else float("nan")


def _lexical_similarity(left: str, right: str) -> float:
    left_terms = set(re.findall(r"[a-z0-9]{3,}", left.casefold()))
    right_terms = set(re.findall(r"[a-z0-9]{3,}", right.casefold()))
    return len(left_terms & right_terms) / max(len(left_terms | right_terms), 1)


def _normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.casefold()).strip()


def _text_token_f1(reference: str, candidate: str) -> float:
    reference_tokens = set(_normalize(reference).split())
    candidate_tokens = set(_normalize(candidate).split())
    overlap = len(reference_tokens & candidate_tokens)
    return _prf(overlap, len(candidate_tokens), len(reference_tokens))[2]


__all__ = [
    "CorpusMetrics",
    "MatchJudgePackage",
    "MatchJudgeResponse",
    "MatchLabel",
    "PairMetrics",
    "PointMatchDecision",
    "build_blind_match_package",
    "evaluate_corpus",
    "evaluate_review_pair",
    "validate_match_judge_response",
    "wrong_paper_shuffle",
]

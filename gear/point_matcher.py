"""One-to-one production matching for independent review branch points."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .review_contracts import BranchReview, FusionMatch, ReviewPoint

TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u3400-\u9fff]")
NEGATIVE_MARKERS = {
    "not",
    "no",
    "without",
    "lack",
    "lacks",
    "missing",
    "unsupported",
    "unclear",
    "fail",
    "fails",
}
Relation = Literal["SAME_POINT", "PARTIAL", "CONTRADICTORY", "NO_MATCH"]


@dataclass(frozen=True)
class _LocatedPoint:
    section: str
    point: ReviewPoint


class PointMatcher:
    """Match branch points without using evaluation prompts or human references."""

    def match(self, agent: BranchReview, qwen: BranchReview) -> list[FusionMatch]:
        agent_points = _located_points(agent)
        qwen_points = _located_points(qwen)
        candidates: list[tuple[float, int, int, Relation]] = []
        for agent_index, left in enumerate(agent_points):
            for qwen_index, right in enumerate(qwen_points):
                if left.point.aspect != right.point.aspect:
                    continue
                score = _similarity(left.point, right.point)
                relation = _relation(left.point, right.point, score)
                if relation != "NO_MATCH":
                    candidates.append((score, agent_index, qwen_index, relation))
        used_agent: set[int] = set()
        used_qwen: set[int] = set()
        matches: list[FusionMatch] = []
        for _, agent_index, qwen_index, relation in sorted(candidates, reverse=True):
            if agent_index in used_agent or qwen_index in used_qwen:
                continue
            used_agent.add(agent_index)
            used_qwen.add(qwen_index)
            matches.append(
                FusionMatch(
                    agent_point_id=agent_points[agent_index].point.point_id,
                    qwen_point_id=qwen_points[qwen_index].point.point_id,
                    relation=relation,
                )
            )
        matches.extend(
            FusionMatch(agent_point_id=item.point.point_id, relation="NO_MATCH")
            for index, item in enumerate(agent_points)
            if index not in used_agent
        )
        matches.extend(
            FusionMatch(qwen_point_id=item.point.point_id, relation="NO_MATCH")
            for index, item in enumerate(qwen_points)
            if index not in used_qwen
        )
        return matches


def branch_sections(review: BranchReview) -> dict[str, str]:
    return {item.point.point_id: item.section for item in _located_points(review)}


def _located_points(review: BranchReview) -> list[_LocatedPoint]:
    return [
        *(
            _LocatedPoint("novelty_support", point)
            for point in review.novelty.supporting_points
        ),
        *(
            _LocatedPoint("novelty_limit", point)
            for point in review.novelty.limiting_points
        ),
        *(
            _LocatedPoint("questions", point)
            for point in review.novelty.uncertain_points
        ),
        *(_LocatedPoint("strengths", point) for point in review.strengths),
        *(_LocatedPoint("weaknesses", point) for point in review.weaknesses),
        *(_LocatedPoint("questions", point) for point in review.questions),
    ]


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in TOKEN_RE.findall(text)}


def _similarity(left: ReviewPoint, right: ReviewPoint) -> float:
    left_tokens = _tokens(left.text)
    right_tokens = _tokens(right.text)
    lexical = len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)
    left_spans = {key for key in left.evidence_keys if key.startswith("P:")}
    right_spans = {key for key in right.evidence_keys if key.startswith("P:")}
    span_overlap = 1.0 if left_spans & right_spans else 0.0
    severity = 1.0 if left.severity == right.severity else 0.0
    return 0.65 * lexical + 0.25 * span_overlap + 0.10 * severity


def _relation(left: ReviewPoint, right: ReviewPoint, score: float) -> Relation:
    left_tokens = _tokens(left.text)
    right_tokens = _tokens(right.text)
    left_negative = bool(left_tokens & NEGATIVE_MARKERS)
    right_negative = bool(right_tokens & NEGATIVE_MARKERS)
    span_overlap = bool(set(left.evidence_keys) & set(right.evidence_keys))
    if span_overlap and left_negative != right_negative and score >= 0.25:
        return "CONTRADICTORY"
    if score >= 0.72:
        return "SAME_POINT"
    if score >= 0.30 or span_overlap:
        return "PARTIAL"
    return "NO_MATCH"


__all__ = ["PointMatcher", "branch_sections"]

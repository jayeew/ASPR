from __future__ import annotations

import pytest
from pydantic import ValidationError

from gear.review_contracts import (
    NoveltyAssessment,
    NoveltyJudgment,
    PointSeverity,
    ReviewAspect,
    ReviewPoint,
    ReviewSummary,
    StructuredReview,
)


def _point(index: int, *, aspect: ReviewAspect = ReviewAspect.METHOD):
    return ReviewPoint(
        point_id=f"RP-{index}",
        aspect=aspect,
        text=f"Atomic point {index} is supported by the cited manuscript span.",
        severity=PointSeverity.MINOR,
        evidence_keys=["P:S-test"],
    )


def _review() -> StructuredReview:
    support = _point(0, aspect=ReviewAspect.NOVELTY_PRIOR_ART).model_copy(
        update={"external_verification_required": True}
    )
    return StructuredReview(
        paper_id="paper",
        summary=ReviewSummary(
            text="A concise contribution.", evidence_keys=["P:S-test"]
        ),
        novelty=NoveltyAssessment(
            judgment=NoveltyJudgment.POSITIVE,
            supporting_points=[support],
            limiting_points=[],
        ),
        strengths=[_point(1)],
    )


def test_contract_rejects_recommendation_and_extra_fields():
    payload = _review().model_dump(mode="json")
    payload["recommendation"] = "accept"
    with pytest.raises(ValidationError):
        StructuredReview.model_validate(payload)


def test_novelty_direction_is_independent_from_verification_points():
    support = _point(0, aspect=ReviewAspect.NOVELTY_PRIOR_ART)
    limit = _point(1, aspect=ReviewAspect.NOVELTY_PRIOR_ART)
    assessment = NoveltyAssessment(
        judgment=NoveltyJudgment.POSITIVE,
        supporting_points=[support],
        limiting_points=[limit],
    )
    assert assessment.judgment == NoveltyJudgment.POSITIVE


def test_contract_enforces_point_and_word_limits():
    with pytest.raises(ValidationError):
        ReviewPoint(
            point_id="too-long",
            aspect=ReviewAspect.METHOD,
            text=" ".join(["word"] * 121),
        )
    review = _review().model_dump(mode="json")
    review["strengths"] = [
        _point(index + 10).model_dump(mode="json") for index in range(24)
    ]
    with pytest.raises(ValidationError):
        StructuredReview.model_validate(review)


def test_major_point_requires_evidence():
    with pytest.raises(ValidationError):
        ReviewPoint(
            point_id="major",
            aspect=ReviewAspect.EXPERIMENT_EVIDENCE,
            text="The central experiment lacks a necessary control.",
            severity=PointSeverity.MAJOR,
        )


def test_duplicate_evidence_keys_are_deduplicated_in_original_order():
    point = ReviewPoint(
        point_id="duplicate-evidence",
        aspect=ReviewAspect.METHOD,
        text="The method point cites two distinct manuscript spans.",
        evidence_keys=["P:S-first", "P:S-second", "P:S-first"],
    )
    assert point.evidence_keys == ["P:S-first", "P:S-second"]

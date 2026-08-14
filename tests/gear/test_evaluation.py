from __future__ import annotations

from experiments.gear.review_reconstruction.evaluation import (
    MatchJudgeResponse,
    MatchLabel,
    PointMatchDecision,
    build_blind_match_package,
    evaluate_review_pair,
    validate_match_judge_response,
    wrong_paper_shuffle,
)
from gear.review_contracts import (
    NoveltyAssessment,
    NoveltyJudgment,
    ReviewAspect,
    ReviewPoint,
    ReviewSummary,
    StructuredReview,
)


def _review(point_id: str) -> StructuredReview:
    point = ReviewPoint(
        point_id=point_id,
        aspect=ReviewAspect.METHOD,
        text="The method uses a clearly described comparison protocol.",
        evidence_keys=["P:S-test"],
    )
    return StructuredReview(
        paper_id="paper",
        summary=ReviewSummary(text="Same summary.", evidence_keys=["P:S-test"]),
        novelty=NoveltyAssessment(
            judgment=NoveltyJudgment.NOT_DISCUSSED,
            supporting_points=[],
            limiting_points=[],
        ),
        strengths=[point],
    )


def test_identical_structures_score_one_and_matching_is_one_to_one():
    reference = _review("R")
    candidate = _review("C")
    decisions = [
        PointMatchDecision(
            paper_id="paper",
            reference_point_id="R",
            candidate_point_id="C",
            label=MatchLabel.SAME_POINT,
        ),
        PointMatchDecision(
            paper_id="paper",
            reference_point_id="R",
            candidate_point_id="C",
            label=MatchLabel.SAME_POINT,
            confidence=0.5,
        ),
    ]
    metrics = evaluate_review_pair(
        reference,
        candidate,
        decisions,
        valid_evidence_keys={"P:S-test"},
        semantically_supported_point_ids={"C"},
    )
    assert metrics.atomic_precision == 1.0
    assert metrics.atomic_recall == 1.0
    assert metrics.atomic_f1 == 1.0
    assert metrics.major_weakness_question_recall == 1.0
    assert metrics.section_f1["strengths"] == 1.0
    assert metrics.section_coverage["summary"] == 1.0
    assert metrics.valid_evidence_key_ratio == 1.0
    assert metrics.unsupported_major_rate == 0.0
    assert "composite" not in metrics.model_dump_json()


def test_wrong_paper_shuffle_preserves_blinded_comparison_identity():
    left = _review("L")
    right = _review("R").model_copy(update={"paper_id": "paper-2"})
    pairs = wrong_paper_shuffle([left, right], [left, right])
    assert all(
        reference.paper_id == candidate.paper_id for reference, candidate in pairs
    )
    assert pairs[0][1].strengths[0].point_id == "R"


def test_blind_match_response_requires_exact_pair_coverage():
    reference = _review("R")
    candidate = _review("C")
    package = build_blind_match_package(reference, candidate)
    response = MatchJudgeResponse(
        task_id=package.task_id,
        model_id="judge",
        conversation_hash="sha256:" + "a" * 64,
        decisions=[
            PointMatchDecision(
                paper_id=package.paper_id_hash,
                reference_point_id="R",
                candidate_point_id="C",
                label=MatchLabel.SAME_POINT,
            )
        ],
    )
    validate_match_judge_response(package, response)

from __future__ import annotations

from datetime import date

from experiments.gear.revision_audit import (
    AuditPoint,
    RevisionAuditCase,
    build_blind_package,
    score_pairs,
)
from gear.contracts import PaperMetadata, ReviewRequest


def _point(point_id: str, section: str = "weaknesses") -> AuditPoint:
    return AuditPoint(
        point_id=point_id,
        section=section,
        aspect="method",
        severity="major",
        text=point_id,
        suggested_action=None,
    )


def test_submission_date_precedes_publication_date() -> None:
    request = ReviewRequest(
        paper_path="paper.md",
        metadata=PaperMetadata(
            submission_date=date(2021, 1, 2), publication_date=date(2024, 3, 4)
        ),
    )
    assert request.evidence_date == date(2021, 1, 2)
    assert request.evidence_date_source == "submission_date"


def test_empty_reviews_are_not_a_perfect_match() -> None:
    metrics = score_pairs([], [], [])
    assert metrics["strict"]["f1"] is None
    assert metrics["strict"]["both_empty"] is True


def test_soft_assignment_is_one_to_one_and_cross_section_is_separate() -> None:
    left = [_point("H1", "weaknesses"), _point("H2", "questions")]
    right = [_point("A1", "questions"), _point("A2", "weaknesses")]
    decisions = [
        {
            "left_id": "L-H1",
            "right_id": "R-A1",
            "label": "SAME_POINT",
            "confidence": 1.0,
        },
        {
            "left_id": "L-H1",
            "right_id": "R-A2",
            "label": "PARTIAL_POINT",
            "confidence": 1.0,
        },
        {
            "left_id": "L-H2",
            "right_id": "R-A1",
            "label": "PARTIAL_POINT",
            "confidence": 1.0,
        },
        {
            "left_id": "L-H2",
            "right_id": "R-A2",
            "label": "SAME_POINT",
            "confidence": 1.0,
        },
    ]
    metrics = score_pairs(left, right, decisions)
    assert metrics["strict_match_count"] == 2
    assert metrics["strict"]["f1"] == 1.0
    assert metrics["section_correct_rate"] == 0.0


def test_blind_package_uses_opaque_ids() -> None:
    case = RevisionAuditCase(
        paper_id="10.1038/s41467-023-36025-x",
        manuscript_path="paper.md",
        metadata_path="metadata.json",
        reconstruction_dir="reconstruction",
        agent_run_dir="agent",
        cutoff_date="2021-07-14",
    )
    human = _point("human-private-id")
    human = AuditPoint(**{**human.__dict__, "text": "human scientific point"})
    agent = _point("ai-private-id")
    agent = AuditPoint(**{**agent.__dict__, "text": "agent scientific point"})
    package = build_blind_package(case, [human], [agent], kind="retained", chunk=0)
    assert "human-private-id" not in str(package["left"])
    assert "ai-private-id" not in str(package["right"])
    assert package["mapping"]

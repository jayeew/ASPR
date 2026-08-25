from __future__ import annotations

import json

import pytest

from gear.codex_critic import FORBIDDEN_DECISION_TEXT
from gear.paper_extraction import PaperRubricBuilder
from gear.review_contracts import ReviewSource
from gear.reviewers import ASPRQwenReviewer, CodexAgentReviewer
from gear.reviewers.base import assert_graph_blind_payload


def _empty_review(paper_ir):
    span = paper_ir.spans[0]
    return {
        "schema_version": "aspr_gear",
        "paper_id": paper_ir.paper_id,
        "summary": {
            "schema_version": "aspr_gear",
            "text": "The manuscript presents a bounded evidence review method.",
            "evidence_keys": [f"P:{span.span_id}"],
        },
        "novelty": {
            "schema_version": "aspr_gear",
            "judgment": "not_discussed",
            "supporting_points": [],
            "limiting_points": [],
        },
        "strengths": [],
        "weaknesses": [],
        "questions": [],
    }


def test_agent_input_is_graph_blind(gear_config, paper_ir) -> None:
    reviewer = CodexAgentReviewer(
        gear_config, generator=lambda system, user: _empty_review(paper_ir)
    )
    branch = reviewer.review(paper_ir, PaperRubricBuilder().build(paper_ir))
    payload = json.dumps(reviewer.last_payload, sort_keys=True).casefold()
    assert branch.source == ReviewSource.AGENT
    assert branch.graph_blind is True
    for forbidden in ("graph_prior", "aspr_score", "p_uptake", "d5_percentile"):
        assert forbidden not in payload


def test_graph_blind_check_rejects_fields_but_allows_scientific_prose() -> None:
    assert_graph_blind_payload(
        {"spans": [{"text": "This creates an opportunity for future work."}]}
    )
    with pytest.raises(ValueError, match="score_0_100"):
        assert_graph_blind_payload({"nested": {"score_0_100": 75.0}})


def test_scientific_rejection_is_not_editorial_decision_language() -> None:
    assert FORBIDDEN_DECISION_TEXT.search("99% salt rejection") is None
    assert FORBIDDEN_DECISION_TEXT.search("the paper should be rejected") is not None


def test_agent_forces_external_verification_for_novelty_points(
    gear_config, paper_ir
) -> None:
    payload = _empty_review(paper_ir)
    payload["novelty"] = {
        "schema_version": "aspr_gear",
        "judgment": "positive",
        "supporting_points": [
            {
                "schema_version": "aspr_gear",
                "point_id": "novelty-one",
                "aspect": "novelty_prior_art",
                "text": "The manuscript presents a potentially novel review method.",
                "severity": "none",
                "suggested_action": "",
                "evidence_keys": [f"P:{paper_ir.spans[0].span_id}"],
                "external_verification_required": False,
            }
        ],
        "limiting_points": [],
    }
    payload["weaknesses"] = [
        {
            "schema_version": "aspr_gear",
            "point_id": "misplaced-novelty",
            "aspect": "novelty_prior_art",
            "text": "The prior-art comparison requires independent verification.",
            "severity": "minor",
            "suggested_action": "",
            "evidence_keys": [f"P:{paper_ir.spans[0].span_id}"],
            "external_verification_required": False,
        }
    ]
    reviewer = CodexAgentReviewer(gear_config, generator=lambda system, user: payload)
    branch = reviewer.review(paper_ir, PaperRubricBuilder().build(paper_ir))
    assert branch.failures == []
    assert branch.novelty.supporting_points[0].external_verification_required is True
    assert branch.weaknesses[0].external_verification_required is True


def test_agent_preserves_explicit_novelty_direction(gear_config, paper_ir) -> None:
    payload = _empty_review(paper_ir)
    payload["novelty"] = {
        "schema_version": "aspr_gear",
        "judgment": "positive",
        "supporting_points": [
            {
                "schema_version": "aspr_gear",
                "point_id": "support",
                "aspect": "novelty_prior_art",
                "text": "The manuscript uses a distinct empirical setting.",
                "severity": "none",
                "suggested_action": "",
                "evidence_keys": [f"P:{paper_ir.spans[0].span_id}"],
            }
        ],
        "limiting_points": [
            {
                "schema_version": "aspr_gear",
                "point_id": "limit",
                "aspect": "novelty_prior_art",
                "text": "The underlying mechanism should be compared with prior work.",
                "severity": "minor",
                "suggested_action": "",
                "evidence_keys": [f"P:{paper_ir.spans[0].span_id}"],
                "external_verification_required": True,
            }
        ],
    }
    reviewer = CodexAgentReviewer(gear_config, generator=lambda system, user: payload)
    branch = reviewer.review(paper_ir, PaperRubricBuilder().build(paper_ir))
    assert branch.failures == []
    assert branch.novelty.judgment.value == "positive"
    assert branch.novelty.verification_status.value == "not_assessed"


def test_agent_separates_pending_verification_from_novelty_direction(
    gear_config, paper_ir
) -> None:
    payload = _empty_review(paper_ir)
    payload["novelty"] = {
        "schema_version": "aspr_gear",
        "judgment": "uncertain",
        "verification_status": "not_assessed",
        "supporting_points": [
            {
                "schema_version": "aspr_gear",
                "point_id": "support",
                "aspect": "novelty_prior_art",
                "text": "The manuscript demonstrates a bounded technical delta.",
                "severity": "minor",
                "suggested_action": "",
                "evidence_keys": [f"P:{paper_ir.spans[0].span_id}"],
            }
        ],
        "limiting_points": [
            {
                "schema_version": "aspr_gear",
                "point_id": "limit",
                "aspect": "novelty_prior_art",
                "text": "The nearest antecedent still requires external comparison.",
                "severity": "minor",
                "suggested_action": "",
                "evidence_keys": [f"P:{paper_ir.spans[0].span_id}"],
                "external_verification_required": True,
            }
        ],
    }

    reviewer = CodexAgentReviewer(gear_config, generator=lambda system, user: payload)
    branch = reviewer.review(paper_ir, PaperRubricBuilder().build(paper_ir))

    assert branch.failures == []
    assert branch.novelty.judgment.value == "positive"
    assert branch.novelty.verification_status.value == "not_assessed"


def test_agent_keeps_manuscript_grounded_overlap_mixed(gear_config, paper_ir) -> None:
    payload = _empty_review(paper_ir)
    payload["novelty"] = {
        "schema_version": "aspr_gear",
        "judgment": "uncertain",
        "verification_status": "not_assessed",
        "supporting_points": [
            {
                "schema_version": "aspr_gear",
                "point_id": "support",
                "aspect": "novelty_prior_art",
                "text": "The manuscript demonstrates a bounded technical delta.",
                "severity": "minor",
                "suggested_action": "",
                "evidence_keys": [f"P:{paper_ir.spans[0].span_id}"],
            }
        ],
        "limiting_points": [
            {
                "schema_version": "aspr_gear",
                "point_id": "limit",
                "aspect": "novelty_prior_art",
                "text": "The manuscript identifies material overlap with its antecedent.",
                "severity": "major",
                "suggested_action": "Bound the residual delta.",
                "evidence_keys": [f"P:{paper_ir.spans[0].span_id}"],
                "external_verification_required": False,
            }
        ],
    }

    branch = CodexAgentReviewer(
        gear_config, generator=lambda system, user: payload
    ).review(paper_ir, PaperRubricBuilder().build(paper_ir))

    assert branch.failures == []
    assert branch.novelty.judgment.value == "mixed"


def test_agent_normalizes_novelty_point_aspect(gear_config, paper_ir) -> None:
    payload = _empty_review(paper_ir)
    payload["novelty"]["supporting_points"] = [
        {
            "schema_version": "aspr_gear",
            "point_id": "support",
            "aspect": "method",
            "text": "The manuscript uses a distinct empirical setting.",
            "severity": "none",
            "suggested_action": "",
            "evidence_keys": [f"P:{paper_ir.spans[0].span_id}"],
        }
    ]
    reviewer = CodexAgentReviewer(gear_config, generator=lambda system, user: payload)
    branch = reviewer.review(paper_ir, PaperRubricBuilder().build(paper_ir))
    assert branch.failures == []
    assert branch.novelty.supporting_points[0].aspect.value == "novelty_prior_art"


def test_agent_repairs_mistyped_paper_evidence_key(gear_config, paper_ir) -> None:
    payload = _empty_review(paper_ir)
    payload["summary"]["evidence_keys"] = ["P:S-model-invented"]
    reviewer = CodexAgentReviewer(gear_config, generator=lambda system, user: payload)

    branch = reviewer.review(paper_ir, PaperRubricBuilder().build(paper_ir))

    allowed = {f"P:{span.span_id}" for span in paper_ir.spans}
    assert branch.failures == []
    assert set(branch.summary.evidence_keys) <= allowed


def test_qwen_branch_is_optional(gear_config, paper_ir) -> None:
    reviewer = ASPRQwenReviewer(gear_config)
    assert reviewer.enabled is False
    assert reviewer.review(paper_ir, PaperRubricBuilder().build(paper_ir)) is None

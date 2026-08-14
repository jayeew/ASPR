from __future__ import annotations

import hashlib

import pytest

from experiments.gear.review_reconstruction.contracts import (
    ReconstructionSessionResponse,
    ReferenceTrace,
    ResolutionStatus,
)
from experiments.gear.review_reconstruction.sessions import (
    build_reconstruction_package,
    validate_session_package,
    validate_session_response,
    write_session_handoff,
)
from gear.review_contracts import (
    NoveltyAssessment,
    NoveltyJudgment,
    ReviewAspect,
    ReviewPoint,
    ReviewSummary,
    StructuredReview,
)
from gear.trace import sha256_value


def _case(sample_md, tmp_path):
    review = tmp_path / "review.md"
    review.write_text(
        "# Reviewer 1\n\nThe method is clearly specified and reproducible.\n\n"
        "# Reviewer 2\n\nThe evaluation needs a stronger baseline comparison.\n\n"
        "# Author response\n\nWe added the requested baseline and updated the results.",
        encoding="utf-8",
    )
    return {"paper_path": str(sample_md), "review_path": str(review)}


def _response(package):
    quote = package.reviewer_spans[0]
    evidence = package.paper_context.spans[0].evidence_key
    point = ReviewPoint(
        point_id="RP-reconstructed",
        aspect=ReviewAspect.METHOD,
        text="The method is specified clearly enough to support reproducibility.",
        evidence_keys=[evidence],
    )
    review = StructuredReview(
        paper_id=package.paper_id,
        summary=ReviewSummary(
            text="The paper presents a review method.", evidence_keys=[evidence]
        ),
        novelty=NoveltyAssessment(
            judgment=NoveltyJudgment.NOT_DISCUSSED,
            supporting_points=[],
            limiting_points=[],
        ),
        strengths=[point],
    )
    trace = ReferenceTrace(
        trace_id="TRACE-1",
        paper_id=package.paper_id,
        point_id=point.point_id,
        reviewer_quote_keys=[quote.source_key],
        round_ids=[quote.round_id],
        reviewer_id_hashes=[quote.reviewer_id_hash],
        resolution_status=ResolutionStatus.PERSISTS,
    )
    response = ReconstructionSessionResponse(
        package_id=package.package_id,
        session_kind=package.session_kind,
        paper_id=package.paper_id,
        model_id="independent-codex-test",
        conversation_hash="sha256:"
        + hashlib.sha256(b"independent-session").hexdigest(),
        prompt_sha256=package.prompt_sha256,
        schema_sha256=package.schema_sha256,
        input_sha256=package.input_sha256,
        review=review,
        reference_traces=[trace],
        output_sha256="pending",
    )
    return response.model_copy(
        update={"output_sha256": sha256_value(response.hash_payload())}
    )


def test_role_separation_and_hash_bound_response(sample_md, tmp_path, gear_config):
    package, paper_ir = build_reconstruction_package(
        _case(sample_md, tmp_path), config=gear_config
    )
    assert package.reviewer_spans
    assert package.author_response_spans
    response = _response(package)
    validate_session_response(package, response)
    handoff = write_session_handoff(package, paper_ir, tmp_path / "handoff")
    assert (handoff / "response.schema.json").is_file()


def test_author_response_cannot_be_used_as_reviewer_quote(
    sample_md, tmp_path, gear_config
):
    package, _ = build_reconstruction_package(
        _case(sample_md, tmp_path), config=gear_config
    )
    response = _response(package)
    bad_trace = response.reference_traces[0].model_copy(
        update={"reviewer_quote_keys": [package.author_response_spans[0].source_key]}
    )
    bad = response.model_copy(update={"reference_traces": [bad_trace]})
    bad = bad.model_copy(update={"output_sha256": sha256_value(bad.hash_payload())})
    with pytest.raises(ValueError, match="non-reviewer"):
        validate_session_response(package, bad)


def test_unknown_final_paper_span_fails_closed(sample_md, tmp_path, gear_config):
    package, _ = build_reconstruction_package(
        _case(sample_md, tmp_path), config=gear_config
    )
    response = _response(package)
    forged_point = response.review.strengths[0].model_copy(
        update={"evidence_keys": ["P:S-forged"]}
    )
    forged_review = response.review.model_copy(update={"strengths": [forged_point]})
    forged = response.model_copy(update={"review": forged_review})
    forged = forged.model_copy(
        update={"output_sha256": sha256_value(forged.hash_payload())}
    )
    with pytest.raises(ValueError, match="unknown final-paper"):
        validate_session_response(package, forged)


def test_package_input_hash_drift_fails_closed(sample_md, tmp_path, gear_config):
    package, _ = build_reconstruction_package(
        _case(sample_md, tmp_path), config=gear_config
    )
    drifted = package.model_copy(update={"instructions": package.instructions + "x"})
    with pytest.raises(ValueError, match="prompt hash drift"):
        validate_session_package(drifted)


def test_reconstructed_point_requires_paper_evidence(sample_md, tmp_path, gear_config):
    package, _ = build_reconstruction_package(
        _case(sample_md, tmp_path), config=gear_config
    )
    response = _response(package)
    point = response.review.strengths[0].model_copy(update={"evidence_keys": []})
    review = response.review.model_copy(update={"strengths": [point]})
    empty = response.model_copy(update={"review": review})
    empty = empty.model_copy(
        update={"output_sha256": sha256_value(empty.hash_payload())}
    )
    with pytest.raises(ValueError, match="requires final-paper evidence"):
        validate_session_response(package, empty)


def test_domain_specific_rejection_term_is_not_decision_language(
    sample_md, tmp_path, gear_config
):
    package, _ = build_reconstruction_package(
        _case(sample_md, tmp_path), config=gear_config
    )
    response = _response(package)
    summary = response.review.summary.model_copy(
        update={"text": "The membrane achieves high salt rejection."}
    )
    review = response.review.model_copy(update={"summary": summary})
    technical = response.model_copy(update={"review": review})
    technical = technical.model_copy(
        update={"output_sha256": sha256_value(technical.hash_payload())}
    )
    validate_session_response(package, technical)


@pytest.mark.parametrize(
    "text",
    [
        "I recommend this paper for publication.",
        "The manuscript should be rejected.",
        "I recommend this paper for rejection.",
    ],
)
def test_publication_decision_language_fails_closed(
    sample_md, tmp_path, gear_config, text
):
    package, _ = build_reconstruction_package(
        _case(sample_md, tmp_path), config=gear_config
    )
    response = _response(package)
    summary = response.review.summary.model_copy(update={"text": text})
    review = response.review.model_copy(update={"summary": summary})
    leaked = response.model_copy(update={"review": review})
    leaked = leaked.model_copy(
        update={"output_sha256": sha256_value(leaked.hash_payload())}
    )
    with pytest.raises(ValueError, match="publication/decision"):
        validate_session_response(package, leaked)

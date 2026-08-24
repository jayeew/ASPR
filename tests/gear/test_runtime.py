from __future__ import annotations

import json
from datetime import date

from gear.cli import _validate_command
from gear.codex_critic import CodexCliCritic
from gear.contracts import PaperMetadata, ReviewRequest, ReviewStatus
from gear.graph_context import build_graph_review_context
from gear.graph_prior_contracts import GraphResultV3
from gear.paper_compiler import PaperCompiler
from gear.paper_extraction import PaperRubricBuilder
from gear.prior_art import PriorArtService, RelationClassifier
from gear.review_contracts import (
    CanonicalReviewPoint,
    NoveltyAssessment,
    NoveltyJudgment,
    ReviewAspect,
    ReviewPoint,
    ReviewSummary,
    StructuredReview,
)
from gear.review_pipeline import ServiceRegistry, review_paper
from gear.review_state import initialize_review_state_v3
from gear.review_verifier import ReviewVerifier
from gear.trace import EvidenceStore


def _draft(paper_ir):
    span = paper_ir.spans[0]
    point = ReviewPoint(
        point_id="RP-strength",
        aspect=ReviewAspect.CONTRIBUTION,
        text="The contribution is stated clearly in the manuscript.",
        evidence_keys=[f"P:{span.span_id}"],
    )
    return StructuredReview(
        paper_id=paper_ir.paper_id,
        summary=ReviewSummary(
            text="The manuscript presents an evidence-grounded review method.",
            evidence_keys=[f"P:{span.span_id}"],
        ),
        novelty=NoveltyAssessment(
            judgment=NoveltyJudgment.NOT_DISCUSSED,
            supporting_points=[],
            limiting_points=[],
        ),
        strengths=[point],
    )


def test_graph_projection_excludes_opportunity_and_controls(
    paper_ir, calibration_factory
):
    packet = calibration_factory(paper_ir.paper_id, score=99.0)
    graph = build_graph_review_context(packet)
    serialized = graph.model_dump_json()
    assert "opportunity" not in serialized
    assert "context_control" not in serialized
    assert graph.d5_percentile == 99.0
    assert _draft(paper_ir).novelty.judgment == NoveltyJudgment.NOT_DISCUSSED


def test_codex_cli_returns_entire_review(gear_config, paper_ir, calibration_factory):
    graph = build_graph_review_context(
        calibration_factory(paper_ir.paper_id, score=90.0)
    )
    critic = CodexCliCritic(
        gear_config,
        generator=lambda system, user: _draft(paper_ir).model_dump(mode="json"),
    )
    review = critic.review(paper_ir, graph)
    assert review == _draft(paper_ir)
    assert critic.last_failures == []
    assert critic.metadata.critic_source.value == "codex_cli"


def test_pipeline_outputs_only_five_part_contract(
    tmp_path, gear_config, sample_md, calibration_factory
):
    output = tmp_path / "current"
    compiler = PaperCompiler(gear_config)
    request = ReviewRequest(
        paper_path=sample_md,
        metadata=PaperMetadata(
            title="current manuscript",
            publication_date=date(2010, 1, 2),
        ),
    )
    paper_ir = compiler.compile(request)

    class CalibrationFake:
        def build_packet(self, supplied, **kwargs):
            return calibration_factory(supplied.paper_id, score=55.0)

    reviewer = CodexCliCritic(
        gear_config,
        generator=lambda system, user: _draft(paper_ir).model_dump(mode="json"),
    )
    services = ServiceRegistry(
        evidence_store=EvidenceStore(output),
        paper_compiler=compiler,
        calibration_service=CalibrationFake(),
        reviewer=reviewer,
        prior_art=PriorArtService(gear_config),
        relation_classifier=RelationClassifier(gear_config),
        verifier=ReviewVerifier(
            gear_config,
            semantic_checker=lambda system, user: {
                "unsupported_point_ids": [],
                "summary_supported": True,
            },
        ),
    )
    bundle = review_paper(
        request, output_dir=output, config=gear_config, services=services
    )
    assert bundle.status == ReviewStatus.COMPLETE
    assert bundle.schema_version == "aspr_gear"
    payload = bundle.structured_review.model_dump(mode="json")
    assert set(payload) == {
        "schema_version",
        "schema_revision",
        "paper_id",
        "summary",
        "novelty",
        "strengths",
        "weaknesses",
        "questions",
    }
    assert "recommend" not in bundle.review_markdown.casefold()
    assert "graph calibration" not in bundle.review_markdown.casefold()
    assert bundle.graph_result is not None
    assert bundle.state_v3 is not None
    assert (output / "graph_result.json").is_file()
    assert not (output / "graph_prior.json").exists()
    assert not (output / "graph_prior_audit.json").exists()
    serialized_bundle = json.loads(
        (output / "review_bundle.json").read_text(encoding="utf-8")
    )
    assert serialized_bundle["contract"] == "aspr_gear_review_bundle_v3"
    for legacy_field in ("calibration", "graph_context", "graph_prior", "state_v2"):
        assert legacy_field not in serialized_bundle
    assert services.evidence_store.validate_manifest() == []


def test_semantic_verifier_failure_forces_limited(
    tmp_path, gear_config, sample_md, calibration_factory
):
    output = tmp_path / "semantic-failed"
    compiler = PaperCompiler(gear_config)
    request = ReviewRequest(paper_path=sample_md)
    paper_ir = compiler.compile(request)

    class CalibrationFake:
        def build_packet(self, supplied, **kwargs):
            return calibration_factory(supplied.paper_id, score=55.0)

    reviewer = CodexCliCritic(
        gear_config,
        generator=lambda system, user: _draft(paper_ir).model_dump(mode="json"),
    )

    def failed_semantic_checker(system, user):
        raise ValueError("invalid semantic response")

    services = ServiceRegistry(
        evidence_store=EvidenceStore(output),
        paper_compiler=compiler,
        calibration_service=CalibrationFake(),
        reviewer=reviewer,
        prior_art=PriorArtService(gear_config),
        relation_classifier=RelationClassifier(gear_config),
        verifier=ReviewVerifier(gear_config, semantic_checker=failed_semantic_checker),
    )
    bundle = review_paper(
        request, output_dir=output, config=gear_config, services=services
    )
    assert bundle.status == ReviewStatus.LIMITED
    assert bundle.verification.limited is True
    assert "semantic_verification_unavailable_or_failed" in (
        bundle.process_diagnostic or {}
    ).get("blocking_reasons", [])


def test_delete_only_semantic_repair_reaches_final_compiler(
    tmp_path, gear_config, sample_md, calibration_factory
):
    output = tmp_path / "semantic-delete-repair"
    compiler = PaperCompiler(gear_config)
    request = ReviewRequest(paper_path=sample_md)
    paper_ir = compiler.compile(request)

    class CalibrationFake:
        def build_packet(self, supplied, **kwargs):
            return calibration_factory(supplied.paper_id, score=55.0)

    def reject_visible_points(system, user):
        del system
        review = json.loads(user)["review"]
        points = [
            *review["novelty"]["supporting_points"],
            *review["novelty"]["limiting_points"],
            *review["novelty"]["uncertain_points"],
            *review["strengths"],
            *review["weaknesses"],
            *review["questions"],
        ]
        return {
            "unsupported_point_ids": [point["point_id"] for point in points],
            "summary_supported": True,
        }

    services = ServiceRegistry(
        evidence_store=EvidenceStore(output),
        paper_compiler=compiler,
        calibration_service=CalibrationFake(),
        reviewer=CodexCliCritic(
            gear_config,
            generator=lambda system, user: _draft(paper_ir).model_dump(mode="json"),
        ),
        prior_art=PriorArtService(gear_config),
        relation_classifier=RelationClassifier(gear_config),
        verifier=ReviewVerifier(gear_config, semantic_checker=reject_visible_points),
    )

    bundle = review_paper(
        request, output_dir=output, config=gear_config, services=services
    )

    assert bundle.verification.passed is True
    assert bundle.structured_review.all_points() == []
    assert bundle.state_v3 is not None
    assert bundle.state_v3.phase.value == "compiled"


def test_wrong_paper_relation_target_is_blocking(
    tmp_path, gear_config, paper_ir, paper_request
):
    graph = GraphResultV3(
        paper_id=paper_ir.paper_id,
        score_0_100=50,
        p_uptake=0.5,
        conditional_diffusion=0.5,
        feature_coverage=1.0,
    )
    state = initialize_review_state_v3(
        paper_ir,
        PaperRubricBuilder().build(paper_ir),
        graph,
        paper_request.evidence_date,
    )
    point = CanonicalReviewPoint(
        point_id="CP-wrong",
        section="novelty_limit",
        aspect=ReviewAspect.NOVELTY_PRIOR_ART,
        severity="major",
        proposition="A prior work may overlap.",
        agent_support=True,
        relation_evidence_keys=["R:wrong"],
    )
    state.canonical_points[point.point_id] = point
    store = EvidenceStore(tmp_path / "wrong-relation")
    store.add_evidence(
        "R:wrong",
        "relation_card",
        {
            "target_span_id": "S-from-another-paper",
            "prior_span_id": "RSPAN-1",
            "difference_dimensions": ["method"],
            "temporal_valid": True,
        },
    )
    verifier = ReviewVerifier(gear_config, semantic_checker=None)
    issues = verifier._relation_issues(state, store, paper_ir)
    assert any(issue.code == "relation_target_span_mismatch" for issue in issues)


def test_graph_exception_is_limited_with_explicit_reason(
    tmp_path, gear_config, sample_md
):
    output = tmp_path / "graph-failed"
    compiler = PaperCompiler(gear_config)
    request = ReviewRequest(paper_path=sample_md)
    paper_ir = compiler.compile(request)

    class BrokenGraph:
        def score(self, supplied, cutoff):
            raise RuntimeError("injected graph failure")

    reviewer = CodexCliCritic(
        gear_config,
        generator=lambda system, user: _draft(paper_ir).model_dump(mode="json"),
    )
    services = ServiceRegistry(
        evidence_store=EvidenceStore(output),
        paper_compiler=compiler,
        graph_scorer=BrokenGraph(),
        reviewer=reviewer,
        prior_art=PriorArtService(gear_config),
        relation_classifier=RelationClassifier(gear_config),
        verifier=ReviewVerifier(
            gear_config,
            semantic_checker=lambda system, user: {
                "unsupported_point_ids": [],
                "summary_supported": True,
            },
        ),
    )
    bundle = review_paper(
        request, output_dir=output, config=gear_config, services=services
    )
    assert bundle.status == ReviewStatus.LIMITED
    assert "graph_unavailable" in (bundle.process_diagnostic or {}).get(
        "blocking_reasons", []
    )


def test_validate_run_rejects_pre_contract_bundle(tmp_path, capsys):
    run_dir = tmp_path / "old-run"
    run_dir.mkdir()
    (run_dir / "review_bundle.json").write_text(
        json.dumps({"schema_version": "aspr_gear"}), encoding="utf-8"
    )

    exit_code = _validate_command(type("Args", (), {"run_dir": run_dir})())

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["code"] == "unsupported_schema_revision"


def test_missing_codex_cli_is_explicit_limited(
    tmp_path, gear_config, sample_md, calibration_factory
):
    output = tmp_path / "limited-current"

    class CalibrationFake:
        def build_packet(self, supplied, **kwargs):
            return calibration_factory(supplied.paper_id, score=55.0)

    def unavailable(system, user):
        raise ValueError("Codex CLI unavailable")

    reviewer = CodexCliCritic(gear_config, generator=unavailable)
    services = ServiceRegistry(
        evidence_store=EvidenceStore(output),
        paper_compiler=PaperCompiler(gear_config),
        calibration_service=CalibrationFake(),
        reviewer=reviewer,
        prior_art=PriorArtService(gear_config),
        relation_classifier=RelationClassifier(gear_config),
        verifier=ReviewVerifier(
            gear_config,
            semantic_checker=lambda system, user: {
                "unsupported_point_ids": [],
                "summary_supported": True,
            },
        ),
    )
    bundle = review_paper(
        ReviewRequest(paper_path=sample_md),
        output_dir=output,
        config=gear_config,
        services=services,
    )
    assert bundle.status == ReviewStatus.LIMITED
    assert bundle.critic.critic_source.value == "unavailable"
    assert bundle.structured_review.novelty.judgment == NoveltyJudgment.NOT_DISCUSSED

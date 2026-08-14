from __future__ import annotations

from datetime import date

from gear.contracts import PaperMetadata, ReviewRequest, ReviewStatus
from gear.review_contracts import (
    NoveltyAssessment,
    NoveltyJudgment,
    ReviewAspect,
    ReviewPoint,
    ReviewSummary,
    StructuredReview,
)
from gear.graph_context import build_graph_review_context
from gear.paper_compiler import PaperCompiler
from gear.review_pipeline import ServiceRegistry, review_paper
from gear.prior_art import PriorArtService, RelationClassifier
from gear.codex_critic import CodexCliCritic
from gear.trace import EvidenceStore
from gear.review_verifier import ReviewVerifier


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


def test_codex_cli_returns_entire_review(
    gear_config, paper_ir, calibration_factory
):
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
        "paper_id",
        "summary",
        "novelty",
        "strengths",
        "weaknesses",
        "questions",
    }
    assert "recommend" not in bundle.review_markdown.casefold()
    assert "graph calibration" not in bundle.review_markdown.casefold()
    assert services.evidence_store.validate_manifest() == []


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

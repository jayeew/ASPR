"""Default GEAR paper-to-structured-review runtime."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

from .config import GearConfig, load_config
from .contracts import (
    ActionRecord,
    FailureRecord,
    ParseStatus,
    ReviewRequest,
    ReviewStatus,
)
from .evidence_supervisor import EvidenceSupervisor
from .graph_context import build_graph_review_context
from .graph_prior import GraphPriorService
from .graph_prior_contracts import GraphPriorResult
from .paper_compiler import PaperCompiler
from .paper_extraction import HybridPaperExtractor, PaperRubricBuilder
from .prior_art import PriorArtService, RelationClassifier
from .process_diagnostic import diagnose_process
from .review_compiler import ReviewCompiler, calibration_evidence_key, render_markdown
from .review_contracts import (
    BranchReview,
    CriticRunMetadata,
    CriticSource,
    EvidenceBudget,
    GraphReviewContext,
    ReviewBundle,
    ReviewSource,
    StructuredReview,
)
from .review_controller import ReviewController
from .review_fusion import ReviewFusion
from .review_state import initialize_review_state, initialize_review_state_v2
from .review_verifier import ReviewVerifier
from .reviewers import ASPRQwenReviewer, CodexAgentReviewer
from .reviewers.base import build_graph_blind_payload
from .submission_calibration import SubmissionCalibrationService
from .trace import EvidenceStore, sha256_file, sha256_value


class StructuredReviewer(Protocol):
    model_name: str
    last_failures: list[str]

    @property
    def metadata(self) -> CriticRunMetadata: ...

    def review(
        self, paper_ir: Any, graph_context: GraphReviewContext
    ) -> StructuredReview: ...


@dataclass
class ServiceRegistry:
    evidence_store: EvidenceStore
    paper_compiler: Optional[Any] = None
    calibration_service: Optional[Any] = None
    reviewer: Optional[StructuredReviewer] = None
    prior_art: Optional[PriorArtService] = None
    relation_classifier: Optional[RelationClassifier] = None
    controller: Optional[ReviewController] = None
    compiler: Optional[ReviewCompiler] = None
    verifier: Optional[ReviewVerifier] = None
    agent_reviewer: Optional[Any] = None
    qwen_reviewer: Optional[Any] = None
    graph_prior: Optional[GraphPriorService] = None
    fusion: Optional[ReviewFusion] = None
    supervisor: Optional[EvidenceSupervisor] = None
    paper_extractor: Optional[HybridPaperExtractor] = None


def review_paper(
    request: ReviewRequest,
    *,
    output_dir: Optional[Path] = None,
    config: Optional[GearConfig] = None,
    services: Optional[ServiceRegistry] = None,
) -> ReviewBundle:
    """Run the independent three-source ASPR-ESR workflow."""
    resolved_config = config or load_config()
    target_dir = Path(
        output_dir or _default_output_dir(request, resolved_config)
    ).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    store = services.evidence_store if services else EvidenceStore(target_dir)
    if store.output_dir != target_dir:
        raise ValueError("injected EvidenceStore output_dir must match output_dir")
    paper_compiler = (
        services.paper_compiler
        if services and services.paper_compiler is not None
        else PaperCompiler(resolved_config)
    )
    paper_extractor = (
        services.paper_extractor
        if services and services.paper_extractor is not None
        else HybridPaperExtractor()
    )
    graph_prior_service = _graph_prior_service(resolved_config, services)
    agent_reviewer = (
        services.agent_reviewer
        if services and services.agent_reviewer is not None
        else None
    )
    legacy_reviewer: Optional[StructuredReviewer] = (
        services.reviewer
        if services and services.reviewer is not None and agent_reviewer is None
        else None
    )
    if agent_reviewer is None and legacy_reviewer is None:
        agent_reviewer = CodexAgentReviewer(resolved_config)
    qwen_reviewer = (
        services.qwen_reviewer
        if services and services.qwen_reviewer is not None
        else ASPRQwenReviewer(resolved_config)
    )
    prior_art = (
        services.prior_art
        if services and services.prior_art is not None
        else PriorArtService(resolved_config)
    )
    relation_classifier = (
        services.relation_classifier
        if services and services.relation_classifier is not None
        else RelationClassifier(resolved_config)
    )
    fusion = services.fusion if services and services.fusion else ReviewFusion()
    supervisor = (
        services.supervisor
        if services and services.supervisor
        else EvidenceSupervisor(resolved_config)
    )
    compiler = (
        services.compiler
        if services and services.compiler is not None
        else ReviewCompiler()
    )
    verifier = (
        services.verifier
        if services and services.verifier is not None
        else ReviewVerifier(resolved_config)
    )
    started = time.monotonic()
    paper_ir = paper_compiler.compile(request)
    paper_ir = paper_extractor.enrich(paper_ir)
    _append_stage(store, "paper_review_compiler", request, paper_ir, started)
    rubric = PaperRubricBuilder().build(paper_ir)
    started = time.monotonic()
    graph_prior = graph_prior_service.score(paper_ir, request.evidence_date)
    calibration_failure = graph_prior_service.last_failure
    calibration = graph_prior_service.last_packet
    if calibration is None:
        calibration = SubmissionCalibrationService(resolved_config).build_from_paper_ir(
            paper_ir,
            reason=calibration_failure or "graph_prior_unavailable",
        )
    graph_context = build_graph_review_context(calibration)
    _append_stage(
        store,
        "graph_prior",
        calibration,
        graph_prior,
        started,
        failure=calibration_failure,
    )
    _register_initial_evidence(store, paper_ir, calibration, graph_context)
    store.add_evidence("G:PRIOR", "graph_prior_public", graph_prior)
    if graph_prior_service.last_audit is not None:
        store.add_evidence(
            "G:AUDIT", "graph_prior_internal_audit", graph_prior_service.last_audit
        )
    state_v2 = initialize_review_state_v2(
        paper_ir,
        rubric,
        graph_prior,
        request.evidence_date,
        action_budget=EvidenceBudget(
            normal_per_claim_max=resolved_config.retrieval.normal_max,
            counterfactual_per_claim_max=resolved_config.retrieval.contrastive_max,
            citation_per_claim_max=resolved_config.retrieval.citation_expansion_max,
            relation_cards_max=resolved_config.retrieval.relation_cards_max,
            total_actions_max=resolved_config.retrieval.total_actions_max,
        ),
    )
    started = time.monotonic()
    if legacy_reviewer is not None:
        draft = legacy_reviewer.review(paper_ir, graph_context)
        agent_branch = BranchReview.from_structured(
            draft,
            source=ReviewSource.AGENT,
            model_id=legacy_reviewer.model_name,
            prompt_sha256="sha256:legacy_v1_prompt",
            input_sha256=sha256_value(build_graph_blind_payload(paper_ir, rubric)),
            failures=list(legacy_reviewer.last_failures),
        )
        agent_payload = build_graph_blind_payload(paper_ir, rubric)
        critic_metadata = legacy_reviewer.metadata
    else:
        if agent_reviewer is None:
            raise RuntimeError("required Agent Reviewer was not configured")
        agent_branch = agent_reviewer.review(paper_ir, rubric)
        draft = _structured_from_branch(agent_branch)
        agent_payload = dict(agent_reviewer.last_payload)
        critic_metadata = CriticRunMetadata(
            critic_source=(
                CriticSource.UNAVAILABLE
                if agent_branch.failures
                else (
                    CriticSource.CODEX_CLI
                    if resolved_config.model_backend == "codex_cli"
                    else CriticSource.OPENAI_COMPATIBLE_API
                )
            ),
            model_id=agent_branch.model_id,
        )
    _append_stage(
        store,
        "agent_reviewer",
        agent_payload,
        agent_branch,
        started,
        model=agent_branch.model_id,
        failure=";".join(agent_branch.failures) or None,
    )
    store.add_evidence(
        "A:BRANCH",
        "agent_branch_review",
        {"review": agent_branch, "input_payload": agent_payload},
    )
    qwen_branch = qwen_reviewer.review(paper_ir, rubric)
    if qwen_branch is not None:
        store.add_evidence(
            "Q:BRANCH",
            "qwen_branch_review",
            {"review": qwen_branch, "input_payload": qwen_reviewer.last_payload},
        )
    started = time.monotonic()
    state_v2, fusion_report = fusion.fuse(state_v2, agent_branch, qwen_branch)
    store.add_evidence("F:FUSION", "review_fusion", fusion_report)
    _append_stage(store, "review_fusion", agent_branch, fusion_report, started)
    store.snapshot_state(state_v2)
    started = time.monotonic()
    state_v2 = supervisor.resolve(
        state_v2,
        paper_ir,
        store,
        prior_art=prior_art,
        relation_classifier=relation_classifier,
    )
    _append_stage(store, "evidence_supervisor", fusion_report, state_v2, started)
    store.snapshot_state(state_v2)
    started = time.monotonic()
    structured = compiler.compile_v2(state_v2)
    _append_stage(store, "review_compiler_v2", state_v2, structured, started)
    started = time.monotonic()
    verification = verifier.verify_state(structured, state_v2, paper_ir, store)
    if not verification.passed and verifier.reject_failed_points(
        state_v2, verification
    ):
        structured = compiler.compile_v2(state_v2)
        verification = verifier.verify_state(structured, state_v2, paper_ir, store)
    _append_stage(
        store,
        "review_verifier_v2",
        structured,
        verification,
        started,
    )
    if verification.passed:
        structured = compiler.compile_verified(state_v2)
    markdown = render_markdown(structured)
    diagnostic = diagnose_process(state_v2)
    status = _bundle_status_v2(
        paper_ir,
        graph_prior,
        agent_branch,
        qwen_branch,
        resolved_config,
        verification,
        diagnostic.status,
    )
    state = initialize_review_state(
        paper_ir, graph_context, draft, request.evidence_date
    )
    state.finalized = True
    bundle = ReviewBundle(
        status=status,
        paper_ir=paper_ir,
        calibration=calibration,
        graph_context=graph_context,
        critic=critic_metadata,
        state=state,
        structured_review=structured,
        review_markdown=markdown,
        verification=verification,
        agent_review=agent_branch,
        qwen_review=qwen_branch,
        graph_prior=graph_prior,
        fusion_report=fusion_report,
        state_v2=state_v2,
        process_diagnostic=diagnostic.features,
    )
    return _persist_bundle(bundle, request, resolved_config, store, target_dir)


def _graph_prior_service(
    config: GearConfig,
    services: Optional[ServiceRegistry],
) -> GraphPriorService:
    if services and services.graph_prior is not None:
        return services.graph_prior
    if services and services.calibration_service is not None:
        return GraphPriorService(
            config, calibration_factory=lambda: services.calibration_service
        )
    return GraphPriorService(config)


def _structured_from_branch(branch: BranchReview) -> StructuredReview:
    return StructuredReview(
        paper_id=branch.paper_id,
        summary=branch.summary,
        novelty=branch.novelty,
        strengths=branch.strengths,
        weaknesses=branch.weaknesses,
        questions=branch.questions,
    )


def _register_initial_evidence(
    store: EvidenceStore,
    paper_ir: Any,
    calibration: Any,
    graph_context: GraphReviewContext,
) -> None:
    for span in paper_ir.spans:
        store.add_evidence(f"P:{span.span_id}", "paper_span", span)
    for part, kind, payload in (
        ("profile", "graph_calibration_profile", calibration.measurement),
        ("forecast", "graph_calibration_forecast", calibration.forecast),
        ("applicability", "graph_calibration_applicability", calibration.reliability),
        ("provenance", "graph_calibration_provenance", calibration.provenance),
    ):
        store.add_evidence(calibration_evidence_key(calibration, part), kind, payload)
    store.add_evidence("G:CTX", "graph_review_context", graph_context)


def _bundle_status(
    paper_ir: Any,
    graph_context: GraphReviewContext,
    critic: CriticRunMetadata,
    reviewer_failures: list[str],
    state_failures: list[FailureRecord],
    verification: Any,
) -> ReviewStatus:
    if paper_ir.parse_status == ParseStatus.UNAVAILABLE:
        return ReviewStatus.FAILED
    limited = (
        paper_ir.parse_status != ParseStatus.READY
        or graph_context.limited
        or critic.critic_source == CriticSource.UNAVAILABLE
        or bool(reviewer_failures)
        or bool(state_failures)
        or verification.limited
        or not verification.passed
    )
    return ReviewStatus.LIMITED if limited else ReviewStatus.COMPLETE


def _bundle_status_v2(
    paper_ir: Any,
    graph_prior: GraphPriorResult,
    agent: BranchReview,
    qwen: Optional[BranchReview],
    config: GearConfig,
    verification: Any,
    diagnostic_status: str,
) -> ReviewStatus:
    if paper_ir.parse_status == ParseStatus.UNAVAILABLE:
        return ReviewStatus.FAILED
    qwen_required_missing = (
        config.aspr_qwen.enabled and config.aspr_qwen.required and qwen is None
    )
    limited = (
        paper_ir.parse_status != ParseStatus.READY
        or bool(agent.failures)
        or graph_prior.status == "unavailable"
        or qwen_required_missing
        or verification.limited
        or not verification.passed
        or diagnostic_status != "sufficient"
    )
    return ReviewStatus.LIMITED if limited else ReviewStatus.COMPLETE


def _persist_bundle(
    bundle: ReviewBundle,
    request: ReviewRequest,
    config: GearConfig,
    store: EvidenceStore,
    target_dir: Path,
) -> ReviewBundle:
    paths = {
        "paper_ir": target_dir / "paper_ir.json",
        "paper_markdown": target_dir / "paper.md",
        "agent_review": target_dir / "agent_review.json",
        "graph_prior": target_dir / "graph_prior.json",
        "graph_prior_audit": target_dir / "graph_prior_audit.json",
        "fusion_report": target_dir / "fusion_report.json",
        "review_state": target_dir / "review_state.json",
        "process_diagnostic": target_dir / "process_diagnostic.json",
        "structured_review": target_dir / "review.json",
        "validation_report": target_dir / "validation_report.json",
        "review_markdown": target_dir / "review.md",
        "review_bundle": target_dir / "review_bundle.json",
        "run_manifest": target_dir / "run_manifest.json",
    }
    if bundle.qwen_review is not None:
        paths["qwen_review"] = target_dir / "qwen_review.json"
    output_files = {name: str(path) for name, path in paths.items()}
    bundle = bundle.model_copy(update={"output_files": output_files})
    _write_model(paths["paper_ir"], bundle.paper_ir)
    paths["paper_markdown"].write_text(
        bundle.paper_ir.markdown + "\n", encoding="utf-8"
    )
    if bundle.agent_review is not None:
        _write_model(paths["agent_review"], bundle.agent_review)
    if bundle.graph_prior is not None:
        _write_model(paths["graph_prior"], bundle.graph_prior)
    audit_record = store.get("G:AUDIT")
    paths["graph_prior_audit"].write_text(
        json.dumps(
            audit_record.payload if audit_record is not None else {},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if bundle.fusion_report is not None:
        _write_model(paths["fusion_report"], bundle.fusion_report)
    if bundle.state_v2 is not None:
        _write_model(paths["review_state"], bundle.state_v2)
        diagnostic = diagnose_process(bundle.state_v2)
        _write_model(paths["process_diagnostic"], diagnostic)
    if bundle.qwen_review is not None:
        _write_model(paths["qwen_review"], bundle.qwen_review)
    _write_model(paths["structured_review"], bundle.structured_review)
    _write_model(paths["validation_report"], bundle.verification)
    paths["review_markdown"].write_text(bundle.review_markdown, encoding="utf-8")
    paths["review_bundle"].write_text(
        bundle.model_dump_json(
            indent=2,
            exclude={"calibration", "graph_context", "state"},
        )
        + "\n",
        encoding="utf-8",
    )
    artifact_hashes = {
        name: sha256_file(path)
        for name, path in paths.items()
        if name != "run_manifest" and path.is_file()
    }
    store.write_manifest(
        {
            "contract": "aspr_gear_run_manifest",
            "schema_version": "aspr_gear",
            "config_version": config.config_version,
            "evidence_policy": config.evidence_policy,
            "deprecated_fig4_to_fig10_used": False,
            "request_sha256": sha256_value(request),
            "paper_sha256": bundle.paper_ir.paper_sha256,
            "state_sha256": (
                sha256_value(bundle.state) if bundle.state is not None else None
            ),
            "state_v2_sha256": (
                sha256_value(bundle.state_v2) if bundle.state_v2 is not None else None
            ),
            "agent_prompt_sha256": (
                bundle.agent_review.prompt_sha256
                if bundle.agent_review is not None
                else None
            ),
            "agent_input_sha256": (
                bundle.agent_review.input_sha256
                if bundle.agent_review is not None
                else None
            ),
            "qwen_prompt_sha256": (
                bundle.qwen_review.prompt_sha256
                if bundle.qwen_review is not None
                else None
            ),
            "qwen_input_sha256": (
                bundle.qwen_review.input_sha256
                if bundle.qwen_review is not None
                else None
            ),
            "graph_prior_status": (
                bundle.graph_prior.status if bundle.graph_prior is not None else None
            ),
            "critic_source": bundle.critic.critic_source.value,
            "status": bundle.status.value,
            "output_files": output_files,
            "output_file_sha256": artifact_hashes,
        }
    )
    return bundle


def _write_model(path: Path, model: Any) -> None:
    path.write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _default_output_dir(request: ReviewRequest, config: GearConfig) -> Path:
    identity = f"current|{Path(request.paper_path).resolve()}|{request.evidence_date.isoformat()}"
    run_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return config.resolve_path(config.output_root) / run_id


def _append_stage(
    store: EvidenceStore,
    stage: str,
    input_value: Any,
    output_value: Any,
    started: float,
    *,
    model: Optional[str] = None,
    failure: Optional[str] = None,
) -> None:
    input_hash = sha256_value(input_value)
    output_hash = sha256_value(output_value)
    identity = f"{stage}|{input_hash}|{output_hash}"
    store.append_action(
        ActionRecord(
            action_id="STAGE-"
            + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:18],
            stage=stage,
            input_sha256=input_hash,
            output_sha256=output_hash,
            model=model,
            duration_ms=int((time.monotonic() - started) * 1000),
            failure=failure,
        )
    )


__all__ = ["ServiceRegistry", "StructuredReviewer", "review_paper"]

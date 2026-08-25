"""Default GEAR paper-to-structured-review runtime."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .config import GearConfig, load_config
from .contracts import (
    ActionRecord,
    FailureRecord,
    ParseStatus,
    ReviewRequest,
    ReviewStatus,
)
from .evidence_supervisor import EvidenceSupervisor, build_retrieval_claim
from .graph_guidance import GraphGuidancePlanner, is_graph_guidance_target
from .graph_prior import (
    GraphScorer,
    GraphService,
    GraphUnavailableError,
    graph_runtime_packet,
)
from .graph_prior_contracts import GraphResourceCapsV1, GraphRuntimePacketV1
from .grounding import GroundingWorkflow
from .paper_compiler import PaperCompiler
from .paper_extraction import (
    HybridPaperExtractor,
    PaperRubricBuilder,
    configured_paper_extractor,
)
from .prior_art import PriorArtService, RelationClassifier
from .process_diagnostic import diagnose_process
from .review_compiler import ReviewCompiler, render_markdown
from .review_contracts import (
    BranchReview,
    CriticRunMetadata,
    CriticSource,
    EvidenceBudget,
    GraphReviewContext,
    ReviewBundle,
    ReviewPhase,
    ReviewSource,
    StructuredReview,
)
from .review_controller import ReviewController
from .review_fusion import ReviewFusion
from .review_state import initialize_review_state_v3
from .review_verifier import ReviewVerifier
from .reviewers import ASPRQwenReviewer, CodexAgentReviewer
from .reviewers.base import build_graph_blind_payload
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
    paper_compiler: Any | None = None
    calibration_service: Any | None = None
    reviewer: StructuredReviewer | None = None
    prior_art: PriorArtService | None = None
    relation_classifier: RelationClassifier | None = None
    controller: ReviewController | None = None
    compiler: ReviewCompiler | None = None
    verifier: ReviewVerifier | None = None
    agent_reviewer: Any | None = None
    qwen_reviewer: Any | None = None
    graph_prior: GraphScorer | None = None
    graph_scorer: GraphScorer | None = None
    fusion: ReviewFusion | None = None
    supervisor: EvidenceSupervisor | None = None
    paper_extractor: HybridPaperExtractor | None = None


def review_paper(
    request: ReviewRequest,
    *,
    output_dir: Path | None = None,
    config: GearConfig | None = None,
    services: ServiceRegistry | None = None,
    full_artifacts: bool = True,
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
        else (
            HybridPaperExtractor()
            if services is not None
            else configured_paper_extractor(resolved_config)
        )
    )
    graph_scorer = _graph_scorer(resolved_config, services)
    agent_reviewer = (
        services.agent_reviewer
        if services and services.agent_reviewer is not None
        else None
    )
    legacy_reviewer: StructuredReviewer | None = (
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
    graph_result: GraphRuntimePacketV1 | None = None
    graph_failure: str | None = None
    try:
        candidate = graph_scorer.score(paper_ir, request.evidence_date)
        graph_result = graph_runtime_packet(candidate)
        if graph_result.paper_id != paper_ir.paper_id:
            raise ValueError("Graph result paper_id mismatch")
    except (GraphUnavailableError, OSError, RuntimeError, TypeError, ValueError) as exc:
        graph_result = None
        graph_failure = f"{type(exc).__name__}:{exc}"
    _append_stage(
        store,
        "graph_result",
        paper_ir,
        graph_result or {},
        started,
        failure=graph_failure,
    )
    _register_initial_evidence(store, paper_ir)
    if graph_result is not None:
        store.add_evidence("G:RESULT", "graph_result", graph_result)
    grounding_report = (
        GroundingWorkflow().run(paper_ir, store) if full_artifacts else None
    )
    state_v3 = initialize_review_state_v3(
        paper_ir,
        rubric,
        graph_result,
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
        draft = legacy_reviewer.review(
            paper_ir, _legacy_graph_context(paper_ir.paper_id, graph_result)
        )
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
    if (
        resolved_config.aspr_qwen.enabled
        and resolved_config.aspr_qwen.required
        and qwen_branch is None
    ):
        state_v3.failures.append(
            FailureRecord(stage="qwen_reviewer", reason="qwen_required_unavailable")
        )
    if qwen_branch is not None:
        store.add_evidence(
            "Q:BRANCH",
            "qwen_branch_review",
            {"review": qwen_branch, "input_payload": qwen_reviewer.last_payload},
        )
    started = time.monotonic()
    state_v3, fusion_report = fusion.fuse(state_v3, agent_branch, qwen_branch)
    store.add_evidence("F:FUSION", "review_fusion", fusion_report)
    _append_stage(store, "review_fusion", agent_branch, fusion_report, started)
    if state_v3.graph_result is not None:
        guidance_config = resolved_config.graph_guidance
        search_frames = {}
        prepare_search_frame = getattr(prior_art, "prepare_search_frame", None)
        for point in state_v3.canonical_points.values():
            if not (is_graph_guidance_target(point) and callable(prepare_search_frame)):
                continue
            try:
                claim, target_span = build_retrieval_claim(point, paper_ir)
                frame = prepare_search_frame(claim, target_span, paper_ir)
                search_frames[point.point_id] = frame
                store.add_evidence(
                    f"QF:{claim.claim_id}", "scientific_search_frame", frame
                )
            except (TypeError, ValueError) as exc:
                point.validation_notes.append(
                    f"graph_claim_alignment_frame_degraded:{exc}"
                )
        guidance_plan = GraphGuidancePlanner(
            resource_caps=GraphResourceCapsV1(
                provider_searches=guidance_config.provider_searches,
                direct_fetches=guidance_config.direct_fetches,
                neighbor_expansions=guidance_config.neighbor_expansions,
                fulltext_candidates=guidance_config.fulltext_candidates,
                relation_classifications=guidance_config.relation_classifications,
            ),
            policy_version=guidance_config.policy_version,
        ).plan(
            state_v3,
            search_frames=search_frames,
            enable_profile=guidance_config.profile_enabled,
            enable_topology=guidance_config.topology_enabled,
        )
        state_v3.graph_guidance_plan = guidance_plan
        if state_v3.resource_ledger is not None:
            state_v3.resource_ledger.caps = guidance_plan.resource_caps
        store.add_evidence("G:PLAN", "graph_guidance_plan", guidance_plan)
    if full_artifacts:
        store.snapshot_state(state_v3)
    started = time.monotonic()
    state_v3 = supervisor.resolve(
        state_v3,
        paper_ir,
        store,
        prior_art=prior_art,
        relation_classifier=relation_classifier,
    )
    if state_v3.resource_ledger is not None:
        store.add_evidence("G:LEDGER", "resource_ledger", state_v3.resource_ledger)
    _append_stage(store, "evidence_supervisor", fusion_report, state_v3, started)
    if full_artifacts:
        store.snapshot_state(state_v3)
    started = time.monotonic()
    structured = compiler.compile_v3(state_v3)
    _append_stage(store, "review_compiler_v3", state_v3, structured, started)
    started = time.monotonic()
    verification = verifier.verify_state(structured, state_v3, paper_ir, store)
    if not verification.passed and verifier.reject_failed_points(
        state_v3, verification
    ):
        structured = compiler.compile_v3(state_v3)
        deterministic_issues = verifier._deterministic_v2_issues(
            structured, state_v3, paper_ir, store
        )
        verification = verification.model_copy(
            update={
                "passed": not deterministic_issues,
                "issues": [
                    issue
                    for issue in verification.issues
                    if issue.code.startswith("semantic_verifier_")
                ]
                + deterministic_issues,
                "unsupported_major_count": sum(
                    issue.code == "unsupported_major" for issue in deterministic_issues
                ),
            }
        )
    _append_stage(
        store,
        "review_verifier_v3",
        structured,
        verification,
        started,
    )
    if verification.passed:
        # A delete-only verifier repair recomputes the report without making a
        # second semantic model call.  Keep the state machine in sync with that
        # successful repaired report before invoking the final compiler.
        state_v3.phase = ReviewPhase.VERIFIED
        state_v3.process_features.semantic_verifier_passed = (
            verification.semantic_verification_available
        )
        structured = compiler.compile_verified(state_v3)
    markdown = render_markdown(structured)
    diagnostic = diagnose_process(state_v3, paper_ir)
    status = _bundle_status_v3(
        paper_ir,
        graph_result,
        agent_branch,
        qwen_branch,
        resolved_config,
        verification,
        diagnostic.status,
    )
    bundle = ReviewBundle(
        status=status,
        paper_ir=paper_ir,
        critic=critic_metadata,
        structured_review=structured,
        review_markdown=markdown,
        verification=verification,
        agent_review=agent_branch,
        qwen_review=qwen_branch,
        graph_result=graph_result,
        fusion_report=fusion_report,
        state_v3=state_v3,
        process_diagnostic=diagnostic.model_dump(mode="json"),
        grounding_report=(
            grounding_report.model_dump(mode="json")
            if grounding_report is not None
            else None
        ),
    )
    return _persist_bundle(
        bundle,
        request,
        resolved_config,
        store,
        target_dir,
        full_artifacts=full_artifacts,
    )


def _graph_scorer(
    config: GearConfig,
    services: ServiceRegistry | None,
) -> GraphScorer:
    if services and services.graph_scorer is not None:
        return services.graph_scorer
    if services and services.graph_prior is not None:
        return services.graph_prior
    if services and services.calibration_service is not None:
        return GraphService(
            config, calibration_factory=lambda: services.calibration_service
        )
    return GraphService(config)


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
) -> None:
    for span in paper_ir.spans:
        store.add_evidence(f"P:{span.span_id}", "paper_span", span)


def _legacy_graph_context(
    paper_id: str, graph_result: GraphRuntimePacketV1 | None
) -> GraphReviewContext:
    """Compatibility-only projection for an injected legacy reviewer."""
    return GraphReviewContext(
        paper_id=paper_id,
        p_uptake=graph_result.p_uptake if graph_result is not None else None,
        conditional_diffusion=(
            graph_result.conditional_diffusion if graph_result is not None else None
        ),
        d5_percentile=(graph_result.score_0_100 if graph_result is not None else None),
        applicability_mode="v3_result" if graph_result is not None else "unavailable",
        feature_coverage=(
            graph_result.feature_coverage if graph_result is not None else 0.0
        ),
        limited=graph_result is None,
    )


def _bundle_status_v3(
    paper_ir: Any,
    graph_result: GraphRuntimePacketV1 | None,
    agent: BranchReview,
    qwen: BranchReview | None,
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
        or graph_result is None
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
    *,
    full_artifacts: bool,
) -> ReviewBundle:
    paths = {
        "paper_ir": target_dir / "paper_ir.json",
        "paper_markdown": target_dir / "paper.md",
        "agent_review": target_dir / "agent_review.json",
        "fusion_report": target_dir / "fusion_report.json",
        "review_state": target_dir / "review_state.json",
        "process_diagnostic": target_dir / "process_diagnostic.json",
        "grounding_report": target_dir / "grounding_report.json",
        "structured_review": target_dir / "review.json",
        "validation_report": target_dir / "validation_report.json",
        "review_markdown": target_dir / "review.md",
        "review_bundle": target_dir / "review_bundle.json",
        "run_manifest": target_dir / "run_manifest.json",
    }
    if not full_artifacts:
        for name in (
            "paper_ir",
            "paper_markdown",
            "agent_review",
            "fusion_report",
            "review_state",
            "process_diagnostic",
            "grounding_report",
        ):
            paths.pop(name)
    if bundle.graph_result is not None:
        paths["graph_runtime_packet"] = target_dir / "graph_runtime_packet.json"
    if bundle.qwen_review is not None and full_artifacts:
        paths["qwen_review"] = target_dir / "qwen_review.json"
    output_files = {name: str(path) for name, path in paths.items()}
    bundle = bundle.model_copy(update={"output_files": output_files})
    if "paper_ir" in paths:
        _write_model(paths["paper_ir"], bundle.paper_ir)
    if "paper_markdown" in paths:
        paths["paper_markdown"].write_text(
            bundle.paper_ir.markdown + "\n", encoding="utf-8"
        )
    if bundle.agent_review is not None and "agent_review" in paths:
        _write_model(paths["agent_review"], bundle.agent_review)
    if bundle.graph_result is not None:
        _write_model(paths["graph_runtime_packet"], bundle.graph_result)
    if bundle.fusion_report is not None and "fusion_report" in paths:
        _write_model(paths["fusion_report"], bundle.fusion_report)
    if bundle.state_v3 is not None and "review_state" in paths:
        _write_model(paths["review_state"], bundle.state_v3)
        diagnostic = diagnose_process(bundle.state_v3, bundle.paper_ir)
        _write_model(paths["process_diagnostic"], diagnostic)
    if bundle.grounding_report is not None and "grounding_report" in paths:
        paths["grounding_report"].write_text(
            json.dumps(bundle.grounding_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if bundle.qwen_review is not None:
        _write_model(paths["qwen_review"], bundle.qwen_review)
    _write_model(paths["structured_review"], bundle.structured_review)
    _write_model(paths["validation_report"], bundle.verification)
    paths["review_markdown"].write_text(bundle.review_markdown, encoding="utf-8")
    paths["review_bundle"].write_text(
        bundle.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    artifact_hashes = {
        name: sha256_file(path)
        for name, path in paths.items()
        if name != "run_manifest" and path.is_file()
    }
    store.write_manifest(
        {
            "contract": "aspr_gear_run_manifest_v3",
            "schema_version": "aspr_gear",
            "config_version": config.config_version,
            "evidence_policy": config.evidence_policy,
            "deprecated_fig4_to_fig10_used": False,
            "request_sha256": sha256_value(request),
            "paper_sha256": bundle.paper_ir.paper_sha256,
            "state_v3_sha256": (
                sha256_value(bundle.state_v3) if bundle.state_v3 is not None else None
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
            "graph_available": bundle.graph_result is not None,
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
    model: str | None = None,
    failure: str | None = None,
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

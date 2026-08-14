"""Default GEAR paper-to-structured-review runtime."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

from .calibration import CalibrationService
from .codex_critic import CodexCliCritic
from .review_compiler import ReviewCompiler, calibration_evidence_key, render_markdown
from .config import GearConfig, load_config
from .contracts import (
    ActionRecord,
    CalibrationMode,
    FailureRecord,
    ParseStatus,
    ReviewRequest,
    ReviewStatus,
)
from .review_contracts import (
    CriticRunMetadata,
    CriticSource,
    GraphReviewContext,
    ReviewBundle,
    StructuredReview,
)
from .review_controller import ReviewController
from .graph_context import build_graph_review_context
from .paper_compiler import PaperCompiler
from .prior_art import PriorArtService, RelationClassifier
from .review_state import initialize_review_state
from .submission_calibration import SubmissionCalibrationService
from .trace import EvidenceStore, sha256_file, sha256_value
from .review_verifier import ReviewVerifier


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


def review_paper(
    request: ReviewRequest,
    *,
    output_dir: Optional[Path] = None,
    config: Optional[GearConfig] = None,
    services: Optional[ServiceRegistry] = None,
) -> ReviewBundle:
    """Run the evidence-traceable GEAR review workflow."""
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
    calibration_service = (
        services.calibration_service
        if services and services.calibration_service is not None
        else CalibrationService(resolved_config)
    )
    reviewer: StructuredReviewer = (
        services.reviewer
        if services and services.reviewer is not None
        else CodexCliCritic(resolved_config)
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
    controller = (
        services.controller
        if services and services.controller is not None
        else ReviewController(resolved_config)
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
    _append_stage(store, "paper_review_compiler", request, paper_ir, started)
    started = time.monotonic()
    calibration_failure: Optional[str] = None
    try:
        calibration = calibration_service.build_packet(
            paper_ir, cutoff_date=request.evidence_date
        )
        if calibration.reliability.mode == CalibrationMode.UNAVAILABLE:
            calibration = SubmissionCalibrationService(
                resolved_config
            ).build_from_paper_ir(
                paper_ir,
                reason=";".join(calibration.reliability.quality_flags)
                or "exact_calibration_unavailable",
            )
    except Exception as exc:  # boundary converts service failure to limited packet.
        calibration_failure = f"{type(exc).__name__}:{exc}"
        calibration = SubmissionCalibrationService(resolved_config).build_from_paper_ir(
            paper_ir,
            reason=f"calibration_service_failed:{type(exc).__name__}",
        )
    graph_context = build_graph_review_context(calibration)
    _append_stage(
        store,
        "graph_context",
        calibration,
        graph_context,
        started,
        failure=calibration_failure,
    )
    _register_initial_evidence(store, paper_ir, calibration, graph_context)
    started = time.monotonic()
    draft = reviewer.review(paper_ir, graph_context)
    critic_metadata = reviewer.metadata
    _append_stage(
        store,
        "structured_reviewer",
        {"paper_sha256": paper_ir.paper_sha256, "graph": graph_context},
        draft,
        started,
        model=reviewer.model_name,
        failure=";".join(reviewer.last_failures) or None,
    )
    store.add_evidence(
        f"D:{paper_ir.paper_id}",
        "structured_review_draft",
        {"review": draft, "critic": critic_metadata},
    )
    state = initialize_review_state(
        paper_ir, graph_context, draft, request.evidence_date
    )
    if calibration_failure:
        state.failure_ledger.append(
            FailureRecord(
                stage="calibration",
                reason=calibration_failure,
                recoverable=True,
            )
        )
    state.failure_ledger.extend(
        FailureRecord(stage="structured_reviewer", reason=reason, recoverable=True)
        for reason in reviewer.last_failures
    )
    store.snapshot_state(state)
    started = time.monotonic()
    state = controller.run(
        state,
        paper_ir,
        store,
        prior_art=prior_art,
        relation_classifier=relation_classifier,
    )
    _append_stage(store, "review_review_controller", draft, state, started)
    store.snapshot_state(state)
    started = time.monotonic()
    structured = compiler.compile(state)
    _append_stage(store, "review_review_compiler", state, structured, started)
    started = time.monotonic()
    verification = verifier.verify(structured, state, paper_ir, store)
    _append_stage(
        store,
        "review_review_verifier",
        structured,
        verification,
        started,
    )
    markdown = render_markdown(structured)
    status = _bundle_status(
        paper_ir,
        graph_context,
        critic_metadata,
        reviewer.last_failures,
        state.failure_ledger,
        verification,
    )
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
    )
    return _persist_bundle(bundle, request, resolved_config, store, target_dir)


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
        "calibration_packet": target_dir / "calibration_packet.json",
        "graph_context": target_dir / "graph_context.json",
        "structured_review": target_dir / "review.json",
        "validation_report": target_dir / "validation_report.json",
        "review_markdown": target_dir / "review.md",
        "review_bundle": target_dir / "review_bundle.json",
        "run_manifest": target_dir / "run_manifest.json",
    }
    output_files = {name: str(path) for name, path in paths.items()}
    bundle = bundle.model_copy(update={"output_files": output_files})
    _write_model(paths["paper_ir"], bundle.paper_ir)
    paths["paper_markdown"].write_text(
        bundle.paper_ir.markdown + "\n", encoding="utf-8"
    )
    _write_model(paths["calibration_packet"], bundle.calibration)
    _write_model(paths["graph_context"], bundle.graph_context)
    _write_model(paths["structured_review"], bundle.structured_review)
    _write_model(paths["validation_report"], bundle.verification)
    paths["review_markdown"].write_text(bundle.review_markdown, encoding="utf-8")
    _write_model(paths["review_bundle"], bundle)
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
            "state_sha256": sha256_value(bundle.state),
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
    identity = (
        f"current|{Path(request.paper_path).resolve()}|{request.evidence_date.isoformat()}"
    )
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

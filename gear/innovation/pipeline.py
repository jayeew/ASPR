"""Resumable v2 branches sharing claims, with append-only per-claim evidence."""

from __future__ import annotations

import json
import os
import sqlite3
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path
from uuid import uuid4

from gear.artifacts import read_model, write_json, write_model
from gear.claim_attribution import ClaimGraphRuntime
from gear.config import GearConfig
from gear.contracts import PaperIR, ScientificSearchFrame
from gear.evidence_supervisor import EvidenceSupervisor
from gear.local_ranking import LocalScientificRanker
from gear.review_contracts import (
    BranchStatus,
    GearClaim,
    GraphClaim,
    GraphFactCard,
    InnovationPaperInput,
)
from gear.trace import EvidenceStore, sha256_value

from .analysis import assess, evidence_payloads
from .contracts import AnalysisResult, Assessment, ClaimSet
from .joint_graph import run_joint
from .locking import stage_lock
from .shared import prepare_shared

ERRORS = (
    ImportError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    KeyError,
    sqlite3.Error,
)


def _manuscript(paper: PaperIR, root: Path) -> dict[str, object]:
    store = EvidenceStore(root)
    for span in paper.spans:
        store.add_evidence(f"P:{span.span_id}", "manuscript_span", span)
    return evidence_payloads(root)


def _assess_saved(
    config: GearConfig,
    claim_id: str,
    text: str,
    sources: dict[str, object],
    mode: str,
    path: Path,
) -> Assessment:
    result = assess(config, claim_id, text, sources, mode)
    write_model(path, result)
    return result


def _gear_assess(
    config: GearConfig,
    claim: GearClaim,
    paper: PaperIR,
    item: InnovationPaperInput,
    root: Path,
    claim_dir: Path,
    local_ranker: LocalScientificRanker | None,
    prepared_search_root: Path | None,
) -> Assessment:
    from .usage import log_progress, progress_scope

    with progress_scope(f"claim={claim.claim_id}"):
        log_progress("[Claim处理开始]")
        recovery_path = claim_dir / "recovery_source.json"
        recovery = (
            json.loads(recovery_path.read_text()) if recovery_path.exists() else {}
        )
        if (claim_dir / "evidence_trace.jsonl").exists() and not (
            claim_dir / "gear_card.json"
        ).exists():
            archive = root / "gear_attempts" / claim_dir.name / uuid4().hex
            archive.parent.mkdir(parents=True, exist_ok=True)
            claim_dir.rename(archive)
            if recovery:
                write_json(recovery_path, recovery)
        sources = _manuscript(paper, claim_dir)
        store = EvidenceStore(claim_dir)
        card_path = claim_dir / "gear_card.json"
        if card_path.exists():
            from gear.review_contracts import GearClaimCard

            card = read_model(card_path, GearClaimCard)
            log_progress("[Claim证据复用] 已存在GEAR card")
        else:
            write_json(
                claim_dir / "retrieval_policy.json",
                {
                    "historical_pdf_enabled": config.retrieval.openalex_pdf_enabled,
                    "external_fulltext_enabled": config.retrieval.external_fulltext_enabled,
                    "external_fulltext_max_works": config.retrieval.external_fulltext_max_works,
                    "coverage_protocol": "normal_contrastive_reserved_v2",
                    "candidate_budget": config.retrieval.fulltext_max,
                    "recovery_source": recovery.get("source"),
                },
            )
            supervisor = EvidenceSupervisor(
                config,
                store,
                local_ranker=local_ranker,
            )
            if prepared_search_root is not None:
                frame_path = (
                    prepared_search_root
                    / claim.claim_id.rsplit("::", 1)[-1]
                    / "search_frame.json"
                )
                if not frame_path.is_file():
                    raise FileNotFoundError(
                        f"Prepared search frame is missing: {frame_path}"
                    )
                frame = read_model(frame_path, ScientificSearchFrame)
                supervisor.prior_art._frames[claim.claim_id] = frame
                log_progress("[检索准备复用] %s", frame_path)
            if recovery:
                card = supervisor.evaluate(
                    claim,
                    paper,
                    item.cutoff_date,
                    recovery_source=Path(recovery["source"]),
                )
            else:
                card = supervisor.evaluate(claim, paper, item.cutoff_date)
            write_model(card_path, card)
            write_json(
                claim_dir / "execution.json",
                {
                    "status": "evidence_complete",
                    "errors": list(dict.fromkeys(supervisor.execution_errors)),
                },
            )
        store.add_evidence(f"GEAR:{claim.claim_id}", "gear_claim_card", card)
        sources = evidence_payloads(claim_dir)
        log_progress("[Claim总结开始]")
        assessment = _assess_saved(
            config,
            claim.claim_id,
            claim.normalized_claim_text,
            sources,
            "gear",
            claim_dir / "assessment.json",
        )
        execution_path = claim_dir / "execution.json"
        if execution_path.exists():
            execution = json.loads(execution_path.read_text(encoding="utf-8"))
            execution["status"] = "failed" if execution.get("errors") else "complete"
            write_json(execution_path, execution)
        log_progress("[Claim处理完成] status=%s", assessment.overall_stance.value)
        return assessment


def run_branch(
    item: InnovationPaperInput,
    root: Path,
    config: GearConfig,
    paper: PaperIR,
    shared: ClaimSet,
    mode: str,
    graph_root: Path,
    embedding_model: Path,
    prepared_search_root: Path | None = None,
    *,
    shared_local_ranker: LocalScientificRanker | None = None,
) -> AnalysisResult:
    with stage_lock(root / ".locks" / mode):
        return _run_branch(
            item,
            root,
            config,
            paper,
            shared,
            mode,
            graph_root,
            embedding_model,
            prepared_search_root,
            shared_local_ranker,
        )


def _run_branch(
    item: InnovationPaperInput,
    root: Path,
    config: GearConfig,
    paper: PaperIR,
    shared: ClaimSet,
    mode: str,
    graph_root: Path,
    embedding_model: Path,
    prepared_search_root: Path | None = None,
    shared_local_ranker: LocalScientificRanker | None = None,
) -> AnalysisResult:
    branch = root / mode
    result_path = branch / "analysis.json"
    fingerprint = (
        sha256_value(shared) if config.resume_fingerprint_checks_enabled else ""
    )
    if mode == "graph":
        policy = {
            "version": "threshold_parent_path_v2",
            "top_k": config.graph_top_k,
            "min_similarity": config.graph_min_similarity,
        }
        policy_path = branch / "insertion_policy.json"
        if policy_path.exists():
            if json.loads(policy_path.read_text()) != policy:
                raise ValueError(
                    "Graph insertion policy changed; use a new run directory"
                )
        elif result_path.exists() or any(branch.glob("*/evidence_trace.jsonl")):
            raise ValueError(
                "Legacy graph cache lacks threshold policy; use a new run directory"
            )
        write_json(policy_path, policy)
    if result_path.exists():
        saved = read_model(result_path, AnalysisResult)
        if (
            config.resume_fingerprint_checks_enabled
            and saved.claim_fingerprint != fingerprint
        ):
            raise ValueError("Branch/shared claim fingerprint mismatch")
        if len(saved.assessments) == len(shared.claims):
            return saved
    runtime = (
        ClaimGraphRuntime(
            graph_root, embedding_model, config.graph_top_k, config.graph_min_similarity
        )
        if mode == "graph"
        else None
    )
    result = AnalysisResult(
        paper_id=item.paper_id,
        system=mode,
        claim_fingerprint=fingerprint,
        status=BranchStatus.COMPLETE,
    )
    worker_count = (
        max(1, min(8, int(os.environ.get("GEAR_CLAIM_WORKERS", "2"))))
        if mode == "gear"
        else 8
    )
    local_ranker = (
        shared_local_ranker
        or LocalScientificRanker(
            config.retrieval.recall_model_path,
            config.retrieval.reranker_model_path,
        )
        if mode == "gear"
        and config.retrieval.local_recall_enabled
        and config.retrieval.local_reranker_enabled
        else None
    )
    executor = ThreadPoolExecutor(max_workers=worker_count)
    jobs: list[tuple[str, Future[Assessment]]] = []
    try:
        for claim in shared.claims:
            claim_dir = branch / claim.claim_id.rsplit("::", 1)[-1]
            saved_path = claim_dir / "assessment.json"
            try:
                if saved_path.exists():
                    result.assessments.append(read_model(saved_path, Assessment))
                    continue
                if mode == "graph":
                    assert runtime is not None
                    spans = paper.span_map()
                    target = GraphClaim(
                        claim_id=claim.claim_id,
                        paper_id=item.paper_id,
                        claim_type=claim.claim_type,
                        claim_text=claim.normalized_claim_text,
                        source_sentence_ids=claim.source_span_ids,
                        source_sentence_texts=[
                            spans[x].text for x in claim.source_span_ids
                        ],
                    )
                    store = EvidenceStore(claim_dir)
                    key = f"GRAPH:{claim.claim_id}"
                    if key in store._evidence:
                        fact = GraphFactCard.model_validate(
                            store._evidence[key].payload
                        )
                        if fact.claim != target:
                            raise ValueError(
                                "Stored graph fact has different claim input"
                            )
                        if fact.insertion_policy != runtime.insertion_policy:
                            raise ValueError(
                                "Cached graph fact uses a different insertion policy"
                            )
                    else:
                        fact = runtime.insert(target, item)
                        store.add_evidence(key, "graph_fact", fact)
                    sources = evidence_payloads(claim_dir)
                    if not fact.neighbors:
                        raise ValueError("No eligible historical graph neighbors")
                else:
                    future = executor.submit(
                        copy_context().run,
                        _gear_assess,
                        config,
                        claim,
                        paper,
                        item,
                        root,
                        claim_dir,
                        local_ranker,
                        prepared_search_root,
                    )
                    jobs.append((claim.claim_id, future))
                    continue
                future = executor.submit(
                    copy_context().run,
                    _assess_saved,
                    config,
                    claim.claim_id,
                    claim.normalized_claim_text,
                    sources,
                    mode,
                    saved_path,
                )
                jobs.append((claim.claim_id, future))
            except ERRORS as exc:
                result.limitations.append(
                    f"{claim.claim_id}:{type(exc).__name__}:{exc}"
                )
                write_json(claim_dir / "failure.json", {"error": str(exc)})
        if runtime is not None:
            runtime.close()
            runtime = None
        for claim_id, future in jobs:
            try:
                result.assessments.append(future.result())
            except ERRORS as exc:
                result.limitations.append(f"{claim_id}:{type(exc).__name__}:{exc}")
        order = {claim.claim_id: index for index, claim in enumerate(shared.claims)}
        result.assessments.sort(key=lambda row: order[row.claim_id])
    finally:
        executor.shutdown(wait=True)
        if local_ranker is not None and local_ranker is not shared_local_ranker:
            local_ranker.close()
        if runtime:
            runtime.close()
    if result.limitations or any(x.limitations for x in result.assessments):
        result.status = BranchStatus.LIMITED
    write_model(result_path, result)
    return result


def fuse(
    item: InnovationPaperInput, root: Path, config: GearConfig, shared: ClaimSet
) -> AnalysisResult:
    with stage_lock(root / ".locks" / "fusion"):
        return _fuse(item, root, config, shared)


def _fuse(
    item: InnovationPaperInput, root: Path, config: GearConfig, shared: ClaimSet
) -> AnalysisResult:
    fingerprint = (
        sha256_value(shared) if config.resume_fingerprint_checks_enabled else ""
    )
    branches = {
        mode: read_model(root / mode / "analysis.json", AnalysisResult)
        for mode in ("gear", "graph")
    }
    if any(row.paper_id != item.paper_id for row in branches.values()) or (
        config.resume_fingerprint_checks_enabled
        and any(row.claim_fingerprint != fingerprint for row in branches.values())
    ):
        raise ValueError("Cannot fuse different paper/claim runs")
    fusion_root = root / "fusion"
    input_hash = sha256_value(branches)
    input_path = fusion_root / "inputs.json"
    recorded = json.loads(input_path.read_text()) if input_path.exists() else {}
    if fusion_root.exists() and recorded.get("branch_hash") != input_hash:
        # Retain old append-only evidence; changed branch results need a new attempt.
        archive = root / "fusion_attempts" / uuid4().hex
        archive.parent.mkdir(parents=True, exist_ok=True)
        fusion_root.rename(archive)
    write_json(input_path, {"branch_hash": input_hash})
    path = fusion_root / "analysis.json"
    if path.exists():
        saved = read_model(path, AnalysisResult)
        if saved.claim_fingerprint != fingerprint:
            raise ValueError("Fusion fingerprint mismatch")
        if len(saved.assessments) == len(shared.claims):
            return saved
    result = AnalysisResult(
        paper_id=item.paper_id,
        system="fusion",
        claim_fingerprint=fingerprint,
        status=BranchStatus.COMPLETE,
    )

    def one(claim: GearClaim) -> Assessment:
        directory = root / "fusion" / claim.claim_id.rsplit("::", 1)[-1]
        saved_path = directory / "assessment.json"
        if saved_path.exists():
            return read_model(saved_path, Assessment)
        sources: dict[str, object] = {}
        store = EvidenceStore(directory)
        for mode, branch in branches.items():
            assessment = next(
                (x for x in branch.assessments if x.claim_id == claim.claim_id), None
            )
            store.add_evidence(
                f"{mode.upper()}_ANALYSIS:{claim.claim_id}",
                "branch_analysis",
                {
                    "assessment": assessment,
                    "limitations": branch.limitations,
                    "status": branch.status,
                },
            )
            for key, payload in evidence_payloads(
                root / mode / claim.claim_id.rsplit("::", 1)[-1]
            ).items():
                store.add_evidence(f"{mode}:{key}", "branch_source", payload)
        sources = evidence_payloads(directory)
        return _assess_saved(
            config,
            claim.claim_id,
            claim.normalized_claim_text,
            sources,
            "fusion",
            saved_path,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = [
            (claim.claim_id, pool.submit(copy_context().run, one, claim))
            for claim in shared.claims
        ]
        for claim_id, future in jobs:
            try:
                result.assessments.append(future.result())
            except ERRORS as exc:
                result.limitations.append(f"{claim_id}:{exc}")
    if result.limitations or any(
        x.status is not BranchStatus.COMPLETE for x in branches.values()
    ):
        result.status = BranchStatus.LIMITED
    write_model(path, result)
    lines = [f"# {item.title}", f"状态：{result.status.value}"]
    for row in result.assessments:
        lines += [
            f"\n## {row.claim_text}",
            f"支持范围：{row.supported_scope}",
            f"总体判断：{row.overall_stance.value} — {row.overall_reason}",
        ]
        lines += [
            f"- {f.dimension}：{f.text}（依据：{', '.join(f.evidence_keys)}）"
            for f in row.findings
        ]
        lines += [f"- 限制：{x}" for x in row.limitations]
    lines += [f"- 运行限制：{x}" for x in result.limitations]
    (root / "fusion" / "innovation_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return result


def run(
    item: InnovationPaperInput,
    root: Path,
    config: GearConfig,
    stage: str,
    graph_root: Path,
    embedding_model: Path,
    targets: list[str] | None = None,
) -> dict[str, str]:
    with stage_lock(root / ".locks" / "shared"):
        paper, shared = prepare_shared(item, root, config, targets)
    paths = {"shared": str(root / "shared" / "claims.json")}
    for mode in ("gear", "graph"):
        if stage in ("all", mode):
            run_branch(
                item, root, config, paper, shared, mode, graph_root, embedding_model
            )
            paths[mode] = str(root / mode / "analysis.json")
            if mode == "graph":
                try:
                    paths["graph_joint"] = str(
                        run_joint(config, root, shared, graph_root, embedding_model)
                    )
                except ERRORS as exc:
                    failure = root / "graph" / "joint" / "status.json"
                    write_json(failure, {"status": "limited", "error": str(exc)})
                    paths["graph_joint"] = str(failure)
    if stage in ("all", "fusion"):
        fuse(item, root, config, shared)
        paths["fusion"] = str(root / "fusion" / "analysis.json")
    return paths

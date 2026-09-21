"""Separate batched graph encoding/facts from a flat language-model work queue."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from gear.artifacts import read_model, write_json, write_model
from gear.claim_attribution import ClaimGraphRuntime
from gear.config import GearConfig
from gear.contracts import PaperIR
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
from .joint_graph import analyze_prepared_joint, run_joint
from .locking import stage_lock
from .pipeline import ERRORS
from .usage import log_progress


def pending_claims(
    item: InnovationPaperInput,
    root: Path,
    config: GearConfig,
    runtime: ClaimGraphRuntime,
) -> list[tuple[GraphClaim, InnovationPaperInput, Path]]:
    paper = cast(PaperIR, read_model(root / "shared/paper_ir.json", PaperIR))
    shared = cast(ClaimSet, read_model(root / "shared/claims.json", ClaimSet))
    policy = {
        "top_k": config.graph_top_k,
        "min_similarity": config.graph_min_similarity,
    }
    branch = root / "graph"
    policy_path = branch / "insertion_policy.json"
    if policy_path.exists():
        if json.loads(policy_path.read_text()) != policy:
            raise ValueError("Graph insertion policy changed; use a new run directory")
    elif (branch / "analysis.json").exists() or any(
        branch.glob("*/evidence_trace.jsonl")
    ):
        raise ValueError(
            "Legacy graph cache lacks threshold policy; use a new run directory"
        )
    write_json(policy_path, policy)
    spans = paper.span_map()
    pending = []
    for claim in shared.claims:
        target = GraphClaim(
            claim_id=claim.claim_id,
            paper_id=item.paper_id,
            claim_type=claim.claim_type,
            claim_text=claim.normalized_claim_text,
            source_sentence_ids=claim.source_span_ids,
            source_sentence_texts=[spans[x].text for x in claim.source_span_ids],
        )
        directory = branch / claim.claim_id.rsplit("::", 1)[-1]
        row = EvidenceStore(directory)._evidence.get(f"GRAPH:{claim.claim_id}")
        if row is None:
            pending.append((target, item, directory))
        else:
            card = GraphFactCard.model_validate(row.payload)
            if (
                card.claim != target
                or card.insertion_policy != runtime.insertion_policy
            ):
                raise ValueError("Prepared graph fact input/policy mismatch")
    return pending


def prepare_graph(
    papers: list[tuple[InnovationPaperInput, Path]],
    config: GearConfig,
    graph_root: Path,
    embedding_model: Path,
    batch_size: int,
) -> list[dict[str, str]]:
    """One encoder for the study, including batches spanning paper boundaries."""
    if batch_size < 1:
        raise ValueError("Embedding batch size must be positive")
    failures = []
    runtime = ClaimGraphRuntime(
        graph_root, embedding_model, config.graph_top_k, config.graph_min_similarity
    )
    try:
        pending = []
        ready = []
        for item, root in papers:
            try:
                pending.extend(pending_claims(item, root, config, runtime))
                ready.append((item, root))
            except ERRORS as exc:
                failures.append({"paper_id": item.paper_id, "error": str(exc)})
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset : offset + batch_size]
            log_progress("[Graph embedding] %d/%d claims", offset, len(pending))
            try:
                vectors = runtime.encode_batch(
                    [claim.claim_text for claim, _, _ in batch], batch_size
                )
                if len(vectors) != len(batch):
                    raise ValueError("Embedding batch returned wrong number of vectors")
            except ERRORS as exc:
                for claim, _, directory in batch:
                    write_json(directory / "failure.json", {"error": str(exc)})
                    failures.append({"claim_id": claim.claim_id, "error": str(exc)})
                # Encoder failures usually affect the whole study (model, dimensions, GPU).
                # Stop here; saved successful batches remain resumable.
                break
            for (claim, item, directory), vector in zip(batch, vectors, strict=True):
                try:
                    card = runtime.insert_vector(claim, item, vector)
                    EvidenceStore(directory).add_evidence(
                        f"GRAPH:{claim.claim_id}", "graph_fact", card
                    )
                    (directory / "failure.json").unlink(missing_ok=True)
                except ERRORS as exc:
                    write_json(directory / "failure.json", {"error": str(exc)})
                    failures.append({"claim_id": claim.claim_id, "error": str(exc)})
    finally:
        runtime.close()
    for item, root in ready:
        try:
            shared = cast(ClaimSet, read_model(root / "shared/claims.json", ClaimSet))
            # Incomplete evidence must remain retryable; do not seal a partial joint result.
            missing = [
                c.claim_id
                for c in shared.claims
                if f"GRAPH:{c.claim_id}"
                not in EvidenceStore(
                    root / "graph" / c.claim_id.rsplit("::", 1)[-1]
                )._evidence
            ]
            if missing:
                raise ValueError(f"Missing prepared graph facts: {missing}")
            run_joint(
                config, root, shared, graph_root, embedding_model, prepare_only=True
            )
        except ERRORS as exc:
            failures.append({"paper_id": item.paper_id, "error": str(exc)})
    return failures


def model_tasks(
    papers: list[tuple[InnovationPaperInput, Path]],
) -> list[dict[str, str]]:
    tasks: list[dict[str, str]] = []
    for item, root in papers:
        shared = cast(ClaimSet, read_model(root / "shared/claims.json", ClaimSet))
        tasks.extend(
            {
                "task_id": c.claim_id,
                "paper_id": item.paper_id,
                "root": str(root),
                "claim_id": c.claim_id,
            }
            for c in shared.claims
        )
        tasks.append(
            {
                "task_id": f"{item.paper_id}::joint",
                "paper_id": item.paper_id,
                "root": str(root),
                "claim_id": "",
            }
        )
    return tasks


def prepared_sources(
    directory: Path, shared: ClaimSet, claim: GearClaim, config: GearConfig
) -> tuple[dict[str, object], GraphFactCard]:
    sources = evidence_payloads(directory)
    card = GraphFactCard.model_validate(sources[f"GRAPH:{claim.claim_id}"])
    expected = f"threshold_parent_path:k={config.graph_top_k}:cosine>{config.graph_min_similarity}"
    if (
        card.insertion_policy != expected
        or card.claim.claim_id != claim.claim_id
        or card.claim.paper_id != shared.paper_id
        or card.claim.claim_type != claim.claim_type
        or card.claim.source_sentence_ids != claim.source_span_ids
        or card.claim.claim_text != claim.normalized_claim_text
    ):
        raise ValueError("Prepared graph fact input/policy mismatch")
    return sources, card


def saved_assessment(
    path: Path, claim: GearClaim, sources: dict[str, object]
) -> Assessment:
    saved = cast(Assessment, read_model(path, Assessment))
    if (
        saved.claim_id != claim.claim_id
        or saved.claim_text != claim.normalized_claim_text
    ):
        raise ValueError("Saved assessment inputs changed")
    if not saved.findings or any(
        not f.evidence_keys or not set(f.evidence_keys) <= sources.keys()
        for f in saved.findings
    ):
        raise ValueError("Saved assessment has unbound evidence keys")
    return saved


def analyze_task(task: dict[str, str], config: GearConfig) -> dict[str, object]:
    root = Path(task["root"])
    shared = cast(ClaimSet, read_model(root / "shared/claims.json", ClaimSet))
    if not task["claim_id"]:
        return {"output": str(analyze_prepared_joint(config, root, shared))}
    claim = next(c for c in shared.claims if c.claim_id == task["claim_id"])
    directory = root / "graph" / claim.claim_id.rsplit("::", 1)[-1]
    with stage_lock(root / ".locks" / f"graph_{directory.name}"):
        sources, card = prepared_sources(directory, shared, claim, config)
        if not card.neighbors:
            return {
                "status": "limited",
                "reason": "No eligible historical graph neighbors",
            }
        path = directory / "assessment.json"
        if path.exists():
            saved_assessment(path, claim, sources)
            return {"skipped": True, "output": str(path)}
        result = assess(
            config, claim.claim_id, claim.normalized_claim_text, sources, "graph"
        )
        write_model(path, result)
        return {"output": str(path)}


def finalize_graph(
    item: InnovationPaperInput, root: Path, config: GearConfig
) -> AnalysisResult:
    shared = cast(ClaimSet, read_model(root / "shared/claims.json", ClaimSet))
    result = AnalysisResult(
        paper_id=item.paper_id,
        system="graph",
        claim_fingerprint=(
            sha256_value(shared) if config.resume_fingerprint_checks_enabled else ""
        ),
        status=BranchStatus.COMPLETE,
    )
    for claim in shared.claims:
        path = root / "graph" / claim.claim_id.rsplit("::", 1)[-1] / "assessment.json"
        try:
            sources, card = prepared_sources(path.parent, shared, claim, config)
            if not card.neighbors:
                raise ValueError("No eligible historical graph neighbors")
            result.assessments.append(saved_assessment(path, claim, sources))
        except ERRORS as exc:
            result.limitations.append(f"{claim.claim_id}: {type(exc).__name__}: {exc}")
    if result.limitations or any(a.limitations for a in result.assessments):
        result.status = BranchStatus.LIMITED
    write_model(root / "graph/analysis.json", result)
    return result

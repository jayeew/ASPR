"""Build and validate a blinded, auditable expert annotation pack.

The pack is intentionally label-free.  It can only be released after the full
Stage-B, Gate-1, and randomized Stage-C inputs pass conservative readiness
checks.  Public tasks contain aliases and evidence text; identities, arm names,
and numeric Graph/model scores remain in a separately sealed key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import Field, model_validator

from gear.contracts import StrictModel

RelationAuditLabel = Literal["DIRECT", "PARTIAL", "PARALLEL", "DISTANT", "UNVERIFIABLE"]
OrdinalAuditLabel = Literal["YES", "PARTIAL", "NO", "UNVERIFIABLE"]

CODEBOOK = """# ASPR–GEAR blinded expert annotation codebook

## Independence and blinding

Each task is completed independently by at least two subject-matter experts.
Do not discuss a task before both initial annotations are frozen.  Public task
files deliberately omit paper identifiers, method/arm names, Graph values,
model scores, future outcomes, and action assignments.  The sealed key is held
by the study custodian and is not released to annotators.

## Claim B — evidence validity

Assess the proposed claim inventory as a whole (`inventory_complete`) and then
every proposed claim.  `inventory_valid` asks whether the item is a distinct,
material manuscript claim.  Relation labels mean: DIRECT—same central claim or
mechanism already present; PARTIAL—substantial overlap but a material residual;
PARALLEL—similar goal/result through a distinct mechanism or setting;
DISTANT—contextual only; UNVERIFIABLE—provided evidence is insufficient.
`residual_novelty`, `manuscript_support`, and `trace_complete` use YES, PARTIAL,
NO, or UNVERIFIABLE.  A YES residual-novelty judgment requires a stated,
evidence-supported difference after the closest supplied antecedent.  Record a
0–1 confidence, a substantive rationale, and every evidence key actually used.

## Claim C — pairwise integration utility

Compare LEFT and RIGHT only on evidence-supported structural innovation:
correctly bounded contribution, credible direct/partial/parallel/distant prior
relations, useful residual novelty, manuscript support, and complete trace.
Choose LEFT, RIGHT, TIE, or UNVERIFIABLE.  Ignore prose style and list length.
Record confidence, rationale, and evidence keys.  Side order is independently
randomized per task; neither side name reveals the generating method.

## Adjudication

After two frozen independent annotations, disagreements are sent to a third
expert who has not authored either initial annotation.  The adjudicator reviews
both rationales and cited evidence, records a final decision, confidence,
rationale, and evidence keys.  Agreement needs no adjudication row.
"""


class EvidenceExcerpt(StrictModel):
    evidence_key: str = Field(min_length=1)
    evidence_kind: str = Field(min_length=1)
    excerpt: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=0)


class ClaimCandidate(StrictModel):
    claim_alias: str
    claim_text: str = Field(min_length=1)
    manuscript_evidence: list[EvidenceExcerpt] = Field(min_length=1)
    relation_evidence: list[EvidenceExcerpt] = Field(default_factory=list)


class ClaimBTask(StrictModel):
    contract: Literal["gear_claim_b_blind_task_v1"] = "gear_claim_b_blind_task_v1"
    task_id: str
    paper_alias: str
    manuscript_packet: list[EvidenceExcerpt] = Field(min_length=1)
    claims: list[ClaimCandidate] = Field(min_length=1)


class PairwiseSide(StrictModel):
    side: Literal["LEFT", "RIGHT"]
    claims: list[ClaimCandidate] = Field(min_length=1)


class ClaimCTask(StrictModel):
    contract: Literal["gear_claim_c_blind_pairwise_task_v1"] = (
        "gear_claim_c_blind_pairwise_task_v1"
    )
    task_id: str
    paper_alias: str
    left: PairwiseSide
    right: PairwiseSide


class ClaimAssessment(StrictModel):
    claim_alias: str
    inventory_valid: OrdinalAuditLabel
    relation: RelationAuditLabel
    residual_novelty: OrdinalAuditLabel
    manuscript_support: OrdinalAuditLabel
    trace_complete: OrdinalAuditLabel
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)
    evidence_keys: list[str] = Field(min_length=1)


class ClaimBAnnotation(StrictModel):
    contract: Literal["gear_claim_b_expert_annotation_v1"] = (
        "gear_claim_b_expert_annotation_v1"
    )
    task_id: str
    annotator_id: str = Field(min_length=1)
    inventory_complete: OrdinalAuditLabel
    inventory_rationale: str = Field(min_length=1)
    assessments: list[ClaimAssessment] = Field(min_length=1)


class ClaimCAnnotation(StrictModel):
    contract: Literal["gear_claim_c_expert_annotation_v1"] = (
        "gear_claim_c_expert_annotation_v1"
    )
    task_id: str
    annotator_id: str = Field(min_length=1)
    preference: Literal["LEFT", "RIGHT", "TIE", "UNVERIFIABLE"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)
    evidence_keys: list[str] = Field(min_length=1)


class Adjudication(StrictModel):
    contract: Literal["gear_expert_adjudication_v1"] = "gear_expert_adjudication_v1"
    task_id: str
    claim: Literal["B", "C"]
    adjudicator_id: str = Field(min_length=1)
    initial_annotator_ids: list[str] = Field(min_length=2, max_length=2)
    final_decision: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)
    evidence_keys: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def independent_adjudicator(self) -> Adjudication:
        if len(set(self.initial_annotator_ids)) != 2:
            raise ValueError("initial annotators must be distinct")
        if self.adjudicator_id in self.initial_annotator_ids:
            raise ValueError("adjudicator must be independent")
        return self


class AnnotationPackNotReady(ValueError):
    """Raised when an annotation release would rely on partial inputs."""

    def __init__(self, failures: list[str]):
        self.failures = failures
        super().__init__("annotation pack not ready: " + "; ".join(failures))


def generate_annotation_pack(
    stage_b_claims_path: Path,
    gate1_path: Path,
    stage_c_log_path: Path,
    stage_b_manifest_path: Path,
    stage_c_manifest_path: Path,
    runs_dir: Path,
    output_dir: Path,
    *,
    claim_b_papers: int = 30,
    claim_c_pairs: int = 30,
    expected_stage_b_papers: int = 241,
    expected_stage_c_cases: int = 150,
    seed: int = 20260828,
) -> dict[str, Any]:
    """Generate immutable public tasks, empty templates, schemas, and seal."""
    paths = [
        stage_b_claims_path,
        gate1_path,
        stage_c_log_path,
        stage_b_manifest_path,
        stage_c_manifest_path,
    ]
    missing_paths = [str(path) for path in paths if not path.is_file()]
    if missing_paths:
        raise AnnotationPackNotReady(
            [f"missing_input:{path}" for path in missing_paths]
        )
    claims = pd.read_parquet(stage_b_claims_path)
    gate1 = pd.read_parquet(gate1_path)
    stage_c = pd.read_parquet(stage_c_log_path)
    stage_b_manifest = _json(stage_b_manifest_path)
    stage_c_manifest = _json(stage_c_manifest_path)
    failures = _readiness_failures(
        claims,
        gate1,
        stage_c,
        stage_b_manifest,
        stage_c_manifest,
        expected_stage_b_papers,
        expected_stage_c_cases,
    )
    if failures:
        raise AnnotationPackNotReady(failures)
    joined = _join_claim_inputs(claims, gate1)
    b_ids = _stratified_papers(joined, claim_b_papers, seed, "claim-b")
    holdout_ids = set(
        stage_c.loc[
            stage_c["experiment_split"].eq("confirmatory_holdout"), "paper_id"
        ].astype(str)
    )
    c_source = joined[joined["paper_id"].astype(str).isin(holdout_ids)]
    c_ids = _stratified_papers(c_source, claim_c_pairs, seed, "claim-c")
    b_tasks, b_seal = _build_claim_b_tasks(joined, b_ids, runs_dir, seed)
    c_tasks, c_seal = _build_claim_c_tasks(joined, c_ids, runs_dir, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    public_files = _write_pack_files(output_dir, b_tasks, c_tasks)
    seal_path = output_dir / "sealed_assignment_key.json"
    _write_json(
        seal_path,
        {
            "contract": "gear_expert_annotation_sealed_key_v1",
            "do_not_release_to_annotators": True,
            "claim_b": b_seal,
            "claim_c": c_seal,
        },
    )
    source_paths = {
        "stage_b_claims": stage_b_claims_path,
        "gate1": gate1_path,
        "stage_c_log": stage_c_log_path,
        "stage_b_manifest": stage_b_manifest_path,
        "stage_c_manifest": stage_c_manifest_path,
    }
    manifest = {
        "contract": "gear_expert_annotation_pack_manifest_v1",
        "status": "ready_for_annotation",
        "labels_included": False,
        "blinding": {
            "randomized_left_right": True,
            "graph_scores_hidden": True,
            "method_names_hidden": True,
            "future_outcomes_hidden": True,
            "sealed_key_sha256": _sha256_file(seal_path),
        },
        "review_design": {
            "independent_experts_per_task": 2,
            "adjudication_on_disagreement": True,
            "independent_adjudicator_required": True,
        },
        "sampling": {
            "strategy": "deterministic_round_robin_across_split_domain_decile_year_bin_verification",
            "claim_b": _selection_audit(joined, b_ids),
            "claim_c": _selection_audit(joined, c_ids),
        },
        "counts": {"claim_b_tasks": len(b_tasks), "claim_c_tasks": len(c_tasks)},
        "seed": seed,
        "source_sha256": {
            name: _sha256_file(path) for name, path in source_paths.items()
        },
        "file_sha256": {
            path.name: _sha256_file(path) for path in [*public_files, seal_path]
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    validate_annotation_pack(output_dir)
    return manifest


def validate_annotation_pack(
    pack_dir: Path,
    *,
    require_completed: bool = False,
) -> dict[str, Any]:
    """Validate hashes, blindness, templates, and optionally expert labels."""
    manifest_path = pack_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("annotation pack manifest is missing")
    manifest = _json(manifest_path)
    if manifest.get("status") != "ready_for_annotation":
        raise ValueError("annotation pack is not release-ready")
    for name, expected in manifest.get("file_sha256", {}).items():
        path = pack_dir / name
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"annotation pack hash mismatch: {name}")
    b_tasks = [
        ClaimBTask.model_validate(row)
        for row in _read_jsonl(pack_dir / "claim_b_tasks.jsonl")
    ]
    c_tasks = [
        ClaimCTask.model_validate(row)
        for row in _read_jsonl(pack_dir / "claim_c_tasks.jsonl")
    ]
    _validate_public_blinding(b_tasks, c_tasks)
    _validate_templates(pack_dir, b_tasks, c_tasks)
    annotation_hashes: dict[str, str] = {}
    if require_completed:
        _validate_completed(pack_dir, b_tasks, c_tasks)
        for name in (
            "claim_b_annotations.jsonl",
            "claim_c_annotations.jsonl",
            "adjudications.jsonl",
        ):
            path = pack_dir / name
            if path.is_file():
                annotation_hashes[name] = _sha256_file(path)
    return {
        "contract": "gear_expert_annotation_pack_validation_v1",
        "valid": True,
        "completed_annotations_validated": require_completed,
        "claim_b_tasks": len(b_tasks),
        "claim_c_tasks": len(c_tasks),
        "annotation_sha256": annotation_hashes,
    }


def _readiness_failures(
    claims: pd.DataFrame,
    gate1: pd.DataFrame,
    stage_c: pd.DataFrame,
    stage_b_manifest: dict[str, Any],
    stage_c_manifest: dict[str, Any],
    expected_stage_b: int,
    expected_stage_c: int,
) -> list[str]:
    failures: list[str] = []
    claim_required = {
        "paper_id",
        "claim_id",
        "claim_text",
        "gear_run_path",
        "verification_passed",
    }
    gate_required = {
        "paper_id",
        "claim_id",
        "domain12",
        "publication_year",
        "integration_split",
        "graph_percentile",
        "structural_innovation_score",
        "structural_score_at_zero",
    }
    stage_c_required = {
        "paper_id",
        "assigned_action",
        "propensity",
        "matched_budget",
        "experiment_split",
        "context_id",
    }
    for name, frame, required in (
        ("stage_b", claims, claim_required),
        ("gate1", gate1, gate_required),
        ("stage_c", stage_c, stage_c_required),
    ):
        missing = sorted(required - set(frame))
        if missing:
            failures.append(f"{name}_columns_missing:{','.join(missing)}")
    if failures:
        return failures
    b_papers = set(claims["paper_id"].astype(str))
    g_papers = set(gate1["paper_id"].astype(str))
    if len(b_papers) != expected_stage_b:
        failures.append(f"stage_b_incomplete:{len(b_papers)}/{expected_stage_b}")
    if len(g_papers) != expected_stage_b:
        failures.append(f"gate1_incomplete:{len(g_papers)}/{expected_stage_b}")
    if b_papers != g_papers:
        failures.append("stage_b_gate1_paper_mismatch")
    if claims[["paper_id", "claim_id"]].duplicated().any():
        failures.append("stage_b_duplicate_claims")
    if gate1[["paper_id", "claim_id"]].duplicated().any():
        failures.append("gate1_duplicate_claims")
    if stage_b_manifest.get("selection_uses_future_outcomes") is not False:
        failures.append("stage_b_manifest_not_outcome_blind")
    cases = stage_c_manifest.get("cases", [])
    if len(cases) != expected_stage_c:
        failures.append(f"stage_c_manifest_incomplete:{len(cases)}/{expected_stage_c}")
    if stage_c_manifest.get("randomization_precedes_outcomes") is not True:
        failures.append("stage_c_randomization_not_pre_outcome")
    if len(stage_c) != expected_stage_c:
        failures.append(f"stage_c_log_incomplete:{len(stage_c)}/{expected_stage_c}")
    if stage_c["paper_id"].astype(str).duplicated().any():
        failures.append("stage_c_duplicate_papers")
    manifest_ids = {str(case.get("paper_id")) for case in cases}
    if set(stage_c["paper_id"].astype(str)) != manifest_ids:
        failures.append("stage_c_manifest_log_mismatch")
    if set(stage_c["experiment_split"].astype(str)) != {
        "development",
        "confirmatory_holdout",
    }:
        failures.append("stage_c_independent_holdout_missing")
    action_counts = stage_c.groupby(["experiment_split", "assigned_action"]).size()
    split_balanced = action_counts.size == 12 and all(
        group.nunique() == 1
        for _, group in action_counts.groupby(level="experiment_split")
    )
    if not split_balanced:
        failures.append("stage_c_actions_not_split_balanced")
    if not pd.to_numeric(stage_c["propensity"], errors="coerce").eq(1.0 / 6.0).all():
        failures.append("stage_c_propensity_invalid")
    if stage_c["matched_budget"].nunique() != 1:
        failures.append("stage_c_budget_not_matched")
    return failures


def _join_claim_inputs(claims: pd.DataFrame, gate1: pd.DataFrame) -> pd.DataFrame:
    gate_columns = [
        "paper_id",
        "claim_id",
        "domain12",
        "publication_year",
        "integration_split",
        "graph_percentile",
        "structural_innovation_score",
        "structural_score_at_zero",
    ]
    return claims.merge(
        gate1[gate_columns],
        on=["paper_id", "claim_id"],
        how="inner",
        validate="one_to_one",
    )


def _stratified_papers(
    frame: pd.DataFrame, count: int, seed: int, purpose: str
) -> list[str]:
    papers = frame.drop_duplicates("paper_id").copy()
    if len(papers) < count:
        raise AnnotationPackNotReady(
            [f"{purpose}_eligible_papers:{len(papers)}/{count}"]
        )
    papers["decile"] = (
        (pd.to_numeric(papers["graph_percentile"], errors="coerce") / 10)
        .fillna(-1)
        .clip(-1, 9)
        .astype(int)
    )
    papers["year_bin"] = (pd.to_numeric(papers["publication_year"]) // 5).astype(int)
    papers["stratum"] = papers.apply(
        lambda row: "|".join(
            [
                str(row["integration_split"]),
                str(row["domain12"]),
                str(row["decile"]),
                str(row["year_bin"]),
                str(bool(row.get("verification_passed", True))),
            ]
        ),
        axis=1,
    )
    buckets: dict[str, list[str]] = defaultdict(list)
    for row in papers.to_dict(orient="records"):
        buckets[str(row["stratum"])].append(str(row["paper_id"]))
    for key, values in buckets.items():
        values.sort(key=lambda value: _stable_digest(seed, purpose, key, value))
    selected: list[str] = []
    keys = sorted(buckets, key=lambda key: _stable_digest(seed, purpose, key))
    while len(selected) < count:
        progressed = False
        for key in keys:
            if buckets[key] and len(selected) < count:
                selected.append(buckets[key].pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def _selection_audit(frame: pd.DataFrame, paper_ids: list[str]) -> dict[str, Any]:
    selected = frame[frame["paper_id"].astype(str).isin(paper_ids)].drop_duplicates(
        "paper_id"
    )
    deciles = (
        (pd.to_numeric(selected["graph_percentile"], errors="coerce") / 10)
        .clip(0, 9)
        .astype(int)
        .value_counts()
        .sort_index()
    )
    years = (pd.to_numeric(selected["publication_year"]) // 5 * 5).astype(int)
    return {
        "papers": len(selected),
        "integration_splits": selected["integration_split"]
        .astype(str)
        .value_counts()
        .sort_index()
        .to_dict(),
        "domains": selected["domain12"]
        .astype(str)
        .value_counts()
        .sort_index()
        .to_dict(),
        "graph_deciles": {str(key): int(value) for key, value in deciles.items()},
        "publication_year_bins": {
            str(key): int(value)
            for key, value in years.value_counts().sort_index().items()
        },
        "verification_status": selected["verification_passed"]
        .astype(bool)
        .value_counts()
        .sort_index()
        .astype(int)
        .to_dict(),
    }


def _build_claim_b_tasks(
    frame: pd.DataFrame, paper_ids: list[str], runs_dir: Path, seed: int
) -> tuple[list[ClaimBTask], list[dict[str, str]]]:
    tasks: list[ClaimBTask] = []
    seal: list[dict[str, str]] = []
    for index, paper_id in enumerate(paper_ids, start=1):
        group = frame[frame["paper_id"].astype(str).eq(paper_id)]
        paper_alias = f"PB-{index:04d}"
        task_id = "CB-" + _stable_digest(seed, "claim-b-task", paper_id)[:16]
        candidates = _claim_candidates(group, runs_dir, rank_column=None)
        manuscript_packet = _full_manuscript_packet(group, runs_dir)
        tasks.append(
            ClaimBTask(
                task_id=task_id,
                paper_alias=paper_alias,
                manuscript_packet=manuscript_packet,
                claims=candidates,
            )
        )
        seal.append(
            {"task_id": task_id, "paper_alias": paper_alias, "paper_id": paper_id}
        )
    return tasks, seal


def _build_claim_c_tasks(
    frame: pd.DataFrame, paper_ids: list[str], runs_dir: Path, seed: int
) -> tuple[list[ClaimCTask], list[dict[str, str]]]:
    tasks: list[ClaimCTask] = []
    seal: list[dict[str, str]] = []
    for index, paper_id in enumerate(paper_ids, start=1):
        group = frame[frame["paper_id"].astype(str).eq(paper_id)]
        evidence_only = _claim_candidates(group, runs_dir, "structural_score_at_zero")
        joint = _claim_candidates(group, runs_dir, "structural_innovation_score")
        joint_left = int(_stable_digest(seed, "claim-c-side", paper_id), 16) % 2 == 0
        left_claims, right_claims = (
            (joint, evidence_only) if joint_left else (evidence_only, joint)
        )
        task_id = "CC-" + _stable_digest(seed, "claim-c-task", paper_id)[:16]
        paper_alias = f"PC-{index:04d}"
        tasks.append(
            ClaimCTask(
                task_id=task_id,
                paper_alias=paper_alias,
                left=PairwiseSide(side="LEFT", claims=left_claims),
                right=PairwiseSide(side="RIGHT", claims=right_claims),
            )
        )
        seal.append(
            {
                "task_id": task_id,
                "paper_alias": paper_alias,
                "paper_id": paper_id,
                "left_arm": (
                    "evidence_gated_fusion" if joint_left else "gear_evidence_only"
                ),
                "right_arm": (
                    "gear_evidence_only" if joint_left else "evidence_gated_fusion"
                ),
            }
        )
    return tasks, seal


def _claim_candidates(
    group: pd.DataFrame, runs_dir: Path, rank_column: str | None
) -> list[ClaimCandidate]:
    ordered = group.sort_values(
        [rank_column, "claim_id"] if rank_column else ["claim_id"],
        ascending=[False, True] if rank_column else [True],
    )
    run_path = Path(str(ordered.iloc[0]["gear_run_path"]))
    if not run_path.is_absolute():
        run_path = runs_dir / run_path
    trace = _load_trace(run_path / "evidence_trace.jsonl")
    bundle = _json(run_path / "review_bundle.json")
    relation_keys = sorted(
        {
            str(key)
            for point in (
                bundle.get("state", {}).get("canonical_points") or {}
            ).values()
            for key in point.get("relation_evidence_keys", [])
        }
    )
    relation = [_excerpt(trace[key]) for key in relation_keys if key in trace]
    candidates: list[ClaimCandidate] = []
    for alias_index, row in enumerate(ordered.to_dict(orient="records"), start=1):
        keys = _list_value(row.get("manuscript_evidence_keys"))
        if not keys:
            inventory = bundle.get("state", {}).get("claim_inventory") or []
            match = next(
                (
                    item
                    for item in inventory
                    if str(item.get("claim_id")) == str(row["claim_id"])
                ),
                None,
            )
            keys = [
                str(key) for key in (match or {}).get("manuscript_evidence_keys", [])
            ]
        manuscript = [_excerpt(trace[key]) for key in keys if key in trace]
        if not manuscript:
            raise AnnotationPackNotReady(
                [f"missing_claim_evidence:{row['paper_id']}:{row['claim_id']}"]
            )
        candidates.append(
            ClaimCandidate(
                claim_alias=f"CL-{alias_index:02d}",
                claim_text=str(row["claim_text"]),
                manuscript_evidence=manuscript,
                relation_evidence=relation,
            )
        )
    return candidates


def _full_manuscript_packet(
    group: pd.DataFrame, runs_dir: Path
) -> list[EvidenceExcerpt]:
    run_path = Path(str(group.iloc[0]["gear_run_path"]))
    if not run_path.is_absolute():
        run_path = runs_dir / run_path
    trace = _load_trace(run_path / "evidence_trace.jsonl")
    packet = [
        _excerpt(record)
        for record in trace.values()
        if str(record.get("kind")) == "paper_span"
    ]
    if not packet:
        raise AnnotationPackNotReady(
            [f"missing_manuscript_packet:{group.iloc[0]['paper_id']}"]
        )
    return packet


def _load_trace(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise AnnotationPackNotReady([f"missing_evidence_trace:{path}"])
    rows = _read_jsonl(path)
    return {str(row["evidence_id"]): row for row in rows if row.get("evidence_id")}


def _excerpt(record: dict[str, Any]) -> EvidenceExcerpt:
    payload = record.get("payload") or {}
    texts = _collect_text(payload)
    excerpt = " | ".join(dict.fromkeys(texts))[:4000]
    if not excerpt:
        excerpt = "Evidence record available; inspect sealed source trace."
    page = payload.get("page") if isinstance(payload, dict) else None
    return EvidenceExcerpt(
        evidence_key=str(record["evidence_id"]),
        evidence_kind=str(record.get("kind") or "unknown"),
        excerpt=excerpt,
        page=int(page) if isinstance(page, int) and page >= 0 else None,
    )


def _collect_text(value: Any, key: str = "") -> list[str]:
    forbidden = {"relation", "relation_label", "confidence", "score", "rationale"}
    if key.casefold() in forbidden:
        return []
    if isinstance(value, str) and key.casefold() in {
        "text",
        "title",
        "claim",
        "target_claim",
        "candidate_excerpt",
        "shared_base",
        "residual_delta",
        "abstract",
    }:
        return [value.strip()] if value.strip() else []
    if isinstance(value, dict):
        return [
            text
            for child_key, child in value.items()
            for text in _collect_text(child, str(child_key))
        ]
    if isinstance(value, list):
        return [text for child in value for text in _collect_text(child, key)]
    return []


def _write_pack_files(
    output_dir: Path, b_tasks: list[ClaimBTask], c_tasks: list[ClaimCTask]
) -> list[Path]:
    paths: list[Path] = []
    b_path = output_dir / "claim_b_tasks.jsonl"
    c_path = output_dir / "claim_c_tasks.jsonl"
    _write_jsonl(b_path, [task.model_dump(mode="json") for task in b_tasks])
    _write_jsonl(c_path, [task.model_dump(mode="json") for task in c_tasks])
    paths.extend([b_path, c_path])
    b_templates = [
        _blank_b_template(task.task_id, slot, task.claims)
        for task in b_tasks
        for slot in (1, 2)
    ]
    c_templates = [
        _blank_c_template(task.task_id, slot) for task in c_tasks for slot in (1, 2)
    ]
    for name, rows in (
        ("claim_b_annotation_template.jsonl", b_templates),
        ("claim_c_annotation_template.jsonl", c_templates),
        (
            "adjudication_template.jsonl",
            [_blank_adjudication(task.task_id, "B") for task in b_tasks]
            + [_blank_adjudication(task.task_id, "C") for task in c_tasks],
        ),
    ):
        path = output_dir / name
        _write_jsonl(path, rows)
        paths.append(path)
    schema_path = output_dir / "schemas.json"
    _write_json(
        schema_path,
        {
            "claim_b_task": ClaimBTask.model_json_schema(),
            "claim_c_task": ClaimCTask.model_json_schema(),
            "claim_b_annotation": ClaimBAnnotation.model_json_schema(),
            "claim_c_annotation": ClaimCAnnotation.model_json_schema(),
            "adjudication": Adjudication.model_json_schema(),
        },
    )
    codebook_path = output_dir / "CODEBOOK.md"
    codebook_path.write_text(CODEBOOK, encoding="utf-8")
    paths.extend([schema_path, codebook_path])
    return paths


def _blank_b_template(
    task_id: str, slot: int, claims: list[ClaimCandidate]
) -> dict[str, Any]:
    return {
        "contract": "gear_claim_b_expert_annotation_v1",
        "task_id": task_id,
        "annotation_slot": slot,
        "annotator_id": None,
        "inventory_complete": None,
        "inventory_rationale": None,
        "assessments": [
            {
                "claim_alias": claim.claim_alias,
                "inventory_valid": None,
                "relation": None,
                "residual_novelty": None,
                "manuscript_support": None,
                "trace_complete": None,
                "confidence": None,
                "rationale": None,
                "evidence_keys": [],
            }
            for claim in claims
        ],
    }


def _blank_c_template(task_id: str, slot: int) -> dict[str, Any]:
    return {
        "contract": "gear_claim_c_expert_annotation_v1",
        "task_id": task_id,
        "annotation_slot": slot,
        "annotator_id": None,
        "preference": None,
        "confidence": None,
        "rationale": None,
        "evidence_keys": [],
    }


def _blank_adjudication(task_id: str, claim: str) -> dict[str, Any]:
    return {
        "contract": "gear_expert_adjudication_v1",
        "task_id": task_id,
        "claim": claim,
        "adjudicator_id": None,
        "initial_annotator_ids": [],
        "final_decision": None,
        "confidence": None,
        "rationale": None,
        "evidence_keys": [],
        "required_only_if_initial_annotations_disagree": True,
    }


def _validate_public_blinding(
    b_tasks: list[ClaimBTask], c_tasks: list[ClaimCTask]
) -> None:
    forbidden_keys = {
        "paper_id",
        "graph_percentile",
        "graph_score",
        "hgb",
        "method",
        "arm",
        "future_outcome",
        "assigned_action",
    }
    for task in [*b_tasks, *c_tasks]:
        keys = _all_keys(task.model_dump(mode="json"))
        overlap = forbidden_keys & {key.casefold() for key in keys}
        if overlap:
            raise ValueError(f"public task leaks blinded fields: {sorted(overlap)}")
    for task in c_tasks:
        if task.left.side != "LEFT" or task.right.side != "RIGHT":
            raise ValueError("pairwise side labels are malformed")


def _validate_templates(
    pack_dir: Path, b_tasks: list[ClaimBTask], c_tasks: list[ClaimCTask]
) -> None:
    b_rows = _read_jsonl(pack_dir / "claim_b_annotation_template.jsonl")
    c_rows = _read_jsonl(pack_dir / "claim_c_annotation_template.jsonl")
    if len(b_rows) != 2 * len(b_tasks) or len(c_rows) != 2 * len(c_tasks):
        raise ValueError("exactly two independent annotation slots are required")
    for rows in (b_rows, c_rows):
        by_task: dict[str, set[int]] = defaultdict(set)
        for row in rows:
            if row.get("annotator_id") is not None:
                raise ValueError("annotation template unexpectedly contains labels")
            by_task[str(row.get("task_id"))].add(int(row.get("annotation_slot", 0)))
        if any(slots != {1, 2} for slots in by_task.values()):
            raise ValueError("annotation template slots must be 1 and 2")


def _validate_completed(
    pack_dir: Path, b_tasks: list[ClaimBTask], c_tasks: list[ClaimCTask]
) -> None:
    b_annotations = [
        ClaimBAnnotation.model_validate(row)
        for row in _read_jsonl(pack_dir / "claim_b_annotations.jsonl")
    ]
    c_annotations = [
        ClaimCAnnotation.model_validate(row)
        for row in _read_jsonl(pack_dir / "claim_c_annotations.jsonl")
    ]
    b_index = {task.task_id: task for task in b_tasks}
    c_index = {task.task_id: task for task in c_tasks}
    _require_two_experts(b_annotations, set(b_index))
    _require_two_experts(c_annotations, set(c_index))
    for annotation in b_annotations:
        task = b_index[annotation.task_id]
        if {row.claim_alias for row in annotation.assessments} != {
            row.claim_alias for row in task.claims
        }:
            raise ValueError(
                "Claim B annotation does not cover every claim exactly once"
            )
        if len(annotation.assessments) != len(task.claims):
            raise ValueError("Claim B annotation contains duplicate claim assessments")
        allowed = _task_evidence_keys(task)
        for row in annotation.assessments:
            if not set(row.evidence_keys) <= allowed:
                raise ValueError("Claim B annotation cites unknown evidence")
    for c_annotation in c_annotations:
        if not set(c_annotation.evidence_keys) <= _task_evidence_keys(
            c_index[c_annotation.task_id]
        ):
            raise ValueError("Claim C annotation cites unknown evidence")
    disagreements = _disagreements(b_annotations, c_annotations)
    adjudication_path = pack_dir / "adjudications.jsonl"
    adjudications = (
        [Adjudication.model_validate(row) for row in _read_jsonl(adjudication_path)]
        if adjudication_path.is_file()
        else []
    )
    by_task = {row.task_id: row for row in adjudications}
    if set(by_task) != disagreements:
        raise ValueError("adjudication rows must exactly match disagreements")
    expert_ids = _expert_ids_by_task([*b_annotations, *c_annotations])
    task_index: dict[str, ClaimBTask | ClaimCTask] = {**b_index, **c_index}
    for task_id, adjudication in by_task.items():
        if set(adjudication.initial_annotator_ids) != expert_ids[task_id]:
            raise ValueError("adjudication initial annotators mismatch")
        if not set(adjudication.evidence_keys) <= _task_evidence_keys(
            task_index[task_id]
        ):
            raise ValueError("adjudication cites unknown evidence")


def _require_two_experts(rows: list[Any], task_ids: set[str]) -> None:
    grouped = _expert_ids_by_task(rows)
    if set(grouped) != task_ids or any(len(ids) != 2 for ids in grouped.values()):
        raise ValueError("every task requires exactly two distinct expert annotations")
    if len(rows) != 2 * len(task_ids):
        raise ValueError("duplicate expert annotations detected")


def _expert_ids_by_task(rows: list[Any]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        grouped[row.task_id].add(row.annotator_id)
    return grouped


def _disagreements(
    b_rows: list[ClaimBAnnotation], c_rows: list[ClaimCAnnotation]
) -> set[str]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    combined: list[Any] = [*b_rows, *c_rows]
    for row in combined:
        grouped[row.task_id].append(row)
    disagreements: set[str] = set()
    for task_id, rows in grouped.items():
        if isinstance(rows[0], ClaimCAnnotation):
            decisions = {row.preference for row in rows}
        else:
            decisions = {
                json.dumps(
                    [
                        row.inventory_complete,
                        *[
                            (
                                a.inventory_valid,
                                a.relation,
                                a.residual_novelty,
                                a.manuscript_support,
                                a.trace_complete,
                            )
                            for a in row.assessments
                        ],
                    ],
                    sort_keys=True,
                )
                for row in rows
            }
        if len(decisions) > 1:
            disagreements.add(task_id)
    return disagreements


def _task_evidence_keys(task: ClaimBTask | ClaimCTask) -> set[str]:
    candidates = (
        task.claims
        if isinstance(task, ClaimBTask)
        else [*task.left.claims, *task.right.claims]
    )
    keys = {
        evidence.evidence_key
        for claim in candidates
        for evidence in [*claim.manuscript_evidence, *claim.relation_evidence]
    }
    if isinstance(task, ClaimBTask):
        keys.update(row.evidence_key for row in task.manuscript_packet)
    return keys


def _all_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            key for child in value.values() for key in _all_keys(child)
        }
    if isinstance(value, list):
        return {key for child in value for key in _all_keys(child)}
    return set()


def _list_value(value: Any) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [str(item) for item in value] if isinstance(value, list) else []


def _stable_digest(seed: int, *values: str) -> str:
    payload = "|".join([str(seed), *values]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"JSONL file is missing: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--stage-b-claims", type=Path, required=True)
    build.add_argument("--gate1", type=Path, required=True)
    build.add_argument("--stage-c-log", type=Path, required=True)
    build.add_argument("--stage-b-manifest", type=Path, required=True)
    build.add_argument("--stage-c-manifest", type=Path, required=True)
    build.add_argument("--runs-dir", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--claim-b-papers", type=int, default=30)
    build.add_argument("--claim-c-pairs", type=int, default=30)
    build.add_argument("--expected-stage-b-papers", type=int, default=241)
    build.add_argument("--expected-stage-c-cases", type=int, default=150)
    build.add_argument("--seed", type=int, default=20260828)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--pack-dir", type=Path, required=True)
    validate.add_argument("--require-completed", action="store_true")
    validate.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "validate":
        result = validate_annotation_pack(
            args.pack_dir, require_completed=args.require_completed
        )
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            _write_json(args.output, result)
        print(json.dumps(result, indent=2))
        return 0
    try:
        report = generate_annotation_pack(
            args.stage_b_claims,
            args.gate1,
            args.stage_c_log,
            args.stage_b_manifest,
            args.stage_c_manifest,
            args.runs_dir,
            args.output_dir,
            claim_b_papers=args.claim_b_papers,
            claim_c_pairs=args.claim_c_pairs,
            expected_stage_b_papers=args.expected_stage_b_papers,
            expected_stage_c_cases=args.expected_stage_c_cases,
            seed=args.seed,
        )
    except AnnotationPackNotReady as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            args.output_dir / "annotation_pack_readiness.json",
            {
                "contract": "gear_expert_annotation_pack_readiness_v1",
                "status": "not_ready",
                "claim_allowed": False,
                "failures": exc.failures,
            },
        )
        print(json.dumps({"status": "not_ready", "failures": exc.failures}, indent=2))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "Adjudication",
    "AnnotationPackNotReady",
    "ClaimBAnnotation",
    "ClaimBTask",
    "ClaimCAnnotation",
    "ClaimCTask",
    "generate_annotation_pack",
    "validate_annotation_pack",
]

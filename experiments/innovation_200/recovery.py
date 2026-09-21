"""Explicit, reversible GEAR cleanup that preserves reusable claim evidence."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from experiments.innovation_200.common import write_json
from experiments.innovation_200.contracts import SYSTEMS
from gear.contracts import QuerySpec
from gear.innovation.contracts import AnalysisResult, Assessment, ClaimSet
from gear.innovation.locking import stage_lock
from gear.review_contracts import BranchStatus, GearClaim, GearClaimCard
from gear.trace import EvidenceStore

DEPENDENT_SYSTEMS = tuple(s for s in SYSTEMS if s not in ("direct_llm", "graph"))


@contextmanager
def study_gear_lock(study: Path) -> Iterator[None]:
    """Fail immediately if another study runner or cleanup owns the write lock."""
    path = study / ".locks/run_gear"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another GEAR runner/cleanup holds this study lock"
            ) from exc
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(str(os.getpid()))
            handle.flush()
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _analysis(path: Path) -> AnalysisResult | None:
    if not path.is_file():
        return None
    try:
        return AnalysisResult.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def _claim_problem(directory: Path, claim: GearClaim) -> str | None:
    """Reuse only complete, source-bound, matching assessments without limitations."""
    try:
        assessment = Assessment.model_validate_json(
            (directory / "assessment.json").read_text(encoding="utf-8")
        )
        card = GearClaimCard.model_validate_json(
            (directory / "gear_card.json").read_text(encoding="utf-8")
        )
        if assessment.claim_id != claim.claim_id or (
            assessment.claim_text != claim.normalized_claim_text or card.claim != claim
        ):
            return "claim_identity_mismatch"
        if assessment.limitations or card.limitations:
            return "claim_limitations"
        if (directory / "failure.json").exists():
            return "recorded_failure"
        execution = directory / "execution.json"
        if execution.exists():
            record = json.loads(execution.read_text(encoding="utf-8"))
            if record.get("status") != "complete" or record.get("errors"):
                return "execution_failure"
        if not (directory / "evidence_trace.jsonl").is_file():
            return "missing_evidence_trace"
        evidence = EvidenceStore(directory)._evidence
        if not assessment.findings or any(
            not finding.evidence_keys
            or not set(finding.evidence_keys).issubset(evidence)
            for finding in assessment.findings
        ):
            return "missing_finding_evidence"
        if not set(card.evidence_keys).issubset(evidence):
            return "missing_card_evidence"
        stored_card = evidence.get(f"GEAR:{claim.claim_id}")
        if stored_card is None or stored_card.payload != card.model_dump(mode="json"):
            return "missing_or_mismatched_stored_card"
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return f"invalid_or_missing_artifact:{type(exc).__name__}"
    return None


def _healthy_card_problem(directory: Path, claim: GearClaim) -> str | None:
    try:
        card = GearClaimCard.model_validate_json(
            (directory / "gear_card.json").read_text(encoding="utf-8")
        )
        if card.claim != claim:
            return "claim_identity_mismatch"
        if not (directory / "evidence_trace.jsonl").is_file():
            return "missing_evidence_trace"
        evidence = EvidenceStore(directory)._evidence
        if not set(card.evidence_keys).issubset(evidence):
            return "missing_card_evidence"
        stored_card = evidence.get(f"GEAR:{claim.claim_id}")
        if stored_card and stored_card.payload != card.model_dump(mode="json"):
            return "mismatched_stored_card"
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return f"invalid_card:{type(exc).__name__}"
    return None


def _recorded_execution_problem(directory: Path) -> str | None:
    path = directory / "execution.json"
    if not path.exists():
        return None
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("status") == "failed" or row.get("errors"):
            return "execution_failure"
    except (OSError, ValueError, TypeError, AttributeError):
        return "invalid_execution_record"
    return None


def _assessment_problem(directory: Path, claim: GearClaim) -> str | None:
    try:
        assessment = Assessment.model_validate_json(
            (directory / "assessment.json").read_text(encoding="utf-8")
        )
        if (
            assessment.claim_id != claim.claim_id
            or assessment.claim_text != claim.normalized_claim_text
        ):
            return "claim_identity_mismatch"
        evidence = EvidenceStore(directory)._evidence
        if not assessment.findings or any(
            not finding.evidence_keys
            or not set(finding.evidence_keys).issubset(evidence)
            for finding in assessment.findings
        ):
            return "missing_finding_evidence"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return f"invalid_or_missing_assessment:{type(exc).__name__}"
    return None


def _retry_failure_claim(
    study: Path,
    root: Path,
    claim: GearClaim,
    branch_errors: list[str],
    archive: Path,
    manifest: dict[str, Any],
) -> tuple[str, str | None]:
    directory = root / "gear" / claim.claim_id.rsplit("::", 1)[-1]
    execution_error = _recorded_execution_problem(directory)
    card_error = _healthy_card_problem(directory, claim)
    assessment_error = (
        _assessment_problem(directory, claim) if not card_error else "card_unavailable"
    )
    branch_error = any(
        error.startswith(f"{claim.claim_id}:") for error in branch_errors
    )
    if execution_error is None and card_error is None and assessment_error is not None:
        # Evidence acquisition succeeded; only the final model summary needs retry.
        for name in ("assessment.json", "failure.json"):
            _move(directory / name, study, archive, manifest)
        return "reassess_saved_card", assessment_error
    reason = execution_error or card_error
    if reason is None and (branch_error or (directory / "failure.json").exists()):
        reason = "recorded_execution_failure"
    if reason:
        _move(directory, study, archive, manifest)
        return "retry_evidence", reason
    return "kept", None


def _retry_paper(
    study: Path, paper_id: str, archive: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    root = study / "papers" / paper_id
    branch = root / "gear"
    record: dict[str, Any] = {
        "paper_id": paper_id,
        "kept_claims": [],
        "retry_claims": [],
    }
    if not branch.exists():
        record["action"] = "not_started"
        return record
    result = _analysis(branch / "analysis.json")
    shared = ClaimSet.model_validate_json(
        (root / "shared/claims.json").read_text(encoding="utf-8")
    )
    if shared.paper_id != paper_id:
        raise ValueError(f"Shared paper identity mismatch: {paper_id}")
    for claim in shared.claims:
        action, problem = _retry_failure_claim(
            study, root, claim, result.limitations if result else [], archive, manifest
        )
        if action == "kept":
            record["kept_claims"].append(claim.claim_id)
        else:
            record["retry_claims"].append(
                {"claim_id": claim.claim_id, "reason": problem, "action": action}
            )
    needs_summary = (
        result is None
        or bool(result.limitations)
        or len(result.assessments) != len(shared.claims)
    )
    if record["retry_claims"] or needs_summary:
        _move(branch / "analysis.json", study, archive, manifest)
        for path in _dependent_paths(study, paper_id):
            _move(path, study, archive, manifest)
        record["action"] = "cleaned"
    else:
        record["action"] = "kept_existing_paper"
    return record


def _move(path: Path, study: Path, archive: Path, manifest: dict[str, Any]) -> None:
    if not path.exists():
        return
    relative = path.relative_to(study)
    destination = archive / "artifacts" / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    path.rename(destination)
    manifest["moved"].append(str(relative))
    write_json(archive / "manifest.json", manifest)


def _dependent_paths(study: Path, paper_id: str) -> list[Path]:
    paths = [study / "papers" / paper_id / "fusion"]
    for category in ("reports", "human_evaluation"):
        for system in DEPENDENT_SYSTEMS:
            for suffix in (".json", ".md", ".json.tmp"):
                paths.append(study / category / system / f"{paper_id}{suffix}")
    paths.extend((study / "pairwise").glob(f"{paper_id}__*.json"))
    return paths


def _clean_paper(
    study: Path, paper_id: str, archive: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    root = study / "papers" / paper_id
    branch = root / "gear"
    result = _analysis(branch / "analysis.json")
    record: dict[str, Any] = {
        "paper_id": paper_id,
        "kept_claims": [],
        "retry_claims": [],
    }
    if result is not None and result.status == BranchStatus.COMPLETE:
        record.update(action="kept_complete_paper", assessments=len(result.assessments))
        return record
    if not branch.exists():
        record["action"] = "not_started"
        return record
    shared = ClaimSet.model_validate_json(
        (root / "shared/claims.json").read_text(encoding="utf-8")
    )
    if shared.paper_id != paper_id:
        raise ValueError(f"Shared paper identity mismatch: {paper_id}")
    expected = {claim.claim_id.rsplit("::", 1)[-1]: claim for claim in shared.claims}
    errors = result.limitations if result is not None else []
    for suffix, claim in expected.items():
        directory = branch / suffix
        problem = _claim_problem(directory, claim)
        if any(error.startswith(f"{claim.claim_id}:") for error in errors):
            problem = "recorded_branch_execution_failure"
        if problem is None:
            record["kept_claims"].append(claim.claim_id)
            continue
        record["retry_claims"].append({"claim_id": claim.claim_id, "reason": problem})
        _move(directory, study, archive, manifest)
    for child in list(branch.iterdir()):
        if child.is_file() or child.name not in expected:
            _move(child, study, archive, manifest)
    for path in _dependent_paths(study, paper_id):
        _move(path, study, archive, manifest)
    record["action"] = "cleaned"
    return record


def _invalidate_status(
    study: Path, archive: Path, paper_ids: set[str], manifest: dict[str, Any]
) -> None:
    for stage in (
        "run_gear",
        "run_fusion",
        "generate_reports",
        "evaluate_human",
        "compare_reports",
    ):
        path = study / "status" / f"{stage}.json"
        if not path.is_file():
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            continue
        changed = False
        for row in rows:
            task_id = str(row.get("paper_id", ""))
            paper_id, _, system = task_id.partition("__")
            if paper_id not in paper_ids or system in ("direct_llm", "graph"):
                continue
            if row.get("status") == "pending" and row.get("reason") == "gear_cleanup":
                continue
            row.clear()
            row.update(paper_id=task_id, status="pending", reason="gear_cleanup")
            changed = True
        if changed:
            backup = archive / "status_before_cleanup" / path.name
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
            write_json(path, rows)
            manifest["invalidated_status_files"].append(str(path.relative_to(study)))


def _recover(
    study: Path, paper_ids: list[str], *, failed_only: bool, coverage_only: bool = False
) -> dict[str, Any]:
    """Caller must hold study_gear_lock; all evidence moves retain complete directories."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    archive = study / "status/gear_recovery" / f"{stamp}_{uuid4().hex[:8]}"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "mode": (
            "repair_coverage"
            if coverage_only
            else ("retry_failed" if failed_only else "clean_limited")
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "archive": str(archive),
        "status": "running",
        "moved": [],
        "papers": [],
        "invalidated_status_files": [],
    }
    write_json(archive / "manifest.json", manifest)
    try:
        for paper_id in paper_ids:
            if Path(paper_id).name != paper_id or paper_id in (".", ".."):
                raise ValueError("Paper ID must be a single path component")
            with stage_lock(study / "papers" / paper_id / ".locks/gear"):
                clean = (
                    _repair_coverage_paper
                    if coverage_only
                    else (_retry_paper if failed_only else _clean_paper)
                )
                record = clean(study, paper_id, archive, manifest)
                manifest["papers"].append(record)
                write_json(archive / "manifest.json", manifest)
        changed = {
            row["paper_id"] for row in manifest["papers"] if row["action"] == "cleaned"
        }
        _invalidate_status(study, archive, changed, manifest)
        if changed and manifest["moved"]:
            for path in (
                study / "summary.md",
                study / "summary.json",
                study / "tables",
            ):
                _move(path, study, archive, manifest)
        manifest["status"] = "complete"
    except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}:{exc}")
        write_json(archive / "manifest.json", manifest)
        raise
    write_json(archive / "manifest.json", manifest)
    write_json(study / "status/gear_cleanup.json", manifest)
    return manifest


def clean_limited(study: Path, paper_ids: list[str]) -> dict[str, Any]:
    """One-time broad cleanup; caller holds study_gear_lock."""
    return _recover(study, paper_ids, failed_only=False)


def retry_failed(study: Path, paper_ids: list[str]) -> dict[str, Any]:
    """Retry recorded failures while preserving scientific limitations and healthy cards."""
    return _recover(study, paper_ids, failed_only=True)


def execution_summary(
    root: Path, result: AnalysisResult, expected: int
) -> dict[str, Any]:
    errors = list(result.limitations)
    known = 0
    coverage: dict[str, Any] = {}
    for path in (root / "gear").glob("*/execution.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        known += 1
        errors.extend(str(error) for error in row.get("errors", []))
        if row.get("status") == "failed" and not row.get("errors"):
            errors.append(f"{path.parent.name}:execution_failed")
        if row.get("status") not in ("complete", "failed"):
            errors.append(f"{path.parent.name}:execution_incomplete")
        evidence = EvidenceStore(path.parent)
        for key in evidence.ids():
            record = evidence.get(key)
            if record is None:
                continue
            if record.kind == "retrieval_coverage":
                payload = record.payload
                coverage[path.parent.name] = {
                    "sufficient": payload["coverage_sufficient"],
                    "missing_query_roles": sorted(
                        set(payload["required_query_roles"])
                        - set(payload["completed_query_roles"])
                    ),
                    "unique_eligible_count": payload["unique_eligible_count"],
                    "compared_count": len(payload["compared_work_ids"]),
                }
    if len(result.assessments) != expected:
        errors.append(f"missing_assessments:{len(result.assessments)}/{expected}")
    return {
        "branch_status": result.status.value,
        "scientific_limited": any(row.limitations for row in result.assessments),
        "execution_status": (
            "completed_with_errors"
            if errors
            else ("completed" if known == expected else "legacy_unknown")
        ),
        "execution_errors": errors,
        "expected_claims": expected,
        "coverage_status": (
            "sufficient"
            if len(coverage) == expected
            and all(x["sufficient"] for x in coverage.values())
            else "limited_or_unknown"
        ),
        "coverage_by_claim": coverage,
    }


def _repair_coverage_paper(
    study: Path, paper_id: str, archive: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    root = study / "papers" / paper_id
    branch = root / "gear"
    record: dict[str, Any] = {
        "paper_id": paper_id,
        "kept_claims": [],
        "retry_claims": [],
    }
    if not branch.exists():
        return {**record, "action": "not_started"}
    shared = ClaimSet.model_validate_json((root / "shared/claims.json").read_text())
    for claim in shared.claims:
        directory = branch / claim.claim_id.rsplit("::", 1)[-1]
        if not directory.exists() or (directory / "recovery_source.json").exists():
            continue
        issue = _recorded_execution_problem(directory) or _healthy_card_problem(
            directory, claim
        )
        coverage = None
        if issue is None:
            coverage = EvidenceStore(directory).get(f"COVERAGE:{claim.claim_id}")
        missing = (
            coverage is not None
            and "legacy_contrastive" not in coverage.payload["completed_query_roles"]
        )
        if issue is None and not missing:
            record["kept_claims"].append(claim.claim_id)
            continue
        destination = archive / "artifacts" / directory.relative_to(study)
        _move(directory, study, archive, manifest)
        if issue is None:
            write_json(
                directory / "recovery_source.json",
                {
                    "source": str(destination.resolve()),
                    "claim_id": claim.claim_id,
                    "reason": "missing_legacy_contrastive",
                    "mode": "supplement_evidence",
                },
            )
        record["retry_claims"].append(
            {
                "claim_id": claim.claim_id,
                "action": "retry_evidence" if issue else "supplement_evidence",
                "reason": issue or "missing_legacy_contrastive",
            }
        )
    if record["retry_claims"]:
        _move(branch / "analysis.json", study, archive, manifest)
        for path in _dependent_paths(study, paper_id):
            _move(path, study, archive, manifest)
        record["action"] = "cleaned"
    else:
        record["action"] = "kept_existing_paper"
    return record


def repair_coverage(study: Path, paper_ids: list[str]) -> dict[str, Any]:
    """Archive affected attempts and seed healthy evidence for contrastive repair."""
    return _recover(study, paper_ids, failed_only=False, coverage_only=True)


def supplement_claims(study: Path, plan: list[dict[str, Any]]) -> dict[str, Any]:
    """Archive only explicitly selected claims, retaining evidence for new queries.

    Caller holds study_gear_lock. Validate the entire plan before moving files.
    Replaying the same plan resumes its existing attempt, never cleans it again.
    """
    from experiments.innovation_200.common import experiment_config

    maximum = experiment_config().retrieval.normal_max
    seen: set[str] = set()
    selected: list[tuple[str, Path, dict[str, Any]]] = []
    for row in plan:
        claim_id = str(row["claim_id"])
        parts = claim_id.split("::CLAIM::")
        if len(parts) != 2 or any(Path(p).name != p or p in (".", "..") for p in parts):
            raise ValueError("Invalid supplemental claim ID")
        if claim_id in seen:
            raise ValueError("Duplicate supplemental claim ID")
        seen.add(claim_id)
        paper_id, suffix = parts
        root = study / "papers" / paper_id
        claims = ClaimSet.model_validate_json((root / "shared/claims.json").read_text())
        claim = next((c for c in claims.claims if c.claim_id == claim_id), None)
        if claim is None:
            raise ValueError(f"Unknown supplemental claim: {claim_id}")
        queries = [QuerySpec.model_validate(q) for q in row["queries"]]
        if not 1 <= len(queries) <= maximum or any(
            q.claim_id != claim_id
            or q.search_mode not in ("text", "semantic")
            or not q.query.strip()
            for q in queries
        ):
            raise ValueError(f"Invalid supplemental queries: {claim_id}")
        payload = [q.model_dump(mode="json") for q in queries]
        directory = root / "gear" / suffix
        marker = directory / "recovery_source.json"
        if marker.exists():
            previous = json.loads(marker.read_text())
            if previous.get("supplemental_queries") == payload:
                continue
            if (
                not (directory / "gear_card.json").is_file()
                or not (directory / "assessment.json").is_file()
            ):
                raise ValueError(
                    f"Claim already has an unfinished recovery attempt: {claim_id}"
                )
        issue = _recorded_execution_problem(directory) or _healthy_card_problem(
            directory, claim
        )
        issue = issue or _assessment_problem(directory, claim)
        coverage = EvidenceStore(directory).get(f"COVERAGE:{claim_id}")
        if issue or coverage is None or coverage.payload["coverage_sufficient"]:
            raise ValueError(f"Not a healthy coverage-gap claim: {claim_id}: {issue}")
        selected.append(
            (
                paper_id,
                directory,
                {"claim_id": claim_id, "supplemental_queries": payload},
            )
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    archive = study / "status/gear_recovery" / f"{stamp}_{uuid4().hex[:8]}"
    manifest: dict[str, Any] = {
        "mode": "targeted_supplement",
        "archive": str(archive),
        "moved": [],
        "claims": [],
        "invalidated_status_files": [],
        "status": "running",
    }
    write_json(archive / "manifest.json", manifest)
    for paper_id, directory, marker in selected:
        with stage_lock(study / "papers" / paper_id / ".locks/gear"):
            source = archive / "artifacts" / directory.relative_to(study)
            _move(directory, study, archive, manifest)
            write_json(
                directory / "recovery_source.json",
                {
                    **marker,
                    "source": str(source.resolve()),
                    "mode": "supplement_evidence",
                    "reason": "targeted_candidate_shortfall",
                },
            )
            manifest["claims"].append(marker["claim_id"])
            _move(directory.parent / "analysis.json", study, archive, manifest)
            for path in _dependent_paths(study, paper_id):
                _move(path, study, archive, manifest)
    _invalidate_status(study, archive, {p for p, _, _ in selected}, manifest)
    manifest["status"] = "complete"
    write_json(archive / "manifest.json", manifest)
    return manifest

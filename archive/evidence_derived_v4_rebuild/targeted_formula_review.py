from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import indicators
from build_targeted_formula_completion_worklist_v3 import (
    PROTECTED_FIELDS,
    REVIEW_FIELDS,
)
from common import (
    DATABASE_PATH,
    OUTPUT_DIR,
    normalize_term,
    parse_bool,
    read_json,
    sha256_file,
    utc_now,
    write_csv,
    write_json,
)
from database import (
    initialize,
    invalidate_stages,
    log_event,
    snapshot_import_file,
)


ROOT = Path(__file__).resolve().parent
H1_PROTOCOL = ROOT / "INDEPENDENT_CODEX_TARGETED_FORMULA_H1_PROTOCOL_V3.json"
H2_PROTOCOL = ROOT / "INDEPENDENT_CODEX_TARGETED_FORMULA_H2_PROTOCOL_V3.json"
DEFAULT_H1_BLANK = (
    OUTPUT_DIR
    / "human_tasks"
    / "formal_terminal_targeted_formula_H1_BLIND_v3.csv"
)
DEFAULT_H1_REVIEWED = (
    OUTPUT_DIR
    / "independent_codex_review_v3"
    / "formal_terminal_targeted_formula_H1_REVIEWED_v3.csv"
)
DEFAULT_H2_WORKLIST = (
    OUTPUT_DIR
    / "human_tasks"
    / "formal_terminal_targeted_formula_H2_ADJUDICATE_v3.csv"
)
DEFAULT_H2_REVIEWED = (
    OUTPUT_DIR
    / "independent_codex_review_v3"
    / "formal_terminal_targeted_formula_H2_REVIEWED_v3.csv"
)
DEFAULT_APPLICATION_REPORT = (
    OUTPUT_DIR / "targeted_formula_review_application_v3.json"
)

INDEPENDENT_PROVENANCE_FIELDS = (
    "draft_method",
    "independent_ai_review_status",
    "independent_ai_reviewer_id",
    "independent_ai_reviewed_at",
    "independent_ai_review_action",
    "independent_ai_review_note",
    "independent_ai_run_id",
    "independent_ai_model",
    "independent_ai_prompt_sha256",
)
H1_FIELDS = (*PROTECTED_FIELDS, *REVIEW_FIELDS)
H2_FIELDS = (
    *PROTECTED_FIELDS,
    "h1_payload_json",
    *REVIEW_FIELDS,
    *INDEPENDENT_PROVENANCE_FIELDS,
)
H1_DECISIONS = {
    "approve_formula",
    "reject_construct",
    "reject_formula_missing",
    "reject_data_mismatch",
    "reject_future_information",
    "uncertain",
}
H2_DECISIONS = H1_DECISIONS - {"uncertain"}
FORMULA_AUTHORIZING_SOURCE_ROLES = {
    "original_definition",
    "original_application",
    "mathematical_foundation",
}
APPROVAL_SOURCE_ROLES = {
    *FORMULA_AUTHORIZING_SOURCE_ROLES,
    "validation",
}
FORMULA_SCOPE_ROLES = {
    "direct_innovation",
    "t0_substantive",
    "t0_opportunity",
    "context_control",
}
SELECTION_PRIORITY = {
    "original_definition": 0,
    "mathematical_foundation": 1,
    "original_application": 2,
    "validation": 3,
}
EVIDENCE_LEVELS = {
    "systematic_review_plus_primary",
    "high",
    "moderate",
    "limited",
    "weak",
    "unknown",
}
BIAS_POLICY = {
    "direct_innovation": "allowed_core",
    "t0_substantive": "allowed_core",
    "t0_opportunity": "allowed_opportunity",
    "context_control": "allowed_context",
}
FORMULA_LOCATION_PATTERN = re.compile(
    r"\b(?:p(?:age)?s?\.?\s*\d+|table\s+\w+|"
    r"eq(?:uation)?\.?\s*\w+|"
    r"appendix\s+\w+|section\s+\w+)",
    flags=re.IGNORECASE,
)


def _rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _manifest_path(artifact_path: Path) -> Path:
    return artifact_path.with_suffix(".manifest.json")


def _require_manifest(
    manifest_path: Path,
    artifact_path: Path,
    input_path: Path,
    role: str,
    protocol_path: Path,
    row_count: int,
) -> Dict[str, Any]:
    manifest = read_json(manifest_path)
    required = {
        "run_id",
        "artifact_path",
        "artifact_sha256",
        "input_path",
        "input_sha256",
        "reviewer_role",
        "reviewer_id",
        "model",
        "model_digest",
        "prompt_sha256",
        "parameters",
        "item_count",
        "completed_at",
        "status",
    }
    missing = sorted(
        field
        for field in required
        if manifest.get(field) is None
        or (
            field != "parameters"
            and not str(manifest.get(field) or "").strip()
        )
    )
    if missing:
        raise ValueError(f"Review manifest lacks fields: {missing}")
    checks = {
        "artifact": (
            sha256_file(artifact_path),
            str(manifest["artifact_sha256"]),
        ),
        "input": (sha256_file(input_path), str(manifest["input_sha256"])),
        "protocol": (
            sha256_file(protocol_path),
            str(manifest["prompt_sha256"]),
        ),
    }
    for label, (observed, expected) in checks.items():
        if observed.casefold() != expected.casefold():
            raise ValueError(f"{role} {label} hash mismatch")
    if str(manifest["reviewer_role"]).upper() != role:
        raise ValueError(f"Manifest role is not {role}")
    if int(manifest["item_count"]) != row_count:
        raise ValueError(f"{role} manifest item_count mismatch")
    if str(manifest["status"]).casefold() != "complete":
        raise ValueError(f"{role} review manifest is not complete")
    if not isinstance(manifest["parameters"], dict):
        raise ValueError(f"{role} manifest parameters must be an object")
    identity = (
        f"{manifest['model']} {manifest['model_digest']}".casefold()
    )
    if any(value in identity for value in ("qwen", "ollama")):
        raise ValueError(f"{role} used a prohibited model")
    return manifest


def _assert_same_order(
    expected: Sequence[Mapping[str, str]],
    observed: Sequence[Mapping[str, str]],
) -> None:
    expected_ids = [str(row.get("target_id") or "") for row in expected]
    observed_ids = [str(row.get("target_id") or "") for row in observed]
    if (
        not expected_ids
        or "" in expected_ids
        or len(set(expected_ids)) != len(expected_ids)
        or expected_ids != observed_ids
    ):
        raise ValueError("Target row set or order changed")


def _assert_fields_unchanged(
    expected: Sequence[Mapping[str, str]],
    observed: Sequence[Mapping[str, str]],
    fields: Iterable[str],
) -> None:
    for line_number, (before, after) in enumerate(
        zip(expected, observed),
        start=2,
    ):
        for field in fields:
            if str(before.get(field) or "") != str(after.get(field) or ""):
                raise ValueError(
                    f"Protected field {field} changed at line {line_number}"
                )


def _required(row: Mapping[str, str], field: str) -> str:
    value = str(row.get(field) or "").strip()
    if not value:
        raise ValueError(f"{row.get('target_id')} requires {field}")
    return value


def _json_list(row: Mapping[str, str], field: str) -> List[str]:
    try:
        value = json.loads(_required(row, field))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{row.get('target_id')} has invalid {field}"
        ) from error
    if not isinstance(value, list) or not value:
        raise ValueError(f"{row.get('target_id')} requires a nonempty {field}")
    normalized = [str(item).strip() for item in value]
    if any(not item for item in normalized):
        raise ValueError(f"{row.get('target_id')} has blank {field} values")
    return normalized


def _verify_fulltext_spans(row: Mapping[str, str]) -> None:
    path = Path(_required(row, "fulltext_local_path")).resolve()
    digest = _required(row, "fulltext_sha256").casefold()
    if not path.is_file() or sha256_file(path).casefold() != digest:
        raise ValueError(f"Full-text hash mismatch: {row.get('target_id')}")
    text = indicators._extract_fulltext_text(path, digest)
    normalized_text = indicators.normalize_text(text)
    for field in ("evidence_span", "research_group_evidence"):
        span = indicators.normalize_text(_required(row, field))
        if len(span.split()) < 2 or span not in normalized_text:
            raise ValueError(
                f"{field} is absent from full text: {row.get('target_id')}"
            )


def _validate_approval(row: Mapping[str, str], role: str) -> None:
    target_id = str(row.get("target_id") or "")
    required_fields = (
        "raw_name_en",
        "label_zh",
        "research_group",
        "research_group_id",
        "research_group_evidence",
        "source_role",
        "formula_location",
        "evidence_span",
        "formula",
        "units",
        "parameters",
        "direction",
        "source_reported_missing_rule",
        "required_data_json",
        "maximum_information_time",
        "scope_role",
        "validation_summary",
        "evidence_direction",
        "evidence_strength",
        "stability_score",
        "stability_basis",
        "selection_priority",
        "redundancy_family",
    )
    for field in required_fields:
        _required(row, field)
    source_role = str(row["source_role"]).casefold()
    scope_role = str(row["scope_role"]).casefold()
    if source_role not in APPROVAL_SOURCE_ROLES:
        raise ValueError(f"Non-authorizing source role: {target_id}")
    if scope_role not in FORMULA_SCOPE_ROLES:
        raise ValueError(f"Out-of-scope formula approval: {target_id}")
    if int(row["selection_priority"]) != SELECTION_PRIORITY[source_role]:
        raise ValueError(f"Selection priority mismatch: {target_id}")
    if not FORMULA_LOCATION_PATTERN.search(str(row["formula_location"])):
        raise ValueError(f"Formula location is not auditable: {target_id}")
    if str(row["maximum_information_time"]).casefold() != "t0":
        raise ValueError(f"Formula is not bounded at T0: {target_id}")
    booleans = {
        field: parse_bool(row.get(field), field)
        for field in (
            "article_level",
            "primary_or_foundational_evidence",
            "formula_reproducible",
            "t0_computable",
            "requires_future",
            "fatal_validity_concern",
        )
    }
    if not all(
        booleans[field]
        for field in (
            "article_level",
            "formula_reproducible",
            "t0_computable",
        )
    ) or booleans["requires_future"]:
        raise ValueError(
            f"Formula approval fails T0/source gates: {target_id}"
        )
    if (
        role == "H2"
        and source_role in FORMULA_AUTHORIZING_SOURCE_ROLES
        and not booleans["primary_or_foundational_evidence"]
    ):
        raise ValueError(
            f"H2 formula-authorizing source is not primary: {target_id}"
        )
    _json_list(row, "required_data_json")
    evidence_strength = str(row["evidence_strength"]).casefold()
    allowed_strengths = (
        EVIDENCE_LEVELS | {"strong"}
        if role == "H1"
        else EVIDENCE_LEVELS
    )
    if evidence_strength not in allowed_strengths:
        raise ValueError(f"Invalid evidence strength: {target_id}")
    score = float(row["stability_score"])
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"Invalid stability score: {target_id}")
    _verify_fulltext_spans(row)


def _validate_review_rows(
    rows: Sequence[Mapping[str, str]],
    role: str,
) -> Counter[str]:
    allowed = H1_DECISIONS if role == "H1" else H2_DECISIONS
    counts: Counter[str] = Counter()
    prompt_sha = sha256_file(H1_PROTOCOL if role == "H1" else H2_PROTOCOL)
    for row in rows:
        if str(row.get("reviewer_role") or "").upper() != role:
            raise ValueError(f"Review row role is not {role}")
        decision = _required(row, "decision").casefold()
        if decision not in allowed:
            raise ValueError(
                f"Invalid {role} decision for {row.get('target_id')}"
            )
        _required(row, "review_reason")
        _required(row, "reviewer_provenance")
        if role == "H2":
            for field in INDEPENDENT_PROVENANCE_FIELDS:
                _required(row, field)
            if (
                str(row["independent_ai_prompt_sha256"]).casefold()
                != prompt_sha
            ):
                raise ValueError("H2 row-level prompt hash mismatch")
        if decision == "approve_formula":
            _validate_approval(row, role)
        counts[decision] += 1
    return counts


def validate_h1_review(
    blank_path: Path,
    reviewed_path: Path,
    manifest_path: Path,
) -> Dict[str, Any]:
    blank = _rows(blank_path)
    reviewed = _rows(reviewed_path)
    if len(blank) != len(reviewed):
        raise ValueError("H1 row count changed")
    _assert_same_order(blank, reviewed)
    _assert_fields_unchanged(blank, reviewed, PROTECTED_FIELDS)
    counts = _validate_review_rows(reviewed, "H1")
    manifest = _require_manifest(
        manifest_path,
        reviewed_path,
        blank_path,
        "H1",
        H1_PROTOCOL,
        len(reviewed),
    )
    return {
        "rows": reviewed,
        "counts": dict(sorted(counts.items())),
        "manifest": manifest,
    }


def build_h2_worklist(
    blank_path: Path,
    h1_path: Path,
    h1_manifest_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    result = validate_h1_review(blank_path, h1_path, h1_manifest_path)
    rows: List[Dict[str, str]] = []
    for h1_row in result["rows"]:
        row = {field: "" for field in H2_FIELDS}
        for field in PROTECTED_FIELDS:
            row[field] = str(h1_row.get(field) or "")
        row["h1_payload_json"] = json.dumps(
            {
                field: str(h1_row.get(field) or "")
                for field in REVIEW_FIELDS
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        row["reviewer_role"] = "H2"
        rows.append(row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_path, rows, H2_FIELDS)
    manifest_path = _manifest_path(output_path)
    manifest = {
        "schema_version": "targeted_formula_h2_worklist_manifest_v3",
        "created_at": utc_now(),
        "artifact_path": str(output_path.resolve()),
        "artifact_sha256": sha256_file(output_path),
        "item_count": len(rows),
        "h1_reviewed_path": str(h1_path.resolve()),
        "h1_reviewed_sha256": sha256_file(h1_path),
        "h1_manifest_path": str(h1_manifest_path.resolve()),
        "h1_manifest_sha256": sha256_file(h1_manifest_path),
        "h2_protocol_path": str(H2_PROTOCOL.resolve()),
        "h2_protocol_sha256": sha256_file(H2_PROTOCOL),
        "protected_fields": [*PROTECTED_FIELDS, "h1_payload_json"],
        "review_fields": list(REVIEW_FIELDS),
        "provenance_fields": list(INDEPENDENT_PROVENANCE_FIELDS),
        "round_13": False,
        "target_count_is_not_a_selection_quota": True,
    }
    write_json(manifest_path, manifest)
    return {
        "output": str(output_path),
        "sha256": sha256_file(output_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "rows": len(rows),
        "h1_counts": result["counts"],
    }


def validate_h2_review(
    worklist_path: Path,
    reviewed_path: Path,
    manifest_path: Path,
) -> Dict[str, Any]:
    worklist = _rows(worklist_path)
    reviewed = _rows(reviewed_path)
    if len(worklist) != len(reviewed):
        raise ValueError("H2 row count changed")
    _assert_same_order(worklist, reviewed)
    _assert_fields_unchanged(
        worklist,
        reviewed,
        (*PROTECTED_FIELDS, "h1_payload_json"),
    )
    counts = _validate_review_rows(reviewed, "H2")
    manifest = _require_manifest(
        manifest_path,
        reviewed_path,
        worklist_path,
        "H2",
        H2_PROTOCOL,
        len(reviewed),
    )
    return {
        "rows": reviewed,
        "counts": dict(sorted(counts.items())),
        "manifest": manifest,
    }


def _family_rows(
    connection: sqlite3.Connection,
    canonical_name: str,
) -> List[sqlite3.Row]:
    return connection.execute(
        """
        SELECT * FROM indicator_mentions
        WHERE canonical_name_en = ? AND status != 'excluded'
        ORDER BY mention_id
        """,
        (canonical_name,),
    ).fetchall()


def _refresh_family(
    connection: sqlite3.Connection,
    feature_id: str,
    canonical_name: str,
) -> None:
    rows = _family_rows(connection, canonical_name)
    if not rows:
        raise ValueError(f"No retained mentions for {feature_id}")
    representative = indicators._representative_mention(rows)
    formula_rows = [
        row
        for row in rows
        if indicators._literature_formula_complete_except_missing_rule(row)
    ]
    formula_representative = (
        indicators._representative_mention(formula_rows)
        if formula_rows
        else representative
    )
    aliases = sorted(
        {str(row["raw_name_en"]).strip() for row in rows},
        key=normalize_term,
    )
    research_groups = sorted(
        {
            normalize_term(row["research_group_id"])
            for row in rows
            if normalize_term(row["research_group_id"])
            and row["english_fulltext_verified"]
            and row["h2_approved"]
            and str(row["research_group_evidence"]).strip()
        }
    )
    formula_complete = any(
        row["formula_reproducible"]
        and indicators._is_reported_formula_value(row["missing_rule"])
        for row in formula_rows
    )
    connection.execute(
        """
        UPDATE indicator_families SET
            alias_names_json = ?, mention_ids_json = ?, formula = ?,
            units = ?, parameters = ?, direction = ?, missing_rule = ?,
            required_data_json = ?, maximum_information_time = ?,
            scope_role = ?, article_level = ?,
            primary_or_foundational_evidence = ?,
            formula_reproducible = ?, t0_computable = ?,
            requires_future = ?, data_status = ?, bias_policy = ?,
            fatal_validity_concern = ?, uses_outcome_for_selection = ?,
            quality_audit_status = ?, nonconstant = ?,
            english_fulltext_verified = ?, h2_approved = ?,
            evidence_strength = ?, stability_score = ?,
            stability_basis = ?, selection_priority = ?,
            redundancy_family = ?, research_groups_json = ?,
            status = 'candidate'
        WHERE feature_id = ? AND canonical_name_en = ?
        """,
        (
            json.dumps(aliases, ensure_ascii=False),
            json.dumps(
                sorted(str(row["mention_id"]) for row in rows),
                ensure_ascii=False,
            ),
            formula_representative["formula"],
            formula_representative["units"],
            formula_representative["parameters"],
            formula_representative["direction"],
            formula_representative["missing_rule"],
            formula_representative["required_data_json"],
            formula_representative["maximum_information_time"],
            formula_representative["scope_role"],
            int(formula_representative["article_level"]),
            int(any(row["primary_or_foundational_evidence"] for row in rows)),
            int(formula_complete),
            int(formula_representative["t0_computable"]),
            int(formula_representative["requires_future"]),
            formula_representative["data_status"],
            formula_representative["bias_policy"],
            int(any(row["fatal_validity_concern"] for row in rows)),
            int(any(row["uses_outcome_for_selection"] for row in rows)),
            formula_representative["quality_audit_status"],
            int(formula_representative["nonconstant"]),
            int(bool(formula_rows)),
            int(bool(formula_rows)),
            max(
                (str(row["evidence_strength"]) for row in rows),
                key=indicators._evidence_rank,
            ),
            float(representative["stability_score"]),
            representative["stability_basis"],
            int(representative["selection_priority"]),
            representative["redundancy_family"],
            json.dumps(research_groups, ensure_ascii=False),
            feature_id,
            canonical_name,
        ),
    )
    if connection.execute("SELECT changes()").fetchone()[0] != 1:
        raise ValueError(f"Family identity changed for {feature_id}")


def _mention_payload(
    connection: sqlite3.Connection,
    row: Mapping[str, str],
    mention_id: str,
) -> Dict[str, Any]:
    scope_role = str(row["scope_role"]).casefold()
    required_data = _json_list(row, "required_data_json")
    fulltext = indicators._register_fulltext_evidence(
        connection,
        {
            **row,
            "fulltext_source_url": row["fulltext_source_url"],
            "fulltext_local_path": row["fulltext_local_path"],
            "fulltext_sha256": row["fulltext_sha256"],
            "fulltext_license": row["fulltext_license"],
        },
        mention_id,
        str(row["target_id"]),
    )
    return {
        "mention_id": mention_id,
        "record_key": row["record_key"],
        "raw_name_en": row["raw_name_en"].strip(),
        "canonical_name_en": row["canonical_name_en"],
        "label_zh": row["label_zh"].strip(),
        "source_id": row["record_key"],
        "research_group": row["research_group"].strip(),
        "research_group_id": normalize_term(row["research_group_id"]),
        "research_group_evidence": row["research_group_evidence"].strip(),
        "source_role": row["source_role"].casefold(),
        "formula_location": row["formula_location"].strip(),
        "evidence_span": row["evidence_span"].strip(),
        "formula": row["formula"].strip(),
        "units": row["units"].strip(),
        "parameters": row["parameters"].strip(),
        "direction": row["direction"].strip(),
        "missing_rule": row["source_reported_missing_rule"].strip(),
        "required_data_json": json.dumps(required_data, ensure_ascii=False),
        "maximum_information_time": "T0",
        "scope_role": scope_role,
        "validation_summary": row["validation_summary"].strip(),
        "evidence_direction": row["evidence_direction"].casefold(),
        "negative_evidence": str(row.get("negative_evidence") or "").strip(),
        **fulltext,
        "english_fulltext_verified": 1,
        "article_level": 1,
        "primary_or_foundational_evidence": int(
            parse_bool(
                row["primary_or_foundational_evidence"],
                "primary_or_foundational_evidence",
            )
        ),
        "formula_reproducible": 1,
        "t0_computable": 1,
        "requires_future": 0,
        "data_status": "unassessed",
        "bias_policy": BIAS_POLICY[scope_role],
        "fatal_validity_concern": int(
            parse_bool(row["fatal_validity_concern"], "fatal_validity_concern")
        ),
        "uses_outcome_for_selection": 0,
        "quality_audit_status": "pending",
        "nonconstant": 0,
        "h2_approved": 1,
        "evidence_strength": row["evidence_strength"].casefold(),
        "stability_score": float(row["stability_score"]),
        "stability_basis": row["stability_basis"].strip(),
        "selection_priority": int(row["selection_priority"]),
        "redundancy_family": row["redundancy_family"].strip(),
        "extracted_by": "H1",
        "verified_by": "H1|H2",
        "verification_notes": json.loads(row["h1_payload_json"])[
            "review_reason"
        ],
        "adjudication_notes": row["review_reason"].strip(),
        "status": "candidate",
    }


def _upsert_mention(
    connection: sqlite3.Connection,
    values: Mapping[str, Any],
) -> None:
    fields = tuple(values)
    placeholders = ", ".join("?" for _ in fields)
    updates = ", ".join(
        f"{field} = excluded.{field}"
        for field in fields
        if field != "mention_id"
    )
    connection.execute(
        f"""
        INSERT INTO indicator_mentions({", ".join(fields)})
        VALUES ({placeholders})
        ON CONFLICT(mention_id) DO UPDATE SET {updates}
        """,
        tuple(values[field] for field in fields),
    )


def _store_target_review(
    connection: sqlite3.Connection,
    row: Mapping[str, str],
    role: str,
    artifact_sha256: str,
) -> None:
    payload = {
        field: str(row.get(field) or "")
        for field in REVIEW_FIELDS
    }
    connection.execute(
        """
        INSERT INTO targeted_formula_reviews(
            target_id, feature_id, record_key, reviewer_role, decision,
            payload_json, reason, artifact_sha256, reviewed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(target_id, reviewer_role) DO UPDATE SET
            feature_id = excluded.feature_id,
            record_key = excluded.record_key,
            decision = excluded.decision,
            payload_json = excluded.payload_json,
            reason = excluded.reason,
            artifact_sha256 = excluded.artifact_sha256,
            reviewed_at = excluded.reviewed_at
        """,
        (
            row["target_id"],
            row["feature_id"],
            row["record_key"],
            role,
            row["decision"].casefold(),
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            row["review_reason"].strip(),
            artifact_sha256,
            utc_now(),
        ),
    )


def _store_mention_reviews(
    connection: sqlite3.Connection,
    mention_id: str,
    h1_row: Mapping[str, str],
    h2_row: Mapping[str, str],
) -> None:
    h1_decision = (
        "candidate"
        if str(h1_row["decision"]).casefold() == "approve_formula"
        else "excluded"
    )
    indicators._store_mention_review(
        connection,
        mention_id,
        "H1",
        h1_decision,
        json.dumps(dict(h1_row), ensure_ascii=False, sort_keys=True),
        h1_row["review_reason"].strip(),
    )
    indicators._store_mention_review(
        connection,
        mention_id,
        "H2",
        "approve",
        json.dumps(dict(h2_row), ensure_ascii=False, sort_keys=True),
        h2_row["review_reason"].strip(),
    )


def _snapshot_reviews(
    connection: sqlite3.Connection,
    paths: Sequence[tuple[Path, str]],
) -> None:
    for path, role in paths:
        snapshot_import_file(connection, path, role)


def apply_reviews(
    connection: sqlite3.Connection,
    blank_path: Path,
    h1_path: Path,
    h1_manifest_path: Path,
    h2_worklist_path: Path,
    h2_path: Path,
    h2_manifest_path: Path,
    report_path: Path,
) -> Dict[str, Any]:
    h1_result = validate_h1_review(
        blank_path,
        h1_path,
        h1_manifest_path,
    )
    h2_result = validate_h2_review(
        h2_worklist_path,
        h2_path,
        h2_manifest_path,
    )
    h1_rows = {
        row["target_id"]: row for row in h1_result["rows"]
    }
    h1_sha = sha256_file(h1_path)
    h2_sha = sha256_file(h2_path)
    prior = connection.execute(
        """
        SELECT COUNT(*) AS n,
               COUNT(DISTINCT h1_artifact_sha256) AS h1_n,
               COUNT(DISTINCT h2_artifact_sha256) AS h2_n,
               MIN(h1_artifact_sha256) AS h1_sha,
               MIN(h2_artifact_sha256) AS h2_sha
        FROM targeted_formula_decisions
        """
    ).fetchone()
    if prior["n"]:
        if (
            prior["h1_n"] == 1
            and prior["h2_n"] == 1
            and prior["h1_sha"] == h1_sha
            and prior["h2_sha"] == h2_sha
            and prior["n"] == len(h2_result["rows"])
        ):
            return read_json(report_path)
        raise ValueError(
            "Targeted formula decisions are frozen; use a correction "
            "artifact rather than overwriting them"
        )
    _snapshot_reviews(
        connection,
        (
            (h1_path, "targeted_formula_h1_review"),
            (h1_manifest_path, "targeted_formula_h1_manifest"),
            (h2_path, "targeted_formula_h2_review"),
            (h2_manifest_path, "targeted_formula_h2_manifest"),
        ),
    )
    approved: List[Dict[str, str]] = []
    rejected: List[Dict[str, str]] = []
    affected: Dict[str, str] = {}
    for h2_row in h2_result["rows"]:
        target_id = h2_row["target_id"]
        h1_row = h1_rows[target_id]
        _store_target_review(connection, h1_row, "H1", h1_sha)
        _store_target_review(connection, h2_row, "H2", h2_sha)
        decision = h2_row["decision"].casefold()
        mention_id = ""
        if decision == "approve_formula":
            mention_id = f"MENTION_{target_id}"
            values = _mention_payload(connection, h2_row, mention_id)
            _upsert_mention(connection, values)
            _store_mention_reviews(
                connection,
                mention_id,
                h1_row,
                h2_row,
            )
            affected[h2_row["feature_id"]] = h2_row["canonical_name_en"]
            approved.append(
                {
                    "target_id": target_id,
                    "feature_id": h2_row["feature_id"],
                    "mention_id": mention_id,
                }
            )
        else:
            rejected.append(
                {
                    "target_id": target_id,
                    "feature_id": h2_row["feature_id"],
                    "decision": decision,
                    "reason": h2_row["review_reason"],
                }
            )
        connection.execute(
            """
            INSERT INTO targeted_formula_decisions(
                target_id, feature_id, record_key, final_decision,
                mention_id, reason, h1_artifact_sha256,
                h2_artifact_sha256, decided_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                target_id,
                h2_row["feature_id"],
                h2_row["record_key"],
                decision,
                mention_id,
                h2_row["review_reason"].strip(),
                h1_sha,
                h2_sha,
                utc_now(),
            ),
        )
    for feature_id, canonical_name in sorted(affected.items()):
        _refresh_family(connection, feature_id, canonical_name)
    if affected:
        placeholders = ",".join("?" for _ in affected)
        connection.execute(
            f"""
            DELETE FROM feature_operationalization_reviews
            WHERE feature_id IN ({placeholders})
            """,
            tuple(affected),
        )
        connection.execute(
            f"""
            DELETE FROM feature_data_audit
            WHERE feature_id IN ({placeholders})
            """,
            tuple(affected),
        )
    connection.execute("DELETE FROM dimension_coding")
    connection.execute("DELETE FROM candidate_dimensions")
    connection.execute("DELETE FROM feature_decisions")
    connection.execute("DELETE FROM dimension_decisions")
    invalidate_stages(
        connection,
        (
            "operationalizations_reviewed",
            "dimensions_derived",
            "features_selected",
            "audit_complete",
        ),
        "H2-targeted formula evidence changed",
    )
    log_event(
        connection,
        "targeted_formula_reviews_applied",
        "terminal_search_frame",
        "round_12",
        {
            "h1_artifact_sha256": h1_sha,
            "h2_artifact_sha256": h2_sha,
            "approved_targets": len(approved),
            "rejected_targets": len(rejected),
            "affected_existing_families": len(affected),
            "new_indicator_families": 0,
            "round_13": False,
        },
    )
    report = {
        "schema_version": "targeted_formula_review_application_v3",
        "applied_at": utc_now(),
        "terminal_search_round": 12,
        "round_13": False,
        "h1_artifact_path": str(h1_path.resolve()),
        "h1_artifact_sha256": h1_sha,
        "h2_artifact_path": str(h2_path.resolve()),
        "h2_artifact_sha256": h2_sha,
        "h1_decision_counts": h1_result["counts"],
        "h2_decision_counts": h2_result["counts"],
        "approved_targets": approved,
        "rejected_targets": rejected,
        "affected_existing_family_count": len(affected),
        "new_indicator_family_count": 0,
        "target_count_is_not_a_selection_quota": True,
        "model_outcomes_used": False,
    }
    write_json(report_path, report)
    connection.commit()
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_h2 = subparsers.add_parser("build-h2")
    build_h2.add_argument("--h1-blank", type=Path, default=DEFAULT_H1_BLANK)
    build_h2.add_argument(
        "--h1-reviewed",
        type=Path,
        default=DEFAULT_H1_REVIEWED,
    )
    build_h2.add_argument("--h1-manifest", type=Path)
    build_h2.add_argument("--output", type=Path, default=DEFAULT_H2_WORKLIST)
    validate_h1 = subparsers.add_parser("validate-h1")
    validate_h1.add_argument(
        "--h1-blank",
        type=Path,
        default=DEFAULT_H1_BLANK,
    )
    validate_h1.add_argument(
        "--h1-reviewed",
        type=Path,
        default=DEFAULT_H1_REVIEWED,
    )
    validate_h1.add_argument("--h1-manifest", type=Path)
    apply = subparsers.add_parser("apply")
    apply.add_argument("--database", type=Path, default=DATABASE_PATH)
    apply.add_argument("--h1-blank", type=Path, default=DEFAULT_H1_BLANK)
    apply.add_argument(
        "--h1-reviewed",
        type=Path,
        default=DEFAULT_H1_REVIEWED,
    )
    apply.add_argument("--h1-manifest", type=Path)
    apply.add_argument(
        "--h2-worklist",
        type=Path,
        default=DEFAULT_H2_WORKLIST,
    )
    apply.add_argument(
        "--h2-reviewed",
        type=Path,
        default=DEFAULT_H2_REVIEWED,
    )
    apply.add_argument("--h2-manifest", type=Path)
    apply.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_APPLICATION_REPORT,
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    h1_manifest = (
        args.h1_manifest.resolve()
        if args.h1_manifest
        else _manifest_path(args.h1_reviewed.resolve())
    )
    if args.command == "validate-h1":
        result = validate_h1_review(
            args.h1_blank.resolve(),
            args.h1_reviewed.resolve(),
            h1_manifest,
        )
        print(json.dumps(result["counts"], indent=2, sort_keys=True))
        return
    if args.command == "build-h2":
        result = build_h2_worklist(
            args.h1_blank.resolve(),
            args.h1_reviewed.resolve(),
            h1_manifest,
            args.output.resolve(),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    h2_manifest = (
        args.h2_manifest.resolve()
        if args.h2_manifest
        else _manifest_path(args.h2_reviewed.resolve())
    )
    connection = initialize(args.database.resolve())
    try:
        result = apply_reviews(
            connection,
            args.h1_blank.resolve(),
            args.h1_reviewed.resolve(),
            h1_manifest,
            args.h2_worklist.resolve(),
            args.h2_reviewed.resolve(),
            h2_manifest,
            args.report.resolve(),
        )
    finally:
        connection.close()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

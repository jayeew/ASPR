from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import textwrap
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


DECISIONS = {"include", "exclude"}
LANGUAGES = {"en", "non_en", "uncertain"}
EXCLUSION_REASONS = {
    "E_LANGUAGE_NON_ENGLISH",
    "E_NOT_PAPER_LEVEL_INNOVATION_OR_POTENTIAL_IMPACT",
    "E_FUTURE_OUTCOME_ONLY",
    "E_NOT_ARTICLE_LEVEL",
    "E_NOT_INDICATOR_PREDICTOR_VALIDATION",
    "E_DUPLICATE",
    "E_INSUFFICIENT_METADATA",
}
EXCLUSION_CHOICES = {
    "1": "E_NOT_PAPER_LEVEL_INNOVATION_OR_POTENTIAL_IMPACT",
    "2": "E_FUTURE_OUTCOME_ONLY",
    "3": "E_NOT_ARTICLE_LEVEL",
    "4": "E_NOT_INDICATOR_PREDICTOR_VALIDATION",
    "5": "E_DUPLICATE",
    "6": "E_INSUFFICIENT_METADATA",
}
TERM_RELATIONS = {
    "canonical",
    "synonym",
    "abbreviation",
    "full_form",
    "historical_name",
    "morphological_variant",
    "parameter_variant",
}
REVIEW_FIELDS = (
    "human_reviewer_id",
    "human_reviewed_at",
    "human_review_action",
    "human_review_note",
)
CATEGORY_ORDER = {
    "unresolved_uncertain": 0,
    "decision_disagreement": 1,
    "language_disagreement": 2,
    "reason_disagreement": 3,
    "mandatory_audit_or_include": 4,
}


def _read_csv(path: Path) -> tuple[List[str], List[Dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def _atomic_write_csv(
    path: Path,
    fields: Sequence[str],
    rows: Iterable[Mapping[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(fields),
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                dict(payload),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _review_fields(fields: Sequence[str]) -> List[str]:
    result = list(fields)
    for field in REVIEW_FIELDS:
        if field not in result:
            result.append(field)
    return result


def _load_target(
    input_path: Path,
    output_path: Path,
) -> tuple[List[str], List[Dict[str, str]]]:
    load_path = output_path if output_path.exists() else input_path
    fields, rows = _read_csv(load_path)
    if "draft_method" not in fields:
        raise ValueError("Input is not an assisted H2 draft")
    if not rows:
        raise ValueError("Input contains no review rows")
    return _review_fields(fields), rows


def _source_text(row: Mapping[str, str]) -> str:
    return "\n".join(
        (str(row.get("title") or ""), str(row.get("abstract") or ""))
    )


def _exact_span(
    label: str,
    row: Mapping[str, str],
    default: str,
) -> str:
    source = _source_text(row)
    fallback = default if default in source else str(
        row.get("title") or ""
    ).strip()
    while True:
        value = input(f"{label} [Enter=current/title]: ").strip() or fallback
        if value and value in source:
            return value
        print("Copy an exact span from the displayed title or abstract.")


def _mark_reviewed(
    row: Dict[str, str],
    reviewer_id: str,
    action: str,
    note: str,
) -> None:
    row["human_review_status"] = "reviewed"
    row["human_reviewer_id"] = reviewer_id
    row["human_reviewed_at"] = _now()
    row["human_review_action"] = action
    row["human_review_note"] = note


def screening_category(row: Mapping[str, str]) -> str:
    """Return one exclusive risk-priority category for an H2 row."""
    if str(row.get("decision") or "").strip().casefold() == "uncertain":
        return "unresolved_uncertain"
    ai_decision = str(row.get("ai_decision") or "").strip().casefold()
    h1_decision = str(row.get("h1_decision") or "").strip().casefold()
    if ai_decision != h1_decision:
        return "decision_disagreement"
    ai_language = str(
        row.get("ai_language_judgment") or ""
    ).strip().casefold()
    h1_language = str(
        row.get("h1_language_judgment") or ""
    ).strip().casefold()
    if ai_language != h1_language:
        return "language_disagreement"
    ai_reason = str(row.get("ai_exclusion_reason") or "").strip()
    h1_reason = str(row.get("h1_exclusion_reason") or "").strip()
    if ai_reason != h1_reason:
        return "reason_disagreement"
    return "mandatory_audit_or_include"


def status_screening(path: Path) -> Dict[str, int]:
    """Summarize H2 screening work by exclusive risk category."""
    _, rows = _read_csv(path)
    categories = Counter(screening_category(row) for row in rows)
    reviewed = sum(
        str(row.get("human_review_status") or "").strip().casefold()
        == "reviewed"
        for row in rows
    )
    result = {
        "rows": len(rows),
        "reviewed": reviewed,
        "unreviewed": len(rows) - reviewed,
    }
    result.update(
        {
            category: categories[category]
            for category in CATEGORY_ORDER
        }
    )
    return result


def _screening_row_errors(row: Mapping[str, str]) -> List[str]:
    key = str(row.get("record_key") or "").strip() or "<missing-key>"
    errors: List[str] = []
    if str(row.get("reviewer_role") or "").strip().upper() != "H2":
        errors.append(f"{key}: reviewer_role must be H2")
    if (
        str(row.get("human_review_status") or "").strip().casefold()
        != "reviewed"
    ):
        errors.append(f"{key}: row is not human-reviewed")
    for field in (
        "human_reviewer_id",
        "human_reviewed_at",
        "human_review_action",
    ):
        if not str(row.get(field) or "").strip():
            errors.append(f"{key}: missing {field}")
    decision = str(row.get("decision") or "").strip().casefold()
    language = str(
        row.get("language_judgment") or ""
    ).strip().casefold()
    reason = str(row.get("exclusion_reason") or "").strip()
    if decision not in DECISIONS:
        errors.append(f"{key}: H2 decision must be include or exclude")
    if language not in LANGUAGES:
        errors.append(f"{key}: invalid language_judgment")
    if language == "uncertain":
        errors.append(f"{key}: H2 language cannot remain uncertain")
    if decision == "exclude" and reason not in EXCLUSION_REASONS:
        errors.append(f"{key}: invalid exclusion_reason")
    if decision == "include" and reason:
        errors.append(f"{key}: included row has exclusion_reason")
    if language == "non_en" and (
        decision != "exclude"
        or reason != "E_LANGUAGE_NON_ENGLISH"
    ):
        errors.append(f"{key}: non_en requires the language exclusion")
    if reason == "E_LANGUAGE_NON_ENGLISH" and language != "non_en":
        errors.append(f"{key}: language exclusion requires non_en")
    source = _source_text(row)
    for field in ("language_evidence", "evidence_span"):
        value = str(row.get(field) or "").strip()
        if not value:
            errors.append(f"{key}: missing {field}")
        elif value not in source:
            errors.append(f"{key}: {field} is not an exact source span")
    return errors


def _raise_errors(label: str, errors: Sequence[str]) -> None:
    if not errors:
        return
    sample = "\n".join(errors[:50])
    remainder = len(errors) - min(len(errors), 50)
    suffix = f"\n... plus {remainder} more" if remainder else ""
    raise ValueError(
        f"{label} has {len(errors)} validation errors:\n{sample}{suffix}"
    )


def validate_screening(path: Path) -> Dict[str, int]:
    """Validate a completed assisted H2 screening artifact."""
    fields, rows = _read_csv(path)
    missing_fields = sorted(
        {"draft_method", "human_review_status", *REVIEW_FIELDS} - set(fields)
    )
    if missing_fields:
        raise ValueError(
            "Reviewed screening file lacks fields: "
            + ", ".join(missing_fields)
        )
    errors = [
        error
        for row in rows
        for error in _screening_row_errors(row)
    ]
    reviewer_ids = {
        str(row.get("human_reviewer_id") or "").strip()
        for row in rows
        if str(row.get("human_reviewer_id") or "").strip()
    }
    if len(reviewer_ids) != 1:
        errors.append("artifact must contain exactly one human reviewer ID")
    _raise_errors("H2 screening artifact", errors)
    return {
        "rows": len(rows),
        "reviewed": len(rows),
        "errors": 0,
    }


def _print_screening(
    row: Mapping[str, str],
    position: int,
    total: int,
    width: int,
) -> None:
    print("\n" + "=" * min(width, 100))
    print(
        f"[{position}/{total}] {screening_category(row)}  "
        f"{row.get('record_key', '')}"
    )
    print(f"DOI={row.get('doi', '')}")
    print("\nTITLE")
    print(textwrap.fill(str(row.get("title") or ""), width=width))
    print("\nABSTRACT")
    print(
        textwrap.fill(
            str(row.get("abstract") or "") or "[No abstract]",
            width=width,
        )
    )
    print("\nAI")
    print(
        f"{row.get('ai_language_judgment', '')} / "
        f"{row.get('ai_decision', '')} / "
        f"{row.get('ai_exclusion_reason', '')}"
    )
    print(textwrap.fill(str(row.get("ai_notes") or ""), width=width))
    print("\nH1")
    print(
        f"{row.get('h1_language_judgment', '')} / "
        f"{row.get('h1_decision', '')} / "
        f"{row.get('h1_exclusion_reason', '')}"
    )
    print(textwrap.fill(str(row.get("h1_notes") or ""), width=width))
    print("\nDRAFT")
    print(
        f"{row.get('language_judgment', '')} / "
        f"{row.get('decision', '')} / "
        f"{row.get('exclusion_reason', '')}"
    )


def _adopt_screening_source(
    row: Dict[str, str],
    source: str,
) -> None:
    if source == "draft":
        return
    prefix = f"{source}_"
    language = str(
        row.get(prefix + "language_judgment") or ""
    ).strip().casefold()
    decision = str(
        row.get(prefix + "decision") or ""
    ).strip().casefold()
    reason = str(row.get(prefix + "exclusion_reason") or "").strip()
    evidence = str(row.get(prefix + "evidence_span") or "").strip()
    if decision == "uncertain":
        raise ValueError(
            f"{source.upper()} is uncertain; make an explicit H2 decision"
        )
    row["language_judgment"] = language
    row["decision"] = decision
    row["exclusion_reason"] = reason
    row["evidence_span"] = evidence
    row["language_evidence"] = (
        evidence
        if evidence in _source_text(row)
        else str(row.get("title") or "").strip()
    )


def review_screening(
    input_path: Path,
    output_path: Path,
    reviewer_id: str,
    width: int = 96,
) -> Dict[str, int]:
    """Review H2 screening rows in risk order with per-row checkpoints."""
    if input_path.resolve() == output_path.resolve():
        raise ValueError("H2 reviewed output must not overwrite the draft")
    fields, rows = _load_target(input_path, output_path)
    while True:
        pending = [
            index
            for index, row in enumerate(rows)
            if str(
                row.get("human_review_status") or ""
            ).strip().casefold()
            != "reviewed"
        ]
        if not pending:
            break
        pending.sort(
            key=lambda index: (
                CATEGORY_ORDER[screening_category(rows[index])],
                str(rows[index].get("record_key") or ""),
            )
        )
        index = pending[0]
        row = rows[index]
        _print_screening(row, len(rows) - len(pending) + 1, len(rows), width)
        print(
            "\nAction: h=adopt H1, a=adopt AI, d=adopt draft, "
            "i=include, n=non-English, 1-6=exclude reason, "
            "s=skip, q=quit"
        )
        choice = input("> ").strip().casefold()
        if choice == "q":
            break
        if choice == "s":
            rows.append(rows.pop(index))
            _atomic_write_csv(output_path, fields, rows)
            continue
        try:
            if choice in {"h", "a", "d"}:
                source = {"h": "h1", "a": "ai", "d": "draft"}[choice]
                _adopt_screening_source(row, source)
                action = f"adopt_{source}"
            elif choice == "i":
                row["language_judgment"] = "en"
                row["decision"] = "include"
                row["exclusion_reason"] = ""
                action = "explicit_include"
            elif choice == "n":
                row["language_judgment"] = "non_en"
                row["decision"] = "exclude"
                row["exclusion_reason"] = "E_LANGUAGE_NON_ENGLISH"
                action = "explicit_non_english_exclusion"
            elif choice in EXCLUSION_CHOICES:
                row["language_judgment"] = "en"
                row["decision"] = "exclude"
                row["exclusion_reason"] = EXCLUSION_CHOICES[choice]
                action = "explicit_exclusion"
            else:
                print("Unknown action.")
                continue
            if choice not in {"h", "a", "d"}:
                row["language_evidence"] = _exact_span(
                    "Exact language evidence",
                    row,
                    str(row.get("language_evidence") or ""),
                )
                row["evidence_span"] = _exact_span(
                    "Exact decision evidence",
                    row,
                    str(row.get("evidence_span") or ""),
                )
            note = input("Optional H2 note: ").strip()
            _mark_reviewed(row, reviewer_id, action, note)
            errors = _screening_row_errors(row)
            if errors:
                row["human_review_status"] = "unreviewed"
                raise ValueError("\n".join(errors))
        except ValueError as error:
            print(str(error))
            continue
        _atomic_write_csv(output_path, fields, rows)
    _atomic_write_csv(output_path, fields, rows)
    return status_screening(output_path)


def _term_row_errors(row: Mapping[str, str]) -> List[str]:
    key = str(row.get("term_id") or "").strip() or "<missing-term>"
    errors: List[str] = []
    if str(row.get("coder_role") or "").strip().upper() != "H2":
        errors.append(f"{key}: coder_role must be H2")
    if (
        str(row.get("human_review_status") or "").strip().casefold()
        != "reviewed"
    ):
        errors.append(f"{key}: row is not human-reviewed")
    for field in (
        "human_reviewer_id",
        "human_reviewed_at",
        "human_review_action",
    ):
        if not str(row.get(field) or "").strip():
            errors.append(f"{key}: missing {field}")
    decision = str(row.get("decision") or "").strip().casefold()
    if decision not in DECISIONS:
        errors.append(f"{key}: decision must be include or exclude")
    if not str(row.get("reason") or "").strip():
        errors.append(f"{key}: reason is required")
    verbatim = str(row.get("verbatim_term") or "").strip()
    evidence = str(row.get("source_evidence_span") or "")
    if not verbatim or verbatim not in evidence:
        errors.append(f"{key}: verbatim term is absent from evidence span")
    if decision == "include":
        for field in (
            "canonical_term",
            "term_family_label",
            "search_domain_label",
            "search_domain_definition",
            "query_family_label",
        ):
            if not str(row.get(field) or "").strip():
                errors.append(f"{key}: included term lacks {field}")
        relation = str(row.get("term_relation") or "").strip().casefold()
        if relation not in TERM_RELATIONS:
            errors.append(f"{key}: invalid term_relation")
    return errors


def validate_terms(path: Path) -> Dict[str, int]:
    """Validate a completed assisted H2 term-adjudication artifact."""
    fields, rows = _read_csv(path)
    missing_fields = sorted(
        {"draft_method", "human_review_status", *REVIEW_FIELDS} - set(fields)
    )
    if missing_fields:
        raise ValueError(
            "Reviewed term file lacks fields: " + ", ".join(missing_fields)
        )
    errors = [
        error for row in rows for error in _term_row_errors(row)
    ]
    reviewer_ids = {
        str(row.get("human_reviewer_id") or "").strip()
        for row in rows
        if str(row.get("human_reviewer_id") or "").strip()
    }
    if len(reviewer_ids) != 1:
        errors.append("artifact must contain exactly one human reviewer ID")
    _raise_errors("H2 term artifact", errors)
    return {"rows": len(rows), "reviewed": len(rows), "errors": 0}


def _crossref_row_errors(row: Mapping[str, str]) -> List[str]:
    key = str(row.get("record_key") or "").strip() or "<missing-key>"
    errors: List[str] = []
    if str(row.get("reviewer_role") or "").strip().upper() != "H2":
        errors.append(f"{key}: reviewer_role must be H2")
    if (
        str(row.get("human_review_status") or "").strip().casefold()
        != "reviewed"
    ):
        errors.append(f"{key}: row is not human-reviewed")
    for field in (
        "human_reviewer_id",
        "human_reviewed_at",
        "human_review_action",
    ):
        if not str(row.get(field) or "").strip():
            errors.append(f"{key}: missing {field}")
    resolution = str(row.get("resolution") or "").strip().casefold()
    if resolution not in {
        "accept_openalex",
        "accept_crossref",
        "manual_bibliographic_resolution",
        "exclude_mapping_error",
    }:
        errors.append(f"{key}: invalid resolution")
    if not str(row.get("resolution_notes") or "").strip():
        errors.append(f"{key}: resolution_notes is required")
    return errors


def validate_crossref(path: Path) -> Dict[str, int]:
    """Validate a completed assisted H2 Crossref-resolution artifact."""
    fields, rows = _read_csv(path)
    missing_fields = sorted(
        {"draft_method", "human_review_status", *REVIEW_FIELDS} - set(fields)
    )
    if missing_fields:
        raise ValueError(
            "Reviewed Crossref file lacks fields: "
            + ", ".join(missing_fields)
        )
    errors = [
        error for row in rows for error in _crossref_row_errors(row)
    ]
    reviewer_ids = {
        str(row.get("human_reviewer_id") or "").strip()
        for row in rows
        if str(row.get("human_reviewer_id") or "").strip()
    }
    if len(reviewer_ids) != 1:
        errors.append("artifact must contain exactly one human reviewer ID")
    _raise_errors("H2 Crossref artifact", errors)
    return {"rows": len(rows), "reviewed": len(rows), "errors": 0}


def validate_reviewed(path: Path, kind: str) -> Dict[str, int]:
    if kind == "screening":
        return validate_screening(path)
    if kind == "terms":
        return validate_terms(path)
    return validate_crossref(path)


def attestation_template(
    artifact_path: Path,
    output_path: Path,
    kind: str,
) -> Dict[str, str]:
    """Create a pending human attestation bound to the reviewed CSV hash."""
    validate_reviewed(artifact_path, kind)
    _, rows = _read_csv(artifact_path)
    reviewer_id = str(rows[0]["human_reviewer_id"]).strip()
    digest = _sha256(artifact_path)
    payload = {
        "attestation_id": f"H2_{kind.upper()}_{digest[:20].upper()}",
        "artifact_path": str(artifact_path.resolve()),
        "artifact_sha256": digest,
        "reviewer_role": "H2",
        "reviewer_id": reviewer_id,
        "provenance_type": "human_reviewed_automated_draft",
        "attestation_statement": (
            "I personally reviewed every row in this automated draft and "
            "adopt the recorded H2 decisions as my human review."
        ),
        "attested_at": _now(),
        "status": "pending",
        "review_kind": kind,
        "instructions": (
            "The named human reviewer must inspect this exact-hash record, "
            "then change status to accepted. Registration rejects pending "
            "attestations and any later change to the reviewed CSV."
        ),
    }
    _atomic_write_json(output_path, payload)
    return {
        "output": str(output_path.resolve()),
        "artifact_sha256": digest,
        "status": "pending",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Local H2 review helper for assisted screening, term, and "
            "Crossref worksheets. It checkpoints every human decision."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    status = subparsers.add_parser("status-screening")
    status.add_argument("--input", required=True)

    review = subparsers.add_parser("review-screening")
    review.add_argument("--input", required=True)
    review.add_argument("--output", required=True)
    review.add_argument("--reviewer-id", required=True)
    review.add_argument("--width", type=int, default=96)

    validate = subparsers.add_parser("validate-reviewed")
    validate.add_argument(
        "--kind",
        choices=("screening", "terms", "crossref"),
        required=True,
    )
    validate.add_argument("--input", required=True)

    attestation = subparsers.add_parser("attestation-template")
    attestation.add_argument(
        "--kind",
        choices=("screening", "terms", "crossref"),
        required=True,
    )
    attestation.add_argument("--input", required=True)
    attestation.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    input_path = Path(args.input).resolve()
    if args.command == "status-screening":
        result = status_screening(input_path)
    elif args.command == "review-screening":
        result = review_screening(
            input_path,
            Path(args.output).resolve(),
            str(args.reviewer_id).strip(),
            width=max(int(args.width), 60),
        )
    elif args.command == "validate-reviewed":
        result = validate_reviewed(input_path, args.kind)
    else:
        result = attestation_template(
            input_path,
            Path(args.output).resolve(),
            args.kind,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import os
import tempfile
import textwrap
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


DECISIONS = {"include", "exclude", "uncertain"}
LANGUAGE_JUDGMENTS = {"en", "non_en", "uncertain"}
EXCLUSION_CHOICES = {
    "1": "E_NOT_PAPER_LEVEL_INNOVATION_OR_POTENTIAL_IMPACT",
    "2": "E_FUTURE_OUTCOME_ONLY",
    "3": "E_NOT_ARTICLE_LEVEL",
    "4": "E_NOT_INDICATOR_PREDICTOR_VALIDATION",
    "5": "E_DUPLICATE",
    "6": "E_INSUFFICIENT_METADATA",
}
EXCLUSION_REASONS = {
    "E_LANGUAGE_NON_ENGLISH",
    *EXCLUSION_CHOICES.values(),
}
REQUIRED_FIELDS = {
    "record_key",
    "doi",
    "title",
    "abstract",
    "reviewer_role",
    "language_judgment",
    "language_evidence",
    "decision",
    "exclusion_reason",
    "evidence_span",
    "notes",
}
TERM_RELATION_CHOICES = {
    "1": "canonical",
    "2": "synonym",
    "3": "abbreviation",
    "4": "full_form",
    "5": "historical_name",
    "6": "morphological_variant",
    "7": "parameter_variant",
}
TERM_RELATIONS = set(TERM_RELATION_CHOICES.values())
TERM_REQUIRED_FIELDS = {
    "term_id",
    "verbatim_term",
    "source_type",
    "coder_role",
    "canonical_term",
    "term_family_label",
    "term_relation",
    "search_domain_label",
    "search_domain_definition",
    "query_family_label",
    "cross_domain",
    "decision",
    "reason",
    "source_id",
    "source_location",
    "source_evidence_span",
    "proposed_role",
}


def _read_rows(path: Path) -> tuple[List[str], List[Dict[str, str]]]:
    """Read one blind H1 worksheet and reject leaked comparison columns."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    missing = sorted(REQUIRED_FIELDS - set(fields))
    if missing:
        raise ValueError(
            "Screening worksheet lacks fields: " + ", ".join(missing)
        )
    leaked = [
        field
        for field in fields
        if field.casefold().startswith(("ai_", "h2_"))
    ]
    if leaked:
        raise ValueError(
            "Blind H1 review refuses comparison columns: "
            + ", ".join(leaked)
        )
    invalid_roles = sorted(
        {
            str(row.get("reviewer_role") or "").strip().upper()
            for row in rows
            if str(row.get("reviewer_role") or "").strip().upper() != "H1"
        }
    )
    if invalid_roles:
        raise ValueError(
            "This blind reviewer accepts only H1 rows; found: "
            + ", ".join(invalid_roles)
        )
    return fields, rows


def _read_term_rows(
    path: Path,
) -> tuple[List[str], List[Dict[str, str]]]:
    """Read a blind H1 term worksheet and reject comparison columns."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    missing = sorted(TERM_REQUIRED_FIELDS - set(fields))
    if missing:
        raise ValueError(
            "Term worksheet lacks fields: " + ", ".join(missing)
        )
    leaked = [
        field
        for field in fields
        if field.casefold().startswith(("ai_", "h2_"))
    ]
    if leaked:
        raise ValueError(
            "Blind H1 term coding refuses comparison columns: "
            + ", ".join(leaked)
        )
    invalid_roles = sorted(
        {
            str(row.get("coder_role") or "").strip().upper()
            for row in rows
            if str(row.get("coder_role") or "").strip().upper() != "H1"
        }
    )
    if invalid_roles:
        raise ValueError(
            "This blind term coder accepts only H1 rows; found: "
            + ", ".join(invalid_roles)
        )
    return fields, rows


def _atomic_write(
    path: Path,
    fields: Sequence[str],
    rows: Iterable[Mapping[str, str]],
) -> None:
    """Atomically checkpoint the current worksheet."""
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


def _source_text(row: Mapping[str, str]) -> str:
    return "\n".join(
        (
            str(row.get("title") or ""),
            str(row.get("abstract") or ""),
        )
    )


def _default_evidence(row: Mapping[str, str]) -> str:
    title = str(row.get("title") or "").strip()
    if title:
        return title
    return str(row.get("abstract") or "").strip()[:240]


def validate_row(row: Mapping[str, str]) -> List[str]:
    """Return all completeness/provenance errors for one H1 decision."""
    errors: List[str] = []
    key = str(row.get("record_key") or "").strip() or "<missing-key>"
    decision = str(row.get("decision") or "").strip().casefold()
    language = str(
        row.get("language_judgment") or ""
    ).strip().casefold()
    reason = str(row.get("exclusion_reason") or "").strip()
    language_evidence = str(row.get("language_evidence") or "").strip()
    evidence = str(row.get("evidence_span") or "").strip()
    source = _source_text(row)
    if decision not in DECISIONS:
        errors.append(f"{key}: invalid or blank decision")
    if language not in LANGUAGE_JUDGMENTS:
        errors.append(f"{key}: invalid or blank language_judgment")
    if not language_evidence:
        errors.append(f"{key}: language_evidence is required")
    elif language_evidence not in source:
        errors.append(f"{key}: language_evidence is not an exact source span")
    if not evidence:
        errors.append(f"{key}: evidence_span is required")
    elif evidence not in source:
        errors.append(f"{key}: evidence_span is not an exact source span")
    if decision == "exclude":
        if reason not in EXCLUSION_REASONS:
            errors.append(f"{key}: invalid or blank exclusion_reason")
    elif reason:
        errors.append(f"{key}: non-exclusion row has exclusion_reason")
    if language == "non_en" and (
        decision != "exclude"
        or reason != "E_LANGUAGE_NON_ENGLISH"
    ):
        errors.append(
            f"{key}: non_en must use exclude/E_LANGUAGE_NON_ENGLISH"
        )
    if reason == "E_LANGUAGE_NON_ENGLISH" and language != "non_en":
        errors.append(f"{key}: language exclusion must use non_en")
    return errors


def validate_worksheet(path: Path) -> Dict[str, int]:
    """Validate every completed row without importing or reading AI output."""
    _, rows = _read_rows(path)
    errors: List[str] = []
    completed = 0
    for row in rows:
        if not str(row.get("decision") or "").strip():
            errors.append(
                f"{row.get('record_key', '<missing-key>')}: blank decision"
            )
            continue
        completed += 1
        errors.extend(validate_row(row))
    if errors:
        sample = "\n".join(errors[:50])
        remainder = max(len(errors) - 50, 0)
        suffix = f"\n... plus {remainder} more" if remainder else ""
        raise ValueError(
            f"Worksheet has {len(errors)} validation errors:\n"
            f"{sample}{suffix}"
        )
    return {"rows": len(rows), "completed": completed, "errors": 0}


def validate_term_row(row: Mapping[str, str]) -> List[str]:
    """Return completeness errors for one independent H1 term code."""
    errors: List[str] = []
    key = str(row.get("term_id") or "").strip() or "<missing-term-id>"
    decision = str(row.get("decision") or "").strip().casefold()
    reason = str(row.get("reason") or "").strip()
    cross_domain = str(row.get("cross_domain") or "").strip().casefold()
    if decision not in {"include", "exclude"}:
        errors.append(f"{key}: invalid or blank decision")
    if not reason:
        errors.append(f"{key}: reason is required")
    if cross_domain not in {"true", "false"}:
        errors.append(f"{key}: cross_domain must be true or false")
    evidence = str(row.get("source_evidence_span") or "")
    verbatim = str(row.get("verbatim_term") or "").strip()
    if not verbatim or verbatim not in evidence:
        errors.append(
            f"{key}: verbatim_term is not in source_evidence_span"
        )
    if decision == "include":
        required = (
            "canonical_term",
            "term_family_label",
            "search_domain_label",
            "search_domain_definition",
            "query_family_label",
        )
        for field in required:
            if not str(row.get(field) or "").strip():
                errors.append(f"{key}: included term requires {field}")
        relation = str(row.get("term_relation") or "").strip().casefold()
        if relation not in TERM_RELATIONS:
            errors.append(f"{key}: invalid term_relation")
        domains = [
            value.strip()
            for value in str(
                row.get("search_domain_label") or ""
            ).replace(";", "|").split("|")
            if value.strip()
        ]
        if cross_domain == "true" and len(set(domains)) < 2:
            errors.append(
                f"{key}: cross-domain term needs at least two domains"
            )
        if cross_domain == "false" and len(set(domains)) > 1:
            errors.append(
                f"{key}: multiple domains require cross_domain=true"
            )
    return errors


def validate_term_worksheet(path: Path) -> Dict[str, int]:
    """Validate a completed blind term worksheet."""
    _, rows = _read_term_rows(path)
    errors: List[str] = []
    completed = 0
    for row in rows:
        if not str(row.get("decision") or "").strip():
            errors.append(
                f"{row.get('term_id', '<missing-term-id>')}: blank decision"
            )
            continue
        completed += 1
        errors.extend(validate_term_row(row))
    if errors:
        sample = "\n".join(errors[:50])
        remainder = max(len(errors) - 50, 0)
        suffix = f"\n... plus {remainder} more" if remainder else ""
        raise ValueError(
            f"Term worksheet has {len(errors)} validation errors:\n"
            f"{sample}{suffix}"
        )
    return {"rows": len(rows), "completed": completed, "errors": 0}


def worksheet_status(path: Path) -> Dict[str, int]:
    """Count review progress without validating incomplete rows."""
    _, rows = _read_rows(path)
    decisions = Counter(
        str(row.get("decision") or "").strip().casefold() or "blank"
        for row in rows
    )
    return {
        "rows": len(rows),
        "blank": decisions["blank"],
        "include": decisions["include"],
        "exclude": decisions["exclude"],
        "uncertain": decisions["uncertain"],
    }


def term_worksheet_status(path: Path) -> Dict[str, int]:
    """Count term-coding progress without opening any AI output."""
    _, rows = _read_term_rows(path)
    decisions = Counter(
        str(row.get("decision") or "").strip().casefold() or "blank"
        for row in rows
    )
    return {
        "rows": len(rows),
        "blank": decisions["blank"],
        "include": decisions["include"],
        "exclude": decisions["exclude"],
    }


def _print_record(
    row: Mapping[str, str],
    index: int,
    total: int,
    width: int,
) -> None:
    print("\n" + "=" * min(width, 100))
    print(
        f"[{index + 1}/{total}] {row.get('record_key', '')}  "
        f"DOI={row.get('doi', '')}"
    )
    print(
        f"year={row.get('publication_year', '')}  "
        f"type={row.get('work_type', '')}  "
        f"OpenAlex-language={row.get('openalex_language', '')}"
    )
    print("\nTITLE")
    print(textwrap.fill(str(row.get("title") or ""), width=width))
    print("\nABSTRACT")
    abstract = str(row.get("abstract") or "") or "[No abstract]"
    print(textwrap.fill(abstract, width=width))
    existing = str(row.get("decision") or "").strip()
    if existing:
        print(
            "\nCURRENT: "
            f"{existing} / {row.get('exclusion_reason', '')}"
        )


def _prompt_exact_span(
    label: str,
    row: Mapping[str, str],
) -> str:
    default = _default_evidence(row)
    source = _source_text(row)
    while True:
        value = input(f"{label} [Enter=title]: ").strip() or default
        if value and value in source:
            return value
        print("The value must be copied exactly from title or abstract.")


def _apply_choice(row: Dict[str, str], choice: str) -> bool:
    """Prompt for one decision; return False for a navigation command."""
    if choice == "i":
        language = "en"
        decision = "include"
        reason = ""
    elif choice == "u":
        language_choice = input(
            "Language [e=English, n=non-English, u=uncertain]: "
        ).strip().casefold()
        language = {
            "e": "en",
            "n": "non_en",
            "u": "uncertain",
        }.get(language_choice, "uncertain")
        decision = "uncertain"
        reason = ""
        if language == "non_en":
            decision = "exclude"
            reason = "E_LANGUAGE_NON_ENGLISH"
    elif choice == "n":
        language = "non_en"
        decision = "exclude"
        reason = "E_LANGUAGE_NON_ENGLISH"
    elif choice in EXCLUSION_CHOICES:
        language = "en"
        decision = "exclude"
        reason = EXCLUSION_CHOICES[choice]
    else:
        return False
    row["language_judgment"] = language
    row["language_evidence"] = _prompt_exact_span(
        "Exact language evidence",
        row,
    )
    row["decision"] = decision
    row["exclusion_reason"] = reason
    row["evidence_span"] = _prompt_exact_span(
        "Exact decision evidence",
        row,
    )
    row["notes"] = input("Optional notes: ").strip()
    errors = validate_row(row)
    if errors:
        raise ValueError("\n".join(errors))
    return True


def review_screening(
    input_path: Path,
    output_path: Path | None = None,
    width: int = 96,
) -> Dict[str, int]:
    """Interactively review an H1-blind screening worksheet."""
    target = output_path or input_path
    load_path = target if target.exists() else input_path
    fields, rows = _read_rows(load_path)
    index = next(
        (
            position
            for position, row in enumerate(rows)
            if not str(row.get("decision") or "").strip()
        ),
        len(rows),
    )
    while index < len(rows):
        row = rows[index]
        _print_record(row, index, len(rows), width)
        print(
            "\nDecision: i=include, u=uncertain, n=non-English, "
            "1=not target, 2=future-only, 3=not article-level, "
            "4=not indicator/predictor/validation, 5=duplicate, "
            "6=insufficient metadata, s=skip, b=back, q=quit"
        )
        choice = input("> ").strip().casefold()
        if choice == "q":
            break
        if choice == "s":
            index += 1
            continue
        if choice == "b":
            index = max(index - 1, 0)
            continue
        try:
            if not _apply_choice(row, choice):
                print("Unknown choice.")
                continue
        except ValueError as error:
            print(str(error))
            continue
        _atomic_write(target, fields, rows)
        index += 1
    _atomic_write(target, fields, rows)
    return worksheet_status(target)


def _prompt_value(label: str, default: str = "") -> str:
    prompt = label
    if default:
        prompt += f" [Enter={default}]"
    prompt += ": "
    while True:
        value = input(prompt).strip() or default
        if value:
            return value
        print(f"{label} is required.")


def _print_term(
    row: Mapping[str, str],
    index: int,
    total: int,
    width: int,
) -> None:
    print("\n" + "=" * min(width, 100))
    print(
        f"[{index + 1}/{total}] {row.get('term_id', '')}  "
        f"source={row.get('source_id', '')}"
    )
    print(
        f"type={row.get('source_type', '')}  "
        f"role={row.get('proposed_role', '')}  "
        f"location={row.get('source_location', '')}"
    )
    print("\nVERBATIM TERM")
    print(textwrap.fill(str(row.get("verbatim_term") or ""), width=width))
    print("\nSOURCE EVIDENCE")
    print(
        textwrap.fill(
            str(row.get("source_evidence_span") or ""),
            width=width,
        )
    )
    if str(row.get("decision") or "").strip():
        print(
            "\nCURRENT: "
            f"{row.get('decision', '')}; "
            f"domain={row.get('search_domain_label', '')}; "
            f"family={row.get('term_family_label', '')}"
        )


def _included_term_values(row: Mapping[str, str]) -> Dict[str, str]:
    verbatim = str(row.get("verbatim_term") or "").strip()
    canonical = _prompt_value(
        "Canonical English term",
        str(row.get("canonical_term") or "").strip() or verbatim,
    )
    family = _prompt_value(
        "Term-family label",
        str(row.get("term_family_label") or "").strip() or canonical,
    )
    print(
        "Relation: 1=canonical, 2=synonym, 3=abbreviation, "
        "4=full form, 5=historical name, 6=morphological variant, "
        "7=parameter variant"
    )
    relation_raw = _prompt_value(
        "Relation number or name",
        str(row.get("term_relation") or "").strip() or "1",
    ).casefold()
    domains = _prompt_value(
        "Search-domain label(s; use | for multiple)",
        str(row.get("search_domain_label") or "").strip(),
    )
    definition = _prompt_value(
        "Domain definition",
        str(row.get("search_domain_definition") or "").strip(),
    )
    query_family = _prompt_value(
        "Logical query-family label",
        str(row.get("query_family_label") or "").strip() or domains,
    )
    cross_default = str(
        row.get("cross_domain") or "false"
    ).strip().casefold()
    cross_raw = _prompt_value(
        "Cross-domain? [y/n]",
        "y" if cross_default == "true" else "n",
    ).casefold()
    return {
        "canonical_term": canonical,
        "term_family_label": family,
        "term_relation": TERM_RELATION_CHOICES.get(
            relation_raw,
            relation_raw,
        ),
        "search_domain_label": domains,
        "search_domain_definition": definition,
        "query_family_label": query_family,
        "cross_domain": (
            "true" if cross_raw in {"y", "yes", "true"} else "false"
        ),
        "decision": "include",
        "reason": _prompt_value(
            "Evidence-based coding reason",
            str(row.get("reason") or "").strip(),
        ),
    }


def _apply_term_choice(row: Dict[str, str], choice: str) -> bool:
    if choice == "e":
        row.update(
            {
                "canonical_term": "",
                "term_family_label": "",
                "term_relation": "",
                "search_domain_label": "",
                "search_domain_definition": "",
                "query_family_label": "",
                "cross_domain": "false",
                "decision": "exclude",
                "reason": _prompt_value("Exclusion reason"),
            }
        )
    elif choice == "i":
        row.update(_included_term_values(row))
    else:
        return False
    errors = validate_term_row(row)
    if errors:
        raise ValueError("\n".join(errors))
    return True


def review_terms(
    input_path: Path,
    output_path: Path | None = None,
    width: int = 96,
) -> Dict[str, int]:
    """Interactively complete an H1-blind term-coding worksheet."""
    target = output_path or input_path
    load_path = target if target.exists() else input_path
    fields, rows = _read_term_rows(load_path)
    index = next(
        (
            position
            for position, row in enumerate(rows)
            if not str(row.get("decision") or "").strip()
        ),
        len(rows),
    )
    while index < len(rows):
        row = rows[index]
        _print_term(row, index, len(rows), width)
        print(
            "\nDecision: i=include/code, e=exclude, "
            "s=skip, b=back, q=quit"
        )
        choice = input("> ").strip().casefold()
        if choice == "q":
            break
        if choice == "s":
            index += 1
            continue
        if choice == "b":
            index = max(index - 1, 0)
            continue
        try:
            if not _apply_term_choice(row, choice):
                print("Unknown choice.")
                continue
        except ValueError as error:
            print(str(error))
            continue
        _atomic_write(target, fields, rows)
        index += 1
    _atomic_write(target, fields, rows)
    return term_worksheet_status(target)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Blind, local H1 screening and term-coding helper. It never "
            "opens AI outputs or the project database."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    review = subparsers.add_parser("screen")
    review.add_argument("--input", required=True)
    review.add_argument("--output")
    review.add_argument("--width", type=int, default=96)
    status = subparsers.add_parser("status")
    status.add_argument("--input", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--input", required=True)
    term_review = subparsers.add_parser("code-terms")
    term_review.add_argument("--input", required=True)
    term_review.add_argument("--output")
    term_review.add_argument("--width", type=int, default=96)
    term_status = subparsers.add_parser("term-status")
    term_status.add_argument("--input", required=True)
    term_validate = subparsers.add_parser("validate-terms")
    term_validate.add_argument("--input", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    input_path = Path(args.input).resolve()
    if args.command == "screen":
        output_path = Path(args.output).resolve() if args.output else None
        result = review_screening(
            input_path,
            output_path=output_path,
            width=max(int(args.width), 60),
        )
    elif args.command == "code-terms":
        output_path = Path(args.output).resolve() if args.output else None
        result = review_terms(
            input_path,
            output_path=output_path,
            width=max(int(args.width), 60),
        )
    elif args.command == "status":
        result = worksheet_status(input_path)
    elif args.command == "validate":
        result = validate_worksheet(input_path)
    elif args.command == "term-status":
        result = term_worksheet_status(input_path)
    else:
        result = validate_term_worksheet(input_path)
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

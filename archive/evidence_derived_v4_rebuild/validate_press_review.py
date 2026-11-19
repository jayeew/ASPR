from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from common import (
    iter_csv,
    normalize_term,
    parse_bool,
    read_json,
    sha256_file,
)


MUTABLE_FIELDS = {
    "concepts_complete",
    "boolean_logic_valid",
    "spelling_valid",
    "phrases_valid",
    "limits_justified",
    "covered_by_logical_query_id",
    "logical_coverage_verified",
    "result_set_coverage_verified",
    "independent_construct_role",
    "decision",
    "notes",
    "draft_method",
    "independent_ai_review_status",
    "independent_ai_reviewer_id",
    "independent_ai_reviewed_at",
    "independent_ai_review_action",
    "independent_ai_review_note",
    "independent_ai_run_id",
    "independent_ai_model",
    "independent_ai_prompt_sha256",
}
PROVENANCE_FIELDS = {
    "draft_method",
    "independent_ai_review_status",
    "independent_ai_reviewer_id",
    "independent_ai_reviewed_at",
    "independent_ai_review_action",
    "independent_ai_review_note",
    "independent_ai_run_id",
    "independent_ai_model",
    "independent_ai_prompt_sha256",
}
REQUIRED_MANIFEST_FIELDS = (
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
)
NONAUTHORIZING_SOURCE_TYPES = {
    "pilot_v2_indicator",
    "pilot_v2_literature",
    "development_seed_hint",
}


def _assert_hash(path: Path, expected: Any, label: str) -> None:
    if not path.is_file() or sha256_file(path) != str(expected):
        raise ValueError(f"{label} hash does not match: {path}")


def _manifest_counts(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    counts = manifest.get("counts")
    if isinstance(counts, dict):
        return counts
    parameters = manifest.get("parameters")
    if not isinstance(parameters, dict):
        return {}
    nested = parameters.get("counts")
    return nested if isinstance(nested, dict) else parameters


def _assert_manifest(
    manifest: Mapping[str, Any],
    row_count: int,
    prompt_sha: str,
) -> None:
    missing = [
        field
        for field in REQUIRED_MANIFEST_FIELDS
        if manifest.get(field) is None
        or (
            field != "parameters"
            and not str(manifest.get(field) or "").strip()
        )
    ]
    if missing:
        raise ValueError(
            "Manifest is not registration-compatible; missing: "
            + ", ".join(missing)
        )
    if not isinstance(manifest["parameters"], dict):
        raise ValueError("Manifest parameters must be a JSON object")
    if int(manifest["item_count"]) != row_count:
        raise ValueError("Manifest item_count differs from reviewed rows")
    if str(manifest["status"]).casefold() != "complete":
        raise ValueError("Manifest status must be complete")
    if str(manifest["reviewer_role"]).upper() != "H2":
        raise ValueError("Manifest reviewer_role must be H2")
    if str(manifest["prompt_sha256"]).casefold() != prompt_sha:
        raise ValueError("Manifest prompt hash differs from protocol")
    identity = (
        f"{manifest['model']} {manifest['model_digest']}".casefold()
    )
    if any(token in identity for token in ("qwen", "ollama")):
        raise ValueError("Qwen/Ollama review manifests are forbidden")


def _assert_expression(row: Mapping[str, str]) -> None:
    expression = str(row["logical_expression"])
    if expression.count("(") != expression.count(")"):
        raise ValueError(
            f"Unbalanced query expression: {row['logical_query_id']}"
        )
    if expression.count(" AND ") < 2:
        raise ValueError(
            f"Query lacks the three-block structure: "
            f"{row['logical_query_id']}"
        )
    blocks = (
        json.loads(row["domain_terms_json"]),
        json.loads(row["object_terms_json"]),
        json.loads(row["context_terms_json"]),
    )
    for terms in blocks:
        if not isinstance(terms, list) or not terms:
            raise ValueError(
                f"Query contains an empty term block: "
                f"{row['logical_query_id']}"
            )
        for term in terms:
            if normalize_term(str(term)) not in normalize_term(expression):
                raise ValueError(
                    f"Expression omits exported term {term}: "
                    f"{row['logical_query_id']}"
                )
    filters = json.loads(row["physical_filter_expressions_json"])
    if (
        not isinstance(filters, list)
        or not filters
        or any(
            "to_publication_date:" not in value or "type:" not in value
            for value in filters
        )
    ):
        raise ValueError(
            f"Missing frozen date/type filters: {row['logical_query_id']}"
        )
    evidence = json.loads(row["term_evidence_json"])
    if int(row["term_evidence_count"]) != len(evidence) or not evidence:
        raise ValueError(
            f"Term-evidence count mismatch: {row['logical_query_id']}"
        )
    required = {
        "term_id",
        "source_id",
        "source_type",
        "evidence_span",
        "canonical_term",
        "term_family_label",
    }
    if any(
        not required.issubset(item)
        or not all(str(item[field]).strip() for field in required)
        for item in evidence
    ):
        raise ValueError(
            f"Incomplete source evidence: {row['logical_query_id']}"
        )
    if any(
        item["source_type"] not in NONAUTHORIZING_SOURCE_TYPES
        and not str(item.get("source_record_key") or "").strip()
        for item in evidence
    ):
        raise ValueError(
            f"Authorizing evidence lacks a record key: "
            f"{row['logical_query_id']}"
        )
    if all(
        item["source_type"] in NONAUTHORIZING_SOURCE_TYPES
        for item in evidence
    ):
        raise ValueError(
            f"Pilot-only family lacks authorizing evidence: "
            f"{row['logical_query_id']}"
        )


def validate_press_review(
    input_path: Path,
    output_path: Path,
    protocol_path: Path,
    manifest_path: Path,
) -> Dict[str, Any]:
    """Validate one independent PRESS artifact before registration/import."""
    protocol = read_json(protocol_path)
    manifest = read_json(manifest_path)
    prompt_sha = sha256_file(protocol_path)
    authorized = protocol["authorized_input"]
    _assert_hash(input_path, authorized["sha256"], "PRESS input")
    _assert_hash(input_path, manifest.get("input_sha256"), "Manifest input")
    _assert_hash(
        output_path,
        manifest.get("artifact_sha256"),
        "PRESS output",
    )
    input_rows = list(iter_csv(input_path))
    output_rows = list(iter_csv(output_path))
    if not input_rows or len(input_rows) != int(authorized["rows"]):
        raise ValueError("PRESS input row count differs from protocol")
    if len(output_rows) != len(input_rows):
        raise ValueError("PRESS output row count differs from input")
    input_ids = [row["logical_query_id"] for row in input_rows]
    output_ids = [row["logical_query_id"] for row in output_rows]
    if (
        "" in input_ids
        or len(set(input_ids)) != len(input_ids)
        or output_ids != input_ids
    ):
        raise ValueError("PRESS logical-query ID set or order changed")
    _assert_manifest(manifest, len(output_rows), prompt_sha)
    valid_ids = set(input_ids)
    counts = {"pass": 0, "revise": 0, "archive_redundant": 0}
    protected_fields = set(input_rows[0]) - MUTABLE_FIELDS
    for line_number, (source, reviewed) in enumerate(
        zip(input_rows, output_rows),
        start=2,
    ):
        for field in protected_fields:
            if str(source.get(field) or "") != str(
                reviewed.get(field) or ""
            ):
                raise ValueError(
                    f"Protected field {field} changed at line {line_number}"
                )
        _assert_expression(reviewed)
        if str(reviewed.get("reviewer_role") or "").upper() != "H2":
            raise ValueError(f"reviewer_role is not H2 at line {line_number}")
        if any(
            not str(reviewed.get(field) or "").strip()
            for field in PROVENANCE_FIELDS
        ):
            raise ValueError(
                f"Incomplete independent provenance at line {line_number}"
            )
        if (
            str(reviewed["independent_ai_prompt_sha256"]).casefold()
            != prompt_sha
        ):
            raise ValueError(f"Prompt hash mismatch at line {line_number}")
        decision = str(reviewed.get("decision") or "").casefold()
        if decision not in counts:
            raise ValueError(f"Invalid PRESS decision at line {line_number}")
        counts[decision] += 1
        checks = [
            parse_bool(reviewed.get(field), field)
            for field in (
                "concepts_complete",
                "boolean_logic_valid",
                "spelling_valid",
                "phrases_valid",
                "limits_justified",
            )
        ]
        covered_by = str(
            reviewed.get("covered_by_logical_query_id") or ""
        ).strip()
        logical_coverage = parse_bool(
            reviewed.get("logical_coverage_verified") or "false",
            "logical_coverage_verified",
        )
        result_coverage = parse_bool(
            reviewed.get("result_set_coverage_verified") or "false",
            "result_set_coverage_verified",
        )
        independent_role = parse_bool(
            reviewed.get("independent_construct_role"),
            "independent_construct_role",
        )
        notes = str(reviewed.get("notes") or "").strip()
        if decision == "pass":
            if not all(checks) or covered_by or not independent_role:
                raise ValueError(
                    f"Invalid PRESS pass at line {line_number}"
                )
        elif decision == "revise":
            if all(checks) or covered_by or not independent_role or not notes:
                raise ValueError(
                    f"Invalid PRESS revise at line {line_number}"
                )
        else:
            if (
                covered_by not in valid_ids
                or covered_by == reviewed["logical_query_id"]
                or not logical_coverage
                or result_coverage
                or independent_role
                or not notes
            ):
                raise ValueError(
                    f"Invalid redundancy archival at line {line_number}"
                )
        if not notes:
            raise ValueError(f"PRESS notes are blank at line {line_number}")
    reported = _manifest_counts(manifest)
    for key, value in counts.items():
        candidates: Sequence[str] = (key, f"{key}_count")
        for candidate in candidates:
            if candidate in reported:
                if int(reported[candidate]) != value:
                    raise ValueError(
                        f"Manifest {candidate} differs from observed count"
                    )
                break
    return {
        "status": "pass",
        "rows": len(output_rows),
        "decision_counts": counts,
        "input_sha256": sha256_file(input_path),
        "output_sha256": sha256_file(output_path),
        "manifest_sha256": sha256_file(manifest_path),
        "protocol_sha256": prompt_sha,
    }


def main() -> None:
    """Run the standalone PRESS artifact validator."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    result = validate_press_review(
        Path(args.input).resolve(),
        Path(args.output).resolve(),
        Path(args.protocol).resolve(),
        Path(args.manifest).resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

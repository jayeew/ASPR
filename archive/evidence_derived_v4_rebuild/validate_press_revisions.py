from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from common import iter_csv, normalize_term, read_json, sha256_file


MUTABLE_FIELDS = {
    "revision_decision",
    "revised_domain_terms_json",
    "revision_rationale",
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
STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "between",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}
FORBIDDEN_TOKENS = {"t0", "amp", "lt", "gt"}


def _inflection_variants(token: str) -> set[str]:
    """Return conservative English singular/plural variants for evidence matching."""
    variants = {token}
    if token.endswith("ies") and len(token) > 3:
        variants.add(f"{token[:-3]}y")
    elif token.endswith(("ses", "xes", "zes", "ches", "shes")):
        variants.add(token[:-2])
    elif token.endswith("s") and not token.endswith("ss") and len(token) > 2:
        variants.add(token[:-1])
    else:
        variants.add(f"{token}s")
    return variants


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


def _source_vocabulary(row: Mapping[str, str]) -> set[str]:
    evidence = json.loads(row["term_evidence_json"])
    text = " ".join(
        (
            row["search_domain_label"],
            row["search_domain_definition"],
            row["family_label"],
            row["press_notes"],
            " ".join(json.loads(row["old_domain_terms_json"])),
            *(
                " ".join(
                    (
                        str(item.get("evidence_span") or ""),
                        str(item.get("canonical_term") or ""),
                        str(item.get("term_family_label") or ""),
                        str(item.get("source_title") or ""),
                    )
                )
                for item in evidence
            ),
        )
    )
    vocabulary: set[str] = set()
    for token in normalize_term(text).split():
        vocabulary.update(_inflection_variants(token))
    return vocabulary


def validate_press_revisions(
    input_path: Path,
    output_path: Path,
    protocol_path: Path,
    manifest_path: Path,
) -> Dict[str, Any]:
    """Validate a PRESS query-surface correction artifact end to end."""
    protocol = read_json(protocol_path)
    manifest = read_json(manifest_path)
    prompt_sha = sha256_file(protocol_path)
    authorized = protocol["authorized_input"]
    _assert_hash(input_path, authorized["sha256"], "Revision input")
    _assert_hash(input_path, manifest.get("input_sha256"), "Manifest input")
    _assert_hash(
        output_path,
        manifest.get("artifact_sha256"),
        "Revision output",
    )
    input_rows = list(iter_csv(input_path))
    output_rows = list(iter_csv(output_path))
    if len(input_rows) != int(authorized["rows"]):
        raise ValueError("Revision input row count differs from protocol")
    if len(output_rows) != len(input_rows):
        raise ValueError("Revision output row count differs from input")
    input_ids = [row["logical_query_id"] for row in input_rows]
    output_ids = [row["logical_query_id"] for row in output_rows]
    if (
        "" in input_ids
        or len(set(input_ids)) != len(input_ids)
        or output_ids != input_ids
    ):
        raise ValueError("Revision logical-query ID set or order changed")
    _assert_manifest(manifest, len(output_rows), prompt_sha)
    counts = {"replace_terms": 0, "archive_unsupported": 0}
    protected = set(input_rows[0]) - MUTABLE_FIELDS
    changed_terms = 0
    for line_number, (source, reviewed) in enumerate(
        zip(input_rows, output_rows),
        start=2,
    ):
        for field in protected:
            if str(source.get(field) or "") != str(
                reviewed.get(field) or ""
            ):
                raise ValueError(
                    f"Protected field {field} changed at line {line_number}"
                )
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
        decision = str(
            reviewed.get("revision_decision") or ""
        ).casefold()
        if decision not in counts:
            raise ValueError(
                f"Invalid revision decision at line {line_number}"
            )
        counts[decision] += 1
        rationale = str(
            reviewed.get("revision_rationale") or ""
        ).strip()
        if not rationale:
            raise ValueError(f"Blank revision rationale at line {line_number}")
        old_terms = json.loads(source["old_domain_terms_json"])
        revised = json.loads(
            reviewed.get("revised_domain_terms_json") or "[]"
        )
        if not isinstance(revised, list):
            raise ValueError(f"Revised terms are not a list at line {line_number}")
        revised = [str(value).strip() for value in revised]
        if any(not value for value in revised):
            raise ValueError(f"Blank revised term at line {line_number}")
        normalized = [normalize_term(value) for value in revised]
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"Duplicate revised term at line {line_number}")
        if decision == "archive_unsupported":
            if revised:
                raise ValueError(
                    f"Archived revision retains terms at line {line_number}"
                )
            continue
        if not revised or normalized == [
            normalize_term(value) for value in old_terms
        ]:
            raise ValueError(
                f"Replacement does not change terms at line {line_number}"
            )
        vocabulary = _source_vocabulary(reviewed)
        old_normalized = {normalize_term(value) for value in old_terms}
        for term, normalized_term in zip(revised, normalized):
            tokens = set(normalized_term.split())
            if tokens & FORBIDDEN_TOKENS:
                raise ValueError(
                    f"Artifact token remains at line {line_number}: {term}"
                )
            if {"and", "or"} & {value.casefold() for value in term.split()}:
                # Natural-language "and/or" is allowed; symbolic Boolean
                # operators are rejected below by exact uppercase spelling.
                pass
            if " AND " in term or " OR " in term:
                raise ValueError(
                    f"Boolean operator embedded in term at line {line_number}"
                )
            if normalized_term in old_normalized:
                continue
            unsupported = {
                token
                for token in tokens
                if token not in STOPWORDS
                and token not in vocabulary
                and not token.replace(".", "").isdigit()
            }
            if unsupported:
                raise ValueError(
                    f"Added term lacks source/PRESS vocabulary at line "
                    f"{line_number}: {sorted(unsupported)}"
                )
            changed_terms += 1
    reported = _manifest_counts(manifest)
    for key, value in counts.items():
        for candidate in (key, f"{key}_count"):
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
        "added_or_repaired_term_rows": changed_terms,
        "input_sha256": sha256_file(input_path),
        "output_sha256": sha256_file(output_path),
        "manifest_sha256": sha256_file(manifest_path),
        "protocol_sha256": prompt_sha,
    }


def main() -> None:
    """Run the standalone PRESS-revision validator."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    result = validate_press_revisions(
        Path(args.input).resolve(),
        Path(args.output).resolve(),
        Path(args.protocol).resolve(),
        Path(args.manifest).resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

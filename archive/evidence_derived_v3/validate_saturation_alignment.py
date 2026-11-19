from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from common import (
    iter_csv,
    normalize_term,
    parse_bool,
    read_json,
    sha256_file,
)


PROVENANCE_FIELDS = {
    "reviewer_role",
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
INDICATOR_MUTABLE_FIELDS = {
    "h2_decision",
    "canonical_family_label",
    "adjudication_notes",
}
TERM_MUTABLE_FIELDS = {
    "canonical_term",
    "term_family_label",
    "term_relation",
    "search_domain_label",
    "search_domain_definition",
    "query_family_label",
    "cross_domain",
    "decision",
    "reason",
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
REQUIRED_REGISTRATION_MANIFEST_FIELDS = (
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


def _resolve(protocol_path: Path, value: Any) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = protocol_path.resolve().parent / path
    return path.resolve()


def _assert_hash(path: Path, expected: Any, label: str) -> None:
    if not path.is_file() or sha256_file(path) != str(expected):
        raise ValueError(f"{label} hash does not match: {path}")


def _assert_unique_order(
    input_rows: Sequence[Mapping[str, str]],
    output_rows: Sequence[Mapping[str, str]],
    key: str,
) -> None:
    input_ids = [str(row.get(key) or "") for row in input_rows]
    output_ids = [str(row.get(key) or "") for row in output_rows]
    if "" in input_ids or len(set(input_ids)) != len(input_ids):
        raise ValueError(f"Input requires unique nonblank {key}")
    if input_ids != output_ids:
        raise ValueError(f"Output {key} set or order differs from input")


def _assert_protected_fields(
    input_rows: Sequence[Mapping[str, str]],
    output_rows: Sequence[Mapping[str, str]],
    mutable_fields: set[str],
) -> None:
    protected = set(input_rows[0]) - mutable_fields
    for line_number, (source, reviewed) in enumerate(
        zip(input_rows, output_rows),
        start=2,
    ):
        for field in protected:
            if str(source.get(field) or "") != str(reviewed.get(field) or ""):
                raise ValueError(
                    f"Protected field {field} changed at line {line_number}"
                )


def _assert_provenance(
    rows: Iterable[Mapping[str, str]],
    prompt_sha: str,
) -> None:
    for line_number, row in enumerate(rows, start=2):
        if str(row.get("reviewer_role") or "").strip().upper() != "H2":
            raise ValueError(f"reviewer_role is not H2 at line {line_number}")
        if (
            str(row.get("independent_ai_prompt_sha256") or "")
            .strip()
            .casefold()
            != prompt_sha
        ):
            raise ValueError(
                f"Prompt hash differs from protocol at line {line_number}"
            )


def _assert_registration_compatible_manifest(
    manifest: Mapping[str, Any],
    row_count: int,
) -> None:
    """Reject review manifests that cannot be registered downstream."""
    missing = [
        field
        for field in REQUIRED_REGISTRATION_MANIFEST_FIELDS
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
        raise ValueError(
            "Manifest item_count differs from the reviewed row count"
        )
    if str(manifest["status"]).strip().casefold() != "complete":
        raise ValueError("Manifest status must be complete")
    if str(manifest["reviewer_role"]).strip().upper() != "H2":
        raise ValueError("Manifest reviewer_role must be H2")
    model_identity = (
        f"{manifest['model']} {manifest['model_digest']}".casefold()
    )
    if any(token in model_identity for token in ("qwen", "ollama")):
        raise ValueError("Qwen/Ollama review manifests are forbidden")


def _manifest_counts(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    value = manifest.get("counts")
    if isinstance(value, dict):
        return value
    parameters = manifest.get("parameters")
    if not isinstance(parameters, dict):
        return {}
    nested = parameters.get("counts")
    return nested if isinstance(nested, dict) else parameters


def _assert_reported_count(
    counts: Mapping[str, Any],
    key: str,
    observed: int,
    aliases: Sequence[str] = (),
) -> None:
    for candidate in (key, *aliases):
        if candidate not in counts:
            continue
        if int(counts[candidate]) != observed:
            raise ValueError(
                f"Manifest {candidate}={counts[candidate]} differs from "
                f"observed={observed}"
            )
        return


def validate_alignment(
    kind: str,
    input_path: Path,
    output_path: Path,
    protocol_path: Path,
    manifest_path: Path,
) -> Dict[str, Any]:
    """Validate one H2 saturation-alignment artifact end to end."""
    normalized_kind = kind.strip().casefold()
    if normalized_kind not in {"indicator", "term"}:
        raise ValueError("kind must be indicator or term")
    protocol = read_json(protocol_path)
    manifest = read_json(manifest_path)
    prompt_sha = sha256_file(protocol_path)
    if str(manifest.get("prompt_sha256") or "").casefold() != prompt_sha:
        raise ValueError("Manifest prompt hash differs from protocol")
    _assert_hash(
        output_path,
        manifest.get("artifact_sha256"),
        "Output artifact",
    )
    _assert_hash(input_path, manifest.get("input_sha256"), "Input artifact")
    reference = protocol["prior_round_reference"]
    reference_path = _resolve(protocol_path, reference["path"])
    reference_manifest = _resolve(
        protocol_path,
        reference["export_manifest_path"],
    )
    _assert_hash(reference_path, reference["sha256"], "Codebook reference")
    _assert_hash(
        reference_manifest,
        reference["export_manifest_sha256"],
        "Codebook export manifest",
    )
    input_rows = list(iter_csv(input_path))
    output_rows = list(iter_csv(output_path))
    if not input_rows or not output_rows:
        raise ValueError("Alignment input and output must both contain rows")
    _assert_registration_compatible_manifest(manifest, len(output_rows))
    key = "candidate_id" if normalized_kind == "indicator" else "term_id"
    _assert_unique_order(input_rows, output_rows, key)
    mutable_fields = (
        INDICATOR_MUTABLE_FIELDS
        if normalized_kind == "indicator"
        else TERM_MUTABLE_FIELDS
    )
    _assert_protected_fields(input_rows, output_rows, mutable_fields)
    _assert_provenance(output_rows, prompt_sha)
    prior_rows = list(iter_csv(reference_path))
    included = []
    excluded = []
    mapped_rows = []
    new_rows = []
    if normalized_kind == "indicator":
        prior_roles: Dict[str, set[str]] = {}
        for row in prior_rows:
            label = normalize_term(row["canonical_family_label"])
            prior_roles.setdefault(label, set()).add(row["proposed_role"])
        for row in output_rows:
            decision = str(row.get("h2_decision") or "").casefold()
            family = str(row.get("canonical_family_label") or "").strip()
            notes = str(row.get("adjudication_notes") or "").strip()
            if decision == "include":
                if not family or not notes:
                    raise ValueError(f"Included indicator lacks fields: {row[key]}")
                included.append(row)
                family_key = normalize_term(family)
                if family_key in prior_roles:
                    if row["proposed_role"] not in prior_roles[family_key]:
                        raise ValueError(
                            "Mapped indicator lacks a same-role prior "
                            f"exemplar: {row[key]}"
                        )
                    mapped_rows.append(row)
                else:
                    new_rows.append(row)
            elif decision == "exclude":
                if family or not notes:
                    raise ValueError(f"Excluded indicator has invalid fields: {row[key]}")
                excluded.append(row)
            else:
                raise ValueError(f"Invalid indicator decision: {row[key]}")
        family_field = "canonical_family_label"
    else:
        prior_assignments: Dict[tuple[str, str, str], set[str]] = {}
        for row in prior_rows:
            assignment = (
                normalize_term(row["term_family_label"]),
                normalize_term(row["search_domain_label"]),
                normalize_term(row["query_family_label"]),
            )
            prior_assignments.setdefault(assignment, set()).add(
                str(row.get("proposed_role") or "")
            )
        prior_families = {value[0] for value in prior_assignments}
        for row in output_rows:
            decision = str(row.get("decision") or "").casefold()
            family = str(row.get("term_family_label") or "").strip()
            if decision == "include":
                required = (
                    row.get("canonical_term"),
                    family,
                    row.get("term_relation"),
                    row.get("search_domain_label"),
                    row.get("search_domain_definition"),
                    row.get("query_family_label"),
                    row.get("reason"),
                )
                if not all(str(value or "").strip() for value in required):
                    raise ValueError(f"Included term lacks fields: {row[key]}")
                if str(row["term_relation"]).casefold() not in TERM_RELATIONS:
                    raise ValueError(f"Invalid term relation: {row[key]}")
                domain_labels = {
                    value.strip()
                    for value in str(row["search_domain_label"])
                    .replace(";", "|")
                    .split("|")
                    if value.strip()
                }
                cross_domain = parse_bool(
                    str(row.get("cross_domain") or ""),
                    "cross_domain",
                )
                if cross_domain != (len(domain_labels) > 1):
                    raise ValueError(
                        "cross_domain does not match the number of assigned "
                        f"domains: {row[key]}"
                    )
                included.append(row)
                assignment = (
                    normalize_term(family),
                    normalize_term(row["search_domain_label"]),
                    normalize_term(row["query_family_label"]),
                )
                if assignment in prior_assignments:
                    prior_roles = prior_assignments[assignment]
                    if (
                        "" not in prior_roles
                        and str(row.get("proposed_role") or "")
                        not in prior_roles
                    ):
                        raise ValueError(
                            "Mapped term lacks a same-role prior exemplar: "
                            f"{row[key]}"
                        )
                    mapped_rows.append(row)
                elif assignment[0] in prior_families:
                    raise ValueError(
                        "Reused term family has a different domain/query "
                        f"assignment: {row[key]}"
                    )
                else:
                    new_rows.append(row)
            elif decision == "exclude":
                if not str(row.get("reason") or "").strip():
                    raise ValueError(f"Excluded term lacks reason: {row[key]}")
                excluded.append(row)
            else:
                raise ValueError(f"Invalid term decision: {row[key]}")
        family_field = "term_family_label"
    mapped_families = {
        normalize_term(row[family_field]) for row in mapped_rows
    }
    new_families = {normalize_term(row[family_field]) for row in new_rows}
    counts = _manifest_counts(manifest)
    observed = {
        "include_count": len(included),
        "exclude_count": len(excluded),
        "mapped_to_existing_row_count": len(mapped_rows),
        "genuinely_new_row_count": len(new_rows),
        "mapped_to_existing_family_count": len(mapped_families),
        "genuinely_new_family_count": len(new_families),
    }
    aliases = {
        "include_count": ("include_rows", "included_term_count"),
        "exclude_count": ("exclude_rows", "excluded_term_count"),
        "mapped_to_existing_row_count": ("mapped_to_existing_rows",),
        "genuinely_new_row_count": ("genuinely_new_rows",),
        "mapped_to_existing_family_count": (
            "mapped_to_existing_term_families",
        ),
        "genuinely_new_family_count": ("genuinely_new_term_families",),
    }
    for count_key, value in observed.items():
        _assert_reported_count(
            counts,
            count_key,
            value,
            aliases.get(count_key, ()),
        )
    return {
        "status": "pass",
        "kind": normalized_kind,
        "rows": len(output_rows),
        **observed,
        "input_sha256": sha256_file(input_path),
        "output_sha256": sha256_file(output_path),
        "manifest_sha256": sha256_file(manifest_path),
        "protocol_sha256": prompt_sha,
        "reference_sha256": sha256_file(reference_path),
    }


def main() -> None:
    """Run deterministic post-review alignment validation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("indicator", "term"), required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    result = validate_alignment(
        args.kind,
        Path(args.input).resolve(),
        Path(args.output).resolve(),
        Path(args.protocol).resolve(),
        Path(args.manifest).resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

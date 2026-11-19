"""Build protected H2 sheets from JSON-embedded independent full-text reviews."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from common import sha256_file, write_csv, write_json

MENTION_FIELDS = (
    "raw_name_en",
    "canonical_name_en",
    "source_role",
    "formula_location",
    "evidence_span",
    "formula",
    "parameters",
    "required_data",
    "maximum_information_time",
    "scope_role",
    "requires_future",
    "extraction_notes",
)


def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a UTF-8 review CSV."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def _index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Index frozen source rows by record key."""
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        key = str(row.get("record_key", "")).strip()
        if not key or key in result:
            raise ValueError(f"Blank or duplicate record key: {key!r}")
        result[key] = row
    return result


def _mentions(row: dict[str, str], role: str) -> list[dict[str, str]]:
    """Validate one reviewer's JSON candidate payload."""
    raw = str(row.get(f"{role}_candidate_mentions_json", ""))
    values = json.loads(raw)
    if not isinstance(values, list):
        raise TypeError(f"{role} candidate payload is not a list")
    parsed: list[dict[str, str]] = []
    for value in values:
        if not isinstance(value, dict):
            raise TypeError(f"{role} candidate payload item is not an object")
        item = {field: str(value.get(field, "")) for field in MENTION_FIELDS}
        if not item["canonical_name_en"].strip():
            raise ValueError(f"{role} candidate has blank canonical name")
        parsed.append(item)
    return parsed


def _review_payload(
    rows: list[dict[str, str]], role: str
) -> tuple[dict[str, dict[str, str]], dict[tuple[str, str], dict[str, str]]]:
    """Read either embedded JSON or explicit source/mention review rows."""
    if rows and "row_type" in rows[0]:
        source_rows = [row for row in rows if row.get("row_type") == "source_review"]
        mention_rows = [
            row for row in rows if row.get("row_type") == "candidate_mention"
        ]
        sources = _index(source_rows)
        mentions: dict[tuple[str, str], dict[str, str]] = {}
        for row in mention_rows:
            key = str(row.get("record_key", "")).strip()
            name = " ".join(str(row.get("canonical_name_en", "")).casefold().split())
            if not key or not name or (key, name) in mentions:
                raise ValueError(f"Invalid duplicate/blank {role} mention identity")
            mentions[(key, name)] = {
                field: str(row.get(field, "")) for field in MENTION_FIELDS
            }
        return sources, mentions
    sources = _index(rows)
    mentions = {}
    for key, row in sources.items():
        for item in _mentions(row, role):
            identity = (key, " ".join(item["canonical_name_en"].casefold().split()))
            if identity in mentions:
                raise ValueError(f"Duplicate {role} mention identity: {identity}")
            mentions[identity] = item
    return sources, mentions


def build(
    input_path: Path,
    ai_path: Path,
    h1_path: Path,
    source_output: Path,
    mention_output: Path,
    manifest: Path,
) -> dict[str, Any]:
    """Create H2-only editable source and union-candidate sheets."""
    fields, frozen_rows = _read(input_path)
    _, ai_rows = _read(ai_path)
    _, h1_rows = _read(h1_path)
    frozen = _index(frozen_rows)
    ai, ai_mentions = _review_payload(ai_rows, "AI")
    h1, h1_mentions = _review_payload(h1_rows, "H1")
    if set(frozen) != set(ai) or set(frozen) != set(h1):
        raise ValueError("Review record sets differ from frozen input")

    source_rows: list[dict[str, str]] = []
    candidates: dict[tuple[str, str], dict[str, str]] = {}
    for base in frozen_rows:
        key = str(base["record_key"])
        row = dict(base)
        for lower, upper, review in (("ai", "AI", ai[key]), ("h1", "H1", h1[key])):
            disposition = (
                str(review.get(f"{upper}_source_disposition", "")).strip()
                or str(review.get("source_disposition", "")).strip()
            )
            notes = (
                str(review.get(f"{upper}_source_notes", "")).strip()
                or str(review.get("source_notes", "")).strip()
            )
            if not disposition or not notes:
                raise ValueError(f"{upper} source decision is incomplete: {key}")
            row[f"{lower}_source_disposition"] = disposition
            row[f"{lower}_source_notes"] = notes
        for lower, reviewer_mentions in (("ai", ai_mentions), ("h1", h1_mentions)):
            for (candidate_key, name), item in reviewer_mentions.items():
                if candidate_key != key:
                    continue
                identity = (candidate_key, name)
                candidates.setdefault(identity, {"record_key": candidate_key})
                for field, value in item.items():
                    candidates[identity][f"{lower}_{field}"] = value
        row["h2_final_source_disposition"] = ""
        row["h2_final_source_notes"] = ""
        source_rows.append(row)
    source_fields = fields + [
        "ai_source_disposition",
        "ai_source_notes",
        "h1_source_disposition",
        "h1_source_notes",
        "h2_final_source_disposition",
        "h2_final_source_notes",
    ]
    write_csv(source_output, source_rows, source_fields)

    mention_rows: list[dict[str, str]] = []
    for identity in sorted(candidates):
        row = candidates[identity]
        for role in ("ai", "h1"):
            for field in MENTION_FIELDS:
                row.setdefault(f"{role}_{field}", "")
        row["h2_decision"] = ""
        for field in MENTION_FIELDS:
            row[field] = ""
        mention_rows.append(row)
    mention_fields = (
        ["record_key"]
        + [f"{role}_{field}" for role in ("ai", "h1") for field in MENTION_FIELDS]
        + ["h2_decision", *MENTION_FIELDS]
    )
    write_csv(mention_output, mention_rows, mention_fields)
    result: dict[str, Any] = {
        "schema_version": "contextual_fulltext_h2_json_batch_input_v4",
        "source_count": len(source_rows),
        "union_candidate_count": len(mention_rows),
        "inputs": {
            str(path.resolve()): sha256_file(path)
            for path in (input_path, ai_path, h1_path)
        },
        "outputs": {
            str(path.resolve()): sha256_file(path)
            for path in (source_output, mention_output)
        },
    }
    write_json(manifest, result)
    return result


def main() -> None:
    """Run the JSON-review H2 input builder."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--ai", type=Path, required=True)
    parser.add_argument("--h1", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--mention-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.input,
                args.ai,
                args.h1,
                args.source_output,
                args.mention_output,
                args.manifest,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

"""Join independent batch-three full-text extraction reviews for H2 adjudication."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from common import ROOT, sha256_file, write_csv, write_json

OUTPUT_DIR = ROOT / "outputs"
SOURCE_INPUT = OUTPUT_DIR / "contextual_fulltext_extraction_input_batch3_v4.csv"
AI_SOURCES = OUTPUT_DIR / "contextual_fulltext_source_review_AI_batch3_v4.csv"
H1_SOURCES = OUTPUT_DIR / "contextual_fulltext_source_review_H1_batch3_v4.csv"
AI_MENTIONS = OUTPUT_DIR / "contextual_fulltext_indicator_mentions_AI_batch3_v4.csv"
H1_MENTIONS = OUTPUT_DIR / "contextual_fulltext_indicator_mentions_H1_batch3_v4.csv"
SOURCE_OUTPUT = OUTPUT_DIR / "contextual_fulltext_source_review_H2_batch3_v4.csv"
MENTION_OUTPUT = OUTPUT_DIR / "contextual_fulltext_indicator_mentions_H2_batch3_v4.csv"
MANIFEST = OUTPUT_DIR / "contextual_fulltext_h2_batch3_input_manifest_v4.json"
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
    """Read a UTF-8 CSV with a mandatory header."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def _index_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Index source review rows by unique record key."""
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        key = str(row.get("record_key", "")).strip()
        if not key or key in indexed:
            raise ValueError(f"Duplicate/blank source-review record key: {key!r}")
        indexed[key] = row
    return indexed


def _mention_key(row: dict[str, str]) -> tuple[str, str]:
    """Use source plus normalized canonical name for a conservative union."""
    return (
        str(row.get("record_key", "")).strip(),
        " ".join(str(row.get("canonical_name_en", "")).casefold().split()),
    )


def _prefixed(
    row: dict[str, str] | None, prefix: str, fields: tuple[str, ...]
) -> dict[str, str]:
    """Prefix a blind review payload while representing a missing proposal explicitly."""
    return {
        f"{prefix}_{field}": "" if row is None else str(row.get(field, ""))
        for field in fields
    }


def _blind_source_payload(row: dict[str, str], role: str) -> dict[str, str]:
    """Read either the standard or explicitly role-prefixed blind source fields."""
    values: dict[str, str] = {}
    role_upper = role.upper()
    values["source_disposition"] = (
        str(row.get(f"{role_upper}_source_disposition", "")).strip()
        or str(row.get(f"{role_upper}_disposition", "")).strip()
        or str(row.get("source_disposition", "")).strip()
    )
    values["source_notes"] = (
        str(row.get(f"{role_upper}_source_notes", "")).strip()
        or str(row.get(f"{role_upper}_rationale", "")).strip()
        or str(row.get("source_notes", "")).strip()
    )
    if not values["source_disposition"] or not values["source_notes"]:
        raise ValueError(f"{role} source review has blank decision or notes")
    return values


def build(
    source_input: Path,
    ai_sources: Path,
    h1_sources: Path,
    ai_mentions: Path,
    h1_mentions: Path,
    source_output: Path,
    mention_output: Path,
    manifest: Path,
) -> dict[str, Any]:
    """Create H2-editable sheets without altering any AI/H1 evidence fields."""
    source_headers, base_sources = _read(source_input)
    _, ai_source_rows = _read(ai_sources)
    _, h1_source_rows = _read(h1_sources)
    base, ai, h1 = (
        _index_rows(base_sources),
        _index_rows(ai_source_rows),
        _index_rows(h1_source_rows),
    )
    if set(base) != set(ai) or set(base) != set(h1):
        raise ValueError("AI/H1 source-review identity set differs from frozen batch")
    source_rows: list[dict[str, str]] = []
    for frozen in base_sources:
        key = str(frozen["record_key"])
        row = dict(frozen)
        for role, review in (("ai", ai[key]), ("h1", h1[key])):
            for field, value in _blind_source_payload(review, role).items():
                row[f"{role}_{field}"] = value
        row["h2_final_source_disposition"] = ""
        row["h2_final_source_notes"] = ""
        source_rows.append(row)
    source_fields = source_headers + [
        "ai_source_disposition",
        "ai_source_notes",
        "h1_source_disposition",
        "h1_source_notes",
        "h2_final_source_disposition",
        "h2_final_source_notes",
    ]
    write_csv(source_output, source_rows, source_fields)

    _, ai_candidate_rows = _read(ai_mentions)
    _, h1_candidate_rows = _read(h1_mentions)
    ai_candidates = {_mention_key(row): row for row in ai_candidate_rows}
    h1_candidates = {_mention_key(row): row for row in h1_candidate_rows}
    if len(ai_candidates) != len(ai_candidate_rows) or len(h1_candidates) != len(
        h1_candidate_rows
    ):
        raise ValueError("Duplicate source/canonical candidate in blind extraction")
    mentions: list[dict[str, str]] = []
    for key in sorted(set(ai_candidates) | set(h1_candidates)):
        row = {"record_key": key[0]}
        row.update(_prefixed(ai_candidates.get(key), "ai", MENTION_FIELDS))
        row.update(_prefixed(h1_candidates.get(key), "h1", MENTION_FIELDS))
        row["h2_decision"] = ""
        row.update({field: "" for field in MENTION_FIELDS})
        mentions.append(row)
    mention_fields = (
        ["record_key"]
        + [f"{role}_{field}" for role in ("ai", "h1") for field in MENTION_FIELDS]
        + ["h2_decision", *MENTION_FIELDS]
    )
    write_csv(mention_output, mentions, mention_fields)
    result = {
        "schema_version": "contextual_fulltext_h2_batch3_input_v4",
        "source_count": len(source_rows),
        "union_candidate_count": len(mentions),
        "inputs": {
            str(path.resolve()): sha256_file(path)
            for path in (source_input, ai_sources, h1_sources, ai_mentions, h1_mentions)
        },
        "outputs": {
            str(path.resolve()): sha256_file(path)
            for path in (source_output, mention_output)
        },
        "h2_editable_source_fields": [
            "h2_final_source_disposition",
            "h2_final_source_notes",
        ],
        "h2_editable_mention_fields": ["h2_decision", *MENTION_FIELDS],
    }
    write_json(manifest, result)
    return result


def main() -> None:
    """Build and report the H2 input artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-input", type=Path, default=SOURCE_INPUT)
    parser.add_argument("--ai-sources", type=Path, default=AI_SOURCES)
    parser.add_argument("--h1-sources", type=Path, default=H1_SOURCES)
    parser.add_argument("--ai-mentions", type=Path, default=AI_MENTIONS)
    parser.add_argument("--h1-mentions", type=Path, default=H1_MENTIONS)
    parser.add_argument("--source-output", type=Path, default=SOURCE_OUTPUT)
    parser.add_argument("--mention-output", type=Path, default=MENTION_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    result = build(
        args.source_input.resolve(),
        args.ai_sources.resolve(),
        args.h1_sources.resolve(),
        args.ai_mentions.resolve(),
        args.h1_mentions.resolve(),
        args.source_output.resolve(),
        args.mention_output.resolve(),
        args.manifest.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

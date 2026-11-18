"""Aggregate completed full-report theme screening, without calling an API.

Default inputs: theme_codebook.json, theme_matches/Pxxx.json, raw_reports.jsonl,
and ../sample.jsonl. Missing/invalid screenings are NEVER negative observations.
The matrix retains eligible papers with blank theme cells when not completed.

Optional semantic_adjustments.json ledger (raw screening files stay untouched)::

    {"version": 1, "adjustments": [
      {"adjustment_id": "review-001", "action": "exclude", "kind": "match",
       "pilot_id": "P001", "theme_id": "T01", "reason": "exclusion rationale",
       "reviewer": "human reviewer or audit process identifier"},
      {"adjustment_id": "review-002", "action": "add", "kind": "match",
       "pilot_id": "P001", "reason": "addition rationale", "reviewer": "name",
       "evidence": {"theme_id": "T02", "reason": "why the definition is met",
         "source_block_id": "P001_R1_1", "reviewer_id": "1",
         "source_path": "/absolute/original/source.md",
         "source_start_char": 100, "source_end_char": 200,
         "quote": "exact source slice", "selected_quote": "exact contained text"}}
    ]}

For exclusions, target_record_id from theme_evidence/unmapped_concerns can replace
(or disambiguate) theme_id. kind='other' uses label instead of theme_id. Additions
may also provide block offsets; when present they are checked against raw_reports.
Only completed papers can be adjusted: a manual hit is not a full-paper screening.
All adjustments require distinct IDs, rationale and reviewer attribution. Record
IDs depend on paper/kind/identity/source span/selected text, not output row order.

Outputs: theme_statistics.csv, theme_by_group.csv, paper_theme_matrix.csv,
theme_evidence.jsonl, unmapped_concerns.jsonl, aggregation_audit.json.
Percentages are percentage points (0–100); an empty denominator gives a blank.
Source-level provenance validation establishes location, not semantic correctness.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EXPECTED = {"all_994": 994}
Record = dict[str, Any]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[Record]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def unique(rows: list[Record], key: str, label: str) -> dict[str, Record]:
    ids = [row[key] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate {key} in {label}")
    return dict(zip(ids, rows, strict=True))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_inputs(input_dir: Path, sample_path: Path) -> tuple[dict, dict, dict]:
    book = read_json(input_dir / "theme_codebook.json")
    themes = book["themes"] if isinstance(book, dict) else book
    if not themes:
        raise ValueError("Empty theme codebook")
    theme_map = unique(themes, "theme_id", "theme codebook")
    sample = unique(read_jsonl(sample_path), "pilot_id", "sample")
    unique(list(sample.values()), "paper_id", "sample")
    reports = unique(read_jsonl(input_dir / "raw_reports.jsonl"), "pilot_id", "eligible reports")
    if not set(reports) <= set(sample):
        raise ValueError("Eligible reports contain IDs absent from sample")
    eligible = {pid: sample[pid] for pid in reports}
    for pid, report in reports.items():
        if report["paper_id"] != eligible[pid]["paper_id"]:
            raise ValueError(f"Paper identity mismatch: {pid}")
        unique(report["blocks"], "block_id", pid)
    if any(not re.fullmatch(r"P\d{3,}", pid) for pid in eligible):
        raise ValueError("Invalid paper identifier")
    actual = {"all_994": len(eligible)}
    if dict(actual) != EXPECTED:
        raise ValueError(f"Eligible roster differs from required population: {dict(actual)}")
    return theme_map, eligible, reports


def source_text(path: Path, cache: dict[str, str]) -> str:
    key = str(path)
    if key not in cache:
        cache[key] = path.read_text(encoding="utf-8")
    return cache[key]


def validate_quote(item: Record, row: Record, report: Record, cache: dict[str, str]) -> Record:
    """Verify exact source location plus selected passage and available block offsets."""
    result = dict(item)
    quote, selected = item["quote"], item["selected_quote"]
    if not isinstance(quote, str) or not quote.strip() or not isinstance(selected, str) or not selected.strip():
        raise ValueError("Empty/non-string quote or selected_quote")
    if selected not in quote:
        raise ValueError("selected_quote is not an exact substring of quote")
    path = Path(item.get("source_path") or row["review_path"])
    if not path.is_absolute():
        raise ValueError(f"Source path must be absolute: {path}")
    start, end = item["source_start_char"], item["source_end_char"]
    text = source_text(path, cache)
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
        raise ValueError("Invalid source character offsets")
    if text[start:end] != quote:
        raise ValueError(f"quote does not match source slice: {path}:{start}:{end}")
    blocks = {b["block_id"]: b for b in report["blocks"]}
    block = blocks[item["source_block_id"]]
    if str(item["reviewer_id"]) != str(block["reviewer_id"]):
        raise ValueError("Reviewer identity differs from source block")
    for a, b, value in [("start_in_block", "end_in_block", quote),
                         ("selected_start_in_block", "selected_end_in_block", selected)]:
        if (a in item) != (b in item):
            raise ValueError(f"Incomplete offset pair: {a}/{b}")
        if a in item:
            x, y = item[a], item[b]
            if type(x) is not int or type(y) is not int or not 0 <= x < y <= len(block["text"]):
                raise ValueError(f"Invalid block offsets: {a}/{b}")
            if block["text"][x:y] != value:
                raise ValueError(f"Block quote mismatch: {a}/{b}")
    if quote not in block["text"] or selected not in block["text"]:
        raise ValueError("Citation not contained in its declared reviewer block")
    result.update(source_path=str(path), source_quote_verified=True)
    return result


def make_record(item: Record, pid: str, kind: str, origin: str, themes: dict) -> Record:
    identity = item["theme_id"] if kind == "match" else item["label"]
    if kind == "match" and identity not in themes:
        raise ValueError(f"Unknown theme_id: {identity}")
    if not identity or not item.get("reason", "").strip():
        raise ValueError("Missing concern identity or reason")
    key = [pid, kind, identity, item["source_path"], item["source_start_char"],
           item["source_end_char"], item["selected_quote"]]
    token = hashlib.sha256(json.dumps(key, ensure_ascii=False).encode()).hexdigest()[:20]
    return dict(item, record_id=f"{pid}-{kind}-{token}", pilot_id=pid, kind=kind,
                origin=origin, included=True, adjustment_ids=[])


def load_completed(input_dir: Path, themes: dict, sample: dict, reports: dict,
                   cache: dict[str, str]) -> tuple[dict, list[Record], dict]:
    completed, records, invalid = {}, [], {}
    files = sorted((input_dir / "theme_matches").glob("*.json"))
    unexpected = [str(path) for path in files if path.stem not in sample]
    for path in files:
        pid = path.stem
        if pid not in sample:
            continue
        try:
            result = read_json(path)
            if result["pilot_id"] != pid or result["paper_id"] != sample[pid]["paper_id"]:
                raise ValueError("Result filename/paper identity mismatch")
            if result["read_complete"] is not True:
                raise ValueError("read_complete is not true")
            unique(result["matches"], "theme_id", str(path))
            paper_records = []
            for kind, field in [("match", "matches"), ("other", "other_concerns")]:
                for hit in result[field]:
                    bound = validate_quote(hit, sample[pid], reports[pid], cache)
                    paper_records.append(make_record(bound, pid, kind, "raw", themes))
            unique(paper_records, "record_id", str(path))
            records.extend(paper_records)
            completed[pid] = {"path": str(path), "sha256": digest(path),
                              "source_warnings": result.get("source_warnings", []),
                              "input_fingerprint": result.get("input_fingerprint"),
                              "completed_at": result.get("completed_at")}
        except (OSError, ValueError, TypeError, KeyError) as exc:
            invalid[pid] = {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
    unique(records, "record_id", "all raw evidence")
    audit = {"invalid_files": invalid, "unexpected_files": unexpected,
             "missing_result_ids": sorted(set(sample) - set(completed) - set(invalid))}
    return completed, records, audit


def apply_adjustments(path: Path, records: list[Record], completed: dict, themes: dict,
                      sample: dict, reports: dict, cache: dict[str, str]) -> list[Record]:
    if not path.exists():
        return []
    ledger = read_json(path)
    if ledger.get("version") != 1:
        raise ValueError("semantic_adjustments.json requires version=1")
    entries = ledger["adjustments"]
    unique(entries, "adjustment_id", "semantic adjustment ledger")
    for entry in entries:
        pid, aid = entry["pilot_id"], entry["adjustment_id"]
        if not aid or not entry.get("reason", "").strip() or not entry.get("reviewer", "").strip():
            raise ValueError(f"Adjustment requires ID, rationale and reviewer: {aid}")
        if pid not in completed:
            raise ValueError(f"Cannot adjust an incomplete/invalid paper: {pid}")
        kind = entry.get("kind", "match")
        if kind not in {"match", "other"}:
            raise ValueError(f"Invalid adjustment kind: {kind}")
        if entry["action"] == "add":
            hit = validate_quote(entry["evidence"], sample[pid], reports[pid], cache)
            record = make_record(hit, pid, kind, "manual_addition", themes)
            if any(r["record_id"] == record["record_id"] for r in records):
                raise ValueError(f"Duplicate evidence addition: {aid}")
            record["adjustment_ids"] = [aid]
            records.append(record)
        elif entry["action"] == "exclude":
            key = "theme_id" if kind == "match" else "label"
            candidates = [r for r in records if r["pilot_id"] == pid and r["kind"] == kind and r["included"]]
            if "target_record_id" in entry:
                candidates = [r for r in candidates if r["record_id"] == entry["target_record_id"]]
            else:
                candidates = [r for r in candidates if r[key] == entry[key]]
            if len(candidates) != 1:
                raise ValueError(f"Exclusion must identify exactly one active record: {aid}")
            candidates[0]["included"] = False
            candidates[0]["adjustment_ids"].append(aid)
        else:
            raise ValueError(f"Unknown adjustment action: {entry['action']}")
    unique(records, "record_id", "adjusted evidence")
    return entries


def presence(records: list[Record], completed: dict, raw: bool) -> dict[str, set[str]]:
    result = {pid: set() for pid in completed}
    for record in records:
        eligible = record["origin"] == "raw" if raw else record["included"]
        if record["kind"] == "match" and eligible:
            result[record["pilot_id"]].add(record["theme_id"])
    return result


def statistic(theme: Record, ids: set[str], completed: dict, raw: dict,
              effective: dict) -> Record:
    available = ids & set(completed)
    raw_n = sum(theme["theme_id"] in raw[pid] for pid in available)
    effective_n = sum(theme["theme_id"] in effective[pid] for pid in available)
    denominator = len(available)
    return {"theme_id": theme["theme_id"], "name_zh": theme["name_zh"],
            "eligible_papers": len(ids), "completed_papers": denominator,
            "not_completed_papers": len(ids) - denominator,
            "raw_present_papers": raw_n, "present_papers": effective_n,
            "raw_percentage": 100 * raw_n / denominator if denominator else "",
            "percentage": 100 * effective_n / denominator if denominator else "",
            "adjustment_delta_papers": effective_n - raw_n}


def aggregate(themes: dict, sample: dict, completed: dict, raw: dict,
              effective: dict) -> tuple[list[Record], list[Record], dict]:
    stats, groups, counts = [], [], {}
    for name in EXPECTED:
        ids = set(sample)
        available = ids & set(completed)
        counts[name] = {"eligible": len(ids), "completed": len(available),
                        "not_completed": len(ids - available)}
        for theme in themes.values():
            stats.append(statistic(theme, ids, completed, raw, effective))
        for dimension, metadata in [("journal", "journal_name"), ("field", "field_name")]:
            partitions: dict[str, set[str]] = {}
            for pid in ids:
                value = str(sample[pid].get(metadata) or "(missing)")
                partitions.setdefault(value, set()).add(pid)
            if sum(len(x) for x in partitions.values()) != len(ids):
                raise ValueError(f"Group eligible totals mismatch: {name}/{dimension}")
            if sum(len(x & available) for x in partitions.values()) != len(available):
                raise ValueError(f"Group completed totals mismatch: {name}/{dimension}")
            for value, members in sorted(partitions.items()):
                for theme in themes.values():
                    groups.append(dict(group_dimension=dimension, group_value=value,
                                       **statistic(theme, members, completed, raw, effective)))
    return stats, groups, counts


def matrix_rows(sample: dict, completed: dict, invalid: dict, themes: dict,
                raw: dict, effective: dict) -> list[Record]:
    rows = []
    for pid, metadata in sorted(sample.items()):
        done = pid in completed
        row = {"pilot_id": pid, "paper_id": metadata["paper_id"],
               "journal": metadata.get("journal_name", ""), "field": metadata.get("field_name", ""),
               "status": "completed" if done else "invalid" if pid in invalid else "missing",
               "raw_theme_count": len(raw[pid]) if done else "",
               "theme_count": len(effective[pid]) if done else ""}
        for tid in themes:
            row[f"raw_{tid}"] = int(tid in raw[pid]) if done else ""
            row[tid] = int(tid in effective[pid]) if done else ""
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[Record]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[Record]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=HERE)
    parser.add_argument("--sample", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--adjustments", type=Path, default=None)
    parser.add_argument("--require-complete", action="store_true", help="Write available results, then exit 2 unless all 994 are valid")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.input_dir.resolve()
    sample_path = args.sample or root.parent / "sample.jsonl"
    output = args.output_dir or root
    ledger_path = args.adjustments or root / "semantic_adjustments.json"
    themes, sample, reports = read_inputs(root, sample_path)
    cache: dict[str, str] = {}
    completed, records, audit = load_completed(root, themes, sample, reports, cache)
    adjustments = apply_adjustments(ledger_path, records, completed, themes, sample, reports, cache)
    raw, effective = presence(records, completed, True), presence(records, completed, False)
    stats, groups, counts = aggregate(themes, sample, completed, raw, effective)
    matrix = matrix_rows(sample, completed, audit["invalid_files"], themes, raw, effective)
    sources = {path: hashlib.sha256(text.encode()).hexdigest() for path, text in cache.items()}
    audit.update(schema_version=2, generated_at=datetime.now(timezone.utc).isoformat(),
                 expected_population=EXPECTED, population_status=counts, themes=len(themes),
                 completed_results=completed, completed_papers=len(completed),
                 all_complete=len(completed) == len(sample), matrix_rows=len(matrix),
                 raw_evidence_records=sum(r["origin"] == "raw" for r in records),
                 active_match_records=sum(r["kind"] == "match" and r["included"] for r in records),
                 active_unmapped_records=sum(r["kind"] == "other" and r["included"] for r in records),
                 adjustments=adjustments, adjustments_path=str(ledger_path),
                 adjustments_sha256=digest(ledger_path) if ledger_path.exists() else None,
                 codebook_sha256=digest(root / "theme_codebook.json"), sample_sha256=digest(sample_path),
                 eligible_reports_sha256=digest(root / "raw_reports.jsonl"),
                 source_text_sha256=sources, unit="paper-theme presence; multiple hits count once",
                 denominator="valid read_complete papers only; missing/invalid files excluded, never counted as zero",
                 checks={"unique_ids": True, "source_quotes_located": True,
                         "population_count": True, "group_denominators_partition_population": True},
                 limitations=["Exact source location does not validate semantic coding.",
                              "Source warnings remain attached to completed papers; no missing historical rounds are inferred.",
                              "This script does not recompute the model input fingerprint or independently certify read_complete."])
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "theme_statistics.csv", stats)
    write_csv(output / "theme_by_group.csv", groups)
    write_csv(output / "paper_theme_matrix.csv", matrix)
    write_jsonl(output / "theme_evidence.jsonl", [r for r in records if r["kind"] == "match"])
    write_jsonl(output / "unmapped_concerns.jsonl", [r for r in records if r["kind"] == "other"])
    (output / "aggregation_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"population": counts, "themes": len(themes), "invalid": len(audit["invalid_files"]),
                      "adjustments": len(adjustments), "output_dir": str(output)}, ensure_ascii=False))
    if args.require_complete and not audit["all_complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

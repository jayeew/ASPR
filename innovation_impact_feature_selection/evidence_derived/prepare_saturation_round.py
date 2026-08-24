#!/usr/bin/env python3
"""Create a deterministic, decision-blind saturation-round worksheet."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from pathlib import Path

try:
    from .core import EvidenceProtocol, canonical_json, normalize_text
except ImportError:
    from core import (  # type: ignore[no-redef]
        EvidenceProtocol,
        canonical_json,
        normalize_text,
    )


TARGET_BLOCKS = {
    "innovation": ("novelty", "innovation", "innovative"),
    "impact": ("potential impact", "citation impact", "scientific influence"),
    "quality": ("research quality", "paper quality"),
    "generic": (),
}
EVIDENCE_BLOCKS = {
    "measurement": ("measure", "indicator", "metric", "index", "score"),
    "prediction": ("predictor", "feature", "determinant", "prediction"),
    "validation": ("validation", "validate", "review"),
    "generic": (),
}


def year_band(year: int | None) -> str:
    if year is None or year < 2000:
        return "pre_2000"
    if year < 2010:
        return "2000_2009"
    if year < 2020:
        return "2010_2019"
    return "2020_2026_cutoff"


def first_block(text: str, blocks: dict[str, tuple[str, ...]]) -> str:
    normalized = normalize_text(text)
    for label, terms in blocks.items():
        if terms and any(term in normalized for term in terms):
            return label
    return "generic"


def prepare(database: Path, round_no: int, per_stratum: int, output: Path) -> int:
    engine = EvidenceProtocol(database, output.parent)
    engine.initialize()
    prior_rounds = list(
        engine.connection.execute(
            "SELECT round_no,fully_reviewed,decision FROM saturation_rounds "
            "WHERE round_no<? ORDER BY round_no",
            (round_no,),
        )
    )
    if [int(row[0]) for row in prior_rounds] != list(range(1, round_no)):
        raise RuntimeError("Every prior saturation round must be imported consecutively")
    if any(int(row[1]) != 1 for row in prior_rounds):
        raise RuntimeError("Every prior saturation round must be fully reviewed")
    if any(row[2] == "stop" for row in prior_rounds):
        raise RuntimeError("Cannot prepare a round after the protocol stop")
    assigned = {
        row[0]
        for row in engine.connection.execute(
            "SELECT work_id FROM discovery_round_records"
        )
    }
    protocol_hash = str(engine.get_metadata("protocol_hash"))
    candidates: dict[str, list[tuple[str, sqlite3.Row]]] = {}
    for row in engine.connection.execute(
        "SELECT * FROM provider_cache_records WHERE cache_status='raw_cache_only_no_decisions' "
        "AND language='en' AND publication_year<=2026 ORDER BY record_key"
    ):
        work_id = row["record_key"]
        if work_id in assigned:
            continue
        text = f"{row['title']} {row['abstract']}"
        raw_payload = json.loads(row["raw_json"])
        publication_date = str(raw_payload.get("publication_date") or "")
        if row["publication_year"] == 2026 and (
            not publication_date or publication_date > "2026-07-28"
        ):
            continue
        stratum = "__".join(
            (
                year_band(row["publication_year"]),
                row["work_type"] or "unknown",
                first_block(text, TARGET_BLOCKS),
                first_block(text, EVIDENCE_BLOCKS),
            )
        )
        rank = hashlib.sha256(
            f"{protocol_hash}|{work_id}|{stratum}".encode()
        ).hexdigest()
        candidates.setdefault(stratum, []).append((rank, row))
    selected: list[tuple[str, str, sqlite3.Row]] = []
    for stratum, rows in sorted(candidates.items()):
        for rank, row in sorted(rows, key=lambda item: item[0])[:per_stratum]:
            selected.append((stratum, rank, row))
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "round_no",
        "work_id",
        "stratum_id",
        "stable_rank",
        "doi",
        "openalex_id",
        "title",
        "abstract",
        "language",
        "publication_year",
        "publication_date",
        "work_type",
        "primary_decision",
        "exclusion_code",
        "evidence",
        "reason",
        "term_mentions_json",
        "indicator_mentions_json",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for stratum, rank, row in sorted(selected, key=lambda item: (item[0], item[1])):
            publication_date = str(
                json.loads(row["raw_json"]).get("publication_date") or ""
            )
            writer.writerow(
                {
                    "round_no": round_no,
                    "work_id": row["record_key"],
                    "stratum_id": stratum,
                    "stable_rank": rank,
                    "doi": row["doi"],
                    "openalex_id": row["provider_id"],
                    "title": row["title"],
                    "abstract": row["abstract"],
                    "language": row["language"],
                    "publication_year": row["publication_year"],
                    "publication_date": publication_date,
                    "work_type": row["work_type"],
                    "primary_decision": "",
                    "exclusion_code": "",
                    "evidence": "",
                    "reason": "",
                    "term_mentions_json": "[]",
                    "indicator_mentions_json": "[]",
                }
            )
            engine.connection.execute(
                "INSERT INTO discovery_round_records VALUES(?,?,?,?,?)",
                (round_no, row["record_key"], stratum, rank, row["payload_sha256"]),
            )
    engine.connection.commit()
    manifest = {
        "round_no": round_no,
        "rows": len(selected),
        "strata": len({item[0] for item in selected}),
        "per_stratum": per_stratum,
        "protocol_hash": protocol_hash,
        "worksheet_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "decisions_in_input": False,
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    engine.close()
    return len(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--per-stratum", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = prepare(args.database, args.round, args.per_stratum, args.output)
    print(
        canonical_json({"round": args.round, "rows": rows, "output": str(args.output)})
    )


if __name__ == "__main__":
    main()

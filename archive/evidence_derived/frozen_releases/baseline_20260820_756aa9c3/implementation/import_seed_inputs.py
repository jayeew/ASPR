#!/usr/bin/env python3
"""Import development and hidden seeds with source provenance, not decisions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    from .core import EvidenceProtocol, ProtocolError, canonical_json, file_hash
except ImportError:
    from core import (  # type: ignore[no-redef]
        EvidenceProtocol,
        ProtocolError,
        canonical_json,
        file_hash,
    )


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def import_seeds(database: Path, development: Path, hidden: Path) -> dict[str, int]:
    engine = EvidenceProtocol(database)
    engine.initialize()
    inputs = (("development", development), ("hidden", hidden))
    counts: dict[str, int] = {}
    for cohort, path in inputs:
        source_hash = file_hash(path)
        imported = 0
        for row in rows(path):
            if cohort == "hidden" and row.get("eligibility_status") != "eligible":
                continue
            doi = row.get("doi", "").strip().lower()
            if not doi or row.get("language", "en") != "en":
                raise ProtocolError(f"Invalid {cohort} seed row: {row.get('seed_id')}")
            match = engine.connection.execute(
                "SELECT record_key FROM provider_cache_records WHERE lower(doi)=? "
                "ORDER BY record_key LIMIT 1",
                (doi,),
            ).fetchone()
            work_id = str(match[0]) if match else ""
            engine.connection.execute(
                "INSERT INTO seed_inputs VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    row["seed_id"], cohort, doi, row.get("citation", ""),
                    int(row["publication_year"]) if row.get("publication_year") else None,
                    row.get("language", "en"), str(path.resolve()), source_hash,
                    "seed_input_only_no_legacy_decisions",
                ),
            )
            engine.connection.execute(
                "INSERT INTO seed_recall VALUES(?,?,?,?,?,?,?)",
                (
                    row["seed_id"], cohort, work_id,
                    "indexable_cache_match" if match else "unchecked",
                    "unchecked", "", canonical_json([]),
                ),
            )
            imported += 1
        counts[cohort] = imported
    if counts != {"development": 53, "hidden": 3}:
        raise ProtocolError(f"Unexpected seed cohort counts: {counts}")
    engine.connection.commit()
    engine.close()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--hidden", type=Path, required=True)
    args = parser.parse_args()
    print(canonical_json(import_seeds(args.database, args.development, args.hidden)))


if __name__ == "__main__":
    main()

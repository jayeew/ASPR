#!/usr/bin/env python3
"""Import integrity-matching v4 provider payloads as decision-free cache rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

try:
    from .core import EvidenceProtocol, file_hash, utc_now
except ImportError:
    from core import EvidenceProtocol, file_hash, utc_now  # type: ignore[no-redef]


def payload_hash(row: sqlite3.Row) -> str:
    payload = "\0".join(
        (str(row["abstract"]), str(row["referenced_works_json"]), str(row["raw_json"]))
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def bridge(source: Path, database: Path, output: Path) -> dict[str, Any]:
    source_hash = file_hash(source)
    source_connection = sqlite3.connect(source)
    source_connection.row_factory = sqlite3.Row
    engine = EvidenceProtocol(database, output.parent)
    engine.initialize()
    expected = {
        (row["provider"], row["record_key"]): row["payload_sha256"]
        for row in source_connection.execute("SELECT * FROM record_payload_digests")
    }
    inserted = 0
    mismatched: list[str] = []
    for row in source_connection.execute(
        "SELECT * FROM records ORDER BY provider,record_key"
    ):
        digest = payload_hash(row)
        if expected.get((row["provider"], row["record_key"])) != digest:
            mismatched.append(row["record_key"])
            continue
        engine.connection.execute(
            "INSERT OR REPLACE INTO provider_cache_records VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                row["provider"],
                row["record_key"],
                row["provider_id"],
                row["doi"],
                row["title"],
                row["abstract"],
                row["language"],
                row["publication_year"],
                row["work_type"],
                row["source_url"],
                row["referenced_works_json"],
                row["raw_json"],
                row["retrieval_route"],
                digest,
                source_hash,
                "raw_cache_only_no_decisions",
            ),
        )
        inserted += 1
    engine.connection.commit()
    engine.set_metadata(
        "provider_cache_bridge",
        {
            "source_database": str(source.resolve()),
            "source_database_sha256": source_hash,
            "imported_records": inserted,
            "digest_mismatches_quarantined": len(mismatched),
            "decisions_imported": 0,
            "created_at": utc_now(),
        },
    )
    engine.connection.commit()
    engine.close()
    source_connection.close()
    payload = {
        "status": "complete",
        "source_database_sha256": source_hash,
        "imported_records": inserted,
        "digest_mismatches_quarantined": len(mismatched),
        "quarantined_record_keys": mismatched,
        "decisions_imported": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            bridge(args.source, args.database, args.output),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

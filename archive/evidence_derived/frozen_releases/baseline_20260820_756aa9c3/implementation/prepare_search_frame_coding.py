#!/usr/bin/env python3
"""Export a decision-blind term-family worksheet after saturation stops."""

from __future__ import annotations

import argparse
import csv
import json
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


def prepare(database: Path, output: Path) -> dict[str, object]:
    engine = EvidenceProtocol(database)
    engine.initialize()
    stop = engine.connection.execute(
        "SELECT round_no,stop_basis FROM saturation_rounds "
        "WHERE decision='stop' ORDER BY round_no LIMIT 1"
    ).fetchone()
    if not stop:
        raise ProtocolError("Search-frame coding requires a protocol saturation stop")
    fields = [
        "family_id", "canonical_label", "aliases_json", "source_work_ids_json",
        "source_titles_json", "evidence_json", "first_seen_round",
        "coding_disposition", "term_role", "canonical_group",
        "proposed_domain", "evidence_quote", "reason",
    ]
    rows: list[dict[str, object]] = []
    for family in engine.connection.execute(
        "SELECT * FROM discovery_term_families ORDER BY normalized_label"
    ):
        work_ids = json.loads(family["source_work_ids_json"])
        titles: list[str] = []
        for work_id in work_ids:
            record = engine.connection.execute(
                "SELECT title FROM provider_cache_records WHERE record_key=?",
                (work_id,),
            ).fetchone()
            if record and record[0] not in titles:
                titles.append(str(record[0]))
        rows.append(
            {
                "family_id": family["family_id"],
                "canonical_label": family["canonical_label"],
                "aliases_json": family["aliases_json"],
                "source_work_ids_json": family["source_work_ids_json"],
                "source_titles_json": canonical_json(titles),
                "evidence_json": family["evidence_json"],
                "first_seen_round": family["first_seen_round"],
                "coding_disposition": "",
                "term_role": "",
                "canonical_group": "",
                "proposed_domain": "",
                "evidence_quote": "",
                "reason": "",
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": "search_frame_coding_blind_1",
        "rows": len(rows),
        "saturation_stop_round": int(stop[0]),
        "saturation_stop_basis": stop[1],
        "worksheet_sha256": file_hash(output),
        "decisions_in_input": False,
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    engine.close()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(canonical_json(prepare(args.database, args.output)))


if __name__ == "__main__":
    main()

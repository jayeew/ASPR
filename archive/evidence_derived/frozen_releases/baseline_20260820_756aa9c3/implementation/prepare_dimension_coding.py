#!/usr/bin/env python3
"""Export included formal literature for blind construct and dimension coding."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    from .core import EvidenceProtocol, canonical_json, file_hash
except ImportError:
    from core import (  # type: ignore[no-redef]
        EvidenceProtocol,
        canonical_json,
        file_hash,
    )


def prepare(database: Path, output_dir: Path) -> dict[str, object]:
    with EvidenceProtocol(database, output_dir) as engine:
        engine.initialize()
        rows = list(
            engine.connection.execute(
                "SELECT w.work_id,w.doi,w.openalex_id,w.title,w.abstract,w.publication_year,"
                "w.language,w.work_type,p.routes_json,p.query_ids_json,p.stable_rank "
                "FROM screening_final f JOIN works w USING(work_id) "
                "JOIN formal_pool_records p USING(work_id) WHERE f.decision='include' "
                "ORDER BY p.stable_rank"
            )
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "formal_included_dimension_blind.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(rows[0].keys() if rows else [])
            writer.writerows(tuple(row) for row in rows)
        manifest = {
            "artifact": "formal_included_dimension_blind",
            "row_count": len(rows),
            "unique_work_ids": len({row["work_id"] for row in rows}),
            "input_screening_final_count": engine.connection.execute(
                "SELECT COUNT(*) FROM screening_final"
            ).fetchone()[0],
            "sha256": file_hash(path),
            "contains_primary_dimension_decisions": False,
            "contains_independent_dimension_decisions": False,
            "model_outcomes_consulted": False,
        }
        manifest_path = output_dir / "formal_included_dimension_blind.manifest.json"
        manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
        engine.set_metadata("stage", "derive-dimensions-coding")
        engine.connection.commit()
        return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(canonical_json(prepare(args.database, args.output_dir)))


if __name__ == "__main__":
    main()

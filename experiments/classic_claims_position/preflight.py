"""Read-only source-asset audit for the classic contribution preparation."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/classic_claims_position_2025_2026/preparation"


def main() -> None:
    source = ROOT / "data/claim_graph/claim_graph_index.sqlite"
    groups = [
        ("G01", "超导电路", 2, "2025-10-07"),
        ("G02", "MOF", 4, "2025-10-08"),
        ("G03", "免疫耐受", 4, "2025-10-06"),
        ("G04", "IceCube", 2, "2026-10-06"),
        ("G05", "手性化学", 3, "2026-10-07"),
        ("G06", "光遗传学", 3, "2026-10-05"),
    ]
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as db:
        lo, hi, count = db.execute(
            "SELECT MIN(publication_date), MAX(publication_date), COUNT(*) FROM claim_nodes"
        ).fetchone()
        rows = []
        for ident, label, size, cutoff in groups:
            eligible = db.execute(
                "SELECT COUNT(*) FROM claim_nodes WHERE publication_date < ?", (cutoff,)
            ).fetchone()[0]
            rows.append(
                {
                    "group_id": ident,
                    "topic": label,
                    "expected_claims": size,
                    "cutoff_exclusive": cutoff,
                    "eligible_existing_claims": eligible,
                    "snapshot_required": hi >= cutoff,
                    "cutoff_verification": "pending",
                }
            )
    OUT.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "preflight_only",
        "source": str(source),
        "min_date": lo,
        "max_date": hi,
        "source_claim_count": count,
        "groups": rows,
        "analysis_started": False,
        "snapshots_built": False,
        "evidence_packets_prepared": False,
    }
    (OUT / "preflight.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Run native builder, restricting motif queries to existing semantic endpoints."""

from __future__ import annotations
from figure_pipeline.paths import PROJECT_ROOT

import importlib.util
import sqlite3
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

ROOT = PROJECT_ROOT
SNAP = ROOT / "outputs/fig10_reference/snapshot"


def main() -> None:
    spec = importlib.util.spec_from_file_location(
        "native_builder", ROOT / "scripts/claim_graph/07_build_claim_graph.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.motifs
    canonical = pq.read_table(
        SNAP / "canonical_target_works.parquet",
        columns=["nature_article_id", "work_id"],
    ).to_pylist()
    ids = {r["nature_article_id"]: r["work_id"] for r in canonical}
    allowed: dict[str, set[str]] = {}
    for r in pq.read_table(
        SNAP / "semantic_claim_edges.parquet",
        columns=["earlier_paper_id", "later_paper_id"],
    ).to_pylist():
        allowed.setdefault(ids[r["later_paper_id"]], set()).add(
            ids[r["earlier_paper_id"]]
        )

    class Restricted:
        def __init__(self, db: sqlite3.Connection, works: set[str]) -> None:
            self.db = db
            self.works = sorted(works)

        def execute(self, sql: str, parameters: tuple[Any, ...]) -> sqlite3.Cursor:
            clause = " AND p.work_id IN (" + ",".join("?" for _ in self.works) + ")"
            sql = (
                sql.replace(" GROUP BY", clause + " GROUP BY")
                if " GROUP BY" in sql
                else sql + clause
            )
            return self.db.execute(sql, (*parameters, *self.works))

    def restricted(db: sqlite3.Connection, work: str, date: str, refs: int) -> dict:
        return original(Restricted(db, allowed.get(work, set())), work, date, refs)

    # Check numerical equivalence for three observed semantic endpoint sets.
    db = sqlite3.connect(f'file:{SNAP / "paper_graph_index.sqlite"}?mode=ro', uri=True)
    for work in sorted(allowed)[:3]:
        date, refs = db.execute(
            "SELECT publication_date,referenced_works_count FROM paper_nodes WHERE work_id=?",
            (work,),
        ).fetchone()
        expected = {
            k: v
            for k, v in original(db, work, date, refs).items()
            if k in allowed[work]
        }
        assert expected == restricted(db, work, date, refs), work
    db.close()
    module.motifs = restricted
    raise SystemExit(module.main())


if __name__ == "__main__":
    main()

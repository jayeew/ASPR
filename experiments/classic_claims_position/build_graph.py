"""Run native builder, restricting motif queries to existing semantic endpoints."""

from __future__ import annotations

import importlib.util
import math
import sqlite3
import sys
from functools import lru_cache
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
SNAP = Path(sys.argv[sys.argv.index("--output-root") + 1]).resolve()


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

    @lru_cache(maxsize=100000)
    def references(db: sqlite3.Connection, work: str) -> frozenset[str]:
        return frozenset(
            r[0]
            for r in db.execute(
                "SELECT cited_work_id FROM paper_edges WHERE citing_work_id=?", (work,)
            )
        )

    @lru_cache(maxsize=100000)
    def info(db: sqlite3.Connection, work: str) -> tuple | None:
        return db.execute(
            "SELECT publication_date,referenced_works_count,is_nature_target FROM paper_nodes WHERE work_id=?",
            (work,),
        ).fetchone()

    def restricted(db: sqlite3.Connection, work: str, date: str, refs: int) -> dict:
        # Exact set intersections implement the native three SQL joins on unique edges.
        candidates = {w: info(db, w) for w in allowed.get(work, set())}
        candidates = {
            w: v for w, v in candidates.items() if v and v[2] == 1 and v[0] < date
        }
        direct = references(db, work)
        result = {}

        def item(w: str) -> dict:
            return result.setdefault(
                w,
                {
                    "direct": False,
                    "min_path": None,
                    "two_count": 0,
                    "two_ra": 0.0,
                    "shared_count": 0,
                    "shared_salton": 0.0,
                },
            )

        for w in direct.intersection(candidates):
            item(w).update(direct=True, min_path=1)
        for mid in sorted(direct):
            metadata = info(db, mid)
            if metadata is None:
                continue
            for w in references(db, mid).intersection(candidates):
                row = item(w)
                row["two_count"] += 1
                row["two_ra"] += 1.0 / max(metadata[1], 1)
                row["min_path"] = min(row["min_path"] or 2, 2)
        for w, metadata in candidates.items():
            count = len(direct.intersection(references(db, w)))
            if count:
                item(w).update(
                    shared_count=count,
                    shared_salton=count
                    / math.sqrt(max(refs, 1) * max(int(metadata[1] or 0), 1)),
                )
        return result

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
    references.cache_clear()
    info.cache_clear()
    db.close()
    module.motifs = restricted
    raise SystemExit(module.main())


if __name__ == "__main__":
    main()

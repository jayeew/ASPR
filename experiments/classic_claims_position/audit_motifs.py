"""Compare nonempty snapshot annotations against unmodified native SQL."""

from __future__ import annotations

import importlib.util
import json
import math
import sqlite3

import pyarrow.parquet as pq

from .preflight import OUT, ROOT


def main() -> None:
    spec = importlib.util.spec_from_file_location(
        "native", ROOT / "scripts/claim_graph/07_build_claim_graph.py"
    )
    assert spec and spec.loader
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    results = []
    for cutoff in ["2025-10-06", "2025-10-07", "2025-10-08"]:
        root = OUT / "snapshots" / cutoff
        paths = pq.read_table(root / "paper_path_claim_edges.parquet").to_pylist()
        # Distributed samples include chunks resumed under both implementations.
        selected = [
            paths[int((len(paths) - 1) * f)] for f in [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
        ]
        ids = {
            r["nature_article_id"]: r["work_id"]
            for r in pq.read_table(
                root / "canonical_target_works.parquet",
                columns=["nature_article_id", "work_id"],
            ).to_pylist()
        }
        mapping = {
            "paper_direct_citation": "direct",
            "paper_min_path_length": "min_path",
            "paper_directed_two_hop_count": "two_count",
            "paper_two_hop_ra_weight": "two_ra",
            "paper_shared_reference_count": "shared_count",
            "paper_shared_reference_salton": "shared_salton",
        }
        with sqlite3.connect(
            f"file:{root}/paper_graph_index.sqlite?mode=ro", uri=True
        ) as db:
            for row in selected:
                later = ids[row["later_paper_id"]]
                earlier = ids[row["earlier_paper_id"]]
                refs = db.execute(
                    "SELECT referenced_works_count FROM paper_nodes WHERE work_id=?",
                    (later,),
                ).fetchone()[0]
                native_row = native.motifs(
                    db, later, str(row["later_publication_date"]), refs
                )[earlier]
                for actual, key in mapping.items():
                    a, b = row[actual], native_row[key]
                    assert (
                        math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-7)
                        if isinstance(a, float)
                        else a == b
                    ), (cutoff, actual, a, b)
                results.append(
                    {
                        "cutoff": cutoff,
                        "earlier_claim": row["earlier_claim_id"],
                        "later_claim": row["later_claim_id"],
                        "native_equivalent": True,
                        "tolerance": "float32 serialization: rtol=1e-6, atol=1e-7; counts/flags exact",
                        "fields": list(mapping),
                    }
                )
        print(cutoff, "7 nonempty annotations match native SQL", flush=True)
    (OUT / "native_motif_equivalence.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

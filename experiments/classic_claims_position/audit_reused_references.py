"""Record reference metadata anomalies separately from claim corpus coverage."""

from __future__ import annotations

import json
import sqlite3

from .preflight import OUT, ROOT


def main() -> None:
    query = "SELECT COUNT(*) FROM paper_edges e JOIN paper_nodes future ON future.work_id=e.cited_work_id JOIN paper_nodes n ON n.work_id=e.citing_work_id WHERE future.publication_date >= '2026-10-05' AND n.is_nature_target=1"
    with sqlite3.connect(
        f"file:{ROOT}/data/claim_graph/paper_graph_index.sqlite?mode=ro", uri=True
    ) as db:
        future = db.execute(
            "SELECT work_id,publication_date,is_nature_target FROM paper_nodes WHERE publication_date >= '2026-10-05'"
        ).fetchall()
        unknown = db.execute(
            "SELECT COUNT(*) FROM paper_nodes WHERE publication_date IS NULL"
        ).fetchone()[0]
        cited = db.execute(query).fetchone()[0]
    result = {
        "cutoff": "2026-10-05",
        "future_nodes": future,
        "future_node_count": len(future),
        "undated_node_count": unknown,
        "future_nodes_cited_by_nature_targets": cited,
        "sql": query,
        "impact": "Post-cutoff reference nodes are outside native target path annotations when no Nature target cites them. Unknown-date reference metadata remains unresolved and is not scientific support.",
    }
    (OUT / "reused_reference_anomalies.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    assert (
        cited == 0
    ), "Future references require a separate filtered reuse view before delivery"


if __name__ == "__main__":
    main()

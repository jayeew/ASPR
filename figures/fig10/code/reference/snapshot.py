"""Build an isolated pre-publication graph using existing native builders."""

from __future__ import annotations
from figure_pipeline.paths import PROJECT_ROOT

import json
import sqlite3
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = PROJECT_ROOT
SOURCE = ROOT / "data/claim_graph"
OUT = ROOT / "outputs/fig10_reference/snapshot"
CUTOFF = "2024-05-08"


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    canonical = pq.read_table(SOURCE / "canonical_target_works.parquet")
    keep = [
        bool(r["publication_date"]) and str(r["publication_date"])[:10] < CUTOFF
        for r in canonical.to_pylist()
    ]
    canonical = canonical.filter(pa.array(keep))
    pq.write_table(canonical, OUT / "canonical_target_works.parquet")
    claims = pq.read_table(SOURCE / "claim_nodes.parquet")
    claims = claims.filter(
        pc.is_in(claims["parent_paper_id"], value_set=canonical["nature_article_id"])
    )
    pq.write_table(claims, OUT / "claim_nodes.parquet")
    embeddings = pq.read_table(SOURCE / "claim_embeddings.parquet")
    embeddings = embeddings.filter(
        pc.is_in(embeddings["claim_id"], value_set=claims["claim_id"])
    )
    pq.write_table(embeddings, OUT / "claim_embeddings.parquet")
    target = OUT / "paper_graph_index.sqlite"
    if not target.exists():
        source = sqlite3.connect(
            f'file:{SOURCE / "paper_graph_index.sqlite"}?mode=ro', uri=True
        )
        dest = sqlite3.connect(target)
        source.backup(dest)
        dest.execute(
            "DELETE FROM paper_nodes WHERE publication_date IS NULL OR publication_date >= ?",
            (CUTOFF,),
        )
        dest.execute(
            "DELETE FROM paper_edges WHERE citing_work_id NOT IN (SELECT work_id FROM paper_nodes) OR cited_work_id NOT IN (SELECT work_id FROM paper_nodes)"
        )
        dest.commit()
        dest.close()
        source.close()
    (OUT / "snapshot_manifest.json").write_text(
        json.dumps(
            {
                "cutoff_exclusive": CUTOFF,
                "claim_count": len(claims),
                "paper_count": len(canonical),
                "source": str(SOURCE),
                "communities": "Recomputed on pre-cutoff corpus; not inherited",
                "percentiles": "Not used by thresholded runtime",
            },
            indent=2,
        )
    )


def main() -> None:
    prepare()
    python = ROOT / "outputs/fig10_reference/snapshot_env/bin/python"
    subprocess.run(
        [
            str(python),
            str(ROOT / "figure_pipeline/fig10_reference/build_snapshot_graph.py"),
            "--claims",
            str(OUT / "claim_nodes.parquet"),
            "--embeddings",
            str(OUT / "claim_embeddings.parquet"),
            "--canonical-targets",
            str(OUT / "canonical_target_works.parquet"),
            "--paper-index",
            str(OUT / "paper_graph_index.sqlite"),
            "--output-root",
            str(OUT),
            "--device",
            "cpu",
            "--matrix-batch-size",
            "128",
            "--resume",
        ],
        check=True,
    )
    args = [
        str(python),
        str(ROOT / "scripts/claim_graph/08_build_claim_graph_statistics.py"),
        "--output-root",
        str(OUT),
        "--workers",
        "2",
    ]
    for flag, name in [
        ("claims", "claim_nodes.parquet"),
        ("edges", "claim_edges.parquet"),
        ("communities", "claim_communities.parquet"),
        ("embeddings", "claim_embedding_matrix.npy"),
        ("embedding-index", "claim_embedding_index.parquet"),
        ("graph-index", "claim_graph_index.sqlite"),
    ]:
        args += ["--" + flag, str(OUT / name)]
    subprocess.run(args, check=True)


if __name__ == "__main__":
    main()

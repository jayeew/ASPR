from __future__ import annotations

import gzip
import importlib.util
import json
import logging
import sqlite3
from pathlib import Path
from types import ModuleType

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gear.claim_attribution import ClaimGraphRuntime

ROOT = Path(__file__).resolve().parents[2]


def _script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts/claim_graph" / f"{name}.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_raw_import_retains_anomaly_source_and_only_cross_work_edges(
    tmp_path: Path,
) -> None:
    build = _script("02_build_paper_graph")
    build._WORKER_PHASE = "pass1"
    build._WORKER_CHUNK_DIR = tmp_path
    build._TARGET_DOIS = {"10.1234/example": "article"}
    shard = tmp_path / "source.gz"
    with gzip.open(shard, "wt", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "id": "https://openalex.org/W1",
                    "doi": "10.1234/example",
                    "referenced_works": ["W1", "W2", "https://openalex.org/W2", "bad"],
                }
            )
            + "\n"
        )
    build.scan_shard((1, str(shard)))
    rows = [
        json.loads(line) for line in (tmp_path / "00001.jsonl").read_text().splitlines()
    ]
    edges = [row["row"] for row in rows if row["kind"] == "edge"]
    assert [(row["citing_work_id"], row["cited_work_id"]) for row in edges] == [
        ("W1", "W2")
    ]
    anomalies = [row["row"] for row in rows if row["kind"] == "anomaly"]
    assert {row["reason"] for row in anomalies} == {
        "same_work_citation",
        "duplicate_reference",
        "invalid_work_id",
    }
    assert all(
        row["source_shard"] == str(shard) and row["source_line"] == 1
        for row in anomalies
    )


def test_resumed_old_chunks_are_cleaned_with_audit_and_isolated_nodes_retained(
    tmp_path: Path,
) -> None:
    build = _script("02_build_paper_graph")
    chunk = tmp_path / "chunks/paper_graph/pass1/00001.jsonl"
    chunk.parent.mkdir(parents=True)
    rows = [
        {"kind": "edge", "row": {"citing_work_id": a, "cited_work_id": b}}
        for a, b in [("W1", "W1"), ("W1", "W2"), ("W1", "W2"), ("W1", "W9")]
    ]
    rows += [{"kind": "node", "row": {"work_id": "W3"}}]
    chunk.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    edges = list(build.normalized_edge_rows(tmp_path, {"W1", "W2", "W3"}, set(), set()))
    assert [(row["citing_work_id"], row["cited_work_id"]) for row in edges] == [
        ("W1", "W2")
    ]
    anomalies = [
        json.loads(line)
        for line in (tmp_path / "paper_edge_anomalies.jsonl").read_text().splitlines()
    ]
    assert {row["reason"] for row in anomalies} == {
        "same_work_citation",
        "duplicate_citation",
        "missing_endpoint",
    }
    assert all(row["source_chunk"] == str(chunk) for row in anomalies)
    assert list(build.normalized_node_rows(tmp_path, {"W3"}, set(), set())) == [
        {"work_id": "W3"}
    ]


def test_index_quarantines_bad_edges_and_resume_is_stable(tmp_path: Path) -> None:
    index = _script("06_build_paper_graph_index")
    connection = index.connect(tmp_path / "clean.sqlite")
    for work in ("W1", "W2", "W3"):
        connection.execute(
            "INSERT INTO paper_nodes(work_id, referenced_works_count, is_nature_target, hop_min) VALUES (?,0,0,0)",
            (work,),
        )
    connection.commit()
    path = tmp_path / "edges.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"citing_work_id": a, "cited_work_id": b}
                for a, b in [
                    ("W1", "W1"),
                    ("W1", "W2"),
                    ("W1", "W2"),
                    ("W1", "W9"),
                    ("bad", "W2"),
                ]
            ]
        ),
        path,
    )
    logger = logging.getLogger("test.paper-index")
    index.import_edges(connection, path, 2, False, logger)
    assert connection.execute("SELECT * FROM paper_edges").fetchall() == [("W1", "W2")]
    assert connection.execute("SELECT COUNT(*) FROM paper_nodes").fetchone()[0] == 3
    assert (
        connection.execute("SELECT COUNT(*) FROM paper_edge_anomalies").fetchone()[0]
        == 4
    )
    index.import_edges(connection, path, 2, True, logger)
    assert (
        connection.execute("SELECT COUNT(*) FROM paper_edge_anomalies").fetchone()[0]
        == 4
    )
    connection.close()


def test_legacy_invalid_index_requires_separate_rebuild(tmp_path: Path) -> None:
    index = _script("06_build_paper_graph_index")
    connection = index.connect(tmp_path / "legacy.sqlite")
    connection.execute("INSERT INTO paper_edges VALUES ('W1','W1')")
    connection.commit()
    with pytest.raises(ValueError, match="separate output"):
        index.import_edges(
            connection,
            tmp_path / "unused.parquet",
            100,
            True,
            logging.getLogger("test"),
        )
    assert connection.execute("SELECT * FROM paper_edges").fetchall() == [("W1", "W1")]
    connection.close()


def test_runtime_excludes_selfloops_from_shared_references_and_paths(
    tmp_path: Path,
) -> None:
    runtime = ClaimGraphRuntime(tmp_path, tmp_path / "unused")
    runtime._paper_db = sqlite3.connect(":memory:")
    runtime._paper_db.execute(
        "CREATE TABLE paper_edges (citing_work_id TEXT, cited_work_id TEXT)"
    )
    runtime._paper_db.executemany(
        "INSERT INTO paper_edges VALUES (?, ?)",
        [("W1", "W1"), ("W1", "W3"), ("W2", "W1"), ("W2", "W1")],
    )
    result = runtime._paper_path(["W1", "W2", "W3"], "W1")
    assert result["direct_citation"] is True
    assert result["shared_reference_count"] == 1
    assert result["two_hop_path_count"] == 1
    assert (
        runtime._paper_db.execute("SELECT COUNT(*) FROM paper_edges").fetchone()[0] == 4
    )
    runtime.close()

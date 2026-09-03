#!/usr/bin/env python3
"""构建历史 Claim Graph：语义边、书目路径注释、语义骨架社区和运行时索引。"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import sqlite3
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "data" / "claim_graph"
EDGE_SCHEMA = pa.schema([
    pa.field("earlier_claim_id", pa.string(), False), pa.field("later_claim_id", pa.string(), False),
    pa.field("earlier_paper_id", pa.string(), False), pa.field("later_paper_id", pa.string(), False),
    pa.field("earlier_claim_type", pa.string(), False), pa.field("later_claim_type", pa.string(), False),
    pa.field("is_cross_type", pa.bool_(), False), pa.field("earlier_publication_date", pa.string(), False),
    pa.field("later_publication_date", pa.string(), False), pa.field("cosine_similarity", pa.float32(), False),
    pa.field("semantic_rank", pa.int32()), pa.field("from_semantic", pa.bool_(), False),
    pa.field("from_paper_path", pa.bool_(), False), pa.field("paper_direct_citation", pa.bool_(), False),
    pa.field("paper_min_path_length", pa.int8()), pa.field("paper_directed_two_hop_count", pa.int32(), False),
    pa.field("paper_two_hop_ra_weight", pa.float32(), False), pa.field("paper_shared_reference_count", pa.int32(), False),
    pa.field("paper_shared_reference_salton", pa.float32(), False),
])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", type=Path, default=ROOT / "claim_nodes.parquet")
    parser.add_argument("--embeddings", type=Path, default=ROOT / "claim_embeddings.parquet")
    parser.add_argument("--canonical-targets", type=Path, default=ROOT / "canonical_target_works.parquet")
    parser.add_argument("--paper-index", type=Path, default=ROOT / "paper_graph_index.sqlite")
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--device", default="cuda", help="语义矩阵计算设备，默认 cuda")
    parser.add_argument("--semantic-top-k", type=int, default=10)
    parser.add_argument("--community-top-k", type=int, default=10)
    parser.add_argument("--matrix-batch-size", type=int, default=128)
    parser.add_argument("--paper-group-size", type=int, default=100)
    parser.add_argument("--stage", choices=("all", "semantic", "paper-path", "merge", "communities", "statistics", "index"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def logger_for(path: Path, verbose: bool) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("claim_graph.phase7")
    logger.handlers.clear(); logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")):
        handler.setFormatter(formatter); handler.setLevel(logging.DEBUG if verbose else logging.INFO); logger.addHandler(handler)
    return logger


def require_files(paths: Iterable[Path]) -> None:
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"输入不存在：{path}")


def state_db(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE IF NOT EXISTS phase_state(stage TEXT PRIMARY KEY, completed INTEGER NOT NULL)")
    connection.commit()
    return connection


def completed(connection: sqlite3.Connection, stage: str) -> bool:
    row = connection.execute("SELECT completed FROM phase_state WHERE stage=?", (stage,)).fetchone()
    return row is not None and bool(row[0])


def mark_completed(connection: sqlite3.Connection, stage: str) -> None:
    connection.execute("INSERT INTO phase_state VALUES (?, 1) ON CONFLICT(stage) DO UPDATE SET completed=1", (stage,)); connection.commit()


def load_claims(path: Path, canonical_path: Path) -> list[dict[str, Any]]:
    canonical = pq.read_table(
        canonical_path,
        columns=["nature_article_id", "publication_date"],
    ).to_pylist()
    article_dates = {
        str(row["nature_article_id"]): str(row["publication_date"])
        for row in canonical
        if row.get("nature_article_id") and row.get("publication_date")
    }
    table = pq.read_table(path, columns=["claim_id", "parent_paper_id", "claim_type", "publication_date"])
    rows = []
    for row in table.to_pylist():
        paper_id = str(row["parent_paper_id"])
        publication_date = article_dates.get(paper_id)
        if publication_date is None:
            raise ValueError(f"Claim {row['claim_id']} 的父论文 {paper_id} 缺少规范 OpenAlex 发表日期")
        rows.append({"claim_id": str(row["claim_id"]), "paper_id": paper_id, "claim_type": str(row["claim_type"]), "date": publication_date})
    if len({row["claim_id"] for row in rows}) != len(rows):
        raise ValueError("claim_nodes.parquet 存在重复 claim_id")
    return rows


def load_vectors(path: Path, claims: list[dict[str, Any]]) -> np.ndarray:
    table = pq.read_table(path, columns=["claim_id", "embedding"])
    if table.num_rows != len(claims):
        raise ValueError(f"embedding 行数={table.num_rows}，Claim 行数={len(claims)}，无法对齐")
    array = table["embedding"].combine_chunks()
    if not isinstance(array, pa.FixedSizeListArray):
        raise ValueError("embedding 必须是 fixed-size float list")
    vectors = np.asarray(array.values.to_numpy(zero_copy_only=False), dtype=np.float32).reshape(table.num_rows, array.type.list_size)
    embedding_ids = [str(value) for value in table["claim_id"].to_pylist()]
    positions = {claim_id: index for index, claim_id in enumerate(embedding_ids)}
    if len(positions) != len(embedding_ids) or set(positions) != {row["claim_id"] for row in claims}:
        raise ValueError("Claim 节点与 embedding 的 claim_id 集合不一致")
    vectors = vectors[np.asarray([positions[row["claim_id"]] for row in claims], dtype=np.int64)]
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(vectors).all() or np.any(norms == 0):
        raise ValueError("embedding 存在非有限值或零向量")
    return vectors / norms[:, None]


def write_table(rows: list[dict[str, Any]], schema: pa.Schema, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), temporary, compression="zstd", use_dictionary=False)
    temporary.replace(path)


def torch_vectors(vectors: np.ndarray, device: str):
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("缺少 torch，无法构建精确语义近邻") from error
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA 但未发现 GPU")
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
    return torch, torch.from_numpy(vectors).to(device)


def semantic_edges(claims: list[dict[str, Any]], vectors: np.ndarray, args: argparse.Namespace, output: Path, logger: logging.Logger) -> None:
    torch, matrix = torch_vectors(vectors, args.device)
    order = sorted(range(len(claims)), key=lambda index: (claims[index]["date"], claims[index]["claim_id"]))
    dates = [claims[index]["date"] for index in order]
    records: list[dict[str, Any]] = []
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and dates[end] == dates[start]:
            end += 1
        if start:
            history = order[:start]
            history_tensor = matrix[history]
            history_papers = np.asarray([claims[index]["paper_id"] for index in history], dtype=object)
            take = min(args.semantic_top_k, len(history))
            for batch_start in range(start, end, args.matrix_batch_size):
                current = order[batch_start:min(batch_start + args.matrix_batch_size, end)]
                scores = matrix[current] @ history_tensor.T
                for row_index, claim_index in enumerate(current):
                    same_paper = np.flatnonzero(history_papers == claims[claim_index]["paper_id"])
                    if len(same_paper):
                        scores[row_index, torch.as_tensor(same_paper, device=scores.device)] = -torch.inf
                values, positions = torch.topk(scores, k=take, dim=1)
                for row_index, later_index in enumerate(current):
                    for rank, (score, position) in enumerate(zip(values[row_index].tolist(), positions[row_index].tolist(), strict=True), start=1):
                        if not math.isfinite(score):
                            continue
                        earlier_index = history[position]
                        records.append(edge_record(claims, earlier_index, later_index, float(score), rank, True, False, {}))
        start = end
        logger.info("[语义边] 日期进度=%d/%d，当前边=%d", end, len(order), len(records))
    write_table(records, EDGE_SCHEMA, output)
    save_embedding_assets(vectors, claims, output.parent, logger)


def save_embedding_assets(vectors: np.ndarray, claims: list[dict[str, Any]], root: Path, logger: logging.Logger) -> None:
    np.save(root / "claim_embedding_matrix.npy", vectors)
    mapping = pa.Table.from_pylist([{"claim_id": row["claim_id"], "claim_row": index} for index, row in enumerate(claims)])
    pq.write_table(mapping, root / "claim_embedding_index.parquet", compression="zstd")
    try:
        import faiss
    except ImportError as error:
        raise RuntimeError("缺少 faiss；请安装 faiss-cpu 后重新运行 Phase 7") from error
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(vectors.shape[1]))
    index.add_with_ids(vectors, np.arange(len(vectors), dtype=np.int64))
    faiss.write_index(index, str(root / "claim_semantic_index.faiss"))
    logger.info("[语义边] 已写入精确 FAISS IndexFlatIP")


def edge_record(claims: list[dict[str, Any]], earlier: int, later: int, cosine: float, rank: int | None, semantic: bool, paper: bool, motif: dict[str, Any]) -> dict[str, Any]:
    first, second = claims[earlier], claims[later]
    return {
        "earlier_claim_id": first["claim_id"], "later_claim_id": second["claim_id"], "earlier_paper_id": first["paper_id"], "later_paper_id": second["paper_id"],
        "earlier_claim_type": first["claim_type"], "later_claim_type": second["claim_type"], "is_cross_type": first["claim_type"] != second["claim_type"],
        "earlier_publication_date": first["date"], "later_publication_date": second["date"], "cosine_similarity": cosine, "semantic_rank": rank,
        "from_semantic": semantic, "from_paper_path": paper, "paper_direct_citation": bool(motif.get("direct", False)),
        "paper_min_path_length": motif.get("min_path"), "paper_directed_two_hop_count": int(motif.get("two_count", 0)),
        "paper_two_hop_ra_weight": float(motif.get("two_ra", 0.0)), "paper_shared_reference_count": int(motif.get("shared_count", 0)),
        "paper_shared_reference_salton": float(motif.get("shared_salton", 0.0)),
    }


def motifs(connection: sqlite3.Connection, later_work: str, later_date: str, later_refs: int) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = defaultdict(lambda: {"direct": False, "min_path": None, "two_count": 0, "two_ra": 0.0, "shared_count": 0, "shared_salton": 0.0})
    direct_sql = "SELECT e.cited_work_id FROM paper_edges e JOIN paper_nodes p ON p.work_id=e.cited_work_id WHERE e.citing_work_id=? AND p.is_nature_target=1 AND p.publication_date<?"
    for (work_id,) in connection.execute(direct_sql, (later_work, later_date)):
        result[work_id].update({"direct": True, "min_path": 1})
    two_sql = "SELECT e2.cited_work_id, COUNT(*), SUM(1.0 / MAX(x.referenced_works_count, 1)) FROM paper_edges e1 JOIN paper_edges e2 ON e1.cited_work_id=e2.citing_work_id JOIN paper_nodes p ON p.work_id=e2.cited_work_id JOIN paper_nodes x ON x.work_id=e1.cited_work_id WHERE e1.citing_work_id=? AND p.is_nature_target=1 AND p.publication_date<? GROUP BY e2.cited_work_id"
    for work_id, count, ra in connection.execute(two_sql, (later_work, later_date)):
        item = result[work_id]; item["two_count"] = int(count); item["two_ra"] = float(ra or 0.0); item["min_path"] = min(item["min_path"] or 2, 2)
    shared_sql = "SELECT e2.citing_work_id, COUNT(*), p.referenced_works_count FROM paper_edges e1 JOIN paper_edges e2 ON e1.cited_work_id=e2.cited_work_id JOIN paper_nodes p ON p.work_id=e2.citing_work_id WHERE e1.citing_work_id=? AND p.is_nature_target=1 AND p.publication_date<? GROUP BY e2.citing_work_id"
    for work_id, count, earlier_refs in connection.execute(shared_sql, (later_work, later_date)):
        item = result[work_id]; item["shared_count"] = int(count); item["shared_salton"] = float(count) / math.sqrt(max(later_refs, 1) * max(int(earlier_refs or 0), 1))
    return result


def paper_path_edges(canonical: Path, index_path: Path, root: Path, args: argparse.Namespace, logger: logging.Logger) -> None:
    """只为既有语义边补充父论文路径证据，不创建纯 Paper-path Claim 边。"""
    targets = pq.read_table(canonical, columns=["work_id", "nature_article_id", "publication_date", "referenced_works_count"]).to_pylist()
    article_to_work = {str(row["nature_article_id"]): str(row["work_id"]) for row in targets if row.get("nature_article_id")}
    semantic_rows = pq.read_table(root / "semantic_claim_edges.parquet", schema=EDGE_SCHEMA).to_pylist()
    by_later_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in semantic_rows:
        by_later_paper[str(row["later_paper_id"])].append(row)
    target_articles = sorted(by_later_paper)
    chunks = root / "paper_path_chunks"; chunks.mkdir(exist_ok=True)
    connection = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    try:
        for group_start in range(0, len(target_articles), args.paper_group_size):
            group_number = group_start // args.paper_group_size
            chunk = chunks / f"{group_number:05d}.parquet"
            if args.resume and chunk.exists():
                continue
            records: list[dict[str, Any]] = []
            for article in target_articles[group_start:group_start + args.paper_group_size]:
                if article not in article_to_work:
                    raise ValueError(f"语义边父论文 {article} 无法映射 OpenAlex Work ID")
                later_work = article_to_work[article]
                row = connection.execute("SELECT referenced_works_count FROM paper_nodes WHERE work_id=?", (later_work,)).fetchone()
                later_date = str(by_later_paper[article][0]["later_publication_date"])
                paper_motifs = motifs(connection, later_work, later_date, int(row[0] if row else 0))
                for semantic_row in by_later_paper[article]:
                    earlier_article = str(semantic_row["earlier_paper_id"])
                    earlier_work = article_to_work.get(earlier_article)
                    if earlier_work is None or earlier_work not in paper_motifs:
                        continue
                    motif = paper_motifs[earlier_work]
                    enriched = dict(semantic_row)
                    enriched.update({
                        "from_semantic": True,
                        "from_paper_path": True,
                        "paper_direct_citation": bool(motif.get("direct", False)),
                        "paper_min_path_length": motif.get("min_path"),
                        "paper_directed_two_hop_count": int(motif.get("two_count", 0)),
                        "paper_two_hop_ra_weight": float(motif.get("two_ra", 0.0)),
                        "paper_shared_reference_count": int(motif.get("shared_count", 0)),
                        "paper_shared_reference_salton": float(motif.get("shared_salton", 0.0)),
                    })
                    records.append(enriched)
            write_table(records, EDGE_SCHEMA, chunk)
            logger.info("[Paper-path] 分组=%d/%d，边=%d", group_number + 1, math.ceil(len(target_articles) / args.paper_group_size), len(records))
    finally:
        connection.close()
    writer: pq.ParquetWriter | None = None; temporary = root / "paper_path_claim_edges.parquet.tmp"
    try:
        for chunk in sorted(chunks.glob("*.parquet")):
            table = pq.read_table(chunk, schema=EDGE_SCHEMA)
            if writer is None: writer = pq.ParquetWriter(temporary, EDGE_SCHEMA, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None: writer.close()
    if writer is None: pq.write_table(pa.Table.from_pylist([], schema=EDGE_SCHEMA), temporary, compression="zstd")
    temporary.replace(root / "paper_path_claim_edges.parquet")


def graph_db(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL"); connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("CREATE TABLE IF NOT EXISTS claim_nodes (claim_row INTEGER PRIMARY KEY, claim_id TEXT UNIQUE, parent_paper_id TEXT, publication_date TEXT, claim_type TEXT, community_id INTEGER)")
    connection.execute("CREATE TABLE IF NOT EXISTS claim_edges (earlier_row INTEGER, later_row INTEGER, cosine REAL, semantic_rank INTEGER, from_semantic INTEGER, from_paper_path INTEGER, direct INTEGER, min_path INTEGER, two_count INTEGER, two_ra REAL, shared_count INTEGER, shared_salton REAL, PRIMARY KEY(earlier_row,later_row)) WITHOUT ROWID")
    connection.execute("CREATE TABLE IF NOT EXISTS semantic_backbone_edges (claim_row_a INTEGER, claim_row_b INTEGER, cosine REAL, PRIMARY KEY(claim_row_a,claim_row_b)) WITHOUT ROWID")
    connection.execute("CREATE TABLE IF NOT EXISTS semantic_backbone_adjacency (source_row INTEGER, target_row INTEGER, cosine REAL, PRIMARY KEY(source_row,target_row)) WITHOUT ROWID")
    connection.commit(); return connection


def merge_edges(claims: list[dict[str, Any]], root: Path, logger: logging.Logger) -> None:
    database = root / "claim_graph_index.sqlite"
    if database.exists(): database.unlink()
    connection = graph_db(database)
    mapping = {row["claim_id"]: index for index, row in enumerate(claims)}
    with connection:
        connection.executemany("INSERT INTO claim_nodes(claim_row,claim_id,parent_paper_id,publication_date,claim_type) VALUES (?, ?, ?, ?, ?)", [(index, row["claim_id"], row["paper_id"], row["date"], row["claim_type"]) for index, row in enumerate(claims)])
    for source in (root / "semantic_claim_edges.parquet", root / "paper_path_claim_edges.parquet"):
        parquet = pq.ParquetFile(source)
        for batch in parquet.iter_batches(batch_size=50_000):
            values = []
            for row in batch.to_pylist():
                values.append((mapping[row["earlier_claim_id"]], mapping[row["later_claim_id"]], float(row["cosine_similarity"]), row.get("semantic_rank"), int(bool(row["from_semantic"])), int(bool(row["from_paper_path"])), int(bool(row["paper_direct_citation"])), row.get("paper_min_path_length"), int(row["paper_directed_two_hop_count"]), float(row["paper_two_hop_ra_weight"]), int(row["paper_shared_reference_count"]), float(row["paper_shared_reference_salton"])))
            with connection:
                connection.executemany("INSERT INTO claim_edges VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(earlier_row,later_row) DO UPDATE SET cosine=MAX(cosine,excluded.cosine), semantic_rank=COALESCE(claim_edges.semantic_rank,excluded.semantic_rank), from_semantic=MAX(from_semantic,excluded.from_semantic), from_paper_path=MAX(from_paper_path,excluded.from_paper_path), direct=MAX(direct,excluded.direct), min_path=MIN(COALESCE(min_path,99),COALESCE(excluded.min_path,99)), two_count=MAX(two_count,excluded.two_count), two_ra=MAX(two_ra,excluded.two_ra), shared_count=MAX(shared_count,excluded.shared_count), shared_salton=MAX(shared_salton,excluded.shared_salton)", values)
    connection.execute("CREATE INDEX claim_edges_by_later ON claim_edges(later_row, earlier_row)"); connection.execute("CREATE INDEX claim_edges_by_paper ON claim_edges(from_paper_path, later_row)"); connection.commit()
    cursor = connection.execute("SELECT n1.claim_id,n2.claim_id,n1.parent_paper_id,n2.parent_paper_id,n1.claim_type,n2.claim_type,n1.publication_date,n2.publication_date,e.cosine,e.semantic_rank,e.from_semantic,e.from_paper_path,e.direct,e.min_path,e.two_count,e.two_ra,e.shared_count,e.shared_salton FROM claim_edges e JOIN claim_nodes n1 ON n1.claim_row=e.earlier_row JOIN claim_nodes n2 ON n2.claim_row=e.later_row ORDER BY e.earlier_row,e.later_row")
    temporary = root / "claim_edges.parquet.tmp"; writer = pq.ParquetWriter(temporary, EDGE_SCHEMA, compression="zstd"); count = 0
    try:
        while True:
            rows = cursor.fetchmany(50_000)
            if not rows:
                break
            records = [{"earlier_claim_id": row[0], "later_claim_id": row[1], "earlier_paper_id": row[2], "later_paper_id": row[3], "earlier_claim_type": row[4], "later_claim_type": row[5], "is_cross_type": row[4] != row[5], "earlier_publication_date": row[6], "later_publication_date": row[7], "cosine_similarity": row[8], "semantic_rank": row[9], "from_semantic": bool(row[10]), "from_paper_path": bool(row[11]), "paper_direct_citation": bool(row[12]), "paper_min_path_length": None if row[13] == 99 else row[13], "paper_directed_two_hop_count": row[14], "paper_two_hop_ra_weight": row[15], "paper_shared_reference_count": row[16], "paper_shared_reference_salton": row[17]} for row in rows]
            writer.write_table(pa.Table.from_pylist(records, schema=EDGE_SCHEMA)); count += len(records)
    finally:
        writer.close()
    temporary.replace(root / "claim_edges.parquet"); connection.close(); logger.info("[合并] 正式 Claim 边=%d", count)


def communities(claims: list[dict[str, Any]], vectors: np.ndarray, args: argparse.Namespace, root: Path, logger: logging.Logger) -> None:
    try:
        import igraph as ig
        import leidenalg
    except ImportError as error:
        raise RuntimeError("缺少 igraph 或 leidenalg；请安装 python-igraph leidenalg") from error
    torch, matrix = torch_vectors(vectors, args.device); count = len(claims); neighbors = [set() for _ in range(count)]
    papers = np.asarray([row["paper_id"] for row in claims], dtype=object)
    for start in range(0, count, args.matrix_batch_size):
        current = list(range(start, min(start + args.matrix_batch_size, count))); scores = matrix[current] @ matrix.T
        for local, index in enumerate(current): scores[local, torch.as_tensor(np.flatnonzero(papers == papers[index]), device=scores.device)] = -torch.inf
        positions = torch.topk(scores, k=min(args.community_top_k, count - 1), dim=1).indices.tolist()
        for index, values in zip(current, positions, strict=True): neighbors[index].update(values)
        logger.info("[社区骨架] 进度=%d/%d", min(start + args.matrix_batch_size, count), count)
    edges = [(left, right) for left in range(count) for right in neighbors[left] if left < right and left in neighbors[right]]
    weights = [float(np.dot(vectors[left], vectors[right])) for left, right in edges]
    connected_nodes = sorted({node for edge in edges for node in edge})
    if not connected_nodes:
        raise RuntimeError("semantic mutual-kNN backbone 没有产生任何边")
    compact_row = {claim_row: index for index, claim_row in enumerate(connected_nodes)}
    compact_edges = [(compact_row[left], compact_row[right]) for left, right in edges]
    graph = ig.Graph(n=len(connected_nodes), edges=compact_edges, directed=False); graph.es["weight"] = weights
    partition = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition, weights="weight", seed=0)
    membership: list[int | None] = [None] * count
    for compact_index, claim_row in enumerate(connected_nodes):
        membership[claim_row] = int(partition.membership[compact_index])
    backbone = [{"claim_id_a": claims[left]["claim_id"], "claim_id_b": claims[right]["claim_id"], "cosine_similarity": weight} for (left, right), weight in zip(edges, weights, strict=True)]
    write_table(backbone, pa.schema([pa.field("claim_id_a", pa.string()), pa.field("claim_id_b", pa.string()), pa.field("cosine_similarity", pa.float32())]), root / "claim_backbone_edges.parquet")
    community_rows = [{"claim_id": row["claim_id"], "community_id": membership[index]} for index, row in enumerate(claims)]
    write_table(community_rows, pa.schema([pa.field("claim_id", pa.string()), pa.field("community_id", pa.int32())]), root / "claim_communities.parquet")
    profile_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for index, row in enumerate(claims):
        community_id = membership[index]
        if community_id is not None:
            profile_counts[community_id][row["claim_type"]] += 1
    profiles = [{"community_id": community, "claim_count": sum(counts.values()), "claim_type_counts_json": json.dumps(dict(sorted(counts.items())), ensure_ascii=False)} for community, counts in sorted(profile_counts.items())]
    write_table(profiles, pa.schema([pa.field("community_id", pa.int32()), pa.field("claim_count", pa.int32()), pa.field("claim_type_counts_json", pa.string())]), root / "community_profiles.parquet")
    connection = sqlite3.connect(root / "claim_graph_index.sqlite")
    with connection:
        connection.execute("UPDATE claim_nodes SET community_id=NULL")
        connection.execute("DELETE FROM semantic_backbone_edges")
        connection.execute("DELETE FROM semantic_backbone_adjacency")
        connection.executemany(
            "INSERT INTO semantic_backbone_edges VALUES (?, ?, ?)",
            [(left, right, weight) for (left, right), weight in zip(edges, weights, strict=True)],
        )
        adjacency_rows = [
            directed
            for (left, right), weight in zip(edges, weights, strict=True)
            for directed in ((left, right, weight), (right, left, weight))
        ]
        connection.executemany(
            "INSERT INTO semantic_backbone_adjacency VALUES (?, ?, ?)",
            adjacency_rows,
        )
        connection.execute("CREATE INDEX IF NOT EXISTS semantic_backbone_by_b ON semantic_backbone_edges(claim_row_b, claim_row_a)")
        connection.execute("CREATE INDEX IF NOT EXISTS semantic_backbone_adjacency_by_target ON semantic_backbone_adjacency(target_row, source_row)")
        connection.executemany("UPDATE claim_nodes SET community_id=? WHERE claim_id=?", [(membership[index], row["claim_id"]) for index, row in enumerate(claims)])
    connection.close(); logger.info("[社区] backbone边=%d，参与节点=%d，孤立节点=%d，社区=%d", len(edges), len(connected_nodes), count-len(connected_nodes), len(set(value for value in membership if value is not None)))


def statistics(root: Path, logger: logging.Logger) -> None:
    connection = sqlite3.connect(root / "claim_graph_index.sqlite")
    groups: dict[int, list[tuple[int, float, int, int]]] = defaultdict(list)
    query = "SELECT e.later_row,n.community_id,e.cosine,e.from_paper_path,e.from_semantic FROM claim_edges e JOIN claim_nodes n ON n.claim_row=e.earlier_row WHERE e.cosine>0 AND n.community_id IS NOT NULL ORDER BY e.later_row"
    for later, community, cosine, paper, semantic in connection.execute(query): groups[int(later)].append((int(community), float(cosine), int(paper), int(semantic)))
    profiles=[]; pair_counts: dict[tuple[int,int], int] = defaultdict(int)
    for later, values in groups.items():
        weights: dict[int,float] = defaultdict(float)
        for community, cosine, _, _ in values: weights[community] += cosine
        total=sum(weights.values()); probabilities=[weight/total for weight in weights.values()] if total else []
        communities_here=sorted(weights)
        for pair in combinations(communities_here,2): pair_counts[pair]+=1
        semantic_values=[value[1] for value in values if value[3]]
        profiles.append({"claim_id": connection.execute("SELECT claim_id FROM claim_nodes WHERE claim_row=?", (later,)).fetchone()[0], "nearest_prior_similarity": max(semantic_values) if semantic_values else None, "community_variety": len(communities_here), "effective_community_count": 1.0/sum(p*p for p in probabilities) if probabilities else None, "paper_path_neighbor_count": sum(value[2] for value in values), "dual_source_edge_count": sum(value[2] and value[3] for value in values)})
    write_table(profiles, pa.schema([pa.field("claim_id",pa.string()),pa.field("nearest_prior_similarity",pa.float32()),pa.field("community_variety",pa.int32()),pa.field("effective_community_count",pa.float32()),pa.field("paper_path_neighbor_count",pa.int32()),pa.field("dual_source_edge_count",pa.int32())]), root / "historical_insertion_profiles.parquet")
    write_table([{"community_a":a,"community_b":b,"observed_claim_count":count} for (a,b),count in pair_counts.items()], pa.schema([pa.field("community_a",pa.int32()),pa.field("community_b",pa.int32()),pa.field("observed_claim_count",pa.int32())]), root / "community_pair_history.parquet")
    type_rows = connection.execute("SELECT a.claim_type,b.claim_type,COUNT(*),AVG(e.cosine),SUM(e.from_semantic),SUM(e.from_paper_path) FROM claim_edges e JOIN claim_nodes a ON a.claim_row=e.earlier_row JOIN claim_nodes b ON b.claim_row=e.later_row GROUP BY a.claim_type,b.claim_type").fetchall()
    present = {(row[0], row[1]): row for row in type_rows}
    types = ("METHOD", "FINDING", "MECHANISM", "RESOURCE", "THEORY")
    type_stats = []
    for earlier_type in types:
        for later_type in types:
            row = present.get((earlier_type, later_type), (earlier_type, later_type, 0, None, 0, 0))
            type_stats.append({"earlier_claim_type": earlier_type, "later_claim_type": later_type, "edge_count": int(row[2]), "mean_cosine_similarity": row[3], "semantic_edge_count": int(row[4]), "paper_path_edge_count": int(row[5])})
    write_table(type_stats, pa.schema([pa.field("earlier_claim_type",pa.string()),pa.field("later_claim_type",pa.string()),pa.field("edge_count",pa.int64()),pa.field("mean_cosine_similarity",pa.float32()),pa.field("semantic_edge_count",pa.int64()),pa.field("paper_path_edge_count",pa.int64())]), root / "claim_edge_type_stats.parquet")
    connection.execute("CREATE INDEX IF NOT EXISTS claim_nodes_by_date ON claim_nodes(publication_date)"); connection.execute("ANALYZE"); connection.commit(); connection.close(); logger.info("[统计] 历史画像=%d，社区对=%d",len(profiles),len(pair_counts))


def run_stage(name: str, args: argparse.Namespace, state: sqlite3.Connection, action, logger: logging.Logger) -> None:
    if args.stage not in ("all", name): return
    if args.resume and completed(state, name): logger.info("[%s] 已完成，跳过", name); return
    logger.info("[%s] 开始", name); started=time.monotonic(); action(); mark_completed(state,name); logger.info("[%s] 完成，耗时=%.1fs",name,time.monotonic()-started)


def main() -> int:
    args=parse_args()
    if args.resume and args.restart: raise ValueError("--resume 与 --restart 不能同时使用")
    if min(args.semantic_top_k,args.community_top_k,args.matrix_batch_size,args.paper_group_size)<=0: raise ValueError("所有规模参数必须为正数")
    require_files([args.claims,args.embeddings,args.canonical_targets,args.paper_index])
    root=args.output_root.resolve(); root.mkdir(parents=True,exist_ok=True); logger=logger_for(root/"logs"/"phase7_build_claim_graph.log",args.verbose)
    state_path=root/"phase7_build_state.sqlite"
    if args.restart:
        for path in (state_path,state_path.with_name(state_path.name+"-wal"),state_path.with_name(state_path.name+"-shm"),root/"semantic_claim_edges.parquet",root/"paper_path_claim_edges.parquet",root/"claim_edges.parquet",root/"claim_edge_type_stats.parquet",root/"claim_backbone_edges.parquet",root/"claim_communities.parquet",root/"community_profiles.parquet",root/"community_pair_history.parquet",root/"historical_insertion_profiles.parquet",root/"claim_graph_index.sqlite",root/"claim_graph_index.sqlite-wal",root/"claim_graph_index.sqlite-shm",root/"claim_embedding_matrix.npy",root/"claim_embedding_index.parquet",root/"claim_semantic_index.faiss"):
            path.unlink(missing_ok=True)
        shutil.rmtree(root/"paper_path_chunks",ignore_errors=True)
    elif state_path.exists() and not args.resume: raise FileExistsError("Phase 7 已有状态；继续请传 --resume，重建请传 --restart")
    claims=load_claims(args.claims,args.canonical_targets); vectors=load_vectors(args.embeddings,claims); state=state_db(state_path)
    try:
        run_stage("semantic",args,state,lambda:semantic_edges(claims,vectors,args,root/"semantic_claim_edges.parquet",logger),logger)
        run_stage("paper-path",args,state,lambda:paper_path_edges(args.canonical_targets,args.paper_index,root,args,logger),logger)
        run_stage("merge",args,state,lambda:merge_edges(claims,root,logger),logger)
        run_stage("communities",args,state,lambda:communities(claims,vectors,args,root,logger),logger)
        run_stage("statistics",args,state,lambda:statistics(root,logger),logger)
        run_stage("index",args,state,lambda:logger.info("运行时 SQLite 与 FAISS 已由前序步骤生成"),logger)
    finally: state.close()
    return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except (FileNotFoundError,FileExistsError,RuntimeError,ValueError,OSError,sqlite3.Error,pa.ArrowException) as error:
        print(f"Claim Graph 构建失败：{error}",file=sys.stderr); raise SystemExit(1) from error

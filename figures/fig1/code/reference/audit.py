"""Independent full-source numerical audit: python -m figure_pipeline.fig1_reference.audit."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

import duckdb
import networkx as nx
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .data import ASSETS, ROLES, ROOT, read, source, write


class Audit:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def check(self, name: str, observed: Any, expected: Any) -> None:
        passed = (
            math.isclose(observed, expected, rel_tol=1e-6, abs_tol=1e-8)
            if isinstance(observed, float) and isinstance(expected, (int, float))
            else observed == expected
        )
        self.rows.append(
            {
                "check": name,
                "observed": observed,
                "expected": expected,
                "passed": bool(passed),
            }
        )


def paper_counts(audit: Audit, stats: dict[str, Any]) -> None:
    """Exact fresh scan, including pair deduplication; no saved aggregate reused."""
    with duckdb.connect() as db:
        db.execute("SET memory_limit='6GB'")
        db.execute("SET threads=4")
        db.from_parquet(str(ASSETS / "paper_nodes.parquet")).create_view("papers")
        db.from_parquet(str(ASSETS / "paper_edges.parquet")).create_view("citations")
        print("Auditing 13.9M paper identities and 64M raw citation rows…", flush=True)
        rows, distinct, missing = db.execute(
            "SELECT count(*), count(DISTINCT work_id), count(*) FILTER (WHERE work_id IS NULL OR work_id='') FROM papers"
        ).fetchone()
        audit.check("c.paper_raw_nodes", rows, stats["paper_nodes"])
        audit.check("c.paper_distinct_nodes", distinct, stats["paper_nodes"])
        audit.check("c.paper_missing_ids", missing, 0)
        raw, nonself, clean, missing = db.execute(
            "SELECT count(*), count(*) FILTER (WHERE citing_work_id!=cited_work_id), "
            "count(DISTINCT (citing_work_id,cited_work_id)) FILTER (WHERE citing_work_id!=cited_work_id), "
            "count(*) FILTER (WHERE citing_work_id IS NULL OR cited_work_id IS NULL) FROM citations"
        ).fetchone()
        audit.check("c.paper_raw_edges", raw, stats["paper_raw_edges"])
        audit.check(
            "c.paper_clean_distinct_nonself_edges", clean, stats["paper_clean_edges"]
        )
        audit.check("c.paper_duplicate_nonself_rows", nonself - clean, 0)
        audit.check("c.paper_missing_endpoints", missing, 0)
        audit.check(
            "c.paper_self_loops_excluded",
            raw - nonself,
            stats["paper_raw_edges"] - stats["paper_clean_edges"],
        )


def historical_counts(audit: Audit, snapshot: dict[str, Any]) -> None:
    stats = snapshot["statistics"]
    nodes = pd.read_parquet(
        ASSETS / "claim_nodes.parquet",
        columns=["claim_id", "parent_paper_id", "claim_type", "publication_date"],
    )
    assignment = pd.read_parquet(ASSETS / "claim_communities.parquet")
    frame = nodes.merge(assignment, on="claim_id", how="left", validate="one_to_one")
    audit.check("c.claim_count", len(nodes), stats["claims"])
    audit.check("c.unique_claim_ids", nodes.claim_id.nunique(), len(nodes))
    audit.check("c.claim_roles", dict(Counter(nodes.claim_type)), stats["roles"])
    targets = pq.ParquetFile(ASSETS / "nature_targets.parquet").metadata.num_rows
    parents = Counter(nodes.parent_paper_id)
    distribution = dict(Counter(parents.values()))
    distribution[0] = targets - len(parents)
    audit.check("c.historical_target_papers", targets, stats["target_papers"])
    audit.check("c.papers_with_claims", len(parents), stats["papers_with_claims"])
    audit.check(
        "c.claims_per_paper",
        distribution,
        {int(k): v for k, v in stats["claims_per_paper"].items()},
    )
    groups = frame.dropna(subset=["community_id"]).groupby("community_id")
    sizes = groups.size()
    audit.check("d.community_count", len(sizes), stats["community_count"])
    audit.check(
        "d.community_histogram",
        dict(Counter(sizes.tolist())),
        {int(k): v for k, v in stats["community_size_histogram"].items()},
    )
    audit.check(
        "d.unassigned_claims", int(frame.community_id.isna().sum()), stats["unassigned"]
    )
    audit.check("d.printed_median_size", float(sizes.median()), 2)
    audit.check("d.printed_maximum_size", int(sizes.max()), 2544)
    audit.check(
        "d.printed_unassigned_percent",
        round(stats["unassigned"] / len(nodes) * 100, 1),
        21.2,
    )
    for row in stats["community_ccdf"]:
        audit.check(
            f"d.ccdf_at_{row['size']}",
            float((sizes >= row["size"]).mean()),
            row["share_at_least_size"],
        )
    for community in stats["communities"]:
        group = groups.get_group(community["community_id"])
        audit.check(f"d.{community['display']}.size", len(group), community["size"])
        audit.check(
            f"d.{community['display']}.roles",
            dict(Counter(group.claim_type)),
            community["role_counts"],
        )
    edge_counts(audit, snapshot, frame)


def edge_counts(audit: Audit, snapshot: dict[str, Any], frame: pd.DataFrame) -> None:
    stats = snapshot["statistics"]
    native = frame.set_index("claim_id")
    semantic = pd.read_parquet(ASSETS / "semantic_claim_edges.parquet")
    audit.check("e.semantic_edge_rows", len(semantic), stats["semantic_edges"])
    audit.check(
        "e.semantic_unique_pairs",
        len(semantic.drop_duplicates(["earlier_claim_id", "later_claim_id"])),
        len(semantic),
    )
    earlier = semantic.earlier_claim_id.map(native.claim_type)
    later = semantic.later_claim_id.map(native.claim_type)
    audit.check(
        "e.earlier_role_metadata_matches_nodes",
        bool((earlier == semantic.earlier_claim_type).all()),
        True,
    )
    audit.check(
        "e.later_role_metadata_matches_nodes",
        bool((later == semantic.later_claim_type).all()),
        True,
    )
    dates = native.publication_date.astype(str).str[:10]
    audit.check(
        "e.role_matrix_temporal_orientation",
        bool(
            (
                semantic.earlier_claim_id.map(dates)
                <= semantic.later_claim_id.map(dates)
            ).all()
        ),
        True,
    )
    counts = Counter(zip(earlier, later))
    matrix = np.array([[counts[(a, b)] for b in ROLES] for a in ROLES])
    audit.check(
        "e.role_matrix_from_node_roles", matrix.tolist(), stats["role_matrix_counts"]
    )
    audit.check(
        "e.column_normalization_from_raw_counts",
        bool(
            np.allclose(
                matrix / matrix.sum(axis=0), stats["role_matrix_column_normalized"]
            )
        ),
        True,
    )
    path_rows = pq.ParquetFile(
        ASSETS / "paper_path_claim_edges.parquet"
    ).metadata.num_rows
    audit.check(
        "e.saved_path_annotation_rows", path_rows, stats["saved_paper_path_annotations"]
    )
    backbone = pd.read_parquet(ASSETS / "claim_backbone_edges.parquet")
    edges = {
        tuple(sorted(pair))
        for pair in backbone[["claim_id_a", "claim_id_b"]].itertuples(
            index=False, name=None
        )
    }
    audit.check(
        "e.backbone_unique_undirected_edges", len(edges), stats["backbone_edges"]
    )
    focus = snapshot["atlas"]["focus_communities"]
    mixing = np.zeros((len(focus), len(focus)), dtype=int)
    communities = native.community_id.to_dict()
    for a, b in edges:
        ca, cb = communities[a], communities[b]
        if ca in focus and cb in focus:
            i, j = focus.index(ca), focus.index(cb)
            mixing[i, j] += 1
            if i != j:
                mixing[j, i] += 1
    audit.check(
        "d.community_mixing_native_deduplicated",
        mixing.tolist(),
        stats["community_mixing"],
    )


def cases_and_display(audit: Audit, snapshot: dict[str, Any], out: Path) -> None:
    index = pd.read_parquet(ASSETS / "community_centroid_index.parquet")
    rows = dict(zip(index.community_id.astype(int), index.centroid_row.astype(int)))
    centroids = np.load(ASSETS / "community_centroid_matrix.npy", mmap_mode="r")
    for case in snapshot["cases"]:
        label, metrics = case["label"], case["metrics"]
        neighbors = case["neighbors"]
        weights: Counter[int] = Counter()
        for node in neighbors:
            if node["community_id"] is not None:
                weights[int(node["community_id"])] += node["cosine_similarity"]
        probabilities = np.array(list(weights.values())) / sum(weights.values())
        vectors = np.array([centroids[rows[k]] for k in weights], dtype=float)
        distance = np.clip(1 - vectors @ vectors.T, 0, 1)
        np.fill_diagonal(distance, 0)
        audit.check(
            f"b.{label}.centroids_normalized",
            bool(np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)),
            True,
        )
        audit.check(
            f"b.{label}.nearest_similarity",
            max(n["cosine_similarity"] for n in neighbors),
            metrics["nearest_prior_similarity"],
        )
        audit.check(
            f"b.{label}.effective_communities",
            float(1 / (probabilities @ probabilities)),
            metrics["effective_community_count"],
        )
        audit.check(
            f"b.{label}.rao_stirling",
            float(probabilities @ distance @ probabilities),
            metrics["community_rao_stirling"],
        )
        audit.check(
            f"b.{label}.cross_role_share",
            sum(n["claim_type"] != case["claim"]["claim_type"] for n in neighbors)
            / len(neighbors),
            metrics["cross_type_neighbor_share"],
        )
        graph = nx.Graph()
        graph.add_nodes_from(n["claim_id"] for n in neighbors)
        graph.add_edges_from(case["edges"])
        keys = list(graph)
        new_pairs = sum(
            not nx.has_path(graph, a, b)
            for i, a in enumerate(keys)
            for b in keys[i + 1 :]
        )
        audit.check(
            f"b.{label}.new_pairs_independent_reachability",
            new_pairs,
            metrics["newly_connected_neighbor_pair_count"],
        )
        audit.check(
            f"b.{label}.new_pair_fraction",
            new_pairs / math.comb(len(keys), 2),
            metrics["connected_pair_share"],
        )
        for record in case["sources"]:
            audit.check(
                f"b.{label}.source_hash.{Path(record['path']).name}",
                source(Path(record["path"]))["sha256"],
                record["sha256"],
            )
    layer = snapshot["atlas"]["paper_layer"]
    graph = nx.DiGraph(layer["edges"])
    for path in layer["connecting_paths"]:
        audit.check(
            "a.preserved_connector." + path[0] + "." + path[-1],
            all(
                graph.has_edge(a, b) or graph.has_edge(b, a) for a, b in pairwise(path)
            ),
            True,
        )
    graph.add_nodes_from(p["work_id"] for p in layer["nodes"])
    audit.check("a.paper_display_connected", nx.is_weakly_connected(graph), True)
    audit.check("a.paper_display_isolated_nodes", len(list(nx.isolates(graph))), 0)
    audit.check("a.paper_display_budget", len(layer["nodes"]), 90)
    svg = (out / "final/Fig1.svg").read_text()
    audit.check(
        "a.semantic_and_citation_description",
        "Semantic relations and" in svg and "citation context" in svg,
        True,
    )
    audit.check(
        "b.grey_case_footers_removed",
        all(f"b_{label}_provenance" not in svg for label in "ABCD"),
        True,
    )


def audit_all(out: Path) -> bool:
    snapshot = read(out / "data/snapshot.json")
    audit = Audit()
    historical_counts(audit, snapshot)
    cases_and_display(audit, snapshot, out)
    paper_counts(audit, snapshot["statistics"])
    failures = [r["check"] for r in audit.rows if not r["passed"]]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "all_passed": not failures,
        "failures": failures,
        "checks": audit.rows,
        "scope": "Exact numerical and provenance audit of the displayed figure against local sources; does not independently validate scientific conclusions or correctness of bibliographic records.",
    }
    write(out / "qa/data_audit.json", report)
    lines = [
        "# Fig.1 全图数据复核",
        "",
        f"检查时间：{report['generated_at_utc']}。通过 {len(audit.rows) - len(failures)}/{len(audit.rows)} 项。",
        "",
        "Paper Graph 节点和清洁引用边由原始 Parquet 全量重新扫描、去自环并去重；其他图表从原始 claim、社区、骨架、语义边、质心及案例源记录独立复算。",
        "",
        "| 检查 | 结果 |",
        "| --- | --- |",
    ]
    lines += [
        f"| {r['check']} | {'通过' if r['passed'] else '不一致'} |" for r in audit.rows
    ]
    lines += [
        "",
        "范围：核对数值、身份、引用记录、方向和绘图口径；不把数据库中的引用记录等同于已验证的科学承接关系。297,063 为保存的历史路径注释数。",
        "",
        "复现：`python -m figure_pipeline.fig1_reference.audit`。详细观测值与预期值见 `qa/data_audit.json`。",
    ]
    (out / "qa/data_audit_zh.md").write_text("\n".join(lines) + "\n")
    print(
        f"Full data audit: {len(audit.rows)} checks; failures: {failures}", flush=True
    )
    return not failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/fig1_reference"
    )
    if not audit_all(parser.parse_args().output_dir.resolve()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

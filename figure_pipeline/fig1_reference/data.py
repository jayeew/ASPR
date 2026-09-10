"""Read-only scientific inputs and versioned figure derivatives; no model calls."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "data/claim_graph"
STUDY = ROOT / "outputs/innovation_200_20260907"
ROLES = ["FINDING", "METHOD", "MECHANISM", "RESOURCE", "THEORY"]
COLORS = [
    "#398bea",
    "#fa993b",
    "#54a76a",
    "#eb596e",
    "#ab7bd2",
    "#e6b549",
    "#30b8c5",
    "#e985b6",
]
ROLE_COLORS = ["#398bea", "#ffa74b", "#48ad83", "#a780d6", "#f296c5"]
SELECTIONS = {
    "A": (
        "s41467-026-68852-z::CLAIM::03",
        "High similarity; ten historical neighbors already connected.",
    ),
    "B": (
        "s41467-026-68975-3::CLAIM::02",
        "VSIG10L epithelial mechanism: community diversity and complete saved citation context; replaces the reference-missing cryo-EM example.",
    ),
    "C": (
        "s41467-026-68663-2::CLAIM::01",
        "Sensight probe-design method: seven cross-role neighbors, concentrated community composition and saved citation context; replaces the reference-missing climate example.",
    ),
    "D": (
        "s41467-026-69273-8::CLAIM::02",
        "Three substantial historical components; same-community example separates connectivity from community mixing.",
    ),
}


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def source(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def connect(name: str) -> sqlite3.Connection:
    db = sqlite3.connect(f"file:{ASSETS / name}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


def bare(value: str) -> str:
    return value.rstrip("/").rsplit("/", 1)[-1].upper()


def local_metrics(
    neighbors: list[dict[str, Any]], edges: list[list[str]], role: str
) -> dict[str, Any]:
    identities = {n["claim_id"] for n in neighbors}
    if len(identities) != len(neighbors):
        raise ValueError("Duplicate neighbor identity")
    if any(u not in identities or v not in identities or u == v for u, v in edges):
        raise ValueError(
            "Historical edge outside the retained neighborhood or self-loop"
        )
    if len({tuple(sorted(e)) for e in edges}) != len(edges):
        raise ValueError("Duplicate undirected historical edge")
    graph = nx.Graph()
    graph.add_nodes_from(n["claim_id"] for n in neighbors)
    graph.add_edges_from(edges)
    sizes = sorted((len(c) for c in nx.connected_components(graph)), reverse=True)
    n = len(neighbors)
    pairs = math.comb(n, 2)
    new_pairs = pairs - sum(math.comb(s, 2) for s in sizes)
    weights: Counter[int] = Counter()
    for item in neighbors:
        if item["community_id"] is not None:
            weights[int(item["community_id"])] += item["cosine_similarity"]
    total = sum(weights.values())
    probabilities = {k: v / total for k, v in weights.items()} if total else {}
    similarities = sorted((x["cosine_similarity"] for x in neighbors), reverse=True)
    return {
        "neighbor_count": n,
        "component_sizes": sizes,
        "components_before": len(sizes),
        "components_after": 1 if n else 0,
        "component_merge_count": max(len(sizes) - 1, 0),
        "newly_connected_neighbor_pair_count": new_pairs,
        "connected_pair_share": new_pairs / pairs if pairs else None,
        "nearest_prior_similarity": max(similarities) if n else None,
        "mean_top5_similarity": float(np.mean(similarities[:5])) if n else None,
        "effective_community_count": 1 / sum(p * p for p in probabilities.values())
        if probabilities
        else None,
        "community_probabilities": probabilities,
        "community_assignment_coverage": sum(
            x["community_id"] is not None for x in neighbors
        )
        / n
        if n
        else None,
        "cross_type_neighbor_share": sum(x["claim_type"] != role for x in neighbors) / n
        if n
        else None,
        "role_counts": dict(Counter(x["claim_type"] for x in neighbors)),
        "neighbor_induced_density": len(edges) / pairs if pairs else None,
    }


def rao(probabilities: dict[int, float]) -> float | None:
    index = pd.read_parquet(ASSETS / "community_centroid_index.parquet")
    rows = dict(zip(index.community_id.astype(int), index.centroid_row.astype(int)))
    matrix = np.load(ASSETS / "community_centroid_matrix.npy", mmap_mode="r")
    if not probabilities or any(k not in rows for k in probabilities):
        return None
    keys = sorted(probabilities)
    return float(
        sum(
            2
            * probabilities[a]
            * probabilities[b]
            * np.clip(1 - float(np.dot(matrix[rows[a]], matrix[rows[b]])), 0, 1)
            for i, a in enumerate(keys)
            for b in keys[i + 1 :]
        )
    )


def paths(
    db: sqlite3.Connection, item: dict[str, Any], neighbor: dict[str, Any]
) -> dict[str, Any]:
    target = bare(item.get("openalex_work_id") or item["paper_id"])
    refs = {bare(r) for r in item["reference_work_ids"]} - {target}
    dest = neighbor.get("parent_openalex_work_id")
    if not refs or not dest:
        return {
            "status": "reference_ids_unavailable"
            if not refs
            else "parent_mapping_unavailable",
            "witnesses": [],
            "counts": None,
        }
    dest = bare(dest)
    other_refs = {
        r[0]
        for r in db.execute(
            "SELECT cited_work_id FROM paper_edges WHERE citing_work_id=? AND cited_work_id!=citing_work_id",
            (dest,),
        )
    }
    witnesses = []
    if dest in refs:
        witnesses.append(
            {
                "kind": "direct",
                "edges": [[target, dest]],
                "target_edge_source": "innovation_input.reference_work_ids",
            }
        )
    for ref in sorted(refs - {dest}):
        if db.execute(
            "SELECT 1 FROM paper_edges WHERE citing_work_id=? AND cited_work_id=?",
            (ref, dest),
        ).fetchone():
            witnesses.append(
                {
                    "kind": "two_hop",
                    "edges": [[target, ref], [ref, dest]],
                    "target_edge_source": "innovation_input.reference_work_ids",
                }
            )
    for ref in sorted((refs & other_refs) - {dest, target}):
        witnesses.append(
            {
                "kind": "shared_reference",
                "edges": [[target, ref], [dest, ref]],
                "target_edge_source": "innovation_input.reference_work_ids",
            }
        )
    return {
        "status": "observed_in_saved_references",
        "witnesses": witnesses,
        "counts": dict(Counter(w["kind"] for w in witnesses)),
    }


def load_cases(
    graph: nx.Graph, nodes: dict[str, Any], db: sqlite3.Connection
) -> tuple[list[dict[str, Any]], list[Any]]:
    observations = read(ROOT / "outputs/FROM_WEB/data/fig01_claim_observations.json")
    by_id = {r["claim_id"]: r for r in observations}
    cases = []
    for label, (cid, reason) in SELECTIONS.items():
        row = by_id[cid]
        folder = STUDY / "papers" / row["paper_id"]
        trace = ROOT / row["source_path"]
        records = [json.loads(line) for line in trace.read_text().splitlines()]
        evidence = next(
            e
            for e in reversed(records)
            if e.get("kind") == "graph_fact"
            and e["payload"]["claim"]["claim_id"] == cid
        )
        fact = evidence["payload"]
        item = read(folder / "innovation_input.json")
        shared = next(
            c
            for c in read(folder / "shared/claims.json")["claims"]
            if c["claim_id"] == cid
        )
        ir = read(folder / "shared/paper_ir.json")
        spans = [s for s in ir["spans"] if s["span_id"] in shared["source_span_ids"]]
        assert len(spans) == len(shared["source_span_ids"]), cid
        neighbors = fact["neighbors"]
        for n in neighbors:
            assert n["claim_id"] in nodes and n["parent_paper_id"] != item["paper_id"]
            assert (
                n["publication_date"] < item["cutoff_date"]
                and n["cosine_similarity"] > 0.5
            )
            n["path_recomputed"] = paths(db, item, n)
        edges = [
            sorted(e) for e in graph.subgraph(n["claim_id"] for n in neighbors).edges
        ]
        metrics = local_metrics(neighbors, edges, fact["claim"]["claim_type"])
        metrics["community_rao_stirling"] = rao(metrics["community_probabilities"])
        cases.append(
            {
                "label": label,
                "selection_reason": reason,
                "claim": fact["claim"],
                "input": item,
                "shared_claim": shared,
                "source_spans": spans,
                "neighbors": neighbors,
                "edges": sorted(edges),
                "metrics": metrics,
                "original_metrics": fact["metrics"],
                "evidence_id": evidence["evidence_id"],
                "original_policy": fact["insertion_policy"],
                "derived_policy": "fig1_saved_neighbors_clean_paths_v1; not a retrieval/model rerun",
                "sources": [
                    source(trace),
                    source(folder / "innovation_input.json"),
                    source(folder / "shared/claims.json"),
                    source(folder / "shared/paper_ir.json"),
                ],
                "historical_neighbor_texts": [nodes[n["claim_id"]] for n in neighbors],
            }
        )
    return cases, observations


def scale_positions(
    pos: dict[str, Any], width: float, height: float
) -> dict[str, list[float]]:
    keys = list(pos)
    values = np.array([pos[k] for k in keys], dtype=float)
    values -= values.min(axis=0)
    values /= np.maximum(values.max(axis=0), 1e-9)
    return {k: [float(x * width), float(y * height)] for k, (x, y) in zip(keys, values)}


def ordered_subgraph(graph: nx.Graph, identities: Any) -> nx.Graph:
    keys = sorted(identities)
    result = nx.Graph()
    result.add_nodes_from(keys)
    result.add_edges_from(sorted(tuple(sorted(e)) for e in graph.subgraph(keys).edges))
    return result


def separate_positions(
    pos: dict[str, Any],
    width: float,
    height: float,
    min_distance: float,
    fixed: dict[str, list[float]] | None = None,
    excluded: list[tuple[float, float, float, float]] | None = None,
) -> dict[str, list[float]]:
    """Resolve display collisions; never change graph membership or edge sets."""
    keys = list(pos)
    points = np.array([pos[k] for k in keys], dtype=float)
    fixed = fixed or {}
    rng = np.random.default_rng(409)
    points += rng.normal(0, 0.05, points.shape)
    for _ in range(300):
        delta = points[:, None, :] - points[None, :, :]
        distance = np.sqrt((delta**2).sum(axis=2))
        np.fill_diagonal(distance, 1e9)
        magnitude = (
            np.maximum(min_distance - distance, 0) / np.maximum(distance, 0.001) * 0.24
        )
        shift = (delta * magnitude[:, :, None]).sum(axis=1)
        points += np.clip(shift, -2, 2)
        points = np.clip(points, [0, 0], [width, height])
        for left, top, right, bottom in excluded or []:
            mask = (
                (points[:, 0] >= left)
                & (points[:, 0] <= right)
                & (points[:, 1] >= top)
                & (points[:, 1] <= bottom)
            )
            for i in np.flatnonzero(mask):
                x, y = points[i]
                options = [(left - 1, y), (right + 1, y), (x, top - 1), (x, bottom + 1)]
                options = [
                    p for p in options if 0 <= p[0] <= width and 0 <= p[1] <= height
                ]
                points[i] = min(options, key=lambda p: np.linalg.norm(points[i] - p))
        for key, xy in fixed.items():
            points[keys.index(key)] = xy
    return {key: list(map(float, xy)) for key, xy in zip(keys, points)}


def local_layout(case: dict[str, Any], seed: int) -> dict[str, Any]:
    graph = nx.Graph()
    graph.add_nodes_from(n["claim_id"] for n in case["neighbors"])
    graph.add_edges_from(case["edges"])
    components = sorted(
        nx.connected_components(graph), key=lambda c: (-len(c), sorted(c))
    )
    pos = {}
    for i, group in enumerate(components):
        angle = 2 * np.pi * i / len(components) - 0.4
        center = np.array([np.cos(angle), np.sin(angle)]) * (
            0 if len(components) == 1 else 0.75
        )
        sub = nx.spring_layout(
            ordered_subgraph(graph, group), seed=seed + i, iterations=160
        )
        radius = 0.85 if len(components) == 1 else 0.20 + 0.11 * math.sqrt(len(group))
        pos.update({k: center + radius * xy for k, xy in sub.items()})
    xy = scale_positions(pos, 86, 94)
    # Place the new node in a free gap; never move a historical coordinate.
    points = np.array(list(xy.values()))
    candidates = [
        (x, y) for x in np.linspace(20, 66, 12) for y in np.linspace(20, 74, 12)
    ]
    target = max(
        candidates,
        key=lambda p: (
            min(np.linalg.norm(points - p, axis=1))
            - 0.15 * np.linalg.norm(np.array(p) - [43, 47])
        ),
    )
    return {
        "before": xy,
        "after_historical": xy.copy(),
        "target": list(map(float, target)),
        "seed": seed,
    }


def atlas_data(
    graph: nx.Graph, nodes: dict[str, Any], cases: list[Any], db: sqlite3.Connection
) -> dict[str, Any]:
    sizes = Counter(
        n["community_id"] for n in nodes.values() if n["community_id"] is not None
    )
    focus = []
    for case in cases:
        dominant = max(
            case["metrics"]["community_probabilities"],
            key=case["metrics"]["community_probabilities"].get,
        )
        if dominant not in focus:
            focus.append(dominant)
    case_communities = Counter(
        n["community_id"]
        for c in cases
        for n in c["neighbors"]
        if n["community_id"] is not None
    )
    focus += [
        c
        for c, _ in sorted(case_communities.items(), key=lambda kv: (-kv[1], kv[0]))
        if c not in focus
    ][: 8 - len(focus)]
    focus += [c for c, _ in sizes.most_common() if c not in focus][: 8 - len(focus)]
    chosen = {n["claim_id"] for c in cases for n in c["neighbors"]}
    for community in focus:
        group = {k for k, v in nodes.items() if v["community_id"] == community}
        seeds = sorted(chosen & group)
        if not seeds:
            seeds = [max(group, key=lambda k: (graph.degree(k), k))]
        queue = list(seeds)
        visited = set(seeds)
        while queue and len(visited) < 59:
            node = queue.pop(0)
            for neighbor in sorted(graph[node], key=lambda k: (-graph.degree(k), k)):
                if neighbor in group and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    if len(visited) >= 59:
                        break
        chosen.update(visited)
    rng = np.random.default_rng(109)
    background = sorted(k for k, n in nodes.items() if n["community_id"] not in focus)
    starts = rng.choice(background, size=150, replace=False)
    for k in starts:
        chosen.add(str(k))
        chosen.update(sorted(graph[k])[:2])
        if len(chosen) >= 600:
            break
    sub = ordered_subgraph(graph, chosen)
    # Reference-like anchors are display geometry, not measured scientific coordinates.
    anchors = np.array(
        [
            [0.17, 0.35],
            [0.39, 0.25],
            [0.61, 0.20],
            [0.84, 0.40],
            [0.23, 0.70],
            [0.48, 0.65],
            [0.69, 0.73],
            [0.88, 0.78],
        ]
    )
    pos = {}
    for i, c in enumerate(focus):
        group = sorted(k for k in chosen if nodes[k]["community_id"] == c)
        internal = nx.spring_layout(
            ordered_subgraph(sub, group), seed=90 + i, iterations=180, k=0.4
        )
        pos.update(
            {k: anchors[i] + np.array([0.14, 0.21]) * xy for k, xy in internal.items()}
        )
    for k in sorted(chosen):
        if k not in pos:
            pos[k] = rng.uniform([0.03, 0.08], [0.98, 0.97])
    pos = nx.spring_layout(sub, pos=pos, seed=91, iterations=12, k=0.058)
    positions = separate_positions(
        scale_positions(pos, 582, 303),
        582,
        303,
        7.7,
        excluded=[(0, 0, 165, 58), (0, 220, 165, 278)],
    )
    node_rows = [
        {
            **nodes[k],
            "x": positions[k][0],
            "y": positions[k][1],
            "backbone_degree": graph.degree(k),
        }
        for k in sorted(chosen)
    ]
    parents = {}
    for pid in sorted({n["parent_paper_id"] for n in node_rows}):
        p = db.execute(
            "SELECT * FROM paper_nodes WHERE nature_article_id=?", (pid,)
        ).fetchone()
        if p:
            parents[pid] = dict(p)
    return {
        "claim_nodes": node_rows,
        "claim_edges": sorted([sorted(e) for e in sub.edges]),
        "focus_communities": focus,
        "parent_papers_all": list(parents.values()),
        "ownership": [
            {
                "claim_id": n["claim_id"],
                "parent_paper_id": n["parent_paper_id"],
                "work_id": parents.get(n["parent_paper_id"], {}).get("work_id"),
            }
            for n in node_rows
        ],
        "selection": "Eight community BFS samples plus independent seeded background and all four case neighborhoods; display sample, not a population estimator.",
        "layout_seed": 91,
    }


def paper_layer(
    atlas: dict[str, Any], cases: list[Any], db: sqlite3.Connection
) -> dict[str, Any]:
    from .paper_display import build_paper_layer

    return build_paper_layer(atlas, cases, db)


def community_ccdf(histogram: dict[Any, int]) -> list[dict[str, Any]]:
    counts = sorted((int(size), int(count)) for size, count in histogram.items())
    if not counts or any(size < 1 or count <= 0 for size, count in counts):
        raise ValueError("Community histogram requires positive sizes and counts")
    total = sum(count for _, count in counts)
    remaining = total
    rows = []
    for size, count in counts:
        rows.append(
            {
                "size": size,
                "frequency": count,
                "communities_at_least_size": remaining,
                "share_at_least_size": remaining / total,
            }
        )
        remaining -= count
    return rows


def prepare(out: Path) -> None:
    frame = pd.read_parquet(ASSETS / "claim_nodes.parquet")
    assignment = pd.read_parquet(ASSETS / "claim_communities.parquet")
    frame = frame.merge(assignment, on="claim_id", validate="one_to_one")
    frame["publication_date"] = frame["publication_date"].astype(str).str[:10]
    nodes = json.loads(frame.to_json(orient="records"))
    nodes = {n["claim_id"]: n for n in nodes}
    for n in nodes.values():
        if n["community_id"] is not None:
            n["community_id"] = int(n["community_id"])
    edges = pd.read_parquet(ASSETS / "claim_backbone_edges.parquet")
    graph = nx.Graph()
    graph.add_nodes_from(sorted(nodes))
    graph.add_edges_from(
        edges[["claim_id_a", "claim_id_b"]].itertuples(index=False, name=None)
    )
    with connect("paper_graph_index.sqlite") as db:
        cases, observations = load_cases(graph, nodes, db)
        atlas = atlas_data(graph, nodes, cases, db)
        atlas["paper_layer"] = paper_layer(atlas, cases, db)
    for i, c in enumerate(cases):
        c["layout"] = local_layout(c, 41 + i)
        write(out / "layouts" / f"case_{c['label']}.json", c["layout"])
    semantic = pd.read_parquet(
        ASSETS / "semantic_claim_edges.parquet",
        columns=["earlier_claim_type", "later_claim_type"],
    )
    role_matrix = pd.crosstab(
        semantic.earlier_claim_type, semantic.later_claim_type
    ).reindex(index=ROLES, columns=ROLES, fill_value=0)
    communities = []
    focus = atlas["focus_communities"]
    mixing = np.zeros((len(focus), len(focus)), dtype=int)
    for a, b in graph.edges:
        ca, cb = nodes[a]["community_id"], nodes[b]["community_id"]
        if ca in focus and cb in focus:
            i, j = focus.index(ca), focus.index(cb)
            mixing[i, j] += 1
            if i != j:
                mixing[j, i] += 1
    for i, cid in enumerate(focus):
        group = [n for n in nodes.values() if n["community_id"] == cid]
        communities.append(
            {
                "display": f"K{i + 1}",
                "community_id": cid,
                "size": len(group),
                "role_counts": dict(Counter(n["claim_type"] for n in group)),
                "internal_backbone_edges": int(mixing[i, i]),
            }
        )
    with zipfile.ZipFile(ROOT / "outputs/GRAPH_STATISTICS.zip") as archive:
        paper_stats = json.loads(
            archive.read("GRAPH_STATISTICS/paper_graph_statistics.json")
        )
        write(out / "data/paper_statistics_source.json", paper_stats)
    sizes = Counter(
        n["community_id"] for n in nodes.values() if n["community_id"] is not None
    )
    aggregate = read(ROOT / "docs/Fig1_design_package/verified_aggregate_inputs.json")
    if (
        aggregate["paper_clean_edges"]
        != paper_stats["edges"]["clean_distinct_nonself_edges"]
    ):
        raise ValueError("Paper edge aggregate sources disagree")
    for record in paper_stats["sources"]:
        path = ROOT / record["path"]
        if path.stat().st_size != record["bytes"]:
            raise ValueError(f"Aggregate input size changed: {path}")
    statistics = {
        "claims": len(nodes),
        "semantic_edges": len(semantic),
        "backbone_edges": len(edges),
        "roles": dict(Counter(n["claim_type"] for n in nodes.values())),
        "communities": communities,
        "community_mixing": mixing.tolist(),
        "community_size_histogram": dict(sorted(Counter(sizes.values()).items())),
        "community_count": len(sizes),
        "unassigned": sum(n["community_id"] is None for n in nodes.values()),
        "role_matrix_counts": role_matrix.to_numpy().tolist(),
        "role_matrix_column_normalized": (role_matrix / role_matrix.sum(axis=0))
        .to_numpy()
        .tolist(),
        "claims_per_paper": dict(
            Counter(Counter(n["parent_paper_id"] for n in nodes.values()).values())
        ),
        "target_papers": len(pd.read_parquet(ASSETS / "nature_targets.parquet")),
        "papers_with_claims": frame.parent_paper_id.nunique(),
        "paper_nodes": aggregate["paper_nodes"],
        "paper_clean_edges": aggregate["paper_clean_edges"],
        "saved_paper_path_annotations": len(
            pd.read_parquet(
                ASSETS / "paper_path_claim_edges.parquet", columns=["earlier_claim_id"]
            )
        ),
        "paper_raw_edges": paper_stats["edges"]["raw_rows"],
        "historical_max_date": max(n["publication_date"] for n in nodes.values()),
    }
    statistics["community_ccdf"] = community_ccdf(
        statistics["community_size_histogram"]
    )
    write(out / "data/community_ccdf.json", statistics["community_ccdf"])
    pd.DataFrame(statistics["community_ccdf"]).to_csv(
        out / "data/community_ccdf.csv", index=False
    )
    statistics["claims_per_paper"][0] = (
        statistics["target_papers"] - statistics["papers_with_claims"]
    )
    snapshot = {
        "cases": cases,
        "atlas": atlas,
        "statistics": statistics,
        "sources": [
            source(ASSETS / name)
            for name in [
                "claim_nodes.parquet",
                "claim_communities.parquet",
                "claim_backbone_edges.parquet",
                "semantic_claim_edges.parquet",
                "community_centroid_index.parquet",
                "community_centroid_matrix.npy",
            ]
        ],
        "aggregate_source": source(ROOT / "outputs/GRAPH_STATISTICS.zip"),
        "reference": source(
            Path("/mnt/c/Users/jayee/Downloads/ChatGPT Image 2026年9月9日 20_25_04.png")
        ),
    }
    write(out / "data/snapshot.json", snapshot)
    selection_details(out, observations)
    write(
        out / "data/selection_manifest.json",
        {
            "selection": SELECTIONS,
            "candidate_count": len(observations),
            "filter": "n>=5; community coverage>=0.8; verified source spans; distinct papers; prefer n=10",
            "selection_uses": "structural facts and source text only, not reviewer approval, citations or GEAR judgment",
            "candidates": [
                {
                    k: r[k]
                    for k in [
                        "claim_id",
                        "neighbor_count",
                        "nearest_prior_similarity",
                        "effective_community_count",
                        "component_merge_count",
                        "cross_type_neighbor_count",
                    ]
                }
                for r in observations
            ],
            "nonselection": "Not selected for this four-case illustrative gallery; not evidence of low novelty.",
        },
    )
    write(
        out / "data/paper_path_witnesses.json",
        [
            {"case": c["label"], "neighbor": n["claim_id"], **n["path_recomputed"]}
            for c in cases
            for n in c["neighbors"]
        ],
    )
    write(
        out / "layouts/atlas.json",
        {
            "claims": [
                {k: n[k] for k in ["claim_id", "x", "y"]} for n in atlas["claim_nodes"]
            ],
            "papers": atlas["paper_layer"]["nodes"],
        },
    )
    print(
        f"Prepared {len(atlas['claim_nodes'])} atlas claims, {len(atlas['paper_layer']['nodes'])} papers and four cases.",
        flush=True,
    )


def selection_details(out: Path, observations: list[dict[str, Any]]) -> None:
    pools: dict[str, list[str]] = {label: [] for label in SELECTIONS}
    excluded = []
    for row in observations:
        n = row["neighbor_count"]
        if n < 5 or row["community_assignment_coverage"] < 0.8:
            excluded.append(
                {
                    "claim_id": row["claim_id"],
                    "reason": "fewer than five neighbors"
                    if n < 5
                    else "community coverage below 80%",
                }
            )
            continue
        if row["component_merge_count"] == 0 and row["nearest_prior_similarity"] >= 0.8:
            pools["A"].append(row["claim_id"])
        if (
            row["effective_community_count"] > 2.5
            and row["community_rao_stirling"] > 0.1
            and row["component_merge_count"] <= 5
        ):
            pools["B"].append(row["claim_id"])
        if (
            row["claim_type"] == "METHOD"
            and row["cross_type_neighbor_count"] / n >= 0.7
            and row["effective_community_count"] < 2
        ):
            pools["C"].append(row["claim_id"])
        if (
            n == 10
            and row["component_merge_count"] == 2
            and row["newly_connected_neighbor_pair_count"] >= 25
        ):
            pools["D"].append(row["claim_id"])
    write(
        out / "data/case_alternatives.json",
        {
            "structural_pools": pools,
            "excluded": excluded,
            "status": "Alternative pools screened on structure only; source-span review performed for the four selected cases.",
            "ranking_policy": "Illustrative coverage of four views, not a novelty ranking or representative outcome sample.",
            "selected": {
                k: {"claim_id": v[0], "reason": v[1]} for k, v in SELECTIONS.items()
            },
        },
    )

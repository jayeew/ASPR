"""Read existing study artifacts and freeze figure-specific data; never run models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import networkx as nx

from figure_pipeline.fig1_reference.data import connect, paths

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "outputs/innovation_200_20260907"
PAPER = "s42003-026-09574-2"
REFERENCE = Path(
    "/mnt/c/Users/jayee/Downloads/ChatGPT Image 2026年9月14日 11_57_06.png"
)
DEFAULT_OUT = ROOT / "outputs/fig2_reference"
METRICS = [
    ("nearest_prior_similarity", "Nearest-prior similarity"),
    ("mean_top5_similarity", "Top-5 mean similarity"),
    ("effective_community_count", "Effective community count"),
    ("community_rao_stirling", "Community disparity"),
    ("first_observed_recent_nature_pair_share", "Unseen community-pair share"),
    ("community_pair_mean_surprisal", "Community-pair surprisal"),
    ("cross_boundary_weight_share", "Non-dominant community weight"),
    ("cross_type_neighbor_count", "Cross-role neighbors"),
    ("neighbor_induced_density", "Historical-neighborhood density"),
    ("component_merge_count", "Local component merges"),
    ("newly_connected_neighbor_pair_count", "Newly connected neighbor pairs"),
    ("direct_citation_neighbor_count", "Direct-citation neighbors"),
    ("two_hop_neighbor_count", "Two-hop neighbors"),
    ("co_citation_neighbor_count", "Shared-reference neighbors"),
]


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def file_source(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def topology(nodes: list[str], edges: list[list[str]]) -> dict[str, Any]:
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    sizes = [len(c) for c in nx.connected_components(graph)]
    pairs = len(nodes) * (len(nodes) - 1) // 2
    return {
        "density": nx.density(graph),
        "components": len(sizes),
        "merges": max(len(sizes) - 1, 0),
        "new_pairs": pairs - sum(n * (n - 1) // 2 for n in sizes),
    }


def network_layout(
    nodes: list[str], edges: list[list[str]], seed: int
) -> dict[str, list[float]]:
    graph = nx.Graph()
    graph.add_nodes_from(sorted(nodes))
    graph.add_edges_from(sorted(edges))
    targets = [n for n in sorted(nodes) if "::CLAIM::" in n]
    if len(targets) > 1:
        initial = nx.spring_layout(graph, seed=seed, iterations=200)
        anchors = [[-0.7, -0.6], [0.7, -0.6], [0.7, 0.7], [-0.7, 0.7]]
        for node, anchor in zip(targets, anchors):
            initial[node] = anchor
        positions = nx.spring_layout(
            graph, pos=initial, fixed=targets, seed=seed, iterations=600, k=0.32
        )
    else:
        positions = nx.spring_layout(
            graph, seed=seed, iterations=500, k=1.4 / len(nodes) ** 0.5
        )
    xs, ys = [v[0] for v in positions.values()], [v[1] for v in positions.values()]
    return {
        n: [
            (float(p[0]) - min(xs)) / max(max(xs) - min(xs), 1e-6),
            (float(p[1]) - min(ys)) / max(max(ys) - min(ys), 1e-6),
        ]
        for n, p in positions.items()
    }


def prepare(out: Path) -> dict[str, Any]:
    paper = STUDY / "papers" / PAPER
    used: list[Path] = []

    def load(relative: str) -> Any:
        path = paper / relative
        used.append(path)
        return read(path)

    item, shared = load("innovation_input.json"), load("shared/claims.json")
    ir = load("shared/paper_ir.json")
    claims = shared["claims"]
    facts, assessments = [], {"gear": [], "graph": []}
    for claim in claims:
        num = claim["claim_id"].rsplit("::", 1)[-1]
        trace = paper / "graph" / num / "evidence_trace.jsonl"
        used.append(trace)
        facts.append(
            next(
                r["payload"]
                for r in reversed(records(trace))
                if r["kind"] == "graph_fact"
            )
        )
        for branch, entries in assessments.items():
            entries.append(load(f"{branch}/{num}/assessment.json"))
    joint = load("graph/joint/facts.json")
    joint_analysis = load("graph/joint/analysis.json")
    card = load("gear/01/gear_card.json")
    trace = paper / "gear/01/evidence_trace.jsonl"
    used.append(trace)
    gear_records = records(trace)
    report_path = STUDY / "reports/fusion" / f"{PAPER}.json"
    used.append(report_path)
    report = read(report_path)
    historical = {n["claim_id"]: n for fact in facts for n in fact["neighbors"]}
    aliases = {c["claim_id"]: f"C{i + 1}" for i, c in enumerate(claims)}
    order = [n["claim_id"] for n in facts[0]["neighbors"]]
    order += sorted(set(historical) - set(order))
    aliases.update({n: f"H{i + 1}" for i, n in enumerate(order)})
    source_spans = [
        claims[0]["source_span_ids"][0],
        "S-468a0ca5373a9634a1ef",
        "S-40f6fd9cda209eb9b384",
    ]
    aliases.update({sid: f"S{i + 1}" for i, sid in enumerate(source_spans)})
    selected_work_ids = ["W2789458740", "W4220699785", "W4404674953"]
    works = []
    for i, wid in enumerate(selected_work_ids):
        work = next(
            r
            for r in gear_records
            if r["kind"] == "retrieved_work" and r["evidence_id"].endswith(wid)
        )
        relation = next(
            r
            for r in gear_records
            if r["kind"] == "relation_card" and r["evidence_id"].endswith(wid)
        )
        works.append({"alias": f"W{i + 1}", "record": work, "relation": relation})
        aliases[work["payload"]["work_id"]] = f"W{i + 1}"
    witnesses = {}
    with connect("paper_graph_index.sqlite") as db:
        for neighbor in facts[0]["neighbors"]:
            witnesses[neighbor["claim_id"]] = paths(db, item, neighbor)
    # Recompute the path counts on the clean, deduplicated rule for this display.
    metrics = {m["name"]: m["value"] for m in facts[0]["metrics"]}
    path_kinds = {
        "direct_citation_neighbor_count": "direct",
        "two_hop_neighbor_count": "two_hop",
        "co_citation_neighbor_count": "shared_reference",
    }
    path_diffs = {}
    for name, kind in path_kinds.items():
        value = sum(
            any(w["kind"] == kind for w in r["witnesses"]) for r in witnesses.values()
        )
        if value != metrics[name]:
            path_diffs[name] = {"saved": metrics[name], "display": value}
        metrics[name] = value
    witness = next(
        {"neighbor_id": n, **w}
        for n, r in witnesses.items()
        for w in r["witnesses"]
        if w["kind"] == "shared_reference"
    )
    atlas_source = ROOT / "outputs/fig1_reference/data/snapshot.json"
    atlas_layout = ROOT / "outputs/fig1_reference/layouts/atlas.json"
    used.extend([atlas_source, atlas_layout, REFERENCE])
    atlas = read(atlas_source)["atlas"]
    snapshot = {
        "paper": item,
        "claims": claims,
        "facts": facts,
        "metrics": metrics,
        "joint": joint,
        "joint_analysis": joint_analysis,
        "assessments": assessments,
        "gear_card": card,
        "gear_records": gear_records,
        "report": report,
        "spans": [s for s in ir["spans"] if s["span_id"] in source_spans],
        "aliases": aliases,
        "historical": historical,
        "works": works,
        "witness": witness,
        "path_witnesses": witnesses,
        "path_differences": path_diffs,
        "candidate_count": len(shared["candidate_records"]),
        "consolidated_count": len(shared["consolidation_records"]),
        "fusion_available": (paper / "fusion/01/assessment.json").exists(),
        "atlas": {
            "nodes": atlas["claim_nodes"],
            "edges": atlas["claim_edges"],
            "focus_communities": atlas["focus_communities"],
            "layout": read(atlas_layout)["claims"],
        },
        "sources": [file_source(p) for p in used],
        "notes": [
            "An existing study example, not a new system evaluation.",
            "English compact texts are figure summaries of the preserved source artifacts.",
            "Standard claim fusion is shown as a schema because this study did not produce it.",
            "GEAR status and assessment are retained separately; missing residual_contribution is not filled.",
            "Path values are figure derivatives checked against deduplicated, loop-free witnesses.",
        ],
    }
    write(out / "data/snapshot.json", snapshot)
    write(out / "data/source_manifest.json", snapshot["sources"])
    write(out / "data/aliases.json", aliases)
    write(out / "data/path_witnesses.json", witnesses)
    local_nodes = order[: len(facts[0]["neighbors"])] + [claims[0]["claim_id"]]
    local_edges = facts[0]["neighbor_edges"] + [
        [claims[0]["claim_id"], n] for n in local_nodes[:-1]
    ]
    layouts = {
        "local": network_layout(local_nodes, local_edges, 129),
        "joint": network_layout(
            list(historical) + [c["claim_id"] for c in claims],
            joint["historical_edges"] + joint["insertion_edges"],
            68,
        ),
    }
    write(out / "layouts/networks.json", layouts)
    print(
        f"Prepared {len(claims)} claims, {len(historical)} union neighbors, {len(metrics)} descriptors.",
        flush=True,
    )
    return snapshot

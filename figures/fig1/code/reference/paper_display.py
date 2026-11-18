"""Connected citation-context selection; all displayed edges remain source records."""

from __future__ import annotations

import sqlite3
from collections import Counter
from typing import Any

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

from .data import bare, separate_positions

# Not declared a database error: this cross-topic record has not been corroborated
# and is unnecessary for the display. Never use it as a shortcut between fields.
EXCLUDED_BRIDGES = {("W4295546423", "W2998840182")}
DISPLAY_BUDGET = 90


def connected_selection(
    graph: nx.DiGraph, roots: list[str], preferred: set[str], budget: int
) -> tuple[list[str], list[list[str]]]:
    undirected = graph.to_undirected()
    paths = {root: nx.single_source_shortest_path(undirected, root) for root in roots}
    if not all(root in paths[roots[0]] for root in roots):
        raise ValueError("Observed citation pool does not connect all target papers")
    closure = nx.Graph()
    for i, u in enumerate(roots):
        for v in roots[i + 1 :]:
            closure.add_edge(u, v, weight=len(paths[u][v]) - 1)
    connecting_paths = [paths[u][v] for u, v in nx.minimum_spanning_tree(closure).edges]
    selected = {node for path in connecting_paths for node in path}
    if len(selected) > budget:
        raise ValueError("Citation paths exceed the display budget")
    regions = {
        node: min(
            range(len(roots)),
            key=lambda i: len(paths[roots[i]].get(node, range(10**6))),
        )
        for node in nx.node_connected_component(undirected, roots[0])
    }
    frontier: Counter[str] = Counter(
        v for u in selected for v in undirected[u] if v not in selected
    )
    while len(selected) < budget and frontier:
        counts = Counter(regions[node] for node in selected)
        region = min(range(len(roots)), key=lambda i: (counts[i], i))
        candidates = [node for node in frontier if regions[node] == region] or list(
            frontier
        )
        winner = max(
            candidates,
            key=lambda node: (
                min(frontier[node], 3),
                node in preferred,
                -len(paths[roots[regions[node]]][node]),
                min(graph.degree(node), 40),
                node,
            ),
        )
        selected.add(winner)
        del frontier[winner]
        frontier.update(v for v in undirected[winner] if v not in selected)
    return sorted(selected), connecting_paths


def bounded_layout(graph: nx.Graph, roots: list[str]) -> dict[str, list[float]]:
    keys = sorted(graph)
    rng = np.random.default_rng(144)
    points = rng.uniform([8, 8], [512, 131], (len(keys), 2))
    anchors = {
        root: [[75, 280, 365, 450][i], 70 + (10 if i % 2 else -10)]
        for i, root in enumerate(roots)
    }
    adjacency = nx.to_numpy_array(graph, nodelist=keys)
    for step in range(550):
        delta = points[:, None] - points[None, :]
        distance = np.maximum(np.linalg.norm(delta, axis=2), 0.1)
        forces = (
            delta * (12**2 / distance**2 - adjacency * distance / 12)[:, :, None]
        ).sum(axis=1)
        forces -= [0.1, 0.8] * (points - [260, 69.5])
        for axis, extent, strength in [(0, 520, 2000), (1, 139, 1500)]:
            forces[:, axis] += strength / np.maximum(points[:, axis], 1) ** 2
            forces[:, axis] -= strength / np.maximum(extent - points[:, axis], 1) ** 2
        norm = np.maximum(np.linalg.norm(forces, axis=1), 0.1)
        points += forces / norm[:, None] * (2 * (1 - step / 550) + 0.02)
        points = np.clip(points, [4, 4], [516, 135])
        for root, xy in anchors.items():
            points[keys.index(root)] = xy
    positions = separate_positions(dict(zip(keys, points)), 520, 139, 12, anchors)
    return spread_to_slots(positions, anchors, rng)


def spread_to_slots(
    positions: dict[str, list[float]],
    anchors: dict[str, list[float]],
    rng: np.random.Generator,
) -> dict[str, list[float]]:
    """Fill the narrow display without boundary piles; coordinates carry no metric."""
    others = [node for node in positions if node not in anchors]
    slots: list[list[float]] = []
    for _ in range(30000):
        point = rng.uniform([6, 8], [514, 131])
        if all(
            np.linalg.norm(point - np.array(existing)) > 18
            for existing in slots + list(anchors.values())
        ):
            slots.append(point.tolist())
        if len(slots) == len(others):
            break
    if len(slots) != len(others):
        raise ValueError("Paper display cannot fit the requested node spacing")
    points = np.array([positions[node] for node in others])
    destinations = np.array(slots)
    rows, columns = linear_sum_assignment(
        ((points[:, None] - destinations[None, :]) ** 2).sum(axis=2)
    )
    positions.update({others[r]: slots[c] for r, c in zip(rows, columns)})
    return positions


def build_paper_layer(
    atlas: dict[str, Any], cases: list[Any], db: sqlite3.Connection
) -> dict[str, Any]:
    roots = []
    metadata = {p["work_id"]: p for p in atlas["parent_papers_all"]}
    graph = nx.DiGraph()
    for case in cases:
        item = case["input"]
        root = bare(item.get("openalex_work_id") or item["paper_id"])
        refs = sorted({bare(r) for r in item["reference_work_ids"]} - {root})
        if not refs:
            raise ValueError(f"Case {case['label']} has no saved reference IDs")
        roots.append(root)
        graph.add_edges_from((root, ref) for ref in refs)
        metadata[root] = {
            "work_id": root,
            "nature_article_id": item["paper_id"],
            "title": item["title"],
            "publication_date": item["publication_date"],
            "target_case": case["label"],
        }
    queried = set(roots)
    for _ in range(2):
        frontier = sorted(set(graph) - queried)
        queried.update(frontier)
        for node in frontier:
            graph.add_edges_from(
                (node, row[0])
                for row in db.execute(
                    "SELECT cited_work_id FROM paper_edges WHERE citing_work_id=? AND citing_work_id!=cited_work_id",
                    (node,),
                )
                if (node, row[0]) not in EXCLUDED_BRIDGES and row[0] not in roots
            )
    selected, bridges = connected_selection(
        graph, roots, set(metadata) - set(roots), DISPLAY_BUDGET
    )
    selected_set = set(selected)
    # Fetch the full directed induced edge set for selected non-target nodes.
    for node in selected:
        if node not in roots:
            graph.add_edges_from(
                (node, row[0])
                for row in db.execute(
                    "SELECT cited_work_id FROM paper_edges WHERE citing_work_id=? AND citing_work_id!=cited_work_id",
                    (node,),
                )
                if row[0] in selected_set and (node, row[0]) not in EXCLUDED_BRIDGES
            )
    display = graph.subgraph(selected).copy()
    pos = bounded_layout(display.to_undirected(), roots)
    for node in selected:
        if node not in metadata:
            row = db.execute(
                "SELECT * FROM paper_nodes WHERE work_id=?", (node,)
            ).fetchone()
            metadata[node] = (
                dict(row)
                if row
                else {"work_id": node, "metadata_status": "unavailable"}
            )
        metadata[node].update(x=pos[node][0], y=pos[node][1])
    return {
        "nodes": [metadata[k] for k in selected],
        "edges": [list(edge) for edge in sorted(display.edges)],
        "witness_edges": [
            list(edge) for edge in sorted(display.edges) if edge[0] in roots
        ],
        "connecting_paths": bridges,
        "candidate_nodes": len(graph),
        "candidate_edges": graph.number_of_edges(),
        "omitted_parent_papers": len(
            {p["work_id"] for p in atlas["parent_papers_all"]} - selected_set
        ),
        "weak_components": nx.number_weakly_connected_components(display),
        "isolated_nodes": list(nx.isolates(display)),
        "target_degrees": {root: display.degree(root) for root in roots},
        "seed": 144,
        "layout": {
            "method": "Constrained force layout, then minimum-displacement assignment to deterministic spaced slots; no metric interpretation.",
            "extent": [520, 139],
            "minimum_spacing": 18,
            "iterations": 550,
        },
        "selection": "Three citation hops from each target; minimum connector paths then balanced connected growth. Display selection, not an unbiased sample.",
        "edge_source": "read-only paper_graph_index; target edges from saved input references",
        "excluded_display_edges": [
            {
                "edge": list(edge),
                "reason": "Cross-topic citation shortcut not independently corroborated; omitted from display only.",
            }
            for edge in sorted(EXCLUDED_BRIDGES)
        ],
    }

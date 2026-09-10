"""Frozen-snapshot graph audit and readable Fig. 1 / descriptive Fig. 3.

This module never changes historical assets.  Its cleaned atlas is a versioned
derived graph; local claim metrics remain their recorded snapshot values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Polygon
from scipy.stats import spearmanr

from figure_pipeline.renderers.theme import PALETTE

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, EDGE = PALETTE[0], "#555555", "#b2b2b2"
CMAP = LinearSegmentedColormap.from_list("user_sequential", PALETTE)
MARKERS = ["o", "s", "^", "D", "v", "P"]
MM = 1 / 25.4
VERSION = "graph_revision_v2.1"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_location(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 10,
        "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": EDGE,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": .6, "lines.linewidth": 1.2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def clean_atlas(atlas: dict[str, Any], source: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Quarantine invalid citation records before deriving any graph statistics."""
    nodes = {row["work_id"]: dict(row) for row in atlas["paper_nodes"]}
    accepted, seen, quarantined, dates = [], set(), [], []
    for index, edge in enumerate(atlas["paper_edges"]):
        u, v = edge
        reason = None
        if not re.fullmatch(r"W\d+", u) or not re.fullmatch(r"W\d+", v):
            reason = "invalid_work_id"
        elif u not in nodes or v not in nodes:
            reason = "missing_endpoint"
        elif u == v:
            reason = "same_work_citation_self_loop"
        elif (u, v) in seen:
            reason = "duplicate_directed_edge"
        record = {"source_file": source_location(source), "source_sha256": sha256(source),
                  "source_pointer": f"/paper_edges/{index}", "citing_work_id": u, "cited_work_id": v}
        if reason:
            quarantined.append({**record, "reason": reason, "source_node": nodes.get(u), "target_node": nodes.get(v)})
            continue
        accepted.append([u, v])
        seen.add((u, v))
        date_u, date_v = nodes[u].get("publication_date"), nodes[v].get("publication_date")
        if date_u and date_v and date_u < date_v:
            dates.append({**record, "citing_date": date_u, "cited_date": date_v,
                          "action": "retained_pending_version_and_online_first_date_review"})
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(accepted)
    for node, row in nodes.items():
        row.update(clean_induced_in_degree=graph.in_degree(node), clean_induced_out_degree=graph.out_degree(node))
    undirected = graph.to_undirected()
    components = sorted((sorted(c) for c in nx.connected_components(undirected)), key=lambda c: (-len(c), c))
    stats = {"paper_nodes": len(nodes), "original_paper_edges": len(atlas["paper_edges"]),
             "clean_directed_citation_edges": len(accepted), "quarantined": len(quarantined),
             "quarantine_reasons": dict(Counter(row["reason"] for row in quarantined)),
             "isolated_paper_nodes": len(list(nx.isolates(graph))), "weak_components": len(components),
             "weak_component_size_counts": dict(Counter(map(len, components))),
             "reachable_ordered_pairs": sum(len(nx.descendants(graph, n)) for n in graph),
             "date_order_anomalies_pending_review": len(dates)}
    result = {**atlas, "paper_nodes": list(nodes.values()), "paper_edges": accepted,
              "derived_graph_version": VERSION, "clean_statistics": stats,
              "source_sha256": sha256(source), "paper_components": components,
              "scope": "Frozen atlas induced subgraph only; historical database remains unchanged."}
    audit = {"statistics": stats, "quarantined_edges": quarantined, "date_anomalies": dates,
             "upstream_location": "data/claim_graph/paper_graph_index.sqlite:paper_edges(citing_work_id,cited_work_id)",
             "limitation": "Source snapshots locate each anomaly, but do not identify the original ingestion/alias cause; no full-database repair is claimed."}
    return result, audit


def connected_pair_share(graph: nx.Graph) -> float | None:
    n = len(graph)
    if n < 2:
        return None
    return 1 - sum(len(c) * (len(c) - 1) for c in nx.connected_components(graph)) / (n * (n - 1))


def structural_audit(rows: list[dict[str, Any]], draws: int = 2000) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Use matched n,m G(n,m) as a topology diagnostic, never content ground truth."""
    valid = [r for r in rows if r["neighbor_count"] >= 2 and r.get("connected_pair_share") is not None]
    strata: dict[tuple[int, int], dict[str, Any]] = {}
    for row in valid:
        n = int(row["neighbor_count"])
        m = round(row["neighbor_induced_density"] * math.comb(n, 2))
        if (n, m) in strata:
            continue
        samples = np.array([connected_pair_share(nx.gnm_random_graph(n, m, seed=9309 + 10000*n + 101*m + i)) for i in range(draws)])
        strata[n, m] = {"n": n, "m": m, "draws": draws, "mean": float(samples.mean()),
                        "sd": float(samples.std()), "q025": float(np.quantile(samples, .025)),
                        "q975": float(np.quantile(samples, .975))}
    normalized = []
    for row in rows:
        n = row["neighbor_count"]
        item = {**row, "metric_version": VERSION, "retained_neighbor_count": n,
                "at_retention_cap": n == 10, "threshold_eligible_candidate_count": None,
                "top_k_truncated": None, "candidate_count_status": "not_exported_in_frozen_snapshot",
                "connected_pair_share": row.get("connected_pair_share") if n >= 2 else None}
        if n >= 2 and row.get("neighbor_induced_density") is not None:
            m = round(row["neighbor_induced_density"] * math.comb(n, 2))
            null = strata[n, m]
            item.update(induced_edge_count=m, null_model=null,
                        connected_pair_residual=float(item["connected_pair_share"] - null["mean"]))
        normalized.append(item)
    rho_density = spearmanr([r["neighbor_induced_density"] for r in valid], [r["connected_pair_share"] for r in valid]).statistic
    rho_merge = spearmanr([r["component_merge_count"] for r in valid], [r["connected_pair_share"] for r in valid]).statistic
    audit = {"claims": len(rows), "papers": len({r["paper_id"] for r in rows}), "valid_pair_share_claims": len(valid),
             "neighbor_count_distribution": dict(Counter(r["neighbor_count"] for r in rows)),
             "at_cap_count": sum(r["neighbor_count"] == 10 for r in rows),
             "at_cap_share": sum(r["neighbor_count"] == 10 for r in rows) / len(rows),
             "median_connected_pair_share": float(np.median([r["connected_pair_share"] for r in valid])),
             "rho_density_connected_pair": float(rho_density), "rho_merge_connected_pair": float(rho_merge),
             "null_model": {"name": "G(n,m)", "conditioned_on": ["retained_neighbor_count", "induced_edge_count"],
                            "draws_per_stratum": draws, "seed_rule": "9309 + 10000*n + 101*m + draw_index",
                            "strata": list(strata.values()), "scientific_labels": False},
             "interpretation": "Local connected-pair change is mechanically constrained by induced component sizes; the null does not control similarity, date, field, degree sequence or content.",
             "formula": "1 - sum(s_i*(s_i-1))/(n*(n-1)); n<2 is NA",
             "unavailable_fields": ["threshold_eligible_candidate_count", "top_k_truncated", "independent_content_verification"]}
    return normalized, audit


def community_style(communities: list[int]) -> dict[int, tuple[str, str]]:
    return {c: (PALETTE[i % len(PALETTE)], MARKERS[(i // len(PALETTE)) % len(MARKERS)]) for i, c in enumerate(sorted(set(communities)))}


def draw_edges(ax: plt.Axes, edges: list[Any], pos: dict[Any, np.ndarray], **kwargs: Any) -> None:
    segments = [(pos[u], pos[v]) for u, v in edges if u in pos and v in pos]
    ax.add_collection(LineCollection(segments, **kwargs))


def node_groups(ax: plt.Axes, nodes: list[dict[str, Any]], key: str, pos: dict[Any, np.ndarray], styles: dict[int, tuple[str, str]], size: float) -> None:
    for community in sorted({n["community_id"] for n in nodes}):
        group = [n for n in nodes if n["community_id"] == community]
        xy = np.array([pos[n[key]] for n in group])
        color, marker = styles[community]
        artist = ax.scatter(xy[:, 0], xy[:, 1], s=size, c=color, marker=marker,
                            edgecolors=INK if color == PALETTE[-1] else "white", linewidths=.2, zorder=4)
        register_node_ids(ax, artist, [str(n[key]) for n in group])


def register_node_ids(ax: plt.Axes, artist: matplotlib.artist.Artist, ids: list[str]) -> None:
    """Record stable node IDs for SVG use elements without changing their geometry."""
    registry = getattr(ax.figure, "_aspr_node_ids", {})
    group_id = f"nodes-axis{ax.figure.axes.index(ax)}-group{len(registry)}"
    artist.set_gid(group_id)
    registry[group_id] = ids
    ax.figure._aspr_node_ids = registry


def identify_svg_nodes(path: Path, fig: plt.Figure) -> None:
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    tree = ET.parse(path)
    registry = getattr(fig, "_aspr_node_ids", {})
    for group in tree.getroot().iter("{http://www.w3.org/2000/svg}g"):
        group_id = group.attrib.get("id")
        if group_id not in registry:
            continue
        elements = list(group.iter("{http://www.w3.org/2000/svg}use"))
        ids = registry[group_id]
        if len(elements) != len(ids):
            raise ValueError(f"SVG node count mismatch for {group_id}: {len(elements)} != {len(ids)}")
        for element, node_id in zip(elements, ids, strict=True):
            safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", node_id)
            element.set("id", f"{group_id}-node-{safe_id}")
    tree.write(path,encoding="utf-8",xml_declaration=True)


def packed_paper_positions(atlas: dict[str, Any]) -> dict[str, np.ndarray]:
    """Pack genuine connected components; retain every isolate in a separate strip."""
    graph = nx.Graph()
    graph.add_nodes_from(n["work_id"] for n in atlas["paper_nodes"])
    graph.add_edges_from(atlas["paper_edges"])
    components = [c for c in atlas["paper_components"] if len(c) > 1]
    pos: dict[str, np.ndarray] = {}
    x, y, height = 0.0, 0.0, 0.0
    for idx, component in enumerate(components):
        scale = .08 + .020 * math.sqrt(len(component))
        if x + scale * 2 > 1.9:
            x, y, height = 0, y + height + .045, 0
        local = nx.spring_layout(graph.subgraph(component), seed=9309 + idx, iterations=100)
        for node, xy in local.items():
            pos[node] = np.array([x + scale + xy[0]*scale*.84, y + scale + xy[1]*scale*.84])
        x += 2*scale + .025
        height = max(height, 2*scale)
    for idx, node in enumerate(sorted(nx.isolates(graph))):
        pos[node] = np.array([.025 + (idx % 60)*.031, -.13 - (idx // 60)*.038])
    return pos


def atlas_positions(atlas: dict[str, Any]) -> tuple[dict[int, np.ndarray], dict[str, np.ndarray]]:
    claims = {n["claim_row"]: np.array([n["layout_x"], n["layout_y"]]) for n in atlas["claim_nodes"]}
    xy = np.array(list(claims.values()))
    minimum, span = xy.min(axis=0), np.ptp(xy, axis=0)
    # One affine projection per layer, applied to nodes, edges and envelopes.
    claims = {k: np.array([((p-minimum)/span)[0]*1.8 + ((p-minimum)/span)[1]*.28,
                           ((p-minimum)/span)[1]*.93 + 1.75]) for k, p in claims.items()}
    papers = packed_paper_positions(atlas)
    pxy = np.array(list(papers.values()))
    top = pxy[:, 1].max()
    papers = {k: np.array([p[0]+max(p[1], 0)*.15, p[1]/top*.92 + .32]) for k, p in papers.items()}
    isolated = sorted(n["work_id"] for n in atlas["paper_nodes"] if n["clean_induced_in_degree"] + n["clean_induced_out_degree"] == 0)
    for index, node in enumerate(isolated):
        papers[node] = np.array([.025 + index%60*.031, .20 - (index//60)*.06])
    return claims, papers


def panel_title(ax: plt.Axes, letter: str, title: str) -> None:
    ax.set_title(f"{letter}  {title}", loc="left", fontweight="bold", pad=8, fontsize=10)


def draw_atlas(ax: plt.Axes, atlas: dict[str, Any], output: Path) -> None:
    claim_pos, paper_pos = atlas_positions(atlas)
    styles = community_style([n["community_id"] for n in atlas["claim_nodes"]])
    for bounds in [[(-.07, 1.69), (1.94, 1.69), (2.24, 2.74), (.22, 2.74)],
                   [(-.07, .26), (1.95, .26), (2.14, 1.30), (.12, 1.30)]]:
        ax.add_patch(Polygon(bounds, fill=False, ec="#dddddd", lw=.65, zorder=0))
    draw_edges(ax, [(e["source"], e["target"]) for e in atlas["claim_edges"]], claim_pos, colors=EDGE, alpha=.43, linewidths=.38, zorder=1)
    node_groups(ax, atlas["claim_nodes"], "claim_row", claim_pos, styles, size=6)
    for index, (u, v) in enumerate(atlas["paper_edges"]):
        arrow = FancyArrowPatch(paper_pos[u], paper_pos[v], arrowstyle="-|>", mutation_scale=3.7,
                                color=PALETTE[2], alpha=.5, lw=.45, shrinkA=1.4, shrinkB=1.4, zorder=2)
        arrow.set_gid(f"citation-{index}-{u}-{v}")
        ax.add_patch(arrow)
    papers_xy = np.array(list(paper_pos.values()))
    paper_artist = ax.scatter(papers_xy[:, 0], papers_xy[:, 1], s=5, c=PALETTE[3], marker="s", linewidths=0, zorder=3)
    register_node_ids(ax,paper_artist,list(paper_pos))
    parent_to_work = {n["nature_article_id"]: n["work_id"] for n in atlas["paper_nodes"]}
    paper_degrees = {n["work_id"]: n["clean_induced_in_degree"] + n["clean_induced_out_degree"] for n in atlas["paper_nodes"]}
    representatives: dict[int, dict[str, Any]] = {}
    for node in sorted(atlas["claim_nodes"], key=lambda n: (-paper_degrees.get(parent_to_work.get(n["parent_paper_id"]), 0), -n["historical_degree"], n["claim_id"])):
        representatives.setdefault(node["community_id"], node)
    selected = sorted(representatives.values(), key=lambda n: (-sum(x["community_id"] == n["community_id"] for x in atlas["claim_nodes"]), n["community_id"]))[:8]
    ownership = []
    for node in selected:
        work = parent_to_work[node["parent_paper_id"]]
        link = FancyArrowPatch(claim_pos[node["claim_row"]], paper_pos[work], connectionstyle="arc3,rad=.06",
                               arrowstyle="-", ls=(0, (1.5, 3)), lw=.65, color=styles[node["community_id"]][0], alpha=.45, zorder=.5)
        link.set_gid(f"ownership-{node['claim_id']}-{work}")
        ax.add_patch(link)
        ownership.append({"claim_id": node["claim_id"], "work_id": work, "community_id": node["community_id"]})
    ax.text(-.035, 2.89, f"Claim Graph  ·  {len(atlas['claim_nodes']):,} claims / {len(atlas['claim_edges']):,} semantic edges", fontsize=8, weight="bold")
    ax.text(-.035, 1.45, f"Paper Graph  ·  {len(atlas['paper_nodes']):,} papers / {len(atlas['paper_edges']):,} citations", fontsize=8, weight="bold")
    ax.text(-.035, -.23, f"{atlas['clean_statistics']['isolated_paper_nodes']} isolated papers in the displayed induced subgraph", fontsize=7.5, color=MUTED)
    ax.text(-.035, -.40, "Solid: historical relation    Dotted: selected claim → parent ownership", fontsize=7.5, color=MUTED)
    ax.text(-.035, -.58, "Claim color + shape: historical community    Paper squares: publications", fontsize=7.5, color=MUTED)
    ax.set(xlim=(-.08, 2.26), ylim=(-.68, 3.02))
    ax.axis("off")
    dump(output / "layouts/fig01_atlas_layout.json", {
        "claim_positions": {str(k): p.tolist() for k, p in claim_pos.items()},
        "paper_positions": {k: p.tolist() for k, p in paper_pos.items()}, "ownership_display_edges": ownership,
        "ownership_selection": "One representative per community, prioritized by induced paper degree then historical claim degree; eight largest communities displayed.",
        "community_encoding": {str(k): {"color": v[0], "marker": v[1]} for k, v in styles.items()},
        "claim_projection": "Frozen historical x/y normalized then affine (1.8*x + .28*y, .93*y + 1.75)",
        "paper_layout": "Each true weak component laid out independently; deterministic size-based packing; all isolates preserved in a strip."})


def local_positions(case: dict[str, Any]) -> tuple[nx.Graph, dict[str, np.ndarray], np.ndarray]:
    graph = nx.Graph()
    graph.add_nodes_from(n["claim_id"] for n in case["graph"]["neighbors"])
    graph.add_edges_from(case["graph"]["neighbor_edges"])
    if len(graph) == 0:
        return graph, {}, np.array([0.0, 0.0])
    components = sorted((sorted(c) for c in nx.connected_components(graph)), key=lambda c: (-len(c), c))
    if len(components[0]) >= 3:
        # The dense component gets readable space instead of collapsing beside an isolate.
        pos = {node: xy*.76 + np.array([.12, -.08]) for node, xy in nx.circular_layout(graph.subgraph(components[0])).items()}
        remaining = [node for component in components[1:] for node in component]
        for index, node in enumerate(remaining):
            angle = math.pi*.85 + index*2*math.pi/max(1, len(remaining))
            pos[node] = np.array([math.cos(angle), math.sin(angle)])
    else:
        pos = nx.circular_layout(graph)
    # Target is chosen after the historical layout and never moves old nodes.
    target = np.array([0.0, 0.0])
    if min(np.linalg.norm(xy-target) for xy in pos.values()) < .25:
        target = np.array([0.0, -.30])
    return graph, pos, target


def draw_local(ax: plt.Axes, case: dict[str, Any], output: Path) -> None:
    graph, pos, target = local_positions(case)
    neighbors = case["graph"]["neighbors"]
    styles = community_style([n["community_id"] for n in neighbors])
    display_pos = {}
    for side, offset in [("Before", -.63), ("After", .63)]:
        local = {node: xy*.42 + np.array([offset, .35]) for node, xy in pos.items()}
        display_pos[side] = {node: xy.tolist() for node, xy in local.items()}
        draw_edges(ax, list(graph.edges), local, colors=EDGE, linewidths=.75, zorder=1)
        if neighbors:
            node_groups(ax, neighbors, "claim_id", local, styles, size=14)
        ax.text(offset, 1.01, side, ha="center", fontsize=7.5, color=MUTED)
        if side == "After":
            target_xy = target*.42 + np.array([offset, .35])
            local["__target__"] = target_xy
            draw_edges(ax, [("__target__", node) for node in graph], local, colors=PALETTE[0], linewidths=.7, linestyles="dashed", alpha=.8, zorder=2)
            ax.scatter(*target_xy, marker="*", s=67, c=PALETTE[0], edgecolors="white", linewidths=.6, zorder=6)
    row = case["row"]
    count, share = row["neighbor_count"], row.get("connected_pair_share")
    ax.text(-1.12, -.37, f"Neighbors {count}/10", fontsize=7.5)
    ax.text(-1.12, -.70, "Connected-pair share", fontsize=7.5)
    ax.plot([-.99, .96], [-.99, -.99], lw=4, color="#ededed", solid_capstyle="round")
    if share is not None:
        ax.plot([-.99, -.99+1.95*share], [-.99, -.99], lw=4, color=PALETTE[3], solid_capstyle="round")
    ax.text(.99, -.67, f"{share:.2f}" if share is not None else "NA", fontsize=7.5, ha="right")
    labels = {
        "C1": ("MLCC stability", "Dense historical neighborhood", "GEAR: bounded search;\nno verified antecedent"),
        "C2": ("Molecular stiffness", "Fragmented neighborhood", "GEAR: partial antecedents;\nremaining scope difference"),
        "C3": ("AGN mass estimates", "No eligible neighbors", "GEAR: unavailable;\nfirstness is unresolved"),
    }
    title, description, evidence = labels[case["display_id"]]
    ax.text(-1.12, 1.52, title, fontsize=8, weight="bold")
    ax.text(-1.12, 1.23, description, fontsize=7.5, color=MUTED)
    ax.text(-1.12, -1.29, evidence, va="top", fontsize=7.5, linespacing=1.35)
    ax.set(xlim=(-1.17, 1.17), ylim=(-1.98, 1.7))
    ax.axis("off")
    dump(output / f"layouts/fig01_{case['display_id']}_layout.json", {
        "claim_id": row["claim_id"], "paper_id": row["paper_id"],
        "history_positions": {node: xy.tolist() for node, xy in pos.items()},
        "display_positions": display_pos, "historical_edges": list(graph.edges),
        "inserted_target": target.tolist(), "inserted_edges": [[row["claim_id"], node] for node in graph],
        "community_encoding": {str(k): {"color": v[0], "marker": v[1]} for k, v in styles.items()}})


def visible_text(fig: plt.Figure) -> list[matplotlib.text.Text]:
    """Return drawn text, excluding virtual off-axis ticks on hidden axes."""
    items = list(fig.texts)
    for ax in fig.axes:
        items.extend(ax.texts)
        items.extend([ax.title, ax._left_title, ax._right_title])
        if ax.axison:
            items.extend([ax.xaxis.label, ax.yaxis.label, ax.xaxis.get_offset_text(), ax.yaxis.get_offset_text()])
            for axis, limits in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
                low, high = sorted(limits)
                for tick in axis.get_major_ticks():
                    if low <= tick.get_loc() <= high:
                        items.extend([tick.label1, tick.label2])
    return [text for text in dict.fromkeys(items) if text.get_visible() and text.get_text()]


def export(fig: plt.Figure, stem: str, output: Path) -> None:
    """Keep the physical page size stable; do not crop-rescale to a tight bbox."""
    fig.canvas.draw()
    text_artists = visible_text(fig)
    fonts = [t.get_fontsize() for t in text_artists]
    renderer = fig.canvas.get_renderer()
    outside = []
    for text in text_artists:
        box = text.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width+1 or box.y1 > fig.bbox.height+1:
            outside.append(text.get_text())
    for suffix in ["png", "svg", "pdf"]:
        fig.savefig(output / f"{stem}.{suffix}", dpi=300)
    identify_svg_nodes(output/f"{stem}.svg",fig)
    dump(output / f"qa/{stem}_physical.json", {"width_mm": fig.get_figwidth()*25.4,
         "height_mm": fig.get_figheight()*25.4, "minimum_font_pt": min(fonts),
         "minimum_font_threshold_pt": 7.5, "font_size_passed": min(fonts) >= 7.5,
         "text_outside_canvas": outside, "text_inside_canvas_passed": not outside,
         "visual_review": "requires rendered-image inspection; typography threshold is not a collision verdict"})
    plt.close(fig)


def fig1(atlas: dict[str, Any], rows: list[dict[str, Any]], cases: list[dict[str, Any]], output: Path) -> None:
    fig = plt.figure(figsize=(180*MM, 231*MM))
    grid = fig.add_gridspec(3, 1, left=.075, right=.97, top=.965, bottom=.085,
                          height_ratios=[1.72, 1.13, .80], hspace=.24)
    atlas_ax = fig.add_subplot(grid[0])
    panel_title(atlas_ax, "a", "A historical atlas with two explicit relation layers")
    draw_atlas(atlas_ax, atlas, output)
    local_outer = fig.add_subplot(grid[1]); local_outer.axis("off")
    panel_title(local_outer, "b", "Fixed history; three different insertion conditions")
    local_grid = grid[1].subgridspec(1, 3, wspace=.12)
    for index, case in enumerate(cases):
        draw_local(fig.add_subplot(local_grid[index]), case, output)
    lower = grid[2].subgridspec(1, 2, width_ratios=[1.35, 1], wspace=.65)
    ax = fig.add_subplot(lower[0]); panel_title(ax, "c", "The local metric space")
    valid = [r for r in rows if r.get("connected_pair_share") is not None]
    points = ax.hexbin([r["neighbor_induced_density"] for r in valid], [r["connected_pair_share"] for r in valid],
                       gridsize=(16, 10), mincnt=1, cmap=CMAP, linewidths=.15)
    ax.set(xlabel="Historical-neighborhood density", ylabel="Newly connected pair share", xlim=(-.035,1.035), ylim=(-.035,1.055))
    ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])
    cb = fig.colorbar(points, ax=ax, fraction=.065, pad=.035); cb.set_label("Claims", fontsize=7.5); cb.solids.set_rasterized(False)
    ax2 = fig.add_subplot(lower[1])
    counts = Counter(r["neighbor_count"] for r in rows)
    ax2.bar(range(11), [counts[n] for n in range(11)], color=[PALETTE[2]]*10+[PALETTE[0]], width=.76)
    ax2.set(xlabel="Retained neighbors", ylabel="Claims", xlim=(-.6,10.6))
    ax2.set_xticks([0,5,10]); ax2.set_yticks([0,600,1200]); ax2.set_ylim(0,1500)
    cap = counts[10]/len(rows)
    ax2.text(.03, .96, f"{counts[10]:,}/{len(rows):,} at cap\n({cap:.1%}); candidates unknown", transform=ax2.transAxes, va="top", fontsize=7.5)
    fig.text(.075, .018, f"200-paper frozen cohort · {len(valid):,} valid pair-share records; {len(rows)-len(valid)} NA (n < 2). Local structure is not a novelty score.", fontsize=7.5, color=MUTED)
    export(fig, "fig01_revised", output)


def metric_heatmap(ax: plt.Axes, rows: list[dict[str, Any]]) -> None:
    roles = sorted({r["claim_type"] for r in rows})
    fields = ["nearest_prior_similarity", "neighbor_induced_density", "connected_pair_share", "cross_boundary_weight_share"]
    labels = ["Nearest\nsimilarity", "Induced\ndensity", "New-pair\nshare", "Cross-community\nweight"]
    medians = np.array([[np.median([r[f] for r in rows if r["claim_type"] == role and r.get(f) is not None]) for f in fields] for role in roles])
    ax.pcolormesh(np.arange(len(fields)+1)-.5, np.arange(len(roles)+1)-.5, medians, vmin=0, vmax=1, cmap=CMAP)
    ax.set_xlim(-.5,len(fields)-.5); ax.set_ylim(len(roles)-.5,-.5)
    ax.set_xticks(range(len(fields)), labels, fontsize=7.5)
    ax.set_yticks(range(len(roles)), [f"{role.title()}\nn={sum(r['claim_type']==role for r in rows)}" for role in roles], fontsize=7.5)
    for i in range(len(roles)):
        for j in range(len(fields)):
            value = medians[i,j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value < .52 else INK, fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


def fig3(rows: list[dict[str, Any]], audit: dict[str, Any], output: Path) -> None:
    fig = plt.figure(figsize=(180*MM, 160*MM))
    grid = fig.add_gridspec(2, 2, left=.16, right=.96, top=.92, bottom=.14, height_ratios=[.8, 1], hspace=.70, wspace=.62)
    ax = fig.add_subplot(grid[0,:]); panel_title(ax, "a", "Role-level descriptions of the recorded local graphs")
    metric_heatmap(ax, rows)
    valid = [r for r in rows if r.get("connected_pair_residual") is not None]
    ax = fig.add_subplot(grid[1,0]); panel_title(ax, "b", "Structural redundancy")
    ax.hexbin([r["neighbor_induced_density"] for r in valid], [r["connected_pair_share"] for r in valid],
              gridsize=(16, 10), mincnt=1, cmap=CMAP, linewidths=.15)
    ax.set(xlabel="Historical-neighborhood density", ylabel="Newly connected pair share", xlim=(-.025,1.025),ylim=(-.025,1.04))
    ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1])
    ax.text(.97,.96,f"Spearman ρ = {audit['rho_density_connected_pair']:.3f}",transform=ax.transAxes,ha="right",va="top",fontsize=7.5)
    ax = fig.add_subplot(grid[1,1]); panel_title(ax,"c", "n,m-matched null")
    residual = np.sort([r["connected_pair_residual"] for r in valid])
    ax.step(residual,np.arange(1,len(residual)+1)/len(residual),where="post",color=PALETTE[2],lw=1.5)
    ax.axvline(0,color=MUTED,lw=.8,ls="--")
    ax.set(xlabel="Observed − G(n,m) mean",ylabel="Cumulative share of claims",ylim=(0,1.02))
    ax.set_yticks([0,.5,1])
    ax.text(.97,.05,f"{len(valid):,} claims\n{audit['null_model']['draws_per_stratum']:,} draws / stratum",transform=ax.transAxes,fontsize=7.5,va="bottom",ha="right")
    fig.text(.07,.055,"Exploratory structure diagnostics · Role labels are extracted claim types; the null controls only n and m.",fontsize=7.5,color=MUTED)
    fig.text(.07,.025,"Independent knowledge-relation / explanation audits are absent. The intended Fig.3 validation remains incomplete.",fontsize=7.5,color=MUTED)
    export(fig,"fig03_descriptive_draft",output)


def build(source: Path, output: Path, mode: str = "draft", draws: int = 2000) -> dict[str, Any]:
    setup_style()
    output.mkdir(parents=True,exist_ok=True)
    atlas_file = source / "fig01_atlas.json"
    atlas, atlas_audit = clean_atlas(load(atlas_file),atlas_file)
    rows, structure_audit = structural_audit(load(source/"fig01_claim_observations.json"),draws)
    cases = load(source/"fig01_selected_cases.json")
    dump(output/"data/fig01_clean_atlas.json",atlas)
    dump(output/"data/fig01_atlas_audit.json",atlas_audit)
    dump(output/"data/fig01_graph_observations_v2.json",rows)
    dump(output/"data/fig03_structural_audit.json",structure_audit)
    profile_fields = ["nearest_prior_similarity", "neighbor_induced_density", "connected_pair_share", "cross_boundary_weight_share"]
    role_profiles = []
    for role in sorted({r["claim_type"] for r in rows}):
        group = [r for r in rows if r["claim_type"] == role]
        for field in profile_fields:
            values = [r[field] for r in group if r.get(field) is not None]
            role_profiles.append({"claim_type":role,"metric":field,"median":float(np.median(values)),
                                  "requested_claims":len(group),"valid_claims":len(values),"missing_claims":len(group)-len(values)})
    dump(output/"data/fig03_role_profiles_v2.json",role_profiles)
    local_audits = []
    for case in cases:
        graph, _, _ = local_positions(case)
        value=connected_pair_share(graph)
        local_audits.append({"claim_id":case["row"]["claim_id"],"nodes":len(graph),"edges":graph.number_of_edges(),
                             "component_sizes":sorted(map(len,nx.connected_components(graph)),reverse=True),
                             "recomputed_connected_pair_share":value,"saved_connected_pair_share":case["row"]["connected_pair_share"],
                             "match": value is None if case["row"]["connected_pair_share"] is None else abs(value-case["row"]["connected_pair_share"])<1e-12})
    dump(output/"data/fig01_selected_case_audit.json",local_audits)
    fig1(atlas,rows,cases,output)
    if mode == "draft":
        fig3(rows,structure_audit,output)
    result = {"module_version":VERSION,"mode":mode,"source_hashes":{name:sha256(source/name) for name in ["fig01_atlas.json","fig01_claim_observations.json","fig01_selected_cases.json"]},
              "fig01":{"rendered":True,"panels":3,"claim_graph_above_paper_graph":True},
              "fig03":{"rendered":mode=="draft","publication_blocked":True,"reason":"Independent content/relationship and explanation verification absent; these panels are descriptive/null-model diagnostics only."},
              "atlas_statistics":atlas_audit["statistics"],"local_case_integrity":all(a["match"] for a in local_audits)}
    dump(output/"data/graph_revision_manifest.json",result)
    (output/"data/graph_panel_captions_zh.md").write_text("""# Fig.1 / Fig.3 逐 panel 读图说明

- Fig.1a：说明历史 claim 语义关系与论文引用关系是不同的图层，先看上层 Claim Graph 再看下层 Paper Graph，实线为各层真实关系、点线为按社区代表性选取的归属链接，底部 356 个孤立论文保留且不表示首次性。
- Fig.1b：说明相同插入规则可遇到密集、碎片化或无邻居三种真实情形，对照每例固定历史位置的 Before/After 与新增虚线，再读邻居数、连通对比例和 GEAR 的已有证据边界。
- Fig.1c：说明局部连通变化与邻域密度高度相关且邻居数受检索上限约束，左侧颜色越亮代表该结构区间 claim 越多，右侧 10 邻居柱表示达到保留上限而非历史证据充分。
- Fig.3a：说明不同抽取角色的局部结构描述，按行读角色、按列读各指标中位数和原始数值，所有指标虽在 0–1 范围但含义不同，角色不等于已核验的知识关系。
- Fig.3b：说明密度与新增连通对占比存在强结构冗余，观察二维点质量的下降轨迹与 Spearman 相关，不能把两个相关量当作两份独立创新证据。
- Fig.3c：说明匹配邻居数 n 与历史边数 m 后的机械连通变化背景，横轴为观察值减 G(n,m) 均值、纵轴为累计 claim 比例，偏离零只表示该粗零模型未解释的结构差异而非科学创新正确性。

Fig.3 为描述与零模型诊断草稿；独立原文关系与解释核验尚缺，publication 模式不导出该验证主图。
""",encoding="utf-8")
    return result


def main() -> dict[str, Any]:
    """Default integration entry point; deliberately does not parse parent CLI args."""
    return build(ROOT/"outputs/FROM_WEB/data",ROOT/"outputs/FROM_WEB_v2")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=ROOT/"outputs/FROM_WEB/data")
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/FROM_WEB_v2")
    parser.add_argument("--mode",choices=["draft","publication"],default="draft")
    parser.add_argument("--draws",type=int,default=2000)
    args=parser.parse_args()
    print(json.dumps(build(args.source,args.output,args.mode,args.draws),ensure_ascii=False,indent=2))

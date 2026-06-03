from __future__ import annotations

import argparse
import math
import os
import random
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/aspr_matplotlib_cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "kg_perturbation_fig2"
DEFAULT_FIG1_DATA_ROOT = PROJECT_ROOT / "outputs" / "kg_perturbation_fig1"

TEXT_DARK = "#111827"
TEXT_MID = "#374151"
TEXT_LIGHT = "#6B7280"
BORDER = "#9CA3AF"
GRID = "#D1D5DB"
PANEL_FACE = "#FFFFFF"


@dataclass(frozen=True)
class MetricSpec:
    """Visual and semantic metadata for one retained Fig. 2 indicator."""

    key: str
    label: str
    color: str
    family: str


@dataclass(frozen=True)
class ScreeningStage:
    """One stage in the 92-to-7 indicator screening funnel."""

    title: str
    n: int
    criterion: str
    removed_examples: str
    color: str


@dataclass(frozen=True)
class CoverageCell:
    """Mechanism coverage level for one indicator-by-mechanism matrix cell."""

    level: str
    delayed: bool = False


@dataclass(frozen=True)
class FingerprintRow:
    """A stylized seven-indicator profile for one paper/control class."""

    label: str
    values: Tuple[float, ...]
    icon: str
    color: str


@dataclass(frozen=True)
class MechanismCluster:
    """A non-redundant mechanism cluster and its representative indicator."""

    label: str
    selected_metric: str
    candidates: Tuple[str, ...]
    color: str
    xy: Tuple[float, float]


@dataclass
class DomainFig1Data:
    """Real Fig. 1 exported data for one domain."""

    slug: str
    works: pd.DataFrame
    paper_edges: pd.DataFrame
    topic_nodes: pd.DataFrame
    topic_edges: pd.DataFrame
    metrics: pd.DataFrame
    anchor_year: int


@dataclass(frozen=True)
class Fig2Data:
    """Container for all conceptual data needed to draw Fig. 2."""

    metrics: List[MetricSpec]
    screening_stages: List[ScreeningStage]
    coverage: Dict[str, Dict[str, CoverageCell]]
    fingerprint_rows: List[FingerprintRow]
    clusters: List[MechanismCluster]
    real_domains: Tuple[DomainFig1Data, ...]
    coverage_strengths: Dict[str, Dict[str, float]]
    metric_correlations: Dict[Tuple[str, str], float]
    metric_correlation_matrix: pd.DataFrame
    drop_one_losses: Dict[str, float]
    effect_sizes: Dict[str, Dict[str, float]]
    group_counts: Dict[str, int]
    validation_weights: Dict[str, float]
    data_note: str


def setup_style() -> None:
    """Configure matplotlib defaults for a compact publication-style figure."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.linewidth": 0.7,
            "axes.edgecolor": BORDER,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "text.color": TEXT_DARK,
        }
    )


def wrap_text(text: str, width: int) -> str:
    """Wrap text without breaking compact metric labels."""
    if not text:
        return ""
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def blend_with_white(color: str, amount: float = 0.82) -> str:
    """Return a light tint of a color by blending it with white."""
    rgb = np.array(mcolors.to_rgb(color), dtype=float)
    out = rgb * (1.0 - amount) + np.ones(3) * amount
    return mcolors.to_hex(out)


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    facecolor: str = "white",
    edgecolor: str = BORDER,
    linewidth: float = 0.8,
    radius: float = 0.018,
    linestyle: str = "-",
    alpha: float = 1.0,
    zorder: int = 1,
) -> mpatches.FancyBboxPatch:
    """Draw a rounded rectangle in panel coordinates."""
    patch = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        transform=ax.transAxes,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        zorder=zorder,
        clip_on=False,
    )
    ax.add_patch(patch)
    return patch


def panel_frame(ax: plt.Axes, label: str, title: str) -> None:
    """Initialize a rounded panel with a panel letter and title."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    rounded_box(ax, 0.0, 0.0, 1.0, 1.0, PANEL_FACE, "#7A7A7A", 0.8, 0.025, zorder=0)
    ax.text(0.022, 0.965, label, ha="left", va="top", fontsize=16, fontweight="bold")
    ax.text(0.100, 0.955, title, ha="left", va="top", fontsize=9.4, fontweight="bold")


def draw_arrow(
    ax: plt.Axes,
    start: Tuple[float, float],
    end: Tuple[float, float],
    color: str = "#4B5563",
    lw: float = 1.1,
    mutation_scale: float = 12.0,
    linestyle: str = "-",
    connectionstyle: str = "arc3,rad=0.0",
    zorder: int = 4,
) -> None:
    """Draw an arrow in panel coordinates."""
    arrow = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
        linestyle=linestyle,
        connectionstyle=connectionstyle,
        shrinkA=1,
        shrinkB=1,
        zorder=zorder,
    )
    ax.add_patch(arrow)


def draw_pill(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    color: str,
    width: float,
    height: float = 0.055,
    fontsize: float = 7.0,
    facecolor: Optional[str] = None,
    fontweight: str = "bold",
    zorder: int = 5,
) -> None:
    """Draw a labeled metric/category pill."""
    fill = facecolor or blend_with_white(color, 0.88)
    rounded_box(ax, x, y, width, height, fill, color, 0.8, radius=height * 0.45, zorder=zorder)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color if fontweight == "bold" else TEXT_DARK,
        fontweight=fontweight,
        zorder=zorder + 1,
    )


def draw_icon_expansion(ax: plt.Axes, cx: float, cy: float, color: str, scale: float = 1.0) -> None:
    """Draw outward arrows indicating expansion."""
    r = 0.022 * scale
    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        draw_arrow(
            ax,
            (cx + 0.25 * r * dx, cy + 0.25 * r * dy),
            (cx + 1.75 * r * dx, cy + 1.75 * r * dy),
            color=color,
            lw=1.0,
            mutation_scale=7 * scale,
            zorder=8,
        )


def draw_icon_bridging(ax: plt.Axes, cx: float, cy: float, color: str, scale: float = 1.0) -> None:
    """Draw a small bridge icon."""
    w = 0.080 * scale
    h = 0.052 * scale
    ax.plot([cx - w / 2, cx + w / 2], [cy - h / 2, cy - h / 2], color=color, lw=1.2, transform=ax.transAxes)
    ax.plot([cx - w / 2, cx - w / 2], [cy - h / 2, cy + h / 2], color=color, lw=1.2, transform=ax.transAxes)
    ax.plot([cx + w / 2, cx + w / 2], [cy - h / 2, cy + h / 2], color=color, lw=1.2, transform=ax.transAxes)
    arc = mpatches.Arc((cx, cy - h * 0.10), w * 0.90, h * 1.65, theta1=200, theta2=340, color=color, lw=1.3, transform=ax.transAxes)
    ax.add_patch(arc)
    for x in np.linspace(cx - w * 0.28, cx + w * 0.28, 3):
        ax.plot([x, x], [cy - h * 0.48, cy + h * 0.18], color=color, lw=0.8, transform=ax.transAxes)


def draw_icon_reconfiguration(ax: plt.Axes, cx: float, cy: float, color: str, scale: float = 1.0) -> None:
    """Draw circular arrows indicating reconfiguration."""
    r = 0.033 * scale
    for angle in [20, 140, 260]:
        theta1 = math.radians(angle)
        theta2 = math.radians(angle + 85)
        start = (cx + r * math.cos(theta1), cy + r * math.sin(theta1))
        end = (cx + r * math.cos(theta2), cy + r * math.sin(theta2))
        draw_arrow(
            ax,
            start,
            end,
            color=color,
            lw=1.0,
            mutation_scale=7 * scale,
            connectionstyle="arc3,rad=0.28",
            zorder=8,
        )


def draw_icon_compression(ax: plt.Axes, cx: float, cy: float, color: str, scale: float = 1.0) -> None:
    """Draw inward arrows indicating compression."""
    r = 0.040 * scale
    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        draw_arrow(
            ax,
            (cx + r * dx, cy + r * dy),
            (cx + 0.22 * r * dx, cy + 0.22 * r * dy),
            color=color,
            lw=1.0,
            mutation_scale=7 * scale,
            zorder=8,
        )


def draw_schematic_graph(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    mode: str,
    seed: int = 42,
) -> None:
    """Draw a small stylized prior/publication/future graph snapshot."""
    rng = np.random.default_rng(seed)
    communities = [
        ((x + 0.27 * w, y + 0.70 * h), "#5BAE52"),
        ((x + 0.70 * w, y + 0.70 * h), "#3478D4"),
        ((x + 0.30 * w, y + 0.32 * h), "#8E44AD"),
        ((x + 0.72 * w, y + 0.32 * h), "#FF7F0E"),
    ]
    all_nodes: List[Tuple[float, float, str]] = []
    for ci, ((cx, cy), color) in enumerate(communities):
        radius = 0.090 * min(w, h) if mode == "future" else 0.115 * min(w, h)
        n_nodes = 6 if mode != "future" else 5
        pts: List[Tuple[float, float]] = []
        for j in range(n_nodes):
            theta = 2 * math.pi * j / n_nodes + 0.25 * ci
            jitter = rng.normal(0, 0.004, size=2)
            px = cx + radius * math.cos(theta) + float(jitter[0])
            py = cy + radius * math.sin(theta) + float(jitter[1])
            pts.append((px, py))
            all_nodes.append((px, py, color))
        for j, p0 in enumerate(pts):
            p1 = pts[(j + 1) % len(pts)]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#9CA3AF", lw=0.45, alpha=0.55, zorder=2)
        if len(pts) > 4:
            ax.plot([pts[0][0], pts[3][0]], [pts[0][1], pts[3][1]], color="#9CA3AF", lw=0.35, alpha=0.40, zorder=2)

    weak_edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    for i, j in weak_edges:
        if mode == "prior" and (i, j) in [(1, 3), (0, 2)]:
            continue
        x0, y0 = communities[i][0]
        x1, y1 = communities[j][0]
        ax.plot([x0, x1], [y0, y1], color="#C7CBD1", lw=0.55, alpha=0.45, linestyle="--", zorder=1)

    if mode in {"publication", "future"}:
        star_x, star_y = x + 0.50 * w, y + 0.52 * h
        refs = [all_nodes[k] for k in [1, 5, 9, 14, 18, 21] if k < len(all_nodes)]
        for px, py, _ in refs:
            ax.plot([star_x, px], [star_y, py], color="#4B5563", lw=0.55, alpha=0.58, zorder=3)
        ax.scatter([star_x], [star_y], marker="*", s=190, color="#EF1D1D", edgecolors="white", linewidths=0.55, zorder=7)
        ax.text(star_x + 0.018, star_y + 0.020, "p", fontsize=8.0, fontstyle="italic", fontweight="bold", zorder=8)

    if mode == "future":
        for i, j in [(0, 3), (1, 2), (0, 1), (1, 3)]:
            x0, y0 = communities[i][0]
            x1, y1 = communities[j][0]
            ax.plot([x0, x1], [y0, y1], color="#64748B", lw=0.75, alpha=0.52, zorder=2)

    xs = [p[0] for p in all_nodes]
    ys = [p[1] for p in all_nodes]
    colors = [p[2] for p in all_nodes]
    ax.scatter(xs, ys, s=18, color=colors, edgecolors="#F8FAFC", linewidths=0.45, zorder=5)


def safe_int(value: object) -> int | None:
    """Convert a possibly missing CSV value to int."""
    if value is None or pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def normalize_layout_positions(topic_nodes: pd.DataFrame, x: float, y: float, w: float, h: float) -> Dict[int, Tuple[float, float]]:
    """Map exported Fig. 1 topic coordinates into a target panel box."""
    if topic_nodes.empty or "x" not in topic_nodes or "y" not in topic_nodes:
        return {}
    xs = topic_nodes["x"].astype(float).to_numpy()
    ys = topic_nodes["y"].astype(float).to_numpy()
    xmin, xmax = float(np.nanmin(xs)), float(np.nanmax(xs))
    ymin, ymax = float(np.nanmin(ys)), float(np.nanmax(ys))
    xspan = xmax - xmin if xmax > xmin else 1.0
    yspan = ymax - ymin if ymax > ymin else 1.0
    pos: Dict[int, Tuple[float, float]] = {}
    for row in topic_nodes.itertuples(index=False):
        comm = safe_int(getattr(row, "community"))
        if comm is None:
            continue
        px = x + 0.14 * w + ((float(getattr(row, "x")) - xmin) / xspan) * 0.72 * w
        py = y + 0.14 * h + ((float(getattr(row, "y")) - ymin) / yspan) * 0.72 * h
        pos[comm] = (px, py)
    return pos


def community_colors(comm_ids: Iterable[int]) -> Dict[int, str]:
    """Assign stable Fig. 1-style tabular colors to displayed communities."""
    palette: List[str] = []
    for cmap_name in ["tab20", "tab20b", "tab20c"]:
        cmap = plt.get_cmap(cmap_name)
        palette.extend(mcolors.to_hex(cmap(i)) for i in range(cmap.N))
    return {comm: palette[i % len(palette)] for i, comm in enumerate(sorted(set(comm_ids)))}


def active_display_works(domain: DomainFig1Data, end_year: int) -> pd.DataFrame:
    """Return real works with a display community and year up to a snapshot end."""
    works = domain.works.copy()
    if "display_community" not in works:
        return pd.DataFrame()
    mask = works["display_community"].notna() & works["year"].notna() & (works["year"].astype(float) <= end_year)
    out = works.loc[mask].copy()
    out["display_community_int"] = out["display_community"].map(safe_int)
    out = out[out["display_community_int"].notna()]
    out["display_community_int"] = out["display_community_int"].astype(int)
    return out


def fig1_disc_points(key: object, n: int, radius: float) -> np.ndarray:
    """Match Fig. 1's central point plus ring bead layout."""
    rng = random.Random(str(key))
    if n <= 0:
        return np.zeros((0, 2), dtype=float)
    pts = []
    for i in range(n):
        if i == 0:
            rr = 0.08 * radius
            theta = rng.random() * 2 * math.pi
        else:
            rr = radius * (0.34 + 0.55 * ((i - 1) / max(1, n - 1)))
            theta = 2 * math.pi * (i - 1) / max(1, n - 1) + rng.uniform(-0.25, 0.25)
        pts.append([rr * math.cos(theta), rr * math.sin(theta)])
    return np.asarray(pts, dtype=float)


def log_scaled_radius(values: Sequence[float], rmin: float, rmax: float) -> List[float]:
    """Match Fig. 1's log-compressed topic halo radius scaling."""
    arr = np.log1p(np.asarray([max(0.0, float(v)) for v in values], dtype=float))
    if len(arr) == 0:
        return []
    lo, hi = float(arr.min()), float(arr.max())
    if abs(hi - lo) < 1e-9:
        return [0.5 * (rmin + rmax)] * len(arr)
    return list(rmin + (arr - lo) / (hi - lo) * (rmax - rmin))


def axes_xy_aspect(ax: plt.Axes) -> float:
    """Return physical x/y scale ratio for axes-coordinate geometry."""
    bbox = ax.get_position()
    return max(1e-9, (bbox.width * ax.figure.get_figwidth()) / (bbox.height * ax.figure.get_figheight()))


def add_axes_circle(
    ax: plt.Axes,
    xy: Tuple[float, float],
    radius: float,
    aspect: float,
    **kwargs: object,
) -> None:
    """Draw a physically circular halo while using axes coordinates."""
    patch = mpatches.Ellipse(
        xy,
        width=2.0 * radius / aspect,
        height=2.0 * radius,
        transform=ax.transAxes,
        **kwargs,
    )
    ax.add_patch(patch)


def build_snapshot_topic_graph(domain: DomainFig1Data, active_comms: set[int], counts: Mapping[int, int]) -> nx.Graph:
    """Build a small topic graph for Fig. 1-style backbone selection."""
    graph = nx.Graph()
    for comm in active_comms:
        graph.add_node(comm, n_papers=int(counts.get(comm, 1)))
    for edge in domain.topic_edges.itertuples(index=False):
        u = safe_int(getattr(edge, "source_community"))
        v = safe_int(getattr(edge, "target_community"))
        if u in active_comms and v in active_comms:
            graph.add_edge(u, v, weight=float(getattr(edge, "weight", 1.0)))
    return graph


def select_fig1_backbone_edges(
    graph: nx.Graph,
    previous_graph: nx.Graph,
    anchor_nodes: set[int],
    max_edges: int,
    extra_edges: int,
) -> List[Tuple[int, int, Mapping[str, float], bool]]:
    """Match Fig. 1's maximum-spanning-tree plus anchor/strong-edge backbone."""
    if graph.number_of_edges() == 0:
        return []
    chosen: Dict[Tuple[int, int], Tuple[int, int, Mapping[str, float]]] = {}

    def add_edge(u: int, v: int, d: Mapping[str, float]) -> None:
        chosen[tuple(sorted((int(u), int(v))))] = (int(u), int(v), d)

    for comp in nx.connected_components(graph):
        sub = graph.subgraph(comp).copy()
        if sub.number_of_edges() == 0:
            continue
        tree = nx.maximum_spanning_tree(sub, weight="weight")
        for u, v, d in tree.edges(data=True):
            add_edge(int(u), int(v), d)

    anchor_edges = []
    for u, v, d in graph.edges(data=True):
        if int(u) in anchor_nodes or int(v) in anchor_nodes:
            anchor_edges.append((float(d.get("weight", 1.0)), int(u), int(v), d))
    for _, u, v, d in sorted(anchor_edges, reverse=True)[: max(3, extra_edges)]:
        add_edge(u, v, d)

    all_edges = sorted(graph.edges(data=True), key=lambda item: float(item[2].get("weight", 1.0)), reverse=True)
    for u, v, d in all_edges:
        add_edge(int(u), int(v), d)
        if len(chosen) >= max_edges:
            break

    previous_edges = {tuple(sorted((int(u), int(v)))) for u, v in previous_graph.edges()}
    out = []
    for key, (u, v, d) in chosen.items():
        out.append((u, v, d, key not in previous_edges))
    out.sort(key=lambda item: (item[3], float(item[2].get("weight", 1.0))), reverse=False)
    return out[:max_edges]


def snapshot_year(domain: DomainFig1Data, mode: str) -> int:
    """Resolve prior, publication-day and future snapshot years."""
    max_year = int(domain.works["year"].dropna().astype(int).max())
    if mode == "prior":
        return max(int(domain.anchor_year) - 1, int(domain.works["year"].dropna().astype(int).min()))
    if mode == "publication":
        return int(domain.anchor_year)
    return min(int(domain.anchor_year) + 5, max_year)


def draw_real_graph_snapshot(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    domain: DomainFig1Data,
    mode: str,
    seed: int,
) -> None:
    """Draw a Fig. 1 style real-data topic snapshot inside a panel box."""
    end_year = snapshot_year(domain, mode)
    active = active_display_works(domain, end_year)
    if active.empty:
        draw_schematic_graph(ax, x, y, w, h, mode=mode, seed=seed)
        return

    topic_nodes = domain.topic_nodes.copy()
    topic_nodes["community_int"] = topic_nodes["community"].map(safe_int)
    active_comms = set(int(v) for v in active["display_community_int"].astype(int))
    topic_nodes = topic_nodes[topic_nodes["community_int"].isin(active_comms)]
    if topic_nodes.empty:
        draw_schematic_graph(ax, x, y, w, h, mode=mode, seed=seed)
        return

    pos = normalize_layout_positions(topic_nodes, x, y, w, h)
    colors = community_colors(pos.keys())
    counts = active.groupby("display_community_int").size().to_dict()
    aspect = axes_xy_aspect(ax)

    previous_end = None
    if mode == "publication":
        previous_end = max(int(domain.anchor_year) - 1, int(domain.works["year"].dropna().astype(int).min()))
    elif mode == "future":
        previous_end = int(domain.anchor_year)
    previous_active = active_display_works(domain, previous_end) if previous_end is not None else pd.DataFrame()
    previous_comms = set(int(v) for v in previous_active["display_community_int"].astype(int)) if not previous_active.empty else set()
    previous_counts = previous_active.groupby("display_community_int").size().to_dict() if not previous_active.empty else {}
    topic_graph = build_snapshot_topic_graph(domain, active_comms, counts)
    previous_graph = build_snapshot_topic_graph(domain, previous_comms, previous_counts)

    anchor_rows = domain.works[domain.works.get("anchor_label", pd.Series(dtype=object)).notna()].copy()
    anchor_rows["display_community_int"] = anchor_rows.get("display_community", pd.Series(dtype=object)).map(safe_int)
    anchor_rows = anchor_rows[anchor_rows["display_community_int"].notna()]
    anchor_nodes = set()
    if mode in {"publication", "future"} and not anchor_rows.empty:
        anchor_nodes.add(int(anchor_rows.sort_values("year").iloc[0]["display_community_int"]))

    selected_edges = select_fig1_backbone_edges(
        topic_graph,
        previous_graph,
        anchor_nodes=anchor_nodes,
        max_edges=8 if mode == "prior" else 13,
        extra_edges=4 if mode == "prior" else 6,
    )
    for idx, (u, v, edge_data, is_new) in enumerate(selected_edges):
        if u not in pos or v not in pos:
            continue
        weight = float(edge_data.get("weight", 1.0))
        lw = 0.35 + 0.70 * min(1.0, math.log1p(weight) / 8.0)
        alpha = 0.28 if not is_new else 0.55
        color = "#9CA3AF" if not is_new else "#3F3F46"
        rad = (0.10 + 0.03 * (idx % 3)) * (-1 if idx % 2 else 1)
        patch = FancyArrowPatch(
            pos[u],
            pos[v],
            transform=ax.transAxes,
            arrowstyle="-",
            connectionstyle=f"arc3,rad={rad}",
            linewidth=lw,
            color=color,
            alpha=alpha,
            shrinkA=6,
            shrinkB=6,
            zorder=2,
        )
        ax.add_patch(patch)

    topic_order = [safe_int(v) for v in topic_nodes["community"].tolist()]
    topic_order = [int(v) for v in topic_order if v is not None and int(v) in pos]
    radii_values = [counts.get(comm, 1) for comm in topic_order]
    radii = {comm: radius for comm, radius in zip(topic_order, log_scaled_radius(radii_values, 0.036, 0.063))}
    for row in topic_nodes.itertuples(index=False):
        comm = safe_int(getattr(row, "community"))
        if comm is None or comm not in pos:
            continue
        cx, cy = pos[comm]
        count = int(counts.get(comm, 1))
        radius = radii.get(comm, 0.046)
        color = colors[comm]
        add_axes_circle(
            ax,
            (cx, cy),
            radius=radius,
            aspect=aspect,
            facecolor=mcolors.to_rgba(color, 0.11 if mode == "prior" else 0.15),
            edgecolor=mcolors.to_rgba(color, 0.72),
            linewidth=1.55 if comm in anchor_nodes else 0.85,
            zorder=3,
        )
        n_beads = max(4, min(7, int(round(3 + math.log1p(count)))))
        offsets = fig1_disc_points(comm, n_beads, radius * 0.66)
        beads = np.column_stack((cx + offsets[:, 0] / aspect, cy + offsets[:, 1]))
        if len(beads) > 2:
            for j in range(1, len(beads)):
                ax.plot([beads[0, 0], beads[j, 0]], [beads[0, 1], beads[j, 1]], color="#9CA3AF", lw=0.42, alpha=0.38, zorder=4)
        ax.scatter(beads[:, 0], beads[:, 1], s=20, color=color, edgecolors="white", linewidths=0.50, alpha=0.94, zorder=5)

    if mode in {"publication", "future"} and not anchor_rows.empty:
        anchor_row = anchor_rows.sort_values("year").iloc[0]
        anchor_comm = int(anchor_row["display_community_int"])
        if anchor_comm in pos:
            star_x, star_y = pos[anchor_comm]
            anchor_id = str(anchor_row["id"])
            neighbor_comms: Dict[int, float] = {}
            if not domain.paper_edges.empty:
                incident = domain.paper_edges[(domain.paper_edges["source"] == anchor_id) | (domain.paper_edges["target"] == anchor_id)]
                id_to_comm = dict(zip(active["id"].astype(str), active["display_community_int"].astype(int)))
                for edge in incident.itertuples(index=False):
                    other = str(getattr(edge, "target")) if str(getattr(edge, "source")) == anchor_id else str(getattr(edge, "source"))
                    comm = id_to_comm.get(other)
                    if comm is not None and comm in pos and comm != anchor_comm:
                        neighbor_comms[comm] = neighbor_comms.get(comm, 0.0) + float(getattr(edge, "weight", 1.0))
            top_neighbors = sorted(neighbor_comms, key=neighbor_comms.get, reverse=True)[:5]
            for comm in top_neighbors:
                px, py = pos[comm]
                ax.plot([star_x, px], [star_y, py], color="#374151", lw=0.55, alpha=0.62, zorder=6)
            add_axes_circle(
                ax,
                pos[anchor_comm],
                radius=radii.get(anchor_comm, 0.046) * 1.28,
                aspect=aspect,
                facecolor="none",
                edgecolor="#DC2626",
                linewidth=0.85,
                linestyle="--",
                alpha=0.55,
                zorder=6,
            )
            ax.scatter([star_x], [star_y], marker="*", s=170, color="#DC2626", edgecolors="white", linewidths=0.70, zorder=8)
            ax.text(star_x + 0.014, star_y + 0.012, "p", fontsize=7.2, fontstyle="italic", fontweight="bold", zorder=9)

    if w >= 0.18:
        labels = topic_nodes.sort_values("n_papers", ascending=False).head(2)
        for row in labels.itertuples(index=False):
            comm = safe_int(getattr(row, "community"))
            if comm is None or comm not in pos:
                continue
            label = str(getattr(row, "label", f"Topic {comm}")).replace(" / ", "\n")
            cx, cy = pos[comm]
            label_y = min(cy + 0.058, y + h - 0.012)
            ax.text(
                cx,
                label_y,
                wrap_text(label, 12),
                ha="center",
                va="bottom",
                fontsize=4.0,
                color=TEXT_DARK,
                linespacing=0.88,
                bbox=dict(boxstyle="round,pad=0.08", facecolor="white", edgecolor="none", alpha=0.70),
                zorder=9,
            )

    ax.text(
        x + 0.012,
        y + 0.012,
        f"{domain.slug.replace('_', ' ')}\n≤{end_year}, n={len(active):,}",
        ha="left",
        va="bottom",
        fontsize=4.5,
        color=TEXT_LIGHT,
        linespacing=0.95,
        zorder=9,
    )


def draw_panel_a(ax: plt.Axes, data: Fig2Data, seed: int) -> None:
    """Draw Panel a: publication-day measurement setting."""
    panel_frame(ax, "a", "Publication-day measurement setting")
    focus_domain = data.real_domains[0] if data.real_domains else None
    graph_w = 0.285
    graph_h = 0.540
    graph_y = 0.292
    boxes = [
        (0.022, graph_y, graph_w, graph_h, "Prior graph G−", "Existing papers\nand communities", "#2E7D32", "prior"),
        (0.358, graph_y, graph_w, graph_h, "Publication-day graph G0", "New paper p and references", "#1D4ED8", "publication"),
        (0.694, graph_y, graph_w, graph_h, "Future graph G+τ", "Graph after τ time", "#6D4C8D", "future"),
    ]
    for x, y, w, h, title, subtitle, color, mode in boxes:
        rounded_box(ax, x, y, w, h, blend_with_white(color, 0.93), color, 0.75, 0.018, zorder=1)
        ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top", fontsize=7.6, fontweight="bold", color=TEXT_DARK)
        ax.text(x + w / 2, y + h - 0.083, subtitle, ha="center", va="top", fontsize=5.9, color=TEXT_DARK, linespacing=1.05)
        if focus_domain is not None:
            draw_real_graph_snapshot(ax, x + 0.018, y + 0.055, w - 0.036, h - 0.150, focus_domain, mode=mode, seed=seed + len(title))
        else:
            draw_schematic_graph(ax, x + 0.018, y + 0.055, w - 0.036, h - 0.150, mode=mode, seed=seed + len(title))

    draw_arrow(ax, (0.310, 0.555), (0.354, 0.555), color="#5F6875", lw=2.1, mutation_scale=22)
    draw_arrow(ax, (0.646, 0.555), (0.690, 0.555), color="#5F6875", lw=2.1, mutation_scale=22)
    ax.text(0.500, 0.223, "Seven indicators computed here", ha="center", va="center", fontsize=6.2, color="#0B4FA3", fontweight="bold")

    pill_widths = [0.060, 0.060, 0.072, 0.072, 0.066, 0.082, 0.064]
    start = 0.232
    gap = 0.010
    x = start
    for metric, width in zip(data.metrics, pill_widths):
        draw_pill(ax, x, 0.165, metric.label, metric.color, width, height=0.050, fontsize=6.4)
        x += width + gap

    principles = [
        ("No future leakage", "Metrics use only information\navailable on publication day.", "#2E7D32", "shield"),
        ("Reference-only", "Computed from reference graph\n(no authors, venues, or text).", "#0B4FA3", "link"),
        ("Graph-grounded", "Each metric maps to a concrete\nperturbation mechanism.", "#6B2A8F", "graph"),
    ]
    for i, (title, desc, color, icon) in enumerate(principles):
        x0 = 0.022 + i * 0.322
        rounded_box(ax, x0, 0.035, 0.300, 0.105, "#FFFFFF", "#CBD5E1", 0.65, 0.015, zorder=2)
        if icon == "shield":
            shield = mpatches.RegularPolygon((x0 + 0.040, 0.087), 5, radius=0.030, orientation=math.pi / 2, transform=ax.transAxes, facecolor=blend_with_white(color, 0.82), edgecolor=color, lw=1.4, zorder=4)
            ax.add_patch(shield)
            ax.text(x0 + 0.040, 0.087, "✓", ha="center", va="center", fontsize=12, color=color, fontweight="bold", zorder=5)
        elif icon == "link":
            ax.add_patch(
                mpatches.Ellipse(
                    (x0 + 0.030, 0.091),
                    0.048,
                    0.025,
                    angle=135,
                    transform=ax.transAxes,
                    facecolor="none",
                    edgecolor=color,
                    lw=1.5,
                    zorder=4,
                )
            )
            ax.add_patch(
                mpatches.Ellipse(
                    (x0 + 0.052, 0.082),
                    0.048,
                    0.025,
                    angle=135,
                    transform=ax.transAxes,
                    facecolor="none",
                    edgecolor=color,
                    lw=1.5,
                    zorder=4,
                )
            )
            ax.plot([x0 + 0.038, x0 + 0.045], [0.088, 0.085], color=color, lw=1.3, transform=ax.transAxes, zorder=5)
        else:
            for dx, dy in [(0, 0.018), (-0.023, -0.010), (0.023, -0.010), (0.0, -0.033)]:
                ax.scatter([x0 + 0.040 + dx], [0.092 + dy], s=16, color=blend_with_white(color, 0.18), edgecolors=color, linewidths=0.5, transform=ax.transAxes, zorder=5)
            ax.plot([x0 + 0.017, x0 + 0.040, x0 + 0.063], [0.082, 0.110, 0.082], color=color, lw=0.8, transform=ax.transAxes, zorder=4)
            ax.plot([x0 + 0.040, x0 + 0.040], [0.110, 0.059], color=color, lw=0.8, transform=ax.transAxes, zorder=4)
        ax.text(x0 + 0.082, 0.110, title, ha="left", va="center", fontsize=6.7, color=color, fontweight="bold")
        ax.text(x0 + 0.082, 0.067, desc, ha="left", va="center", fontsize=5.7, color=TEXT_DARK, linespacing=1.05)


def draw_dot_cloud(ax: plt.Axes, x: float, y: float, w: float, h: float, n: int, seed: int) -> None:
    """Draw a compact deterministic dot cloud for a screening count."""
    rng = np.random.default_rng(seed)
    count = max(6, min(34, int(round(n / 3))))
    pts = rng.uniform([x + 0.020, y + 0.010], [x + w - 0.020, y + h - 0.010], size=(count, 2))
    sizes = rng.uniform(8, 18, size=count)
    ax.scatter(pts[:, 0], pts[:, 1], s=sizes, color="#8C8F96", alpha=0.58, linewidths=0, zorder=4)


def draw_stage_box(ax: plt.Axes, stage: ScreeningStage, index: int, x: float, y: float, w: float, h: float) -> None:
    """Draw one rounded shrinking funnel card."""
    fill = blend_with_white(stage.color, 0.91)
    rounded_box(ax, x, y, w, h, fill, stage.color, 0.9, radius=0.024, zorder=2)
    rounded_box(ax, x + 0.010, y + 0.030, w - 0.020, 0.118, "#FFFFFF", "#D1D5DB", 0.45, radius=0.014, zorder=3)
    stage_label = f"Stage {index + 1}"
    ax.text(x + w / 2, y + h - 0.060, stage_label, ha="center", va="top", fontsize=5.7, color=stage.color, fontweight="bold", zorder=5)
    ax.text(x + w / 2, y + h - 0.128, wrap_text(stage.title, 14), ha="center", va="top", fontsize=5.7, color=TEXT_DARK, fontweight="bold", linespacing=1.03, zorder=5)
    ax.text(x + w / 2, y + 0.390, f"{stage.n}", ha="center", va="center", fontsize=17.0, fontweight="bold", color=TEXT_DARK, zorder=5)
    ax.text(x + w / 2, y + 0.305, "retained metrics", ha="center", va="center", fontsize=4.9, color=TEXT_LIGHT, zorder=5)
    prefix = "source" if index == 0 else "remove"
    ax.text(
        x + w / 2,
        y + 0.089,
        f"{prefix}: {wrap_text(stage.removed_examples, 15)}",
        ha="center",
        va="center",
        fontsize=4.6,
        color=TEXT_MID,
        linespacing=0.98,
        zorder=5,
    )


def draw_panel_b(ax: plt.Axes, data: Fig2Data) -> None:
    """Draw Panel b: multi-stage metric screening."""
    panel_frame(ax, "b", "Multi-stage screening of candidate metrics")
    categories = [
        ("Bibliometrics", "#5B8C4A", 0.115),
        ("Network science", "#4C77B8", 0.145),
        ("Diversity", "#6B4AA1", 0.105),
        ("Novelty", "#7B3FA1", 0.095),
        ("Entropy", "#E85D04", 0.090),
        ("Semantic metrics", "#2A7F7F", 0.145),
    ]
    x = 0.033
    for label, color, width in categories:
        draw_pill(ax, x, 0.845, label, color, width, height=0.054, fontsize=5.8, fontweight="normal")
        x += width + 0.018

    start_x = 0.028
    stage_widths = [0.142, 0.132, 0.122, 0.112, 0.102, 0.092]
    gap = 0.014
    y = 0.070
    h = 0.720
    sx = start_x
    for i, stage in enumerate(data.screening_stages):
        stage_w = stage_widths[i]
        draw_stage_box(ax, stage, i, sx, y, stage_w, h)
        if i < len(data.screening_stages) - 1:
            draw_arrow(ax, (sx + stage_w + 0.002, 0.430), (sx + stage_w + gap - 0.003, 0.430), color="#374151", lw=1.1, mutation_scale=11)
        sx += stage_w + gap

    final_x = 0.902
    ax.text(final_x + 0.050, 0.755, "Final\nbasis", ha="center", va="center", fontsize=7.0, fontweight="bold")
    for i, metric in enumerate(data.metrics):
        draw_pill(ax, final_x, 0.666 - i * 0.078, metric.label, metric.color, 0.080, height=0.052, fontsize=6.0)


def draw_coverage_dot(
    ax: plt.Axes,
    x: float,
    y: float,
    cell: CoverageCell,
    color: str,
) -> None:
    """Draw one coverage marker as a solid or delayed/dashed circle."""
    sizes = {"strong": 180, "moderate": 110, "weak": 58, "weak-moderate": 82}
    alphas = {"strong": 1.0, "moderate": 0.68, "weak": 0.35, "weak-moderate": 0.48}
    size = sizes.get(cell.level, 40)
    alpha = alphas.get(cell.level, 0.4)
    if cell.delayed:
        radius = {180: 0.020, 110: 0.017, 82: 0.015, 58: 0.013}.get(size, 0.014)
        circ = mpatches.Circle(
            (x, y),
            radius=radius,
            transform=ax.transAxes,
            facecolor="white",
            edgecolor="#6B7280",
            linewidth=0.85,
            linestyle=(0, (3, 2)),
            alpha=0.98,
            zorder=5,
        )
        ax.add_patch(circ)
        if cell.level in {"strong", "moderate", "weak-moderate"}:
            ax.scatter([x], [y], s=size * 0.22, color=color, alpha=0.23, linewidths=0, transform=ax.transAxes, zorder=4)
    else:
        ax.scatter([x], [y], s=size, color=color, alpha=alpha, edgecolors="white", linewidths=0.55, transform=ax.transAxes, zorder=5)


def draw_panel_c(ax: plt.Axes, data: Fig2Data) -> None:
    """Draw Panel c: mechanistic coverage matrix."""
    panel_frame(ax, "c", "Mechanistic coverage of the seven-parameter basis")
    columns = [
        ("Expansion", "#2E7D32"),
        ("Bridging", "#0B4FA3"),
        ("Reconfiguration", "#E85D04"),
        ("Delayed\ncompression", "#4B5563"),
    ]
    row_labels = [metric.label for metric in data.metrics]
    row_keys = [metric.key for metric in data.metrics]
    x0, x1 = 0.160, 0.970
    y0, y1 = 0.240, 0.730
    header_y0, header_y1 = 0.748, 0.875
    nrows = len(row_labels)
    ncols = len(columns)
    col_w = (x1 - x0) / ncols
    row_h = (y1 - y0) / nrows

    rounded_box(ax, x0, header_y0, x1 - x0, header_y1 - header_y0, "#F8FAFC", "#E5E7EB", 0.55, radius=0.006, zorder=1)
    for i, (title, color) in enumerate(columns):
        cx = x0 + (i + 0.5) * col_w
        if title == "Expansion":
            draw_icon_expansion(ax, cx, 0.835, color, scale=0.64)
        elif title == "Bridging":
            draw_icon_bridging(ax, cx, 0.834, color, scale=0.55)
        elif title == "Reconfiguration":
            draw_icon_reconfiguration(ax, cx, 0.834, color, scale=0.58)
        else:
            draw_icon_compression(ax, cx, 0.834, color, scale=0.55)
        ax.text(cx, 0.780, title, ha="center", va="center", fontsize=5.2, color=color, fontweight="bold", linespacing=0.92)

    for i in range(ncols + 1):
        xx = x0 + i * col_w
        ax.plot([xx, xx], [y0, header_y1], color=GRID, lw=0.55, transform=ax.transAxes, zorder=1)
    for j in range(nrows + 1):
        yy = y1 - j * row_h
        ax.plot([0.060, x1], [yy, yy], color=GRID, lw=0.55, transform=ax.transAxes, zorder=1)
    ax.plot([0.060, x1], [header_y0, header_y0], color=GRID, lw=0.55, transform=ax.transAxes, zorder=1)

    for r, (label, key) in enumerate(zip(row_labels, row_keys)):
        cy = y1 - (r + 0.5) * row_h
        ax.text(0.085, cy, label, ha="left", va="center", fontsize=6.5, fontweight="bold")
        for c, (title, color) in enumerate(columns):
            cx = x0 + (c + 0.5) * col_w
            lookup_title = "Compression" if "compression" in title.lower() else title
            draw_coverage_dot(ax, cx, cy, data.coverage[key][lookup_title], color)

    legend_y = 0.115
    legend_items = [("Strong", 95, 1.0), ("Moderate", 58, 0.68), ("Weak", 34, 0.35)]
    lx = 0.095
    for color, heading in [("#2E7D32", "Expansion"), ("#0B4FA3", "Bridging"), ("#E85D04", "Reconfiguration")]:
        for j, (label, size, alpha) in enumerate(legend_items):
            ax.scatter([lx], [legend_y - j * 0.040], s=size, color=color, alpha=alpha, edgecolors="white", linewidths=0.4, transform=ax.transAxes)
            ax.text(lx + 0.032, legend_y - j * 0.040, label, ha="left", va="center", fontsize=5.0)
        lx += 0.190
    circ = mpatches.Circle((0.690, legend_y - 0.010), radius=0.017, transform=ax.transAxes, facecolor="white", edgecolor="#6B7280", linewidth=0.8, linestyle=(0, (3, 2)))
    ax.add_patch(circ)
    ax.text(0.722, legend_y - 0.010, "Moderate\n(delayed)", ha="left", va="center", fontsize=5.0, linespacing=1.0)
    circ2 = mpatches.Circle((0.850, legend_y - 0.010), radius=0.014, transform=ax.transAxes, facecolor="white", edgecolor="#6B7280", linewidth=0.8, linestyle=(0, (3, 2)))
    ax.add_patch(circ2)
    ax.text(0.882, legend_y - 0.010, "Weak-moderate\n(delayed)", ha="left", va="center", fontsize=5.0, linespacing=1.0)


def draw_cluster(ax: plt.Axes, cluster: MechanismCluster, metric_lookup: Mapping[str, MetricSpec]) -> None:
    """Draw one mechanism cluster with representative and candidate nodes."""
    cx, cy = cluster.xy
    halo = mpatches.Ellipse(
        (cx, cy),
        width=0.245,
        height=0.245,
        transform=ax.transAxes,
        facecolor=blend_with_white(cluster.color, 0.93),
        edgecolor=cluster.color,
        linewidth=0.65,
        linestyle=(0, (2, 2)),
        alpha=0.92,
        zorder=1,
    )
    ax.add_patch(halo)

    G = nx.Graph()
    G.add_node(cluster.selected_metric)
    for cand in cluster.candidates:
        G.add_edge(cluster.selected_metric, cand)
    angles = np.linspace(math.pi * 0.12, math.pi * 2.12, len(cluster.candidates), endpoint=False)
    cand_pos: Dict[str, Tuple[float, float]] = {}
    for cand, angle in zip(cluster.candidates, angles):
        cand_pos[cand] = (cx + 0.090 * math.cos(angle), cy + 0.082 * math.sin(angle))
    for _, cand in G.edges(cluster.selected_metric):
        px, py = cand_pos[cand]
        ax.plot([cx, px], [cy, py], color="#9CA3AF", lw=0.55, alpha=0.70, transform=ax.transAxes, zorder=2)
        ax.scatter([px], [py], s=20, color="#9CA3AF", edgecolors="white", linewidths=0.35, transform=ax.transAxes, zorder=3)
        ax.text(px, py + 0.026, wrap_text(cand, 10), ha="center", va="bottom", fontsize=4.9, color=TEXT_MID, linespacing=0.95)

    metric = metric_lookup[cluster.selected_metric]
    ax.scatter([cx], [cy], s=410, color=metric.color, edgecolors="white", linewidths=0.9, transform=ax.transAxes, zorder=5)
    center_label = "Burt\nIP" if metric.label == "Burt IP" else metric.label
    center_size = 6.4 if "\n" in center_label else 7.4
    ax.text(cx, cy, center_label, ha="center", va="center", fontsize=center_size, color="white", fontweight="bold", linespacing=0.92, zorder=6)
    ax.text(cx, cy - 0.132, wrap_text(cluster.label, 18), ha="center", va="top", fontsize=5.2, color=metric.color, fontweight="bold", linespacing=0.98)


def draw_panel_d(ax: plt.Axes, data: Fig2Data) -> None:
    """Draw Panel d: redundancy heatmap plus unique contribution bars."""
    panel_frame(ax, "d", "Redundancy and complementarity")
    metric_lookup = {metric.key: metric for metric in data.metrics}
    order = [key for key in REDUNDANCY_METRIC_ORDER if key in metric_lookup]
    labels = [metric_lookup[key].label for key in order]
    corr = data.metric_correlation_matrix.reindex(index=order, columns=order) if not data.metric_correlation_matrix.empty else pd.DataFrame(np.eye(len(order)), index=order, columns=order)

    ax.text(0.070, 0.875, "Pairwise indicator correlation", ha="left", va="center", fontsize=7.2, fontweight="bold")
    left, bottom, size = 0.105, 0.230, 0.505
    n = len(order)
    cell = size / n
    cmap = plt.get_cmap("RdBu_r")
    norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
    for r, row_key in enumerate(order):
        for c, col_key in enumerate(order):
            value = float(corr.loc[row_key, col_key]) if row_key in corr.index and col_key in corr.columns else (1.0 if row_key == col_key else 0.0)
            x = left + c * cell
            y = bottom + (n - 1 - r) * cell
            ax.add_patch(mpatches.Rectangle((x, y), cell, cell, transform=ax.transAxes, facecolor=cmap(norm(value)), edgecolor="white", linewidth=0.70, zorder=2))
            ax.text(x + cell / 2, y + cell / 2, f"{value:.2f}", ha="center", va="center", fontsize=4.8, color=TEXT_DARK if abs(value) < 0.68 else "white", fontweight="bold" if r == c else "normal", zorder=3)
    rounded_box(ax, left, bottom, size, size, "none", "#9CA3AF", 0.75, radius=0.004, zorder=4)
    for i, label in enumerate(labels):
        ax.text(left - 0.020, bottom + (n - 0.5 - i) * cell, label, ha="right", va="center", fontsize=5.5, color=metric_lookup[order[i]].color, fontweight="bold")
        ax.text(left + (i + 0.5) * cell, bottom + size + 0.020, label, ha="center", va="bottom", fontsize=5.2, color=metric_lookup[order[i]].color, fontweight="bold", rotation=45)

    cb_x, cb_y, cb_w, cb_h = left + 0.030, 0.145, size - 0.060, 0.028
    for i in range(90):
        frac = i / 89
        ax.add_patch(mpatches.Rectangle((cb_x + frac * cb_w, cb_y), cb_w / 90.0, cb_h, transform=ax.transAxes, facecolor=cmap(frac), edgecolor="none", zorder=2))
    ax.text(cb_x, cb_y - 0.016, "-1", ha="center", va="top", fontsize=5.0)
    ax.text(cb_x + cb_w / 2, cb_y - 0.016, "Spearman ρ", ha="center", va="top", fontsize=5.0, color=TEXT_LIGHT)
    ax.text(cb_x + cb_w, cb_y - 0.016, "+1", ha="center", va="top", fontsize=5.0)

    ax.text(0.670, 0.875, "Drop-one unique contribution", ha="left", va="center", fontsize=7.2, fontweight="bold")
    bar_left, bar_right = 0.700, 0.955
    bar_top, bar_step = 0.780, 0.075
    max_loss = max(data.drop_one_losses.values()) if data.drop_one_losses else 0.0
    max_loss = max(max_loss, 1e-6)
    for i, key in enumerate(order):
        y = bar_top - i * bar_step
        loss = float(data.drop_one_losses.get(key, 0.0))
        width = (bar_right - bar_left) * (loss / max_loss)
        ax.text(bar_left - 0.020, y, metric_lookup[key].label, ha="right", va="center", fontsize=5.8, color=metric_lookup[key].color, fontweight="bold")
        rounded_box(ax, bar_left, y - 0.020, bar_right - bar_left, 0.038, "#F3F4F6", "#E5E7EB", 0.35, radius=0.010, zorder=1)
        rounded_box(ax, bar_left, y - 0.020, width, 0.038, blend_with_white(metric_lookup[key].color, 0.45), metric_lookup[key].color, 0.35, radius=0.010, zorder=2)
        ax.text(bar_left + width + 0.008, y, f"{loss:.2f}", ha="left", va="center", fontsize=5.2, color=TEXT_MID)
    ax.text(0.700, 0.160, "Bar length = validation-correlation loss when the\nindicator is removed from the composite score.", ha="left", va="bottom", fontsize=5.2, color=TEXT_LIGHT, linespacing=1.02)


def draw_row_icon(ax: plt.Axes, x: float, y: float, icon: str, color: str) -> None:
    """Draw a compact row icon for the fingerprint heatmap."""
    if icon == "star":
        ax.scatter([x], [y], marker="*", s=180, color=color, edgecolors="white", linewidths=0.55, transform=ax.transAxes, zorder=5)
    elif icon == "chart":
        for i, height in enumerate([0.026, 0.046, 0.070]):
            ax.add_patch(mpatches.Rectangle((x - 0.027 + i * 0.022, y - 0.035), 0.012, height, transform=ax.transAxes, facecolor=blend_with_white(color, 0.25), edgecolor=color, lw=0.5, zorder=4))
        draw_arrow(ax, (x - 0.036, y - 0.020), (x + 0.036, y + 0.043), color=color, lw=0.9, mutation_scale=8, zorder=5)
    elif icon == "book":
        ax.add_patch(mpatches.Rectangle((x - 0.032, y - 0.040), 0.028, 0.075, transform=ax.transAxes, facecolor=blend_with_white(color, 0.80), edgecolor=color, lw=0.7))
        ax.add_patch(mpatches.Rectangle((x + 0.004, y - 0.040), 0.028, 0.075, transform=ax.transAxes, facecolor=blend_with_white(color, 0.88), edgecolor=color, lw=0.7))
        ax.plot([x, x], [y - 0.040, y + 0.035], color=color, lw=0.7, transform=ax.transAxes)
    elif icon == "paper":
        ax.add_patch(mpatches.Rectangle((x - 0.026, y - 0.039), 0.052, 0.078, transform=ax.transAxes, facecolor="white", edgecolor=color, lw=0.75))
        for k in range(4):
            ax.plot([x - 0.015, x + 0.017], [y + 0.020 - k * 0.017, y + 0.020 - k * 0.017], color="#9CA3AF", lw=0.55, transform=ax.transAxes)
    else:
        xs = np.linspace(x - 0.035, x + 0.035, 26)
        ys = y + 0.014 * np.sin(np.linspace(0, 5 * math.pi, 26))
        ax.plot(xs, ys, color=color, lw=1.0, transform=ax.transAxes)
        for dx in [-0.024, 0.0, 0.024]:
            ax.plot([x + dx, x + dx], [y - 0.030, y + 0.030], color="#9CA3AF", lw=0.45, transform=ax.transAxes)


def draw_panel_e(ax: plt.Axes, data: Fig2Data) -> None:
    """Draw Panel e: empirical effect-size forest plot."""
    panel_frame(ax, "e", "Empirical support for retained indicators")
    metric_lookup = {metric.key: metric for metric in data.metrics}
    order = [key for key in REDUNDANCY_METRIC_ORDER if key in metric_lookup]
    domain_names = [domain.slug for domain in data.real_domains]

    ax.text(0.085, 0.880, "Discrimination of peak perturbation windows vs controls", ha="left", va="center", fontsize=7.0, fontweight="bold")
    plot_left, plot_right = 0.270, 0.940
    y_top, row_step = 0.790, 0.082
    x_min, x_max = 0.48, 1.02

    def map_x(value: float) -> float:
        clipped = max(x_min, min(x_max, value))
        return plot_left + (clipped - x_min) / (x_max - x_min) * (plot_right - plot_left)

    chance_x = map_x(0.5)
    ax.plot([chance_x, chance_x], [0.205, 0.815], color="#6B7280", lw=0.75, linestyle=(0, (3, 3)), transform=ax.transAxes, zorder=1)
    for tick in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        if x_min <= tick <= x_max:
            tx = map_x(tick)
            ax.plot([tx, tx], [0.205, 0.815], color="#E5E7EB", lw=0.45, transform=ax.transAxes, zorder=0)
            ax.text(tx, 0.178, f"{tick:.1f}", ha="center", va="top", fontsize=5.2, color=TEXT_LIGHT)
    ax.text((plot_left + plot_right) / 2, 0.130, "Direction-free AUROC (0.5 = random; higher = stronger separation)", ha="center", va="center", fontsize=5.5, color=TEXT_MID)

    domain_palette = ["#64748B", "#94A3B8", "#475569", "#CBD5E1", "#334155"]
    for i, key in enumerate(order):
        y = y_top - i * row_step
        metric = metric_lookup[key]
        ax.text(0.220, y, metric.label, ha="right", va="center", fontsize=6.3, color=metric.color, fontweight="bold")
        ax.plot([plot_left, plot_right], [y, y], color="#F1F5F9", lw=0.75, transform=ax.transAxes, zorder=0)
        effects = data.effect_sizes.get(key, {})
        for j, domain in enumerate(domain_names):
            value = float(effects.get(domain, 0.0))
            jitter = (j - (len(domain_names) - 1) / 2) * 0.008
            ax.scatter([map_x(value)], [y + jitter], s=22, color=domain_palette[j % len(domain_palette)], edgecolors="white", linewidths=0.45, transform=ax.transAxes, zorder=3)
        overall = float(effects.get("overall", 0.0))
        diamond_x = map_x(overall)
        diamond = mpatches.RegularPolygon((diamond_x, y), numVertices=4, radius=0.018, orientation=math.pi / 4, transform=ax.transAxes, facecolor=metric.color, edgecolor="white", linewidth=0.60, zorder=4)
        ax.add_patch(diamond)
        ax.text(min(plot_right + 0.012, 0.970), y, f"{overall:.2f}", ha="left", va="center", fontsize=5.2, color=TEXT_MID)

    legend_y = 0.075
    ax.scatter([0.280], [legend_y], s=22, color="#64748B", edgecolors="white", linewidths=0.45, transform=ax.transAxes, zorder=3)
    ax.text(0.300, legend_y, "domain", ha="left", va="center", fontsize=5.3, color=TEXT_MID)
    ax.add_patch(mpatches.RegularPolygon((0.405, legend_y), numVertices=4, radius=0.015, orientation=math.pi / 4, transform=ax.transAxes, facecolor="#111827", edgecolor="white", linewidth=0.50, zorder=4))
    ax.text(0.427, legend_y, "overall", ha="left", va="center", fontsize=5.3, color=TEXT_MID)
    counts = data.group_counts or {"target": 0, "control": 0}
    ax.text(0.575, legend_y, f"target n={counts.get('target', 0)}; control n={counts.get('control', 0)}", ha="left", va="center", fontsize=5.3, color=TEXT_LIGHT)


def draw_normal_curve(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    """Draw a tiny field-year normalization curve."""
    xs = np.linspace(-3, 3, 100)
    ys = np.exp(-0.5 * xs**2)
    xs2 = x + (xs + 3.0) / 6.0 * w
    ys2 = y + ys / ys.max() * h
    ax.plot(xs2, ys2, color="#111827", lw=0.8, transform=ax.transAxes, zorder=5)
    ax.plot([x, x + w], [y, y], color="#111827", lw=0.55, transform=ax.transAxes, zorder=5)
    ax.plot([x + 0.50 * w, x + 0.50 * w], [y, y + 0.78 * h], color="#9CA3AF", lw=0.55, linestyle="--", transform=ax.transAxes, zorder=5)


def flow_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str = "",
    edgecolor: str = "#9CA3AF",
) -> None:
    """Draw a standard flow-chart box."""
    rounded_box(ax, x, y, w, h, "#FFFFFF", edgecolor, 0.75, 0.018, zorder=2)
    ax.text(x + w / 2, y + h - 0.034, title, ha="center", va="top", fontsize=6.2, fontweight="bold", linespacing=1.04, zorder=5)
    if subtitle:
        ax.text(x + w / 2, y + h - 0.085, subtitle, ha="center", va="top", fontsize=5.6, color=TEXT_DARK, linespacing=1.04, zorder=5)


def draw_panel_f(ax: plt.Axes, data: Fig2Data) -> None:
    """Draw Panel f: complete scoring, validation, and robustness framework."""
    panel_frame(ax, "f", "From indicators to evaluation framework")
    ax.text(0.035, 0.895, "Score construction", ha="left", va="center", fontsize=7.0, fontweight="bold")
    ax.text(0.035, 0.435, "Validation and robustness", ha="left", va="center", fontsize=7.0, fontweight="bold")

    top_y, top_h = 0.575, 0.275
    boxes = [
        (0.025, top_y, 0.165, top_h, "Seven indicators\n(G0)", ""),
        (0.235, top_y, 0.135, top_h, "Field-year\nnormalization", ""),
        (0.415, top_y, 0.215, top_h, "Mechanism scores\n(7 → 4)", ""),
        (0.675, top_y, 0.145, top_h, "Composite\nscores", ""),
    ]
    for x, y, w, h, title, subtitle in boxes:
        flow_box(ax, x, y, w, h, title, subtitle)

    pill_positions = [(0.045, 0.765), (0.112, 0.765), (0.045, 0.715), (0.112, 0.715), (0.045, 0.665), (0.112, 0.665), (0.079, 0.615)]
    pill_widths = [0.055, 0.055, 0.065, 0.060, 0.055, 0.070, 0.060]
    for metric, (px, py), pw in zip(data.metrics, pill_positions, pill_widths):
        draw_pill(ax, px, py, metric.label, metric.color, pw, height=0.036, fontsize=5.2)

    draw_normal_curve(ax, 0.267, 0.660, 0.070, 0.075)

    mechanisms = [
        ("Expansion", "#2E7D32", "exp"),
        ("Bridging", "#0B4FA3", "bridge"),
        ("Reconfiguration", "#E85D04", "reconfig"),
        ("Delayed\ncompression", "#4B5563", "comp"),
    ]
    for i, (label, color, icon) in enumerate(mechanisms):
        yy = 0.765 - i * 0.047
        rounded_box(ax, 0.438, yy, 0.165, 0.036, blend_with_white(color, 0.91), color, 0.50, radius=0.009, zorder=4)
        if icon == "exp":
            draw_icon_expansion(ax, 0.455, yy + 0.018, color, 0.30)
        elif icon == "bridge":
            draw_icon_bridging(ax, 0.455, yy + 0.018, color, 0.28)
        elif icon == "reconfig":
            draw_icon_reconfiguration(ax, 0.455, yy + 0.018, color, 0.30)
        else:
            draw_icon_compression(ax, 0.455, yy + 0.018, color, 0.28)
        ax.text(0.535, yy + 0.018, label, ha="center", va="center", fontsize=5.2, color=color, fontweight="bold", linespacing=0.88)

    score_cards = [("Novelty\npotential", "#6B2A8F"), ("Importance\npotential", "#2C7FB8")]
    for i, (label, color) in enumerate(score_cards):
        yy = 0.740 - i * 0.085
        rounded_box(ax, 0.698, yy, 0.095, 0.060, blend_with_white(color, 0.88), color, 0.70, radius=0.012, zorder=4)
        ax.text(0.746, yy + 0.030, label, ha="center", va="center", fontsize=5.4, color=color, fontweight="bold", linespacing=0.92)

    for start, end in [((0.190, 0.710), (0.235, 0.710)), ((0.370, 0.710), (0.415, 0.710)), ((0.630, 0.710), (0.675, 0.710))]:
        draw_arrow(ax, start, end, color="#111827", lw=1.1, mutation_scale=12)

    rounded_box(ax, 0.835, 0.655, 0.140, 0.145, "#FFFFFF", "#9CA3AF", 0.70, radius=0.014, zorder=4)
    ax.text(0.905, 0.755, r"$N(p)=\sum_k w_k^{N}z_k(p)$", ha="center", va="center", fontsize=6.7, fontstyle="italic", zorder=5)
    ax.text(0.905, 0.705, r"$I(p)=\sum_k w_k^{I}z_k(p)$", ha="center", va="center", fontsize=6.7, fontstyle="italic", zorder=5)
    if data.validation_weights:
        top_weights = sorted(data.validation_weights.items(), key=lambda item: item[1], reverse=True)[:3]
        ax.text(0.905, 0.668, "empirical w: " + ", ".join(f"{k} {v:.2f}" for k, v in top_weights), ha="center", va="center", fontsize=4.5, color=TEXT_LIGHT, zorder=5)

    lower = [
        (0.040, 0.105, 0.245, 0.270, "Future validation", ["diffusion / adoption", "citation impact", "graph reconfiguration", "expert / landmark labels"], "#7B3FA1"),
        (0.350, 0.105, 0.245, 0.270, "Robustness checks", ["matched controls", "cross-domain validation", "temporal robustness", "drop-one ablation"], "#0B4FA3"),
        (0.660, 0.105, 0.245, 0.270, "Model calibration", ["learn weights w", "audit residual errors", "report uncertainty", "update score rules"], "#E85D04"),
    ]
    for x, y, w, h, title, items, color in lower:
        rounded_box(ax, x, y, w, h, "#FFFFFF", color, 0.70, radius=0.016, zorder=2)
        ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top", fontsize=6.3, color=color, fontweight="bold", zorder=5)
        for i, item in enumerate(items):
            yy = y + h - 0.085 - i * 0.047
            ax.scatter([x + 0.030], [yy], s=16, color=blend_with_white(color, 0.20), edgecolors=color, linewidths=0.45, transform=ax.transAxes, zorder=5)
            ax.text(x + 0.052, yy, item, ha="left", va="center", fontsize=5.3, color=TEXT_DARK, zorder=5)

    for start, end in [((0.285, 0.240), (0.350, 0.240)), ((0.595, 0.240), (0.660, 0.240))]:
        draw_arrow(ax, start, end, color="#4B5563", lw=1.0, mutation_scale=11)
    draw_arrow(ax, (0.785, 0.375), (0.745, 0.575), color="#6B7280", lw=0.85, mutation_scale=10, linestyle=(0, (4, 3)), connectionstyle="arc3,rad=-0.28")


METRIC_PROXY_COLUMNS: Dict[str, str] = {
    "B": "B_proxy_raw",
    "RS": "RS_proxy_raw",
    "DeltaQ": "DeltaQ_directionality_raw",
    "Uzzi": "Uzzi_proxy_raw",
    "RTD": "RTD_proxy_raw",
    "BurtIP": "BurtIP_proxy_raw",
    "PDE": "PDE_proxy_raw",
}

MECHANISM_INDEX_COLUMNS: Dict[str, str] = {
    "Expansion": "Expansion_index",
    "Bridging": "Bridging_index",
    "Reconfiguration": "Reconfiguration_index",
    "Compression": "Compression_index",
}

REDUNDANCY_METRIC_ORDER = ["B", "BurtIP", "RTD", "RS", "PDE", "Uzzi", "DeltaQ"]


def load_fig1_domain_data(data_root: Path, focus_domain: str = "crispr") -> Tuple[DomainFig1Data, ...]:
    """Load real Fig. 1 exports that can drive Fig. 2 panels."""
    required = {
        "works": "works_selected.csv",
        "paper_edges": "paper_edges.csv",
        "topic_nodes": "topic_nodes.csv",
        "topic_edges": "topic_edges.csv",
        "metrics": "perturbation_metrics.csv",
    }
    domains: List[DomainFig1Data] = []
    if not data_root.exists():
        return tuple()
    for domain_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        if not all((domain_dir / name).exists() for name in required.values()):
            continue
        works = pd.read_csv(domain_dir / required["works"])
        paper_edges = pd.read_csv(domain_dir / required["paper_edges"])
        topic_nodes = pd.read_csv(domain_dir / required["topic_nodes"])
        topic_edges = pd.read_csv(domain_dir / required["topic_edges"])
        metrics = pd.read_csv(domain_dir / required["metrics"])
        anchors = works[works.get("anchor_label", pd.Series(dtype=object)).notna()]
        if anchors.empty or anchors["year"].dropna().empty:
            anchor_year = int(metrics["rolling_end"].median())
        else:
            anchor_year = int(anchors["year"].dropna().astype(int).min())
        domains.append(
            DomainFig1Data(
                slug=domain_dir.name,
                works=works,
                paper_edges=paper_edges,
                topic_nodes=topic_nodes,
                topic_edges=topic_edges,
                metrics=metrics,
                anchor_year=anchor_year,
            )
        )
    if not domains:
        return tuple()
    domains.sort(key=lambda item: (0 if item.slug == focus_domain else 1, item.slug))
    return tuple(domains)


def robust_minmax(values: Sequence[float]) -> np.ndarray:
    """Normalize numeric values to [0, 1] with zero protection."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    arr = np.where(np.isfinite(arr), arr, np.nan)
    if np.isnan(arr).all():
        return np.zeros_like(arr)
    fill = float(np.nanmedian(arr))
    arr = np.where(np.isfinite(arr), arr, fill)
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if hi - lo < 1e-12:
        return np.full_like(arr, 0.5, dtype=float)
    return (arr - lo) / (hi - lo)


def real_metric_window_table(domains: Sequence[DomainFig1Data], metrics: Sequence[MetricSpec]) -> pd.DataFrame:
    """Build a normalized seven-indicator table from real Fig. 1 domain windows."""
    rows: List[Dict[str, object]] = []
    for domain in domains:
        df = domain.metrics.copy()
        for _, row in df.iterrows():
            record: Dict[str, object] = {
                "domain": domain.slug,
                "window_index": int(row.get("window_index", len(rows) + 1)),
                "window": f"{int(row['rolling_start'])}-{int(row['rolling_end'])}",
                "n_rolling_papers": float(row.get("n_rolling_papers", np.nan)),
            }
            for metric in metrics:
                col = METRIC_PROXY_COLUMNS[metric.key]
                val = float(row.get(col, np.nan))
                if metric.key == "DeltaQ":
                    val = abs(val)
                record[metric.key] = val
            for mechanism, col in MECHANISM_INDEX_COLUMNS.items():
                record[mechanism] = float(row.get(col, np.nan))
            rows.append(record)
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    for metric in metrics:
        table[f"{metric.key}_norm"] = robust_minmax(table[metric.key].to_numpy(dtype=float))
    for mechanism in MECHANISM_INDEX_COLUMNS:
        table[f"{mechanism}_norm"] = robust_minmax(table[mechanism].to_numpy(dtype=float))
    norm_cols = [f"{metric.key}_norm" for metric in metrics]
    table["perturbation_composite"] = table[norm_cols].mean(axis=1)
    mech_cols = [f"{mechanism}_norm" for mechanism in MECHANISM_INDEX_COLUMNS]
    table["validation_composite"] = table[mech_cols].mean(axis=1)
    return table


def pearson_abs(x: Sequence[float], y: Sequence[float]) -> float:
    """Return finite absolute Pearson correlation, or zero when undefined."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    if mask.sum() < 3:
        return 0.0
    xa = xa[mask]
    ya = ya[mask]
    if float(np.std(xa)) < 1e-12 or float(np.std(ya)) < 1e-12:
        return 0.0
    return float(abs(np.corrcoef(xa, ya)[0, 1]))


def spearman_corr_matrix(table: pd.DataFrame, metrics: Sequence[MetricSpec]) -> pd.DataFrame:
    """Compute signed Spearman correlations for retained indicators."""
    if table.empty:
        return pd.DataFrame()
    cols = [f"{metric.key}_norm" for metric in metrics]
    labels = [metric.key for metric in metrics]
    ranked = table[cols].rank(axis=0, method="average")
    corr = ranked.corr(method="pearson").fillna(0.0)
    corr.index = labels
    corr.columns = labels
    for key in labels:
        corr.loc[key, key] = 1.0
    return corr


def target_control_masks(table: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Define peak perturbation windows and low-perturbation controls."""
    if table.empty:
        return pd.Series(dtype=bool), pd.Series(dtype=bool)
    high = float(table["perturbation_composite"].quantile(0.75))
    low = float(table["perturbation_composite"].quantile(0.25))
    target = table["perturbation_composite"] >= high
    control = table["perturbation_composite"] <= low
    return target, control


def cohen_d(target: Sequence[float], control: Sequence[float]) -> float:
    """Compute standardized mean separation between target and control groups."""
    x = np.asarray(target, dtype=float)
    y = np.asarray(control, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return 0.0
    pooled = math.sqrt(((len(x) - 1) * float(np.var(x, ddof=1)) + (len(y) - 1) * float(np.var(y, ddof=1))) / max(1, len(x) + len(y) - 2)) if len(x) > 1 and len(y) > 1 else 0.0
    if pooled < 1e-12:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled)


def rank_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Compute AUROC from ranks without an external ML dependency."""
    score_series = pd.Series(scores, dtype=float)
    label_arr = np.asarray(labels, dtype=int)
    mask = np.isfinite(score_series.to_numpy()) & np.isfinite(label_arr)
    score_series = score_series.loc[mask]
    label_arr = label_arr[mask]
    n_pos = int(np.sum(label_arr == 1))
    n_neg = int(np.sum(label_arr == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = score_series.rank(method="average").to_numpy()
    pos_rank_sum = float(np.sum(ranks[label_arr == 1]))
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(max(0.0, min(1.0, auc)))


def build_drop_one_losses(table: pd.DataFrame, metrics: Sequence[MetricSpec]) -> Dict[str, float]:
    """Estimate unique contribution as validation-correlation loss when dropping one indicator."""
    if table.empty:
        return {}
    keys = [metric.key for metric in metrics]
    cols = [f"{key}_norm" for key in keys]
    base_scores = table[cols].mean(axis=1).to_numpy(dtype=float)
    validation = table["validation_composite"].to_numpy(dtype=float)
    base_perf = pearson_abs(base_scores, validation)
    losses: Dict[str, float] = {}
    for key in keys:
        keep_cols = [f"{other}_norm" for other in keys if other != key]
        drop_scores = table[keep_cols].mean(axis=1).to_numpy(dtype=float)
        losses[key] = max(0.0, base_perf - pearson_abs(drop_scores, validation))
    return losses


def build_effect_sizes(table: pd.DataFrame, metrics: Sequence[MetricSpec]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, int]]:
    """Compute overall and per-domain direction-free AUROC separation."""
    if table.empty:
        return {}, {}
    target, control = target_control_masks(table)
    effects: Dict[str, Dict[str, float]] = {}
    group_counts = {
        "target": int(target.sum()),
        "control": int(control.sum()),
    }
    for metric in metrics:
        col = f"{metric.key}_norm"
        subset = table.loc[target | control]
        labels = np.where(target.loc[subset.index].to_numpy(), 1, 0)
        auc = rank_auc(subset[col].to_numpy(dtype=float), labels)
        metric_effects: Dict[str, float] = {
            "overall": max(auc, 1.0 - auc)
        }
        for domain, group in table.groupby("domain"):
            domain_target, domain_control = target_control_masks(group)
            domain_subset = group.loc[domain_target | domain_control]
            domain_labels = np.where(domain_target.loc[domain_subset.index].to_numpy(), 1, 0)
            domain_auc = rank_auc(domain_subset[col].to_numpy(dtype=float), domain_labels)
            metric_effects[str(domain)] = max(domain_auc, 1.0 - domain_auc)
        effects[metric.key] = metric_effects
    return effects, group_counts


def level_from_strength(strength: float) -> str:
    """Map empirical association strength onto the visual coverage levels."""
    if strength >= 0.70:
        return "strong"
    if strength >= 0.42:
        return "moderate"
    if strength >= 0.22:
        return "weak-moderate"
    return "weak"


def build_empirical_coverage(
    table: pd.DataFrame,
    metrics: Sequence[MetricSpec],
    fallback: Mapping[str, Mapping[str, CoverageCell]],
) -> Tuple[Dict[str, Dict[str, CoverageCell]], Dict[str, Dict[str, float]]]:
    """Infer the coverage matrix from metric-mechanism associations across real windows."""
    if table.empty:
        return {k: dict(v) for k, v in fallback.items()}, {}
    coverage: Dict[str, Dict[str, CoverageCell]] = {}
    strengths: Dict[str, Dict[str, float]] = {}
    for metric in metrics:
        coverage[metric.key] = {}
        strengths[metric.key] = {}
        x = table[f"{metric.key}_norm"].to_numpy(dtype=float)
        for mechanism in MECHANISM_INDEX_COLUMNS:
            y = table[f"{mechanism}_norm"].to_numpy(dtype=float)
            strength = pearson_abs(x, y)
            strengths[metric.key][mechanism] = strength
            coverage[metric.key][mechanism] = CoverageCell(
                level=level_from_strength(strength),
                delayed=(mechanism == "Compression"),
            )
    return coverage, strengths


def build_empirical_metric_correlations(table: pd.DataFrame, metrics: Sequence[MetricSpec]) -> Dict[Tuple[str, str], float]:
    """Compute retained-metric correlations across real domain windows."""
    if table.empty:
        return {}
    out: Dict[Tuple[str, str], float] = {}
    keys = [metric.key for metric in metrics]
    for i, left in enumerate(keys):
        for right in keys[i + 1 :]:
            out[(left, right)] = pearson_abs(table[f"{left}_norm"], table[f"{right}_norm"])
    return out


def mean_profile(table: pd.DataFrame, mask: pd.Series, metrics: Sequence[MetricSpec]) -> Tuple[float, ...]:
    """Average normalized seven-indicator values for a selected real cohort."""
    cols = [f"{metric.key}_norm" for metric in metrics]
    subset = table.loc[mask, cols]
    if subset.empty:
        subset = table[cols]
    return tuple(float(v) for v in subset.mean(axis=0).clip(0.0, 1.0).to_numpy())


def build_empirical_fingerprints(table: pd.DataFrame, metrics: Sequence[MetricSpec]) -> List[FingerprintRow]:
    """Create heatmap rows from real Fig. 1 domain-window cohorts."""
    if table.empty:
        return []
    top_cut = float(table["perturbation_composite"].quantile(0.75))
    low_cut = float(table["perturbation_composite"].quantile(0.25))
    median_composite = float(table["perturbation_composite"].median())
    high_output_cut = float(table["n_rolling_papers"].quantile(0.75))
    expansion_cut = float(table["Expansion_norm"].quantile(0.75))
    compression_cut = float(table["Compression_norm"].quantile(0.75))

    return [
        FingerprintRow(
            "Peak perturbation\nwindows",
            mean_profile(table, table["perturbation_composite"] >= top_cut, metrics),
            "star",
            "#6B2A8F",
        ),
        FingerprintRow(
            "High-output\nconventional",
            mean_profile(table, (table["n_rolling_papers"] >= high_output_cut) & (table["perturbation_composite"] <= median_composite), metrics),
            "chart",
            "#0B4FA3",
        ),
        FingerprintRow(
            "Expansion-dominant\nwindows",
            mean_profile(table, table["Expansion_norm"] >= expansion_cut, metrics),
            "book",
            "#0F766E",
        ),
        FingerprintRow(
            "Delayed compression\nwindows",
            mean_profile(table, table["Compression_norm"] >= compression_cut, metrics),
            "paper",
            "#6B7280",
        ),
        FingerprintRow(
            "Low-perturbation\nbaseline",
            mean_profile(table, table["perturbation_composite"] <= low_cut, metrics),
            "wave",
            "#6B7280",
        ),
    ]


def build_empirical_validation_weights(table: pd.DataFrame, metrics: Sequence[MetricSpec]) -> Dict[str, float]:
    """Estimate simple validation weights from metric-outcome correlations."""
    if table.empty:
        return {}
    raw = {
        metric.key: pearson_abs(table[f"{metric.key}_norm"], table["validation_composite"])
        for metric in metrics
    }
    total = sum(raw.values())
    if total <= 1e-12:
        return {key: 0.0 for key in raw}
    return {key: value / total for key, value in raw.items()}


def build_fig2_data(
    data_root: Path = DEFAULT_FIG1_DATA_ROOT,
    focus_domain: str = "crispr",
    use_real_data: bool = True,
) -> Fig2Data:
    """Build static conceptual data for the Fig. 2 drawing."""
    metrics = [
        MetricSpec("B", "B", "#2E7D32", "Global bridging position"),
        MetricSpec("RS", "RS", "#4B8B3B", "Distance-weighted diversity"),
        MetricSpec("DeltaQ", "ΔQ0", "#E85D04", "Community boundary perturbation"),
        MetricSpec("Uzzi", "Uzzi", "#7B3FA1", "Atypical combinations"),
        MetricSpec("RTD", "RTD", "#2C7FB8", "Reference target diversity"),
        MetricSpec("BurtIP", "Burt IP", "#0B4FA3", "Structural holes"),
        MetricSpec("PDE", "PDE", "#E64A19", "Prospective diffusion entropy"),
    ]

    screening_stages = [
        ScreeningStage("Candidate metric universe", 92, "All candidate indicators", "bibliometrics, network science, diversity", "#5B8C4A"),
        ScreeningStage("No future leakage", 67, "Publication-day observable", "future citations, CD index, burst", "#2C7FB8"),
        ScreeningStage("Reference-only", 49, "Reference-graph computable", "author, journal, institution", "#2A7F7F"),
        ScreeningStage("Graph perturbation", 29, "Mechanistically linked", "generic controls", "#7B3FA1"),
        ScreeningStage("Non-redundant clusters", 12, "Merge overlapping metrics", "overlapping metrics", "#E85D04"),
        ScreeningStage("Interpretable and robust", 7, "Validation-ready", "final seven-parameter basis", "#F97316"),
    ]

    coverage = {
        "B": {
            "Expansion": CoverageCell("weak"),
            "Bridging": CoverageCell("strong"),
            "Reconfiguration": CoverageCell("moderate"),
            "Compression": CoverageCell("moderate", delayed=True),
        },
        "RS": {
            "Expansion": CoverageCell("strong"),
            "Bridging": CoverageCell("moderate"),
            "Reconfiguration": CoverageCell("weak"),
            "Compression": CoverageCell("moderate", delayed=True),
        },
        "DeltaQ": {
            "Expansion": CoverageCell("weak"),
            "Bridging": CoverageCell("weak"),
            "Reconfiguration": CoverageCell("strong"),
            "Compression": CoverageCell("moderate", delayed=True),
        },
        "Uzzi": {
            "Expansion": CoverageCell("moderate"),
            "Bridging": CoverageCell("weak"),
            "Reconfiguration": CoverageCell("strong"),
            "Compression": CoverageCell("moderate", delayed=True),
        },
        "RTD": {
            "Expansion": CoverageCell("weak"),
            "Bridging": CoverageCell("strong"),
            "Reconfiguration": CoverageCell("moderate"),
            "Compression": CoverageCell("weak-moderate", delayed=True),
        },
        "BurtIP": {
            "Expansion": CoverageCell("weak"),
            "Bridging": CoverageCell("strong"),
            "Reconfiguration": CoverageCell("moderate"),
            "Compression": CoverageCell("weak-moderate", delayed=True),
        },
        "PDE": {
            "Expansion": CoverageCell("strong"),
            "Bridging": CoverageCell("weak-moderate"),
            "Reconfiguration": CoverageCell("weak"),
            "Compression": CoverageCell("moderate", delayed=True),
        },
    }

    fingerprint_rows = [
        FingerprintRow("Landmark innovation", (0.95, 0.88, 0.92, 0.89, 0.93, 0.90, 0.86), "star", "#6B2A8F"),
        FingerprintRow("High-citation conventional", (0.44, 0.52, 0.42, 0.37, 0.45, 0.40, 0.58), "chart", "#0B4FA3"),
        FingerprintRow("Review article", (0.45, 0.50, 0.56, 0.47, 0.49, 0.46, 0.50), "book", "#0F766E"),
        FingerprintRow("Incremental paper", (0.36, 0.41, 0.39, 0.42, 0.44, 0.42, 0.43), "paper", "#6B7280"),
        FingerprintRow("Noise control", (0.20, 0.18, 0.16, 0.24, 0.22, 0.19, 0.21), "wave", "#6B7280"),
    ]

    clusters = [
        MechanismCluster("Expansion", "RS", ("PageRank", "In-degree", "# refs", "Out-degree"), "#4B8B3B", (0.275, 0.710)),
        MechanismCluster("Bridging", "B", ("Betweenness", "Tie span", "Bridge score", "k-core span"), "#0B4FA3", (0.535, 0.725)),
        MechanismCluster("Reconfiguration", "DeltaQ", ("Modularity change", "Community surprise", "Link rewiring"), "#E85D04", (0.790, 0.670)),
        MechanismCluster("Uzzi diversity", "Uzzi", ("Journal z", "Atypical tail", "Conventionality"), "#7B3FA1", (0.250, 0.345)),
        MechanismCluster("Reference-time dispersion", "RTD", ("Ref age variance", "Cited half-life", "Community Simpson"), "#2C7FB8", (0.500, 0.335)),
        MechanismCluster("Burt structural holes", "BurtIP", ("Constraint", "Effective size", "Structural hole score"), "#0B4FA3", (0.650, 0.360)),
        MechanismCluster("Potential diversity entropy", "PDE", ("Shannon entropy", "Field dispersion", "Simpson index"), "#E64A19", (0.845, 0.350)),
    ]

    real_domains: Tuple[DomainFig1Data, ...] = tuple()
    coverage_strengths: Dict[str, Dict[str, float]] = {}
    metric_correlations: Dict[Tuple[str, str], float] = {}
    metric_correlation_matrix = pd.DataFrame()
    drop_one_losses: Dict[str, float] = {}
    effect_sizes: Dict[str, Dict[str, float]] = {}
    group_counts: Dict[str, int] = {}
    validation_weights: Dict[str, float] = {}
    data_note = "Static methodology fallback"
    if use_real_data:
        real_domains = load_fig1_domain_data(data_root, focus_domain=focus_domain)
        real_table = real_metric_window_table(real_domains, metrics)
        if not real_table.empty:
            coverage, coverage_strengths = build_empirical_coverage(real_table, metrics, coverage)
            empirical_rows = build_empirical_fingerprints(real_table, metrics)
            if empirical_rows:
                fingerprint_rows = empirical_rows
            metric_correlations = build_empirical_metric_correlations(real_table, metrics)
            metric_correlation_matrix = spearman_corr_matrix(real_table, metrics)
            drop_one_losses = build_drop_one_losses(real_table, metrics)
            effect_sizes, group_counts = build_effect_sizes(real_table, metrics)
            validation_weights = build_empirical_validation_weights(real_table, metrics)
            data_note = f"Real Fig. 1 exports: {len(real_domains)} domains, {len(real_table)} rolling windows"

    return Fig2Data(
        metrics=metrics,
        screening_stages=screening_stages,
        coverage=coverage,
        fingerprint_rows=fingerprint_rows,
        clusters=clusters,
        real_domains=real_domains,
        coverage_strengths=coverage_strengths,
        metric_correlations=metric_correlations,
        metric_correlation_matrix=metric_correlation_matrix,
        drop_one_losses=drop_one_losses,
        effect_sizes=effect_sizes,
        group_counts=group_counts,
        validation_weights=validation_weights,
        data_note=data_note,
    )


def make_axes(fig: plt.Figure) -> Dict[str, plt.Axes]:
    """Create manually placed axes for the six panels."""
    positions = {
        "a": [0.015, 0.535, 0.345, 0.390],
        "b": [0.365, 0.535, 0.355, 0.390],
        "c": [0.725, 0.535, 0.260, 0.390],
        "d": [0.015, 0.055, 0.315, 0.455],
        "e": [0.335, 0.055, 0.285, 0.455],
        "f": [0.625, 0.055, 0.360, 0.455],
    }
    return {key: fig.add_axes(pos) for key, pos in positions.items()}


def draw_fig2(
    data: Fig2Data,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
    seed: int,
    width: float = 19.0,
    height: float = 10.5,
) -> List[Path]:
    """Draw and save Fig. 2 in the requested formats."""
    setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(width, height), dpi=dpi)
    fig.text(0.5, 0.985, "Fig. 2 | Why these seven indicators?", ha="center", va="top", fontsize=19, fontweight="bold")
    fig.text(
        0.5,
        0.944,
        "Selection and validation logic for graph-perturbation metrics",
        ha="center",
        va="top",
        fontsize=11.5,
        color=TEXT_LIGHT,
        fontstyle="italic",
    )

    axes = make_axes(fig)
    draw_panel_a(axes["a"], data, seed)
    draw_panel_b(axes["b"], data)
    draw_panel_c(axes["c"], data)
    draw_panel_d(axes["d"], data)
    draw_panel_e(axes["e"], data)
    draw_panel_f(axes["f"], data)
    if data.data_note:
        fig.text(0.985, 0.012, data.data_note, ha="right", va="bottom", fontsize=6.2, color=TEXT_LIGHT)

    saved: List[Path] = []
    for ext in formats:
        suffix = ext.lower().lstrip(".")
        path = output_dir / f"fig2_seven_indicators.{suffix}"
        fig.savefig(path, dpi=dpi)
        saved.append(path)
    plt.close(fig)
    return saved


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Draw Fig. 2 seven-indicator selection and validation logic.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for generated figure files.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_FIG1_DATA_ROOT, help="Root directory containing Fig. 1 exported CSV folders.")
    parser.add_argument("--domain", default="crispr", help="Focus Fig. 1 domain to use for Panel a real graph snapshots.")
    parser.add_argument("--no-real-data", action="store_true", help="Use the static methodology fallback instead of Fig. 1 exports.")
    parser.add_argument("--formats", nargs="+", default=["png", "svg", "pdf"], help="Output formats to write.")
    parser.add_argument("--dpi", type=int, default=300, help="Raster output resolution.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic schematic jitter.")
    parser.add_argument("--width", type=float, default=19.0, help="Figure width in inches.")
    parser.add_argument("--height", type=float, default=10.5, help="Figure height in inches.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    data = build_fig2_data(data_root=args.data_root, focus_domain=args.domain, use_real_data=not args.no_real_data)
    saved = draw_fig2(
        data=data,
        output_dir=args.output_dir,
        formats=args.formats,
        dpi=args.dpi,
        seed=args.seed,
        width=args.width,
        height=args.height,
    )
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()

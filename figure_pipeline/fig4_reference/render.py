"""Millimetre-layout Fig.4 with observed points, stored intervals and vector tiles."""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.text import Text
from scipy.stats import gaussian_kde

from figure_pipeline.fig1_reference.export import fonts

from .data import ACTIONS, COMPONENTS, ORDER, OUT, read

W, H = 216.0, 158.0
SCALE = 1.5
PT = 72 / 25.4 * SCALE
INK, GREY, GRID, PURPLE = "#24282D", "#737B85", "#E4ECF2", "#713BCB"
EDGE, HEADER = "#A9CDE4", "#EEF6FC"
COMPONENT_COLORS = ["#E16C35", "#1269DF", PURPLE, "#8896A8", "#40A7BE"]
PALETTE = ["#327AC5", "#8DBADF", "#F6EABF", "#EDAB65", "#D85A43"]
DIVERGING = LinearSegmentedColormap.from_list("information_effect", PALETTE)
BLUE = LinearSegmentedColormap.from_list("row_share", ["#FAFCFD", "#ADCFE1", "#3D6FA6"])
SOURCES = ["#E4773F", "#397CCB", "#9870CA"]
SOURCE_NAMES = ["Retained", "Newly eligible", "New to branches"]
ORIGINAL_PANELS = {"a": (3, 6, 63, 48), "b": (70, 6, 110, 48), "c": (3, 58, 72, 36),
          "d": (79, 58, 101, 37), "e": (3, 104, 101, 42), "f": (108, 104, 72, 42)}
PANELS = {"a": (4, 15, 98, 48), "b": (106, 15, 106, 48),
          "c": (4, 68, 98, 37), "d": (106, 68, 106, 37),
          "e": (4, 116, 120, 39), "f": (128, 116, 84, 39)}


def position(gid: str, x: float, y: float) -> tuple[float, float]:
    """Place each existing component inside the reference's three-row card grid."""
    if gid[:1] not in ORIGINAL_PANELS or len(gid) < 2 or not (gid[1].isdigit() or gid[1] == "_"):
        return x, y
    ox, oy, ow, oh = ORIGINAL_PANELS[gid[0]]
    nx, ny, nw, nh = PANELS[gid[0]]
    return nx + (x - ox) * nw / ow, ny + (y - oy) * nh / oh


def positioned_box(gid: str, box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, w, h = box
    x1, y1 = position(gid, x, y)
    x2, y2 = position(gid, x + w, y + h)
    return x1, y1, x2 - x1, y2 - y1


BOXES = {"a1": (3, 11, 24, 43), "a2": (29, 11, 37, 43),
         "b1": (84, 11, 30, 41), "b2": (116, 11, 30, 41), "b3": (148, 11, 32, 41),
         "c1": (3, 62, 39, 32), "c2": (45, 62, 30, 32),
         "d1": (79, 62, 101, 20), "d2": (79, 83, 101, 12),
         "e1": (3, 109, 48, 37), "e2": (53, 109, 51, 37),
         "f1": (108, 109, 72, 13), "f2": (108, 123, 72, 23)}
TITLES = {"a": "Configurations and endpoints", "b": "Paired component effects", "c": "Additive interaction",
          "d": "Structure-to-report translation", "e": "Fusion actions and information sources",
          "f": "Relation evidence and report treatment"}


class Figure:
    def __init__(self) -> None:
        for name in ["arial.ttf", "arialbd.ttf", "times.ttf", "timesbd.ttf"]:
            font_manager.fontManager.addfont("/mnt/c/Windows/Fonts/" + name)
        mpl.rcParams.update({"font.family": "Arial", "font.size": 6.5, "axes.labelsize": 6.5,
            "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.linewidth": .55,
            "text.color": INK, "axes.labelcolor": INK, "xtick.color": GREY, "ytick.color": GREY,
            "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.facecolor": "white"})
        self.fig = plt.figure(figsize=(W * SCALE / 25.4, H * SCALE / 25.4), facecolor="white")
        self.serial = 0
        self.text("figure_heading", W / 2, 2.4,
            "Fig. 4 | Controlled information effects and evidence-to-report mechanisms",
            12.5, bold=True, ha="center")
        self.text("domain_new", W / 2, 9.5,
            "Matched downstream task · 100 papers × 7 configurations · existing analysis reused",
            7, color=GREY, ha="center")
        self.fig.add_artist(plt.Line2D([4 / W, 212 / W], [1 - 111 / H] * 2,
            transform=self.fig.transFigure, color=EDGE, lw=.65, ls=(0, (4, 3)), gid="domain_old_rule"))
        self.text("domain_old", W / 2, 108.6,
            "Earlier Fig.3 reports · descriptive mechanism analysis",
            7.6, color=INK, bold=True, ha="center",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 3})
        for panel, (x, y, w, h) in PANELS.items():
            self.card(panel + "_card", (x - 1.1, y - 1.5, w + 2.2, h + 2.2), mapped=False)
            self.fig.add_artist(Rectangle(((x - .8) / W, 1 - (y + 3.5) / H),
                (w + 1.6) / W, 4.5 / H, transform=self.fig.transFigure,
                fc=HEADER, ec="none", zorder=-4, gid=panel + "_card_header"))
            ox, oy, _, _ = ORIGINAL_PANELS[panel]
            self.text(panel + "_title", ox, oy, panel, 10, bold=True)
            self.text(panel + "_title", ox + 4, oy + .35, TITLES[panel], 7.5, bold=True)
        self.card("a1_frame", (3, 11, 24, 43))
        self.card("a2_frame", (29, 11, 37, 43))
        self.text("a1", 15, 11.5, "Controlled-contrast map", 6.6, bold=True, ha="center")
        self.text("a2", 47.5, 11.5, "Endpoints by configuration", 6.6, bold=True, ha="center")

    def card(self, gid: str, box: tuple[float, float, float, float], mapped: bool = True) -> None:
        x, y, w, h = positioned_box(gid, box) if mapped else box
        self.fig.add_artist(FancyBboxPatch((x / W, 1 - (y + h) / H), w / W, h / H,
            boxstyle="round,pad=0,rounding_size=0.005", transform=self.fig.transFigure,
            facecolor="#FCFDFF", edgecolor=EDGE, linewidth=.65,
            zorder=-5, gid=gid))

    def text(self, gid: str, x: float, y: float, value: str, size: float = 6.5,
             color: str = INK, bold: bool = False, **kwargs: Any) -> None:
        self.serial += 1
        x, y = position(gid, x, y)
        if bold:
            kwargs.setdefault("fontfamily", "Times New Roman")
        self.fig.text(x / W, 1 - y / H, value, fontsize=size, color=color,
            weight="bold" if bold else "normal", va="top", gid=f"{gid}_text{self.serial}", **kwargs)

    def ax(self, gid: str, box: tuple[float, float, float, float], projection: str | None = None) -> Axes:
        x, y, w, h = positioned_box(gid, box)
        ax = self.fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H], projection=projection)
        ax.set_gid(gid)
        if projection is None:
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(length=2, pad=1.5, width=.5)
        return ax

    def data(self, name: str) -> pd.DataFrame:
        return pd.read_csv(OUT / "data" / (name + ".csv"), escapechar="\\")


def panel_a(f: Figure) -> None:
    ax = f.ax("a1", (3.8, 17, 22.4, 29)); ax.axis("off"); ax.set(xlim=(0, 1), ylim=(0, 1.08))
    nodes = {"T": (.19, .20), "E": (.19, .88), "G": (.72, .20), "F": (.72, .88)}
    colors = {"T": GREY, "E": COMPONENT_COLORS[0], "G": COMPONENT_COLORS[1], "F": PURPLE}
    names = {"T": "T\nText", "E": "E\nGEAR", "G": "G\nGraph", "F": "F\nFull"}
    for key, (x, y) in nodes.items():
        ax.add_patch(FancyBboxPatch((x - .16, y - .09), .32, .18,
            boxstyle="round,pad=0,rounding_size=0.015", ec=colors[key],
            fc=mpl.colors.to_rgba(colors[key], .12), lw=.8))
        ax.text(x, y, names[key], va="center", ha="center", color=colors[key],
            weight="bold", fontsize=7, linespacing=1.15)
    for left, right in [("T", "E"), ("T", "G"), ("E", "F")]:
        a, b = nodes[left], nodes[right]
        horizontal = a[1] == b[1]
        start = (a[0] + .18, a[1]) if horizontal else (a[0], a[1] + .11)
        end = (b[0] - .18, b[1]) if horizontal else (b[0], b[1] - .11)
        color = COMPONENT_COLORS[1] if horizontal else COMPONENT_COLORS[0]
        ax.annotate("", end, start, arrowprops={"arrowstyle": "-|>", "lw": 1.1, "color": color})
    ax.plot([.89, .97, .97], [.20, .20, .88], color=COMPONENT_COLORS[0], lw=1.1)
    ax.annotate("", (.89, .88), (.97, .88), arrowprops={"arrowstyle": "-|>", "lw": 1.1, "color": COMPONENT_COLORS[0]})
    ax.text(.45, .99, "+Graph", ha="center", fontsize=6.5, color=COMPONENT_COLORS[1])
    ax.text(.45, .08, "+Graph", ha="center", fontsize=6.5, color=COMPONENT_COLORS[1])
    ax.text(.055, .54, "+GEAR", rotation=90, ha="center", va="center", fontsize=6.5, color=COMPONENT_COLORS[0])
    ax.plot([.72, .72], [.78, .40], color="#C1B0DC", lw=.7, zorder=0)
    for y, value in [(.65, "F−J · No Joint"), (.51, "F−M · No metrics"), (.37, "F−P · No paths")]:
        ax.add_patch(FancyBboxPatch((.38, y - .055), .52, .11,
            boxstyle="round,pad=0,rounding_size=0.01", ec="#BCC9D4", fc="#F0F3F6", lw=.6))
        ax.text(.64, y, value, ha="center", va="center", fontsize=5.8)
    f.text("a1", 4, 48.2, "Same sources and writing task", 6.2, color=GREY)
    f.text("a1", 4, 51.1, "Component information varied", 6.2, color=GREY)
    data = f.data("a_condition_endpoints").set_index("condition").loc[ORDER]
    ax = f.ax("a2", (29.6, 17, 35.8, 29)); ax.axis("off"); ax.set(xlim=(0, 4.7), ylim=(7.5, -.9))
    names = ["T · Text", "E · GEAR", "G · Graph", "F · Full", "F−J · No Joint", "F−M · No metrics", "F−P · No paths"]
    for j, title in enumerate(["H ↑", "V ↑", "R ↓"]):
        ax.add_patch(Rectangle((1.55+j, -.9), 1, .9, fc=HEADER, ec="#C9D7E3", lw=.4))
        ax.text(2 + j, -.45, title, ha="center", va="center", weight="bold", fontsize=7)
    for i, (_, row) in enumerate(data.iterrows()):
        ax.add_patch(Rectangle((0, i), 1.55, 1, fc="#EDE9F5" if i == 3 else "#F0F3F6", ec="#C9D7E3", lw=.4))
        ax.text(.05, i + .5, names[i], va="center", fontsize=6.5, weight="bold" if i == 3 else "normal")
        for j, col in enumerate(["H", "V", "R"]):
            v = row[col]; color = DIVERGING(v / (8 if col == "V" else 100))
            ax.add_patch(Rectangle((1.55 + j, i), 1, 1, fc=(*color[:3], .23), ec="#C9D7E3", lw=.4))
            ax.text(2 + j, i + .5, f"{v:.1f}" if col == "H" else f"{v:.2f}", ha="center", va="center", fontsize=6.5)
        if i == 3: ax.add_patch(Rectangle((-.02, i), 4.6, .96, fill=False, ec=PURPLE, lw=.5))
    ax.text(2, 7.3, "%", ha="center", fontsize=6); ax.text(3, 7.3, "clusters", ha="center", fontsize=6); ax.text(4, 7.3, "%", ha="center", fontsize=6)
    f.text("a2", 29, 48.2, "Unknown support: 56.5–60.9%", 6.5)
    f.text("a2", 29, 51.1, "1−R is not accuracy", 6.5, color=GREY)


def stacked_points(values: np.ndarray, metric: str) -> tuple[np.ndarray, np.ndarray]:
    if metric == "R":
        return values, -.11 - (np.arange(len(values)) * .61803398875 % 1) * .18
    x, y = [], []
    for value in np.unique(values):
        count = int(np.sum(values == value))
        for k in range(count):
            x.append(value); y.append(-.07 - .20 * (k + .5) / max(count, 1))
    return np.array(x), np.array(y)


def panel_b(f: Figure) -> None:
    paper = f.data("b_paper_effects_improvement"); summary = f.data("b_effect_summary_improvement")
    row_labels = ["GEAR", "Graph\nbranch", "Joint", "Structural\nsummaries", "Paper\npaths"]
    for i, label in enumerate(row_labels): f.text("b_labels", 70, 20.6 + i * 6.65, label, 6.5, color=COMPONENT_COLORS[i], bold=True)
    for index, (metric, title, limits, ticks) in enumerate([
        ("H", "ΔH (pp)", (-105, 105), [-100, -50, 0, 50, 100]),
        ("V", "ΔV (clusters/report)", (-24, 20), [-20, 0, 20]),
        ("R", "Reduction in R (pp)", (-75, 75), [-75, 0, 75])], 1):
        gid = f"b{index}"; x, _, w, _ = BOXES[gid]
        f.card(gid + "_frame", (x, 11, w, 41))
        f.text(gid, x + w / 2, 11.4, title, 6.5, bold=True, ha="center")
        ax = f.ax(gid, (x + 1, 18, w - 2, 33.5)); ax.set(xlim=limits, ylim=(4.53, -.50))
        ax.set_yticks([]); ax.set_xticks(ticks); ax.xaxis.tick_top(); ax.spines[["left", "bottom"]].set_visible(False)
        ax.axvline(0, color=GREY, lw=.55, ls=(0, (2, 2)))
        for i, component in enumerate(COMPONENTS):
            vals = paper[(paper.metric == metric) & (paper.component == component)].display_delta.to_numpy()
            row = summary[(summary.metric == metric) & (summary.component == component)].iloc[0]
            row_color = COMPONENT_COLORS[i]
            ax.axhspan(i - .48, i + .48, color="#EFF4F8" if i % 2 == 0 else "#FFFFFF", zorder=-3)
            ax.axhline(i + .50, color="#DAE5EE", lw=.45, zorder=-2)
            if metric != "R":
                bins = np.array([-125, -75, -25, 25, 75, 125]) if metric == "H" else np.arange(-24.5, 21.5, 1)
                count, edges = np.histogram(vals, bins=bins)
                centers = (edges[:-1] + edges[1:]) / 2
                width = 7 if metric == "H" else .88
                ax.bar(centers, -.23 * count / max(count.max(), 1), width=width, bottom=i + .06,
                       color=row_color, alpha=.30, linewidth=0)
            else:
                grid = np.linspace(max(vals.min(), limits[0]), min(vals.max(), limits[1]), 160)
                density = gaussian_kde(vals)(grid)
                ax.fill_between(grid, i + .07, i + .07 + .23 * density / density.max(), color=row_color, alpha=.25, lw=.4)
            px, py = stacked_points(vals, metric)
            ax.scatter(px, i + py, s=3.4, c=row_color, alpha=.55, linewidths=.10, edgecolors="white")
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            ax.add_patch(Rectangle((q1, i - .045), q3 - q1, .09, fc="white", ec=INK, lw=.5, zorder=5))
            ax.plot([med, med], [i - .065, i + .065], color=INK, lw=.7, zorder=6)
            estimate = f"{row.display_mean:+.2f} [{row.display_ci_low:.2f}, {row.display_ci_high:.2f}]".replace("-", "−")
            ax.text(.98, i + .45, estimate, transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=6, color=INK, bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8, "pad": .3})
        ax.tick_params(labelsize=6.5)
    f.text("b_labels", 70, 52.3, "100 paired papers; all 15 CIs include 0 · labels: mean [95% CI]", 6.5)


def panel_c(f: Figure) -> None:
    ax = f.ax("c1", (3, 63, 39, 23), "3d")
    a = f.data("a_condition_endpoints").set_index("condition").V.to_dict()
    xx, yy = np.meshgrid(np.linspace(0, 1, 16), np.linspace(0, 1, 16))
    interaction = a["F"] - a["E"] - a["G"] + a["T"]
    additive = a["T"] + (a["E"] - a["T"]) * xx + (a["G"] - a["T"]) * yy
    ax.plot_surface(xx, yy, additive + interaction * xx * yy, cmap=LinearSegmentedColormap.from_list("interaction_surface", ["#307AC0", "#79B4E2", "#FFE59A", "#EE9B53"]), vmin=5.45, vmax=6.2,
        alpha=.85, linewidth=.15, edgecolor="#BCC5CA", antialiased=True)
    ax.plot_wireframe(xx, yy, additive, color=GREY, linewidth=.35, rstride=5, cstride=5)
    expected = a["E"] + a["G"] - a["T"]
    ax.plot([1, 1], [1, 1], [expected, a["F"]], color=PURPLE, lw=.9)
    for name, x, y, color in [("T", 0, 0, GREY), ("E", 1, 0, "#D97732"), ("G", 0, 1, "#3F78B5"), ("F", 1, 1, PURPLE)]:
        ax.scatter(x, y, a[name], s=10, color=color, depthshade=False)
    ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 8), xticks=[0, 1], yticks=[0, 1], zticks=[0, 4, 8])
    ax.set_xlabel("", labelpad=-5); ax.set_ylabel("", labelpad=-5); ax.set_zlabel("V", labelpad=-6)
    ax.tick_params(pad=-2, labelsize=6.5); ax.view_init(elev=24, azim=-125); ax.set_proj_type("ortho"); ax.set_box_aspect((1, 1, .7))
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False; axis._axinfo["grid"]["linewidth"] = .3
    f.text("c1", 6, 87, "Graph (0/1)", 6.5)
    f.text("c1", 26, 87, "GEAR (0/1)", 6.5)
    for name, x, y, color in [("T", 20, 79, GREY), ("E", 35, 74, COMPONENT_COLORS[0]),
                              ("G", 5, 67, COMPONENT_COLORS[1]), ("F", 28, 65.7, PURPLE)]:
        f.text("c1", x, y, f"{name}={a[name]:.2f}", 6.5, color=color,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": .2})
    f.text("c1", 3, 91, "4 conditions; interpolation only", 6.5)
    f.text("c1", 4, 62, "V and additive reference", 6.5, bold=True)
    values = f.data("c_interaction_paper").I.to_numpy()
    stored = read(OUT / "data/c_interaction_summary.json")["statistics"]["I"]
    ax = f.ax("c2", (49, 69, 25, 15))
    ax.hist(values, bins=np.arange(-20.5, 26.5, 2), color=PURPLE, alpha=.45, edgecolor="white", lw=.3)
    ax.axvspan(stored["low"], stored["high"], color=PURPLE, alpha=.12, lw=0)
    ax.axvline(0, color=GREY, lw=.5, ls="--"); ax.axvline(values.mean(), color=PURPLE, lw=.8)
    ax.plot(values, np.full(len(values), -.65), "|", color=PURPLE, ms=2, mew=.35, clip_on=False)
    ax.set(xlim=(-22, 27), xticks=[-20, 0, 20], ylabel="Papers")
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3, integer=True)); ax.set_ylim(0, ax.get_ylim()[1])
    ax.tick_params(labelsize=6.5)
    f.text("c2", 45, 62, "I = F − E − G + T", 6.5, bold=True)
    f.text("c2", 45, 65.4, "+0.74; 95% CI [−0.88, 2.37]", 6.5)
    f.text("c2", 49, 87, "Interaction (clusters)", 6.5)
    f.text("c2", 45, 91, "+ / 0 / − : 53 / 4 / 43", 6.5)


def panel_d(f: Figure) -> None:
    data = f.data("d_structure_and_report_tracks")
    ax = f.ax("d1", (89, 64, 90, 14))
    ax.vlines(data.structural_rank, 0, data.J_percent, color="#DEE5EB", lw=.4, zorder=0)
    ax.scatter(data.structural_rank, data.J_percent, c=data.J_percent, cmap=LinearSegmentedColormap.from_list("structure", ["#337DC8", "#7772C9", "#E59A55", "#D75343"]), vmin=0, vmax=85,
        s=7, lw=0, alpha=.75)
    selected = data[data.delta_cross.ne(0)]
    ax.scatter(selected.structural_rank, selected.J_percent, s=13, facecolor="none", edgecolor=PURPLE, lw=.65)
    ax.set(xlim=(.5, 100.5), ylim=(0, 100), yticks=[0, 50, 100], xticks=[1, 50, 100])
    ax.set_ylabel("J (%)", labelpad=1); ax.grid(axis="y", color=GRID, lw=.35)
    ax.text(.02, .96, "Median 31.32%; range 4.01–81.12%", transform=ax.transAxes, va="top", fontsize=6.5)
    f.text("d1", 89, 61.7, "Joint-exclusive connectivity · 100 papers", 6.5, bold=True)
    ax = f.ax("d2", (89, 83, 90, 5)); ax.axis("off"); ax.set(xlim=(.5, 100.5), ylim=(2, 0))
    for i, column in enumerate(["V_cross_F", "V_cross_F_noJ"]):
        for r in data.itertuples():
            value = getattr(r, column)
            ax.add_patch(Rectangle((r.structural_rank - .5, i + .1), .96, .72, fc=PURPLE if value else "#EDF0F2", ec="white", lw=.15))
    f.text("d2", 79, 83.1, "Full", 6.5); f.text("d2", 79, 85.6, "−Joint", 6.5)
    f.text("d2", 179, 89, "Full: 1/100; −Joint: 3/100", 6.5, ha="right")
    f.text("d2", 89, 80.6, "Paper order by joint-exclusive connectivity", 6.5)
    f.text("d2", 179, 90.9, "■ Present   □ Absent", 6, color=PURPLE, ha="right")
    f.text("d2", 79, 92, "Strict cross-claim content · Δ: 1 positive / 96 zero / 3 negative", 6.5)


def panel_e(f: Figure) -> None:
    cells = f.data("e1_action_transition_cells")
    counts = cells.pivot(index="decision_action", columns="report_action", values="count").loc[ACTIONS, ACTIONS].to_numpy()
    percents = cells.pivot(index="decision_action", columns="report_action", values="row_percent").loc[ACTIONS, ACTIONS].to_numpy()
    ax = f.ax("e1", (13, 119, 32, 20))
    for i in range(5):
        for j in range(5):
            value = percents[i, j]; color = BLUE(value / 100)
            ax.add_patch(Rectangle((j, i), 1, 1, fc=color, ec="#BECDDC", lw=.5))
            ax.text(j + .5, i + .5, f"{counts[i,j]}", ha="center", va="center", color="white" if value > 65 else INK, fontsize=6.5)
    ax.add_patch(Rectangle((1, 0), 1, 1, fill=False, ec=PURPLE, lw=.8))
    labels = ["Retain", "Merge", "Correct", "Omit", "Unres."]
    ax.set(xlim=(0, 5), ylim=(5, 0), xticks=np.arange(5) + .5, yticks=np.arange(5) + .5,
        xticklabels=labels, yticklabels=labels)
    ax.xaxis.tick_top(); ax.tick_params(length=0, labelsize=6, pad=4)
    ax.spines[:].set_visible(False)
    for i, n in enumerate(counts.sum(axis=1)): ax.text(5.13, i + .5, f"{n:,}", va="center", fontsize=6, clip_on=False)
    for j, n in enumerate(counts.sum(axis=0)): ax.text(j + .5, 5.25, f"{n:,}", ha="center", va="top", fontsize=6, clip_on=False)
    f.text("e1", 3, 109, "Finding-action records: 9,860", 6.5, bold=True)
    f.text("e1", 13, 112, "Final-report action →", 6.5)
    f.text("e1", 3, 119, "Fusion action", 6.5, rotation=90)
    f.text("e1", 46.2, 115, "Total", 6, color=GREY)
    f.text("e1", 3, 144, "Color: row %; text: n", 6.5)
    data = f.data("e2_information_sources_paper")
    cols = ["retained", "newly_supported", "added"]; totals = data[cols].sum().to_numpy()
    f.text("e2_total", 53, 109, "Full: 1,800 eligible clusters", 6.5, bold=True)
    ax = f.ax("e2_total", (53, 113, 51, 4)); ax.axis("off"); ax.set(xlim=(0, 1800), ylim=(0, 1))
    left = 0
    for total, color in zip(totals, SOURCES):
        ax.add_patch(Rectangle((left, 0), total, 1, fc=color, ec="white", lw=.4))
        ax.text(left + total / 2, .5, f"{total} ({total / 1800:.1%})", va="center", ha="center", fontsize=6, color="white")
        left += total
    for i, (name, color) in enumerate(zip(SOURCE_NAMES, SOURCES)):
        f.text("e2_total", 53 + [0, 16, 33][i], 118.2, "■ " + name, 6, color=color)
    ax = f.ax("e2_paper", (59, 125, 44, 11))
    left = np.zeros(len(data))
    for col, color in zip(cols, SOURCES):
        vals = data[col].to_numpy(); ax.bar(np.arange(1, 101), vals, bottom=left, color=color, width=.85, lw=0); left += vals
    ax.set(xlim=(.2, 100.8), ylim=(0, max(left) * 1.03), xticks=[1, 50, 100], ylabel="Clusters")
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3, integer=True)); ax.tick_params(labelsize=6.5)
    f.text("e2_paper", 53, 123, "Per-paper information sources", 6.5)
    f.text("e2_paper", 59, 140.1, "Papers ordered by ID", 6.5)
    f.text("e2_paper", 53, 143.6, "Union: 659 = 368 retained + 291 not retained", 6)


TREATMENTS = {"same_scope": ("#377B61", "Same scope", ""),
    "narrowed_or_corrected": ("#D4B654", "Narrowed/corrected", ""),
    "partial": ("#DC9367", "Partial", ""), "different_or_conflicting": ("#B95045", "Conflicting", ""),
    "explicitly_unresolved": ("#9878B3", "Explicitly unresolved", "/"),
    "uncertain": ("#D3D6DA", "Trace uncertain", "..")}


def panel_f(f: Figure) -> None:
    states = ["supported", "supported_after_narrowing", "insufficient_material", "not_submitted_source_gap"]
    evidence_colors = ["#397BB6", "#8DC4D5", "#EAC67A", "#D3D6DA"]
    counts = f.data("f1_evidence_status_counts").set_index("verified_state")["count"]
    ax = f.ax("f1", (108, 112, 72, 4.2)); ax.axis("off"); ax.set(xlim=(0, 76), ylim=(0, 1))
    left = 0
    for state, color in zip(states, evidence_colors):
        count = int(counts[state])
        ax.add_patch(Rectangle((left, 0), count, 1, fc=color, ec="white", lw=.4, hatch="///" if state == states[-1] else None))
        ax.text(left + count / 2, .5, str(count), ha="center", va="center", fontsize=6.5, color="white" if state == "supported" else INK)
        left += count
    f.text("f1", 108, 109, "76 candidates · 44 supported/narrowed", 6.5, bold=True)
    ax.plot([0, 0, 52, 52], [-.09, -.22, -.22, -.09], color="#397BB6", lw=.65, clip_on=False)
    ax.text(26, -.3, "52 source-audited", color="#397BB6", fontsize=5.8,
        va="top", ha="center", clip_on=False)
    f.text("f1", 108, 120, "11 supported · 33 narrowed · 8 insufficient · 24 source gap", 5.7)
    data = f.data("f2_relation_trace_tiles")
    gear = data[data.method.eq("gear")].sort_values("relation_rank")
    full = data[data.method.eq("fusion")].sort_values("relation_rank")
    ax = f.ax("f2", (120, 126, 44, 7)); ax.axis("off"); ax.set(xlim=(.5, 44.5), ylim=(3, 0))
    for row in gear.itertuples():
        color = evidence_colors[0 if row.verified_state == "supported" else 1]
        ax.add_patch(Rectangle((row.relation_rank - .5, .1), .96, .75, fc=color, lw=0))
    for i, method in enumerate([gear, full], 1):
        for row in method.itertuples():
            color, _, hatch = TREATMENTS[row.treatment]
            ax.add_patch(Rectangle((row.relation_rank - .5, i + .1), .96, .75, fc=color, ec="#8D949C" if hatch else "white", hatch=hatch, lw=.15))
    boundaries = gear[gear.paper_id.ne(gear.paper_id.shift())].relation_rank
    for rank in boundaries: ax.axvline(rank - .5, color=INK, lw=.3)
    for i, label in enumerate(["Evidence", "GEAR", "Full"]): f.text("f2", 108, 126 + i * 2.33, label, 6.5)
    f.text("f2", 166, 128.2, "13/44", 6.5); f.text("f2", 166, 130.5, "17/44", 6.5)
    f.text("f2", 108, 122.4, "44 relations · source and report treatment", 6.5, bold=True)
    f.text("f2", 166, 122.8, "Same /\nnarrowed", 6)
    for rank in [1, 20, 44]:
        f.text("f2", 120 + (rank - .5) / 44 * 44, 133.3, str(rank), 6, ha="center")
    for i, (_, (color, label, hatch)) in enumerate(TREATMENTS.items()):
        x = 108 + (i % 2) * 36; y = 135.5 + (i // 2) * 2.5
        f.text("f2", x, y, "■ " + label, 6, color=color if i != 5 else GREY)
    f.text("f2", 108, 143.5, "Trace uncertain: GEAR 27/44; Full 24/44", 6)


def crop_svg(source: Path, destination: Path, box: tuple[float, float, float, float], prefixes: list[str]) -> None:
    root = ET.parse(source).getroot()
    namespace = "{http://www.w3.org/2000/svg}"
    figure = root.find(namespace + "g")
    for child in list(figure):
        gid = child.get("id", "")
        if gid != "patch_1" and not any(gid == p or gid.startswith(p + "_") for p in prefixes):
            figure.remove(child)
    x, y, w, h = box
    x, y, w, h = x - .8, y - .8, w + 1.6, h + 1.6
    root.set("viewBox", f"{x * PT} {y * PT} {w * PT} {h * PT}")
    root.set("width", f"{w * SCALE}mm"); root.set("height", f"{h * SCALE}mm")
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)


def render() -> None:
    import cairosvg

    fonts(OUT)
    f = Figure()
    for draw in [panel_a, panel_b, panel_c, panel_d, panel_e, panel_f]: draw(f)
    for artist in f.fig.findobj(Text):
        artist.set_fontsize(artist.get_fontsize() * SCALE)
    for artist in f.fig.findobj(PathCollection):
        artist.set_sizes(artist.get_sizes() * SCALE * SCALE)
    final = OUT / "final"; final.mkdir(parents=True, exist_ok=True)
    master = final / "Fig4.svg"
    f.fig.savefig(master)
    with mpl.rc_context({"svg.fonttype": "path"}): f.fig.savefig(final / "Fig4_outlined.svg")
    f.fig.savefig(final / "Fig4.pdf")
    for name, scale in [("Fig4.png", 4), ("Fig4_6x.png", 6), ("Fig4_10x.png", 10)]:
        cairosvg.svg2png(url=str(master), write_to=str(final / name), scale=scale)
    for name, box in PANELS.items():
        path = OUT / "panels" / (name + ".svg")
        crop_svg(master, path, (box[0] - 1.2, box[1] - 1.6, box[2] + 2.4, box[3] + 2.4), [name + "_card", name + "_title"] + [k for k in BOXES if k.startswith(name)] + (["b_labels"] if name == "b" else []))
        cairosvg.svg2pdf(url=str(path), write_to=str(path.with_suffix(".pdf")))
        cairosvg.svg2png(url=str(path), write_to=str(path.with_suffix(".png")), scale=6)
    for name, box in BOXES.items():
        path = OUT / "components" / (name + ".svg")
        crop_box = positioned_box(name, box)
        if name.startswith("b"):
            x, y, w, h = crop_box
            crop_box = (x - 1, y, w + 2, h)
        crop_svg(master, path, crop_box, [name])
        cairosvg.svg2pdf(url=str(path), write_to=str(path.with_suffix(".pdf")))
        cairosvg.svg2png(url=str(path), write_to=str(path.with_suffix(".png")), scale=8)
    (OUT / "layouts/style.json").write_text(json.dumps({"canvas_mm": [W*SCALE, H*SCALE], "style": "reference-rounded-cards", "layout_scale": SCALE, "panels": {k: [v*SCALE for v in box] for k,box in PANELS.items()}, "components": {k: [v*SCALE for v in positioned_box(k, box)] for k,box in BOXES.items()}}, indent=2))
    plt.close(f.fig)
    print(f"Rendered {master}; 6 panels and {len(BOXES)} components", flush=True)

"""Coordinate-driven renderer that reproduces the approved Fig. 4 reference."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/fig04/new"
DATA = OUT / "data_20260829"
PACK = ROOT / "outputs/gear/graph_rescue_replication_20260828/expert_annotation_pack"
STRUCTURAL = (
    ROOT
    / "data/calibration/graph_calibration/gear_structural_head_release_v1/validation_report.json"
)

C = {
    "ink": "#182532",
    "text2": "#526575",
    "gear": "#0B3A73",
    "gear_mid": "#6E9FD0",
    "gear_light": "#D8E6F1",
    "aspr": "#07877D",
    "aspr_mid": "#6CB8B1",
    "aspr_light": "#D9F0ED",
    "joint": "#E46800",
    "joint_mid": "#F0A36E",
    "joint_light": "#FFF1E5",
    "red": "#C52E26",
    "gray": "#BFC4C8",
    "gray2": "#E5E8EA",
    "gray3": "#F4F5F6",
    "grid": "#D9DEE2",
    "frame": "#AEBBC4",
    "white": "#FFFFFF",
}
PANELS = {
    "a": [0.006, 0.515, 0.294, 0.415],
    "b": [0.305, 0.515, 0.340, 0.415],
    "c": [0.651, 0.515, 0.343, 0.415],
    "d": [0.006, 0.020, 0.438, 0.480],
    "e": [0.452, 0.020, 0.258, 0.480],
    "f": [0.718, 0.020, 0.276, 0.480],
}


def configure() -> None:
    """Set the compact reference-figure typography."""
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 5.7,
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": C["white"],
            "savefig.facecolor": C["white"],
            "text.color": C["ink"],
        }
    )


def load_data() -> dict[str, Any]:
    """Read only the frozen summaries required by the figure."""
    rows = [
        json.loads(line)
        for line in (PACK / "claim_b_annotations.jsonl").read_text().splitlines()
        if line
    ]
    grouped: dict[str, list[dict[str, dict[str, Any]]]] = defaultdict(list)
    for row in rows:
        grouped[row["task_id"]].append(
            {a["claim_alias"]: a for a in row["assessments"]}
        )
    distributions: dict[str, Counter[tuple[str, str]]] = {}
    for field in ("inventory_valid", "manuscript_support"):
        counter: Counter[tuple[str, str]] = Counter()
        for first, second in grouped.values():
            for claim in first:
                counter[(first[claim][field], second[claim][field])] += 1
        distributions[field] = counter
    review = pd.read_csv(DATA / "reviewer_soft_alignment.csv").set_index("metric")[
        "value"
    ]
    claim = pd.read_csv(DATA / "claim_adoption_validity.csv")
    integration = pd.read_csv(DATA / "integration_validity.csv")
    contrasts = pd.read_csv(DATA / "integration_contrasts.csv")
    preferences = pd.read_csv(DATA / "claim_c_independent_preferences.csv")[
        "preferred_arm"
    ].value_counts()
    graph = pd.read_csv(DATA / "graph_predictive_validity.csv").iloc[0].to_dict()
    structural = json.loads(STRUCTURAL.read_text())["metrics"]
    return {
        "b": distributions,
        "review": review,
        "claim": claim,
        "integration": integration,
        "contrasts": contrasts,
        "preferences": preferences,
        "graph": graph,
        "structural": structural,
    }


def panel(fig: Figure, key: str, title: str, color: str = C["ink"]) -> Axes:
    """Create a framed coordinate panel with a shared header treatment."""
    ax = fig.add_axes(PANELS[key], facecolor=C["white"] if key != "e" else "#FFFDFC")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            fill=False,
            edgecolor=C["frame"],
            lw=0.55,
            transform=ax.transAxes,
        )
    )
    ax.text(0.028, 0.962, key, fontsize=10, weight="bold", va="top", color=color)
    ax.text(0.083, 0.962, title, fontsize=8.7, weight="bold", va="top", color=color)
    return ax


def rect(
    ax: Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    color: str,
    text: str,
    *,
    size: float = 6.1,
    bold: bool = True,
    dashed: bool = False,
) -> None:
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            fill=True,
            facecolor=C["white"],
            edgecolor=color,
            lw=0.8,
            ls=(0, (3, 2)) if dashed else "-",
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color=color,
        weight="bold" if bold else "normal",
        linespacing=1.18,
    )


def connector(
    ax: Axes, points: list[tuple[float, float]], color: str = C["ink"]
) -> None:
    ax.plot(*zip(*points), color=color, lw=0.65, solid_capstyle="butt")


def panel_a(ax: Axes) -> None:
    rect(ax, 0.285, 0.775, 0.41, 0.105, C["gear"], "Submission-time\ninputs only")
    connector(ax, [(0.49, 0.775), (0.49, 0.735), (0.23, 0.735), (0.23, 0.69)])
    connector(ax, [(0.49, 0.735), (0.75, 0.735), (0.75, 0.69)])
    rect(ax, 0.055, 0.585, 0.36, 0.105, C["gear"], "GEAR evidence\nassessment")
    rect(ax, 0.055, 0.415, 0.36, 0.105, C["gear"], "Human expert\nevidence reference")
    rect(ax, 0.585, 0.585, 0.36, 0.105, C["aspr"], "ASPR graph\nforecast")
    rect(ax, 0.585, 0.415, 0.36, 0.105, C["aspr"], "Real future graph\noutcomes")
    connector(ax, [(0.235, 0.585), (0.235, 0.52)])
    connector(ax, [(0.765, 0.585), (0.765, 0.52)])
    connector(ax, [(0.235, 0.415), (0.235, 0.365), (0.49, 0.365), (0.49, 0.305)])
    connector(ax, [(0.765, 0.415), (0.765, 0.365), (0.49, 0.365)])
    rect(
        ax,
        0.285,
        0.18,
        0.41,
        0.12,
        C["joint"],
        "End-to-end held-out\nvalidation",
        size=6.3,
    )
    ax.text(
        0.735,
        0.235,
        "Conditional integration:\nASPR applies after the\nevidence gate.",
        fontsize=4.8,
        color=C["text2"],
        va="center",
    )


def stacked(
    ax: Axes, y: float, values: list[int], colors: list[str], heading: str
) -> None:
    ax.text(
        0.035,
        y + 0.055,
        heading,
        color=C["gear"],
        fontsize=5.8,
        weight="bold",
        va="bottom",
    )
    x, w, h = 0.035, 0.89, 0.052
    total = sum(values)
    for value, color in zip(values, colors):
        width = w * value / total
        ax.add_patch(
            Rectangle((x, y), width, h, facecolor=color, edgecolor=C["white"], lw=0.35)
        )
        ax.text(
            x + width / 2,
            y + h / 2,
            str(value),
            ha="center",
            va="center",
            fontsize=6.7,
            color=C["white"] if color != C["gray2"] else C["ink"],
            weight="bold",
        )
        x += width
    for p, lab in zip((0.035, 0.48, 0.925), ("0%", "50%", "100%")):
        ax.text(p, y - 0.025, lab, fontsize=4.8, ha="center", color=C["text2"])


def panel_b(ax: Axes, data: dict[str, Any]) -> None:
    inv = data["b"]["inventory_valid"]
    sup = data["b"]["manuscript_support"]
    stacked(
        ax,
        0.79,
        [inv[("YES", "YES")], inv[("NO", "NO")], 2],
        [C["gear"], C["gear_mid"], C["gray2"]],
        "Claim inventory validity — 178 / 180 = 98.9% agreement",
    )
    stacked(
        ax,
        0.62,
        [
            sup[("YES", "YES")],
            sup[("PARTIAL", "PARTIAL")],
            sup[("PARTIAL", "YES")] + sup[("YES", "PARTIAL")],
        ],
        [C["gear"], C["gear_mid"], C["gray2"]],
        "Manuscript support — 164 / 180 = 91.1% agreement",
    )
    legend = [
        (C["gear"], "Both valid / both supported"),
        (C["gear_mid"], "Both invalid / both partial"),
        (C["gray2"], "Disagreement"),
    ]
    for i, (color, text) in enumerate(legend):
        x = 0.045 + i * 0.35
        ax.add_patch(
            Rectangle(
                (x, 0.525), 0.028, 0.018, facecolor=color, edgecolor=C["frame"], lw=0.2
            )
        )
        ax.text(x + 0.04, 0.534, text, fontsize=4.6, va="center")
    ax.plot([0.03, 0.97], [0.49, 0.49], color=C["text2"], lw=0.55, ls=(0, (3, 2)))
    ax.text(
        0.035,
        0.445,
        "30-paper audit matrix",
        fontsize=6.2,
        weight="bold",
        color=C["gear"],
    )
    for i in range(30):
        row, col = divmod(i, 6)
        color = C["gear"] if i < 8 else C["gear_mid"] if i < 14 else C["gray2"]
        ax.add_patch(
            Rectangle(
                (0.035 + col * 0.07, 0.37 - row * 0.05),
                0.057,
                0.036,
                facecolor=color,
                edgecolor=C["frame"],
                lw=0.35,
            )
        )
    ax.text(
        0.525, 0.405, "30 audited papers", fontsize=6.6, weight="bold", color=C["gear"]
    )
    ax.text(
        0.525,
        0.325,
        "14 with comparable\nprior art",
        fontsize=6.5,
        weight="bold",
        color=C["gear_mid"],
        linespacing=1.15,
    )
    ax.text(
        0.525,
        0.215,
        "8 completed\ncomparisons",
        fontsize=6.5,
        weight="bold",
        color=C["text2"],
        linespacing=1.15,
    )
    rect(
        ax,
        0.03,
        0.03,
        0.94,
        0.075,
        C["text2"],
        "Blinded AI development audit; not external human validation.",
        size=4.8,
        bold=False,
        dashed=True,
    )


def panel_c(ax: Axes, data: dict[str, Any]) -> None:
    ax.text(
        0.105,
        0.795,
        "published human peer reviews vs paper-mismatched\nnegative control",
        fontsize=5,
        color=C["text2"],
        linespacing=1.25,
    )
    recovery = float(data["review"]["correct_pair_mean_soft_recall"])
    left = ax.inset_axes([0.14, 0.16, 0.22, 0.42])
    left.set_ylim(0, 1)
    left.set_xlim(0, 1)
    left.set_xticks([])
    left.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0", "25", "50", "75", "100"])
    left.grid(axis="y", color=C["grid"], lw=0.5, ls=(0, (2, 2)))
    left.spines[["top", "right"]].set_visible(False)
    left.set_ylabel("Recovery of human concerns (%)", fontsize=5.5)
    left.errorbar(
        0.35, recovery, fmt="o", color=C["gear"], markersize=5.5, capsize=3, lw=0.8
    )
    left.text(
        0.55,
        recovery,
        f"{recovery:.0%}",
        fontsize=8,
        color=C["gear"],
        weight="bold",
        va="center",
    )
    left.set_title("Human concern\nrecovery", fontsize=5.8, weight="bold", pad=7)
    right = ax.inset_axes([0.55, 0.25, 0.35, 0.32])
    right.set_xlim(0, 1)
    right.set_ylim(-0.5, 1.5)
    right.set_yticks([1, 0], ["Correct-paper\nsoft match", "Wrong-paper\ncontrol"])
    right.set_xticks([0, 0.5, 1])
    right.spines[["top", "right", "left"]].set_visible(False)
    right.axvline(0, color=C["text2"], lw=0.5, ls=(0, (3, 2)))
    for y, key, color in [
        (1, "correct_pair_mean_soft_f1", C["gear"]),
        (0, "wrong_paper_mean_soft_f1", C["red"]),
    ]:
        value = float(data["review"][key])
        right.hlines(y, 0, value, color=color, lw=1)
        right.scatter(value, y, color=color, s=30, zorder=3)
        right.text(1.03, y, f"{value:.2f}", color=color, fontsize=6.6, va="center")
    right.set_xlabel("Spearman ρ", fontsize=5.5)
    right.set_title(
        "AI review soft-match to human\nconcerns (Spearman ρ)",
        fontsize=5.8,
        weight="bold",
        pad=8,
    )
    rect(
        ax,
        0.03,
        0.03,
        0.94,
        0.072,
        C["gear"],
        "Correct-paper match  ≫  wrong-paper control",
        size=5.8,
    )


def ci(
    ax: Axes,
    x: float,
    y: float,
    value: float,
    low: float,
    high: float,
    color: str,
    *,
    marker: str = "o",
    xlim: tuple[float, float] = (-0.2, 0.6),
    width: float = 0.18,
    open_marker: bool = False,
) -> None:
    scale = lambda z: x + width * (z - xlim[0]) / (xlim[1] - xlim[0])
    ax.plot([scale(low), scale(high)], [y, y], color=color, lw=0.9)
    ax.plot([scale(low), scale(low)], [y - 0.012, y + 0.012], color=color, lw=0.7)
    ax.plot([scale(high), scale(high)], [y - 0.012, y + 0.012], color=color, lw=0.7)
    ax.scatter(
        scale(value),
        y,
        s=26,
        facecolor=C["white"] if open_marker else color,
        edgecolor=color,
        marker=marker,
        zorder=3,
    )


def panel_d(ax: Axes, data: dict[str, Any]) -> None:
    rect(
        ax,
        0.02,
        0.825,
        0.94,
        0.065,
        C["aspr_mid"],
        "317,262 OOF papers        │        Spearman ρ = 0.7396        │        top-decile lift = 1.734×",
        size=6.2,
    )
    ax.text(
        0.02,
        0.745,
        "Claim adoption attribution generalization",
        color=C["aspr"],
        fontsize=6.4,
        weight="bold",
    )
    ax.text(
        0.02, 0.705, "(Spearman ρ on held-out claims)", fontsize=5.1, color=C["text2"]
    )
    claim = data["claim"].set_index("axis")
    for y, axis, label, fill in [
        (0.625, "temporal", "Strict temporal", True),
        (0.405, "domain", "Leave-domain-out", False),
    ]:
        row = claim.loc[axis]
        ax.text(0.05, y + 0.02, label, fontsize=5.6, weight="bold", color=C["gear"])
        ci(
            ax,
            0.23,
            y,
            row.advantage_over_permutation,
            row.advantage_ci95_low,
            row.advantage_ci95_high,
            C["gear"],
            marker="o",
            open_marker=not fill,
        )
        ax.text(
            0.05,
            y - 0.06,
            f"+{row.advantage_over_permutation:.3f} [{row.advantage_ci95_low:.3f}, {row.advantage_ci95_high:.3f}]",
            fontsize=5.0,
        )
        ax.text(0.05, y - 0.105, f"raw ρ = {row.spearman_rho:.3f}", fontsize=4.9)
        ax.text(
            0.05,
            y - 0.145,
            f"{int(row.papers)} papers / {int(row.claims)} claims",
            fontsize=4.8,
            color=C["text2"],
        )
    ax.plot([0.275, 0.275], [0.18, 0.69], color=C["text2"], lw=0.5, ls=(0, (3, 2)))
    ax.text(0.23, 0.13, "−0.2   0   0.2   0.4   0.6", fontsize=4.7)
    ax.text(0.25, 0.08, "Spearman ρ", fontsize=5.0)
    ax.text(
        0.205,
        0.03,
        "← Worse than chance      Better than chance →",
        fontsize=4.2,
        color=C["text2"],
    )
    ax.text(
        0.59,
        0.745,
        "Future structural consequence prediction",
        color=C["aspr"],
        fontsize=6.2,
        weight="bold",
    )
    ax.text(
        0.59, 0.705, "(Spearman ρ on held-out outcomes)", fontsize=5.0, color=C["text2"]
    )
    cmap = LinearSegmentedColormap.from_list(
        "teal", ["#EFF8F7", "#BBDDD9", "#68B9AE", "#07877D"]
    )
    norm = Normalize(0, 0.5)
    names = [("d_excess", "Excess\ndiffusion"), ("perturbation", "Perturbation")]
    cols = [
        ("forward_temporal_latest", "Temporal"),
        ("leave_one_domain_out", "Domain-out"),
    ]
    for r, (metric, label) in enumerate(names):
        ax.text(0.585, 0.57 - r * 0.19, label, ha="left", va="center", fontsize=5.6)
        for c, (fold, col) in enumerate(cols):
            m = data["structural"][fold][metric]
            x, y = 0.68 + c * 0.14, 0.50 - r * 0.19
            ax.add_patch(
                Rectangle(
                    (x, y),
                    0.14,
                    0.19,
                    facecolor=cmap(norm(m["spearman"])),
                    edgecolor=C["text2"],
                    lw=0.45,
                )
            )
            ax.text(
                x + 0.07,
                y + 0.118,
                f"{m['spearman']:.3f}",
                ha="center",
                fontsize=6.2,
                weight="bold",
                color=C["white"] if m["spearman"] > 0.32 else C["ink"],
            )
            ax.text(
                x + 0.07,
                y + 0.063,
                f"[{m['spearman_ci95_low']:.3f}, {m['spearman_ci95_high']:.3f}]",
                ha="center",
                fontsize=4.5,
            )
            if r == 0:
                ax.text(x + 0.07, 0.69, col, ha="center", fontsize=5.2)
    cb = ax.inset_axes([0.68, 0.12, 0.25, 0.022])
    mpl.colorbar.ColorbarBase(
        cb, cmap=cmap, norm=norm, orientation="horizontal", ticks=[0, 0.25, 0.5]
    )
    cb.tick_params(labelsize=4.5, length=2)
    ax.text(0.805, 0.065, "Spearman ρ (effect size)", ha="center", fontsize=4.8)


def panel_e(ax: Axes, data: dict[str, Any]) -> None:
    ax.text(
        0.05,
        0.84,
        "Stage A development cohort (n = 241 papers)",
        fontsize=5.4,
        color=C["joint"],
    )
    validity = data["integration"].set_index(["cohort", "arm"])
    contrast = data["contrasts"].set_index(["cohort", "contrast"])
    g = validity.loc[("overall_241", "GEAR-only")]
    j = validity.loc[("overall_241", "GEAR+Graph")]
    ax.text(0.05, 0.78, "GEAR only", fontsize=5.5, weight="bold", color=C["gear"])
    ax.text(
        0.05,
        0.72,
        f"{g.spearman_rho:.3f}\n[{g.spearman_ci95_low:.3f}, {g.spearman_ci95_high:.3f}]",
        fontsize=5.0,
        color=C["gear"],
    )
    ax.text(
        0.05, 0.64, "GEAR +\nGraph/HGB", fontsize=5.5, weight="bold", color=C["joint"]
    )
    ax.text(
        0.05,
        0.56,
        f"{j.spearman_rho:.3f}\n[{j.spearman_ci95_low:.3f}, {j.spearman_ci95_high:.3f}]",
        fontsize=5.0,
        color=C["joint"],
    )
    plot = ax.inset_axes([0.32, 0.55, 0.62, 0.23])
    plot.set_xlim(-0.1, 0.3)
    plot.set_ylim(-0.2, 1.2)
    plot.set_yticks([])
    plot.set_xticks([-0.1, 0, 0.1, 0.2, 0.3])
    plot.tick_params(labelsize=4.8)
    plot.spines[["top", "right", "left"]].set_visible(False)
    plot.axvline(0, color=C["text2"], lw=0.5, ls=(0, (2, 2)))
    plot.plot([g.spearman_rho, j.spearman_rho], [0.75, 0.25], color=C["gray"], lw=0.75)
    plot.errorbar(
        g.spearman_rho,
        0.75,
        xerr=[
            [g.spearman_rho - g.spearman_ci95_low],
            [g.spearman_ci95_high - g.spearman_rho],
        ],
        fmt="o",
        color=C["gear"],
        capsize=2,
    )
    plot.errorbar(
        j.spearman_rho,
        0.25,
        xerr=[
            [j.spearman_rho - j.spearman_ci95_low],
            [j.spearman_ci95_high - j.spearman_rho],
        ],
        fmt="o",
        color=C["joint"],
        capsize=2,
    )
    plot.set_xlabel("Spearman ρ", fontsize=5)
    main = contrast.loc[("overall_241", "GEAR+Graph minus GEAR-only")]
    shuffle = contrast.loc[("overall_241", "GEAR+Graph minus GEAR+shuffled-Graph")]
    ax.text(
        0.05,
        0.45,
        "Bootstrap improvement (Δρ)",
        fontsize=5.8,
        weight="bold",
        color=C["joint"],
    )
    ax.text(
        0.05,
        0.35,
        f"Δ = +{main.delta_spearman_rho:.3f}\n[{main.delta_ci95_low:.3f}, {main.delta_ci95_high:.3f}]",
        fontsize=5.3,
        color=C["joint"],
        weight="bold",
    )
    sub = ax.inset_axes([0.36, 0.29, 0.26, 0.17])
    sub.set_xlim(-0.1, 0.3)
    sub.set_ylim(0, 1)
    sub.set_yticks([])
    sub.set_xticks([-0.1, 0, 0.1, 0.2, 0.3])
    sub.tick_params(labelsize=4.5)
    sub.spines[["top", "right", "left"]].set_visible(False)
    sub.axvline(0, color=C["text2"], lw=0.5, ls=(0, (2, 2)))
    sub.errorbar(
        main.delta_spearman_rho,
        0.27,
        xerr=[
            [main.delta_spearman_rho - main.delta_ci95_low],
            [main.delta_ci95_high - main.delta_spearman_rho],
        ],
        fmt="o",
        color=C["joint"],
        capsize=2,
    )
    rect(
        ax,
        0.66,
        0.28,
        0.28,
        0.18,
        C["joint"],
        f"Shuffled control\n(real − shuffled)\n\n+{shuffle.delta_spearman_rho:.3f}\n[{shuffle.delta_ci95_low:.3f}, {shuffle.delta_ci95_high:.3f}]",
        size=4.8,
    )
    ax.add_patch(
        Rectangle(
            (0.04, 0.02),
            0.92,
            0.21,
            fill=False,
            edgecolor=C["joint_mid"],
            lw=0.7,
            ls=(0, (3, 2)),
        )
    )
    ax.text(
        0.06,
        0.19,
        "Held-out directionality (Δρ)",
        fontsize=5.5,
        color=C["joint"],
        weight="bold",
    )
    for x, cohort, label in [
        (0.11, "temporal_49", "Temporal"),
        (0.58, "domain_68", "Domain"),
    ]:
        row = contrast.loc[(cohort, "GEAR+Graph minus GEAR-only")]
        ax.text(x, 0.135, label, fontsize=5.1, weight="bold", color=C["gear"])
        ax.text(
            x,
            0.085,
            f"+{row.delta_spearman_rho:.3f}\n[{row.delta_ci95_low:.3f}, {row.delta_ci95_high:.3f}]",
            fontsize=4.7,
            color=C["joint"],
        )
        tiny = ax.inset_axes([x + 0.16, 0.045, 0.2, 0.09])
        tiny.set_xlim(-0.2, 0.2)
        tiny.set_ylim(0, 1)
        tiny.axis("off")
        tiny.axvline(0, color=C["text2"], lw=0.5, ls=(0, (2, 2)))
        tiny.errorbar(
            row.delta_spearman_rho,
            0.5,
            xerr=[
                [row.delta_spearman_rho - row.delta_ci95_low],
                [row.delta_ci95_high - row.delta_spearman_rho],
            ],
            fmt="o",
            color=C["joint"],
            capsize=2,
        )


def panel_f(ax: Axes, data: dict[str, Any]) -> None:
    ax.text(
        0.08,
        0.84,
        "Independent blinded AI judge; source labels hidden.",
        fontsize=5.1,
        color=C["text2"],
    )
    counts = data["preferences"]
    order = [
        ("GEAR+Graph", C["joint"], "Integrated preferred"),
        ("GEAR-only", C["gear"], "GEAR-only preferred"),
        ("TIE", C["gray"], "Similar / tie"),
    ]
    for i, (key, color, _label) in enumerate(order):
        for j in range(int(counts[key])):
            r, c = divmod(sum(int(counts[x[0]]) for x in order[:i]) + j, 13)
            ax.add_patch(
                Rectangle(
                    (0.05 + c * 0.048, 0.74 - r * 0.048),
                    0.036,
                    0.032,
                    facecolor=color,
                    edgecolor=C["white"],
                    lw=0.25,
                )
            )
    for i, (key, color, label) in enumerate(order):
        y = 0.72 - i * 0.12
        ax.add_patch(Rectangle((0.68, y), 0.05, 0.05, facecolor=color, edgecolor=color))
        n = int(counts[key])
        ax.text(0.77, y + 0.037, label, fontsize=5.6, weight="bold", color=color)
        ax.text(0.77, y - 0.005, f"{n} ({n/78:.1%})", fontsize=5.7, color=C["ink"])
    ax.text(
        0.05,
        0.20,
        "n = 78 total comparisons",
        fontsize=5.5,
        weight="bold",
        color=C["gear"],
    )
    rect(
        ax,
        0.08,
        0.07,
        0.78,
        0.15,
        C["joint"],
        "72.3% integrated win rate\namong 65 decisive comparisons",
        size=7.0,
    )
    ax.text(
        0.47,
        0.02,
        "(decisive = integrated preferred or\nGEAR-only preferred)",
        ha="center",
        fontsize=4.6,
        color=C["text2"],
    )


def build() -> Figure:
    """Assemble the fixed-coordinate six-panel manuscript figure."""
    configure()
    data = load_data()
    fig = plt.figure(figsize=(14.48, 10.86))
    fig.text(
        0.006,
        0.985,
        "Fig. 4 | Complementary validation of evidence-gated structural innovation",
        fontsize=13.5,
        weight="bold",
        va="top",
    )
    fig.text(
        0.006,
        0.955,
        "GEAR evaluates whether contributions are evidence-defensible; conditional on that gate, ASPR/HGB forecasts future adoption, diffusion and structural perturbation.",
        fontsize=7.6,
        color=C["text2"],
        va="top",
    )
    a = panel(fig, "a", "Dual-reference validation design")
    panel_a(a)
    b = panel(fig, "b", "Reliability of GEAR-extracted contributions")
    panel_b(b, data)
    c = panel(fig, "c", "AI review recovers paper-specific\nhuman review concerns")
    panel_c(c, data)
    d = panel(
        fig, "d", "Graph/HGB predicts future adoption and structural consequences"
    )
    panel_d(d, data)
    e = panel(
        fig, "e", "Evidence-gated integration improves\nprospective ranking", C["joint"]
    )
    panel_e(e, data)
    f = panel(
        fig, "f", "Blinded comparison favors\nintegrated final summaries", C["joint"]
    )
    panel_f(f, data)
    return fig


def save(fig: Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in ((".png", {"dpi": 600}), (".pdf", {}), (".svg", {})):
        fig.savefig(
            stem.with_suffix(suffix), facecolor=C["white"], bbox_inches=None, **kwargs
        )


def main() -> int:
    fig = build()
    save(fig, OUT / "figure_full")
    plt.close(fig)
    data = load_data()
    drawers = {
        "a": ("Dual-reference validation design", C["ink"], panel_a),
        "b": ("Reliability of GEAR-extracted contributions", C["ink"], panel_b),
        "c": (
            "AI review recovers paper-specific\nhuman review concerns",
            C["ink"],
            panel_c,
        ),
        "d": (
            "Graph/HGB predicts future adoption and structural consequences",
            C["ink"],
            panel_d,
        ),
        "e": (
            "Evidence-gated integration improves\nprospective ranking",
            C["joint"],
            panel_e,
        ),
        "f": (
            "Blinded comparison favors\nintegrated final summaries",
            C["joint"],
            panel_f,
        ),
    }
    for key, (title, color, drawer) in drawers.items():
        original = PANELS[key]
        PANELS[key] = [0.02, 0.02, 0.96, 0.96]
        panel_figure = plt.figure(figsize=(5.2, 4.2))
        axis = panel(panel_figure, key, title, color)
        if key == "a":
            drawer(axis)
        else:
            drawer(axis, data)
        save(panel_figure, OUT / "panels" / f"fig04_{key}")
        plt.close(panel_figure)
        PANELS[key] = original
    contract = {
        "figure_id": 4,
        "version": "fig4new-reference-layout-v2",
        "renderer": "draw_reference.py",
        "reference_layout": "explicit normalized coordinates; top a/b/c and bottom d/e/f",
        "quantitative_limits": [
            "Panel c displays no unsupported confidence interval or p value.",
            "Panel e uses frozen integration intervals, including holdout intervals crossing zero.",
            "Panel f is an independent blinded AI audit, not human validation.",
        ],
    }
    (OUT / "chart_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

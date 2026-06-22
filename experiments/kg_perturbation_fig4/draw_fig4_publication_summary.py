"""Draw a publication-oriented Fig. 4 validation summary from metric tables."""
from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS = PROJECT_ROOT / "outputs" / "kg_perturbation_fig4_demo50" / "fig4_metrics_summary.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "kg_perturbation_fig4_publication"

ASPECT_COLUMNS = [
    ("Novelty", "novelty_semantic_coverage"),
    ("Significance", "significance_semantic_coverage"),
    ("Prior art", "prior_art_semantic_coverage"),
    ("Evidence / rigor", "evidence_rigor_semantic_coverage"),
    ("Limitations", "limitations_semantic_coverage"),
    ("Future work", "future_work_semantic_coverage"),
]
RELATION_COLUMNS = [
    ("Fully entailed", "entailed_points", "#2563eb"),
    ("Related", "related_points", "#60a5fa"),
    ("No match", "no_match_points", "#d1d5db"),
    ("Contradicted", "contradicted_points", "#ef4444"),
]
TOKENS = {
    "surface": "#f8fafc",
    "panel": "#ffffff",
    "ink": "#0f172a",
    "muted": "#64748b",
    "grid": "#e2e8f0",
    "axis": "#94a3b8",
    "blue": "#2563eb",
    "light_blue": "#bfdbfe",
    "red": "#ef4444",
    "green": "#16a34a",
    "amber": "#f59e0b",
    "grey": "#94a3b8",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dpi", type=int, default=320)
    return parser.parse_args(argv)


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["DejaVu Sans", "sans-serif"],
            "axes.facecolor": TOKENS["panel"],
            "figure.facecolor": TOKENS["surface"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "xtick.color": TOKENS["muted"],
            "ytick.color": TOKENS["muted"],
            "axes.titleweight": "bold",
            "axes.titlecolor": TOKENS["ink"],
        }
    )


def numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    values = df[column].astype(str).str.lower().str.strip()
    return values.isin({"true", "1", "yes", "y"})


def finite_values(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return out


def safe_mean(values: Iterable[Any]) -> Optional[float]:
    nums = finite_values(values)
    return float(np.mean(nums)) if nums else None


def rounded(value: Optional[float], digits: int = 2) -> Optional[float]:
    return None if value is None else round(float(value), digits)


def summarize(df: pd.DataFrame) -> Dict[str, Any]:
    included = bool_series(df, "included_in_main")
    screen = bool_series(df, "screen_pass")
    agent = bool_series(df, "agent_success")
    graph = bool_series(df, "graph_metric_valid")
    local = df.get("retrieval_source", pd.Series([""] * len(df))).astype(str).eq("local_fallback")
    relation_totals = {
        label: float(numeric(df, column).sum(skipna=True))
        for label, column, _ in RELATION_COLUMNS
    }
    return {
        "n_cases": int(len(df)),
        "n_screen_pass": int(screen.sum()),
        "n_agent_success": int(agent.sum()),
        "n_graph_metric_valid": int(graph.sum()),
        "n_local_fallback": int(local.sum()),
        "n_included_main": int(included.sum()),
        "structured_semantic_consistency_mean": rounded(safe_mean(numeric(df, "structured_semantic_consistency_mean")), 3),
        "semantic_claim_alignment_mean": rounded(safe_mean(numeric(df, "semantic_claim_alignment")), 3),
        "text_cosine_mean": rounded(safe_mean(numeric(df, "consistency_cosine")), 3),
        "overclaiming_score_mean": rounded(safe_mean(numeric(df, "overclaiming_score_1_5")), 3),
        "missing_peer_point_rate_mean": rounded(safe_mean(numeric(df, "missing_peer_point_rate")), 3),
        "relation_totals": relation_totals,
    }


def panel_title(ax: plt.Axes, letter: str, title: str, subtitle: str = "") -> None:
    ax.set_title(f"{letter}  {title}", loc="left", fontsize=12, fontweight="bold", pad=10)
    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=8.5, color=TOKENS["muted"], va="bottom")


def style_panel(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", color=TOKENS["grid"], linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])


def draw_sample_funnel(ax: plt.Axes, summary: Dict[str, Any]) -> None:
    labels = ["Audited", "Screen pass", "Agent success", "Graph valid", "Main N"]
    values = [
        summary["n_cases"],
        summary["n_screen_pass"],
        summary["n_agent_success"],
        summary["n_graph_metric_valid"],
        summary["n_included_main"],
    ]
    colors = [TOKENS["grey"], TOKENS["light_blue"], "#93c5fd", TOKENS["blue"], TOKENS["green"]]
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, edgecolor="white")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    xmax = max(values) if values else 1
    ax.set_xlim(0, max(1, xmax * 1.18))
    for idx, value in enumerate(values):
        ax.text(value + xmax * 0.02, idx, str(value), va="center", fontsize=9, color=TOKENS["ink"])
    ax.set_xlabel("Papers")
    panel_title(ax, "a", "Evaluation design and usable sample", "No-leakage dossier -> ASPR -> independent peer labels")
    style_panel(ax)


def draw_alignment_distribution(ax: plt.Axes, df: pd.DataFrame, summary: Dict[str, Any]) -> None:
    values = finite_values(numeric(df, "structured_semantic_consistency_mean"))
    if not values:
        values = finite_values(numeric(df, "semantic_claim_alignment"))
        bins = np.linspace(0, 1, 11)
        xlabel = "Claim alignment"
    else:
        bins = np.linspace(1, 5, 9)
        xlabel = "Structured semantic consistency (1-5)"
    ax.hist(values, bins=bins, color="#93c5fd", edgecolor=TOKENS["blue"], alpha=0.85)
    mean_text = summary.get("structured_semantic_consistency_mean")
    cosine_text = summary.get("text_cosine_mean")
    ax.text(
        0.04,
        0.92,
        f"Mean structured = {mean_text}\nText cosine (supp.) = {cosine_text}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=TOKENS["ink"],
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Papers")
    panel_title(ax, "b", "Structured semantic consistency", "Full-text similarity is supplementary")
    style_panel(ax)


def draw_aspect_coverage(ax: plt.Axes, df: pd.DataFrame) -> None:
    values = [safe_mean(numeric(df, column)) or 0.0 for _, column in ASPECT_COLUMNS]
    labels = [label for label, _ in ASPECT_COLUMNS]
    ax.bar(np.arange(len(values)), values, color="#60a5fa", edgecolor=TOKENS["blue"])
    ax.set_xticks(np.arange(len(values)), labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.02)
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.03, f"{value:.2f}", ha="center", fontsize=8, color=TOKENS["ink"])
    ax.set_ylabel("Semantic coverage")
    panel_title(ax, "c", "Aspect-level reviewer point coverage", "Mean entailment/relatedness by aspect")
    style_panel(ax)


def draw_stance_heatmap(ax: plt.Axes, df: pd.DataFrame) -> None:
    peer = numeric(df, "peer_innovation_stance_1_5").round().astype("Int64")
    agent = numeric(df, "agent_innovation_stance_1_5").round().astype("Int64")
    table = pd.crosstab(peer, agent).reindex(index=range(1, 6), columns=range(1, 6), fill_value=0)
    image = ax.imshow(table.values, cmap="Blues", vmin=0)
    ax.set_xticks(range(5), range(1, 6))
    ax.set_yticks(range(5), range(1, 6))
    for i in range(5):
        for j in range(5):
            value = int(table.values[i, j])
            if value:
                ax.text(j, i, str(value), ha="center", va="center", fontsize=8, color=TOKENS["ink"])
    ax.set_xlabel("ASPR stance")
    ax.set_ylabel("Peer stance")
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    panel_title(ax, "d", "Human vs ASPR innovation stance", "Discrete 1-5 stance pairs")


def draw_relation_mix(ax: plt.Axes, df: pd.DataFrame, summary: Dict[str, Any]) -> None:
    totals = np.array([summary["relation_totals"].get(label, 0.0) for label, _, _ in RELATION_COLUMNS], dtype=float)
    denom = totals.sum()
    shares = totals / denom if denom > 0 else np.zeros_like(totals)
    left = 0.0
    for share, (label, _, color) in zip(shares, RELATION_COLUMNS):
        ax.barh([0], [share], left=left, color=color, edgecolor="white", height=0.34, label=label)
        if share >= 0.08:
            ax.text(left + share / 2, 0, f"{share:.0%}", ha="center", va="center", fontsize=8, color=TOKENS["ink"])
        left += share
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Share of peer-review points")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.45), ncol=2, fontsize=8, frameon=False)
    ax.text(0, 0.34, f"Matched peer points counted: {int(denom)}", fontsize=8, color=TOKENS["muted"])
    panel_title(ax, "e", "Claim-level relation composition", "Low alignment means missing specific reviewer points")
    style_panel(ax)


def draw_graph_alignment_scatter(ax: plt.Axes, df: pd.DataFrame) -> None:
    x = numeric(df, "weighted_score_fig3")
    y = numeric(df, "semantic_claim_alignment")
    included = bool_series(df, "included_in_main")
    ax.scatter(x[~included], y[~included], s=35, color="#cbd5e1", edgecolor="white", label="held out / excluded")
    ax.scatter(x[included], y[included], s=42, color=TOKENS["blue"], edgecolor="white", label="included")
    valid = x.notna() & y.notna()
    if valid.sum() >= 3:
        rho = x[valid].corr(y[valid], method="spearman")
        ax.text(0.04, 0.92, f"Spearman rho = {rho:.2f}", transform=ax.transAxes, fontsize=9, color=TOKENS["ink"])
    ax.set_xlabel("Fig.3 graph perturbation score")
    ax.set_ylabel("Semantic claim alignment")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    panel_title(ax, "f", "Graph evidence is contextual", "Graph score versus peer-point coverage")
    style_panel(ax)


def draw_figure(df: pd.DataFrame, out_dir: Path, dpi: int) -> Dict[str, Any]:
    summary = summarize(df)
    set_style()
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 8.1), dpi=dpi)
    fig.patch.set_facecolor(TOKENS["surface"])
    fig.suptitle(
        "Fig. 4 | Claim-level validation of automated innovation review",
        x=0.03,
        y=0.985,
        ha="left",
        fontsize=17,
        fontweight="bold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.03,
        0.945,
        textwrap.fill(
            "Nature-family transparent peer-review cases; current run reports usable N, semantic consistency, "
            "specific reviewer-point coverage, and failure modes.",
            150,
        ),
        ha="left",
        va="top",
        fontsize=9.5,
        color=TOKENS["muted"],
    )
    draw_sample_funnel(axes[0, 0], summary)
    draw_alignment_distribution(axes[0, 1], df, summary)
    draw_aspect_coverage(axes[0, 2], df)
    draw_stance_heatmap(axes[1, 0], df)
    draw_relation_mix(axes[1, 1], df, summary)
    draw_graph_alignment_scatter(axes[1, 2], df)
    fig.text(
        0.03,
        0.018,
        "Interpretation: Fig.4 validates a calibrated innovation-diagnostic signal, not replacement of full peer review.",
        ha="left",
        va="bottom",
        fontsize=8,
        color=TOKENS["muted"],
    )
    fig.tight_layout(rect=(0.02, 0.045, 0.99, 0.9), h_pad=3.0, w_pad=2.2)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "fig4_claim_validation_summary.png"
    svg_path = out_dir / "fig4_claim_validation_summary.svg"
    json_path = out_dir / "fig4_claim_validation_summary.json"
    fig.savefig(png_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    fig.savefig(svg_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"png": str(png_path), "svg": str(svg_path), "json": str(json_path), "summary": summary}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    metrics = args.metrics.resolve()
    if not metrics.exists():
        raise FileNotFoundError(f"Missing Fig. 4 metrics table: {metrics}")
    df = pd.read_csv(metrics, low_memory=False)
    result = draw_figure(df, args.out_dir.resolve(), int(args.dpi))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

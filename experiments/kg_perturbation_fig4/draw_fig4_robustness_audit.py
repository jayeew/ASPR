"""Draw a robustness and uncertainty audit for Fig. 4 validation metrics."""
from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


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
    ("Entailed", "entailed_points", "#2563eb"),
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
    "green": "#16a34a",
    "red": "#ef4444",
    "amber": "#f59e0b",
    "grey": "#94a3b8",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dpi", type=int, default=320)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260617)
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


def bootstrap_ci(values: Iterable[Any], reps: int, seed: int) -> Dict[str, Optional[float]]:
    nums = np.array(finite_values(values), dtype=float)
    if len(nums) == 0:
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    if len(nums) == 1:
        value = float(nums[0])
        return {"n": 1, "mean": value, "lo": value, "hi": value}
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=float)
    for idx in range(reps):
        means[idx] = rng.choice(nums, size=len(nums), replace=True).mean()
    return {
        "n": int(len(nums)),
        "mean": float(nums.mean()),
        "lo": float(np.quantile(means, 0.025)),
        "hi": float(np.quantile(means, 0.975)),
    }


def value_counts(df: pd.DataFrame, column: str, top_n: int = 6) -> Dict[str, int]:
    if column not in df.columns:
        return {}
    counts = df[column].fillna("").astype(str).replace("", "missing").value_counts().head(top_n)
    return {str(key): int(value) for key, value in counts.items()}


def summarize(df: pd.DataFrame, reps: int, seed: int) -> Dict[str, Any]:
    included = bool_series(df, "included_in_main")
    main_df = df[included].copy()
    source_df = main_df if len(main_df) else df
    intervals = {
        "semantic_claim_alignment": bootstrap_ci(numeric(source_df, "semantic_claim_alignment"), reps, seed),
        "structured_consistency": bootstrap_ci(numeric(source_df, "structured_semantic_consistency_mean"), reps, seed + 1),
        "stance_agreement": bootstrap_ci(numeric(source_df, "innovation_stance_agreement"), reps, seed + 2),
        "missing_peer_point_rate": bootstrap_ci(numeric(source_df, "missing_peer_point_rate"), reps, seed + 3),
        "overclaiming_score": bootstrap_ci(numeric(source_df, "overclaiming_score_1_5"), reps, seed + 4),
    }
    relation_totals = {
        label: float(numeric(source_df, column).sum(skipna=True))
        for label, column, _ in RELATION_COLUMNS
    }
    aspect_means = {
        label: safe_mean(numeric(source_df, column))
        for label, column in ASPECT_COLUMNS
    }
    return {
        "n_cases": int(len(df)),
        "n_included_main": int(included.sum()),
        "n_excluded_or_held_out": int((~included).sum()),
        "n_agent_success": int(bool_series(df, "agent_success").sum()),
        "n_screen_pass": int(bool_series(df, "screen_pass").sum()),
        "n_graph_metric_valid": int(bool_series(df, "graph_metric_valid").sum()),
        "n_readability_available": int(bool_series(df, "readability_available").sum()) if "readability_available" in df.columns else 0,
        "source_for_intervals": "included_in_main" if len(main_df) else "all_rows",
        "intervals": intervals,
        "aspect_means": aspect_means,
        "relation_totals": relation_totals,
        "retrieval_source_counts": value_counts(df, "retrieval_source"),
        "excluded_reason_counts": value_counts(df, "exclusion_reason"),
        "screen_reason_counts": value_counts(df, "screen_reason"),
        "readability_failure_reason_counts": value_counts(df, "readability_failure_reason"),
    }


def panel_title(ax: plt.Axes, letter: str, title: str, subtitle: str = "") -> None:
    ax.set_title(f"{letter}  {title}", loc="left", fontsize=12, fontweight="bold", pad=10)
    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=8.5, color=TOKENS["muted"], va="bottom")


def style_panel(ax: plt.Axes) -> None:
    ax.grid(True, axis="x", color=TOKENS["grid"], linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])


def draw_interval_panel(ax: plt.Axes, summary: Dict[str, Any]) -> None:
    items = [
        ("Claim alignment", "semantic_claim_alignment"),
        ("Structured consistency", "structured_consistency"),
        ("Stance agreement", "stance_agreement"),
        ("Missing point rate", "missing_peer_point_rate"),
        ("Overclaiming", "overclaiming_score"),
    ]
    y = np.arange(len(items))
    means = [summary["intervals"][key]["mean"] for _, key in items]
    lows = [summary["intervals"][key]["lo"] for _, key in items]
    highs = [summary["intervals"][key]["hi"] for _, key in items]
    for idx, (mean, lo, hi) in enumerate(zip(means, lows, highs)):
        if mean is None:
            continue
        ax.plot([lo, hi], [idx, idx], color=TOKENS["blue"], linewidth=3)
        ax.scatter([mean], [idx], color=TOKENS["ink"], s=28, zorder=3)
        ax.text(mean, idx + 0.22, f"{mean:.2f}", ha="center", fontsize=8, color=TOKENS["ink"])
    ax.set_yticks(y, [label for label, _ in items])
    ax.invert_yaxis()
    ax.set_xlim(0, 5.1 if max([m or 0 for m in means]) > 1.2 else 1.02)
    ax.set_xlabel("Bootstrap mean and 95% interval")
    panel_title(ax, "a", "Uncertainty of main validation metrics", f"Intervals from {summary['source_for_intervals']} rows")
    style_panel(ax)


def draw_group_panel(ax: plt.Axes, summary: Dict[str, Any]) -> None:
    labels = ["All", "Screen pass", "Agent success", "Graph valid", "Main"]
    values = [
        summary["n_cases"],
        summary["n_screen_pass"],
        summary["n_agent_success"],
        summary["n_graph_metric_valid"],
        summary["n_included_main"],
    ]
    ax.bar(labels, values, color=[TOKENS["grey"], "#93c5fd", "#60a5fa", TOKENS["blue"], TOKENS["green"]])
    for idx, value in enumerate(values):
        ax.text(idx, value + max(values) * 0.03, str(value), ha="center", fontsize=8)
    ax.set_ylabel("Papers")
    ax.tick_params(axis="x", rotation=20)
    panel_title(ax, "b", "Inclusion sensitivity", "Main N is stricter than processed N")
    style_panel(ax)


def draw_aspect_panel(ax: plt.Axes, summary: Dict[str, Any]) -> None:
    labels = list(summary["aspect_means"])
    values = [summary["aspect_means"][label] or 0.0 for label in labels]
    ax.barh(labels, values, color="#60a5fa", edgecolor=TOKENS["blue"])
    ax.set_xlim(0, 1.02)
    ax.invert_yaxis()
    for idx, value in enumerate(values):
        ax.text(value + 0.02, idx, f"{value:.2f}", va="center", fontsize=8)
    ax.set_xlabel("Mean semantic coverage")
    panel_title(ax, "c", "Aspect coverage audit", "Reviewer-point coverage differs by dimension")
    style_panel(ax)


def draw_relation_panel(ax: plt.Axes, summary: Dict[str, Any]) -> None:
    labels = list(summary["relation_totals"])
    totals = np.array([summary["relation_totals"][label] for label in labels], dtype=float)
    colors = [color for _, _, color in RELATION_COLUMNS]
    denom = totals.sum()
    shares = totals / denom if denom > 0 else np.zeros_like(totals)
    ax.bar(labels, shares, color=colors, edgecolor="white")
    for idx, share in enumerate(shares):
        ax.text(idx, share + 0.03, f"{share:.0%}", ha="center", fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Share of matched peer points")
    ax.tick_params(axis="x", rotation=20)
    panel_title(ax, "d", "Claim-matching relation mix", "Related/no-match are not full validation hits")
    style_panel(ax)


def draw_quality_panel(ax: plt.Axes, summary: Dict[str, Any]) -> None:
    counts = summary.get("retrieval_source_counts", {})
    labels = list(counts) or ["missing"]
    values = [counts.get(label, 0) for label in labels]
    ax.barh(labels, values, color="#93c5fd", edgecolor=TOKENS["blue"])
    ax.invert_yaxis()
    ax.set_xlabel("Papers")
    panel_title(ax, "e", "Retrieval provenance", "Local fallback should be excluded from main claims")
    style_panel(ax)


def _top_text(counts: Dict[str, int], limit: int = 4) -> str:
    if not counts:
        return "No recorded issue."
    return "; ".join(f"{key}: {value}" for key, value in list(counts.items())[:limit])


def draw_reason_panel(ax: plt.Axes, summary: Dict[str, Any]) -> None:
    ax.set_axis_off()
    ax.set_title("f  Audit notes for interpretation", loc="left", fontsize=12, fontweight="bold", color=TOKENS["ink"])
    notes = [
        f"Screened cases: {summary['n_cases']}; included main cases: {summary['n_included_main']}; held out/excluded: {summary['n_excluded_or_held_out']}.",
        f"Graph metric valid: {summary['n_graph_metric_valid']}/{summary['n_cases']}; readability available: {summary['n_readability_available']}/{summary['n_cases']}.",
        "Interpretation: current Fig.4 supports a cautious diagnostic-use claim, not replacement of peer review.",
    ]
    y = 0.88
    for note in notes:
        ax.text(0.03, y, textwrap.fill(note, width=72), transform=ax.transAxes, ha="left", va="top", fontsize=9, color=TOKENS["ink"])
        y -= 0.14
    ax.add_patch(Rectangle((0.03, 0.05), 0.92, 0.27, transform=ax.transAxes, facecolor="#f1f5f9", edgecolor="#cbd5e1"))
    ax.text(0.06, 0.27, "Categorized data-quality issues", transform=ax.transAxes, fontsize=8.5, fontweight="bold", color=TOKENS["ink"])
    ax.text(
        0.06,
        0.18,
        textwrap.fill(f"Exclusions: {_top_text(summary.get('excluded_reason_counts', {}))}", width=76),
        transform=ax.transAxes,
        fontsize=8,
        color=TOKENS["muted"],
    )
    ax.text(
        0.06,
        0.09,
        textwrap.fill(f"Screen: {_top_text(summary.get('screen_reason_counts', {}))}", width=76),
        transform=ax.transAxes,
        fontsize=8,
        color=TOKENS["muted"],
    )


def draw_figure(df: pd.DataFrame, summary: Dict[str, Any], out_dir: Path, dpi: int) -> List[Path]:
    set_style()
    fig = plt.figure(figsize=(15.8, 9.2), facecolor=TOKENS["surface"])
    grid = fig.add_gridspec(2, 3, left=0.055, right=0.985, top=0.84, bottom=0.09, hspace=0.54, wspace=0.38)
    axes = [fig.add_subplot(grid[i, j]) for i in range(2) for j in range(3)]
    draw_interval_panel(axes[0], summary)
    draw_group_panel(axes[1], summary)
    draw_aspect_panel(axes[2], summary)
    draw_relation_panel(axes[3], summary)
    draw_quality_panel(axes[4], summary)
    draw_reason_panel(axes[5], summary)
    fig.text(
        0.035,
        0.95,
        "Fig. 4 audit | Robustness and measurement limits of claim-level validation",
        fontsize=22,
        fontweight="bold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.035,
        0.917,
        "Bootstrap uncertainty, inclusion sensitivity, failure-mode composition, and pipeline availability for the validation run.",
        fontsize=12,
        color=TOKENS["muted"],
    )
    fig.text(
        0.035,
        0.025,
        "Source: Fig.4 metrics summary. This audit is descriptive and bounds the claims made from the demo run.",
        fontsize=9,
        color=TOKENS["muted"],
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "fig4_robustness_audit.png"
    svg = out_dir / "fig4_robustness_audit.svg"
    pdf = out_dir / "fig4_robustness_audit.pdf"
    tif = out_dir / "fig4_robustness_audit.tif"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(tif, dpi=dpi, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    return [png, svg, pdf, tif]


def write_summary(summary: Dict[str, Any], out_dir: Path, metrics_path: Path, image_paths: List[Path]) -> Path:
    payload = dict(summary)
    payload["metrics_path"] = str(metrics_path)
    payload["image_paths"] = [str(path) for path in image_paths]
    json_path = out_dir / "fig4_robustness_audit.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return json_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.metrics.exists():
        raise FileNotFoundError(f"Missing Fig. 4 metrics table: {args.metrics}")
    df = pd.read_csv(args.metrics, low_memory=False)
    summary = summarize(df, reps=int(args.bootstrap_reps), seed=int(args.seed))
    image_paths = draw_figure(df, summary, args.out_dir, dpi=int(args.dpi))
    json_path = write_summary(summary, args.out_dir, args.metrics, image_paths)
    for path in image_paths:
        print(path)
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

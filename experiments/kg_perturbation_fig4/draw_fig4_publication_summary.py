"""Draw a compact publication-oriented Fig.4 validation summary."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS = PROJECT_ROOT / "outputs" / "kg_perturbation_fig4" / "fig4_metrics_summary.csv"
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


def numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([float("nan")] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    return df[column].astype(str).str.lower().str.strip().isin({"true", "1", "yes", "y"})


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
    return float(sum(nums) / len(nums)) if nums else None


def rounded(value: Optional[float], digits: int = 3) -> Optional[float]:
    return None if value is None else round(float(value), digits)


def summarize(df: pd.DataFrame) -> Dict[str, Any]:
    included = bool_series(df, "included_in_main")
    analysis_df = df[included].copy() if bool(included.any()) else df.copy()
    screen = bool_series(df, "screen_pass")
    agent = bool_series(df, "agent_success")
    graph = bool_series(df, "graph_metric_valid")
    relation_totals = {label: float(numeric(analysis_df, column).sum(skipna=True)) for label, column, _ in RELATION_COLUMNS}
    relation_total = sum(relation_totals.values())
    strict_claim_recall = safe_mean(numeric(analysis_df, "strict_claim_recall"))
    soft_claim_recall = safe_mean(numeric(analysis_df, "soft_claim_recall"))
    if strict_claim_recall is None and relation_total:
        strict_claim_recall = relation_totals["Fully entailed"] / relation_total
    if soft_claim_recall is None and relation_total:
        soft_claim_recall = (relation_totals["Fully entailed"] + relation_totals["Related"]) / relation_total
    overclaiming_flag_mean = safe_mean(numeric(analysis_df, "overclaiming_flag"))
    low_overclaiming_rate = None if overclaiming_flag_mean is None else 1.0 - overclaiming_flag_mean
    return {
        "n_cases": int(len(df)),
        "n_screen_pass": int(screen.sum()),
        "n_agent_success": int(agent.sum()),
        "n_graph_metric_valid": int(graph.sum()),
        "n_included_main": int(included.sum()),
        "semantic_claim_alignment_mean": rounded(safe_mean(numeric(analysis_df, "semantic_claim_alignment"))),
        "strict_claim_recall_mean": rounded(strict_claim_recall),
        "soft_claim_recall_mean": rounded(soft_claim_recall),
        "stance_exact_agreement_mean": rounded(safe_mean(numeric(analysis_df, "stance_exact_agreement"))),
        "stance_within_one_agreement_mean": rounded(safe_mean(numeric(analysis_df, "stance_within_one_agreement"))),
        "quadratic_weighted_kappa_mean": rounded(safe_mean(numeric(analysis_df, "quadratic_weighted_kappa"))),
        "text_cosine_mean": rounded(safe_mean(numeric(analysis_df, "consistency_cosine"))),
        "overclaiming_score_mean": rounded(safe_mean(numeric(analysis_df, "overclaiming_score_1_5"))),
        "overclaiming_flag_rate_mean": rounded(overclaiming_flag_mean),
        "low_overclaiming_rate_mean": rounded(low_overclaiming_rate),
        "claim_validation_pass_rate_mean": rounded(safe_mean(numeric(analysis_df, "claim_validation_pass"))),
        "contradiction_rate_mean": rounded(safe_mean(numeric(analysis_df, "contradiction_rate"))),
        "missing_peer_point_rate_mean": rounded(safe_mean(numeric(analysis_df, "missing_peer_point_rate"))),
        "relation_totals": relation_totals,
    }


def draw_figure(df: pd.DataFrame, out_dir: Path, dpi: int = 320) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(df)
    included = bool_series(df, "included_in_main")
    analysis_df = df[included].copy() if bool(included.any()) else df.copy()

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        _draw_fallback(summary, out_dir)
        return {"summary": summary, "out_dir": str(out_dir)}

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    colors = ["#94a3b8", "#60a5fa", "#2563eb", "#16a34a"]
    funnel_labels = ["All", "Screen", "Agent", "Main"]
    funnel_values = [summary["n_cases"], summary["n_screen_pass"], summary["n_agent_success"], summary["n_included_main"]]
    axes[0, 0].barh(funnel_labels, funnel_values, color=colors)
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_title("a  No-leakage validation sample")
    axes[0, 0].set_xlabel("Papers")

    score_items = [
        ("Strict recall", summary["strict_claim_recall_mean"]),
        ("Soft recall", summary["soft_claim_recall_mean"]),
        ("Within-1 stance", summary["stance_within_one_agreement_mean"]),
        ("Low overclaiming", summary["low_overclaiming_rate_mean"]),
        ("Validation pass", summary["claim_validation_pass_rate_mean"]),
    ]
    axes[0, 1].barh([item[0] for item in score_items], [0.0 if item[1] is None else float(item[1]) for item in score_items], color="#2563eb")
    axes[0, 1].set_xlim(0, 1.05)
    axes[0, 1].set_title("b  Innovation-review agreement")

    aspect_values = []
    for _, column in ASPECT_COLUMNS:
        value = safe_mean(numeric(analysis_df, column))
        aspect_values.append(0.0 if value is None else value)
    axes[1, 0].bar([label for label, _ in ASPECT_COLUMNS], aspect_values, color="#60a5fa")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].tick_params(axis="x", rotation=25)
    axes[1, 0].set_title("c  Aspect-level coverage")

    relation_values = [summary["relation_totals"][label] for label, _, _ in RELATION_COLUMNS]
    relation_colors = [color for _, _, color in RELATION_COLUMNS]
    axes[1, 1].bar([label for label, _, _ in RELATION_COLUMNS], relation_values, color=relation_colors)
    axes[1, 1].tick_params(axis="x", rotation=20)
    axes[1, 1].set_title("d  Claim-evidence labels")

    for ax in axes.flatten():
        ax.grid(True, axis="y", color="#e2e8f0", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(out_dir / f"fig4_claim_validation_summary.{suffix}", dpi=dpi)
    plt.close(fig)
    (out_dir / "fig4_claim_validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"summary": summary, "out_dir": str(out_dir)}


def _draw_fallback(summary: Mapping[str, Any], out_dir: Path) -> None:
    from PIL import Image, ImageDraw
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    lines = [
        "Fig.4 claim validation summary",
        f"Main N: {summary.get('n_included_main')}",
        f"Strict recall: {summary.get('strict_claim_recall_mean')}",
        f"Soft recall: {summary.get('soft_claim_recall_mean')}",
        f"Low overclaiming: {summary.get('low_overclaiming_rate_mean')}",
    ]
    image = Image.new("RGB", (1100, 720), "white")
    draw = ImageDraw.Draw(image)
    for idx, line in enumerate(lines):
        draw.text((60, 70 + idx * 52), str(line), fill=(15, 23, 42))
    image.save(out_dir / "fig4_claim_validation_summary.png")
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="720">',
        '<rect width="1100" height="720" fill="white"/>',
    ]
    for idx, line in enumerate(lines):
        svg.append(f'<text x="60" y="{90 + idx * 52}" font-size="28" fill="#0f172a">{line}</text>')
    svg.append("</svg>")
    (out_dir / "fig4_claim_validation_summary.svg").write_text("\n".join(svg) + "\n", encoding="utf-8")
    pdf = canvas.Canvas(str(out_dir / "fig4_claim_validation_summary.pdf"), pagesize=letter)
    y = 720
    for line in lines:
        pdf.drawString(72, y, str(line))
        y -= 28
    pdf.save()
    (out_dir / "fig4_claim_validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dpi", type=int, default=320)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    draw_figure(pd.read_csv(args.metrics), args.out_dir, dpi=args.dpi)


if __name__ == "__main__":
    main()

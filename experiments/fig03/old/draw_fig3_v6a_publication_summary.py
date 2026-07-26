from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "fig03"
    / "old"
    / "work"
    / "exact_v6a_locked"
    / "runs"
    / "moderate__RGPM_latent_future_percentile__publication_day_plus__linear"
)
DEFAULT_DECISION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "fig03"
    / "old"
    / "work"
    / "exact_v6a_locked"
    / "fig3_v6a_probe_decision.json"
)
DEFAULT_GATE_PATH = (
    PROJECT_ROOT
    / "data"
    / "knowledge_corpus"
    / "v2_publication_v6a_locked_candidate"
    / "performance_gate_decision_v6a.json"
)
DEFAULT_OUT_DIR = (
    PROJECT_ROOT / "outputs" / "fig03" / "old" / "work" / "v6a_locked"
)


DOMAIN_LABELS = {
    "crispr": "CRISPR",
    "exoplanets": "Exoplanets",
    "gamma_ray_bursts_and_supernovae": "GRB / supernovae",
    "genetics_aging_and_longevity_in_model_organisms": "Longevity genetics",
    "graphene_2d_materials": "Graphene / 2D",
    "ipsc_reprogramming": "iPSC",
    "microbiome_metagenomics": "Microbiome",
    "perovskite_solar_cells": "Perovskite solar",
    "topological_insulators": "Topological insulators",
    "ubiquitin_and_proteasome_pathways": "Ubiquitin / proteasome",
}

VALIDATION_LABELS = {
    "final_materialized": "final",
    "locked_v4_final_bio_methods_phys10": "bio/methods/phys10",
    "independent_v3_all12": "independent all12",
    "independent_v3_strong11_no_magnetic": "strong11 no magnetic",
}


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def add_panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(-0.04, 1.16, label, transform=ax.transAxes, fontsize=18, fontweight="bold", va="top")
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=12)


def draw_flow(ax: plt.Axes, metrics: Dict[str, Any]) -> None:
    ax.axis("off")
    add_panel_label(ax, "a", "Locked v6A main-result policy")
    boxes = [
        ("Publication-day inputs", "7 core indicators\n+ legal graph features"),
        ("Reliability gate", "future citers >= 5\ncontrols >= 75\nstable deltas"),
        ("Latent target", "future graph-delta\nlatent percentile"),
        ("Locked validation", f"OOF rho = {metrics['learned_oof_spearman']:.3f}\nlatest = {metrics['latest_fold_test_spearman']:.3f}"),
    ]
    colors = ["#E8F3FF", "#EAF7EA", "#FFF2DE", "#FCECEC"]
    x_positions = np.linspace(0.03, 0.78, len(boxes))
    for i, ((heading, body), color, x) in enumerate(zip(boxes, colors, x_positions)):
        rect = plt.Rectangle((x, 0.36), 0.19, 0.34, facecolor=color, edgecolor="#6B7280", lw=1.1)
        ax.add_patch(rect)
        ax.text(x + 0.095, 0.61, heading, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(x + 0.095, 0.48, body, ha="center", va="center", fontsize=9)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + 0.23, 0.53), xytext=(x + 0.195, 0.53), arrowprops={"arrowstyle": "->", "lw": 1.4})
    ax.text(
        0.03,
        0.18,
        "Main claim: publication-day graph structure predicts a reliability-gated latent future graph-perturbation percentile.",
        fontsize=9,
        color="#374151",
    )


def draw_domain_balance(ax: plt.Axes, oof: pd.DataFrame, metrics: Dict[str, Any]) -> None:
    add_panel_label(ax, "b", "Reliability-gated 10-domain cohort")
    counts = oof["domain"].astype(str).value_counts().sort_values()
    y = np.arange(len(counts))
    ax.barh(y, counts.values, color="#4C78A8", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels([DOMAIN_LABELS.get(name, name.replace("_", " ")) for name in counts.index], fontsize=8)
    ax.tick_params(axis="y", pad=2)
    ax.set_xlabel("OOF rows")
    ax.grid(axis="x", alpha=0.2)
    ax.text(
        0.98,
        0.06,
        f"rows={metrics['n_rows']}\ndomains={metrics['n_domains']}\nmin/domain={metrics['min_rows_per_domain']}\nmax share={metrics['max_domain_share']:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#CBD5E1"},
    )


def draw_decile_response(ax: plt.Axes, oof: pd.DataFrame, metrics: Dict[str, Any]) -> None:
    add_panel_label(ax, "c", "Out-of-fold score separates latent future perturbation")
    frame = oof[["S_v5_oof", "RGPM_latent_future_percentile"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    frame["decile"] = pd.qcut(frame["S_v5_oof"].rank(method="first"), 10, labels=False) + 1
    grouped = frame.groupby("decile", as_index=False).agg(
        target_mean=("RGPM_latent_future_percentile", "mean"),
        target_median=("RGPM_latent_future_percentile", "median"),
    )
    ax.plot(grouped["decile"], grouped["target_mean"] * 100, marker="o", color="#2563EB", lw=2, label="mean")
    ax.plot(grouped["decile"], grouped["target_median"] * 100, marker="s", color="#111827", lw=1.6, label="median")
    ax.set_xlabel("OOF score decile")
    ax.set_ylabel("Latent future percentile")
    ax.set_ylim(0, 100)
    ax.set_xticks(range(1, 11))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    ax.text(
        0.04,
        0.92,
        f"OOF Spearman = {metrics['learned_oof_spearman']:.3f}\nlearned vs equal = {metrics['learned_vs_equal_delta']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#CBD5E1"},
    )


def draw_fold_validation(ax: plt.Axes, cv: pd.DataFrame, metrics: Dict[str, Any]) -> None:
    add_panel_label(ax, "d", "Time-block validation remains stable")
    cv = cv.copy()
    ax.bar(cv["fold"].astype(str), cv["test_spearman"], color="#10B981", alpha=0.8)
    ax.axhline(0.35, color="#DC2626", ls="--", lw=1.0, label="gate 0.35")
    ax.set_ylabel("Fold test Spearman")
    ax.set_xlabel("Held-out time fold")
    ax.set_ylim(0, max(0.7, float(cv["test_spearman"].max()) + 0.06))
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    ax.text(
        0.96,
        0.92,
        f"latest fold = {metrics['latest_fold_test_spearman']:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#CBD5E1"},
    )


def draw_enrichment(ax: plt.Axes, effect: Dict[str, Any]) -> None:
    add_panel_label(ax, "e", "Top-tail enrichment")
    labels = ["bottom top20", "top top20", "bottom top10", "top top10"]
    values = [
        float(effect.get("bottom_score_decile_rgpm_top20_rate", 0.0)) * 100,
        float(effect.get("top_score_decile_rgpm_top20_rate", 0.0)) * 100,
        float(effect.get("bottom_score_decile_rgpm_top10_rate", 0.0)) * 100,
        float(effect.get("top_score_decile_rgpm_top10_rate", 0.0)) * 100,
    ]
    colors = ["#CBD5E1", "#2563EB", "#FECACA", "#DC2626"]
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Rate (%)")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.2)
    ax.text(
        0.04,
        0.92,
        f"top20 enrichment = {float(effect.get('top_vs_bottom_score_decile_rgpm_top20_enrichment', 0.0)):.1f}x",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#CBD5E1"},
    )


def draw_validations(ax: plt.Axes, gate: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    add_panel_label(ax, "f", "Locked policy replicates across validation corpora")
    rows = [
        {
            "label": "final materialized",
            "oof": metrics["learned_oof_spearman"],
            "latest": metrics["latest_fold_test_spearman"],
        }
    ]
    for row in gate.get("independent_validations", []):
        label = Path(str(row.get("probe_dir", "validation"))).name
        label = label.replace("fig3_v6a_", "").replace("_locked", "")
        rows.append(
            {
                "label": VALIDATION_LABELS.get(label, label),
                "oof": row.get("learned_oof_spearman"),
                "latest": row.get("latest_fold_test_spearman"),
            }
        )
    x = np.arange(len(rows))
    width = 0.36
    ax.bar(x - width / 2, [r["oof"] for r in rows], width, label="OOF", color="#7C3AED", alpha=0.8)
    ax.bar(x + width / 2, [r["latest"] for r in rows], width, label="latest", color="#F97316", alpha=0.8)
    ax.axhline(0.45, color="#111827", ls="--", lw=1.0, label="OOF gate")
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in rows], rotation=18, ha="right", fontsize=8)
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("Spearman")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=8, ncol=3)


def draw(args: argparse.Namespace) -> None:
    run_dir = args.run_dir
    oof = pd.read_csv(run_dir / "fig3_v6a_oof_score_table.csv", low_memory=False)
    cv = pd.read_csv(run_dir / "fig3_v6a_cv_summary.csv")
    effect = read_json(run_dir / "fig3_v6a_effect_summary.json")
    decision = read_json(args.decision_path)
    gate = read_json(args.gate_path)
    metrics = decision["best_run"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(19, 11.5), dpi=300)
    gs = fig.add_gridspec(3, 2, hspace=0.44, wspace=0.32)
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
    draw_flow(axes[0], metrics)
    draw_domain_balance(axes[1], oof, metrics)
    draw_decile_response(axes[2], oof, metrics)
    draw_fold_validation(axes[3], cv, metrics)
    draw_enrichment(axes[4], effect)
    draw_validations(axes[5], gate, metrics)
    fig.suptitle("Fig. 3 v6A | Publication-day graph signatures predict latent future perturbation", fontsize=18, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.958,
        "Locked moderate reliability cohort; seven core indicators plus legal publication-day graph features; no future features used as inputs.",
        ha="center",
        va="center",
        fontsize=10,
        color="#374151",
    )
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(args.out_dir / f"fig3_v6a_publication_summary.{ext}", bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw Fig3 v6A publication summary from locked outputs.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--decision-path", type=Path, default=DEFAULT_DECISION_PATH)
    parser.add_argument("--gate-path", type=Path, default=DEFAULT_GATE_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    draw(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

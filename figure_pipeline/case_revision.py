"""Frozen-source Fig.2/8/9/10 redraws; no model calls or invented outcomes.

Run ``python3 -m figure_pipeline.case_revision`` from the project root.
All coordinates, source records and derived statistics are written separately.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.text import Text

from figure_pipeline.renderers.theme import PALETTE, SEQUENTIAL

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/FROM_WEB/data"
OUT = ROOT / "outputs/FROM_WEB_v2"
INK, MUTED, GRID = "#252525", "#616161", "#dddddd"
CMAP = SEQUENTIAL
CASE = "s41467-026-68288-5"
ROLES = ["METHOD", "FINDING", "MECHANISM", "RESOURCE", "THEORY"]
VENUES = ["Nature Communications", "Communications Biology", "Communications Chemistry", "Communications Physics", "Communications Materials", "Communications Earth & Environment"]
VNAMES = ["Nature Commun.", "Commun. Biology", "Commun. Chemistry", "Commun. Physics", "Commun. Materials", "Commun. Earth & Env."]
SOURCE_HASHES: dict[str, str] = {}


def read(name: str) -> Any:
    """Read immutable inputs and record hashes."""
    path = SOURCE / name
    raw = path.read_bytes()
    SOURCE_HASHES[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    if path.suffix == ".csv":
        return list(csv.DictReader(raw.decode().splitlines()))
    return json.loads(raw)


def dump(name: str, value: Any, folder: str = "data") -> None:
    directory = OUT / folder
    directory.mkdir(parents=True, exist_ok=True)
    (directory / (name + ".json")).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def configure() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.2, "axes.titlesize": 10,
                        "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                        "legend.fontsize": 7.5, "svg.fonttype": "none", "pdf.fonttype": 42,
                        "axes.spines.top": False, "axes.spines.right": False,
                        "axes.edgecolor": MUTED, "text.color": INK, "axes.labelcolor": INK,
                        "axes.linewidth": .6, "savefig.facecolor": "white"})


def canvas(number: int, title: str, subtitle: str, height_mm: float) -> plt.Figure:
    fig = plt.figure(figsize=(180 / 25.4, height_mm / 25.4), dpi=150)
    fig.text(.055, .984, f"Fig. {number} | {title}", va="top", fontsize=12, weight="bold")
    fig.text(.055, .956, subtitle, va="top", fontsize=7.5, color=MUTED)
    return fig


def panel(ax: plt.Axes, letter: str, title: str) -> None:
    ax.set_title(f"{letter}  {title}", loc="left", pad=12, fontsize=10, weight="bold")


def txt(ax: plt.Axes, x: float, y: float, message: str, width: int = 50,
        size: float = 8.2, color: str = INK, bold: bool = False) -> Text:
    message = "\n".join(textwrap.fill(line, width=width, break_long_words=False) for line in message.split("\n"))
    return ax.text(x, y, message, transform=ax.transAxes, va="top", fontsize=size,
                   color=color, weight="bold" if bold else "normal", linespacing=1.3)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float],
          color: str = MUTED) -> None:
    ax.add_patch(FancyArrowPatch(start, end, transform=ax.transAxes, arrowstyle="-|>",
                                mutation_scale=9, lw=.9, color=color))


def block(ax: plt.Axes, rect: tuple[float, float, float, float], title: str, body: str,
          color: str = PALETTE[2], width: int = 27) -> None:
    x, y, w, h = rect
    ax.add_patch(Rectangle((x, y), w, h, transform=ax.transAxes, facecolor="white", edgecolor=GRID, lw=.8))
    ax.plot([x, x+w], [y+h, y+h], transform=ax.transAxes, color=color, lw=2)
    height_mm = ax.figure.get_size_inches()[1]*25.4*ax.get_position().height
    txt(ax, x+.012, y+h-2/height_mm, title, width, 8.5, color, True)
    txt(ax, x+.012, y+h-7/height_mm, body, width, 8)


def export(fig: plt.Figure, number: int, name: str, metadata: dict[str, Any]) -> None:
    """Export exact physical dimensions and explicitly scoped QA results."""
    fig.canvas.draw()
    candidates = list(fig.texts)
    for ax in fig.axes:
        candidates += list(ax.texts)+[ax.title, ax._left_title, ax._right_title]
        if ax.axison:
            candidates += [ax.xaxis.label, ax.yaxis.label]+ax.get_xticklabels()+ax.get_yticklabels()
        if ax.get_legend() is not None:
            candidates += ax.get_legend().get_texts()
    for legend in fig.legends:
        candidates += legend.get_texts()
    text_objects = [x for x in candidates if x.get_text().strip() and x.get_visible()]
    sizes = [x.get_fontsize() for x in text_objects]
    renderer = fig.canvas.get_renderer()
    boundary = fig.bbox
    outside = []
    for x in text_objects:
        bb = x.get_window_extent(renderer)
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > boundary.width+1 or bb.y1 > boundary.height+1:
            outside.append(x.get_text())
    if min(sizes) < 7.5:
        raise ValueError("Final-size typography below 7.5 pt")
    if outside:
        raise ValueError(f"Text outside physical canvas: {outside[:4]}")
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"fig{number:02d}_{name}.{suffix}", dpi=220)
    metadata.update({"source_hashes": SOURCE_HASHES.copy(), "width_mm": 180,
                     "height_mm": float(fig.get_size_inches()[1]*25.4), "minimum_font_pt": min(sizes),
                     "full_fusion_results": "unavailable_in_frozen_snapshot"})
    dump(f"fig{number:02d}_revision_manifest", metadata)
    dump(f"fig{number:02d}_physical_export", {"format": "passed", "font_size": "passed",
         "canvas_bounds": "passed", "text_count": len(text_objects), "minimum_font_pt": min(sizes),
         "label_collisions": "requires_visual_review", "scientific_validity": "not_established_by_export"}, "qa")
    plt.close(fig)


def case_records() -> dict[str, Any]:
    return read("fig10_case_source_records.json")


def source_span(case: dict[str, Any], index: int, span_id: str) -> dict[str, Any]:
    return next(r["payload"] for r in case["gear_traces"][index]
                if r["kind"] == "manuscript_span" and r["payload"]["span_id"] == span_id)


def fig02() -> None:
    case = case_records()
    fig = canvas(2, "Sources, independent branches and synthesis", "Implemented mechanism and one saved evidence chain; whole-paper fusion is a template here.", 248)
    a = fig.add_axes([.055, .48, .89, .43]); a.axis("off"); panel(a, "a", "Processing architecture")
    block(a, (0, .79, 1, .17), "OFFLINE | Historical abstracts + paper citations",
          "Claim Graph, Paper Graph, semantic index and eligibility dates", width=80)
    block(a, (0, .54, .28, .18), "PaperIR", "Parse target full text\nPreserve manuscript spans", width=24)
    block(a, (.36, .54, .28, .18), "Shared grounded claims", "Stable claim identity\nSupported scope + span IDs", width=25)
    arrow(a, (.29, .63), (.35, .63))
    block(a, (.04, .28, .39, .20), "GEAR | Independent branch", "Retrieve and compare prior art\nManuscript support + increment", PALETTE[4], 33)
    block(a, (.56, .28, .40, .20), "Graph | Independent branch", "Eligible neighbors + local insertion\nFacts, paths and interpretation", PALETTE[2], 33)
    arrow(a, (.44, .535), (.23, .49)); arrow(a, (.56, .535), (.76, .49))
    arrow(a, (.92, .785), (.92, .485)); arrow(a, (.30, .785), (.30, .485))
    block(a, (.57, .025, .39, .19), "Joint Graph", "Union neighborhood of claims\nGraph facts + interpretation", PALETTE[2], 32)
    arrow(a, (.76, .275), (.76, .22), PALETTE[2])
    block(a, (.04, .025, .39, .19), "Synthesis outputs", "Claim fusion: GEAR + Graph\nStudy report: + joint Graph", PALETTE[1], 32)
    arrow(a, (.23, .275), (.23, .22), PALETTE[4]); arrow(a, (.55, .34), (.44, .14)); arrow(a, (.56, .12), (.44, .12))
    txt(a, .36, .765, "Shared inputs; independent branch judgments", 55, 7.5, MUTED)
    b = fig.add_axes([.055, .26, .89, .145]); b.axis("off"); panel(b, "b", "One conclusion traced to terminal sources")
    block(b, (0, .05, .29, .84), "Saved finding | C05", "51.3%: optimized versus\nnon-optimized scenarios\n\nGEAR internal-support finding", PALETTE[4], 24)
    block(b, (.35, .05, .28, .84), "Wrapper dependency", "GEAR claim card\n+ direct P: span key\n\nRepeated keys are not new evidence", PALETTE[1], 23)
    block(b, (.69, .05, .31, .84), "Terminal manuscript", "Results, page 5\nS-6b6d9474eb3be5e95c4c\n\nSame source underlies both keys", PALETTE[2], 25)
    arrow(b, (.30, .5), (.34, .5)); arrow(b, (.64, .5), (.68, .5))
    c = fig.add_axes([.055, .06, .89, .13]); c.axis("off"); panel(c, "c", "Output fields and source locations")
    txt(c, 0, .95, "CLAIM ASSESSMENT", 35, 8.5, PALETTE[2], True)
    txt(c, 0, .72, "Identity + source scope\nHistorical comparison + structural interpretation\nEvidence keys + explicit limitations", 48, 8)
    txt(c, .56, .95, "WHOLE-PAPER TEMPLATE", 35, 8.5, PALETTE[1], True)
    txt(c, .56, .72, "Claim-level findings + joint interpretation\nEvidence-linked report passages\nUnresolved questions", 39, 8)
    fig.text(.055, .025, "Reviewer references enter evaluation only. No numeric novelty score; graph paths do not prove antecedence.", fontsize=7.5, color=MUTED)
    span = source_span(case, 4, "S-6b6d9474eb3be5e95c4c")
    dump("fig02_terminal_provenance", {"claim_id": case["claims"][4]["claim_id"],
         "finding": case["gear_assessments"][4]["findings"][0], "terminal_source": span,
         "wrapper_key": "GEAR:"+case["claims"][4]["claim_id"], "independent_source_count_for_displayed_keys": 1})
    export(fig, 2, "mechanism_revision", {"panels": 3, "evidence_status": "implemented_architecture_and_saved_example",
           "claims_not_allowed": ["full fusion empirical efficacy", "key validity implies scientific correctness"]})


def forecast_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    papers = read("fig08_title_corpus.csv")
    rows = read("fig08_direction_results.csv")
    for row in rows:
        for key in ["count_2023", "count_2024", "count_2025", "n_2023", "n_2024", "n_2025", "trend_rank"]:
            row[key] = int(row[key])
        for key in ["frequency_score", "trend_score", "share_2025", "growth_pp", "x", "y"]:
            row[key] = float(row[key])
        row["growth_event"] = row["growth_event"] == "True"
    original = read("fig08_baseline_performance.csv")
    for row in original:
        row["k"] = int(row["k"])
        for key in ("precision_at_k", "ndcg_at_k"):
            row[key] = float(row[key])
    return papers, rows, original


def deduplicate_phrases(papers: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge nested phrases only when pre-cutoff document membership is identical."""
    candidates = {r["phrase"] for r in rows}
    membership: dict[str, set[str]] = {p: set() for p in candidates}
    for paper in papers:
        if int(paper["year"]) >= 2025:
            continue
        tokens = re.findall(r"[a-z]+(?:-[a-z]+)?", paper["title"].lower())
        phrases = {" ".join(tokens[i:i+w]) for w in (2, 3) for i in range(len(tokens)-w+1)}
        for phrase in phrases & candidates:
            membership[phrase].add(paper["paper_id"])
    mapping = []
    for phrase in sorted(candidates):
        equivalents = [p for p in candidates if f" {phrase} " in f" {p} "
                       and membership[p] == membership[phrase]]
        canonical = sorted(equivalents, key=lambda p: (-len(p.split()), -len(p), p))[0]
        mapping.append({"phrase": phrase, "canonical": canonical, "merged": phrase != canonical,
                        "history_document_count": len(membership[phrase]),
                        "reason": "nested_phrase_identical_2023_2024_document_membership" if phrase != canonical else "retained"})
    lookup = {r["phrase"]: r for r in rows}
    kept = [lookup[p] for p in sorted({m["canonical"] for m in mapping})]
    dump("fig08_history_only_deduplication", mapping)
    dump("fig08_deduplicated_candidates", kept)
    return kept


def baseline_performance(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    ideal = sorted((r["share_2025"] for r in rows), reverse=True)
    for method, key in [("Historical frequency", "frequency_score"), ("Linear trend", "trend_score")]:
        ranked = sorted(rows, key=lambda r: (-r[key], r["phrase"]))
        for k in [5, 10, 20, 30, 50, 75, 100]:
            discount = 1 / np.log2(np.arange(k)+2)
            output.append({"method": method, "k": k, "precision_at_k": sum(r["growth_event"] for r in ranked[:k])/k,
                           "ndcg_at_k": float(np.dot([r["share_2025"] for r in ranked[:k]], discount)/np.dot(ideal[:k], discount)),
                           "candidate_pool_size": len(rows), "n_matured_origins": 1})
    return output


def collision_cloud(ax: plt.Axes, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic measured spiral layout, with physical text-box collision tests."""
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    renderer = ax.figure.canvas.get_renderer()
    occupied = []
    layout = []
    for index, row in enumerate(sorted(rows, key=lambda r: (-r["trend_score"], r["phrase"]))[:18]):
        size = 8 + 8*math.sqrt(row["trend_score"]/max(r["trend_score"] for r in rows))
        placed = False
        for attempt in range(600):
            theta = attempt*.53
            radius = .017*math.sqrt(attempt)
            x, y = .5+radius*math.cos(theta), .5+radius*.80*math.sin(theta)
            artist = ax.text(x, y, row["phrase"], fontsize=size, ha="center", va="center",
                             color=PALETTE[index % 5], transform=ax.transAxes)
            bb = artist.get_window_extent(renderer).expanded(1.045, 1.14)
            if (ax.bbox.contains(bb.x0, bb.y0) and ax.bbox.contains(bb.x1, bb.y1)
                    and not any(bb.overlaps(other) for other in occupied)):
                occupied.append(bb); placed = True
                layout.append({"phrase": row["phrase"], "x": x, "y": y, "font_pt": size,
                               "trend_score": row["trend_score"], "layout": "measured_spiral"})
                break
            artist.remove()
            if attempt in (199, 399):
                size = max(7.5, size-1.5)
        if not placed:
            layout.append({"phrase": row["phrase"], "displayed": False, "reason": "no_collision_free_space"})
    return layout


def fig08() -> None:
    papers, original_rows, original_perf = forecast_rows()
    rows = deduplicate_phrases(papers, original_rows)
    perf = baseline_performance(rows)
    dump("fig08_revised_baseline_performance", perf)
    dump("fig08_original_baseline_performance", original_perf)
    n = Counter(int(p["year"]) for p in papers)
    original_random = sum(r["growth_event"] for r in original_rows)/len(original_rows)
    revised_random = sum(r["growth_event"] for r in rows)/len(rows)
    fig = canvas(8, "Title-phrase forecasting baselines", "Fixed 2024 origin; 2025 titles evaluate existing phrases, not scientific innovation or GEAR predictions.", 266)
    a = fig.add_axes([.07, .84, .86, .065]); a.axis("off"); panel(a, "a", "Time split and candidate definition")
    for x, year, label in [(0, 2023, "Candidate history"), (.35, 2024, "Scores frozen"), (.70, 2025, "Evaluation only")]:
        txt(a, x, .95, f"{year} | {n[year]:,} titles", 26, 10, PALETTE[2], True)
        txt(a, x, .52, label, 27, 8)
    txt(a, 0, .05, f"2023 ≥3 and 2024 ≥5 occurrences; {len(original_rows)} original → {len(rows)} history-deduplicated candidates", 95, 7.5, MUTED)
    b = fig.add_axes([.12, .565, .78, .20]); panel(b, "b", "Historical frequency and momentum")
    xs = np.array([100*r["frequency_score"] for r in rows]); ys = np.array([100*(r["frequency_score"]-r["count_2023"]/r["n_2023"]) for r in rows])
    b.axhline(0, color=MUTED, lw=.7)
    b.scatter(xs, ys, s=14, color=PALETTE[2], alpha=.60, edgecolors="white", linewidths=.3)
    b.set_xlabel("2024 title share (%)"); b.set_ylabel("2024 − 2023 share (pp)")
    b.set_ylim(min(ys)-.05, max(ys)+.1); b.grid(alpha=.2)
    for r in sorted(rows, key=lambda r: -r["trend_score"])[:3]:
        b.annotate(r["phrase"], (100*r["frequency_score"], 100*(r["frequency_score"]-r["count_2023"]/r["n_2023"])), xytext=(3, 5), textcoords="offset points", fontsize=7.5)
    c = fig.add_axes([.055, .29, .56, .18]); panel(c, "c", "Predicted title phrases")
    fig.canvas.draw(); cloud_layout = collision_cloud(c, rows); dump("fig08_phrase_cloud_layout", cloud_layout, "layouts")
    cr = fig.add_axes([.67, .29, .28, .18]); cr.axis("off")
    top = sorted(rows, key=lambda r: (-r["trend_score"], r["phrase"]))[:6]
    txt(cr, 0, 1, "Exact predicted share", 29, 8.5, PALETTE[2], True)
    for i, row in enumerate(top):
        txt(cr, 0, .86-i*.145, f"{i+1}. {row['phrase']}\n{100*row['trend_score']:.3f}%", 26, 7.5)
    d1 = fig.add_axes([.105, .090, .35, .128]); panel(d1, "d", "Growth event")
    d2 = fig.add_axes([.60, .090, .35, .128]); d2.set_title("Later popularity", loc="left", fontsize=10, pad=12)
    for data, style in [(original_perf, "--"), (perf, "-")]:
        for method, color in [("Historical frequency", PALETTE[2]), ("Linear trend", PALETTE[4])]:
            sub = [r for r in data if r["method"] == method]
            for ax, key in [(d1, "precision_at_k"), (d2, "ndcg_at_k")]:
                ax.plot([r["k"] for r in sub], [r[key] for r in sub], linestyle=style, color=color,
                        lw=1.1, marker="o" if style == "-" else None, markersize=2.5)
    d1.axhline(revised_random, color=INK, linestyle=":", lw=.8)
    d1.set(xlabel="Top k", ylabel="Precision@k", ylim=(0, 1.03)); d2.set(xlabel="Top k", ylabel="nDCG@k", ylim=(.50, 1.03))
    for ax in (d1, d2):
        ax.set_xticks([10, 50, 100]); ax.grid(alpha=.18)
    handles = [Line2D([], [], color=PALETTE[2], label="Frequency"), Line2D([], [], color=PALETTE[4], label="Trend"),
               Line2D([], [], color=MUTED, label="Deduplicated"), Line2D([], [], color=MUTED, ls="--", label="Original")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .025), ncol=4, frameon=False, columnspacing=1.2)
    fig.text(.055, .009, f"Random growth: original {original_random:.1%}; deduplicated {revised_random:.1%} (dotted). One origin; no temporal-generalization CI.", fontsize=7.5, color=MUTED)
    export(fig, 8, "title_baselines_revision", {"panels": 4, "original_candidates": len(original_rows), "deduplicated_candidates": len(rows),
           "original_random_growth": original_random, "revised_random_growth": revised_random,
           "candidate_rule": "nested and identical pre-2025 document membership; no future values used",
           "space": "two measured historical coordinates, not semantic embedding", "cloud": "measured spiral fallback; WordCloud unavailable",
           "claims_not_allowed": ["frontier novelty", "Graph or fusion forecast benefit", "time-out generalization"]})


def field_group(value: str) -> str:
    if value in ["Biochemistry, Genetics and Molecular Biology", "Medicine", "Neuroscience", "Immunology and Microbiology", "Agricultural and Biological Sciences", "Nursing", "Pharmacology, Toxicology and Pharmaceutics"]:
        return "Life / health"
    if value == "Physics and Astronomy": return "Physics"
    if value in ["Chemistry", "Chemical Engineering"]: return "Chemistry"
    if value in ["Materials Science", "Engineering"]: return "Materials / eng."
    if value in ["Environmental Science", "Earth and Planetary Sciences", "Energy"]: return "Earth / env."
    return "Other"


def mean_ci(values: list[float], seed: int = 20260909) -> tuple[float, float, float]:
    a = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(a, size=(5000, len(a)), replace=True), axis=1)
    lo, hi = np.quantile(means, [.025, .975])
    return float(a.mean()), float(lo), float(hi)


def paper_intervals(ax: plt.Axes, papers: list[dict[str, Any]], metric: str, labels: bool,
                    maximum: float, results: list[dict[str, Any]]) -> None:
    rng = np.random.default_rng(520)
    for i, venue in enumerate(VENUES):
        values = [float(p[metric]) for p in papers if p["journal"] == venue and p[metric] not in ("", None)]
        mean, lo, hi = mean_ci(values)
        ax.scatter(values, i+rng.uniform(-.15, .15, len(values)), s=5, alpha=.25, color=PALETTE[2])
        ax.plot([lo, hi], [i, i], color=PALETTE[0], lw=1.7)
        ax.plot(mean, i, "o", ms=3.7, color=PALETTE[0], markeredgecolor="white", markeredgewidth=.5)
        results.append({"journal": venue, "metric": metric, "n_papers": len(values), "mean": mean, "lo": lo, "hi": hi})
    ax.set(ylim=(5.5, -.5), xlim=(-.025*maximum, 1.04*maximum))
    ax.set_yticks(range(6), VNAMES if labels else [""]*6); ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=.2)


def fig09() -> None:
    papers = read("fig09_paper_rows.csv")
    fig = canvas(9, "Journal samples and contribution profiles", "Frozen 200-paper study cohort: 150 Nature Communications + five specialty groups of ten.", 260)
    results: list[dict[str, Any]] = []
    a = fig.add_axes([.27, .70, .34, .20]); panel(a, "a", "Sample and field composition")
    groups = ["Life / health", "Physics", "Chemistry", "Materials / eng.", "Earth / env.", "Other"]
    counts = np.array([[sum(p["journal"] == venue and field_group(p["field"]) == group for p in papers) for group in groups] for venue in VENUES])
    shares = counts/counts.sum(axis=1, keepdims=True)
    left = np.zeros(6)
    for j, group in enumerate(groups):
        a.barh(range(6), shares[:, j], left=left, color=PALETTE[j], height=.66, label=group, edgecolor="white", linewidth=.3)
        left += shares[:, j]
    a.set_yticks(range(6), VNAMES); a.invert_yaxis(); a.set_xlim(0, 1); a.set_xlabel("Within-journal field share")
    a.legend(ncol=3, loc="upper left", bbox_to_anchor=(-.65, -.24), frameon=False, columnspacing=1)
    ap = fig.add_axes([.66, .70, .30, .20]); ap.axis("off")
    txt(ap, 0, 1.03, "n      OpenAlex / model", 28, 8, PALETTE[2], True)
    for i, venue in enumerate(VENUES):
        sub = [p for p in papers if p["journal"] == venue]
        oa = sum(p["field_source"] == "openalex" for p in sub)
        txt(ap, 0, .94-i*.155, f"{len(sub):3d}      {oa:3d} / {len(sub)-oa}", 30, 8)
    fig.text(.055, .592, "b  Contribution roles | paper proportions, means and 95% bootstrap intervals", fontsize=10, weight="bold")
    for i, role in enumerate(ROLES):
        ax = fig.add_axes([.27+i*.138, .398, .112, .16])
        paper_intervals(ax, papers, role, i == 0, 1, results)
        ax.set_title({"MECHANISM": "Mech.", "RESOURCE": "Resource"}.get(role, role.title()), fontsize=8.5, pad=8)
        ax.set_xticks([0, .5, 1]); ax.set_xlabel("Share")
    fig.text(.055, .338, "c  Structural profiles | each axis keeps its own physical unit", fontsize=10, weight="bold")
    metrics = [("nearest_prior_similarity", "Nearest prior\nsimilarity", 1), ("effective_community_count", "Effective\ncommunities", 10),
               ("connected_pair_share", "Connected-pair\nshare", 1), ("observed_path_share", "Citation-path\nshare", 1)]
    for i, (metric, label, maximum) in enumerate(metrics):
        ax = fig.add_axes([.27+i*.173, .105, .145, .17])
        paper_intervals(ax, papers, metric, i == 0, maximum, results)
        ax.set_title(label, fontsize=8.5, pad=8); ax.set_xticks([0, maximum/2, maximum])
    fig.text(.055, .06, "Dots: paper means; dark points: journal means; intervals resample papers within each journal (5,000 draws).", fontsize=7.5, color=MUTED)
    fig.text(.055, .042, "Fields mix OpenAlex and model assignments. Small specialty samples support description, not journal ranking.", fontsize=7.5, color=MUTED)
    fig.text(.055, .024, "Connected-pair share describes local connectivity. Cross-journal GEAR comparisons await comparable evidence coverage.", fontsize=7.5, color=MUTED)
    dump("fig09_paper_level_intervals", results)
    dump("fig09_field_provenance", {"counts": dict(Counter(p["field_source"] for p in papers)),
         "papers": [{k: p[k] for k in ["paper_id", "journal", "field", "field_source"]} for p in papers]})
    export(fig, 9, "journal_profiles_revision", {"panels": 3, "unit": "paper", "bootstrap_resamples": 5000,
           "field_sources": dict(Counter(p["field_source"] for p in papers)), "claims_not_allowed": ["journal quality ranking", "adjusted GEAR journal effect"]})


def joint_layout(case: dict[str, Any]) -> tuple[nx.Graph, dict[str, np.ndarray], dict[str, int]]:
    historical: dict[str, dict[str, Any]] = {}
    for facts in case["facts"]:
        for neighbor in facts["neighbors"]:
            historical[neighbor["claim_id"]] = neighbor
    graph = nx.Graph(); graph.add_nodes_from(sorted(historical)); graph.add_edges_from(case["joint_facts"]["historical_edges"])
    if len(graph) != case["joint_facts"]["historical_neighbor_count"]:
        raise ValueError("Union historical node inventory mismatch")
    initial = nx.spring_layout(graph, seed=9820, iterations=100, k=1.0, scale=1)
    augmented = graph.copy(); augmented.add_nodes_from(c["claim_id"] for c in case["claims"])
    augmented.add_edges_from(case["joint_facts"]["insertion_edges"])
    seeds = dict(initial)
    for i, claim in enumerate(case["claims"]):
        neighbors = [n["claim_id"] for n in case["facts"][i]["neighbors"]]
        center = np.mean([initial[n] for n in neighbors], axis=0)
        seeds[claim["claim_id"]] = center+np.array([.10*math.cos(i), .10*math.sin(i)])
    positions = nx.spring_layout(augmented, pos=seeds, fixed=list(graph), seed=92, iterations=180, k=.60)
    communities = {node: int(historical[node]["community_id"]) for node in historical}
    dump("fig10_fixed_historical_layout", {"positions": {k: list(map(float, v)) for k, v in positions.items()},
         "historical_nodes": sorted(graph), "historical_edges": list(graph.edges()),
         "historical_isolates": list(nx.isolates(graph)), "communities": communities,
         "layout_rule": "historical spring layout frozen; target nodes optimized with historical positions fixed"}, "layouts")
    return graph, positions, communities


def network(ax: plt.Axes, historical: list[str], historical_edges: list[list[str]], targets: list[str],
            inserted: list[list[str]], pos: dict[str, np.ndarray], communities: dict[str, int]) -> None:
    graph = nx.Graph(); graph.add_nodes_from(historical+targets); graph.add_edges_from(historical_edges+inserted)
    cmap = {v: PALETTE[i % 7] for i, v in enumerate(sorted(set(communities.values())))}
    nx.draw_networkx_edges(graph, pos, edgelist=historical_edges, ax=ax, edge_color=MUTED, width=.8, alpha=.7)
    nx.draw_networkx_edges(graph, pos, edgelist=inserted, ax=ax, edge_color=PALETTE[3], width=.7, alpha=.6, style="dashed")
    nx.draw_networkx_nodes(graph, pos, nodelist=historical, ax=ax, node_size=23,
                           node_color=[cmap[communities[n]] for n in historical], edgecolors="white", linewidths=.4)
    nx.draw_networkx_nodes(graph, pos, nodelist=targets, ax=ax, node_size=70, node_shape="D",
                           node_color="white", edgecolors=PALETTE[0], linewidths=1)
    for target in targets:
        x, y = pos[target]
        ax.annotate("C"+target.split("::")[-1], (x, y), xytext=(4, 5), textcoords="offset points", fontsize=7.5,
                    color=PALETTE[0], weight="bold", bbox={"facecolor": "white", "edgecolor": "none", "pad": .2, "alpha": .8})
    bounds = np.array(list(pos.values())); padding = .17
    ax.set(xlim=(bounds[:, 0].min()-padding, bounds[:, 0].max()+padding), ylim=(bounds[:, 1].min()-padding, bounds[:, 1].max()+padding))
    ax.set_aspect("equal"); ax.axis("off")


def fig10() -> None:
    case = case_records(); snapshot = read("fig04_fig06_fig07_snapshot.json")
    claim = case["claims"][4]
    span = source_span(case, 4, "S-6b6d9474eb3be5e95c4c")
    optimized = re.search(r"(\d+\.?\d*)%", claim["normalized_claim_text"]).group(1)
    extracted = re.search(r"(\d+\.?\d*)%", claim["author_claim_text"]).group(1)
    source_text = " ".join(span["text"].split())
    random = re.search(r"This estimate is (\d+\.?\d*)% higher", source_text).group(1)
    assert optimized+"%" in source_text and random+"%" in source_text
    assert random+"%" in claim["narrowing_reason"]
    reference = next(r for r in case["human_references"]["references"] if r["reference_id"].endswith("0006"))
    assert reference["use_in_main"] is True
    match = next(m for m in snapshot["EVAL"]["graph"][CASE]["evaluation"]["matches"] if m["reference_id"] == reference["reference_id"])
    reviewer_excerpt = "potential underestimation of future carbon sequestration due to overlooked edge effects"
    cleaned_quote = " ".join(reference["source_quote"].split()).replace("o f", "of")
    assert reviewer_excerpt in cleaned_quote
    graph, positions, communities = joint_layout(case)
    fig = canvas(10, "Forestation: a traceable six-claim case", "Enhancing carbon sinks in China using a spatially-optimized forestation strategy | 12 January 2026", 310)
    a = fig.add_axes([.055, .758, .89, .15]); a.axis("off"); panel(a, "a", "C05: the comparator changes the claim")
    txt(a, 0, .98, "SAVED EXTRACTION", 29, 8.5, PALETTE[1], True)
    txt(a, 0, .79, f"≈{extracted}% versus random forestation", 29, 10, PALETTE[1], True)
    txt(a, 0, .47, "Extractor wording; not an author quotation", 28, 7.5, MUTED)
    txt(a, .50, .98, "GROUNDED MANUSCRIPT SCOPE", 38, 8.5, PALETTE[2], True)
    txt(a, .50, .79, f"{optimized}% versus non-optimized\n{random}% versus random forestation", 38, 10, PALETTE[2], True)
    arrow(a, (.40, .67), (.48, .67))
    txt(a, .50, .43, "Different quantities and comparator scopes; projections to 2060", 44, 7.5, MUTED)
    txt(a, 0, .21, "C01  Workflow   C02  Edge pattern   C03  Mechanism\nC04  Planting rules   C05  Carbon projection   C06  Spillover", 87, 8)
    txt(a, 0, -.03, f'Manuscript p.{span["page"]}: “This estimate is {random}% higher ... random forestation scenarios”\n{span["span_id"]} · corrected during shared grounding', 105, 7.5, MUTED)
    b = fig.add_axes([.055, .584, .89, .11]); b.axis("off"); panel(b, "b", "GEAR: comparison evidence and its scope")
    txt(b, 0, 1.02, "C01 | Piao et al., 2005", 36, 8.5, PALETTE[4], True)
    txt(b, 0, .79, "BUILDING_BLOCK · abstract evidence\nShared: inventory-based carbon estimates\nUncovered: edge mapping, random-forest\nscenarios and spatial optimization", 45, 8)
    txt(b, .52, 1.02, "C05 | Forest management, 2020", 40, 8.5, PALETTE[4], True)
    txt(b, .52, .79, "SUPPORT · full-text evidence\nShared: new forests accumulate carbon\nUncovered: optimized 2060 prediction\nand quantitative edge attribution", 42, 8)
    txt(b, 0, .07, "W2005221849 / RS-f028ea96f2511cc544       W3000715334 / RS-3ce0f87db9b1f2769b", 103, 7.5, MUTED)
    txt(b, 0, -.11, "Saved model-coded comparisons; both independent_verification_passed = false.", 103, 7.5, MUTED)
    ca = fig.add_axes([.055, .409, .89, .11]); ca.axis("off"); panel(ca, "c", "Local neighborhoods in the same historical coordinates")
    for j, index in enumerate([0, 3, 4]):
        facts = case["facts"][index]; neighbors = [n["claim_id"] for n in facts["neighbors"]]; target = facts["claim"]["claim_id"]
        ax = fig.add_axes([.055+j*.302, .408, .285, .119])
        network(ax, neighbors, facts["neighbor_edges"], [target], [[target, n] for n in neighbors], positions, communities)
        metric = {m["name"]: m["value"] for m in facts["metrics"]}
        fig.text(.065+j*.302, .397, f"C{index+1:02d} | n={len(neighbors)} | nearest={metric['nearest_prior_similarity']:.3f}", fontsize=7.5)
    d = fig.add_axes([.055, .205, .89, .135]); d.axis("off"); panel(d, "d", "Joint Graph: observed union connectivity")
    dn = fig.add_axes([.055, .193, .45, .165]); targets = [c["claim_id"] for c in case["claims"]]
    joint = case["joint_facts"]
    network(dn, list(graph), joint["historical_edges"], targets, joint["insertion_edges"], positions, communities)
    dm = fig.add_axes([.60, .237, .21, .10]); matrix = np.zeros((6, 6))
    for pair in joint["claim_pairs"]:
        i, j = sorted(targets.index(k) for k in pair["claim_ids"])
        matrix[i, j] = pair["shared_neighbor_count"]; matrix[j, i] = int(pair["connected_via_historical_graph"])
    dm.pcolormesh(np.arange(7)-.5, np.arange(7)-.5,
                  np.where(np.triu(np.ones((6, 6)), 1), matrix, 0), cmap=CMAP, vmin=0, vmax=10,
                  edgecolors="white", linewidth=.3)
    dm.set_xlim(-.5, 5.5); dm.set_ylim(5.5, -.5)
    dm.set_xticks(range(6), [str(i+1) for i in range(6)]); dm.set_yticks(range(6), [str(i+1) for i in range(6)]); dm.tick_params(length=0)
    for i in range(6):
        for j in range(6):
            value = "–" if i == j else str(int(matrix[i, j])) if i < j else "●" if matrix[i, j] else "○"
            dm.text(j, i, value, ha="center", va="center", fontsize=7.5, color="white" if matrix[i, j] < 4 or i >= j else INK)
    txt(d, .53, .16, f"{len(graph)} historical nodes · {joint['historical_components_before']} → {joint['historical_components_after']} components\n{joint['joint_newly_connected_historical_pairs']} newly connected historical pairs", 47, 8, PALETTE[2], True)
    fig.text(.535, .185, "Upper: shared neighbors; lower: historical reachability.\nNumbers describe this temporary union only.", fontsize=7.5, color=MUTED)
    handles = [Line2D([], [], color=MUTED, label="Historical semantic edge"), Line2D([], [], color=PALETTE[3], ls="--", label="Temporary insertion")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.045, .19), frameon=False, ncol=1, fontsize=7.5)
    e = fig.add_axes([.055, .030, .89, .090]); e.axis("off"); panel(e, "e", "Saved Graph + joint report and a main reviewer reference")
    txt(e, 0, 1, "GRAPH + JOINT | translated condensation", 49, 8, PALETTE[2], True)
    txt(e, 0, .76, f"The {optimized}% increase cannot be stated as relative to random forestation; projected carbon gains are not realized ecological outcomes.", 49, 8)
    txt(e, .54, 1, f"REVIEWER {reference['reviewer_id']} · ROUND {reference['round_number']} · HR{reference['reference_id'].split('::')[-1]} · TIER {reference['tier']}", 47, 8, PALETTE[2], True)
    txt(e, .54, .76, f'“{reviewer_excerpt}”', 44, 8)
    txt(e, .54, .24, f"Saved AI match: {match['scope']} scope; {match['reason_coverage']} reason.\nMain reference; no new independent verification.", 49, 7.5, MUTED)
    fig.text(.055, .01, "DOI 10.1038/s41467-026-68288-5 | Original passages and saved matches accompany the figure; full fusion is absent.", fontsize=7.5, color=MUTED)
    dump("fig10_display_texts", {"texts": [t.get_text() for ax in fig.axes for t in ax.texts]+[t.get_text() for t in fig.texts],
         "kind": "English condensation or translation except explicitly quoted source fragments",
         "checks": {"percentages_verified_against_manuscript": [optimized, random], "extracted_percentage": extracted,
                    "main_reference_use_in_main": reference["use_in_main"], "reviewer_quote_verified": True,
                    "reviewer_quote_normalization": "collapse whitespace; repair OCR o f to of"},
         "source_pointers": {"C05_comparators": "fig10_case_source_records.json/claims/4 + gear_traces/4/manuscript_span/"+span["span_id"],
                             "report_condensation": "fig10_case_source_records.json/graph_report/body (fourth contribution and final limitations)",
                             "reviewer_quote": "fig10_case_source_records.json/human_references/references/5/source_quote",
                             "saved_match": "fig04_fig06_fig07_snapshot.json/EVAL/graph/"+CASE+"/evaluation/matches (HR0006)"}})
    dump("fig10_case_trace", {"paper_id": CASE, "claim": case["claims"][4], "manuscript_span": span,
         "graph_report": case["graph_report"], "main_reference": reference, "saved_ai_match": match,
         "display_text_kind": "English condensation or translation except marked reviewer quotation",
         "correction_stage": "shared_claim_grounding", "matrix": matrix.tolist(), "historical_node_count": len(graph),
         "historical_isolates_retained": list(nx.isolates(graph))})
    export(fig, 10, "forestation_revision", {"panels": 5, "paper_id": CASE, "historical_nodes": len(graph),
           "isolated_historical_nodes": len(list(nx.isolates(graph))), "main_reference": reference["reference_id"],
           "use_in_main": reference["use_in_main"], "claims_not_allowed": ["grounding correction caused by fusion", "connectivity proves scientific synergy", "AI match is independent human correctness"]})


EXPLANATIONS = {
    "fig02": {"a": "说明原文解析、共享主张、两个独立分支及联合图如何进入综合输出，按箭头阅读并注意联合图仅接收Graph信息。", "b": "说明一条C05判断如何穿透GEAR包装键回到同一稿件段落，沿三列追溯且不要把重复键当作独立证据。", "c": "说明单主张评估和整篇输出各自应包含哪些可追溯字段，左右对照身份、范围、比较、解释和限制，整篇栏为模板。"},
    "fig08": {"a": "说明既有标题短语的时间切分与历史候选筛选，从2023到2025阅读并注意只有一个预测起点且新出现短语不在候选集。", "b": "说明候选在历史热度与历史增长上的位置，横轴读2024占比、纵轴读2024减2023占比而非语义距离。", "c": "说明仅由线性趋势分数决定的短语优先级，看实际避碰词云的字号并用右栏读取前六项精确预测占比。", "d": "说明增长事件与后续热度是不同预测目标，分别比较Precision和nDCG、虚实线对应原候选和去重候选，并保留历史频率更强的热度结果。"},
    "fig09": {"a": "说明六个期刊样本的学科组成与字段来源，先看样本数再看比例及OpenAlex/model计数，不能当作全刊普查。", "b": "说明各期刊论文内归一后的贡献角色分布，看淡点的论文差异和深色均值区间而非把主张当作独立论文。", "c": "说明各期刊的局部图结构画像，在每个独立单位轴上比较论文点和均值区间，不能跨列比较色深或解释成期刊质量。"},
    "fig10": {"a": "说明C05抽取稿把两种对照范围混在一起，比较51.3%非优化与34.2%随机并沿页码追溯，共享grounding已完成这项修正。", "b": "说明C01与C05的历史共同点及未覆盖范围，分清摘要和全文证据并注意关系来自保存的模型判断而未独立核验。", "c": "说明三条真实主张连接哪些历史邻居，在共同冻结坐标中比较相同节点、历史实线和临时插入虚线。", "d": "说明六条主张同时插入29个历史节点后的局部连通事实，读网络和矩阵上下三角但不要把333个新连通对当作科学协同证据。", "e": "说明实际Graph加joint报告与纳入主分析的审稿参考如何对应，核对报告限定范围、HR0006原话及保存AI匹配，不能视作新增人工核验。"},
}


def main(argv: list[str] | None = None) -> None:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--figures", nargs="+", type=int, default=[2, 8, 9, 10])
    args = parser.parse_args([] if argv is None else argv); OUT = args.output_dir.resolve(); OUT.mkdir(parents=True, exist_ok=True)
    configure()
    for number in args.figures:
        {2: fig02, 8: fig08, 9: fig09, 10: fig10}[number]()
    dump("fig02_fig08_fig09_fig10_explanations_zh", EXPLANATIONS)
    lines = ["# Fig.2/8/9/10：逐panel一句话说明", ""]
    for figure, panels in EXPLANATIONS.items():
        lines += [f"## {figure}", ""]+[f"- **{letter}**：{description}" for letter, description in panels.items()]+[""]
    lines += ["## 数据及解释边界", "", "全部数据读取FROM_WEB冻结输入，源SHA-256记入各图manifest；可重绘不等于完整系统科研复现。", "", "Fig8只合并历史文档集合完全相同的嵌套短语，未完成全部术语语义审定；坐标改用实际历史频率和动量，未伪造语义嵌入。", "", "Fig9区间按论文重采样，无法消除小样本、便利抽样或字段来源差异；跨期刊GEAR效果未估计。", "", "Fig10保留29个历史节点及孤立点，C05纠错来自共享grounding；GEAR关系与审稿匹配都是保存模型结果。"]
    (OUT / "fig02_fig08_fig09_fig10_notes_zh.md").write_text("\n".join(lines), encoding="utf-8")
    print("Rendered", args.figures, "to", OUT)


if __name__ == "__main__":
    main(sys.argv[1:])

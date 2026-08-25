#!/usr/bin/env python3
"""Build reproducible Fig.4 readiness diagnostics from the frozen dev100 run."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVALUATION = PROJECT_ROOT / "outputs/gear/evaluation/nature_dev100_v1_20260824"
DEFAULT_HUMAN = (
    PROJECT_ROOT
    / "outputs/gear/human_review_reconstruction/nature_dev100_human_v2_20260824"
)
DEFAULT_GRAPH = (
    PROJECT_ROOT
    / "outputs/gear/aspr_scoring/work_nature_dev100_20260821/graph_runtime_packets.jsonl"
)
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "outputs/gear/evaluation_setup/nature_dev100_v1/evaluation_manifest_with_runs.json"
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _all_points(review: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    output = [
        ("novelty", point)
        for bucket in ("supporting_points", "limiting_points", "uncertain_points")
        for point in review["novelty"][bucket]
    ]
    output.extend(
        (section, point)
        for section in ("strengths", "weaknesses", "questions")
        for point in review[section]
    )
    return output


def _matched_reference_ids(
    reference_ids: set[str], decisions: list[dict[str, Any]], labels: set[str]
) -> list[dict[str, Any]]:
    ranked = sorted(
        (
            row
            for row in decisions
            if row["reference_point_id"] in reference_ids and row["label"] in labels
        ),
        key=lambda row: float(row["confidence"]),
        reverse=True,
    )
    used_reference: set[str] = set()
    used_candidate: set[str] = set()
    output = []
    for row in ranked:
        if (
            row["reference_point_id"] in used_reference
            or row["candidate_point_id"] in used_candidate
        ):
            continue
        used_reference.add(row["reference_point_id"])
        used_candidate.add(row["candidate_point_id"])
        output.append(row)
    return output


def _family_coverage(
    human: dict[str, dict[str, Any]],
    decisions: dict[str, list[dict[str, Any]]],
    case_to_paper: dict[str, str],
) -> list[dict[str, Any]]:
    totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {"reference": 0.0, "strict": 0.0, "soft": 0.0, "weighted": 0.0}
    )
    for case_id, paper_id in case_to_paper.items():
        points = _all_points(human[paper_id])
        families = {
            "Novelty": {p["point_id"] for section, p in points if section == "novelty"},
            "Significance proxy": {
                p["point_id"]
                for _, p in points
                if p["aspect"] in {"contribution", "results_conclusion"}
            },
            "Evidence": {
                p["point_id"] for _, p in points if p["aspect"] == "experiment_evidence"
            },
            "Limitations / concerns": {
                p["point_id"]
                for section, p in points
                if section in {"weaknesses", "questions"}
            },
        }
        for family, ids in families.items():
            strict = _matched_reference_ids(ids, decisions[case_id], {"SAME_POINT"})
            soft = _matched_reference_ids(
                ids, decisions[case_id], {"SAME_POINT", "PARTIAL_POINT"}
            )
            totals[family]["reference"] += len(ids)
            totals[family]["strict"] += len(strict)
            totals[family]["soft"] += len(soft)
            totals[family]["weighted"] += sum(
                1.0 if row["label"] == "SAME_POINT" else 0.5 for row in soft
            )
    rows = []
    for family, values in totals.items():
        for measure in ("strict", "weighted", "soft"):
            rows.append(
                {
                    "family": family,
                    "measure": measure.title(),
                    "coverage": values[measure] / values["reference"],
                    "reference_points": int(values["reference"]),
                }
            )
    return rows


def _candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    unique = np.unique(scores)
    return np.r_[unique[0] - 1.0, (unique[:-1] + unique[1:]) / 2, unique[-1] + 1.0]


def _threshold_metrics(
    scores: np.ndarray, labels: np.ndarray, threshold: float, *, high_positive: bool
) -> dict[str, Any]:
    predictions = (
        (scores >= threshold).astype(int)
        if high_positive
        else (scores < threshold).astype(int)
    )
    return {
        "threshold": float(threshold),
        "orientation": (
            "higher_score_predicts_positive"
            if high_positive
            else "lower_score_predicts_positive"
        ),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "confusion_mixed_positive": confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).tolist(),
    }


def _best_threshold(
    scores: np.ndarray, labels: np.ndarray, *, high_positive: bool
) -> dict[str, Any]:
    rows = [
        _threshold_metrics(scores, labels, threshold, high_positive=high_positive)
        for threshold in _candidate_thresholds(scores)
    ]
    return max(
        rows,
        key=lambda row: (
            row["balanced_accuracy"],
            row["macro_f1"],
            row["accuracy"],
        ),
    )


def _cross_validated_threshold(
    scores: np.ndarray, labels: np.ndarray, *, high_positive: bool
) -> dict[str, Any]:
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=20260824)
    rows = []
    thresholds = []
    for train, test in splitter.split(scores, labels):
        fitted = _best_threshold(
            scores[train], labels[train], high_positive=high_positive
        )
        thresholds.append(float(fitted["threshold"]))
        rows.append(
            _threshold_metrics(
                scores[test],
                labels[test],
                fitted["threshold"],
                high_positive=high_positive,
            )
        )
    return {
        "fold_count": len(rows),
        "threshold_median": float(np.median(thresholds)),
        "threshold_iqr": [
            float(value) for value in np.quantile(thresholds, [0.25, 0.75])
        ],
        "threshold_range": [min(thresholds), max(thresholds)],
        "balanced_accuracy_mean": mean(row["balanced_accuracy"] for row in rows),
        "accuracy_mean": mean(row["accuracy"] for row in rows),
        "macro_f1_mean": mean(row["macro_f1"] for row in rows),
    }


def _bootstrap_auc(
    scores: np.ndarray, labels: np.ndarray, *, samples: int = 5000
) -> list[float]:
    rng = np.random.default_rng(20260824)
    values = []
    while len(values) < samples:
        index = rng.integers(0, len(labels), len(labels))
        sampled_labels = labels[index]
        if len(np.unique(sampled_labels)) != 2:
            continue
        positive = scores[index][sampled_labels == 1]
        mixed = scores[index][sampled_labels == 0]
        pairwise = positive[:, None] - mixed[None, :]
        values.append(float(np.mean((pairwise > 0) + 0.5 * (pairwise == 0))))
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def _threshold_curve(scores: np.ndarray, labels: np.ndarray) -> list[dict[str, Any]]:
    output = []
    for threshold in range(40, 101, 2):
        for high_positive in (True, False):
            row = _threshold_metrics(
                scores, labels, float(threshold), high_positive=high_positive
            )
            output.append(
                {
                    "threshold": threshold,
                    "orientation": (
                        "Higher score → positive"
                        if high_positive
                        else "Lower score → positive"
                    ),
                    "balanced_accuracy": row["balanced_accuracy"],
                    "accuracy": row["accuracy"],
                }
            )
    return output


def _report_artifact(
    summary: dict[str, Any],
    datasets: dict[str, list[dict[str, Any]]],
    metrics: dict[str, Any],
    ablation: dict[str, Any],
) -> dict[str, Any]:
    paper_macro = metrics["paper_macro"]
    headline = [
        {
            "completed_coverage": metrics["completed_coverage"],
            "human_concern_coverage": paper_macro["human_concern_coverage"],
            "major_support_precision": paper_macro["major_support_precision"],
            "aspr_auc": summary["aspr_auc_higher_predicts_positive"],
            "directional_coverage": metrics["novelty_judgment"]["directional_coverage"],
        }
    ]
    aggregate = ablation["aggregate"]["full-score_only"]
    graph_effects = []
    for label, key in (
        ("Verified relations", "relation_count_delta"),
        ("Independent prior works", "independent_prior_count_delta"),
        ("Analytical Quality", "analytical_quality_delta"),
        ("Major support precision", "major_support_precision_delta"),
        ("Novelty reasoning soft F1", "novelty_reasoning_soft_f1_delta"),
        ("Unsupported major count", "unsupported_major_count_delta"),
    ):
        row = aggregate[key]
        graph_effects.append(
            {
                "metric": label,
                "mean_delta": row["mean"],
                "ci_low": row["bootstrap_95_ci"][0],
                "ci_high": row["bootstrap_95_ci"][1],
                "n": row["n"],
            }
        )
    high = summary["best_in_sample_high_positive_threshold"]
    high_cv = summary["cross_validated_high_positive_threshold"]
    reverse = summary["best_in_sample_reverse_threshold"]
    reverse_cv = summary["cross_validated_reverse_threshold"]
    threshold_summary = [
        {
            "rule": "Always positive baseline",
            "threshold": None,
            "accuracy": summary["majority_positive_baseline_accuracy"],
            "balanced_accuracy": 0.5,
            "macro_f1": None,
        },
        {
            "rule": "Higher score → positive (in-sample)",
            "threshold": high["threshold"],
            "accuracy": high["accuracy"],
            "balanced_accuracy": high["balanced_accuracy"],
            "macro_f1": high["macro_f1"],
        },
        {
            "rule": "Higher score → positive (repeated CV)",
            "threshold": high_cv["threshold_median"],
            "accuracy": high_cv["accuracy_mean"],
            "balanced_accuracy": high_cv["balanced_accuracy_mean"],
            "macro_f1": high_cv["macro_f1_mean"],
        },
        {
            "rule": "Lower score → positive (in-sample)",
            "threshold": reverse["threshold"],
            "accuracy": reverse["accuracy"],
            "balanced_accuracy": reverse["balanced_accuracy"],
            "macro_f1": reverse["macro_f1"],
        },
        {
            "rule": "Lower score → positive (repeated CV)",
            "threshold": reverse_cv["threshold_median"],
            "accuracy": reverse_cv["accuracy_mean"],
            "balanced_accuracy": reverse_cv["balanced_accuracy_mean"],
            "macro_f1": reverse_cv["macro_f1_mean"],
        },
    ]
    datasets = {
        **datasets,
        "headline": headline,
        "graph_effects": graph_effects,
        "threshold_summary": threshold_summary,
    }
    source = {
        "id": "fig4_diagnostics",
        "label": "Frozen Nature dev100 GEAR evaluation and Fig.4 diagnostics",
        "path": "outputs/gear/evaluation/nature_dev100_v1_20260824/fig4_diagnostics/source.sql",
    }
    cards = [
        {
            "id": "completion",
            "description": "Share of frozen dev100 papers with a valid COMPLETE clean review.",
            "dataset": "headline",
            "sourceId": source["id"],
            "metrics": [
                {
                    "label": "Completed clean runs",
                    "field": "completed_coverage",
                    "format": "percent",
                }
            ],
        },
        {
            "id": "concern_coverage",
            "description": "Human concerns matched by an AI concern at SAME or PARTIAL semantic granularity.",
            "dataset": "headline",
            "sourceId": source["id"],
            "metrics": [
                {
                    "label": "Human concern coverage",
                    "field": "human_concern_coverage",
                    "format": "percent",
                }
            ],
        },
        {
            "id": "major_support",
            "description": "Major or critical AI points judged semantically supported.",
            "dataset": "headline",
            "sourceId": source["id"],
            "metrics": [
                {
                    "label": "Major support precision",
                    "field": "major_support_precision",
                    "format": "percent",
                }
            ],
        },
        {
            "id": "aspr_auc",
            "description": "Direction discrimination when a higher ASPR score is treated as evidence of human positive novelty.",
            "dataset": "headline",
            "sourceId": source["id"],
            "metrics": [
                {
                    "label": "ASPR novelty-direction AUC",
                    "field": "aspr_auc",
                    "format": "number",
                }
            ],
        },
        {
            "id": "directional_coverage",
            "description": "Papers where both human and final GEAR novelty labels are directional positive, mixed, or negative.",
            "dataset": "headline",
            "sourceId": source["id"],
            "metrics": [
                {
                    "label": "Final directional coverage",
                    "field": "directional_coverage",
                    "format": "percent",
                }
            ],
        },
    ]
    charts = [
        {
            "id": "alignment_family",
            "title": "AI–human semantic coverage by review family",
            "subtitle": "100 papers; significance is a contribution/results-conclusion proxy rather than a native label.",
            "type": "bar",
            "dataset": "family_coverage",
            "sourceId": source["id"],
            "encodings": {
                "x": {"field": "family", "type": "nominal", "label": "Review family"},
                "y": {
                    "field": "coverage",
                    "type": "quantitative",
                    "label": "Coverage",
                    "format": "percent",
                },
                "color": {
                    "field": "measure",
                    "type": "nominal",
                    "label": "Match definition",
                },
                "tooltip": [
                    {
                        "field": "reference_points",
                        "type": "quantitative",
                        "label": "Human points",
                    }
                ],
            },
            "yAxisTitle": "Human-point coverage",
            "valueFormat": "percent",
            "layout": "full",
        },
        {
            "id": "novelty_confusion",
            "title": "Human and final GEAR novelty judgments",
            "subtitle": "Counts across the frozen 100-paper dev set.",
            "type": "stackedBar",
            "dataset": "novelty_confusion",
            "sourceId": source["id"],
            "encodings": {
                "x": {
                    "field": "human_novelty",
                    "type": "nominal",
                    "label": "Human judgment",
                },
                "y": {
                    "field": "paper_count",
                    "type": "quantitative",
                    "label": "Papers",
                },
                "color": {
                    "field": "final_novelty",
                    "type": "nominal",
                    "label": "Final GEAR judgment",
                },
            },
            "yAxisTitle": "Paper count",
            "valueFormat": "number",
            "layout": "full",
        },
        {
            "id": "novelty_transition",
            "title": "Novelty point buckets before and after evidence verification",
            "subtitle": "Reviewer branch versus final compiled review; counts are review points.",
            "type": "stackedBar",
            "dataset": "novelty_point_transition",
            "sourceId": source["id"],
            "encodings": {
                "x": {"field": "stage", "type": "ordinal", "label": "Stage"},
                "y": {
                    "field": "point_count",
                    "type": "quantitative",
                    "label": "Points",
                },
                "color": {
                    "field": "point_bucket",
                    "type": "nominal",
                    "label": "Novelty bucket",
                },
            },
            "yAxisTitle": "Novelty point count",
            "valueFormat": "number",
            "layout": "full",
        },
        {
            "id": "aspr_distribution",
            "title": "ASPR score distribution by human novelty judgment",
            "subtitle": "Ten-point score bins; human labels contain only positive and mixed cases.",
            "type": "stackedBar",
            "dataset": "score_distribution",
            "sourceId": source["id"],
            "encodings": {
                "x": {
                    "field": "score_bin",
                    "type": "ordinal",
                    "label": "ASPR score bin",
                },
                "y": {
                    "field": "paper_count",
                    "type": "quantitative",
                    "label": "Papers",
                },
                "color": {
                    "field": "human_novelty",
                    "type": "nominal",
                    "label": "Human novelty",
                },
            },
            "yAxisTitle": "Paper count",
            "valueFormat": "number",
            "layout": "full",
        },
        {
            "id": "threshold_curve",
            "title": "ASPR threshold sensitivity for positive-versus-mixed novelty",
            "subtitle": "In-sample balanced accuracy; 0.5 is chance-level balance.",
            "type": "line",
            "dataset": "threshold_curve",
            "sourceId": source["id"],
            "encodings": {
                "x": {
                    "field": "threshold",
                    "type": "quantitative",
                    "label": "ASPR threshold",
                },
                "y": {
                    "field": "balanced_accuracy",
                    "type": "quantitative",
                    "label": "Balanced accuracy",
                },
                "color": {
                    "field": "orientation",
                    "type": "nominal",
                    "label": "Threshold orientation",
                },
            },
            "referenceLines": [{"value": 0.5, "label": "Chance balance"}],
            "yAxisTitle": "Balanced accuracy",
            "valueFormat": "percent",
            "layout": "full",
        },
    ]
    tables = [
        {
            "id": "threshold_summary",
            "title": "ASPR threshold diagnostics",
            "subtitle": "Positive versus mixed human novelty labels; thresholds selected on dev100 are exploratory.",
            "dataset": "threshold_summary",
            "sourceId": source["id"],
            "defaultSort": {"field": "balanced_accuracy", "direction": "desc"},
            "density": "comfortable",
            "layout": "full",
            "columns": [
                {"field": "rule", "label": "Rule", "type": "text"},
                {"field": "threshold", "label": "Threshold", "format": "number"},
                {"field": "accuracy", "label": "Accuracy", "format": "percent"},
                {
                    "field": "balanced_accuracy",
                    "label": "Balanced accuracy",
                    "format": "percent",
                },
                {"field": "macro_f1", "label": "Macro-F1", "format": "number"},
            ],
        },
        {
            "id": "graph_effects",
            "title": "Graph hints versus score-only ablation",
            "subtitle": "Full minus score-only macro deltas with paper-cluster bootstrap 95% intervals.",
            "dataset": "graph_effects",
            "sourceId": source["id"],
            "defaultSort": {"field": "mean_delta", "direction": "desc"},
            "density": "comfortable",
            "layout": "full",
            "columns": [
                {"field": "metric", "label": "Metric", "type": "text"},
                {
                    "field": "mean_delta",
                    "label": "Mean delta",
                    "format": "number",
                    "movement": True,
                },
                {"field": "ci_low", "label": "CI low", "format": "number"},
                {"field": "ci_high", "label": "CI high", "format": "number"},
                {"field": "n", "label": "Papers", "format": "number"},
            ],
        },
    ]
    blocks = [
        {
            "id": "title",
            "type": "markdown",
            "body": "# GEAR Fig.4 readiness: effectiveness, alignment, and ASPR calibration",
        },
        {
            "id": "technical_summary",
            "type": "markdown",
            "sourceId": source["id"],
            "body": "## Technical summary\n\n- **The current data support a GEAR effectiveness panel based on semantic concern coverage and evidence support.** Human-concern coverage is 62.4%, major-support precision is 78.9%, and all 100 clean runs completed.\n- **The current data do not support a claim that final GEAR novelty direction agrees with Nature reviewers.** Directional coverage is 1%, because post-review verification removes or downgrades almost every directional novelty point.\n- **Graph hints demonstrably improve evidence discovery, but not final review quality.** They add verified relations and independent prior works without increasing unsupported-major or leakage counts; quality deltas remain near zero with intervals crossing zero.\n- **A raw ASPR threshold should not be presented as a novelty verdict.** Higher ASPR predicts human positive versus mixed novelty with AUC 0.437, below chance orientation and with a bootstrap interval spanning 0.5.",
        },
        {
            "id": "headline_metrics",
            "type": "metric-strip",
            "cardIds": [
                "completion",
                "concern_coverage",
                "major_support",
                "aspr_auc",
                "directional_coverage",
            ],
        },
        {
            "id": "alignment_finding",
            "type": "markdown",
            "sourceId": source["id"],
            "body": "## Semantic overlap is strongest for evidence, but exact novelty reproduction remains low\n\nSoft one-to-one coverage reaches 84.5% for evidence, 74.1% for the significance proxy, 70.9% for novelty, and 69.4% for limitations/concerns. Weighted coverage is materially lower, showing that many matches share the same scientific concern but differ in granularity or evidentiary scope. The significance result is provisional because the current review contract has no native significance/impact field.",
        },
        {
            "id": "alignment_chart",
            "type": "chart",
            "chartId": "alignment_family",
            "layout": "full",
        },
        {
            "id": "direction_finding",
            "type": "markdown",
            "sourceId": source["id"],
            "body": "## Verification collapses directional novelty into uncertainty\n\nThe graph-blind Reviewer initially produced `mixed` for all 100 papers, with 109 supporting, 108 limiting, and 61 uncertain novelty points. Evidence supervision then left 163 novelty points unresolved and unretained, while 114 were retained as inconclusive questions. The final output contains 60 `uncertain`, 39 `not_discussed`, and only one `mixed` judgment. This is a pipeline-state transition, not evidence that the Reviewer saw no novelty.",
        },
        {
            "id": "confusion_chart",
            "type": "chart",
            "chartId": "novelty_confusion",
            "layout": "full",
        },
        {
            "id": "transition_chart",
            "type": "chart",
            "chartId": "novelty_transition",
            "layout": "full",
        },
        {
            "id": "threshold_finding",
            "type": "markdown",
            "sourceId": source["id"],
            "body": "## No ASPR cutoff is accurate enough to determine novelty color\n\nHuman labels are 61 positive and 39 mixed. Positive papers have a lower mean ASPR score (83.93) than mixed papers (85.77), and the difference is not significant (Mann–Whitney p=0.292). The intuitive in-sample cutoff near 80.2 reaches 60.0% accuracy and 55.7% balanced accuracy, below the 61.0% accuracy of always predicting positive. Repeated cross-validation yields only 54.2% balanced accuracy. The reverse cutoff looks better in-sample but drops to 52.4% balanced accuracy in cross-validation and conflicts with ASPR's propagation interpretation.",
        },
        {
            "id": "score_chart",
            "type": "chart",
            "chartId": "aspr_distribution",
            "layout": "full",
        },
        {
            "id": "threshold_chart",
            "type": "chart",
            "chartId": "threshold_curve",
            "layout": "full",
        },
        {
            "id": "threshold_table",
            "type": "table",
            "tableId": "threshold_summary",
            "layout": "full",
        },
        {
            "id": "graph_finding",
            "type": "markdown",
            "sourceId": source["id"],
            "body": "## Graph supports a retrieval-efficiency claim, not a novelty-accuracy claim\n\nCompared with score-only, full Graph hints add 1.70 verified relations and 0.77 independent prior works per paper; 62% of papers change their verified relation set. Analytical Quality changes by -0.004 and major-support precision by -0.0235, with both intervals crossing zero. ASPR score itself is uncorrelated with relation gain (Spearman ρ=-0.032, p=0.750), while p_uptake is nearly saturated at 0.998 and feature coverage is 1.0 for every paper. The useful component is currently the seed-work guidance, not the scalar score.",
        },
        {
            "id": "graph_table",
            "type": "table",
            "tableId": "graph_effects",
            "layout": "full",
        },
        {
            "id": "definitions",
            "type": "markdown",
            "body": "## Scope and metric definitions\n\nThe population is the frozen 100-paper Nature Communications development set with reconstructed transparent review histories. `Soft` coverage counts SAME_POINT and PARTIAL_POINT equally; `Weighted` coverage counts them as 1 and 0.5; `Strict` counts SAME_POINT only. `Significance proxy` combines contribution and results/conclusion aspects because significance is not explicitly represented. Results are descriptive and development-only, not confirmatory or causal.",
        },
        {
            "id": "limitations",
            "type": "markdown",
            "body": "## Limitations and robustness boundaries\n\n- Human novelty labels contain no negative cases, so negative-direction calibration cannot be evaluated.\n- Thresholds and category definitions were inspected on the same dev100 set; any cutoff requires a new frozen holdout.\n- The significance and limitations panels currently rely on proxies rather than native reconstruction labels.\n- ASPR measures expected scientific uptake/diffusion, not paper quality, novelty truth, significance, or acceptance probability.\n- Graph ablation proves additional evidence acquisition, but final quality improvements are not statistically resolved.",
        },
        {
            "id": "recommendations",
            "type": "markdown",
            "body": "## Recommended fixes before drawing the publication figure\n\n1. **Separate novelty evidence state from novelty direction.** Preserve the Reviewer direction and attach `verified`, `partially_verified`, or `insufficient_coverage` instead of converting all unresolved directional points into an `uncertain` judgment.\n2. **Add native `significance_impact` and `limitations_scope` labels to the human and AI evaluation sidecars.** Rejudge these families rather than publishing proxy labels as ground truth.\n3. **Use ASPR thresholds only for action routing.** Freeze percentile gates at dev100 Q1=77.37 and Q3=97.14: low prioritizes adoption/barrier checks, middle uses balanced retrieval, and high prioritizes direct-antecedent/downstream-diffusion checks. Do not map these gates to positive/mixed/negative novelty.\n4. **Repair Graph search terms and reduce scalar saturation.** Search-term unique prior yield is zero, p_uptake is almost constant, and feature coverage has no variance. Validate recalibration on a confirmatory holdout before claiming score effectiveness.\n5. **Rerun Fig.4 metrics after direction preservation.** Require directional coverage above a predeclared floor and report confusion, semantic coverage, support guardrails, and Graph ablation together.",
        },
        {
            "id": "questions",
            "type": "markdown",
            "body": "## Further questions\n\n- Can the final novelty direction be preserved without allowing unsupported priority claims into the review text?\n- Which transparent-review statements should define significance independently of contribution accuracy?\n- Does a newly frozen holdout contain enough negative novelty cases to calibrate three-way direction?\n- Do graph seeds selected by bibliographic coupling outperform title/search-term hints under the same retrieval budget?",
        },
    ]
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "GEAR Fig.4 readiness: effectiveness, alignment, and ASPR calibration",
            "description": "Technical diagnostic for publication-figure claims using the frozen Nature dev100 evaluation.",
            "generatedAt": "2026-08-24T11:31:00+08:00",
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": [source],
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": "2026-08-24T11:31:00+08:00",
            "status": "ready",
            "datasets": datasets,
        },
        "sources": [source],
    }


def _novelty_stage_diagnostics(
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    initial_judgments: Counter[str] = Counter()
    final_judgments: Counter[str] = Counter()
    initial_points: Counter[str] = Counter()
    final_points: Counter[str] = Counter()
    canonical_sections: Counter[str] = Counter()
    canonical_status: Counter[str] = Counter()
    notes: Counter[str] = Counter()
    for case in manifest["cases"]:
        bundle = json.loads(
            (Path(case["clean_run_dir"]) / "review_bundle.json").read_text()
        )
        initial = bundle["agent_review"]["novelty"]
        final = bundle["structured_review"]["novelty"]
        initial_judgments[initial["judgment"]] += 1
        final_judgments[final["judgment"]] += 1
        initial_ids = set()
        for bucket in ("supporting_points", "limiting_points", "uncertain_points"):
            initial_points[bucket] += len(initial[bucket])
            final_points[bucket] += len(final[bucket])
            initial_ids.update(point["point_id"] for point in initial[bucket])
        for point in bundle["state_v3"]["canonical_points"].values():
            source_ids = point["source_point_ids"].get("agent_reviewer", [])
            if not initial_ids.intersection(source_ids):
                continue
            canonical_sections[point["section"]] += 1
            canonical_status[
                f"{point['validation_status']}|retained={point['retained']}"
            ] += 1
            for note in point.get("validation_notes", []):
                notes[note.split(":", 1)[0]] += 1
    transition_rows = []
    for stage, counts in (("Reviewer", initial_points), ("Final", final_points)):
        for bucket, count in counts.items():
            transition_rows.append(
                {"stage": stage, "point_bucket": bucket, "point_count": count}
            )
    return (
        {
            "initial_judgments": dict(initial_judgments),
            "final_judgments": dict(final_judgments),
            "initial_points": dict(initial_points),
            "final_points": dict(final_points),
            "canonical_final_sections": dict(canonical_sections),
            "canonical_validation_status": dict(canonical_status),
            "top_validation_notes": dict(notes.most_common(12)),
        },
        transition_rows,
    )


def _graph_diagnostics(
    graph_rows: list[dict[str, Any]],
    ablation: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    scores = [float(row["score_0_100"]) for row in graph_rows]
    comparisons = [
        row for row in ablation["comparisons"] if row["comparison"] == "full-score_only"
    ]
    by_case = {row["case_id"]: row for row in comparisons}
    full_rows = {row["case_id"]: row for row in manifest["cases"]}
    paired_scores = []
    relation_deltas = []
    quality_deltas = []
    for case_id, row in by_case.items():
        paired_scores.append(float(full_rows[case_id]["graph_result"]["score_0_100"]))
        relation_deltas.append(float(row["relation_count_delta"]))
        quality_deltas.append(float(row["analytical_quality_delta"]))
    relation_corr = spearmanr(paired_scores, relation_deltas)
    quality_corr = spearmanr(paired_scores, quality_deltas)
    return {
        "score_mean": mean(scores),
        "score_median": median(scores),
        "score_range": [min(scores), max(scores)],
        "score_quartiles": [
            float(value) for value in np.quantile(scores, [0.25, 0.5, 0.75])
        ],
        "mean_p_uptake": mean(float(row["p_uptake"]) for row in graph_rows),
        "mean_conditional_diffusion": mean(
            float(row["conditional_diffusion"]) for row in graph_rows
        ),
        "feature_coverage_values": sorted(
            {float(row["feature_coverage"]) for row in graph_rows}
        ),
        "score_relation_delta_spearman": {
            "rho": float(relation_corr.statistic),
            "pvalue": float(relation_corr.pvalue),
        },
        "score_quality_delta_spearman": {
            "rho": float(quality_corr.statistic),
            "pvalue": float(quality_corr.pvalue),
        },
        "guidance_acceptance_met": ablation["graph_guidance_acceptance_met"],
        "usefulness_criterion_met": ablation["graph_usefulness_criterion_met"],
        "shuffled_graph_harm_rate": ablation["shuffled_graph_harm_rate"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--human-dir", type=Path, default=DEFAULT_HUMAN)
    parser.add_argument("--graph-results", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or (args.evaluation_dir / "fig4_diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)

    human_rows = _jsonl(args.human_dir / "human_structured_reviews.jsonl")
    human = {row["paper_id"]: row for row in human_rows}
    graph_rows = _jsonl(args.graph_results)
    graph = {row["paper_id"]: row for row in graph_rows}
    manifest = json.loads(args.manifest.read_text())
    case_to_paper = {row["case_id"]: row["paper_id"] for row in manifest["cases"]}
    judge_rows = _jsonl(args.evaluation_dir / "judge_decisions.jsonl")
    decisions = {
        row["case_id"]: row["decision"]["decisions"]
        for row in judge_rows
        if row["kind"] == "matches"
    }
    scores = np.array(
        [float(graph[row["paper_id"]]["score_0_100"]) for row in human_rows]
    )
    labels = np.array(
        [int(row["novelty"]["judgment"] == "positive") for row in human_rows]
    )
    human_labels = [row["novelty"]["judgment"] for row in human_rows]
    positive_scores = scores[labels == 1]
    mixed_scores = scores[labels == 0]
    mann_whitney = mannwhitneyu(positive_scores, mixed_scores, alternative="two-sided")
    initial_best = _best_threshold(scores, labels, high_positive=True)
    reverse_best = _best_threshold(scores, labels, high_positive=False)
    stage, transition_rows = _novelty_stage_diagnostics(manifest)
    family_rows = _family_coverage(human, decisions, case_to_paper)
    ablation = json.loads(
        (args.evaluation_dir / "graph_ablation_metrics.json").read_text()
    )
    metrics = json.loads((args.evaluation_dir / "metrics.json").read_text())

    score_distribution = []
    for lower in range(30, 101, 10):
        upper = lower + 10
        for label in ("mixed", "positive"):
            score_distribution.append(
                {
                    "score_bin": f"{lower}-{upper}",
                    "human_novelty": label,
                    "paper_count": int(
                        sum(
                            lower <= score < upper and human_label == label
                            for score, human_label in zip(
                                scores, human_labels, strict=True
                            )
                        )
                    ),
                }
            )
    confusion_rows = [
        {
            "human_novelty": left,
            "final_novelty": right,
            "paper_count": count,
        }
        for key, count in metrics["novelty_judgment"]["judgment_confusion"].items()
        for left, right in [key.split("->", 1)]
    ]
    summary = {
        "contract": "gear_fig4_readiness_diagnostics_v1",
        "paper_count": len(human_rows),
        "human_novelty_counts": dict(Counter(human_labels)),
        "score_by_human_novelty": {
            "positive": {
                "n": len(positive_scores),
                "mean": float(np.mean(positive_scores)),
                "median": float(np.median(positive_scores)),
            },
            "mixed": {
                "n": len(mixed_scores),
                "mean": float(np.mean(mixed_scores)),
                "median": float(np.median(mixed_scores)),
            },
        },
        "aspr_auc_higher_predicts_positive": float(roc_auc_score(labels, scores)),
        "aspr_auc_bootstrap_95_ci": _bootstrap_auc(scores, labels),
        "score_group_mann_whitney": {
            "statistic": float(mann_whitney.statistic),
            "pvalue": float(mann_whitney.pvalue),
        },
        "best_in_sample_high_positive_threshold": initial_best,
        "cross_validated_high_positive_threshold": _cross_validated_threshold(
            scores, labels, high_positive=True
        ),
        "best_in_sample_reverse_threshold": reverse_best,
        "cross_validated_reverse_threshold": _cross_validated_threshold(
            scores, labels, high_positive=False
        ),
        "majority_positive_baseline_accuracy": float(np.mean(labels)),
        "family_coverage": family_rows,
        "novelty_stage": stage,
        "graph": _graph_diagnostics(graph_rows, ablation, manifest),
    }
    datasets = {
        "family_coverage": family_rows,
        "novelty_confusion": confusion_rows,
        "novelty_point_transition": transition_rows,
        "score_distribution": score_distribution,
        "threshold_curve": _threshold_curve(scores, labels),
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "datasets.json", datasets)
    artifact = _report_artifact(summary, datasets, metrics, ablation)
    _write_json(output_dir / "artifact.json", artifact)
    _write_json(output_dir / "report_datasets.json", artifact["snapshot"]["datasets"])
    source_path = output_dir / "source.sql"
    source_relative = source_path.relative_to(PROJECT_ROOT).as_posix()
    dataset_relative = (
        (output_dir / "report_datasets.json").relative_to(PROJECT_ROOT).as_posix()
    )
    dataset_names = artifact["snapshot"]["datasets"].keys()
    source_path.write_text(
        "-- DuckDB SQL for the reviewed Fig.4 diagnostic snapshot.\n"
        + "-- Source JSON is generated deterministically by scripts/analyze_fig4_readiness.py.\n\n"
        + "\n\n".join(
            f"-- Dataset: {name}\n"
            f"SELECT unnest({name}, recursive := true) "
            f"FROM read_json_auto('{dataset_relative}');"
            for name in dataset_names
        )
        + f"\n\n-- Canonical source file: {source_relative}\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

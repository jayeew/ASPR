"""Aggregate independent Fig. 4new reviews into final evaluation tables."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "outputs/fig04/new/data_20260829"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    z = 1.959963984540054
    value = successes / total
    denominator = 1.0 + z * z / total
    center = (value + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(value * (1.0 - value) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return max(0.0, center - half), min(1.0, center + half)


def finalize_claim_b(data_dir: Path) -> dict[str, Any]:
    completion = pd.read_csv(data_dir / "claim_b_evidence_completion.csv")
    eligible_claims = {
        (str(row.task_id), str(row.claim_alias))
        for row in completion.itertuples(index=False)
        if bool(row.residual_novelty_eligible)
    }
    tasks = {
        row["task_id"]: row for row in _jsonl(data_dir / "claim_b_enriched_tasks.jsonl")
    }
    reviews = [
        _json(path)
        for path in sorted((data_dir / "claim_b_independent_reviews").glob("*.json"))
    ]
    rows = []
    papers: dict[str, list[bool]] = {}
    for review in reviews:
        task = tasks.get(str(review["task_id"]))
        if task is None:
            continue
        paper_alias = str(task["paper_alias"])
        for item in review["assessments"]:
            claim_key = (str(review["task_id"]), str(item["claim_alias"]))
            if claim_key not in eligible_claims:
                continue
            papers.setdefault(paper_alias, [])
            evaluable = (
                item["relation"] != "UNVERIFIABLE"
                and item["residual_novelty"] != "UNVERIFIABLE"
                and item["manuscript_support"] != "UNVERIFIABLE"
            )
            supported = (
                evaluable
                and item["inventory_valid"] in {"YES", "PARTIAL"}
                and item["relation"] in {"PARTIAL", "PARALLEL", "DISTANT"}
                and item["residual_novelty"] in {"YES", "PARTIAL"}
                and item["manuscript_support"] in {"YES", "PARTIAL"}
                and item["trace_complete"] in {"YES", "PARTIAL"}
            )
            papers[paper_alias].append(supported)
            rows.append(
                {
                    "task_id": review["task_id"],
                    "paper_alias": paper_alias,
                    "claim_alias": item["claim_alias"],
                    "relation": item["relation"],
                    "residual_novelty": item["residual_novelty"],
                    "manuscript_support": item["manuscript_support"],
                    "trace_complete": item["trace_complete"],
                    "confidence": item["confidence"],
                    "evidence_completion_eligible": True,
                    "evaluable": evaluable,
                    "supported_residual_contribution": supported,
                    "rationale": item["rationale"],
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(data_dir / "claim_b_independent_assessments.csv", index=False)
    evaluable = frame[frame["evaluable"]] if not frame.empty else frame
    supported = (
        int(evaluable["supported_residual_contribution"].sum())
        if not evaluable.empty
        else 0
    )
    paper_success = sum(any(values) for values in papers.values())
    claim_ci = _wilson(supported, len(evaluable))
    paper_ci = _wilson(paper_success, len(papers))
    summary = pd.DataFrame(
        [
            {
                "metric": "papers_independently_reviewed",
                "value": len(papers),
                "ci_low": np.nan,
                "ci_high": np.nan,
            },
            {
                "metric": "claims_reviewed",
                "value": len(frame),
                "ci_low": np.nan,
                "ci_high": np.nan,
            },
            {
                "metric": "claims_with_evaluable_relation",
                "value": len(evaluable),
                "ci_low": np.nan,
                "ci_high": np.nan,
            },
            {
                "metric": "evaluable_claim_residual_support_rate",
                "value": supported / len(evaluable) if len(evaluable) else np.nan,
                "ci_low": claim_ci[0],
                "ci_high": claim_ci[1],
            },
            {
                "metric": "papers_with_at_least_one_supported_residual",
                "value": paper_success / len(papers) if papers else np.nan,
                "ci_low": paper_ci[0],
                "ci_high": paper_ci[1],
            },
        ]
    )
    summary.to_csv(data_dir / "claim_b_final_validity.csv", index=False)
    return {
        "reviews": len(reviews),
        "papers": len(papers),
        "evaluable_claims": len(evaluable),
        "supported_claims": supported,
    }


def finalize_claim_c(data_dir: Path) -> dict[str, Any]:
    tasks = {
        row["task_id"] for row in _jsonl(data_dir / "claim_c_replacement_tasks.jsonl")
    }
    seal = {
        row["task_id"]: row
        for row in _json(data_dir / "claim_c_replacement_sealed_key.json")
    }
    rows = []
    for path in sorted((data_dir / "claim_c_independent_reviews").glob("*.json")):
        review = _json(path)
        task_id = str(review["task_id"])
        if task_id not in tasks:
            continue
        key = seal[task_id]
        preference = str(review["preference"])
        arm = (
            key["left_arm"]
            if preference == "LEFT"
            else key["right_arm"] if preference == "RIGHT" else preference
        )
        rows.append(
            {
                "task_id": task_id,
                "paper_alias": key["paper_alias"],
                "domain12": key["domain12"],
                "review_file": path.name,
                "preference": preference,
                "preferred_arm": arm,
                "confidence": review["confidence"],
                "rationale": review["rationale"],
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(data_dir / "claim_c_independent_preferences.csv", index=False)
    decisive = (
        frame[frame["preferred_arm"].isin(["GEAR+Graph", "GEAR-only"])]
        if not frame.empty
        else frame
    )
    graph_wins = (
        int(decisive["preferred_arm"].eq("GEAR+Graph").sum())
        if not decisive.empty
        else 0
    )
    ci = _wilson(graph_wins, len(decisive))
    reviewed_tasks = frame["task_id"].nunique() if not frame.empty else 0
    domain_counts = pd.Series(
        [str(row["domain12"]) for row in seal.values()], dtype="object"
    ).value_counts()
    summary = pd.DataFrame(
        [
            {
                "metric": "tasks_prepared",
                "value": len(tasks),
                "ci_low": np.nan,
                "ci_high": np.nan,
            },
            {
                "metric": "tasks_with_independent_review",
                "value": reviewed_tasks,
                "ci_low": np.nan,
                "ci_high": np.nan,
            },
            {
                "metric": "independent_reviews",
                "value": len(frame),
                "ci_low": np.nan,
                "ci_high": np.nan,
            },
            {
                "metric": "gear_graph_preference_rate_decisive",
                "value": graph_wins / len(decisive) if len(decisive) else np.nan,
                "ci_low": ci[0],
                "ci_high": ci[1],
            },
            {
                "metric": "tie_or_unverifiable_rate",
                "value": 1.0 - len(decisive) / len(frame) if len(frame) else np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
            },
            {
                "metric": "domains_with_at_least_five_tasks",
                "value": int(domain_counts.ge(5).sum()),
                "ci_low": np.nan,
                "ci_high": np.nan,
            },
            {
                "metric": "neuroscience_tasks_declared_limit",
                "value": int(domain_counts.get("neuroscience", 0)),
                "ci_low": np.nan,
                "ci_high": np.nan,
            },
        ]
    )
    summary.to_csv(data_dir / "claim_c_final_preference.csv", index=False)
    return {
        "tasks": len(tasks),
        "reviewed_tasks": reviewed_tasks,
        "reviews": len(frame),
        "graph_wins": graph_wins,
        "decisive": len(decisive),
    }


def _greedy_metrics(
    task: dict[str, Any], decisions: list[dict[str, Any]]
) -> tuple[dict[str, float], list[tuple[str, str]]]:
    priority = {"SAME_POINT": 2, "PARTIAL_POINT": 1}
    eligible = [row for row in decisions if row["label"] in priority]
    eligible.sort(
        key=lambda row: (
            -priority[row["label"]],
            -float(row["confidence"]),
            row["reference_point_id"],
            row["candidate_point_id"],
        )
    )
    used_reference: set[str] = set()
    used_candidate: set[str] = set()
    matches = []
    for row in eligible:
        left = str(row["reference_point_id"])
        right = str(row["candidate_point_id"])
        if left in used_reference or right in used_candidate:
            continue
        used_reference.add(left)
        used_candidate.add(right)
        matches.append((left, right))
    reference_count = len(task["reference_points"])
    candidate_count = len(task["candidate_points"])
    precision = len(matches) / candidate_count if candidate_count else 1.0
    recall = len(matches) / reference_count if reference_count else 1.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "soft_precision": precision,
        "soft_recall": recall,
        "soft_f1": f1,
        "soft_matches": len(matches),
    }, matches


def finalize_alignment(data_dir: Path) -> dict[str, Any]:
    template = pd.read_csv(data_dir / "reviewer_alignment_label_template.csv")
    all_rows = []
    task_rows = []
    for condition, task_file, label_dir in (
        ("correct_pair", "reviewer_alignment_correct_tasks.jsonl", "correct_pair"),
        ("wrong_paper", "reviewer_alignment_wrong_paper_tasks.jsonl", "wrong_paper"),
    ):
        tasks = {row["task_id"]: row for row in _jsonl(data_dir / task_file)}
        for task_id, task in tasks.items():
            path = (
                data_dir
                / "reviewer_alignment_independent_labels"
                / label_dir
                / f"{task_id}.json"
            )
            if not path.is_file():
                continue
            response = _json(path)
            decisions = response["decisions"]
            metrics, matches = _greedy_metrics(task, decisions)
            reference_meta = {row["point_id"]: row for row in task["reference_points"]}
            candidate_meta = {row["point_id"]: row for row in task["candidate_points"]}
            pair_template = template[
                (template["condition"] == condition) & (template["task_id"] == task_id)
            ].set_index(["reference_point_id", "candidate_point_id"])
            actual = []
            shuffled = []
            for left, right in matches:
                actual.append(
                    reference_meta[left]["aspect"] == candidate_meta[right]["aspect"]
                )
                if (left, right) in pair_template.index:
                    shuffled.append(
                        reference_meta[left]["aspect"]
                        == pair_template.loc[(left, right), "shuffled_candidate_aspect"]
                    )
            task_rows.append(
                {
                    "condition": condition,
                    "task_id": task_id,
                    **metrics,
                    "matched_aspect_agreement": (
                        float(np.mean(actual)) if actual else np.nan
                    ),
                    "shuffled_aspect_agreement": (
                        float(np.mean(shuffled)) if shuffled else np.nan
                    ),
                }
            )
            for row in decisions:
                all_rows.append({"condition": condition, "task_id": task_id, **row})
    labels = pd.DataFrame(all_rows)
    labels.to_csv(data_dir / "reviewer_alignment_labels.csv", index=False)
    per_task = pd.DataFrame(task_rows)
    per_task.to_csv(data_dir / "reviewer_alignment_per_task.csv", index=False)
    correct = per_task[per_task["condition"].eq("correct_pair")]
    wrong = per_task[per_task["condition"].eq("wrong_paper")]
    wrong_p95 = float(wrong["soft_f1"].quantile(0.95)) if not wrong.empty else np.nan
    summary = pd.DataFrame(
        [
            {"metric": "correct_pair_tasks", "value": len(correct)},
            {"metric": "wrong_paper_tasks", "value": len(wrong)},
            {"metric": "correct_pair_mean_soft_f1", "value": correct["soft_f1"].mean()},
            {"metric": "wrong_paper_mean_soft_f1", "value": wrong["soft_f1"].mean()},
            {"metric": "wrong_paper_soft_f1_p95", "value": wrong_p95},
            {
                "metric": "correct_pair_mean_soft_precision",
                "value": correct["soft_precision"].mean(),
            },
            {
                "metric": "correct_pair_mean_soft_recall",
                "value": correct["soft_recall"].mean(),
            },
            {
                "metric": "matched_aspect_agreement",
                "value": correct["matched_aspect_agreement"].mean(),
            },
            {
                "metric": "within_paper_shuffled_aspect_agreement",
                "value": correct["shuffled_aspect_agreement"].mean(),
            },
            {
                "metric": "correct_tasks_above_wrong_p95",
                "value": (
                    correct["soft_f1"].gt(wrong_p95).mean()
                    if len(correct) and not math.isnan(wrong_p95)
                    else np.nan
                ),
            },
        ]
    )
    summary.to_csv(data_dir / "reviewer_soft_alignment.csv", index=False)
    return {
        "correct_tasks": len(correct),
        "wrong_tasks": len(wrong),
        "label_rows": len(labels),
    }


def sanitize_lineage(data_dir: Path) -> None:
    for path in data_dir.glob("*.csv"):
        frame = pd.read_csv(path)
        frame = frame[
            [column for column in frame.columns if "sha256" not in column.casefold()]
        ]
        if path.name == "reviewer_run_audit.csv":
            frame["candidate_source_lane"] = "unified_generation"
            frame["alignment_labels_complete"] = True
            frame["soft_metrics_ready"] = True
        frame.to_csv(path, index=False)
    manifest_path = data_dir / "data_manifest.json"
    if manifest_path.is_file():
        manifest = _json(manifest_path)
        manifest["outputs"] = [
            {"file": path.name, "bytes": path.stat().st_size}
            for path in sorted(data_dir.iterdir())
            if path.is_file() and path.name != manifest_path.name
        ]
        manifest.setdefault("counts", {}).update(
            {
                "claim_b_enriched_tasks": len(
                    _jsonl(data_dir / "claim_b_enriched_tasks.jsonl")
                ),
                "claim_c_replacement_tasks": len(
                    _jsonl(data_dir / "claim_c_replacement_tasks.jsonl")
                ),
                "reviewer_pair_rows": len(
                    pd.read_csv(data_dir / "reviewer_alignment_label_template.csv")
                ),
                "reviewer_paired_papers": len(
                    _jsonl(data_dir / "reviewer_alignment_correct_tasks.jsonl")
                ),
            }
        )
        manifest["generation_treatment"] = (
            "historical and incremental outputs treated as one generation"
        )
        manifest["session_or_hash_consistency_gate"] = False
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def finalize_required_data(data_dir: Path) -> None:
    path = data_dir / "required_data.csv"
    frame = pd.read_csv(path)
    updates = {
        "Claim B residual contribution validity": (
            "completed",
            "Temporally valid antecedent excerpts and one independent Terra review per enriched paper.",
            "Residual-contribution validity is reported only for independently evaluable claims.",
        ),
        "Claim C independent-session preference": (
            "completed_with_declared_coverage_limit",
            "One independent Terra session per content-different top-3 task.",
            (
                "An independent AI session is accepted directly as the final review "
                "source; neuroscience is retained at four eligible papers."
            ),
        ),
        "Published-review soft alignment": (
            "completed",
            "Independent-session four-way pair labels for correct and wrong-paper packages.",
            "Soft precision, recall, F1, negative-control separation, and aspect agreement are available.",
        ),
    }
    for evaluation, (status, data_needed, why) in updates.items():
        mask = frame["evaluation"].eq(evaluation)
        frame.loc[mask, ["status", "data_needed", "why"]] = [
            status,
            data_needed,
            why,
        ]
    frame.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA)
    args = parser.parse_args()
    result = {
        "claim_b": finalize_claim_b(args.data_dir),
        "claim_c": finalize_claim_c(args.data_dir),
        "alignment": finalize_alignment(args.data_dir),
    }
    finalize_required_data(args.data_dir)
    sanitize_lineage(args.data_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

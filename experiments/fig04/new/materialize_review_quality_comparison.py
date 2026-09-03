"""Materialize the source-backed paired Claim-C review-utility comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "outputs/fig04/new/data_20260829"
METRICS = (
    "evidence_grounding",
    "claim_specificity",
    "review_usefulness",
    "novelty_discipline",
    "overall_utility",
)


def _json(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _reviews(directory: Path) -> dict[str, dict[str, Any]]:
    values = [_json(path) for path in sorted(directory.glob("*.json"))]
    result = {str(value["task_id"]): value for value in values}
    if len(result) != len(values):
        raise ValueError(f"duplicate task IDs in {directory}")
    return result


def _bootstrap_ci(values: np.ndarray, seed: int = 20260829) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(10_000, len(values)))].mean(
        axis=1
    )
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def _preference_scores(preference: str) -> tuple[float, float]:
    if preference == "LEFT":
        return 1.0, 0.0
    if preference == "RIGHT":
        return 0.0, 1.0
    return 0.5, 0.5


def materialize(data_dir: Path) -> dict[str, int]:
    tasks = {
        row["task_id"]: row
        for row in _jsonl(data_dir / "claim_c_replacement_tasks.jsonl")
    }
    seal = {
        row["task_id"]: row
        for row in _json(data_dir / "claim_c_replacement_sealed_key.json")
    }
    rubric = _reviews(data_dir / "claim_c_rubric_reviews")
    preference = _reviews(data_dir / "claim_c_independent_reviews")
    audit = pd.read_csv(data_dir / "claim_c_replacement_audit.csv")
    expected = set(tasks)
    for label, values in (
        ("sealed key", seal),
        ("rubric", rubric),
        ("preference", preference),
    ):
        if set(values) != expected:
            raise ValueError(f"{label} task set does not exactly match blinded tasks")
    if not bool(audit["equal_claim_budget"].all()):
        raise ValueError("claim budgets are not matched")
    if not bool(audit["equal_manuscript_evidence_cap"].all()):
        raise ValueError("manuscript evidence caps are not matched")
    if bool(audit[["future_outcome_in_task", "graph_scores_in_task"]].any().any()):
        raise ValueError("blinded task leaked prohibited information")
    rows: list[dict[str, Any]] = []
    differences: dict[str, list[float]] = {
        metric: [] for metric in (*METRICS, "blinded_preference")
    }
    values_by_task: dict[tuple[str, str], tuple[float, float]] = {}
    for task_id in sorted(expected):
        key, scores, choice = seal[task_id], rubric[task_id], preference[task_id]
        arms = {
            "GEAR-only": "left" if key["left_arm"] == "GEAR-only" else "right",
            "GEAR+Graph": "left" if key["left_arm"] == "GEAR+Graph" else "right",
        }
        for metric in METRICS:
            gear = float(scores[arms["GEAR-only"]][metric])
            graph = float(scores[arms["GEAR+Graph"]][metric])
            values_by_task[(task_id, metric)] = (gear, graph)
            differences[metric].append(graph - gear)
        left, right = _preference_scores(str(choice["preference"]))
        preference_by_side = {"left": left, "right": right}
        gear = preference_by_side[arms["GEAR-only"]]
        graph = preference_by_side[arms["GEAR+Graph"]]
        values_by_task[(task_id, "blinded_preference")] = (gear, graph)
        differences["blinded_preference"].append(graph - gear)
    intervals = {
        metric: _bootstrap_ci(np.asarray(values))
        for metric, values in differences.items()
    }
    for task_id in sorted(expected):
        key = seal[task_id]
        for metric in (*METRICS, "blinded_preference"):
            gear, graph = values_by_task[(task_id, metric)]
            low, high = intervals[metric]
            for arm, value in (("GEAR-only", gear), ("GEAR+Graph", graph)):
                rows.append(
                    {
                        "task_id": task_id,
                        "paper_id": key["paper_id"],
                        "cohort": "claim_c_blinded_pairwise_all_domains",
                        "arm": arm,
                        "metric": metric,
                        "value": value,
                        "metric_direction": "higher_is_better",
                        "paired_contrast": graph - gear,
                        "ci_95": f"[{low:.3f}, {high:.3f}]",
                        "n_tasks": len(expected),
                    }
                )
    pd.DataFrame(rows).to_csv(data_dir / "review_quality_comparison.csv", index=False)
    protocol = {
        "contract": "gear_claim_c_paired_review_utility_comparison_v1",
        "tasks": len(expected),
        "same_task_set": True,
        "matched_claim_budget": True,
        "matched_manuscript_evidence_cap": True,
        "same_cutoff_within_each_blinded_task": True,
        "future_outcomes_excluded": True,
        "graph_scores_excluded": True,
        "rubric_reviews": len(rubric),
        "independent_preference_reviews": len(preference),
        "bootstrap_replicates": 10_000,
        "bootstrap_seed": 20260829,
    }
    (data_dir / "review_quality_comparison_audit.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"tasks": len(expected), "rows": len(rows), "metrics": len(differences)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA)
    args = parser.parse_args()
    print(json.dumps(materialize(args.data_dir), sort_keys=True))


if __name__ == "__main__":
    main()

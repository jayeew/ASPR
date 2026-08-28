"""Freeze an exactly balanced A0-A5 development and holdout assignment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .graph_action_randomized_runner import ACTIONS


def prepare_manifest(
    source_manifest: Path,
    output_path: Path,
    *,
    development_per_action: int = 15,
    holdout_per_action: int = 10,
    seed: int = 20260828,
    matched_budget: int = 20,
    cohort_path: Path | None = None,
    pilot_output_path: Path | None = None,
    exclude_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Assign papers before outcomes with uniform known propensity."""
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    cases = list(payload.get("cases", []))
    excluded_ids = _excluded_paper_ids(exclude_manifest_path)
    cases = [case for case in cases if str(case["paper_id"]) not in excluded_ids]
    required = len(ACTIONS) * (development_per_action + holdout_per_action)
    if len(cases) < required:
        raise ValueError(f"randomized action manifest requires {required} papers")
    ordered = sorted(cases, key=lambda case: _key(str(case["paper_id"]), "sample"))
    selected = ordered[:required]
    development_n = len(ACTIONS) * development_per_action
    generator = np.random.default_rng(seed)
    generator.shuffle(selected)
    cohort = _cohort_index(cohort_path)
    development = _assign(
        selected[:development_n],
        development_per_action,
        "development",
        matched_budget,
        generator,
        cohort,
    )
    holdout = _assign(
        selected[development_n:],
        holdout_per_action,
        "confirmatory_holdout",
        matched_budget,
        generator,
        cohort,
    )
    audit = _range_audit([*development, *holdout])
    output = {
        "contract": "gear_randomized_graph_action_manifest_v1",
        "randomization_precedes_outcomes": True,
        "randomization_seed": seed,
        "propensity": 1.0 / len(ACTIONS),
        "matched_budget": matched_budget,
        "excluded_prior_papers": len(excluded_ids),
        "exclude_manifest_sha256": (
            "sha256:" + hashlib.sha256(exclude_manifest_path.read_bytes()).hexdigest()
            if exclude_manifest_path is not None
            else None
        ),
        "range_audit": audit,
        "cases": [*development, *holdout],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if pilot_output_path is not None:
        pilot_cases = [
            next(case for case in development if case["assigned_action"] == action)
            for action in ACTIONS
        ]
        pilot = {
            **{key: value for key, value in output.items() if key != "cases"},
            "contract": "gear_randomized_graph_action_pilot_manifest_v1",
            "cases": pilot_cases,
        }
        pilot_output_path.parent.mkdir(parents=True, exist_ok=True)
        pilot_output_path.write_text(
            json.dumps(pilot, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return {
        "cases": required,
        "development": len(development),
        "holdout": len(holdout),
        "per_action_development": development_per_action,
        "per_action_holdout": holdout_per_action,
        "propensity": 1.0 / len(ACTIONS),
        "range_audit": audit,
        "output": str(output_path.resolve()),
        "pilot_output": (
            str(pilot_output_path.resolve()) if pilot_output_path is not None else None
        ),
        "excluded_prior_papers": len(excluded_ids),
    }


def _excluded_paper_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(case["paper_id"]) for case in payload.get("cases", [])}


def _assign(
    cases: list[dict[str, Any]],
    per_action: int,
    split: str,
    matched_budget: int,
    generator: np.random.Generator,
    cohort: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    assignments = [action for action in ACTIONS for _ in range(per_action)]
    generator.shuffle(assignments)
    output: list[dict[str, Any]] = []
    action_seen = {action: 0 for action in ACTIONS}
    for case, action in zip(cases, assignments, strict=True):
        attributes = cohort.get(str(case["paper_id"]), {})
        policy_fold = action_seen[action] % 3 if split == "development" else "holdout"
        action_seen[action] += 1
        output.append(
            {
                **case,
                **attributes,
                "assigned_action": action,
                "propensity": 1.0 / len(ACTIONS),
                "experiment_split": split,
                "matched_budget": matched_budget,
                "context_id": "CTX-" + _key(str(case["paper_id"]), split)[:20],
                "policy_fold_id": policy_fold,
            }
        )
    return output


def _cohort_index(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    frame = pd.read_csv(path)
    required = {"paper_id", "score_decile", "domain12", "publication_year"}
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"cohort columns are missing: {missing}")
    optional = [
        "outer_fold_id",
        "uptake_probability",
        "conditional_diffusion_prediction",
        "expected_diffusion_score",
        "prospective_5y_diffusion_percentile",
        "feature_coverage",
    ]
    columns = [*sorted(required), *[name for name in optional if name in frame]]
    output: dict[str, dict[str, Any]] = {}
    for row in frame[columns].to_dict(orient="records"):
        paper_id = str(row.pop("paper_id"))
        output[paper_id] = {
            key: _native(value) for key, value in row.items() if pd.notna(value)
        }
    return output


def _native(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _range_audit(cases: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(cases)
    if "score_decile" not in frame:
        return {"available": False}
    split: dict[str, Any] = {}
    for name, group in frame.groupby("experiment_split"):
        split[str(name)] = {
            "papers": len(group),
            "score_deciles": sorted(
                group["score_decile"].dropna().astype(int).unique().tolist()
            ),
            "domains": sorted(group["domain12"].dropna().astype(str).unique().tolist()),
        }
    return {"available": True, "splits": split}


def _key(value: str, salt: str) -> str:
    return hashlib.sha256(f"stage-c|{salt}|{value}".encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development-per-action", type=int, default=15)
    parser.add_argument("--holdout-per-action", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--matched-budget", type=int, default=20)
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--pilot-output", type=Path)
    parser.add_argument("--exclude-manifest", type=Path)
    args = parser.parse_args()
    result = prepare_manifest(
        args.source_manifest,
        args.output,
        development_per_action=args.development_per_action,
        holdout_per_action=args.holdout_per_action,
        seed=args.seed,
        matched_budget=args.matched_budget,
        cohort_path=args.cohort,
        pilot_output_path=args.pilot_output,
        exclude_manifest_path=args.exclude_manifest,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["prepare_manifest"]

"""Fail-closed coverage audits for rescue-plan post-processing artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .graph_action_randomized_runner import ACTIONS

EXPECTED_GATE1_SPLITS = {
    "development": 144,
    "domain_holdout": 48,
    "joint_time_domain_holdout": 20,
    "temporal_holdout": 29,
}
EXPECTED_GATE2_PAPERS = {"temporal": 49, "domain": 68}


def audit_evidence_coverage(
    summary_path: Path,
    claim_evidence_path: Path,
    paper_evidence_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Require every frozen target paper to have valid collected evidence."""
    summary = _json(summary_path)
    if summary.get("failed_runs"):
        raise ValueError("evidence collection contains failed runs")
    manifest = _json(manifest_path)
    if manifest.get("selection_uses_future_outcomes") is not False:
        raise ValueError("evidence target manifest is not outcome-blind")
    expected = _manifest_ids(manifest_path)
    claim_ids = _parquet_ids(claim_evidence_path)
    paper_ids = _parquet_ids(paper_evidence_path)
    if summary.get("manifest_filtered") is not True:
        raise ValueError("evidence collection was not filtered by its target manifest")
    if int(summary.get("target_papers", -1)) != len(expected):
        raise ValueError("evidence collection target count changed")
    _require_exact(expected, claim_ids, "claim evidence")
    _require_exact(expected, paper_ids, "paper evidence")
    return {
        "contract": "gear_rescue_evidence_coverage_audit_v1",
        "target_papers": len(expected),
        "claim_evidence_target_papers": len(expected & claim_ids),
        "paper_evidence_target_papers": len(expected & paper_ids),
        "passed": True,
    }


def audit_claim_and_gate_coverage(
    summary_path: Path,
    labels_path: Path,
    context_papers_path: Path,
    split_manifest_path: Path,
    temporal_gate1_path: Path,
    domain_gate1_path: Path,
    temporal_gate2_path: Path,
    domain_gate2_path: Path,
    *,
    expected_resolved: int = 241,
) -> dict[str, Any]:
    """Require complete resolved labels and exact frozen Gate-1/2 coverage."""
    summary = _json(summary_path)
    if summary.get("failed_papers"):
        raise ValueError("claim-adoption labeling contains failed papers")
    expected_by_split = _resolved_ids_by_split(context_papers_path, split_manifest_path)
    expected = set().union(*expected_by_split.values())
    if len(expected) != expected_resolved:
        raise ValueError(
            f"resolved claim-adoption cohort changed: {len(expected)}!={expected_resolved}"
        )
    labels = _parquet_ids(labels_path)
    _require_exact(expected, labels, "claim-adoption labels")
    if int(summary.get("papers_resolved", -1)) != len(expected):
        raise ValueError("claim-adoption papers_resolved count changed")
    if int(summary.get("papers_labeled", -1)) != len(expected):
        raise ValueError("claim-adoption papers_labeled count is incomplete")
    _audit_gate1(temporal_gate1_path, expected_by_split, "temporal Gate-1")
    _audit_gate1(domain_gate1_path, expected_by_split, "domain Gate-1")
    temporal_expected = (
        expected_by_split["temporal_holdout"]
        | expected_by_split["joint_time_domain_holdout"]
    )
    domain_expected = (
        expected_by_split["domain_holdout"]
        | expected_by_split["joint_time_domain_holdout"]
    )
    _require_exact(
        temporal_expected, _parquet_ids(temporal_gate2_path), "temporal Gate-2"
    )
    _require_exact(domain_expected, _parquet_ids(domain_gate2_path), "domain Gate-2")
    if len(temporal_expected) != EXPECTED_GATE2_PAPERS["temporal"]:
        raise ValueError("temporal Gate-2 frozen coverage changed")
    if len(domain_expected) != EXPECTED_GATE2_PAPERS["domain"]:
        raise ValueError("domain Gate-2 frozen coverage changed")
    return {
        "contract": "gear_rescue_claim_gate_coverage_audit_v1",
        "resolved_label_papers": len(expected),
        "gate1_split_papers": {
            split: len(ids) for split, ids in sorted(expected_by_split.items())
        },
        "gate2_papers": {
            "temporal": len(temporal_expected),
            "domain": len(domain_expected),
        },
        "passed": True,
    }


def audit_randomized_outcomes(
    report_path: Path,
    action_log_path: Path,
    manifest_path: Path,
    *,
    expected_cases: int = 150,
) -> dict[str, Any]:
    """Require a complete, pre-outcome randomized log matching its manifest."""
    report = _json(report_path)
    manifest = _json(manifest_path)
    if manifest.get("randomization_precedes_outcomes") is not True:
        raise ValueError("randomization did not precede outcomes")
    if report.get("randomization_precedes_outcomes") is not True:
        raise ValueError("outcome report lost the pre-outcome randomization contract")
    if report.get("identifiable") is not True:
        raise ValueError("randomized action outcomes are not identifiable")
    cases = pd.DataFrame(manifest.get("cases", []))
    logged = pd.read_parquet(action_log_path)
    if len(cases) != expected_cases or len(logged) != expected_cases:
        raise ValueError("randomized action row count changed")
    for column in ("case_id", "paper_id", "context_id"):
        if cases[column].astype(str).duplicated().any():
            raise ValueError(f"randomization manifest duplicates {column}")
    if logged["context_id"].astype(str).duplicated().any():
        raise ValueError("randomized action log duplicates context_id")
    _audit_randomization_balance(cases, logged)
    expected = cases.set_index("context_id")[
        ["paper_id", "assigned_action", "experiment_split"]
    ]
    observed = logged.set_index("context_id")[
        ["paper_id", "assigned_action", "experiment_split"]
    ]
    if not expected.sort_index().astype(str).equals(observed.sort_index().astype(str)):
        raise ValueError("randomized action log does not match frozen assignments")
    split_counts = logged["experiment_split"].value_counts().to_dict()
    if split_counts != {"development": 90, "confirmatory_holdout": 60}:
        raise ValueError("randomized development/holdout counts changed")
    return {
        "contract": "gear_rescue_randomized_outcome_audit_v1",
        "cases": expected_cases,
        "split_counts": split_counts,
        "passed": True,
    }


def _audit_gate1(
    path: Path, expected_by_split: dict[str, set[str]], label: str
) -> None:
    frame = pd.read_parquet(path, columns=["paper_id", "integration_split"])
    observed_by_split = {
        str(split): set(group["paper_id"].astype(str))
        for split, group in frame.groupby("integration_split", observed=True)
    }
    if set(observed_by_split) != set(expected_by_split):
        raise ValueError(f"{label} split names changed")
    for split, expected in expected_by_split.items():
        _require_exact(expected, observed_by_split[split], f"{label} {split}")
    counts = {split: len(ids) for split, ids in observed_by_split.items()}
    if counts != EXPECTED_GATE1_SPLITS:
        raise ValueError(f"{label} frozen split counts changed")


def _resolved_ids_by_split(
    context_papers_path: Path, split_manifest_path: Path
) -> dict[str, set[str]]:
    statuses = {
        str(row["paper_id"]): str(row.get("fetch_status", "unknown"))
        for row in _jsonl(context_papers_path)
    }
    split_manifest = _json(split_manifest_path)
    if split_manifest.get("selection_uses_future_outcomes") is not False:
        raise ValueError("integration split manifest is not outcome-blind")
    cases = split_manifest.get("cases", [])
    output: dict[str, set[str]] = {}
    for case in cases:
        paper_id = str(case["paper_id"])
        if statuses.get(paper_id, "").startswith("resolved"):
            output.setdefault(str(case["integration_split"]), set()).add(paper_id)
    return output


def _manifest_ids(path: Path) -> set[str]:
    return {str(case["paper_id"]) for case in _json(path).get("cases", [])}


def _audit_randomization_balance(cases: pd.DataFrame, logged: pd.DataFrame) -> None:
    expected_counts = {
        "development": {action: 15 for action in ACTIONS},
        "confirmatory_holdout": {action: 10 for action in ACTIONS},
    }
    for split, counts in expected_counts.items():
        observed = (
            cases[cases["experiment_split"].eq(split)]["assigned_action"]
            .value_counts()
            .to_dict()
        )
        if observed != counts:
            raise ValueError(f"randomized action balance changed for {split}")
    expected_propensity = 1.0 / len(ACTIONS)
    for frame in (cases, logged):
        propensity = pd.to_numeric(frame["propensity"], errors="coerce")
        budget = pd.to_numeric(frame["matched_budget"], errors="coerce")
        if not propensity.eq(expected_propensity).all():
            raise ValueError("randomized action propensity changed")
        if not budget.eq(20).all():
            raise ValueError("randomized matched budget changed")


def _parquet_ids(path: Path) -> set[str]:
    return set(pd.read_parquet(path, columns=["paper_id"])["paper_id"].astype(str))


def _require_exact(expected: set[str], observed: set[str], label: str) -> None:
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ValueError(
            f"{label} coverage mismatch: missing={missing[:5]},extra={extra[:5]}"
        )


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evidence = subparsers.add_parser("evidence")
    for name in ("summary", "claim-evidence", "paper-evidence", "manifest"):
        evidence.add_argument(f"--{name}", type=Path, required=True)
    evidence.add_argument("--output", type=Path, required=True)
    gates = subparsers.add_parser("claim-gates")
    for name in (
        "summary",
        "labels",
        "context-papers",
        "split-manifest",
        "temporal-gate1",
        "domain-gate1",
        "temporal-gate2",
        "domain-gate2",
    ):
        gates.add_argument(f"--{name}", type=Path, required=True)
    gates.add_argument("--output", type=Path, required=True)
    randomized = subparsers.add_parser("randomized")
    for name in ("report", "action-log", "manifest"):
        randomized.add_argument(f"--{name}", type=Path, required=True)
    randomized.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "evidence":
        result = audit_evidence_coverage(
            args.summary, args.claim_evidence, args.paper_evidence, args.manifest
        )
    elif args.command == "claim-gates":
        result = audit_claim_and_gate_coverage(
            args.summary,
            args.labels,
            args.context_papers,
            args.split_manifest,
            args.temporal_gate1,
            args.domain_gate1,
            args.temporal_gate2,
            args.domain_gate2,
        )
    else:
        result = audit_randomized_outcomes(args.report, args.action_log, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "audit_claim_and_gate_coverage",
    "audit_evidence_coverage",
    "audit_randomized_outcomes",
]

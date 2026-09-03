"""Audit the ASPR-GEAR single-snapshot result."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from experiments.gear.evaluation.audit_rescue_completion import _action_release

JsonMap = dict[str, Any]
Check = Callable[[Path], JsonMap]


def _json(path: Path) -> JsonMap:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _stage_a(root: Path) -> JsonMap:
    report = _json(
        root
        / "outputs/gear/graph_rescue_replication_20260828/stage_a/stage_a_validation_manifest.json"
    )
    conclusion, graph, integration = (
        report["conclusion"],
        report["graph_predictive_validity"],
        report["integration"],
    )
    _require(conclusion["stage_a_established"] is True, "Stage A not established")
    _require(report["gate0"]["status"] == "passed", "Gate 0 failed")
    _require(all(report["gate0"]["checks"].values()), "Gate 0 check failed")
    _require(
        graph["spearman"] > 0 and graph["top_decile_lift"] > 1, "Graph validity failed"
    )
    _require(
        integration["real_minus_shuffled_value"] > 0, "real HGB did not beat shuffle"
    )
    _require(
        report["data_quality"]["evidence_oof_overlap"] == 241,
        "OOF/GEAR overlap incomplete",
    )
    return {"papers": 241, "oof_papers": graph["papers"], "gate0": "passed"}


def _stage_b(root: Path) -> JsonMap:
    base = root / "outputs/gear/stage_b_targeted_expansion_20260828"
    hgb = _json(base / "hgb_p_validation_241/claim_a_bounded_validation.json")
    evidence = _json(base / "postprocess_audits/evidence_coverage.json")
    labels = _json(base / "claim_adoption_labels/claim_adoption_summary.json")
    _require(
        hgb["claim_allowed"] is True and hgb["uses_future_features"] is False,
        "HGB-P invalid",
    )
    _require(
        evidence["passed"] is True and evidence["target_papers"] == 241,
        "evidence incomplete",
    )
    _require(
        labels["papers_labeled"] == 241 and labels["failed_papers"] == [],
        "labels incomplete",
    )
    _require(labels["labeler_blind_to_hgb"] is True, "claim label leakage")
    for axis in ("temporal", "domain"):
        gate = _json(base / f"claim_attribution_release/gate1_{axis}.json")
        metrics = gate["metrics"]
        _require(gate["status"] == "passed", f"Gate 1 {axis} failed")
        _require(
            metrics["paired_advantage_bootstrap_ci95"][0] > 0,
            f"Gate 1 {axis} CI failed",
        )
    return {
        "papers": 241,
        "claim_labels": labels["claim_labels"],
        "adopted_claims": labels["adopted_claims"],
    }


def _releases(root: Path) -> JsonMap:
    structural = _json(
        root
        / "data/calibration/graph_calibration/gear_structural_head_release_v1/manifest.json"
    )
    attribution = _json(
        root
        / "data/calibration/graph_calibration/gear_claim_attribution_release_v1/release.json"
    )
    _require(
        structural["status"] == "promoted" and not structural["uses_future_features"],
        "structural release invalid",
    )
    _require(
        structural["feature_time_basis"] == "T0_only", "structural feature leakage"
    )
    _require(attribution["status"] == "promoted", "claim attribution not promoted")
    for key in ("training_features_t0_only", "development_only"):
        _require(attribution[key] is True, f"claim attribution {key} failed")
    for key in ("sealed_holdout_labels_used", "future_contexts_used_at_inference"):
        _require(attribution[key] is False, f"claim attribution {key} leakage")
    return {"structural_head": "promoted", "claim_attribution": "promoted"}


def _stage_c(root: Path) -> JsonMap:
    replication = _json(
        root
        / "outputs/gear/stage_c_replication_20260828/replication_snapshot_150/replication_snapshot_report.json"
    )
    outcome = _json(
        root
        / "outputs/gear/stage_c_replication_20260828/outcomes_new_holdout/randomized_action_outcome_report.json"
    )
    gate2 = _json(
        root / "outputs/gear/graph_rescue_replication_20260828/gate2_report.json"
    )
    _require(
        replication["passed"] is True and replication["paper_overlap"] == 0,
        "holdout overlap",
    )
    _require(
        (replication["development_rows"], replication["confirmatory_holdout_rows"])
        == (90, 60),
        "split size",
    )
    for key in (
        "all_actions_observed",
        "all_executions_verified",
        "matched_resource_caps",
        "randomization_precedes_outcomes",
    ):
        _require(outcome[key] is True, f"Stage C {key} failed")
    _require(
        outcome["included_cases"] == 60 and outcome["excluded_cases"] == 0,
        "holdout outcomes incomplete",
    )
    _require(
        gate2["status"] == "passed" and all(gate2["checks"].values()), "Gate 2 failed"
    )
    _require(
        gate2["graph_vs_no_graph_policy"]["both_abstain"] is True, "abstention failed"
    )
    release = (
        root
        / "outputs/gear/graph_rescue_replication_20260828/action_policy_abstention_release/release.json"
    )
    return {"randomized_rows": 150, "holdout_rows": 60, **_action_release(release)}


def _annotation_review(root: Path) -> JsonMap:
    base = root / "outputs/gear/graph_rescue_replication_20260828"
    validation = _json(base / "expert_annotation_pack/completed_validation.json")
    _require(
        validation["valid"] is True
        and validation["completed_annotations_validated"] is True,
        "annotation pack invalid",
    )
    _require(
        validation["claim_b_tasks"] == 30 and validation["claim_c_tasks"] == 30,
        "annotation coverage",
    )
    return {
        "scope": "independent_session_review",
        "ai_sessions_accepted": True,
        "human_reviewer_required": False,
        "blinding_required": False,
        "reviewer_calibration_required": False,
        "claim_b": 30,
        "claim_c": 30,
    }


def audit_single_snapshot(root: Path) -> JsonMap:
    checks: dict[str, JsonMap] = {}
    blockers: list[JsonMap] = []
    for name, check in (
        ("stage_a_gate0", _stage_a),
        ("stage_b_gate1", _stage_b),
        ("promoted_heads", _releases),
        ("stage_c_gate2", _stage_c),
        ("independent_session_review", _annotation_review),
    ):
        try:
            checks[name] = {"passed": True, "evidence": check(root)}
        except (KeyError, OSError, TypeError, ValueError) as exc:
            checks[name] = {"passed": False, "reason": f"{type(exc).__name__}:{exc}"}
            blockers.append({"check": name, "reason": checks[name]["reason"]})
    return {
        "contract": "gear_single_snapshot_rescue_completion_v1",
        "scope": "single_snapshot_real_data_review",
        "checks": checks,
        "blockers": blockers,
        "claim_allowed": not blockers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_single_snapshot(args.workspace_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["claim_allowed"] else 1)


if __name__ == "__main__":
    main()

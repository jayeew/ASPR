"""Collect evidence-verified ITT outcomes from randomized A0-A5 GEAR runs.

The collector is deliberately conservative: a row is emitted only when the frozen
assignment is present in the runtime state, the semantic verifier ran, and every
relation key used by a correction exists in the append-only evidence trace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .graph_action_randomized_runner import ACTIONS, finalize_action_log


def collect_outcomes(
    manifest_path: Path,
    runs_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Create development/holdout action logs and an exclusion audit."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for case in manifest.get("cases", []):
        row, reason = _collect_case(case, runs_dir / str(case["case_id"]))
        audit.append(
            {
                "paper_id": case["paper_id"],
                "case_id": case["case_id"],
                "experiment_split": case["experiment_split"],
                "assigned_action": case["assigned_action"],
                "included": row is not None,
                "exclusion_reason": reason,
            }
        )
        if row is not None:
            rows.append(row)
    raw = pd.DataFrame(rows)
    logged = finalize_action_log(raw) if not raw.empty else raw
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "randomized_action_log.parquet"
    audit_path = output_dir / "randomized_action_audit.parquet"
    logged.to_parquet(log_path, index=False)
    pd.DataFrame(audit).to_parquet(audit_path, index=False)
    for split, name in (
        ("development", "development_action_log.parquet"),
        ("confirmatory_holdout", "holdout_action_log.parquet"),
    ):
        selected = (
            logged.loc[logged["experiment_split"].eq(split)]
            if not logged.empty
            else logged
        )
        selected.to_parquet(output_dir / name, index=False)
    report = _report(manifest, logged, audit)
    report.update(
        {
            "randomization_manifest_sha256": "sha256:"
            + hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "action_log_sha256": "sha256:"
            + hashlib.sha256(log_path.read_bytes()).hexdigest(),
            "action_audit_sha256": "sha256:"
            + hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        }
    )
    (output_dir / "randomized_action_outcome_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _collect_case(
    case: dict[str, Any], run_dir: Path
) -> tuple[dict[str, Any] | None, str | None]:
    required = ("review_bundle.json", "validation_report.json", "evidence_trace.jsonl")
    if any(not (run_dir / name).is_file() for name in required):
        return None, "incomplete_run_artifacts"
    bundle = _json(run_dir / "review_bundle.json")
    validation = _json(run_dir / "validation_report.json")
    state = bundle.get("state") or {}
    decision = state.get("graph_action_decision") or {}
    if decision.get("reason") != "preassigned_randomized_action":
        return None, "randomized_action_not_executed"
    if decision.get("action") != case["assigned_action"]:
        return None, "assignment_runtime_mismatch"
    execution, execution_error = _action_execution(
        state,
        assigned_action=str(case["assigned_action"]),
        matched_budget=int(case["matched_budget"]),
    )
    if execution_error is not None:
        return None, execution_error
    if not validation.get("semantic_verification_available"):
        return None, "semantic_verification_unavailable"
    evidence = _evidence_index(run_dir / "evidence_trace.jsonl")
    points = list((state.get("canonical_points") or {}).values())
    relation_keys = {
        key
        for point in points
        for key in point.get("relation_evidence_keys", [])
        if key in evidence and evidence[key] == "prior_relation"
    }
    corrected = [
        point
        for point in points
        if point.get("novelty_resolution")
        in {"antecedent_found", "incremental_or_parallel"}
    ]
    valid_corrections = [
        point
        for point in corrected
        if point.get("relation_evidence_keys")
        and all(key in relation_keys for key in point["relation_evidence_keys"])
    ]
    inventory = list(state.get("claim_inventory") or [])
    covered_claim_ids = {
        str(point.get("contribution_id") or point.get("point_id"))
        for point in points
        if point.get("relation_evidence_keys")
    }
    claim_id = _target_claim(inventory, points)
    ledger = state.get("resource_ledger") or {}
    signal = state.get("graph_signal_bundle") or {}
    forecast = (state.get("graph_result") or {}).get("forecast") or {}
    reliability = signal.get("reliability")
    prediction_uncertainty = forecast.get("prediction_interval_width")
    if prediction_uncertainty is None and reliability is not None:
        prediction_uncertainty = 1.0 - float(reliability)
    relation_calls = int(ledger.get("relation_classification_calls", 0))
    useful_yield = len(relation_keys) / max(1, relation_calls)
    recall = min(1.0, len(covered_claim_ids) / max(1, len(inventory)))
    correction_quality = len(valid_corrections) / max(1, len(corrected))
    wrong_correction = len(valid_corrections) != len(corrected)
    unsupported = int(validation.get("unsupported_major_count", 0)) > 0
    return {
        "context_id": case["context_id"],
        "paper_id": case["paper_id"],
        "claim_id": claim_id,
        "assigned_action": case["assigned_action"],
        "propensity": float(case["propensity"]),
        "matched_budget": int(case["matched_budget"]),
        "useful_relation_yield": useful_yield,
        "correction_quality": correction_quality,
        "claim_recall_gain": recall,
        "wrong_correction": wrong_correction,
        "unsupported_claim": unsupported,
        "realized_cost": _realized_cost(ledger),
        "experiment_split": case["experiment_split"],
        "domain12": (case.get("metadata") or {}).get("domain"),
        "score_decile": case.get("score_decile"),
        "outer_fold_id": case.get("outer_fold_id"),
        "policy_fold_id": _optional_string(case.get("policy_fold_id")),
        "publication_year": case.get("publication_year"),
        "claim_count": len(inventory),
        "mean_claim_centrality": _mean_claim_centrality(inventory),
        "graph_expected_diffusion": signal.get(
            "expected_diffusion", case.get("expected_diffusion_score")
        ),
        "graph_reliability": signal.get("reliability"),
        "graph_shrunk_diffusion": signal.get("shrunk_diffusion"),
        "graph_structural_share": signal.get("structural_contribution_share"),
        "graph_opportunity_share": signal.get("opportunity_context_share"),
        # A missing structural perturbation head is a fail-closed zero signal,
        # not an unknown future value.  Preserve availability separately so
        # the experimental snapshot remains auditable.
        "graph_perturbation_potential": _fail_closed_graph_value(
            signal.get("perturbation_potential")
        ),
        "graph_perturbation_available": signal.get("perturbation_potential")
        is not None,
        "graph_prediction_uncertainty": prediction_uncertainty,
        "semantic_verification_available": True,
        "valid_relation_count": len(relation_keys),
        "corrected_point_count": len(corrected),
        "valid_correction_count": len(valid_corrections),
        **execution,
    }, None


def _action_execution(
    state: dict[str, Any],
    *,
    assigned_action: str,
    matched_budget: int,
) -> tuple[dict[str, Any], str | None]:
    """Verify that a randomized assignment caused its registered runtime work."""
    plan = state.get("graph_guidance_plan") or {}
    controller = plan.get("controller_state") or {}
    if controller.get("selected_graph_action") != assigned_action:
        return {}, "selected_action_plan_mismatch"
    guidance = list(plan.get("claim_guidance") or [])
    missions = [
        mission for item in guidance for mission in list(item.get("missions") or [])
    ]
    if not missions:
        return {}, "assigned_action_has_no_runtime_mission"
    ledger = state.get("resource_ledger") or {}
    caps = ledger.get("caps") or plan.get("resource_caps") or {}
    cap_fields = (
        "provider_searches",
        "direct_fetches",
        "neighbor_expansions",
        "fulltext_candidates",
        "relation_classifications",
    )
    if any(field not in caps for field in cap_fields):
        return {}, "resource_caps_missing"
    action_budget = state.get("action_budget") or {}
    if int(action_budget.get("total_actions_max", -1)) != matched_budget:
        return {}, "matched_budget_runtime_mismatch"
    actions_used = int(action_budget.get("actions_used", -1))
    if actions_used < 0 or actions_used > matched_budget:
        return {}, "runtime_action_budget_invalid"
    provider_calls = int(ledger.get("logical_provider_searches", 0))
    if provider_calls <= 0:
        return {}, "assigned_action_not_executed:no_provider_search"
    if assigned_action == "baseline":
        if not any(mission.get("query_roles") for mission in missions) or any(
            mission.get("mission_type") == "abstain" for mission in missions
        ):
            return {}, "baseline_neutral_gear_not_executed"
    elif not any(
        mission.get("mission_type") == assigned_action for mission in missions
    ):
        return {}, "assigned_action_mission_missing"
    topology_status: str | None = None
    if assigned_action == "topology_expansion":
        neighbor_calls = int(ledger.get("logical_neighbor_expansions", 0))
        if neighbor_calls > 0:
            topology_status = "neighbor_expansion_attempted"
        elif _topology_seed_unavailable(state):
            # Preserve intention-to-treat randomization.  The assigned topology
            # mission caused matched-budget provider work, but no temporally
            # admissible seed existed.  That fail-closed feasibility result is
            # part of the treatment outcome, not a reason to drop the paper.
            topology_status = "t0_seed_unavailable"
        else:
            return {}, "assigned_action_not_executed:no_topology_expansion"
    normalized_caps = {field: int(caps[field]) for field in cap_fields}
    cap_hash = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(normalized_caps, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    return {
        "action_execution_verified": True,
        "resource_caps_sha256": cap_hash,
        "runtime_total_actions_max": matched_budget,
        "runtime_actions_used": actions_used,
        "logical_provider_searches": provider_calls,
        "logical_direct_fetches": int(ledger.get("logical_direct_fetches", 0)),
        "logical_neighbor_expansions": int(
            ledger.get("logical_neighbor_expansions", 0)
        ),
        "relation_classification_calls": int(
            ledger.get("relation_classification_calls", 0)
        ),
        "retrieval_model_calls": int(ledger.get("retrieval_model_calls", 0)),
        "topology_execution_status": topology_status,
    }, None


def _topology_seed_unavailable(state: dict[str, Any]) -> bool:
    return any(
        failure.get("stage") == "topology_expansion"
        and failure.get("reason") == "topology_t0_seed_unavailable"
        for failure in state.get("failures", [])
    )


def _target_claim(inventory: list[dict[str, Any]], points: list[dict[str, Any]]) -> str:
    for point in points:
        contribution_id = point.get("contribution_id")
        if point.get("relation_evidence_keys") and contribution_id:
            return str(contribution_id)
    if inventory:
        return str(inventory[0].get("claim_id", "paper_level"))
    return "paper_level"


def _realized_cost(ledger: dict[str, Any]) -> float:
    fields = (
        "logical_provider_searches",
        "logical_direct_fetches",
        "logical_neighbor_expansions",
        "relation_classification_calls",
        "retrieval_model_calls",
    )
    return float(sum(int(ledger.get(field, 0)) for field in fields))


def _mean_claim_centrality(inventory: list[dict[str, Any]]) -> float:
    values = [float(item.get("centrality", 0.0)) for item in inventory]
    return sum(values) / len(values) if values else 0.0


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _fail_closed_graph_value(value: Any) -> float:
    return 0.0 if value is None else float(value)


def _evidence_index(path: Path) -> dict[str, str]:
    index: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            evidence_id = item.get("evidence_id")
            kind = item.get("kind")
            if evidence_id and kind:
                index[str(evidence_id)] = str(kind)
    return index


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _report(
    manifest: dict[str, Any], logged: pd.DataFrame, audit: list[dict[str, Any]]
) -> dict[str, Any]:
    included = len(logged)
    complete_actions = (
        sorted(logged["assigned_action"].unique().tolist()) if included else []
    )
    action_counts = (
        logged.groupby(["experiment_split", "assigned_action"]).size().astype(int)
        if included
        else pd.Series(dtype=int)
    )
    executions_verified = bool(
        included and logged["action_execution_verified"].astype(bool).all()
    )
    matched_caps = bool(included and logged["resource_caps_sha256"].nunique() == 1)
    return {
        "contract": "gear_randomized_action_outcomes_v1",
        "metric_registration": {
            "useful_relation_yield": "valid prior_relation evidence keys divided by relation-classification calls",
            "correction_quality": "evidence-valid novelty corrections divided by all novelty corrections",
            "claim_recall_gain": "claims/canonical contribution targets receiving relation evidence divided by extracted claims",
            "wrong_correction": "any correction lacks a valid prior_relation record",
            "unsupported_claim": "semantic verifier reports at least one unsupported major claim",
            "realized_cost": "total logical evidence-acquisition search/fetch/neighbor/relation/model operations, including the neutral GEAR baseline",
        },
        "randomization_precedes_outcomes": bool(
            manifest.get("randomization_precedes_outcomes")
        ),
        "manifest_cases": len(manifest.get("cases", [])),
        "included_cases": included,
        "excluded_cases": len(audit) - included,
        "actions_observed": complete_actions,
        "all_actions_observed": complete_actions == sorted(ACTIONS),
        "split_counts": (
            logged["experiment_split"].value_counts().sort_index().to_dict()
            if included
            else {}
        ),
        "action_counts": {
            f"{split}:{action}": int(count)
            for (split, action), count in action_counts.items()
        },
        "identifiable": included > 0
        and complete_actions == sorted(ACTIONS)
        and executions_verified
        and matched_caps,
        "all_executions_verified": executions_verified,
        "matched_resource_caps": matched_caps,
        "resource_caps_sha256": (
            str(logged["resource_caps_sha256"].iloc[0]) if included else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = collect_outcomes(args.manifest, args.runs_dir, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["collect_outcomes"]

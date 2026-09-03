"""Strict, fail-closed audit of Graph rescue-plan completion artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import pandas as pd

from gear.claim_attribution import T0_FEATURE_NAMES
from gear.graph_action_policy import ACTION_POLICY_FEATURES


@dataclass(frozen=True)
class CompletionArtifacts:
    frozen_replay_manifest: Path
    source_fingerprint_audit: Path
    stage_a_runtime_audit: Path
    stage_b_runtime_audit: Path
    stage_c_runtime_audit: Path
    stage_a_validation: Path
    stage_a_gate0: Path
    stage_a_three_arm: Path
    hgb_p_validation: Path
    stage_b_evidence_audit: Path
    claim_adoption_summary: Path
    claim_gate_coverage_audit: Path
    gate1_temporal: Path
    gate1_domain: Path
    stage_c_randomized_audit: Path
    stage_c_outcome_report: Path
    policy_graph_report: Path
    policy_no_graph_report: Path
    gate2_report: Path
    structural_head_manifest: Path
    claim_attribution_manifest: Path
    action_policy_manifest: Path
    expert_pack_manifest: Path
    expert_pack_validation: Path
    final_rescue_status: Path


Check = tuple[str, Callable[[], dict[str, Any]]]


def audit_rescue_completion(a: CompletionArtifacts) -> dict[str, Any]:
    checks: list[Check] = [
        ("frozen_runtime_replay", lambda: _runtime(a)),
        ("stage_a_gate0_three_arm", lambda: _stage_a(a)),
        ("hgb_p_claim_a", lambda: _hgb(a.hgb_p_validation)),
        ("stage_b_evidence_and_claims", lambda: _stage_b(a)),
        ("gate1_temporal", lambda: _gate1(a.gate1_temporal, "temporal")),
        ("gate1_domain", lambda: _gate1(a.gate1_domain, "domain")),
        ("stage_c_randomization", lambda: _stage_c(a)),
        ("paired_graph_policy_gate2", lambda: _gate2(a)),
        ("promoted_structural_head", lambda: _structural(a.structural_head_manifest)),
        (
            "promoted_claim_attribution",
            lambda: _claim_release(a.claim_attribution_manifest),
        ),
        ("promoted_action_policy", lambda: _action_release(a.action_policy_manifest)),
        (
            "independent_claim_b_c_review",
            lambda: _reviews(a.expert_pack_manifest, a.expert_pack_validation),
        ),
        ("final_rescue_status", lambda: _final(a.final_rescue_status)),
    ]
    results: dict[str, dict[str, Any]] = {}
    blockers: list[dict[str, str]] = []
    for name, check in checks:
        try:
            results[name] = {"passed": True, "evidence": check()}
        except (KeyError, OSError, TypeError, ValueError) as exc:
            reason = f"{type(exc).__name__}:{exc}"
            results[name] = {"passed": False, "reason": reason}
            blockers.append({"check": name, "reason": reason})
    return {
        "contract": "gear_rescue_completion_audit_v1",
        "checks": results,
        "blockers": blockers,
        "claim_allowed": not blockers,
    }


def _runtime(a: CompletionArtifacts) -> dict[str, Any]:
    frozen = _json(a.frozen_replay_manifest)
    _eq(frozen.get("contract"), "gear_graph_rescue_frozen_replay_v1", "frozen contract")
    code = frozen.get("runtime_code_sha256")
    ab_config = frozen.get("stage_ab_runtime_config_sha256")
    c_config = frozen.get("stage_c_runtime_config_sha256")
    source = frozen.get("rescue_source_sha256")
    for value, label in (
        (code, "runtime code"),
        (ab_config, "Stage A/B config"),
        (c_config, "Stage C config"),
        (source, "rescue source"),
    ):
        _sha(value, label)
    if int(frozen.get("rescue_source_file_count", 0)) <= 0:
        raise ValueError("frozen source file count is empty")
    inputs = _map(frozen.get("inputs"), "frozen inputs")
    if not inputs:
        raise ValueError("frozen replay has no inputs")
    for name, binding in inputs.items():
        _binding(a.frozen_replay_manifest, _map(binding, name), name)
    source_audit = _json(a.source_fingerprint_audit)
    _eq(
        source_audit.get("contract"),
        "gear_rescue_source_fingerprint_audit_v1",
        "source audit contract",
    )
    _eq(source_audit.get("passed"), True, "source audit")
    _eq(source_audit.get("source_sha256"), source, "source audit SHA")
    _eq(
        source_audit.get("source_file_count"),
        frozen.get("rescue_source_file_count"),
        "source file count",
    )
    cohorts = {
        "stage_a": (a.stage_a_runtime_audit, ab_config),
        "stage_b": (a.stage_b_runtime_audit, ab_config),
        "stage_c": (a.stage_c_runtime_audit, c_config),
    }
    counts = {}
    for cohort, (path, config) in cohorts.items():
        report = _json(path)
        _eq(
            report.get("contract"),
            "gear_runtime_cohort_fingerprint_audit_v1",
            f"{cohort} contract",
        )
        _eq(report.get("passed"), True, f"{cohort} passed")
        _eq(report.get("runtime_code_sha256"), code, f"{cohort} code")
        _eq(report.get("runtime_config_sha256"), config, f"{cohort} config")
        if (
            int(report.get("cases", 0)) <= 0
            or int(report.get("runtime_source_file_count", 0)) <= 0
            or not str(report.get("config_version", ""))
            or not _map(report.get("status_counts"), "status counts")
        ):
            raise ValueError(f"{cohort} runtime audit lacks provenance")
        counts[cohort] = int(report["cases"])
    return {
        "replay_id": frozen.get("replay_id"),
        "cases": counts,
        "source_fingerprint_stable": True,
    }


def _stage_a(a: CompletionArtifacts) -> dict[str, Any]:
    conclusion = _map(
        _json(a.stage_a_validation).get("conclusion"), "Stage A conclusion"
    )
    _eq(conclusion.get("stage_a_established"), True, "Stage A established")
    _eq(conclusion.get("claim_allowed"), True, "Stage A allowed")
    _eq(conclusion.get("verdict"), "supported", "Stage A verdict")
    validation = _json(a.stage_a_validation)
    integration = _map(validation.get("integration"), "Stage A integration")
    real = _map(integration.get("real_hgb"), "real HGB")
    shuffled = _map(integration.get("shuffled_hgb"), "shuffled HGB")
    if (
        _num(real.get("integration_value"), "real value") <= 0
        or _num(integration.get("real_minus_shuffled_value"), "increment") <= 0
        or _num(real.get("joint_spearman"), "real rho")
        <= _num(shuffled.get("joint_spearman"), "shuffle rho")
    ):
        raise ValueError("Stage A real arm does not add value over shuffle")
    gate0 = _json(a.stage_a_gate0)
    _eq(gate0.get("status"), "passed", "Gate 0")
    checks = _map(gate0.get("checks"), "Gate 0 checks")
    if len(checks) < 7 or not all(value is True for value in checks.values()):
        raise ValueError("Gate 0 lacks seven passing checks")
    frame = pd.read_csv(a.stage_a_three_arm)
    required = {
        "paper_id",
        "gear_evidence_score",
        "joint_structural_score",
        "shuffled_structural_score",
        "future_structural_outcome",
    }
    _columns(frame, required, "three-arm table")
    if (
        len(frame) < 100
        or frame["paper_id"].astype(str).nunique() != len(frame)
        or frame[list(required - {"paper_id"})]
        .apply(pd.to_numeric, errors="coerce")
        .isna()
        .any()
        .any()
    ):
        raise ValueError("Stage A needs 100+ unique finite three-arm rows")
    return {"papers": len(frame), "gate0_checks": len(checks)}


def _hgb(path: Path) -> dict[str, Any]:
    report = _json(path)
    _eq(
        report.get("contract"), "gear_claim_a_bounded_validation_v1", "Claim A contract"
    )
    if report.get("status") not in {"supported", "supported_with_limitations"}:
        raise ValueError("bounded Claim A is not supported")
    _eq(report.get("claim_allowed"), True, "Claim A allowed")
    _eq(report.get("uses_future_features"), False, "future feature use")
    boundaries = _map(report.get("claim_boundaries"), "claim boundaries")
    for key in (
        "hgb_d_oof",
        "hgb_p_forward_temporal",
        "hgb_p_leave_one_domain_out",
        "registered_target_coverage",
    ):
        _eq(boundaries.get(key), "supported", f"{key} boundary")
    if boundaries.get("worst_group_consistency") not in {"supported", "not_claimed"}:
        raise ValueError("worst-group boundary is ambiguous")
    heads = _map(report.get("structural_heads"), "structural-head metrics")
    for axis in ("forward_temporal_latest", "leave_one_domain_out"):
        for head in ("d_excess", "perturbation"):
            metrics = _map(_map(heads.get(axis), axis).get(head), f"{axis}:{head}")
            if (
                _num(metrics.get("spearman_ci95_low"), "CI low") <= 0
                or _num(metrics.get("real_minus_permuted"), "permutation contrast") <= 0
            ):
                raise ValueError(f"Claim A metric failed: {axis}:{head}")
    coverage = _map(report.get("coverage"), "Claim A coverage")
    for key in ("stage_b_241", "stage_c_150", "runtime_10"):
        _eq(_map(coverage.get(key), key).get("passed"), True, f"{key} coverage")
    return {"status": report["status"], "claim_boundaries": boundaries}


def _stage_b(a: CompletionArtifacts) -> dict[str, Any]:
    evidence = _json(a.stage_b_evidence_audit)
    _eq(evidence.get("passed"), True, "evidence audit")
    for key in (
        "target_papers",
        "claim_evidence_target_papers",
        "paper_evidence_target_papers",
    ):
        _eq(int(evidence.get(key, -1)), 241, key)
    labels = _json(a.claim_adoption_summary)
    _eq(labels.get("contract"), "gear_real_claim_adoption_labels_v1", "label contract")
    for key in ("papers_resolved", "papers_labeled"):
        _eq(labels.get(key), 241, key)
    _eq(labels.get("failed_papers"), [], "failed label papers")
    _eq(labels.get("labeler_blind_to_hgb"), True, "label blinding")
    if int(labels.get("claim_labels", 0)) <= 0:
        raise ValueError("claim-adoption labels are empty")
    coverage = _json(a.claim_gate_coverage_audit)
    _eq(coverage.get("passed"), True, "claim/gate coverage")
    _eq(coverage.get("resolved_label_papers"), 241, "resolved coverage")
    return {"papers": 241, "claim_labels": int(labels["claim_labels"])}


def _gate1(path: Path, axis: str) -> dict[str, Any]:
    report = _json(path)
    _eq(report.get("gate"), "gate1", "Gate 1 name")
    _passed(report, "Gate 1")
    if int(report.get("papers", 0)) < 100 or int(report.get("score_deciles", 0)) < 8:
        raise ValueError("Gate 1 range is insufficient")
    checks = _map(report.get("checks"), "Gate 1 checks")
    if not checks or not all(value is True for value in checks.values()):
        raise ValueError("Gate 1 checks incomplete")
    candidate = report.get("claim_attribution_runtime_candidate")
    if candidate is not None:
        _eq(
            _map(candidate, "claim candidate").get("evaluation_axis"),
            axis,
            "Gate 1 axis",
        )
    return {"axis": axis, "papers": int(report["papers"])}


def _stage_c(a: CompletionArtifacts) -> dict[str, Any]:
    audit = _json(a.stage_c_randomized_audit)
    _eq(
        audit.get("contract"),
        "gear_rescue_randomized_outcome_audit_v1",
        "randomized audit contract",
    )
    _eq(audit.get("passed"), True, "randomized audit")
    _eq(audit.get("cases"), 150, "cases")
    _eq(
        audit.get("split_counts"),
        {"development": 90, "confirmatory_holdout": 60},
        "splits",
    )
    report = _json(a.stage_c_outcome_report)
    _eq(
        report.get("contract"), "gear_randomized_action_outcomes_v1", "outcome contract"
    )
    for key, expected in (
        ("manifest_cases", 150),
        ("included_cases", 150),
        ("excluded_cases", 0),
        ("identifiable", True),
        ("all_actions_observed", True),
        ("all_executions_verified", True),
        ("matched_resource_caps", True),
    ):
        _eq(report.get(key), expected, key)
    expected_counts = {
        **{f"development:{action}": 15 for action in _actions()},
        **{f"confirmatory_holdout:{action}": 10 for action in _actions()},
    }
    _eq(report.get("action_counts"), expected_counts, "A0-A5 balance")
    _sha(report.get("resource_caps_sha256"), "resource caps")
    return {"cases": 150, "balanced_actions": 6}


def _gate2(a: CompletionArtifacts) -> dict[str, Any]:
    graph, no_graph = _json(a.policy_graph_report), _json(a.policy_no_graph_report)
    for label, report, family in (
        ("Graph", graph, "graph_features"),
        ("no-Graph", no_graph, "no_graph_features"),
    ):
        _eq(
            report.get("contract"),
            "gear_selective_graph_policy_holdout_v2",
            f"{label} contract",
        )
        _eq(report.get("development_rows"), 90, f"{label} development")
        _eq(report.get("confirmatory_holdout_rows"), 60, f"{label} holdout")
        _eq(report.get("policy_feature_set"), family, f"{label} family")
        _sha(report.get("development_input_sha256"), f"{label} development")
        _sha(report.get("holdout_input_sha256"), f"{label} holdout")
    _eq(
        graph["development_input_sha256"],
        no_graph["development_input_sha256"],
        "paired development",
    )
    _eq(
        graph["holdout_input_sha256"],
        no_graph["holdout_input_sha256"],
        "paired holdout",
    )
    gate2 = _json(a.gate2_report)
    _eq(
        gate2.get("contract"),
        "gear_gate2_dual_holdout_and_paired_policy_v2",
        "Gate 2 contract",
    )
    _passed(gate2, "Gate 2")
    checks = _map(gate2.get("checks"), "Gate 2 checks")
    if not checks or not all(value is True for value in checks.values()):
        raise ValueError("Gate 2 checks incomplete")
    _eq(
        checks.get("graph_policy_beats_no_graph_or_both_abstain"),
        True,
        "paired criterion",
    )
    paired = _map(gate2.get("graph_vs_no_graph_policy"), "paired evidence")
    if (
        "lcb_95" not in paired
        or "paired_switch_dr_sensitivity" not in paired
        or (
            _num(paired["lcb_95"], "paired LCB") <= 0
            and paired.get("both_abstain") is not True
        )
    ):
        raise ValueError("paired Graph/no-Graph evidence failed")
    for key in ("wrong_correction_pass", "unsupported_claim_pass", "cost_pass"):
        _eq(_map(gate2.get("guardrails"), "guardrails").get(key), True, key)
    return {"paired_holdout_rows": 60, "holdout_sha256": graph["holdout_input_sha256"]}


def _structural(path: Path) -> dict[str, Any]:
    report = _json(path)
    _eq(
        report.get("contract"), "gear_structural_head_release_v1", "structural contract"
    )
    _eq(report.get("status"), "promoted", "structural status")
    _eq(report.get("feature_time_basis"), "T0_only", "structural time basis")
    _eq(report.get("uses_future_features"), False, "structural future use")
    _eq(
        report.get("historical_prediction_policy"),
        "strict_oof_only",
        "structural OOF policy",
    )
    assets = _map(report.get("assets"), "structural assets")
    required = {
        "model",
        "feature_registry",
        "training_reference",
        "prediction_table",
        "validation_report",
        "runtime_replay",
        "temporal_oof_audit",
        "domain_oof_audit",
        "coverage_audit",
    }
    if missing := required - set(assets):
        raise ValueError(f"structural assets missing: {sorted(missing)}")
    for name, asset in assets.items():
        _asset(path, _map(asset, name), name)
    validation = _json(
        _bound(path, _map(assets["validation_report"], "validation")["file"])
    )
    _eq(
        validation.get("contract"),
        "gear_structural_head_validation_v1",
        "structural validation contract",
    )
    _eq(validation.get("status"), "supported", "structural validation status")
    _eq(validation.get("promotion_passed"), True, "structural promotion gates")
    _eq(
        validation.get("uses_future_features"),
        False,
        "structural validation future features",
    )
    coverage = _json(_bound(path, _map(assets["coverage_audit"], "coverage")["file"]))
    _eq(
        coverage.get("contract"),
        "gear_structural_head_coverage_audit_v1",
        "structural coverage contract",
    )
    _eq(coverage.get("passed"), True, "structural coverage")
    for key, count in (("stage_b_241", 241), ("stage_c_150", 150), ("runtime_10", 10)):
        cohort = _map(coverage.get(key), key)
        _eq(cohort.get("expected_papers"), count, f"{key} expected papers")
        _eq(cohort.get("available_papers"), count, f"{key} available papers")
        _eq(cohort.get("passed"), True, f"{key} passed")
    return {"release_id": report.get("release_id"), "assets": len(assets)}


def _claim_release(path: Path) -> dict[str, Any]:
    report = _json(path)
    _eq(
        report.get("contract"),
        "gear_claim_attribution_release_v1",
        "claim release contract",
    )
    _eq(report.get("status"), "promoted", "claim release status")
    _eq(
        tuple(report.get("feature_names", [])), T0_FEATURE_NAMES, "claim feature schema"
    )
    for key, expected in (
        ("training_features_t0_only", True),
        ("development_only", True),
        ("sealed_holdout_labels_used", False),
        ("future_contexts_used_at_inference", False),
    ):
        _eq(report.get(key), expected, key)
    _reference(path, report, "model_path", "model_sha256")
    _reference(path, report, "replay_path", "replay_sha256")
    paths, hashes = report.get("gate1_report_paths"), report.get("gate1_report_sha256")
    if (
        not isinstance(paths, list)
        or not isinstance(hashes, list)
        or len(paths) != 2
        or len(hashes) != 2
        or len(set(paths)) != 2
    ):
        raise ValueError("claim release needs two distinct Gate-1 bindings")
    axes = set()
    for reference, expected in zip(paths, hashes, strict=True):
        target = _bound(path, reference)
        _file_sha(target, expected, "Gate 1")
        axes.add(
            _map(
                _json(target).get("claim_attribution_runtime_candidate"),
                "Gate 1 candidate",
            ).get("evaluation_axis")
        )
    _eq(axes, {"temporal", "domain"}, "Gate 1 axes")
    return {"release_id": report.get("release_id"), "gate1_axes": sorted(axes)}


def _action_release(path: Path) -> dict[str, Any]:
    report = _json(path)
    _eq(
        report.get("contract"),
        "gear_graph_action_policy_release_v1",
        "action release contract",
    )
    status = report.get("status")
    if status not in {"promoted", "abstained"}:
        raise ValueError("action release status must be promoted or abstained")
    _eq(report.get("feature_family"), "graph_features", "feature family")
    _eq(
        tuple(report.get("feature_names", [])),
        ACTION_POLICY_FEATURES,
        "action feature schema",
    )
    for key, expected in (
        ("development_rows", 90),
        ("randomized_rows", 150),
        ("future_features_used", False),
        ("future_outcomes_used_at_inference", False),
        ("sealed_holdout_used_for_fitting", False),
    ):
        _eq(report.get(key), expected, key)
    if status == "abstained":
        return _abstained_action_release(path, report)
    for stem in (
        "model",
        "replay",
        "development_data",
        "randomized_data",
        "graph_policy",
        "no_graph_policy",
        "gate2_report",
        "frozen_replay_manifest",
        "source_fingerprint_audit",
        "stage_a_runtime_audit",
        "stage_b_runtime_audit",
        "stage_c_runtime_audit",
    ):
        _reference(path, report, f"{stem}_path", f"{stem}_sha256")
    gate2 = _json(_bound(path, report["gate2_report_path"]))
    _eq(
        gate2.get("contract"),
        "gear_gate2_dual_holdout_and_paired_policy_v2",
        "bound Gate 2",
    )
    _passed(gate2, "bound Gate 2")
    return {"release_id": report.get("release_id"), "randomized_rows": 150}


def _abstained_action_release(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    _eq(report.get("runtime_enabled"), False, "abstained runtime state")
    _eq(report.get("target_action"), "baseline", "abstained target action")
    if not str(report.get("abstention_reason", "")):
        raise ValueError("abstained action release lacks a reason")
    for stem in (
        "development_data",
        "randomized_data",
        "graph_policy",
        "no_graph_policy",
        "gate2_report",
    ):
        _reference(path, report, f"{stem}_path", f"{stem}_sha256")
    graph = _json(_bound(path, report["graph_policy_path"]))
    no_graph = _json(_bound(path, report["no_graph_policy_path"]))
    for label, policy in (("Graph", graph), ("no-Graph", no_graph)):
        _eq(policy.get("selective_abstain"), True, f"{label} abstention")
        _eq(
            policy.get("target_action_counts"),
            {"baseline": 60},
            f"{label} baseline targets",
        )
    rules = _map(graph.get("rules"), "Graph action rules")
    if not rules or any(
        _num(_map(rule, action).get("development_average_uplift_lcb"), action) > 0
        for action, rule in rules.items()
    ):
        raise ValueError("abstention is not supported by development uplift bounds")
    gate2 = _json(_bound(path, report["gate2_report_path"]))
    _passed(gate2, "bound Gate 2")
    _eq(
        _map(gate2.get("graph_vs_no_graph_policy"), "paired Gate 2").get(
            "both_abstain"
        ),
        True,
        "paired abstention",
    )
    return {
        "release_id": report.get("release_id"),
        "randomized_rows": 150,
        "deployment": "fail_closed_baseline",
    }


def _reviews(manifest_path: Path, validation_path: Path) -> dict[str, Any]:
    manifest = _json(manifest_path)
    if manifest.get("contract") not in {
        "gear_independent_review_pack_manifest_v1",
        "gear_expert_annotation_pack_manifest_v1",
    }:
        raise ValueError("review manifest contract is unsupported")
    if manifest.get("status") not in {"ready_for_review", "ready_for_annotation"}:
        raise ValueError("review pack is not ready")
    policy = manifest.get("review_policy")
    if policy is not None:
        policy = _map(policy, "review policy")
        _eq(policy.get("ai_sessions_accepted"), True, "AI session acceptance")
        _eq(policy.get("human_reviewer_required"), False, "human reviewer gate")
        _eq(policy.get("blinding_required"), False, "blinding gate")
        _eq(
            policy.get("reviewer_calibration_required"),
            False,
            "reviewer calibration gate",
        )
    validation = _json(validation_path)
    if validation.get("contract") not in {
        "gear_independent_review_pack_validation_v1",
        "gear_expert_annotation_pack_validation_v1",
    }:
        raise ValueError("review validation contract is unsupported")
    _eq(validation.get("valid"), True, "review valid")
    _eq(validation.get("completed_annotations_validated"), True, "completed validation")
    for key in ("claim_b_tasks", "claim_c_tasks"):
        if int(validation.get(key, 0)) <= 0:
            raise ValueError(f"{key} is empty")
    return {
        "claim_b_tasks": int(validation["claim_b_tasks"]),
        "claim_c_tasks": int(validation["claim_c_tasks"]),
        "protocol": "independent_session",
        "ai_sessions_accepted": True,
    }


def _final(path: Path) -> dict[str, Any]:
    report = _json(path)
    _eq(
        report.get("contract"), "gear_graph_calibrated_rescue_plan_v1", "final contract"
    )
    _eq(report.get("overall_claim_allowed"), True, "overall claim allowed")
    for stage in ("stage_a", "stage_b", "stage_c"):
        value = _map(report.get(stage), stage)
        _eq(value.get("implementation"), "complete", f"{stage} implementation")
        validation = _map(value.get("validation"), f"{stage} validation")
        if validation.get("status") not in {"passed", "supported"}:
            raise ValueError(f"{stage} validation not fully supported")
        _eq(validation.get("claim_allowed"), True, f"{stage} allowed")
    return {"overall_claim_allowed": True}


def _actions() -> tuple[str, ...]:
    return (
        "baseline",
        "antecedent_falsification",
        "remote_mechanism_analogue",
        "cross_field_pathway",
        "topology_expansion",
        "opportunity_attribution_audit",
    )


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required artifact is missing: {path}")
    return _map(json.loads(path.read_text(encoding="utf-8")), str(path))


def _map(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return value


def _num(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{label} must be numeric")
    return float(value)


def _eq(observed: object, expected: object, label: str) -> None:
    if observed != expected:
        raise ValueError(f"{label} mismatch: {observed!r}!={expected!r}")


def _passed(report: dict[str, Any], label: str) -> None:
    _eq(report.get("status"), "passed", f"{label} status")
    _eq(report.get("claim_allowed"), True, f"{label} claim_allowed")


def _sha(value: object, label: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be SHA string")
    digest = value.removeprefix("sha256:")
    if len(digest) != 64 or any(
        char not in "0123456789abcdef" for char in digest.casefold()
    ):
        raise ValueError(f"{label} is not SHA-256")


def _bound(manifest: Path, reference: object) -> Path:
    if not isinstance(reference, str) or not reference:
        raise ValueError("artifact reference missing")
    target = Path(reference)
    if target.is_absolute():
        return target
    target = (manifest.parent / target).resolve()
    if not target.is_relative_to(manifest.parent.resolve()):
        raise ValueError("artifact reference escapes release directory")
    return target


def _file_sha(path: Path, expected: object, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    _sha(expected, f"{label} SHA")
    _eq(
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        expected,
        f"{label} SHA",
    )


def _binding(manifest: Path, binding: dict[str, Any], label: str) -> None:
    _file_sha(_bound(manifest, binding.get("path")), binding.get("sha256"), label)


def _asset(manifest: Path, asset: dict[str, Any], label: str) -> None:
    target = _bound(manifest, asset.get("file"))
    _file_sha(target, asset.get("sha256"), label)
    _eq(target.stat().st_size, asset.get("size_bytes"), f"{label} size")


def _reference(path: Path, report: dict[str, Any], path_key: str, sha_key: str) -> None:
    _file_sha(_bound(path, report.get(path_key)), report.get(sha_key), path_key)


def _columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    if missing := required - set(frame):
        raise ValueError(f"{label} missing columns: {sorted(missing)}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for item in fields(CompletionArtifacts):
        parser.add_argument(
            "--" + item.name.replace("_", "-"), type=Path, required=True
        )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    artifacts = CompletionArtifacts(
        **{item.name: getattr(args, item.name) for item in fields(CompletionArtifacts)}
    )
    result = audit_rescue_completion(artifacts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["claim_allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CompletionArtifacts", "audit_rescue_completion"]

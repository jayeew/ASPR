"""Fail-closed Gate 1 and Gate 2 evaluators for the rescue plan."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .claim_attribution_eval import evaluate_claim_attribution
from .joint_structural_innovation_eval import evaluate_joint_value
from .off_policy_value_eval import (
    attach_target_policy_values,
    doubly_robust_value,
    paired_doubly_robust_contrast,
    paired_switch_doubly_robust_contrast,
    switch_doubly_robust_value,
)

POLICY_HOLDOUT_ROWS = 60
POLICY_ACTIONS = {
    "baseline",
    "antecedent_falsification",
    "remote_mechanism_analogue",
    "cross_field_pathway",
    "topology_expansion",
    "opportunity_attribution_audit",
}

GATE1_REQUIRED = {
    "paper_id",
    "claim_id",
    "attribution_weight",
    "future_adoption",
    "evidence_gate",
    "diffusion_potential",
    "structural_innovation_score",
    "shuffled_structural_score",
    "future_structural_outcome",
    "graph_percentile",
    "structural_contribution_share",
    "opportunity_context_share",
    "anatomy_limited",
    "context_observation_status",
}


def evaluate_gate1(
    frame: pd.DataFrame,
    *,
    minimum_papers: int = 100,
    minimum_deciles: int = 8,
    minimum_attribution_papers: int = 30,
    minimum_profile_papers_per_class: int = 10,
    minimum_nontruncated_papers: int = 50,
) -> dict[str, Any]:
    """Evaluate attribution, fusion mechanism, range, and shuffle degradation."""
    missing = sorted(GATE1_REQUIRED - set(frame))
    if missing:
        return _blocked("gate1", f"missing_columns:{','.join(missing)}")
    data = frame.dropna(subset=list(GATE1_REQUIRED)).copy()
    if "integration_split" in data:
        data = data[data["integration_split"].eq("development")].copy()
    papers = int(data["paper_id"].nunique())
    deciles = _decile_count(data["graph_percentile"])
    if papers < minimum_papers or deciles < minimum_deciles:
        return _blocked(
            "gate1",
            f"insufficient_range:papers={papers},deciles={deciles}",
        )
    attribution = evaluate_claim_attribution(data)
    real_rho = _spearman(data, "structural_innovation_score")
    shuffled_rho = _spearman(data, "shuffled_structural_score")
    monotone_violations = _monotone_violations(data)
    profile = _profile_diagnostics(data)
    sensitivity = _nontruncated_sensitivity(data)
    checks = {
        "claim_attribution_conserved": attribution["max_conservation_error"] <= 1e-6,
        "claim_attribution_identifiable": attribution["top_claim_eligible_papers"]
        >= minimum_attribution_papers,
        "claim_attribution_better_than_uniform": attribution["top_claim_accuracy"]
        > _uniform_top_claim_accuracy(data),
        "graph_profile_distinguishes_drivers": min(
            profile["structural_driven_papers"],
            profile["opportunity_driven_papers"],
        )
        >= minimum_profile_papers_per_class,
        "graph_profile_shares_conserved": profile["max_share_conservation_error"]
        <= 1e-6,
        "nontruncated_sensitivity_identifiable": sensitivity["papers"]
        >= minimum_nontruncated_papers,
        "nontruncated_shuffle_degrades_ranking": sensitivity["real_spearman"]
        > sensitivity["shuffled_spearman"],
        "fusion_monotone": monotone_violations == 0,
        "hgb_shuffle_degrades_ranking": real_rho > shuffled_rho,
        "full_score_range": deciles >= minimum_deciles,
    }
    return {
        "gate": "gate1",
        "status": "passed" if all(checks.values()) else "failed",
        "claim_allowed": all(checks.values()),
        "papers": papers,
        "score_deciles": deciles,
        "checks": checks,
        "attribution": attribution,
        "real_joint_spearman": real_rho,
        "shuffled_joint_spearman": shuffled_rho,
        "monotone_violations": monotone_violations,
        "profile": profile,
        "nontruncated_sensitivity": sensitivity,
    }


def evaluate_gate2(
    integration_frame: pd.DataFrame,
    policy_frame: pd.DataFrame,
    *,
    importance_threshold: float = 10.0,
) -> dict[str, Any]:
    """Evaluate frozen integration value, selective policy uplift, and guardrails."""
    integration_required = {
        "future_structural_outcome",
        "gear_evidence_score",
        "joint_structural_score",
        "shuffled_structural_score",
    }
    missing_integration = sorted(integration_required - set(integration_frame))
    policy_required = {
        "target_action",
        "logged_action",
        "outcome",
        "propensity",
        "q_logged",
        "q_target",
        "q_baseline",
        "wrong_correction",
        "unsupported_claim",
        "realized_cost",
    }
    missing_policy = sorted(policy_required - set(policy_frame))
    if missing_integration or missing_policy:
        reason = (
            f"missing_integration:{','.join(missing_integration)};"
            f"missing_policy:{','.join(missing_policy)}"
        )
        return _blocked("gate2", reason)
    real = evaluate_joint_value(integration_frame)
    shuffled = evaluate_joint_value(
        integration_frame.rename(
            columns={
                "joint_structural_score": "real_joint_structural_score",
                "shuffled_structural_score": "joint_structural_score",
            }
        )
    )
    policy_dr = doubly_robust_value(policy_frame, expected_n=POLICY_HOLDOUT_ROWS)
    policy_switch = switch_doubly_robust_value(
        policy_frame,
        importance_threshold=importance_threshold,
        expected_n=POLICY_HOLDOUT_ROWS,
    )
    baseline_actions = pd.Series("baseline", index=policy_frame.index)
    baseline = attach_target_policy_values(policy_frame, baseline_actions)
    baseline_dr = doubly_robust_value(baseline, expected_n=POLICY_HOLDOUT_ROWS)
    paired_uplift = paired_doubly_robust_contrast(
        policy_frame, baseline, expected_n=POLICY_HOLDOUT_ROWS
    )
    paired_switch_uplift = paired_switch_doubly_robust_contrast(
        policy_frame,
        baseline,
        importance_threshold=importance_threshold,
        expected_n=POLICY_HOLDOUT_ROWS,
    )
    uplift = paired_uplift["value"]
    uplift_lcb = paired_uplift["lcb_95"]
    guardrails = _guardrails(policy_frame)
    selective_abstain = bool(policy_frame["target_action"].eq("baseline").all())
    policy_check = uplift_lcb > 0.0 or selective_abstain
    checks = {
        "joint_beats_gear_only": real["integration_value"] > 0.0,
        "real_hgb_beats_shuffle": real["joint_spearman"] > shuffled["joint_spearman"],
        "policy_positive_lcb_or_abstains": policy_check,
        "wrong_correction_not_worse": guardrails["wrong_correction_pass"],
        "unsupported_claim_not_worse": guardrails["unsupported_claim_pass"],
        "cost_not_worse": guardrails["cost_pass"],
    }
    return {
        "gate": "gate2",
        "status": "passed" if all(checks.values()) else "failed",
        "claim_allowed": all(checks.values()),
        "checks": checks,
        "integration": {"real": real, "shuffled": shuffled},
        "policy": {
            "doubly_robust": policy_dr,
            "switch_dr": policy_switch,
            "baseline_doubly_robust": baseline_dr,
            "paired_doubly_robust_uplift": paired_uplift,
            "paired_switch_dr_uplift": paired_switch_uplift,
            "uplift": uplift,
            "uplift_lcb_95": uplift_lcb,
            "selective_abstain": selective_abstain,
        },
        "guardrails": guardrails,
    }


def evaluate_gate2_confirmatory(
    temporal_integration_frame: pd.DataFrame,
    domain_integration_frame: pd.DataFrame,
    policy_frame: pd.DataFrame,
    no_graph_policy_frame: pd.DataFrame | None = None,
    *,
    importance_threshold: float = 10.0,
    minimum_papers_per_holdout: int = 20,
) -> dict[str, Any]:
    """Require both frozen temporal and domain holdouts to pass Gate 2."""
    temporal_error = _validate_confirmatory_integration(
        temporal_integration_frame,
        allowed_splits={"temporal_holdout", "joint_time_domain_holdout"},
        minimum_papers=minimum_papers_per_holdout,
    )
    domain_error = _validate_confirmatory_integration(
        domain_integration_frame,
        allowed_splits={"domain_holdout", "joint_time_domain_holdout"},
        minimum_papers=minimum_papers_per_holdout,
    )
    graph_policy_error = _validate_confirmatory_policy(
        policy_frame, expected_feature_set="graph_features"
    )
    no_graph_policy_error = (
        "missing_no_graph_policy"
        if no_graph_policy_frame is None
        else _validate_confirmatory_policy(
            no_graph_policy_frame, expected_feature_set="no_graph_features"
        )
    )
    pairing_error = (
        _policy_provenance_pairing_error(policy_frame, no_graph_policy_frame)
        if not graph_policy_error
        and not no_graph_policy_error
        and no_graph_policy_frame is not None
        else None
    )
    if (
        temporal_error
        or domain_error
        or graph_policy_error
        or no_graph_policy_error
        or pairing_error
    ):
        return _blocked(
            "gate2",
            ";".join(
                (
                    f"temporal:{temporal_error or 'ok'}",
                    f"domain:{domain_error or 'ok'}",
                    f"graph_policy:{graph_policy_error or 'ok'}",
                    f"no_graph_policy:{no_graph_policy_error or 'ok'}",
                    f"policy_pairing:{pairing_error or 'ok'}",
                )
            ),
        )
    temporal = evaluate_gate2(
        temporal_integration_frame,
        policy_frame,
        importance_threshold=importance_threshold,
    )
    domain = evaluate_gate2(
        domain_integration_frame,
        policy_frame,
        importance_threshold=importance_threshold,
    )
    if (
        temporal["status"] == "not_identifiable"
        or domain["status"] == "not_identifiable"
    ):
        return _blocked(
            "gate2",
            "temporal_or_domain_evaluation_not_identifiable",
        )
    assert no_graph_policy_frame is not None
    graph_vs_no_graph = paired_doubly_robust_contrast(
        policy_frame,
        no_graph_policy_frame,
        expected_n=POLICY_HOLDOUT_ROWS,
    )
    graph_vs_no_graph_switch = paired_switch_doubly_robust_contrast(
        policy_frame,
        no_graph_policy_frame,
        importance_threshold=importance_threshold,
        expected_n=POLICY_HOLDOUT_ROWS,
    )
    both_abstain = bool(
        policy_frame["target_action"].eq("baseline").all()
        and no_graph_policy_frame["target_action"].eq("baseline").all()
    )
    integration_checks = {
        "temporal_joint_beats_gear_only": temporal["checks"]["joint_beats_gear_only"],
        "temporal_real_hgb_beats_shuffle": temporal["checks"]["real_hgb_beats_shuffle"],
        "domain_joint_beats_gear_only": domain["checks"]["joint_beats_gear_only"],
        "domain_real_hgb_beats_shuffle": domain["checks"]["real_hgb_beats_shuffle"],
    }
    policy_checks = {
        key: value
        for key, value in temporal["checks"].items()
        if key not in {"joint_beats_gear_only", "real_hgb_beats_shuffle"}
    }
    policy_checks["graph_policy_beats_no_graph_or_both_abstain"] = bool(
        graph_vs_no_graph["lcb_95"] > 0.0 or both_abstain
    )
    checks = {**integration_checks, **policy_checks}
    return {
        "gate": "gate2",
        "contract": "gear_gate2_dual_holdout_and_paired_policy_v2",
        "status": "passed" if all(checks.values()) else "failed",
        "claim_allowed": all(checks.values()),
        "checks": checks,
        "temporal_holdout": temporal["integration"],
        "domain_holdout": domain["integration"],
        "policy": temporal["policy"],
        "graph_vs_no_graph_policy": {
            **graph_vs_no_graph,
            "both_abstain": both_abstain,
            "paired_switch_dr_sensitivity": graph_vs_no_graph_switch,
        },
        "guardrails": temporal["guardrails"],
    }


def _monotone_violations(frame: pd.DataFrame) -> int:
    counterfactual = {"structural_score_at_zero", "structural_score_at_one"}
    if counterfactual.issubset(frame):
        low = pd.to_numeric(frame["structural_score_at_zero"]).to_numpy(float)
        observed = pd.to_numeric(frame["structural_innovation_score"]).to_numpy(float)
        high = pd.to_numeric(frame["structural_score_at_one"]).to_numpy(float)
        return int(((observed < low - 1e-12) | (high < observed - 1e-12)).sum())
    data = frame.copy()
    data["gate_bin"] = pd.to_numeric(data["evidence_gate"]).round(6)
    violations = 0
    for _, group in data.groupby(["paper_id", "claim_id", "gate_bin"]):
        ordered = group.sort_values("diffusion_potential")
        score = pd.to_numeric(ordered["structural_innovation_score"]).to_numpy(float)
        violations += int((np.diff(score) < -1e-12).sum())
    return violations


def _uniform_top_claim_accuracy(frame: pd.DataFrame) -> float:
    adoption_range = frame.groupby("paper_id")["future_adoption"].agg(
        lambda values: pd.to_numeric(values).max() - pd.to_numeric(values).min()
    )
    eligible = adoption_range[adoption_range.gt(0.0)].index
    counts = (
        frame[frame["paper_id"].isin(eligible)]
        .groupby("paper_id")["claim_id"]
        .nunique()
    )
    if counts.empty:
        return float("nan")
    return float((1.0 / counts).mean())


def _profile_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    paper = frame[
        [
            "paper_id",
            "structural_contribution_share",
            "opportunity_context_share",
            "anatomy_limited",
        ]
    ].drop_duplicates("paper_id")
    paper = paper[~paper["anatomy_limited"].astype(bool)].copy()
    structural = pd.to_numeric(paper["structural_contribution_share"])
    opportunity = pd.to_numeric(paper["opportunity_context_share"])
    return {
        "nonlimited_papers": len(paper),
        "structural_driven_papers": int(structural.gt(opportunity).sum()),
        "opportunity_driven_papers": int(opportunity.ge(structural).sum()),
        "max_share_conservation_error": float(
            (structural + opportunity - 1.0).abs().max()
        ),
    }


def _nontruncated_sensitivity(frame: pd.DataFrame) -> dict[str, Any]:
    data = frame[frame["context_observation_status"].ne("resolved_truncated")]
    attribution = evaluate_claim_attribution(data)
    return {
        "papers": int(data["paper_id"].nunique()),
        "top_claim_eligible_papers": attribution["top_claim_eligible_papers"],
        "top_claim_accuracy": attribution["top_claim_accuracy"],
        "uniform_top_claim_accuracy": _uniform_top_claim_accuracy(data),
        "real_spearman": _spearman(data, "structural_innovation_score"),
        "shuffled_spearman": _spearman(data, "shuffled_structural_score"),
    }


def _spearman(frame: pd.DataFrame, score: str) -> float:
    return float(
        pd.to_numeric(frame[score]).corr(
            pd.to_numeric(frame["future_structural_outcome"]), method="spearman"
        )
    )


def _decile_count(values: pd.Series) -> int:
    decile = np.clip(np.floor(pd.to_numeric(values) / 10.0), 0, 9)
    return int(pd.Series(decile).dropna().nunique())


def _guardrails(frame: pd.DataFrame) -> dict[str, Any]:
    baseline = frame["logged_action"].eq("baseline")
    selected = frame["logged_action"].eq(frame["target_action"])
    if not baseline.any() or not selected.any():
        return {
            "wrong_correction_pass": False,
            "unsupported_claim_pass": False,
            "cost_pass": False,
            "reason": "no_observed_baseline_or_target_matches",
        }

    def mean(column: str, mask: pd.Series) -> float:
        values = pd.to_numeric(frame.loc[mask, column], errors="coerce")
        propensities = pd.to_numeric(frame.loc[mask, "propensity"], errors="coerce")
        valid = values.notna() & propensities.notna() & propensities.gt(0.0)
        if not valid.any():
            return float("nan")
        weights = 1.0 / propensities.loc[valid]
        return float(np.average(values.loc[valid], weights=weights))

    policy_wrong = mean("wrong_correction", selected)
    baseline_wrong = mean("wrong_correction", baseline)
    policy_unsupported = mean("unsupported_claim", selected)
    baseline_unsupported = mean("unsupported_claim", baseline)
    policy_cost = mean("realized_cost", selected)
    baseline_cost = mean("realized_cost", baseline)
    return {
        "policy_wrong_correction_rate": policy_wrong,
        "baseline_wrong_correction_rate": baseline_wrong,
        "wrong_correction_pass": policy_wrong <= baseline_wrong,
        "policy_unsupported_claim_rate": policy_unsupported,
        "baseline_unsupported_claim_rate": baseline_unsupported,
        "unsupported_claim_pass": policy_unsupported <= baseline_unsupported,
        "policy_cost": policy_cost,
        "baseline_cost": baseline_cost,
        "cost_pass": policy_cost <= baseline_cost,
        "estimator": "self_normalized_inverse_propensity_weighted",
    }


def _validate_confirmatory_integration(
    frame: pd.DataFrame,
    *,
    allowed_splits: set[str],
    minimum_papers: int,
) -> str | None:
    required = {
        "paper_id",
        "integration_split",
        "future_structural_outcome",
        "gear_evidence_score",
        "joint_structural_score",
        "shuffled_structural_score",
    }
    missing = sorted(required - set(frame))
    if missing:
        return f"missing_columns:{','.join(missing)}"
    data = frame.dropna(subset=list(required)).copy()
    observed_splits = set(data["integration_split"].astype(str))
    unexpected = sorted(observed_splits - allowed_splits)
    if unexpected:
        return f"unexpected_splits:{','.join(unexpected)}"
    papers = int(data["paper_id"].astype(str).nunique())
    if papers < minimum_papers:
        return f"insufficient_papers:{papers}<{minimum_papers}"
    return None


def _validate_confirmatory_policy(
    frame: pd.DataFrame, *, expected_feature_set: str
) -> str | None:
    required = {
        "paper_id",
        "context_id",
        "experiment_split",
        "policy_fold_id",
        "policy_development_input_sha256",
        "policy_holdout_input_sha256",
        "policy_feature_set",
        "matched_budget",
        "logged_action",
        "target_action",
        "outcome",
        "propensity",
        "q_logged",
        "q_target",
        "q_baseline",
        "wrong_correction",
        "unsupported_claim",
        "realized_cost",
    }
    missing = sorted(required - set(frame))
    if missing:
        return f"missing_columns:{','.join(missing)}"
    if len(frame) != POLICY_HOLDOUT_ROWS:
        return f"row_count:{len(frame)}!={POLICY_HOLDOUT_ROWS}"
    if frame[["paper_id", "context_id"]].isna().any().any():
        return "null_paper_or_context"
    if frame["paper_id"].astype(str).duplicated().any():
        return "duplicate_paper_id"
    if frame["context_id"].astype(str).duplicated().any():
        return "duplicate_context_id"
    if not frame["experiment_split"].astype(str).eq("confirmatory_holdout").all():
        return "unexpected_experiment_split"
    if not frame["policy_fold_id"].astype(str).eq("holdout").all():
        return "unexpected_policy_fold"
    counts = frame["logged_action"].astype(str).value_counts().to_dict()
    if counts != {action: 10 for action in POLICY_ACTIONS}:
        return f"unbalanced_logged_actions:{counts}"
    unknown_targets = set(frame["target_action"].astype(str)) - POLICY_ACTIONS
    if unknown_targets:
        return f"unknown_target_actions:{','.join(sorted(unknown_targets))}"
    propensity = pd.to_numeric(frame["propensity"], errors="coerce")
    if not propensity.eq(1.0 / len(POLICY_ACTIONS)).all():
        return "invalid_propensity"
    if not pd.to_numeric(frame["matched_budget"], errors="coerce").eq(20).all():
        return "invalid_matched_budget"
    if set(frame["policy_feature_set"].astype(str)) != {expected_feature_set}:
        return "unexpected_policy_feature_set"
    for column in (
        "policy_development_input_sha256",
        "policy_holdout_input_sha256",
    ):
        values = set(frame[column].astype(str))
        if len(values) != 1 or not _is_sha256(next(iter(values))):
            return f"invalid_{column}"
    numeric = frame[
        ["outcome", "propensity", "q_logged", "q_target", "q_baseline", "realized_cost"]
    ].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(float)).all():
        return "nonfinite_off_policy_values"
    return None


def _is_sha256(value: str) -> bool:
    prefix = "sha256:"
    digest = value.removeprefix(prefix)
    return (
        value.startswith(prefix)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    )


def _policy_provenance_pairing_error(
    graph: pd.DataFrame, no_graph: pd.DataFrame
) -> str | None:
    for column in (
        "policy_development_input_sha256",
        "policy_holdout_input_sha256",
    ):
        if set(graph[column].astype(str)) != set(no_graph[column].astype(str)):
            return f"input_sha_mismatch:{column}"
    try:
        paired_doubly_robust_contrast(graph, no_graph, expected_n=POLICY_HOLDOUT_ROWS)
    except ValueError as exc:
        return f"unpaired_holdout:{exc}"
    return None


def _blocked(gate: str, reason: str) -> dict[str, Any]:
    return {
        "gate": gate,
        "status": "not_identifiable",
        "reason": reason,
        "claim_allowed": False,
    }


__all__ = ["evaluate_gate1", "evaluate_gate2", "evaluate_gate2_confirmatory"]

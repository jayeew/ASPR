from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.gear.evaluation.claim_attribution_eval import (
    evaluate_claim_attribution,
)
from experiments.gear.evaluation.graph_action_randomized_runner import (
    ACTIONS,
    assign_randomized_actions,
    finalize_action_log,
)
from experiments.gear.evaluation.off_policy_value_eval import (
    doubly_robust_value,
    paired_doubly_robust_contrast,
    switch_doubly_robust_value,
)
from experiments.gear.evaluation.policy_training import (
    apply_selective_policy,
    fit_action_promotion_rules,
)
from experiments.gear.evaluation.rescue_stage_gates import (
    evaluate_gate1,
    evaluate_gate2,
    evaluate_gate2_confirmatory,
)
from experiments.gear.evaluation.run_real_perturbation_validation import (
    _latest_temporal_split,
    _temporal_hybrid,
)
from experiments.gear.evaluation.structural_head_training import (
    run_cross_fitted_structural_heads,
)
from gear.graph_action_policy import RandomizedGraphActionSelector
from gear.nature_multihorizon.perturbation_targets import (
    build_perturbation_components,
)
from gear.nature_multihorizon.targets_v6 import (
    DIFFUSION_COMPONENTS,
    PERTURBATION_COMPONENTS,
    build_fold_local_structural_targets,
)


def _target_frame(rows: int, offset: float = 0.0) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    frame = pd.DataFrame(
        {
            "horizon": 5,
            "future_uptake": (index.astype(int) % 3 != 0).astype(int),
            "n_future_citers": index + 1,
            "domain12": np.where(index.astype(int) % 2, "a", "b"),
            "publication_year": 2000 + index.astype(int) % 4,
            "opportunity_score": (index % 5) / 5,
        }
    )
    for position, column in enumerate(DIFFUSION_COMPONENTS):
        frame[column] = index + position + offset
    for position, column in enumerate(PERTURBATION_COMPONENTS):
        frame[column] = index / max(rows, 1) + position / 10 + offset
    return frame


def test_stage_b_targets_are_fold_local_and_emit_provenance() -> None:
    training = _target_frame(30)
    evaluation = _target_frame(8, offset=1000.0)

    train_targets, evaluation_targets, provenance = build_fold_local_structural_targets(
        training,
        evaluation,
        outer_fold_id="fold-1",
        horizon_years=5,
    )

    assert len(train_targets) == 30
    assert len(evaluation_targets) == 8
    assert {item.head for item in provenance} == {
        "diffusion",
        "excess_diffusion",
        "perturbation",
    }
    assert all(item.fit_scope == "outer_training_fold_only" for item in provenance)
    assert all(item.training_rows == 30 for item in provenance)


def test_temporal_hybrid_replaces_latest_block_with_forward_predictions() -> None:
    frame = pd.DataFrame(
        {
            "paper_id": ["early", "latest"],
            "outer_fold_id": [4, 6],
            "publication_year": [2011, 2016],
        }
    )
    domain = pd.DataFrame(
        {
            "paper_id": ["early", "latest"],
            "domain12": ["a", "b"],
            "publication_year": [2011, 2016],
            "perturbation_target_fold": [0.1, 0.2],
            "perturbation_head_p": [0.3, 0.4],
            "shuffled_perturbation_head_p": [0.5, 0.6],
            "heldout_domain12": ["a", "b"],
        }
    )
    forward = domain.loc[domain["paper_id"].eq("latest")].copy()
    forward["perturbation_head_p"] = 0.9

    output = _temporal_hybrid(domain, forward).set_index("paper_id")

    assert _latest_temporal_split(frame) == 6
    assert output.at["early", "perturbation_head_p"] == 0.3
    assert output.at["early", "prediction_protocol"] == "domain_oof_development"
    assert output.at["latest", "perturbation_head_p"] == 0.9
    assert (
        output.at["latest", "prediction_protocol"] == "forward_temporal_latest_holdout"
    )


def test_perturbation_components_are_bounded_and_auditable() -> None:
    raw = pd.DataFrame(
        {
            "future_new_community_count": [2],
            "total_future_community_count": [4],
            "outsider_citer_share": [0.75],
            "pre_cross_community_edge_rate": [0.1],
            "post_cross_community_edge_rate": [0.4],
            "focal_only_citers": [8],
            "focal_and_predecessor_citers": [2],
            "pre_shortest_path": [5],
            "post_shortest_path": [3],
            "claim_adoption_breadth": [0.6],
        }
    )

    result = build_perturbation_components(raw)

    assert result.at[0, "boundary_expansion"] == pytest.approx(0.625)
    assert result.at[0, "community_mixing_change"] == pytest.approx(0.3)
    assert result.at[0, "dependency_displacement"] == pytest.approx(0.8)
    assert result.at[0, "path_shortening"] == pytest.approx(0.4)
    assert result.at[0, "claim_adoption_breadth"] == pytest.approx(0.6)
    assert result.at[0, "path_shortening_claim_adoption"] == pytest.approx(0.5)


def test_missing_claim_adoption_does_not_erase_graph_perturbation() -> None:
    raw = pd.DataFrame(
        {
            "future_new_community_count": [2],
            "total_future_community_count": [4],
            "outsider_citer_share": [0.5],
            "pre_cross_community_edge_rate": [0.2],
            "post_cross_community_edge_rate": [0.4],
            "focal_only_citers": [3],
            "focal_and_predecessor_citers": [1],
            "pre_shortest_path": [4],
            "post_shortest_path": [2],
            "claim_adoption_breadth": [np.nan],
        }
    )

    result = build_perturbation_components(raw)

    assert result.at[0, "path_shortening"] == pytest.approx(0.5)
    assert pd.isna(result.at[0, "path_shortening_claim_adoption"])


def test_randomized_action_selector_preserves_assignment_and_propensity() -> None:
    decision = RandomizedGraphActionSelector("cross_field_pathway", 1.0 / 6.0).decide(
        object()
    )

    assert decision.action == "cross_field_pathway"
    assert decision.propensity == pytest.approx(1.0 / 6.0)
    assert decision.selected is True
    assert decision.reason == "preassigned_randomized_action"


def test_structural_heads_cross_fit_and_reject_future_features() -> None:
    rows = 90
    x = np.linspace(0.0, 1.0, rows)
    frame = pd.DataFrame(
        {
            "paper_id": [f"paper-{index}" for index in range(rows)],
            "outer_fold_id": np.arange(rows) % 3,
            "t0_feature": x,
            "future_uptake": (x > 0.45).astype(int),
            "excess_diffusion_fold": x,
            "perturbation_fold": np.sqrt(x),
        }
    )

    predictions, manifest = run_cross_fitted_structural_heads(
        frame, feature_columns=["t0_feature"]
    )

    assert len(predictions) == rows
    assert manifest["folds"] == 3
    assert predictions["aspr_joint"].between(0.0, 1.0).all()
    with pytest.raises(ValueError, match="target leakage"):
        run_cross_fitted_structural_heads(frame, feature_columns=["future_uptake"])


def test_randomized_action_logging_and_switch_dr() -> None:
    contexts = pd.DataFrame(
        {
            "context_id": [f"context-{index}" for index in range(120)],
            "paper_id": [f"paper-{index // 2}" for index in range(120)],
            "claim_id": [f"claim-{index}" for index in range(120)],
            "domain": ["a", "b"] * 60,
        }
    )
    assigned = assign_randomized_actions(
        contexts, seed=17, budget=8, stratify_by=["domain"]
    )
    assigned["useful_relation_yield"] = 1.0
    assigned["correction_quality"] = 0.8
    assigned["claim_recall_gain"] = 0.1
    assigned["wrong_correction"] = False
    assigned["unsupported_claim"] = False
    assigned["realized_cost"] = 8.0
    logged = finalize_action_log(assigned)
    logged["target_action"] = logged["logged_action"]
    logged["q_logged"] = logged["outcome"]
    logged["q_target"] = logged["outcome"]

    assert set(logged["assigned_action"]).issubset(ACTIONS)
    assert logged["propensity"].eq(1.0 / len(ACTIONS)).all()
    assert doubly_robust_value(logged)["value"] == pytest.approx(
        logged["outcome"].mean()
    )
    assert switch_doubly_robust_value(logged)["value"] == pytest.approx(
        logged["outcome"].mean()
    )


def test_gate1_and_gate2_fail_closed_when_confirmatory_data_are_missing() -> None:
    gate1 = evaluate_gate1(pd.DataFrame({"paper_id": ["paper-1"]}))
    gate2 = evaluate_gate2(pd.DataFrame(), pd.DataFrame())

    assert gate1["status"] == "not_identifiable"
    assert gate1["claim_allowed"] is False
    assert gate2["status"] == "not_identifiable"
    assert gate2["claim_allowed"] is False


def test_claim_attribution_top_accuracy_excludes_all_tied_adoption_papers() -> None:
    frame = pd.DataFrame(
        {
            "paper_id": ["informative", "informative", "zero", "zero"],
            "claim_id": ["a", "b", "a", "b"],
            "attribution_weight": [0.2, 0.8, 0.9, 0.1],
            "future_adoption": [0.0, 1.0, 0.0, 0.0],
        }
    )

    result = evaluate_claim_attribution(frame)

    assert result["papers"] == 2
    assert result["top_claim_eligible_papers"] == 1
    assert result["top_claim_accuracy"] == 1.0


def _integration_holdout(split: str, rows: int = 24) -> pd.DataFrame:
    outcome = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "paper_id": [f"{split}-{index}" for index in range(rows)],
            "integration_split": split,
            "future_structural_outcome": outcome,
            "gear_evidence_score": np.roll(outcome, rows // 2),
            "joint_structural_score": outcome,
            "shuffled_structural_score": outcome[::-1],
        }
    )


def _abstaining_policy_holdout(
    feature_set: str = "graph_features",
) -> pd.DataFrame:
    logged = np.repeat(ACTIONS, 10)
    frame = pd.DataFrame(
        {
            "paper_id": [f"policy-paper-{index}" for index in range(60)],
            "context_id": [f"policy-context-{index}" for index in range(60)],
            "experiment_split": "confirmatory_holdout",
            "policy_fold_id": "holdout",
            "policy_development_input_sha256": "sha256:" + "a" * 64,
            "policy_holdout_input_sha256": "sha256:" + "b" * 64,
            "policy_feature_set": feature_set,
            "matched_budget": 20,
            "logged_action": logged,
            "target_action": "baseline",
            "outcome": 0.0,
            "propensity": 1.0 / len(ACTIONS),
            "q_logged": 0.0,
            "q_target": 0.0,
            "q_baseline": 0.0,
            "wrong_correction": False,
            "unsupported_claim": False,
            "realized_cost": 8.0,
        }
    )
    return frame


def test_gate2_confirmatory_requires_and_passes_both_frozen_holdouts() -> None:
    temporal = _integration_holdout("temporal_holdout")
    domain = _integration_holdout("domain_holdout")

    result = evaluate_gate2_confirmatory(
        temporal,
        domain,
        _abstaining_policy_holdout(),
        _abstaining_policy_holdout("no_graph_features"),
    )

    assert result["status"] == "passed"
    assert result["claim_allowed"] is True
    assert result["contract"] == "gear_gate2_dual_holdout_and_paired_policy_v2"
    assert result["graph_vs_no_graph_policy"]["both_abstain"] is True


def test_gate2_confirmatory_rejects_development_rows() -> None:
    result = evaluate_gate2_confirmatory(
        _integration_holdout("development"),
        _integration_holdout("domain_holdout"),
        _abstaining_policy_holdout(),
        _abstaining_policy_holdout("no_graph_features"),
    )

    assert result["status"] == "not_identifiable"
    assert result["claim_allowed"] is False
    assert "unexpected_splits:development" in result["reason"]


def test_gate2_confirmatory_rejects_unpaired_policy_input_sha() -> None:
    graph = _abstaining_policy_holdout()
    no_graph = _abstaining_policy_holdout("no_graph_features")
    no_graph["policy_holdout_input_sha256"] = "sha256:" + "c" * 64

    result = evaluate_gate2_confirmatory(
        _integration_holdout("temporal_holdout"),
        _integration_holdout("domain_holdout"),
        graph,
        no_graph,
    )

    assert result["status"] == "not_identifiable"
    assert "input_sha_mismatch" in result["reason"]


def test_off_policy_rejects_any_nonfinite_score() -> None:
    frame = _abstaining_policy_holdout()
    frame.loc[0, "q_target"] = np.nan

    with pytest.raises(ValueError, match="numeric columns must be finite"):
        doubly_robust_value(frame, expected_n=60)


def test_paired_dr_uses_covariance_and_exact_identity() -> None:
    candidate = _abstaining_policy_holdout()
    reference = _abstaining_policy_holdout("no_graph_features")
    candidate["q_target"] = 0.25

    contrast = paired_doubly_robust_contrast(candidate, reference, expected_n=60)

    assert contrast["n"] == 60
    assert contrast["value"] == pytest.approx(0.25)
    assert contrast["standard_error"] == pytest.approx(0.0)


def test_selective_policy_promotes_only_safe_positive_uplift() -> None:
    actions = np.repeat(ACTIONS, 20)
    outcome_by_action = {
        action: (1.0 if action == "antecedent_falsification" else 0.5)
        for action in ACTIONS
    }
    development = pd.DataFrame(
        {
            "logged_action": actions,
            "outcome": [outcome_by_action[action] for action in actions],
            "wrong_correction": False,
            "unsupported_claim": False,
            "realized_cost": 8.0,
        }
    )
    for action in ACTIONS:
        development[f"q_{action}"] = outcome_by_action[action]
    rules = fit_action_promotion_rules(development)
    holdout = development.head(5).copy()

    selected = apply_selective_policy(holdout, rules)

    assert selected.eq("antecedent_falsification").all()

"""Evaluate a frozen selective Graph policy on an independent holdout log."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.linear_model import Ridge

from gear.graph_action_policy import (
    ACTION_POLICY_FEATURE_SCHEMA,
    ACTION_POLICY_FEATURES,
    GRAPH_ACTIONS,
    ActionPolicyRule,
    GraphActionQModel,
)

from .graph_action_randomized_runner import ACTIONS
from .off_policy_value_eval import (
    attach_target_policy_values,
    doubly_robust_value,
    fit_development_and_holdout_action_values,
    paired_doubly_robust_contrast,
    paired_switch_doubly_robust_contrast,
    switch_doubly_robust_value,
)
from .policy_training import apply_selective_policy, fit_action_promotion_rules

EXPECTED_DEVELOPMENT_ROWS = 90
EXPECTED_HOLDOUT_ROWS = 60
EXPECTED_BUDGET = 20
GRAPH_POLICY_T0_FEATURES = list(ACTION_POLICY_FEATURES)


def run_policy_evaluation(
    development_path: Path,
    holdout_path: Path,
    output_dir: Path,
    *,
    feature_columns: list[str] | None = None,
    fold_column: str = "policy_fold_id",
) -> dict[str, Any]:
    """Fit promotion margins on development data and score a frozen holdout."""
    development = _read(development_path)
    holdout = _read(holdout_path)
    _validate_frozen_inputs(development, holdout, fold_column=fold_column)
    development_sha256 = _sha256(development_path)
    holdout_sha256 = _sha256(holdout_path)
    runtime_model: GraphActionQModel | None = None
    if feature_columns and any(
        column.startswith("graph_") for column in feature_columns
    ):
        if feature_columns != GRAPH_POLICY_T0_FEATURES:
            raise ValueError(
                "Graph policy features must equal the frozen runtime T0 schema"
            )
        development, holdout, runtime_model = _fit_runtime_linear_q_model(
            development,
            holdout,
            feature_columns=feature_columns,
            fold_column=fold_column,
        )
    elif feature_columns:
        development, holdout = fit_development_and_holdout_action_values(
            development,
            holdout,
            feature_columns=feature_columns,
            fold_column=fold_column,
        )
    rules = fit_action_promotion_rules(development)
    target = apply_selective_policy(holdout, rules)
    evaluated = attach_target_policy_values(holdout, target)
    baseline = attach_target_policy_values(
        holdout, pd.Series("baseline", index=holdout.index)
    )
    policy_dr = doubly_robust_value(evaluated, expected_n=EXPECTED_HOLDOUT_ROWS)
    baseline_dr = doubly_robust_value(baseline, expected_n=EXPECTED_HOLDOUT_ROWS)
    paired_uplift = paired_doubly_robust_contrast(
        evaluated, baseline, expected_n=EXPECTED_HOLDOUT_ROWS
    )
    paired_switch_uplift = paired_switch_doubly_robust_contrast(
        evaluated, baseline, expected_n=EXPECTED_HOLDOUT_ROWS
    )
    policy_kind = (
        "graph_features"
        if feature_columns
        and any(column.startswith("graph_") for column in feature_columns)
        else "no_graph_features"
    )
    evaluated["policy_development_input_sha256"] = development_sha256
    evaluated["policy_holdout_input_sha256"] = holdout_sha256
    evaluated["policy_feature_set"] = policy_kind
    result = {
        "contract": "gear_selective_graph_policy_holdout_v2",
        "development_rows": len(development),
        "confirmatory_holdout_rows": len(holdout),
        "feature_columns": feature_columns or [],
        "policy_feature_set": policy_kind,
        "fold_column": fold_column,
        "development_input_sha256": development_sha256,
        "holdout_input_sha256": holdout_sha256,
        "rules": rules,
        "target_action_counts": target.value_counts().to_dict(),
        "doubly_robust": policy_dr,
        "switch_dr": switch_doubly_robust_value(
            evaluated, expected_n=EXPECTED_HOLDOUT_ROWS
        ),
        "baseline_doubly_robust": baseline_dr,
        "paired_doubly_robust_uplift": paired_uplift,
        "paired_switch_dr_uplift": paired_switch_uplift,
        "uplift": paired_uplift["value"],
        "uplift_lcb_95": paired_uplift["lcb_95"],
        "selective_abstain": bool(target.eq("baseline").all()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = output_dir / "policy_holdout_scored.parquet"
    evaluated.to_parquet(policy_path, index=False)
    if runtime_model is not None:
        model_path = output_dir / "graph_action_q_model.json"
        replay_path = output_dir / "graph_action_policy_replay.json"
        model_path.write_text(
            runtime_model.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        _write_runtime_replay(runtime_model, holdout, replay_path)
        result["runtime_candidate"] = {
            "model_path": model_path.name,
            "model_sha256": _sha256(model_path),
            "replay_path": replay_path.name,
            "replay_sha256": _sha256(replay_path),
            "q_model_family": "linear_t0_v1",
            "feature_schema_version": ACTION_POLICY_FEATURE_SCHEMA,
            "feature_family": "graph_features",
            "future_features_used": False,
            "future_outcomes_used_at_inference": False,
            "sealed_holdout_used_for_fitting": False,
            "training_rows": EXPECTED_DEVELOPMENT_ROWS,
            "training_scope": "development_only",
            "gear_evidence_gap_status": (
                "phase_one_excluded_not_available_at_pre_retrieval_decision"
            ),
        }
    (output_dir / "policy_holdout_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _fit_runtime_linear_q_model(
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    *,
    feature_columns: list[str],
    fold_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, GraphActionQModel]:
    x_development = _finite_design(development, feature_columns, "development")
    x_holdout = _finite_design(holdout, feature_columns, "holdout")
    scored_development = development.copy()
    for action in ACTIONS:
        scored_development[f"q_{action}"] = float("nan")
    for fold in ("0", "1", "2"):
        test = development[fold_column].astype(str).eq(fold)
        train = ~test
        for action in ACTIONS:
            action_train = train & development["logged_action"].astype(str).eq(action)
            if int(action_train.sum()) != 10:
                raise ValueError(
                    "runtime Q OOF requires ten training rows per action/fold"
                )
            fitted = Ridge(alpha=1.0).fit(
                x_development.loc[action_train],
                pd.to_numeric(development.loc[action_train, "outcome"]),
            )
            scored_development.loc[test, f"q_{action}"] = fitted.predict(
                x_development.loc[test]
            )
    scored_development["q_logged"] = [
        scored_development.at[index, f"q_{action}"]
        for index, action in scored_development["logged_action"].items()
    ]
    scored_holdout = holdout.copy()
    intercepts: dict[str, float] = {}
    coefficients: dict[str, list[float]] = {}
    for action in ACTIONS:
        action_rows = development["logged_action"].astype(str).eq(action)
        if int(action_rows.sum()) != 15:
            raise ValueError(
                "runtime Q fit requires fifteen development rows per action"
            )
        fitted = Ridge(alpha=1.0).fit(
            x_development.loc[action_rows],
            pd.to_numeric(development.loc[action_rows, "outcome"]),
        )
        intercepts[action] = float(fitted.intercept_)
        coefficients[action] = [float(value) for value in fitted.coef_]
        scored_holdout[f"q_{action}"] = fitted.predict(x_holdout)
    scored_holdout["q_logged"] = [
        scored_holdout.at[index, f"q_{action}"]
        for index, action in scored_holdout["logged_action"].items()
    ]
    rules = fit_action_promotion_rules(scored_development)
    model = GraphActionQModel(
        feature_schema_version=ACTION_POLICY_FEATURE_SCHEMA,
        feature_family="graph_features",
        feature_names=feature_columns,
        intercepts=intercepts,
        coefficients=coefficients,
        rules={
            action: ActionPolicyRule(
                uplift_margin=float(rules[action]["uplift_margin"]),
                development_rows=15,
                development_average_uplift=float(
                    rules[action]["development_average_uplift"]
                ),
                development_average_uplift_lcb=float(
                    rules[action]["development_average_uplift_lcb"]
                ),
                development_positive_uplift_pass=bool(
                    rules[action]["development_positive_uplift_pass"]
                ),
                wrong_correction_pass=bool(rules[action]["wrong_correction_pass"]),
                unsupported_claim_pass=bool(rules[action]["unsupported_claim_pass"]),
                cost_pass=bool(rules[action]["cost_pass"]),
            )
            for action in GRAPH_ACTIONS
        },
        selection_rule="max_positive_q_minus_baseline_minus_uplift_margin_v1",
        tie_break="uplift_lcb_then_uplift_then_action_lexicographic_v1",
        fallback_action="abstain",
        future_features_used=False,
        training_rows=90,
        training_scope="development_only",
        sealed_holdout_used_for_fitting=False,
        gear_evidence_gap_status=(
            "phase_one_excluded_not_available_at_pre_retrieval_decision"
        ),
    )
    return scored_development, scored_holdout, model


def _finite_design(frame: pd.DataFrame, columns: list[str], label: str) -> pd.DataFrame:
    design = frame[columns].apply(pd.to_numeric, errors="coerce")
    if design.isna().any().any() or not all(
        math.isfinite(value) for value in design.to_numpy().ravel()
    ):
        raise ValueError(f"{label} runtime policy T0 features are incomplete")
    return design


def _write_runtime_replay(
    model: GraphActionQModel, holdout: pd.DataFrame, path: Path
) -> None:
    rows = []
    for _, row in holdout.iterrows():
        features = [float(row[column]) for column in ACTION_POLICY_FEATURES]
        rows.append(
            {
                "features": features,
                "expected_q_values": model.predict(features),
                "expected_decision": model.decision(features).model_dump(mode="json"),
            }
        )
    path.write_text(json.dumps({"rows": rows}, sort_keys=True) + "\n", encoding="utf-8")


def _read(path: Path) -> pd.DataFrame:
    return (
        pd.read_csv(path) if path.suffix.casefold() == ".csv" else pd.read_parquet(path)
    )


def _validate_frozen_inputs(
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    *,
    fold_column: str,
) -> None:
    required = {
        "paper_id",
        "context_id",
        "experiment_split",
        "logged_action",
        "propensity",
        "matched_budget",
        fold_column,
    }
    for label, frame, expected_rows, expected_split in (
        (
            "development",
            development,
            EXPECTED_DEVELOPMENT_ROWS,
            "development",
        ),
        (
            "holdout",
            holdout,
            EXPECTED_HOLDOUT_ROWS,
            "confirmatory_holdout",
        ),
    ):
        missing = sorted(required - set(frame))
        if missing:
            raise ValueError(f"{label} policy columns are missing: {missing}")
        if len(frame) != expected_rows:
            raise ValueError(
                f"{label} row count changed: {len(frame)} != {expected_rows}"
            )
        if not frame["experiment_split"].astype(str).eq(expected_split).all():
            raise ValueError(f"{label} contains rows outside its frozen split")
        _validate_identity_and_assignment(frame, label=label)
    for identity in ("paper_id", "context_id"):
        overlap = set(development[identity].astype(str)) & set(
            holdout[identity].astype(str)
        )
        if overlap:
            raise ValueError(f"development/holdout {identity} overlap: {len(overlap)}")
    _validate_folds(development, holdout, fold_column=fold_column)


def _validate_identity_and_assignment(frame: pd.DataFrame, *, label: str) -> None:
    identity = ["paper_id", "context_id"]
    if frame[identity].isna().any().any():
        raise ValueError(f"{label} paper/context identities must be non-null")
    if frame["paper_id"].astype(str).duplicated().any():
        raise ValueError(f"{label} paper_id must be unique")
    if frame["context_id"].astype(str).duplicated().any():
        raise ValueError(f"{label} context_id must be unique")
    counts = frame["logged_action"].astype(str).value_counts().to_dict()
    expected = 15 if label == "development" else 10
    if counts != {action: expected for action in ACTIONS}:
        raise ValueError(f"{label} action allocation changed: {counts}")
    propensity = pd.to_numeric(frame["propensity"], errors="coerce")
    if not propensity.eq(1.0 / len(ACTIONS)).all():
        raise ValueError(f"{label} propensity changed from 1/{len(ACTIONS)}")
    budget = pd.to_numeric(frame["matched_budget"], errors="coerce")
    if not budget.eq(EXPECTED_BUDGET).all():
        raise ValueError(f"{label} matched budget changed")


def _validate_folds(
    development: pd.DataFrame, holdout: pd.DataFrame, *, fold_column: str
) -> None:
    development_folds = development[fold_column].astype(str)
    if set(development_folds) != {"0", "1", "2"}:
        raise ValueError("development policy folds must be exactly 0,1,2")
    fold_action_counts = (
        development.assign(_fold=development_folds)
        .groupby(["_fold", "logged_action"])
        .size()
    )
    if any(
        int(fold_action_counts.get((fold, action), 0)) != 5
        for fold in ("0", "1", "2")
        for action in ACTIONS
    ):
        raise ValueError("development folds are not action-balanced at five rows each")
    if not holdout[fold_column].astype(str).eq("holdout").all():
        raise ValueError("confirmatory holdout fold must be exactly 'holdout'")


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature", action="append")
    parser.add_argument("--fold-column", default="policy_fold_id")
    args = parser.parse_args()
    result = run_policy_evaluation(
        args.development,
        args.holdout,
        args.output_dir,
        feature_columns=args.feature,
        fold_column=args.fold_column,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

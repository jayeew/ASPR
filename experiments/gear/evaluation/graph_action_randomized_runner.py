"""Matched-budget randomized logging design for Graph actions A0-A5."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

ACTIONS = (
    "baseline",
    "antecedent_falsification",
    "remote_mechanism_analogue",
    "cross_field_pathway",
    "topology_expansion",
    "opportunity_attribution_audit",
)


def assign_randomized_actions(
    contexts: pd.DataFrame,
    *,
    seed: int,
    budget: int,
    action_probabilities: Mapping[str, float] | None = None,
    stratify_by: Sequence[str] = (),
) -> pd.DataFrame:
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if "context_id" not in contexts:
        raise ValueError("contexts requires context_id")
    missing_strata = sorted(set(stratify_by) - set(contexts))
    if missing_strata:
        raise ValueError(f"randomization strata are missing: {missing_strata}")
    probabilities = _probabilities(action_probabilities)
    generator = np.random.default_rng(seed)
    output = contexts.copy()
    output["assigned_action"] = ""
    if stratify_by:
        grouping: str | list[str] = (
            stratify_by[0] if len(stratify_by) == 1 else list(stratify_by)
        )
        groups = output.groupby(grouping, observed=True, dropna=False).groups
    else:
        groups = {"all": output.index}
    for indexes in groups.values():
        selected = generator.choice(
            ACTIONS,
            size=len(indexes),
            replace=True,
            p=[probabilities[action] for action in ACTIONS],
        )
        output.loc[indexes, "assigned_action"] = selected
    output["propensity"] = output["assigned_action"].map(probabilities).astype(float)
    output["matched_budget"] = int(budget)
    output["randomization_seed"] = int(seed)
    return output


def finalize_action_log(
    assignments: pd.DataFrame,
    *,
    utility_weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Validate observed action outcomes and derive the registered utility."""
    required = {
        "context_id",
        "paper_id",
        "claim_id",
        "assigned_action",
        "propensity",
        "matched_budget",
        "useful_relation_yield",
        "correction_quality",
        "claim_recall_gain",
        "wrong_correction",
        "unsupported_claim",
        "realized_cost",
    }
    missing = sorted(required - set(assignments))
    if missing:
        raise ValueError(f"action log columns are missing: {missing}")
    weights = {
        "useful_relation_yield": 1.0,
        "correction_quality": 1.0,
        "claim_recall_gain": 1.0,
        "wrong_correction": -2.0,
        "unsupported_claim": -2.0,
        "realized_cost": -0.05,
        **(utility_weights or {}),
    }
    output = assignments.copy()
    output["outcome"] = sum(
        float(weight) * pd.to_numeric(output[column], errors="coerce")
        for column, weight in weights.items()
    )
    output["logged_action"] = output["assigned_action"]
    if output[list(required)].isna().any().any() or output["outcome"].isna().any():
        raise ValueError("action log contains missing required observations")
    if not output["assigned_action"].isin(ACTIONS).all():
        raise ValueError("action log contains an unknown action")
    return output


def _probabilities(values: Mapping[str, float] | None) -> dict[str, float]:
    if values is None:
        return {action: 1.0 / len(ACTIONS) for action in ACTIONS}
    if set(values) != set(ACTIONS):
        raise ValueError("action probabilities must cover A0-A5 exactly")
    probabilities = {action: float(values[action]) for action in ACTIONS}
    if any(value <= 0.0 for value in probabilities.values()) or not np.isclose(
        sum(probabilities.values()), 1.0
    ):
        raise ValueError("action probabilities must be positive and sum to one")
    return probabilities


__all__ = ["ACTIONS", "assign_randomized_actions", "finalize_action_log"]

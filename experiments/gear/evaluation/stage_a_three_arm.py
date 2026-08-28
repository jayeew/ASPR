"""Matched-cohort GEAR-only, GEAR+HGB, and shuffled-HGB evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .joint_structural_innovation_eval import evaluate_joint_value

REQUIRED_COLUMNS = {
    "paper_id",
    "domain12",
    "publication_year",
    "gear_evidence_score",
    "mechanism_validity",
    "antecedent_risk",
    "score_decile",
    "graph_expected_diffusion",
    "future_structural_outcome",
}


def run_three_arm_experiment(
    frame: pd.DataFrame,
    *,
    seed: int = 20260827,
    epsilon: float = 0.1,
    minimum_rows: int = 100,
    minimum_deciles: int = 8,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the registered arms or fail closed when identification is absent."""
    missing = sorted(REQUIRED_COLUMNS - set(frame))
    if missing:
        return pd.DataFrame(), _blocked(f"missing_columns:{','.join(missing)}")
    data = frame.dropna(subset=list(REQUIRED_COLUMNS)).copy()
    if len(data) < minimum_rows:
        return pd.DataFrame(), _blocked(
            f"eligible_rows_below_minimum:{len(data)}<{minimum_rows}"
        )
    deciles = pd.to_numeric(data["score_decile"], errors="coerce")
    valid_deciles = deciles.notna() & deciles.between(0, 9) & deciles.mod(1).eq(0)
    if not bool(valid_deciles.all()):
        return pd.DataFrame(), _blocked("invalid_graph_percentile_deciles")
    data["score_decile"] = deciles.astype(int)
    if data["score_decile"].nunique() < minimum_deciles:
        return pd.DataFrame(), _blocked(
            "graph_score_range_restricted:"
            f"{data['score_decile'].nunique()}<{minimum_deciles}"
        )
    shuffled = _shuffle_within_field_year(data, seed=seed)
    gate = pd.to_numeric(data["gear_evidence_score"]).clip(0.0, 1.0)
    mechanism = pd.to_numeric(data["mechanism_validity"]).clip(0.0, 1.0)
    antecedent = pd.to_numeric(data["antecedent_risk"]).clip(0.0, 1.0)
    gate = gate * (1.0 - antecedent)
    evidence_only = gate * np.sqrt(mechanism)
    joint = evidence_only * _diffusion_factor(
        pd.to_numeric(data["graph_expected_diffusion"]), epsilon
    )
    placebo = evidence_only * _diffusion_factor(shuffled, epsilon)
    direct = antecedent >= 0.999
    evidence_only.loc[direct] = 0.0
    joint.loc[direct] = 0.0
    placebo.loc[direct] = 0.0
    arms = data[
        ["paper_id", "domain12", "publication_year", "future_structural_outcome"]
    ].copy()
    arms["gear_evidence_score"] = evidence_only
    arms["joint_structural_score"] = joint
    arms["shuffled_structural_score"] = placebo
    arms["graph_expected_diffusion"] = data["graph_expected_diffusion"]
    arms["shuffled_graph_expected_diffusion"] = shuffled
    real = evaluate_joint_value(arms)
    shuffled_eval = evaluate_joint_value(
        arms.rename(
            columns={
                "joint_structural_score": "real_joint",
                "shuffled_structural_score": "joint_structural_score",
            }
        )
    )
    return arms, {
        "status": "estimated",
        "eligible_rows": len(arms),
        "score_deciles": int(data["score_decile"].nunique()),
        "real_hgb": real,
        "shuffled_hgb": shuffled_eval,
        "real_minus_shuffled_value": real["joint_spearman"]
        - shuffled_eval["joint_spearman"],
    }


def _shuffle_within_field_year(frame: pd.DataFrame, *, seed: int) -> pd.Series:
    generator = np.random.default_rng(seed)
    year_bin = (pd.to_numeric(frame["publication_year"]) // 5 * 5).astype(int)
    values = pd.to_numeric(frame["graph_expected_diffusion"]).copy()
    output = values.copy()
    strata = pd.DataFrame(
        {"domain12": frame["domain12"].astype(str), "year_bin": year_bin},
        index=frame.index,
    )
    for indexes in strata.groupby(["domain12", "year_bin"]).groups.values():
        indexes = list(indexes)
        output.loc[indexes] = generator.permutation(values.loc[indexes].to_numpy())
    if output.equals(values):
        output.loc[:] = generator.permutation(values.to_numpy())
    return output


def _diffusion_factor(values: pd.Series, epsilon: float) -> pd.Series:
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be in (0, 1)")
    return epsilon + (1.0 - epsilon) * values.clip(0.0, 1.0)


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "status": "not_identifiable",
        "reason": reason,
        "claim_allowed": False,
    }


__all__ = ["run_three_arm_experiment"]

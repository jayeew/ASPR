"""Construct auditable HGB-P components from future-graph summary statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd

RAW_PERTURBATION_COLUMNS: tuple[str, ...] = (
    "future_new_community_count",
    "total_future_community_count",
    "outsider_citer_share",
    "pre_cross_community_edge_rate",
    "post_cross_community_edge_rate",
    "focal_only_citers",
    "focal_and_predecessor_citers",
    "pre_shortest_path",
    "post_shortest_path",
    "claim_adoption_breadth",
)


def build_perturbation_components(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the four registered P dimensions without cross-paper fitting."""
    missing = sorted(set(RAW_PERTURBATION_COLUMNS) - set(frame))
    if missing:
        raise ValueError(f"raw perturbation inputs are missing: {missing}")
    numeric = frame[list(RAW_PERTURBATION_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    )
    new_share = _safe_ratio(
        numeric["future_new_community_count"],
        numeric["total_future_community_count"],
    )
    boundary = _mean_complete(new_share, numeric["outsider_citer_share"])
    mixing = (
        numeric["post_cross_community_edge_rate"]
        - numeric["pre_cross_community_edge_rate"]
    ).clip(lower=0.0, upper=1.0)
    displacement = _safe_ratio(
        numeric["focal_only_citers"],
        numeric["focal_only_citers"] + numeric["focal_and_predecessor_citers"],
    )
    path_gain = _safe_ratio(
        (numeric["pre_shortest_path"] - numeric["post_shortest_path"]).clip(lower=0.0),
        numeric["pre_shortest_path"],
    )
    adoption = numeric["claim_adoption_breadth"].clip(lower=0.0, upper=1.0)
    path_adoption = _mean_complete(path_gain, adoption)
    output = pd.DataFrame(
        {
            "boundary_expansion": boundary,
            "community_mixing_change": mixing,
            "dependency_displacement": displacement,
            "path_shortening": path_gain,
            "claim_adoption_breadth": adoption,
            "path_shortening_claim_adoption": path_adoption,
            "perturbation_component_scope": "future_graph_audit_summary",
        },
        index=frame.index,
    )
    return output


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.replace(0.0, np.nan)
    return result.clip(lower=0.0, upper=1.0)


def _mean_complete(first: pd.Series, second: pd.Series) -> pd.Series:
    output = (first + second) / 2.0
    return output.where(first.notna() & second.notna())


__all__ = ["RAW_PERTURBATION_COLUMNS", "build_perturbation_components"]

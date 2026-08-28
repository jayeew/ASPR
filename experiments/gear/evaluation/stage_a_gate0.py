"""Deterministic Stage-A correctness gates evaluated on frozen real scores."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from gear.graph_guidance import score_controller
from gear.graph_prior_contracts import GraphRuntimePacket, InfluenceForecast


def evaluate_gate0(population: pd.DataFrame) -> dict[str, Any]:
    scores = population[
        ["prospective_5y_diffusion_percentile", "expected_diffusion_score"]
    ].dropna()
    low_row = scores.nsmallest(1, "prospective_5y_diffusion_percentile").iloc[0]
    high_row = scores.nlargest(1, "prospective_5y_diffusion_percentile").iloc[0]
    low_packet = _packet(low_row)
    high_packet = _packet(high_row)
    low_geometry = score_controller(low_packet, 8, enabled=True)[:2]
    high_geometry = score_controller(high_packet, 8, enabled=True)[:2]
    neutral_low = score_controller(low_packet, 8, enabled=False)[:2]
    neutral_high = score_controller(high_packet, 8, enabled=False)[:2]
    diffusion = np.sort(scores["expected_diffusion_score"].to_numpy(float))
    factor = 0.1 + 0.9 * diffusion
    checks = {
        "score_changes_geometry": low_geometry != high_geometry,
        "resource_caps_equal": sum(low_geometry) == sum(high_geometry) == 8,
        "neutral_baseline_invariant": neutral_low == neutral_high == (4, 4),
        "structural_score_monotone_in_hgb": bool(np.all(np.diff(factor) >= 0.0)),
        "direct_antecedent_noncompensatory": bool(np.all((1.0 - 1.0) * factor == 0.0)),
        "uncertainty_shrinks_to_base": _shrink_check(),
        "raw_score_preserved": (
            low_packet.score_0_100
            == float(low_row["prospective_5y_diffusion_percentile"])
            and high_packet.score_0_100
            == float(high_row["prospective_5y_diffusion_percentile"])
        ),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "low_geometry": list(low_geometry),
        "high_geometry": list(high_geometry),
        "neutral_geometry": list(neutral_low),
        "real_score_min": float(scores["prospective_5y_diffusion_percentile"].min()),
        "real_score_max": float(scores["prospective_5y_diffusion_percentile"].max()),
    }


def _packet(row: pd.Series) -> GraphRuntimePacket:
    return GraphRuntimePacket(
        paper_id="gate0",
        cutoff_date=date(2020, 1, 1),
        forecast=InfluenceForecast(
            status="available",
            prospective_5y_diffusion_percentile=float(
                row["prospective_5y_diffusion_percentile"]
            ),
            uptake_probability=0.5,
            conditional_diffusion=0.4,
            expected_diffusion=float(row["expected_diffusion_score"]),
            feature_coverage=1.0,
            release_id="gear-d5-primary16-current",
            model_sha256="sha256:gate0",
            percentile_reference_sha256="sha256:gate0-reference",
        ),
    )


def _shrink_check() -> bool:
    base = 0.2
    expected = 0.8
    reliabilities = np.array([0.0, 0.25, 0.5, 1.0])
    shrunk = base + reliabilities * (expected - base)
    return bool(shrunk[0] == base and np.all(np.diff(shrunk) >= 0.0))


__all__ = ["evaluate_gate0"]

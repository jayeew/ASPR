"""Deterministic scalar ablations for the registered monotone fusion."""

from __future__ import annotations

import pandas as pd


def fusion_curve(
    *,
    evidence_gate: float,
    mechanism_validity: float,
    diffusion_values: list[float],
    perturbation: float | None = None,
    epsilon: float = 0.1,
) -> pd.DataFrame:
    if not 0.0 <= evidence_gate <= 1.0 or not 0.0 <= mechanism_validity <= 1.0:
        raise ValueError("fusion inputs must be in [0, 1]")
    rows = []
    for diffusion in diffusion_values:
        if not 0.0 <= diffusion <= 1.0:
            raise ValueError("diffusion values must be in [0, 1]")
        score = evidence_gate * (epsilon + (1.0 - epsilon) * diffusion)
        if perturbation is not None:
            score *= epsilon + (1.0 - epsilon) * perturbation
        score *= mechanism_validity**0.5
        rows.append({"diffusion_potential": diffusion, "structural_score": score})
    return pd.DataFrame(rows)


__all__ = ["fusion_curve"]

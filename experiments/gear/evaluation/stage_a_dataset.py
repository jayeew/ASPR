"""Build a deterministic score-decile cohort for Stage-A integration tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

OOF_PATH = Path(
    "data/calibration/releases/gear-d5-primary16-current/oof_predictions.parquet"
)
SCORE_PATH = Path(
    "data/calibration/releases/gear-d5-primary16-current/score_table.parquet"
)
NATURE_SOURCE_MANIFEST = Path(
    "outputs/gear/human_review_reconstruction/"
    "nature_dev100_human_v2_20260824/source_manifest.json"
)


def load_stage_a_population(
    oof_path: Path = OOF_PATH,
    score_path: Path = SCORE_PATH,
) -> pd.DataFrame:
    """Load frozen D5 OOF outcomes joined one-to-one to display percentiles."""
    oof = pd.read_parquet(oof_path)
    score = pd.read_parquet(
        score_path,
        columns=[
            "paper_id",
            "prospective_5y_diffusion_percentile",
            "feature_coverage",
        ],
    )
    if oof["paper_id"].duplicated().any() or score["paper_id"].duplicated().any():
        raise ValueError("Stage-A population inputs must be unique at paper grain")
    population = oof.merge(score, on="paper_id", how="left", validate="one_to_one")
    population["score_decile"] = np.clip(
        np.floor(population["prospective_5y_diffusion_percentile"] / 10.0),
        0,
        9,
    ).astype("Int64")
    population["stable_key"] = population["paper_id"].map(_stable_key)
    return population


def build_score_stratified_cohort(
    population: pd.DataFrame,
    *,
    per_decile: int = 20,
    evidence_inventory: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Select a domain-diverse real-paper cohort without reading outcomes."""
    if per_decile < 1:
        raise ValueError("per_decile must be positive")
    eligible = population.dropna(
        subset=["score_decile", "prospective_5y_diffusion_percentile"]
    ).copy()
    eligible["domain_rank"] = eligible.groupby(
        ["score_decile", "domain12"], observed=True
    )["stable_key"].rank(method="first")
    selected = (
        eligible.sort_values(["score_decile", "domain_rank", "domain12", "stable_key"])
        .groupby("score_decile", observed=True, group_keys=False)
        .head(per_decile)
        .copy()
    )
    if selected["score_decile"].nunique() != 10:
        raise ValueError("frozen population does not cover all ten score deciles")
    selected = selected.drop(columns=["domain_rank"])
    selected["manuscript_path"] = pd.NA
    selected["review_history_path"] = pd.NA
    selected["gear_evidence_available"] = False
    if evidence_inventory is not None and not evidence_inventory.empty:
        evidence = evidence_inventory[
            ["paper_id", "source_paper_path", "source_review_history_path"]
        ].drop_duplicates("paper_id")
        selected = selected.drop(
            columns=[
                "manuscript_path",
                "review_history_path",
                "gear_evidence_available",
            ]
        ).merge(evidence, on="paper_id", how="left", validate="one_to_one")
        selected = selected.rename(
            columns={
                "source_paper_path": "manuscript_path",
                "source_review_history_path": "review_history_path",
            }
        )
        selected["gear_evidence_available"] = selected["manuscript_path"].notna()
    selected["integration_eligible"] = (
        selected["gear_evidence_available"]
        & selected["realized_diffusion_target"].notna()
    )
    selected["evidence_status"] = np.where(
        selected["integration_eligible"],
        "ready",
        "manuscript_and_gear_evidence_required",
    )
    return selected.sort_values(["score_decile", "domain12", "stable_key"]).reset_index(
        drop=True
    )


def load_evidence_inventory(path: Path = NATURE_SOURCE_MANIFEST) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("evidence inventory must contain a list")
    return pd.DataFrame(payload)


def cohort_quality(cohort: pd.DataFrame) -> dict[str, Any]:
    counts = cohort["score_decile"].value_counts().sort_index()
    return {
        "rows": len(cohort),
        "score_deciles_covered": int(cohort["score_decile"].nunique()),
        "minimum_per_decile": int(counts.min()) if len(counts) else 0,
        "domains_covered": int(cohort["domain12"].nunique()),
        "publication_year_min": int(cohort["publication_year"].min()),
        "publication_year_max": int(cohort["publication_year"].max()),
        "gear_evidence_available": int(cohort["gear_evidence_available"].sum()),
        "integration_eligible": int(cohort["integration_eligible"].sum()),
    }


def _stable_key(paper_id: str) -> str:
    return hashlib.sha256(f"stage-a-v1|{paper_id}".encode()).hexdigest()


__all__ = [
    "build_score_stratified_cohort",
    "cohort_quality",
    "load_evidence_inventory",
    "load_stage_a_population",
]

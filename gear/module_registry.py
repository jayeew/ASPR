"""Stable module names, exchange contracts, and canonical artifact locations."""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

from .config import PROJECT_ROOT


class ModuleSpec(NamedTuple):
    artifact: str
    primary_file: str
    description: str


MODULES: dict[str, ModuleSpec] = {
    "indicator_definition": ModuleSpec(
        "indicator_release", "feature_registry.json", "Frozen indicator definitions"
    ),
    "aspr_scoring": ModuleSpec(
        "score_release", "paper_scores.parquet", "Paper-level ASPR score release"
    ),
    "review_reconstruction": ModuleSpec(
        "reference_reviews",
        "human_structured_reviews.jsonl",
        "Revision-aware human StructuredReview labels",
    ),
    "gear_agent": ModuleSpec(
        "review_run", "agent_structured_reviews.jsonl", "GEAR agent reviews"
    ),
    "review_evaluation": ModuleSpec(
        "agent_human_agreement",
        "corpus_metrics.json",
        "Agent-versus-human comparison metrics",
    ),
}


def artifact_root() -> Path:
    value = os.getenv("ASPR_GEAR_ARTIFACT_ROOT")
    return Path(value).resolve() if value else PROJECT_ROOT / "outputs/gear/artifacts"


def reference_root() -> Path:
    value = os.getenv("ASPR_GEAR_REFERENCE_ROOT")
    return (
        Path(value).resolve() if value else PROJECT_ROOT / "outputs/gear/artifact_refs"
    )


__all__ = ["MODULES", "ModuleSpec", "artifact_root", "reference_root"]

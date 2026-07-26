"""Frozen measurement definitions shared by every new figure."""

from __future__ import annotations

from typing import Dict, Tuple


PRIMARY_FEATURES: Tuple[str, ...] = (
    "reference_overlap_novelty_t0",
    "hypergeom_conventionality_median_t0",
    "first_time_source_pair_share",
    "field_gini_balance",
    "reference_other_field_share",
    "field_variety",
    "field_disparity_cosine_mean",
    "rao_stirling_integration",
)

ANGLE_FEATURES: Dict[str, Tuple[str, ...]] = {
    "A1_COMBINATION_RARITY": ("reference_overlap_novelty_t0",),
    "A2_ATYPICALITY_CONVENTIONALITY": (
        "hypergeom_conventionality_median_t0",
    ),
    "A3_FIRST_TIME_COMBINATION": ("first_time_source_pair_share",),
    "A4_KNOWLEDGE_BREADTH_BALANCE": (
        "field_gini_balance",
        "reference_other_field_share",
        "field_variety",
    ),
    "A5_COGNITIVE_DISTANCE_INTEGRATION": (
        "field_disparity_cosine_mean",
        "rao_stirling_integration",
    ),
}

# Direction is fixed by the measurement interpretation, never by OOF results.
FEATURE_DIRECTION: Dict[str, int] = {
    "reference_overlap_novelty_t0": 1,
    "hypergeom_conventionality_median_t0": -1,
    "first_time_source_pair_share": 1,
    "field_gini_balance": 1,
    "reference_other_field_share": 1,
    "field_variety": 1,
    "field_disparity_cosine_mean": 1,
    "rao_stirling_integration": 1,
}

SUPPORTED_FIGURES: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 9, 10)

STATUS_COMPLETE = "COMPLETE"
STATUS_DESCRIPTIVE = "DESCRIPTIVE_ONLY"
STATUS_DRAFT_LABELS = "DRAFT_HUMAN_LABELS"
STATUS_BLOCKED_COMPARABILITY = "BLOCKED_COMPARABILITY"
STATUS_BLOCKED_MODEL = "BLOCKED_MISSING_MODEL"


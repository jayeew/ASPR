"""Frozen labels and visual conventions for the selected-case Fig. 1."""

from __future__ import annotations

from typing import Dict, Mapping, Tuple


DOMAIN_LABELS: Mapping[str, str] = {
    "crispr": "CRISPR–Cas genome editing",
    "graphene_2d_materials": "Graphene and 2D materials",
    "ipsc_reprogramming": "iPSC reprogramming",
    "exoplanets": "Exoplanet discovery",
    "microbiome_metagenomics": "Microbiome / metagenomics",
    "click_chemistry_cuaac": "Click chemistry (CuAAC)",
    "genome_wide_association_studies": "Genome-wide association studies",
    "mass_spectrometry_techniques_and_applications": (
        "Electrospray mass spectrometry"
    ),
    "organoids": "Organoids",
    "topological_insulators": "Topological insulators",
}

FEATURE_LABELS: Mapping[str, str] = {
    "reference_overlap_novelty_t0": "Reference-overlap novelty",
    "hypergeom_conventionality_median_t0": "Low conventionality",
    "first_time_source_pair_share": "First-time source pairs",
    "field_gini_balance": "Field balance (1−Gini)",
    "reference_other_field_share": "Out-of-field references",
    "field_variety": "Field variety",
    "field_disparity_cosine_mean": "Mean cognitive distance",
    "rao_stirling_integration": "Rao–Stirling integration",
}

FEATURE_SHORT_LABELS: Mapping[str, str] = {
    "reference_overlap_novelty_t0": "Overlap novelty",
    "hypergeom_conventionality_median_t0": "Low conventionality",
    "first_time_source_pair_share": "First-time pairs",
    "field_gini_balance": "Field balance",
    "reference_other_field_share": "Out-of-field share",
    "field_variety": "Field variety",
    "field_disparity_cosine_mean": "Cognitive distance",
    "rao_stirling_integration": "Rao–Stirling",
}

FEATURE_STYLES: Dict[str, Tuple[str, str, str]] = {
    "reference_overlap_novelty_t0": ("#315F8C", "o", "-"),
    "hypergeom_conventionality_median_t0": ("#C67722", "s", "--"),
    "first_time_source_pair_share": ("#B44A6A", "^", "-."),
    "field_gini_balance": ("#708238", "D", ":"),
    "reference_other_field_share": ("#247C78", "v", "-"),
    "field_variety": ("#7B5AA6", "P", "--"),
    "field_disparity_cosine_mean": ("#A44A3F", "X", "-."),
    "rao_stirling_integration": ("#4E6E81", "h", ":"),
}

STAGE_KEYS: Tuple[str, ...] = (
    "pre",
    "landmark",
    "early_post",
    "late_post",
)

STAGE_LABELS: Mapping[str, str] = {
    "pre": "Pre-landmark",
    "landmark": "Landmark window",
    "early_post": "Early diffusion",
    "late_post": "Later diffusion",
}

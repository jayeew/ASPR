"""Materialize the source-backed publication-time features used by final Fig. 1."""

from __future__ import annotations

import math
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

_TRAINING_MATRIX = import_module(
    "innovation_impact_feature_selection.evidence_derived_v3."
    "experiments.oof_feature_set_comparison_v3.build_training_matrix"
)
descriptor = _TRAINING_MATRIX.descriptor
load_feature_sets = _TRAINING_MATRIX.load_feature_sets
title_features = _TRAINING_MATRIX.title_features


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIG1_DATA = PROJECT_ROOT / "data" / "knowledge_corpus" / "fig1_quasicausal_v1"
OOF_OUTPUTS = (
    PROJECT_ROOT
    / "innovation_impact_feature_selection"
    / "evidence_derived_v3"
    / "experiments"
    / "oof_feature_set_comparison_v3"
    / "outputs"
)

LOCAL_SOURCE_ALIASES: Mapping[str, str] = {
    "EF0307": "publication_year",
    "EF0309": "rao_stirling_integration",
    "EF0312": "field_gini_balance",
    "EF0314": "valid_reference_count",
    "EF0315": "field_disparity_cosine_mean",
    "EF0318": "field_variety",
}

DISPLAY_EXCLUDED_SOURCES = {
    "article_type_constant",
    "english_language_constant",
    "field_prior_volume",
    "publication_year",
    "venue_prior_volume",
}

PREFERRED_SOURCE_REPRESENTATIVES: Mapping[str, str] = {
    "field_div_index": "EF0117",
    "field_mapping_coverage": "EF0311",
    "reference_overlap_novelty_t0": "EF0340",
    "uzzi_conventionality_median_t0": "EF0211",
}

TIER_PRIORITY: Mapping[str, int] = {
    "source_formula_existing": 0,
    "source_formula_local_surrogate": 1,
    "structured_construct_proxy": 2,
    "title_taxonomy_lexical_proxy": 3,
}


def selected_cases(panel_data: Path) -> pd.DataFrame:
    """Load the four frozen display cases in deterministic order."""
    frame = pd.read_csv(panel_data / "domain_selection.csv")
    return (
        frame.loc[frame["selected"].astype(bool)]
        .sort_values("selection_rank", kind="stable")
        .reset_index(drop=True)
    )


def _base_frame() -> pd.DataFrame:
    features = pd.read_parquet(FIG1_DATA / "indicator_features_v6_1.parquet")
    works = pd.read_parquet(
        FIG1_DATA / "focal_works.parquet",
        columns=[
            "work_id",
            "title",
            "publication_year",
            "domain",
            "primary_topic_name",
            "primary_subfield_name",
            "primary_field_name",
            "venue_source_id",
        ],
    )
    works = (
        works.sort_values(["work_id", "domain"], kind="stable")
        .drop_duplicates(["work_id", "domain"])
    )
    frame = features.merge(
        works,
        left_on=["paper_id", "publication_year", "domain"],
        right_on=["work_id", "publication_year", "domain"],
        how="left",
        validate="one_to_one",
    )
    if frame["title"].isna().any():
        raise ValueError("Fig. 1 corpus contains missing titles")
    return frame.sort_values(
        ["publication_year", "paper_id"], kind="stable"
    ).reset_index(drop=True)


def _add_prior_volume(frame: pd.DataFrame, key: str) -> np.ndarray:
    counts: Dict[str, int] = {}
    values = np.zeros(len(frame), dtype=np.float32)
    years = frame["publication_year"].astype(int).to_numpy()
    keys = frame[key].astype("string").fillna("missing").astype(str).to_numpy()
    for year in sorted(set(years)):
        positions = np.flatnonzero(years == year)
        for position in positions:
            values[position] = math.log1p(counts.get(keys[position], 0))
        for position in positions:
            counts[keys[position]] = counts.get(keys[position], 0) + 1
    return values


def _augment_frame(frame: pd.DataFrame) -> pd.DataFrame:
    titles = title_features(
        frame[["paper_id", "publication_year", "title"]].copy()
    )
    for column in titles.columns[1:]:
        frame[column] = titles[column].to_numpy()
    frame["venue_prior_volume"] = _add_prior_volume(frame, "venue_source_id")
    frame["field_prior_volume"] = _add_prior_volume(
        frame, "primary_subfield_name"
    )
    frame["additive_entropy_diversity_local"] = (
        pd.to_numeric(frame["field_shannon_entropy"], errors="coerce")
        + pd.to_numeric(frame["field_pielou_evenness"], errors="coerce")
        - pd.to_numeric(
            frame["field_disparity_cosine_mean"], errors="coerce"
        )
    )
    frame["article_type_constant"] = 1.0
    frame["english_language_constant"] = 1.0
    return frame


def _lexical_values(
    frame: pd.DataFrame,
    feature_ids: Sequence[str],
    library: Mapping[str, Mapping[str, str]],
    feature_to_dimension: Mapping[str, str],
    dimensions: Mapping[str, Mapping[str, str]],
) -> np.ndarray:
    if not feature_ids:
        return np.empty((len(frame), 0), dtype=np.float32)
    descriptions = [
        descriptor(feature_id, library, feature_to_dimension, dimensions)
        for feature_id in feature_ids
    ]
    texts = (
        frame[
            [
                "title",
                "primary_field_name",
                "primary_subfield_name",
                "primary_topic_name",
                "domain",
            ]
        ]
        .astype("string")
        .fillna("")
        .agg(" ".join, axis=1)
        .tolist()
    )
    word = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        norm="l2",
    )
    char = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=32_768,
        sublinear_tf=True,
        norm="l2",
    )
    indicator_word = word.fit_transform(descriptions)
    indicator_char = char.fit_transform(descriptions)
    values = np.empty((len(frame), len(feature_ids)), dtype=np.float32)
    for start in range(0, len(frame), 5_000):
        stop = min(start + 5_000, len(frame))
        word_score = word.transform(texts[start:stop]) @ indicator_word.T
        char_score = char.transform(texts[start:stop]) @ indicator_char.T
        values[start:stop] = (
            0.7 * word_score.toarray() + 0.3 * char_score.toarray()
        ).astype(np.float32)
    return values


def materialize_values() -> Tuple[
    pd.DataFrame,
    Dict[str, np.ndarray],
    pd.DataFrame,
    Mapping[str, Mapping[str, str]],
    Mapping[str, str],
    Mapping[str, Mapping[str, str]],
]:
    """Materialize all source-set values available in the local Fig. 1 corpus."""
    feature_sets, library, feature_to_dimension, dimensions = load_feature_sets()
    source_ids = feature_sets["source_154"]
    audit = pd.read_csv(OOF_OUTPUTS / "operationalization_audit.csv")
    audit = audit.loc[audit["feature_id"].isin(source_ids)].copy()
    audit_lookup = audit.set_index("feature_id")
    frame = _augment_frame(_base_frame())
    lexical_ids = [
        feature_id
        for feature_id in source_ids
        if str(audit_lookup.loc[feature_id, "source_column"])
        == "lexical_similarity"
    ]
    lexical = _lexical_values(
        frame,
        lexical_ids,
        library,
        feature_to_dimension,
        dimensions,
    )
    values: Dict[str, np.ndarray] = {
        feature_id: lexical[:, index]
        for index, feature_id in enumerate(lexical_ids)
    }
    rows: List[Dict[str, Any]] = []
    for feature_id in source_ids:
        row = audit_lookup.loc[feature_id]
        source_column = str(row["source_column"])
        local_column = LOCAL_SOURCE_ALIASES.get(source_column, source_column)
        if feature_id in values:
            status = "materialized"
            reason = ""
        elif local_column in frame.columns:
            values[feature_id] = pd.to_numeric(
                frame[local_column], errors="coerce"
            ).to_numpy()
            status = "materialized"
            reason = ""
        else:
            status = "unavailable_in_balanced_fig1_corpus"
            reason = f"missing local source column: {local_column}"
        rows.append(
            {
                "feature_id": feature_id,
                "canonical_name_en": library[feature_id]["canonical_name_en"],
                "dimension_id": feature_to_dimension[feature_id],
                "dimension_label": dimensions[
                    feature_to_dimension[feature_id]
                ]["label"],
                "tier": str(row["tier"]),
                "registered_source_column": source_column,
                "local_source_column": local_column,
                "materialization_status": status,
                "materialization_reason": reason,
                "main_display_eligible_tier": bool(
                    str(row["tier"]) != "title_taxonomy_lexical_proxy"
                ),
            }
        )
    return (
        frame,
        values,
        pd.DataFrame(rows),
        library,
        feature_to_dimension,
        dimensions,
    )


__all__ = [
    "DISPLAY_EXCLUDED_SOURCES",
    "PREFERRED_SOURCE_REPRESENTATIVES",
    "TIER_PRIORITY",
    "materialize_values",
    "selected_cases",
]

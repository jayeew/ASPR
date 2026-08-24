"""Tests for the recovered evidence-v3 OOF matrix contract."""

from __future__ import annotations

from innovation_impact_feature_selection.evidence_derived_v3.experiments.oof_feature_set_comparison_v3.build_training_matrix import (
    descriptor,
)

DIMENSIONS = {
    "CD001": {
        "label": "Example dimension",
        "definition": "Example definition.",
    }
}
FEATURE_TO_DIMENSION = {"EF0001": "CD001"}


def test_descriptor_accepts_full_library_schema() -> None:
    library = {
        "EF0001": {
            "canonical_name_en": "Example indicator",
            "alias_names_json": '["Example alias"]',
            "required_data_json": '["publication year"]',
            "formula": "x / y",
        }
    }

    value = descriptor("EF0001", library, FEATURE_TO_DIMENSION, DIMENSIONS)

    assert value == (
        "Example indicator Example alias Example dimension "
        "Example definition. x / y publication year"
    )


def test_descriptor_accepts_recovery_library_schema() -> None:
    library = {
        "EF0001": {
            "canonical_name_en": "Recovered indicator",
            "formula_text": "recovered formula",
        }
    }

    value = descriptor("EF0001", library, FEATURE_TO_DIMENSION, DIMENSIONS)

    assert value == (
        "Recovered indicator  Example dimension Example definition. "
        "recovered formula "
    )

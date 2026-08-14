"""Dimension-safe registry for the four frozen Graph feature families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class FeatureSetSpec:
    name: str
    dimension: int
    public_primary: bool = False

    def validate_vector(self, values: Sequence[object]) -> None:
        if len(values) != self.dimension:
            raise ValueError(
                f"{self.name} requires {self.dimension} features; received {len(values)}"
            )

    def validate_model(self, model_feature_count: int) -> None:
        if model_feature_count != self.dimension:
            raise ValueError(
                f"model dimension {model_feature_count} does not match "
                f"{self.name} dimension {self.dimension}"
            )


FEATURE_SET_SPECS: Mapping[str, FeatureSetSpec] = {
    "strict_7": FeatureSetSpec("strict_7", 7),
    "fulltext_16": FeatureSetSpec("fulltext_16", 16, public_primary=True),
    "source_154": FeatureSetSpec("source_154", 154),
    "ultrarelaxed_221": FeatureSetSpec("ultrarelaxed_221", 221),
}


def feature_set_spec(name: str) -> FeatureSetSpec:
    try:
        return FEATURE_SET_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"unknown Graph feature set: {name}") from exc


def validate_model_assignment(
    feature_set: str,
    values: Sequence[object],
    *,
    model_feature_count: int,
) -> None:
    spec = feature_set_spec(feature_set)
    spec.validate_vector(values)
    spec.validate_model(model_feature_count)


__all__ = [
    "FEATURE_SET_SPECS",
    "FeatureSetSpec",
    "feature_set_spec",
    "validate_model_assignment",
]

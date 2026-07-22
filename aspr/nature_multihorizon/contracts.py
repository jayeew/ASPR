"""Typed contracts for the Nature Portfolio multi-horizon pipeline.

The models in this module are deliberately independent from pandas and the
pipeline implementation.  They are the stable boundary shared by data
building, model training, scoring, and figure export.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from typing import Dict, List, Mapping, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SCHEMA_VERSION = "1.0.0"

CORE_FEATURES: Tuple[str, ...] = (
    "delta_q0_shock",
    "rtd_simpson",
    "field_log_variety",
    "field_evenness",
    "field_disparity",
    "pair_atypicality_tail",
    "pair_conventionality_median",
    "burt_efficiency",
)

AUXILIARY_FEATURES: Tuple[str, ...] = (
    "log_reference_count",
    "reference_age_median",
    "reference_age_iqr",
    "recent_reference_share_5y",
    "classic_reference_share_20y",
    "prior_graph_degree_median",
    "prior_graph_degree_p90",
    "prior_obscure_reference_share",
    "prior_component_size_log",
    "reference_induced_density",
)

MECHANISM_FEATURES: Dict[str, Tuple[str, ...]] = {
    "boundary_perturbation": ("delta_q0_shock",),
    "community_diffusion": ("rtd_simpson",),
    "interdisciplinarity": (
        "field_log_variety",
        "field_evenness",
        "field_disparity",
    ),
    "knowledge_recombination": (
        "pair_atypicality_tail",
        "pair_conventionality_median",
    ),
    "knowledge_brokerage": ("burt_efficiency",),
}

DEFAULT_DENYLIST: Tuple[str, ...] = (
    "cited_by_count",
    "n_future_citers",
    "future_field_reach",
    "future_subfield_reach",
    "future_topic_reach",
    "future_field_simpson",
    "future_topic_simpson",
    "rgpm",
)


class FrozenContract(BaseModel):
    """Base model for immutable, strict pipeline contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=True)


class ReleaseChannel(str, Enum):
    """Publication state for an evidence release."""

    CANDIDATE = "candidate"
    FROZEN = "frozen"


class StageStatus(str, Enum):
    """Completion state recorded in manifests."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class HorizonSpec(FrozenContract):
    """Configuration for one independently trained prediction horizon."""

    tau: int = Field(gt=0)
    complete_publication_end_year: int = Field(ge=1900, le=2200)
    development_end_year: int = Field(ge=1900, le=2200)
    sealed_test_start_year: int = Field(ge=1900, le=2200)
    sealed_test_end_year: int = Field(ge=1900, le=2200)
    min_future_citers: int = Field(default=10, ge=1)
    target_name: str = Field(min_length=1)
    incremental_source: Optional[str] = None

    @model_validator(mode="after")
    def validate_years(self) -> "HorizonSpec":
        """Require a non-overlapping development and sealed test period."""
        if self.development_end_year >= self.sealed_test_start_year:
            raise ValueError("development_end_year must precede sealed_test_start_year")
        if self.sealed_test_start_year > self.sealed_test_end_year:
            raise ValueError("sealed test years must be ordered")
        if self.sealed_test_end_year > self.complete_publication_end_year:
            raise ValueError("sealed test cannot exceed complete publication coverage")
        return self


class FeatureSpec(FrozenContract):
    """Registry of scientific, auxiliary, calibration, and forbidden fields."""

    definition_version: str = "nature-multihorizon-feature-v1"
    core_features: Tuple[str, ...] = CORE_FEATURES
    mechanisms: Dict[str, Tuple[str, ...]] = Field(
        default_factory=lambda: dict(MECHANISM_FEATURES)
    )
    auxiliary_features: Tuple[str, ...] = AUXILIARY_FEATURES
    calibration_features: Tuple[str, ...] = ("domain12", "publication_year")
    denylist: Tuple[str, ...] = DEFAULT_DENYLIST
    require_strictly_prior_year: bool = True

    @model_validator(mode="after")
    def validate_registry(self) -> "FeatureSpec":
        """Reject duplicates and mechanisms that reference unknown features."""
        if len(set(self.core_features)) != len(self.core_features):
            raise ValueError("core_features contains duplicates")
        if len(set(self.auxiliary_features)) != len(self.auxiliary_features):
            raise ValueError("auxiliary_features contains duplicates")
        overlap = set(self.core_features) & set(self.auxiliary_features)
        if overlap:
            raise ValueError(f"core and auxiliary features overlap: {sorted(overlap)}")
        if len(self.mechanisms) != 5:
            raise ValueError("exactly five mechanism channels are required")
        members = [item for values in self.mechanisms.values() for item in values]
        if set(members) != set(self.core_features) or len(members) != len(set(members)):
            raise ValueError("mechanism channels must partition the core features")
        forbidden = {item.casefold() for item in self.denylist}
        selected = {
            item.casefold()
            for item in self.core_features
            + self.auxiliary_features
            + self.calibration_features
        }
        collision = selected & forbidden
        if collision:
            raise ValueError(f"selected features violate denylist: {sorted(collision)}")
        return self

    @property
    def prediction_features(self) -> Tuple[str, ...]:
        """Return the 18 scientific and bibliographic prediction variables."""
        return self.core_features + self.auxiliary_features


class TargetSpec(FrozenContract):
    """Definition of one diffusion target and its optional structure validation."""

    horizon: int = Field(gt=0)
    target_name: str = Field(min_length=1)
    definition_version: str = "nature-multihorizon-target-v1"
    breadth_components: Tuple[str, ...] = (
        "future_field_reach",
        "future_subfield_reach",
        "future_topic_reach",
    )
    evenness_components: Tuple[str, ...] = (
        "future_field_simpson",
        "future_topic_simpson",
    )
    breadth_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    evenness_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    adjust_for_future_citer_count: bool = True
    structure_components: Tuple[str, ...] = (
        "modularity_shock",
        "boundary_mixing_change",
        "partition_change",
        "path_shortening",
    )

    @model_validator(mode="after")
    def validate_weights(self) -> "TargetSpec":
        """Require a fixed convex combination for the diffusion target."""
        if abs((self.breadth_weight + self.evenness_weight) - 1.0) > 1e-12:
            raise ValueError("breadth_weight and evenness_weight must sum to one")
        if not self.breadth_components or not self.evenness_components:
            raise ValueError("breadth and evenness components cannot be empty")
        return self


class CohortSpec(FrozenContract):
    """Reliability and observability requirements for a modeling cohort."""

    horizons: Tuple[int, ...] = (3, 5, 8)
    primary_horizon: int = 5
    min_future_citers: int = Field(default=10, ge=1)
    min_valid_references: int = Field(default=10, ge=1)
    min_reference_metadata_coverage: float = Field(default=0.60, ge=0.0, le=1.0)
    high_quality_reference_coverage: float = Field(default=0.80, ge=0.0, le=1.0)
    require_future_fetch_success: bool = True
    allowed_work_types: Tuple[str, ...] = ("article",)

    @model_validator(mode="after")
    def validate_cohort(self) -> "CohortSpec":
        """Validate horizons and nested quality thresholds."""
        if not self.horizons or len(self.horizons) != len(set(self.horizons)):
            raise ValueError("horizons must be a non-empty unique sequence")
        if any(horizon <= 0 for horizon in self.horizons):
            raise ValueError("all horizons must be positive")
        if self.primary_horizon not in self.horizons:
            raise ValueError("primary_horizon must be included in horizons")
        if self.high_quality_reference_coverage < self.min_reference_metadata_coverage:
            raise ValueError("high-quality coverage cannot be below the main threshold")
        return self


class SplitSpec(FrozenContract):
    """Nested-CV, stratification, holdout, and uncertainty settings."""

    outer_folds: int = Field(default=5, ge=2)
    inner_folds: int = Field(default=4, ge=2)
    seed: int = 20260710
    stratification_fields: Tuple[str, ...] = (
        "domain12",
        "publication_year_bin",
        "venue_family",
    )
    year_bin_width: int = Field(default=5, ge=1)
    min_conditional_cell_size: int = Field(default=30, ge=2)
    min_domain_oof_size: int = Field(default=50, ge=2)
    bootstrap_iterations: int = Field(default=2000, ge=100)
    sealed_holdout_years: Dict[int, Tuple[int, int]] = Field(
        default_factory=lambda: {
            3: (2019, 2022),
            5: (2017, 2020),
            8: (2014, 2017),
        }
    )

    @field_validator("sealed_holdout_years")
    @classmethod
    def validate_holdouts(
        cls, value: Mapping[int, Tuple[int, int]]
    ) -> Dict[int, Tuple[int, int]]:
        """Validate each horizon's inclusive holdout interval."""
        normalized = {int(key): tuple(years) for key, years in value.items()}
        for horizon, years in normalized.items():
            if horizon <= 0 or len(years) != 2 or years[0] > years[1]:
                raise ValueError("invalid sealed holdout interval")
        return normalized


class ScorePacket(FrozenContract):
    """Versioned dual-score response consumed by figures and online scoring."""

    paper_id: str = Field(min_length=1)
    horizon: int = Field(gt=0)
    mechanism_channels: Dict[str, float]
    score_mechanism: Optional[float] = None
    score_performance_raw: Optional[float] = None
    score_performance_calibrated: Optional[float] = None
    score_performance_percentile: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    model_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    quality_flags: Tuple[str, ...] = ()
    claim_scope: str = "Nature Portfolio; pre-publication-year graph"

    @field_validator(
        "score_mechanism",
        "score_performance_raw",
        "score_performance_calibrated",
        "score_performance_percentile",
    )
    @classmethod
    def validate_finite_score(cls, value: Optional[float]) -> Optional[float]:
        """Keep unavailable scores as null, never as NaN or infinity."""
        if value is not None and not isfinite(value):
            raise ValueError("scores must be finite or null")
        return value

    @field_validator("mechanism_channels")
    @classmethod
    def validate_channels(cls, value: Mapping[str, float]) -> Dict[str, float]:
        """Require finite values for every supplied mechanism channel."""
        normalized = {str(key): float(score) for key, score in value.items()}
        expected = set(MECHANISM_FEATURES)
        if set(normalized) != expected:
            missing = sorted(expected - set(normalized))
            extra = sorted(set(normalized) - expected)
            raise ValueError(
                f"mechanism channels must contain exactly the locked five; missing={missing}, extra={extra}"
            )
        if any(not isfinite(score) for score in normalized.values()):
            raise ValueError("mechanism channel scores must be finite")
        return normalized


class ArtifactRecord(FrozenContract):
    """One immutable file referenced by a stage or release manifest."""

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    row_count: Optional[int] = Field(default=None, ge=0)
    primary_key: Tuple[str, ...] = ()

    @field_validator("path")
    @classmethod
    def require_relative_path(cls, value: str) -> str:
        """Prevent absolute and parent-traversal paths in manifests."""
        normalized = value.replace("\\", "/")
        parts = tuple(part for part in normalized.split("/") if part not in {"", "."})
        if value.startswith(("/", "\\")) or ".." in parts:
            raise ValueError("artifact path must be relative and remain inside the release")
        return "/".join(parts)


class StageManifest(FrozenContract):
    """Manifest written when an atomic pipeline stage completes."""

    schema_version: str = SCHEMA_VERSION
    dataset_id: str = Field(min_length=1)
    stage_name: str = Field(min_length=1)
    source_snapshot_id: Optional[str] = None
    config_hash: Optional[str] = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    code_hash: Optional[str] = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    dirty_diff_hash: Optional[str] = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    stage_status: StageStatus = StageStatus.COMPLETE
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    input_artifact_ids: Tuple[str, ...] = ()
    artifacts: Dict[str, ArtifactRecord] = Field(default_factory=dict)
    output_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    row_counts: Dict[str, int] = Field(default_factory=dict)
    primary_keys: Dict[str, Tuple[str, ...]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_complete(self) -> "StageManifest":
        """Only completed stages are publishable immutable artifacts."""
        if self.stage_status is not StageStatus.COMPLETE:
            raise ValueError("persisted stage manifests must be complete")
        return self


class ReleaseManifest(FrozenContract):
    """Immutable evidence-release contract used by all downstream figures."""

    schema_version: str = SCHEMA_VERSION
    source_snapshot_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    analysis_id: str = Field(min_length=1)
    channel: ReleaseChannel
    config_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    code_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    dirty_diff_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    input_artifact_ids: Tuple[str, ...] = ()
    artifacts: Dict[str, ArtifactRecord] = Field(default_factory=dict)
    output_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    row_counts: Dict[str, int] = Field(default_factory=dict)
    primary_keys: Dict[str, Tuple[str, ...]] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    stage_status: StageStatus = StageStatus.COMPLETE

    @model_validator(mode="after")
    def require_complete(self) -> "ReleaseManifest":
        """A published release must represent a successfully completed analysis."""
        if self.stage_status is not StageStatus.COMPLETE:
            raise ValueError("published releases must have complete stage status")
        return self


__all__ = [
    "AUXILIARY_FEATURES",
    "ArtifactRecord",
    "CORE_FEATURES",
    "CohortSpec",
    "DEFAULT_DENYLIST",
    "FeatureSpec",
    "HorizonSpec",
    "MECHANISM_FEATURES",
    "ReleaseChannel",
    "ReleaseManifest",
    "SCHEMA_VERSION",
    "ScorePacket",
    "SplitSpec",
    "StageManifest",
    "StageStatus",
    "TargetSpec",
]

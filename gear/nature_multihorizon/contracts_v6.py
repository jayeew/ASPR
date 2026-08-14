"""Strict v6 contracts for evidence-governed innovation assessment.

These contracts deliberately separate publication-time innovation evidence
from future-influence forecasts.  They coexist with the legacy v1 score
packet so historical releases remain readable without lending old proxy
metrics the names of source-defined constructs.
"""

from __future__ import annotations

from enum import Enum
from math import isfinite
from typing import Dict, Mapping, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


V6_SCHEMA_VERSION = "6.0.0"


class FrozenV6Contract(BaseModel):
    """Immutable strict base class for v6 public contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=True)


class EvidenceClass(str, Enum):
    """Strength of the published evidence supporting a metric."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"


class MetricRole(str, Enum):
    """The only roles a registered variable may play."""

    DIRECT_INNOVATION = "direct_innovation"
    SUPPORTING_CONTEXT = "supporting_context"
    STRUCTURAL_LEADING = "structural_leading"
    PREDICTION_ONLY = "prediction_only"
    CONTROL = "control"
    OUTCOME = "outcome"
    EXCLUDED = "excluded"


class RegistryStatus(str, Enum):
    """Definition-stage eligibility of a dimension or metric."""

    CONFIRMATORY = "confirmatory"
    CANDIDATE_CONFIRMATORY = "candidate_confirmatory"
    CONDITIONAL = "conditional"
    EXPLORATORY = "exploratory"
    REGISTERED = "registered"
    LEGACY_BASELINE = "legacy_baseline"
    EXCLUDED = "excluded"


class ImplementationFidelity(str, Enum):
    """Relationship between code and the cited source definition."""

    EXACT_SOURCE = "exact_source"
    REGISTERED_ADAPTATION = "registered_adaptation"
    PROJECT_PROXY = "project_proxy"
    NOT_IMPLEMENTED = "not_implemented"


class ModelUse(str, Enum):
    """How an admitted metric may enter a predictive model."""

    PRIMARY = "primary"
    PROFILE_ONLY = "profile_only"
    SENSITIVITY_ONLY = "sensitivity_only"
    NOT_USED = "not_used"


class SourceReference(FrozenV6Contract):
    """A verifiable primary or validation source."""

    source_id: str = Field(min_length=1)
    citation: str = Field(min_length=1)
    doi: Optional[str] = None
    url: Optional[str] = None
    source_role: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_locator(self) -> "SourceReference":
        """Require a DOI or stable URL for auditability."""
        if not self.doi and not self.url:
            raise ValueError("each source requires a DOI or stable URL")
        return self


class QualityRule(FrozenV6Contract):
    """Outcome-blind rule that determines whether a metric is reportable."""

    rule_id: str = Field(min_length=1)
    field: str = Field(min_length=1)
    operator: str = Field(pattern=r"^(ge|gt|le|lt|eq|in)$")
    threshold: float | int | str | Tuple[str, ...]
    provenance: str = Field(min_length=1)


class DimensionDefinition(FrozenV6Contract):
    """Source-backed construct and its scientific admission decision."""

    dimension_id: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    label: str = Field(min_length=1)
    construct_definition: str = Field(
        min_length=1, validation_alias="construct", serialization_alias="construct"
    )
    role: MetricRole
    status: RegistryStatus
    foundational_source_ids: Tuple[str, ...] = Field(min_length=1)
    paper_level_source_ids: Tuple[str, ...] = Field(min_length=1)
    distinct_signal_families: Tuple[str, ...] = Field(min_length=1)
    validity_plan: str = Field(min_length=1)
    admission_checks: Dict[str, bool]
    admission_decision: str = Field(min_length=1)

    @model_validator(mode="after")
    def enforce_confirmatory_dimension_rules(self) -> "DimensionDefinition":
        """Encode D1-D7 rather than leaving them as prose-only guidance."""
        expected = {f"D{index}" for index in range(1, 8)}
        if set(self.admission_checks) != expected:
            raise ValueError("dimension admission_checks must contain D1-D7 exactly")
        if self.status in {
            RegistryStatus.CONFIRMATORY,
            RegistryStatus.CANDIDATE_CONFIRMATORY,
        }:
            if not all(self.admission_checks.values()):
                raise ValueError(
                    "confirmatory-candidate dimensions must pass D1-D7"
                )
            if len(self.distinct_signal_families) < 2:
                raise ValueError(
                    "confirmatory-candidate dimensions need two mathematically "
                    "distinct families"
                )
            if not self.foundational_source_ids or not self.paper_level_source_ids:
                raise ValueError(
                    "confirmatory-candidate dimensions need foundational and "
                    "paper-level sources"
                )
        if self.role in {MetricRole.CONTROL, MetricRole.OUTCOME, MetricRole.EXCLUDED}:
            raise ValueError("profile dimensions cannot be controls, outcomes, or excluded")
        return self


class MetricDefinition(FrozenV6Contract):
    """Machine-auditable inclusion/exclusion record for one metric."""

    metric_id: str = Field(pattern=r"^[A-Z][A-Z0-9_.-]*$")
    code_name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    dimension_id: Optional[str] = None
    role: MetricRole
    status: RegistryStatus
    evidence_class: EvidenceClass
    source_ids: Tuple[str, ...] = ()
    formula: str = Field(min_length=1)
    algebraic_family: str = Field(min_length=1)
    model_use: ModelUse
    graph_layer: str = Field(min_length=1)
    cutoff_rule: str = Field(min_length=1)
    source_max_year_policy: str = "source_max_year < publication_year"
    parameters: Dict[str, str | int | float | bool] = Field(default_factory=dict)
    missingness_rule: str = Field(min_length=1)
    structural_undefined_rule: str = Field(min_length=1)
    quality_rules: Tuple[QualityRule, ...] = ()
    fidelity: ImplementationFidelity
    implementation_name: Optional[str] = None
    unit_test_ids: Tuple[str, ...] = ()
    admission_checks: Dict[str, bool]
    disposition_reason: str = Field(min_length=1)
    future_information_allowed: bool = False

    @model_validator(mode="after")
    def enforce_indicator_rules(self) -> "MetricDefinition":
        """Encode the I1-I10 gates and prevent semantic laundering."""
        expected = {f"I{index}" for index in range(1, 11)}
        if set(self.admission_checks) != expected:
            raise ValueError("metric admission_checks must contain I1-I10 exactly")
        included = self.status in {
            RegistryStatus.CONFIRMATORY,
            RegistryStatus.CANDIDATE_CONFIRMATORY,
            RegistryStatus.CONDITIONAL,
            RegistryStatus.EXPLORATORY,
        }
        if included and self.role not in {MetricRole.CONTROL, MetricRole.OUTCOME}:
            if not self.dimension_id:
                raise ValueError("included innovation metrics require a dimension_id")
        if self.status in {
            RegistryStatus.CONFIRMATORY,
            RegistryStatus.CANDIDATE_CONFIRMATORY,
        }:
            if not all(self.admission_checks.values()):
                raise ValueError(
                    "confirmatory-candidate metrics must pass I1-I10"
                )
            if self.evidence_class in {EvidenceClass.D, EvidenceClass.E}:
                raise ValueError("D/E evidence cannot be confirmatory candidates")
            if not self.source_ids:
                raise ValueError(
                    "confirmatory-candidate metrics require a verified source"
                )
            if self.fidelity is not ImplementationFidelity.EXACT_SOURCE:
                raise ValueError(
                    "confirmatory-candidate metrics require exact-source fidelity"
                )
            if not self.implementation_name or not self.unit_test_ids:
                raise ValueError(
                    "confirmatory-candidate metrics require an implementation "
                    "and toy-data tests"
                )
            if not self.quality_rules:
                raise ValueError(
                    "confirmatory-candidate metrics require outcome-blind quality rules"
                )
        if self.fidelity is ImplementationFidelity.PROJECT_PROXY and self.status not in {
            RegistryStatus.EXPLORATORY,
            RegistryStatus.REGISTERED,
            RegistryStatus.LEGACY_BASELINE,
            RegistryStatus.EXCLUDED,
        }:
            raise ValueError("project proxies cannot be confirmatory or conditional")
        if self.future_information_allowed and self.role is not MetricRole.OUTCOME:
            raise ValueError("future information is allowed only for registered outcomes")
        if self.role is MetricRole.EXCLUDED and self.status is not RegistryStatus.EXCLUDED:
            raise ValueError("excluded role and status must agree")
        if self.status is RegistryStatus.EXCLUDED and self.role is not MetricRole.EXCLUDED:
            raise ValueError("excluded status and role must agree")
        if self.status is RegistryStatus.EXCLUDED and all(
            self.admission_checks.values()
        ):
            raise ValueError("excluded metrics must record at least one failed gate")
        if self.model_use is ModelUse.PRIMARY and self.status not in {
            RegistryStatus.CONFIRMATORY,
            RegistryStatus.CANDIDATE_CONFIRMATORY,
            RegistryStatus.CONDITIONAL,
            RegistryStatus.REGISTERED,
        }:
            raise ValueError(
                "primary model inputs must be confirmatory-candidate, conditional, "
                "or registered prediction variables"
            )
        return self


class EvidenceValue(FrozenV6Contract):
    """One publication-time metric value with lineage and quality state."""

    metric_id: str = Field(min_length=1)
    value: Optional[float] = None
    field_year_percentile: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    uncertainty_low: Optional[float] = None
    uncertainty_high: Optional[float] = None
    source_max_year: Optional[int] = None
    artifact_id: str = Field(min_length=1)
    quality_flags: Tuple[str, ...] = ()

    @field_validator("value", "uncertainty_low", "uncertainty_high")
    @classmethod
    def finite_or_null(cls, value: Optional[float]) -> Optional[float]:
        """Represent unavailable values as null rather than NaN or infinity."""
        if value is not None and not isfinite(value):
            raise ValueError("metric values must be finite or null")
        return value

    @model_validator(mode="after")
    def order_interval(self) -> "EvidenceValue":
        """Reject incomplete or reversed uncertainty intervals."""
        bounds = (self.uncertainty_low, self.uncertainty_high)
        if (bounds[0] is None) != (bounds[1] is None):
            raise ValueError("uncertainty bounds must be supplied together")
        if bounds[0] is not None and bounds[0] > bounds[1]:
            raise ValueError("uncertainty interval is reversed")
        return self


class InnovationEvidenceProfile(FrozenV6Contract):
    """Primary v6 output: evidence values without a learned total score."""

    schema_version: str = V6_SCHEMA_VERSION
    paper_id: str = Field(min_length=1)
    publication_year: int = Field(ge=1800, le=2200)
    domain12: str = Field(min_length=1)
    evidence: Dict[str, EvidenceValue]
    registry_version: str = Field(min_length=1)
    claim_scope: str = (
        "publication-time knowledge-base and knowledge-structure evidence; "
        "not total scientific innovation, quality, or acceptance probability"
    )

    @model_validator(mode="after")
    def enforce_profile_integrity(self) -> "InnovationEvidenceProfile":
        """Require key agreement and strictly prior source information."""
        if not self.evidence:
            raise ValueError("innovation evidence profile cannot be empty")
        for metric_id, item in self.evidence.items():
            if metric_id != item.metric_id:
                raise ValueError("evidence mapping key must equal metric_id")
            if item.source_max_year is not None and item.source_max_year >= self.publication_year:
                raise ValueError("source_max_year must be strictly before publication_year")
        return self


class InfluenceForecast(FrozenV6Contract):
    """Secondary v6 output: calibrated future-influence forecast."""

    schema_version: str = V6_SCHEMA_VERSION
    paper_id: str = Field(min_length=1)
    horizon: int = Field(gt=0)
    uptake_probability: float = Field(ge=0.0, le=1.0)
    diffusion_score_if_uptake: Optional[float] = None
    expected_diffusion_score: float
    prediction_interval_low: Optional[float] = None
    prediction_interval_high: Optional[float] = None
    model_version: str = Field(min_length=1)
    calibration_version: str = Field(min_length=1)
    feature_artifact_id: str = Field(min_length=1)
    quality_flags: Tuple[str, ...] = ()
    claim_scope: str = (
        "forecast of future knowledge-graph influence; not an innovation score "
        "or Nature acceptance probability"
    )

    @field_validator(
        "diffusion_score_if_uptake",
        "expected_diffusion_score",
        "prediction_interval_low",
        "prediction_interval_high",
    )
    @classmethod
    def validate_finite(cls, value: Optional[float]) -> Optional[float]:
        """Keep absent conditional forecasts null and reject non-finite values."""
        if value is not None and not isfinite(value):
            raise ValueError("forecast values must be finite or null")
        return value

    @model_validator(mode="after")
    def validate_forecast(self) -> "InfluenceForecast":
        """Validate horizon and prediction interval semantics."""
        if self.horizon not in {3, 5, 8}:
            raise ValueError("v6 forecasts are registered only for D3, D5, and D8")
        bounds = (self.prediction_interval_low, self.prediction_interval_high)
        if (bounds[0] is None) != (bounds[1] is None):
            raise ValueError("prediction interval bounds must be supplied together")
        if bounds[0] is not None and bounds[0] > bounds[1]:
            raise ValueError("prediction interval is reversed")
        return self


__all__ = [
    "DimensionDefinition",
    "EvidenceClass",
    "EvidenceValue",
    "FrozenV6Contract",
    "ImplementationFidelity",
    "InfluenceForecast",
    "InnovationEvidenceProfile",
    "MetricDefinition",
    "MetricRole",
    "ModelUse",
    "QualityRule",
    "RegistryStatus",
    "SourceReference",
    "V6_SCHEMA_VERSION",
]

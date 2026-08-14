"""Strict loader for the ASPR v6.1 candidate-indicator registry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator


V6_1_REGISTRY_SCHEMA = "6.1.0"
EXPECTED_ANGLES = {
    "A1_COMBINATION_RARITY",
    "A2_ATYPICALITY_CONVENTIONALITY",
    "A3_FIRST_TIME_COMBINATION",
    "A4_KNOWLEDGE_BREADTH_BALANCE",
    "A5_COGNITIVE_DISTANCE_INTEGRATION",
}
REQUIRED_GATE_IDS = tuple(f"I{index}" for index in range(1, 11))


class FrozenModel(BaseModel):
    """Immutable strict contract base."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class CandidateSource(FrozenModel):
    """Peer-reviewed source or discovery-only catalog."""

    source_id: str = Field(min_length=1)
    citation: str = Field(min_length=1)
    doi: Optional[str] = None
    url: Optional[str] = None
    peer_reviewed: bool
    source_role: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_locator(self) -> "CandidateSource":
        if not self.doi and not self.url:
            raise ValueError("every source requires a DOI or stable URL")
        return self


class ObservationAngle(FrozenModel):
    """One retained source-backed observation angle."""

    angle_id: str
    label_zh: str = Field(min_length=1)
    meaning: str = Field(min_length=1)
    source_ids: Tuple[str, ...] = Field(min_length=1)
    inclusion_rule: str = Field(min_length=1)
    exclusion_rule: str = Field(min_length=1)


class EmpiricalScreen(FrozenModel):
    """Outcome-blind measurements frozen before OOF."""

    total_n: Optional[int] = Field(default=None, ge=0)
    eligible_n: Optional[int] = Field(default=None, ge=0)
    coverage_denominator_policy: Optional[str] = None
    raw_overall_coverage: Optional[float] = Field(
        default=None, ge=0.0, le=1.0
    )
    raw_minimum_domain_coverage: Optional[float] = Field(
        default=None, ge=0.0, le=1.0
    )
    overall_coverage: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    minimum_domain_coverage: Optional[float] = Field(
        default=None, ge=0.0, le=1.0
    )
    stability_spearman: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    stability_median_relative_error: Optional[float] = Field(
        default=None, ge=0.0
    )
    relative_error_denominator_policy: Optional[str] = None
    relative_error_scale_floor: Optional[float] = Field(
        default=None, ge=0.0
    )
    approximation_spearman: Optional[float] = Field(
        default=None, ge=-1.0, le=1.0
    )
    approximation_median_relative_error: Optional[float] = Field(
        default=None, ge=0.0
    )
    toy_test_pass: bool
    temporal_test_pass: bool
    nondegenerate_test_pass: bool
    screening_artifact_id: Optional[str] = None


class CandidateMetric(FrozenModel):
    """Complete inclusion/exclusion record for one candidate."""

    candidate_id: str = Field(pattern=r"^[A-Z][A-Z0-9_.-]*$")
    code_name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    angle_id: str
    mathematical_family: str = Field(min_length=1)
    knowledge_unit: str = Field(min_length=1)
    formula: str = Field(min_length=1)
    direction: Literal["higher_more", "lower_more", "two_axis"]
    parameters: Dict[str, str | int | float | bool]
    missing_definition: str = Field(min_length=1)
    original_source_ids: Tuple[str, ...] = ()
    paper_application_source_ids: Tuple[str, ...] = ()
    validation_source_ids: Tuple[str, ...] = ()
    required_data: Tuple[str, ...] = Field(min_length=1)
    maximum_information_time: str = Field(min_length=1)
    local_computability: Literal[
        "F0_existing",
        "F1_existing_tables",
        "F2_local_snapshot",
        "F3_external_required",
    ]
    implementation_name: Optional[str] = None
    redundancy_group: str = Field(min_length=1)
    gate_checks: Dict[str, bool]
    empirical_screen: EmpiricalScreen
    final_role: Literal[
        "primary", "sensitivity", "exploratory", "excluded"
    ]
    decision_reason: str = Field(min_length=1)
    oof_used_for_selection: Literal[False] = False

    @model_validator(mode="after")
    def validate_gates(self) -> "CandidateMetric":
        if set(self.gate_checks) != set(REQUIRED_GATE_IDS):
            raise ValueError(
                f"{self.candidate_id} must record I1-I10 exactly"
            )
        if self.final_role in {"primary", "sensitivity"} and (
            not self.original_source_ids
            or not self.paper_application_source_ids
        ):
            raise ValueError(
                f"{self.candidate_id} cannot be {self.final_role} without "
                "both original-formula and paper-application sources"
            )
        if self.final_role == "primary":
            if not all(self.gate_checks.values()):
                raise ValueError(
                    f"primary candidate {self.candidate_id} failed a gate"
                )
            if not self.implementation_name:
                raise ValueError(
                    f"primary candidate {self.candidate_id} is not implemented"
                )
            screen = self.empirical_screen
            required = (
                screen.overall_coverage,
                screen.minimum_domain_coverage,
                screen.stability_spearman,
                screen.stability_median_relative_error,
            )
            if any(value is None for value in required):
                raise ValueError(
                    f"primary candidate {self.candidate_id} lacks screening values"
                )
            if not (
                screen.toy_test_pass
                and screen.temporal_test_pass
                and screen.nondegenerate_test_pass
            ):
                raise ValueError(
                    f"primary candidate {self.candidate_id} failed runtime tests"
                )
        return self


class ScreeningThresholds(FrozenModel):
    """Outcome-blind primary-admission thresholds."""

    overall_coverage_min: float = Field(ge=0.0, le=1.0)
    each_domain_coverage_min: float = Field(ge=0.0, le=1.0)
    stability_spearman_min: float = Field(ge=-1.0, le=1.0)
    stability_median_relative_error_max: float = Field(ge=0.0)
    approximation_spearman_min: float = Field(ge=-1.0, le=1.0)
    approximation_median_relative_error_max: float = Field(ge=0.0)


class CandidateRegistryV61(FrozenModel):
    """The complete v6.1 candidate universe and frozen decisions."""

    schema_version: Literal["6.1.0"]
    registry_version: str = Field(min_length=1)
    registry_stage: Literal[
        "candidate_catalog",
        "posthoc_versioned_extension_frozen_before_oof",
    ]
    literature_search_cutoff: Literal["2026-07-24"]
    search_log_path: str = Field(min_length=1)
    search_log_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    network_policy_for_experiment: Literal["forbidden"]
    raw_data_policy: Literal["local_frozen_only"]
    disclosure: str = Field(min_length=1)
    selection_principle: str = Field(min_length=1)
    thresholds: ScreeningThresholds
    sources: Dict[str, CandidateSource]
    observation_angles: Dict[str, ObservationAngle]
    candidates: Dict[str, CandidateMetric]

    @model_validator(mode="after")
    def validate_registry(self) -> "CandidateRegistryV61":
        if set(self.observation_angles) != EXPECTED_ANGLES:
            raise ValueError("v6.1 registry must contain exactly five angles")
        for key, source in self.sources.items():
            if key != source.source_id:
                raise ValueError("source key differs from source_id")
        for key, angle in self.observation_angles.items():
            if key != angle.angle_id:
                raise ValueError("angle key differs from angle_id")
            missing = set(angle.source_ids) - set(self.sources)
            if missing:
                raise ValueError(f"{key} has unknown sources: {sorted(missing)}")
        code_names = []
        primary_keys = []
        for key, candidate in self.candidates.items():
            if key != candidate.candidate_id:
                raise ValueError("candidate key differs from candidate_id")
            if candidate.angle_id not in self.observation_angles:
                raise ValueError(f"{key} has an unknown angle")
            used_sources = (
                set(candidate.original_source_ids)
                | set(candidate.paper_application_source_ids)
                | set(candidate.validation_source_ids)
            )
            missing = used_sources - set(self.sources)
            if missing:
                raise ValueError(
                    f"{key} has unknown sources: {sorted(missing)}"
                )
            code_names.append(candidate.code_name)
            if candidate.final_role == "primary":
                primary_keys.append(
                    (candidate.angle_id, candidate.mathematical_family)
                )
                self._validate_primary_thresholds(candidate)
        if len(code_names) != len(set(code_names)):
            raise ValueError("candidate code names must be unique")
        duplicates = {
            key for key in primary_keys if primary_keys.count(key) > 1
        }
        if duplicates:
            raise ValueError(
                "only one primary implementation is allowed per family: "
                f"{sorted(duplicates)}"
            )
        if (
            self.registry_stage
            == "posthoc_versioned_extension_frozen_before_oof"
        ):
            active_angles = {
                candidate.angle_id
                for candidate in self.candidates.values()
                if candidate.final_role == "primary"
            }
            if active_angles != EXPECTED_ANGLES:
                raise ValueError(
                    "every retained observation angle needs a primary metric"
                )
        return self

    def _validate_primary_thresholds(
        self, candidate: CandidateMetric
    ) -> None:
        screen = candidate.empirical_screen
        thresholds = self.thresholds
        if float(screen.overall_coverage or 0.0) < thresholds.overall_coverage_min:
            raise ValueError(f"{candidate.candidate_id} fails overall coverage")
        if (
            float(screen.minimum_domain_coverage or 0.0)
            < thresholds.each_domain_coverage_min
        ):
            raise ValueError(f"{candidate.candidate_id} fails domain coverage")
        if (
            float(
                screen.stability_spearman
                if screen.stability_spearman is not None
                else -1.0
            )
            < thresholds.stability_spearman_min
        ):
            raise ValueError(f"{candidate.candidate_id} fails stability rho")
        if (
            float(
                screen.stability_median_relative_error
                if screen.stability_median_relative_error is not None
                else float("inf")
            )
            > thresholds.stability_median_relative_error_max
        ):
            raise ValueError(f"{candidate.candidate_id} fails stability error")
        if screen.approximation_spearman is not None:
            if (
                screen.approximation_spearman
                < thresholds.approximation_spearman_min
            ):
                raise ValueError(
                    f"{candidate.candidate_id} fails approximation rho"
                )
            if (
                float(
                    screen.approximation_median_relative_error
                    if screen.approximation_median_relative_error is not None
                    else float("inf")
                )
                > thresholds.approximation_median_relative_error_max
            ):
                raise ValueError(
                    f"{candidate.candidate_id} fails approximation error"
                )

    @property
    def primary_feature_names(self) -> Tuple[str, ...]:
        """Return primary innovation features in registered order."""
        return tuple(
            candidate.code_name
            for candidate in self.candidates.values()
            if candidate.final_role == "primary"
        )

    @property
    def provisional_core8_names(self) -> Tuple[str, ...]:
        """Return the eight v6 features retained for the B0 comparison."""
        expected_ids = (
            "A1.NOVELTY_U",
            "A2.UZZI_P10",
            "A2.UZZI_MEDIAN",
            "A3.FIRST_SHARE",
            "A4.VARIETY",
            "A4.PIELOU",
            "A5.MEAN_DISTANCE",
            "A5.RAO_STIRLING",
        )
        return tuple(self.candidates[item].code_name for item in expected_ids)


def load_candidate_registry_v6_1(path: Path) -> CandidateRegistryV61:
    """Load and validate a frozen v6.1 registry."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("candidate registry must be a JSON object")
    return CandidateRegistryV61.model_validate(payload)


def candidate_registry_sha256(registry: CandidateRegistryV61) -> str:
    """Return canonical content hash for OOF lineage."""
    payload = registry.model_dump(mode="json")
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def verify_search_log(registry: CandidateRegistryV61, root: Path) -> Path:
    """Verify that the registered literature-search log is unchanged."""
    path = Path(registry.search_log_path)
    path = path if path.is_absolute() else Path(root) / path
    digest = f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"
    if digest != registry.search_log_sha256:
        raise ValueError("literature-search log hash differs from registry")
    return path


__all__ = [
    "CandidateRegistryV61",
    "CandidateMetric",
    "EXPECTED_ANGLES",
    "V6_1_REGISTRY_SCHEMA",
    "candidate_registry_sha256",
    "load_candidate_registry_v6_1",
    "verify_search_log",
]

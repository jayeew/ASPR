"""Public and audit-only contracts for the isolated Graph prior branch."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import Field, model_validator

from .contracts import StrictModel


class InfluenceForecast(StrictModel):
    """A future-diffusion forecast; never evidence for a review judgment."""

    status: Literal["available", "limited", "unavailable"]
    prospective_5y_diffusion_percentile: float | None = Field(
        default=None, ge=0.0, le=100.0
    )
    uptake_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    conditional_diffusion: float | None = Field(default=None, ge=0.0, le=1.0)
    expected_diffusion: float | None = Field(default=None, ge=0.0, le=1.0)
    feature_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    release_id: str | None = None
    model_sha256: str | None = None
    feature_registry_sha256: str | None = None
    training_snapshot_sha256: str | None = None
    percentile_reference_sha256: str | None = None
    diagnostics: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def complete_when_available(self) -> InfluenceForecast:
        required = (
            self.prospective_5y_diffusion_percentile,
            self.uptake_probability,
            self.conditional_diffusion,
            self.expected_diffusion,
            self.release_id,
            self.model_sha256,
            self.percentile_reference_sha256,
        )
        if self.status == "available" and any(value is None for value in required):
            raise ValueError("available forecast lacks values or release provenance")
        return self


class ForecastAnatomy(StrictModel):
    """Non-evidentiary decomposition of a frozen Primary16 HGB forecast."""

    contract: Literal["gear_forecast_anatomy_v1"] = "gear_forecast_anatomy_v1"
    paper_id: str
    target_field: str | None = None
    uptake_percentile: float | None = Field(default=None, ge=0.0, le=100.0)
    conditional_diffusion_percentile: float | None = Field(
        default=None, ge=0.0, le=100.0
    )
    expected_diffusion_percentile: float | None = Field(default=None, ge=0.0, le=100.0)
    uptake_role_contributions: dict[str, float] = Field(default_factory=dict)
    conditional_role_contributions: dict[str, float] = Field(default_factory=dict)
    role_coverage: dict[str, float] = Field(default_factory=dict)
    baseline_id: str | None = None
    feature_input_sha256: str | None = None
    anatomy_release_id: str | None = None
    limited: bool = False

    @model_validator(mode="after")
    def role_sets_are_complete_when_available(self) -> ForecastAnatomy:
        roles = {"substantive_innovation", "t0_potential", "opportunity", "context"}
        if not self.limited:
            if set(self.uptake_role_contributions) != roles:
                raise ValueError("uptake anatomy lacks a Primary16 role")
            if set(self.conditional_role_contributions) != roles:
                raise ValueError("conditional anatomy lacks a Primary16 role")
            if set(self.role_coverage) != roles:
                raise ValueError("anatomy lacks role coverage")
        return self


class CalibrationTension(StrictModel):
    """A process-only safeguard; it is never review evidence."""

    kind: Literal["opportunity_dominant", "integration_dominant"]
    active: bool
    score: float = Field(ge=0.0, le=1.0)
    review_effect: Literal["antecedent_attribution_check", "cross_field_bridge_check"]
    non_evidentiary: Literal[True] = True


class AnalogSeed(StrictModel):
    """A cutoff-safe HGB-conditioned candidate entrance, not evidence."""

    claim_id: str
    work_id: str
    title: str
    lane: Literal["local_adoption", "cross_field_bridge"]
    semantic_score: float = Field(ge=0.0, le=1.0)
    anatomy_score: float = Field(ge=0.0, le=1.0)
    combined_score: float = Field(ge=0.0, le=1.0)
    publication_year: int = Field(ge=1500)
    cutoff_date: date
    source_snapshot_id: str
    source_snapshot_sha256: str
    text_sha256: str | None = None
    text_version: str | None = None
    cutoff_valid: bool
    non_evidentiary: Literal[True] = True

    @model_validator(mode="after")
    def is_conservatively_pre_cutoff(self) -> AnalogSeed:
        if not self.cutoff_valid:
            raise ValueError("analog seed is not cutoff valid")
        if self.publication_year >= self.cutoff_date.year:
            raise ValueError("year-only analog seed is not conservatively pre-cutoff")
        return self


class TopologySeed(StrictModel):
    """Point-in-time candidate entrance; it is not review evidence."""

    work_id: str
    title: str
    publication_date: date | None = None
    publication_year: int | None = Field(default=None, ge=0)
    shared_reference_ids: list[str] = Field(default_factory=list)
    field_ids: list[str] = Field(default_factory=list)
    as_of_date: date
    source_snapshot_id: str
    source_snapshot_sha256: str
    source_max_date: date | None = None
    source_max_year: int | None = Field(default=None, ge=0)
    cutoff_valid: bool
    validation_reasons: list[str] = Field(default_factory=list)

    @property
    def shared_reference_count(self) -> int:
        return len(self.shared_reference_ids)

    @property
    def anchor_field_ids(self) -> list[str]:
        return self.field_ids


class GraphRuntimePacket(StrictModel):
    """The only Graph packet accepted by the review runtime."""

    contract: Literal["gear_graph_runtime_packet"] = "gear_graph_runtime_packet"
    paper_id: str
    cutoff_date: date
    forecast: InfluenceForecast
    forecast_anatomy: ForecastAnatomy | None = None
    calibration_tensions: list[CalibrationTension] = Field(default_factory=list)
    topology_seeds: list[TopologySeed] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)

    @property
    def score_0_100(self) -> float:
        value = self.forecast.prospective_5y_diffusion_percentile
        return 50.0 if value is None else value

    @property
    def p_uptake(self) -> float:
        return self.forecast.uptake_probability or 0.0

    @property
    def conditional_diffusion(self) -> float:
        return self.forecast.conditional_diffusion or 0.0

    @property
    def feature_coverage(self) -> float:
        return self.forecast.feature_coverage

    @property
    def diagnostic_flags(self) -> list[str]:
        return self.diagnostics

    @model_validator(mode="after")
    def point_in_time_safe(self) -> GraphRuntimePacket:
        for seed in self.topology_seeds:
            if not seed.cutoff_valid:
                raise ValueError(
                    "runtime packet contains a cutoff-invalid topology seed"
                )
            if seed.as_of_date > self.cutoff_date:
                raise ValueError("topology snapshot is later than the review cutoff")
            if seed.source_max_date is not None:
                if seed.source_max_date >= self.cutoff_date:
                    raise ValueError("topology source contains post-cutoff data")
            elif (
                seed.source_max_year is None
                or seed.source_max_year >= self.cutoff_date.year
            ):
                raise ValueError(
                    "year-only topology source is not conservatively pre-cutoff"
                )
            if seed.publication_date is not None:
                if seed.publication_date >= self.cutoff_date:
                    raise ValueError("topology candidate is not pre-cutoff")
            elif (
                seed.publication_year is None
                or seed.publication_year >= self.cutoff_date.year
            ):
                raise ValueError("year-only topology candidate is not pre-cutoff")
        return self


class GraphResourceCaps(StrictModel):
    provider_searches: int = Field(default=8, ge=0)
    direct_fetches: int = Field(default=8, ge=0)
    neighbor_expansions: int = Field(default=2, ge=0)
    fulltext_candidates: int = Field(default=12, ge=0)
    relation_classifications: int = Field(default=12, ge=0)


class RoutedCandidate(StrictModel):
    candidate_id: str
    pool: Literal["local", "remote", "topology", "analog"]
    pool_rank: int = Field(ge=0)
    final_rank: int | None = Field(default=None, ge=0)
    semantic_relevance: float = Field(ge=0.0, le=1.0)
    selected_for_verification: bool
    reason: str


class RetrievalRoutingPlan(StrictModel):
    contract: Literal["gear_retrieval_routing_plan"] = "gear_retrieval_routing_plan"
    paper_id: str
    variant: Literal[
        "neutral",
        "topology_only",
        "scalar_score",
        "hgb_analog",
        "full_calibrated",
        "shuffled_hgb",
    ]
    draft_sha256: str
    resource_caps: GraphResourceCaps = Field(default_factory=GraphResourceCaps)
    q_effective: float = Field(ge=0.0, le=1.0)
    local_weight: float = Field(ge=0.0, le=1.0)
    remote_weight: float = Field(ge=0.0, le=1.0)
    topology_query_replacements: list[str] = Field(default_factory=list, max_length=2)
    candidates: list[RoutedCandidate] = Field(default_factory=list)

    @model_validator(mode="after")
    def weights_and_budget_are_consistent(self) -> RetrievalRoutingPlan:
        if abs(self.local_weight + self.remote_weight - 1.0) > 1e-9:
            raise ValueError("routing weights must sum to one")
        if len([row for row in self.candidates if row.selected_for_verification]) > (
            self.resource_caps.relation_classifications
        ):
            raise ValueError("routing plan exceeds relation-classification budget")
        return self


class RetrievalMission(StrictModel):
    mission_id: str
    mission_type: Literal[
        "local_nearest_antecedent",
        "remote_mechanism_analogue",
        "topology_seed",
    ]
    origin: Literal["score", "topology", "calibration"]
    target_claim_id: str
    orientation: Literal["neutral"]
    query_roles: list[str] = Field(default_factory=list)
    seed_work_ids: list[str] = Field(default_factory=list)
    traversal: Literal["none", "references", "citations"] = "none"
    expected_relation_types: list[str] = Field(default_factory=list)
    stop_rule: str


class ClaimGuidance(StrictModel):
    review_point_id: str
    claim_id: str
    claim_relevance: float = Field(ge=0.0, le=1.0)
    allocated_local_query_slots: int = Field(ge=0)
    allocated_remote_query_slots: int = Field(ge=0)
    missions: list[RetrievalMission] = Field(default_factory=list)
    analog_seeds: list[AnalogSeed] = Field(default_factory=list, max_length=2)


class RetrievalGuidancePlan(StrictModel):
    contract: Literal["gear_retrieval_guidance_plan"] = "gear_retrieval_guidance_plan"
    paper_id: str
    policy_version: str = "safe_graph_admission_v2"
    source_packet_evidence_key: str
    controller_state: dict[str, float | int | str | bool] = Field(default_factory=dict)
    resource_caps: GraphResourceCaps = Field(default_factory=GraphResourceCaps)
    claim_guidance: list[ClaimGuidance] = Field(default_factory=list)
    no_effect_reason: str | None = None


class ResourceLedger(StrictModel):
    contract: Literal["gear_resource_ledger"] = "gear_resource_ledger"
    paper_id: str
    caps: GraphResourceCaps = Field(default_factory=GraphResourceCaps)
    logical_provider_searches: int = Field(default=0, ge=0)
    network_provider_attempts: int = Field(default=0, ge=0)
    logical_direct_fetches: int = Field(default=0, ge=0)
    network_direct_fetch_attempts: int = Field(default=0, ge=0)
    logical_neighbor_expansions: int = Field(default=0, ge=0)
    network_neighbor_attempts: int = Field(default=0, ge=0)
    fulltext_candidates_retained: int = Field(default=0, ge=0)
    relation_classification_calls: int = Field(default=0, ge=0)
    retrieval_model_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cache_hits: int = Field(default=0, ge=0)


__all__ = [
    "AnalogSeed",
    "CalibrationTension",
    "ClaimGuidance",
    "ForecastAnatomy",
    "GraphResourceCaps",
    "GraphRuntimePacket",
    "InfluenceForecast",
    "ResourceLedger",
    "RetrievalGuidancePlan",
    "RetrievalMission",
    "RetrievalRoutingPlan",
    "RoutedCandidate",
    "TopologySeed",
]

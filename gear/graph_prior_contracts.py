"""Public and audit-only contracts for the isolated Graph prior branch."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .contracts import FeatureScalar, StrictModel

FULLTEXT16_FEATURE_IDS = {
    "EF0017",
    "EF0038",
    "EF0052",
    "EF0083",
    "EF0186",
    "EF0188",
    "EF0197",
    "EF0238",
    "EF0240",
    "EF0307",
    "EF0309",
    "EF0312",
    "EF0314",
    "EF0315",
    "EF0318",
    "EF0319",
}
STRUCTURAL_FEATURE_IDS = {
    "EF0017",
    "EF0052",
    "EF0240",
    "EF0309",
    "EF0312",
    "EF0315",
    "EF0318",
}


class GraphPriorProvenance(StrictModel):
    calibration_release_id: str | None = None
    model_id: str | None = None
    model_sha256: str | None = None
    score_table_sha256: str | None = None
    feature_matrix_sha256: str | None = None
    evidence_policy: Literal["fig1_fig2_fig3_current_only"] = (
        "fig1_fig2_fig3_current_only"
    )


class GraphPriorResult(StrictModel):
    """Read-only compatibility contract for persisted V2 artifacts."""

    contract: Literal["aspr_graph_prior_v2"] = "aspr_graph_prior_v2"
    paper_id: str
    status: Literal["exact_lookup", "eligible_inference", "unavailable"]
    score_0_100: float | None = Field(default=None, ge=0.0, le=100.0)
    primary_feature_set: Literal["fulltext_16"] = "fulltext_16"
    model_id: str | None = None
    feature_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    drift_flags: list[str] = Field(default_factory=list)
    quality_flags: list[str] = Field(default_factory=list)
    provenance: GraphPriorProvenance = Field(default_factory=GraphPriorProvenance)


class GraphResultV3(StrictModel):
    """The single Graph result consumed by the current review runtime."""

    contract: Literal["aspr_graph_result_v3"] = "aspr_graph_result_v3"
    paper_id: str
    score_0_100: float = Field(ge=0.0, le=100.0)
    p_uptake: float = Field(ge=0.0, le=1.0)
    conditional_diffusion: float = Field(ge=0.0, le=1.0)
    feature_coverage: float = Field(ge=0.0, le=1.0)


class GraphResultV4(StrictModel):
    """Graph score plus deterministic directions for prior-art retrieval."""

    contract: Literal["aspr_graph_result_v4"] = "aspr_graph_result_v4"
    paper_id: str
    score_0_100: float = Field(ge=0.0, le=100.0)
    p_uptake: float = Field(ge=0.0, le=1.0)
    conditional_diffusion: float = Field(ge=0.0, le=1.0)
    feature_coverage: float = Field(ge=0.0, le=1.0)
    seed_work_ids: list[str] = Field(default_factory=list)
    search_terms: list[str] = Field(default_factory=list)


class GraphTopologySeedV1(StrictModel):
    """Cutoff-safe topology anchor; never review evidence by itself."""

    work_id: str
    title: str = ""
    publication_year: int | None = Field(default=None, ge=0)
    shared_reference_count: int = Field(default=0, ge=0)
    shared_reference_ids: list[str] = Field(default_factory=list)
    anchor_field_ids: list[str] = Field(default_factory=list)


class GraphRuntimePacketV1(StrictModel):
    """Source-agnostic Graph runtime packet used by guidance policy."""

    contract: Literal["aspr_graph_runtime_packet_v1"] = "aspr_graph_runtime_packet_v1"
    paper_id: str
    score_semantics: Literal["prospective_structural_innovation_percentile"] = (
        "prospective_structural_innovation_percentile"
    )
    score_0_100: float = Field(ge=0.0, le=100.0)
    raw_expected_diffusion: float = Field(ge=0.0, le=1.0)
    p_uptake: float = Field(ge=0.0, le=1.0)
    conditional_diffusion: float = Field(ge=0.0, le=1.0)
    feature_version: Literal["fulltext16_v3"] = "fulltext16_v3"
    feature_values: dict[str, FeatureScalar] = Field(default_factory=dict)
    historical_bands: dict[str, str] = Field(default_factory=dict)
    missing_feature_ids: list[str] = Field(default_factory=list)
    diagnostic_flags: list[str] = Field(default_factory=list)
    topology_seeds: list[GraphTopologySeedV1] = Field(default_factory=list)

    @property
    def feature_coverage(self) -> float:
        """Coverage is derived, never trusted as an independently stored scalar."""
        return max(0.0, 1.0 - len(set(self.missing_feature_ids)) / 16.0)

    @model_validator(mode="after")
    def consistent_profile(self) -> GraphRuntimePacketV1:
        feature_ids = set(self.feature_values)
        missing_ids = set(self.missing_feature_ids)
        if not feature_ids.issubset(FULLTEXT16_FEATURE_IDS):
            raise ValueError("runtime packet contains an unknown Full-text-16 feature")
        if len(missing_ids) != len(
            self.missing_feature_ids
        ) or not missing_ids.issubset(FULLTEXT16_FEATURE_IDS):
            raise ValueError("missing_feature_ids are invalid or duplicated")
        observed_none = {
            feature_id
            for feature_id, value in self.feature_values.items()
            if value is None
        }
        observed_values = feature_ids - observed_none
        if not observed_none.issubset(missing_ids) or observed_values & missing_ids:
            raise ValueError("feature values and missing_feature_ids disagree")
        if not set(self.historical_bands).issubset(STRUCTURAL_FEATURE_IDS):
            raise ValueError("historical bands are restricted to structural features")
        return self


class GraphResourceCapsV1(StrictModel):
    provider_searches: int = Field(default=8, ge=0)
    direct_fetches: int = Field(default=8, ge=0)
    neighbor_expansions: int = Field(default=2, ge=0)
    fulltext_candidates: int = Field(default=12, ge=0)
    relation_classifications: int = Field(default=12, ge=0)


class GraphMissionV1(StrictModel):
    mission_id: str
    mission_type: Literal[
        "local_nearest_antecedent",
        "remote_mechanism_analogue",
        "terminology_free_counterfactual",
        "reference_structure_diversity",
        "historical_lineage",
        "recent_direct_predecessor",
        "terminology_lineage",
        "topology_seed",
        "remote_rescue",
    ]
    origin: Literal["score", "profile", "topology", "rescue"]
    target_claim_id: str
    orientation: Literal["falsification", "rescue", "neutral"]
    query_roles: list[str] = Field(default_factory=list)
    seed_work_ids: list[str] = Field(default_factory=list)
    traversal: Literal["none", "references", "citations"] = "none"
    expected_relation_types: list[str] = Field(default_factory=list)
    stop_rule: str


class GraphClaimGuidanceV1(StrictModel):
    review_point_id: str
    claim_id: str
    claim_relevance: float = Field(ge=0.0, le=1.0)
    allocated_local_query_slots: int = Field(ge=0)
    allocated_remote_query_slots: int = Field(ge=0)
    missions: list[GraphMissionV1] = Field(default_factory=list)


class GraphGuidancePlanV1(StrictModel):
    contract: Literal["aspr_graph_guidance_plan_v1"] = "aspr_graph_guidance_plan_v1"
    paper_id: str
    policy_version: str = "score_profile_topology_v22_claim_aligned"
    source_packet_evidence_key: str
    controller_state: dict[str, float | int | str | bool] = Field(default_factory=dict)
    resource_caps: GraphResourceCapsV1 = Field(default_factory=GraphResourceCapsV1)
    claim_guidance: list[GraphClaimGuidanceV1] = Field(default_factory=list)
    no_effect_reason: str | None = None


class ResourceLedgerV1(StrictModel):
    contract: Literal["aspr_resource_ledger_v1"] = "aspr_resource_ledger_v1"
    paper_id: str
    caps: GraphResourceCapsV1 = Field(default_factory=GraphResourceCapsV1)
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


class FeatureSetAudit(StrictModel):
    feature_set: Literal["strict_7", "fulltext_16", "source_154", "ultrarelaxed_221"]
    expected_dimension: int
    observed_dimension: int
    coverage: float = Field(ge=0.0, le=1.0)
    model_id: str | None = None
    score_0_100: float | None = Field(default=None, ge=0.0, le=100.0)
    quality_flags: list[str] = Field(default_factory=list)


class GraphPriorAudit(StrictModel):
    """Sensitive reproduction data that never enters reviewer prompts or prose."""

    contract: Literal["aspr_graph_prior_audit_v1"] = "aspr_graph_prior_audit_v1"
    paper_id: str
    feature_values: dict[str, FeatureScalar] = Field(default_factory=dict)
    p_uptake: float | None = None
    conditional_diffusion: float | None = None
    feature_sets: list[FeatureSetAudit] = Field(default_factory=list)


__all__ = [
    "FeatureSetAudit",
    "GraphClaimGuidanceV1",
    "GraphGuidancePlanV1",
    "GraphMissionV1",
    "GraphPriorAudit",
    "GraphPriorProvenance",
    "GraphPriorResult",
    "GraphResourceCapsV1",
    "GraphResultV3",
    "GraphResultV4",
    "GraphRuntimePacketV1",
    "GraphTopologySeedV1",
    "ResourceLedgerV1",
]

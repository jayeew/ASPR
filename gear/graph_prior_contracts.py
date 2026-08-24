"""Public and audit-only contracts for the isolated Graph prior branch."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .contracts import FeatureScalar, StrictModel


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
    "GraphPriorAudit",
    "GraphPriorProvenance",
    "GraphPriorResult",
    "GraphResultV3",
    "GraphResultV4",
]

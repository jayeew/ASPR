"""Public and audit-only contracts for the isolated Graph prior branch."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import Field

from .contracts import FeatureScalar, StrictModel


class GraphPriorProvenance(StrictModel):
    calibration_release_id: Optional[str] = None
    model_id: Optional[str] = None
    model_sha256: Optional[str] = None
    score_table_sha256: Optional[str] = None
    feature_matrix_sha256: Optional[str] = None
    evidence_policy: Literal["fig1_fig2_fig3_current_only"] = (
        "fig1_fig2_fig3_current_only"
    )


class GraphPriorResult(StrictModel):
    """The only Graph payload visible outside the Graph branch."""

    contract: Literal["aspr_graph_prior_v2"] = "aspr_graph_prior_v2"
    paper_id: str
    status: Literal["exact_lookup", "eligible_inference", "unavailable"]
    score_0_100: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    primary_feature_set: Literal["fulltext_16"] = "fulltext_16"
    model_id: Optional[str] = None
    feature_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    drift_flags: List[str] = Field(default_factory=list)
    quality_flags: List[str] = Field(default_factory=list)
    provenance: GraphPriorProvenance = Field(default_factory=GraphPriorProvenance)


class FeatureSetAudit(StrictModel):
    feature_set: Literal["strict_7", "fulltext_16", "source_154", "ultrarelaxed_221"]
    expected_dimension: int
    observed_dimension: int
    coverage: float = Field(ge=0.0, le=1.0)
    model_id: Optional[str] = None
    score_0_100: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    quality_flags: List[str] = Field(default_factory=list)


class GraphPriorAudit(StrictModel):
    """Sensitive reproduction data that never enters reviewer prompts or prose."""

    contract: Literal["aspr_graph_prior_audit_v1"] = "aspr_graph_prior_audit_v1"
    paper_id: str
    feature_values: Dict[str, FeatureScalar] = Field(default_factory=dict)
    p_uptake: Optional[float] = None
    conditional_diffusion: Optional[float] = None
    feature_sets: List[FeatureSetAudit] = Field(default_factory=list)


__all__ = [
    "FeatureSetAudit",
    "GraphPriorAudit",
    "GraphPriorProvenance",
    "GraphPriorResult",
]

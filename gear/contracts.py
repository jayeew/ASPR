"""Strict, versioned contracts shared by every ASPR-GEAR stage."""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION: Literal["aspr_gear"] = "aspr_gear"
FeatureScalar = Union[float, int, str, bool, None]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    """Base contract that rejects silently ignored fields."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class EvidenceModel(StrictModel):
    """Base for serialized evidence and state objects."""

    schema_version: Literal["aspr_gear"] = SCHEMA_VERSION
    schema_revision: Literal["evidence_state_delta_v2"] = "evidence_state_delta_v2"


class ParseStatus(str, Enum):
    READY = "ready"
    DEGRADED = "parse_degraded"
    UNAVAILABLE = "parse_unavailable"


class EvidenceReadiness(str, Enum):
    READY = "ready"
    LIMITED = "limited"
    UNAVAILABLE = "unavailable"


class ClaimType(str, Enum):
    NOVELTY = "novelty_claim"
    METHOD = "method_claim"
    RESULT = "result_claim"
    SCOPE = "scope_claim"
    CAUSAL = "causal_claim"
    SIGNIFICANCE = "significance_claim"


class ClaimStrength(str, Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


class CalibrationMode(str, Enum):
    EXACT_LOOKUP = "exact_lookup"
    ELIGIBLE_INFERENCE = "eligible_inference"
    PROFILE_ONLY = "profile_only"
    UNAVAILABLE = "unavailable"


class RelationLabel(str, Enum):
    BACKGROUND = "BACKGROUND"
    BUILDING_BLOCK = "BUILDING_BLOCK"
    DIRECT_ANTECEDENT = "DIRECT_ANTECEDENT"
    PARTIAL_ANTECEDENT = "PARTIAL_ANTECEDENT"
    EXTENSION = "EXTENSION"
    PARALLEL = "PARALLEL"
    SUPPORT = "SUPPORT"
    CONFLICT = "CONFLICT"
    DISTANT = "DISTANT"
    UNRESOLVED = "UNRESOLVED"


class EvidenceLevel(str, Enum):
    FULLTEXT = "fulltext_evidence"
    ABSTRACT = "abstract_evidence"
    CITATION_CONTEXT = "citation_context_evidence"
    METADATA_ONLY = "metadata_only"


class ReviewStatus(str, Enum):
    COMPLETE = "complete"
    LIMITED = "limited"
    FAILED = "failed"


class PaperMetadata(EvidenceModel):
    title: str = ""
    authors: List[str] = Field(default_factory=list)
    doi: Optional[str] = None
    openalex_id: Optional[str] = None
    publication_date: Optional[date] = None
    submission_date: Optional[date] = None
    venue: Optional[str] = None
    domain: Optional[str] = None

    @field_validator("doi", mode="before")
    @classmethod
    def normalize_doi(cls, value: Any) -> Optional[str]:
        text = str(value or "").strip()
        for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
            if text.casefold().startswith(prefix):
                text = text[len(prefix) :]
                break
        return text or None


class ReviewRequest(EvidenceModel):
    paper_path: Path
    metadata: PaperMetadata = Field(default_factory=PaperMetadata)
    evaluation_date: date = Field(default_factory=date.today)

    @property
    def evidence_date(self) -> date:
        """Resolve the prior-art boundary without exposing a user mode switch."""
        return (
            self.metadata.submission_date
            or self.metadata.publication_date
            or self.evaluation_date
        )

    @property
    def evidence_date_source(
        self,
    ) -> Literal["publication_date", "submission_date", "evaluation_date"]:
        if self.metadata.submission_date is not None:
            return "submission_date"
        if self.metadata.publication_date is not None:
            return "publication_date"
        return "evaluation_date"


class PageText(EvidenceModel):
    page: int = Field(ge=1)
    text: str


class EvidenceSpan(EvidenceModel):
    span_id: str
    source_id: str
    page: int = Field(ge=1)
    section_path: List[str] = Field(default_factory=list)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    text: str
    text_sha256: str

    @model_validator(mode="after")
    def validate_offsets(self) -> "EvidenceSpan":
        if self.char_end < self.char_start:
            raise ValueError("char_end must be >= char_start")
        observed = "sha256:" + hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if self.text_sha256 != observed:
            raise ValueError("EvidenceSpan text_sha256 does not match text")
        return self


class PaperClaim(EvidenceModel):
    claim_id: str
    claim_type: ClaimType
    span_id: str
    text: str
    strength: ClaimStrength = ClaimStrength.MODERATE
    dependency_span_ids: List[str] = Field(default_factory=list)
    required_evidence: List[str] = Field(default_factory=list)


class ReferenceEntry(EvidenceModel):
    reference_id: str
    raw_text: str
    source_span_id: str
    citation_number: Optional[int] = Field(default=None, ge=1)
    title: Optional[str] = None
    doi: Optional[str] = None
    publication_date: Optional[date] = None
    publication_year: Optional[int] = None


class MethodResultLedger(EvidenceModel):
    research_question: List[str] = Field(default_factory=list)
    dataset_sample: List[str] = Field(default_factory=list)
    design_comparator: List[str] = Field(default_factory=list)
    model_algorithm: List[str] = Field(default_factory=list)
    baselines_metrics_statistics: List[str] = Field(default_factory=list)
    ablation_robustness: List[str] = Field(default_factory=list)
    main_results: List[str] = Field(default_factory=list)
    stated_limitations: List[str] = Field(default_factory=list)
    figures_tables: List[str] = Field(default_factory=list)


class PaperQualityReport(EvidenceModel):
    """Observable PaperIR quality gates; never infer quality from completion alone."""

    evidence_readiness: EvidenceReadiness = EvidenceReadiness.READY
    section_count: int = Field(default=0, ge=0)
    document_only_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    table_figure_anchor_count: int = Field(default=0, ge=0)
    semantic_extraction_ready: bool = False
    blocking_reasons: List[str] = Field(default_factory=list)
    advisories: List[str] = Field(default_factory=list)


class ClaimLedger(EvidenceModel):
    novelty_claim_ids: List[str] = Field(default_factory=list)
    method_claim_ids: List[str] = Field(default_factory=list)
    result_claim_ids: List[str] = Field(default_factory=list)
    scope_claim_ids: List[str] = Field(default_factory=list)
    causal_claim_ids: List[str] = Field(default_factory=list)
    significance_claim_ids: List[str] = Field(default_factory=list)
    method_span_ids: List[str] = Field(default_factory=list)
    result_span_ids: List[str] = Field(default_factory=list)
    table_span_ids: List[str] = Field(default_factory=list)
    reference_ids: List[str] = Field(default_factory=list)


class PaperIR(EvidenceModel):
    paper_id: str
    paper_path: Path
    paper_sha256: str
    source_format: Literal["markdown", "pdf"]
    markdown: str
    metadata: PaperMetadata
    pages: List[PageText]
    spans: List[EvidenceSpan]
    claims: List[PaperClaim]
    claim_ledger: ClaimLedger
    method_result_ledger: MethodResultLedger
    references: List[ReferenceEntry] = Field(default_factory=list)
    parse_status: ParseStatus
    quality_flags: List[str] = Field(default_factory=list)
    quality_report: PaperQualityReport = Field(default_factory=PaperQualityReport)

    def span_map(self) -> Dict[str, EvidenceSpan]:
        return {span.span_id: span for span in self.spans}

    @model_validator(mode="after")
    def validate_evidence_graph(self) -> "PaperIR":
        page_map = {page.page: page.text for page in self.pages}
        if len(page_map) != len(self.pages):
            raise ValueError("PaperIR page numbers must be unique")
        span_map = self.span_map()
        if len(span_map) != len(self.spans):
            raise ValueError("PaperIR span IDs must be unique")
        for span in self.spans:
            if span.page not in page_map:
                raise ValueError(f"span references an unknown page: {span.span_id}")
            if page_map[span.page][span.char_start : span.char_end] != span.text:
                raise ValueError(f"span coordinates do not match text: {span.span_id}")
        claim_ids = {claim.claim_id for claim in self.claims}
        if len(claim_ids) != len(self.claims):
            raise ValueError("PaperIR claim IDs must be unique")
        if any(claim.span_id not in span_map for claim in self.claims):
            raise ValueError("PaperIR claim references an unknown span")
        reference_ids = {reference.reference_id for reference in self.references}
        if len(reference_ids) != len(self.references):
            raise ValueError("PaperIR reference IDs must be unique")
        if any(
            reference.source_span_id not in span_map for reference in self.references
        ):
            raise ValueError("PaperIR reference references an unknown span")
        ledger_claim_ids = {
            *self.claim_ledger.novelty_claim_ids,
            *self.claim_ledger.method_claim_ids,
            *self.claim_ledger.result_claim_ids,
            *self.claim_ledger.scope_claim_ids,
            *self.claim_ledger.causal_claim_ids,
            *self.claim_ledger.significance_claim_ids,
        }
        if not ledger_claim_ids.issubset(claim_ids):
            raise ValueError("ClaimLedger references an unknown claim")
        ledger_span_ids = {
            *self.claim_ledger.method_span_ids,
            *self.claim_ledger.result_span_ids,
            *self.claim_ledger.table_span_ids,
        }
        method_result_span_ids = {
            span_id
            for field_name in type(self.method_result_ledger).model_fields
            if field_name not in {"schema_version", "schema_revision"}
            for span_id in getattr(self.method_result_ledger, field_name)
        }
        if not ledger_span_ids.issubset(
            span_map
        ) or not method_result_span_ids.issubset(span_map):
            raise ValueError("PaperIR ledger references an unknown span")
        if not set(self.claim_ledger.reference_ids).issubset(reference_ids):
            raise ValueError("ClaimLedger references an unknown reference")
        return self


class CalibrationCutoff(EvidenceModel):
    publication_date: Optional[date] = None
    publication_year: Optional[int] = None
    source_max_year: Optional[int] = None
    granularity: Literal["year", "day", "unknown"] = "unknown"


class CalibrationMeasurement(EvidenceModel):
    feature_set: str = "fulltext_16"
    feature_version: str = "evidence_v3"
    substantive_innovation: Dict[str, FeatureScalar] = Field(default_factory=dict)
    t0_potential: Dict[str, FeatureScalar] = Field(default_factory=dict)
    opportunity: Dict[str, FeatureScalar] = Field(default_factory=dict)
    context_control: Dict[str, FeatureScalar] = Field(default_factory=dict)
    local_surrogates: List[str] = Field(default_factory=list)
    historical_bands: Dict[str, str] = Field(default_factory=dict)


class CalibrationForecast(EvidenceModel):
    horizon: int = 5
    p_uptake: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    conditional_diffusion: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    raw_expected_diffusion: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    aspr_score_0_100: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    reference_corpus: str = "nature-mature-d5"


class CalibrationReliability(EvidenceModel):
    mode: CalibrationMode
    domain: Optional[str] = None
    domain_support_n: Optional[int] = None
    temporal_block: Optional[str] = None
    overall_oof_spearman: Optional[float] = None
    fold_oof_spearman: Optional[float] = None
    domain_oof_spearman: Optional[float] = None
    feature_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    missing_features: List[str] = Field(default_factory=list)
    drift_flags: List[str] = Field(default_factory=list)
    quality_flags: List[str] = Field(default_factory=list)


class CalibrationInterpretation(EvidenceModel):
    allowed: List[str] = Field(
        default_factory=lambda: [
            "publication-time structural profile",
            "relative prospective scientific diffusion signal",
        ]
    )
    prohibited: List[str] = Field(
        default_factory=lambda: [
            "direct novelty truth",
            "causal impact",
            "paper quality",
            "acceptance probability",
            "social impact",
        ]
    )


class CalibrationProvenance(EvidenceModel):
    calibration_release_id: Optional[str] = None
    model_family: str = "hgb"
    model_sha256: Optional[str] = None
    score_table_sha256: Optional[str] = None
    feature_matrix_sha256: Optional[str] = None
    reference_corpus_sha256: Optional[str] = None
    oof_metrics_sha256: Optional[str] = None
    oof_fold_metrics_sha256: Optional[str] = None
    oof_domain_metrics_sha256: Optional[str] = None
    runtime_replay_manifest_sha256: Optional[str] = None
    runtime_matrix_sha256: Optional[str] = None
    context_snapshot_sha256: Optional[str] = None
    evidence_policy: str = "fig1_fig2_fig3_current_only"
    deprecated_fig4_to_fig10_used: bool = False


class CalibrationPacketV3(EvidenceModel):
    contract: Literal["aspr_calibration_packet_v3"] = "aspr_calibration_packet_v3"
    paper_id: str
    cutoff: CalibrationCutoff
    measurement: CalibrationMeasurement
    forecast: CalibrationForecast
    reliability: CalibrationReliability
    interpretation: CalibrationInterpretation = Field(
        default_factory=CalibrationInterpretation
    )
    provenance: CalibrationProvenance = Field(default_factory=CalibrationProvenance)


class SubmissionCalibrationPacketV1(EvidenceModel):
    """Calibration contract for fields observable before journal publication.

    This packet is deliberately separate from ``CalibrationPacketV3`` so a
    submission-time profile can never be mistaken for an exact Fig.3 lookup.
    """

    contract: Literal["aspr_submission_calibration_packet_v1"] = (
        "aspr_submission_calibration_packet_v1"
    )
    paper_id: str
    cutoff: CalibrationCutoff
    measurement: CalibrationMeasurement
    forecast: CalibrationForecast
    reliability: CalibrationReliability
    interpretation: CalibrationInterpretation = Field(
        default_factory=CalibrationInterpretation
    )
    provenance: CalibrationProvenance = Field(default_factory=CalibrationProvenance)


class QuerySpec(EvidenceModel):
    query_id: str
    claim_id: str
    family: Literal["lexical", "semantic", "contrastive", "citation"]
    query_role: Literal[
        "author_terminology",
        "object_problem",
        "mechanism_outcome",
        "purpose_semantic",
        "legacy_contrastive",
        "author_citation",
        "citation_neighbor",
        "graph_focus",
        "graph_seed",
    ] = "object_problem"
    query: str
    search_mode: Literal["text", "semantic", "direct_id"] = "text"
    source_span_ids: List[str] = Field(default_factory=list)
    anchor_fields: List[str] = Field(default_factory=list)
    transformation: str = ""
    parent_query_id: Optional[str] = None


class ScientificSearchFrame(EvidenceModel):
    """Paper-grounded scientific intent used only to formulate retrieval."""

    target_object: List[str] = Field(default_factory=list)
    task_problem: List[str] = Field(default_factory=list)
    mechanism: List[str] = Field(default_factory=list)
    population_input: List[str] = Field(default_factory=list)
    outcome_observable: List[str] = Field(default_factory=list)
    comparator: List[str] = Field(default_factory=list)
    author_terms: List[str] = Field(default_factory=list)
    brand_terms: List[str] = Field(default_factory=list)
    legacy_terms: List[str] = Field(default_factory=list)
    claimed_delta: str = ""
    citation_seed_ids: List[str] = Field(default_factory=list)
    source_span_ids: List[str] = Field(min_length=1)

    @field_validator("claimed_delta", mode="before")
    @classmethod
    def normalize_claimed_delta(cls, value: object) -> str:
        """Accept a model's enumerated deltas as one retrieval description."""
        if isinstance(value, list):
            return "; ".join(str(item).strip() for item in value if str(item).strip())
        return str(value or "").strip()

    @model_validator(mode="after")
    def require_scientific_anchors(self) -> "ScientificSearchFrame":
        groups = (
            self.target_object,
            self.task_problem,
            self.mechanism,
            self.population_input,
            self.outcome_observable,
        )
        if sum(bool(group) for group in groups) < 2:
            raise ValueError("scientific search frame requires at least two anchors")
        return self


class RetrievalHit(EvidenceModel):
    hit_id: str
    target_claim_id: str
    work_id: str
    query_id: str
    query_family: str
    search_mode: str
    provider_rank: int = Field(ge=1)
    provider_relevance: Optional[float] = None
    fused_score: float = Field(default=0.0, ge=0.0)
    selection_stage: Literal[
        "retrieved",
        "temporal_excluded",
        "metadata_only",
        "recall_filtered",
        "rerank_filtered",
        "compared",
    ] = "retrieved"
    gate_label: Optional[Literal["comparable", "partial", "distant"]] = None
    matched_fields: List[str] = Field(default_factory=list)
    gate_reason: str = ""
    claim_alignment: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    essential_claim_facets: List[str] = Field(default_factory=list)
    recall_score: Optional[float] = None
    rerank_score: Optional[float] = None


class RetrievedSpan(EvidenceModel):
    span_id: str
    text: str
    text_sha256: str
    source: EvidenceLevel

    @model_validator(mode="after")
    def validate_text_hash(self) -> "RetrievedSpan":
        observed = "sha256:" + hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if self.text_sha256 != observed:
            raise ValueError("RetrievedSpan text_sha256 does not match text")
        return self


class RetrievedWork(EvidenceModel):
    work_id: str
    target_claim_id: str
    title: str
    abstract: str = ""
    authors: List[str] = Field(default_factory=list)
    venue: str = ""
    publication_date: Optional[date] = None
    publication_year: Optional[int] = None
    doi: Optional[str] = None
    cited_work_ids: List[str] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    spans: List[RetrievedSpan] = Field(default_factory=list)
    # Kept for in-process compatibility. EvidenceStore persists query provenance
    # separately as RetrievalHit records, so this field is never part of W:*.
    retrieval_query_id: str
    source_query_ids: List[str] = Field(default_factory=list)
    retrieval_source: str


class RelationCard(EvidenceModel):
    relation_id: str
    target_claim_id: str
    target_span_id: str
    prior_work_id: str
    prior_span_id: Optional[str] = None
    prior_work_date: Optional[date] = None
    prior_work_year: Optional[int] = None
    relation_label: RelationLabel
    evidence_level: EvidenceLevel
    difference_dimensions: List[str] = Field(min_length=1)
    common_dimensions: List[str] = Field(default_factory=list)
    retrieval_query_id: str
    source_query_ids: List[str] = Field(default_factory=list)
    rationale: str = ""
    temporal_valid: bool = False
    temporal_order_unresolved: bool = False
    essential_facet_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    independent_verification_passed: bool = False


class RetrievalCoverageCard(EvidenceModel):
    """Auditable search coverage for one external-verification target."""

    coverage_id: str
    target_claim_id: str
    cutoff_date: date
    required_query_roles: List[str] = Field(default_factory=list)
    completed_query_roles: List[str] = Field(default_factory=list)
    query_ids: List[str] = Field(default_factory=list)
    retrieved_count: int = Field(default=0, ge=0)
    unique_eligible_count: int = Field(default=0, ge=0)
    temporal_excluded_count: int = Field(default=0, ge=0)
    metadata_only_count: int = Field(default=0, ge=0)
    compared_work_ids: List[str] = Field(default_factory=list)
    direct_or_partial_found: bool = False
    whole_paper_ranking_completed: bool = False
    purpose_ranking_completed: bool = False
    ranker: str = ""
    degraded: bool = False
    service_failed: bool = False
    exhaustive_provider_results: bool = False
    coverage_sufficient: bool = False
    advisory_notes: List[str] = Field(default_factory=list)


class RetrievalBudget(EvidenceModel):
    normal_used: int = Field(default=0, ge=0)
    contrastive_used: int = Field(default=0, ge=0)
    citation_expansion_used: int = Field(default=0, ge=0)
    fulltext_kept: int = Field(default=0, ge=0)
    normal_max: int = Field(default=4, ge=0)
    contrastive_max: int = Field(default=1, ge=0)
    citation_expansion_max: int = Field(default=1, ge=0)
    fulltext_max: int = Field(default=12, ge=0)

    @model_validator(mode="after")
    def validate_usage(self) -> "RetrievalBudget":
        for prefix in ("normal", "contrastive", "citation_expansion"):
            if getattr(self, f"{prefix}_used") > getattr(self, f"{prefix}_max"):
                raise ValueError(f"{prefix} retrieval budget exceeded")
        if self.fulltext_kept > self.fulltext_max:
            raise ValueError("retained evidence budget exceeded")
        return self


class FailureRecord(EvidenceModel):
    stage: str
    reason: str
    claim_id: Optional[str] = None
    recoverable: bool = True


class EvidenceRecord(EvidenceModel):
    evidence_id: str
    kind: str
    payload: Dict[str, Any]
    payload_sha256: str
    created_at: datetime = Field(default_factory=_utc_now)


class ActionRecord(EvidenceModel):
    action_id: str
    stage: str
    input_sha256: str
    output_sha256: str
    model: Optional[str] = None
    query: Optional[str] = None
    cache_hit: bool = False
    duration_ms: int = 0
    failure: Optional[str] = None
    reason_code: Optional[str] = None
    target_id: Optional[str] = None
    budget_before: Optional[int] = None
    budget_after: Optional[int] = None
    created_at: datetime = Field(default_factory=_utc_now)


class StateSnapshot(EvidenceModel):
    state_sha256: str
    state: Dict[str, Any]
    created_at: datetime = Field(default_factory=_utc_now)

"""Scientific records, not source-matching or completeness gates."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Record(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Evidence(Record):
    source_id: str
    source_type: str
    quote: str
    location: str
    scope: str
    reason: str


class ComparisonEvidence(Evidence):
    source_id: Literal['MANUSCRIPT', 'HISTORICAL']


class CoreItem(Record):
    core_id: str
    description: str
    manuscript_quote: str
    selection_reason: str


class Core(Record):
    status: Literal['selected', 'no_selectable_core']
    reason: str
    items: list[CoreItem] = Field(max_length=2)


class CoreMappingItem(Record):
    core_id: str
    claim_ids: list[str]
    reason: str


class CoreMapping(Record):
    items: list[CoreMappingItem]


class CoreChoice(Record):
    candidate_ids: list[str] = Field(max_length=2)
    reason: str


class CoreSentence(Record):
    sentence_id: str
    description: str
    selection_reason: str


class CoreSentences(Record):
    items: list[CoreSentence] = Field(max_length=2)


class FusionDecision(Record):
    finding_key: str
    action: Literal['retain', 'merge', 'correct', 'omit', 'unresolved']
    reason: str
    evidence_keys: list[str]


class FusionDecisions(Record):
    decisions: list[FusionDecision]


class NewError(Record):
    newly_introduced: bool
    branch_quotes: list[str]
    reason: str


class Queries(Record):
    queries: list[str] = Field(max_length=4)
    # Citations are independently extracted from this method's own manuscript input.
    cited_works: list[str]


class Analysis(Record):
    analysis: str
    evidence: list[Evidence]
    limitations: list[str]


class Report(Record):
    body: str
    cited_source_ids: list[str]


class ReportFinding(Record):
    finding_key: str
    action: Literal['retained', 'merged', 'corrected', 'omitted', 'unresolved']
    report_quote: str
    reason: str


class FullReport(Report):
    finding_outcomes: list[ReportFinding]


class ReviewSection(Record):
    section_id: str
    role: Literal['reviewer', 'author', 'editor', 'unknown']
    reviewer_id: str
    round_number: int
    manuscript_scope: str
    identity_explicit: bool
    quote: str


class ReviewSections(Record):
    sections: list[ReviewSection]


class Concern(Record):
    concern_id: str
    section_id: str
    reviewer_id: str
    round_number: int
    quote: str
    object_description: str
    scope: str
    dimension: Literal['novelty', 'evidence', 'scope']
    stance: Literal['recognized', 'limited', 'challenged', 'unresolved']
    reasons: list[str]
    reason_quotes: list[str]
    prior_work_queries: list[str]
    applies_to_input: Literal['yes', 'no', 'uncertain']
    manuscript_scope: str
    version_reason: str
    evidence: list[Evidence]


class Checklist(Record):
    concerns: list[Concern]


RState = Literal['supported_difference', 'substantially_covered', 'bounded_increment', 'insufficient_material']
PState = Literal['positive_increment', 'substantially_known', 'limited_increment', 'explicit_abstention', 'not_addressed']


class ReferenceItem(Record):
    core_id: str
    state: RState
    common_ground: str
    difference: str
    necessary_scope: str
    historical_comparison_applicable: bool
    scope_applicable: bool
    known_antecedent_ids: list[str]
    evidence: list[Evidence]
    limitations: list[str]


class Reference(Record):
    items: list[ReferenceItem]


class HistoricalComparison(Record):
    source_id: str
    relevant: bool
    common_ground: str
    difference: str
    scope: str
    evidence: list[ComparisonEvidence] = Field(max_length=2)
    limitation: str


class Unit(Record):
    unit_id: str
    quote: str
    claim_ids: list[str]
    needs_verification: bool
    kind: Literal['manuscript_fact', 'historical_comparison', 'increment', 'structure', 'scope']
    insight_type: Literal['historical_increment', 'cross_work_relation', 'scope_correction', 'cross_contribution', 'none']
    substantive: bool
    paraphrase: bool
    original_locator: str


class CorePrediction(Record):
    core_id: str
    state: PState
    quotes: list[str]
    conflicting_quotes: list[str]
    asserts_increment: bool
    difference: str
    stated_scope: str


class Units(Record):
    units: list[Unit]
    predictions: list[CorePrediction]


class Predictions(Record):
    predictions: list[CorePrediction]


class QuoteSpan(Record):
    quote_key: str
    present_in_source: bool
    first_clause_id: str
    last_clause_id: str
    reason: str


class QuoteSpans(Record):
    spans: list[QuoteSpan]


class ReportFindingLocation(Record):
    finding_key: str
    action: Literal['retained', 'merged', 'corrected', 'omitted', 'unresolved']
    first_clause_id: str
    last_clause_id: str
    reason: str


class ReportFindingLocations(Record):
    outcomes: list[ReportFindingLocation]


SupportState = Literal['supported', 'partly_supported', 'not_verifiable', 'contradicted']
ErrorKind = Literal['unsupported_definitive', 'false_antecedence', 'false_firstness', 'semantic_causal', 'omitted_scope']


class SupportItem(Record):
    unit_id: str
    support: SupportState
    scope_correct: bool | None
    evidence: list[Evidence]
    originally_substantiated: bool
    original_locator: str
    nonparaphrase_insight: bool
    errors: list[ErrorKind]
    reason: str


class Support(Record):
    units: list[SupportItem]


class NoveltyItem(Record):
    core_id: str
    difference_correct: bool | None
    scope_correct: bool | None
    historical_comparison_correct: bool | None
    false_firstness: bool | None
    false_antecedence: bool | None
    material_overclaim: bool | None
    effective_increment: bool | None
    antecedent_ids_covered: list[str]
    evidence: list[Evidence]
    reason: str


class Novelty(Record):
    items: list[NoveltyItem]


class ConcernMatch(Record):
    concern_id: str
    scope: Literal['same', 'partial', 'none']
    report_quote: str
    reason_coverage: Literal['complete', 'partial', 'none', 'not_applicable']
    stance: Literal['recognized', 'limited', 'challenged', 'unresolved', 'absent']
    disagreement: Literal['none', 'system_supported', 'system_error', 'review_unsubstantiated', 'unresolved']
    evidence: list[Evidence]
    reason: str


class ConcernMatches(Record):
    matches: list[ConcernMatch]


class QualityCheck(Record):
    dimension: Literal['contribution_fidelity', 'historical_increment', 'knowledge_relations', 'scope_uncertainty', 'whole_paper_synthesis']
    criteria: list[str]
    evidence: list[Evidence]


class QualityChecklist(Record):
    dimensions: list[QualityCheck]


class QualityScore(Record):
    dimension: Literal['contribution_fidelity', 'historical_increment', 'knowledge_relations', 'scope_uncertainty', 'whole_paper_synthesis']
    score: int = Field(ge=0, le=3)
    report_quotes: list[str]
    evidence: list[Evidence]
    reason: str


class Quality(Record):
    dimensions: list[QualityScore]


class ToneItem(Record):
    section_id: str
    tone: Literal['positive', 'neutral', 'negative', 'mixed']
    scientific_stance: Literal['recognized', 'limited', 'challenged', 'unresolved']
    quote: str
    reason: str


class Tones(Record):
    items: list[ToneItem]


class ReviewChange(Record):
    concern_id: str
    reviewer_id: str
    earlier_round: int
    later_round: int | None
    earlier_stance: str
    later_stance: str
    author_response_quote: str
    later_reviewer_quote: str
    same_reviewer_explicit: bool
    resolution: Literal['explicitly_resolved', 'partly_resolved', 'unresolved', 'no_explicit_followup', 'not_applicable']
    reason: str


class ReviewDynamics(Record):
    changes: list[ReviewChange]


class Cluster(Record):
    cluster_id: str
    unit_keys: list[str]
    summary: str
    kind: Literal['historical_increment', 'cross_work_relation', 'scope_correction', 'cross_contribution']


class Clusters(Record):
    clusters: list[Cluster]


class ImportanceItem(Record):
    cluster_id: str
    important: bool
    reason: str


class Importance(Record):
    items: list[ImportanceItem]


class ErrorCluster(Record):
    cluster_id: str
    unit_keys: list[str]
    error_type: ErrorKind
    reason: str


class ErrorClusters(Record):
    clusters: list[ErrorCluster]


class ErrorFlow(Record):
    branch_unit_key: str
    status: Literal['explicitly_corrected', 'not_propagated', 'propagated', 'unknown']
    full_unit_keys: list[str]
    full_quote: str
    evidence: list[Evidence]
    reason: str


class FusionJudgment(Record):
    errors: list[ErrorFlow]
    newly_introduced_error_keys: list[str]
    reason: str


Choice = Literal['A', 'B', 'tie', 'cannot_judge']


class Preference(Record):
    overall: Choice
    increment_clarity: Choice
    evidence_traceability: Choice
    knowledge_usefulness: Choice
    appropriate_limitations: Choice
    quote_a: str
    quote_b: str
    reason: str


class ControlPair(Record):
    kind: Literal['source_mismatch', 'firstness_overreach', 'scope_deletion']
    original: str
    altered: str
    original_source_ids: list[str]
    altered_source_ids: list[str]
    alteration_reason: str


class Controls(Record):
    pairs: list[ControlPair] = Field(max_length=3)


class ControlVerdict(Record):
    support: SupportState
    scope_correct: bool | None
    errors: list[ErrorKind]
    reason: str
    evidence: list[Evidence]


class Digest(Record):
    text: str = Field(max_length=2200)
    source_ids: list[str]
    limitations: list[str]


class SharedClaim(Record):
    claim_id: str
    claim_type: Literal['METHOD', 'FINDING', 'MECHANISM', 'RESOURCE', 'THEORY']
    normalized_claim_text: str
    author_claim_text: str
    source_span_ids: list[str]
    manuscript_quote: str


class SharedClaims(Record):
    claims: list[SharedClaim] = Field(max_length=8)


class ManuscriptSupport(Record):
    status: Literal['supported', 'partially_supported', 'internally_unsupported', 'unresolved']
    supported_scope: str
    evidence: list[Evidence]
    reason: str


class HistoricalRelation(Record):
    work_id: str
    relation: Literal['DIRECT_ANTECEDENT', 'PARTIAL_ANTECEDENT', 'EXTENSION', 'PARALLEL',
                      'SUPPORT', 'CONFLICT', 'BUILDING_BLOCK', 'DISTANT', 'UNRESOLVED']
    common_dimensions: list[str]
    difference_dimensions: list[str]
    essential_facet_coverage: float
    evidence: list[Evidence]
    rationale: str


class HistoricalRelations(Record):
    relations: list[HistoricalRelation]


class AntecedentVerification(Record):
    confirmed: bool
    missing_facets: list[str]
    rationale: str


class ConcernBrief(Record):
    concern_id: str
    brief: str = Field(max_length=1000)


class ConcernBriefs(Record):
    concerns: list[ConcernBrief]

"""Only additional report provenance, relation and fusion-event contracts."""
from __future__ import annotations

from typing import Literal

from pydantic import Field

from figure_pipeline.fig3_revision.models import Analysis, Evidence, Record, Report, Units


class Organized(Analysis):
    findings: list[dict] = Field(default_factory=list)


class Outcome(Record):
    finding_key: str
    action: Literal['retained', 'merged', 'corrected', 'omitted', 'unresolved']
    report_quote: str
    reason: str


class TracedReport(Report):
    finding_outcomes: list[Outcome]


class Outcomes(Record):
    items: list[Outcome]


class FindingAtoms(Record):
    finding_key: str
    extraction: Units


class FindingAtomBatches(Record):
    items: list[FindingAtoms]


class Relation(Record):
    unit_key: str
    claim_ids: list[str]
    proposition: str
    quote: str
    support: Literal['supported', 'partly_supported', 'not_verifiable', 'contradicted']
    scope_correct: bool | None
    evidence: list[Evidence]
    reason: str


class Relations(Record):
    items: list[Relation]


class Issue(Record):
    issue_id: str
    proposition: str
    scope: str
    unit_keys: list[str]


class Issues(Record):
    issues: list[Issue]


class Transition(Record):
    issue_id: str
    full_status: Literal['corrected', 'not_propagated', 'propagated', 'unresolved']
    full_unit_keys: list[str]
    full_quote: str
    correction_present: bool
    error_propagated: bool
    without_other_branch_status: Literal['corrected', 'not_propagated', 'propagated', 'unresolved', 'not_applicable']
    other_branch_counterevidence: list[Evidence]
    counterevidence_unit_keys: list[str]
    branch_dependent_correction: bool
    evidence: list[Evidence]
    reason: str


class Transitions(Record):
    items: list[Transition]


class RiskEvent(Record):
    event_id: str
    risk_type: Literal['scope_inflation', 'false_antecedence', 'semantic_causal', 'other_new_error',
                       'unsupported_downgrade', 'wrong_modification', 'context_dropped', 'omission_only']
    branch_unit_keys: list[str]
    full_unit_keys: list[str]
    branch_quotes: list[str]
    full_quote: str
    confirmed: bool
    evidence: list[Evidence]
    reason: str


class RiskAssessment(Record):
    risk_type: str
    eligible_unit_count: int
    assessment_complete: bool
    unresolved_unit_count: int
    applicability: bool


class Risks(Record):
    events: list[RiskEvent]
    assessments: list[RiskAssessment]


class IdentityGroup(Record):
    member_ids: list[str]
    summary: str
    scope: str
    kind: Literal['historical_increment', 'cross_work_relation', 'scope_correction', 'cross_contribution', 'error']


class IdentityGroups(Record):
    groups: list[IdentityGroup]


class IdentityLink(Record):
    candidate_id: str
    representative_id: str


class IdentityLinks(Record):
    items: list[IdentityLink]


class BranchPresence(Record):
    full_unit_key: str
    already_in_branch: bool
    complete: bool
    matched_finding_keys: list[str]
    reason: str


class BranchPresences(Record):
    items: list[BranchPresence]

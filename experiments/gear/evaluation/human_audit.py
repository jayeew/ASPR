"""Blinded Nature-only human audit selection, scoring, and release gates."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Literal, Sequence

from pydantic import Field, model_validator

from gear.contracts import StrictModel

IssueStatus = Literal["persists", "partially_resolved", "resolved", "unverifiable"]
RelationLabel = Literal["DIRECT", "PARTIAL", "PARALLEL", "DISTANT", "UNVERIFIABLE"]
Variant = Literal["neutral", "score", "score_topology", "placebo_graph"]

RELATION_GAIN: dict[str, int] = {
    "DIRECT": 3,
    "PARTIAL": 2,
    "PARALLEL": 1,
    "DISTANT": 0,
    "UNVERIFIABLE": 0,
}


class TraceableIssue(StrictModel):
    issue_id: str
    aspect: Literal["novelty_prior_art", "contribution"]
    status: IssueStatus
    reviewer_quote_keys: list[str] = Field(min_length=1)
    reviewer_ids: list[str] = Field(min_length=1)
    round_ids: list[str] = Field(min_length=1)
    author_response_keys: list[str] = Field(default_factory=list)
    final_paper_evidence_keys: list[str] = Field(min_length=1)


class AuditEligibleCase(StrictModel):
    paper_id: str
    percentile: float = Field(ge=0.0, le=100.0)
    cutoff_safe: bool
    trace_complete: bool
    issues: list[TraceableIssue] = Field(min_length=1)

    @property
    def preferred_status(self) -> bool:
        return any(
            issue.status in {"persists", "partially_resolved"} for issue in self.issues
        )


class AuditCandidate(StrictModel):
    candidate_id: str
    work_id: str
    title: str
    target_claim: str
    candidate_excerpt: str
    relation_rationale: str
    correction_text: str | None = None


class BlindedVariant(StrictModel):
    alias: Literal["A", "B", "C", "D"]
    candidates: list[AuditCandidate]


class BlindedAuditPack(StrictModel):
    contract: Literal["gear_blinded_graph_audit_pack"] = "gear_blinded_graph_audit_pack"
    paper_id: str
    issue_ids: list[str] = Field(min_length=1)
    variants: list[BlindedVariant] = Field(min_length=4, max_length=4)
    sealed_variant_key_sha256: str

    @model_validator(mode="after")
    def unique_aliases(self) -> BlindedAuditPack:
        if {row.alias for row in self.variants} != {"A", "B", "C", "D"}:
            raise ValueError("audit pack must contain four blinded aliases")
        return self


class HumanCandidateJudgment(StrictModel):
    candidate_id: str
    relation: RelationLabel
    claim_relevant: bool
    material_review_change: bool
    wrong_paper_contamination: bool = False


class HumanVariantJudgment(StrictModel):
    paper_id: str
    variant: Variant
    judgments: list[HumanCandidateJudgment]
    unsupported_major_or_critical: int = Field(default=0, ge=0)
    post_cutoff_evidence: int = Field(default=0, ge=0)
    issue_recall: float = Field(ge=0.0, le=1.0)
    major_evidence_precision: float = Field(ge=0.0, le=1.0)
    resolved_resurrection: int = Field(default=0, ge=0)


class HumanVariantMetrics(StrictModel):
    paper_id: str
    variant: Variant
    ndcg_at_10: float = Field(ge=0.0, le=1.0)
    precision_at_5: float = Field(ge=0.0, le=1.0)
    valid_relation_count: int = Field(ge=0)
    material_change_count: int = Field(ge=0)
    issue_recall: float = Field(ge=0.0, le=1.0)
    major_evidence_precision: float = Field(ge=0.0, le=1.0)
    resolved_resurrection: int = Field(ge=0)
    safety_violations: int = Field(ge=0)


def stable_key(value: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}|{value}".encode()).hexdigest()


def select_quick_gate(cases: Sequence[AuditEligibleCase]) -> list[AuditEligibleCase]:
    eligible = sorted(
        (row for row in cases if row.cutoff_safe and row.trace_complete),
        key=lambda row: (row.percentile, stable_key(row.paper_id, "gear-quick3")),
    )
    if len(eligible) < 3:
        raise ValueError("quick gate requires at least three eligible Nature cases")
    targets = (0, (len(eligible) - 1) // 2, len(eligible) - 1)
    selected: list[AuditEligibleCase] = []
    for target in targets:
        ordered = sorted(
            (
                row
                for row in eligible
                if row.paper_id not in {x.paper_id for x in selected}
            ),
            key=lambda row: (
                not row.preferred_status,
                abs(eligible.index(row) - target),
                stable_key(row.paper_id, "gear-quick3"),
            ),
        )
        selected.append(ordered[0])
    return selected


def select_engineering_gate(
    cases: Sequence[AuditEligibleCase],
    quick_cases: Sequence[AuditEligibleCase],
) -> list[AuditEligibleCase]:
    eligible = sorted(
        (row for row in cases if row.cutoff_safe and row.trace_complete),
        key=lambda row: (row.percentile, stable_key(row.paper_id, "gear-confirm10")),
    )
    if len(eligible) < 10:
        raise ValueError("engineering gate requires at least ten eligible Nature cases")
    boundaries = (len(eligible) // 3, 2 * len(eligible) // 3)
    strata = (
        eligible[: boundaries[0]],
        eligible[boundaries[0] : boundaries[1]],
        eligible[boundaries[1] :],
    )
    quotas = (3, 4, 3)
    selected = list(dict.fromkeys(row.paper_id for row in quick_cases))
    by_id = {row.paper_id: row for row in eligible}
    if any(paper_id not in by_id for paper_id in selected):
        raise ValueError("quick-gate case is no longer eligible")
    chosen = [by_id[paper_id] for paper_id in selected]
    for stratum, quota in zip(strata, quotas):
        present = sum(row in stratum for row in chosen)
        candidates = sorted(
            (row for row in stratum if row.paper_id not in selected),
            key=lambda row: _status_priority(row, chosen),
        )
        for row in candidates[: max(0, quota - present)]:
            chosen.append(row)
            selected.append(row.paper_id)
    if len(chosen) != 10:
        raise ValueError("could not satisfy stable 3/4/3 sampling quotas")
    counts = Counter(issue.status for row in chosen for issue in row.issues)
    if any(
        counts[status] < 2 for status in ("resolved", "partially_resolved", "persists")
    ):
        raise ValueError(
            "selected cases do not cover two issues in each required status"
        )
    return chosen


def _status_priority(
    candidate: AuditEligibleCase, selected: Sequence[AuditEligibleCase]
) -> tuple[int, str]:
    counts = Counter(issue.status for row in selected for issue in row.issues)
    deficits = {
        status
        for status in ("resolved", "partially_resolved", "persists")
        if counts[status] < 2
    }
    covers = any(issue.status in deficits for issue in candidate.issues)
    return (not covers, stable_key(candidate.paper_id, "gear-confirm10"))


def rank_metrics(judgment: HumanVariantJudgment) -> HumanVariantMetrics:
    gains = [
        RELATION_GAIN[row.relation] if row.claim_relevant else 0
        for row in judgment.judgments[:10]
    ]
    ideal = sorted(gains, reverse=True)
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    idcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(ideal))
    top5 = judgment.judgments[:5]
    valid = [
        row
        for row in judgment.judgments
        if row.claim_relevant and RELATION_GAIN[row.relation] > 0
    ]
    return HumanVariantMetrics(
        paper_id=judgment.paper_id,
        variant=judgment.variant,
        ndcg_at_10=(dcg / idcg if idcg else 0.0),
        precision_at_5=(
            sum(row.claim_relevant and RELATION_GAIN[row.relation] > 0 for row in top5)
            / len(top5)
            if top5
            else 0.0
        ),
        valid_relation_count=len(valid),
        material_change_count=sum(
            row.material_review_change for row in judgment.judgments
        ),
        issue_recall=judgment.issue_recall,
        major_evidence_precision=judgment.major_evidence_precision,
        resolved_resurrection=judgment.resolved_resurrection,
        safety_violations=(
            judgment.unsupported_major_or_critical
            + judgment.post_cutoff_evidence
            + sum(row.wrong_paper_contamination for row in judgment.judgments)
        ),
    )


def quick_gate_passes(rows: Sequence[HumanVariantMetrics]) -> bool:
    grouped = _group(rows)
    if set(grouped) != {"neutral", "score", "score_topology", "placebo_graph"}:
        raise ValueError("quick gate requires all four variants")
    papers = set(grouped["neutral"])
    if len(papers) != 3 or any(set(grouped[name]) != papers for name in grouped):
        raise ValueError("quick gate requires the same three papers in every variant")
    ndcg_wins = sum(
        grouped["score_topology"][paper].ndcg_at_10
        > grouped["neutral"][paper].ndcg_at_10
        for paper in papers
    )
    unique_valid = sum(
        grouped["score_topology"][paper].valid_relation_count
        > grouped["neutral"][paper].valid_relation_count
        for paper in papers
    )
    return (
        ndcg_wins >= 2
        and unique_valid >= 2
        and all(row.safety_violations == 0 for row in rows)
    )


def engineering_gate(rows: Sequence[HumanVariantMetrics]) -> dict[str, bool]:
    grouped = _group(rows)
    papers = set(grouped.get("neutral", {}))
    if len(papers) != 10 or any(
        set(grouped.get(name, {})) != papers
        for name in ("score", "score_topology", "placebo_graph")
    ):
        raise ValueError("engineering gate requires four variants for ten papers")
    score_delta = [
        grouped["score"][paper].ndcg_at_10 - grouped["neutral"][paper].ndcg_at_10
        for paper in papers
    ]
    topology_delta = [
        grouped["score_topology"][paper].ndcg_at_10 - grouped["score"][paper].ndcg_at_10
        for paper in papers
    ]
    placebo_delta = [
        grouped["placebo_graph"][paper].ndcg_at_10 - grouped["score"][paper].ndcg_at_10
        for paper in papers
    ]
    score_default = (
        _median(score_delta) > 0.0
        and sum(value >= 0.0 for value in score_delta) >= 7
        and _median_metric(grouped["score"], "precision_at_5")
        >= _median_metric(grouped["neutral"], "precision_at_5")
        and _median_metric(grouped["score"], "major_evidence_precision")
        >= _median_metric(grouped["neutral"], "major_evidence_precision")
        and _median_metric(grouped["score"], "issue_recall")
        >= _median_metric(grouped["neutral"], "issue_recall")
    )
    unique_topology = sum(
        grouped["score_topology"][paper].valid_relation_count
        > grouped["score"][paper].valid_relation_count
        or grouped["score_topology"][paper].material_change_count
        > grouped["score"][paper].material_change_count
        for paper in papers
    )
    topology_default = (
        _median(topology_delta) > 0.0
        and unique_topology >= 3
        and _median(placebo_delta) < _median(topology_delta)
        and all(row.safety_violations == 0 for row in rows)
    )
    return {"score_default": score_default, "topology_default": topology_default}


def _group(
    rows: Sequence[HumanVariantMetrics],
) -> dict[str, dict[str, HumanVariantMetrics]]:
    grouped: dict[str, dict[str, HumanVariantMetrics]] = {}
    for row in rows:
        bucket = grouped.setdefault(row.variant, {})
        if row.paper_id in bucket:
            raise ValueError("duplicate human metric row")
        bucket[row.paper_id] = row
    return grouped


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )


def _median_metric(rows: dict[str, HumanVariantMetrics], field: str) -> float:
    return _median([float(getattr(row, field)) for row in rows.values()])


__all__ = [
    "AuditEligibleCase",
    "BlindedAuditPack",
    "HumanVariantJudgment",
    "HumanVariantMetrics",
    "TraceableIssue",
    "engineering_gate",
    "quick_gate_passes",
    "rank_metrics",
    "select_engineering_gate",
    "select_quick_gate",
]

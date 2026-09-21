"""Restore validated evidence into a new, separately archived GEAR attempt."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from gear.contracts import RelationCard, RetrievalBudget, RetrievedWork
from gear.review_contracts import GearClaim, GearClaimCard
from gear.trace import EvidenceStore

if TYPE_CHECKING:
    from gear.evidence_supervisor import EvidenceSupervisor


def restore_evidence(
    supervisor: EvidenceSupervisor,
    source: Path,
    claim: GearClaim,
    cutoff: date,
    works: dict[str, RetrievedWork],
    relations: dict[str, RelationCard],
    budget: RetrievalBudget,
) -> tuple[bool, bool]:
    old_card = GearClaimCard.model_validate_json(
        (source / "gear_card.json").read_text(encoding="utf-8")
    )
    if old_card.claim != claim:
        raise ValueError("Recovery claim identity/content mismatch")
    old = EvidenceStore(source)
    row = old.get(f"COVERAGE:{claim.claim_id}")
    if row is None or row.payload["cutoff_date"] != cutoff.isoformat():
        raise ValueError("Recovery cutoff/coverage mismatch")
    cov = row.payload
    preserved = {
        "manuscript_span",
        "retrieved_work",
        "retrieved_work_fulltext",
        "fulltext_acquisition",
        "relation_card",
        "target_version_exclusion",
    }
    for key in old.ids():
        record = old.get(key)
        if record is None:
            continue
        if record.kind not in preserved:
            continue
        supervisor.store.add_evidence(key, record.kind, record.payload)
        if record.kind in {"retrieved_work", "retrieved_work_fulltext"}:
            work = RetrievedWork.model_validate(record.payload)
            if work.target_claim_id != claim.claim_id:
                raise ValueError("Recovery work identity mismatch")
            if work.work_id not in works or record.kind == "retrieved_work_fulltext":
                works[work.work_id] = work
        elif record.kind == "relation_card":
            relation = RelationCard.model_validate(record.payload)
            if relation.target_claim_id != claim.claim_id:
                raise ValueError("Recovery relation identity mismatch")
            relations[relation.prior_work_id] = relation
        elif record.kind == "fulltext_acquisition":
            supervisor._fulltext_requested.add(key.split(f"{claim.claim_id}:", 1)[-1])
        elif record.kind == "target_version_exclusion":
            supervisor._identity_exclusions.append(record.payload["work"]["work_id"])
    state = {
        "cutoff_date": cutoff.isoformat(),
        "roles": set(cov["completed_query_roles"]),
        "query_ids": set(cov["query_ids"]),
        "retrieved": cov["retrieved_count"],
        "temporal": cov["temporal_excluded_count"],
        "metadata": cov["metadata_only_count"],
        # Unknown discarded candidate IDs cannot be reconstructed from counts.
        "eligible_ids": set(works),
        "compared_ids": set(relations),
        "whole_ranked": cov["whole_paper_ranking_completed"],
        "purpose_ranked": cov["purpose_ranking_completed"],
        "ranker": cov["ranker"],
        "degraded": cov["degraded"],
        "service_failed": cov["service_failed"],
        "exhaustive": cov["exhaustive_provider_results"],
        "advisories": [
            x
            for x in cov["advisory_notes"]
            if not x.startswith("contrastive_query_coverage_gap:")
        ]
        + ["resumed_unique_eligible_count_is_lower_bound"],
        "prior_eligible_count": max(
            0,
            cov["unique_eligible_count"]
            - len(set(supervisor._identity_exclusions) & set(cov["compared_work_ids"])),
        ),
    }
    supervisor.prior_art._coverage_state[claim.claim_id] = state
    budget.fulltext_kept = len(works)
    roles = state["roles"]
    normal_done = (
        len(
            roles
            & {
                "author_terminology",
                "object_problem",
                "mechanism_outcome",
                "purpose_semantic",
            }
        )
        >= 3
    )
    contrastive_done = "legacy_contrastive" in roles
    if normal_done:
        budget.normal_used = budget.normal_max
    if contrastive_done:
        budget.contrastive_used = budget.contrastive_max
    return normal_done, contrastive_done

"""Claim-scoped, bounded GEAR prior-art verification."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from datetime import date
from pathlib import Path

from gear.config import GearConfig
from gear.contracts import (
    ClaimStrength,
    ClaimType,
    EvidenceSpan,
    PaperClaim,
    PaperIR,
    RelationCard,
    RelationLabel,
    RetrievalBudget,
    RetrievedWork,
)
from gear.local_ranking import LocalScientificRanker
from gear.model_client import LazyRoleClient
from gear.prior_art import PriorArtService, RelationClassifier
from gear.trace import EvidenceStore, sha256_value
from gear.work_identity import version_identity

from .review_contracts import (
    GearClaim,
    GearClaimCard,
    GearEvidenceStatus,
    InternalSupportStatus,
    SupervisorAction,
)


def _digest(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


MAX_CANDIDATE_WORKS_PER_CLAIM = 10


def _claim_type(_: GearClaim) -> ClaimType:
    return ClaimType.NOVELTY


class EvidenceSupervisor:
    """Run a finite evidence loop; the planner proposes but cannot exceed budgets."""

    def __init__(
        self,
        config: GearConfig,
        store: EvidenceStore,
        *,
        local_ranker: LocalScientificRanker | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.prior_art = PriorArtService(config, local_ranker=local_ranker)
        self.classifier = RelationClassifier(config)
        self.planner = LazyRoleClient(config, "supervisor_planner")
        self.execution_errors: list[str] = []
        self._fulltext_requested: set[str] = set()

    def _record_identity(
        self, claim: GearClaim, work: RetrievedWork, reason: str
    ) -> None:
        payload = {"work": work.model_dump(mode="json"), "reason": reason}
        # Repeated retrievals can update query provenance or fulltext of the same work.
        digest = sha256_value(payload).split(":")[-1]
        self.store.add_evidence(
            f"IDENTITY:{claim.claim_id}:{work.work_id}:{digest}",
            "target_version_exclusion",
            payload,
        )

    def evaluate(
        self,
        claim: GearClaim,
        paper: PaperIR,
        cutoff: date,
        *,
        seed_work_ids: list[str] | None = None,
        recovery_source: Path | None = None,
    ) -> GearClaimCard:
        from gear.innovation.usage import log_progress

        log_progress("[Claim开始] internal_support=%s", claim.internal_support.value)
        if claim.internal_support is InternalSupportStatus.UNSUPPORTED:
            log_progress("[Claim结束] 正文证据不足，跳过外部检索")
            return GearClaimCard(
                claim=claim,
                status=GearEvidenceStatus.INTERNALLY_UNSUPPORTED,
                summary="该主张未被论文内部证据支持，未进入外部先验检索。",
                limitations=["internal_evidence_failed"],
            )
        target_span, paper_claim = self._adapters(claim, paper)
        budget = RetrievalBudget(
            normal_max=self.config.retrieval.normal_max,
            contrastive_max=self.config.retrieval.contrastive_max,
            citation_expansion_max=self.config.retrieval.citation_expansion_max,
            fulltext_max=min(
                MAX_CANDIDATE_WORKS_PER_CLAIM,
                self.config.retrieval.fulltext_max,
            ),
        )
        self._target_metadata = paper.metadata
        self._identity_exclusions: list[str] = []
        works: dict[str, RetrievedWork] = {}
        self._works = works
        relations: dict[str, RelationCard] = {}
        actions: list[SupervisorAction] = []
        normal_done = contrastive_done = expanded_done = stability_done = False
        if recovery_source is not None:
            from gear.innovation.gear_resume import restore_evidence

            normal_done, contrastive_done = restore_evidence(
                self, recovery_source, claim, cutoff, works, relations, budget
            )
            log_progress(
                "[补检索证据复用] 文献=%d，关系=%d", len(works), len(relations)
            )
        for _ in range(12):
            unclassified = [
                work for key, work in works.items() if key not in relations
            ][: max(0, MAX_CANDIDATE_WORKS_PER_CLAIM - len(relations))]
            antecedents = self._antecedents(relations.values())
            if not normal_done:
                legal = ["normal_search"]
            elif unclassified:
                legal = ["verify_relation"]
            elif not contrastive_done:
                legal = ["contrastive_search"]
            elif (
                antecedents
                and self.config.relation_stability_check_enabled
                and not stability_done
            ):
                legal = ["stability_check"]
            elif not antecedents and works and not expanded_done:
                legal = ["citation_expand", "finalize"]
            else:
                legal = ["finalize"]
            action, reason = self._choose_action(claim, legal, works, relations)
            if action == "normal_search":
                self._search(
                    action,
                    paper_claim,
                    target_span,
                    paper,
                    cutoff,
                    budget,
                    works,
                    actions,
                    seed_work_ids or [],
                )
                normal_done = True
            elif action == "contrastive_search":
                self._search(
                    action,
                    paper_claim,
                    target_span,
                    paper,
                    cutoff,
                    budget,
                    works,
                    actions,
                    [],
                    family="contrastive",
                )
                contrastive_done = True
            elif action == "verify_relation":
                self._classify(
                    target_span, claim, unclassified, cutoff, relations, actions
                )
            elif action == "citation_expand":
                seed = max(works.values(), key=lambda work: bool(work.abstract))
                found = self.prior_art.expand_neighbors(
                    seed, paper_claim, cutoff, budget
                )
                self._record_retrieval_errors()
                for work in found:
                    if (
                        work.work_id not in works
                        and len(works) >= MAX_CANDIDATE_WORKS_PER_CLAIM
                    ):
                        continue
                    identity = version_identity(work, paper.metadata)
                    if identity:
                        self._identity_exclusions.append(work.work_id)
                        self._record_identity(claim, work, identity)
                        continue
                    is_new = work.work_id not in works
                    works[work.work_id] = work
                    if not is_new:
                        continue
                    self.store.add_evidence(
                        f"WORK:{claim.claim_id}:{work.work_id}",
                        "retrieved_work",
                        work.model_dump(mode="json"),
                    )
                actions.append(
                    SupervisorAction(
                        step=len(actions) + 1,
                        action=action,
                        reason=reason,
                        input_ids=[seed.work_id],
                        output_ids=[x.work_id for x in found],
                    )
                )
                expanded_done = True
            elif action == "stability_check":
                self._stability_check(
                    target_span, claim, cutoff, works, relations, actions
                )
                stability_done = True
            else:
                actions.append(
                    SupervisorAction(
                        step=len(actions) + 1, action="finalize", reason=reason
                    )
                )
                break
            if actions:
                actions[-1].reason = reason
        coverage = self.prior_art.coverage_card(
            claim.claim_id,
            cutoff,
            require_contrastive=True,
            direct_or_partial_found=bool(self._antecedents(relations.values())),
        )
        coverage_key = f"COVERAGE:{claim.claim_id}"
        self.store.add_evidence(
            coverage_key, "retrieval_coverage", coverage.model_dump(mode="json")
        )
        if not actions or actions[-1].action != "finalize":
            actions.append(
                SupervisorAction(
                    step=len(actions) + 1,
                    action="finalize",
                    reason="Deterministic action limit reached.",
                )
            )
        card = self._card(
            claim,
            list(relations.values()),
            coverage_key,
            coverage.coverage_sufficient and not self._identity_exclusions,
            works,
            actions,
        )
        log_progress(
            "[Claim证据完成] 历史文献=%d，关系=%d，status=%s",
            len(works),
            len(relations),
            card.status.value,
        )
        return card

    def _adapters(
        self, claim: GearClaim, paper: PaperIR
    ) -> tuple[EvidenceSpan, PaperClaim]:
        primary_id = (
            claim.support_span_ids[0]
            if claim.support_span_ids
            else claim.source_span_ids[0]
        )
        primary = paper.span_map()[primary_id]
        return primary, PaperClaim(
            claim_id=claim.claim_id,
            claim_type=_claim_type(claim),
            span_id=primary.span_id,
            text=claim.normalized_claim_text,
            strength=ClaimStrength.MODERATE,
            dependency_span_ids=list(
                dict.fromkeys(claim.source_span_ids + claim.support_span_ids)
            ),
        )

    def _search(
        self,
        name: str,
        claim: PaperClaim,
        span: EvidenceSpan,
        paper: PaperIR,
        cutoff: date,
        budget: RetrievalBudget,
        works: dict[str, RetrievedWork],
        actions: list[SupervisorAction],
        seeds: list[str],
        family: str = "normal",
    ) -> None:
        from gear.innovation.usage import log_progress, progress_scope

        log_progress("[检索开始] family=%s", family)
        with progress_scope(f"retrieval={family}"):
            found = self.prior_art.retrieve(
                claim,
                cutoff,
                budget,
                family=family,
                target_span=span,
                paper_ir=paper,
                graph_seed_work_ids=seeds,
                graph_neighbor_slots=len(seeds),
            )
        self._record_retrieval_errors()
        log_progress(
            "[检索完成] family=%s，候选=%d，查询=%d，失败=%d",
            family,
            len(found),
            len(self.prior_art.last_query_specs),
            len(self.prior_art.last_failures),
        )
        for work in found:
            if (
                work.work_id not in works
                and len(works) >= MAX_CANDIDATE_WORKS_PER_CLAIM
            ):
                continue
            identity = version_identity(work, self._target_metadata)
            if identity:
                self._identity_exclusions.append(work.work_id)
                self._record_identity(claim, work, identity)
                continue
            is_new = work.work_id not in works
            if not is_new:
                continue
            works[work.work_id] = work
            self.store.add_evidence(
                f"WORK:{claim.claim_id}:{work.work_id}",
                "retrieved_work",
                work.model_dump(mode="json"),
            )
        actions.append(
            SupervisorAction(
                step=len(actions) + 1,
                action=name,
                reason="Bounded evidence acquisition.",
                input_ids=seeds,
                output_ids=[x.work_id for x in found],
            )
        )

    def _record_retrieval_errors(self) -> None:
        self.execution_errors.extend(self.prior_art.last_failures)
        failure_prefixes = (
            "query_planner_degraded:",
            "comparability_audit_degraded:",
            "local_ranker_degraded:",
            "global_ranker_degraded:",
        )
        self.execution_errors.extend(
            note
            for note in self.prior_art.last_advisories
            if note.startswith(failure_prefixes)
        )

    def _classify(
        self,
        span: EvidenceSpan,
        claim: GearClaim,
        works: Iterable[RetrievedWork],
        cutoff: date,
        relations: dict[str, RelationCard],
        actions: list[SupervisorAction],
    ) -> None:
        new_ids: list[str] = []
        remaining = max(0, MAX_CANDIDATE_WORKS_PER_CLAIM - len(relations))
        pending = [work for work in works if work.work_id not in relations][:remaining]

        from gear.innovation.usage import log_progress, progress_scope

        pending = self._prepare_fulltext(claim, pending)
        log_progress("[关系批量判断开始] 文献=%d", len(pending))
        with progress_scope("operation=relation_batch"):
            cards = self.classifier.classify_many(
                span,
                pending,
                target_claim_id=claim.claim_id,
                cutoff=cutoff,
                target_claim_text=claim.normalized_claim_text,
            )
        log_progress("[关系批量判断完成] 文献=%d", len(cards))
        if getattr(self.classifier, "last_failure", None):
            self.execution_errors.append(
                f"relation_model:{self.classifier.last_failure}"
            )
        for work, card in zip(pending, cards):
            relations[work.work_id] = card
            key = f"RELATION:{claim.claim_id}:{work.work_id}"
            self.store.add_evidence(key, "relation_card", card.model_dump(mode="json"))
            new_ids.append(key)
        if new_ids:
            actions.append(
                SupervisorAction(
                    step=len(actions) + 1,
                    action="verify_relation",
                    reason="Paired target/prior text verification.",
                    output_ids=new_ids,
                )
            )

    def _prepare_fulltext(
        self, claim: GearClaim, pending: list[RetrievedWork]
    ) -> list[RetrievedWork]:
        from gear.innovation.usage import log_progress

        limits = self.config.retrieval
        if not (limits.external_fulltext_enabled or limits.openalex_pdf_enabled):
            log_progress("[历史全文下载关闭] 使用已有摘要；文献=%d", len(pending))
            return pending
        maximum = (
            limits.external_fulltext_max_works
            if limits.external_fulltext_enabled
            else limits.openalex_pdf_max_downloads
        )
        selected = [
            work for work in pending if work.work_id not in self._fulltext_requested
        ][: max(0, maximum - len(self._fulltext_requested))]
        self._fulltext_requested.update(work.work_id for work in selected)
        upgraded_by_id: dict[str, RetrievedWork] = {}
        log_progress(
            "[历史全文队列] 本批=%d，Claim预算=%d，外部来源=%s",
            len(selected),
            maximum,
            limits.external_fulltext_enabled,
        )
        with ThreadPoolExecutor(max_workers=2) as executor:
            jobs = [
                (
                    work,
                    executor.submit(
                        copy_context().run,
                        self.prior_art.upgrade_fulltext,
                        work,
                        claim.normalized_claim_text,
                    ),
                )
                for work in selected
            ]
            for work, future in jobs:
                upgraded = future.result()
                attempt = self.prior_art.fulltext_attempts.get(work.work_id)
                if attempt:
                    self.store.add_evidence(
                        f"FULLTEXT_FETCH:{claim.claim_id}:{work.work_id}",
                        "fulltext_acquisition",
                        attempt,
                    )
                    log_progress(
                        "[外部全文结果] work_id=%s，status=%s，provider=%s，cache=%s",
                        work.work_id,
                        attempt.get("status"),
                        attempt.get("provider"),
                        attempt.get("cache_hit"),
                    )
                if upgraded is not work:
                    self.store.add_evidence(
                        f"FULLTEXT:{claim.claim_id}:{work.work_id}",
                        "retrieved_work_fulltext",
                        upgraded.model_dump(mode="json"),
                    )
                    self._works[work.work_id] = upgraded
                upgraded_by_id[work.work_id] = upgraded
        return [upgraded_by_id.get(work.work_id, work) for work in pending]

    @staticmethod
    def _antecedents(relations: object) -> list[RelationCard]:
        return [
            card
            for card in relations
            if card.relation_label
            in {RelationLabel.DIRECT_ANTECEDENT, RelationLabel.PARTIAL_ANTECEDENT}
        ]

    def _stability_check(
        self,
        span: EvidenceSpan,
        claim: GearClaim,
        cutoff: date,
        works: dict[str, RetrievedWork],
        relations: dict[str, RelationCard],
        actions: list[SupervisorAction],
    ) -> None:
        critical = self._antecedents(relations.values())[:2]
        stable: list[str] = []
        evidence_ids: list[str] = []
        for first in critical:
            work = works.get(first.prior_work_id)
            if work is None:
                continue
            second = self.classifier.classify(
                span,
                work,
                target_claim_id=claim.claim_id,
                cutoff=cutoff,
                target_claim_text=claim.normalized_claim_text,
                replicate="independent-relation-check-1",
            )
            evidence_id = f"STABILITY:{claim.claim_id}:{first.prior_work_id}"
            self.store.add_evidence(
                evidence_id,
                "relation_stability",
                {
                    "first": first.model_dump(mode="json"),
                    "second": second.model_dump(mode="json"),
                    "stable": second.relation_label == first.relation_label,
                },
            )
            evidence_ids.append(evidence_id)
            if second.relation_label == first.relation_label:
                stable.append(first.prior_work_id)
                continue
            relations[first.prior_work_id] = first.model_copy(
                update={
                    "relation_label": RelationLabel.UNRESOLVED,
                    "rationale": "Independent repeated classification was unstable; antecedence is unresolved.",
                }
            )
        if critical:
            actions.append(
                SupervisorAction(
                    step=len(actions) + 1,
                    action="stability_check",
                    reason="Repeated classification of conclusion-changing relations.",
                    input_ids=[x.prior_work_id for x in critical],
                    output_ids=evidence_ids,
                )
            )

    def _choose_action(
        self,
        claim: GearClaim,
        legal: list[str],
        works: dict[str, RetrievedWork],
        relations: dict[str, RelationCard],
    ) -> tuple[str, str]:
        if len(legal) == 1:
            return legal[0], "Only one legal action; selected without a planner call."
        try:
            raw = self.planner.generate_json(
                system="Choose one legal evidence action that most reduces uncertainty. Do not exceed the supplied action set. Return JSON.",
                user=json.dumps(
                    {
                        "claim_id": claim.claim_id,
                        "legal_actions": legal,
                        "retrieved_work_count": len(works),
                        "relation_labels": [
                            x.relation_label.value for x in relations.values()
                        ],
                    }
                ),
                response_schema={
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": legal},
                        "reason": {"type": "string"},
                    },
                    "required": ["action", "reason"],
                    "additionalProperties": False,
                },
            )
            return str(raw["action"]), str(raw["reason"])
        except (RuntimeError, TypeError, ValueError, KeyError) as exc:
            self.execution_errors.append(f"planner:{exc}")
            return (
                legal[0],
                f"Planner unavailable; deterministic legal action selected: {exc}",
            )

    def _card(
        self,
        claim: GearClaim,
        relations: list[RelationCard],
        coverage_key: str,
        sufficient: bool,
        works: dict[str, RetrievedWork],
        actions: list[SupervisorAction],
    ) -> GearClaimCard:
        antecedents = self._antecedents(relations)
        direct = [
            x
            for x in antecedents
            if x.relation_label is RelationLabel.DIRECT_ANTECEDENT
            and x.independent_verification_passed
        ]
        partial = [
            x
            for x in antecedents
            if x.relation_label is RelationLabel.PARTIAL_ANTECEDENT
        ]
        if direct:
            status, summary = (
                GearEvidenceStatus.ANTECEDENT_FOUND,
                "发现经文本对照和独立复核的直接先例。",
            )
        elif partial:
            status, summary = (
                GearEvidenceStatus.RESIDUAL_EXTENSION,
                "发现部分先例；差异维度构成待评估的剩余扩展。",
            )
        elif (
            sufficient
            and relations
            and all(
                x.temporal_valid and x.relation_label is not RelationLabel.UNRESOLVED
                for x in relations
            )
        ):
            status, summary = (
                GearEvidenceStatus.BOUNDED_NO_ANTECEDENT,
                "在明确检索边界内未发现可验证先例。",
            )
        else:
            status, summary = (
                GearEvidenceStatus.INCONCLUSIVE,
                "检索或证据覆盖不足，不能形成否定先例的结论。",
            )
        relation_keys = [
            f"RELATION:{claim.claim_id}:{x.prior_work_id}" for x in relations
        ]
        stability_keys = [
            key
            for action in actions
            for key in action.output_ids
            if key.startswith("STABILITY:")
        ]
        residual = (
            "; ".join(
                dict.fromkeys(d for x in partial for d in x.difference_dimensions)
            )
            or None
        )
        return GearClaimCard(
            claim=claim,
            status=status,
            summary=summary,
            strongest_relation=(
                (direct or partial or relations or [None])[0].relation_label.value
                if relations
                else None
            ),
            antecedent_work_ids=[x.prior_work_id for x in direct + partial],
            residual_contribution=residual,
            evidence_keys=[coverage_key, *relation_keys, *stability_keys],
            assessed_work_ids=sorted(works),
            actions=actions,
            limitations=[] if sufficient else ["retrieval_coverage_insufficient"],
        )

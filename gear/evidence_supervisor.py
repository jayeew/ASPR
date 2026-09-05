"""Claim-scoped, bounded GEAR prior-art verification."""

from __future__ import annotations

import hashlib
import json
from datetime import date

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
from gear.prior_art import PriorArtService, RelationClassifier
from gear.trace import EvidenceStore

from .review_contracts import (
    GearClaim,
    GearClaimCard,
    GearEvidenceStatus,
    InternalSupportStatus,
    SupervisorAction,
)
from gear.model_client import LazyRoleClient


def _digest(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _claim_type(_: GearClaim) -> ClaimType:
    return ClaimType.NOVELTY


class EvidenceSupervisor:
    """Run a finite evidence loop; the planner proposes but cannot exceed budgets."""

    def __init__(self, config: GearConfig, store: EvidenceStore) -> None:
        self.config = config
        self.store = store
        self.prior_art = PriorArtService(config)
        self.classifier = RelationClassifier(config)
        self.planner = LazyRoleClient(config, "supervisor_planner")

    def evaluate(
        self,
        claim: GearClaim,
        paper: PaperIR,
        cutoff: date,
        *,
        seed_work_ids: list[str] | None = None,
    ) -> GearClaimCard:
        if claim.internal_support is InternalSupportStatus.UNSUPPORTED:
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
            fulltext_max=self.config.retrieval.fulltext_max,
        )
        works: dict[str, RetrievedWork] = {}
        relations: dict[str, RelationCard] = {}
        actions: list[SupervisorAction] = []
        normal_done = contrastive_done = expanded_done = stability_done = False
        for _ in range(12):
            unclassified = [work for key, work in works.items() if key not in relations]
            antecedents = self._antecedents(relations.values())
            if not normal_done:
                legal = ["normal_search"]
            elif unclassified:
                legal = ["verify_relation"]
            elif not contrastive_done:
                legal = ["contrastive_search"]
            elif antecedents and not stability_done:
                legal = ["stability_check"]
            elif not antecedents and works and not expanded_done:
                legal = ["citation_expand", "finalize"]
            else:
                legal = ["finalize"]
            action, reason = self._choose_action(claim, legal, works, relations)
            if action == "normal_search":
                self._search(action, paper_claim, target_span, paper, cutoff, budget, works, actions, seed_work_ids or [])
                normal_done = True
            elif action == "contrastive_search":
                self._search(action, paper_claim, target_span, paper, cutoff, budget, works, actions, [], family="contrastive")
                contrastive_done = True
            elif action == "verify_relation":
                self._classify(target_span, claim, unclassified, cutoff, relations, actions)
            elif action == "citation_expand":
                seed = max(works.values(), key=lambda work: bool(work.abstract))
                found = self.prior_art.expand_neighbors(seed, paper_claim, cutoff, budget)
                for work in found:
                    is_new = work.work_id not in works
                    works[work.work_id] = work
                    if not is_new:
                        continue
                    self.store.add_evidence(f"WORK:{claim.claim_id}:{work.work_id}", "retrieved_work", work.model_dump(mode="json"))
                actions.append(SupervisorAction(step=len(actions) + 1, action=action, reason=reason, input_ids=[seed.work_id], output_ids=[x.work_id for x in found]))
                expanded_done = True
            elif action == "stability_check":
                self._stability_check(target_span, claim, cutoff, works, relations, actions)
                stability_done = True
            else:
                actions.append(SupervisorAction(step=len(actions) + 1, action="finalize", reason=reason))
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
        self.store.add_evidence(coverage_key, "retrieval_coverage", coverage.model_dump(mode="json"))
        if not actions or actions[-1].action != "finalize":
            actions.append(SupervisorAction(step=len(actions) + 1, action="finalize", reason="Deterministic action limit reached."))
        return self._card(claim, list(relations.values()), coverage_key, coverage.coverage_sufficient, works, actions)

    def _adapters(self, claim: GearClaim, paper: PaperIR) -> tuple[EvidenceSpan, PaperClaim]:
        primary_id = claim.support_span_ids[0] if claim.support_span_ids else claim.source_span_ids[0]
        primary = paper.span_map()[primary_id]
        span = EvidenceSpan(
            span_id=f"SYNTH:{claim.claim_id}", source_id=paper.paper_id,
            page=primary.page, section_path=primary.section_path, char_start=0,
            char_end=len(claim.normalized_claim_text), text=claim.normalized_claim_text,
            text_sha256=_digest(claim.normalized_claim_text),
        )
        return span, PaperClaim(
            claim_id=claim.claim_id, claim_type=_claim_type(claim), span_id=span.span_id,
            text=claim.normalized_claim_text, strength=ClaimStrength.MODERATE,
            dependency_span_ids=claim.support_span_ids,
        )

    def _search(self, name: str, claim: PaperClaim, span: EvidenceSpan, paper: PaperIR,
                cutoff: date, budget: RetrievalBudget, works: dict[str, RetrievedWork],
                actions: list[SupervisorAction], seeds: list[str], family: str = "normal") -> None:
        found = self.prior_art.retrieve(
            claim, cutoff, budget, family=family, target_span=span, paper_ir=paper,
            graph_seed_work_ids=seeds, graph_neighbor_slots=len(seeds),
        )
        for work in found:
            is_new = work.work_id not in works
            works[work.work_id] = work
            if not is_new:
                continue
            self.store.add_evidence(f"WORK:{claim.claim_id}:{work.work_id}", "retrieved_work", work.model_dump(mode="json"))
        actions.append(SupervisorAction(step=len(actions) + 1, action=name, reason="Bounded evidence acquisition.", input_ids=seeds, output_ids=[x.work_id for x in found]))

    def _classify(self, span: EvidenceSpan, claim: GearClaim,
                  works: object, cutoff: date, relations: dict[str, RelationCard],
                  actions: list[SupervisorAction]) -> None:
        new_ids: list[str] = []
        for work in works:
            if work.work_id in relations:
                continue
            card = self.classifier.classify(span, work, target_claim_id=claim.claim_id, cutoff=cutoff)
            relations[work.work_id] = card
            key = f"RELATION:{claim.claim_id}:{work.work_id}"
            self.store.add_evidence(key, "relation_card", card.model_dump(mode="json"))
            new_ids.append(key)
        if new_ids:
            actions.append(SupervisorAction(step=len(actions) + 1, action="verify_relation", reason="Paired target/prior text verification.", output_ids=new_ids))

    @staticmethod
    def _antecedents(relations: object) -> list[RelationCard]:
        return [card for card in relations if card.relation_label in {RelationLabel.DIRECT_ANTECEDENT, RelationLabel.PARTIAL_ANTECEDENT}]

    def _stability_check(self, span: EvidenceSpan, claim: GearClaim, cutoff: date,
                         works: dict[str, RetrievedWork], relations: dict[str, RelationCard],
                         actions: list[SupervisorAction]) -> None:
        critical = self._antecedents(relations.values())[:2]
        stable: list[str] = []
        evidence_ids: list[str] = []
        for first in critical:
            work = works.get(first.prior_work_id)
            if work is None:
                continue
            second = self.classifier.classify(span, work, target_claim_id=claim.claim_id, cutoff=cutoff)
            evidence_id = f"STABILITY:{claim.claim_id}:{first.prior_work_id}"
            self.store.add_evidence(evidence_id, "relation_stability", {
                "first": first.model_dump(mode="json"),
                "second": second.model_dump(mode="json"),
                "stable": second.relation_label == first.relation_label,
            })
            evidence_ids.append(evidence_id)
            if second.relation_label == first.relation_label:
                stable.append(first.prior_work_id)
                continue
            relations[first.prior_work_id] = first.model_copy(
                update={"relation_label": RelationLabel.UNRESOLVED,
                        "rationale": "Independent repeated classification was unstable; antecedence is unresolved."}
            )
        if critical:
            actions.append(SupervisorAction(
                step=len(actions) + 1, action="stability_check",
                reason="Repeated classification of conclusion-changing relations.",
                input_ids=[x.prior_work_id for x in critical], output_ids=evidence_ids,
            ))

    def _choose_action(self, claim: GearClaim, legal: list[str],
                       works: dict[str, RetrievedWork], relations: dict[str, RelationCard]) -> tuple[str, str]:
        try:
            raw = self.planner.generate_json(
                system="Choose one legal evidence action that most reduces uncertainty. Do not exceed the supplied action set. Return JSON.",
                user=json.dumps({"claim_id": claim.claim_id, "legal_actions": legal, "retrieved_work_count": len(works), "relation_labels": [x.relation_label.value for x in relations.values()]}),
                response_schema={"type": "object", "properties": {"action": {"type": "string", "enum": legal}, "reason": {"type": "string"}}, "required": ["action", "reason"], "additionalProperties": False},
            )
            return str(raw["action"]), str(raw["reason"])
        except (RuntimeError, TypeError, ValueError, KeyError) as exc:
            return legal[0], f"Planner unavailable; deterministic legal action selected: {exc}"

    def _card(self, claim: GearClaim, relations: list[RelationCard], coverage_key: str,
              sufficient: bool, works: dict[str, RetrievedWork], actions: list[SupervisorAction]) -> GearClaimCard:
        antecedents = self._antecedents(relations)
        direct = [x for x in antecedents if x.relation_label is RelationLabel.DIRECT_ANTECEDENT and x.independent_verification_passed]
        partial = [x for x in antecedents if x.relation_label is RelationLabel.PARTIAL_ANTECEDENT]
        if direct:
            status, summary = GearEvidenceStatus.ANTECEDENT_FOUND, "发现经文本对照和独立复核的直接先例。"
        elif partial:
            status, summary = GearEvidenceStatus.RESIDUAL_EXTENSION, "发现部分先例；差异维度构成待评估的剩余扩展。"
        elif sufficient:
            status, summary = GearEvidenceStatus.BOUNDED_NO_ANTECEDENT, "在明确检索边界内未发现可验证先例。"
        else:
            status, summary = GearEvidenceStatus.INCONCLUSIVE, "检索或证据覆盖不足，不能形成否定先例的结论。"
        relation_keys = [f"RELATION:{claim.claim_id}:{x.prior_work_id}" for x in relations]
        stability_keys = [key for action in actions for key in action.output_ids if key.startswith("STABILITY:")]
        residual = "; ".join(dict.fromkeys(d for x in partial for d in x.difference_dimensions)) or None
        return GearClaimCard(
            claim=claim, status=status, summary=summary,
            strongest_relation=(direct or partial or relations or [None])[0].relation_label.value if relations else None,
            antecedent_work_ids=[x.prior_work_id for x in direct + partial],
            residual_contribution=residual, evidence_keys=[coverage_key, *relation_keys, *stability_keys],
            assessed_work_ids=sorted(works), actions=actions,
            limitations=[] if sufficient else ["retrieval_coverage_insufficient"],
        )

"""Passive and active fusion of independently produced Graph and GEAR branches."""

from __future__ import annotations

import json
import re
from pathlib import Path

from gear.config import GearConfig
from gear.contracts import PaperIR
from gear.trace import EvidenceStore

from .review_contracts import (
    AlignmentLink,
    BranchStatus,
    FusionResult,
    GearBranchResult,
    GearClaim,
    GearEvidenceStatus,
    GraphBranchResult,
    InternalSupportStatus,
    JointInnovationClaimCard,
    InnovationPaperInput,
)
from .evidence_supervisor import EvidenceSupervisor
from gear.artifacts import read_model, write_model
from gear.model_client import LazyRoleClient


ALIGN_SYSTEM = """Align abstract Graph Claims with full-text GEAR Claims many-to-many.
Create a link only for substantively overlapping scientific contributions, not merely shared topic.
Allowed relations: equivalent, graph_broader, gear_broader, partial_overlap. Return JSON only."""


class BranchFusion:
    def __init__(self, config: GearConfig) -> None:
        self.client = LazyRoleClient(config, "relation_fusion")
        self.config = config

    def run(self, output_dir: Path, mode: str) -> FusionResult:
        fusion_dir = output_dir / "fusion"
        for name in ("evidence_trace.jsonl", "action_trace.jsonl", "state_trace.jsonl"):
            (fusion_dir / name).unlink(missing_ok=True)
        graph = read_model(output_dir / "graph" / "graph_branch_result.json", GraphBranchResult)
        gear = read_model(output_dir / "gear" / "gear_branch_result.json", GearBranchResult)
        fusion_limitations: list[str] = []
        try:
            links = self._align(graph, gear)
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
            links = []
            fusion_limitations.append(f"claim_alignment_unavailable:{exc}")
        recovered: list[str] = []
        rechecks: list[dict[str, object]] = []
        if mode == "active" and graph.status is not BranchStatus.FAILED and gear.status is not BranchStatus.FAILED:
            try:
                recovered, rechecks, gear = self._active(output_dir, graph, gear, links)
            except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
                fusion_limitations.append(f"active_fusion_limited:{exc}")
        cards = self._joint_cards(graph, gear, links)
        status = BranchStatus.COMPLETE
        limitations = [*graph.limitations, *gear.limitations, *fusion_limitations]
        if graph.status is not BranchStatus.COMPLETE or gear.status is not BranchStatus.COMPLETE or fusion_limitations:
            status = BranchStatus.LIMITED
        result = FusionResult(
            paper_id=graph.paper_id or gear.paper_id, mode=mode, status=status,
            alignments=links, joint_claim_cards=cards,
            recovered_claim_ids=recovered, graph_triggered_rechecks=rechecks,
            limitations=limitations,
        )
        mode_result_path = fusion_dir / f"fusion_result_{mode}.json"
        mode_report_path = fusion_dir / f"innovation_report_{mode}.md"
        write_model(mode_result_path, result)
        write_model(fusion_dir / "fusion_result.json", result)
        report = self._report(result, mode_report_path)
        result.report_path = str(report)
        write_model(mode_result_path, result)
        write_model(fusion_dir / "fusion_result.json", result)
        return result

    def _align(self, graph: GraphBranchResult, gear: GearBranchResult) -> list[AlignmentLink]:
        if not graph.claims or not gear.claims:
            return []
        raw = self.client.generate_json(
            system=ALIGN_SYSTEM,
            user=json.dumps({"graph_claims": [x.model_dump(mode="json") for x in graph.claims], "gear_claims": [x.model_dump(mode="json") for x in gear.claims]}, ensure_ascii=False),
            response_schema={"type": "object", "properties": {"links": {"type": "array", "items": {
                "type": "object", "properties": {
                    "graph_claim_id": {"type": "string"}, "gear_claim_id": {"type": "string"},
                    "relation": {"type": "string", "enum": ["equivalent", "graph_broader", "gear_broader", "partial_overlap"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}, "rationale": {"type": "string"},
                }, "required": ["graph_claim_id", "gear_claim_id", "relation", "confidence", "rationale"], "additionalProperties": False,
            }}}, "required": ["links"], "additionalProperties": False},
        )
        graph_ids = {x.claim_id for x in graph.claims}
        gear_ids = {x.claim_id for x in gear.claims}
        output = []
        for row in raw.get("links", []):
            if row.get("graph_claim_id") not in graph_ids or row.get("gear_claim_id") not in gear_ids:
                continue
            output.append(AlignmentLink.model_validate(row))
        return output

    def _active(self, output_dir: Path, graph: GraphBranchResult, gear: GearBranchResult,
                links: list[AlignmentLink]) -> tuple[list[str], list[dict[str, object]], GearBranchResult]:
        paper = read_model(output_dir / "gear" / "paper_ir.json", PaperIR)
        runtime_input = read_model(output_dir / "innovation_input.json", InnovationPaperInput)
        linked_graph = {x.graph_claim_id for x in links}
        recovered: list[str] = []
        for graph_claim in graph.claims:
            if graph_claim.claim_id in linked_graph:
                continue
            candidate = self._reground(graph_claim, paper)
            if candidate is not None:
                gear.claims.append(candidate)
                recovered.append(candidate.claim_id)
                links.append(AlignmentLink(graph_claim_id=graph_claim.claim_id, gear_claim_id=candidate.claim_id, relation="graph_recovered", confidence=1.0, rationale="Abstract claim was re-grounded in full text."))
        store = EvidenceStore(output_dir / "fusion")
        supervisor = EvidenceSupervisor(self.config, store)
        existing = {card.claim.claim_id: card for card in gear.claim_cards}
        rechecks: list[dict[str, object]] = []
        graph_facts = {fact.claim.claim_id: fact for fact in graph.fact_cards}
        for claim_id, card in list(existing.items()):
            linked_ids = [
                link.graph_claim_id
                for link in links
                if link.gear_claim_id == claim_id
            ]
            facts = [graph_facts[item] for item in linked_ids if item in graph_facts]
            if not facts:
                continue
            unseen = list(dict.fromkeys(
                neighbor.parent_openalex_work_id
                for fact in facts
                for neighbor in fact.neighbors
                if neighbor.parent_openalex_work_id
                and neighbor.parent_openalex_work_id not in card.assessed_work_ids
            ))
            if card.status not in {GearEvidenceStatus.BOUNDED_NO_ANTECEDENT, GearEvidenceStatus.INCONCLUSIVE} or not unseen:
                continue
            updated = supervisor.evaluate(card.claim, paper, runtime_input.cutoff_date, seed_work_ids=unseen[:10])
            existing[card.claim.claim_id] = updated
            rechecks.append({"gear_claim_id": card.claim.claim_id, "seed_work_ids": unseen[:10], "result_status": updated.status.value})
        for claim in gear.claims:
            if claim.claim_id in existing:
                continue
            existing[claim.claim_id] = supervisor.evaluate(claim, paper, runtime_input.cutoff_date)
        gear.claim_cards = list(existing.values())
        return recovered, rechecks, gear

    def _reground(self, graph_claim: object, paper: PaperIR) -> GearClaim | None:
        spans = [x for x in paper.spans if "reference" not in " ".join(x.section_path).casefold()]
        tokens = set(re.findall(r"[A-Za-z][A-Za-z0-9_-]+", graph_claim.claim_text.casefold()))
        ranked = sorted(
            spans,
            key=lambda span: len(tokens & set(re.findall(r"[A-Za-z][A-Za-z0-9_-]+", span.text.casefold()))),
            reverse=True,
        )[:80]
        raw = self.client.generate_json(
            system="Determine whether the abstract Claim is a central contribution explicitly supported by full-text spans. Do not infer. Return JSON.",
            user=json.dumps({"claim": graph_claim.model_dump(mode="json"), "spans": [{"span_id": x.span_id, "text": x.text} for x in ranked]}, ensure_ascii=False),
            response_schema={"type": "object", "properties": {"supported": {"type": "boolean"}, "normalized_claim_text": {"type": "string"}, "support_span_ids": {"type": "array", "items": {"type": "string"}}, "reason": {"type": "string"}}, "required": ["supported", "normalized_claim_text", "support_span_ids", "reason"], "additionalProperties": False},
        )
        known = {x.span_id for x in spans}
        ids = [str(x) for x in raw["support_span_ids"] if str(x) in known]
        if not raw["supported"] or not ids:
            return None
        return GearClaim(
            claim_id=graph_claim.claim_id.replace("::GRAPH::", "::RECOVERED::"),
            claim_type=graph_claim.claim_type, author_claim_text=graph_claim.claim_text,
            normalized_claim_text=str(raw["normalized_claim_text"]), source_span_ids=ids,
            support_span_ids=ids, internal_support=InternalSupportStatus.SUPPORTED,
            narrowing_reason=str(raw["reason"]),
        )

    @staticmethod
    def _joint_cards(graph: GraphBranchResult, gear: GearBranchResult,
                     links: list[AlignmentLink]) -> list[JointInnovationClaimCard]:
        graph_map = {x.claim.claim_id: x for x in graph.fact_cards}
        gear_map = {x.claim.claim_id: x for x in gear.claim_cards}
        groups: dict[str, list[AlignmentLink]] = {}
        for link in links:
            groups.setdefault(link.gear_claim_id, []).append(link)
        output = []
        for claim_id, card in gear_map.items():
            aligned = groups.get(claim_id, [])
            facts = [graph_map[x.graph_claim_id] for x in aligned if x.graph_claim_id in graph_map]
            output.append(JointInnovationClaimCard(
                joint_claim_id=f"JOINT:{claim_id}", graph_claim_ids=[x.graph_claim_id for x in aligned],
                gear_claim_ids=[claim_id], statement=card.claim.normalized_claim_text,
                evidence_status=card.status, graph_facts=[m for fact in facts for m in fact.metrics],
                graph_neighbor_ids=list(dict.fromkeys(n.claim_id for fact in facts for n in fact.neighbors)),
                evidence_keys=card.evidence_keys,
                interpretation=card.summary + BranchFusion._graph_interpretation(facts),
                limitations=card.limitations,
            ))
        return output

    @staticmethod
    def _graph_interpretation(facts: list[object]) -> str:
        if not facts:
            return ""
        metrics: dict[str, object] = {}
        path_count = 0
        for fact in facts:
            metrics.update({metric.name: metric.value for metric in fact.metrics})
            path_count += sum(
                neighbor.direct_citation or neighbor.two_hop_path_count > 0 or neighbor.shared_reference_count > 0
                for neighbor in fact.neighbors
            )
        nearest = metrics.get("nearest_prior_similarity")
        communities = metrics.get("effective_community_count")
        merge = metrics.get("component_merge_count")
        return (
            f" Graph 将该主张定位到历史 Claim 邻域：最近语义相似度={nearest}，"
            f"有效社区数={communities}，连接前分量合并数={merge}，"
            f"具有父论文书目通路的近邻={path_count}；这些是结构事实，不替代文本先验核验。"
        )

    @staticmethod
    def _report(result: FusionResult, path: Path) -> Path:
        lines = ["# 论文创新性评价", "", f"状态：{result.status.value}；融合模式：{result.mode}。", ""]
        for index, card in enumerate(result.joint_claim_cards, 1):
            lines.extend([f"## 创新 Claim {index}", "", card.statement, "", card.interpretation, "", f"GEAR 证据状态：{card.evidence_status.value if card.evidence_status else 'unavailable'}。", f"Graph 历史近邻数：{len(card.graph_neighbor_ids)}。", f"证据键：{', '.join(card.evidence_keys) or '无'}。", ""])
        if result.limitations:
            lines.extend(["## 局限", "", *[f"- {x}" for x in result.limitations], ""])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

"""Hierarchical full-text innovation Claim extraction and internal grounding."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Iterable

from gear.claim_graph.contracts import InnovationClaimType
from gear.config import GearConfig
from gear.contracts import EvidenceSpan, PaperIR

from .review_contracts import (
    ClaimCandidate,
    GearClaim,
    InternalSupportStatus,
)
from gear.model_client import LazyRoleClient


MINER_SYSTEM = """You extract candidate scientific contributions from one manuscript chunk.
Use only supplied spans. Extract what the authors present as a method, finding, mechanism,
resource, or theory contribution. Do not judge novelty and do not import outside facts.
Every candidate must cite exact span_ids. Return JSON only."""

CONSOLIDATOR_SYSTEM = """Consolidate chunk candidates into 1-8 atomic core innovation claims.
Merge paraphrases, split compound claims when independently testable, and discard background,
motivation, routine procedure, and generic significance. Preserve the authors' meaning. Every
claim must cite candidate_ids and manuscript span_ids. Return JSON only."""

VERIFIER_SYSTEM = """Verify one proposed innovation claim against supplied manuscript spans.
Mark unsupported if no span entails it. If partly supported, narrow normalized_claim_text to the
strongest entailed statement; never broaden, improve, or infer causality. Keep author_claim_text
unchanged. Return exact support_span_ids and a brief narrowing_reason. Return JSON only."""


def _span_payload(spans: Iterable[EvidenceSpan]) -> list[dict[str, object]]:
    return [
        {
            "span_id": span.span_id,
            "section": span.section_path,
            "page": span.page,
            "text": span.text,
        }
        for span in spans
    ]


def _eligible_spans(paper: PaperIR) -> list[EvidenceSpan]:
    output: list[EvidenceSpan] = []
    for span in paper.spans:
        section = " ".join(span.section_path).casefold()
        if "reference" in section or not span.text.strip():
            continue
        output.append(span)
    return output


def _chunks(paper: PaperIR, maximum_chars: int = 12_000) -> list[list[EvidenceSpan]]:
    grouped: dict[str, list[EvidenceSpan]] = defaultdict(list)
    for span in _eligible_spans(paper):
        grouped[" / ".join(span.section_path) or "Document"].append(span)
    chunks: list[list[EvidenceSpan]] = []
    for spans in grouped.values():
        current: list[EvidenceSpan] = []
        size = 0
        for span in spans:
            if current and size + len(span.text) > maximum_chars:
                chunks.append(current)
                current, size = [], 0
            current.append(span)
            size += len(span.text)
        if current:
            chunks.append(current)
    return chunks


class FullTextClaimMiner:
    """Mine locally grounded candidates, then consolidate and verify them."""

    def __init__(self, config: GearConfig) -> None:
        self.miner = LazyRoleClient(config, "claim_miner")
        self.consolidator = LazyRoleClient(config, "claim_consolidator")
        self.verifier = LazyRoleClient(config, "internal_verifier")

    def extract(self, paper: PaperIR) -> list[GearClaim]:
        candidates = self._mine(paper)
        consolidated = self._consolidate(paper, candidates)
        return [self._verify(paper, index, raw) for index, raw in enumerate(consolidated, 1)]

    def _mine(self, paper: PaperIR) -> list[ClaimCandidate]:
        output: list[ClaimCandidate] = []
        for chunk_index, spans in enumerate(_chunks(paper), 1):
            raw = self.miner.generate_json(
                system=MINER_SYSTEM,
                user=json.dumps({"paper_id": paper.paper_id, "spans": _span_payload(spans)}, ensure_ascii=False),
                response_schema={
                    "type": "object",
                    "properties": {"candidates": {"type": "array", "items": {
                        "type": "object", "properties": {
                            "claim_type": {"type": "string", "enum": [x.value for x in InnovationClaimType]},
                            "author_claim_text": {"type": "string"},
                            "source_span_ids": {"type": "array", "items": {"type": "string"}},
                        }, "required": ["claim_type", "author_claim_text", "source_span_ids"], "additionalProperties": False,
                    }}},
                    "required": ["candidates"],
                    "additionalProperties": False,
                },
            )
            known = {span.span_id for span in spans}
            for item_index, item in enumerate(raw.get("candidates", []), 1):
                ids = [str(value) for value in item.get("source_span_ids", []) if str(value) in known]
                text = str(item.get("author_claim_text", "")).strip()
                if not text or not ids:
                    continue
                output.append(ClaimCandidate(
                    candidate_id=f"CAND-{chunk_index:03d}-{item_index:03d}",
                    claim_type=InnovationClaimType(str(item.get("claim_type", "FINDING")).upper()),
                    author_claim_text=text,
                    source_span_ids=ids,
                ))
        if not output:
            raise ValueError("全文中没有抽取到可绑定的贡献候选")
        return output

    def _consolidate(self, paper: PaperIR, candidates: list[ClaimCandidate]) -> list[dict[str, object]]:
        raw = self.consolidator.generate_json(
            system=CONSOLIDATOR_SYSTEM,
            user=json.dumps({"paper_id": paper.paper_id, "candidates": [x.model_dump(mode="json") for x in candidates]}, ensure_ascii=False),
            response_schema={
                "type": "object",
                "properties": {"claims": {"type": "array", "minItems": 1, "maxItems": 8, "items": {
                    "type": "object", "properties": {
                        "claim_type": {"type": "string", "enum": [x.value for x in InnovationClaimType]},
                        "author_claim_text": {"type": "string"},
                        "source_span_ids": {"type": "array", "items": {"type": "string"}},
                        "candidate_ids": {"type": "array", "items": {"type": "string"}},
                    }, "required": ["claim_type", "author_claim_text", "source_span_ids", "candidate_ids"], "additionalProperties": False,
                }}},
                "required": ["claims"],
                "additionalProperties": False,
            },
        )
        claims = list(raw.get("claims", []))[:8]
        if not claims:
            raise ValueError("候选合并后没有核心创新 Claim")
        return claims

    def _verify(self, paper: PaperIR, index: int, raw: dict[str, object]) -> GearClaim:
        span_map = paper.span_map()
        source_ids = [str(value) for value in raw.get("source_span_ids", []) if str(value) in span_map]
        if not source_ids:
            raise ValueError(f"第 {index} 条全文 Claim 没有有效原文绑定")
        author_text = str(raw.get("author_claim_text", "")).strip()
        response = self.verifier.generate_json(
            system=VERIFIER_SYSTEM,
            user=json.dumps({"author_claim_text": author_text, "spans": _span_payload(span_map[x] for x in source_ids)}, ensure_ascii=False),
            response_schema={
                "type": "object",
                "properties": {
                    "internal_support": {"type": "string", "enum": [x.value for x in InternalSupportStatus]},
                    "normalized_claim_text": {"type": "string"},
                    "support_span_ids": {"type": "array", "items": {"type": "string"}},
                    "narrowing_reason": {"type": "string"},
                },
                "required": ["internal_support", "normalized_claim_text", "support_span_ids", "narrowing_reason"],
                "additionalProperties": False,
            },
        )
        support_ids = [str(x) for x in response["support_span_ids"] if str(x) in span_map]
        status = InternalSupportStatus(str(response["internal_support"]))
        if status is not InternalSupportStatus.UNSUPPORTED and not support_ids:
            status = InternalSupportStatus.UNSUPPORTED
        normalized = str(response["normalized_claim_text"]).strip() or author_text
        return GearClaim(
            claim_id=f"{paper.paper_id}::GEAR::{index:02d}",
            claim_type=InnovationClaimType(str(raw.get("claim_type", "FINDING")).upper()),
            author_claim_text=author_text,
            normalized_claim_text=normalized,
            source_span_ids=source_ids,
            support_span_ids=support_ids,
            internal_support=status,
            narrowing_reason=str(response["narrowing_reason"]),
        )

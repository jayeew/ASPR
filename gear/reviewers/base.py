"""Shared contracts and payload construction for graph-blind reviewers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from ..contracts import PaperIR
from ..review_contracts import BranchReview, PaperSpecificRubric

GRAPH_FORBIDDEN_FIELDS = {
    "aspr_score",
    "aspr_score_0_100",
    "score_0_100",
    "graph_context",
    "graph_prior",
    "graph_result",
    "p_uptake",
    "conditional_diffusion",
    "d5_percentile",
    "opportunity",
    "context_control",
}


class AgentReviewer(Protocol):
    model_name: str
    last_failures: list[str]
    last_payload: dict[str, Any]

    def review(
        self, paper_ir: PaperIR, rubric: PaperSpecificRubric
    ) -> BranchReview: ...


def build_graph_blind_payload(
    paper_ir: PaperIR,
    rubric: PaperSpecificRubric,
    *,
    max_spans: int = 60,
) -> dict[str, Any]:
    span_map = paper_ir.span_map()
    selected: list[str] = []
    for claim in paper_ir.claims:
        if claim.span_id not in selected:
            selected.append(claim.span_id)
    for field_name in type(paper_ir.method_result_ledger).model_fields:
        if field_name in {"schema_version", "schema_revision"}:
            continue
        for span_id in getattr(paper_ir.method_result_ledger, field_name):
            if span_id in span_map and span_id not in selected:
                selected.append(span_id)
    if not selected:
        selected = [span.span_id for span in paper_ir.spans[:max_spans]]
    payload = {
        "paper_id": paper_ir.paper_id,
        "paper_sha256": paper_ir.paper_sha256,
        "metadata": paper_ir.metadata.model_dump(mode="json"),
        "rubric": rubric.model_dump(mode="json"),
        "claims": [
            {
                "claim_id": claim.claim_id,
                "claim_type": claim.claim_type.value,
                "strength": claim.strength.value,
                "evidence_key": f"P:{claim.span_id}",
                "text": claim.text,
            }
            for claim in paper_ir.claims
        ],
        "spans": [
            {
                "evidence_key": f"P:{span_map[span_id].span_id}",
                "span_id": span_map[span_id].span_id,
                "page": span_map[span_id].page,
                "section_path": list(span_map[span_id].section_path),
                "text": span_map[span_id].text,
                "text_sha256": span_map[span_id].text_sha256,
            }
            for span_id in selected[:max_spans]
        ],
    }
    assert_graph_blind_payload(payload)
    return payload


def assert_graph_blind_payload(payload: dict[str, Any]) -> None:
    leaked: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                normalized = str(key).casefold()
                if normalized in GRAPH_FORBIDDEN_FIELDS:
                    leaked.add(normalized)
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    if leaked:
        raise ValueError(f"reviewer payload contains Graph fields: {sorted(leaked)}")


__all__ = [
    "GRAPH_FORBIDDEN_FIELDS",
    "AgentReviewer",
    "assert_graph_blind_payload",
    "build_graph_blind_payload",
]

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from gear.paper_extraction import configured_paper_extractor


class _RecordingClient:
    def __init__(self, span_id: str, span_text: str) -> None:
        self.response_schema: Mapping[str, Any] | None = None
        self.span_id = span_id
        self.span_text = span_text

    def generate_json(
        self,
        *,
        system: str,
        user: str,
        response_schema: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.response_schema = response_schema
        return {
            "claims": [
                {
                    "claim_id": "C-test",
                    "claim_type": "result_claim",
                    "span_id": self.span_id,
                    "text": self.span_text,
                    "strength": "moderate",
                    "dependency_span_ids": [],
                    "required_evidence": [],
                }
            ]
        }


def test_configured_extractor_passes_paper_claim_response_schema(
    monkeypatch, gear_config, paper_ir
) -> None:
    span = paper_ir.spans[0]
    client = _RecordingClient(span.span_id, span.text)
    monkeypatch.setattr(
        "gear.paper_extraction.build_json_model_client", lambda config: client
    )

    enriched = configured_paper_extractor(gear_config).enrich(paper_ir)

    assert enriched.quality_report.semantic_extraction_ready is True
    assert client.response_schema is not None
    claim_types = client.response_schema["$defs"]["ClaimType"]["enum"]
    assert claim_types == [
        "novelty_claim",
        "method_claim",
        "result_claim",
        "scope_claim",
        "causal_claim",
        "significance_claim",
    ]


def test_semantic_claim_may_paraphrase_its_evidence_span(
    monkeypatch, gear_config, paper_ir
) -> None:
    span = paper_ir.spans[0]
    client = _RecordingClient(
        span.span_id,
        "A concise paraphrase of the manuscript claim.",
    )
    monkeypatch.setattr(
        "gear.paper_extraction.build_json_model_client", lambda config: client
    )

    enriched = configured_paper_extractor(gear_config).enrich(paper_ir)

    assert enriched.quality_report.semantic_extraction_ready is True
    assert enriched.claims[0].text == "A concise paraphrase of the manuscript claim."

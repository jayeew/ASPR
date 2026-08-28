from __future__ import annotations

import pandas as pd

from experiments.gear.evaluation.label_claim_adoption import (
    _aggregate,
    _generate_valid_chunk,
)


class _RepairingClient:
    def __init__(self) -> None:
        self.users: list[str] = []

    def generate_json(self, *, system: str, user: str, response_schema: object) -> dict:
        self.users.append(user)
        claim_id = None if len(self.users) == 1 else "claim-a"
        return {
            "judgments": [
                {
                    "context_id": "ctx-1",
                    "claim_id": claim_id,
                    "relation": "extension",
                    "confidence": 0.9,
                    "rationale": "explicit extension",
                }
            ]
        }


def test_invalid_model_judgment_is_retried_with_validation_feedback() -> None:
    claims = pd.DataFrame({"claim_id": ["claim-a"], "claim_text": ["method A"]})
    contexts = pd.DataFrame({"context_id": ["ctx-1"], "context": ["extends method A"]})
    client = _RepairingClient()

    response = _generate_valid_chunk(
        client,
        paper_id="paper-1",
        claim_payload=claims.to_dict(orient="records"),
        context_payload=contexts.to_dict(orient="records"),
        claims=claims,
        contexts=contexts,
    )

    assert response.judgments[0].claim_id == "claim-a"
    assert len(client.users) == 2
    assert "correction_required" in client.users[1]


def test_claim_adoption_aggregation_preserves_evidence_and_conservation() -> None:
    claims = pd.DataFrame(
        {
            "paper_id": ["paper-1", "paper-1"],
            "claim_id": ["claim-a", "claim-b"],
            "claim_text": ["method A", "result B"],
            "claim_centrality": [0.75, 0.25],
        }
    )
    contexts = pd.DataFrame(
        {
            "paper_id": ["paper-1", "paper-1", "paper-1"],
            "context_id": ["ctx-1", "ctx-1b", "ctx-2"],
            "context": ["uses method A", "extends method A", "mentions result B"],
            "citing_paper_id": ["citer-1", "citer-1", "citer-2"],
            "citing_fields": [["Chemistry"], ["Chemistry"], ["Physics"]],
        }
    )
    judgments = [
        {
            "paper_id": "paper-1",
            "context_id": "ctx-1",
            "claim_id": "claim-a",
            "relation": "method_or_result_use",
            "confidence": 0.9,
            "rationale": "explicit use",
            "citing_paper_id": "citer-1",
            "citing_fields": ["Chemistry"],
        },
        {
            "paper_id": "paper-1",
            "context_id": "ctx-1b",
            "claim_id": "claim-a",
            "relation": "extension",
            "confidence": 0.9,
            "rationale": "explicit extension",
            "citing_paper_id": "citer-1",
            "citing_fields": ["Chemistry"],
        },
        {
            "paper_id": "paper-1",
            "context_id": "ctx-2",
            "claim_id": "claim-b",
            "relation": "background",
            "confidence": 0.8,
            "rationale": "mention only",
            "citing_paper_id": "citer-2",
            "citing_fields": ["Physics"],
        },
    ]

    labels, papers = _aggregate(
        claims,
        contexts,
        judgments,
        context_statuses={"paper-1": "resolved_truncated"},
    )

    assert labels["attribution_weight"].sum() == 1.0
    claim_a = labels.set_index("claim_id").loc["claim-a"]
    assert claim_a["future_adoption"] == 0.8
    assert claim_a["adopting_context_count"] == 2
    assert claim_a["adopting_paper_count"] == 1
    assert labels.set_index("claim_id").loc[
        "claim-a", "adoption_evidence_context_ids"
    ] == ["ctx-1", "ctx-1b"]
    assert papers.iloc[0]["claim_adoption_breadth"] == 0.5
    assert labels["context_observation_status"].eq("resolved_truncated").all()
    assert bool(papers.iloc[0]["adoption_is_lower_bound"]) is True


def test_resolved_zero_context_paper_is_a_zero_label_not_missing() -> None:
    claims = pd.DataFrame(
        {
            "paper_id": ["paper-zero", "paper-zero"],
            "claim_id": ["claim-a", "claim-b"],
            "claim_text": ["method A", "result B"],
            "claim_centrality": [0.6, 0.4],
        }
    )
    contexts = pd.DataFrame(
        columns=[
            "paper_id",
            "context_id",
            "context",
            "citing_paper_id",
            "citing_fields",
        ]
    )

    labels, papers = _aggregate(
        claims, contexts, [], completed_paper_ids={"paper-zero"}
    )

    assert len(labels) == 2
    assert labels["future_adoption"].eq(0.0).all()
    assert papers.iloc[0]["claim_adoption_breadth"] == 0.0

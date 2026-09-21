import pytest

from gear.config import GearConfig
from gear.innovation.analysis import assess
from gear.model_client import LazyRoleClient


def payload(keys):
    return {
        "claim_id": "c",
        "claim_text": "Claim",
        "supported_scope": "Claim",
        "findings": [
            {
                "dimension": "increment",
                "text": "Difference",
                "evidence_keys": keys,
                "stance": None,
            }
        ],
        "overall_stance": "unresolved",
        "overall_reason": "Limited",
        "limitations": ["abstract only"],
    }


def test_analysis_rejects_invented_evidence(monkeypatch):
    monkeypatch.setattr(
        LazyRoleClient, "generate_json", lambda *a, **k: payload(["invented"])
    )
    with pytest.raises(ValueError, match="unbound"):
        assess(GearConfig(), "c", "Claim", {"real": {}}, "gear")


def test_graph_analysis_is_independent(monkeypatch):
    captured = {}

    def call(self, **kwargs):
        captured.update(kwargs)
        return payload(["graph"])

    monkeypatch.setattr(LazyRoleClient, "generate_json", call)
    result = assess(GearConfig(), "c", "Claim", {"graph": {"neighbors": []}}, "graph")
    assert result.claim_id == "c"
    assert "LOCAL" in captured["system"]
    assert "gear_card" not in captured["user"]


def test_text_ablation_removes_structural_information():
    from gear.innovation.experiments import graph_view

    fact = {
        "claim": {},
        "neighbors": [
            {
                "claim_id": "n",
                "claim_text": "text",
                "community_id": 8,
                "direct_citation": True,
            }
        ],
        "metrics": [{"value": 10}],
        "neighbor_edges": [["n", "m"]],
    }
    result = graph_view(fact, "fusion_text_only")
    assert result["neighbors"] == [{"claim_id": "n", "claim_text": "text"}]
    assert "metrics" not in result and "neighbor_edges" not in result


def test_intervention_preserves_text_and_does_not_modify_original():
    from gear.innovation.diagnostics import collapse_communities

    fact = {
        "neighbors": [{"claim_text": "old fact", "community_id": 8}],
        "community_ids": [8],
        "metrics": [{"value": 1}],
    }
    result = collapse_communities(fact)
    assert result["neighbors"][0]["claim_text"] == "old fact"
    assert result["neighbors"][0]["community_id"] == 0
    assert fact["neighbors"][0]["community_id"] == 8


def test_claim_text_is_constrained_and_mismatch_still_rejected(monkeypatch):
    def call(self, **kwargs):
        assert kwargs["response_schema"]["properties"]["claim_text"]["enum"] == [
            "Claim"
        ]
        raw = payload(["real"])
        raw["claim_text"] = "Changed claim"
        return raw

    monkeypatch.setattr(LazyRoleClient, "generate_json", call)
    with pytest.raises(ValueError, match="identity/text"):
        assess(GearConfig(), "c", "Claim", {"real": {}}, "gear")


def test_quoted_claim_text_is_bound_without_schema_literal(monkeypatch) -> None:
    text = 'A "quoted" claim'

    def call(self, **kwargs):
        schema = kwargs['response_schema']
        assert 'claim_text' not in schema['properties']
        assert 'claim_text' not in schema['required']
        raw = payload(['real'])
        del raw['claim_text']
        return raw

    monkeypatch.setattr(LazyRoleClient, 'generate_json', call)
    assert assess(GearConfig(), 'c', text, {'real': {}}, 'graph').claim_text == text

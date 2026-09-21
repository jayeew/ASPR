import sqlite3
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from gear.claim_attribution import ClaimGraphRuntime
from gear.config import GearConfig
from gear.innovation.joint_graph import analyze_joint, joint_structure
from gear.review_contracts import (
    GraphClaim,
    GraphFactCard,
    GraphNeighbor,
    InnovationPaperInput,
)


def card(target, neighbors):
    return GraphFactCard(
        claim=GraphClaim(
            claim_id=target,
            paper_id="new",
            claim_text=target,
            claim_type="FINDING",
            source_sentence_ids=[],
            source_sentence_texts=[],
        ),
        neighbors=[
            GraphNeighbor(
                claim_id=n,
                parent_paper_id="old",
                claim_text=n,
                claim_type="FINDING",
                publication_date=date(2025, 1, 1),
                cosine_similarity=0.8,
                semantic_rank=i + 1,
            )
            for i, n in enumerate(neighbors)
        ],
        metrics=[],
    )


def test_joint_counts_union_not_sum_and_uses_cross_neighborhood_edges():
    cards = [card("x", ["a", "b"]), card("y", ["b", "c"])]
    facts = joint_structure(cards, [])
    assert facts["joint_component_merge_count"] == 2
    assert facts["joint_newly_connected_historical_pairs"] == 3
    assert facts["shared_historical_neighbors"] == {"b": 2}
    # Existing a-c edge must be included even though neither single neighborhood contains it.
    facts = joint_structure(cards, [["a", "c"]])
    assert facts["joint_component_merge_count"] == 1
    assert facts["joint_newly_connected_historical_pairs"] == 2


def test_disjoint_paper_claims_do_not_invent_internal_edges():
    facts = joint_structure([card("x", ["a"]), card("y", ["b"]), card("z", [])], [])
    assert facts["joint_component_merge_count"] == 0
    assert facts["joint_newly_connected_historical_pairs"] == 0
    assert facts["claims_without_neighbors"] == ["z"]
    assert all(not p["connected_via_historical_graph"] for p in facts["claim_pairs"])
    assert facts["insertion_edges"] == [["x", "a"], ["y", "b"]]


def test_redundant_claims_do_not_double_count():
    facts = joint_structure([card("x", ["a", "b"]), card("y", ["a", "b"])], [])
    assert facts["joint_component_merge_count"] == 1
    assert facts["joint_newly_connected_historical_pairs"] == 1


@pytest.mark.parametrize("cache_hit", [True, False])
def test_threshold_and_parent_cache_policy(tmp_path, monkeypatch, cache_hit):
    runtime = ClaimGraphRuntime(tmp_path, Path("."), top_k=10, min_similarity=0.5)
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute(
        "CREATE TABLE claim_nodes(claim_row INTEGER,claim_id TEXT,parent_paper_id TEXT,publication_date TEXT,claim_type TEXT,community_id INTEGER)"
    )
    for i in range(3):
        db.execute(
            "INSERT INTO claim_nodes VALUES(?,?,?,?,?,?)",
            (i, f"c{i}", f"p{i}", "2025-01-01", "FINDING", 1),
        )
    runtime._claim_db = db
    runtime._paper_id_map = {"p0": "W1"} if cache_hit else {}
    monkeypatch.setattr(runtime, "_connections", lambda: None)
    monkeypatch.setattr(runtime, "_index_size", lambda: 3)
    monkeypatch.setattr(
        runtime,
        "_semantic_search",
        lambda v, n: (np.array([0.8, 0.5, 0.49]), np.array([0, 1, 2])),
    )
    monkeypatch.setattr(runtime, "_claim_text", lambda c: c)
    calls = []

    def path(refs, parent):
        calls.append(parent)
        return {"direct_citation": True}

    monkeypatch.setattr(runtime, "_paper_path", path)
    item = InnovationPaperInput(
        paper_id="new",
        paper_path=Path("none"),
        title="new",
        publication_date=date(2026, 1, 1),
        cutoff_date=date(2026, 1, 1),
        reference_work_ids=["W1"],
        abstract_text="test",
        abstract_source="test",
    )
    neighbors = runtime._neighbors(np.ones(2), item)
    assert len(neighbors) == 1  # Exactly .5 is excluded; no forced top10 filling.
    assert neighbors[0].parent_id_cache_hit == cache_hit
    assert neighbors[0].edge_type == (
        "semantic_and_paper_path" if cache_hit else "semantic_only"
    )
    assert len(calls) == int(cache_hit)
    runtime.close()


def test_new_threshold_does_not_reuse_old_percentiles():
    runtime = ClaimGraphRuntime(Path("."), Path("."))
    metric = runtime._metric_fact(
        "component_merge_count", 3, card("x", []).claim.claim_type
    )
    assert metric.value == 3 and metric.global_percentile is None


def test_joint_rejects_invented_claims(monkeypatch):
    class Client:
        def __init__(self, *args):
            pass

        def generate_json(self, **kwargs):
            return {
                "paper_id": "p",
                "knowledge_summary": "x",
                "findings": [
                    {
                        "question": "q",
                        "claim_ids": ["invented"],
                        "observation": "x",
                        "interpretation": "y",
                        "evidence_keys": ["source"],
                        "limitations": [],
                    }
                ],
                "limitations": [],
            }

    monkeypatch.setattr("gear.innovation.joint_graph.LazyRoleClient", Client)
    with pytest.raises(ValueError, match="invented"):
        analyze_joint(GearConfig(), "p", {"source": {}}, {"real"})


def test_joint_retries_invalid_keys_and_constrains_target_ids(monkeypatch) -> None:
    requests = []

    class Client:
        def __init__(self, *args):
            pass

        def generate_json(self, **kwargs):
            requests.append(kwargs)
            fields = kwargs['response_schema']['$defs']['JointFinding']['properties']
            assert fields['claim_ids']['items']['enum'] == ['target']
            assert fields['evidence_keys']['items']['enum'] == ['source']
            return {
                'paper_id': 'p', 'knowledge_summary': 'summary', 'limitations': [],
                'findings': [{'question': 'q', 'claim_ids': ['target'],
                              'observation': 'o', 'interpretation': 'i',
                              'evidence_keys': ['unknown' if len(requests) == 1 else 'source'],
                              'limitations': []}],
            }

    monkeypatch.setattr('gear.innovation.joint_graph.LazyRoleClient', Client)
    result = analyze_joint(GearConfig(), 'p', {'source': {}}, {'target'})
    assert len(requests) == 2
    assert 'unknown' in requests[1]['user']
    assert result.findings[0].evidence_keys == ['source']

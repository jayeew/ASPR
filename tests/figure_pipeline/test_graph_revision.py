"""Regression checks for graph cleaning, missingness and frozen insertion layouts."""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np

from figure_pipeline.graph_revision import clean_atlas, connected_pair_share, local_positions, structural_audit


def test_clean_atlas_quarantines_before_degree_and_path_calculation(tmp_path: Path) -> None:
    nodes = [{"work_id": f"W{i}", "publication_date": f"2023-01-0{i}"} for i in range(1, 5)]
    atlas = {"paper_nodes": nodes, "paper_edges": [["W1","W1"],["W2","W1"],["W2","W1"],["W3","W2"],["W1","missing"]]}
    source = tmp_path/"atlas.json"
    source.write_text(json.dumps(atlas))
    clean, audit = clean_atlas(atlas,source)
    assert clean["paper_edges"] == [["W2","W1"],["W3","W2"]]
    assert len(clean["paper_nodes"]) == 4
    assert audit["statistics"]["isolated_paper_nodes"] == 1
    assert audit["statistics"]["reachable_ordered_pairs"] == 3
    assert audit["statistics"]["quarantined"] == 3
    assert atlas["paper_edges"][0] == ["W1","W1"]
    assert all(row["source_pointer"].startswith("/paper_edges/") for row in audit["quarantined_edges"])


def test_connected_pairs_require_two_history_nodes() -> None:
    assert connected_pair_share(nx.empty_graph(0)) is None
    assert connected_pair_share(nx.empty_graph(1)) is None
    graph = nx.Graph([(0,1),(1,2)])
    graph.add_node(3)
    assert connected_pair_share(graph) == .5
    assert connected_pair_share(nx.complete_graph(4)) == 0


def test_date_inversion_is_flagged_but_not_declared_future_leakage(tmp_path: Path) -> None:
    atlas = {"paper_nodes":[{"work_id":"W1","publication_date":"2023-01-01"},
                            {"work_id":"W2","publication_date":"2023-01-02"}],
             "paper_edges":[["W1","W2"]]}
    source=tmp_path/"atlas.json"
    source.write_text(json.dumps(atlas))
    clean,audit=clean_atlas(atlas,source)
    assert clean["paper_edges"] == [["W1","W2"]]
    assert audit["quarantined_edges"] == []
    assert audit["date_anomalies"][0]["action"] == "retained_pending_version_and_online_first_date_review"


def test_at_cap_does_not_imply_known_candidate_count_or_truncation() -> None:
    rows = [{"paper_id":"P1","neighbor_count":10,"neighbor_induced_density":0.,
             "connected_pair_share":1.,"component_merge_count":9},
            {"paper_id":"P1","neighbor_count":2,"neighbor_induced_density":1.,
             "connected_pair_share":0.,"component_merge_count":0},
            {"paper_id":"P2","neighbor_count":1,"neighbor_induced_density":None,
             "connected_pair_share":0.,"component_merge_count":None}]
    records, audit = structural_audit(rows,draws=10)
    assert records[0]["at_retention_cap"] is True
    assert records[0]["threshold_eligible_candidate_count"] is None
    assert records[0]["top_k_truncated"] is None
    assert records[2]["connected_pair_share"] is None
    assert audit["valid_pair_share_claims"] == 2


def test_local_layout_preserves_an_isolated_historical_node() -> None:
    case = {"graph":{"neighbors":[{"claim_id":"A"},{"claim_id":"B"},{"claim_id":"C"}],
                     "neighbor_edges":[["A","B"]]}}
    graph, first, target = local_positions(case)
    _, second, _ = local_positions(case)
    assert set(graph) == {"A","B","C"}
    assert set(first) == set(graph)
    assert graph.degree("C") == 0
    assert all(np.array_equal(first[node],second[node]) for node in graph)
    assert target.shape == (2,)

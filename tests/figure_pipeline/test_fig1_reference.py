"""Scientific scope and export contracts for the component-based Fig.1."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import networkx as nx
import pytest

from figure_pipeline.fig1_reference.data import (
    community_ccdf,
    local_layout,
    local_metrics,
    paths,
)
from figure_pipeline.fig1_reference.export import reassemble_components
from figure_pipeline.fig1_reference.paper_display import connected_selection
from figure_pipeline.fig1_reference.svg import Scene


def test_observations_follow_current_roster_and_latest_graph_fact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from figure_pipeline.fig1_reference import data

    monkeypatch.setattr(data, "STUDY", tmp_path)
    (tmp_path / "papers.jsonl").write_text('{"paper_id": "current"}\n')
    for pid in ("current", "removed"):
        folder = tmp_path / "papers" / pid
        (folder / "shared").mkdir(parents=True)
        cid = f"{pid}::CLAIM::01"
        (folder / "shared/claims.json").write_text(
            json.dumps({"claims": [{"claim_id": cid}]})
        )
        directory = folder / "graph/01"
        directory.mkdir(parents=True)
        records = [
            {
                "kind": "graph_fact", "evidence_id": f"GRAPH:{cid}",
                "payload": {
                    "claim": {"claim_id": cid, "claim_type": "FINDING"},
                    "neighbors": [{"community_id": 1}, {"community_id": None}],
                    "metrics": [{"name": "nearest_prior_similarity", "value": value}],
                },
            }
            for value in (0.6, 0.8)
        ]
        (directory / "evidence_trace.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in records)
        )
    rows = data.study_observations()
    assert [row["claim_id"] for row in rows] == ["current::CLAIM::01"]
    assert rows[0]["nearest_prior_similarity"] == 0.8
    assert rows[0]["neighbor_count"] == 2
    assert rows[0]["community_assignment_coverage"] == 0.5


def neighbor(
    ident: str, community: int | None = 1, role: str = "FINDING"
) -> dict[str, Any]:
    return {
        "claim_id": ident,
        "community_id": community,
        "claim_type": role,
        "cosine_similarity": 0.8,
    }


def test_connectivity_matches_explicit_insertion() -> None:
    neighbors = [neighbor(str(i)) for i in range(10)]
    edges = [[str(i), str(i + 1)] for i in [0, 1, 2, 4, 5, 7, 8]]
    metrics = local_metrics(neighbors, edges, "METHOD")
    before = nx.Graph()
    before.add_nodes_from(str(i) for i in range(10))
    before.add_edges_from(edges)
    after = before.copy()
    after.add_edges_from(("target", str(i)) for i in range(10))
    newly_reachable = sum(
        not nx.has_path(before, str(i), str(j)) and nx.has_path(after, str(i), str(j))
        for i in range(10)
        for j in range(i + 1, 10)
    )
    assert metrics["components_before"] == 3
    assert metrics["newly_connected_neighbor_pair_count"] == newly_reachable == 33
    assert metrics["connected_pair_share"] == 33 / math.comb(10, 2)
    assert metrics["cross_type_neighbor_share"] == 1


@pytest.mark.parametrize("count", [0, 1])
def test_undefined_pairs_remain_na(count: int) -> None:
    metrics = local_metrics(
        [neighbor(str(i), None) for i in range(count)], [], "FINDING"
    )
    assert metrics["connected_pair_share"] is None
    assert metrics["effective_community_count"] is None


@pytest.mark.parametrize(
    "edges", [[["a", "outside"]], [["a", "a"]], [["a", "b"], ["b", "a"]]]
)
def test_reject_invalid_historical_edges(edges: list[list[str]]) -> None:
    with pytest.raises(ValueError):
        local_metrics([neighbor("a"), neighbor("b")], edges, "METHOD")


def test_local_layout_keeps_historical_nodes_fixed() -> None:
    case = {
        "neighbors": [neighbor(str(i)) for i in range(10)],
        "edges": [[str(i), str(i + 1)] for i in range(8)],
    }
    first = local_layout(case, 19)
    second = local_layout(case, 19)
    assert first == second
    assert first["before"] == first["after_historical"]
    assert set(first["before"]) == {str(i) for i in range(10)}


def test_missing_references_do_not_query_or_become_zero() -> None:
    result = paths(
        None,
        {"paper_id": "p", "reference_work_ids": []},
        {"parent_openalex_work_id": "W1"},
    )  # type: ignore[arg-type]
    assert result["status"] == "reference_ids_unavailable"
    assert result["counts"] is None


def test_component_preserves_editable_identity_and_dimensions(tmp_path: Path) -> None:
    component = Scene(120, 100, "before")
    component.node(20, 40, ident="real-claim-id")
    component.text(10, 90, "Before insertion")
    path = tmp_path / "component.svg"
    component.save(path)
    text = path.read_text()
    assert 'viewBox="0 0 120 100"' in text
    assert 'id="real-claim-id"' in text
    assert "<text" in text


def test_long_paragraph_does_not_silently_clip() -> None:
    s = Scene(30, 30, "metric")
    with pytest.raises(ValueError, match="requires"):
        s.paragraph(0, 10, "A long metric title that cannot fit", 20, max_lines=1)


def test_assembly_consumes_editable_components(tmp_path: Path) -> None:
    component = Scene(80, 60, "mini")
    component.root.set("data-component", "mini")
    component.text(5, 30, "Old label")
    page = Scene(120, 100, "page")
    page.use(component, 10, 15)
    page.save(tmp_path / "layouts/templates/Fig1.svg")
    changed = Scene(80, 60, "mini")
    changed.root.set("data-component", "mini")
    changed.text(5, 30, "Updated label")
    changed.save(tmp_path / "components/layers/mini.svg")
    (tmp_path / "final").mkdir()
    reassemble_components(tmp_path)
    text = (tmp_path / "final/Fig1.svg").read_text()
    assert "Updated label" in text and "Old label" not in text
    assert "translate(10 15)" in text


def test_community_ccdf_preserves_counts_and_thresholds() -> None:
    rows = community_ccdf({"9": 1, "2": 3, "5": 2})
    assert [r["size"] for r in rows] == [2, 5, 9]
    assert [r["communities_at_least_size"] for r in rows] == [6, 3, 1]
    assert [r["share_at_least_size"] for r in rows] == pytest.approx([1, 0.5, 1 / 6])


@pytest.mark.parametrize("histogram", [{}, {0: 1}, {2: 0}, {2: -1}])
def test_community_ccdf_rejects_invalid_counts(histogram: dict[int, int]) -> None:
    with pytest.raises(ValueError):
        community_ccdf(histogram)


def test_connected_paper_selection_preserves_observed_edges() -> None:
    graph = nx.DiGraph([(r, "shared") for r in ["A", "B", "C", "D"]])
    graph.add_edges_from(("shared", f"ref{i}") for i in range(10))
    graph.add_edge("outside", "disconnected")
    original = set(graph.edges)
    selected, connectors = connected_selection(graph, ["A", "B", "C", "D"], set(), 9)
    assert len(selected) == 9
    assert nx.is_weakly_connected(graph.subgraph(selected))
    assert not list(nx.isolates(graph.subgraph(selected)))
    assert all(set(p) <= set(selected) for p in connectors)
    assert set(graph.edges) == original
    assert "outside" not in selected


def test_disconnected_targets_are_not_artificially_linked() -> None:
    graph = nx.DiGraph([("A", "a"), ("B", "b")])
    with pytest.raises(ValueError, match="does not connect"):
        connected_selection(graph, ["A", "B"], set(), 8)

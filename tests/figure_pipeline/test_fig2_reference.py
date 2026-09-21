"""Tests of graph counting and editable reassembly, independent of model execution."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from figure_pipeline.fig1_reference.svg import NS, Scene
from figure_pipeline.fig2_reference.data import METRICS, network_layout, topology
from figure_pipeline.fig2_reference.export import reassemble


def test_topology_counts_pairs_not_added_edges() -> None:
    facts = topology(["a", "b", "c", "d", "e"], [["a", "b"], ["b", "c"], ["d", "e"]])
    assert facts == {"density": 0.3, "components": 2, "merges": 1, "new_pairs": 6}


def test_duplicate_historical_edges_do_not_change_connectivity() -> None:
    once = topology(["a", "b", "c"], [["a", "b"]])
    twice = topology(["a", "b", "c"], [["a", "b"], ["b", "a"]])
    assert once == twice
    assert once["new_pairs"] == 2


def test_metric_names_are_complete_and_unique() -> None:
    assert len(METRICS) == len({name for name, _ in METRICS}) == 14
    assert METRICS[-1] == ("co_citation_neighbor_count", "Shared-reference neighbors")


def test_deterministic_coordinates_use_only_real_nodes() -> None:
    nodes = ["H1", "H2", "H3", "paper::CLAIM::01"]
    edges = [[nodes[-1], n] for n in nodes[:-1]]
    first = network_layout(nodes, edges, 3)
    assert first == network_layout(nodes[::-1], edges[::-1], 3)
    assert set(first) == set(nodes)
    assert all(0 <= coordinate <= 1 for point in first.values() for coordinate in point)


def test_layer_edit_reaches_standalone_panel_and_final(tmp_path: Path) -> None:
    component = Scene(50, 20, "test_component")
    component.root.set("data-component", "test_component")
    component.text(4, 14, "Original", 9)
    panel = Scene(100, 40, "test_panel")
    panel.use(component, 20, 10)
    panel.save(tmp_path / "layouts/templates/a.svg")
    panel.save(tmp_path / "layouts/templates/Fig2.svg")
    (tmp_path / "panels").mkdir()
    (tmp_path / "final").mkdir()
    for node in component.root.iter(f"{{{NS}}}text"):
        node.text = "Edited layer"
    component.save(tmp_path / "components/layers/test_component.svg")
    reassemble(tmp_path)
    for path in ["components/test_component.svg", "panels/a.svg", "final/Fig2.svg"]:
        root = ET.parse(tmp_path / path).getroot()
        assert [n.text for n in root.iter(f"{{{NS}}}text")] == ["Edited layer"]
    assert "translate(20 10)" in (tmp_path / "final/Fig2.svg").read_text()

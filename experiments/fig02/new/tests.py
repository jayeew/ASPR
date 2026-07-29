"""Tests for Fig.2."""

from pathlib import Path

from experiments.common.new.adapters.builders import build_new_bundle
from experiments.common.new.adapters.runtime import load_figure_context
from experiments.common.new.adapters.tests_support import run_tests


def _assert_fig2_contract(config_path: Path) -> None:
    """Check the outcome-free Fig.2 data contract before rendering."""
    config, paths, _ = load_figure_context(config_path)
    bundle = build_new_bundle(2, config, paths)
    forbidden = ("oof", "future", "known_group", "measurement_scene")
    assert not [
        table
        for table in bundle.tables
        if any(token in table.lower() for token in forbidden)
    ]
    stages = bundle.tables["fig2_selection_stages"].sort_values("stage_order")
    assert stages["count"].astype(int).tolist() == [50, 30, 20, 18, 8]
    flows = bundle.tables["fig2_candidate_role_flows"]
    assert int(flows["candidate_count"].sum()) == 50
    assert (
        flows.groupby("role")["candidate_count"].sum().astype(int).to_dict()
        == {
            "primary": 8,
            "sensitivity": 18,
            "exploratory": 4,
            "excluded": 20,
        }
    )
    ledger = bundle.tables["fig2_indicator_ledger"]
    assert len(ledger) == 8
    assert ledger["angle_id"].nunique() == 5
    assert ledger["all_primary_gates_pass"].all()
    relations = bundle.tables["fig2_relation_edges"]
    assert len(relations) == 7
    incident_ids = set(relations["source_id"]) | set(relations["target_id"])
    assert "I1" not in incident_ids


if __name__ == "__main__":
    figure_config = Path(__file__).with_name("config.json")
    _assert_fig2_contract(figure_config)
    run_tests(2, figure_config)

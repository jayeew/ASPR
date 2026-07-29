from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from select_indicators import ROOT, run_selection


def by_id(rows: List[Dict[str, Any]], field: str) -> Dict[str, Dict[str, Any]]:
    """Index decision rows by a stable identifier."""
    return {str(row[field]): row for row in rows}


def test_selected_metrics_obey_invariants(result: Dict[str, Any]) -> None:
    """Selected metrics must be direct, T0, local, and outcome-blind."""
    selected = [
        row for row in result["metric_decisions"] if bool(row["selected"])
    ]
    assert selected
    for row in selected:
        assert row["scope_role"] == "direct_novelty"
        assert row["signal_role"] == "novelty"
        assert row["t0_computable"] is True
        assert row["requires_future"] is False
        assert row["local_data_status"] == "audited_available"
        assert row["failed_gates"] == []


def test_dimension_selection_has_no_empty_dimension(result: Dict[str, Any]) -> None:
    """Every selected dimension must have at least one selected metric."""
    for row in result["dimension_decisions"]:
        if row["selected"]:
            assert row["scope_role"] == "direct_novelty"
            assert row["selected_metric_ids"]


def test_important_exclusions_are_rule_driven(result: Dict[str, Any]) -> None:
    """Regression-check the conceptual and measurement boundary decisions."""
    metrics = by_id(result["metric_decisions"], "metric_id")
    assert "G13_STABILITY_RHO" in metrics[
        "M012_NOVELTY_U_COMMONNESS_TAIL"
    ]["failed_gates"]
    assert "G02_NOVELTY_SIGNAL" in metrics[
        "M014_CONVENTIONALITY_Z_CENTER"
    ]["failed_gates"]
    assert "G01_DIRECT_NOVELTY" in metrics["M036_RAO_STIRLING_INTEGRATION"][
        "failed_gates"
    ]
    assert "G08_NO_FUTURE_INFORMATION" in metrics["M040_DISRUPTION_INDEX"][
        "failed_gates"
    ]


def test_no_quota_or_outcome_selection(result: Dict[str, Any]) -> None:
    """The summary must explicitly attest that forbidden selectors were absent."""
    summary = result["summary"]
    assert summary["selection_used_dimension_quota"] is False
    assert summary["selection_used_indicator_quota"] is False
    assert summary["selection_used_future_outcomes"] is False
    assert summary["selection_used_prediction_performance"] is False


def test_materialized_outputs_are_deterministic() -> None:
    """Two clean runs from frozen inputs must produce byte-identical outputs."""
    with (
        tempfile.TemporaryDirectory() as first,
        tempfile.TemporaryDirectory() as second,
    ):
        first_path = Path(first)
        second_path = Path(second)
        run_selection(first_path)
        run_selection(second_path)
        first_files = sorted(path.name for path in first_path.iterdir())
        second_files = sorted(path.name for path in second_path.iterdir())
        assert first_files == second_files
        for name in first_files:
            assert (first_path / name).read_bytes() == (second_path / name).read_bytes()


def main() -> None:
    """Run lightweight standalone tests without pytest."""
    result = run_selection()
    test_selected_metrics_obey_invariants(result)
    test_dimension_selection_has_no_empty_dimension(result)
    test_important_exclusions_are_rule_driven(result)
    test_no_quota_or_outcome_selection(result)
    test_materialized_outputs_are_deterministic()
    protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    print(f"PASS: {protocol['protocol_id']}")


if __name__ == "__main__":
    main()

"""Checks for the active expanded Nature multi-horizon data contract."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gear.nature_multihorizon.active_dataset import (
    load_active_dataset,
)


def test_active_expanded_dataset() -> None:
    """Require the active registry, contract, and quality report to agree."""
    active = load_active_dataset(PROJECT_ROOT)
    assert (
        active["active_dataset_version"]
        == active["contract_payload"]["dataset_version"]
    )
    assert active["horizon_publication_year_max"] == {
        "3": 2022,
        "5": 2020,
        "8": 2017,
    }
    assert active["quality_payload"]["overall_pass"] is True
    for horizon, maximum_year in active["horizon_publication_year_max"].items():
        profile = active["quality_payload"]["counts_by_horizon"][horizon]
        assert int(profile["cohort_rows"]) > 0
        assert int(profile["publication_year_max"]) == int(maximum_year)


if __name__ == "__main__":
    test_active_expanded_dataset()
    print("active expanded dataset: PASS")

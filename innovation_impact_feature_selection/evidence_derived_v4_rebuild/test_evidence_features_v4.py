"""Deterministic edge-case tests for ``materialize_evidence_features_v4``."""

from __future__ import annotations

import json
import math
from pathlib import Path

from materialize_evidence_features_v4 import (
    _cross_disciplinary_ratio,
    _distance_values,
    _distribution_values,
)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "evidence_features_v4_tests.json"


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def main() -> None:
    balance, simpson, entropy, variety = _distribution_values(["a", "a", "b"])
    assert _close(balance, 5.0 / 6.0)
    assert _close(simpson, 4.0 / 9.0)
    assert _close(entropy, -(2.0 / 3.0) * math.log(2.0 / 3.0) - (1.0 / 3.0) * math.log(1.0 / 3.0))
    assert _close(variety, 2.0)
    empty = _distribution_values([])
    assert all(math.isnan(value) for value in empty)
    partial = _distribution_values(["a", "", "b"])
    assert _close(partial[0], 1.0)
    assert _close(partial[1], 0.5)
    assert _close(partial[2], math.log(2.0))
    assert _close(partial[3], 2.0)
    distances = {("a", "b"): 0.6}
    average, rao = _distance_values(["a", "a", "b"], distances)
    assert _close(average, 0.4)
    assert _close(rao, 4.0 / 15.0)
    missing_distance = _distance_values(["a", "b"], {})
    assert all(math.isnan(value) for value in missing_distance)
    assert _close(_cross_disciplinary_ratio("a", ["a", "b"], 2), 0.5)
    assert math.isnan(_cross_disciplinary_ratio("", ["a"], 1))
    assert math.isnan(_cross_disciplinary_ratio("a", [], 0))
    payload = {
        "status": "pass",
        "tests": [
            "distribution_metrics_known_counts",
            "empty_mapped_reference_set_missing",
            "incomplete_category_coverage_uses_observed_categories_without_imputation",
            "ordered_pair_average_and_rao_scaling",
            "missing_distance_returns_missing",
            "cross_disciplinary_ratio_boundary_cases",
        ],
        "outcome_columns_used": False,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

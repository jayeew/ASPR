from __future__ import annotations

import csv
from pathlib import Path

import pytest

from aspr.nature_multihorizon.finalize_v6 import (
    canonical_artifact_id,
    read_gate_results,
)


def test_canonical_artifact_id_ignores_existing_id() -> None:
    payload = {"artifact_kind": "test", "value": 7}
    first = canonical_artifact_id(payload)
    second = canonical_artifact_id({**payload, "artifact_id": "stale"})
    assert first == second
    assert first.startswith("sha256:")


def test_gate_reader_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "gates.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["gate_id", "passed"])
        writer.writeheader()
        writer.writerow({"gate_id": "G1", "passed": "1"})
        writer.writerow({"gate_id": "G2", "passed": "0"})
    with pytest.raises(RuntimeError, match="G2"):
        read_gate_results(path)


def test_gate_reader_accepts_unique_passing_roster(tmp_path: Path) -> None:
    path = tmp_path / "gates.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["gate_id", "passed"])
        writer.writeheader()
        writer.writerow({"gate_id": "G1", "passed": "1"})
        writer.writerow({"gate_id": "G2", "passed": "true"})
    result = read_gate_results(path)
    assert result["all_pass"] is True
    assert result["n_gates"] == 2

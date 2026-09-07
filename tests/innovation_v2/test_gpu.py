from __future__ import annotations

import fcntl
from pathlib import Path

import pytest

from gear.innovation.gpu import acquire_graph_slots


def test_graph_wait_does_not_hold_partial_gpu_slots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEAR_GPU_MAX_PROCESSES", "2")
    monkeypatch.setenv("GEAR_GPU_LEASE_DIR", str(tmp_path))
    monkeypatch.setenv("GEAR_GPU_LEASE_TIMEOUT_SECONDS", "0")
    with (tmp_path / "cuda0_slot_1.lock").open("a+") as busy:
        fcntl.flock(busy.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(TimeoutError):
            acquire_graph_slots()
        with (tmp_path / "cuda0_slot_0.lock").open("a+") as free:
            fcntl.flock(free.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    handles = acquire_graph_slots()
    assert len(handles) == 2
    for handle in handles:
        handle.close()

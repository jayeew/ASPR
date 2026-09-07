"""Exclusive graph embedding lease compatible with existing GEAR GPU slots."""

from __future__ import annotations

import fcntl
import os
import time
from pathlib import Path
from typing import TextIO


def acquire_graph_slots() -> list[TextIO]:
    count = int(os.environ.get("GEAR_GPU_MAX_PROCESSES", "0"))
    if count <= 0:
        return []
    root = Path(os.environ.get("GEAR_GPU_LEASE_DIR", "/tmp/aspr_gear_gpu_leases"))
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    deadline = time.monotonic() + float(
        os.environ.get("GEAR_GPU_LEASE_TIMEOUT_SECONDS", "3600")
    )
    while True:
        acquired: list[TextIO] = []
        try:
            for slot in range(count):
                handle = (root / f"cuda0_slot_{slot}.lock").open("a+", encoding="utf-8")
                acquired.append(handle)
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return acquired
        except BlockingIOError:
            for acquired_handle in acquired:
                acquired_handle.close()
        if time.monotonic() >= deadline:
            raise TimeoutError("Graph embedding GPU lease unavailable")
        time.sleep(0.25)

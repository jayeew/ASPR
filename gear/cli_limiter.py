"""Cross-process lease for bounded Codex CLI concurrency."""

from __future__ import annotations

import fcntl
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO


@contextmanager
def _lease(env_name: str, prefix: str) -> Iterator[None]:
    limit = int(os.environ.get(env_name, "0"))
    if limit <= 0:
        yield
        return
    root = Path(os.environ.get("GEAR_CLI_LEASE_DIR", "/tmp/aspr_gear_cli_leases"))
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    deadline = time.monotonic() + float(
        os.environ.get("GEAR_CLI_LEASE_TIMEOUT_SECONDS", "7200")
    )
    handle: TextIO | None = None
    while handle is None:
        for slot in range(limit):
            candidate = (root / f"{prefix}_{slot}.lock").open("a+", encoding="utf-8")
            try:
                fcntl.flock(candidate.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                candidate.close()
                continue
            handle = candidate
            break
        if handle is None:
            if time.monotonic() >= deadline:
                raise TimeoutError("Codex CLI concurrency lease unavailable")
            time.sleep(0.1)
    try:
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


@contextmanager
def codex_cli_lease() -> Iterator[None]:
    with _lease("GEAR_CLI_MAX_PROCESSES", "codex"):
        yield


@contextmanager
def network_request_lease() -> Iterator[None]:
    with _lease("GEAR_NETWORK_MAX_PROCESSES", "network"):
        yield

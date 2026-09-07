"""Delay new postprocessing calls while Linux has little available memory."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path


def wait_for_memory(logger: logging.Logger | None = None) -> None:
    minimum = float(os.environ.get("GEAR_POSTPROCESS_MIN_AVAILABLE_GIB", "5"))
    if minimum <= 0:
        return
    started = time.monotonic()
    last_log = -60.0
    while True:
        rows = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
        available = (
            next(int(row.split()[1]) for row in rows if row.startswith("MemAvailable:"))
            / 1024**2
        )
        if available >= minimum:
            return
        elapsed = time.monotonic() - started
        if logger and elapsed - last_log >= 60:
            logger.warning(
                "[等待内存] 可用=%.1f GiB，保留=%.1f GiB；暂停新增模型任务",
                available,
                minimum,
            )
            last_log = elapsed
        if elapsed >= 7200:
            raise TimeoutError(
                "Postprocessing memory reserve unavailable for two hours"
            )
        time.sleep(2)

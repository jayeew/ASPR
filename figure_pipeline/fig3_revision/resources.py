"""Physical-resource admission for new tasks; running work is never killed."""
from __future__ import annotations

import os
from pathlib import Path

from .config import Config


def available_memory_gib() -> float:
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        values[key] = int(value.strip().split()[0])*1024
    available = values['MemAvailable']
    # Respect a container/WSL process memory limit when cgroup v2 exposes one.
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        if line.startswith('0::'):
            directory = Path('/sys/fs/cgroup')/line[3:].lstrip('/')
            while directory.is_relative_to(Path('/sys/fs/cgroup')):
                limit_path, current_path = directory/'memory.max', directory/'memory.current'
                if limit_path.exists() and current_path.exists():
                    limit = limit_path.read_text().strip()
                    if limit != 'max':
                        available = min(available, max(0, int(limit)-int(current_path.read_text())))
                if directory == Path('/sys/fs/cgroup'):
                    break
                directory = directory.parent
    return available/(1024**3)


def admission(config: Config, requested: int, active: int) -> tuple[int, float, int]:
    """Return additional tasks allowed now, available GiB, and CPU-derived ceiling."""
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1
    ceiling = min(requested, max(1, cpus*config.workers_per_cpu))
    available = available_memory_gib()
    slots = max(0, int((available-config.memory_reserve_gib)/config.task_memory_gib))
    return max(0, min(ceiling-active, slots)), available, ceiling

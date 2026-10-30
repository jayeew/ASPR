"""Release completed batch allocations before an otherwise idle consumer waits."""
from __future__ import annotations

import ctypes
import gc
import os
from pathlib import Path

import pyarrow as pa


def release_batch_memory() -> dict[str, int]:
    def resident() -> int:
        return int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
    before = resident()
    gc.collect()
    pa.default_memory_pool().release_unused()
    libc = ctypes.CDLL(None)
    trim = libc.malloc_trim
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    trim(0)
    return {'rss_before': before, 'rss_after': resident()}

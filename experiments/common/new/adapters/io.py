"""Deterministic serialization and provenance helpers."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd


def json_ready(value: Any) -> Any:
    """Convert common scientific Python values to strict JSON values."""
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA or value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write sorted UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return path


def sha256_file(path: Path) -> str:
    """Hash a local file without reading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(text: str, seed: int) -> str:
    """Return a deterministic sortable hash."""
    return hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()


def write_tables(
    tables: Mapping[str, pd.DataFrame],
    panel_data_dir: Path,
) -> Dict[str, Dict[str, Any]]:
    """Write every panel table and return its immutable record."""
    panel_data_dir.mkdir(parents=True, exist_ok=True)
    records: Dict[str, Dict[str, Any]] = {}
    for name, frame in sorted(tables.items()):
        if len(frame) >= 50_000 or frame.size >= 400_000:
            path = panel_data_dir / f"{name}.parquet"
            frame.to_parquet(path, index=False, compression="zstd")
            file_format = "parquet"
        else:
            path = panel_data_dir / f"{name}.csv"
            frame.to_csv(path, index=False, float_format="%.12g")
            file_format = "csv"
        records[name] = {
            "path": str(path.resolve()),
            "format": file_format,
            "rows": int(len(frame)),
            "columns": [str(column) for column in frame.columns],
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
    return records


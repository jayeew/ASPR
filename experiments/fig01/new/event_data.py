"""Deterministic hashing and JSON helpers used by the canonical Fig.1 run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    """Hash one JSON-serializable value deterministically."""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_seed(*values: object) -> int:
    """Create a positive deterministic seed from stable text values."""
    digest = hashlib.sha256(
        "::".join(str(value) for value in values).encode("utf-8")
    ).hexdigest()
    return int(digest[:12], 16) % 2_147_483_647


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one JSON mapping atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)

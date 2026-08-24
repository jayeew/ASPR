"""Stable read-only entry point for the currently preferred OOF release."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_RESULT_RELEASE = ROOT / "current_best"
DEFAULT_RESULT_MANIFEST = ROOT / "current_release.json"


def load_current_release() -> dict[str, Any]:
    """Load and validate the mutable pointer to an immutable release."""
    payload = json.loads(DEFAULT_RESULT_MANIFEST.read_text(encoding="utf-8"))
    release = DEFAULT_RESULT_RELEASE.resolve(strict=True)
    expected = Path(payload["release_path"]).resolve(strict=True)
    if release != expected:
        raise ValueError("current_best symlink differs from current_release.json")
    if not (release / "frozen_release_manifest.json").is_file():
        raise ValueError("current release is not a frozen release")
    return {**payload, "resolved_release_path": str(release)}


def current_artifact(name: str) -> Path:
    """Resolve one registered artifact inside the preferred frozen release."""
    payload = load_current_release()
    relative = (payload.get("artifacts") or {}).get(name)
    if not relative:
        raise KeyError(f"unregistered current-release artifact: {name}")
    path = DEFAULT_RESULT_RELEASE / str(relative)
    resolved = path.resolve(strict=True)
    release = DEFAULT_RESULT_RELEASE.resolve(strict=True)
    if release not in resolved.parents:
        raise ValueError("current-release artifact escapes the frozen release")
    return path


__all__ = [
    "DEFAULT_RESULT_MANIFEST",
    "DEFAULT_RESULT_RELEASE",
    "current_artifact",
    "load_current_release",
]

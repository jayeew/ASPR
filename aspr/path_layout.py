"""Canonical path resolution for versioned ASPR experiment artifacts.

Frozen manifests retain the absolute paths recorded when an experiment ran.
After the repository layout migration, readers should resolve those historical
locations to the canonical ``common/{old,new}`` and ``figXX/{old,new}`` trees
without editing the frozen manifest or changing its provenance hash.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple


_LEGACY_OUTPUT_PREFIXES: Tuple[Tuple[str, str], ...] = (
    (
        "outputs/common/new/model/v6_1_r5",
        "outputs/common/new/model/v6_1_r5",
    ),
    (
        "outputs/common/old/model/v6_1_r4",
        "outputs/common/old/model/v6_1_r4",
    ),
    (
        "outputs/common/old/model/v6_1_r3",
        "outputs/common/old/model/v6_1_r3",
    ),
    (
        "outputs/common/old/model/v6_1_r2",
        "outputs/common/old/model/v6_1_r2",
    ),
    (
        "outputs/common/old/model/v6_1_r1",
        "outputs/common/old/model/v6_1_r1",
    ),
    (
        "outputs/common/old/model/v6_1_initial",
        "outputs/common/old/model/v6_1_initial",
    ),
    (
        "outputs/common/old/model/v6",
        "outputs/common/old/model/v6",
    ),
    (
        "outputs/common/new/base_suite",
        "outputs/common/new/base_suite",
    ),
    (
        "outputs/common/new/baseline_suite_r1",
        "outputs/common/new/baseline_suite_r1",
    ),
)


def resolve_artifact_path(
    value: str | Path,
    *,
    project_root: Optional[Path] = None,
) -> Path:
    """Resolve a current or pre-migration artifact path.

    Existing paths are returned unchanged. A missing path is relocated only
    when it is under one of the exact, registered legacy output roots.

    Args:
        value: Absolute path from a manifest or a project-relative path.
        project_root: Repository root. Defaults to the parent of ``aspr``.

    Returns:
        The resolved current path, which may still be absent when no registered
        relocation applies.
    """
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[1]
    )
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    if candidate.exists() or candidate.is_symlink():
        return candidate

    try:
        relative = candidate.relative_to(root).as_posix()
    except ValueError:
        return candidate

    for legacy_prefix, canonical_prefix in _LEGACY_OUTPUT_PREFIXES:
        if relative == legacy_prefix:
            return (root / canonical_prefix).resolve()
        marker = f"{legacy_prefix}/"
        if relative.startswith(marker):
            suffix = relative[len(marker) :]
            return (root / canonical_prefix / suffix).resolve()
    return candidate


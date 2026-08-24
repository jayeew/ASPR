#!/usr/bin/env python3
"""Replace verified mutable result duplicates with frozen-release symlinks."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
BEST_RELEASE = (
    ROOT / "frozen_releases" / "hgb_nested_tuned_7_16_153_219_20260820_b48936af"
)
BASELINE_RELEASE = ROOT / "frozen_releases" / "baseline_20260820_756aa9c3"
FIXED_RELEASE = ROOT / "frozen_releases" / "relaxed_7_16_153_219_fixed_hgb_20260820"

DIRECTORY_DUPLICATES = {
    ROOT / "experiments" / "hgb_nested_tuned_7_16_153_219_20260820": BEST_RELEASE,
    ROOT / "experiments" / "relaxed_7_16_153_219_20260820": FIXED_RELEASE,
    ROOT
    / "outputs"
    / "hgb_oof_canonical": (BASELINE_RELEASE / "outputs" / "hgb_oof_canonical"),
}

FILE_DUPLICATES = {
    ROOT / "outputs" / name: BASELINE_RELEASE / "outputs" / name
    for name in (
        "final_training_features_strict.parquet",
        "final_training_features_primary.parquet",
        "final_training_features_expanded.parquet",
        "final_training_features_broad_t0.parquet",
        "mapped_training_source.parquet",
    )
}

CACHE_DIRECTORIES = (
    ROOT / ".pytest_cache",
    ROOT / ".ruff_cache",
    ROOT / "__pycache__",
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _verify_directory_duplicate(source: Path, frozen: Path) -> int:
    if source.is_symlink():
        if source.resolve() != frozen.resolve():
            raise ValueError(f"existing symlink has wrong target: {source}")
        return 0
    if not source.is_dir() or not frozen.is_dir():
        raise ValueError(f"directory duplicate is missing: {source} / {frozen}")
    source_files = sorted(item for item in source.rglob("*") if item.is_file())
    for item in source_files:
        counterpart = frozen / item.relative_to(source)
        if not counterpart.is_file() or sha256_file(item) != sha256_file(counterpart):
            raise ValueError(f"directory is not an exact frozen duplicate: {item}")
    return _directory_size(source)


def _replace_with_symlink(source: Path, frozen: Path) -> None:
    if source.is_symlink():
        return
    if source.is_dir():
        shutil.rmtree(source)
    else:
        source.unlink()
    source.symlink_to(frozen.resolve(), target_is_directory=frozen.is_dir())


def cleanup() -> dict[str, Any]:
    """Verify every target before deleting duplicate bytes and linking frozen data."""
    reclaimed = 0
    for source, frozen in DIRECTORY_DUPLICATES.items():
        reclaimed += _verify_directory_duplicate(source, frozen)
    for source, frozen in FILE_DUPLICATES.items():
        if source.is_symlink():
            if source.resolve() != frozen.resolve():
                raise ValueError(f"existing symlink has wrong target: {source}")
            continue
        if not source.is_file() or sha256_file(source) != sha256_file(frozen):
            raise ValueError(f"file is not an exact frozen duplicate: {source}")
        reclaimed += source.stat().st_size
    for source, frozen in DIRECTORY_DUPLICATES.items():
        _replace_with_symlink(source, frozen)
    for source, frozen in FILE_DUPLICATES.items():
        _replace_with_symlink(source, frozen)
    for cache in CACHE_DIRECTORIES:
        if cache.is_dir() and not cache.is_symlink():
            reclaimed += _directory_size(cache)
            shutil.rmtree(cache)
    current = ROOT / "current_best"
    if current.is_symlink():
        if current.resolve() != BEST_RELEASE.resolve():
            current.unlink()
    elif current.exists():
        raise ValueError("current_best exists but is not a symlink")
    if not current.exists():
        current.symlink_to(BEST_RELEASE.resolve(), target_is_directory=True)
    return {
        "contract": "evidence_derived_duplicate_cleanup_v1",
        "reclaimed_bytes": reclaimed,
        "default_release": str(BEST_RELEASE.resolve()),
        "directory_links": len(DIRECTORY_DUPLICATES),
        "file_links": len(FILE_DUPLICATES),
        "frozen_releases_modified": False,
    }


def main() -> int:
    """Run the verified cleanup."""
    print(json.dumps(cleanup(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

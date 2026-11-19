#!/usr/bin/env python3
"""Create a read-only, content-hashed release of the current protocol outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "outputs"
DEFAULT_RELEASE = ROOT / "frozen_releases" / "baseline_20260820_756aa9c3"


class FreezeError(RuntimeError):
    """Raised when the current result is not safe to freeze."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_source(source: Path) -> dict[str, Any]:
    audit = source / "audit_report.md"
    validation = source / "hgb_oof_canonical" / "validation_report.json"
    run_manifest = source / "hgb_oof_canonical" / "run_manifest.json"
    if "Status: **COMPLETE**" not in audit.read_text(encoding="utf-8"):
        raise FreezeError("Protocol audit is not COMPLETE")
    validation_payload = json.loads(validation.read_text(encoding="utf-8"))
    run_payload = json.loads(run_manifest.read_text(encoding="utf-8"))
    if validation_payload.get("passed") is not True:
        raise FreezeError("HGB OOF validation did not pass")
    if run_payload.get("checkpoint_count") != 84:
        raise FreezeError("HGB OOF checkpoint count is not 84")
    return {
        "audit_deterministic_hash": "756aa9c34bb6018f294fe02d9ddf6ad52a6118c500a9d4fdf54b671f3fd0292b",
        "feature_set_freeze_hash": run_payload["feature_set_freeze_hash"],
        "hgb_validation_contract": validation_payload["contract"],
        "hgb_validation_passed": True,
        "checkpoint_count": 84,
    }


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        mode = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
        if path.is_dir():
            mode |= stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        path.chmod(mode)
    root.chmod(
        stat.S_IRUSR
        | stat.S_IRGRP
        | stat.S_IROTH
        | stat.S_IXUSR
        | stat.S_IXGRP
        | stat.S_IXOTH
    )


def freeze(source: Path, release: Path) -> dict[str, Any]:
    """Copy the validated result and implementation into a new read-only release."""
    source = source.resolve()
    release = release.resolve()
    if release.exists():
        raise FreezeError(f"Release already exists and will not be overwritten: {release}")
    validation = _validate_source(source)
    release.mkdir(parents=True)
    try:
        shutil.copytree(source, release / "outputs", copy_function=shutil.copy2)
        implementation = release / "implementation"
        implementation.mkdir()
        for path in sorted(ROOT.iterdir()):
            if path.is_file() and path.suffix in {".py", ".sql", ".json"}:
                shutil.copy2(path, implementation / path.name)
        files = sorted(path for path in release.rglob("*") if path.is_file())
        manifest = {
            "contract": "evidence_derived_frozen_release_v1",
            "release_id": release.name,
            "source": str(source),
            "read_only": True,
            "future_experiments_must_use_new_directories": True,
            **validation,
            "files": {
                str(path.relative_to(release)): {
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in files
            },
        }
        manifest_path = release / "frozen_release_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest["manifest_sha256"] = _sha256(manifest_path)
        _make_read_only(release)
        return manifest
    except Exception:
        if release.exists():
            shutil.rmtree(release)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    args = parser.parse_args()
    result = freeze(args.source, args.release)
    print(
        json.dumps(
            {
                "release": str(args.release.resolve()),
                "file_count": len(result["files"]),
                "manifest_sha256": result["manifest_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Freeze a completed four-set evidence-derived experiment as a read-only release."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
from pathlib import Path
from typing import Any

EXPECTED_FEATURE_COUNTS = {
    "strict": 7,
    "primary": 16,
    "expanded": 153,
    "broad_t0": 219,
}


class FreezeError(RuntimeError):
    """Raised when an experiment is incomplete or cannot be frozen safely."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_fixed_experiment(source: Path) -> dict[str, Any]:
    """Validate the fixed-parameter four-set experiment layout."""
    audit = source / "audit_report.md"
    matrix_manifest = source / "training_matrix_manifest.json"
    run_manifest = source / "hgb_oof" / "run_manifest.json"
    validation_report = source / "hgb_oof" / "validation_report.json"
    required = (audit, matrix_manifest, run_manifest, validation_report)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FreezeError(f"required freeze inputs are missing: {missing}")
    if "Status: **COMPLETE**" not in audit.read_text(encoding="utf-8"):
        raise FreezeError("experiment audit is not COMPLETE")
    matrices = json.loads(matrix_manifest.read_text(encoding="utf-8"))
    observed_counts = {
        name: len(definition.get("feature_names") or [])
        for name, definition in (matrices.get("sets") or {}).items()
    }
    if observed_counts != EXPECTED_FEATURE_COUNTS:
        raise FreezeError(f"feature counts differ from 7/16/153/219: {observed_counts}")
    run = json.loads(run_manifest.read_text(encoding="utf-8"))
    validation = json.loads(validation_report.read_text(encoding="utf-8"))
    if run.get("checkpoint_count") != 84:
        raise FreezeError("HGB OOF checkpoint count is not 84")
    if validation.get("passed") is not True:
        raise FreezeError("HGB OOF validation did not pass")
    return {
        "result_scope": "fixed_parameter_four_set_oof",
        "feature_counts": observed_counts,
        "feature_set_freeze_hash": run.get("feature_set_freeze_hash"),
        "protocol_hash": run.get("protocol_hash"),
        "checkpoint_count": 84,
        "hgb_validation_contract": validation.get("contract"),
        "hgb_validation_passed": True,
    }


def _validate_nested_tuned_experiment(source: Path) -> dict[str, Any]:
    """Validate the horizon-specific nested tuning experiment layout."""
    run_path = source / "run_manifest.json"
    validation_path = source / "validation_report.json"
    tuning_validation_path = source / "tuning_validation_report.json"
    comparison_path = source / "comparison_vs_fixed.csv"
    required = (run_path, validation_path, tuning_validation_path, comparison_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FreezeError(f"required tuned freeze inputs are missing: {missing}")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    tuning_validation = json.loads(tuning_validation_path.read_text(encoding="utf-8"))
    if run.get("result_scope") != (
        "horizon_specific_nested_temporal_tuned_four_set_oof"
    ):
        raise FreezeError("result is not the registered nested tuning scope")
    if run.get("feature_counts") != EXPECTED_FEATURE_COUNTS:
        raise FreezeError("tuned feature counts differ from 7/16/153/219")
    tuning = run.get("parameter_tuning") or {}
    if tuning.get("outer_test_labels_used_for_selection") is not False:
        raise FreezeError("outer test labels may have entered parameter selection")
    if run.get("checkpoint_count") != 84:
        raise FreezeError("tuned HGB OOF checkpoint count is not 84")
    if validation.get("passed") is not True:
        raise FreezeError("canonical tuned OOF validation did not pass")
    if tuning_validation.get("passed") is not True:
        raise FreezeError("nested tuning validation did not pass")
    return {
        "result_scope": run["result_scope"],
        "feature_counts": run["feature_counts"],
        "feature_set_freeze_hash": run.get("feature_set_freeze_hash"),
        "protocol_hash": run.get("protocol_hash"),
        "search_space_hash": tuning.get("search_space_hash"),
        "checkpoint_count": 84,
        "prediction_rows": validation.get("prediction_rows"),
        "hgb_validation_contract": validation.get("contract"),
        "hgb_validation_passed": True,
        "tuning_validation_contract": tuning_validation.get("contract"),
        "tuning_validation_passed": True,
    }


def validate_source(source: Path) -> dict[str, Any]:
    """Fail closed unless a supported experiment layout is complete."""
    if (source / "tuning_validation_report.json").is_file():
        return _validate_nested_tuned_experiment(source)
    return _validate_fixed_experiment(source)


def make_read_only(root: Path) -> None:
    """Remove write bits from all files and directories in a release."""
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_dir():
            path.chmod(0o555)
        else:
            path.chmod(0o444)
    root.chmod(0o555)


def freeze(source: Path, release: Path) -> dict[str, Any]:
    """Copy a validated experiment into a new immutable directory."""
    source = source.resolve()
    release = release.resolve()
    if release.exists():
        raise FreezeError(f"release already exists: {release}")
    validation = validate_source(source)
    release.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(source, release, copy_function=shutil.copy2)
        files = sorted(path for path in release.rglob("*") if path.is_file())
        manifest: dict[str, Any] = {
            "contract": "evidence_derived_experiment_frozen_release_v1",
            "release_id": release.name,
            "source": str(source),
            "read_only": True,
            "future_experiments_must_use_new_directories": True,
            **validation,
            "files": {
                str(path.relative_to(release)): {
                    "sha256": sha256_file(path),
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
        manifest["manifest_sha256"] = sha256_file(manifest_path)
        make_read_only(release)
        return manifest
    except Exception:
        if release.exists():
            for path in release.rglob("*"):
                path.chmod(path.stat().st_mode | stat.S_IWUSR)
            release.chmod(release.stat().st_mode | stat.S_IWUSR)
            shutil.rmtree(release)
        raise


def main() -> int:
    """Run the command-line freezer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
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

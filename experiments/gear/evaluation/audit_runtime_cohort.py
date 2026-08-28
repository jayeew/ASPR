"""Require one immutable runtime and configuration fingerprint across a cohort."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

SHA256_LENGTH = len("sha256:") + 64


def audit_runtime_cohort(
    manifest_path: Path,
    runs_dir: Path,
    output_path: Path,
    *,
    expected_code_sha256: str | None = None,
    expected_config_sha256: str | None = None,
) -> dict[str, Any]:
    """Fail closed unless every assigned case was produced by one code snapshot."""
    cases = json.loads(manifest_path.read_text(encoding="utf-8")).get("cases", [])
    if not cases:
        raise ValueError("cohort manifest has no cases")
    rows = [_read_run_manifest(runs_dir, str(case["case_id"])) for case in cases]
    code_hashes = Counter(str(row.get("runtime_code_sha256", "")) for row in rows)
    config_hashes = Counter(str(row.get("runtime_config_sha256", "")) for row in rows)
    config_versions = Counter(str(row.get("config_version", "")) for row in rows)
    for label, values in (
        ("runtime code", code_hashes),
        ("runtime config", config_hashes),
    ):
        if any(not _valid_sha256(value) for value in values):
            raise ValueError(f"{label} fingerprint missing or invalid")
        if len(values) != 1:
            raise ValueError(f"mixed {label} fingerprints: {sorted(values)}")
    if len(config_versions) != 1 or "" in config_versions:
        raise ValueError("mixed or missing config versions")
    actual_code_sha256 = next(iter(code_hashes))
    actual_config_sha256 = next(iter(config_hashes))
    if expected_code_sha256 is not None and actual_code_sha256 != expected_code_sha256:
        raise ValueError("runtime code fingerprint differs from frozen expectation")
    if (
        expected_config_sha256 is not None
        and actual_config_sha256 != expected_config_sha256
    ):
        raise ValueError("runtime config fingerprint differs from frozen expectation")
    source_counts = {int(row.get("runtime_source_file_count", 0)) for row in rows}
    if len(source_counts) != 1 or next(iter(source_counts)) <= 0:
        raise ValueError("runtime source file count is missing or inconsistent")
    statuses = Counter(str(row.get("status", "unknown")) for row in rows)
    result = {
        "contract": "gear_runtime_cohort_fingerprint_audit_v1",
        "passed": True,
        "cases": len(cases),
        "runtime_code_sha256": actual_code_sha256,
        "runtime_config_sha256": actual_config_sha256,
        "runtime_source_file_count": next(iter(source_counts)),
        "config_version": next(iter(config_versions)),
        "status_counts": dict(sorted(statuses.items())),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _read_run_manifest(runs_dir: Path, case_id: str) -> dict[str, Any]:
    path = runs_dir / case_id / "run_manifest.json"
    if not path.is_file():
        raise ValueError(f"run manifest missing: {case_id}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("contract") != "aspr_gear_run_manifest_v3":
        raise ValueError(f"run manifest contract changed: {case_id}")
    return payload


def _valid_sha256(value: str) -> bool:
    if len(value) != SHA256_LENGTH or not value.startswith("sha256:"):
        return False
    try:
        int(value.removeprefix("sha256:"), 16)
    except ValueError:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-code-sha256")
    parser.add_argument("--expected-config-sha256")
    args = parser.parse_args()
    result = audit_runtime_cohort(
        args.manifest,
        args.runs_dir,
        args.output,
        expected_code_sha256=args.expected_code_sha256,
        expected_config_sha256=args.expected_config_sha256,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["audit_runtime_cohort"]

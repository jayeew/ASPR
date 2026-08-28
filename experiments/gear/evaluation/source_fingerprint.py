"""Fingerprint every source file that can affect the rescue replay or evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOTS = (
    PROJECT_ROOT / "gear",
    PROJECT_ROOT / "experiments/gear",
    PROJECT_ROOT / "scripts",
)
SOURCE_SUFFIXES = {".py", ".sh"}


def rescue_source_fingerprint() -> tuple[str, int]:
    """Hash runtime, evaluation, orchestration, and promotion source trees."""
    paths = sorted(
        (
            path
            for root in SOURCE_ROOTS
            for path in root.rglob("*")
            if path.is_file() and path.suffix in SOURCE_SUFFIXES
        ),
        key=lambda path: path.relative_to(PROJECT_ROOT).as_posix(),
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest(), len(paths)


def audit_expected_source(expected_sha256: str) -> dict[str, Any]:
    """Fail closed when the current tree differs from the frozen snapshot."""
    observed, count = rescue_source_fingerprint()
    if observed != expected_sha256:
        raise ValueError("rescue source fingerprint differs from frozen expectation")
    return {
        "contract": "gear_rescue_source_fingerprint_audit_v1",
        "passed": True,
        "source_sha256": observed,
        "source_file_count": count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.expected_sha256:
        result = audit_expected_source(args.expected_sha256)
    else:
        sha256, count = rescue_source_fingerprint()
        result = {"source_sha256": sha256, "source_file_count": count}
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["audit_expected_source", "rescue_source_fingerprint"]

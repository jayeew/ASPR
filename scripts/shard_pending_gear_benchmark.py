#!/usr/bin/env python3
"""Freeze disjoint manifests for safely parallel GEAR benchmark sessions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _active_case_ids(runs_dir: Path) -> set[str]:
    active: set[str] = set()
    proc = Path("/proc")
    if not proc.is_dir():
        return active
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        args = [item.decode("utf-8", errors="replace") for item in argv if item]
        if not _is_gear_review(args) or "--output-dir" not in args:
            continue
        position = args.index("--output-dir")
        if position + 1 >= len(args):
            continue
        case_dir = Path(args[position + 1]).resolve()
        if case_dir.parent == runs_dir.resolve():
            active.add(case_dir.name)
    return active


def _is_gear_review(args: list[str]) -> bool:
    return any(
        args[index : index + 3] == ["-m", "gear", "review"]
        for index in range(max(0, len(args) - 2))
    )


def _existing_status(case_dir: Path) -> str | None:
    bundle = case_dir / "review_bundle.json"
    if not bundle.is_file():
        return None
    try:
        payload = json.loads(bundle.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    status = payload.get("status")
    return str(status) if status is not None else None


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_shards(
    manifest: Path,
    runs_dir: Path,
    shard_count: int,
    *,
    retry_failed: bool = False,
    reverse_pending: bool = False,
    explicit_exclusions: set[str] | None = None,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list):
        raise TypeError("manifest must be an object containing a cases list")
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    case_ids = [str(case.get("case_id") or "") for case in cases]
    if any(not case_id for case_id in case_ids):
        raise ValueError("every case must have a non-empty case_id")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("manifest contains duplicate case_id values")

    active = _active_case_ids(runs_dir)
    excluded = active | (explicit_exclusions or set())
    completed: list[str] = []
    pending: list[dict[str, Any]] = []
    for case, case_id in zip(cases, case_ids, strict=True):
        status = _existing_status(runs_dir / case_id)
        if status is not None and not (retry_failed and status == "failed"):
            completed.append(case_id)
        elif case_id not in excluded:
            pending.append(case)
    if reverse_pending:
        pending.reverse()

    shards: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    for index, case in enumerate(pending):
        shards[index % shard_count].append(case)
    audit = {
        "schema_version": "gear_benchmark_shards_v1",
        "source_manifest": str(manifest.resolve()),
        "source_manifest_sha256": _source_sha256(manifest),
        "runs_dir": str(runs_dir.resolve()),
        "source_case_count": len(cases),
        "completed_case_count": len(completed),
        "excluded_active_case_ids": sorted(active),
        "explicit_exclusion_case_ids": sorted(explicit_exclusions or set()),
        "pending_case_count": len(pending),
        "pending_order": "reverse" if reverse_pending else "source",
        "shard_case_counts": [len(shard) for shard in shards],
    }
    return shards, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--exclude-case", action="append", default=[])
    args = parser.parse_args()
    shards, audit = build_shards(
        args.manifest,
        args.runs_dir,
        args.shards,
        retry_failed=args.retry_failed,
        reverse_pending=args.reverse,
        explicit_exclusions=set(args.exclude_case),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    for index, cases in enumerate(shards):
        shard_payload = {
            **{key: value for key, value in payload.items() if key != "cases"},
            "cases": cases,
            "shard": {**audit, "shard_index": index},
        }
        path = args.output_dir / f"shard_{index:02d}.json"
        path.write_text(
            json.dumps(shard_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    (args.output_dir / "shard_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

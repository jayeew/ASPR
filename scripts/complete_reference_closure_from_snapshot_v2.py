#!/usr/bin/env python3
"""Complete unresolved reference metadata from every local OpenAlex shard."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

try:
    import orjson
except ImportError:  # pragma: no cover - stdlib fallback remains deterministic
    orjson = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_reference_closure_v5_from_snapshot import (
    iter_snapshot_work_files,
)
from scripts.fetch_openalex_reference_missing_v5 import (
    append_checkpoint_rows,
    load_checkpoint_ids,
    load_reference_ids,
)
from scripts.nature_portfolio_v5 import (
    normalize_openalex_id,
    reference_work_row,
    short_openalex_id,
    utc_now,
)

DEFAULT_SNAPSHOT = Path("/mnt/d/FabCitationData/openalex-snapshot")
DEFAULT_OUTPUT = Path(
    "/mnt/d/aspr_nature_portfolio_v5/openalex_outputs/uncapped_aspr_v2"
)
_WORKER_REFERENCE_IDS: set[str] = set()
_WORK_ID_PREFIX = b'{"id":"'


def _initialize_worker(reference_ids: set[str]) -> None:
    """Freeze the unresolved ID lookup once in each process."""
    global _WORKER_REFERENCE_IDS
    _WORKER_REFERENCE_IDS = reference_ids


def _loads(line: bytes) -> Any:
    return orjson.loads(line) if orjson is not None else json.loads(line)


def _work_id_from_line(line: bytes) -> str:
    """Extract the leading OpenAlex work ID without parsing an unmatched row."""
    if not line.startswith(_WORK_ID_PREFIX):
        return ""
    end = line.find(b'"', len(_WORK_ID_PREFIX))
    if end < 0:
        return ""
    return normalize_openalex_id(
        line[len(_WORK_ID_PREFIX) : end].decode("ascii", errors="ignore")
    )


def _scan_file(path_text: str) -> tuple[str, int, int, list[dict[str, Any]]]:
    """Return normalized in-scope matches from one complete snapshot shard."""
    path = Path(path_text)
    opener = gzip.open if path.suffix == ".gz" else open
    records_seen = 0
    rows: list[dict[str, Any]] = []
    with opener(path, "rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            records_seen += 1
            work_id = _work_id_from_line(line)
            payload = None
            if not work_id:
                payload = _loads(line)
                work_id = (
                    normalize_openalex_id(payload.get("id"))
                    if isinstance(payload, dict)
                    else ""
                )
            if work_id not in _WORKER_REFERENCE_IDS:
                continue
            if payload is None:
                payload = _loads(line)
            if not isinstance(payload, dict):
                continue
            row = reference_work_row(payload, fetched_at=utc_now())
            row["id"] = work_id
            row["short_id"] = short_openalex_id(work_id)
            row["source_provider"] = "openalex_snapshot"
            row["source_dataset"] = (
                "nature_portfolio_v5_reference_closure_snapshot_rescan"
            )
            rows.append(row)
    return str(path), int(path.stat().st_size), records_seen, rows


def _checkpoint_path(root: Path, source_path: Path) -> Path:
    token = hashlib.sha256(str(source_path).encode("utf-8")).hexdigest()[:20]
    return root / f"{token}.json"


def _valid_file_checkpoint(path: Path, source: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("status") == "complete"
        and payload.get("source_file") == str(source)
        and int(payload.get("source_size_bytes", -1)) == int(source.stat().st_size)
    )


def _write_file_checkpoint(
    path: Path,
    *,
    source_file: str,
    source_size_bytes: int,
    records_seen: int,
    matches_added: int,
) -> None:
    payload = {
        "status": "complete",
        "source_file": source_file,
        "source_size_bytes": source_size_bytes,
        "records_seen": records_seen,
        "matches_added": matches_added,
        "completed_at": utc_now(),
    }
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _completed_stats(checkpoint_dir: Path) -> tuple[int, int, int]:
    files = records = matches = 0
    for path in checkpoint_dir.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            continue
        files += 1
        records += int(payload.get("records_seen") or 0)
        matches += int(payload.get("matches_added") or 0)
    return files, records, matches


def complete(args: argparse.Namespace) -> dict[str, Any]:
    """Scan every shard, append newly resolved rows, and freeze an audit."""
    requested_ids = load_reference_ids(args.retry_queue)
    successful_ids = load_checkpoint_ids(args.success_checkpoint)
    unresolved = set(requested_ids) - successful_ids
    files = iter_snapshot_work_files(args.snapshot_dir)
    if args.max_files is not None:
        files = files[: int(args.max_files)]
    args.file_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pending_files = [
        path
        for path in files
        if not _valid_file_checkpoint(
            _checkpoint_path(args.file_checkpoint_dir, path), path
        )
    ]
    print(
        "[Reference snapshot completion] "
        f"unresolved={len(unresolved):,}, files={len(files):,}, "
        f"already_scanned={len(files) - len(pending_files):,}, "
        f"workers={args.workers}",
        flush=True,
    )
    started_at = time.time()
    newly_added = 0
    completed_this_run = 0
    iterator = iter(pending_files)
    pending: dict[Future[tuple[str, int, int, list[dict[str, Any]]]], Path] = {}
    max_pending = max(1, int(args.workers) * 2)

    def consume(result: tuple[str, int, int, list[dict[str, Any]]]) -> None:
        nonlocal newly_added, completed_this_run
        source_file, source_size, records_seen, rows = result
        unique_rows = []
        for row in rows:
            work_id = normalize_openalex_id(row.get("id"))
            if not work_id or work_id in successful_ids:
                continue
            unique_rows.append(row)
            successful_ids.add(work_id)
            unresolved.discard(work_id)
        append_checkpoint_rows(args.success_checkpoint, unique_rows)
        _write_file_checkpoint(
            _checkpoint_path(args.file_checkpoint_dir, Path(source_file)),
            source_file=source_file,
            source_size_bytes=source_size,
            records_seen=records_seen,
            matches_added=len(unique_rows),
        )
        newly_added += len(unique_rows)
        completed_this_run += 1
        if completed_this_run == 1 or completed_this_run % args.progress_every == 0:
            elapsed = max(time.time() - started_at, 0.001)
            print(
                "[Reference snapshot completion] "
                f"files={completed_this_run:,}/{len(pending_files):,}, "
                f"new={newly_added:,}, remaining={len(unresolved):,}, "
                f"files/s={completed_this_run / elapsed:.2f}",
                flush=True,
            )

    with ProcessPoolExecutor(
        max_workers=max(1, int(args.workers)),
        initializer=_initialize_worker,
        initargs=(unresolved,),
    ) as executor:
        while len(pending) < max_pending:
            try:
                path = next(iterator)
            except StopIteration:
                break
            pending[executor.submit(_scan_file, str(path))] = path
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                pending.pop(future)
                consume(future.result())
                try:
                    path = next(iterator)
                except StopIteration:
                    continue
                pending[executor.submit(_scan_file, str(path))] = path

    completed_files, records_seen, checkpoint_matches = _completed_stats(
        args.file_checkpoint_dir
    )
    all_files_scanned = completed_files == len(files)
    manifest = {
        "artifact_kind": "nature_reference_snapshot_completion_v2",
        "created_at": utc_now(),
        "snapshot_dir": str(args.snapshot_dir),
        "retry_queue": str(args.retry_queue),
        "success_checkpoint": str(args.success_checkpoint),
        "file_checkpoint_dir": str(args.file_checkpoint_dir),
        "n_requested_ids": len(requested_ids),
        "n_successful_ids_after_scan": sum(
            work_id in successful_ids for work_id in requested_ids
        ),
        "n_remaining_ids_after_scan": sum(
            work_id not in successful_ids for work_id in requested_ids
        ),
        "n_newly_resolved_this_run": newly_added,
        "n_snapshot_files": len(files),
        "n_snapshot_files_completed": completed_files,
        "n_snapshot_records_scanned": records_seen,
        "n_checkpoint_matches_added": checkpoint_matches,
        "local_snapshot_scan_complete": all_files_scanned,
        "max_files": args.max_files,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not all_files_scanned:
        raise RuntimeError("Not every registered snapshot shard was completed")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument(
        "--retry-queue",
        type=Path,
        default=DEFAULT_OUTPUT / "nature_reference_closure_api_retry_queue.csv",
    )
    parser.add_argument(
        "--success-checkpoint",
        type=Path,
        default=DEFAULT_OUTPUT / "checkpoints" / "reference_missing_online_success.csv",
    )
    parser.add_argument(
        "--file-checkpoint-dir",
        type=Path,
        default=DEFAULT_OUTPUT / "checkpoints" / "reference_snapshot_rescan",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_OUTPUT / "reference_snapshot_rescan_manifest.json",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()
    if args.workers <= 0 or args.progress_every <= 0:
        parser.error("workers and progress-every must be positive")
    return args


def main() -> int:
    manifest = complete(parse_args())
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

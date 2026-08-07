from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    Executor,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from pathlib import Path
from typing import (
    AbstractSet,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
)

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.nature_portfolio_v5 import (  # noqa: E402
    DEFAULT_V5_OUTPUT_DIR,
    normalize_openalex_id,
    read_csv,
    reference_work_row,
    short_openalex_id,
    target_reference_edges,
    utc_now,
    write_json,
)

DEFAULT_OPENALEX_SNAPSHOT_DIR = Path("/mnt/d/FabCitationData/openalex-snapshot")
REFERENCE_WORK_FIELDS = list(
    reference_work_row({"id": "https://openalex.org/W0"}).keys()
)
_PROCESS_REFERENCE_IDS: Set[str] = set()


def iter_snapshot_work_files(snapshot_dir: Path) -> List[Path]:
    works_root = snapshot_dir / "data" / "works"
    if not works_root.exists():
        raise FileNotFoundError(f"OpenAlex works snapshot not found: {works_root}")
    files: List[Path] = []
    for suffix in ("*.jsonl.gz", "*.gz", "*.jsonl"):
        files.extend(path for path in works_root.rglob(suffix) if path.is_file())
    return sorted(set(files))


def iter_work_records(path: Path) -> Iterable[Dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[arg-type]
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"[Snapshot reference closure] JSON 解析失败：{path}:{line_no}: {exc}",
                    file=sys.stderr,
                )
                continue
            if isinstance(payload, dict):
                yield payload


def write_reference_edges_if_needed(args: argparse.Namespace) -> Path:
    edge_path = args.reference_edges or (args.out_dir / "nature_reference_edges.csv")
    if edge_path.exists() and edge_path.stat().st_size > 0 and not args.refresh_edges:
        return edge_path
    targets = read_csv(args.target_works)
    if targets.empty:
        raise FileNotFoundError(f"No target works found: {args.target_works}")
    edges = target_reference_edges(targets)
    if args.max_edges is not None:
        edges = edges.head(int(args.max_edges)).copy()
    edge_path.parent.mkdir(parents=True, exist_ok=True)
    edges.to_csv(edge_path, index=False)
    return edge_path


def collect_reference_ids(
    edge_path: Path, max_reference_ids: Optional[int] = None
) -> Set[str]:
    reference_ids: Set[str] = set()
    with edge_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "target" not in (reader.fieldnames or []):
            raise ValueError(f"Reference edge file has no target column: {edge_path}")
        for row in reader:
            rid = normalize_openalex_id(row.get("target"))
            if rid:
                reference_ids.add(rid)
                if max_reference_ids is not None and len(reference_ids) >= int(
                    max_reference_ids
                ):
                    break
    return reference_ids


def iter_checkpoint_works(checkpoint_jsonl: Path) -> Iterable[Dict[str, Any]]:
    if not checkpoint_jsonl.exists():
        return
    with checkpoint_jsonl.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"[Snapshot reference closure] 跳过 checkpoint 中的坏 JSON 行：{checkpoint_jsonl}:{line_no}",
                    file=sys.stderr,
                )
                continue
            work = payload.get("work", payload) if isinstance(payload, dict) else None
            if isinstance(work, dict):
                yield work


def seed_reference_rows(
    seed_csv: Optional[Path],
    *,
    reference_ids: Set[str],
    found_ids: Set[str],
    writer: csv.DictWriter,
) -> int:
    """Reuse an audited prior reference table before scanning the snapshot."""
    if seed_csv is None:
        return 0
    if not seed_csv.is_file():
        raise FileNotFoundError(f"Seed reference works not found: {seed_csv}")
    added = 0
    with seed_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "id" not in (reader.fieldnames or []):
            raise ValueError(f"Seed reference table has no id column: {seed_csv}")
        for row in reader:
            wid = normalize_openalex_id(row.get("id"))
            if not wid or wid not in reference_ids or wid in found_ids:
                continue
            normalized = {field: row.get(field, "") for field in REFERENCE_WORK_FIELDS}
            normalized["id"] = wid
            writer.writerow(normalized)
            found_ids.add(wid)
            added += 1
    return added


def write_checkpoint_record(handle: Any, work: Dict[str, Any]) -> None:
    handle.write(json.dumps({"work": work}, ensure_ascii=False, sort_keys=True) + "\n")


def scan_snapshot_file(path: Path, reference_ids: AbstractSet[str]) -> Dict[str, Any]:
    matches: List[Dict[str, Any]] = []
    records_seen = 0
    for work in iter_work_records(path):
        records_seen += 1
        wid = normalize_openalex_id(work.get("id"))
        if wid and wid in reference_ids:
            matches.append(work)
    return {"path": str(path), "records_seen": records_seen, "matches": matches}


def initialize_process_reference_ids(reference_ids: AbstractSet[str]) -> None:
    """Initialize one process worker with the immutable lookup set."""
    global _PROCESS_REFERENCE_IDS
    _PROCESS_REFERENCE_IDS = set(reference_ids)


def scan_snapshot_file_process(path: Path) -> Dict[str, Any]:
    """Scan one snapshot shard using the process-local reference lookup."""
    return scan_snapshot_file(path, _PROCESS_REFERENCE_IDS)


def write_matched_work(
    work: Dict[str, Any],
    *,
    reference_ids: Set[str],
    found_ids: Set[str],
    writer: csv.DictWriter,
    checkpoint_handle: Any,
    fetched_at: str,
) -> bool:
    wid = normalize_openalex_id(work.get("id"))
    if not wid or wid not in reference_ids or wid in found_ids:
        return False
    writer.writerow(reference_work_row(work, fetched_at=fetched_at))
    write_checkpoint_record(checkpoint_handle, work)
    found_ids.add(wid)
    return True


def snapshot_reference_closure(args: argparse.Namespace) -> Dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_jsonl = args.checkpoint_jsonl or (
        args.out_dir / "checkpoints" / "reference_works.jsonl"
    )
    checkpoint_jsonl.parent.mkdir(parents=True, exist_ok=True)
    edge_path = write_reference_edges_if_needed(args)
    reference_ids = collect_reference_ids(
        edge_path, max_reference_ids=args.max_reference_ids
    )
    if not reference_ids:
        raise ValueError(f"No reference IDs found from {edge_path}")

    started_at = time.time()
    fetched_at = utc_now()
    found_ids: Set[str] = set()
    reference_csv = args.out_dir / "nature_reference_works.csv"
    tmp_reference_csv = reference_csv.with_suffix(".csv.tmp")
    tmp_reference_csv.parent.mkdir(parents=True, exist_ok=True)

    with tmp_reference_csv.open("w", encoding="utf-8", newline="") as ref_handle:
        writer = csv.DictWriter(
            ref_handle, fieldnames=REFERENCE_WORK_FIELDS, extrasaction="ignore"
        )
        writer.writeheader()
        seed_added = seed_reference_rows(
            args.seed_reference_works,
            reference_ids=reference_ids,
            found_ids=found_ids,
            writer=writer,
        )
        for work in iter_checkpoint_works(checkpoint_jsonl):
            wid = normalize_openalex_id(work.get("id"))
            if wid and wid in reference_ids and wid not in found_ids:
                writer.writerow(reference_work_row(work, fetched_at=fetched_at))
                found_ids.add(wid)

        remaining = set(reference_ids) - found_ids
        work_files = iter_snapshot_work_files(args.snapshot_dir)
        if args.max_snapshot_files is not None:
            work_files = work_files[: int(args.max_snapshot_files)]
        if not args.quiet:
            print(
                f"[Snapshot reference closure] 本地 snapshot 文件数={len(work_files):,}，"
                f"待匹配 reference works={len(remaining):,}，checkpoint 已有={len(found_ids):,}",
                flush=True,
            )
        files_seen = 0
        records_seen = 0
        checkpoint_added = 0
        last_log = time.time()
        workers = max(1, int(args.workers))
        with checkpoint_jsonl.open("a", encoding="utf-8") as checkpoint_handle:
            if workers == 1:
                for path in work_files:
                    files_seen += 1
                    for work in iter_work_records(path):
                        records_seen += 1
                        if write_matched_work(
                            work,
                            reference_ids=remaining,
                            found_ids=found_ids,
                            writer=writer,
                            checkpoint_handle=checkpoint_handle,
                            fetched_at=fetched_at,
                        ):
                            remaining.remove(normalize_openalex_id(work.get("id")))
                            checkpoint_added += 1
                            if checkpoint_added % int(args.flush_every) == 0:
                                checkpoint_handle.flush()
                                ref_handle.flush()
                        if not remaining:
                            break
                    now = time.time()
                    if not args.quiet and (
                        files_seen == 1
                        or files_seen % int(args.log_every_files) == 0
                        or now - last_log >= float(args.progress_every_seconds)
                        or not remaining
                    ):
                        elapsed = max(1.0, now - started_at)
                        print(
                            f"[Snapshot reference closure] 已扫描文件 {files_seen:,}/{len(work_files):,}，"
                            f"记录 {records_seen:,}，本地命中 {len(found_ids):,}/{len(reference_ids):,}，"
                            f"剩余 {len(remaining):,}，速度 {records_seen / elapsed:,.0f} records/s",
                            flush=True,
                        )
                        last_log = now
                    if not remaining:
                        break
            else:
                if not args.quiet:
                    print(
                        f"[Snapshot reference closure] 启用本地并发扫描：workers={workers}",
                        flush=True,
                    )
                scan_reference_ids = frozenset(remaining)
                file_iter = iter(work_files)
                executor: Executor
                scan_callable: Callable[[Path], Dict[str, Any]]
                if args.executor == "process":
                    executor = ProcessPoolExecutor(
                        max_workers=workers,
                        initializer=initialize_process_reference_ids,  # type: ignore[arg-type]
                        initargs=(scan_reference_ids,),  # type: ignore[arg-type]
                    )
                    scan_callable = scan_snapshot_file_process
                else:
                    executor = ThreadPoolExecutor(max_workers=workers)
                    scan_callable = lambda path: scan_snapshot_file(
                        path, scan_reference_ids
                    )
                pending: Dict[Any, Path] = {}

                def submit_next_file() -> bool:
                    try:
                        next_path = next(file_iter)
                    except StopIteration:
                        return False
                    pending[executor.submit(scan_callable, next_path)] = next_path
                    return True

                for _ in range(max(1, workers * 2)):
                    if not submit_next_file():
                        break

                try:
                    while pending:
                        done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
                        for future in done:
                            pending.pop(future, None)
                            result = future.result()
                            files_seen += 1
                            records_seen += int(result.get("records_seen", 0))
                            local_added = 0
                            for work in result.get("matches", []):
                                wid = normalize_openalex_id(work.get("id"))
                                if write_matched_work(
                                    work,
                                    reference_ids=remaining,
                                    found_ids=found_ids,
                                    writer=writer,
                                    checkpoint_handle=checkpoint_handle,
                                    fetched_at=fetched_at,
                                ):
                                    remaining.remove(wid)
                                    checkpoint_added += 1
                                    local_added += 1
                            if (
                                local_added
                                and checkpoint_added % int(args.flush_every) == 0
                            ):
                                checkpoint_handle.flush()
                                ref_handle.flush()
                            now = time.time()
                            if not args.quiet and (
                                files_seen == 1
                                or files_seen % int(args.log_every_files) == 0
                                or now - last_log >= float(args.progress_every_seconds)
                                or not remaining
                            ):
                                elapsed = max(1.0, now - started_at)
                                print(
                                    f"[Snapshot reference closure] 已扫描文件 {files_seen:,}/{len(work_files):,}，"
                                    f"记录 {records_seen:,}，本地命中 {len(found_ids):,}/{len(reference_ids):,}，"
                                    f"剩余 {len(remaining):,}，速度 {records_seen / elapsed:,.0f} records/s，"
                                    f"最近文件命中 {local_added:,}",
                                    flush=True,
                                )
                                last_log = now
                            if not remaining:
                                for queued in pending:
                                    queued.cancel()
                                pending.clear()
                                break
                            submit_next_file()
                        if not remaining:
                            break
                finally:
                    executor.shutdown(wait=True, cancel_futures=True)
                checkpoint_handle.flush()
                ref_handle.flush()
    tmp_reference_csv.replace(reference_csv)

    missing_path = args.out_dir / "nature_reference_closure_local_missing_ids.csv"
    retry_path = args.out_dir / "nature_reference_closure_api_retry_queue.csv"
    missing_rows = [
        {"id": rid, "short_id": short_openalex_id(rid)}
        for rid in sorted(set(reference_ids) - found_ids)
    ]
    pd.DataFrame(missing_rows).to_csv(missing_path, index=False)
    pd.DataFrame(missing_rows).to_csv(retry_path, index=False)

    manifest = {
        "artifact_kind": "nature_portfolio_v5_reference_closure_from_snapshot",
        "created_at": utc_now(),
        "snapshot_dir": str(args.snapshot_dir),
        "target_works": str(args.target_works),
        "reference_edges": str(edge_path),
        "reference_works": str(reference_csv),
        "checkpoint_jsonl": str(checkpoint_jsonl),
        "seed_reference_works": (
            str(args.seed_reference_works) if args.seed_reference_works else None
        ),
        "n_reference_works_seeded": int(seed_added),
        "local_missing_ids": str(missing_path),
        "api_retry_queue": str(retry_path),
        "n_unique_reference_ids": int(len(reference_ids)),
        "n_reference_works_found": int(len(found_ids)),
        "n_reference_works_missing_locally": int(len(missing_rows)),
        "local_snapshot_coverage": float(len(found_ids) / max(1, len(reference_ids))),
        "elapsed_seconds": round(time.time() - started_at, 3),
        "max_reference_ids": args.max_reference_ids,
        "max_snapshot_files": args.max_snapshot_files,
        "workers": int(args.workers),
        "executor": str(args.executor),
    }
    write_json(args.out_dir / "reference_closure_snapshot_manifest.json", manifest)
    if not args.quiet:
        print(
            f"[Snapshot reference closure] 完成：本地命中 {manifest['n_reference_works_found']:,}/"
            f"{manifest['n_unique_reference_ids']:,}，缺失 {manifest['n_reference_works_missing_locally']:,}，"
            f"coverage={manifest['local_snapshot_coverage']:.3f}",
            flush=True,
        )
    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Nature Portfolio v5 reference closure from local OpenAlex snapshot."
    )
    parser.add_argument(
        "--target-works",
        type=Path,
        default=DEFAULT_V5_OUTPUT_DIR / "nature_target_works.csv",
    )
    parser.add_argument("--reference-edges", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_V5_OUTPUT_DIR)
    parser.add_argument(
        "--snapshot-dir", type=Path, default=DEFAULT_OPENALEX_SNAPSHOT_DIR
    )
    parser.add_argument("--checkpoint-jsonl", type=Path, default=None)
    parser.add_argument("--seed-reference-works", type=Path, default=None)
    parser.add_argument("--refresh-edges", action="store_true")
    parser.add_argument("--max-edges", type=int, default=None)
    parser.add_argument("--max-reference-ids", type=int, default=None)
    parser.add_argument("--max-snapshot-files", type=int, default=None)
    parser.add_argument("--log-every-files", type=int, default=25)
    parser.add_argument("--progress-every-seconds", type=float, default=30.0)
    parser.add_argument("--flush-every", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--executor",
        choices=("thread", "process"),
        default="thread",
        help="Use process workers to parallelize gzip and JSON parsing.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    snapshot_reference_closure(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

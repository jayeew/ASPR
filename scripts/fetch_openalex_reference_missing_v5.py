from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import csv
from dataclasses import dataclass
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aspr.env import getenv  # noqa: E402
from scripts.build_openalex_v3_citation_graph import OpenAlexClient, split_api_keys  # noqa: E402
from scripts.nature_portfolio_v5 import (  # noqa: E402
    DEFAULT_V5_OUTPUT_DIR,
    normalize_openalex_id,
    reference_work_row,
    short_openalex_id,
    utc_now,
    write_json,
)


REFERENCE_WORK_FIELDS = list(reference_work_row({"id": "https://openalex.org/W0"}).keys())


@dataclass
class FetchResult:
    """One OpenAlex missing-reference lookup result."""

    reference_id: str
    work: Optional[Dict[str, Any]]


def load_reference_ids(path: Path, max_refs: Optional[int] = None) -> List[str]:
    """Load unique OpenAlex work IDs from a retry queue."""
    if not path.exists():
        raise FileNotFoundError(f"Missing reference retry queue: {path}")
    ids: List[str] = []
    seen: Set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        id_field = "id" if "id" in fields else "target" if "target" in fields else ""
        if not id_field:
            raise ValueError(f"Retry queue must contain an id or target column: {path}")
        for row in reader:
            reference_id = normalize_openalex_id(row.get(id_field))
            if not reference_id or reference_id in seen:
                continue
            seen.add(reference_id)
            ids.append(reference_id)
            if max_refs is not None and len(ids) >= int(max_refs):
                break
    return ids


def load_checkpoint_rows(path: Path) -> Dict[str, Dict[str, str]]:
    """Load the compact normalized success checkpoint."""
    rows: Dict[str, Dict[str, str]] = {}
    if not path.exists() or path.stat().st_size == 0:
        return rows
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            reference_id = normalize_openalex_id(row.get("id"))
            if reference_id:
                row["id"] = reference_id
                rows[reference_id] = row
    return rows


def append_checkpoint_row(path: Path, row: Dict[str, Any]) -> None:
    """Durably append one normalized successful lookup."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REFERENCE_WORK_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def append_failure_rows(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Append failed lookup attempts for audit and later retries."""
    records = list(rows)
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "short_id", "attempted_at", "status"]
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(records)


def fetch_missing_references(
    reference_ids: Sequence[str],
    *,
    checkpoint_path: Path,
    failure_log_path: Path,
    openalex: OpenAlexClient,
    workers: int,
    progress_every: int,
    quiet: bool,
) -> Tuple[Dict[str, Dict[str, str]], Set[str]]:
    """Fetch missing references with bounded concurrency and compact checkpoints."""
    checkpoint_rows = load_checkpoint_rows(checkpoint_path)
    wanted = [reference_id for reference_id in reference_ids if reference_id not in checkpoint_rows]
    failed_ids: Set[str] = set()
    workers = max(1, int(workers))
    started_at = time.time()

    if not quiet:
        print(
            f"[Reference 缺失补抓] 缺失清单={len(reference_ids):,}，"
            f"checkpoint 已成功={len(checkpoint_rows):,}，本轮待抓={len(wanted):,}，workers={workers}",
            flush=True,
        )
    if not wanted:
        return checkpoint_rows, failed_ids

    def fetch_one(reference_id: str) -> FetchResult:
        return FetchResult(reference_id=reference_id, work=openalex.get_work(reference_id))

    completed = 0
    succeeded = 0
    failed = 0
    failure_buffer: List[Dict[str, Any]] = []
    pending: Dict[Future[FetchResult], str] = {}
    iterator = iter(wanted)
    max_pending = max(workers, workers * 4)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for reference_id in iterator:
            pending[executor.submit(fetch_one, reference_id)] = reference_id
            if len(pending) < max_pending:
                continue
            completed_now, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed_now:
                pending.pop(future, None)
                result = future.result()
                completed += 1
                if result.work:
                    row = reference_work_row(result.work)
                    fetched_id = normalize_openalex_id(row.get("id")) or result.reference_id
                    row["id"] = fetched_id
                    row["short_id"] = short_openalex_id(fetched_id)
                    append_checkpoint_row(checkpoint_path, row)
                    checkpoint_rows[fetched_id] = {key: str(value) for key, value in row.items()}
                    succeeded += 1
                else:
                    failed_ids.add(result.reference_id)
                    failed += 1
                    failure_buffer.append(
                        {
                            "id": result.reference_id,
                            "short_id": short_openalex_id(result.reference_id),
                            "attempted_at": utc_now(),
                            "status": "not_found_or_request_failed",
                        }
                    )
                if len(failure_buffer) >= 100:
                    append_failure_rows(failure_log_path, failure_buffer)
                    failure_buffer.clear()
                if not quiet and (completed == 1 or completed % max(1, progress_every) == 0):
                    print_progress(completed, len(wanted), succeeded, failed, started_at)

        while pending:
            completed_now, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed_now:
                pending.pop(future, None)
                result = future.result()
                completed += 1
                if result.work:
                    row = reference_work_row(result.work)
                    fetched_id = normalize_openalex_id(row.get("id")) or result.reference_id
                    row["id"] = fetched_id
                    row["short_id"] = short_openalex_id(fetched_id)
                    append_checkpoint_row(checkpoint_path, row)
                    checkpoint_rows[fetched_id] = {key: str(value) for key, value in row.items()}
                    succeeded += 1
                else:
                    failed_ids.add(result.reference_id)
                    failed += 1
                    failure_buffer.append(
                        {
                            "id": result.reference_id,
                            "short_id": short_openalex_id(result.reference_id),
                            "attempted_at": utc_now(),
                            "status": "not_found_or_request_failed",
                        }
                    )
            if not quiet and (completed == len(wanted) or completed % max(1, progress_every) == 0):
                print_progress(completed, len(wanted), succeeded, failed, started_at)

    append_failure_rows(failure_log_path, failure_buffer)
    return checkpoint_rows, failed_ids


def print_progress(completed: int, total: int, succeeded: int, failed: int, started_at: float) -> None:
    """Print a compact Chinese progress line with rate and ETA."""
    elapsed = max(time.time() - started_at, 0.001)
    rate = completed / elapsed
    remaining = max(0, total - completed)
    eta_seconds = remaining / rate if rate > 0 else 0.0
    print(
        f"[Reference 缺失补抓] 已完成 {completed:,}/{total:,}，成功={succeeded:,}，失败={failed:,}，"
        f"速度={rate:.2f} works/s，预计剩余={eta_seconds / 3600:.2f} 小时",
        flush=True,
    )


def merge_reference_works(
    base_path: Path,
    checkpoint_rows: Dict[str, Dict[str, str]],
    output_path: Path,
) -> int:
    """Stream the base CSV and append successful rows without duplicates."""
    if not base_path.exists():
        raise FileNotFoundError(f"Reference works CSV not found: {base_path}")
    candidate_ids = set(checkpoint_rows)
    existing_candidate_ids: Set[str] = set()
    tmp_path = output_path.with_suffix(output_path.suffix + ".merge.tmp")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0

    with base_path.open("r", encoding="utf-8", newline="") as source, tmp_path.open(
        "w", encoding="utf-8", newline=""
    ) as target:
        reader = csv.DictReader(source)
        fields = list(reader.fieldnames or REFERENCE_WORK_FIELDS)
        writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in reader:
            reference_id = normalize_openalex_id(row.get("id"))
            if reference_id in candidate_ids:
                existing_candidate_ids.add(reference_id)
            writer.writerow(row)
            row_count += 1
        for reference_id, row in checkpoint_rows.items():
            if reference_id in existing_candidate_ids:
                continue
            writer.writerow(row)
            row_count += 1
    tmp_path.replace(output_path)
    return row_count


def write_final_missing(path: Path, reference_ids: Sequence[str], successful_ids: Set[str]) -> int:
    """Write IDs that remain unresolved after all successful checkpoints."""
    remaining = [reference_id for reference_id in reference_ids if reference_id not in successful_ids]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "short_id"])
        writer.writeheader()
        for reference_id in remaining:
            writer.writerow({"id": reference_id, "short_id": short_openalex_id(reference_id)})
    return len(remaining)


def build_missing_reference_topup(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the queue-only online top-up and merge successful rows."""
    args.out_dir.mkdir(parents=True, exist_ok=True)
    reference_ids = load_reference_ids(args.retry_queue, max_refs=args.max_refs)
    openalex = OpenAlexClient(
        api_key=args.openalex_api_key,
        api_keys=split_api_keys(args.openalex_api_keys),
        email=args.openalex_email,
        sleep_seconds=args.sleep_seconds,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
    )
    checkpoint_rows, failed_ids = fetch_missing_references(
        reference_ids,
        checkpoint_path=args.checkpoint_csv,
        failure_log_path=args.failure_log,
        openalex=openalex,
        workers=args.workers,
        progress_every=args.progress_every,
        quiet=args.quiet,
    )
    successful_ids = set(checkpoint_rows) & set(reference_ids)
    if not args.quiet:
        print(f"[Reference 缺失补抓] 开始流式合并参考文献表：{args.reference_works}", flush=True)
    merged_rows = merge_reference_works(args.reference_works, checkpoint_rows, args.reference_works)
    remaining = write_final_missing(args.final_missing, reference_ids, successful_ids)
    manifest = {
        "artifact_kind": "nature_portfolio_v5_reference_missing_online_topup",
        "created_at": utc_now(),
        "retry_queue": str(args.retry_queue),
        "reference_works": str(args.reference_works),
        "checkpoint_csv": str(args.checkpoint_csv),
        "failure_log": str(args.failure_log),
        "final_missing": str(args.final_missing),
        "n_requested_ids": len(reference_ids),
        "n_successful_ids": len(successful_ids),
        "n_failed_this_run": len(failed_ids),
        "n_remaining_ids": remaining,
        "n_merged_reference_rows": merged_rows,
        "workers": int(args.workers),
    }
    write_json(args.manifest, manifest)
    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Online top-up for OpenAlex reference IDs missing from the local snapshot.")
    parser.add_argument(
        "--retry-queue",
        type=Path,
        default=DEFAULT_V5_OUTPUT_DIR / "nature_reference_closure_api_retry_queue.csv",
    )
    parser.add_argument(
        "--reference-works",
        type=Path,
        default=DEFAULT_V5_OUTPUT_DIR / "nature_reference_works.csv",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_V5_OUTPUT_DIR)
    parser.add_argument("--checkpoint-csv", type=Path, default=None)
    parser.add_argument("--failure-log", type=Path, default=None)
    parser.add_argument("--final-missing", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--max-refs", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--openalex-api-key", default=getenv("OPENALEX_API_KEY"))
    parser.add_argument("--openalex-api-keys", default=getenv("OPENALEX_API_KEYS"))
    parser.add_argument("--openalex-email", default=getenv("OPENALEX_EMAIL"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    args.checkpoint_csv = args.checkpoint_csv or args.out_dir / "checkpoints" / "reference_missing_online_success.csv"
    args.failure_log = args.failure_log or args.out_dir / "checkpoints" / "reference_missing_online_failures.csv"
    args.final_missing = args.final_missing or args.out_dir / "nature_reference_closure_final_missing_ids.csv"
    args.manifest = args.manifest or args.out_dir / "reference_missing_online_manifest.json"
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the missing-reference online top-up CLI."""
    args = parse_args(argv)
    manifest = build_missing_reference_topup(args)
    if not args.quiet:
        print(
            f"[Reference 缺失补抓] 完成：成功={manifest['n_successful_ids']:,}/"
            f"{manifest['n_requested_ids']:,}，最终缺失={manifest['n_remaining_ids']:,}，"
            f"合并后参考文献={manifest['n_merged_reference_rows']:,}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

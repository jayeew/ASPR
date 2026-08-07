from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_reference_closure_v5_from_snapshot import (
    collect_reference_ids,
)
from scripts.nature_portfolio_v5 import (
    normalize_openalex_id,
    short_openalex_id,
    utc_now,
)


def audit_partial_reference_csv(
    path: Path,
    reference_ids: set[str],
    *,
    progress_every: int,
) -> set[str]:
    """Validate the interrupted CSV and return its unique in-scope work IDs."""

    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Partial reference table not found: {path}")
    with path.open("rb") as handle:
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) != b"\n":
            raise ValueError(
                f"Partial reference table has an incomplete final row: {path}"
            )

    found_ids: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "id" not in (reader.fieldnames or []):
            raise ValueError(f"Reference table has no id column: {path}")
        expected_columns = len(reader.fieldnames or [])
        for row_number, row in enumerate(reader, start=1):
            if None in row or len(row) != expected_columns:
                raise ValueError(
                    f"Malformed CSV record at data row {row_number:,}: {path}"
                )
            work_id = normalize_openalex_id(row.get("id"))
            if not work_id:
                raise ValueError(f"Blank work ID at data row {row_number:,}: {path}")
            if work_id not in reference_ids:
                raise ValueError(
                    f"Out-of-scope work ID at data row {row_number:,}: {work_id}"
                )
            if work_id in found_ids:
                raise ValueError(
                    f"Duplicate reference work at data row {row_number:,}: {work_id}"
                )
            found_ids.add(work_id)
            if row_number == 1 or row_number % progress_every == 0:
                print(
                    f"[Reference recovery] audited_rows={row_number:,}, "
                    f"unique_ids={len(found_ids):,}",
                    flush=True,
                )
    return found_ids


def write_missing_queue(path: Path, missing_ids: set[str]) -> None:
    """Write a deterministic OpenAlex retry queue."""

    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "short_id"])
        writer.writeheader()
        for work_id in sorted(missing_ids):
            writer.writerow({"id": work_id, "short_id": short_openalex_id(work_id)})
    os.replace(temporary, path)


def recover(args: argparse.Namespace) -> dict[str, object]:
    """Promote an audited interrupted snapshot result and queue its misses."""

    reference_ids = collect_reference_ids(args.reference_edges)
    found_ids = audit_partial_reference_csv(
        args.partial_reference_works,
        reference_ids,
        progress_every=int(args.progress_every),
    )
    missing_ids = reference_ids - found_ids
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_reference_works = args.output_dir / "nature_reference_works.csv"
    if final_reference_works.exists() and not args.overwrite:
        raise FileExistsError(
            f"Final reference table already exists: {final_reference_works}"
        )
    if final_reference_works.exists():
        final_reference_works.unlink()
    os.replace(args.partial_reference_works, final_reference_works)

    local_missing = args.output_dir / "nature_reference_closure_local_missing_ids.csv"
    retry_queue = args.output_dir / "nature_reference_closure_api_retry_queue.csv"
    write_missing_queue(local_missing, missing_ids)
    write_missing_queue(retry_queue, missing_ids)

    manifest: dict[str, object] = {
        "artifact_kind": "nature_portfolio_v5_reference_closure_recovered",
        "created_at": utc_now(),
        "reference_edges": str(args.reference_edges.resolve()),
        "reference_works": str(final_reference_works.resolve()),
        "checkpoint_jsonl": str(args.checkpoint_jsonl.resolve()),
        "n_unique_reference_ids": len(reference_ids),
        "n_reference_works_found": len(found_ids),
        "n_reference_works_missing_locally": len(missing_ids),
        "local_snapshot_coverage": len(found_ids) / max(1, len(reference_ids)),
        "local_snapshot_scan_complete": False,
        "recovery_reason": "worker_memory_exhaustion_after_checkpoint_flush",
        "api_top_up_required": bool(missing_ids),
        "local_missing_ids": str(local_missing.resolve()),
        "api_retry_queue": str(retry_queue.resolve()),
        "primary_key": ["id"],
    }
    manifest_path = args.output_dir / "reference_closure_snapshot_manifest.json"
    temporary_manifest = manifest_path.with_name(
        f".{manifest_path.name}.tmp-{os.getpid()}"
    )
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    print(
        "[Reference recovery] complete: "
        f"found={len(found_ids):,}/{len(reference_ids):,}, "
        f"api_pending={len(missing_ids):,}",
        flush=True,
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recover an interrupted reference-closure CSV after validating every "
            "row, then emit the exact API top-up queue."
        )
    )
    parser.add_argument("--reference-edges", type=Path, required=True)
    parser.add_argument("--partial-reference-works", type=Path, required=True)
    parser.add_argument("--checkpoint-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=250_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.progress_every <= 0:
        parser.error("--progress-every must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    recover(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

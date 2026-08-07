#!/usr/bin/env python3
"""Scan OpenAlex work shards through native Windows paths.

The generated per-shard checkpoints are byte-schema compatible with
``extract_target_metadata`` in ``openalex_controls_v6_1.py``.  This helper is
useful when the snapshot resides on an NTFS volume and WSL's mounted-drive
reader is substantially slower than native Windows Python.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any

_WORK_ID_PREFIX = b'{"id":"'
_TARGET_IDS: set[str] = set()


def normalize_openalex_id(value: Any) -> str:
    """Return a canonical OpenAlex URL or an empty string."""
    text = str(value or "").strip()
    if not text:
        return ""
    suffix = text.rstrip("/").rsplit("/", 1)[-1]
    if not suffix:
        return ""
    return f"https://openalex.org/{suffix}"


def initialize_worker(target_ids_path: str) -> None:
    """Load target identifiers once per worker process."""
    global _TARGET_IDS
    with Path(target_ids_path).open("r", encoding="utf-8") as stream:
        _TARGET_IDS = {line.strip() for line in stream if line.strip()}


def record_work_id(line: bytes) -> str:
    """Extract an OpenAlex work URL without parsing non-target JSON."""
    if not line.startswith(_WORK_ID_PREFIX):
        return ""
    end = line.find(b'"', len(_WORK_ID_PREFIX))
    if end < 0:
        return ""
    return normalize_openalex_id(
        line[len(_WORK_ID_PREFIX) : end].decode("ascii", errors="ignore")
    )


def target_metadata(record: dict[str, Any], source_file: str) -> dict[str, Any]:
    """Project a full work record to the frozen v6.1 control schema."""
    authorships = record.get("authorships") or []
    author_ids = {
        normalize_openalex_id((item.get("author") or {}).get("id"))
        for item in authorships
        if isinstance(item, dict)
    }
    author_ids.discard("")
    institution_ids: set[str] = set()
    countries: set[str] = set()
    for authorship in authorships:
        if not isinstance(authorship, dict):
            continue
        countries.update(
            str(value).strip()
            for value in authorship.get("countries") or []
            if str(value).strip()
        )
        for institution in authorship.get("institutions") or []:
            if not isinstance(institution, dict):
                continue
            institution_id = normalize_openalex_id(institution.get("id"))
            if institution_id:
                institution_ids.add(institution_id)
            country = str(institution.get("country_code") or "").strip()
            if country:
                countries.add(country)
        for affiliation in authorship.get("affiliations") or []:
            if not isinstance(affiliation, dict):
                continue
            institution_ids.update(
                normalized
                for value in affiliation.get("institution_ids") or []
                if (normalized := normalize_openalex_id(value))
            )
    authors_count = record.get("authors_count")
    institutions_count = record.get("institutions_distinct_count")
    countries_count = record.get("countries_distinct_count")
    canonical_record = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "paper_id": normalize_openalex_id(record.get("id")),
        "openalex_updated_date": str(record.get("updated_date") or ""),
        "openalex_author_count": (
            int(authors_count)
            if isinstance(authors_count, (int, float)) and authors_count >= 0
            else len(author_ids)
        ),
        "openalex_institution_count": (
            int(institutions_count)
            if isinstance(institutions_count, (int, float)) and institutions_count >= 0
            else len(institution_ids)
        ),
        "openalex_country_count": (
            int(countries_count)
            if isinstance(countries_count, (int, float)) and countries_count >= 0
            else len(countries)
        ),
        "openalex_author_ids": sorted(author_ids),
        "metadata_source_file": source_file,
        "raw_record_sha256": (f"sha256:{hashlib.sha256(canonical_record).hexdigest()}"),
    }


def scan_one_file(arguments: tuple[str, str]) -> tuple[str, int, list[dict[str, Any]]]:
    """Scan one gzip shard and return only matching metadata rows."""
    native_path, source_file = arguments
    rows: list[dict[str, Any]] = []
    record_count = 0
    with gzip.open(native_path, "rb") as stream:
        for line in stream:
            record_count += 1
            work_id = record_work_id(line)
            if work_id and work_id in _TARGET_IDS:
                rows.append(target_metadata(json.loads(line), source_file))
    return source_file, record_count, rows


def checkpoint_token(source_file: str) -> str:
    """Return the token used by the WSL materializer."""
    return hashlib.sha256(source_file.encode("utf-8")).hexdigest()[:20]


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--posix-snapshot-root", required=True)
    parser.add_argument("--target-ids", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    return parser


def main() -> None:
    """Scan every uncheckpointed work shard and emit an audit summary."""
    args = build_parser().parse_args()
    snapshot_root = args.snapshot_root.resolve()
    checkpoint_dir = args.checkpoint_dir.resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    posix_root = PurePosixPath(args.posix_snapshot_root)
    work_root = snapshot_root / "data" / "works"
    files = sorted(
        work_root.glob("updated_date=*/part_*.gz"),
        key=lambda path: (path.parent.name, path.name),
        reverse=True,
    )
    pending: list[tuple[str, str]] = []
    for path in files:
        relative = path.relative_to(snapshot_root)
        source_file = str(posix_root.joinpath(*relative.parts))
        checkpoint = checkpoint_dir / f"{checkpoint_token(source_file)}.json"
        if not checkpoint.is_file():
            pending.append((str(path), source_file))
    completed = len(files) - len(pending)
    matched = 0
    scanned_records = 0
    print(
        json.dumps(
            {
                "registered_files": len(files),
                "checkpointed_files": completed,
                "pending_files": len(pending),
                "workers": int(args.workers),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    with ProcessPoolExecutor(
        max_workers=max(1, int(args.workers)),
        initializer=initialize_worker,
        initargs=(str(args.target_ids.resolve()),),
    ) as executor:
        futures = {executor.submit(scan_one_file, item): item[1] for item in pending}
        for future in as_completed(futures):
            source_file, record_count, rows = future.result()
            native_path = snapshot_root.joinpath(
                *PurePosixPath(source_file).relative_to(posix_root).parts
            )
            payload = {
                "source_file": source_file,
                "source_size_bytes": native_path.stat().st_size,
                "record_count": int(record_count),
                "rows": rows,
            }
            checkpoint = checkpoint_dir / f"{checkpoint_token(source_file)}.json"
            checkpoint.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            completed += 1
            matched += len(rows)
            scanned_records += int(record_count)
            if completed % 25 == 0 or completed == len(files):
                print(
                    json.dumps(
                        {
                            "completed_files": completed,
                            "registered_files": len(files),
                            "matched_rows_this_run": matched,
                            "records_scanned_this_run": scanned_records,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()

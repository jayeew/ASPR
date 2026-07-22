"""Read-only adapter and resumable recovery for Nature Portfolio v5 assets.

The adapter deliberately treats the v5 directory as a raw source.  It never
uses an unfinished ``*.tmp`` file as a formal input and it unwraps JSONL one
record at a time so the large reference checkpoint is never loaded in memory.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from aspr.corpus import normalize_openalex_id, short_openalex_id


TARGET_FILENAME = "nature_target_works.csv"
PAPER_REFERENCE_FILENAME = "nature_reference_edges.csv"
REFERENCE_FILENAME = "nature_reference_works.csv"
REFERENCE_CHECKPOINT = Path("checkpoints/reference_works.jsonl")
FUTURE_CITERS_FILENAME = "nature_future_citers.csv"
FUTURE_DELTAS_FILENAME = "nature_future_graph_deltas.csv"
RECOVERY_DIRNAME = "reference_closure_recovery"
SNAPSHOT_REFERENCE_MANIFEST = "reference_closure_snapshot_manifest.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _manifest_identity(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(serialized).hexdigest()}"


def audit_reference_recovery(
    source_dir: Path,
    *,
    verify_reference_hash: bool = True,
) -> Dict[str, Any]:
    """Audit the formal V5 recovery marker, manifest, and reference output."""

    root = Path(source_dir)
    recovery_dir = root / RECOVERY_DIRNAME
    manifest_path = recovery_dir / "manifest.json"
    success_path = recovery_dir / "_SUCCESS"
    errors: List[str] = []
    manifest: Dict[str, Any] = {}
    if not manifest_path.is_file():
        errors.append("missing recovery manifest")
    else:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = payload if isinstance(payload, dict) else {}
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid recovery manifest: {exc}")
    if manifest.get("stage_status") != "complete":
        errors.append("recovery stage is not complete")
    if int(manifest.get("bad_json_records_ledger_total") or 0) != 0:
        errors.append("recovery contains malformed snapshot records")
    if not success_path.is_file():
        errors.append("missing recovery _SUCCESS marker")
    elif manifest:
        marker = success_path.read_text(encoding="ascii").strip()
        if marker != _manifest_identity(manifest):
            errors.append("recovery _SUCCESS does not match manifest identity")
    reference_path = root / REFERENCE_FILENAME
    if not reference_path.is_file():
        errors.append("formal reference output is missing")
    expected_hash = str(manifest.get("formal_reference_sha256") or "")
    if not expected_hash:
        errors.append("formal reference hash is missing")
    source_inventory = manifest.get("source_inventory", {})
    verified_inventory: Dict[str, bool] = {}
    expected_inventory = {
        "target_works": root / TARGET_FILENAME,
        "paper_references": root / PAPER_REFERENCE_FILENAME,
        "reference_works": reference_path,
    }
    if verify_reference_hash:
        for name, path in expected_inventory.items():
            record = (
                source_inventory.get(name, {})
                if isinstance(source_inventory, Mapping)
                else {}
            )
            declared_hash = str(record.get("sha256") or "")
            if not declared_hash:
                verified_inventory[name] = False
                continue
            matches = path.is_file() and _sha256(path) == declared_hash
            verified_inventory[name] = bool(matches)
            if not matches:
                errors.append(f"recovery source inventory hash mismatch: {name}")
    elif expected_hash and reference_path.is_file():
        verified_inventory["reference_works"] = True
    return {
        "ok": not errors,
        "errors": errors,
        "manifest": manifest,
        "reference_path": str(reference_path),
        "verified_inventory": verified_inventory,
    }


def audit_snapshot_reference_closure(source_dir: Path) -> Dict[str, Any]:
    """Audit the completed legacy snapshot closure without rerunning it.

    The July 2026 V5 build completed through the legacy snapshot materializer,
    so it has a formal CSV and snapshot manifest but no newer recovery ledger.
    This adapter accepts that provenance only after checking its internal row
    counts, coverage, artifact kind, and formal source files.
    """

    root = Path(source_dir)
    manifest_path = root / SNAPSHOT_REFERENCE_MANIFEST
    errors: List[str] = []
    manifest: Dict[str, Any] = {}
    if not manifest_path.is_file() or manifest_path.is_symlink():
        errors.append("missing snapshot reference-closure manifest")
    else:
        try:
            manifest = _read_json_object(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid snapshot reference-closure manifest: {exc}")
    if manifest.get("artifact_kind") != "nature_portfolio_v5_reference_closure_from_snapshot":
        errors.append("unexpected snapshot reference-closure artifact_kind")
    expected = {
        "target_works": root / TARGET_FILENAME,
        "paper_references": root / PAPER_REFERENCE_FILENAME,
        "reference_works": root / REFERENCE_FILENAME,
    }
    for name, path in expected.items():
        if not path.is_file() or path.is_symlink():
            errors.append(f"formal snapshot source is missing: {name}")
    temporary = root / f"{REFERENCE_FILENAME}.tmp"
    if temporary.exists():
        errors.append("temporary reference CSV exists beside formal output")
    n_unique = int(manifest.get("n_unique_reference_ids") or 0)
    n_found = int(manifest.get("n_reference_works_found") or 0)
    n_missing = int(manifest.get("n_reference_works_missing_locally") or 0)
    coverage = float(manifest.get("local_snapshot_coverage") or 0.0)
    expected_coverage = float(n_found / max(1, n_unique))
    if n_unique <= 0 or n_found <= 0 or n_found + n_missing != n_unique:
        errors.append("snapshot reference counts are inconsistent")
    if abs(coverage - expected_coverage) > 1e-12:
        errors.append("snapshot reference coverage does not match counts")
    return {
        "ok": not errors,
        "errors": errors,
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "coverage": coverage,
        "source_inventory": {
            name: {
                "path": str(path),
                "size_bytes": int(path.stat().st_size) if path.is_file() else 0,
            }
            for name, path in expected.items()
        },
        "provenance_mode": "completed_legacy_snapshot_closure",
    }


def _read_json_object(path: Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def iter_jsonl_records(
    path: Path,
    *,
    unwrap_key: Optional[str] = None,
    strict: bool = True,
    diagnostics: Optional[Dict[str, int]] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield JSON objects from plain or gzip JSONL without buffering the file.

    When ``strict`` is false malformed lines are skipped and counted in the
    supplied diagnostics dictionary.  Formal ingest and recovery use strict
    mode so malformed source records can never silently become missing data.
    """

    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8") as handle:  # type: ignore[arg-type]
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                if diagnostics is not None:
                    diagnostics["bad_json_records"] = diagnostics.get("bad_json_records", 0) + 1
                if strict:
                    raise ValueError(f"Malformed JSONL record {source}:{line_number}: {exc}") from exc
                continue
            if unwrap_key and isinstance(payload, Mapping):
                payload = payload.get(unwrap_key, payload)
            if not isinstance(payload, dict):
                if diagnostics is not None:
                    diagnostics["non_object_records"] = diagnostics.get("non_object_records", 0) + 1
                if strict:
                    raise ValueError(f"JSONL record is not an object: {source}:{line_number}")
                continue
            if diagnostics is not None:
                diagnostics["records"] = diagnostics.get("records", 0) + 1
            yield payload


def iter_v5_checkpoint_works(path: Path, *, strict: bool = True) -> Iterator[Dict[str, Any]]:
    """Stream works from a v5 checkpoint whose rows wrap objects in ``work``."""

    yield from iter_jsonl_records(path, unwrap_key="work", strict=strict)


def _csv_row_count(path: Path) -> int:
    with path.open("rb") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def _csv_unique_count(path: Path, column: str, chunksize: int = 250_000) -> int:
    values: set[str] = set()
    for chunk in pd.read_csv(path, usecols=[column], chunksize=chunksize, dtype=str):
        values.update(chunk[column].dropna().astype(str))
    return len(values)


def audit_v5_source(source_dir: Path, *, deep_jsonl: bool = False) -> Dict[str, Any]:
    """Inventory v5 source assets and expose completion/quality blockers."""

    root = Path(source_dir)
    files: Dict[str, Dict[str, Any]] = {}
    expected = {
        "target_works": root / TARGET_FILENAME,
        "paper_references": root / PAPER_REFERENCE_FILENAME,
        "reference_works": root / REFERENCE_FILENAME,
        "reference_checkpoint": root / REFERENCE_CHECKPOINT,
        "future_citers": root / FUTURE_CITERS_FILENAME,
        "future_deltas": root / FUTURE_DELTAS_FILENAME,
    }
    for name, path in expected.items():
        record: Dict[str, Any] = {
            "path": str(path),
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else 0,
        }
        if (
            path.is_file()
            and path.suffix == ".csv"
            and (deep_jsonl or path.stat().st_size <= 100 * 1024 * 1024)
        ):
            record["rows"] = _csv_row_count(path)
        files[name] = record

    tmp_reference = root / f"{REFERENCE_FILENAME}.tmp"
    diagnostics: Dict[str, int] = {}
    if deep_jsonl and expected["reference_checkpoint"].is_file():
        for _ in iter_v5_checkpoint_works(
            expected["reference_checkpoint"], strict=False
        ):
            diagnostics["checkpoint_works"] = diagnostics.get("checkpoint_works", 0) + 1

    target_manifest_path = root / "nature_target_works_manifest.json"
    target_manifest: Dict[str, Any] = {}
    if target_manifest_path.is_file():
        try:
            loaded_manifest = json.loads(target_manifest_path.read_text(encoding="utf-8"))
            target_manifest = loaded_manifest if isinstance(loaded_manifest, dict) else {}
        except json.JSONDecodeError:
            target_manifest = {}
    target_unique = None
    source_count = None
    roster_path = root / "nature_source_roster.csv"
    if roster_path.is_file():
        roster = pd.read_csv(roster_path, usecols=["source_id"], dtype=str)
        source_count = int(roster["source_id"].dropna().nunique())
    if deep_jsonl and expected["target_works"].is_file():
        target_unique = _csv_unique_count(expected["target_works"], "id")
    target_rows = files["target_works"].get("rows", target_manifest.get("n_target_works"))
    blockers: List[str] = []
    if not expected["target_works"].is_file():
        blockers.append("missing_target_works")
    if not expected["paper_references"].is_file():
        blockers.append("missing_paper_references")
    if not expected["reference_works"].is_file():
        blockers.append("reference_closure_incomplete")
    if tmp_reference.is_file() and not expected["reference_works"].is_file():
        blockers.append("temporary_reference_csv_is_not_formal_input")
    if target_rows is not None and target_unique is not None and target_unique != target_rows:
        blockers.append("duplicate_target_paper_id")
    if diagnostics.get("bad_json_records", 0):
        blockers.append("bad_reference_checkpoint_json")
    if source_count != 42:
        blockers.append("source_roster_is_not_locked_42")

    return {
        "artifact_kind": "nature_multihorizon_v5_source_audit",
        "created_at": _utc_now(),
        "source_dir": str(root),
        "files": files,
        "temporary_reference_csv": {
            "path": str(tmp_reference),
            "exists": tmp_reference.is_file(),
            "size_bytes": tmp_reference.stat().st_size if tmp_reference.is_file() else 0,
        },
        "target_rows": target_rows,
        "target_unique_ids": target_unique,
        "target_uniqueness_verified": bool(target_unique is not None),
        "deep_integrity_checked": bool(deep_jsonl),
        "source_roster_path": str(roster_path),
        "source_count": source_count,
        "source_scope": "42 Nature Portfolio sources; not all journals",
        "jsonl_diagnostics": diagnostics,
        "formal_source_ready": not blockers,
        "blockers": blockers,
    }


def _snapshot_files(snapshot_dir: Path) -> List[Path]:
    works_root = Path(snapshot_dir) / "data" / "works"
    if not works_root.is_dir():
        raise FileNotFoundError(f"OpenAlex works snapshot not found: {works_root}")
    paths: set[Path] = set()
    for pattern in ("*.jsonl.gz", "*.gz", "*.jsonl"):
        paths.update(item for item in works_root.rglob(pattern) if item.is_file())
    return sorted(paths)


def _reference_csv_row(work: Mapping[str, Any]) -> Dict[str, Any]:
    topic = work.get("primary_topic") if isinstance(work.get("primary_topic"), Mapping) else {}
    field = topic.get("field") if isinstance(topic.get("field"), Mapping) else {}
    subfield = topic.get("subfield") if isinstance(topic.get("subfield"), Mapping) else {}
    domain = topic.get("domain") if isinstance(topic.get("domain"), Mapping) else {}
    referenced = work.get("referenced_works") if isinstance(work.get("referenced_works"), list) else []
    return {
        "id": normalize_openalex_id(work.get("id")),
        "short_id": short_openalex_id(work.get("id")),
        "doi": work.get("doi") or "",
        "title": work.get("display_name") or work.get("title") or "",
        "year": work.get("publication_year") or work.get("year") or "",
        "openalex_domain": domain.get("display_name", ""),
        "openalex_primary_field": field.get("display_name", ""),
        "openalex_primary_subfield": subfield.get("display_name", ""),
        "primary_topic": topic.get("display_name", ""),
        "display_topic_id": normalize_openalex_id(topic.get("id")),
        "document_type": work.get("type") or "",
        "reference_count": len(referenced),
        "referenced_works": json.dumps(
            [normalize_openalex_id(item) for item in referenced if normalize_openalex_id(item)],
            ensure_ascii=False,
        ),
        "source_provider": "openalex",
        "source_dataset": "nature_multihorizon_v1_reference_closure",
        "fetched_at": _utc_now(),
        "is_target_work": 0,
    }


def _scan_snapshot_file(path: Path, needed_ids: frozenset[str]) -> Dict[str, Any]:
    matches: List[Dict[str, Any]] = []
    diagnostics: Dict[str, int] = {}
    for work in iter_jsonl_records(path, strict=False, diagnostics=diagnostics):
        work_id = normalize_openalex_id(work.get("id"))
        if work_id and work_id in needed_ids:
            matches.append(work)
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "records_seen": diagnostics.get("records", 0),
        "bad_json_records": diagnostics.get("bad_json_records", 0),
        "matches": matches,
    }


def _initialize_recovery_db(db: sqlite3.Connection) -> None:
    db.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE IF NOT EXISTS needed (id TEXT PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS found (id TEXT PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS processed_files (
            path TEXT PRIMARY KEY,
            size_bytes INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL,
            records_seen INTEGER NOT NULL,
            matches_added INTEGER NOT NULL,
            bad_json_records INTEGER NOT NULL,
            completed_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS emitted (id TEXT PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS checkpoint_index (
            path TEXT PRIMARY KEY,
            size_bytes INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL,
            indexed_offset INTEGER NOT NULL,
            records_indexed INTEGER NOT NULL
        );
        """
    )


def _load_needed_ids(db: sqlite3.Connection, edge_path: Path) -> int:
    with edge_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        target_column = "target" if "target" in (reader.fieldnames or []) else "reference_id"
        if target_column not in (reader.fieldnames or []):
            raise ValueError(f"No reference target column in {edge_path}")
        batch: List[tuple[str]] = []
        for row in reader:
            reference_id = normalize_openalex_id(row.get(target_column))
            if reference_id:
                batch.append((reference_id,))
            if len(batch) >= 50_000:
                db.executemany("INSERT OR IGNORE INTO needed(id) VALUES (?)", batch)
                batch.clear()
        if batch:
            db.executemany("INSERT OR IGNORE INTO needed(id) VALUES (?)", batch)
    db.commit()
    return int(db.execute("SELECT COUNT(*) FROM needed").fetchone()[0])


def _index_checkpoint(db: sqlite3.Connection, checkpoint: Path) -> int:
    if not checkpoint.is_file():
        return 0
    stat = checkpoint.stat()
    previous = db.execute(
        "SELECT indexed_offset,records_indexed FROM checkpoint_index WHERE path=?",
        (str(checkpoint),),
    ).fetchone()
    offset = int(previous[0]) if previous is not None else 0
    records_indexed = int(previous[1]) if previous is not None else 0
    if offset < 0 or offset > stat.st_size:
        db.execute("DELETE FROM found")
        db.execute("DELETE FROM checkpoint_index WHERE path=?", (str(checkpoint),))
        db.commit()
        offset = 0
        records_indexed = 0
    batch: List[tuple[str]] = []
    with checkpoint.open("rb") as handle:
        handle.seek(offset)
        while line := handle.readline():
            try:
                payload = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Malformed checkpoint JSON near byte {handle.tell()} in {checkpoint}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Checkpoint row is not an object: {checkpoint}")
            work = payload.get("work", payload)
            if not isinstance(work, dict):
                raise ValueError(f"Checkpoint work is not an object: {checkpoint}")
            work_id = normalize_openalex_id(work.get("id"))
            if work_id:
                batch.append((work_id,))
            records_indexed += 1
            if len(batch) >= 25_000:
                db.executemany("INSERT OR IGNORE INTO found(id) VALUES (?)", batch)
                batch.clear()
                current = checkpoint.stat()
                db.execute(
                    """
                    INSERT OR REPLACE INTO checkpoint_index
                    (path,size_bytes,mtime_ns,indexed_offset,records_indexed)
                    VALUES (?,?,?,?,?)
                    """,
                    (
                        str(checkpoint),
                        int(current.st_size),
                        int(current.st_mtime_ns),
                        int(handle.tell()),
                        int(records_indexed),
                    ),
                )
                db.commit()
        if batch:
            db.executemany("INSERT OR IGNORE INTO found(id) VALUES (?)", batch)
        current = checkpoint.stat()
        db.execute(
            """
            INSERT OR REPLACE INTO checkpoint_index
            (path,size_bytes,mtime_ns,indexed_offset,records_indexed)
            VALUES (?,?,?,?,?)
            """,
            (
                str(checkpoint),
                int(current.st_size),
                int(current.st_mtime_ns),
                int(handle.tell()),
                int(records_indexed),
            ),
        )
    db.commit()
    return int(db.execute("SELECT COUNT(*) FROM found").fetchone()[0])


def _materialize_reference_csv(
    checkpoint: Path,
    output_path: Path,
    db: sqlite3.Connection,
) -> int:
    temporary = output_path.with_name(f".{output_path.name}.building-{os.getpid()}")
    db.execute("DELETE FROM emitted")
    count = 0
    fieldnames: Optional[List[str]] = None
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer: Optional[csv.DictWriter] = None
            for work in iter_v5_checkpoint_works(checkpoint, strict=True):
                row = _reference_csv_row(work)
                work_id = row["id"]
                if not work_id:
                    continue
                inserted = db.execute(
                    "INSERT OR IGNORE INTO emitted(id) VALUES (?)", (work_id,)
                ).rowcount
                if not inserted:
                    continue
                if writer is None:
                    fieldnames = list(row)
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                writer.writerow(row)
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        if fieldnames is None:
            raise ValueError("Reference checkpoint contains no valid works")
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    db.commit()
    return count


def recover_v5_reference_closure(
    source_dir: Path,
    snapshot_dir: Path,
    *,
    resume: bool = True,
    workers: int = 1,
    max_snapshot_files: Optional[int] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Resume reference closure with a per-snapshot-file durable ledger.

    The existing checkpoint is indexed, not loaded.  Each completed snapshot
    file is committed to SQLite only after its matches have been flushed to the
    append-only JSONL checkpoint.  A resumed run therefore skips only files
    whose size and mtime still match the ledger.
    """

    root = Path(source_dir)
    edge_path = root / PAPER_REFERENCE_FILENAME
    checkpoint = root / REFERENCE_CHECKPOINT
    output_path = root / REFERENCE_FILENAME
    recovery_dir = root / RECOVERY_DIRNAME
    db_path = recovery_dir / "snapshot_ledger.sqlite"
    manifest_path = recovery_dir / "manifest.json"
    success_path = recovery_dir / "_SUCCESS"
    snapshot_files = _snapshot_files(Path(snapshot_dir))
    plan = {
        "artifact_kind": "nature_multihorizon_reference_recovery",
        "created_at": _utc_now(),
        "source_dir": str(root),
        "snapshot_dir": str(snapshot_dir),
        "reference_edges": str(edge_path),
        "checkpoint": str(checkpoint),
        "formal_reference_works": str(output_path),
        "ledger": str(db_path),
        "n_snapshot_files": len(snapshot_files),
        "workers": max(1, int(workers)),
        "resume": bool(resume),
        "dry_run": bool(dry_run),
    }
    if dry_run:
        plan["stage_status"] = "dry_run"
        plan["command_summary"] = "index checkpoint, scan unprocessed snapshot files, atomically materialize canonical reference CSV"
        return plan
    if not edge_path.is_file():
        raise FileNotFoundError(edge_path)
    if not resume and (db_path.exists() or output_path.exists()):
        raise FileExistsError("Reference recovery state exists; pass resume=True")

    recovery_dir.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if success_path.exists():
        success_path.unlink()
    with sqlite3.connect(db_path) as db:
        _initialize_recovery_db(db)
        n_needed = _load_needed_ids(db, edge_path)
        n_checkpoint = _index_checkpoint(db, checkpoint)
        remaining_rows = db.execute(
            "SELECT needed.id FROM needed LEFT JOIN found USING(id) WHERE found.id IS NULL"
        ).fetchall()
        needed_remaining = frozenset(str(row[0]) for row in remaining_rows)
        processed = {
            str(row[0]): (int(row[1]), int(row[2]))
            for row in db.execute(
                "SELECT path,size_bytes,mtime_ns FROM processed_files WHERE bad_json_records=0"
            )
        }
        pending = []
        if needed_remaining or any(
            int(row[0]) > 0
            for row in db.execute("SELECT bad_json_records FROM processed_files")
        ):
            for path in snapshot_files:
                stat = path.stat()
                if processed.get(str(path)) == (stat.st_size, stat.st_mtime_ns):
                    continue
                pending.append(path)
        if max_snapshot_files is not None:
            pending = pending[: max(0, int(max_snapshot_files))]

        scanned = 0
        records_seen = 0
        bad_json = 0
        matches_added = 0
        checkpoint_mode = "a" if checkpoint.exists() else "w"
        with checkpoint.open(checkpoint_mode, encoding="utf-8") as checkpoint_handle:
            worker_count = max(1, int(workers))
            executor = ThreadPoolExecutor(max_workers=worker_count)
            try:
                file_iterator = iter(pending)
                futures: Dict[Any, Path] = {}

                def submit_next() -> bool:
                    try:
                        path = next(file_iterator)
                    except StopIteration:
                        return False
                    futures[executor.submit(_scan_snapshot_file, path, needed_remaining)] = path
                    return True

                for _ in range(worker_count * 2):
                    if not submit_next():
                        break
                remaining_count = len(needed_remaining)
                while futures:
                    future = next(as_completed(tuple(futures)))
                    futures.pop(future, None)
                    result = future.result()
                    local_added = 0
                    for work in result["matches"]:
                        work_id = normalize_openalex_id(work.get("id"))
                        inserted = db.execute(
                            "INSERT OR IGNORE INTO found(id) VALUES (?)", (work_id,)
                        ).rowcount
                        if inserted:
                            checkpoint_handle.write(
                                json.dumps({"work": work}, ensure_ascii=False, sort_keys=True) + "\n"
                            )
                            local_added += 1
                            remaining_count -= 1
                    checkpoint_handle.flush()
                    os.fsync(checkpoint_handle.fileno())
                    db.execute(
                        """
                        INSERT OR REPLACE INTO processed_files
                        (path,size_bytes,mtime_ns,records_seen,matches_added,bad_json_records,completed_at)
                        VALUES (?,?,?,?,?,?,?)
                        """,
                        (
                            result["path"],
                            result["size_bytes"],
                            result["mtime_ns"],
                            result["records_seen"],
                            local_added,
                            result["bad_json_records"],
                            _utc_now(),
                        ),
                    )
                    db.commit()
                    scanned += 1
                    records_seen += int(result["records_seen"])
                    bad_json += int(result["bad_json_records"])
                    matches_added += local_added
                    if remaining_count <= 0:
                        for queued in futures:
                            queued.cancel()
                        futures.clear()
                        break
                    submit_next()
            finally:
                executor.shutdown(wait=True, cancel_futures=False)

        n_found = int(
            db.execute(
                "SELECT COUNT(*) FROM found INNER JOIN needed USING(id)"
            ).fetchone()[0]
        )
        all_files_processed = int(
            db.execute("SELECT COUNT(*) FROM processed_files").fetchone()[0]
        ) >= len(snapshot_files)
        total_bad_json = int(
            db.execute("SELECT COALESCE(SUM(bad_json_records),0) FROM processed_files").fetchone()[0]
        )
        complete = n_found == n_needed or all_files_processed
        reference_rows = None
        if complete and total_bad_json == 0:
            reference_rows = _materialize_reference_csv(checkpoint, output_path, db)

    source_inventory: Dict[str, Any] = {}
    formal_reference_sha256: Optional[str] = None
    if complete and total_bad_json == 0 and output_path.is_file():
        formal_reference_sha256 = _sha256(output_path)
        for name, path in (
            ("target_works", root / TARGET_FILENAME),
            ("paper_references", edge_path),
            ("reference_works", output_path),
        ):
            # Recovery can be exercised against a minimal edge/checkpoint fixture
            # (and against legacy stores where the target inventory was moved).
            # A missing optional inventory member must not invalidate an otherwise
            # complete reference-closure recovery; ingest performs the stricter
            # source contract check before modelling.
            if not path.is_file():
                continue
            source_inventory[name] = {
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "rows": _csv_row_count(path),
                "sha256": (
                    formal_reference_sha256
                    if path == output_path
                    else _sha256(path)
                ),
            }
    manifest = {
        **plan,
        "stage_status": "complete" if complete and total_bad_json == 0 else "partial",
        "n_reference_ids": n_needed,
        "n_checkpoint_ids_before_scan": n_checkpoint,
        "n_reference_ids_found": n_found,
        "n_reference_ids_missing": max(0, n_needed - n_found),
        "coverage": float(n_found / max(1, n_needed)),
        "n_files_scanned_this_run": scanned,
        "n_records_seen_this_run": records_seen,
        "n_matches_added_this_run": matches_added,
        "bad_json_records_this_run": bad_json,
        "bad_json_records_ledger_total": total_bad_json,
        "reference_rows": reference_rows,
        "formal_reference_sha256": formal_reference_sha256,
        "source_inventory": source_inventory,
    }
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp-{os.getpid()}")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    if manifest["stage_status"] == "complete":
        success_path.write_text(
            _manifest_identity(manifest) + "\n", encoding="utf-8"
        )
    return manifest


class _ParquetStreamWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.writer: Optional[pq.ParquetWriter] = None
        self.rows = 0

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = pq.ParquetWriter(self.path, table.schema, compression="zstd")
        self.writer.write_table(table)
        self.rows += len(frame)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def _normalized_paper_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    result = chunk.copy()
    result["paper_id"] = result.get("paper_id", result.get("id", "")).map(normalize_openalex_id)
    result["publication_year"] = pd.to_numeric(
        result.get("publication_year", result.get("year")), errors="coerce"
    ).astype("Int64")
    source_names = result.get(
        "source_display_name",
        pd.Series("", index=result.index),
    ).fillna("").astype(str)
    legacy_families = result.get(
        "journal_family",
        pd.Series("", index=result.index),
    ).fillna("").astype(str)
    venue_family = pd.Series("unmapped_venue_family", index=result.index, dtype="string")
    venue_family.loc[source_names.eq("Nature")] = "nature_flagship"
    venue_family.loc[source_names.eq("Nature Communications")] = "nature_communications"
    venue_family.loc[source_names.eq("Scientific Reports")] = "scientific_reports"
    venue_family.loc[legacy_families.eq("nature_research")] = "nature_specialist_research"
    venue_family.loc[legacy_families.eq("communications")] = "communications_series"
    venue_family.loc[legacy_families.eq("npj")] = "npj_series"
    result["venue_family"] = venue_family
    if "work_type" not in result:
        result["work_type"] = result.get("document_type", "")
    return result.drop(columns=["id"], errors="ignore")


def _json_safe_record(row: Mapping[str, Any]) -> Dict[str, Any]:
    record: Dict[str, Any] = {}
    for key, value in row.items():
        if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
            record[str(key)] = None
        elif hasattr(value, "item"):
            record[str(key)] = value.item()
        else:
            record[str(key)] = value
    return record


def _export_json_table(
    db: sqlite3.Connection,
    query: str,
    output_path: Path,
    *,
    transform: Optional[Any] = None,
) -> int:
    writer = _ParquetStreamWriter(output_path)
    cursor = db.execute(query)
    while batch := cursor.fetchmany(25_000):
        records = [json.loads(str(row[0])) for row in batch]
        frame = pd.DataFrame(records)
        if transform is not None:
            frame = transform(frame)
        writer.write(frame)
    writer.close()
    return writer.rows


def _normalized_reference_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    normalized = chunk.rename(columns={"id": "reference_id", "year": "publication_year"}).copy()
    normalized["reference_id"] = normalized["reference_id"].map(normalize_openalex_id)
    normalized["publication_year"] = pd.to_numeric(
        normalized["publication_year"], errors="coerce"
    ).astype("Int64")
    return normalized


def ingest_v5(
    source_dir: Path,
    output_dir: Path,
    *,
    include_legacy_future: bool = True,
) -> Dict[str, Any]:
    """Materialize normalized Parquet tables from completed v5 raw assets."""

    source = Path(source_dir)
    output = Path(output_dir)
    target_path = source / TARGET_FILENAME
    paper_edge_path = source / PAPER_REFERENCE_FILENAME
    reference_path = source / REFERENCE_FILENAME
    if not target_path.is_file() or not paper_edge_path.is_file():
        raise FileNotFoundError("v5 target works and reference edges are required")
    if not reference_path.is_file():
        tmp_path = source / f"{REFERENCE_FILENAME}.tmp"
        detail = f"; unfinished file exists at {tmp_path}" if tmp_path.exists() else ""
        raise FileNotFoundError(f"Formal reference closure is missing{detail}")
    if reference_path.name.endswith(".tmp"):
        raise ValueError("Formal ingest refuses temporary reference files")
    output.mkdir(parents=True, exist_ok=True)

    row_counts: Dict[str, int] = {}
    raw_counts: Dict[str, int] = {}
    duplicates_removed: Dict[str, int] = {}
    dedup_db_path = output / ".ingest_dedup.sqlite"
    with sqlite3.connect(dedup_db_path) as db:
        db.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            CREATE TABLE IF NOT EXISTS papers (paper_id TEXT PRIMARY KEY, row_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS paper_references (
                paper_id TEXT NOT NULL,
                reference_id TEXT NOT NULL,
                PRIMARY KEY(paper_id, reference_id)
            );
            CREATE TABLE IF NOT EXISTS reference_works (reference_id TEXT PRIMARY KEY, row_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS reference_edges (
                source_reference_id TEXT NOT NULL,
                target_reference_id TEXT NOT NULL,
                edge_year INTEGER,
                PRIMARY KEY(source_reference_id, target_reference_id)
            );
            CREATE TABLE IF NOT EXISTS future_citers (
                paper_id TEXT NOT NULL,
                requested_horizon INTEGER NOT NULL,
                citer_id TEXT NOT NULL,
                row_json TEXT NOT NULL,
                PRIMARY KEY(paper_id, requested_horizon, citer_id)
            );
            """
        )

        raw_counts["papers"] = 0
        for chunk in pd.read_csv(target_path, chunksize=100_000, low_memory=False):
            normalized = _normalized_paper_chunk(chunk)
            records = [_json_safe_record(row) for row in normalized.to_dict("records")]
            raw_counts["papers"] += len(records)
            db.executemany(
                "INSERT OR IGNORE INTO papers(paper_id,row_json) VALUES (?,?)",
                [
                    (record["paper_id"], json.dumps(record, ensure_ascii=False, sort_keys=True))
                    for record in records
                    if record.get("paper_id")
                ],
            )
        unique_papers = int(db.execute("SELECT COUNT(*) FROM papers").fetchone()[0])
        if unique_papers != raw_counts["papers"]:
            raise ValueError(
                f"Target paper_id must be unique: rows={raw_counts['papers']}, unique={unique_papers}"
            )
        row_counts["papers"] = _export_json_table(
            db, "SELECT row_json FROM papers ORDER BY paper_id", output / "papers.parquet"
        )

        raw_counts["paper_references"] = 0
        for chunk in pd.read_csv(paper_edge_path, chunksize=250_000, dtype=str):
            source_col = "source" if "source" in chunk else "paper_id"
            target_col = "target" if "target" in chunk else "reference_id"
            records = [
                (normalize_openalex_id(left), normalize_openalex_id(right))
                for left, right in zip(chunk[source_col], chunk[target_col])
            ]
            records = [record for record in records if record[0] and record[1]]
            raw_counts["paper_references"] += len(records)
            db.executemany(
                "INSERT OR IGNORE INTO paper_references(paper_id,reference_id) VALUES (?,?)",
                records,
            )
        paper_reference_writer = _ParquetStreamWriter(output / "paper_references.parquet")
        cursor = db.execute(
            "SELECT paper_id,reference_id FROM paper_references ORDER BY paper_id,reference_id"
        )
        while batch := cursor.fetchmany(100_000):
            paper_reference_writer.write(
                pd.DataFrame(batch, columns=["paper_id", "reference_id"])
            )
        paper_reference_writer.close()
        row_counts["paper_references"] = paper_reference_writer.rows
        duplicates_removed["paper_references"] = (
            raw_counts["paper_references"] - row_counts["paper_references"]
        )

        # The canonical recovered CSV is authoritative because it is already
        # checkpoint-deduplicated and retains each work's referenced_works.
        raw_counts["reference_works"] = 0
        raw_counts["reference_edges"] = 0
        for chunk in pd.read_csv(reference_path, chunksize=100_000, low_memory=False):
            normalized = _normalized_reference_chunk(chunk)
            records = [_json_safe_record(row) for row in normalized.to_dict("records")]
            raw_counts["reference_works"] += len(records)
            db.executemany(
                "INSERT OR IGNORE INTO reference_works(reference_id,row_json) VALUES (?,?)",
                [
                    (
                        record["reference_id"],
                        json.dumps(record, ensure_ascii=False, sort_keys=True),
                    )
                    for record in records
                    if record.get("reference_id")
                ],
            )
            edge_records: List[tuple[str, str, Optional[int]]] = []
            for record in records:
                try:
                    cited = json.loads(record.get("referenced_works") or "[]")
                except (TypeError, json.JSONDecodeError):
                    cited = []
                source_id = str(record.get("reference_id") or "")
                edge_year = record.get("publication_year")
                for target in cited:
                    target_id = normalize_openalex_id(target)
                    if source_id and target_id and source_id != target_id:
                        edge_records.append((source_id, target_id, edge_year))
            raw_counts["reference_edges"] += len(edge_records)
            db.executemany(
                """
                INSERT OR IGNORE INTO reference_edges
                (source_reference_id,target_reference_id,edge_year) VALUES (?,?,?)
                """,
                edge_records,
            )
        row_counts["reference_works"] = _export_json_table(
            db,
            "SELECT row_json FROM reference_works ORDER BY reference_id",
            output / "reference_works.parquet",
        )
        duplicates_removed["reference_works"] = (
            raw_counts["reference_works"] - row_counts["reference_works"]
        )
        reference_edge_writer = _ParquetStreamWriter(output / "reference_edges.parquet")
        cursor = db.execute(
            """
            SELECT source_reference_id,target_reference_id,edge_year
            FROM reference_edges ORDER BY source_reference_id,target_reference_id
            """
        )
        while batch := cursor.fetchmany(100_000):
            reference_edge_writer.write(
                pd.DataFrame(
                    batch,
                    columns=["source_reference_id", "target_reference_id", "edge_year"],
                )
            )
        reference_edge_writer.close()
        row_counts["reference_edges"] = reference_edge_writer.rows
        duplicates_removed["reference_edges"] = (
            raw_counts["reference_edges"] - row_counts["reference_edges"]
        )

        future_citers_path = source / FUTURE_CITERS_FILENAME
        if include_legacy_future and future_citers_path.is_file():
            raw_counts["future_citers"] = 0
            for chunk in pd.read_csv(future_citers_path, chunksize=100_000, low_memory=False):
                chunk["paper_id"] = chunk["paper_id"].map(normalize_openalex_id)
                chunk["citer_id"] = chunk["citer_id"].map(normalize_openalex_id)
                if "requested_horizon" not in chunk:
                    chunk["requested_horizon"] = 8
                records = [_json_safe_record(row) for row in chunk.to_dict("records")]
                raw_counts["future_citers"] += len(records)
                db.executemany(
                    """
                    INSERT OR IGNORE INTO future_citers
                    (paper_id,requested_horizon,citer_id,row_json) VALUES (?,?,?,?)
                    """,
                    [
                        (
                            record["paper_id"],
                            int(record["requested_horizon"]),
                            record["citer_id"],
                            json.dumps(record, ensure_ascii=False, sort_keys=True),
                        )
                        for record in records
                        if record.get("paper_id") and record.get("citer_id")
                    ],
                )
            row_counts["future_citers"] = _export_json_table(
                db,
                "SELECT row_json FROM future_citers ORDER BY paper_id,requested_horizon,citer_id",
                output / "future_citers.parquet",
            )
            duplicates_removed["future_citers"] = (
                raw_counts["future_citers"] - row_counts["future_citers"]
            )

    for suffix in ("", "-wal", "-shm"):
        candidate = Path(f"{dedup_db_path}{suffix}")
        if candidate.exists():
            candidate.unlink()

    # Materialize explicit request status.  In the legacy fetcher an empty
    # checkpoint is a successful zero; a zero delta without a checkpoint may
    # be a swallowed request failure and is therefore never treated as zero.
    future_delta_path = source / FUTURE_DELTAS_FILENAME
    if include_legacy_future and future_delta_path.is_file():
        from .targets import build_future_fetch_status

        deltas = pd.read_csv(future_delta_path, low_memory=False)
        papers_for_status = pd.read_parquet(output / "papers.parquet", columns=["paper_id"])
        future_frame = (
            pd.read_parquet(output / "future_citers.parquet")
            if (output / "future_citers.parquet").is_file()
            else pd.DataFrame(columns=["paper_id", "citer_id"])
        )
        requested_horizon = int(
            pd.to_numeric(deltas.get("tau", pd.Series([8])), errors="coerce").dropna().max()
        )
        checkpoint_dir = source / "checkpoints" / f"future_citers_tau{requested_horizon}"
        explicit_rows: List[Dict[str, Any]] = []
        for row in deltas.to_dict("records"):
            paper_id = normalize_openalex_id(row.get("paper_id"))
            n_returned = pd.to_numeric(row.get("n_future_citers"), errors="coerce")
            checkpoint_file = checkpoint_dir / f"{short_openalex_id(paper_id)}.jsonl"
            numeric_returned = int(n_returned) if pd.notna(n_returned) else 0
            if checkpoint_file.is_file() or numeric_returned > 0:
                status = "zero_success" if numeric_returned == 0 else "success"
            else:
                status = "not_requested_or_failed"
            explicit_rows.append(
                {
                    "paper_id": paper_id,
                    "requested_horizon": requested_horizon,
                    "fetch_status": status,
                    "n_returned": int(n_returned) if pd.notna(n_returned) and status != "not_requested_or_failed" else np.nan,
                    "cap_hit": 0,
                    "attempt_count": 1 if checkpoint_file.is_file() else 0,
                }
            )
        status = build_future_fetch_status(
            papers_for_status,
            future_frame,
            requested_horizon=requested_horizon,
            explicit_status=pd.DataFrame(explicit_rows),
        )
        status.to_parquet(output / "future_fetch_status.parquet", index=False)
        row_counts["future_fetch_status"] = int(len(status))

    manifest = {
        "artifact_kind": "nature_multihorizon_v5_ingest",
        "created_at": _utc_now(),
        "source_dir": str(source),
        "output_dir": str(output),
        "row_counts": row_counts,
        "raw_counts": raw_counts,
        "duplicates_removed": duplicates_removed,
        "legacy_future_imported": bool(include_legacy_future),
        "primary_keys": {
            "papers": ["paper_id"],
            "paper_references": ["paper_id", "reference_id"],
            "reference_works": ["reference_id"],
            "reference_edges": ["source_reference_id", "target_reference_id"],
            "future_citers": ["paper_id", "requested_horizon", "citer_id"],
            "future_fetch_status": ["paper_id", "requested_horizon"],
        },
    }
    (output / "ingest_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "_SUCCESS").write_text("complete\n", encoding="utf-8")
    return manifest

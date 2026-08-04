from __future__ import annotations

import argparse
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping

from common import (
    DATABASE_PATH,
    sha256_bytes,
    sha256_file,
    utc_now,
    write_json,
)
from database import initialize, log_event, require_complete
from indicators import (
    _download_open_pdf,
    _open_pdf_candidate,
    _safe_public_http_url,
    _store_fulltext_acquisition,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "open_fulltexts"
DEFAULT_REPORT = ROOT / "outputs" / "parallel_fulltext_acquisition_v3.json"


def _source_rows(connection: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Return finally included English sources and prior acquisition state."""
    return [
        dict(row)
        for row in connection.execute(
            """
            SELECT r.record_key, r.raw_json, a.candidate_url,
                   a.local_path, a.sha256, a.status
            FROM records r
            JOIN screening_final s USING(record_key)
            LEFT JOIN fulltext_acquisitions a USING(record_key)
            WHERE s.final_decision = 'include'
              AND s.final_language = 'en'
            ORDER BY r.record_key
            """
        )
    ]


def _values(
    job: Mapping[str, Any],
    *,
    status: str,
    final_url: str = "",
    local_path: str = "",
    digest: str = "",
    content_type: str = "",
    byte_count: int = 0,
    error: str = "",
) -> Dict[str, Any]:
    """Build one acquisition table row without changing evidence semantics."""
    return {
        "record_key": job["record_key"],
        "candidate_url": job["candidate_url"],
        "final_url": final_url,
        "local_path": local_path,
        "sha256": digest,
        "access_statement": job["access_statement"],
        "http_content_type": content_type,
        "byte_count": byte_count,
        "status": status,
        "error": error[:1000],
        "fetched_at": utc_now(),
    }


def _download_job(
    job: Mapping[str, Any],
    timeout_seconds: int,
    maximum_bytes: int,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Fetch one open-PDF candidate without touching SQLite."""
    result = dict(
        _download_open_pdf(
            str(job["candidate_url"]),
            timeout_seconds,
            maximum_bytes,
        )
    )
    return dict(job), result


def _store_success(
    connection: sqlite3.Connection,
    output_dir: Path,
    job: Mapping[str, Any],
    result: Mapping[str, Any],
    maximum_bytes: int,
) -> None:
    """Atomically store and register one validated PDF response."""
    body = bytes(result["body"])
    if len(body) > maximum_bytes or not body.lstrip().startswith(b"%PDF-"):
        raise ValueError("Fetcher returned an invalid PDF payload")
    identity_hash = sha256_bytes(
        str(job["record_key"]).encode("utf-8")
    )[:20]
    local_path = (output_dir / f"{identity_hash}.pdf").resolve()
    temporary_path = local_path.with_suffix(".pdf.part")
    temporary_path.write_bytes(body)
    temporary_path.replace(local_path)
    digest = sha256_file(local_path)
    final_url = _safe_public_http_url(
        str(result.get("final_url") or job["candidate_url"])
    )
    _store_fulltext_acquisition(
        connection,
        _values(
            job,
            status="downloaded",
            final_url=final_url,
            local_path=str(local_path),
            digest=digest,
            content_type=str(result.get("content_type") or ""),
            byte_count=len(body),
        ),
    )
    connection.execute(
        """
        INSERT INTO source_snapshots(
            source_id, path, sha256, role, imported_at
        ) VALUES (?, ?, ?, 'candidate_open_fulltext', ?)
        ON CONFLICT(source_id) DO UPDATE SET
            path = excluded.path,
            sha256 = excluded.sha256,
            role = excluded.role,
            imported_at = excluded.imported_at
        """,
        (
            f"candidate_fulltext_{identity_hash}",
            str(local_path),
            digest,
            utc_now(),
        ),
    )


def acquire(
    connection: sqlite3.Connection,
    output_dir: Path,
    report_path: Path,
    workers: int,
    timeout_seconds: int,
    maximum_bytes: int,
    retry_failed: bool,
) -> Dict[str, Any]:
    """Download eligible open PDFs concurrently and serialize DB writes."""
    require_complete(connection, ["literature_screened"])
    if not 1 <= workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = _source_rows(connection)
    jobs: List[Dict[str, Any]] = []
    counts = {
        "downloaded": 0,
        "resumed": 0,
        "failed": 0,
        "without_open_pdf": 0,
    }
    for source in sources:
        candidate_url, access_statement = _open_pdf_candidate(
            json.loads(str(source["raw_json"]))
        )
        job = {
            "record_key": source["record_key"],
            "candidate_url": candidate_url,
            "access_statement": (
                access_statement
                or (
                    "No OpenAlex location explicitly marked is_oa=true "
                    "with a PDF URL"
                )
            ),
        }
        if not candidate_url:
            _store_fulltext_acquisition(
                connection,
                _values(job, status="no_open_pdf"),
            )
            counts["without_open_pdf"] += 1
            continue
        existing_path = Path(str(source["local_path"] or ""))
        if (
            source["status"] == "downloaded"
            and source["candidate_url"] == candidate_url
            and existing_path.is_file()
            and sha256_file(existing_path) == source["sha256"]
        ):
            counts["resumed"] += 1
            continue
        if (
            source["status"] == "failed"
            and source["candidate_url"] == candidate_url
            and not retry_failed
        ):
            counts["failed"] += 1
            continue
        jobs.append(job)
    connection.commit()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _download_job,
                job,
                timeout_seconds,
                maximum_bytes,
            ): job
            for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                completed_job, result = future.result()
                _store_success(
                    connection,
                    output_dir,
                    completed_job,
                    result,
                    maximum_bytes,
                )
                counts["downloaded"] += 1
            except (
                KeyError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as error:
                _store_fulltext_acquisition(
                    connection,
                    _values(job, status="failed", error=str(error)),
                )
                counts["failed"] += 1
            connection.commit()

    final_counts = {
        str(row["status"]): int(row["n"])
        for row in connection.execute(
            """
            SELECT status, COUNT(*) AS n
            FROM fulltext_acquisitions
            GROUP BY status ORDER BY status
            """
        )
    }
    result = {
        "schema_version": "parallel_open_fulltext_acquisition_v3",
        "eligible_sources": len(sources),
        "download_jobs": len(jobs),
        "workers": workers,
        "timeout_seconds": timeout_seconds,
        "maximum_bytes": maximum_bytes,
        "retry_failed": retry_failed,
        "run_counts": counts,
        "final_status_counts": final_counts,
        "scheduler": (
            "bounded network thread pool; all SQLite and file-registration "
            "writes serialized on the main thread"
        ),
        "semantic_implementation": (
            "reuses indicators._open_pdf_candidate, _download_open_pdf, "
            "_safe_public_http_url, and _store_fulltext_acquisition"
        ),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "completed_at": utc_now(),
    }
    write_json(report_path, result)
    connection.execute(
        """
        INSERT INTO source_snapshots(
            source_id, path, sha256, role, imported_at
        ) VALUES (
            'parallel_fulltext_acquisition_v3', ?, ?,
            'fulltext_acquisition_audit', ?
        )
        ON CONFLICT(source_id) DO UPDATE SET
            path = excluded.path,
            sha256 = excluded.sha256,
            role = excluded.role,
            imported_at = excluded.imported_at
        """,
        (
            str(report_path.resolve()),
            sha256_file(report_path),
            utc_now(),
        ),
    )
    log_event(
        connection,
        "parallel_open_fulltext_acquisition",
        "collection",
        "included_english_sources",
        result,
    )
    connection.commit()
    return result


def main() -> None:
    """Run the resumable concurrent open-full-text acquisition helper."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=20)
    parser.add_argument("--maximum-bytes", type=int, default=100_000_000)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    connection = initialize(args.database.resolve())
    try:
        result = acquire(
            connection,
            args.output_dir.resolve(),
            args.report.resolve(),
            args.workers,
            args.timeout_seconds,
            args.maximum_bytes,
            args.retry_failed,
        )
    finally:
        connection.close()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Acquire lawful open PDFs for H2-included contextual source leads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import requests
from common import DATABASE_PATH, sha256_bytes, sha256_file, utc_now, write_json
from database import initialize
from indicators import (
    _download_open_pdf,
    _open_pdf_candidates,
    _store_fulltext_acquisition,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs" / "contextual_open_fulltexts_v4"
DEFAULT_REPORT = ROOT / "outputs" / "contextual_open_fulltext_acquisition_v4.json"
REQUEST_TIMEOUT_SECONDS = 12


def acquire(
    connection: Any, output: Path, report: Path, maximum: int | None, retry_failed: bool
) -> dict[str, Any]:
    """Download only public OpenAlex PDF locations for H2-included source leads."""
    if maximum is not None and maximum < 1:
        raise ValueError("maximum must be positive")
    output.mkdir(parents=True, exist_ok=True)
    counts = {
        "downloaded": 0,
        "resumed": 0,
        "failed": 0,
        "no_open_pdf": 0,
        "eligible_sources": 0,
    }
    attempts = 0
    for row in connection.execute("""
        SELECT r.record_key, r.raw_json, a.status, a.candidate_url,
               a.local_path, a.sha256
        FROM contextual_source_final f
        JOIN records r USING(record_key)
        LEFT JOIN fulltext_acquisitions a USING(record_key)
        WHERE f.final_decision = 'include_definition_or_review'
        ORDER BY r.record_key
        """):
        counts["eligible_sources"] += 1
        candidates = _open_pdf_candidates(json.loads(row["raw_json"]))
        if not candidates:
            counts["no_open_pdf"] += 1
            continue
        first_url = candidates[0][0]
        local_path = Path(str(row["local_path"] or ""))
        if (
            row["status"] == "downloaded"
            and local_path.is_file()
            and sha256_file(local_path) == row["sha256"]
        ):
            counts["resumed"] += 1
            continue
        if (
            row["status"] == "failed"
            and row["candidate_url"] == first_url
            and not retry_failed
        ):
            counts["failed"] += 1
            continue
        if maximum is not None and attempts >= maximum:
            break
        attempts += 1
        identity = sha256_bytes(str(row["record_key"]).encode("utf-8"))[:20]
        target = (output / f"{identity}.pdf").resolve()
        temporary = target.with_suffix(".pdf.part")
        errors: list[str] = []
        done = False
        for candidate_url, access_statement in candidates:
            try:
                payload = _download_open_pdf(
                    candidate_url, REQUEST_TIMEOUT_SECONDS, 100_000_000
                )
                body = bytes(payload["body"])
                if not body.lstrip().startswith(b"%PDF-"):
                    raise ValueError("Open location did not return a PDF")
                temporary.write_bytes(body)
                temporary.replace(target)
                digest = sha256_file(target)
                _store_fulltext_acquisition(
                    connection,
                    {
                        "record_key": row["record_key"],
                        "candidate_url": candidate_url,
                        "final_url": str(payload["final_url"]),
                        "local_path": str(target),
                        "sha256": digest,
                        "access_statement": access_statement,
                        "http_content_type": str(payload["content_type"]),
                        "byte_count": len(body),
                        "status": "downloaded",
                        "error": "",
                        "fetched_at": utc_now(),
                    },
                )
                connection.execute(
                    """
                    INSERT INTO source_snapshots(source_id, path, sha256, role, imported_at)
                    VALUES (?, ?, ?, 'contextual_candidate_open_fulltext', ?)
                    ON CONFLICT(source_id) DO UPDATE SET path=excluded.path, sha256=excluded.sha256,
                        role=excluded.role, imported_at=excluded.imported_at
                    """,
                    (f"contextual_fulltext_{identity}", str(target), digest, utc_now()),
                )
                counts["downloaded"] += 1
                done = True
                break
            except (
                KeyError,
                OSError,
                TypeError,
                ValueError,
                requests.RequestException,
            ) as error:
                if temporary.exists():
                    temporary.unlink()
                errors.append(f"{candidate_url}: {str(error)[:300]}")
        if not done:
            _store_fulltext_acquisition(
                connection,
                {
                    "record_key": row["record_key"],
                    "candidate_url": first_url,
                    "final_url": "",
                    "local_path": "",
                    "sha256": "",
                    "access_statement": candidates[0][1],
                    "http_content_type": "",
                    "byte_count": 0,
                    "status": "failed",
                    "error": " | ".join(errors)[:1000],
                    "fetched_at": utc_now(),
                },
            )
            counts["failed"] += 1
        connection.commit()
    result = {
        "schema_version": "contextual_open_fulltext_acquisition_v4",
        **counts,
        "new_attempts": attempts,
        "output_dir": str(output.resolve()),
        "output_dir_role": "lawful_open_pdf_only",
        "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
    }
    write_json(report, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--maximum", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    connection = initialize(args.database.resolve())
    try:
        print(
            json.dumps(
                acquire(
                    connection,
                    args.output_dir.resolve(),
                    args.report.resolve(),
                    args.maximum,
                    args.retry_failed,
                ),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        connection.close()


if __name__ == "__main__":
    main()

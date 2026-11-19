from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Mapping

import indicators
from common import (
    DATABASE_PATH,
    OUTPUT_DIR,
    sha256_bytes,
    sha256_file,
    utc_now,
    write_json,
)
from database import initialize, log_event


ROOT = Path(__file__).resolve().parent
DEFAULT_SPEC = ROOT / "targeted_formula_fulltext_sources_v3.json"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "open_fulltexts"
DEFAULT_REPORT = OUTPUT_DIR / "targeted_formula_fulltext_acquisition_v3.json"


def _read_spec(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "targeted_formula_fulltext_sources_v3":
        raise ValueError("Unexpected targeted-formula source schema")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Targeted-formula source list is empty")
    return payload


def _assert_included_source(
    connection: sqlite3.Connection,
    source: Mapping[str, Any],
) -> None:
    record_key = str(source["record_key"])
    row = connection.execute(
        """
        SELECT r.doi, r.title, s.final_language, s.final_decision
        FROM records r
        JOIN screening_final s USING(record_key)
        WHERE r.record_key = ?
        """,
        (record_key,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Targeted source is absent: {record_key}")
    if row["final_language"] != "en" or row["final_decision"] != "include":
        raise ValueError(f"Targeted source is not included English: {record_key}")
    if str(row["doi"]).casefold() != str(source["doi"]).casefold():
        raise ValueError(f"Targeted source DOI mismatch: {record_key}")


def _verify_pdf(path: Path, source: Mapping[str, Any]) -> None:
    digest = sha256_file(path)
    text = indicators._extract_fulltext_text(path, digest).casefold()
    missing = [
        token
        for token in source["expected_text"]
        if str(token).casefold() not in text
    ]
    if missing:
        raise ValueError(
            "Downloaded PDF failed identity verification: "
            + ", ".join(str(value) for value in missing)
        )


def _store_download(
    connection: sqlite3.Connection,
    source: Mapping[str, Any],
    output_dir: Path,
    timeout_seconds: int,
    maximum_bytes: int,
) -> Dict[str, Any]:
    record_key = str(source["record_key"])
    url = indicators._safe_public_http_url(str(source["url"]))
    identity = sha256_bytes(record_key.encode("utf-8"))[:20]
    path = (output_dir / f"{identity}.pdf").resolve()
    result = indicators._download_open_pdf(
        url,
        timeout_seconds,
        maximum_bytes,
    )
    body = bytes(result["body"])
    temporary = path.with_suffix(".part.pdf")
    temporary.write_bytes(body)
    try:
        _verify_pdf(temporary, source)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    digest = sha256_file(path)
    values = {
        "record_key": record_key,
        "candidate_url": url,
        "final_url": indicators._safe_public_http_url(
            str(result.get("final_url") or url)
        ),
        "local_path": str(path),
        "sha256": digest,
        "access_statement": str(source["access_statement"]),
        "http_content_type": str(result.get("content_type") or ""),
        "byte_count": len(body),
        "status": "downloaded",
        "error": "",
        "fetched_at": utc_now(),
    }
    indicators._store_fulltext_acquisition(connection, values)
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
            f"candidate_fulltext_{identity}",
            str(path),
            digest,
            utc_now(),
        ),
    )
    return {
        **values,
        "license": str(source["license"]),
        "completion_route": str(source["completion_route"]),
    }


def acquire(
    connection: sqlite3.Connection,
    spec_path: Path,
    output_dir: Path,
    report_path: Path,
    timeout_seconds: int,
    maximum_bytes: int,
) -> Dict[str, Any]:
    """Acquire and register frozen publisher PDFs for formula completion."""
    spec = _read_spec(spec_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for source in spec["sources"]:
        _assert_included_source(connection, source)
        results.append(
            _store_download(
                connection,
                source,
                output_dir,
                timeout_seconds,
                maximum_bytes,
            )
        )
    report = {
        "schema_version": "targeted_formula_fulltext_acquisition_v3",
        "spec_path": str(spec_path.resolve()),
        "spec_sha256": sha256_file(spec_path),
        "source_count": len(results),
        "sources": results,
        "round_13": False,
        "completed_at": utc_now(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json(report_path, report)
    connection.execute(
        """
        INSERT INTO source_snapshots(
            source_id, path, sha256, role, imported_at
        ) VALUES (
            'targeted_formula_fulltext_acquisition_v3', ?, ?,
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
        "targeted_formula_fulltexts_acquired",
        "collection",
        "terminal_formula_completion",
        report,
    )
    connection.commit()
    return report


def main() -> None:
    """Run the deterministic targeted full-text acquisition."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--maximum-bytes", type=int, default=100_000_000)
    args = parser.parse_args()
    connection = initialize(args.database.resolve())
    try:
        result = acquire(
            connection,
            args.spec.resolve(),
            args.output_dir.resolve(),
            args.report.resolve(),
            args.timeout_seconds,
            args.maximum_bytes,
        )
    finally:
        connection.close()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

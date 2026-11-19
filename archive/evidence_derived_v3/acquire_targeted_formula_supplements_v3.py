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
    normalize_text,
    sha256_bytes,
    sha256_file,
    utc_now,
    write_json,
)
from database import initialize, log_event


ROOT = Path(__file__).resolve().parent
DEFAULT_SPEC = ROOT / "targeted_formula_supplement_sources_v3.json"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "formula_supplement_fulltexts"
DEFAULT_REPORT = OUTPUT_DIR / "targeted_formula_supplement_acquisition_v3.json"


def _read_spec(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != (
        "targeted_formula_supplement_sources_v3"
    ):
        raise ValueError("Unexpected formula-supplement source schema")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Formula-supplement source list is empty")
    if payload.get("round_13") is not False:
        raise ValueError("Formula supplement must not start round 13")
    return payload


def _assert_existing_record_and_families(
    connection: sqlite3.Connection,
    source: Mapping[str, Any],
) -> None:
    record_key = str(source["record_key"])
    row = connection.execute(
        "SELECT doi, title, language FROM records WHERE record_key = ?",
        (record_key,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Formula supplement is absent from records: {record_key}")
    if str(row["doi"]).casefold() != str(source["doi"]).casefold():
        raise ValueError(f"Formula-supplement DOI mismatch: {record_key}")
    if str(row["language"] or "").casefold() != "en":
        raise ValueError(f"Formula supplement is not English: {record_key}")
    for feature_id in source["target_existing_feature_ids"]:
        feature = connection.execute(
            """
            SELECT feature_id FROM indicator_families
            WHERE feature_id = ?
            """,
            (str(feature_id),),
        ).fetchone()
        if feature is None:
            raise ValueError(
                f"Formula supplement targets a missing family: {feature_id}"
            )


def _verify_pdf(path: Path, source: Mapping[str, Any]) -> None:
    digest = sha256_file(path)
    text = normalize_text(
        indicators._extract_fulltext_text(path, digest)
    )
    missing = [
        str(token)
        for token in source["expected_text"]
        if normalize_text(str(token)) not in text
    ]
    if missing:
        raise ValueError(
            "Downloaded supplement failed identity verification: "
            + ", ".join(missing)
        )


def _download_one(
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
    connection.execute(
        """
        INSERT INTO source_snapshots(
            source_id, path, sha256, role, imported_at
        ) VALUES (?, ?, ?, 'targeted_formula_supplement_fulltext', ?)
        ON CONFLICT(source_id) DO UPDATE SET
            path = excluded.path,
            sha256 = excluded.sha256,
            role = excluded.role,
            imported_at = excluded.imported_at
        """,
        (
            f"targeted_formula_supplement_{identity}",
            str(path),
            digest,
            utc_now(),
        ),
    )
    return {
        "record_key": record_key,
        "doi": str(source["doi"]),
        "title": str(source["title"]),
        "fulltext_version_title": str(source["fulltext_version_title"]),
        "fulltext_version_doi": str(source["fulltext_version_doi"]),
        "version_relationship": str(source["version_relationship"]),
        "candidate_url": url,
        "final_url": indicators._safe_public_http_url(
            str(result.get("final_url") or url)
        ),
        "source_landing_page": str(source["source_landing_page"]),
        "local_path": str(path),
        "sha256": digest,
        "access_statement": str(source["access_statement"]),
        "license": str(source["license"]),
        "http_content_type": str(result.get("content_type") or ""),
        "byte_count": len(body),
        "completion_route": str(source["completion_route"]),
        "target_existing_feature_ids": list(
            source["target_existing_feature_ids"]
        ),
        "status": "downloaded",
    }


def acquire(
    connection: sqlite3.Connection,
    spec_path: Path,
    output_dir: Path,
    report_path: Path,
    timeout_seconds: int,
    maximum_bytes: int,
) -> Dict[str, Any]:
    """Acquire frozen English formula supplements without reopening discovery."""
    spec = _read_spec(spec_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = []
    for source in spec["sources"]:
        _assert_existing_record_and_families(connection, source)
        sources.append(
            _download_one(
                connection,
                source,
                output_dir,
                timeout_seconds,
                maximum_bytes,
            )
        )
    report = {
        "schema_version": "targeted_formula_supplement_acquisition_v3",
        "spec_path": str(spec_path.resolve()),
        "spec_sha256": sha256_file(spec_path),
        "source_count": len(sources),
        "sources": sources,
        "creates_new_search_records": False,
        "creates_new_term_families": False,
        "creates_new_indicator_families": False,
        "alters_k_q_p": False,
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
            'targeted_formula_supplement_acquisition_v3', ?, ?,
            'formula_supplement_acquisition_audit', ?
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
        "targeted_formula_supplements_acquired",
        "collection",
        "terminal_formula_completion",
        report,
    )
    connection.commit()
    return report


def main() -> None:
    """Run deterministic acquisition of cited formula supplements."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument("--maximum-bytes", type=int, default=100_000_000)
    args = parser.parse_args()
    connection = initialize(args.database.resolve())
    try:
        report = acquire(
            connection,
            args.spec.resolve(),
            args.output_dir.resolve(),
            args.report.resolve(),
            args.timeout_seconds,
            args.maximum_bytes,
        )
    finally:
        connection.close()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

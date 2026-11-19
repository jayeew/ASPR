"""Extract audit text from downloaded contextual English full texts."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from common import DATABASE_PATH, sha256_file, write_json

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs" / "contextual_fulltext_text_v4"
DEFAULT_REPORT = ROOT / "outputs" / "contextual_fulltext_text_extraction_v4.json"


def extract(
    connection: sqlite3.Connection, output: Path, report: Path
) -> dict[str, Any]:
    """Extract deterministic UTF-8 text only from acquired PDF files."""
    output.mkdir(parents=True, exist_ok=True)
    documents: list[dict[str, Any]] = []
    for row in connection.execute("""
        SELECT r.record_key, r.doi, r.title, a.local_path, a.sha256
        FROM contextual_source_final f
        JOIN records r USING(record_key)
        JOIN fulltext_acquisitions a USING(record_key)
        WHERE f.final_decision = 'include_definition_or_review'
          AND a.status = 'downloaded'
        ORDER BY r.record_key
        """):
        pdf = Path(str(row["local_path"])).resolve()
        if not pdf.is_file() or sha256_file(pdf) != row["sha256"]:
            raise ValueError(f"Missing/hash-mismatched acquired PDF: {pdf}")
        text_path = (output / f"{pdf.stem}.txt").resolve()
        reader = PdfReader(str(pdf))
        text = "\n\f\n".join(page.extract_text() or "" for page in reader.pages)
        text_path.write_text(text, encoding="utf-8")
        documents.append(
            {
                "record_key": row["record_key"],
                "doi": row["doi"],
                "title": row["title"],
                "pdf_path": str(pdf),
                "pdf_sha256": row["sha256"],
                "page_count": len(reader.pages),
                "text_path": str(text_path),
                "text_sha256": sha256_file(text_path),
                "text_characters": len(text),
            }
        )
    result = {
        "schema_version": "contextual_fulltext_text_extraction_v4",
        "document_count": len(documents),
        "documents": documents,
    }
    write_json(report, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    connection = sqlite3.connect(args.database.resolve())
    connection.row_factory = sqlite3.Row
    try:
        print(
            json.dumps(
                extract(connection, args.output_dir.resolve(), args.report.resolve()),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        connection.close()


if __name__ == "__main__":
    main()

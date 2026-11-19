"""Audit independent-team construct corroboration without changing K/Q/P.

The review is narrower than indicator extraction: it cannot create a family or
authorize a new formula.  It only records whether an already-frozen family has
an independently authored, English, peer-reviewed full text supporting the
candidate dimension's construct boundary.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from common import DATABASE_PATH, sha256_file, utc_now, write_csv
from database import initialize, snapshot_import_file


ROOT = Path(__file__).resolve().parent
PDF = ROOT / "outputs" / "formula_supplement_fulltexts_v4" / "60496bf8e0622de45fab.pdf"
TEXT = PDF.with_suffix(".txt")
FIELDS = (
    "support_id", "feature_id", "candidate_dimension_label", "reviewer_role",
    "source_doi", "source_title", "source_role", "source_research_group_id",
    "source_research_group", "fulltext_path", "fulltext_sha256", "evidence_span",
    "support_scope", "decision", "reason",
)
H2_FIELDS = (*FIELDS, "ai_payload_json", "h1_payload_json")
SOURCE = {
    "support_id": "CC001",
    "feature_id": "EF0007",
    "candidate_dimension_label": "Reference knowledge-base interdisciplinarity",
    "source_doi": "10.1007/s11192-023-04822-2",
    "source_title": "Quantifying and addressing uncertainty in the measurement of interdisciplinarity",
    "source_role": "independent_validation",
    "source_research_group_id": "university_otago_information_science",
    "source_research_group": "Department of Information Science, University of Otago",
    "fulltext_path": str(PDF.resolve()),
    "fulltext_sha256": "",
    "evidence_span": (
        "The Rao-Stirling diversity index (RS index), as applied to the disciplines "
        "of a paper’s cited references, is one of the most popular quantitative measures "
        "of IDR in the literature"
    ),
    "support_scope": "dimension_boundary_and_formula_application",
}


def _ensure(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS construct_corroboration_reviews (
          support_id TEXT NOT NULL, reviewer_role TEXT NOT NULL,
          feature_id TEXT NOT NULL, decision TEXT NOT NULL, reason TEXT NOT NULL,
          payload_json TEXT NOT NULL, reviewed_at TEXT NOT NULL,
          PRIMARY KEY (support_id, reviewer_role)
        )
        """
    )
    connection.commit()


def _source(role: str) -> Dict[str, str]:
    row = dict(SOURCE)
    row.update({field: "" for field in ("decision", "reason")})
    row["reviewer_role"] = role
    row["fulltext_sha256"] = sha256_file(PDF)
    return row


def export(connection: sqlite3.Connection, output: Path, role: str) -> None:
    _ensure(connection)
    if role not in {"AI", "H1", "H2"}:
        raise ValueError("role must be AI, H1, or H2")
    row = _source(role)
    fields = FIELDS
    if role == "H2":
        prior = {
            str(item["reviewer_role"]): item
            for item in connection.execute(
                "SELECT reviewer_role, payload_json FROM construct_corroboration_reviews WHERE support_id = 'CC001'"
            )
        }
        if not {"AI", "H1"}.issubset(prior):
            raise RuntimeError("H2 requires independent AI and H1 corroboration reviews")
        row["ai_payload_json"] = str(prior["AI"]["payload_json"])
        row["h1_payload_json"] = str(prior["H1"]["payload_json"])
        fields = H2_FIELDS
    write_csv(output, [row], fields)


def _protected(row: Mapping[str, str], role: str) -> None:
    expected = _source(role)
    for field in FIELDS[:13]:
        if str(row.get(field) or "") != str(expected.get(field) or ""):
            raise ValueError(f"Protected corroboration evidence changed: {field}")


def import_review(connection: sqlite3.Connection, input_path: Path) -> None:
    _ensure(connection)
    snapshot = snapshot_import_file(connection, input_path, "construct_corroboration_review")
    with snapshot.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1 or rows[0].get("support_id") != "CC001":
        raise ValueError("Corroboration review must contain CC001 exactly once")
    row = rows[0]
    role = str(row.get("reviewer_role") or "").upper()
    if role not in {"AI", "H1", "H2"}:
        raise ValueError("Invalid reviewer role")
    _protected(row, role)
    if str(row.get("decision") or "").casefold() not in {"approve", "exclude"}:
        raise ValueError("Decision must be approve or exclude")
    if not str(row.get("reason") or "").strip():
        raise ValueError("Reason is required")
    if role == "H2":
        prior = {str(x[0]) for x in connection.execute("SELECT reviewer_role FROM construct_corroboration_reviews WHERE support_id='CC001'")}
        if not {"AI", "H1"}.issubset(prior):
            raise RuntimeError("H2 needs AI and H1 first")
    connection.execute(
        """INSERT INTO construct_corroboration_reviews
        (support_id, reviewer_role, feature_id, decision, reason, payload_json, reviewed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(support_id, reviewer_role) DO UPDATE SET
        decision=excluded.decision, reason=excluded.reason, payload_json=excluded.payload_json, reviewed_at=excluded.reviewed_at""",
        ("CC001", role, "EF0007", str(row["decision"]).casefold(), str(row["reason"]), json.dumps(dict(row), ensure_ascii=False, sort_keys=True), utc_now()),
    )
    connection.commit()


def apply(connection: sqlite3.Connection, report: Path) -> Dict[str, Any]:
    _ensure(connection)
    h2 = connection.execute("SELECT * FROM construct_corroboration_reviews WHERE support_id='CC001' AND reviewer_role='H2'").fetchone()
    if h2 is None or h2["decision"] != "approve":
        raise RuntimeError("CC001 has no H2 approval")
    family = connection.execute("SELECT research_groups_json FROM indicator_families WHERE feature_id='EF0007'").fetchone()
    groups = sorted(set(json.loads(family["research_groups_json"])) | {SOURCE["source_research_group_id"]})
    connection.execute("UPDATE indicator_families SET research_groups_json=? WHERE feature_id='EF0007'", (json.dumps(groups),))
    payload = {
        "schema_version": "construct_corroboration_application_v4",
        "support_id": "CC001", "feature_id": "EF0007", "h2_decision": "approve",
        "added_independent_research_group": SOURCE["source_research_group_id"],
        "K_Q_P_changed": False, "new_indicator_families": False,
        "applied_at": utc_now(),
    }
    report.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    connection.commit()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("export", "import"):
        item = sub.add_parser(name); item.add_argument("--input" if name == "import" else "--output", type=Path, required=True)
        if name == "export": item.add_argument("--role", required=True, choices=("AI", "H1", "H2"))
    item = sub.add_parser("apply"); item.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(); connection = initialize(args.database.resolve())
    try:
        if args.command == "export": export(connection, args.output.resolve(), args.role)
        elif args.command == "import": import_review(connection, args.input.resolve())
        else: print(json.dumps(apply(connection, args.report.resolve()), ensure_ascii=False, indent=2))
    finally: connection.close()


if __name__ == "__main__": main()

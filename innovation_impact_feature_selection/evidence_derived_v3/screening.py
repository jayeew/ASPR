from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from common import (
    agreement_from_pair_counts,
    deterministic_ten_percent,
    iter_csv,
    utc_now,
    write_csv_iter,
)
from database import (
    assisted_review_file,
    assert_registered_review_attestation,
    invalidate_stages,
    log_event,
    require_complete,
    set_stage,
    snapshot_import_file,
)


EXCLUSION_REASONS = {
    "E_LANGUAGE_NON_ENGLISH",
    "E_NOT_PAPER_LEVEL_INNOVATION_OR_POTENTIAL_IMPACT",
    "E_FUTURE_OUTCOME_ONLY",
    "E_NOT_ARTICLE_LEVEL",
    "E_NOT_INDICATOR_PREDICTOR_VALIDATION",
    "E_DUPLICATE",
    "E_INSUFFICIENT_METADATA",
}
DECISIONS = {"include", "exclude", "uncertain"}
LANGUAGE_JUDGMENTS = {"en", "non_en", "uncertain"}
SCREENING_FIELDS = (
    "record_key",
    "doi",
    "title",
    "abstract",
    "openalex_language",
    "publication_year",
    "work_type",
    "reviewer_role",
    "language_judgment",
    "language_evidence",
    "decision",
    "exclusion_reason",
    "evidence_span",
    "notes",
)


def _formal_record_rows(
    connection: sqlite3.Connection,
) -> Iterable[sqlite3.Row]:
    return connection.execute(
        """
        SELECT r.*
        FROM records r
        JOIN formal_review_records s USING(record_key)
        ORDER BY r.record_key
        """
    )


def _formal_record_count(connection: sqlite3.Connection) -> int:
    return int(
        connection.execute(
            """
            SELECT COUNT(*) FROM formal_review_records
            """
        ).fetchone()[0]
    )


def _missing_primary_screening_count(
    connection: sqlite3.Connection,
) -> int:
    return int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM formal_review_records f
            WHERE NOT EXISTS (
                SELECT 1 FROM screening_decisions d
                WHERE d.record_key = f.record_key
                  AND d.reviewer_role = 'AI'
            )
               OR NOT EXISTS (
                SELECT 1 FROM screening_decisions d
                WHERE d.record_key = f.record_key
                  AND d.reviewer_role = 'H1'
            )
            """
        ).fetchone()[0]
    )


def export_screening(
    connection: sqlite3.Connection,
    output_path: Path,
    reviewer_role: str | None = None,
) -> int:
    """Export blind-compatible title/abstract screening worksheets."""
    require_complete(connection, ["formal_retrieval_complete"])
    if reviewer_role is None:
        raise ValueError(
            "Export one reviewer at a time to preserve independent screening"
        )
    roles = [reviewer_role]
    if any(role not in {"AI", "H1", "H2"} for role in roles):
        raise ValueError("reviewer_role must be AI, H1, or H2")
    if reviewer_role == "H2":
        missing_primary = _missing_primary_screening_count(connection)
        if missing_primary:
            raise RuntimeError(
                "H2 adjudication cannot be exported until AI and H1 have "
                f"screened all records; missing={missing_primary}"
            )
    fields = list(SCREENING_FIELDS)
    if reviewer_role == "H2":
        fields.extend(
            [
                "h2_review_reason",
                "ai_language_judgment",
                "ai_decision",
                "ai_exclusion_reason",
                "ai_evidence_span",
                "ai_notes",
                "h1_language_judgment",
                "h1_decision",
                "h1_exclusion_reason",
                "h1_evidence_span",
                "h1_notes",
            ]
        )

    def iter_rows() -> Iterable[Dict[str, Any]]:
        for record in _formal_record_rows(connection):
            for role in roles:
                existing = connection.execute(
                    """
                    SELECT 1 FROM screening_decisions
                    WHERE record_key = ? AND reviewer_role = ?
                    """,
                    (record["record_key"], role),
                ).fetchone()
                if existing is not None:
                    continue
                if role == "H2":
                    codes = {
                        str(row["reviewer_role"]): row
                        for row in connection.execute(
                            """
                            SELECT * FROM screening_decisions
                            WHERE record_key = ?
                              AND reviewer_role IN ('AI', 'H1')
                            """,
                            (record["record_key"],),
                        )
                    }
                    required, h2_reason = _h2_requirement(
                        record,
                        codes["AI"],
                        codes["H1"],
                    )
                    if not required:
                        continue
                output: Dict[str, Any] = {
                    "record_key": record["record_key"],
                    "doi": record["doi"],
                    "title": record["title"],
                    "abstract": record["abstract"],
                    "openalex_language": record["language"],
                    "publication_year": record["publication_year"],
                    "work_type": record["work_type"],
                    "reviewer_role": role,
                    "language_judgment": "",
                    "language_evidence": "",
                    "decision": "",
                    "exclusion_reason": "",
                    "evidence_span": "",
                    "notes": "",
                }
                if role == "H2":
                    ai = codes["AI"]
                    h1 = codes["H1"]
                    output.update(
                        {
                            "h2_review_reason": h2_reason,
                            "ai_language_judgment": ai[
                                "language_judgment"
                            ],
                            "ai_decision": ai["decision"],
                            "ai_exclusion_reason": ai[
                                "exclusion_reason"
                            ],
                            "ai_evidence_span": ai["evidence_span"],
                            "ai_notes": ai["notes"],
                            "h1_language_judgment": h1[
                                "language_judgment"
                            ],
                            "h1_decision": h1["decision"],
                            "h1_exclusion_reason": h1[
                                "exclusion_reason"
                            ],
                            "h1_evidence_span": h1["evidence_span"],
                            "h1_notes": h1["notes"],
                        }
                    )
                yield output

    return write_csv_iter(output_path, iter_rows(), fields)


def import_screening(
    connection: sqlite3.Connection,
    input_path: Path,
) -> int:
    """Import reviewer decisions with explicit English evidence."""
    if assisted_review_file(input_path):
        assisted_rows = list(iter_csv(input_path))
        assisted_roles = {
            str(row.get("reviewer_role") or "").strip().upper()
            for row in assisted_rows
            if str(row.get("reviewer_role") or "").strip()
        }
        if len(assisted_roles) != 1:
            raise ValueError(
                "One assisted screening import must declare one role"
            )
        assert_registered_review_attestation(
            connection,
            input_path,
            next(iter(assisted_roles)),
        )
    snapshot_path = snapshot_import_file(
        connection,
        input_path,
        "literature_screening",
    )
    rows = iter_csv(snapshot_path)
    imported = 0
    submission_role = ""
    for row in rows:
        record_key = str(row.get("record_key") or "").strip()
        decision = str(row.get("decision") or "").strip().casefold()
        if not record_key or not decision:
            continue
        role = str(row.get("reviewer_role") or "").strip().upper()
        language = str(
            row.get("language_judgment") or ""
        ).strip().casefold()
        reason = str(row.get("exclusion_reason") or "").strip()
        evidence = str(row.get("evidence_span") or "").strip()
        language_evidence = str(
            row.get("language_evidence") or ""
        ).strip()
        if role not in {"AI", "H1", "H2"}:
            raise ValueError(f"Invalid screening reviewer: {role}")
        if submission_role and role != submission_role:
            raise ValueError(
                "One screening import cannot mix AI, H1, and H2 roles"
            )
        submission_role = role
        if role == "H1" and any(
            str(field).casefold().startswith(("ai_", "h2_"))
            for field in row
        ):
            raise ValueError(
                "Blind H1 screening import refuses AI/H2 comparison columns"
            )
        if decision not in DECISIONS:
            raise ValueError(f"Invalid screening decision: {decision}")
        if language not in LANGUAGE_JUDGMENTS:
            raise ValueError(f"Invalid language judgment: {language}")
        record = connection.execute(
            """
            SELECT title, abstract FROM records
            WHERE record_key = ?
            """,
            (record_key,),
        ).fetchone()
        if record is None:
            raise ValueError(f"Unknown record: {record_key}")
        existing_roles = {
            str(value[0])
            for value in connection.execute(
                """
                SELECT reviewer_role FROM screening_decisions
                WHERE record_key = ?
                """,
                (record_key,),
            )
        }
        if role == "H2" and not {"AI", "H1"}.issubset(existing_roles):
            raise ValueError(
                "H2 screening requires earlier independent AI and H1 "
                f"decisions: {record_key}"
            )
        if role in {"AI", "H1"} and "H2" in existing_roles:
            raise ValueError(
                f"{role} screening is frozen after H2 adjudication: "
                f"{record_key}"
            )
        metadata_absent = not (
            str(record["title"]).strip()
            or str(record["abstract"]).strip()
        )
        documented_metadata_exclusion = (
            metadata_absent
            and decision == "exclude"
            and reason == "E_INSUFFICIENT_METADATA"
        )
        if not language_evidence and not documented_metadata_exclusion:
            raise ValueError(
                f"Language evidence is required: {record_key}/{role}"
            )
        if not evidence and not documented_metadata_exclusion:
            raise ValueError(
                f"Screening requires a title/abstract evidence span: "
                f"{record_key}"
            )
        source_text = "\n".join(
            (str(record["title"]), str(record["abstract"]))
        )
        if language_evidence and language_evidence not in source_text:
            raise ValueError(
                "Language evidence is not an exact title/abstract span: "
                f"{record_key}/{role}"
            )
        if evidence and evidence not in source_text:
            raise ValueError(
                f"Screening evidence is not an exact title/abstract span: "
                f"{record_key}/{role}"
            )
        if decision == "exclude":
            if reason not in EXCLUSION_REASONS:
                raise ValueError(
                    f"Invalid or missing exclusion reason: {reason!r}"
                )
        elif reason:
            raise ValueError(
                f"Only excluded decisions may have an exclusion reason: "
                f"{record_key}/{role}"
            )
        if language == "non_en" and (
            decision != "exclude"
            or reason != "E_LANGUAGE_NON_ENGLISH"
        ):
            raise ValueError(
                "A non-English judgment must be excluded with "
                "E_LANGUAGE_NON_ENGLISH"
            )
        connection.execute(
            """
            INSERT INTO screening_decisions(
                record_key, reviewer_role, language_judgment,
                language_evidence, decision, exclusion_reason,
                evidence_span, notes, decided_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(record_key, reviewer_role) DO UPDATE SET
                language_judgment = excluded.language_judgment,
                language_evidence = excluded.language_evidence,
                decision = excluded.decision,
                exclusion_reason = excluded.exclusion_reason,
                evidence_span = excluded.evidence_span,
                notes = excluded.notes,
                decided_at = excluded.decided_at
            """,
            (
                record_key,
                role,
                language,
                language_evidence,
                decision,
                reason,
                evidence,
                str(row.get("notes") or "").strip(),
                utc_now(),
            ),
        )
        imported += 1
    if imported:
        invalidate_stages(
            connection,
            (
                "literature_screened",
                "indicators_extracted",
                "dimensions_derived",
                "features_selected",
                "audit_complete",
            ),
            "literature screening decisions changed",
        )
    connection.commit()
    return imported


def _screening_signature(row: sqlite3.Row) -> str:
    return "|".join(
        (
            str(row["language_judgment"]),
            str(row["decision"]),
            str(row["exclusion_reason"]),
        )
    )


def _h2_requirement(
    record: sqlite3.Row,
    ai: sqlite3.Row,
    h1: sqlite3.Row,
) -> tuple[bool, str]:
    if _screening_signature(ai) != _screening_signature(h1):
        return True, "AI_H1_DISAGREEMENT"
    if ai["decision"] in {"include", "uncertain"}:
        return True, "ALL_INCLUDE_OR_UNCERTAIN"
    identity = str(record["doi"] or record["record_key"])
    if deterministic_ten_percent(identity):
        return True, "DETERMINISTIC_10_PERCENT_CONCORDANT_EXCLUSION"
    return False, "CONCORDANT_EXCLUSION_NOT_SAMPLED"


def finalize_screening(
    connection: sqlite3.Connection,
) -> Dict[str, Any]:
    """Adjudicate every record without promoting AI-only exclusions."""
    require_complete(connection, ["formal_retrieval_complete"])
    total_records = _formal_record_count(connection)
    if total_records == 0:
        raise RuntimeError("No formally retrieved records are available")
    missing_count = 0
    missing_sample: List[str] = []
    decision_pairs: Counter[Tuple[str, str]] = Counter()
    language_pairs: Counter[Tuple[str, str]] = Counter()
    counts = {
        "include": 0,
        "exclude": 0,
        "h2_required": 0,
        "language_exclusions": 0,
    }
    connection.execute("DELETE FROM screening_final")
    for record in _formal_record_rows(connection):
        codes = {
            str(row["reviewer_role"]): row
            for row in connection.execute(
                """
                SELECT * FROM screening_decisions
                WHERE record_key = ?
                """,
                (record["record_key"],),
            )
        }
        if "AI" not in codes or "H1" not in codes:
            missing_count += 1
            if len(missing_sample) < 100:
                missing_sample.append(f"{record['record_key']}:AI/H1")
            continue
        ai = codes["AI"]
        h1 = codes["H1"]
        decision_pairs[(str(ai["decision"]), str(h1["decision"]))] += 1
        language_pairs[
            (
                str(ai["language_judgment"]),
                str(h1["language_judgment"]),
            )
        ] += 1
        h2_required, adjudication_reason = _h2_requirement(record, ai, h1)
        final = ai
        if h2_required:
            counts["h2_required"] += 1
            h2 = codes.get("H2")
            if h2 is None:
                missing_count += 1
                if len(missing_sample) < 100:
                    missing_sample.append(f"{record['record_key']}:H2")
                continue
            if h2["decision"] == "uncertain":
                missing_count += 1
                if len(missing_sample) < 100:
                    missing_sample.append(
                        f"{record['record_key']}:H2_FINAL_UNCERTAIN"
                    )
                continue
            if h2["language_judgment"] == "uncertain":
                missing_count += 1
                if len(missing_sample) < 100:
                    missing_sample.append(
                        f"{record['record_key']}:H2_LANGUAGE_UNCERTAIN"
                    )
                continue
            final = h2
        final_language = str(final["language_judgment"])
        final_decision = str(final["decision"])
        exclusion_reason = str(final["exclusion_reason"])
        if final_language == "non_en":
            final_decision = "exclude"
            exclusion_reason = "E_LANGUAGE_NON_ENGLISH"
        if final_decision == "uncertain":
            missing_count += 1
            if len(missing_sample) < 100:
                missing_sample.append(
                    f"{record['record_key']}:NO_FINAL_DISPOSITION"
                )
            continue
        if final_decision == "exclude" and (
            exclusion_reason not in EXCLUSION_REASONS
        ):
            missing_count += 1
            if len(missing_sample) < 100:
                missing_sample.append(
                    f"{record['record_key']}:INVALID_FINAL_REASON"
                )
            continue
        counts[final_decision] += 1
        counts["language_exclusions"] += int(
            exclusion_reason == "E_LANGUAGE_NON_ENGLISH"
        )
        connection.execute(
            """
            INSERT INTO screening_final(
                record_key, final_language, final_decision,
                exclusion_reason, h2_required, h2_completed,
                adjudication_reason, finalized_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["record_key"],
                final_language,
                final_decision,
                exclusion_reason,
                int(h2_required),
                int(h2_required and "H2" in codes),
                adjudication_reason,
                utc_now(),
            ),
        )
    if missing_count:
        connection.rollback()
        set_stage(
            connection,
            "literature_screened",
            "blocked",
            {
                "missing_or_invalid": missing_sample,
                "count": missing_count,
            },
        )
        connection.commit()
        raise RuntimeError(
            "Literature screening is incomplete: "
            + ", ".join(missing_sample[:25])
        )
    decision_agreement = agreement_from_pair_counts(decision_pairs)
    language_agreement = agreement_from_pair_counts(language_pairs)
    details = {
        **counts,
        "total_records": total_records,
        "decision_agreement": decision_agreement,
        "language_agreement": language_agreement,
    }
    set_stage(connection, "literature_screened", "complete", details)
    log_event(
        connection,
        "screening_finalization",
        "collection",
        "formal_literature",
        details,
    )
    connection.commit()
    return details


def included_record_keys(connection: sqlite3.Connection) -> List[str]:
    """Return finally included record keys."""
    return [
        str(row[0])
        for row in connection.execute(
            """
            SELECT record_key FROM screening_final
            WHERE final_decision = 'include' AND final_language = 'en'
            ORDER BY record_key
            """
        )
    ]


def screening_exclusion_counts(
    connection: sqlite3.Connection,
) -> Dict[str, int]:
    """Return fixed-reason counts for PRISMA reporting."""
    return {
        str(row["exclusion_reason"]): int(row["n"])
        for row in connection.execute(
            """
            SELECT exclusion_reason, COUNT(*) AS n
            FROM screening_final
            WHERE final_decision = 'exclude'
            GROUP BY exclusion_reason
            ORDER BY exclusion_reason
            """
        )
    }

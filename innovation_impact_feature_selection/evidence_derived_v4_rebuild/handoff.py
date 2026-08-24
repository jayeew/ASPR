from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import coding
import indicators
import providers
import saturation
import screening
from common import OUTPUT_DIR, sha256_file, write_csv, write_json


HUMAN_TASK_DIR = OUTPUT_DIR / "human_tasks"
ACTION_FIELDS = (
    "action_id",
    "stage",
    "assignee",
    "status",
    "blocking",
    "items_total",
    "items_completed",
    "input_file",
    "worksheet_sha256",
    "import_command",
    "completion_rule",
    "notes",
)
HUMAN_INPUT_FIELDS = {
    "decision",
    "language_judgment",
    "resolution",
    "resolution_notes",
    "completion_status",
    "canonical_term",
    "source_disposition",
    "h2_approved",
    "audit_status",
    "concepts_complete",
    "dimension_label",
    "extraction_complete",
    "adjudication_notes",
    "disposition_notes",
    "verification_notes",
    "reason",
    "notes",
}


def _worksheet_has_human_input(path: Path) -> bool:
    """Conservatively detect unimported work in a generated CSV."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if "hidden_validation_seeds" in path.name and rows:
        return True
    for row in rows:
        for field in HUMAN_INPUT_FIELDS.intersection(row):
            if str(row.get(field) or "").strip():
                return True
    return False


def _write_once(
    path: Path,
    writer: Any,
    force: bool,
) -> int | None:
    """Create a blank human worksheet without overwriting unimported work."""
    if path.exists() and not force:
        return None
    if path.exists() and force and _worksheet_has_human_input(path):
        raise RuntimeError(
            "Refusing to overwrite a human worksheet containing unimported "
            f"work: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    return int(writer(path))


def _decision_count(
    connection: sqlite3.Connection,
    iteration: int,
    role: str,
) -> int:
    return int(
        connection.execute(
            """
            SELECT COUNT(DISTINCT d.record_key)
            FROM screening_decisions d
            WHERE d.reviewer_role = ?
              AND EXISTS (
                  SELECT 1 FROM discovery_hits h
                  WHERE h.record_key = d.record_key
                    AND h.review_round = ?
              )
            """,
            (role, iteration),
        ).fetchone()[0]
    )


def _h2_screening_required_count(
    connection: sqlite3.Connection,
    iteration: int,
) -> int | None:
    records = saturation._round_records(connection, iteration)
    required = 0
    for record in records:
        decisions = {
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
        if "AI" not in decisions or "H1" not in decisions:
            return None
        needed, _ = screening._h2_requirement(
            record,
            decisions["AI"],
            decisions["H1"],
        )
        required += int(needed)
    return required


def _add_action(
    actions: List[Dict[str, Any]],
    **values: Any,
) -> None:
    row = {field: "" for field in ACTION_FIELDS}
    row.update(values)
    actions.append(row)


def _round_actions(
    connection: sqlite3.Connection,
    force: bool,
    actions: List[Dict[str, Any]],
) -> None:
    invalid_ai_language_spans = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM screening_decisions d
            JOIN records r USING(record_key)
            WHERE d.reviewer_role = 'AI'
              AND (
                    d.language_evidence = ''
                    OR instr(
                        r.title || char(10) || r.abstract,
                        d.language_evidence
                    ) = 0
              )
            """
        ).fetchone()[0]
    )
    if invalid_ai_language_spans:
        _add_action(
            actions,
            action_id="NORMALIZE_AI_LANGUAGE_EVIDENCE",
            stage="discovery_screening_provenance",
            assignee="SYSTEM",
            status="ready",
            blocking="yes",
            items_total=invalid_ai_language_spans,
            items_completed=0,
            input_file="",
            import_command=(
                "python3 pipeline.py normalize-ai-language-evidence"
            ),
            completion_rule=(
                "Every AI language-evidence value is an exact title/abstract "
                "span; no screening decision changes."
            ),
            notes=(
                "The command snapshots legacy AI CSVs and records a "
                "deterministic correction hash before H2 sees the evidence."
            ),
        )
    for round_row in connection.execute(
        "SELECT * FROM discovery_review_rounds ORDER BY iteration"
    ):
        iteration = int(round_row["iteration"])
        assigned = int(round_row["assigned_records"])
        ai_done = _decision_count(connection, iteration, "AI")
        h1_done = _decision_count(connection, iteration, "H1")
        h1_path = (
            HUMAN_TASK_DIR
            / f"round_{iteration:02d}_screening_H1_BLIND.csv"
        )
        if h1_done < assigned:
            _write_once(
                h1_path,
                lambda path: saturation.export_discovery_screening(
                    connection,
                    iteration,
                    "H1",
                    path,
                ),
                force,
            )
            _add_action(
                actions,
                action_id=f"R{iteration:02d}_SCREEN_H1",
                stage="discovery_screening",
                assignee="H1",
                status="ready" if ai_done == assigned else "ready_parallel",
                blocking="yes",
                items_total=assigned,
                items_completed=h1_done,
                input_file=str(h1_path),
                import_command=(
                    "python3 pipeline.py import-screening --input "
                    f"{h1_path}"
                ),
                completion_rule=(
                    "H1 independently codes every row; the worksheet contains "
                    "no AI decision columns."
                ),
                notes=(
                    "Fill language_judgment, language_evidence, decision, "
                    "exclusion_reason when excluded, evidence_span, and notes."
                ),
            )
            continue
        h2_required = _h2_screening_required_count(connection, iteration)
        if h2_required is None:
            continue
        h2_done = _decision_count(connection, iteration, "H2")
        if h2_done < h2_required:
            h2_path = (
                HUMAN_TASK_DIR
                / f"round_{iteration:02d}_screening_H2_ADJUDICATE.csv"
            )
            _write_once(
                h2_path,
                lambda path: saturation.export_discovery_screening(
                    connection,
                    iteration,
                    "H2",
                    path,
                ),
                force,
            )
            _add_action(
                actions,
                action_id=f"R{iteration:02d}_SCREEN_H2",
                stage="discovery_screening_adjudication",
                assignee="H2",
                status="ready",
                blocking="yes",
                items_total=h2_required,
                items_completed=min(h2_done, h2_required),
                input_file=str(h2_path),
                import_command=(
                    "python3 pipeline.py import-screening --input "
                    f"{h2_path}"
                ),
                completion_rule=(
                    "H2 resolves every AI-H1 disagreement, every "
                    "include/uncertain record, and the frozen 10% audit sample."
                ),
                notes="AI and H1 decisions are shown only in the H2 file.",
            )
            continue
        if not bool(round_row["fully_reviewed"]):
            _add_action(
                actions,
                action_id=f"R{iteration:02d}_FINALIZE_SCREEN",
                stage="discovery_screening_finalize",
                assignee="SYSTEM",
                status="ready",
                blocking="yes",
                items_total=assigned,
                items_completed=assigned,
                input_file="",
                import_command=(
                    "python3 pipeline.py finalize-discovery-screening "
                    f"--iteration {iteration}"
                ),
                completion_rule=(
                    "All primary and required adjudication decisions resolve "
                    "to include or exclude."
                ),
                notes="This command is deterministic and makes no new judgment.",
            )
        elif str(round_row["decision"]) == "pending":
            included_records = int(
                connection.execute(
                    """
                    SELECT COUNT(DISTINCT record_key)
                    FROM discovery_hits
                    WHERE review_round = ?
                      AND review_status = 'include'
                    """,
                    (iteration,),
                ).fetchone()[0]
            )
            completed_extractions = int(
                connection.execute(
                    """
                    SELECT COUNT(DISTINCT record_key)
                    FROM discovery_extraction_reviews
                    WHERE review_round = ? AND reviewer_role = 'H1'
                      AND extraction_complete = 1
                    """,
                    (iteration,),
                ).fetchone()[0]
            )
            if completed_extractions >= included_records:
                pending_h2_indicators = int(
                    connection.execute(
                        """
                        SELECT COUNT(*)
                        FROM discovery_indicator_candidates
                        WHERE review_round = ?
                          AND status = 'candidate'
                          AND h1_decision = 'include'
                          AND h2_decision = 'pending'
                        """,
                        (iteration,),
                    ).fetchone()[0]
                )
                if pending_h2_indicators:
                    adjudication_path = (
                        HUMAN_TASK_DIR
                        / (
                            f"round_{iteration:02d}_indicator_"
                            "adjudication_H2.csv"
                        )
                    )
                    _write_once(
                        adjudication_path,
                        lambda path: (
                            saturation
                            .export_discovery_indicator_adjudication(
                                connection,
                                iteration,
                                path,
                            )
                        ),
                        force,
                    )
                    _add_action(
                        actions,
                        action_id=f"R{iteration:02d}_INDICATOR_H2",
                        stage="indicator_family_adjudication",
                        assignee="H2",
                        status="ready",
                        blocking="yes",
                        items_total=pending_h2_indicators,
                        items_completed=0,
                        input_file=str(adjudication_path),
                        import_command=(
                            "python3 pipeline.py "
                            "import-discovery-indicator-adjudication "
                            f"--input {adjudication_path}"
                        ),
                        completion_rule=(
                            "H2 includes/excludes every H1-retained "
                            "indicator name and assigns one canonical family "
                            "to every inclusion."
                        ),
                        notes=(
                            "Canonical-family labels, not raw mentions, are "
                            "used for the saturation endpoint."
                        ),
                    )
                else:
                    _add_action(
                        actions,
                        action_id=f"R{iteration:02d}_SATURATION_H2",
                        stage="dual_novelty_saturation",
                        assignee="H2",
                        status="ready_after_term_coding",
                        blocking="yes",
                        items_total=2,
                        items_completed=0,
                        input_file="",
                        import_command=(
                            "python3 pipeline.py discovery-novelty-status "
                            f"--iteration {iteration}; then use the computed "
                            "counts in record-discovery-saturation"
                        ),
                        completion_rule=(
                            "The database-computed term-family and "
                            "indicator-family novelty counts match H2's "
                            "source audit."
                        ),
                        notes=(
                            "Submitted counts are rejected if they differ "
                            "from adjudicated database evidence."
                        ),
                    )
                continue
            extraction_path = (
                HUMAN_TASK_DIR
                / f"round_{iteration:02d}_term_indicator_extraction_H1.csv"
            )
            _write_once(
                extraction_path,
                lambda path: saturation.export_discovery_extraction(
                    connection,
                    iteration,
                    path,
                    extractor_role="H1",
                ),
                force,
            )
            _add_action(
                actions,
                action_id=f"R{iteration:02d}_EXTRACT_H1",
                stage="term_indicator_extraction",
                assignee="H1",
                status="ready",
                blocking="yes",
                items_total=included_records,
                items_completed=completed_extractions,
                input_file=str(extraction_path),
                import_command=(
                    "python3 pipeline.py import-discovery-extraction --input "
                    f"{extraction_path}; then run prepare-human-tasks again"
                ),
                completion_rule=(
                    "All included records are checked for exact English terms "
                    "and indicator names, including an explicit completed "
                    "no-items disposition when appropriate."
                ),
                notes=(
                    "A freeze is rejected until three consecutive fully "
                    "reviewed rounds have zero new term families and zero new "
                    "indicator families."
                ),
            )


def _term_coding_actions(
    connection: sqlite3.Connection,
    force: bool,
    actions: List[Dict[str, Any]],
) -> None:
    active = int(
        connection.execute(
            "SELECT COUNT(*) FROM raw_terms WHERE status = 'active'"
        ).fetchone()[0]
    )
    if active == 0:
        return
    counts = {
        role: int(
            connection.execute(
                """
                SELECT COUNT(*) FROM term_coding
                WHERE coder_role = ?
                  AND term_id IN (
                      SELECT term_id FROM raw_terms WHERE status = 'active'
                  )
                """,
                (role,),
            ).fetchone()[0]
        )
        for role in ("AI", "H1", "H2")
    }
    if counts["H1"] < active:
        path = HUMAN_TASK_DIR / "term_coding_H1_BLIND.csv"
        _write_once(
            path,
            lambda output: coding.export_term_coding(
                connection,
                output,
                "H1",
            ),
            force,
        )
        _add_action(
            actions,
            action_id="TERM_CODE_H1",
            stage="term_standardization_and_domain_coding",
            assignee="H1",
            status="ready" if counts["AI"] == active else "ready_parallel",
            blocking="yes",
            items_total=active,
            items_completed=counts["H1"],
            input_file=str(path),
            import_command=(
                "python3 pipeline.py import-term-coding --input "
                f"{path}"
            ),
            completion_rule=(
                "H1 independently codes every active source-linked term "
                "without seeing the AI coding."
            ),
            notes=(
                "New terms from later saturation rounds trigger a new version "
                "and additional rows; no target K or Q is supplied. Use "
                "`python3 human_review_cli.py code-terms --input "
                f"{path}` for blind, resumable local coding."
            ),
        )
        return
    if counts["AI"] < active:
        _add_action(
            actions,
            action_id="TERM_CODE_AI",
            stage="term_standardization_and_domain_coding",
            assignee="SYSTEM_AI",
            status="ready",
            blocking="yes",
            items_total=active,
            items_completed=counts["AI"],
            input_file="",
            import_command="python3 pipeline.py ai-code-terms",
            completion_rule="Every active term receives an AI proposal.",
            notes="This does not replace either human role.",
        )
        return
    required_h2 = 0
    h2_complete = 0
    for term in connection.execute(
        "SELECT term_id FROM raw_terms WHERE status = 'active'"
    ):
        codes = {
            str(row["coder_role"]): row
            for row in connection.execute(
                "SELECT * FROM term_coding WHERE term_id = ?",
                (term["term_id"],),
            )
        }
        both_exclude = (
            "AI" in codes
            and "H1" in codes
            and coding._coding_signature(codes["AI"])
            == coding._coding_signature(codes["H1"])
            and codes["AI"]["decision"] == "exclude"
        )
        if not both_exclude:
            required_h2 += 1
            h2_complete += int("H2" in codes)
    if h2_complete < required_h2:
        path = HUMAN_TASK_DIR / "term_coding_H2_ADJUDICATE.csv"
        _write_once(
            path,
            lambda output: coding.export_term_coding(
                connection,
                output,
                "H2",
            ),
            force,
        )
        _add_action(
            actions,
            action_id="TERM_CODE_H2",
            stage="term_standardization_and_domain_coding",
            assignee="H2",
            status="ready",
            blocking="yes",
            items_total=required_h2,
            items_completed=h2_complete,
            input_file=str(path),
            import_command=(
                "python3 pipeline.py import-term-coding --input "
                f"{path}"
            ),
            completion_rule=(
                "H2 approves every retained domain assignment and resolves "
                "all non-exclusion disagreements."
            ),
            notes=(
                "Only after this step may derive-search-frame produce K/Q/P."
            ),
        )


def _validation_actions(
    connection: sqlite3.Connection,
    force: bool,
    actions: List[Dict[str, Any]],
) -> None:
    search_log_status = coding.hidden_seed_search_log_status(connection)
    if search_log_status["missing_routes"]:
        log_path = (
            HUMAN_TASK_DIR
            / "hidden_validation_seed_search_log_H2.csv"
        )
        _write_once(
            log_path,
            coding.export_hidden_seed_search_log_template,
            force,
        )
        _add_action(
            actions,
            action_id="HIDDEN_SEED_SEARCH_LOG_H2",
            stage="external_recall_validation_provenance",
            assignee="H2",
            status="ready",
            blocking="yes",
            items_total=len(search_log_status["required_routes"]),
            items_completed=len(search_log_status["complete_routes"]),
            input_file=str(log_path),
            import_command=(
                "python3 pipeline.py import-hidden-seed-search-log "
                f"--input {log_path}"
            ),
            completion_rule=(
                "H2 documents a completed independent review search plus "
                "completed backward and forward citation tracking, including "
                "exact queries/seeds, flow counts, and every eligible DOI "
                "found on each route."
            ),
            notes=(
                "Missing route classes: "
                + ", ".join(search_log_status["missing_routes"])
                + ". Add rows rather than targeting a seed quota. The "
                "distinct DOI count in eligible_seed_dois must equal "
                "eligible_seed_count."
            ),
        )
    hidden = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM evidence_seeds
            WHERE seed_role = 'validation'
              AND hidden_during_development = 1
              AND supplied_by = 'H2'
              AND eligibility_status = 'eligible'
              AND language = 'en'
            """
        ).fetchone()[0]
    )
    if hidden == 0:
        path = HUMAN_TASK_DIR / "hidden_validation_seeds_H2.csv"
        _write_once(
            path,
            lambda output: (
                coding.export_seed_template(output) or 0
            ),
            force,
        )
        _add_action(
            actions,
            action_id="HIDDEN_SEEDS_H2",
            stage="external_recall_validation",
            assignee="H2",
            status="ready",
            blocking="yes",
            items_total="not preset",
            items_completed=0,
            input_file=str(path),
            import_command=(
                "python3 pipeline.py import-seeds --input "
                f"{path}"
            ),
            completion_rule=(
                "H2 independently supplies eligible English validation "
                "papers that were hidden during initial term generation; "
                "their DOI set exactly matches the completed search logs."
            ),
            notes=(
                "The seed count is not prescribed, but at least one eligible "
                "hidden seed is required. Import every eligible DOI listed "
                "across the three H2 search-log routes."
            ),
        )
    active_queries = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM logical_queries
            WHERE status = 'active' AND logical_query_id LIKE 'L%'
            """
        ).fetchone()[0]
    )
    press_complete = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM logical_queries
            WHERE status = 'active' AND logical_query_id LIKE 'L%'
              AND press_status = 'pass'
            """
        ).fetchone()[0]
    )
    if active_queries and press_complete < active_queries:
        path = HUMAN_TASK_DIR / "PRESS_H2.csv"
        _write_once(
            path,
            lambda output: coding.export_press(connection, output),
            force,
        )
        _add_action(
            actions,
            action_id="PRESS_H2",
            stage="search_frame_validation",
            assignee="H2",
            status="ready",
            blocking="yes",
            items_total=active_queries,
            items_completed=press_complete,
            input_file=str(path),
            import_command=(
                "python3 pipeline.py import-press --input "
                f"{path}"
            ),
            completion_rule=(
                "Every non-redundant logical query passes PRESS with no "
                "unresolved concept, Boolean, spelling, phrase, or limit issue."
            ),
            notes="PRESS is performed only after evidence-derived K/Q exist.",
        )


def _downstream_actions(
    connection: sqlite3.Connection,
    force: bool,
    actions: List[Dict[str, Any]],
) -> None:
    """Create the post-search-frame full-text, dimension, and audit handoff."""
    literature_complete = connection.execute(
        """
        SELECT status FROM stage_status
        WHERE stage = 'literature_screened'
        """
    ).fetchone()[0] == "complete"
    if not literature_complete:
        return
    included = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM screening_final
            WHERE final_decision = 'include' AND final_language = 'en'
            """
        ).fetchone()[0]
    )
    acquisition_coverage = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM screening_final s
            JOIN fulltext_acquisitions a USING(record_key)
            WHERE s.final_decision = 'include' AND s.final_language = 'en'
            """
        ).fetchone()[0]
    )
    if acquisition_coverage < included:
        _add_action(
            actions,
            action_id="OPEN_FULLTEXT_ACQUISITION_SYSTEM",
            stage="open_fulltext_acquisition",
            assignee="SYSTEM",
            status="ready",
            blocking="yes",
            items_total=included,
            items_completed=acquisition_coverage,
            input_file="",
            import_command="python3 pipeline.py acquire-open-fulltexts",
            completion_rule=(
                "Every included English source has a downloaded, failed, or "
                "no-open-PDF acquisition disposition."
            ),
            notes=(
                "Only OpenAlex locations explicitly marked is_oa=true are "
                "downloaded; language and formula decisions remain human."
            ),
        )
        return
    h1_sources = int(
        connection.execute(
            """
            SELECT COUNT(DISTINCT record_key)
            FROM indicator_source_reviews
            WHERE reviewer_role = 'H1'
            """
        ).fetchone()[0]
    )
    h2_sources = int(
        connection.execute(
            """
            SELECT COUNT(DISTINCT record_key)
            FROM indicator_source_reviews
            WHERE reviewer_role = 'H2'
            """
        ).fetchone()[0]
    )
    if h1_sources < included:
        path = HUMAN_TASK_DIR / "indicator_extraction_H1.csv"
        _write_once(
            path,
            lambda output: indicators.export_indicator_extraction(
                connection,
                output,
            ),
            force,
        )
        _add_action(
            actions,
            action_id="INDICATOR_EXTRACTION_H1",
            stage="fulltext_indicator_census",
            assignee="H1",
            status="ready",
            blocking="yes",
            items_total=included,
            items_completed=h1_sources,
            input_file=str(path),
            import_command=(
                "python3 pipeline.py import-indicators --input "
                f"{path}; then run prepare-human-tasks again"
            ),
            completion_rule=(
                "H1 dispositions every included source and extracts all "
                "indicator mentions with exact English full-text evidence."
            ),
            notes=(
                "Duplicate a source row for multiple indicators; reviews may "
                "discover names but cannot authorize formulas."
            ),
        )
        return
    unreviewed_mentions = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM indicator_mention_reviews h1
            LEFT JOIN indicator_mention_reviews h2
              ON h2.mention_id = h1.mention_id
             AND h2.reviewer_role = 'H2'
            WHERE h1.reviewer_role = 'H1'
              AND h1.decision != 'excluded'
              AND h2.mention_id IS NULL
            """
        ).fetchone()[0]
    )
    if h2_sources < included or unreviewed_mentions:
        path = HUMAN_TASK_DIR / "indicator_adjudication_H2.csv"
        _write_once(
            path,
            lambda output: indicators.export_indicator_adjudication(
                connection,
                output,
            ),
            force,
        )
        _add_action(
            actions,
            action_id="INDICATOR_ADJUDICATION_H2",
            stage="fulltext_indicator_adjudication",
            assignee="H2",
            status="ready",
            blocking="yes",
            items_total=included + unreviewed_mentions,
            items_completed=h2_sources,
            input_file=str(path),
            import_command=(
                "python3 pipeline.py import-indicators --input "
                f"{path}; then run prepare-human-tasks again"
            ),
            completion_rule=(
                "H2 reviews every source disposition and approves or excludes "
                "every retained indicator mention with adjudication notes."
            ),
            notes=(
                "The worksheet contains H1 evidence; blank h2_approved and "
                "adjudication_notes are mandatory H2 decisions."
            ),
        )
        return
    indicator_stage = connection.execute(
        """
        SELECT status FROM stage_status
        WHERE stage = 'indicators_extracted'
        """
    ).fetchone()[0]
    if indicator_stage != "complete":
        return
    _dimension_and_data_actions(connection, force, actions)


def _crossref_actions(
    connection: sqlite3.Connection,
    force: bool,
    actions: List[Dict[str, Any]],
) -> None:
    """Expose resumable machine validation and only in-scope H2 conflicts."""
    keys = {
        str(row["record_key"])
        for row in screening._formal_record_rows(connection)
        if str(row["doi"] or "").strip()
    }
    if not keys:
        return
    processed = {
        str(row["record_key"])
        for row in connection.execute(
            "SELECT record_key FROM crossref_validation"
        )
        if str(row["record_key"]) in keys
    }
    if len(processed) < len(keys):
        _add_action(
            actions,
            action_id="CROSSREF_VALIDATION_SYSTEM",
            stage="bibliographic_validation",
            assignee="SYSTEM",
            status="running_or_resumable",
            blocking="yes",
            items_total=len(keys),
            items_completed=len(processed),
            input_file="",
            import_command=(
                "python3 pipeline.py crossref-validate --scope all; then run "
                "prepare-human-tasks again"
            ),
            completion_rule=(
                "Every in-scope DOI has a committed Crossref validation, "
                "conflict, or provider-error disposition."
            ),
            notes="Already processed records are never fetched again.",
        )
        return
    unresolved = [
        str(row["record_key"])
        for row in connection.execute(
            """
            SELECT record_key FROM crossref_validation
            WHERE status IN ('conflict', 'error')
            ORDER BY record_key
            """
        )
        if str(row["record_key"]) in keys
    ]
    if not unresolved:
        return
    path = HUMAN_TASK_DIR / "crossref_conflicts_H2.csv"
    _write_once(
        path,
        lambda output: providers.export_crossref_conflicts(
            connection,
            output,
            record_keys=keys,
        ),
        force,
    )
    _add_action(
        actions,
        action_id="CROSSREF_CONFLICTS_H2",
        stage="bibliographic_conflict_resolution",
        assignee="H2",
        status="ready",
        blocking="yes",
        items_total=len(unresolved),
        items_completed=0,
        input_file=str(path),
        import_command=(
            "python3 pipeline.py import-crossref-resolutions --input "
            f"{path}; then run prepare-human-tasks again"
        ),
        completion_rule=(
            "H2 resolves every in-scope DOI/title/type/provider conflict "
            "with a bibliographic note."
        ),
        notes=(
            "Conflicts from unassigned citation-pool records are excluded "
            "from this queue and from completion gates."
        ),
    )


def _dimension_and_data_actions(
    connection: sqlite3.Connection,
    force: bool,
    actions: List[Dict[str, Any]],
) -> None:
    families = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM indicator_families
            WHERE status = 'candidate'
            """
        ).fetchone()[0]
    )
    if not families:
        return
    counts = {
        role: int(
            connection.execute(
                """
                SELECT COUNT(*) FROM dimension_coding
                WHERE coder_role = ?
                """,
                (role,),
            ).fetchone()[0]
        )
        for role in ("AI", "H1", "H2")
    }
    for role in ("AI", "H1"):
        if counts[role] >= families:
            continue
        if role == "AI":
            _add_action(
                actions,
                action_id="DIMENSION_CODING_AI",
                stage="candidate_dimension_coding",
                assignee="AI",
                status="ready_parallel",
                blocking="yes",
                items_total=families,
                items_completed=counts[role],
                input_file="",
                import_command=(
                    "python3 pipeline.py ai-code-dimensions; then run "
                    "prepare-human-tasks again"
                ),
                completion_rule=(
                    "The declared local AI independently codes every "
                    "canonical indicator family without a target count."
                ),
                notes=(
                    "Model digest, prompt hash, input hash, failures, and "
                    "completed rows are recorded in SQLite."
                ),
            )
            continue
        path = HUMAN_TASK_DIR / f"dimension_coding_{role}_BLIND.csv"
        _write_once(
            path,
            lambda output, actor=role: indicators.export_dimension_coding(
                connection,
                output,
                actor,
            ),
            force,
        )
        _add_action(
            actions,
            action_id=f"DIMENSION_CODING_{role}",
            stage="candidate_dimension_coding",
            assignee=role,
            status="ready_parallel",
            blocking="yes",
            items_total=families,
            items_completed=counts[role],
            input_file=str(path),
            import_command=(
                "python3 pipeline.py import-dimension-coding --input "
                f"{path}; then run prepare-human-tasks again"
            ),
            completion_rule=(
                f"{role} independently codes every canonical indicator "
                "family without a target dimension count."
            ),
            notes=(
                "Formula, T0, source-role, team, and mention evidence are "
                "provided; no other coder's decision is shown."
            ),
        )
    if counts["AI"] >= families and counts["H1"] >= families:
        if counts["H2"] < families:
            path = HUMAN_TASK_DIR / "dimension_adjudication_H2.csv"
            _write_once(
                path,
                lambda output: indicators.export_dimension_coding(
                    connection,
                    output,
                    "H2",
                ),
                force,
            )
            _add_action(
                actions,
                action_id="DIMENSION_ADJUDICATION_H2",
                stage="candidate_dimension_adjudication",
                assignee="H2",
                status="ready",
                blocking="yes",
                items_total=families,
                items_completed=counts["H2"],
                input_file=str(path),
                import_command=(
                    "python3 pipeline.py import-dimension-coding --input "
                    f"{path}; then run prepare-human-tasks again"
                ),
                completion_rule=(
                    "H2 adjudicates every mapping and all dimension "
                    "merge/split or multi-label decisions."
                ),
                notes="AI and H1 codes are exposed only in this H2 file.",
            )
        elif connection.execute(
            """
            SELECT status FROM stage_status
            WHERE stage = 'dimensions_derived'
            """
        ).fetchone()[0] != "complete":
            _add_action(
                actions,
                action_id="DERIVE_DIMENSIONS_SYSTEM",
                stage="derive_candidate_dimensions",
                assignee="SYSTEM",
                status="ready",
                blocking="yes",
                items_total=families,
                items_completed=families,
                input_file="",
                import_command=(
                    "python3 pipeline.py derive-dimensions; then run "
                    "prepare-human-tasks again"
                ),
                completion_rule=(
                    "M is deterministically derived from H2-adjudicated "
                    "indicator-to-construct mappings."
                ),
                notes="No dimension quota or legacy D01-D12 labels are used.",
            )
    audited = int(
        connection.execute(
            "SELECT COUNT(*) FROM feature_data_audit"
        ).fetchone()[0]
    )
    if audited < families:
        path = HUMAN_TASK_DIR / "feature_data_audit_H1.csv"
        _write_once(
            path,
            lambda output: indicators.export_feature_data_audit(
                connection,
                output,
            ),
            force,
        )
        _add_action(
            actions,
            action_id="FEATURE_DATA_AUDIT_H1",
            stage="feature_data_quality",
            assignee="H1",
            status="ready_parallel",
            blocking="yes",
            items_total=families,
            items_completed=audited,
            input_file=str(path),
            import_command=(
                "python3 pipeline.py import-data-audit --input "
                f"{path}; then run prepare-human-tasks again"
            ),
            completion_rule=(
                "Every family has reproducible row/valid/unique/missingness "
                "checks plus frozen derivation and input artifacts."
            ),
            notes=(
                "Unavailable features must fail explicitly; passing claims "
                "without actual hashed artifacts are rejected."
            ),
        )
    dimensions_complete = connection.execute(
        """
        SELECT status FROM stage_status
        WHERE stage = 'dimensions_derived'
        """
    ).fetchone()[0] == "complete"
    features_complete = connection.execute(
        """
        SELECT status FROM stage_status
        WHERE stage = 'features_selected'
        """
    ).fetchone()[0] == "complete"
    if dimensions_complete and audited >= families and not features_complete:
        _add_action(
            actions,
            action_id="SELECT_FINAL_INDICATORS_SYSTEM",
            stage="fixed_gate_selection",
            assignee="SYSTEM",
            status="ready",
            blocking="yes",
            items_total=families,
            items_completed=families,
            input_file="",
            import_command=(
                "python3 pipeline.py select-indicators; "
                "python3 pipeline.py audit"
            ),
            completion_rule=(
                "Frozen hard gates and deterministic redundancy rules produce "
                "D/F and separate opportunity/control/sensitivity sets."
            ),
            notes="Model performance is never consulted for admission.",
        )


def _write_handoff_markdown(
    actions: Iterable[Mapping[str, Any]],
    path: Path,
) -> None:
    rows = list(actions)
    lines = [
        "# Evidence-derived v3 human handoff",
        "",
        "These tasks are the irreducible human gates in the frozen protocol. "
        "AI outputs are proposals. Direct human entry is preferred; an "
        "automated draft is admissible only after a human reviews and adopts "
        "the exact hashed file and the project owner attests that review.",
        "",
        "本文件是人工接力清单。H1 可直接人工编码；若使用自动初稿，必须"
        "逐项人工复核并由项目负责人确认该文件哈希。H2 在 H1 完成后"
        "仲裁。审计中必须如实保留“人工复核的自动初稿”来源，不能冒充"
        "最初即为纯手工盲编。",
        "",
        "## Current blocking actions",
        "",
    ]
    if not rows:
        lines.append("- None.")
    for row in rows:
        lines.extend(
            [
                f"### {row['action_id']} — {row['assignee']}",
                "",
                f"- Status: `{row['status']}`",
                f"- Progress: {row['items_completed']}/{row['items_total']}",
                f"- Worksheet: `{row['input_file']}`",
                f"- Worksheet SHA-256: `{row['worksheet_sha256']}`",
                f"- Completion rule: {row['completion_rule']}",
                f"- Import/next command: `{row['import_command']}`",
                f"- Notes: {row['notes']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Blinding and provenance rules",
            "",
            "- H1 screening and term-coding files contain source evidence but "
            "no AI decisions.",
            "- H2 files are created only after AI and H1 decisions exist and "
            "show both inputs for adjudication.",
            "- Do not enter a human role for an AI-generated decision.",
            "- Preserve exact English evidence spans and import completed CSVs "
            "through the CLI so snapshots and hashes enter SQLite.",
            "",
            "## Fast blind screening / 快速盲筛",
            "",
            "H1 可使用纯本地键盘工具逐条审核。该工具只读取指定的 H1 "
            "盲表，若发现 AI/H2 对照列会拒绝打开，并在每次决定后原子"
            "保存：",
            "",
            "```bash",
            "python3 human_review_cli.py screen --input "
            "outputs/human_tasks/round_01_screening_H1_BLIND.csv",
            "python3 human_review_cli.py status --input "
            "outputs/human_tasks/round_01_screening_H1_BLIND.csv",
            "python3 human_review_cli.py validate --input "
            "outputs/human_tasks/round_01_screening_H1_BLIND.csv",
            "python3 human_review_cli.py code-terms --input "
            "outputs/human_tasks/term_coding_H1_BLIND.csv",
            "python3 human_review_cli.py term-status --input "
            "outputs/human_tasks/term_coding_H1_BLIND.csv",
            "python3 human_review_cli.py validate-terms --input "
            "outputs/human_tasks/term_coding_H1_BLIND.csv",
            "```",
            "",
            "## Screening value guide / 文献筛选取值",
            "",
            "- `language_judgment`: `en`, `non_en`, or `uncertain`.",
            "- `decision`: `include`, `exclude`, or `uncertain`.",
            "- `language_evidence` and `evidence_span`: copy an exact English "
            "span from the title or abstract; do not paraphrase.",
            "- An `exclude` row must use exactly one reason: "
            "`E_LANGUAGE_NON_ENGLISH`, "
            "`E_NOT_PAPER_LEVEL_INNOVATION_OR_POTENTIAL_IMPACT`, "
            "`E_FUTURE_OUTCOME_ONLY`, `E_NOT_ARTICLE_LEVEL`, "
            "`E_NOT_INDICATOR_PREDICTOR_VALIDATION`, `E_DUPLICATE`, or "
            "`E_INSUFFICIENT_METADATA`.",
            "- A `non_en` judgment must pair with `exclude` and "
            "`E_LANGUAGE_NON_ENGLISH`; included/uncertain rows leave "
            "`exclusion_reason` blank.",
            "",
            "## Term coding value guide / 术语编码取值",
            "",
            "- `decision` is `include` or `exclude`; every row requires a "
            "source-based `reason`.",
            "- Included rows must fill `canonical_term`, "
            "`term_family_label`, `search_domain_label`, "
            "`search_domain_definition`, and `query_family_label`.",
            "- `term_relation` is one of: `canonical`, `synonym`, "
            "`abbreviation`, `full_form`, `historical_name`, "
            "`morphological_variant`, or `parameter_variant`.",
            "- `cross_domain` is `true` only when the same evidenced term "
            "genuinely belongs to more than one construct domain; otherwise "
            "use `false`.",
            "- Do not aim for a preferred number of domains or queries. "
            "Use identical labels only when the construct and explanatory "
            "role are genuinely the same.",
            "- `pilot_v2_*` and `development_seed_hint` rows may suggest "
            "synonyms but cannot authorize a domain/query family. At least "
            "one directly verified English title/abstract term is required.",
            "",
            "## Hidden validation seeds / 隐藏验证种子",
            "",
            "- H2 records and adopts the exact review-search expression and "
            "database, then backward and forward citation-tracing sources.",
            "- Every log row records ISO date/time, retrieved, screened, and "
            "eligible counts, plus notes explaining completion.",
            "- List every eligible DOI in `eligible_seed_dois`; its distinct "
            "count must equal `eligible_seed_count`, and the three-route DOI "
            "union must exactly match the eligible English seed file.",
            "- The seed count is not targeted. Include every eligible English "
            "paper found by the documented routes and preserve its DOI, "
            "citation, year, and eligibility decision.",
            "",
            "## Crossref conflict guide / 书目冲突",
            "",
            "- `resolution` is one of `accept_openalex`, "
            "`accept_crossref`, `manual_bibliographic_resolution`, or "
            "`exclude_mapping_error`.",
            "- Every resolution requires `reviewer_role=H2` and a concrete "
            "`resolution_notes` citation or publisher-record explanation.",
            "- Pure online-first/issue-year variants are already classified "
            "automatically and do not appear in this queue.",
            "",
            "## Full-text indicator guide / 全文指标普查",
            "",
            "- H1 gives every included source one disposition: `extracted`, "
            "`no_indicator`, `candidate_fulltext_missing`, or "
            "`excluded_after_fulltext`; duplicate a row for multiple "
            "indicator mentions.",
            "- Verified formula evidence requires a frozen lawful file, an "
            "English evidence span found in that file, formula location, "
            "formula, units, parameters, direction, missing rule, required "
            "data, and `maximum_information_time=T0` when T0-computable.",
            "- `research_group_evidence` must be an author/affiliation span "
            "found in the same frozen English full text. Only H2-approved "
            "full-text-verified teams count toward dimension retention.",
            "- H2 fills every blank `h2_approved`, reviews each source "
            "disposition, and supplies `adjudication_notes`; exclusions use "
            "`status=excluded`.",
            "- H2 rows are accepted only after the matching H1 source and "
            "mention identities exist. Both reviewer records are retained "
            "separately and H1 becomes immutable after adjudication.",
            "- `selection_priority` must match the frozen source-role map in "
            "`screening_rules_v3.json`; `evidence_strength` uses its closed "
            "vocabulary. Give every 0–1 `stability_score` an auditable "
            "`stability_basis`; use zero when no quantitative evidence "
            "exists.",
            "",
            "## Dimension and data guide / 维度与数据",
            "",
            "- H1 maps every canonical indicator family independently; use "
            "only evidence-derived labels and never reuse D01-D12.",
            "- `construct_role` is `substantive_innovation`, `t0_potential`, "
            "`opportunity`, `context_control`, or `sensitivity`.",
            "- H2 reviews the AI/H1 comparison columns and adjudicates all "
            "merge, split, multi-label, and exclusion decisions.",
            "- Passing data audits require actual derivation and input files; "
            "the importer recomputes hashes and count/missingness checks.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def prepare_human_tasks(
    connection: sqlite3.Connection,
    force: bool = False,
) -> Dict[str, Any]:
    """Create only the human worksheets that are currently actionable."""
    HUMAN_TASK_DIR.mkdir(parents=True, exist_ok=True)
    actions: List[Dict[str, Any]] = []
    _round_actions(connection, force, actions)
    _term_coding_actions(connection, force, actions)
    _validation_actions(connection, force, actions)
    _crossref_actions(connection, force, actions)
    _downstream_actions(connection, force, actions)
    for action in actions:
        worksheet = Path(str(action.get("input_file") or ""))
        if str(action.get("input_file") or "") and worksheet.is_file():
            action["worksheet_sha256"] = sha256_file(worksheet)
    register_path = HUMAN_TASK_DIR / "human_action_register_v3.csv"
    write_csv(register_path, actions, ACTION_FIELDS)
    handoff_path = HUMAN_TASK_DIR / "HUMAN_HANDOFF_V3.md"
    _write_handoff_markdown(actions, handoff_path)
    status_path = HUMAN_TASK_DIR / "human_action_status_v3.json"
    payload = {
        "blocking_actions": len(actions),
        "actions": actions,
        "force_overwrite_requested": bool(force),
        "register": str(register_path),
        "handoff": str(handoff_path),
    }
    write_json(status_path, payload)
    return payload

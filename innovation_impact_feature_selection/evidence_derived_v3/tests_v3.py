from __future__ import annotations

import csv
import json
import shutil
import sqlite3
import tempfile
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List

import coding
import handoff
import human_review_cli
import indicators
import local_ai
import pipeline
import providers
import reporting
import retrieval
import saturation
import screening
from common import deterministic_ten_percent, sha256_file, write_csv
from database import initialize, set_stage, snapshot_import_file


ROOT = Path(__file__).resolve().parent
V2_DATABASE = (
    ROOT.parent
    / "expanded_review_v2"
    / "outputs"
    / "expanded_search.sqlite3"
)


def _record(
    identity: str,
    doi: str,
    language: str = "en",
    route: str = "forward_citation_round_1",
) -> Dict[str, Any]:
    return {
        "provider": "OpenAlex",
        "record_key": f"doi:{doi}",
        "provider_id": f"https://openalex.org/{identity}",
        "doi": doi,
        "title": f"Evidence paper {identity}",
        "abstract": "This paper validates an article-level novelty metric.",
        "language": language,
        "publication_year": 2020,
        "work_type": "article",
        "source_url": f"https://example.org/{identity}",
        "referenced_works_json": "[]",
        "raw_json": "{}",
        "retrieval_route": route,
        "first_seen_at": "2026-07-28T00:00:00+00:00",
    }


def _insert_record(
    connection: sqlite3.Connection,
    value: Dict[str, Any],
) -> None:
    connection.execute(
        """
        INSERT INTO records(
            provider, record_key, provider_id, doi, title, abstract,
            language, publication_year, work_type, source_url,
            referenced_works_json, raw_json, retrieval_route, first_seen_at
        ) VALUES (
            :provider, :record_key, :provider_id, :doi, :title, :abstract,
            :language, :publication_year, :work_type, :source_url,
            :referenced_works_json, :raw_json, :retrieval_route,
            :first_seen_at
        )
        """,
        value,
    )


def test_init_is_independent_and_v2_read_only() -> None:
    """Initialization imports provenance without mutating the pilot."""
    before = sha256_file(V2_DATABASE)
    with tempfile.TemporaryDirectory() as temporary:
        connection = initialize(Path(temporary) / "v3.sqlite3")
        result = coding.initialize_project(connection)
        assert result["development_seeds"] == 53
        assert result["pilot_terms"] == 197
        assert result["development_evidence_terms"] == 66
        snapshot = connection.execute(
            """
            SELECT part_count, record_count, content_length_bytes
            FROM local_snapshot_sources
            WHERE snapshot_id = 'openalex_local_works'
            """
        ).fetchone()
        assert snapshot["part_count"] == 2127
        assert snapshot["record_count"] == 492_361_307
        assert snapshot["content_length_bytes"] == 639_189_333_248
        assert (
            connection.execute(
                """
                SELECT COUNT(*) FROM raw_terms
                WHERE source_type = 'pilot_v2_indicator'
                """
            ).fetchone()[0]
            == 197
        )
        assert (
            connection.execute(
                """
                SELECT COUNT(*) FROM raw_terms
                WHERE source_type = 'development_seed_hint'
                """
            ).fetchone()[0]
            == 66
        )
        expression = connection.execute(
            """
            SELECT logical_expression FROM logical_queries
            WHERE logical_query_id = 'B0001_DOMAIN_AGNOSTIC_BOOTSTRAP'
            """
        ).fetchone()[0]
        assert " AND " in expression
        assert "D01_" not in expression
        filter_expression = connection.execute(
            "SELECT filter_expression FROM physical_queries"
        ).fetchone()[0]
        assert "language:" not in filter_expression
        connection.close()
    assert sha256_file(V2_DATABASE) == before


def test_openalex_cursor_resume_and_deduplication() -> None:
    """Interrupted and duplicate pages resume into two unique works."""
    safe_error = providers.safe_provider_error(
        RuntimeError(
            "https://api.openalex.org/works?api_key=TOPSECRET&cursor=x"
        )
    )
    assert "TOPSECRET" not in safe_error
    assert "https://" not in safe_error
    with tempfile.TemporaryDirectory() as temporary:
        connection = initialize(Path(temporary) / "resume.sqlite3")
        connection.execute(
            """
            INSERT INTO logical_queries(
                logical_query_id, query_version, search_domain_id,
                family_label, logical_expression, object_terms_json,
                domain_terms_json, context_terms_json, status,
                archive_reason, press_status, query_hash
            ) VALUES (
                'L0001', 1, 'SD001', 'family', 'novelty',
                '[]', '[]', '[]', 'active', '', 'pass', 'logical'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO physical_queries(
                physical_query_id, logical_query_id, provider, expression,
                filter_expression, status, query_hash
            ) VALUES (
                'L0001__P001', 'L0001', 'OpenAlex', 'novelty',
                'type:article|review', 'active', 'physical'
            )
            """
        )
        first = {
            "id": "https://openalex.org/W1",
            "doi": "https://doi.org/10.1000/a",
            "display_name": "A",
            "publication_year": 2020,
            "type": "article",
            "language": "en",
            "abstract_inverted_index": {},
            "primary_location": {},
            "referenced_works": [],
        }
        second = {
            **first,
            "id": "https://openalex.org/W2",
            "doi": "https://doi.org/10.1000/b",
            "display_name": "B",
        }

        def fake_fetch(url: str) -> Dict[str, Any]:
            parameters = urllib.parse.parse_qs(
                urllib.parse.urlparse(url).query
            )
            assert parameters["per_page"][0] == "100"
            cursor = parameters["cursor"][0]
            if cursor == "*":
                return {
                    "meta": {"count": 3, "next_cursor": "CURSOR_2"},
                    "results": [first],
                }
            assert cursor == "CURSOR_2"
            return {
                "meta": {"count": 3, "next_cursor": None},
                "results": [first, second],
            }

        first_run = providers.retrieve_physical_query(
            connection,
            "L0001__P001",
            "formal",
            max_pages=1,
            fetcher=fake_fetch,
        )
        assert first_run["complete"] == 0
        assert first_run["next_cursor"] == "CURSOR_2"
        second_run = providers.retrieve_physical_query(
            connection,
            "L0001__P001",
            "formal",
            fetcher=fake_fetch,
        )
        assert second_run["complete"] == 1
        assert second_run["retrieved_rows"] == 3
        assert second_run["unique_hits"] == 2
        assert connection.execute(
            "SELECT COUNT(*) FROM records"
        ).fetchone()[0] == 2
        connection.execute(
            """
            INSERT INTO logical_queries(
                logical_query_id, query_version, search_domain_id,
                family_label, logical_expression, object_terms_json,
                domain_terms_json, context_terms_json, status,
                archive_reason, press_status, query_hash
            ) VALUES (
                'L0002', 1, 'SD002', 'shortfall', 'impact',
                '[]', '[]', '[]', 'active', '', 'pass', 'logical2'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO physical_queries(
                physical_query_id, logical_query_id, provider, expression,
                filter_expression, status, query_hash
            ) VALUES (
                'L0002__P001', 'L0002', 'OpenAlex', 'impact',
                'type:article', 'active', 'physical2'
            )
            """
        )

        def short_fetch(_: str) -> Dict[str, Any]:
            return {
                "meta": {"count": 2, "next_cursor": None},
                "results": [first],
            }

        short_run = providers.retrieve_physical_query(
            connection,
            "L0002__P001",
            "formal",
            fetcher=short_fetch,
        )
        assert short_run["complete"] == 0
        assert short_run["stopped_reason"] == "provider_early_termination"
        connection.close()


def _insert_term_codes(
    connection: sqlite3.Connection,
    term_id: str,
    term: str,
    domain: str,
    family: str,
    canonical: str | None = None,
    relation: str = "canonical",
    source_type: str = "bootstrap_literature",
) -> None:
    connection.execute(
        """
        INSERT INTO raw_terms(
            term_id, source_record_key, source_id, source_type,
            source_language_status, source_language_evidence,
            verbatim_term, normalized_term, match_key, location,
            evidence_span, proposed_role, status, exclusion_reason
        ) VALUES (?, ?, ?, ?, 'en',
                  'English title and abstract verified', ?, ?, ?,
                  'abstract', ?, 'measure', 'active', '')
        """,
        (
            term_id,
            f"doi:10.2000/{term_id.casefold()}",
            term_id,
            source_type,
            term,
            term.casefold(),
            term.casefold(),
            f"Evidence for {term}",
        ),
    )
    for role in ("AI", "H1", "H2"):
        connection.execute(
            """
            INSERT INTO term_coding(
                term_id, coder_role, canonical_term, term_family_label,
                term_relation, search_domain_label,
                search_domain_definition, query_family_label,
                cross_domain, decision, reason, coded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 'include',
                      'Evidence-linked construct assignment', ?)
            """,
            (
                term_id,
                role,
                canonical or term,
                family,
                relation,
                domain,
                f"Definition of {domain}",
                family,
                "2026-07-28T00:00:00+00:00",
            ),
        )


def test_hidden_seed_search_log_requires_all_independent_routes() -> None:
    """H2 must document review search plus both citation directions."""
    with tempfile.TemporaryDirectory() as temporary:
        connection = initialize(Path(temporary) / "hidden_log.sqlite3")
        rows = []
        for index, route in enumerate(
            sorted(coding.HIDDEN_SEED_SEARCH_ROUTES),
            start=1,
        ):
            rows.append(
                {
                    "search_run_id": f"H2LOG{index}",
                    "reviewer_role": "H2",
                    "route": route,
                    "source_name": "OpenAlex or source bibliography",
                    "exact_query_or_seed": f"exact source {index}",
                    "executed_at": "2026-07-29T00:00:00+00:00",
                    "retrieved_count": "10",
                    "screened_count": "10",
                    "eligible_seed_count": "1",
                    "eligible_seed_dois": f"10.6000/hidden{index}",
                    "completion_status": "complete",
                    "notes": "Independent H2 route completed.",
                }
            )
        path = Path(temporary) / "hidden_log.csv"
        write_csv(path, rows, coding.HIDDEN_SEED_SEARCH_LOG_FIELDS)
        result = coding.import_hidden_seed_search_log(connection, path)
        assert result["imported"] == 3
        assert result["missing_routes"] == []
        assert result["complete_runs"] == 3
        stored = connection.execute(
            """
            SELECT eligible_seed_dois_json
            FROM hidden_seed_search_log
            ORDER BY search_run_id
            """
        ).fetchall()
        assert json.loads(stored[0][0]) == ["10.6000/hidden1"]
        connection.close()


def test_pilot_terms_cannot_create_a_logical_query_family() -> None:
    """v2 pilots and derived development hints cannot independently raise Q."""
    with tempfile.TemporaryDirectory() as temporary:
        connection = initialize(Path(temporary) / "pilot_query.sqlite3")
        set_stage(connection, "initialized", "complete")
        set_stage(connection, "bootstrap_retrieval_complete", "complete")
        _insert_term_codes(
            connection,
            "FORMAL",
            "atypical combination",
            "Knowledge recombination",
            "supported family",
        )
        _insert_term_codes(
            connection,
            "PILOT",
            "pilot-only phrase",
            "Knowledge recombination",
            "unsupported family",
            source_type="pilot_v2_indicator",
        )
        _insert_term_codes(
            connection,
            "DEVHINT",
            "derived development phrase",
            "Knowledge recombination",
            "unsupported development family",
            source_type="development_seed_hint",
        )
        connection.commit()
        try:
            coding.derive_search_frame(connection)
        except RuntimeError as error:
            assert "cannot independently establish logical query" in str(
                error
            )
        else:
            raise AssertionError("A pilot-only query family increased Q")
        connection.close()


def test_k_q_p_are_derived_and_seed_recall_is_validated() -> None:
    """Domain/query counts emerge from H2-adjudicated source terms."""
    with tempfile.TemporaryDirectory() as temporary:
        connection = initialize(Path(temporary) / "derive.sqlite3")
        set_stage(connection, "initialized", "complete")
        set_stage(connection, "bootstrap_retrieval_complete", "complete")
        connection.execute(
            """
            INSERT INTO logical_queries(
                logical_query_id, query_version, search_domain_id,
                family_label, logical_expression, object_terms_json,
                domain_terms_json, context_terms_json, status,
                archive_reason, press_status, query_hash
            ) VALUES (
                'B0002_TEST_CITATION', 1, 'BOOTSTRAP_CITATION',
                'development citation evidence', '', '[]', '[]', '[]',
                'bootstrap', '', 'not_applicable', 'citation-logical-hash'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO physical_queries(
                physical_query_id, logical_query_id, provider, expression,
                filter_expression, status, query_hash
            ) VALUES (
                'B0002_TEST_CITATION__P001', 'B0002_TEST_CITATION',
                'OpenAlex', '', 'cites:W1', 'active',
                'citation-physical-hash'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO query_runs(
                provider, physical_query_id, run_role, query_hash,
                reported_total, retrieved_rows, unique_hits, pages,
                next_cursor, complete, stopped_reason, error, updated_at
            ) VALUES (
                'OpenAlex', 'B0002_TEST_CITATION__P001',
                'bootstrap_citation', 'citation-physical-hash',
                0, 0, 0, 1, '', 1, '', '',
                '2026-07-29T00:00:00+00:00'
            )
            """
        )
        _insert_term_codes(
            connection,
            "T1",
            "atypical combination",
            "Knowledge recombination",
            "combination rarity",
        )
        _insert_term_codes(
            connection,
            "T2",
            "unprecedented pairing",
            "Knowledge recombination",
            "combination rarity",
            canonical="atypical combination",
            relation="synonym",
        )
        _insert_term_codes(
            connection,
            "T3",
            "semantic distance",
            "Semantic departure",
            "semantic divergence",
        )
        connection.commit()
        result = coding.derive_search_frame(connection)
        assert result["K"] == 2
        assert result["Q"] == 2
        assert result["P"] == 2
        assert connection.execute(
            """
            SELECT COUNT(*) FROM logical_queries
            WHERE logical_query_id = 'B0002_TEST_CITATION'
            """
        ).fetchone()[0] == 1
        assert connection.execute(
            """
            SELECT complete FROM query_runs
            WHERE physical_query_id = 'B0002_TEST_CITATION__P001'
            """
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM term_families"
        ).fetchone()[0] == 2
        assert connection.execute(
            "SELECT COUNT(*) FROM canonical_terms"
        ).fetchone()[0] == 2
        assert not any(
            row[0].startswith("D01")
            for row in connection.execute(
                "SELECT label FROM search_domains"
            )
        )
        press_rows = []
        for query in connection.execute(
            """
            SELECT logical_query_id, search_domain_id, family_label,
                   logical_expression
            FROM logical_queries
            WHERE logical_query_id LIKE 'L%'
            ORDER BY logical_query_id
            """
        ):
            press_rows.append(
                {
                    **dict(query),
                    "reviewer_role": "H2",
                    "concepts_complete": "true",
                    "boolean_logic_valid": "true",
                    "spelling_valid": "true",
                    "phrases_valid": "true",
                    "limits_justified": "true",
                    "covered_by_logical_query_id": "",
                    "logical_coverage_verified": "",
                    "result_set_coverage_verified": "",
                    "independent_construct_role": "",
                    "decision": "pass",
                    "notes": "PRESS checks passed",
                }
            )
        press_path = Path(temporary) / "press.csv"
        write_csv(press_path, press_rows, coding.PRESS_FIELDS)
        press_result = coding.import_press(connection, press_path)
        assert press_result["pass"] == 2
        for seed_id, role, hidden, supplied_by in (
            ("DEV1", "development", 0, "v2_evidence"),
            ("VAL1", "validation", 1, "H2"),
        ):
            connection.execute(
                """
                INSERT INTO evidence_seeds(
                    seed_id, doi, citation, publication_year, language,
                    seed_role, supplied_by, hidden_during_development,
                    eligibility_status
                ) VALUES (?, ?, 'citation', 2020, 'en', ?, ?, ?,
                          'eligible')
                """,
                (
                    seed_id,
                    f"10.3000/{seed_id.casefold()}",
                    role,
                    supplied_by,
                    hidden,
                ),
            )
        for index, route in enumerate(
            sorted(coding.HIDDEN_SEED_SEARCH_ROUTES),
            start=1,
        ):
            connection.execute(
                """
                INSERT INTO hidden_seed_search_log(
                    search_run_id, reviewer_role, route, source_name,
                    exact_query_or_seed, executed_at, retrieved_count,
                    screened_count, eligible_seed_count,
                    eligible_seed_dois_json, completion_status, notes,
                    imported_at
                ) VALUES (?, 'H2', ?, 'independent source', 'exact input',
                          '2026-07-28T00:00:00+00:00', 1, 1, ?,
                          ?,
                          'complete', 'Independent route completed',
                          '2026-07-28T00:00:00+00:00')
                """,
                (
                    f"H2RUN{index}",
                    route,
                    1 if index == 1 else 0,
                    (
                        '["10.3000/val1"]'
                        if index == 1
                        else "[]"
                    ),
                ),
            )
        connection.commit()

        validation_urls: List[str] = []

        def fake_fetch(url: str) -> Dict[str, Any]:
            validation_urls.append(url)
            parameters = urllib.parse.parse_qs(
                urllib.parse.urlparse(url).query
            )
            filter_expression = parameters.get("filter", [""])[0]
            if filter_expression.startswith("doi:"):
                dois = filter_expression.removeprefix("doi:").split("|")
                return {
                    "meta": {"count": len(dois)},
                    "results": [
                        {"doi": f"https://doi.org/{doi}"}
                        for doi in dois
                    ],
                }
            return {"meta": {"count": 1}, "results": [{"id": "W"}]}

        validation = coding.validate_search_frame(
            connection,
            fetcher=fake_fetch,
        )
        assert validation["missed_query_term_seeds"] == []
        assert validation["recalled_seeds"] == 2
        assert validation["seed_validation_api_requests"] == 3
        assert (
            connection.execute(
                """
                SELECT status FROM stage_status
                WHERE stage = 'search_frame_validated'
                """
            ).fetchone()[0]
            == "complete"
        )
        chunks = coding._split_term_block(
            [f"very long evidence term number {index} " + "x" * 80
             for index in range(40)],
            ["paper"],
            ["metric"],
            maximum_length=400,
        )
        assert len(chunks) > 1
        connection.execute(
            """
            UPDATE hidden_seed_search_log
            SET eligible_seed_dois_json = '["10.3000/not-imported"]'
            WHERE search_run_id = 'H2RUN1'
            """
        )
        connection.commit()
        try:
            coding.validate_search_frame(
                connection,
                fetcher=fake_fetch,
            )
        except RuntimeError as error:
            assert "DOI reconciliation failed" in str(error)
        else:
            raise AssertionError(
                "Hidden-seed validation accepted mismatched DOI sets"
            )
        connection.close()


def test_logical_redundancy_uses_complete_result_sets() -> None:
    """A redundancy claim is archived only after a cursor-complete subset."""
    with tempfile.TemporaryDirectory() as temporary:
        connection = initialize(Path(temporary) / "redundancy.sqlite3")
        for index in (1, 2):
            logical_id = f"L{index:04d}"
            connection.execute(
                """
                INSERT INTO logical_queries(
                    logical_query_id, query_version, search_domain_id,
                    family_label, logical_expression, object_terms_json,
                    domain_terms_json, context_terms_json, status,
                    archive_reason, press_status, query_hash
                ) VALUES (?, 1, ?, ?, ?, '[]', '[]', '[]',
                          'active', '', 'pending', ?)
                """,
                (
                    logical_id,
                    f"SD{index:03d}",
                    f"family{index}",
                    f"term{index}",
                    f"logical{index}",
                ),
            )
            connection.execute(
                """
                INSERT INTO physical_queries(
                    physical_query_id, logical_query_id, provider,
                    expression, filter_expression, status, query_hash
                ) VALUES (?, ?, 'OpenAlex', ?, 'type:article',
                          'active', ?)
                """,
                (
                    f"{logical_id}__P001",
                    logical_id,
                    f"term{index}",
                    f"physical{index}",
                ),
            )
        rows = []
        for logical_id, decision in (
            ("L0001", "archive_redundant"),
            ("L0002", "pass"),
        ):
            rows.append(
                {
                    "logical_query_id": logical_id,
                    "search_domain_id": (
                        "SD001" if logical_id == "L0001" else "SD002"
                    ),
                    "family_label": logical_id,
                    "logical_expression": logical_id,
                    "reviewer_role": "H2",
                    "concepts_complete": "true",
                    "boolean_logic_valid": "true",
                    "spelling_valid": "true",
                    "phrases_valid": "true",
                    "limits_justified": "true",
                    "covered_by_logical_query_id": (
                        "L0002" if decision == "archive_redundant" else ""
                    ),
                    "logical_coverage_verified": (
                        "true" if decision == "archive_redundant" else ""
                    ),
                    "result_set_coverage_verified": "false",
                    "independent_construct_role": (
                        "false" if decision == "archive_redundant" else ""
                    ),
                    "decision": decision,
                    "notes": "H2 redundancy assessment",
                }
            )
        path = Path(temporary) / "redundancy_press.csv"
        write_csv(path, rows, coding.PRESS_FIELDS)
        coding.import_press(connection, path)
        item = {
            "id": "https://openalex.org/W77",
            "doi": "https://doi.org/10.3777/same",
            "display_name": "Same work",
            "publication_year": 2020,
            "type": "article",
            "language": "en",
            "abstract_inverted_index": {},
            "primary_location": {},
            "referenced_works": [],
        }

        def fake_fetch(_: str) -> Dict[str, Any]:
            return {
                "meta": {"count": 1, "next_cursor": None},
                "results": [item],
            }

        archived = coding._resolve_redundancy_reviews(
            connection,
            fetcher=fake_fetch,
        )
        assert archived == ["L0001"]
        status = connection.execute(
            """
            SELECT status, press_status FROM logical_queries
            WHERE logical_query_id = 'L0001'
            """
        ).fetchone()
        assert tuple(status) == ("archived", "archived_redundant")
        assert connection.execute(
            """
            SELECT result_set_coverage_verified FROM press_reviews
            WHERE logical_query_id = 'L0001'
            """
        ).fetchone()[0] == 1
        connection.close()


def _screen_row(
    record: Dict[str, Any],
    role: str,
    language: str,
    decision: str,
    reason: str = "",
) -> Dict[str, Any]:
    return {
        "record_key": record["record_key"],
        "doi": record["doi"],
        "title": record["title"],
        "abstract": record["abstract"],
        "openalex_language": record["language"],
        "publication_year": record["publication_year"],
        "work_type": record["work_type"],
        "reviewer_role": role,
        "language_judgment": language,
        "language_evidence": record["title"],
        "decision": decision,
        "exclusion_reason": reason,
        "evidence_span": (
            "This paper validates an article-level novelty metric."
        ),
        "notes": "",
    }


def test_screening_requires_h2_and_retains_language_exclusion() -> None:
    """H2 resolves inclusions/disagreements and language stays in PRISMA."""
    with tempfile.TemporaryDirectory() as temporary:
        connection = initialize(Path(temporary) / "screen.sqlite3")
        set_stage(connection, "formal_retrieval_complete", "complete")
        include = _record("W1", "10.4000/include")
        exclusion_index = 0
        while True:
            excluded_doi = f"10.4000/exclude{exclusion_index}"
            if not deterministic_ten_percent(excluded_doi):
                break
            exclusion_index += 1
        exclude = _record("W2", excluded_doi)
        non_english = _record("W3", "10.4000/nonenglish", "fr")
        for record in (include, exclude, non_english):
            _insert_record(connection, record)
        connection.commit()
        rows = [
            _screen_row(include, "AI", "en", "include"),
            _screen_row(include, "H1", "en", "include"),
            _screen_row(include, "H2", "en", "include"),
            _screen_row(
                exclude,
                "AI",
                "en",
                "exclude",
                "E_NOT_INDICATOR_PREDICTOR_VALIDATION",
            ),
            _screen_row(
                exclude,
                "H1",
                "en",
                "exclude",
                "E_NOT_INDICATOR_PREDICTOR_VALIDATION",
            ),
            _screen_row(non_english, "AI", "uncertain", "uncertain"),
            _screen_row(
                non_english,
                "H1",
                "non_en",
                "exclude",
                "E_LANGUAGE_NON_ENGLISH",
            ),
            _screen_row(
                non_english,
                "H2",
                "non_en",
                "exclude",
                "E_LANGUAGE_NON_ENGLISH",
            ),
        ]
        imported = 0
        for role in ("AI", "H1", "H2"):
            role_rows = [
                row for row in rows if row["reviewer_role"] == role
            ]
            path = Path(temporary) / f"screening_{role}.csv"
            write_csv(path, role_rows, screening.SCREENING_FIELDS)
            imported += screening.import_screening(connection, path)
        assert imported == len(rows)
        result = screening.finalize_screening(connection)
        assert result["include"] == 1
        assert result["exclude"] == 2
        assert result["language_exclusions"] == 1
        include_final = connection.execute(
            """
            SELECT h2_required, h2_completed FROM screening_final
            WHERE record_key = ?
            """,
            (include["record_key"],),
        ).fetchone()
        assert tuple(include_final) == (1, 1)
        assert screening.screening_exclusion_counts(connection)[
            "E_LANGUAGE_NON_ENGLISH"
        ] == 1
        connection.close()


def test_ai_language_evidence_normalization_is_exact_and_audited() -> None:
    """Legacy AI provenance prose is archived and replaced without recoding."""
    with tempfile.TemporaryDirectory() as temporary:
        temporary_path = Path(temporary)
        connection = initialize(temporary_path / "ai_evidence.sqlite3")
        record = _record("WAIEXACT", "10.4000/ai-exact")
        _insert_record(connection, record)
        connection.execute(
            """
            INSERT INTO discovery_queries(
                discovery_query_id, query_role, stratum_label, expression,
                filter_expression, sample_size, random_seed, query_hash,
                status, archive_reason
            ) VALUES ('DS_AI_EXACT', 'base_year_type', 'test', 'test', '',
                      1, 1, 'hash', 'active', '')
            """
        )
        connection.execute(
            """
            INSERT INTO discovery_hits(
                discovery_query_id, record_key, sample_rank,
                selection_hash, review_rank, review_round, review_status
            ) VALUES ('DS_AI_EXACT', ?, 1, 'selection', 1, 1, 'screened')
            """,
            (record["record_key"],),
        )
        connection.execute(
            """
            INSERT INTO screening_decisions(
                record_key, reviewer_role, language_judgment,
                language_evidence, decision, exclusion_reason,
                evidence_span, notes, decided_at
            ) VALUES (?, 'AI', 'en', 'model and prompt provenance',
                      'include', '', ?, 'legacy decision',
                      '2026-07-29T00:00:00+00:00')
            """,
            (
                record["record_key"],
                "This paper validates an article-level novelty metric.",
            ),
        )
        connection.commit()
        output_dir = temporary_path / "outputs"
        output_dir.mkdir()
        old_output = (
            output_dir / "discovery_round_1_screening_ai_completed_v3.csv"
        )
        write_csv(
            old_output,
            [
                {
                    **_screen_row(record, "AI", "en", "include"),
                    "language_evidence": "model and prompt provenance",
                    "review_round": "1",
                    "discovery_query_ids": "DS_AI_EXACT",
                }
            ],
            [
                *screening.SCREENING_FIELDS,
                "review_round",
                "discovery_query_ids",
            ],
        )
        result = local_ai.normalize_ai_language_evidence(
            connection,
            output_dir,
        )
        assert result["changed"] == 1
        assert result["iterations_rewritten"] == [1]
        decision = connection.execute(
            """
            SELECT language_evidence, decision
            FROM screening_decisions
            WHERE record_key = ? AND reviewer_role = 'AI'
            """,
            (record["record_key"],),
        ).fetchone()
        assert tuple(decision) == (record["title"], "include")
        assert connection.execute(
            """
            SELECT COUNT(*) FROM source_snapshots
            WHERE role = 'review_import:'
                         || 'ai_screening_pre_exact_language_evidence'
            """
        ).fetchone()[0] == 1
        with old_output.open(encoding="utf-8", newline="") as handle:
            repaired = next(csv.DictReader(handle))
        assert repaired["language_evidence"] == record["title"]
        assert local_ai.normalize_ai_language_evidence(
            connection,
            output_dir,
        )["changed"] == 0
        connection.close()


def test_crossref_conflicts_require_h2_resolution() -> None:
    """Metadata conflicts are queued and cannot silently overwrite records."""
    with tempfile.TemporaryDirectory() as temporary:
        temporary_path = Path(temporary)
        connection = initialize(temporary_path / "crossref.sqlite3")
        record = _record("W9", "10.4500/conflict")
        _insert_record(connection, record)
        connection.commit()

        def fake_crossref(_: str) -> Dict[str, Any]:
            return {
                "message": {
                    "DOI": "10.4500/conflict",
                    "title": ["A completely different title"],
                    "published": {"date-parts": [[2011, 1, 1]]},
                    "type": "book",
                }
            }

        counts = providers.crossref_validate_scope(
            connection,
            [record["record_key"]],
            fetcher=fake_crossref,
        )
        assert counts["conflict"] == 1
        assert list(
            pipeline._unvalidated_record_keys(
                connection,
                [record["record_key"]],
            )
        ) == []
        conflict_path = temporary_path / "conflicts.csv"
        assert providers.export_crossref_conflicts(
            connection,
            conflict_path,
        ) == 1
        rows = []
        with conflict_path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["reviewer_role"] = "H2"
                row["resolution"] = "accept_openalex"
                row["resolution_notes"] = "Publisher record verified"
                rows.append(row)
        write_csv(conflict_path, rows, list(rows[0]))
        assert providers.import_crossref_resolutions(
            connection,
            conflict_path,
        ) == 1
        assert connection.execute(
            """
            SELECT status FROM crossref_validation WHERE record_key = ?
            """,
            (record["record_key"],),
        ).fetchone()[0] == "resolved"
        date_variant = _record(
            "W10",
            "10.4500/date-variant",
        )
        _insert_record(connection, date_variant)
        connection.commit()

        def fake_date_variant(_: str) -> Dict[str, Any]:
            return {
                "message": {
                    "DOI": "10.4500/date-variant",
                    "title": [date_variant["title"]],
                    "published": {"date-parts": [[2021, 1, 1]]},
                    "type": "journal-article",
                }
            }

        date_counts = providers.crossref_validate_scope(
            connection,
            [date_variant["record_key"]],
            fetcher=fake_date_variant,
        )
        assert date_counts["validated_date_variant"] == 1
        assert connection.execute(
            """
            SELECT status, conflict_reason
            FROM crossref_validation WHERE record_key = ?
            """,
            (date_variant["record_key"],),
        ).fetchone()[:] == (
            "validated_date_variant",
            "year_date_variant",
        )
        connection.execute(
            """
            UPDATE crossref_validation
            SET status = 'conflict', conflict_reason = 'year'
            WHERE record_key = ?
            """,
            (date_variant["record_key"],),
        )
        assert providers.reclassify_crossref_date_variants(
            connection,
            [date_variant["record_key"]],
        ) == 1
        assert providers.export_crossref_conflicts(
            connection,
            temporary_path / "date_variant_conflicts.csv",
            [date_variant["record_key"]],
        ) == 0
        missing = _record("W11", "10.4500/crossref-missing")
        _insert_record(connection, missing)
        connection.commit()

        def fake_not_found(_: str) -> Dict[str, Any]:
            raise RuntimeError(
                "Provider request failed after retries: "
                "HTTPError status=404"
            )

        missing_counts = providers.crossref_validate_scope(
            connection,
            [missing["record_key"]],
            fetcher=fake_not_found,
        )
        assert missing_counts["conflict"] == 1
        assert connection.execute(
            """
            SELECT status, conflict_reason FROM crossref_validation
            WHERE record_key = ?
            """,
            (missing["record_key"],),
        ).fetchone()[:] == (
            "conflict",
            "crossref_doi_not_found",
        )
        connection.execute(
            """
            UPDATE crossref_validation
            SET status = 'error',
                conflict_reason =
                    'RuntimeError: HTTPError status=404'
            WHERE record_key = ?
            """,
            (missing["record_key"],),
        )
        connection.commit()
        assert providers.reclassify_crossref_not_found(
            connection,
            [missing["record_key"]],
        ) == 1
        connection.close()


def _indicator_row(
    record: Dict[str, Any],
    name: str,
    group: str,
    priority: int,
    redundancy: str,
    fulltext_path: Path,
    requires_future: bool = False,
    source_role: str = "original_definition",
    formula_reproducible: bool = True,
) -> Dict[str, Any]:
    row = {field: "" for field in indicators.INDICATOR_FIELDS}
    row.update(
        {
            "record_key": record["record_key"],
            "doi": record["doi"],
            "source_title": record["title"],
            "source_disposition": "extracted",
            "english_fulltext_status": "verified",
            "disposition_notes": "Full text checked",
            "disposition_decided_by": "H1|H2",
            "raw_name_en": name,
            "canonical_name_en": name,
            "label_zh": "测试指标",
            "source_id": f"SRC_{group}_{name}",
            "research_group": group,
            "research_group_id": group,
            "research_group_evidence": (
                f"Corresponding-author and affiliation cluster: {group}"
            ),
            "source_role": source_role,
            "formula_location": (
                "p. 4, Eq. 1" if formula_reproducible else ""
            ),
            "evidence_span": (
                "The indicator is defined by Equation 1."
            ),
            "formula": (
                "x_i = count_i / total_i"
                if formula_reproducible
                else ""
            ),
            "units": "share",
            "parameters": "none",
            "direction": "higher",
            "missing_rule": "missing if total_i = 0",
            "required_data": "audited numerator|audited denominator",
            "maximum_information_time": (
                "T0+2y" if requires_future else "T0"
            ),
            "scope_role": (
                "outcome_only" if requires_future else "direct_innovation"
            ),
            "validation_summary": "Original application",
            "evidence_direction": "definition_only",
            "negative_evidence": "",
            "fulltext_source_url": "https://example.org/lawful-fulltext",
            "fulltext_local_path": str(fulltext_path),
            "fulltext_sha256": "",
            "fulltext_license": "CC-BY-4.0",
            "english_fulltext_verified": "true",
            "article_level": "true",
            "primary_or_foundational_evidence": str(
                formula_reproducible
            ).casefold(),
            "formula_reproducible": str(
                formula_reproducible
            ).casefold(),
            "t0_computable": "false" if requires_future else "true",
            "requires_future": str(requires_future).casefold(),
            "data_status": "materialized_audited",
            "bias_policy": "allowed_core",
            "fatal_validity_concern": "false",
            "uses_outcome_for_selection": "false",
            "quality_audit_status": "pass",
            "nonconstant": "true",
            "h2_approved": "true",
            "evidence_strength": "high",
            "stability_score": "0.95",
            "stability_basis": (
                "Source-reported robustness across field-normalized "
                "specifications; normalized score 0.95."
            ),
            "selection_priority": str(priority),
            "redundancy_family": redundancy,
            "extracted_by": "AI|H1",
            "verified_by": "H1|H2",
            "verification_notes": "H1 verified formula and source location",
            "adjudication_notes": "H2 approved the formula and T0 boundary",
            "status": "candidate",
        }
    )
    return row


def _h1_indicator_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a resolved test row into an independent blind H1 submission."""
    result = dict(row)
    result.update(
        {
            "disposition_decided_by": "H1",
            "verified_by": "H1",
            "h2_approved": "false",
            "adjudication_notes": "",
        }
    )
    return result


def test_open_fulltext_acquisition_is_lawful_hashed_and_resumable() -> None:
    """Only OpenAlex-marked OA PDFs are acquired and frozen for review."""
    with tempfile.TemporaryDirectory() as temporary:
        temporary_path = Path(temporary)
        connection = initialize(temporary_path / "fulltext.sqlite3")
        record = _record("WOA", "10.7000/open-fulltext")
        record["raw_json"] = json.dumps(
            {
                "primary_location": {
                    "is_oa": False,
                    "pdf_url": "",
                    "license": None,
                }
            }
        )
        _insert_record(connection, record)
        connection.execute(
            """
            INSERT INTO screening_final(
                record_key, final_language, final_decision,
                exclusion_reason, h2_required, h2_completed,
                adjudication_reason, finalized_at
            ) VALUES (?, 'en', 'include', '', 1, 1, 'H2 included',
                      '2026-07-29T00:00:00+00:00')
            """,
            (record["record_key"],),
        )
        set_stage(connection, "literature_screened", "complete")
        connection.commit()
        original_openalex_keys = providers.openalex_api_keys
        providers.openalex_api_keys = lambda: ["test-key"]

        def fake_location_fetch(url: str) -> Dict[str, Any]:
            assert "/works/WOA?" in url
            assert "best_oa_location" in url
            return {
                "id": "https://openalex.org/WOA",
                "primary_location": {
                    "is_oa": False,
                    "pdf_url": None,
                },
                "best_oa_location": {
                    "is_oa": True,
                    "pdf_url": "https://example.org/open-paper.pdf",
                    "license": "cc-by",
                },
                "locations": [],
            }

        try:
            hydration = providers.hydrate_openalex_locations(
                connection,
                [record["record_key"]],
                fetcher=fake_location_fetch,
            )
        finally:
            providers.openalex_api_keys = original_openalex_keys
        assert hydration["hydrated"] == 1
        assert connection.execute(
            """
            SELECT status FROM openalex_location_hydration
            WHERE record_key = ?
            """,
            (record["record_key"],),
        ).fetchone()[0] == "complete"

        def fake_pdf(
            url: str,
            timeout_seconds: int,
            maximum_bytes: int,
        ) -> Dict[str, Any]:
            assert url == "https://example.org/open-paper.pdf"
            assert timeout_seconds == 60
            assert maximum_bytes == 100_000_000
            return {
                "body": b"%PDF-1.7\nformula evidence\n%%EOF",
                "final_url": url,
                "content_type": "application/pdf",
            }

        fulltext_dir = temporary_path / "open_fulltexts"
        first = indicators.acquire_open_fulltexts(
            connection,
            fulltext_dir,
            fetcher=fake_pdf,
        )
        assert first["downloaded"] == 1
        acquired = connection.execute(
            """
            SELECT * FROM fulltext_acquisitions
            WHERE record_key = ?
            """,
            (record["record_key"],),
        ).fetchone()
        assert acquired["status"] == "downloaded"
        assert Path(acquired["local_path"]).is_file()
        assert acquired["sha256"] == sha256_file(
            Path(acquired["local_path"])
        )
        assert connection.execute(
            """
            SELECT COUNT(*) FROM source_snapshots
            WHERE role = 'candidate_open_fulltext'
            """
        ).fetchone()[0] == 1
        second = indicators.acquire_open_fulltexts(
            connection,
            fulltext_dir,
            fetcher=fake_pdf,
        )
        assert second["resumed"] == 1
        extraction_path = temporary_path / "indicator_extraction.csv"
        assert indicators.export_indicator_extraction(
            connection,
            extraction_path,
        ) == 1
        with extraction_path.open(encoding="utf-8", newline="") as handle:
            exported = next(csv.DictReader(handle))
        assert exported["candidate_fulltext_url"].endswith(".pdf")
        assert exported["fulltext_local_path"] == acquired["local_path"]
        assert exported["fulltext_sha256"] == acquired["sha256"]
        assert exported["fulltext_license"] == "cc-by"
        assert exported["english_fulltext_verified"] == ""
        verified_text_path = temporary_path / "verified_fulltext.txt"
        verified_text_path.write_text(
            "The indicator is defined by Equation 1.\n"
            "Corresponding-author and affiliation cluster: GroupOA",
            encoding="utf-8",
        )
        h1_row = _h1_indicator_row(
            _indicator_row(
                record,
                "Open evidence metric",
                "GroupOA",
                0,
                "oa_family",
                verified_text_path,
            )
        )
        h1_path = temporary_path / "indicator_h1.csv"
        write_csv(h1_path, [h1_row], indicators.INDICATOR_FIELDS)
        details = indicators.import_indicators(connection, h1_path)
        assert details["sources_without_h2_review"] == [
            record["record_key"]
        ]
        assert details["retained_mentions_without_h2_review"]
        original_task_dir = handoff.HUMAN_TASK_DIR
        handoff.HUMAN_TASK_DIR = temporary_path / "human_tasks"
        try:
            actions: List[Dict[str, Any]] = []
            handoff._downstream_actions(
                connection,
                force=True,
                actions=actions,
            )
        finally:
            handoff.HUMAN_TASK_DIR = original_task_dir
        assert [item["action_id"] for item in actions] == [
            "INDICATOR_ADJUDICATION_H2"
        ]
        family = connection.execute(
            "SELECT * FROM indicator_families"
        ).fetchone()
        original_chat_json = local_ai._chat_json

        def fake_dimension_chat(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            del args, kwargs
            return {
                "decision": "include",
                "dimension_label": "Knowledge recombination novelty",
                "dimension_definition": "Novel combinations at T0",
                "construct_role": "substantive_innovation",
                "information_source": "reference graph",
                "t0_boundary": "T0",
                "bias_risk": "field coverage",
                "reason": "Formula measures a paper-level T0 construct.",
            }

        local_ai._chat_json = fake_dimension_chat
        try:
            ai_code = local_ai._code_one_dimension(family, "test-model")
        finally:
            local_ai._chat_json = original_chat_json
        assert ai_code["construct_role"] == "substantive_innovation"
        assert ai_code["decision"] == "include"
        legacy_response = dict(fake_dimension_chat())
        legacy_response["dimension_label"] = "D01_legacy"
        local_ai._chat_json = lambda *args, **kwargs: legacy_response
        try:
            try:
                local_ai._code_one_dimension(family, "test-model")
            except ValueError as error:
                assert "legacy D01-D12" in str(error)
            else:
                raise AssertionError("AI dimension coder accepted D01 label")
        finally:
            local_ai._chat_json = original_chat_json
        assert connection.execute(
            """
            SELECT status FROM stage_status
            WHERE stage = 'indicators_extracted'
            """
        ).fetchone()[0] == "ready"
        h2_path = temporary_path / "indicator_h2.csv"
        assert indicators.export_indicator_adjudication(
            connection,
            h2_path,
        ) == 1
        with h2_path.open(encoding="utf-8", newline="") as handle:
            h2_row = next(csv.DictReader(handle))
        assert h2_row["h2_approved"] == ""
        assert h2_row["verified_by"] == "H1|H2"
        assert h2_row["disposition_decided_by"] == "H1|H2"
        connection.close()


def test_indicator_gates_redundancy_and_dimension_retention() -> None:
    """One passed representative survives; an empty dimension is removed."""
    with tempfile.TemporaryDirectory() as temporary:
        temporary_path = Path(temporary)
        connection = initialize(temporary_path / "indicators.sqlite3")
        set_stage(connection, "literature_screened", "complete")
        first = _record("W1", "10.5000/a", route="formal:test")
        second = _record("W2", "10.5000/b", route="formal:test")
        fulltext_path = temporary_path / "verified_fulltext.txt"
        fulltext_path.write_text(
            "The indicator is defined by Equation 1.\n"
            + "\n".join(
                f"Corresponding-author and affiliation cluster: {group}"
                for group in (
                    "ReviewTeam",
                    "TeamA",
                    "TeamB",
                    "TeamC",
                    "TeamD",
                    "TeamIncomplete",
                    "TeamAbsent",
                )
            ),
            encoding="utf-8",
        )
        for record in (first, second):
            _insert_record(connection, record)
            connection.execute(
                """
                INSERT INTO screening_final(
                    record_key, final_language, final_decision,
                    exclusion_reason, h2_required, h2_completed,
                    adjudication_reason, finalized_at
                ) VALUES (?, 'en', 'include', '', 1, 1, 'H2', ?)
                """,
                (record["record_key"], "2026-07-28T00:00:00+00:00"),
            )
        connection.commit()
        rows = [
            _indicator_row(
                first,
                "Atypical combination share",
                "ReviewTeam",
                4,
                "combination_share",
                fulltext_path,
                source_role="review_discovery",
                formula_reproducible=False,
            ),
            _indicator_row(
                first,
                "Atypical combination share",
                "TeamA",
                0,
                "combination_share",
                fulltext_path,
            ),
            _indicator_row(
                second,
                "Atypical combination share",
                "TeamB",
                0,
                "combination_share",
                fulltext_path,
            ),
            _indicator_row(
                first,
                "Alternative combination share",
                "TeamC",
                2,
                "combination_share",
                fulltext_path,
                source_role="original_application",
            ),
            _indicator_row(
                first,
                "Future citation count",
                "TeamD",
                0,
                "future_citation",
                fulltext_path,
                requires_future=True,
            ),
        ]
        invalid_priority = _h1_indicator_row(dict(rows[1]))
        invalid_priority["selection_priority"] = "9"
        invalid_priority_path = temporary_path / "invalid_priority.csv"
        write_csv(
            invalid_priority_path,
            [invalid_priority],
            indicators.INDICATOR_FIELDS,
        )
        try:
            indicators.import_indicators(
                connection,
                invalid_priority_path,
            )
        except ValueError as error:
            assert "frozen by source_role" in str(error)
            connection.rollback()
        else:
            raise AssertionError(
                "Reviewer-selected redundancy priority was accepted"
            )
        invalid_stability = _h1_indicator_row(dict(rows[1]))
        invalid_stability["stability_score"] = "1.5"
        invalid_stability_path = temporary_path / "invalid_stability.csv"
        write_csv(
            invalid_stability_path,
            [invalid_stability],
            indicators.INDICATOR_FIELDS,
        )
        try:
            indicators.import_indicators(
                connection,
                invalid_stability_path,
            )
        except ValueError as error:
            assert "between 0 and 1" in str(error)
            connection.rollback()
        else:
            raise AssertionError("Out-of-range stability was accepted")
        h1_indicator_path = temporary_path / "indicators_h1.csv"
        write_csv(
            h1_indicator_path,
            [_h1_indicator_row(row) for row in rows],
            indicators.INDICATOR_FIELDS,
        )
        h1_imported = indicators.import_indicators(
            connection,
            h1_indicator_path,
        )
        assert h1_imported["canonical_indicator_families"] == 3
        assert h1_imported["sources_without_h2_review"]
        incomplete_formula = _h1_indicator_row(
            _indicator_row(
                first,
                "Incomplete formula metric",
                "TeamIncomplete",
                0,
                "incomplete_formula",
                fulltext_path,
            )
        )
        incomplete_formula["direction"] = ""
        incomplete_path = temporary_path / "incomplete_formula.csv"
        write_csv(
            incomplete_path,
            [incomplete_formula],
            indicators.INDICATOR_FIELDS,
        )
        try:
            indicators.import_indicators(connection, incomplete_path)
        except ValueError as error:
            assert "lacks formula" in str(error)
            connection.rollback()
        else:
            raise AssertionError("Incomplete formula evidence was accepted")
        absent_span = _h1_indicator_row(
            _indicator_row(
                first,
                "Absent-span metric",
                "TeamAbsent",
                0,
                "absent_span",
                fulltext_path,
            )
        )
        absent_span["evidence_span"] = (
            "This sentence does not occur in the frozen evidence file."
        )
        absent_span_path = temporary_path / "absent_span.csv"
        write_csv(
            absent_span_path,
            [absent_span],
            indicators.INDICATOR_FIELDS,
        )
        try:
            indicators.import_indicators(connection, absent_span_path)
        except ValueError as error:
            assert "not present in the frozen full text" in str(error)
            connection.rollback()
        else:
            raise AssertionError(
                "A non-verbatim full-text evidence span was accepted"
            )
        absent_group = _h1_indicator_row(
            _indicator_row(
                first,
                "Absent-group metric",
                "TeamNotInFrozenText",
                0,
                "absent_group",
                fulltext_path,
            )
        )
        absent_group_path = temporary_path / "absent_group.csv"
        write_csv(
            absent_group_path,
            [absent_group],
            indicators.INDICATOR_FIELDS,
        )
        try:
            indicators.import_indicators(
                connection,
                absent_group_path,
            )
        except ValueError as error:
            assert "author/affiliation evidence is not present" in str(error)
            connection.rollback()
        else:
            raise AssertionError(
                "Unverified research-group evidence was accepted"
            )
        h2_indicator_path = temporary_path / "indicators_h2.csv"
        write_csv(
            h2_indicator_path,
            rows,
            indicators.INDICATOR_FIELDS,
        )
        imported = indicators.import_indicators(
            connection,
            h2_indicator_path,
        )
        assert imported["canonical_indicator_families"] == 3
        assert imported["unprocessed_included_sources"] == []
        assert imported["sources_without_h1_review"] == []
        assert imported["sources_without_h2_review"] == []
        assert imported["retained_mentions_without_h2_review"] == []
        assert connection.execute(
            "SELECT COUNT(*) FROM indicator_source_reviews"
        ).fetchone()[0] == 4
        assert connection.execute(
            "SELECT COUNT(*) FROM indicator_mention_reviews"
        ).fetchone()[0] == 10
        fulltext_snapshots = connection.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT sha256)
            FROM source_snapshots
            WHERE role = 'indicator_fulltext_evidence'
            """
        ).fetchone()
        assert tuple(fulltext_snapshots) == (5, 1)
        family_rows = connection.execute(
            """
            SELECT feature_id, canonical_name_en
            FROM indicator_families ORDER BY feature_id
            """
        ).fetchall()
        atypical_family = connection.execute(
            """
            SELECT formula, formula_reproducible, h2_approved
            FROM indicator_families
            WHERE canonical_name_en = 'Atypical combination share'
            """
        ).fetchone()
        assert tuple(atypical_family) == (
            "x_i = count_i / total_i",
            1,
            1,
        )
        derivation_artifact = temporary_path / "derive_features.py"
        derivation_artifact.write_text(
            "def derive_feature(rows): return rows\n",
            encoding="utf-8",
        )
        input_snapshot = temporary_path / "input_snapshot.json"
        input_snapshot.write_text(
            '{"snapshot": "frozen-test-input"}\n',
            encoding="utf-8",
        )
        data_audit_rows = [
            {
                "feature_id": family["feature_id"],
                "canonical_name_en": family["canonical_name_en"],
                "data_status": "materialized_audited",
                "row_count": "100",
                "valid_count": "100",
                "unique_count": "10",
                "missing_rate": "0",
                "derivation_artifact_path": str(derivation_artifact),
                "input_snapshot_path": str(input_snapshot),
                "derivation_hash": "",
                "input_snapshot_hash": "",
                "audit_status": "pass",
                "reviewer": "H1|H2",
                "notes": "Recomputed against the frozen test snapshot",
            }
            for family in family_rows
        ]
        data_audit_path = temporary_path / "data_audit.csv"
        write_csv(
            data_audit_path,
            data_audit_rows,
            indicators.DATA_AUDIT_FIELDS,
        )
        assert indicators.import_feature_data_audit(
            connection,
            data_audit_path,
        ) == 3
        artifact_snapshots = connection.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT sha256)
            FROM source_snapshots
            WHERE role IN (
                'feature_derivation_artifact',
                'feature_input_snapshot'
            )
            """
        ).fetchone()
        assert tuple(artifact_snapshots) == (6, 2)
        dimension_rows: List[Dict[str, Any]] = []
        for family in family_rows:
            future = "Future" in family["canonical_name_en"]
            for role in ("AI", "H1", "H2"):
                dimension_rows.append(
                    {
                        "feature_id": family["feature_id"],
                        "canonical_name_en": family["canonical_name_en"],
                        "coder_role": role,
                        "dimension_label": (
                            "Post-publication outcomes"
                            if future
                            else "Recombinational distinctiveness"
                        ),
                        "dimension_definition": (
                            "Future outcomes"
                            if future
                            else "Departure of cited knowledge combinations"
                        ),
                        "construct_role": (
                            "t0_potential"
                            if future
                            else "substantive_innovation"
                        ),
                        "information_source": (
                            "future citations"
                            if future
                            else "reference combinations"
                        ),
                        "t0_boundary": "T0",
                        "bias_risk": "field normalization",
                        "decision": "include",
                        "reason": "H2-approved construct boundary",
                    }
                )
        h1_dimension_path = temporary_path / "dimension_h1_blind.csv"
        assert indicators.export_dimension_coding(
            connection,
            h1_dimension_path,
            "H1",
        ) == 3
        with h1_dimension_path.open(
            encoding="utf-8",
            newline="",
        ) as handle:
            h1_reader = csv.DictReader(handle)
            assert not any(
                field.startswith(("ai_", "h2_"))
                for field in (h1_reader.fieldnames or [])
            )
            h1_evidence = next(h1_reader)
        assert h1_evidence["formula"]
        assert h1_evidence["mention_ids_evidence"]
        for role in ("AI", "H1"):
            role_path = temporary_path / f"dimensions_{role}.csv"
            write_csv(
                role_path,
                [
                    row
                    for row in dimension_rows
                    if row["coder_role"] == role
                ],
                indicators.DIMENSION_FIELDS,
            )
            assert indicators.import_dimension_coding(
                connection,
                role_path,
            ) == 3
        h2_dimension_path = temporary_path / "dimension_h2.csv"
        assert indicators.export_dimension_coding(
            connection,
            h2_dimension_path,
            "H2",
        ) == 3
        with h2_dimension_path.open(
            encoding="utf-8",
            newline="",
        ) as handle:
            h2_dimension = next(csv.DictReader(handle))
        assert h2_dimension["ai_decision"] == "include"
        assert h2_dimension["h1_decision"] == "include"
        completed_h2_dimension_path = (
            temporary_path / "dimensions_H2_completed.csv"
        )
        write_csv(
            completed_h2_dimension_path,
            [
                row
                for row in dimension_rows
                if row["coder_role"] == "H2"
            ],
            indicators.DIMENSION_FIELDS,
        )
        assert indicators.import_dimension_coding(
            connection,
            completed_h2_dimension_path,
        ) == 3
        derived = indicators.derive_dimensions(connection)
        assert derived["M"] == 2
        original_output = indicators.OUTPUT_DIR
        indicators.OUTPUT_DIR = temporary_path / "outputs"
        try:
            selected = indicators.select_indicators(connection)
        finally:
            indicators.OUTPUT_DIR = original_output
        assert selected["D"] == 1
        assert selected["F"] == 1
        assert selected["no_quota_applied"] is True
        failures = {
            row["feature_id"]: json.loads(row["failed_gates_json"])
            for row in connection.execute(
                "SELECT * FROM feature_decisions"
            )
        }
        future_id = next(
            row["feature_id"]
            for row in family_rows
            if "Future" in row["canonical_name_en"]
        )
        assert "G06_NO_FUTURE_INFORMATION" in failures[future_id]
        eliminated = connection.execute(
            """
            SELECT COUNT(*) FROM candidate_dimensions
            WHERE status = 'eliminated'
            """
        ).fetchone()[0]
        assert eliminated == 1
        connection.close()


def test_audit_always_discloses_english_language_bias() -> None:
    """Even an incomplete run emits an explicit language-bias disclosure."""
    with tempfile.TemporaryDirectory() as temporary:
        temporary_path = Path(temporary)
        connection = initialize(temporary_path / "audit.sqlite3")
        original_output = reporting.OUTPUT_DIR
        reporting.OUTPUT_DIR = temporary_path / "outputs"
        try:
            summary = reporting.audit(connection)
            report = (
                reporting.OUTPUT_DIR / "audit_report_v3.md"
            ).read_text(encoding="utf-8")
            with (
                reporting.OUTPUT_DIR / "completion_matrix_v3.csv"
            ).open(encoding="utf-8") as handle:
                completion_rows = list(csv.DictReader(handle))
        finally:
            reporting.OUTPUT_DIR = original_output
        assert not summary["formal_review_complete"]
        assert summary["language_geographic_bias_disclosed"]
        assert "language and geographic coverage bias" in report
        assert len(completion_rows) == 16
        assert completion_rows[0]["requirement_id"] == "R01"
        assert completion_rows[-1]["requirement_id"] == "R16"
        connection.close()


def test_saturation_frame_is_deterministic_and_domain_free() -> None:
    """Frozen strata are reproducible and do not predefine search domains."""
    first = saturation._query_rows()
    second = saturation._query_rows()
    assert first == second
    roles = {
        role: sum(row["query_role"] == role for row in first)
        for role in {
            "base_year_type",
            "target_oversample",
            "evidence_oversample",
            "development_formula_oversample",
        }
    }
    assert roles == {
        "base_year_type": 14,
        "target_oversample": 12,
        "evidence_oversample": 16,
        "development_formula_oversample": 59,
    }
    assert len(first) == 101
    serialized = json.dumps(first, ensure_ascii=False)
    assert "D01_" not in serialized
    assert "D12_" not in serialized
    assert all(int(row["random_seed"]) > 0 for row in first)
    assert all(row["query_hash"] for row in first)


def test_two_key_scheduler_rotates_and_skips_failed_slot() -> None:
    """A failed free-key slot cannot trap retries on the same key."""
    scheduler = object.__new__(saturation.OpenAlexBudgetScheduler)
    scheduler.keys = ["slot-one", "slot-two"]
    scheduler.remaining = {0: 0.5, 1: 0.5}
    scheduler.resets_at = {0: "reset", 1: "reset"}
    scheduler.next_slot = 0
    assert scheduler._available_slot(0.001) == 0
    assert scheduler._available_slot(0.001) == 1
    scheduler.next_slot = 0
    assert scheduler._available_slot(0.001, {0}) == 1


def test_discovery_freeze_requires_three_consecutive_dual_zero_rounds() -> None:
    """H2 cannot stop after one convenient zero-novelty batch."""
    with tempfile.TemporaryDirectory() as temporary:
        connection = initialize(Path(temporary) / "saturation.sqlite3")
        for iteration in range(1, 4):
            connection.execute(
                """
                INSERT INTO discovery_review_rounds(
                    iteration, batch_first_rank, batch_last_rank,
                    assigned_records, fully_reviewed,
                    new_nonredundant_english_terms,
                    new_canonical_indicator_families,
                    consecutive_zero_rounds, reviewer_role, decision,
                    notes, reviewed_at
                ) VALUES (?, ?, ?, 10, 1, -1, -1, 0, 'SYSTEM',
                          'pending', '', '2026-07-29T00:00:00+00:00')
                """,
                (iteration, iteration, iteration),
            )
        for iteration in (1, 2):
            result = saturation.record_discovery_saturation(
                connection,
                iteration,
                new_terms=0,
                new_indicator_families=0,
                decision="continue",
                notes=f"Independent H2 review round {iteration}",
            )
            assert result["consecutive_zero_rounds"] == iteration
        try:
            saturation.record_discovery_saturation(
                connection,
                2,
                new_terms=0,
                new_indicator_families=0,
                decision="freeze",
                notes="Too early",
            )
        except ValueError as error:
            assert "3 consecutive" in str(error)
        else:
            raise AssertionError("Two zero rounds must not permit freezing")
        result = saturation.record_discovery_saturation(
            connection,
            3,
            new_terms=0,
            new_indicator_families=0,
            decision="freeze",
            notes="Third consecutive dual-zero round verified by H2",
        )
        assert result["consecutive_zero_rounds"] == 3
        assert result["decision"] == "freeze"
        connection.execute(
            """
            INSERT INTO discovery_review_rounds(
                iteration, saturation_phase, batch_first_rank,
                batch_last_rank, assigned_records, fully_reviewed,
                new_nonredundant_english_terms,
                new_canonical_indicator_families,
                consecutive_zero_rounds, reviewer_role, decision,
                notes, reviewed_at
            ) VALUES (
                4, 'formal_indicator_discovery', 4, 4, 10, 1,
                -1, -1, 0, 'SYSTEM', 'pending', '',
                '2026-07-29T00:00:00+00:00'
            )
            """
        )
        phase_reset = saturation.record_discovery_saturation(
            connection,
            4,
            new_terms=0,
            new_indicator_families=0,
            decision="continue",
            notes="First dual-zero formal-search round",
        )
        assert phase_reset["saturation_phase"] == (
            "formal_indicator_discovery"
        )
        assert phase_reset["consecutive_zero_rounds"] == 1
        connection.close()


def test_h1_blinding_and_h2_comparison_export() -> None:
    """H1 never sees AI codes; H2 sees both only after independent coding."""
    with tempfile.TemporaryDirectory() as temporary:
        temporary_path = Path(temporary)
        connection = initialize(temporary_path / "handoff.sqlite3")
        record = _record("WBLIND", "10.4000/blind")
        _insert_record(connection, record)
        connection.execute(
            """
            INSERT INTO discovery_queries(
                discovery_query_id, query_role, stratum_label, expression,
                filter_expression, sample_size, random_seed, query_hash,
                status, archive_reason
            ) VALUES ('DS_TEST', 'base_year_type', 'test', 'test', '',
                      1, 1, 'hash', 'active', '')
            """
        )
        connection.execute(
            """
            INSERT INTO discovery_hits(
                discovery_query_id, record_key, sample_rank,
                selection_hash, review_rank, review_round, review_status
            ) VALUES ('DS_TEST', ?, 1, 'selection', 1, 1, 'unassigned')
            """,
            (record["record_key"],),
        )
        h1_path = temporary_path / "h1.csv"
        assert (
            saturation.export_discovery_screening(
                connection, 1, "H1", h1_path
            )
            == 1
        )
        with h1_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            assert "ai_decision" not in (reader.fieldnames or [])
        ai_path = temporary_path / "primary_ai.csv"
        write_csv(
            ai_path,
            [_screen_row(record, "AI", "en", "include")],
            screening.SCREENING_FIELDS,
        )
        h1_decision_path = temporary_path / "primary_h1.csv"
        write_csv(
            h1_decision_path,
            [
                _screen_row(
                    record,
                    "H1",
                    "en",
                    "exclude",
                    "E_NOT_INDICATOR_PREDICTOR_VALIDATION",
                )
            ],
            screening.SCREENING_FIELDS,
        )
        assert screening.import_screening(connection, ai_path) == 1
        assert screening.import_screening(
            connection,
            h1_decision_path,
        ) == 1
        h2_path = temporary_path / "h2.csv"
        assert (
            saturation.export_discovery_screening(
                connection, 1, "H2", h2_path
            )
            == 1
        )
        with h2_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert rows[0]["ai_decision"] == "include"
        assert rows[0]["h1_decision"] == "exclude"
        assert rows[0]["h2_review_reason"] == "AI_H1_DISAGREEMENT"
        connection.close()


def test_discovery_extraction_requires_explicit_no_item_disposition() -> None:
    """A blank worksheet cannot be mistaken for a completed zero-novelty row."""
    with tempfile.TemporaryDirectory() as temporary:
        temporary_path = Path(temporary)
        connection = initialize(temporary_path / "extract.sqlite3")
        record = _record("WEXTRACT", "10.4000/extract")
        _insert_record(connection, record)
        connection.execute(
            """
            INSERT INTO discovery_queries(
                discovery_query_id, query_role, stratum_label, expression,
                filter_expression, sample_size, random_seed, query_hash,
                status, archive_reason
            ) VALUES ('DS_EXTRACT', 'base_year_type', 'test', 'test', '',
                      1, 1, 'hash', 'active', '')
            """
        )
        connection.execute(
            """
            INSERT INTO discovery_hits(
                discovery_query_id, record_key, sample_rank,
                selection_hash, review_rank, review_round, review_status
            ) VALUES ('DS_EXTRACT', ?, 1, 'selection', 1, 1, 'include')
            """,
            (record["record_key"],),
        )
        connection.execute(
            """
            INSERT INTO discovery_review_rounds(
                iteration, batch_first_rank, batch_last_rank,
                assigned_records, fully_reviewed,
                new_nonredundant_english_terms,
                new_canonical_indicator_families,
                consecutive_zero_rounds, reviewer_role, decision,
                notes, reviewed_at
            ) VALUES (1, 1, 1, 1, 1, -1, -1, 0, 'SYSTEM',
                      'pending', '', '2026-07-29T00:00:00+00:00')
            """
        )
        blank_path = temporary_path / "blank.csv"
        blank_row = {
            field: "" for field in saturation.DISCOVERY_EXTRACTION_FIELDS
        }
        blank_row.update(
            {
                "record_key": record["record_key"],
                "review_round": 1,
                "extractor_role": "H1",
                "record_extraction_complete": "true",
                "no_relevant_items": "false",
            }
        )
        write_csv(
            blank_path,
            [blank_row],
            saturation.DISCOVERY_EXTRACTION_FIELDS,
        )
        try:
            saturation.import_discovery_extraction(
                connection,
                blank_path,
            )
        except ValueError as error:
            assert "no_relevant_items=true" in str(error)
        else:
            raise AssertionError("An unmarked blank extraction must fail")
        blank_row["no_relevant_items"] = "true"
        completed_path = temporary_path / "completed.csv"
        write_csv(
            completed_path,
            [blank_row],
            saturation.DISCOVERY_EXTRACTION_FIELDS,
        )
        result = saturation.import_discovery_extraction(
            connection,
            completed_path,
        )
        assert result["completed_record_reviews"] == 1
        novelty = saturation.discovery_novelty_counts(connection, 1)
        assert novelty["new_terms"] == 0
        assert novelty["new_indicator_families"] == 0
        connection.close()


def test_frozen_frame_separates_new_terms_from_indicator_names() -> None:
    """Indicator saturation continues, but new search terms require reopen."""
    with tempfile.TemporaryDirectory() as temporary:
        temporary_path = Path(temporary)
        connection = initialize(temporary_path / "frozen_extract.sqlite3")
        record = _record("WFROZENEXTRACT", "10.4000/frozen-extract")
        _insert_record(connection, record)
        connection.execute(
            """
            INSERT INTO discovery_queries(
                discovery_query_id, query_role, stratum_label, expression,
                filter_expression, sample_size, random_seed, query_hash,
                status, archive_reason
            ) VALUES ('FS_TEST', 'formal_search_family', 'test', 'test', '',
                      1, 1, 'hash', 'active', '')
            """
        )
        connection.execute(
            """
            INSERT INTO discovery_hits(
                discovery_query_id, record_key, sample_rank,
                selection_hash, review_rank, review_round, review_status
            ) VALUES ('FS_TEST', ?, 1, 'selection', 1, 1, 'include')
            """,
            (record["record_key"],),
        )
        connection.execute(
            """
            INSERT INTO metadata(key, value)
            VALUES ('search_frame_frozen_hash', 'frozen-test-hash')
            """
        )
        set_stage(connection, "search_frame_frozen", "complete")
        connection.commit()
        indicator_row = {
            field: ""
            for field in saturation.DISCOVERY_EXTRACTION_FIELDS
        }
        indicator_row.update(
            {
                "record_key": record["record_key"],
                "review_round": "1",
                "item_type": "indicator_candidate",
                "verbatim_name": "novelty metric",
                "location": "abstract",
                "evidence_span": "article-level novelty metric",
                "proposed_role": "indicator_or_measure",
                "extractor_role": "H1",
                "h1_decision": "include",
                "h2_decision": "pending",
                "record_extraction_complete": "true",
                "no_relevant_items": "false",
                "review_notes": "H1 extracted the verbatim name.",
            }
        )
        indicator_path = temporary_path / "indicator_name.csv"
        write_csv(
            indicator_path,
            [indicator_row],
            saturation.DISCOVERY_EXTRACTION_FIELDS,
        )
        result = saturation.import_discovery_extraction(
            connection,
            indicator_path,
        )
        assert result["indicator_candidates"] == 1
        assert connection.execute(
            """
            SELECT status FROM stage_status
            WHERE stage = 'search_frame_frozen'
            """
        ).fetchone()[0] == "complete"
        term_row = dict(indicator_row)
        term_row.update(
            {
                "item_type": "term",
                "verbatim_name": "article-level novelty",
                "evidence_span": "article-level novelty metric",
                "proposed_role": "construct_term",
            }
        )
        term_path = temporary_path / "new_term.csv"
        write_csv(
            term_path,
            [term_row],
            saturation.DISCOVERY_EXTRACTION_FIELDS,
        )
        try:
            saturation.import_discovery_extraction(
                connection,
                term_path,
            )
        except RuntimeError as error:
            assert "reopen-search-frame" in str(error)
            connection.rollback()
        else:
            raise AssertionError(
                "A frozen search frame accepted a new search term"
            )
        assert connection.execute(
            "SELECT COUNT(*) FROM raw_terms"
        ).fetchone()[0] == 0
        connection.close()


def test_formal_queries_register_deterministic_saturation_pools() -> None:
    """Formal search uses frozen per-query pools, not cursoring every hit."""
    with tempfile.TemporaryDirectory() as temporary:
        connection = initialize(Path(temporary) / "formal_pool.sqlite3")
        connection.execute(
            """
            INSERT INTO logical_queries(
                logical_query_id, query_version, search_domain_id,
                family_label, logical_expression, object_terms_json,
                domain_terms_json, context_terms_json, status,
                archive_reason, press_status, query_hash
            ) VALUES (
                'L0001', 1, 'SD001', 'novelty metrics', 'novelty',
                '[]', '["novelty"]', '[]', 'active', '', 'pass',
                'logical_hash'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO physical_queries(
                physical_query_id, logical_query_id, provider, expression,
                filter_expression, status, query_hash
            ) VALUES (
                'L0001__P001', 'L0001', 'OpenAlex', 'novelty',
                'type:article', 'active', 'physical_hash'
            )
            """
        )
        retrieval._register_formal_saturation_pools(
            connection,
            ["L0001__P001"],
        )
        pool = connection.execute(
            """
            SELECT * FROM discovery_queries
            WHERE discovery_query_id = 'FS_V001_L0001__P001'
            """
        ).fetchone()
        assert pool["query_role"] == "formal_search_family"
        assert pool["sample_size"] == 10_000
        assert pool["random_seed"] > 0
        record = _record("WPOOL", "10.4000/pool")
        _insert_record(connection, record)
        connection.execute(
            """
            INSERT INTO discovery_hits(
                discovery_query_id, record_key, sample_rank,
                selection_hash, review_rank, review_round, review_status
            ) VALUES ('FS_V001_L0001__P001', ?, 1, 'selection', 1, 0,
                      'unassigned')
            """,
            (record["record_key"],),
        )
        connection.execute(
            """
            INSERT INTO discovery_query_runs(
                discovery_query_id, query_hash, reported_sample_total,
                retrieved_rows, unique_hits, pages, next_page, complete,
                stopped_reason, error, updated_at
            ) VALUES (
                'FS_V001_L0001__P001', ?, 1, 1, 1, 1, 2, 1,
                'sample_complete', '', '2026-07-29T00:00:00+00:00'
            )
            """,
            (pool["query_hash"],),
        )
        connection.execute(
            """
            INSERT INTO query_runs(
                provider, physical_query_id, run_role, query_hash,
                reported_total, retrieved_rows, unique_hits, pages,
                next_cursor, complete, stopped_reason, error, updated_at
            ) VALUES (
                'OpenAlex', 'L0001__P001',
                'search_frame_validation_inventory', 'physical_hash',
                500000, 0, 0, 1, '', 1, 'inventory_only', '',
                '2026-07-29T00:00:00+00:00'
            )
            """
        )
        retrieval._map_formal_saturation_pools(
            connection,
            ["L0001__P001"],
        )
        run = connection.execute(
            """
            SELECT * FROM query_runs
            WHERE physical_query_id = 'L0001__P001'
              AND run_role = 'formal'
            """
        ).fetchone()
        assert run["reported_total"] == 500_000
        assert run["retrieved_rows"] == 1
        assert run["stopped_reason"] == (
            "deterministic_evidence_saturation_pool"
        )
        connection.execute(
            """
            UPDATE discovery_queries
            SET status = 'archived',
                archive_reason = 'superseded_formal_search_frame'
            WHERE discovery_query_id = 'FS_V001_L0001__P001'
            """
        )
        connection.execute(
            """
            UPDATE logical_queries SET query_version = 2
            WHERE logical_query_id = 'L0001'
            """
        )
        retrieval._register_formal_saturation_pools(
            connection,
            ["L0001__P001"],
        )
        versioned_pools = [
            str(row[0])
            for row in connection.execute(
                """
                SELECT discovery_query_id FROM discovery_queries
                WHERE query_role = 'formal_search_family'
                ORDER BY discovery_query_id
                """
            )
        ]
        assert versioned_pools == [
            "FS_V001_L0001__P001",
            "FS_V002_L0001__P001",
        ]
        connection.close()


def test_formal_phase_uses_offset_ranks_without_refetching() -> None:
    """The first formal round starts at its pool head and resets saturation."""
    with tempfile.TemporaryDirectory() as temporary:
        connection = initialize(Path(temporary) / "formal_round.sqlite3")
        connection.execute(
            """
            INSERT INTO metadata(key, value)
            VALUES ('formal_review_rank_offset', '40')
            """
        )
        connection.execute(
            """
            INSERT INTO discovery_queries(
                discovery_query_id, query_role, stratum_label, expression,
                filter_expression, sample_size, random_seed, query_hash,
                status, archive_reason
            ) VALUES (
                'FS_V001_L0001__P001', 'formal_search_family', 'formal',
                'novelty', 'type:article', 10000, 7, 'pool_hash',
                'active', ''
            )
            """
        )
        for index in range(1, 11):
            record = _record(
                f"WF{index}",
                f"10.4000/formal{index}",
            )
            _insert_record(connection, record)
            connection.execute(
                """
                INSERT INTO discovery_hits(
                    discovery_query_id, record_key, sample_rank,
                    selection_hash, review_rank, review_round, review_status
                ) VALUES (
                    'FS_V001_L0001__P001', ?, ?, ?, ?, 0, 'unassigned'
                )
                """,
                (
                    record["record_key"],
                    index,
                    f"selection-{index}",
                    40 + index,
                ),
            )
        connection.execute(
            """
            INSERT INTO discovery_query_runs(
                discovery_query_id, query_hash, reported_sample_total,
                retrieved_rows, unique_hits, pages, next_page, complete,
                stopped_reason, error, updated_at
            ) VALUES (
                'FS_V001_L0001__P001', 'pool_hash', 10000, 10, 10, 1, 2, 0,
                'review_capacity_loaded', '',
                '2026-07-29T00:00:00+00:00'
            )
            """
        )
        result = saturation.assign_discovery_round(connection, 5)
        assert result["saturation_phase"] == "formal_indicator_discovery"
        assert result["formal_capacity"]["queries_extended"] == 0
        assert result["rank_ranges_by_role"]["formal_search_family"] == [
            41,
            50,
        ]
        assert result["unique_records"] == 10
        connection.close()


def test_discovery_paper_is_never_assigned_to_multiple_rounds() -> None:
    """A paper found in several strata receives one earliest-round review."""
    with tempfile.TemporaryDirectory() as temporary:
        connection = initialize(Path(temporary) / "round_dedup.sqlite3")
        for query_id in ("DS_ONE", "DS_TWO", "DS_THREE"):
            connection.execute(
                """
                INSERT INTO discovery_queries(
                    discovery_query_id, query_role, stratum_label,
                    expression, filter_expression, sample_size,
                    random_seed, query_hash, status, archive_reason
                ) VALUES (?, 'base_year_type', ?, '', '', 1, 7, ?,
                          'active', '')
                """,
                (query_id, query_id, f"hash-{query_id}"),
            )
        record = _record("WROUNDDEDUP", "10.4000/round-dedup")
        _insert_record(connection, record)
        connection.executemany(
            """
            INSERT INTO discovery_hits(
                discovery_query_id, record_key, sample_rank,
                selection_hash, review_rank, review_round, review_status
            ) VALUES (?, ?, ?, ?, ?, ?, 'unassigned')
            """,
            [
                ("DS_ONE", record["record_key"], 1, "s1", 1, 1),
                ("DS_TWO", record["record_key"], 11, "s2", 11, 2),
                ("DS_THREE", record["record_key"], 21, "s3", 21, 0),
            ],
        )
        for iteration in (1, 2):
            connection.execute(
                """
                INSERT INTO discovery_review_rounds(
                    iteration, saturation_phase, batch_first_rank,
                    batch_last_rank, assigned_records, fully_reviewed,
                    new_nonredundant_english_terms,
                    new_canonical_indicator_families,
                    consecutive_zero_rounds, reviewer_role, decision,
                    notes, reviewed_at
                ) VALUES (?, 'search_frame_discovery', ?, ?, 1, 0,
                          -1, -1, 0, 'SYSTEM', 'pending', '',
                          '2026-07-29T00:00:00+00:00')
                """,
                (iteration, (iteration - 1) * 10 + 1, iteration * 10),
            )
        result = saturation.assign_discovery_round(connection, 3)
        assert result["cross_round_duplicates_normalized"] == {
            "records": 1,
            "hits": 1,
        }
        assigned_rounds = connection.execute(
            """
            SELECT discovery_query_id, review_round, review_status
            FROM discovery_hits
            ORDER BY discovery_query_id
            """
        ).fetchall()
        assert [
            (row["discovery_query_id"], row["review_round"])
            for row in assigned_rounds
        ] == [
            ("DS_ONE", 1),
            ("DS_THREE", 0),
            ("DS_TWO", 0),
        ]
        assert assigned_rounds[2]["review_status"] == (
            "duplicate_prior_round"
        )
        assert result["unique_records"] == 0
        connection.close()


def test_citation_network_is_a_pool_not_an_automatic_screening_census() -> None:
    """Only citation records assigned to a saturation round need screening."""
    with tempfile.TemporaryDirectory() as temporary:
        connection = initialize(Path(temporary) / "citation_pool.sqlite3")
        connection.execute(
            """
            INSERT INTO discovery_queries(
                discovery_query_id, query_role, stratum_label, expression,
                filter_expression, sample_size, random_seed, query_hash,
                status, archive_reason
            ) VALUES (
                'DS_CITATIONS', 'development_citation_network', 'citations',
                '', '', 2, 7, 'citation-hash', 'network', ''
            )
            """
        )
        for index, review_round in ((1, 1), (2, 0)):
            record = _record(
                f"WCITATION{index}",
                f"10.4000/citation-pool-{index}",
            )
            _insert_record(connection, record)
            connection.execute(
                """
                INSERT INTO discovery_hits(
                    discovery_query_id, record_key, sample_rank,
                    selection_hash, review_rank, review_round, review_status
                ) VALUES (
                    'DS_CITATIONS', ?, ?, ?, ?, ?, 'unassigned'
                )
                """,
                (
                    record["record_key"],
                    index,
                    f"selection-{index}",
                    index,
                    review_round,
                ),
            )
        assert screening._formal_record_count(connection) == 1
        assert list(pipeline._formal_record_keys(connection)) == [
            "doi:10.4000/citation-pool-1"
        ]
        unassigned_key = connection.execute(
            """
            SELECT record_key FROM discovery_hits
            WHERE review_round = 0
            """
        ).fetchone()[0]
        connection.execute(
            """
            INSERT INTO crossref_validation(
                record_key, doi, status, conflict_reason, validated_at
            ) VALUES (?, '10.4000/citation-pool-2', 'conflict',
                      'test-only', '2026-07-29T00:00:00+00:00')
            """,
            (unassigned_key,),
        )
        blockers = reporting._completion_blockers(
            connection,
            reporting._stage_summary(connection),
        )
        assert not any(
            value.startswith("UNRESOLVED_CROSSREF_CONFLICTS")
            for value in blockers
        )
        assigned_key = connection.execute(
            """
            SELECT record_key FROM discovery_hits
            WHERE review_round = 1
            """
        ).fetchone()[0]
        connection.execute(
            """
            INSERT INTO crossref_validation(
                record_key, doi, status, conflict_reason, validated_at
            ) VALUES (?, '10.4000/citation-pool-1', 'validated', '',
                      '2026-07-29T00:00:00+00:00')
            """,
            (assigned_key,),
        )
        crossref_actions: List[Dict[str, Any]] = []
        handoff._crossref_actions(
            connection,
            force=True,
            actions=crossref_actions,
        )
        assert crossref_actions == []
        connection.close()


def test_h1_review_helper_is_blind_and_validates_exact_spans() -> None:
    """The local helper rejects AI columns and checks source provenance."""
    with tempfile.TemporaryDirectory() as temporary:
        fields = [
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
            "review_round",
            "discovery_query_ids",
        ]
        row = {
            field: ""
            for field in fields
        }
        row.update(
            {
                "record_key": "doi:10.4000/blind",
                "doi": "10.4000/blind",
                "title": "English novelty metric validation",
                "abstract": "We validate a publication-time indicator.",
                "reviewer_role": "H1",
                "language_judgment": "en",
                "language_evidence": "English novelty metric validation",
                "decision": "include",
                "evidence_span": (
                    "We validate a publication-time indicator."
                ),
            }
        )
        valid_path = Path(temporary) / "h1.csv"
        write_csv(valid_path, [row], fields)
        assert human_review_cli.validate_worksheet(valid_path) == {
            "rows": 1,
            "completed": 1,
            "errors": 0,
        }
        leaked_path = Path(temporary) / "h1_leaked.csv"
        leaked_fields = [*fields, "ai_decision"]
        write_csv(
            leaked_path,
            [{**row, "ai_decision": "exclude"}],
            leaked_fields,
        )
        try:
            human_review_cli.worksheet_status(leaked_path)
        except ValueError as error:
            assert "refuses comparison columns" in str(error)
        else:
            raise AssertionError("Blind H1 helper accepted an AI column")
        term_fields = [
            "term_id",
            "verbatim_term",
            "source_type",
            "coder_role",
            "canonical_term",
            "term_family_label",
            "term_relation",
            "search_domain_label",
            "search_domain_definition",
            "query_family_label",
            "cross_domain",
            "decision",
            "reason",
            "source_id",
            "source_location",
            "source_evidence_span",
            "proposed_role",
        ]
        term_row = {
            "term_id": "TERM_BLIND",
            "verbatim_term": "atypical combination",
            "source_type": "bootstrap",
            "coder_role": "H1",
            "canonical_term": "atypical combination",
            "term_family_label": "knowledge recombination novelty",
            "term_relation": "canonical",
            "search_domain_label": "knowledge recombination",
            "search_domain_definition": (
                "Novel combinations of prior knowledge elements"
            ),
            "query_family_label": "recombination novelty measures",
            "cross_domain": "false",
            "decision": "include",
            "reason": "The source defines a publication-level measure.",
            "source_id": "10.4000/source",
            "source_location": "abstract",
            "source_evidence_span": (
                "We quantify atypical combination in scientific papers."
            ),
            "proposed_role": "indicator_or_measure",
        }
        term_path = Path(temporary) / "h1_terms.csv"
        write_csv(term_path, [term_row], term_fields)
        assert human_review_cli.validate_term_worksheet(term_path) == {
            "rows": 1,
            "completed": 1,
            "errors": 0,
        }
        assert human_review_cli.term_worksheet_status(term_path) == {
            "rows": 1,
            "blank": 0,
            "include": 1,
            "exclude": 0,
        }
        term_leak_path = Path(temporary) / "h1_terms_leaked.csv"
        write_csv(
            term_leak_path,
            [{**term_row, "ai_search_domain_label": "leaked"}],
            [*term_fields, "ai_search_domain_label"],
        )
        try:
            human_review_cli.term_worksheet_status(term_leak_path)
        except ValueError as error:
            assert "refuses comparison columns" in str(error)
        else:
            raise AssertionError(
                "Blind H1 term helper accepted an AI column"
            )
        try:
            handoff._write_once(
                valid_path,
                lambda _: 1,
                force=True,
            )
        except RuntimeError as error:
            assert "Refusing to overwrite" in str(error)
        else:
            raise AssertionError(
                "Force handoff generation overwrote completed human work"
            )
        blank_path = Path(temporary) / "blank_h1.csv"
        blank_row = dict(row)
        for field in (
            "language_judgment",
            "language_evidence",
            "decision",
            "exclusion_reason",
            "evidence_span",
            "notes",
        ):
            blank_row[field] = ""
        write_csv(blank_path, [blank_row], fields)
        rewritten = handoff._write_once(
            blank_path,
            lambda path: (
                write_csv(path, [blank_row], fields) or 1
            ),
            force=True,
        )
        assert rewritten == 1
        quarantine = (
            Path(temporary) / "invalidated_automated_h1_trial_test"
        )
        quarantine.mkdir()
        quarantined_file = quarantine / "invalid.csv"
        write_csv(quarantined_file, [row], fields)
        connection = initialize(Path(temporary) / "import_guard.sqlite3")
        try:
            snapshot_import_file(
                connection,
                quarantined_file,
                "screening",
            )
        except RuntimeError as error:
            assert "quarantine cannot be imported" in str(error)
        else:
            raise AssertionError(
                "Invalidated automated-human quarantine was importable"
            )
        unreviewed_h2 = (
            ROOT
            / "outputs"
            / "unreviewed_automated_h2_drafts_20260729"
            / "round_01_screening_H2_DRAFT.csv"
        )
        try:
            snapshot_import_file(
                connection,
                unreviewed_h2,
                "screening_H2",
            )
        except RuntimeError as error:
            assert "unreviewed automated-human quarantine" in str(error)
        else:
            raise AssertionError(
                "Unreviewed H2 draft was importable from quarantine"
            )
        copied_unreviewed = Path(temporary) / "copied_h2_draft.csv"
        shutil.copyfile(unreviewed_h2, copied_unreviewed)
        try:
            snapshot_import_file(
                connection,
                copied_unreviewed,
                "screening_H2",
            )
        except RuntimeError as error:
            assert "unreviewed automated H2 draft" in str(error)
        else:
            raise AssertionError(
                "Unreviewed H2 draft hash was importable outside quarantine"
            )
        attested_file = (
            ROOT
            / "outputs"
            / "human_attested_automated_drafts_20260729"
            / "round_01_screening_H1_HUMAN_REVIEWED.csv"
        )
        snapshot_import_file(
            connection,
            attested_file,
            "screening_H1",
        )
        attested_role = str(
            connection.execute(
                """
                SELECT role FROM source_snapshots
                WHERE sha256 = ?
                """,
                (sha256_file(attested_file),),
            ).fetchone()["role"]
        )
        assert attested_role == (
            "review_import_human_attested_automated_draft:screening_H1"
        )
        connection.close()


def test_citation_tracking_covers_reviews_and_indicator_sources() -> None:
    """Both required source classes route all eligible edges to screening."""
    with tempfile.TemporaryDirectory() as temporary:
        connection = initialize(Path(temporary) / "citations.sqlite3")
        review = _record("W1001", "10.4000/review-source")
        review["work_type"] = "review"
        review["referenced_works_json"] = json.dumps(
            [
                "https://openalex.org/W1003",
                "https://openalex.org/W1004",
            ]
        )
        indicator = _record("W1002", "10.4000/indicator-source")
        local_backward = _record("W1004", "10.4000/local-backward")
        _insert_record(connection, review)
        _insert_record(connection, indicator)
        _insert_record(connection, local_backward)
        connection.execute(
            """
            INSERT INTO screening_final(
                record_key, final_language, final_decision,
                exclusion_reason, h2_required, h2_completed,
                adjudication_reason, finalized_at
            ) VALUES (?, 'en', 'include', '', 1, 1, 'H2', ?)
            """,
            (
                review["record_key"],
                "2026-07-29T00:00:00+00:00",
            ),
        )
        connection.execute(
            """
            INSERT INTO indicator_source_disposition(
                record_key, disposition, english_fulltext_status,
                notes, decided_by, decided_at
            ) VALUES (?, 'extracted', 'verified',
                      'Formula source', 'H2', ?)
            """,
            (
                indicator["record_key"],
                "2026-07-29T00:00:00+00:00",
            ),
        )
        for reviewer_role in ("H1", "H2"):
            connection.execute(
                """
                INSERT INTO indicator_source_reviews(
                    record_key, reviewer_role, disposition,
                    english_fulltext_status, notes, reviewed_at
                ) VALUES (?, ?, 'extracted', 'verified',
                          'Formula source', ?)
                """,
                (
                    indicator["record_key"],
                    reviewer_role,
                    "2026-07-29T00:00:00+00:00",
                ),
            )
        set_stage(connection, "literature_screened", "complete")
        connection.commit()

        def work(
            identity: str,
            doi: str,
            title: str,
            references: List[str] | None = None,
        ) -> Dict[str, Any]:
            return {
                "id": f"https://openalex.org/{identity}",
                "doi": f"https://doi.org/{doi}",
                "display_name": title,
                "publication_year": 2020,
                "type": "article",
                "language": "en",
                "abstract_inverted_index": {},
                "primary_location": {},
                "referenced_works": references or [],
            }

        def fake_fetch(url: str) -> Dict[str, Any]:
            parameters = urllib.parse.parse_qs(
                urllib.parse.urlparse(url).query
            )
            filter_value = parameters["filter"][0]
            if filter_value.startswith("ids.openalex:"):
                return {
                    "meta": {"count": 1, "next_cursor": None},
                    "results": [
                        work(
                            "W1003",
                            "10.4000/backward",
                            "Backward evidence",
                        )
                    ],
                }
            return {
                "meta": {"count": 2, "next_cursor": None},
                "results": [
                    work(
                        "W2001",
                        "10.4000/forward-wreview",
                        "Forward evidence for W1001",
                        ["https://openalex.org/W1001"],
                    ),
                    work(
                        "W2002",
                        "10.4000/forward-windicator",
                        "Forward evidence for W1002",
                        ["https://openalex.org/W1002"],
                    ),
                ],
            }

        details = retrieval.track_citations(
            connection,
            iteration=1,
            scope="reviews_and_indicator_sources",
            fetcher=fake_fetch,
        )
        assert details["citation_sources"] == 2
        assert details["backward_records_in_scope"] == 2
        assert details["backward_records_reused_locally"] == 1
        assert details["forward_records_seen"] == 2
        assert details["citation_review_network_records"] == 4
        assert details["citation_physical_queries"] == 2
        assert details["citation_api_pages"] == 2
        statuses = {
            str(row[0])
            for row in connection.execute(
                "SELECT eligibility_status FROM citation_edges"
            )
        }
        assert statuses == {"pending_screening"}
        query = connection.execute(
            """
            SELECT query_role, sample_size FROM discovery_queries
            WHERE discovery_query_id =
                'CS_REVIEWS_AND_INDICATOR_SOURCES_ITER_001'
            """
        ).fetchone()
        assert tuple(query) == ("citation_tracking_network", 4)
        connection.close()


def test_registered_source_hash_cannot_be_rebased_after_freeze() -> None:
    """Rerunning init cannot hide a post-freeze protocol/code mutation."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        connection = initialize(root / "frozen_source.sqlite3")
        source = root / "protocol.txt"
        source.write_text("version one", encoding="utf-8")
        original_digest = sha256_file(source)
        coding._register_snapshot(
            connection,
            "test_frozen_source",
            source,
            "frozen_protocol",
        )
        connection.execute(
            """
            INSERT INTO metadata(key, value)
            VALUES ('search_frame_frozen_hash', 'frozen-test-hash')
            """
        )
        connection.commit()
        source.write_text("version two", encoding="utf-8")
        try:
            coding._register_snapshot(
                connection,
                "test_frozen_source",
                source,
                "frozen_protocol",
            )
        except RuntimeError as error:
            assert "changed after search-frame freeze" in str(error)
        else:
            raise AssertionError(
                "A post-freeze source hash was silently rebased"
            )
        stored = connection.execute(
            """
            SELECT sha256 FROM source_snapshots
            WHERE source_id = 'test_frozen_source'
            """
        ).fetchone()[0]
        assert stored == original_digest
        assert stored != sha256_file(source)
        connection.close()


def main() -> None:
    """Run all lightweight offline v3 checks."""
    test_init_is_independent_and_v2_read_only()
    test_openalex_cursor_resume_and_deduplication()
    test_hidden_seed_search_log_requires_all_independent_routes()
    test_pilot_terms_cannot_create_a_logical_query_family()
    test_k_q_p_are_derived_and_seed_recall_is_validated()
    test_logical_redundancy_uses_complete_result_sets()
    test_screening_requires_h2_and_retains_language_exclusion()
    test_ai_language_evidence_normalization_is_exact_and_audited()
    test_crossref_conflicts_require_h2_resolution()
    test_open_fulltext_acquisition_is_lawful_hashed_and_resumable()
    test_indicator_gates_redundancy_and_dimension_retention()
    test_audit_always_discloses_english_language_bias()
    test_saturation_frame_is_deterministic_and_domain_free()
    test_two_key_scheduler_rotates_and_skips_failed_slot()
    test_discovery_freeze_requires_three_consecutive_dual_zero_rounds()
    test_h1_blinding_and_h2_comparison_export()
    test_discovery_extraction_requires_explicit_no_item_disposition()
    test_frozen_frame_separates_new_terms_from_indicator_names()
    test_formal_queries_register_deterministic_saturation_pools()
    test_formal_phase_uses_offset_ranks_without_refetching()
    test_discovery_paper_is_never_assigned_to_multiple_rounds()
    test_citation_network_is_a_pool_not_an_automatic_screening_census()
    test_h1_review_helper_is_blind_and_validates_exact_spans()
    test_citation_tracking_covers_reviews_and_indicator_sources()
    test_registered_source_hash_cannot_be_rebased_after_freeze()
    print("All evidence-derived v3 tests passed.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from common import DATABASE_PATH, sha256_file, utc_now


STAGES = (
    "initialized",
    "bootstrap_inventory_complete",
    "bootstrap_retrieval_complete",
    "terms_coded",
    "search_frame_derived",
    "search_frame_validated",
    "search_frame_frozen",
    "formal_retrieval_complete",
    "literature_screened",
    "indicators_extracted",
    "dimensions_derived",
    "features_selected",
    "audit_complete",
)
INVALIDATED_AUTOMATED_HUMAN_HASHES = frozenset(
    {
        # First, unreviewed automated trial. These decision files remain
        # quarantined and can never be imported as H1/H2.
        "218c6f7ee029c710c1e31a3c13af5623d9daa92e076355878856f3042ac3887d",
        "3be75589f7d49ea2144273241a24700ae65eaa5cabf8f797d764dd12edb1361f",
        "ce4f9a4f16299f80a6e1bf1675f5166a6dbcca41bbc130814e814fded4c9159b",
        "be4c983375dadf7f1b8d633eb638e33c9a6281fe2f3c0ab03c4be36f52b900a9",
        "819a54fa66ea3a193d41a7ca123e01616e9b0acd98e0ef47b9f58d9bfaba6c4d",
    }
)
HUMAN_ATTESTED_AUTOMATED_DRAFT_HASHES = frozenset(
    {
        # Generated as drafts, then manually reviewed and adopted. The
        # project-owner attestation is frozen separately.
        "ba672d8e9c39c4a1932f9945a5efc9051edb12a61a019f56892fc682c0b93caa",
        "0fb05ac2a82bf5420b9db74a67830d705ba0c8de7c377c2663b0c6c729d2149a",
        "cc4b38b04f07ba8441674c565c6f9ce327d5ac3cab1e4c1f1d632f18c350da81",
        "99c8e0b87955f7ba6eec1cfe301c3c44b04305f2e3c710ba280eb56666a48e22",
        "8b601e06c7714bc11ad0173720884d0ecab2a35cf8722ecf2d3477701d453744",
        "f469240b19b6310e80a67fa1b329e6aecb1ee0448d5acb9de86ad69a4bd0ddd4",
        "f1c21dc6d012d9c54f216dcc82410ce05ea8ae156d281887c7773d688c48d566",
    }
)
UNREVIEWED_AUTOMATED_H2_HASHES = frozenset(
    {
        "450dde7b7080184dcd8729693e5458ec0479ede88404c97bf76ac7184f136761",
        "9cf8b16458c940e4d0ab6c7115c24462450bd9e9a71d8a6f891473ea9ac1aab5",
        "c1290a5d1b9a9b7aabd556c1ff26e079d5acd5e5f6bb9687ec14d9e22836ad31",
        "85dba2b5fbb30b6fd8b311cc6fa0c4f1c2036c712e9ea5860e0bbb621973e91b",
        "e9ac93836f5a1a5dca06fb63d20ecd3f6cc75645682aa5fd4ede6d498cb158f7",
        "203656f242fe6374b3a0a4a38c4327d0506128f007aac7dfdb8ca08a7c1a5dd6",
    }
)


SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS stage_status (
    stage TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS source_snapshots (
    source_id TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    role TEXT NOT NULL,
    imported_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence_seeds (
    seed_id TEXT PRIMARY KEY,
    doi TEXT NOT NULL,
    citation TEXT NOT NULL,
    publication_year INTEGER,
    language TEXT NOT NULL DEFAULT 'en',
    seed_role TEXT NOT NULL,
    supplied_by TEXT NOT NULL,
    hidden_during_development INTEGER NOT NULL DEFAULT 0,
    eligibility_status TEXT NOT NULL DEFAULT 'eligible',
    indexability_status TEXT NOT NULL DEFAULT 'unchecked',
    recall_status TEXT NOT NULL DEFAULT 'unchecked',
    recall_query_ids TEXT NOT NULL DEFAULT '',
    nonrecall_reason TEXT NOT NULL DEFAULT '',
    UNIQUE(doi, seed_role)
);

CREATE TABLE IF NOT EXISTS hidden_seed_search_log (
    search_run_id TEXT PRIMARY KEY,
    reviewer_role TEXT NOT NULL,
    route TEXT NOT NULL,
    source_name TEXT NOT NULL,
    exact_query_or_seed TEXT NOT NULL,
    executed_at TEXT NOT NULL,
    retrieved_count INTEGER NOT NULL,
    screened_count INTEGER NOT NULL,
    eligible_seed_count INTEGER NOT NULL,
    eligible_seed_dois_json TEXT NOT NULL DEFAULT '[]',
    completion_status TEXT NOT NULL,
    notes TEXT NOT NULL,
    imported_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS records (
    provider TEXT NOT NULL,
    record_key TEXT NOT NULL,
    provider_id TEXT NOT NULL,
    doi TEXT NOT NULL,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL,
    language TEXT NOT NULL,
    publication_year INTEGER,
    work_type TEXT NOT NULL,
    source_url TEXT NOT NULL,
    referenced_works_json TEXT NOT NULL DEFAULT '[]',
    raw_json TEXT NOT NULL,
    retrieval_route TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    PRIMARY KEY(provider, record_key)
);

CREATE TABLE IF NOT EXISTS record_payload_digests (
    provider TEXT NOT NULL,
    record_key TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    PRIMARY KEY(provider, record_key),
    FOREIGN KEY(provider, record_key)
        REFERENCES records(provider, record_key)
);

CREATE TABLE IF NOT EXISTS logical_queries (
    logical_query_id TEXT PRIMARY KEY,
    query_version INTEGER NOT NULL,
    search_domain_id TEXT NOT NULL,
    family_label TEXT NOT NULL,
    logical_expression TEXT NOT NULL,
    object_terms_json TEXT NOT NULL,
    domain_terms_json TEXT NOT NULL,
    context_terms_json TEXT NOT NULL,
    status TEXT NOT NULL,
    archive_reason TEXT NOT NULL DEFAULT '',
    press_status TEXT NOT NULL DEFAULT 'pending',
    press_reviewer TEXT NOT NULL DEFAULT '',
    press_notes TEXT NOT NULL DEFAULT '',
    query_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS physical_queries (
    physical_query_id TEXT PRIMARY KEY,
    logical_query_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    expression TEXT NOT NULL,
    filter_expression TEXT NOT NULL,
    status TEXT NOT NULL,
    query_hash TEXT NOT NULL,
    FOREIGN KEY(logical_query_id) REFERENCES logical_queries(logical_query_id)
);

CREATE TABLE IF NOT EXISTS query_runs (
    provider TEXT NOT NULL,
    physical_query_id TEXT NOT NULL,
    run_role TEXT NOT NULL,
    query_hash TEXT NOT NULL,
    reported_total INTEGER,
    retrieved_rows INTEGER NOT NULL DEFAULT 0,
    unique_hits INTEGER NOT NULL DEFAULT 0,
    pages INTEGER NOT NULL DEFAULT 0,
    next_cursor TEXT NOT NULL DEFAULT '*',
    complete INTEGER NOT NULL DEFAULT 0,
    stopped_reason TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL,
    PRIMARY KEY(provider, physical_query_id, run_role),
    FOREIGN KEY(physical_query_id)
        REFERENCES physical_queries(physical_query_id)
);

CREATE TABLE IF NOT EXISTS query_hits (
    provider TEXT NOT NULL,
    physical_query_id TEXT NOT NULL,
    run_role TEXT NOT NULL,
    record_key TEXT NOT NULL,
    rank INTEGER NOT NULL,
    PRIMARY KEY(provider, physical_query_id, run_role, record_key),
    FOREIGN KEY(provider, record_key)
        REFERENCES records(provider, record_key)
);

CREATE TABLE IF NOT EXISTS citation_edges (
    source_record_key TEXT NOT NULL,
    target_provider_id TEXT NOT NULL,
    direction TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    eligibility_status TEXT NOT NULL DEFAULT 'pending',
    PRIMARY KEY(source_record_key, target_provider_id, direction, iteration)
);

CREATE TABLE IF NOT EXISTS saturation_rounds (
    iteration INTEGER PRIMARY KEY,
    new_records INTEGER NOT NULL,
    new_nonredundant_english_terms INTEGER NOT NULL,
    new_canonical_indicator_families INTEGER NOT NULL,
    reviewer_role TEXT NOT NULL,
    decision TEXT NOT NULL,
    notes TEXT NOT NULL,
    reviewed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS discovery_queries (
    discovery_query_id TEXT PRIMARY KEY,
    query_role TEXT NOT NULL,
    stratum_label TEXT NOT NULL,
    expression TEXT NOT NULL,
    filter_expression TEXT NOT NULL,
    sample_size INTEGER NOT NULL,
    random_seed INTEGER NOT NULL,
    query_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    archive_reason TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS discovery_query_runs (
    discovery_query_id TEXT PRIMARY KEY,
    query_hash TEXT NOT NULL,
    reported_sample_total INTEGER,
    retrieved_rows INTEGER NOT NULL DEFAULT 0,
    unique_hits INTEGER NOT NULL DEFAULT 0,
    pages INTEGER NOT NULL DEFAULT 0,
    next_page INTEGER NOT NULL DEFAULT 1,
    complete INTEGER NOT NULL DEFAULT 0,
    stopped_reason TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL,
    FOREIGN KEY(discovery_query_id)
        REFERENCES discovery_queries(discovery_query_id)
);

CREATE TABLE IF NOT EXISTS discovery_query_evidence (
    discovery_query_id TEXT PRIMARY KEY,
    source_ids_json TEXT NOT NULL,
    source_dois_json TEXT NOT NULL,
    source_phrases_json TEXT NOT NULL,
    derivation_rule TEXT NOT NULL,
    FOREIGN KEY(discovery_query_id)
        REFERENCES discovery_queries(discovery_query_id)
);

CREATE TABLE IF NOT EXISTS discovery_hits (
    discovery_query_id TEXT NOT NULL,
    record_key TEXT NOT NULL,
    sample_rank INTEGER NOT NULL,
    selection_hash TEXT NOT NULL,
    review_rank INTEGER NOT NULL DEFAULT 0,
    review_round INTEGER NOT NULL DEFAULT 0,
    review_status TEXT NOT NULL DEFAULT 'unassigned',
    PRIMARY KEY(discovery_query_id, record_key),
    FOREIGN KEY(discovery_query_id)
        REFERENCES discovery_queries(discovery_query_id),
    FOREIGN KEY(record_key) REFERENCES records(record_key)
);

CREATE TABLE IF NOT EXISTS discovery_review_rounds (
    iteration INTEGER PRIMARY KEY,
    saturation_phase TEXT NOT NULL DEFAULT 'search_frame_discovery',
    batch_first_rank INTEGER NOT NULL,
    batch_last_rank INTEGER NOT NULL,
    assigned_records INTEGER NOT NULL,
    fully_reviewed INTEGER NOT NULL DEFAULT 0,
    new_nonredundant_english_terms INTEGER NOT NULL DEFAULT -1,
    new_canonical_indicator_families INTEGER NOT NULL DEFAULT -1,
    consecutive_zero_rounds INTEGER NOT NULL DEFAULT 0,
    reviewer_role TEXT NOT NULL DEFAULT 'SYSTEM',
    decision TEXT NOT NULL DEFAULT 'pending',
    notes TEXT NOT NULL DEFAULT '',
    reviewed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS discovery_indicator_candidates (
    candidate_id TEXT PRIMARY KEY,
    record_key TEXT NOT NULL,
    review_round INTEGER NOT NULL,
    raw_name_en TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    location TEXT NOT NULL,
    evidence_span TEXT NOT NULL,
    proposed_role TEXT NOT NULL,
    extracted_by TEXT NOT NULL,
    h1_decision TEXT NOT NULL DEFAULT 'pending',
    h2_decision TEXT NOT NULL DEFAULT 'pending',
    canonical_family_label TEXT NOT NULL DEFAULT '',
    adjudication_notes TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'candidate',
    FOREIGN KEY(record_key) REFERENCES records(record_key)
);

CREATE TABLE IF NOT EXISTS discovery_extraction_reviews (
    record_key TEXT NOT NULL,
    review_round INTEGER NOT NULL,
    reviewer_role TEXT NOT NULL,
    extraction_complete INTEGER NOT NULL,
    no_relevant_items INTEGER NOT NULL,
    notes TEXT NOT NULL,
    reviewed_at TEXT NOT NULL,
    PRIMARY KEY(record_key, review_round, reviewer_role),
    FOREIGN KEY(record_key) REFERENCES records(record_key)
);

CREATE TABLE IF NOT EXISTS api_budget_observations (
    observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    key_slot INTEGER NOT NULL,
    daily_budget_usd REAL NOT NULL,
    daily_used_usd REAL NOT NULL,
    daily_remaining_usd REAL NOT NULL,
    resets_at TEXT NOT NULL,
    observed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ai_assistance_runs (
    run_id TEXT PRIMARY KEY,
    task TEXT NOT NULL,
    reviewer_role TEXT NOT NULL,
    model TEXT NOT NULL,
    model_digest TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    output_path TEXT NOT NULL,
    completed_items INTEGER NOT NULL DEFAULT 0,
    failed_items INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS local_snapshot_sources (
    snapshot_id TEXT PRIMARY KEY,
    root_path TEXT NOT NULL,
    manifest_path TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL,
    part_count INTEGER NOT NULL DEFAULT 0,
    record_count INTEGER NOT NULL DEFAULT 0,
    content_length_bytes INTEGER NOT NULL DEFAULT 0,
    maximum_updated_date TEXT NOT NULL,
    role TEXT NOT NULL,
    registered_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS crossref_validation (
    record_key TEXT PRIMARY KEY,
    doi TEXT NOT NULL,
    status TEXT NOT NULL,
    title_match REAL,
    year_match INTEGER,
    type_match INTEGER,
    crossref_title TEXT NOT NULL DEFAULT '',
    crossref_year INTEGER,
    crossref_type TEXT NOT NULL DEFAULT '',
    conflict_reason TEXT NOT NULL DEFAULT '',
    validated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS raw_terms (
    term_id TEXT PRIMARY KEY,
    source_record_key TEXT NOT NULL DEFAULT '',
    source_id TEXT NOT NULL DEFAULT '',
    source_type TEXT NOT NULL,
    source_language_status TEXT NOT NULL,
    source_language_evidence TEXT NOT NULL,
    verbatim_term TEXT NOT NULL,
    normalized_term TEXT NOT NULL,
    match_key TEXT NOT NULL,
    location TEXT NOT NULL,
    evidence_span TEXT NOT NULL,
    proposed_role TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    exclusion_reason TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS term_coding (
    term_id TEXT NOT NULL,
    coder_role TEXT NOT NULL,
    canonical_term TEXT NOT NULL,
    term_family_label TEXT NOT NULL,
    term_relation TEXT NOT NULL,
    search_domain_label TEXT NOT NULL,
    search_domain_definition TEXT NOT NULL,
    query_family_label TEXT NOT NULL,
    cross_domain INTEGER NOT NULL,
    decision TEXT NOT NULL,
    reason TEXT NOT NULL,
    coded_at TEXT NOT NULL,
    PRIMARY KEY(term_id, coder_role),
    FOREIGN KEY(term_id) REFERENCES raw_terms(term_id)
);

CREATE TABLE IF NOT EXISTS term_families (
    term_family_id TEXT PRIMARY KEY,
    label TEXT NOT NULL UNIQUE,
    construct_definition TEXT NOT NULL,
    canonical_term_ids_json TEXT NOT NULL,
    raw_term_ids_json TEXT NOT NULL,
    source_ids_json TEXT NOT NULL,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS canonical_terms (
    canonical_term_id TEXT PRIMARY KEY,
    term_family_id TEXT NOT NULL,
    canonical_term TEXT NOT NULL,
    raw_term_ids_json TEXT NOT NULL,
    relation_map_json TEXT NOT NULL,
    source_ids_json TEXT NOT NULL,
    status TEXT NOT NULL,
    UNIQUE(term_family_id, canonical_term),
    FOREIGN KEY(term_family_id) REFERENCES term_families(term_family_id)
);

CREATE TABLE IF NOT EXISTS search_domains (
    search_domain_id TEXT PRIMARY KEY,
    label TEXT NOT NULL UNIQUE,
    definition TEXT NOT NULL,
    term_ids_json TEXT NOT NULL,
    status TEXT NOT NULL,
    h2_approved INTEGER NOT NULL,
    decision_reason TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS search_frame_versions (
    frame_version INTEGER PRIMARY KEY,
    input_term_hash TEXT NOT NULL,
    frame_hash TEXT NOT NULL,
    counts_json TEXT NOT NULL,
    frame_json TEXT NOT NULL,
    status TEXT NOT NULL,
    derived_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS press_reviews (
    logical_query_id TEXT PRIMARY KEY,
    reviewer_role TEXT NOT NULL,
    concepts_complete INTEGER NOT NULL,
    boolean_logic_valid INTEGER NOT NULL,
    spelling_valid INTEGER NOT NULL,
    phrases_valid INTEGER NOT NULL,
    limits_justified INTEGER NOT NULL,
    covered_by_logical_query_id TEXT NOT NULL,
    logical_coverage_verified INTEGER NOT NULL,
    result_set_coverage_verified INTEGER NOT NULL,
    independent_construct_role INTEGER NOT NULL,
    decision TEXT NOT NULL,
    notes TEXT NOT NULL,
    reviewed_at TEXT NOT NULL,
    FOREIGN KEY(logical_query_id)
        REFERENCES logical_queries(logical_query_id)
);

CREATE TABLE IF NOT EXISTS screening_decisions (
    record_key TEXT NOT NULL,
    reviewer_role TEXT NOT NULL,
    language_judgment TEXT NOT NULL,
    language_evidence TEXT NOT NULL,
    decision TEXT NOT NULL,
    exclusion_reason TEXT NOT NULL,
    evidence_span TEXT NOT NULL,
    notes TEXT NOT NULL,
    decided_at TEXT NOT NULL,
    PRIMARY KEY(record_key, reviewer_role)
);

CREATE TABLE IF NOT EXISTS screening_final (
    record_key TEXT PRIMARY KEY,
    final_language TEXT NOT NULL,
    final_decision TEXT NOT NULL,
    exclusion_reason TEXT NOT NULL,
    h2_required INTEGER NOT NULL,
    h2_completed INTEGER NOT NULL,
    adjudication_reason TEXT NOT NULL,
    finalized_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS indicator_mentions (
    mention_id TEXT PRIMARY KEY,
    record_key TEXT NOT NULL,
    raw_name_en TEXT NOT NULL,
    canonical_name_en TEXT NOT NULL,
    label_zh TEXT NOT NULL,
    source_id TEXT NOT NULL,
    research_group TEXT NOT NULL,
    research_group_id TEXT NOT NULL,
    research_group_evidence TEXT NOT NULL,
    source_role TEXT NOT NULL,
    formula_location TEXT NOT NULL,
    evidence_span TEXT NOT NULL,
    formula TEXT NOT NULL,
    units TEXT NOT NULL,
    parameters TEXT NOT NULL,
    direction TEXT NOT NULL,
    missing_rule TEXT NOT NULL,
    required_data_json TEXT NOT NULL,
    maximum_information_time TEXT NOT NULL,
    scope_role TEXT NOT NULL,
    validation_summary TEXT NOT NULL,
    evidence_direction TEXT NOT NULL,
    negative_evidence TEXT NOT NULL,
    fulltext_source_url TEXT NOT NULL,
    fulltext_local_path TEXT NOT NULL,
    fulltext_sha256 TEXT NOT NULL,
    fulltext_license TEXT NOT NULL,
    english_fulltext_verified INTEGER NOT NULL,
    article_level INTEGER NOT NULL,
    primary_or_foundational_evidence INTEGER NOT NULL,
    formula_reproducible INTEGER NOT NULL,
    t0_computable INTEGER NOT NULL,
    requires_future INTEGER NOT NULL,
    data_status TEXT NOT NULL,
    bias_policy TEXT NOT NULL,
    fatal_validity_concern INTEGER NOT NULL,
    uses_outcome_for_selection INTEGER NOT NULL,
    quality_audit_status TEXT NOT NULL,
    nonconstant INTEGER NOT NULL,
    h2_approved INTEGER NOT NULL,
    evidence_strength TEXT NOT NULL,
    stability_score REAL NOT NULL,
    stability_basis TEXT NOT NULL,
    selection_priority INTEGER NOT NULL,
    redundancy_family TEXT NOT NULL,
    extracted_by TEXT NOT NULL,
    verified_by TEXT NOT NULL,
    verification_notes TEXT NOT NULL,
    adjudication_notes TEXT NOT NULL,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS indicator_source_disposition (
    record_key TEXT PRIMARY KEY,
    disposition TEXT NOT NULL,
    english_fulltext_status TEXT NOT NULL,
    notes TEXT NOT NULL,
    decided_by TEXT NOT NULL,
    decided_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS indicator_source_reviews (
    record_key TEXT NOT NULL,
    reviewer_role TEXT NOT NULL,
    disposition TEXT NOT NULL,
    english_fulltext_status TEXT NOT NULL,
    notes TEXT NOT NULL,
    reviewed_at TEXT NOT NULL,
    PRIMARY KEY(record_key, reviewer_role),
    FOREIGN KEY(record_key) REFERENCES records(record_key)
);

CREATE TABLE IF NOT EXISTS indicator_mention_reviews (
    mention_id TEXT NOT NULL,
    reviewer_role TEXT NOT NULL,
    decision TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    notes TEXT NOT NULL,
    reviewed_at TEXT NOT NULL,
    PRIMARY KEY(mention_id, reviewer_role),
    FOREIGN KEY(mention_id) REFERENCES indicator_mentions(mention_id)
);

CREATE TABLE IF NOT EXISTS fulltext_acquisitions (
    record_key TEXT PRIMARY KEY,
    candidate_url TEXT NOT NULL,
    final_url TEXT NOT NULL,
    local_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    access_statement TEXT NOT NULL,
    http_content_type TEXT NOT NULL,
    byte_count INTEGER NOT NULL,
    status TEXT NOT NULL,
    error TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    FOREIGN KEY(record_key) REFERENCES records(record_key)
);

CREATE TABLE IF NOT EXISTS openalex_location_hydration (
    record_key TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    provider_payload_sha256 TEXT NOT NULL,
    error TEXT NOT NULL,
    hydrated_at TEXT NOT NULL,
    FOREIGN KEY(record_key) REFERENCES records(record_key)
);

CREATE TABLE IF NOT EXISTS indicator_families (
    feature_id TEXT PRIMARY KEY,
    canonical_name_en TEXT NOT NULL UNIQUE,
    label_zh TEXT NOT NULL,
    alias_names_json TEXT NOT NULL,
    mention_ids_json TEXT NOT NULL,
    formula TEXT NOT NULL,
    units TEXT NOT NULL,
    parameters TEXT NOT NULL,
    direction TEXT NOT NULL,
    missing_rule TEXT NOT NULL,
    required_data_json TEXT NOT NULL,
    maximum_information_time TEXT NOT NULL,
    scope_role TEXT NOT NULL,
    article_level INTEGER NOT NULL,
    primary_or_foundational_evidence INTEGER NOT NULL,
    formula_reproducible INTEGER NOT NULL,
    t0_computable INTEGER NOT NULL,
    requires_future INTEGER NOT NULL,
    data_status TEXT NOT NULL,
    bias_policy TEXT NOT NULL,
    fatal_validity_concern INTEGER NOT NULL,
    uses_outcome_for_selection INTEGER NOT NULL,
    quality_audit_status TEXT NOT NULL,
    nonconstant INTEGER NOT NULL,
    english_fulltext_verified INTEGER NOT NULL,
    h2_approved INTEGER NOT NULL,
    evidence_strength TEXT NOT NULL,
    stability_score REAL NOT NULL,
    stability_basis TEXT NOT NULL,
    selection_priority INTEGER NOT NULL,
    redundancy_family TEXT NOT NULL,
    research_groups_json TEXT NOT NULL,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feature_data_audit (
    feature_id TEXT PRIMARY KEY,
    data_status TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    valid_count INTEGER NOT NULL,
    unique_count INTEGER NOT NULL,
    missing_rate REAL NOT NULL,
    derivation_artifact_path TEXT NOT NULL,
    input_snapshot_path TEXT NOT NULL,
    derivation_hash TEXT NOT NULL,
    input_snapshot_hash TEXT NOT NULL,
    audit_status TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    notes TEXT NOT NULL,
    audited_at TEXT NOT NULL,
    FOREIGN KEY(feature_id) REFERENCES indicator_families(feature_id)
);

CREATE TABLE IF NOT EXISTS dimension_coding (
    feature_id TEXT NOT NULL,
    coder_role TEXT NOT NULL,
    dimension_label TEXT NOT NULL,
    dimension_definition TEXT NOT NULL,
    construct_role TEXT NOT NULL,
    information_source TEXT NOT NULL,
    t0_boundary TEXT NOT NULL,
    bias_risk TEXT NOT NULL,
    decision TEXT NOT NULL,
    reason TEXT NOT NULL,
    coded_at TEXT NOT NULL,
    PRIMARY KEY(feature_id, coder_role),
    FOREIGN KEY(feature_id) REFERENCES indicator_families(feature_id)
);

CREATE TABLE IF NOT EXISTS candidate_dimensions (
    dimension_id TEXT PRIMARY KEY,
    label TEXT NOT NULL UNIQUE,
    definition TEXT NOT NULL,
    construct_role TEXT NOT NULL,
    feature_ids_json TEXT NOT NULL,
    research_groups_json TEXT NOT NULL,
    h2_approved INTEGER NOT NULL,
    status TEXT NOT NULL,
    decision_reason TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feature_decisions (
    feature_id TEXT PRIMARY KEY,
    gate_checks_json TEXT NOT NULL,
    failed_gates_json TEXT NOT NULL,
    redundancy_winner_id TEXT NOT NULL,
    final_role TEXT NOT NULL,
    decision_reason TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dimension_decisions (
    dimension_id TEXT PRIMARY KEY,
    selected_feature_ids_json TEXT NOT NULL,
    independent_group_count INTEGER NOT NULL,
    dimension_role TEXT NOT NULL,
    selected INTEGER NOT NULL,
    decision_reason TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS event_log (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    details_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_records_doi
    ON records(provider, doi);
CREATE UNIQUE INDEX IF NOT EXISTS idx_records_record_key
    ON records(record_key);
CREATE INDEX IF NOT EXISTS idx_query_hits_record
    ON query_hits(provider, record_key);
CREATE INDEX IF NOT EXISTS idx_query_hits_record_role
    ON query_hits(provider, record_key, run_role);
CREATE INDEX IF NOT EXISTS idx_query_hits_role_record
    ON query_hits(run_role, provider, record_key);
CREATE INDEX IF NOT EXISTS idx_discovery_hits_record
    ON discovery_hits(record_key);
CREATE INDEX IF NOT EXISTS idx_discovery_hits_round
    ON discovery_hits(review_round, review_status);
CREATE INDEX IF NOT EXISTS idx_discovery_indicator_name
    ON discovery_indicator_candidates(normalized_name);
CREATE INDEX IF NOT EXISTS idx_screening_decision
    ON screening_decisions(decision);
CREATE INDEX IF NOT EXISTS idx_terms_match_key
    ON raw_terms(match_key);
CREATE INDEX IF NOT EXISTS idx_mentions_canonical
    ON indicator_mentions(canonical_name_en);
"""


def connect(path: Path = DATABASE_PATH) -> sqlite3.Connection:
    """Open the v3 database with safe concurrency settings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=90)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA busy_timeout = 90000")
    return connection


def initialize(path: Path = DATABASE_PATH) -> sqlite3.Connection:
    """Create all schema objects and stage rows."""
    connection = connect(path)
    connection.executescript(SCHEMA)
    discovery_columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(discovery_hits)")
    }
    if "review_rank" not in discovery_columns:
        connection.execute(
            """
            ALTER TABLE discovery_hits
            ADD COLUMN review_rank INTEGER NOT NULL DEFAULT 0
            """
        )
    snapshot_columns = {
        str(row["name"])
        for row in connection.execute(
            "PRAGMA table_info(local_snapshot_sources)"
        )
    }
    for column_name in (
        "part_count",
        "record_count",
        "content_length_bytes",
    ):
        if column_name not in snapshot_columns:
            connection.execute(
                f"""
                ALTER TABLE local_snapshot_sources
                ADD COLUMN {column_name} INTEGER NOT NULL DEFAULT 0
                """
            )
    indicator_candidate_columns = {
        str(row["name"])
        for row in connection.execute(
            "PRAGMA table_info(discovery_indicator_candidates)"
        )
    }
    review_round_columns = {
        str(row["name"])
        for row in connection.execute(
            "PRAGMA table_info(discovery_review_rounds)"
        )
    }
    if "saturation_phase" not in review_round_columns:
        connection.execute(
            """
            ALTER TABLE discovery_review_rounds
            ADD COLUMN saturation_phase TEXT NOT NULL
                DEFAULT 'search_frame_discovery'
            """
        )
    if "adjudication_notes" not in indicator_candidate_columns:
        connection.execute(
            """
            ALTER TABLE discovery_indicator_candidates
            ADD COLUMN adjudication_notes TEXT NOT NULL DEFAULT ''
            """
        )
    indicator_mention_columns = {
        str(row["name"])
        for row in connection.execute(
            "PRAGMA table_info(indicator_mentions)"
        )
    }
    for column_name in (
        "fulltext_source_url",
        "fulltext_local_path",
        "fulltext_sha256",
        "fulltext_license",
        "research_group_id",
        "research_group_evidence",
        "stability_basis",
    ):
        if column_name not in indicator_mention_columns:
            connection.execute(
                f"""
                ALTER TABLE indicator_mentions
                ADD COLUMN {column_name} TEXT NOT NULL DEFAULT ''
                """
            )
    indicator_family_columns = {
        str(row["name"])
        for row in connection.execute(
            "PRAGMA table_info(indicator_families)"
        )
    }
    if "stability_basis" not in indicator_family_columns:
        connection.execute(
            """
            ALTER TABLE indicator_families
            ADD COLUMN stability_basis TEXT NOT NULL DEFAULT ''
            """
        )
    data_audit_columns = {
        str(row["name"])
        for row in connection.execute(
            "PRAGMA table_info(feature_data_audit)"
        )
    }
    for column_name in (
        "derivation_artifact_path",
        "input_snapshot_path",
    ):
        if column_name not in data_audit_columns:
            connection.execute(
                f"""
                ALTER TABLE feature_data_audit
                ADD COLUMN {column_name} TEXT NOT NULL DEFAULT ''
                """
            )
    hidden_seed_log_columns = {
        str(row["name"])
        for row in connection.execute(
            "PRAGMA table_info(hidden_seed_search_log)"
        )
    }
    if "eligible_seed_dois_json" not in hidden_seed_log_columns:
        connection.execute(
            """
            ALTER TABLE hidden_seed_search_log
            ADD COLUMN eligible_seed_dois_json TEXT NOT NULL DEFAULT '[]'
            """
        )
    for stage in STAGES:
        connection.execute(
            """
            INSERT OR IGNORE INTO stage_status(
                stage, status, details_json, updated_at
            ) VALUES (?, 'pending', '{}', ?)
            """,
            (stage, utc_now()),
        )
    connection.commit()
    return connection


def set_stage(
    connection: sqlite3.Connection,
    stage: str,
    status: str,
    details: Mapping[str, Any] | None = None,
) -> None:
    """Set one workflow stage and append an audit event."""
    if stage not in STAGES:
        raise ValueError(f"Unknown workflow stage: {stage}")
    if status not in {"pending", "ready", "complete", "blocked"}:
        raise ValueError(f"Invalid stage status: {status}")
    payload = json.dumps(
        dict(details or {}),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    connection.execute(
        """
        UPDATE stage_status
        SET status = ?, details_json = ?, updated_at = ?
        WHERE stage = ?
        """,
        (status, payload, utc_now(), stage),
    )
    log_event(
        connection,
        "stage_status",
        "stage",
        stage,
        {"status": status, "details": dict(details or {})},
    )


def stage_status(
    connection: sqlite3.Connection,
    stage: str,
) -> Dict[str, Any]:
    """Return one stage status record."""
    row = connection.execute(
        "SELECT * FROM stage_status WHERE stage = ?",
        (stage,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Unknown workflow stage: {stage}")
    value = dict(row)
    value["details"] = json.loads(value.pop("details_json"))
    return value


def require_complete(
    connection: sqlite3.Connection,
    stages: Iterable[str],
) -> None:
    """Require prerequisite stages to be complete."""
    incomplete = [
        stage
        for stage in stages
        if stage_status(connection, stage)["status"] != "complete"
    ]
    if incomplete:
        raise RuntimeError(
            "Prerequisite stages are incomplete: " + ", ".join(incomplete)
        )


def invalidate_stages(
    connection: sqlite3.Connection,
    stages: Iterable[str],
    reason: str,
) -> None:
    """Mark downstream results stale after an upstream evidence change."""
    for stage in stages:
        set_stage(
            connection,
            stage,
            "pending",
            {"reason": reason},
        )


def snapshot_import_file(
    connection: sqlite3.Connection,
    input_path: Path,
    role: str,
) -> Path:
    """Copy an imported review file into an immutable hashed audit path."""
    resolved = input_path.resolve()
    digest = sha256_file(resolved)
    if any(
        part.startswith(
            (
                "invalidated_automated_h1_trial",
                "unreviewed_automated_h2_drafts",
            )
        )
        for part in resolved.parts
    ):
        raise RuntimeError(
            "Files in an invalidated or unreviewed automated-human "
            f"quarantine cannot be imported: {resolved}"
        )
    if digest in INVALIDATED_AUTOMATED_HUMAN_HASHES:
        raise RuntimeError(
            "This file hash belongs to an invalidated automated H1/H2 "
            "artifact and cannot be imported"
        )
    if digest in UNREVIEWED_AUTOMATED_H2_HASHES:
        raise RuntimeError(
            "This file hash belongs to an unreviewed automated H2 draft. "
            "It cannot be imported until human review is explicitly "
            "attested and the provenance registry is amended."
        )
    database_file = Path(
        str(
            connection.execute(
                "PRAGMA database_list"
            ).fetchone()["file"]
        )
    )
    snapshot_dir = database_file.parent / "import_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    suffix = resolved.suffix.casefold() or ".dat"
    destination = snapshot_dir / f"{role}_{digest}{suffix}"
    if not destination.exists():
        shutil.copyfile(resolved, destination)
    source_id = f"import_{role}_{digest[:20]}"
    snapshot_role = (
        f"review_import_human_attested_automated_draft:{role}"
        if digest in HUMAN_ATTESTED_AUTOMATED_DRAFT_HASHES
        else f"review_import:{role}"
    )
    connection.execute(
        """
        INSERT OR IGNORE INTO source_snapshots(
            source_id, path, sha256, role, imported_at
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (
            source_id,
            str(destination.resolve()),
            digest,
            snapshot_role,
            utc_now(),
        ),
    )
    return destination


def log_event(
    connection: sqlite3.Connection,
    event_type: str,
    entity_type: str,
    entity_id: str,
    details: Mapping[str, Any],
) -> None:
    """Append one machine-readable audit event."""
    connection.execute(
        """
        INSERT INTO event_log(
            event_type, entity_type, entity_id, details_json, created_at
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (
            event_type,
            entity_type,
            entity_id,
            json.dumps(
                dict(details),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
            utc_now(),
        ),
    )

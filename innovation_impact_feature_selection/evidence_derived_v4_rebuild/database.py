from __future__ import annotations

import csv
import json
import re
import shutil
import sqlite3
from datetime import datetime
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
    "data_correspondence_reviewed",
    "operationalizations_reviewed",
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

CREATE TABLE IF NOT EXISTS source_snapshot_supersessions (
    old_source_id TEXT PRIMARY KEY,
    new_source_id TEXT NOT NULL UNIQUE,
    old_sha256 TEXT NOT NULL,
    observed_current_sha256 TEXT NOT NULL,
    authorization_source_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    superseded_at TEXT NOT NULL,
    FOREIGN KEY(old_source_id) REFERENCES source_snapshots(source_id),
    FOREIGN KEY(new_source_id) REFERENCES source_snapshots(source_id),
    FOREIGN KEY(authorization_source_id)
        REFERENCES source_snapshots(source_id),
    CHECK(old_source_id != new_source_id)
);

CREATE TABLE IF NOT EXISTS human_review_attestations (
    attestation_id TEXT PRIMARY KEY,
    artifact_sha256 TEXT NOT NULL UNIQUE,
    artifact_path TEXT NOT NULL,
    reviewer_role TEXT NOT NULL,
    reviewer_id TEXT NOT NULL,
    provenance_type TEXT NOT NULL,
    attestation_statement TEXT NOT NULL,
    attested_at TEXT NOT NULL,
    status TEXT NOT NULL,
    attestation_file_sha256 TEXT NOT NULL,
    registered_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS independent_ai_review_runs (
    run_id TEXT PRIMARY KEY,
    artifact_sha256 TEXT NOT NULL UNIQUE,
    artifact_path TEXT NOT NULL,
    input_sha256 TEXT NOT NULL,
    input_path TEXT NOT NULL,
    reviewer_role TEXT NOT NULL,
    reviewer_id TEXT NOT NULL,
    model TEXT NOT NULL,
    model_digest TEXT NOT NULL,
    prompt_sha256 TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    item_count INTEGER NOT NULL,
    completed_at TEXT NOT NULL,
    status TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL,
    registered_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS review_run_supersessions (
    old_run_id TEXT PRIMARY KEY,
    new_run_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    recorded_at TEXT NOT NULL
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
    stop_basis TEXT NOT NULL DEFAULT 'not_applicable',
    protocol_amendment_id TEXT NOT NULL DEFAULT '',
    protocol_amendment_sha256 TEXT NOT NULL DEFAULT '',
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

CREATE TABLE IF NOT EXISTS press_query_revisions (
    logical_query_id TEXT PRIMARY KEY,
    source_frame_version INTEGER NOT NULL,
    old_domain_terms_json TEXT NOT NULL,
    revised_domain_terms_json TEXT NOT NULL,
    revision_decision TEXT NOT NULL,
    revision_rationale TEXT NOT NULL,
    reviewer_role TEXT NOT NULL,
    review_artifact_sha256 TEXT NOT NULL,
    review_prompt_sha256 TEXT NOT NULL,
    reviewed_at TEXT NOT NULL,
    FOREIGN KEY(logical_query_id)
        REFERENCES logical_queries(logical_query_id)
);

CREATE TABLE IF NOT EXISTS seed_recall_query_checks (
    frame_version INTEGER NOT NULL,
    physical_query_id TEXT NOT NULL,
    query_hash TEXT NOT NULL,
    seed_set_hash TEXT NOT NULL,
    matched_dois_json TEXT NOT NULL,
    request_count INTEGER NOT NULL,
    complete INTEGER NOT NULL,
    error TEXT NOT NULL,
    checked_at TEXT NOT NULL,
    PRIMARY KEY(frame_version, physical_query_id, seed_set_hash),
    FOREIGN KEY(physical_query_id)
        REFERENCES physical_queries(physical_query_id)
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

CREATE TABLE IF NOT EXISTS targeted_formula_reviews (
    target_id TEXT NOT NULL,
    feature_id TEXT NOT NULL,
    record_key TEXT NOT NULL,
    reviewer_role TEXT NOT NULL,
    decision TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    reason TEXT NOT NULL,
    artifact_sha256 TEXT NOT NULL,
    reviewed_at TEXT NOT NULL,
    PRIMARY KEY(target_id, reviewer_role)
);

CREATE TABLE IF NOT EXISTS targeted_formula_decisions (
    target_id TEXT PRIMARY KEY,
    feature_id TEXT NOT NULL,
    record_key TEXT NOT NULL,
    final_decision TEXT NOT NULL,
    mention_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    h1_artifact_sha256 TEXT NOT NULL,
    h2_artifact_sha256 TEXT NOT NULL,
    decided_at TEXT NOT NULL
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

CREATE TABLE IF NOT EXISTS feature_data_correspondence_reviews (
    feature_id TEXT NOT NULL,
    reviewer_role TEXT NOT NULL,
    decision TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    reason TEXT NOT NULL,
    reviewed_at TEXT NOT NULL,
    PRIMARY KEY(feature_id, reviewer_role),
    FOREIGN KEY(feature_id) REFERENCES indicator_families(feature_id)
);

CREATE TABLE IF NOT EXISTS feature_operationalization_reviews (
    feature_id TEXT NOT NULL,
    reviewer_role TEXT NOT NULL,
    decision TEXT NOT NULL,
    formula_mention_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    reason TEXT NOT NULL,
    reviewed_at TEXT NOT NULL,
    PRIMARY KEY(feature_id, reviewer_role),
    FOREIGN KEY(feature_id) REFERENCES indicator_families(feature_id),
    FOREIGN KEY(formula_mention_id)
        REFERENCES indicator_mentions(mention_id)
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

CREATE TABLE IF NOT EXISTS v3_coverage_recovery_queries (
    v3_feature_id TEXT PRIMARY KEY,
    v3_canonical_name_en TEXT NOT NULL,
    query_expression TEXT NOT NULL,
    filter_expression TEXT NOT NULL,
    query_hash TEXT NOT NULL,
    reported_total INTEGER,
    retrieved_rows INTEGER NOT NULL DEFAULT 0,
    next_cursor TEXT NOT NULL DEFAULT '*',
    complete INTEGER NOT NULL DEFAULT 0,
    error TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS v3_coverage_recovery_hits (
    v3_feature_id TEXT NOT NULL,
    record_key TEXT NOT NULL,
    rank INTEGER NOT NULL,
    PRIMARY KEY(v3_feature_id, record_key),
    FOREIGN KEY(v3_feature_id)
        REFERENCES v3_coverage_recovery_queries(v3_feature_id),
    FOREIGN KEY(record_key) REFERENCES records(record_key)
);

CREATE TABLE IF NOT EXISTS v3_contextual_recovery_queries (
    v3_feature_id TEXT PRIMARY KEY,
    query_expression TEXT NOT NULL,
    filter_expression TEXT NOT NULL,
    query_hash TEXT NOT NULL,
    reported_total INTEGER,
    retrieved_rows INTEGER NOT NULL DEFAULT 0,
    complete INTEGER NOT NULL DEFAULT 0,
    error TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL,
    FOREIGN KEY(v3_feature_id)
        REFERENCES v3_coverage_reconciliation(v3_feature_id)
);

CREATE TABLE IF NOT EXISTS v3_contextual_recovery_hits (
    v3_feature_id TEXT NOT NULL,
    record_key TEXT NOT NULL,
    rank INTEGER NOT NULL,
    PRIMARY KEY(v3_feature_id, record_key),
    FOREIGN KEY(v3_feature_id)
        REFERENCES v3_contextual_recovery_queries(v3_feature_id),
    FOREIGN KEY(record_key) REFERENCES records(record_key)
);

CREATE TABLE IF NOT EXISTS v3_coverage_reconciliation (
    v3_feature_id TEXT PRIMARY KEY,
    v3_canonical_name_en TEXT NOT NULL,
    coverage_disposition TEXT NOT NULL DEFAULT 'pending_source_recovery',
    mapped_v4_feature_id TEXT,
    source_record_keys_json TEXT NOT NULL DEFAULT '[]',
    evidence_status TEXT NOT NULL DEFAULT 'not_reviewed',
    final_reason TEXT NOT NULL DEFAULT '',
    reviewed_by TEXT NOT NULL DEFAULT '',
    reviewed_at TEXT NOT NULL DEFAULT '',
    FOREIGN KEY(mapped_v4_feature_id)
        REFERENCES indicator_families(feature_id)
);

CREATE TABLE IF NOT EXISTS v3_coverage_triage_reviews (
    v3_feature_id TEXT NOT NULL,
    reviewer_role TEXT NOT NULL,
    triage_decision TEXT NOT NULL,
    scope_role_assessment TEXT NOT NULL,
    rationale TEXT NOT NULL,
    minimum_source_evidence_needed TEXT NOT NULL,
    search_terms_en TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    artifact_sha256 TEXT NOT NULL,
    imported_at TEXT NOT NULL,
    PRIMARY KEY(v3_feature_id, reviewer_role),
    FOREIGN KEY(v3_feature_id)
        REFERENCES v3_coverage_reconciliation(v3_feature_id)
);

CREATE TABLE IF NOT EXISTS contextual_source_screening_reviews (
    record_key TEXT NOT NULL,
    reviewer_role TEXT NOT NULL,
    screen_decision TEXT NOT NULL,
    evidence_span TEXT NOT NULL,
    rationale TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    artifact_sha256 TEXT NOT NULL,
    imported_at TEXT NOT NULL,
    PRIMARY KEY(record_key, reviewer_role),
    FOREIGN KEY(record_key) REFERENCES records(record_key)
);

CREATE TABLE IF NOT EXISTS contextual_source_final (
    record_key TEXT PRIMARY KEY,
    final_decision TEXT NOT NULL,
    evidence_span TEXT NOT NULL,
    rationale TEXT NOT NULL,
    h2_required INTEGER NOT NULL,
    h2_completed INTEGER NOT NULL,
    finalized_at TEXT NOT NULL,
    FOREIGN KEY(record_key) REFERENCES records(record_key)
);

CREATE TABLE IF NOT EXISTS contextual_fulltext_source_final (
    record_key TEXT PRIMARY KEY,
    final_disposition TEXT NOT NULL,
    notes TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    artifact_sha256 TEXT NOT NULL,
    finalized_at TEXT NOT NULL,
    FOREIGN KEY(record_key) REFERENCES records(record_key)
);

CREATE TABLE IF NOT EXISTS contextual_fulltext_indicator_candidates (
    candidate_id TEXT PRIMARY KEY,
    record_key TEXT NOT NULL,
    raw_name_en TEXT NOT NULL,
    canonical_name_en TEXT NOT NULL,
    source_role TEXT NOT NULL,
    formula_location TEXT NOT NULL,
    evidence_span TEXT NOT NULL,
    formula TEXT NOT NULL,
    parameters TEXT NOT NULL,
    required_data TEXT NOT NULL,
    maximum_information_time TEXT NOT NULL,
    scope_role TEXT NOT NULL,
    requires_future INTEGER NOT NULL,
    extraction_notes TEXT NOT NULL,
    h2_decision TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    artifact_sha256 TEXT NOT NULL,
    imported_at TEXT NOT NULL,
    UNIQUE(record_key, canonical_name_en),
    FOREIGN KEY(record_key) REFERENCES records(record_key)
);

CREATE TABLE IF NOT EXISTS contextual_candidate_canonicalization_reviews (
    candidate_id TEXT NOT NULL,
    reviewer_role TEXT NOT NULL,
    family_name_en TEXT NOT NULL,
    merge_or_split_reason TEXT NOT NULL,
    formula_reproducible INTEGER NOT NULL,
    t0_computable INTEGER NOT NULL,
    scope_role TEXT NOT NULL,
    missing_rule_status TEXT NOT NULL,
    promotion_decision TEXT NOT NULL,
    rationale TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    artifact_sha256 TEXT NOT NULL,
    reviewed_at TEXT NOT NULL,
    PRIMARY KEY(candidate_id, reviewer_role),
    FOREIGN KEY(candidate_id)
        REFERENCES contextual_fulltext_indicator_candidates(candidate_id)
);

CREATE TABLE IF NOT EXISTS contextual_candidate_canonicalization_final (
    candidate_id TEXT PRIMARY KEY,
    family_name_en TEXT NOT NULL,
    merge_or_split_reason TEXT NOT NULL,
    formula_reproducible INTEGER NOT NULL,
    t0_computable INTEGER NOT NULL,
    scope_role TEXT NOT NULL,
    missing_rule_status TEXT NOT NULL,
    promotion_decision TEXT NOT NULL,
    rationale TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    artifact_sha256 TEXT NOT NULL,
    finalized_at TEXT NOT NULL,
    FOREIGN KEY(candidate_id)
        REFERENCES contextual_fulltext_indicator_candidates(candidate_id)
);

CREATE TABLE IF NOT EXISTS contextual_formalization_reviews (
    candidate_id TEXT NOT NULL,
    reviewer_role TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    artifact_sha256 TEXT NOT NULL,
    reviewed_at TEXT NOT NULL,
    PRIMARY KEY(candidate_id, reviewer_role),
    FOREIGN KEY(candidate_id)
        REFERENCES contextual_fulltext_indicator_candidates(candidate_id)
);

CREATE TABLE IF NOT EXISTS contextual_formalization_final (
    candidate_id TEXT PRIMARY KEY,
    canonical_name_en TEXT NOT NULL,
    label_zh TEXT NOT NULL,
    formula TEXT NOT NULL,
    units TEXT NOT NULL,
    parameters TEXT NOT NULL,
    direction TEXT NOT NULL,
    missing_rule TEXT NOT NULL,
    required_data_json TEXT NOT NULL,
    research_group TEXT NOT NULL,
    research_group_evidence TEXT NOT NULL,
    data_match_decision TEXT NOT NULL,
    local_source_ids_json TEXT NOT NULL,
    local_columns_json TEXT NOT NULL,
    derivation_description TEXT NOT NULL,
    formalization_decision TEXT NOT NULL,
    rationale TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    artifact_sha256 TEXT NOT NULL,
    finalized_at TEXT NOT NULL,
    FOREIGN KEY(candidate_id)
        REFERENCES contextual_fulltext_indicator_candidates(candidate_id)
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

CREATE VIEW IF NOT EXISTS formal_review_records AS
    SELECT DISTINCT record_key
    FROM discovery_hits
    WHERE review_round > 0
    UNION
    SELECT DISTINCT record_key
    FROM query_hits
    WHERE run_role = 'formal' AND rank BETWEEN 1 AND 10
    UNION
    SELECT record_key
    FROM records
    WHERE retrieval_route LIKE '%manual%supplement%'
       OR (
           retrieval_route LIKE '%citation%'
           AND NOT EXISTS (
               SELECT 1 FROM discovery_queries
               WHERE status IN ('active', 'network')
           )
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
CREATE INDEX IF NOT EXISTS idx_v3_coverage_hits_record
    ON v3_coverage_recovery_hits(record_key);
CREATE INDEX IF NOT EXISTS idx_v3_contextual_hits_record
    ON v3_contextual_recovery_hits(record_key);
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
    reconciliation_sql = connection.execute("""
        SELECT sql FROM sqlite_master
        WHERE type = 'table' AND name = 'v3_coverage_reconciliation'
        """).fetchone()
    if (
        reconciliation_sql is not None
        and "mapped_v4_feature_id TEXT NOT NULL DEFAULT ''"
        in str(reconciliation_sql["sql"])
    ):
        # The first development revision encoded an absent mapping as an empty
        # string.  That violates the indicator-family foreign key.  Rebuild
        # the isolated ledger losslessly, changing only empty mappings to SQL
        # NULL (the correct representation of not-yet-reviewed).
        connection.execute(
            "ALTER TABLE v3_coverage_reconciliation "
            "RENAME TO v3_coverage_reconciliation_legacy"
        )
        connection.execute("""
            CREATE TABLE v3_coverage_reconciliation (
                v3_feature_id TEXT PRIMARY KEY,
                v3_canonical_name_en TEXT NOT NULL,
                coverage_disposition TEXT NOT NULL DEFAULT 'pending_source_recovery',
                mapped_v4_feature_id TEXT,
                source_record_keys_json TEXT NOT NULL DEFAULT '[]',
                evidence_status TEXT NOT NULL DEFAULT 'not_reviewed',
                final_reason TEXT NOT NULL DEFAULT '',
                reviewed_by TEXT NOT NULL DEFAULT '',
                reviewed_at TEXT NOT NULL DEFAULT '',
                FOREIGN KEY(mapped_v4_feature_id)
                    REFERENCES indicator_families(feature_id)
            )
            """)
        connection.execute("""
            INSERT INTO v3_coverage_reconciliation(
                v3_feature_id, v3_canonical_name_en, coverage_disposition,
                mapped_v4_feature_id, source_record_keys_json,
                evidence_status, final_reason, reviewed_by, reviewed_at
            )
            SELECT v3_feature_id, v3_canonical_name_en, coverage_disposition,
                   NULLIF(mapped_v4_feature_id, ''), source_record_keys_json,
                   evidence_status, final_reason, reviewed_by, reviewed_at
            FROM v3_coverage_reconciliation_legacy
            """)
        connection.execute("DROP TABLE v3_coverage_reconciliation_legacy")
    discovery_columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(discovery_hits)")
    }
    if "review_rank" not in discovery_columns:
        connection.execute("""
            ALTER TABLE discovery_hits
            ADD COLUMN review_rank INTEGER NOT NULL DEFAULT 0
            """)
    snapshot_columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(local_snapshot_sources)")
    }
    for column_name in (
        "part_count",
        "record_count",
        "content_length_bytes",
    ):
        if column_name not in snapshot_columns:
            connection.execute(f"""
                ALTER TABLE local_snapshot_sources
                ADD COLUMN {column_name} INTEGER NOT NULL DEFAULT 0
                """)
    indicator_candidate_columns = {
        str(row["name"])
        for row in connection.execute(
            "PRAGMA table_info(discovery_indicator_candidates)"
        )
    }
    review_round_columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(discovery_review_rounds)")
    }
    if "saturation_phase" not in review_round_columns:
        connection.execute("""
            ALTER TABLE discovery_review_rounds
            ADD COLUMN saturation_phase TEXT NOT NULL
                DEFAULT 'search_frame_discovery'
            """)
    for column_name, default_value in (
        ("stop_basis", "not_applicable"),
        ("protocol_amendment_id", ""),
        ("protocol_amendment_sha256", ""),
    ):
        if column_name not in review_round_columns:
            connection.execute(f"""
                ALTER TABLE discovery_review_rounds
                ADD COLUMN {column_name} TEXT NOT NULL
                    DEFAULT '{default_value}'
                """)
    if "adjudication_notes" not in indicator_candidate_columns:
        connection.execute("""
            ALTER TABLE discovery_indicator_candidates
            ADD COLUMN adjudication_notes TEXT NOT NULL DEFAULT ''
            """)
    indicator_mention_columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(indicator_mentions)")
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
            connection.execute(f"""
                ALTER TABLE indicator_mentions
                ADD COLUMN {column_name} TEXT NOT NULL DEFAULT ''
                """)
    indicator_family_columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(indicator_families)")
    }
    if "stability_basis" not in indicator_family_columns:
        connection.execute("""
            ALTER TABLE indicator_families
            ADD COLUMN stability_basis TEXT NOT NULL DEFAULT ''
            """)
    data_audit_columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(feature_data_audit)")
    }
    for column_name in (
        "derivation_artifact_path",
        "input_snapshot_path",
    ):
        if column_name not in data_audit_columns:
            connection.execute(f"""
                ALTER TABLE feature_data_audit
                ADD COLUMN {column_name} TEXT NOT NULL DEFAULT ''
                """)
    hidden_seed_log_columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(hidden_seed_search_log)")
    }
    if "eligible_seed_dois_json" not in hidden_seed_log_columns:
        connection.execute("""
            ALTER TABLE hidden_seed_search_log
            ADD COLUMN eligible_seed_dois_json TEXT NOT NULL DEFAULT '[]'
            """)
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
    registered_attestation = connection.execute(
        """
        SELECT attestation_id FROM human_review_attestations
        WHERE artifact_sha256 = ? AND status = 'accepted'
        """,
        (digest,),
    ).fetchone()
    registered_ai_review = connection.execute(
        """
        SELECT run_id FROM independent_ai_review_runs
        WHERE artifact_sha256 = ? AND status = 'complete'
        """,
        (digest,),
    ).fetchone()
    if any(
        part.startswith(
            (
                "invalidated_automated_h1_trial",
                "invalidated_local_qwen_review",
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
    if digest in UNREVIEWED_AUTOMATED_H2_HASHES and registered_attestation is None:
        raise RuntimeError(
            "This file hash belongs to an unreviewed automated H2 draft. "
            "It cannot be imported until human review is explicitly "
            "attested and the provenance registry is amended."
        )
    database_file = Path(
        str(connection.execute("PRAGMA database_list").fetchone()["file"])
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
        else (
            f"review_import_human_attested_assisted_draft:{role}"
            if registered_attestation is not None
            else (
                f"review_import_independent_ai_adjudication:{role}"
                if registered_ai_review is not None
                else f"review_import:{role}"
            )
        )
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


def supersede_source_snapshot(
    connection: sqlite3.Connection,
    old_source_id: str,
    new_source_id: str,
    current_path: Path,
    authorization_path: Path,
    reason: str,
) -> Dict[str, str]:
    """Version a changed source without overwriting its frozen baseline.

    The current bytes and the authorizing protocol are copied to immutable,
    content-addressed paths. The original source row and hash remain intact.
    """
    old = connection.execute(
        """
        SELECT source_id, path, sha256, role
        FROM source_snapshots WHERE source_id = ?
        """,
        (old_source_id,),
    ).fetchone()
    if old is None:
        raise ValueError(f"Unknown frozen source snapshot: {old_source_id}")
    if not re.fullmatch(r"[A-Za-z0-9_.:-]+", new_source_id):
        raise ValueError(f"Invalid new source ID: {new_source_id}")
    if old_source_id == new_source_id:
        raise ValueError("A source supersession requires a new source ID")
    reason = reason.strip()
    if not reason:
        raise ValueError("A source supersession requires a reason")
    resolved = current_path.resolve()
    authorization = authorization_path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    if not authorization.is_file():
        raise FileNotFoundError(authorization)
    try:
        authorization_payload = json.loads(authorization.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            "Source supersession authorization must be valid JSON"
        ) from error
    authorized_ids = authorization_payload.get("authorized_old_source_ids")
    if not isinstance(authorized_ids, list) or old_source_id not in {
        str(value) for value in authorized_ids
    }:
        raise ValueError(
            "Source supersession is not authorized for " f"{old_source_id}"
        )
    if authorization_payload.get("original_hashes_retained") is not True:
        raise ValueError(
            "Source supersession authorization must retain original hashes"
        )
    current_sha256 = sha256_file(resolved)
    if current_sha256 == str(old["sha256"]):
        raise ValueError(
            f"Source has not changed and needs no supersession: {old_source_id}"
        )
    database_file = Path(
        str(connection.execute("PRAGMA database_list").fetchone()["file"])
    )
    snapshot_dir = database_file.parent / "source_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    def immutable_copy(path: Path, source_id: str, digest: str) -> Path:
        suffix = path.suffix.casefold() or ".dat"
        destination = snapshot_dir / f"{source_id}_{digest}{suffix}"
        if not destination.exists():
            shutil.copyfile(path, destination)
        if sha256_file(destination) != digest:
            raise RuntimeError(
                f"Immutable source copy failed verification: {destination}"
            )
        return destination

    current_snapshot = immutable_copy(
        resolved,
        new_source_id,
        current_sha256,
    )
    authorization_sha256 = sha256_file(authorization)
    authorization_source_id = (
        "protocol_amendment_implementation_snapshot_supersession_v3"
    )
    authorization_snapshot = immutable_copy(
        authorization,
        authorization_source_id,
        authorization_sha256,
    )
    existing_new = connection.execute(
        """
        SELECT path, sha256, role FROM source_snapshots
        WHERE source_id = ?
        """,
        (new_source_id,),
    ).fetchone()
    if existing_new is not None and (
        str(existing_new["sha256"]) != current_sha256
        or str(existing_new["role"]) != str(old["role"])
    ):
        raise RuntimeError(
            f"New source ID is already bound to other bytes: {new_source_id}"
        )
    connection.execute(
        """
        INSERT OR IGNORE INTO source_snapshots(
            source_id, path, sha256, role, imported_at
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (
            new_source_id,
            str(current_snapshot.resolve()),
            current_sha256,
            str(old["role"]),
            utc_now(),
        ),
    )
    connection.execute(
        """
        INSERT OR IGNORE INTO source_snapshots(
            source_id, path, sha256, role, imported_at
        ) VALUES (?, ?, ?, 'implementation_versioning_protocol', ?)
        """,
        (
            authorization_source_id,
            str(authorization_snapshot.resolve()),
            authorization_sha256,
            utc_now(),
        ),
    )
    existing = connection.execute(
        """
        SELECT * FROM source_snapshot_supersessions
        WHERE old_source_id = ?
        """,
        (old_source_id,),
    ).fetchone()
    expected = {
        "new_source_id": new_source_id,
        "old_sha256": str(old["sha256"]),
        "observed_current_sha256": current_sha256,
        "authorization_source_id": authorization_source_id,
        "reason": reason,
    }
    if existing is not None and any(
        str(existing[key]) != value for key, value in expected.items()
    ):
        raise RuntimeError(f"Conflicting source supersession: {old_source_id}")
    connection.execute(
        """
        INSERT OR IGNORE INTO source_snapshot_supersessions(
            old_source_id, new_source_id, old_sha256,
            observed_current_sha256, authorization_source_id,
            reason, superseded_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            old_source_id,
            new_source_id,
            str(old["sha256"]),
            current_sha256,
            authorization_source_id,
            reason,
            utc_now(),
        ),
    )
    log_event(
        connection,
        "source_snapshot_superseded",
        "source_snapshot",
        old_source_id,
        expected,
    )
    connection.commit()
    return {
        "old_source_id": old_source_id,
        **expected,
        "snapshot_path": str(current_snapshot.resolve()),
        "authorization_path": str(authorization_snapshot.resolve()),
    }


def _read_assisted_review_rows(
    input_path: Path,
) -> tuple[list[str], list[Dict[str, str]]]:
    """Read an assisted-review CSV and return its header and rows."""
    with input_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return fields, rows


def assisted_review_file(input_path: Path) -> bool:
    """Return whether a CSV declares automated-draft provenance."""
    if input_path.suffix.casefold() != ".csv" or not input_path.is_file():
        return False
    fields, rows = _read_assisted_review_rows(input_path)
    return "draft_method" in fields and any(
        str(row.get("draft_method") or "").strip() for row in rows
    )


def _validate_reviewed_assisted_rows(
    input_path: Path,
    expected_role: str,
    expected_reviewer_id: str = "",
) -> Dict[str, Any]:
    """Validate row-level human review metadata before attestation/import."""
    fields, rows = _read_assisted_review_rows(input_path)
    if not rows:
        raise ValueError("Assisted human-review artifact has no rows")
    required = {
        "draft_method",
        "human_review_status",
        "human_reviewer_id",
        "human_reviewed_at",
        "human_review_action",
    }
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError(
            "Assisted human-review artifact lacks fields: " + ", ".join(missing)
        )
    role = expected_role.strip().upper()
    if role not in {"H1", "H2"}:
        raise ValueError("Expected assisted-review role must be H1 or H2")
    role_field = next(
        (
            candidate
            for candidate in (
                "reviewer_role",
                "coder_role",
                "extractor_role",
            )
            if candidate in fields
        ),
        "",
    )
    if not role_field:
        raise ValueError(
            "Assisted human-review artifact lacks reviewer_role/coder_role"
        )
    reviewer_ids: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        if not str(row.get("draft_method") or "").strip():
            raise ValueError(f"Assisted-review line {line_number} lacks draft_method")
        if str(row.get("human_review_status") or "").strip().casefold() != "reviewed":
            raise ValueError(
                "Every assisted-draft row must have "
                "human_review_status=reviewed; "
                f"first bad line {line_number}"
            )
        observed_role = str(row.get(role_field) or "").strip().upper()
        if observed_role != role:
            raise ValueError(
                "Assisted-review role mismatch at line "
                f"{line_number}: expected {role}, found {observed_role!r}"
            )
        reviewer_id = str(row.get("human_reviewer_id") or "").strip()
        if not reviewer_id:
            raise ValueError(
                f"Assisted-review line {line_number} lacks human_reviewer_id"
            )
        reviewer_ids.add(reviewer_id)
        reviewed_at_raw = str(row.get("human_reviewed_at") or "").strip()
        try:
            reviewed_at = datetime.fromisoformat(reviewed_at_raw.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(
                "Assisted-review human_reviewed_at must be ISO-8601; "
                f"bad line {line_number}"
            ) from error
        if reviewed_at.tzinfo is None:
            raise ValueError(
                "Assisted-review human_reviewed_at requires a UTC offset; "
                f"bad line {line_number}"
            )
        if not str(row.get("human_review_action") or "").strip():
            raise ValueError(
                f"Assisted-review line {line_number} lacks " "human_review_action"
            )
    if len(reviewer_ids) != 1:
        raise ValueError(
            "One assisted-review artifact must use exactly one reviewer ID"
        )
    reviewer_id = next(iter(reviewer_ids))
    if expected_reviewer_id and reviewer_id != expected_reviewer_id:
        raise ValueError("Assisted-review row reviewer ID does not match attestation")
    return {
        "rows": len(rows),
        "reviewer_role": role,
        "reviewer_id": reviewer_id,
    }


def _validate_independent_ai_review_rows(
    input_path: Path,
    expected_role: str,
    expected_reviewer_id: str = "",
    expected_run_id: str = "",
) -> Dict[str, Any]:
    """Validate row provenance for an independently executed AI review."""
    fields, rows = _read_assisted_review_rows(input_path)
    if not rows:
        raise ValueError("Independent-AI review artifact has no rows")
    required = {
        "draft_method",
        "independent_ai_review_status",
        "independent_ai_reviewer_id",
        "independent_ai_reviewed_at",
        "independent_ai_review_action",
        "independent_ai_run_id",
        "independent_ai_model",
        "independent_ai_prompt_sha256",
    }
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError(
            "Independent-AI review artifact lacks fields: " + ", ".join(missing)
        )
    role = expected_role.strip().upper()
    if role not in {"AI", "H1", "H2"}:
        raise ValueError("Expected independent-AI role must be AI, H1, or H2")
    role_field = next(
        (
            candidate
            for candidate in (
                "reviewer_role",
                "coder_role",
                "extractor_role",
            )
            if candidate in fields
        ),
        "",
    )
    if not role_field:
        raise ValueError("Independent-AI review lacks reviewer_role/coder_role")
    reviewer_ids: set[str] = set()
    run_ids: set[str] = set()
    models: set[str] = set()
    prompt_hashes: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        if (
            str(row.get("independent_ai_review_status") or "").strip().casefold()
            != "complete"
        ):
            raise ValueError(
                "Every independent-AI row must be complete; "
                f"first bad line {line_number}"
            )
        if str(row.get(role_field) or "").strip().upper() != role:
            raise ValueError(f"Independent-AI role mismatch at line {line_number}")
        reviewer_id = str(row.get("independent_ai_reviewer_id") or "").strip()
        run_id = str(row.get("independent_ai_run_id") or "").strip()
        model = str(row.get("independent_ai_model") or "").strip()
        prompt_hash = (
            str(row.get("independent_ai_prompt_sha256") or "").strip().casefold()
        )
        if not reviewer_id or not run_id or not model:
            raise ValueError(
                "Independent-AI row lacks reviewer/run/model metadata; "
                f"bad line {line_number}"
            )
        if not re.fullmatch(r"[0-9a-f]{64}", prompt_hash):
            raise ValueError(
                "Independent-AI prompt SHA-256 is invalid; " f"bad line {line_number}"
            )
        reviewed_at_raw = str(row.get("independent_ai_reviewed_at") or "").strip()
        try:
            reviewed_at = datetime.fromisoformat(reviewed_at_raw.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(
                "Independent-AI review time must be ISO-8601; "
                f"bad line {line_number}"
            ) from error
        if reviewed_at.tzinfo is None:
            raise ValueError(
                "Independent-AI review time requires a UTC offset; "
                f"bad line {line_number}"
            )
        if not str(row.get("independent_ai_review_action") or "").strip():
            raise ValueError(
                "Independent-AI row lacks review action; " f"bad line {line_number}"
            )
        reviewer_ids.add(reviewer_id)
        run_ids.add(run_id)
        models.add(model)
        prompt_hashes.add(prompt_hash)
    for label, values in (
        ("reviewer ID", reviewer_ids),
        ("run ID", run_ids),
        ("model", models),
        ("prompt hash", prompt_hashes),
    ):
        if len(values) != 1:
            raise ValueError(f"One independent-AI artifact must use one {label}")
    reviewer_id = next(iter(reviewer_ids))
    run_id = next(iter(run_ids))
    if expected_reviewer_id and reviewer_id != expected_reviewer_id:
        raise ValueError("Independent-AI row reviewer ID does not match manifest")
    if expected_run_id and run_id != expected_run_id:
        raise ValueError("Independent-AI row run ID does not match manifest")
    return {
        "rows": len(rows),
        "reviewer_role": role,
        "reviewer_id": reviewer_id,
        "run_id": run_id,
        "model": next(iter(models)),
        "prompt_sha256": next(iter(prompt_hashes)),
    }


def assert_registered_review_attestation(
    connection: sqlite3.Connection,
    input_path: Path,
    expected_role: str,
) -> Dict[str, Any] | None:
    """Require an exact accepted attestation for an assisted-review CSV."""
    resolved = input_path.resolve()
    if not assisted_review_file(resolved):
        return None
    digest = sha256_file(resolved)
    attestation = connection.execute(
        """
        SELECT * FROM human_review_attestations
        WHERE artifact_sha256 = ? AND status = 'accepted'
        """,
        (digest,),
    ).fetchone()
    if attestation is not None:
        row_metadata = _validate_reviewed_assisted_rows(
            resolved,
            expected_role,
        )
        result = dict(attestation)
        if str(result["reviewer_role"]).strip().upper() != expected_role.upper():
            raise ValueError("Registered attestation role does not match import role")
        if str(result["reviewer_id"]) != row_metadata["reviewer_id"]:
            raise ValueError("Registered attestation reviewer ID does not match rows")
        result["review_rows"] = row_metadata["rows"]
        result["reviewer_type"] = "human"
        return result
    ai_review = connection.execute(
        """
        SELECT * FROM independent_ai_review_runs
        WHERE artifact_sha256 = ? AND status = 'complete'
        """,
        (digest,),
    ).fetchone()
    if ai_review is None:
        raise RuntimeError(
            "Assisted-review artifact has neither an accepted human "
            "attestation nor a registered independent-AI review run for "
            f"its exact SHA-256: {digest}"
        )
    ai_result = dict(ai_review)
    row_metadata = _validate_independent_ai_review_rows(
        resolved,
        expected_role,
        str(ai_result["reviewer_id"]),
        str(ai_result["run_id"]),
    )
    if str(ai_result["reviewer_role"]).strip().upper() != expected_role.upper():
        raise ValueError("Registered independent-AI role does not match import role")
    if str(ai_result["model"]) != row_metadata["model"]:
        raise ValueError("Registered independent-AI model does not match review rows")
    if str(ai_result["prompt_sha256"]).casefold() != row_metadata["prompt_sha256"]:
        raise ValueError("Registered independent-AI prompt hash does not match rows")
    ai_result["review_rows"] = row_metadata["rows"]
    ai_result["reviewer_type"] = "independent_ai"
    return ai_result


def register_human_review_attestation(
    connection: sqlite3.Connection,
    input_path: Path,
) -> Dict[str, Any]:
    """Register one human adoption statement for an exact assisted draft."""
    snapshot_path = snapshot_import_file(
        connection,
        input_path,
        "human_review_attestation",
    )
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Human-review attestation must be a JSON object")
    required = (
        "attestation_id",
        "artifact_path",
        "artifact_sha256",
        "reviewer_role",
        "reviewer_id",
        "provenance_type",
        "attestation_statement",
        "attested_at",
        "status",
    )
    missing = [field for field in required if not str(payload.get(field) or "").strip()]
    if missing:
        raise ValueError("Human-review attestation lacks: " + ", ".join(missing))
    role = str(payload["reviewer_role"]).strip().upper()
    if role not in {"H1", "H2"}:
        raise ValueError("Attestation reviewer_role must be H1 or H2")
    provenance = str(payload["provenance_type"]).strip()
    if provenance != "human_reviewed_automated_draft":
        raise ValueError(
            "Attestation provenance_type must be " "human_reviewed_automated_draft"
        )
    if str(payload["status"]).strip().casefold() != "accepted":
        raise ValueError("Attestation status must be accepted")
    statement = str(payload["attestation_statement"]).strip()
    normalized_statement = statement.casefold()
    if not (
        any(token in normalized_statement for token in ("review", "复核"))
        and any(
            token in normalized_statement
            for token in ("adopt", "accept", "采纳", "采用")
        )
    ):
        raise ValueError(
            "Attestation statement must explicitly state human review and "
            "adoption/acceptance"
        )
    try:
        attested_at = datetime.fromisoformat(
            str(payload["attested_at"]).replace("Z", "+00:00")
        )
    except ValueError as error:
        raise ValueError("Attestation attested_at must be ISO-8601") from error
    if attested_at.tzinfo is None:
        raise ValueError("Attestation attested_at requires a UTC offset")
    artifact_sha256 = str(payload["artifact_sha256"]).strip().casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", artifact_sha256):
        raise ValueError("Attestation artifact_sha256 is invalid")
    artifact_path = Path(str(payload["artifact_path"]))
    if not artifact_path.is_absolute():
        artifact_path = input_path.resolve().parent / artifact_path
    artifact_path = artifact_path.resolve()
    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)
    if sha256_file(artifact_path) != artifact_sha256:
        raise ValueError("Attested artifact SHA-256 does not match its file")
    if artifact_path.suffix.casefold() != ".csv":
        raise ValueError("Assisted human-review artifacts must be CSV files")
    attestation_id = str(payload["attestation_id"]).strip()
    reviewer_id = str(payload["reviewer_id"]).strip()
    reviewed_metadata = _validate_reviewed_assisted_rows(
        artifact_path,
        role,
        reviewer_id,
    )
    attestation_digest = sha256_file(snapshot_path)
    connection.execute(
        """
        INSERT INTO human_review_attestations(
            attestation_id, artifact_sha256, artifact_path,
            reviewer_role, reviewer_id, provenance_type,
            attestation_statement, attested_at, status,
            attestation_file_sha256, registered_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'accepted', ?, ?)
        ON CONFLICT(attestation_id) DO UPDATE SET
            artifact_sha256 = excluded.artifact_sha256,
            artifact_path = excluded.artifact_path,
            reviewer_role = excluded.reviewer_role,
            reviewer_id = excluded.reviewer_id,
            provenance_type = excluded.provenance_type,
            attestation_statement = excluded.attestation_statement,
            attested_at = excluded.attested_at,
            status = excluded.status,
            attestation_file_sha256 = excluded.attestation_file_sha256,
            registered_at = excluded.registered_at
        """,
        (
            attestation_id,
            artifact_sha256,
            str(artifact_path),
            role,
            reviewer_id,
            provenance,
            statement,
            str(payload["attested_at"]),
            attestation_digest,
            utc_now(),
        ),
    )
    log_event(
        connection,
        "human_review_attestation_registered",
        "attestation",
        attestation_id,
        {
            "artifact_sha256": artifact_sha256,
            "reviewer_role": role,
            "reviewer_id": reviewer_id,
            "provenance_type": provenance,
        },
    )
    connection.commit()
    return {
        "attestation_id": attestation_id,
        "artifact_sha256": artifact_sha256,
        "reviewer_role": role,
        "reviewer_id": reviewer_id,
        "review_rows": reviewed_metadata["rows"],
        "status": "accepted",
    }


def register_independent_ai_review_manifest(
    connection: sqlite3.Connection,
    input_path: Path,
) -> Dict[str, Any]:
    """Register one completed, exact-hash independent-AI review artifact."""
    snapshot_path = snapshot_import_file(
        connection,
        input_path,
        "independent_ai_review_manifest",
    )
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Independent-AI review manifest must be an object")
    required = (
        "run_id",
        "artifact_path",
        "artifact_sha256",
        "input_path",
        "input_sha256",
        "reviewer_role",
        "reviewer_id",
        "model",
        "model_digest",
        "prompt_sha256",
        "parameters",
        "item_count",
        "completed_at",
        "status",
    )
    missing = [
        field
        for field in required
        if payload.get(field) is None
        or (field != "parameters" and not str(payload.get(field) or "").strip())
    ]
    if missing:
        raise ValueError("Independent-AI review manifest lacks: " + ", ".join(missing))
    if str(payload["status"]).strip().casefold() != "complete":
        raise ValueError("Independent-AI review status must be complete")
    role = str(payload["reviewer_role"]).strip().upper()
    if role not in {"AI", "H1", "H2"}:
        raise ValueError("Independent-AI reviewer_role must be AI, H1, or H2")
    run_id = str(payload["run_id"]).strip()
    reviewer_id = str(payload["reviewer_id"]).strip()
    model = str(payload["model"]).strip()
    model_digest = str(payload["model_digest"]).strip()
    forbidden_model_tokens = ("qwen", "ollama")
    model_identity = f"{model} {model_digest}".casefold()
    if any(token in model_identity for token in forbidden_model_tokens):
        raise ValueError(
            "Local Qwen/Ollama reviews are forbidden by the reviewer "
            "substitution amendment"
        )
    artifact_digest = str(payload["artifact_sha256"]).strip().casefold()
    source_digest = str(payload["input_sha256"]).strip().casefold()
    prompt_digest = str(payload["prompt_sha256"]).strip().casefold()
    for label, digest in (
        ("artifact_sha256", artifact_digest),
        ("input_sha256", source_digest),
        ("prompt_sha256", prompt_digest),
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Independent-AI {label} is invalid")
    if reviewer_id.startswith(("independent_codex_", "primary_codex_")):
        protocol_source = connection.execute(
            """
            SELECT source_id FROM source_snapshots
            WHERE role = 'independent_review_protocol' AND sha256 = ?
            """,
            (prompt_digest,),
        ).fetchone()
        if protocol_source is None:
            raise ValueError(
                "Independent Codex review prompt hash is not a registered "
                "frozen review protocol"
            )
        if not model_digest.startswith("codex-thread:"):
            raise ValueError(
                "Independent Codex review requires its task/thread ID in "
                "model_digest"
            )
    completed_at_raw = str(payload["completed_at"]).strip()
    try:
        completed_at = datetime.fromisoformat(completed_at_raw.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("Independent-AI completed_at must be ISO-8601") from error
    if completed_at.tzinfo is None:
        raise ValueError("Independent-AI completed_at requires a UTC offset")

    def resolve_manifest_path(value: Any) -> Path:
        path = Path(str(value))
        if not path.is_absolute():
            path = input_path.resolve().parent / path
        return path.resolve()

    artifact_path = resolve_manifest_path(payload["artifact_path"])
    source_path = resolve_manifest_path(payload["input_path"])
    for label, path, expected_digest in (
        ("artifact", artifact_path, artifact_digest),
        ("input", source_path, source_digest),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != expected_digest:
            raise ValueError(f"Independent-AI {label} SHA-256 does not match its file")
    metadata = _validate_independent_ai_review_rows(
        artifact_path,
        role,
        reviewer_id,
        run_id,
    )
    if metadata["model"] != model:
        raise ValueError("Independent-AI model differs between rows and manifest")
    if metadata["prompt_sha256"] != prompt_digest:
        raise ValueError("Independent-AI prompt hash differs between rows and manifest")
    item_count = int(payload["item_count"])
    if item_count != metadata["rows"]:
        raise ValueError("Independent-AI item_count differs from artifact row count")
    parameters = payload["parameters"]
    if not isinstance(parameters, dict):
        raise ValueError("Independent-AI parameters must be an object")
    manifest_digest = sha256_file(snapshot_path)
    connection.execute(
        """
        INSERT INTO independent_ai_review_runs(
            run_id, artifact_sha256, artifact_path,
            input_sha256, input_path, reviewer_role, reviewer_id,
            model, model_digest, prompt_sha256, parameters_json,
            item_count, completed_at, status, manifest_sha256,
            registered_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'complete', ?, ?)
        ON CONFLICT(run_id) DO UPDATE SET
            artifact_sha256 = excluded.artifact_sha256,
            artifact_path = excluded.artifact_path,
            input_sha256 = excluded.input_sha256,
            input_path = excluded.input_path,
            reviewer_role = excluded.reviewer_role,
            reviewer_id = excluded.reviewer_id,
            model = excluded.model,
            model_digest = excluded.model_digest,
            prompt_sha256 = excluded.prompt_sha256,
            parameters_json = excluded.parameters_json,
            item_count = excluded.item_count,
            completed_at = excluded.completed_at,
            status = excluded.status,
            manifest_sha256 = excluded.manifest_sha256,
            registered_at = excluded.registered_at
        """,
        (
            run_id,
            artifact_digest,
            str(artifact_path),
            source_digest,
            str(source_path),
            role,
            reviewer_id,
            model,
            model_digest,
            prompt_digest,
            json.dumps(
                parameters,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
            item_count,
            completed_at_raw,
            manifest_digest,
            utc_now(),
        ),
    )
    log_event(
        connection,
        "independent_ai_review_registered",
        "review_run",
        run_id,
        {
            "artifact_sha256": artifact_digest,
            "input_sha256": source_digest,
            "reviewer_role": role,
            "reviewer_id": reviewer_id,
            "model": model,
            "model_digest": model_digest,
            "prompt_sha256": prompt_digest,
            "item_count": item_count,
        },
    )
    connection.commit()
    return {
        "run_id": run_id,
        "artifact_sha256": artifact_digest,
        "reviewer_role": role,
        "reviewer_id": reviewer_id,
        "model": model,
        "item_count": item_count,
        "status": "complete",
    }


def supersede_independent_ai_review_run(
    connection: sqlite3.Connection,
    old_run_id: str,
    new_run_id: str,
    reason: str,
    allow_superset: bool = False,
) -> Dict[str, Any]:
    """Mark one completed review run as audit-retained but nonfinal."""
    old_id = old_run_id.strip()
    new_id = new_run_id.strip()
    if not old_id or not new_id or old_id == new_id:
        raise ValueError("Supersession requires two distinct run IDs")
    if not reason.strip():
        raise ValueError("Supersession requires a reason")
    old = connection.execute(
        "SELECT * FROM independent_ai_review_runs WHERE run_id = ?",
        (old_id,),
    ).fetchone()
    new = connection.execute(
        """
        SELECT * FROM independent_ai_review_runs
        WHERE run_id = ? AND status = 'complete'
        """,
        (new_id,),
    ).fetchone()
    if old is None or new is None:
        raise ValueError("Both the old run and a complete new run must be registered")
    if str(old["reviewer_role"]) != str(new["reviewer_role"]):
        raise ValueError("Superseding review runs must use the same role")
    old_count = int(old["item_count"])
    new_count = int(new["item_count"])
    scope_mode = "same_scope_correction"
    if old_count != new_count and not allow_superset:
        raise ValueError("Superseding review runs must cover the same item count")
    if old_count != new_count:
        if new_count <= old_count:
            raise ValueError("A superset supersession requires a larger new artifact")
        old_fields, old_rows = _read_assisted_review_rows(
            Path(str(old["artifact_path"]))
        )
        new_fields, new_rows = _read_assisted_review_rows(
            Path(str(new["artifact_path"]))
        )
        key_field = next(
            (
                field
                for field in ("term_id", "candidate_id", "record_key")
                if field in old_fields and field in new_fields
            ),
            "",
        )
        if not key_field:
            raise ValueError("Superset supersession requires a stable artifact key")
        old_by_key = {str(row.get(key_field) or "").strip(): row for row in old_rows}
        new_by_key = {str(row.get(key_field) or "").strip(): row for row in new_rows}
        if (
            "" in old_by_key
            or "" in new_by_key
            or len(old_by_key) != old_count
            or len(new_by_key) != new_count
        ):
            raise ValueError("Superset supersession requires unique nonblank keys")
        if not set(old_by_key).issubset(new_by_key):
            raise ValueError("The new review artifact does not cover every old key")
        comparison_fields = (
            "decision",
            "canonical_term",
            "term_family_label",
            "term_relation",
            "search_domain_label",
            "search_domain_definition",
            "query_family_label",
            "cross_domain",
        )
        changed = [
            key
            for key, old_row in old_by_key.items()
            if any(
                str(old_row.get(field) or "") != str(new_by_key[key].get(field) or "")
                for field in comparison_fields
                if field in old_fields and field in new_fields
            )
        ]
        if changed:
            raise ValueError(
                "Superset supersession changed shared review decisions: "
                + ", ".join(changed[:10])
            )
        scope_mode = "verified_superset_coverage"
    connection.execute(
        """
        INSERT INTO review_run_supersessions(
            old_run_id, new_run_id, reason, recorded_at
        ) VALUES (?, ?, ?, ?)
        ON CONFLICT(old_run_id) DO UPDATE SET
            new_run_id = excluded.new_run_id,
            reason = excluded.reason,
            recorded_at = excluded.recorded_at
        """,
        (old_id, new_id, reason.strip(), utc_now()),
    )
    connection.execute(
        """
        UPDATE independent_ai_review_runs
        SET status = 'superseded'
        WHERE run_id = ?
        """,
        (old_id,),
    )
    log_event(
        connection,
        "independent_ai_review_superseded",
        "review_run",
        old_id,
        {
            "superseded_by": new_id,
            "reason": reason.strip(),
            "scope_mode": scope_mode,
            "old_item_count": old_count,
            "new_item_count": new_count,
        },
    )
    connection.commit()
    return {
        "old_run_id": old_id,
        "new_run_id": new_id,
        "status": "superseded",
        "reason": reason.strip(),
        "scope_mode": scope_mode,
    }


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

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS metadata (
  key TEXT PRIMARY KEY,
  value_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS works (
  work_id TEXT PRIMARY KEY, doi TEXT NOT NULL DEFAULT '', openalex_id TEXT NOT NULL DEFAULT '',
  title TEXT NOT NULL, normalized_title TEXT NOT NULL, publication_year INTEGER,
  language TEXT NOT NULL DEFAULT '', work_type TEXT NOT NULL DEFAULT '', abstract TEXT NOT NULL DEFAULT '',
  source_route TEXT NOT NULL DEFAULT '', payload_hash TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS works_doi_unique ON works(doi) WHERE doi <> '';
CREATE UNIQUE INDEX IF NOT EXISTS works_openalex_unique ON works(openalex_id) WHERE openalex_id <> '';
CREATE INDEX IF NOT EXISTS works_title_year ON works(normalized_title, publication_year);
CREATE TABLE IF NOT EXISTS citations (
  citing_work_id TEXT NOT NULL, cited_work_id TEXT NOT NULL, route TEXT NOT NULL,
  PRIMARY KEY(citing_work_id, cited_work_id, route)
);
CREATE TABLE IF NOT EXISTS work_publication_dates (
  work_id TEXT PRIMARY KEY, publication_date TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS provider_cache_records (
  provider TEXT NOT NULL, record_key TEXT NOT NULL, provider_id TEXT NOT NULL,
  doi TEXT NOT NULL, title TEXT NOT NULL, abstract TEXT NOT NULL,
  language TEXT NOT NULL, publication_year INTEGER, work_type TEXT NOT NULL,
  source_url TEXT NOT NULL, referenced_works_json TEXT NOT NULL,
  raw_json TEXT NOT NULL, retrieval_route TEXT NOT NULL,
  payload_sha256 TEXT NOT NULL, source_database_sha256 TEXT NOT NULL,
  cache_status TEXT NOT NULL,
  PRIMARY KEY(provider, record_key)
);
CREATE INDEX IF NOT EXISTS provider_cache_record_key_idx
  ON provider_cache_records(record_key);
CREATE INDEX IF NOT EXISTS provider_cache_doi_idx
  ON provider_cache_records(doi);
CREATE TABLE IF NOT EXISTS search_runs (
  run_id TEXT PRIMARY KEY, stage TEXT NOT NULL, query_id TEXT NOT NULL DEFAULT '',
  provider TEXT NOT NULL DEFAULT '', key_slot TEXT NOT NULL DEFAULT '', cursor TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL, request_count INTEGER NOT NULL DEFAULT 0, response_hash TEXT NOT NULL DEFAULT '',
  error_code TEXT NOT NULL DEFAULT '', started_at TEXT NOT NULL, completed_at TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS terms (
  term_id TEXT PRIMARY KEY, raw_term TEXT NOT NULL, normalized_term TEXT NOT NULL,
  source_work_id TEXT NOT NULL, evidence_quote TEXT NOT NULL, role TEXT NOT NULL,
  round_no INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS term_families (
  family_id TEXT PRIMARY KEY, canonical_term TEXT NOT NULL UNIQUE, role TEXT NOT NULL,
  source_term_ids_json TEXT NOT NULL, reviewer_status TEXT NOT NULL, merge_reason TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS search_domains (
  domain_id TEXT PRIMARY KEY, label TEXT NOT NULL UNIQUE, definition TEXT NOT NULL,
  term_family_ids_json TEXT NOT NULL, source_work_ids_json TEXT NOT NULL,
  primary_decision TEXT NOT NULL, independent_decision TEXT NOT NULL, status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS logical_queries (
  query_id TEXT PRIMARY KEY, domain_id TEXT NOT NULL, expression TEXT NOT NULL,
  semantic_expression TEXT NOT NULL, evidence_ids_json TEXT NOT NULL,
  press_status TEXT NOT NULL DEFAULT 'pending', redundancy_status TEXT NOT NULL DEFAULT 'active',
  frozen INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS physical_queries (
  physical_query_id TEXT PRIMARY KEY, query_id TEXT NOT NULL, provider TEXT NOT NULL,
  request_expression TEXT NOT NULL, split_reason TEXT NOT NULL DEFAULT '', active INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS seed_recall (
  seed_id TEXT PRIMARY KEY, cohort TEXT NOT NULL, work_id TEXT NOT NULL DEFAULT '',
  indexability TEXT NOT NULL, recall_status TEXT NOT NULL, reason_code TEXT NOT NULL DEFAULT '',
  matched_query_ids_json TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS seed_inputs (
  seed_id TEXT PRIMARY KEY, cohort TEXT NOT NULL, doi TEXT NOT NULL,
  citation TEXT NOT NULL, publication_year INTEGER, language TEXT NOT NULL,
  source_artifact TEXT NOT NULL, source_sha256 TEXT NOT NULL,
  legacy_use TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS screening_decisions (
  work_id TEXT NOT NULL, reviewer_role TEXT NOT NULL, decision TEXT NOT NULL,
  exclusion_code TEXT NOT NULL DEFAULT '', language_evidence TEXT NOT NULL,
  eligibility_evidence TEXT NOT NULL, role TEXT NOT NULL DEFAULT '', t0_judgment TEXT NOT NULL DEFAULT '',
  run_id TEXT NOT NULL, input_hash TEXT NOT NULL, output_hash TEXT NOT NULL,
  model_label TEXT NOT NULL DEFAULT '', reason TEXT NOT NULL, PRIMARY KEY(work_id, reviewer_role)
);
CREATE TABLE IF NOT EXISTS screening_final (
  work_id TEXT PRIMARY KEY, decision TEXT NOT NULL, exclusion_code TEXT NOT NULL DEFAULT '',
  adjudication_reason TEXT NOT NULL, adjudicator_run_id TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS formal_pool_records (
  work_id TEXT PRIMARY KEY, routes_json TEXT NOT NULL, query_ids_json TEXT NOT NULL,
  stable_rank TEXT NOT NULL, payload_sha256 TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS construct_mentions (
  mention_id TEXT PRIMARY KEY, work_id TEXT NOT NULL, construct TEXT NOT NULL, role TEXT NOT NULL,
  information_source TEXT NOT NULL, t0_boundary TEXT NOT NULL, bias_risk TEXT NOT NULL,
  discipline_scope TEXT NOT NULL, indicator_mentions_json TEXT NOT NULL, independent_team TEXT NOT NULL,
  evidence_quote TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS candidate_dimensions (
  dimension_id TEXT PRIMARY KEY, label TEXT NOT NULL UNIQUE, definition TEXT NOT NULL,
  role TEXT NOT NULL, t0_boundary TEXT NOT NULL, source_work_ids_json TEXT NOT NULL,
  independent_teams_json TEXT NOT NULL, merge_split_log_json TEXT NOT NULL,
  primary_approved INTEGER NOT NULL, independent_approved INTEGER NOT NULL,
  independent_non_alias_confirmed INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS indicator_mentions (
  mention_id TEXT PRIMARY KEY, work_id TEXT NOT NULL, dimension_id TEXT NOT NULL,
  raw_name TEXT NOT NULL, definition_evidence TEXT NOT NULL, source_role TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS indicator_families (
  indicator_id TEXT PRIMARY KEY, canonical_name TEXT NOT NULL UNIQUE, aliases_json TEXT NOT NULL,
  dimension_ids_json TEXT NOT NULL, mention_ids_json TEXT NOT NULL, definition TEXT NOT NULL,
  formula TEXT NOT NULL DEFAULT '', definition_source_ids_json TEXT NOT NULL,
  independent_teams_json TEXT NOT NULL, role TEXT NOT NULL, maximum_information_time TEXT NOT NULL,
  missing_rule TEXT NOT NULL, zero_denominator_rule TEXT NOT NULL, empty_set_rule TEXT NOT NULL,
  coverage_rule TEXT NOT NULL, fallback_rule TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'candidate'
);
CREATE TABLE IF NOT EXISTS indicator_evidence (
  evidence_id TEXT PRIMARY KEY, indicator_id TEXT NOT NULL, work_id TEXT NOT NULL,
  evidence_role TEXT NOT NULL, quote TEXT NOT NULL, locator TEXT NOT NULL,
  source_hash TEXT NOT NULL, peer_reviewed INTEGER NOT NULL, team_id TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS indicator_data_mapping (
  indicator_id TEXT PRIMARY KEY, mapping_type TEXT NOT NULL CHECK(mapping_type IN ('direct','derivable','unavailable')),
  fields_json TEXT NOT NULL, derivation TEXT NOT NULL, source_snapshot_hash TEXT NOT NULL,
  coverage REAL, missing_rate REAL, unique_count INTEGER, near_constant INTEGER NOT NULL DEFAULT 0,
  audit_status TEXT NOT NULL DEFAULT 'pending'
);
CREATE TABLE IF NOT EXISTS hard_gate_decisions (
  indicator_id TEXT PRIMARY KEY, h1_scope INTEGER NOT NULL, h2_t0 INTEGER NOT NULL,
  h3_reproducibility INTEGER NOT NULL, h4_computability INTEGER NOT NULL,
  h5_validity_ethics INTEGER NOT NULL, h6_data_integrity INTEGER NOT NULL,
  primary_reason TEXT NOT NULL, independent_reason TEXT NOT NULL,
  deterministic_evidence_json TEXT NOT NULL, all_pass INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS evidence_tiers (
  indicator_id TEXT PRIMARY KEY, tier TEXT NOT NULL CHECK(tier IN ('A','B','C')),
  reason TEXT NOT NULL, independent_approved INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS final_dimensions (
  dimension_id TEXT NOT NULL, set_name TEXT NOT NULL CHECK(set_name IN ('supported','strict')),
  reason TEXT NOT NULL, PRIMARY KEY(dimension_id, set_name)
);
CREATE TABLE IF NOT EXISTS final_features (
  indicator_id TEXT NOT NULL, set_name TEXT NOT NULL CHECK(set_name IN ('all','model','strict','strict_training','primary','expanded','broad_t0')),
  reason TEXT NOT NULL, freeze_hash TEXT NOT NULL DEFAULT '', PRIMARY KEY(indicator_id, set_name)
);
CREATE TABLE IF NOT EXISTS review_sessions (
  review_session_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, reviewer_role TEXT NOT NULL,
  input_hash TEXT NOT NULL, output_hash TEXT NOT NULL, model_label TEXT NOT NULL,
  evidence TEXT NOT NULL, reason TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS saturation_rounds (
  round_no INTEGER PRIMARY KEY, fully_reviewed INTEGER NOT NULL,
  new_term_families INTEGER NOT NULL, new_indicator_families INTEGER NOT NULL,
  decision TEXT NOT NULL, stop_basis TEXT NOT NULL, evidence_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS discovery_round_records (
  round_no INTEGER NOT NULL, work_id TEXT NOT NULL, stratum_id TEXT NOT NULL,
  stable_rank TEXT NOT NULL, payload_sha256 TEXT NOT NULL,
  PRIMARY KEY(round_no, work_id)
);
CREATE TABLE IF NOT EXISTS discovery_decisions (
  round_no INTEGER NOT NULL, work_id TEXT NOT NULL, reviewer_role TEXT NOT NULL,
  decision TEXT NOT NULL, exclusion_code TEXT NOT NULL, evidence TEXT NOT NULL,
  reason TEXT NOT NULL, run_id TEXT NOT NULL,
  PRIMARY KEY(round_no, work_id, reviewer_role)
);
CREATE TABLE IF NOT EXISTS discovery_final (
  round_no INTEGER NOT NULL, work_id TEXT NOT NULL, decision TEXT NOT NULL,
  exclusion_code TEXT NOT NULL, adjudication_reason TEXT NOT NULL,
  adjudicator_run_id TEXT NOT NULL,
  PRIMARY KEY(round_no, work_id)
);
CREATE TABLE IF NOT EXISTS discovery_extractions (
  round_no INTEGER NOT NULL, work_id TEXT NOT NULL,
  term_mentions_json TEXT NOT NULL, indicator_mentions_json TEXT NOT NULL,
  evidence_json TEXT NOT NULL, extractor_role TEXT NOT NULL, run_id TEXT NOT NULL,
  PRIMARY KEY(round_no, work_id, extractor_role)
);
CREATE TABLE IF NOT EXISTS discovery_term_families (
  family_id TEXT PRIMARY KEY, canonical_label TEXT NOT NULL UNIQUE,
  normalized_label TEXT NOT NULL UNIQUE, aliases_json TEXT NOT NULL,
  source_work_ids_json TEXT NOT NULL, evidence_json TEXT NOT NULL,
  first_seen_round INTEGER NOT NULL, independent_confirmed INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS discovery_indicator_families (
  family_id TEXT PRIMARY KEY, canonical_label TEXT NOT NULL UNIQUE,
  normalized_label TEXT NOT NULL UNIQUE, aliases_json TEXT NOT NULL,
  source_work_ids_json TEXT NOT NULL, evidence_json TEXT NOT NULL,
  first_seen_round INTEGER NOT NULL, independent_confirmed INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_manifest (
  audit_id INTEGER PRIMARY KEY AUTOINCREMENT, status TEXT NOT NULL, deterministic_hash TEXT NOT NULL,
  blockers_json TEXT NOT NULL, counts_json TEXT NOT NULL, artifact_hashes_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);

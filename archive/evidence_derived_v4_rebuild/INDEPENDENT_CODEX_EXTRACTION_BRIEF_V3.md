# Independent Codex discovery extraction brief v3

## Independence and allowed inputs

This extraction replaces a nominal H1 manual gate under the project-owner
reviewer-substitution amendment. Use only the reasoning of the assigned
separate Codex task. Do not call Ollama, Qwen, a local model, an external LLM
API, or repository model-running code. Do not read invalidated or unreviewed
automated-review directories.

The primary task will name one or more completed discovery-extraction
worksheets under `outputs/human_tasks/`. Read only those named worksheets and
their embedded English title/abstract evidence. Do not modify an input file,
SQLite, pipeline code, term coding, domains, or search queries.

## Record-level extraction

Every included record must receive a completed disposition. Preserve
`record_key`, DOI, title, abstract, review round, and input order.

Extract all explicit English phrases in the title or abstract that can help
retrieve evidence about an individual scientific paper's novelty,
innovation, or publication-time potential scholarly impact:

- named paper-level novelty, innovation, recombination,
  interdisciplinarity, disruption, diversity, distance, atypicality, or
  related constructs;
- named metrics, measures, scores, indices, models, variables, predictors,
  determinants, validation outcomes, or feature families used at article
  level;
- explicit T0 paper-content, title/abstract, reference, authorship/team,
  topic, network, openness, venue, opportunity, or context features studied
  against scholarly impact;
- exact synonyms, abbreviations, or historical names needed for retrieval.

Do not extract generic prose such as “significant result,” ordinary novelty
claims about the paper's scientific contribution, an unnamed statistical
method, or a future citation count as a T0 feature. A future outcome term may
be retained as a validation/retrieval term only when its role is explicitly
identified as such.

For every extracted item:

- `verbatim_name` must be the exact English phrase occurring in the chosen
  source field;
- `location` must be exactly `title` or `abstract`;
- `evidence_span` must be a concise exact substring of that field and must
  contain `verbatim_name` case-insensitively;
- `proposed_role` must state one of `construct`,
  `indicator_or_measure`, `t0_predictor`, `opportunity_or_context`,
  `control`, or `validation_outcome`;
- `status=active`;
- `record_extraction_complete=true`;
- `no_relevant_items=false`.

Use `item_type=term` for a searchable construct/synonym/predictor phrase. Use
`item_type=indicator_candidate` for an explicit named measure, score, index,
formula-bearing construct, or operational variable. When one exact name is
both a necessary search term and an indicator candidate, create two rows with
the two item types; do not paraphrase it.

For an `indicator_candidate`, set `extractor_role=H1`,
`h1_decision=include`, `h2_decision=pending`, and leave
`canonical_family_label` blank. Indicator-family normalization belongs to the
later independent H2 adjudication.

If a record has no relevant explicit title/abstract item, retain exactly one
blank item row with `record_extraction_complete=true`,
`no_relevant_items=true`, `extractor_role=H1`, and a concise source-based
`review_notes` explanation.

## Provenance

Add these columns to every output row:

- `draft_method=independent_codex_session_review`
- `independent_ai_review_status=complete`
- `independent_ai_reviewer_id=independent_codex_h1_extraction_v3`
- `independent_ai_reviewed_at=<ISO-8601 with UTC offset>`
- `independent_ai_review_action=source_term_indicator_extraction`
- `independent_ai_review_note=<concise rationale>`
- `independent_ai_run_id=<one unique stable ID per output file>`
- `independent_ai_model=<task model name if exposed, otherwise
  codex_configured_default>`
- `independent_ai_prompt_sha256=<SHA-256 of this brief>`

Create a sibling manifest for every completed CSV using the exact manifest
schema in `INDEPENDENT_CODEX_REVIEW_BRIEF_V3.md`, but with
`reviewer_role=H1` and
`reviewer_id=independent_codex_h1_extraction_v3`. If no model digest is
exposed, use the assigned Codex task/thread ID prefixed by `codex-thread:`.

Before marking complete, verify:

- output record keys exactly equal the input record-key set;
- every record has one consistent completed disposition;
- every nonblank name and evidence span passes exact source matching;
- no blank row coexists with an extracted item for the same record;
- all item types, roles, statuses, Boolean fields, and decisions use the
  allowed vocabulary;
- manifest/input/output hashes and row counts are exact.

Extraction produces evidence candidates only. It must not decide the number
of search domains, queries, model dimensions, or final indicators.

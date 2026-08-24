# Independent Codex blind H1 discovery-screening brief v3

## Independence and allowed inputs

Act as the nominal H1 literature screener under the project-owner reviewer
substitution amendment. Use only the named blind H1 discovery-screening
worksheets and their supplied record metadata. Do not inspect AI or H2
decisions, prior screening outputs, SQLite, the search frame, downstream
terms/indicators/dimensions, or invalidated local-model outputs.

Use only the reasoning of the assigned separate Codex task. Do not call
Ollama, Qwen, another local model, an external LLM API, or repository
model-running code. Do not modify the input or project code.

## Eligibility and language

Apply `screening_rules_v3.json` and
`INDEPENDENT_CODEX_REVIEW_LANGUAGE_ADDENDUM_V3.md`.

Include an English journal article, conference article represented by the
OpenAlex article type, or review when its title or abstract supplies evidence
about an individual scientific paper's:

- novelty, innovation, recombination, interdisciplinarity, disruption,
  diversity, distance, atypicality, rigor, openness, or related construct;
- article-level measure, indicator, feature, predictor, determinant,
  opportunity, context, or validation design for potential scholarly impact;
- publication-time text, reference, authorship/team, topic, network,
  openness, venue, policy, or control feature investigated against a
  scholarly-impact or quality outcome.

Exclude with exactly one fixed reason when the record is non-English, not
paper-level, outside novelty/potential-impact scope, future-outcome-only,
not an indicator/predictor/validation study, duplicate, or has insufficient
metadata. Post-publication citations or attention may justify inclusion as a
validation outcome only when the study also defines, measures, or tests a
relevant construct or T0 predictor; they can never be treated as T0 features.

A clearly non-English original title, abstract, or explicit OpenAlex
non-English language is excluded even if an English translation is supplied.
Mojibake alone is not proof of a non-English source. For every row provide an
exact title/abstract `language_evidence` span and an exact eligibility
`evidence_span`.

## Required output and provenance

Preserve every input row, record key, source field, and input order. Complete
`language_judgment`, `decision`, `exclusion_reason`, `evidence_span`, and
`notes` for every row. Use `reviewer_role=H1`.

Add:

- `draft_method=independent_codex_session_review`
- `independent_ai_review_status=complete`
- `independent_ai_reviewer_id=independent_codex_h1_screening_v3`
- `independent_ai_reviewed_at=<ISO-8601 with UTC offset>`
- `independent_ai_review_action=blind_literature_screening`
- `independent_ai_review_note=<concise rationale>`
- `independent_ai_run_id=<one stable ID per output file>`
- `independent_ai_model=<task model name if exposed, otherwise
  codex_configured_default>`
- `independent_ai_prompt_sha256=<SHA-256 of this brief>`

Create a sibling manifest using the schema in
`INDEPENDENT_CODEX_REVIEW_BRIEF_V3.md`, with `reviewer_role=H1` and
`reviewer_id=independent_codex_h1_screening_v3`. Use the assigned task/thread
ID prefixed by `codex-thread:` as `model_digest`.

Before completion verify exact input/output record-key set and order, allowed
decision/reason vocabularies, exact evidence substrings, language boundary,
input/output hashes, row count, and manifest. Screening does not decide any
domain, query, dimension, or indicator count.

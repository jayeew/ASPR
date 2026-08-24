# Independent Codex H2 review brief v3

## Independence boundary

This review must be performed by the reasoning model of a separate Codex
task. Do not call Ollama, `qwen3:8b`, another local model, an external LLM
API, or any model-running script in this repository. Do not read, copy, or
use:

- `outputs/invalidated_local_qwen_review_20260729/`;
- `outputs/unreviewed_automated_h2_drafts_20260729/`;
- any file whose name contains `AI_REVIEWED`, `PROVISIONAL_DRAFT`, or
  `DRAFT_DOMAIN_RECONCILIATION`.

The admissible screening inputs are the blank H2 adjudication worksheets in
`outputs/human_tasks/`. They expose the English source title/abstract and the
already frozen AI/H1 codes. Those primary codes are non-authoritative aids;
H2 must resolve each row from the source evidence.

## Current assignment

Complete these files without editing them in place:

1. `round_01_screening_H2_ADJUDICATE.csv`
2. `round_02_screening_H2_ADJUDICATE.csv`
3. `round_03_screening_H2_ADJUDICATE.csv`
4. `round_04_screening_H2_ADJUDICATE.csv`
5. `crossref_conflicts_H2.csv`

Write reviewed derivatives and manifests only under
`outputs/independent_codex_review_v3/`.

## Screening scope and decisions

The measured or predicted object must be an individual scientific paper,
article, publication, or scholarly work. Include only evidence that defines,
measures, applies, predicts, determines, reviews, or validates:

- paper-level novelty, innovation, recombination, interdisciplinarity,
  disruption, diversity, or a related construct;
- a publication-time (`T0`) paper property, opportunity, or context variable
  studied against potential/later scholarly impact; or
- an article-level metric potentially usable at `T0`.

Later citations, attention, diffusion, or disruption may be validation
outcomes but cannot themselves be T0 features. Ordinary contribution claims
such as “this paper proposes a novel method” are not studies of paper
novelty. Exclude clinical, technical-product, educational, corporate,
patent-only, policy, author-only, journal-only, institution-only,
country-only, or field-only studies unless they explicitly provide an
individual-paper measure or predictor.

Every H2 row must end with:

- `reviewer_role=H2`;
- `language_judgment=en` or `non_en`;
- `decision=include` or `exclude` (never `uncertain`);
- a valid fixed exclusion code when excluded;
- `language_evidence` and `evidence_span` copied exactly from the supplied
  title or abstract;
- a concise source-based note.

For an inclusion, the note must state why the measured object is a paper and
what eligible construct/predictor/metric is present. A self-reference (“this
paper”) alone is insufficient. When title/abstract evidence is inadequate,
exclude with `E_INSUFFICIENT_METADATA`.

## Crossref resolution

Use only supplied DOI and metadata. Choose one:

- `exclude_mapping_error` when OpenAlex and Crossref clearly refer to
  different works;
- `accept_crossref` when registry metadata identifies the same work and the
  conflict is a type, title-style, or date variant;
- `accept_openalex` when Crossref reports DOI-not-found but OpenAlex has a
  syntactically valid DOI and coherent scholarly metadata;
- `manual_bibliographic_resolution` only when the supplied evidence is
  genuinely ambiguous.

Notes must state the observed metadata basis and must not claim a publisher
page was visited.

## Required provenance columns

Add these columns to every reviewed CSV:

- `draft_method=independent_codex_session_review`
- `independent_ai_review_status=complete`
- `independent_ai_reviewer_id=independent_codex_h2_v3`
- `independent_ai_reviewed_at=<ISO-8601 with UTC offset>`
- `independent_ai_review_action=<screening_adjudication or
  bibliographic_resolution>`
- `independent_ai_review_note=<concise rationale>`
- `independent_ai_run_id=<unique stable run ID for that file>`
- `independent_ai_model=<the Codex task model name if exposed, otherwise
  codex_configured_default>`
- `independent_ai_prompt_sha256=<SHA-256 of this brief>`

Preserve every input row and its order. Do not leave any row blank.

## Manifest for each output

Beside every reviewed CSV, write `<stem>.manifest.json` with:

- `run_id`
- absolute `artifact_path` and exact `artifact_sha256`
- absolute `input_path` and exact `input_sha256`
- `reviewer_role: H2`
- `reviewer_id: independent_codex_h2_v3`
- `model`
- `model_digest` (use the task/thread ID if no model digest is exposed)
- `prompt_sha256`
- a JSON `parameters` object including the review method and independence
  boundary
- `item_count`
- `completed_at`
- `status: complete`

The manifest and row metadata must agree exactly. Do not import anything into
SQLite and do not modify pipeline code. The primary task will validate hashes,
register provenance, and import accepted results.

# Independent Codex English term coding and adjudication brief v3

## Purpose and independence

Code evidence-linked English retrieval terms so that the number of search
concept domains and logical query families emerges from the evidence. Do not
target a domain count, query count, dimension count, or indicator count.

When acting as nominal H1, use only the named blind H1 worksheet and its
embedded source evidence. Do not inspect AI coding, H2 coding, search-frame
outputs, invalidated local-model outputs, or an earlier unreviewed coding
draft. When acting as nominal H2, use only the named H2 comparison worksheet
after AI and H1 coding are complete.

Use only the reasoning of the assigned separate Codex task. Do not call
Ollama, Qwen, another local model, an external LLM API, or repository
model-running code. Do not modify the input, SQLite, pipeline code, source
terms, search frame, or any downstream result.

## Coding unit and inclusion boundary

Each `term_id` is one coding unit. Set `decision=include` when the English
phrase is useful for retrieving article-level evidence about scientific-paper
novelty, innovation, or publication-time potential scholarly impact,
including:

- a substantive novelty, recombination, interdisciplinarity, distance,
  diversity, atypicality, rigor, reproducibility, or related construct;
- an explicit article-level measure, feature, predictor, opportunity,
  context, or control family;
- an outcome or validation term needed to locate studies that validate T0
  predictors, provided it is clearly coded as an outcome/validation search
  family and is never represented as a T0 feature;
- an acronym, full form, historical name, or genuine lexical variant needed
  for retrieval.

Exclude generic prose, an ordinary contribution claim, an unnamed statistical
method, a phrase that is not paper-level, an unsupported or corrupted term,
or a source hint that cannot be assigned to any evidence-supported domain.
Every include or exclude decision requires a concise reason.

`development_seed_hint` and `pilot_v2_indicator` are nonauthorizing
development sources. They may supply a synonym to a domain and query family
already supported by direct English discovery literature, but they may not
independently create a domain or query family. Preserve this distinction in
the reason.

## Normalization fields

For every included term populate all required fields:

- `canonical_term`: the preferred English retrieval form;
- `term_family_label`: the synonym/variant family for one theoretical object;
- `term_relation`: exactly one of `canonical`, `synonym`, `abbreviation`,
  `full_form`, `historical_name`, `morphological_variant`, or
  `parameter_variant`;
- `search_domain_label`: one evidence-derived construct domain; use `|`
  separated labels only when the term genuinely spans construct domains;
- `search_domain_definition`: a concise construct boundary, including role
  and T0/outcome distinction where relevant;
- `query_family_label`: one independent semantic/Boolean retrieval purpose
  within that domain;
- `cross_domain`: `true` only for a justified multi-domain assignment;
- `reason`: source- and construct-based rationale.

Apply these fixed merge/split rules:

- merge terms measuring the same theoretical object and serving the same
  explanatory role;
- normally merge data-source, time-window, threshold, transformation, or
  parameter variants, recording the appropriate term relation;
- split substantive paper content, attention/opportunity, background
  control, and future validation outcome roles;
- split when construct definition or the publication-time boundary differs
  materially;
- do not create a query family for a spelling or synonym difference alone;
- create a distinct query family only for a different construct relationship,
  measurement purpose, predictor role, or validation role;
- use stable, concise labels consistently across the complete worksheet.

The granularity must be supported by the terms themselves. Do not reuse the
old v2 domain count or labels merely because they already exist.

## H2 adjudication

For H2 rows, compare the supplied AI and H1 codes. Adjudicate every row in the
worksheet, including agreements that are not jointly excluded. The H2 fields
must contain the final include/exclude decision and, for inclusion, the
complete normalized fields above. Resolve inconsistent labels across rows as
one coherent codebook. H2 may merge, split, or rename domains and query
families under the fixed rules; it may not optimize them against model
performance or choose a desired count.

## Required independent-review provenance

For a separate-session H1 or H2 output, add:

- `draft_method=independent_codex_session_review`
- `independent_ai_review_status=complete`
- `independent_ai_reviewer_id=independent_codex_h1_term_v3` for H1, or
  `independent_codex_h2_term_v3` for H2
- `independent_ai_reviewed_at=<ISO-8601 with UTC offset>`
- `independent_ai_review_action=blind_term_coding` for H1, or
  `term_coding_adjudication` for H2
- `independent_ai_review_note=<concise rationale>`
- `independent_ai_run_id=<one unique stable ID per output file>`
- `independent_ai_model=<task model name if exposed, otherwise
  codex_configured_default>`
- `independent_ai_prompt_sha256=<SHA-256 of this brief>`

Create a sibling manifest using the exact schema in
`INDEPENDENT_CODEX_REVIEW_BRIEF_V3.md`. Use the matching reviewer role and ID.
If no model digest is exposed, use the assigned task/thread ID prefixed by
`codex-thread:`.

Before completion, verify exact input/output `term_id` set equality, allowed
vocabularies, required fields, internally consistent family/domain/query
labels, exact manifest hashes and row counts, and no target-count influence.
Term coding does not itself approve `K`, `Q`, `P`, `M`, `D`, or `F`.

# Independent Codex discovery indicator-family adjudication brief v3

## Independence and allowed inputs

This adjudication replaces a nominal H2 manual gate under the project-owner
reviewer-substitution amendment. Use only the reasoning of the assigned
separate Codex task. Do not call Ollama, Qwen, a local model, an external LLM
API, or repository model-running code. Do not read invalidated local-model
outputs or alter the source extraction, SQLite database, pipeline code,
search domains, search queries, model dimensions, or final feature decisions.

The primary task will name completed
`discovery_indicator_adjudication` worksheets. Use only those worksheets and
their embedded English title, abstract, raw name, role, and exact evidence
span. Preserve every row, `candidate_id`, `record_key`, review round, and
input order.

## Decision boundary

Set `h2_decision=include` only when the supplied title or abstract explicitly
supports an article-level operational measure, score, index, formula-bearing
construct, or publication-time predictor/opportunity/control variable that
could be investigated as a feature of paper novelty or potential scholarly
impact. Inclusion at this stage means “retain as a discovery indicator
family,” not “approve as a final model feature.”

Exclude a candidate when any of the following applies:

- it is generic prose, an unnamed method, or an unoperationalized construct;
- it is not measured at individual-paper level;
- it is only a future citation, attention, diffusion, disruption, or other
  post-publication outcome and therefore cannot be a T0 feature;
- it is an unsupported paraphrase rather than an exact source phrase;
- it describes a population, field, journal, country, institution, or author
  attribute with no paper-level operationalization;
- it is merely a statistical estimator or software method with no substantive
  feature role;
- it is outside paper novelty, publication-time potential, opportunity, or
  background-control scope.

An included opportunity/context/control family must remain explicitly marked
as such in `adjudication_notes`; it must not be represented as a substantive
innovation dimension. A validation outcome may remain a search term elsewhere
but must be excluded from the discovery indicator-family set when it is not a
T0 feature.

## Canonical-family normalization

For every included row, provide one nonblank English
`canonical_family_label`.

- Merge spelling, capitalization, singular/plural, acronym/full-form, coding,
  transformation, threshold, parameter, and time-window variants when they
  measure the same theoretical object and play the same explanatory role.
- Merge data-source variants only when the measured construct and information
  boundary are unchanged.
- Split items that differ materially in construct, causal/explanatory role, or
  T0 availability.
- Do not merge a substantive novelty construct with an attention opportunity,
  background control, or future validation outcome.
- Use a concise, stable construct label rather than copying a paper-specific
  implementation detail.
- Apply the same label consistently across all supplied rounds. Do not target
  or quota a particular number of families.

For an excluded row, leave `canonical_family_label` blank. Every row must have
a concise, evidence-based `adjudication_notes` statement explaining the
decision and, for inclusion, the normalization rationale.

## Required provenance

Add these columns to every output row:

- `draft_method=independent_codex_session_review`
- `independent_ai_review_status=complete`
- `independent_ai_reviewer_id=independent_codex_h2_indicator_v3`
- `independent_ai_reviewed_at=<ISO-8601 with UTC offset>`
- `independent_ai_review_action=discovery_indicator_family_adjudication`
- `independent_ai_review_note=<concise rationale>`
- `independent_ai_run_id=<one unique stable ID per output file>`
- `independent_ai_model=<task model name if exposed, otherwise
  codex_configured_default>`
- `independent_ai_prompt_sha256=<SHA-256 of this brief>`

Create one sibling manifest per output CSV using the exact manifest schema in
`INDEPENDENT_CODEX_REVIEW_BRIEF_V3.md`, with `reviewer_role=H2` and
`reviewer_id=independent_codex_h2_indicator_v3`. If no model digest is
exposed, use the assigned Codex task/thread ID prefixed by `codex-thread:`.

Before completion, verify:

- output candidate IDs exactly equal each input candidate-ID set;
- every row has exactly one include/exclude decision;
- every included row has one family label and every excluded row has none;
- all evidence spans and raw names remain unchanged and exact;
- family labels are consistent across files;
- no family-count target or downstream model result affected a decision;
- input/output hashes, row counts, and manifests are exact.

This step only adjudicates discovery indicator families for saturation. It
must not decide `K`, `Q`, `P`, `M`, `D`, or `F`.

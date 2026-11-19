# Blind v3-coverage scope triage for v4

## Purpose

The attached 432 labels are a **historical coverage benchmark**, not
pre-approved v4 features.  Independently decide only the next recovery action
for each label.  Do not use model performance, the legacy dimension label, or
another reviewer's file.

## Allowed `triage_decision` values

- `recover_priority`: plausibly a paper-level feature of innovation, T0
  substantive potential, T0 opportunity, or a paper-level background control;
  recover original English source evidence.
- `scope_exclude`: clearly a clinical/study-specific outcome, systematic-review
  procedure, non-paper-level construct, post-publication result, or other
  construct outside the v4 target.  State the fixed reason.
- `needs_source_evidence`: label alone is ambiguous; obtain source evidence
  before a scope judgment.

## Required columns

- `scope_role_assessment`: one of `direct_innovation`, `t0_substantive`,
  `t0_opportunity`, `context_control`, `out_of_scope`, or `uncertain`.
- `rationale`: concise construct-level reason.  A legacy role/T0 claim is not
  evidence.
- `minimum_source_evidence_needed`: original application, mathematical
  foundation, validation, or `none_for_clear_scope_exclusion`.
- `search_terms_en`: 2–6 source-search English terms/phrases.  Leave blank
  only for a clear scope exclusion.

## Prohibitions

- This is not formula verification, data mapping, dimension formation, or
  final feature selection.
- Do not approve an item merely because it is in the 432-label archive.
- Do not exclude an ambiguous item solely because its archived formula is
  absent; choose `needs_source_evidence` instead.

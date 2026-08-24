# Independent contextual full-text indicator extraction (v4)

Read each local English text file identified in the input.  The linked v3
labels are discovery leads only, not approved features.

For every source write a source review with exactly one disposition:

- `formula_or_application`: it explicitly defines or applies a paper-level,
  T0-computable innovation/potential-impact/opportunity/control indicator;
- `review_discovery_only`: useful terminology or cited original sources but no
  source-authorized formula/application; or
- `no_relevant_indicator`.

For every candidate formula/application, extract a separate mention row with:
`record_key, raw_name_en, canonical_name_en, source_role,
formula_location, evidence_span, formula, parameters, required_data,
maximum_information_time, scope_role, requires_future, extraction_notes`.

`source_role` must be one of `original_definition`, `original_application`,
`validation`, `review_discovery`, or `mathematical_foundation`.  Do not infer a
formula from prose, do not use a review as sole formula authority, and do not
make a final feature/dimension decision.  Quote a compact, exact English span
and a page/section/equation location whenever a formula is reported.

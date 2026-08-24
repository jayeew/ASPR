# Evidence-derived v4 reconstruction protocol

## Objective

Rebuild the English-only evidence chain for publication-time paper
innovation and potential-impact indicators, then materialize reproducible
features and train HGB out-of-fold models for the fixed twelve experimental
domains and the three, five, and eight-year windows.

## Independence boundary

The v3 source code is a reusable implementation template only. No v3
database, recovered CSV, prior screening decision, prior term assignment,
prior dimension mapping, or prior feature inclusion decision may be imported
into v4. New discovery, screening, independent AI review, adjudication,
search-frame validation, full-text extraction, and feature selection are
stored in `outputs/evidence_derived_v4.sqlite3`.

## Fixed scope

- English journal articles, conference papers, and reviews only.
- Innovation, research quality, and publication-time potential academic
  impact at the paper level.
- No future citation, diffusion, attention, or disruption feature may enter
  a feature matrix.
- OpenAlex is used for discovery and citation tracking; Crossref validates
  DOI, title, year, and type.
- First and second review roles are independent Codex review artifacts;
  neither is represented as human.
- The twelve experimental domains and 3/5/8-year outcome windows are fixed
  only for downstream model evaluation, not for literature-domain discovery.

## Completion conditions

1. Every retained final indicator has an English source, formula or complete
   operational definition, parameters, units/direction, missing-value rule,
   T0 boundary, data mapping, and reproducibility test.
2. Every selected feature column is materialized from the declared audit
   inputs and passes nonconstant and data-quality checks.
3. Search domains, queries, dimensions, and feature counts emerge from the
   v4 evidence database without quotas.
4. The frozen feature sets and data matrices are versioned before HGB fitting.
5. HGB OOF training is completed for every available frozen feature set,
   each of the twelve fixed evaluation domains, and the 3/5/8-year windows.

# Canonical HGB OOF implementation status — 2026-08-20

Status: **IMPLEMENTED AND SYNTHETICALLY VALIDATED; FORMAL TRAINING NOT STARTED**

## Scope

The new adapter consumes only the simplified protocol's frozen
`training_matrix_manifest.json`, `final_feature_sets.json`, and four canonical
Parquet matrices. It does not import evidence-v3 EF rosters or write into the
legacy regression-baseline directory.

The adapter reuses the active Nature multi-horizon D3/D5/D8 cohorts, targets,
fixed forward folds, and HGB parameters. Canonical feature names receive
deterministic internal aliases before merging, preventing collisions with
cohort metadata without changing frozen membership.

## Fail-closed preflight

- exact `Status: **COMPLETE**` protocol audit;
- reconstructed feature-set freeze hash and protocol hash;
- matrix hashes, schemas, row counts, unique paper IDs, and shared row order;
- exact indicator membership from the frozen manifest;
- Strict ⊆ Primary ⊆ Expanded ⊆ Broad T0;
- active-corpus paper alignment;
- no outcome or future-information columns.

## Outputs

- resumable `checkpoints/hgb/D{3,5,8}/{set}/fold_N.parquet`;
- `oof_predictions.parquet`;
- overall, fold, and domain metric CSV files;
- outcome-evaluation-only `model_comparison.csv` with
  `selection_feedback_used=false`;
- fold-valid long-form `paper_scores.parquet`;
- hashed `run_manifest.json` and `validation_report.json`.

The validator recomputes overall metrics, checks exact fold layouts and
temporal boundaries, verifies cross-set paper/fold/label identity, recomputes
paper percentiles, and re-hashes every declared artifact.

## Verification

- canonical adapter synthetic tests: 6 passed;
- existing simplified-protocol regression tests: 22 passed;
- Ruff: passed for the new runner, validator, and tests;
- Black: passed for the new runner, validator, and tests;
- mypy: passed for the new runner and validator modules.

Formal training remains blocked by the intentionally incomplete protocol audit
and absent frozen canonical matrices. No formal new-protocol HGB process was
started.

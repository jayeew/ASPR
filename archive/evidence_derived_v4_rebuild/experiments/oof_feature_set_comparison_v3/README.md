# Evidence-v3 feature-set OOF comparison

This experiment materializes four nested evidence-v3 indicator matrices from
the active `nature-multihorizon-uncapped-v2` Nature Portfolio dataset. The
shared paper-level feature universe contains primary articles through 2022;
training code applies the registered mature window for each outcome:

- D3: publication year through 2022;
- D5: publication year through 2020;
- D8: publication year through 2017.

| Model ID | Evidence rule | Expected indicators | Expected dimensions |
|---|---|---:|---:|
| `strict_7` | all 14 gates | 7 | 4 |
| `fulltext_16` | safe T0 gates + primary/foundational evidence + English full-text formula | 16 | 10 |
| `source_154` | safe T0 gates + primary/foundational evidence | 154 | 48 |
| `ultrarelaxed_221` | literature mention + safe T0 gates | 221 | 55 |

The four physical outputs are `indicator_matrix_7.parquet`,
`indicator_matrix_16.parquet`, `indicator_matrix_154.parquet`, and
`indicator_matrix_221.parquet`. They share the same unique `paper_id` key and
are checked against all three horizon-specific training cohorts.

## Operationalization policy

The relaxed sets intentionally do not require the original v3 local-data gate.
Every selected indicator therefore receives one auditable training column by
the following outcome-blind hierarchy:

1. a source-formula implementation recomputed from expanded-v1 views;
2. a documented local source-formula surrogate;
3. a structured publication-time construct proxy;
4. a title/taxonomy lexical construct proxy.

The audit table records the tier, source columns, coverage, cardinality, and
limitations for every indicator. The quality report additionally compares
missingness before and after 2018. A proxy is never reported as an original
formula implementation.

Only publication-time or strictly pre-publication inputs are used. Future
citations, future attention, D5 labels, and OOF results are forbidden during
feature construction.

## Run

```bash
python3 innovation_impact_feature_selection/evidence_derived_v3/experiments/oof_feature_set_comparison_v3/build_training_matrix.py
python3 innovation_impact_feature_selection/evidence_derived_v3/experiments/oof_feature_set_comparison_v3/run_hgb_comparison.py
python3 innovation_impact_feature_selection/evidence_derived_v3/experiments/oof_feature_set_comparison_v3/validate_hgb_results.py
```

The matrix builder reads the active dataset registry and fails if its quality
audit is not passing. The final runner trains only HGB for all four sets and
all three horizons. D3 uses eight forward folds through 2022, D5 uses seven
through 2020, and D8 uses six through 2017. The formal ASPR model is the
predeclared D5 target with Full-text 16, which has the best D5 OOF Spearman of
the four frozen feature sets. Final artifacts are written to
`outputs/hgb_uncapped_v2`.

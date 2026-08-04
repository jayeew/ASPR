# Evidence-v3 feature-set OOF comparison

This standalone experiment compares the four nested evidence-v3 indicator
sets on the same frozen Nature Portfolio D5 cohort:

| Model ID | Evidence rule | Expected indicators | Expected dimensions |
|---|---|---:|---:|
| `strict_7` | all 14 gates | 7 | 4 |
| `fulltext_16` | safe T0 gates + primary/foundational evidence + English full-text formula | 16 | 10 |
| `source_154` | safe T0 gates + primary/foundational evidence | 154 | 48 |
| `ultrarelaxed_221` | literature mention + safe T0 gates | 221 | 55 |

The experiment does not modify the frozen evidence-v3 database, its exports,
or ASPR business code. All new artifacts stay under this directory.

## Operationalization policy

The relaxed sets intentionally do not require the original v3 local-data gate.
Every selected indicator therefore receives one auditable training column by
the following outcome-blind hierarchy:

1. an existing source-formula implementation;
2. a documented local source-formula surrogate;
3. a structured publication-time construct proxy;
4. a title/taxonomy lexical construct proxy.

The audit table records the tier, source columns, coverage, cardinality, and
limitations for every indicator. A proxy is never reported as an original
formula implementation.

Only publication-time or strictly pre-publication inputs are used. Future
citations, future attention, D5 labels, and OOF results are forbidden during
feature construction.

## Run

```bash
python3 innovation_impact_feature_selection/evidence_derived_v3/experiments/oof_feature_set_comparison_v3/build_training_matrix.py
python3 innovation_impact_feature_selection/evidence_derived_v3/experiments/oof_feature_set_comparison_v3/run_oof_comparison.py
python3 innovation_impact_feature_selection/evidence_derived_v3/experiments/oof_feature_set_comparison_v3/verify_comparison.py
```

The OOF runner is checkpointed by horizon, model, and fold. Re-running resumes
only missing fits.


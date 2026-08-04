# Four-set D5 temporal OOF comparison

## Result

All four nested feature sets were trained with the same 118,057-paper D5
cohort, labels, six expanding-time folds, fixed `medium` model parameters, and
seed `20260724`. The six test folds contain 101,379 OOF predictions per model;
101,350 rows have a rank-valid realized D5 target.

| Rank | Feature set | Indicators | Dimensions | D5 OOF Spearman |
|---:|---|---:|---:|---:|
| 1 | Source evidence | 154 | 48 | **0.7675179552** |
| 2 | Ultra-relaxed | 221 | 55 | 0.7675011027 |
| 3 | English full-text formula | 16 | 10 | 0.7615953226 |
| 4 | Current strict | 7 | 4 | 0.7444429700 |

The 154-indicator set is the nominal winner. Its advantage over the
221-indicator set is only `0.00001685`; the year-block bootstrap interval for
`221 minus 154` is `[-0.00064404, 0.00101828]`. They should therefore be
interpreted as statistically indistinguishable, not as evidence that 154 is
intrinsically superior.

The 154-indicator set improves on the 16-indicator set by `0.00592263`, with a
95% year-block bootstrap interval of `[0.00050182, 0.01107565]`. The
16-indicator set improves on the strict set by `0.01715235`, with interval
`[0.01252551, 0.02046700]`.

## Fold results

| Test years | Strict 7 | Full-text 16 | Source 154 | Ultra-relaxed 221 |
|---|---:|---:|---:|---:|
| 1986–1999 | 0.779795 | 0.795969 | 0.799092 | 0.800712 |
| 2000–2004 | 0.777637 | 0.784529 | 0.789176 | 0.788114 |
| 2005–2009 | 0.824884 | 0.835874 | 0.842994 | 0.843763 |
| 2010–2012 | 0.799277 | 0.817519 | 0.828172 | 0.827843 |
| 2013 | 0.727977 | 0.745951 | 0.753586 | 0.752627 |
| 2014–2017 | 0.645565 | 0.668178 | 0.678039 | 0.677677 |

## Interpretation boundary

The evidence thresholds and the local-data thresholds are different. The
relaxed evidence sets contain indicators that did not pass the original
local-data gate. To preserve all requested indicator IDs without using future
outcomes, this experiment used the frozen operationalization hierarchy below:

| Set | Existing formula columns | Local formula surrogates | Structured construct proxies | Title/taxonomy proxies |
|---|---:|---:|---:|---:|
| Strict 7 | 7 | 0 | 0 | 0 |
| Full-text 16 | 12 | 4 | 0 | 0 |
| Source 154 | 12 | 4 | 79 | 59 |
| Ultra-relaxed 221 | 12 | 4 | 90 | 115 |

Consequently, the empirical comparison directly establishes which *implemented
feature-set scenario* has the higher OOF Spearman. It does not establish that
all 154 or 221 literature indicators have been reproduced from their original
formulas. Every proxy and its limitation is identified in
`outputs/operationalization_audit.csv`.

Given the nominal ranking, statistical tie with 221, and stronger evidence
boundary, the recommended main experimental set is **154 indicators across 48
dimensions**. The 221-indicator set is better retained as a high-recall
sensitivity analysis.

## Reproducibility evidence

- `outputs/completion_audit.json`: all completion checks passed.
- `outputs/oof_metrics.csv`: headline ranking.
- `outputs/oof_fold_metrics.csv`: fold-level results.
- `outputs/paired_year_bootstrap_comparisons.csv`: paired uncertainty.
- `outputs/oof_predictions.parquet`: all four models' paper-level OOF scores.
- `outputs/feature_sets.json`: exact indicator and dimension membership.
- `outputs/operationalization_audit.csv`: one row per indicator.
- `outputs/matrix_manifest.json` and `outputs/oof_run_manifest.json`: hashes and
  lineage.


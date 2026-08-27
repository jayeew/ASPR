# Fig.3 — Evidence-derived tuned HGB performance landscape

This directory implements the final uncapped-data Fig.3 using the HGB model
family exclusively.

数值输入通过 evidence-derived 的 `current_release.json` 解析到自包含的不可变
tuned release，不再读取 evidence v3 或旧 GEAR calibration 路径。

## Figure question

Can publication-time ASPR scores rank subsequent scientific uptake and
cross-field diffusion across three outcome horizons, four frozen feature sets,
twelve scientific domains, and continuous publication years?

The figure supports predictive association and screening. It does not claim
that any indicator causes later impact or that ASPR is a direct novelty label.

The refined renderer uses a 220 × 220 mm surface and reads the frozen panel tables
only. It does not refit models, rerun bootstrap draws, simulate values, or
recompute any displayed statistic.

## Panels

- **a — Score construction.** Primary 16 inputs enter the calibrated two-part
  HGB. The displayed score is the fold-valid D5 expected-diffusion prediction
  mapped to its 0–100 OOF percentile; it is not a probability or novelty label.
- **b — Multi-horizon, multi-set OOF enrichment.** A 3 × 4 small-multiple board
  draws twelve separate full decile curves for every D3/D5/D8 × Strict 7/
  Primary 16/Expanded 153/Broad T0 219 combination. Every curve includes frozen
  95% year-block bootstrap intervals and an exact D10 share/lift callout.
- **c — Performance landscape.** A 3-by-4 heatmap board shows three-year
  trailing domain-level OOF Spearman across D3/D5/D8 and Strict 7, Primary 16,
  Expanded 153, and Broad T0 219. The four heatmap columns use the full available
  panel width; no separate overall-OOF summary matrix is displayed.
- **d — D5 Primary16 terrain.** One continuous semi-transparent surface shows
  the D5 Primary 16 mature-year performance terrain. The surface is smoothed
  only between each domain's first and last reliable observed year: internal
  gaps may be linearly bridged for continuity, but endpoints are never
  extrapolated. Exact, uninterpolated values remain in Panel c. The three
  domains with the highest mean reliable D5 Primary 16 Spearman are labelled
  outside the mountain area with arrows to their peaks.
- **e — D5 gain landscape.** Adjacent nested-set differences show that the main
  local gain occurs from Strict 7 to Primary 16. Expanding further produces
  near-zero median gains and heterogeneous positive and negative cells.

## Frozen inputs

Numeric evidence comes only from the final uncapped-v2 HGB retraining output:

- `oof_predictions.parquet`
- `oof_metrics.csv`
- `paper_scores.parquet`

The model configuration is read only to verify that every training cutoff is
strictly earlier than its test interval. The source files contain only HGB
predictions; no alternative score family is part of the final analysis.

## Reproduction

```bash
python3 -m experiments.fig03.new.run --stage all
python3 -m experiments.fig03.new.tests
```

Stages can also be run separately with `--stage render` or `--stage audit`.
There is deliberately no data-building stage in the frozen v3 renderer.

## Main outputs

- `outputs/fig03/new/figure_full.{png,svg,pdf}`
- `outputs/fig03/new/figure_full_grayscale.png`
- `outputs/fig03/new/figure_full_deuteranopia.png`
- `outputs/fig03/new/panel_data/decile_enrichment.csv`
- `outputs/fig03/new/panel_data/performance_landscape.csv`
- `outputs/fig03/new/panel_data/d5_gain_landscape.csv`
- `outputs/fig03/new/panel_data/d5_gain_summary.csv`
- `outputs/fig03/new/audit_report.json`
- `outputs/fig03/new/run_manifest.json`

Any OOF row without a finite realized-diffusion label remains disclosed in the
model output but is excluded from target ranks, decile enrichment and local
correlations. Landscape `n` therefore means valid prediction–label pairs.

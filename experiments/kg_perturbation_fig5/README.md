# Fig. 5 Forecast Outcome README

Fig. 5 is a result-oriented forecasting experiment. It asks whether a knowledge
graph truncated at 2020 can recover research foci and seed innovations that
become important after 2020.

The implementation is deliberately separate from the Fig. 1-3 computation code:
it reads existing Fig. 3 paper-level scores and builds the five minimal tables
needed for the new Fig. 5 layout.

## Main Command

```bash
python -m experiments.kg_perturbation_fig5.fig5_forecast_outcomes \
  --domain-filter crispr \
  --out-dir outputs/kg_perturbation_fig5/crispr
```

Without `--domain-filter`, the script uses the multi-domain Fig. 3 input.

## Default Inputs

The script auto-detects the newest local Fig. 3 run, defaulting to:

```text
outputs/kg_perturbation_fig3/strong_evidence_tau10_v3/multi_domain/
outputs/kg_perturbation_fig3/strong_evidence_tau10_v3/fig3_input/multi_domain/
```

Required files:

```text
fig3_score_table.csv
works.csv
topics.csv
```

## Outputs

```text
fig5_predicted_focus.csv
fig5_realized_focus.csv
fig5_focus_alignment.csv
fig5_key_innovations.csv
fig5_backtest.csv
fig5_focus_map.csv
fig5_summary.json
fig5_run_config.json
fig5_full.png
fig5_full.svg
fig5_panel_a.png
fig5_panel_b.png
fig5_panel_c.png
fig5_panel_d.png
fig5_panel_e.png
```

The first five CSV files match the minimal Fig. 5 table plan. `fig5_focus_map.csv`
is an audit table for the map panel.

## Panel Mapping

- Panel a: cutoff and validation timeline.
- Panel b: predicted top focus list versus realized hotspot list.
- Panel c: topic-space map of forecast and realized emergence.
- Panel d: representative predicted key innovation cases.
- Panel e: historical backtesting against growth-only, citation-only, and random baselines.

## Strict Forecasting Note

The default demo reuses Fig. 3 publication-day scores. For a strict paper claim,
provide score tables learned only from information available before each cutoff.
The script keeps this layer auditable by writing `fig5_run_config.json` and
`fig5_summary.json`.

If local data end before the requested validation end year, the script keeps the
requested 2021-2026 setting in the metadata but labels the actual plotted
validation span according to the available data.

The same applies to the historical side: the conceptual Fig. 5 setting is
1950-2020, but the demo can only use the years present in the selected
`works.csv`.

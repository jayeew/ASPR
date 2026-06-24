# Fig. 5 Forecast Outcome README

Fig. 5 is a result-oriented forecasting experiment. It asks whether a knowledge
graph truncated at 2020 can recover research foci and seed innovations that
become important after 2020.

The implementation is deliberately separate from the Fig. 1-3 computation code.
The preferred workflow now produces an auditable data package first, then an
image-2 handoff package for drawing the four-panel publication figure. The older
`fig5_forecast_outcomes.py` script remains available for a deterministic
analytical figure.

## Preferred Data + Image-2 Workflow

Build the default multi-domain Fig. 5 data package:

```bash
python -m experiments.kg_perturbation_fig5.build_fig5_plot_data \
  --out-dir outputs/kg_perturbation_fig5/plot_data
```

Build the image-2 handoff package from those tables:

```bash
python -m experiments.kg_perturbation_fig5.build_fig5_image2_handoff \
  --plot-data-dir outputs/kg_perturbation_fig5/plot_data \
  --out-dir outputs/kg_perturbation_fig5/image2_handoff
```

For a CRISPR-Cas case-study figure that visually matches the supplied Fig. 5
reference more closely:

```bash
python -m experiments.kg_perturbation_fig5.build_fig5_plot_data \
  --domain-filter crispr \
  --out-dir outputs/kg_perturbation_fig5/crispr_plot_data

python -m experiments.kg_perturbation_fig5.build_fig5_image2_handoff \
  --plot-data-dir outputs/kg_perturbation_fig5/crispr_plot_data \
  --out-dir outputs/kg_perturbation_fig5/crispr_image2_handoff
```

The handoff package contains:

```text
fig5_image2_prompt.md
fig5_panel_text.json
fig5_visual_reference_notes.md
fig5_layout_draft.png
```

`fig5_panel_text.json` is the source of truth for figure text, rankings, scores,
and card copy. The drawing model should preserve that text and avoid inventing
new scientific claims.

## Legacy Analytical Figure Command

```bash
python -m experiments.kg_perturbation_fig5.fig5_forecast_outcomes \
  --domain-filter crispr \
  --out-dir outputs/kg_perturbation_fig5/crispr
```

Without `--domain-filter`, the script uses the multi-domain Fig. 3 input.

## Default Inputs

The data-package script auto-detects the newest local Fig. 3 run. In this
workspace it prefers:

```text
outputs/redraw_v6a_best_fig3/multi_domain/
outputs/redraw_v6a_best_fig3/fig3_input/multi_domain/
```

Required files:

```text
fig3_score_table.csv
works.csv
topics.csv
```

## Legacy Analytical Outputs

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

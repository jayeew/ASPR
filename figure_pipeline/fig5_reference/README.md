# Fig5 reference

**Historical six-panel version.** The accepted four-panel Fig5 is recorded in
`fig_reference/FINAL_VERSION.json` and delivered in
`outputs/fig1-8_reference_final_20261004/Fig5/`. Its renderer is
`figure_pipeline/fig5_mechanism_study/render_reference.py`, using
`outputs/fig5_mechanism_study/Fig5_formal_statistics.zip`. The commands below
regenerate the older six-panel version and are not the final delivery entry point.

Read-only plotting of completed Fig5 robustness data. Six panels, 15 components, 324 × 252 mm. Reuses existing labels; computes only descriptive plotting summaries.

```bash
/home/jayee/miniconda3/envs/openrlhf/bin/python -m figure_pipeline.fig5_reference all
```

Stages: prepare / render / package. Outputs: `outputs/fig5_reference/`. Full methodological caption and Chinese reading guide accompany the figure. No model calls or changes to experiment outputs.

# Fig.5 — four-panel corrective revision

Reuses `outputs/fig5_robustness`: 100 observational papers, 20 stress papers,
5 repeats. No report/branch generation. Original experiment outputs are read-only.
New outputs: `outputs/fig5_four_panel`.

```bash
/home/jayee/miniconda3/envs/openrlhf/bin/python -m figure_pipeline.fig5_four_panel.prepare
/home/jayee/miniconda3/envs/openrlhf/bin/python -m figure_pipeline.fig5_four_panel.evaluate
# Only after primary evaluation ends: one retry for recorded capacity failures.
/home/jayee/miniconda3/envs/openrlhf/bin/python -m figure_pipeline.fig5_four_panel.recover
/home/jayee/miniconda3/envs/openrlhf/bin/python -m figure_pipeline.fig5_four_panel.aggregate
/home/jayee/miniconda3/envs/openrlhf/bin/python -m figure_pipeline.fig5_four_panel.render
/home/jayee/miniconda3/envs/openrlhf/bin/python -m figure_pipeline.fig5_four_panel.validate
/home/jayee/miniconda3/envs/openrlhf/bin/python -m figure_pipeline.fig5_four_panel.export
python3 -m pytest -q tests/innovation_v2/test_fig5_four_panel.py
```

Evaluation has 20 ordinary requests plus at most 10 additional requests, including
capacity splits and failed attempts. Preflight requires 24 requests for this roster.
All evaluation conditions use gpt-6.1-sol/high and preserved anonymous report IDs.
Reference labels, applicability and grouping do not depend on candidate performance.
The 19 explicitly enumerated material-insufficiency corrections were reviewed from
candidate-free reference reasons before reading corrective evaluation results.
Original labels and reasons are retained in `frozen_history_groups.csv`.

Panel a uses existing Fig.4 evaluations. Panels b–d use corrected evaluations.
Scientific uncertainty is not omitted discussion; scope caution is not abstention.
K5 is evaluated against its own native graph. Only complete technical pairs enter
means; scientific unknowns remain visible within fixed denominators. Common-cost
cohort membership requires all four evaluations and complete generation cost logs.
A missing original E50 report is retained as missing, never replaced or scored zero.

Deliverables include PNG, editable SVG, vector PDF, four source tables, all panel
summaries, original corrective responses and mappings, caption, Chinese audit notes,
and a compact ZIP. See the generated notes for actual request and sample counts.

## Reference-style visual revision

The current figure uses grouped capability profiles, paired paper trajectories with
mean confidence intervals, answer-part transition flows, and quality–cost displacement
with fully named configuration cards. Fonts, rounded cards, header bands and the
purple/blue/orange/teal palette follow Fig.1–4 reference artwork, specifically the
current Fig.4 renderer in `fig4_explanation_reference`. No gray footer notes or
letter-coded metric/configuration labels are rendered. The prior artwork is retained
in `outputs/fig5_four_panel/previous_layout`. Source statistics and model evaluations
are unchanged; observed transition counts are derived from the existing paired parts.

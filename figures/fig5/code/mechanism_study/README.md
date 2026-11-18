Final accepted figure: `fig_reference/Fig5.png`, `.pdf`, and `.svg` (420 × 218 mm). Reproduce with `python3 -m figure_pipeline.fig5_mechanism_study.render_reference`. Superseded plotting scripts and exports were removed at the user’s request; scientific source artifacts remain available.

# Fig.5: model-stage diagnostics and evidence interventions

Authorized on 2026-10-04 (Asia/Shanghai). Original figure and experiments remain
unchanged. New files are under `outputs/fig5_mechanism_study/`.

## Scientific questions

1. Does the observed replacement-model loss originate in branch analyses,
   report synthesis, or their interaction?
2. Does grounded coverage change with evidence availability and native graph
   availability? Do critical deletions differ from source-count-matched controls?
3. Does fresh generation after restoring original evidence recover coverage?

No required ranking, desired curve, or retrospective noninferiority margin is
specified. Actual counterevidence, failures and unresolved evaluations are kept.

## Diagnostic phase

The original 20 stress papers are a previously inspected diagnostic cohort.
Cross the original/replacement branch analyses with original/replacement writers.
All four cells have identical original evidence, native graph, public tasks,
writer prompt and requested effort. The two same-model cells reuse original
reports; 40 cross-model reports are newly generated. Raw writer inputs are checked
for exact equality after excluding only the two branch-interpretation fields.
All four cells receive a fresh anonymous evaluation with the corrected taxonomy.
The protocol and hashes are saved before dispatch. Hard cap: 200 new calls.

## Intervention phase

Following the user’s explicit amendment, use all 100 papers as one primary
cohort for full versus half-source and original versus replacement-analysis
comparisons. Fill the same conditions for the five previously separated papers;
retain exploratory provenance without dividing the main results into 5 and 95.
These are not newly collected papers or an untouched test cohort. The previous 20 exploration papers additionally receive
75%/25% historical-source and 50%/25% graph-parent availability conditions.
Each exploration paper also receives two additional fixed half-source random
masks; these three outcomes are averaged within paper rather than treated as
60 independent papers. Seeds are fixed before dispatch and never rerolled to
obtain stronger effects.
Ten preselected papers receive order changes and two additional independent
full-condition repeats. Forty papers are considered for critical deletion;
eligibility is determined from materials and conditional references, never from
candidate quality. Each eligible paper receives critical deletion, equal-source-
count noncritical deletion, and fresh full-material restoration generation.
Restoration is an independent fresh run, not a conversation-level self-repair.

Existing controls match independent source counts, not necessarily text length
or source type. Report those imbalances instead of describing them as perfectly
matched deletions. New half-source masks use seed 20261004 and preserve complete
independent-work aliases across both branches. Prior established masks remain.
Native historical assets are read-only. All graph changes use existing locally
recomputed graph snapshots, with source texts unchanged.

Adopt reports only when saved actual writer inputs match the new packet and
public tasks (floating-point tolerance 1e-12 only). Adopt branch outputs only if
their corresponding saved inputs match. Conditional references must preserve
all applicable task/part identities and refer to visible sources. Do not inherit
the interrupted old call ledger or restart its automation. New call cap: 1650,
including 150 reserved recovery/audit calls; this is a ceiling, not a target.

## Analysis and completion

Report fixed-denominator grounded coverage, incomplete answers, reasonable
abstention, confirmed unsupported assertions, and evaluator uncertainty separately.
Technical missingness is never scientific zero. Pair at the paper level;
within-paper parts/repeats are not independent sample-size increments.
Inspect nested-reference conflicts before final comparisons; record any repair
and re-evaluate affected conditions. Preserve original model responses.

A symmetric five-paper structured-writer pilot found no consistent benefit;
its prompt was not adopted. The unified 100-paper comparison retains the
original writer prompt. The original 20-paper four-cell diagnostic remains a
subset analysis, and its hybrid reports are reused in the 100-paper paired
evaluation with the same evaluation protocol used for the other papers. Independent model checks are not human gold labels;
actual human review remains a separate deliverable requiring human participation.

Final figure preserves Fig.1–4 fonts, colors and rounded panel cards, uses full
configuration labels, and has no gray footnotes beneath charts. Do not present
incomplete experiment output as the final revised figure.

```bash
python3 -m figure_pipeline.fig5_mechanism_study.diagnostic prepare
python3 -m figure_pipeline.fig5_mechanism_study.diagnostic run
python3 -m figure_pipeline.fig5_mechanism_study.interventions prepare
python3 -m figure_pipeline.fig5_mechanism_study.interventions run
python3 -m pytest -q tests/innovation_v2/test_fig5_mechanism_study.py
```

Unified-cohort amendment: `cohort100.py`; main model call ceiling 400 (350 ordinary
plus 50 recovery). One provider-restricted paper remains planned but receives no
further calls; it is never scored as scientific zero. Final analysis commands:

```bash
python3 -m figure_pipeline.fig5_mechanism_study.analyze
python3 -m figure_pipeline.fig5_mechanism_study.render_reference
```

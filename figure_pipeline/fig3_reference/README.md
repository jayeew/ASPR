# Figure 3

The current figure uses the complete prepared 100-paper study in
`outputs/fig3_revision/derived/`: 717 scientific shared claims (two author-role/funding statements removed), source-selected assessment cores,
600 reports from six methods, and 500 AB/BA preference pairs. It does not
read the historical 200-paper results under `outputs/fig3_reference/data/`.

```bash
/tmp/aspr-fig3-venv/bin/python -m figure_pipeline.fig3_reference all
/tmp/aspr-fig3-venv/bin/python -m figure_pipeline.fig3_reference render --panel e
```

The plotting environment supplies Matplotlib, pandas, NumPy, and CairoSVG.

Rendering directly overwrites `outputs/fig3_reference/final/Fig3.svg`,
`Fig3.pdf`, and `Fig3.png`, together with the eight panel exports in
`outputs/fig3_reference/panels/`. SVG text remains editable; PDF is vector;
PNG is the preview. The master is 420 × 760 mm. Layout and method colors are
in `outputs/fig3_reference/layouts/style.json`; typography and blue panel
frames follow Fig1/2. No model calls, tests, hashes, manifests, or result
versions are created by rendering.

All 40 component positions are backed by current prepared data. Paper-level
plots use actual `paper_metrics.csv` observations, while summary estimates
and intervals use existing component/summary tables. Only the displayed
paired Full-minus-union interval is calculated at draw time, resampling the
100 paper differences 10,000 times with seed 20260922. Counts, proportions,
quality scores, missing denominators, and model-assisted judgments remain
separate. Historical PR has far fewer estimable papers than the full cohort.

The c6 and f5 excerpts display actual original quotations, abbreviated only for space. The e5 panel includes only explicit same-reviewer follow-up in a strictly later round; author replies and general appreciation cannot establish resolution. Counts and missing denominators are read from the current data.

The current repair changes GEAR, Graph and Full generation, while preserving Direct-A, EACL and ReviewGrounder reports. Shared references and evaluation are repaired for the comparison. Full alone has no report-length target; this difference is disclosed in the figure. Panel g distinguishes genuinely new content from branch content that becomes supported, and panel h distinguishes joint-field from individual-field repeat agreement.

Historical downloads that return HTML or browser notices are not full text. Their original abstracts or bibliographic metadata are retained, with unavailable full text explicit; affected GEAR judgments, Full fusion and shared evaluations are recomputed from the corrected material.

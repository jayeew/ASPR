# Fig.1 — Landmark fields and publication-time innovation signals

**Question.** How do four locked landmark fields evolve in their local
knowledge graphs, and which registered innovation indicators change most
across the same field-specific time windows?

**Data.** CRISPR–Cas, graphene, iPSC, and microbiome/metagenomics frozen graph
views plus the current Nature v6.1 publication-time feature materialization.
The experiment does not fetch external data. Exoplanets are not shown because
the frozen 1990–2004 astronomy slice has almost no calculable reference-based
indicator values; it is replaced rather than imputed.

**Figure.** Each field is one row. The four left columns are fixed-layout graph
snapshots restricted to seven temporally representative topics; the right
column shows four or five registered indicators. A candidate must be complete
in all four windows, after which indicators are ranked by the range of their
four window medians. A deterministic largest-gap rule chooses four or five.
Neither D5 labels nor OOF performance enters this selection.

```bash
python3 -m experiments.fig01.new.run --stage all
python3 -m experiments.fig01.new.tests
```

The graph snapshot code is reused only for configuration parsing, layout, and
drawing. No old perturbation metric is computed, loaded, renamed, or plotted.
The figure has two explicitly aligned lenses: frozen landmark-field graphs on
the left and matching `domain12`/OpenAlex-primary-subfield slices from the
Nature v6.1 feature table on the right. They are not claimed to be the same
paper sample.

For display, every raw indicator is converted to an innovation-oriented
percentile within its `domain12 × publication_year` cohort. Each plotted point
is the indicator median for that window. All graph and indicator windows end by
the frozen 2017 feature horizon, so no plotted trajectory is truncated or
imputed.

The audit fails if the eight features differ from the frozen contract, a
registry role is not `primary`, an indicator crosses its publication-time
boundary, a graph snapshot is empty, a plotted indicator cell is missing, or
any overview/detail artifact is absent. Outputs include `figure_full.*` plus
four bundles under `outputs/fig01/new/domains/<slug>/`. The result is
descriptive evidence, not a causal event-study estimate.

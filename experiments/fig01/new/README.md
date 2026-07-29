# Fig.1 — Landmark-field knowledge-graph transitions

## Scientific role

Fig.1 is a descriptive mechanism-motivation figure. It shows how
publication-window topic-coupling structure and frozen publication-time
innovation indicators vary around landmark-paper windows in four deliberately
selected high-change fields.

It is **not** a causal event study and is not intended to represent all fields.
The cases were deliberately selected to make graph and indicator changes
visible. The fourth case was refreshed after an auditable screen of all 11
registered landmark episodes so that every displayed field has at least three
visible indicator contrasts. This descriptive visual-selection step is
disclosed in the figure contracts.

No D5 label, OOF prediction, citation outcome, or future-impact variable is
read by the canonical Fig.1 pipeline.

## Landmark registration and display-case selection

`is_landmark` is not inferred from citation count or from the eight innovation
indicators. It is exact membership in the DOI/OpenAlex-ID-backed v4 landmark
registry. Registry candidates were carried forward from the strict v3 list,
trusted award/expert/manual seeds, and the v4 seed table. Main rows require a
nonblank domain, DOI and year, `include_main=1`, no more than three registered
papers per domain, and a year no later than the registry cutoff. OpenAlex
validation then requires matching DOI, publication year within one year, and
title similarity of at least 0.55. Only `validation_status=passed` rows are
materialized as landmarks.

This validation proves paper identity, not landmark status. The former
manual-only ipilimumab case has therefore been removed. The displayed
replacement is the Wellcome Trust Case Control Consortium's 2007 Nature GWAS
of 14,000 cases and 3,000 shared controls (`10.1038/nature05911`). Its
landmark status is supported independently by a contemporaneous
peer-reviewed Molecular Psychiatry editorial (`10.1038/sj.mp.4002117`) and a
later peer-reviewed history that takes this study as the field-scale starting
point (`10.1016/j.ajhg.2011.11.029`). Evidence strength for the four displayed
cases is now: graphene `nobel_and_field_consensus`, GWAS
`field_consensus`, click chemistry `award_or_field_consensus`, and CRISPR
`field_consensus_manual_doi`.

After registry inclusion, Fig.1 makes a separate **display-case** choice. Four
episodes are frozen from the 11 registered episodes for visible graph and
indicator change. That selection is exploratory and does not redefine which
papers are landmarks.

## Frozen cases and indicators

The current selection is versioned in
`experiments/fig01/new/frozen_selection.json` and protected by SHA-256. It is
frozen before the current render and cannot be changed by the renderer.

| Case | Landmark window | Frozen indicators |
|---|---:|---|
| CRISPR–Cas | 2012–13 | 4 |
| Graphene and 2D materials | 2004 | 5 |
| Click chemistry (CuAAC) | 2002 | 5 |
| Genome-wide association studies | 2007 | 4 |

The first three cases retain their established frozen selections. The fourth
case is the DOI-validated 2007 WTCCC paper. It has an exact landmark topic,
complete graph transitions, and three of its four frozen change-ranked
indicators have an absolute late-minus-pre contrast of at least 9 percentile
points. The fourth frozen candidate is retained in the audit table but hidden
from paired panels b/c by the shared display rule. This replacement was made
after inspecting evidence strength, recency, data sufficiency and the
descriptive results, so it is explicitly a versioned exploratory display
update—not a prospectively registered validation. No D5, OOF, citation
outcome, current citation count or future-impact label was used.

GWAS was already one of the ten fields in the frozen corpus and the frozen
v6.1 feature table. No OpenAlex supplementation and no new experimental
dataset were needed for Fig.1 v6.4.

The graph panel uses four cumulative local knowledge states. The first state
contains six pre-landmark years; each later state adds one non-overlapping
three-year block:

1. `t−6` through `t−1`;
2. all papers through `t+2`;
3. all papers through `t+5`;
4. all papers through `t+8`.

This restores the knowledge-state interpretation used by Fig.1 old. All four
displayed cases contain approximately 600, 900, 1,200 and 1,500 papers per
cumulative snapshot. Panels b and c remain annual/non-cumulative.

## Panel a — fixed-layout transition networks

Papers are linked by cosine-normalized bibliographic coupling and aggregated
to OpenAlex primary topics. Every field uses one union skeleton and fixed topic
coordinates across all four windows.

- pale dotted edge: union-skeleton context;
- dark solid edge: retained;
- amber solid edge: gained;
- magenta dashed edge: lost relative to the previous window;
- amber outline: newly active topic;
- red star and outline: landmark-bearing topic, absent before `t0`, introduced
  in the landmark snapshot and retained thereafter.

Topics that are inactive in a snapshot remain in the frozen union-layout audit
data but are not drawn. Edges are likewise restricted to pairs of topics active
in that snapshot, so no visible line terminates at a hidden topic.

Halo area is proportional to the complete paper count assigned to a displayed
topic. Up to five small beads are real, deterministically selected
representative papers; they are not all papers in the halo.

The combined figure uses eight topics and at most 16 active edges per window.
The four larger field figures use 12 topics and at most 24 active edges.
The renderer deliberately restores the visual language of Fig.1 old—stronger
community halos, representative-paper spokes and beads, curved weighted
backbone edges, light snapshot frames, and progression arrows—without
restoring its historical metric formulae. Each displayed topic now has its own
persistent colour; topics in the same OpenAlex subfield are no longer collapsed
into one colour. A paper-volume-aware collision pass prevents large topic halos
from occupying the same centre.

The displayed topic set is deterministic. It always includes the
landmark-bearing OpenAlex primary topic, interleaves endpoints of the strongest
coupling edges from all four snapshots, and fills remaining slots by paper
volume plus weighted coupling strength. The compact main panel then chooses the
eight-topic subset that maximizes the weakest snapshot's edge support. Text
labels are more selective: after the landmark topic, they prioritize active
topics directly coupled to it, ranked by coupling weight, and then high-volume
field-backbone context. These topics share reference patterns with the
landmark community; they are not necessarily descendants of, cited by, or
caused by the landmark paper. Stage-specific roles are auditable in
`community_label_selection.csv`.

## Panels b and c — annual trajectories and effects

Each selected indicator has 15 annual slots from `t−6` through `t+8`.
Indicators are oriented by their frozen direction and converted to percentiles
within publication year across the ten candidate domains.

The plotted annual value is:

```text
annual median percentile
− median of the annual medians at t−3, t−2, and t−1
```

The ribbon is the annual 25th–75th percentile range shifted by the same
baseline. A year with fewer than 15 valid papers is missing and is never
interpolated.

To avoid compressing most trajectories inside a fixed `−0.70` to `+0.70`
range, a candidate symmetric limit is first calculated for every indicator.
Within each field, all displayed indicators then use the largest of those
candidate limits, so their vertical magnitudes remain directly comparable.
The shared `±xx pp` range is shown once on the first strip's left y-axis and is
frozen in `trajectory_display_scales.csv`. Each candidate limit is the larger
of:

1. the largest absolute annual median plus 5 percentile points;
2. the 90th percentile of the absolute annual IQR bounds;
3. a 20-percentile-point minimum.

It is rounded upward to 5 percentile points and capped at 70 percentile
points. Open boundary triangles mark annual medians or IQR bounds outside that
explicit range. This changes only the display scale, not the values. The
aligned forest column retains one common `−70` to `+70` percentile-point scale
so cross-indicator magnitudes remain directly comparable.

The aligned forest column estimates:

```text
late post (t+6:t+8) − pre landmark (t−3:t−1)
```

Within each of the six publication years, papers are resampled with replacement
for 2,000 deterministic bootstrap draws. Each three-year window is summarized
as the equal-weight mean of its three year-specific medians. The displayed
95% percentile intervals are post-selection descriptive intervals, not
confirmatory significance tests.

Panels b and c use the same display-only filter. Because the two panels are
paired, a registered indicator is shown only when its absolute late-minus-pre
effect is at least 9 percentile points. Annual median peak-to-peak range is
still recorded as a diagnostic, but cannot by itself retain a weak late-minus-
pre contrast. At least three and at most four indicators are retained per field
using a deterministic salience ranking. This filter changes only Fig.1 display
density: all frozen indicator decisions, values and intervals remain in the
panel tables and no model feature is removed.

## Commands

```bash
python3 -m experiments.fig01.new.run --stage prepare
python3 -m experiments.fig01.new.run --stage run
python3 -m experiments.fig01.new.run --stage plot
python3 -m experiments.fig01.new.run --stage audit

# Complete canonical run; it reads only the frozen local corpus and performs
# no network fetch.
python3 -m experiments.fig01.new.run --stage all

# Unit checks only.
python3 -m experiments.fig01.new.tests --unit-only

# Full run followed by all artifact checks.
python3 -m experiments.fig01.new.tests --full
```

## Outputs

Main and field figures:

- `outputs/fig01/new/figure_full.{png,svg,pdf}`;
- `outputs/fig01/new/domains/<domain>/figure_<domain>.{png,svg,pdf}`.

The main figure is exported at exactly 183 × 168 mm. PNG output is 600 dpi;
SVG remains editable and PDF embeds DejaVu Sans.

New auditable data tables include:

- `case_refresh_audit.csv`;
- `annual_indicator_trajectories.csv`;
- `trajectory_display_scales.csv`;
- `indicator_display_filter.csv`;
- `indicator_effects.csv`;
- `indicator_effect_bootstrap.parquet`;
- `representative_papers.parquet`;
- `transition_edges.parquet`;
- `topic_label_audit.csv`;
- `community_label_selection.csv`;
- `frozen_selection_snapshot.json`.

The immediately preceding v6.1 result is preserved in
`outputs/fig01/new/archive_no_inactive_v6_1_20260727/`. Earlier archives,
counterfactual outputs and the GPT layout schematic are not modified or used
as numerical inputs.

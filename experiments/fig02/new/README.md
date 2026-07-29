# Fig.2 — Evidence-governed five-dimension, eight-indicator system

## Scientific question

Fig.2 explains why ASPR v6.1 observes paper innovation through five
publication-time dimensions, why the final implementation contains these eight
primary indicators, and how every decision can be traced to a registered
candidate universe, explicit screening rules and sources.

The active Fig.2 path is deliberately outcome-free. It does not read D5, OOF,
future citations, known-group labels or the former `G−/G0/G+5` scene. Indicator
selection is therefore not optimized against predictive performance.

## Four panels

- **a — Candidate scope and selection.** The upper chain conserves the frozen
  screening counts `50 → 30 → 20 → 18 → 8`. The Alluvial accounts for all 50
  candidates across five observation dimensions and four registered roles
  (`8 primary / 18 sensitivity / 4 exploratory / 20 excluded`). G1–G6 and the
  numerical coverage, stability and approximation gates state the selection
  rules.
- **b — Dimensions, indicators and relations.** Five equal-width outer sectors
  encode observation dimensions; eight equal-width inner nodes encode the
  primary indicators. Exactly seven frozen direction-oriented Spearman
  relations with `|ρ| ≥ 0.40` are drawn. I1 remains isolated at this threshold.
  Equal arc widths prevent the diagram from implying unequal theoretical
  importance.
- **c — Meaning, boundary and sources.** Five source strips give a concise
  meaning, inclusion boundary, exclusion boundary, three key author–year
  sources and the complete registered source count for each dimension. The
  dimensions are evidence-backed observation lenses, not independent causal
  mechanisms.
- **d — Formula and quality ledger.** Eight rows report the registered display
  formula, direction, formula/application/validation counts and four separate
  quality axes. Thresholds are `overall coverage ≥ .70`, `weakest field ≥ .50`,
  `resampling ρ ≥ .90`, and `MRE ≤ .10`. I2 is the only lower-is-stronger
  indicator and the only approximation whose exact-reference agreement is
  shown.

Complete formulas, full dimension definitions, all candidate decisions and
source records remain in `panel_data/`; the rendered panel uses concise display
copy to avoid hiding the evidence structure.

## Rendering stack

The figure is static and fully reproducible:

```bash
python3 -m pip install --user --break-system-packages \
  -r requirements-figures.txt
```

- Matplotlib: layout, Alluvial, source strips, evidence ledger and export.
- pyCirclize: equal-sector circular map and relation ribbons.
- adjustText: relation-value collision avoidance only.
- colorspacious and Pillow: color-vision-deficiency and grayscale QA previews.

The canonical master is `6400 × 5200 px` at 600 dpi. SVG and PDF remain
editable and can be scaled for a journal layout without rerasterization.

## Commands

```bash
python3 -m experiments.fig02.new.run --stage all
python3 -m experiments.fig02.new.tests --full
```

Canonical outputs:

```text
outputs/fig02/new/figure_full.{png,svg,pdf}
outputs/fig02/new/panels/fig02_{a,b,c,d}.{png,svg,pdf}
outputs/fig02/new/panel_data/
outputs/fig02/new/qa/
outputs/fig02/new/{panel_text,chart_contract,run_manifest,audit_report}.json
```

## Claim boundary

The figure supports source traceability, transparent candidate scope,
reproducible selection and observable indicator relations. It does not prove
innovation truth, independence of dimensions, causality or predictive
validity.

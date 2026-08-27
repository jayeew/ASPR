# Fig.2 — Evidence-derived dimensions and feature sets (v3)

Fig.2 explains **what is measured and how that measurement set was fixed**. It
does not show predictive performance, OOF correlations, future citations or the
old v6.1 five-angle/eight-indicator system.

## Figure logic

- **a — Evidence pipeline.** A domain-agnostic English bootstrap query is
  expanded only with author-used terms. The panel records 12 discrete review
  batches, AI–H1–H2 coverage, query/PRESS/seed validation and the full ordered
  chain from terms to dimensions. Round 12 is explicitly labelled as a
  pragmatic marginal-yield amendment (`Δterms=10`, `Δindicator families=9`),
  not as a dual-zero saturation event.
- **b — Classification map.** A quantity-conserving alluvial maps all 432
  canonical indicator families from their extraction role to candidate
  dimension role and then to their exclusive nested-set tier. It is a
  classification map—not a correlation, causal, or importance diagram. The
  inset lists the strict `7 metrics → 4 operational dimensions` mapping.
- **c — Gates and retention.** Fourteen marginal gate-pass counts are shown
  over the shared 432-family denominator. They are intentionally not a funnel:
  an indicator may fail more than one gate. The panel separately shows the
  formal dimension rule: a passing metric, two independent research teams, and
  H2 non-alias confirmation.
- **d — Operating sets.** The strict core, full-text expansion, source-grounded
  expanded set and broad T0 sensitivity ceiling are nested and frozen before
  outcome evaluation. Their implementation tiers are shown separately from
  model roles: direct source formulae, transparent local formula surrogates,
  structured construct proxies and title/taxonomy lexical proxies. In
  particular, the 153-feature expansion is not 153 formula replications.

## Inputs and guardrails

The adapter reads only frozen files under
`innovation_impact_feature_selection/evidence_derived_v3/`, its registered
recovery bundle, the frozen process record in `docs/`, and the tuned membership
manifest. It does **not** read OOF metrics, predictions, fold data, targets or
future outcomes. The source-of-truth result is
checked against:

- `K=42`, `Q=336`, `P=367`, `M=66`, `D=1`, `F=7`;
- `3,615 → 3,170 → 1,102 → 367` terms/families;
- `1,685` census mentions (plus 13 targeted formula-completion records) and
  `432` canonical indicator families;
- nested set sizes `7 / 16 / 153 / 219` and dimensions `4 / 10 / 48 / 55`.

The Figure and its caption must retain three limitations: English-only evidence
can create language/geographic bias; evidence-saturation retrieval is not an
exhaustive database census; and H2 adjudication, rather than raw agreement,
sets final coding labels. The audit ledger also distinguishes seven
human-attested automated draft worksheets from 119 registered independent
Codex-AI review runs; isolated local-Qwen artifacts are excluded from every
reported final count.

## Reproduce

```bash
python3 -m experiments.fig02.new.tests
python3 -m experiments.fig02.new.tests --full
python3 -m experiments.fig02.new.run --stage all
```

The retained publication master is:

- `outputs/fig02/new/Fig2_evidence_architecture.svg`

The command regenerates PNG/PDF copies, panel crops and audit files when needed
for validation; they are intentionally not retained in the final Fig.2 output
directory. The master is laid out for a 183-mm-wide full-page placement.

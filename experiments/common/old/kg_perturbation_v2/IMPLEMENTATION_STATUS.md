# Fig.1--Fig.10 V2 migration status

The shared evidence boundary and V2-native renderers are implemented. All V2
figures require an explicit, audited
`release.json`. The release publisher creates `figure_views/fig01` through
`fig10` before publishing and hashes every CSV, panel specification, caption
statistic, and view manifest. Drawing cannot access OpenAlex, raw checkpoints,
or legacy Fig.3 directories.

The standardized views currently cover:

- Fig.1: five-mechanism trajectories;
- Fig.2: 8+10 feature quality, 8-to-5 mapping, and RGPM-D5/S5 relations;
- Fig.3: all-model OOF, nested selection, three horizons, temporal tests, and
  uncertainty;
- Fig.4: score strata from the dual-score contract plus a provenance-bound
  peer-review external-validity table;
- Fig.5: primary-horizon forecast-score handoff; the AI-frontier join/backtest
  remains a Wave-B run after a real release exists. Development papers use
  OOF scores, sealed papers use temporal-holdout scores, and full-fit scores
  are explicitly separated as descriptive forecasts;
- Fig.6: registered horizon, citation-threshold, graph/community, feature,
  calibration, seed, and fold sensitivity rows; missing runs remain explicit
  placeholders rather than borrowing the Fig.3 model comparison;
- Fig.7: Nature Portfolio venue-family score, diffusion, enrichment,
  five-mechanism, time-migration, and controlled-inference panels, with
  domain-by-period cells requiring at least 30 papers;
- Fig.8: the dual-score ASPR architecture contract;
- Fig.9: the pre-existing fixed DOI case, never a score-selected case; cached
  graph-agent/Qwen/fusion evidence still requires the Wave-C rerun. The fixed
  DOI is not present in the current 211,073-paper V5 target inventory, so the
  base view records `not_in_source_corpus` and draws no zero-valued
  pseudo-profile. `materialize_wave_evidence.py score-case` supplies the formal
  frozen-model out-of-cohort profile once its publication-prior features exist;
- Fig.10: explicit registered ablations only; model-family rows from Fig.3 are
  never relabelled as ablations, and ASPR component ablations remain a Wave-C
  run. `materialize_wave_evidence.py package-table` enforces the complete locked
  ID matrix and provenance assets before those rows enter a release.

Every view now carries a separate scientific `claim_readiness` contract.
Placeholder panels may be rendered only with `--allow-incomplete`, producing a
`_DRAFT` bundle. The final assembler rejects them by default and can never
write `_SUCCESS` for an incomplete draft.

The command below validates the complete view and its hashes, then draws:

```bash
python3 experiments/common/old/kg_perturbation_v2/run_figure.py \
  --release outputs/nature_multihorizon_v1/analyses/<analysis_id>/release.json \
  --figure 3 --draw-only
```

Legacy experiment directories and outputs remain read-only. They are not
baselines, fallback inputs, or model-selection sources for V2.

No real candidate/frozen release or OOF number is claimed yet, but the raw-data
blocker has been removed. The reference closure now covers 3,421,132 of
3,498,552 references (97.787%). The completed common future layer contains
131,777 papers at each of τ=3/5/8 and 22,871,558 detailed future-citer rows.
131,772 requests succeeded; five explicit `missing_checkpoint` papers remain
NA and are excluded from every modeling cohort.

The upstream future report intentionally remains `overall_pass=false` because
it is not 100% complete. The new offline adapter records that value unchanged
and separately records `accepted_for_training=true` under the locked maximum
of five explicit failures. The default common-cohort `run` now imports these
tables without OpenAlex access. A frozen paper release is still prohibited
until the expanded recent-paper windows, sealed/strict tests, τ5 structural
subset, and all quality gates pass.

The current 61-test implementation suite passes and includes dedicated tests for bounded incomplete
future imports, failure-versus-zero handling, precomputed RGPM-D targets, and
streaming structural reads. Stage
and release success markers bind complete manifest identities, candidate
publication requires explicit dataset/analysis IDs, and frozen promotion
requires an explicit audited candidate path. Raw multi-GB closure tables stay
in immutable stage storage rather than being recopied into every evidence
release.

The real 211,073-paper source taxonomy audit maps 195,734 papers into the 12
natural-science domains, excludes 14,644 non-natural papers, and leaves 695
unmapped (99.646% in-scope coverage); 194,320 mapped papers use the official
OpenAlex field hierarchy rather than keyword fallback. Cap-hit papers are
explicitly flagged, each horizon has a 2% maximum cap-hit gate, and τ=5 must
retain its OOF conclusion after cap-hit rows are excluded.

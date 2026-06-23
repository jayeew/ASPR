# Publication Corpus Evidence Bundle

This bundle records the current data-layer state for making Fig1-Fig5 more publication-ready while keeping the figure drawing logic fixed.

## Current State

- Stable 11-domain corpus: `data/knowledge_corpus/v2_publication`
- Latest Fig3-aware candidate corpus: `data/knowledge_corpus/v2_publication_fig3aware12`
- Fig3-aware candidate audit: `outputs/publication_corpus_v5_fig3aware_candidate_audit`
- Fig3-aware diagnostic run: `outputs/kg_perturbation_fig3_v2_publication_fig3aware12_core/multi_domain`
- Fig5 forecast-score runs:
  - `outputs/kg_perturbation_fig5_v2_publication_fig3aware12_forecast_scores_minrefs5`
  - `outputs/kg_perturbation_fig5_v2_publication_fig3aware12_forecast_scores_minrefs1`
- Figure policy: fixed consumer contract; no Fig1-Fig5 plotting-logic rescue.

## Corpus Gates

- Fig3-aware candidate domains: 12
- Domain families: biology/biomedicine 5, materials/chemistry 4, physics/astronomy 3
- Corpus size: 35,050 works
- Citation rows in Fig3/Fig5 views: 522,971
- Clean landmarks in materialized manifest: 18
- Strict corpus audit: `overall_pass=True`
- Main data intervention: closure-aware OpenAlex top-up for mass spectrometry and ubiquitin/proteasome, plus Fig3-aware replacement of autophagy with gamma-ray bursts.

## Fig3 Evidence

The Fig3-aware 12-domain run improves the 11-domain result but still fails the publication gate.

- Learned OOF Spearman: 0.370
- Equal-weight OOF Spearman: 0.210
- Learned vs equal delta: +0.160
- Best single metric OOF Spearman: 0.336
- Learned vs best single delta: +0.033
- Nonlinear upper bound OOF Spearman: 0.380
- Active graph deltas: 8
- Contributing graph deltas: 3
- High vs low tertile RGPM median lift: +35.0 pp
- Top vs bottom score-decile RGPM top20 enrichment: 5.41x
- Quality report overall pass: false

Interpretation: data construction moved Fig3 in the right direction and kept the effect-size panels persuasive, but it is not yet Nature-level evidence. The hard failures are OOF Spearman `<0.45`, contributing deltas `<5`, and weak latest time-block performance.

## Fig3 Subset Screen

A posthoc Fig3 score-table screen found an apparently stronger 10-domain subset, but a full recomputation did not support simple trimming as the final strategy.

- Best posthoc 10-domain subset: Spearman `0.413`, equal-weight `0.238`, learned-vs-equal `+0.175`, top/bottom enrichment `5.94x`.
- Domains in that subset: `gamma_ray_bursts_and_supernovae`, `genetics_aging_and_longevity_in_model_organisms`, `graphene_2d_materials`, `ipsc_reprogramming`, `mass_spectrometry_techniques_and_applications`, `microbiome_metagenomics`, `perovskite_solar_cells`, `spectroscopy_and_quantum_chemical_studies`, `topological_insulators`, `ubiquitin_and_proteasome_pathways`.
- Full recomputation on the same 10 domains: learned OOF Spearman `0.324`, equal-weight `0.142`, learned-vs-equal `+0.182`, best single `0.252`, nonlinear upper `0.345`.
- Full recomputation kept effect-size separation (`4.45x` enrichment; `+37.6 pp` high-low lift) but collapsed mechanism diversity to only `1` contributing graph delta.

Interpretation: pruning the current 12-domain roster alone is not enough. The next data intervention should add or repair domains with better time-block behavior and independent graph-signal channels, not merely remove weak current domains.

## Fig5 Evidence

Fig5 remains the weakest part of the evidence chain.

- Original OOF score coverage on the Fig3-aware 12-domain corpus: 28.3%.
- Forecast score table with strict `min_refs=5` prior-reference scoring: 24,090/35,050 papers, 68.7% coverage.
- Forecast score table with `min_refs=1` prior-reference scoring: 31,411/35,050 papers, 89.6% coverage.
- Strict prior-reference upper bound is below the target 95% unless the data layer either adds more usable prior references or explicitly defines/imputes low-data papers.
- Neither forecast-score run beats growth/citation baselines in the multi-domain Fig5 backtests.
- Per-domain scans mostly show ties or baseline dominance, not robust graph-score wins.

Interpretation: Fig5 should not be a main-text forecasting claim yet. This is not a plotting issue. It is a mismatch between current graph-score weights, topic-level hotspot ranking, and the available time windows.

## Extra Closure Top-Up Probe

An additional local-reference-only OpenAlex top-up was tested on promising non-social, near-ready domains outside the current Fig3-aware roster.

- `magnetic_properties_of_thin_films`: reference closure improved only from about `0.728` to `0.732`.
- `genome_wide_association_studies`: reference closure improved from about `0.597` to `0.639`.
- `immune_checkpoint_therapy`: reference closure stayed near `0.397`.
- The top-up added `317` works and `3,892` local citation edges, which is not enough to move these domains into the main corpus gates.

Interpretation: simple citation-neighborhood top-up is insufficient for the next wave. These domains need broader query design, deeper reference expansion, or landmark/time-window repair before they can strengthen Fig3/Fig5.

## Magnetic Manual Top-Up And Balanced 4-4-4 Probe

A broader, auditable OpenAlex query expansion rescued `magnetic_properties_of_thin_films` as a main-ready physics/astronomy domain.

- Manual query families: giant magnetoresistance, magnetic multilayers, spin valves, magnetic tunnel junctions, spintronics thin films, interlayer exchange coupling.
- Candidate records fetched: `4,387`.
- Records retained after strict local-reference filtering: `1,564`.
- `magnetic_properties_of_thin_films` after repair: `4,117` works, reference closure `0.809`, duplicate DOI rate `0.0046`, topic coverage `1.0`, Fig3-ready with `2,366` eligible metric papers and `1` eligible metric landmark.

This made a 12-domain 4-4-4 family-balanced candidate possible: 4 biology/biomedicine, 4 materials/chemistry, and 4 physics/astronomy domains. The candidate corpus passed corpus audit (`overall_pass=True`) with `36,667` works and `18` clean landmarks.

However, the full Fig3 recomputation rejected this balanced roster as a main-corpus replacement.

- Learned OOF Spearman: `0.248`.
- Equal-weight OOF Spearman: `0.163`.
- Learned vs equal delta: `+0.085`.
- Best single indicator: `0.181`.
- Top/bottom score-decile RGPM top20 enrichment: `4.50x`.
- High/low tertile RGPM lift: `+22.9 pp`.
- Contributing graph deltas: `2` (`community_reach`, `boundary_mixing`).
- Fold degradation is severe: `0.320`, `0.324`, `0.227`, `0.137`.

Interpretation: breadth alone is not enough. The magnetic repair is useful and should stay in the candidate pool, but forcing a 4-4-4 balance weakens Fig3 below both the Fig3-aware 12-domain result and the publication target. The next candidate should be performance-gated first, then constrained for family balance, rather than enforcing equal family counts upfront.

## Recommended Main-Text Position

- Use Fig1/Fig2/Fig4 as corpus and mechanism/context figures only after their data audits pass.
- Use Fig3-aware 12-domain results as a strong diagnostic milestone, not as final main-claim evidence.
- Keep Fig5 out of the main claim until a new data target is met: score coverage `>=95%` under a declared forecast-eligible denominator and graph score beating growth/citation/random in most historical windows.

## Next Data Targets

1. Replace or trim domains/time windows that depress the latest Fig3 time block; the target is OOF Spearman `>=0.45` with at least 5 contributing graph deltas.
2. Build a Fig5-specific forecast dataset with explicit forecast eligibility, not a recycled Fig3 OOF validation table.
3. Add older/mid-period domains with dense prior-reference graphs and clean post-event diffusion, then select 10-12 by holdout behavior.
4. Treat zero-prior-reference papers as a formal data-design decision: either exclude them from the forecast denominator or impute neutral low-confidence scores with a manifest flag.

## Included Files

- `domain_inclusion_table.csv`: 11-domain inclusion table from the previous stable corpus.
- `fig3aware12_materialized_domain_audit.csv`: latest 12-domain materialized audit.
- `fig3aware12_fig3_figure_quality_report.json`: Fig3-aware quality gates.
- `fig3aware12_fig3_effect_summary.csv`: Fig3-aware effect-size summary.
- `fig3aware12_fig3_baseline_comparison.csv`: Fig3-aware baseline comparison.
- `fig3aware12_forecast_score_manifest_minrefs5.json`: strict forecast-score coverage manifest.
- `fig3aware12_forecast_score_manifest_minrefs1.json`: relaxed forecast-score coverage manifest.
- `fig3aware12_fig5_domain_summary_minrefs1.csv`: per-domain Fig5 scan summary.
- `fig3aware12_fig5_domain_backtest_summary_minrefs1.csv`: per-domain/window Fig5 backtest details.
- `fig3aware12_subset_candidate_top25.csv`: top Fig3 posthoc subset candidates.
- `fig3aware12_subset_domain_oof_diagnostics.csv`: per-domain Fig3 OOF diagnostics.
- `fig3aware12_subset_fold_oof_diagnostics.csv`: per-fold Fig3 OOF diagnostics.
- `fig3aware12_subset_diagnostics_manifest.json`: subset screen manifest.
- `fig3aware10_subset_best_quality_report.json`: full recomputation quality report for the best screened 10-domain subset.
- `fig3aware10_subset_best_effect_summary.csv`: full recomputation effect summary for that subset.
- `fig3aware10_subset_best_baseline_comparison.csv`: full recomputation baseline comparison for that subset.
- `extra_closure_topup_domain_status.csv`: candidate audit after the extra top-up probe.
- `extra_closure_topup_manifest.json`: extra top-up manifest.
- `v8_magnetic_manual_topup_domain_status.csv`: audit after broader magnetic query expansion.
- `v8_magnetic_manual_topup_publication_target_domains.json`: roster candidate after magnetic repair.
- `magnetic_manual_topup_records.jsonl`: retained OpenAlex candidate records used for the broader magnetic query expansion.
- `magnetic_manual_topup_records_manifest.json`: query manifest for the magnetic manual top-up records.
- `v9_balanced_444_domain_status.csv`: candidate audit for the 4-4-4 balanced roster.
- `v9_balanced_444_corpus_manifest.json`: manifest for `data/knowledge_corpus/v2_publication_v3_balanced_444`.
- `v9_balanced_444_corpus_quality_report.json`: corpus audit for the 4-4-4 balanced corpus.
- `v9_balanced_444_fig3_figure_quality_report.json`: Fig3 quality report for the 4-4-4 balanced corpus.
- `v9_balanced_444_fig3_cv_summary.csv`: Fig3 fold summary for the 4-4-4 balanced corpus.
- `v9_balanced_444_fig3_effect_summary.csv`: Fig3 effect summary for the 4-4-4 balanced corpus.
- `v9_balanced_444_fig3_baseline_comparison.csv`: Fig3 baseline comparison for the 4-4-4 balanced corpus.
- `v10_balanced_444_fig3_domain_oof_diagnostics.csv`: per-domain Fig3 attribution for the balanced corpus.
- `v10_balanced_444_fig3_fold_oof_diagnostics.csv`: per-fold Fig3 attribution for the balanced corpus.
- `v10_balanced_444_fig3_subset_candidate_top25.csv`: posthoc subset screen for the balanced corpus.
- `cleanup_manifest_2026-06-23.json`: paths, sizes, and reasons for cleaned intermediate artifacts.

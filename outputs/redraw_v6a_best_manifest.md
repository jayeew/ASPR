# Fig1-Fig3 redraw manifest

Created on 2026-06-24 from the current best reproducible corpus.

Note: the first Fig1/Fig2 redraw was superseded by the fixed display redraw in
`outputs/redraw_v6a_display_fixed_manifest.md`, because the first Fig1 redraw
had only three v6A-compatible example domains and contained empty early topic
snapshots.

## Data source

- Main corpus: `data/knowledge_corpus/v2_publication_v6a_locked_candidate`
- Exact Fig3 v6A recompute input: `data/knowledge_corpus/v2_publication_v6a_locked_candidate/reproducibility/fig3_recompute/multi_domain`
- Exact Fig3 v6A decision: `outputs/redraw_v6a_best_fig3_exact_v6a_locked/fig3_v6a_probe_decision.json`

## Outputs

- Fig1, current-best example domains: `outputs/redraw_v6a_best_fig1/fig1_multi_domain_real.png`
- Fig2, current-best 10-domain empirical panels: `outputs/redraw_v6a_best_fig2/fig2_empirical_full.png`
- Fig3, original legacy plotting logic on the current-best 10-domain views: `outputs/redraw_v6a_best_fig3/fig3_selected_weight_learning_full.png`
- Fig3 v6A, publication summary of the locked main result: `outputs/redraw_v6a_best_fig3_v6a_locked/fig3_v6a_publication_summary.png`

Vector/export formats are available beside the PNGs where generated: SVG/PDF for Fig1 and Fig3 v6A, SVG for the legacy Fig3 redraw.

## Locked v6A main-result metrics

- OOF Spearman: `0.546554342701666`
- latest fold Spearman: `0.6421779791052799`
- learned vs equal: `0.41532347704429784`
- contributing deltas: `5`
- domains: `10`
- rows: `2197`
- min rows/domain: `76`
- max domain share: `0.1679563040509786`

## Important distinction

`outputs/redraw_v6a_best_fig3/` is the original Fig3 diagnostic plot, kept for compatibility with the previous Fig1-Fig5 plotting logic. It still reports the legacy structural-residual RGPM result and therefore does not show the v6A OOF 0.5466 result.

`outputs/redraw_v6a_best_fig3_v6a_locked/` is the publication-facing Fig3 summary for the currently best v6A result: seven core indicators plus legal publication-day features, reliability-gated 10-domain cohort, latent future graph-perturbation target, and locked validation.

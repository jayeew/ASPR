# Fixed Fig1-Fig2 display redraw manifest

Created on 2026-06-24 after the first v6A redraw showed empty Fig. 1 snapshots and an empty Fig. 2 panel-a snapshot.

## Data source

- Main corpus: `data/knowledge_corpus/v2_publication_v6a_locked_candidate`
- Fig. 1 source view: `data/knowledge_corpus/v2_publication_v6a_locked_candidate/views/fig1`
- Fig. 2 source view: `data/knowledge_corpus/v2_publication_v6a_locked_candidate/views/fig2`

## Display configs

The fixed Fig. 1 redraw uses four v6A-compatible display configs:

- `experiments/kg_perturbation_fig1/configs/v6a_display_crispr.yaml`
- `experiments/kg_perturbation_fig1/configs/v6a_display_graphene.yaml`
- `experiments/kg_perturbation_fig1/configs/v6a_display_ipsc.yaml`
- `experiments/kg_perturbation_fig1/configs/v6a_display_exoplanets.yaml`

The original CRISPR, graphene, and iPSC configs were not overwritten. The new display configs preserve their domain-specific plotting vocabulary but move the displayed cumulative snapshots to windows with enough v6A topic support.

## Outputs

- Fixed Fig. 1 PNG: `outputs/redraw_v6a_display_fixed_fig1/fig1_multi_domain_real.png`
- Fixed Fig. 1 SVG/PDF: `outputs/redraw_v6a_display_fixed_fig1/fig1_multi_domain_real.svg`, `outputs/redraw_v6a_display_fixed_fig1/fig1_multi_domain_real.pdf`
- Fixed Fig. 2 PNG: `outputs/redraw_v6a_display_fixed_fig2/fig2_empirical_full.png`

## Fig. 1 display windows

- CRISPR: 2010-2014, 2010-2016, 2010-2018, 2010-2021, 2010-2024
- Graphene/2D materials: 2004-2012, 2004-2015, 2004-2018, 2004-2021, 2004-2024
- iPSC reprogramming: 2006-2008, 2006-2011, 2006-2014, 2006-2018, 2006-2024
- Exoplanets: 1995-2005, 1995-2010, 1995-2015, 1995-2020, 1995-2024

## Empty-topic check

`snapshot_delta_metrics.csv` for the fixed Fig. 1 redraw reports no zero-topic snapshots:

- CRISPR displayed topics: 7, 8, 8, 8, 8
- Exoplanets displayed topics: 6, 7, 7, 8, 8
- Graphene/2D materials displayed topics: 5, 8, 8, 8, 8
- iPSC reprogramming displayed topics: 7, 8, 8, 8, 8

Text search over the fixed Fig. 1/Fig. 2 exports found no `No displayed topics` occurrences.

## Fig. 2 panel-a fix

Fig. 2 was redrawn with `--fig1-snapshot-dir outputs/redraw_v6a_display_fixed_fig1/crispr`, replacing the earlier panel-a input that inherited the old empty CRISPR first snapshot.

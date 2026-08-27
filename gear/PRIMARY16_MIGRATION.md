# Primary16 source unification

GEAR now uses the same frozen D5 `primary16` definition and trained model as
Fig.2new and Fig.3new.

## Canonical chain

- Model source: `innovation_impact_feature_selection/evidence_derived/production_releases/d5_primary16_tuned_20260827`
- GEAR release: `data/calibration/releases/gear-d5-primary16-current`
- Recent-paper runtime matrix: `data/calibration/runtime_features/gear-d5-primary16-dev10-v1`
- Full frozen replay: `data/calibration/runtime_replay/primary16_v1`

The runtime score is `prospective_5y_diffusion_percentile`. It is computed
from exactly the 16 Primary16 features and the frozen Primary16 HGB bundle.
The former 19-input GEAR bundle (8 innovation fields plus 11 controls) is not
a supported fallback.

The graph context is frozen through 2022. For every later target, GEAR must
recompute that paper's Primary16 vector at the review cutoff and reuse the
frozen HGB, calibrators, and percentile reference. It may not look up an old
score and may not refit the model. Missing post-2022 graph history remains an
explicit coverage limitation and shrinks routing toward neutral.

## Retired material

The old `gear-d5-v6-1-*` releases and runtime matrices were removed. GEAR
evaluation, smoke, quick-gate, and dev10 evaluation outputs produced with
those scores were also removed because their comparisons are not Primary16
results. The dev10 case selection and traceable review reconstruction inputs
remain available under `outputs/gear/dev10_claim_graph`.

The old gate preparation, analysis, feedback, and dev10 evaluation scripts
were retired as well: they still read candidate text from deleted v6.1 result
caches even after their score-table paths were changed.

Per the migration decision, no three-paper gate was rerun and the existing ten
papers were not re-judged. Therefore this migration establishes a single
model/feature lineage; it does not claim a new Graph-to-GEAR benefit result.

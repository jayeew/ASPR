# Fig.3 — Temporal OOF prediction of future D5 diffusion

**Question.** Can publication-time innovation signals rank future adoption
and cross-field diffusion for papers that were not used in model fitting?

**Data/model.** The frozen D5 label, six expanding time folds, medium
two-part model, eight innovation indicators, and K0/K1/K2 control sets.
Preprocessing and calibration are fit within each training fold.

**Panels.** D5 construction; two-part model and temporal folds; model
performance estimation ladder; OOF prediction/target hexbin; observed D5 by
prediction decile with top-decile enrichment; and angle add/drop plus
fold-stability diagnostics.

**Acceptance.** The main 8+K1 model must reproduce Spearman 0.767039879
within 1e-6 on the registered result.

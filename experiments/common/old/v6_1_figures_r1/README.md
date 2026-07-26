# ASPR v6.1 redesigned experiments

This package rebuilds the ten-paper-figure narrative from the frozen ASPR v6.1
Nature analysis. It does not overwrite the legacy graph-perturbation figures or
the v6.1 OOF release.

The ten experiments answer one question each:

1. What corpus and future-diffusion label are evaluated?
2. How were 50 candidate metrics reduced without consulting outcomes?
3. Are the eight admitted measurements sufficiently covered and stable?
4. Do innovation signals improve D5 out-of-fold ranking beyond controls?
5. Is that improvement consistent at D3, D5 and D8?
6. Does temporal generalization persist across the six expanding folds?
7. Does the conclusion hold across all twelve natural-science domains?
8. How much signal remains when all controls are removed?
9. Which of the five observation angles adds non-redundant predictive signal?
10. Do the registered sensitivity and reproducibility gates support the claim?

Run the complete suite once with:

```bash
python3 experiments/common/old/v6_1_figures_r1/run_all.py \
  --config configs/aspr_v6_1_figures.json \
  --output-dir outputs/common/new/baseline_suite_r1
```

Experiment 9 trains ten fixed-medium D5 temporal-OOF ablation models. Checkpoints
are stored inside the new output directory and are resumable after interruption.
No ablation result changes the frozen primary indicator registry.

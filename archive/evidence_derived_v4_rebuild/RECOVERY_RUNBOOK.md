# Evidence-v3 recovery runbook

The three files below are frozen research definitions. They determine the exact
membership of Strict 7, Full-text 16, Primary 154, and Broad T0 221; never
reconstruct them from model outputs or infer them from a figure.

- `complete_indicator_library_v3.csv`
- `feature_gate_decisions_v3.csv`
- `candidate_dimensions_v3.csv`

## Recovery sequence

1. Recover the three files together from a backup, another ASPR workspace, or
   the upstream evidence-data release. Keep their original bytes unchanged.
2. Validate the recovered directory before it is used:

   ```bash
   python3 innovation_impact_feature_selection/evidence_derived_v3/experiments/oof_feature_set_comparison_v3/verify_recovered_definition_bundle.py \
     --definition-dir /path/to/recovered/outputs \
     --report outputs/recovery/evidence_v3_definition_preflight.json
   ```

   It requires the exact nested counts 7/4, 16/10, 154/48, and 221/55
   (features/dimensions), and records file hashes.
3. Copy only the verified files into
   `innovation_impact_feature_selection/evidence_derived_v3/outputs/`.
4. Build matrices in a new release directory, train four-set HGB OOF models,
   validate forward-chaining splits, then regenerate Fig.3 from those outputs.

## Release hardening

Retain the definition hashes, corpus hashes, matrix hashes, OOF/score hashes,
figure hashes, and source-code Git commit in the release manifest. Update the
active registry only after every audit passes.

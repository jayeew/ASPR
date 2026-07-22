# Wave-B/C figure evidence contract

The model release can be composed with later peer-review, AI-frontier,
robustness, venue-family, and fixed-case runs without changing or retraining the
underlying Nature Multi-Horizon models. Pass one explicit directory to
`publish-release`:

```bash
python3 scripts/run_nature_multihorizon.py publish-release \
  --channel candidate \
  --dataset-id <dataset_id> \
  --analysis-id <training_analysis_id> \
  --figure-evidence-dir outputs/kg_perturbation_v2_evidence/<evidence_run>
```

The content hashes derive a new release analysis ID with an `-ev<hash>` suffix.
The original training analysis ID remains in `run_protocol.json` as
`base_training_analysis_id`. A changed evidence file therefore cannot overwrite
or silently reuse an existing release identity.

The directory may contain the following CSV or Parquet files. Supplying both
formats for one table is rejected.

| Artifact filename stem | Consumed by |
|---|---|
| `fig04_peer_review_validation` | Fig.4 peer-review external validity |
| `fig05_frontier_backtest` | Fig.5 AI-frontier join and backtest |
| `fig06_registered_robustness` | Fig.6 full registered sensitivity matrix |
| `fig07_venue_family_inference` | Fig.7 controlled venue-family inference |
| `fig09_case_profile` | Fig.9 frozen-model score for the fixed external case |
| `fig09_case_evidence` | Fig.9 graph-agent/Qwen/fusion case rerun |
| `fig10_registered_ablations` | Fig.10 model and ASPR-component ablations |

Every row has the following mandatory provenance fields:

```text
evidence_id
metric
value
n
source_artifact_sha256
protocol_hash
ci_low
ci_high
```

`source_artifact_sha256` and `protocol_hash` use the form
`sha256:<64 lowercase hex characters>`. Fig.9 accepts single-case evidence and
does not require a confidence interval; all other tables require a finite
interval containing `value`.

Every declared source/protocol file must also be placed below
`<evidence_run>/assets/`. Asset basenames must be unique. Publication recomputes
their SHA-256 values, refuses any unbound table hash, and copies the assets into
the evidence release. View validation checks the two provenance columns against
those release artifact hashes; a syntactically plausible but absent hash cannot
make a panel claim-ready.

Required evidence IDs are stored in each release-bound `view_manifest.json` and
are recomputed from the CSV data by the validator. The locked IDs are:

- Fig.4: `peer_review_resample_v2`, `new_score_external_validity`.
- Fig.5: `ai_frontier_tau5_join`, `forecast_backtest_v2`.
- Fig.6: horizons, citation threshold, graph snapshot, community algorithm,
  five single-mechanism deletions, auxiliary/calibration deletion, model family,
  seed stability, and fold stability.
- Fig.7: `venue_family_diffusion_enrichment_mechanism_time_panels`.
- Fig.9: `fixed_case_score`, `graph_qwen_fusion_rerun`.
- Fig.10: five mechanism deletions, all-auxiliary deletion, no calibration,
  model-family comparison, and no graph agent/Qwen/fusion-verifier.

Use the standard materializer rather than hand-writing release tables:

```bash
python3 experiments/kg_perturbation_v2/materialize_wave_evidence.py \
  build-case-features --paper <one-row-paper.parquet> \
  --references <case-reference-edges.parquet> \
  --reference-works <case-reference-metadata.parquet> \
  --graph-snapshots <frozen-graph-snapshots.parquet> \
  --output <case-publication-prior-features.parquet>

python3 experiments/kg_perturbation_v2/materialize_wave_evidence.py \
  score-case --release <frozen-release.json> \
  --features <one-row-publication-prior-features.parquet> \
  --case-id <fixed-case-id> --output-dir <evidence-run>

python3 experiments/kg_perturbation_v2/materialize_wave_evidence.py \
  package-table --artifact fig10_registered_ablations \
  --input <ablation-results.csv> --source <raw-result-file> \
  --protocol <locked-protocol.json> --output-dir <evidence-run>
```

`build-case-features` calls the same locked 8+10 feature implementation used by
the corpus. `score-case` calls the frozen τ=5 dual scorer and therefore supplies the fixed
case profile even when the DOI is outside the 211,073-paper modeling corpus.
The case ID and DOI/paper ID must exactly match the frozen `case_registry`; the
producer also requires at least 10 valid references, at least 60% reference
metadata coverage, and finite values for all eight core indicators.
`package-table` checks all locked evidence IDs and rejects Fig.6/Fig.10 metrics
that their renderer would not display. Every formal table must contain exactly
the locked evidence-ID set—extra exploratory rows are rejected rather than
mixed into a paper panel.

Fig.10 is sourced from explicit ablation rows in `evaluation_metrics.parquet`;
ordinary model-comparison rows never qualify. Missing evidence remains a
watermarked placeholder. Batch rendering and final assembly reject placeholders
unless `--allow-incomplete` is supplied, in which case the bundle receives
`_DRAFT` and can never receive `_SUCCESS`.

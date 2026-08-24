# Evidence-derived simplified protocol (four frozen sets)

This directory implements the protocol frozen on 2026-07-28. It is independent
of the older v3/v4 workflow and never imports legacy decisions as current
evidence. Legacy files may only be hash-registered during `bootstrap`.

## State machine

```text
bootstrap → saturate → freeze-search → screen → derive-dimensions
→ census-indicators → select-features → audit
```

Run from this directory, for example:

```bash
python3 pipeline.py --database outputs/evidence_derived.sqlite3 bootstrap \
  --legacy-input ../evidence_derived_v4_rebuild
python3 pipeline.py --database outputs/evidence_derived.sqlite3 saturate \
  --round 1 --new-terms 4 --new-indicators 2 --fully-reviewed \
  --review-artifact outputs/review_round_01.json
python3 pipeline.py --database outputs/evidence_derived.sqlite3 audit
```

Stage inputs are JSON arrays whose keys match the relevant SQLite table. The
importer rejects unknown fields. Independent-review artifacts must populate
`review_sessions`; the engine fails closed when evidence, review, seed recall,
PRESS, data mapping, or quality gates are incomplete.

`select-features` freezes Strict, Primary, Expanded, and Broad T0 definitions
before any model training. Supplying `--training-source` materializes four
Parquet matrices from those frozen definitions. Outcome columns are never read
for selection.

An audit is `COMPLETE` only when every protocol gate passes and two consecutive
audits over unchanged frozen inputs and artifacts have the same deterministic
hash. English-only eligibility and its language/geographic bias are always
reported.

## Seed indexability

Resolve seed DOIs before PRESS/search freeze. The resolver first uses exact DOI
matches in the OpenAlex provider cache, then validates missing bibliography via
Crossref and queries the OpenAlex DOI endpoint when key slots are available.
It never changes `recall_status` from `unchecked`.

```bash
export OPENALEX_API_KEYS="<slot-A>,<slot-B>"
python3 -m innovation_impact_feature_selection.evidence_derived.resolve_seed_indexability \
  --database innovation_impact_feature_selection/evidence_derived/outputs/evidence_derived.sqlite3 \
  --output-dir innovation_impact_feature_selection/evidence_derived/outputs
```

Keys are read only from the process environment. Persisted provenance contains
only `A`/`B` slot labels. Without a configured slot, Crossref-valid seeds remain
explicitly `unchecked` until the resolver is rerun with OpenAlex access.

## Canonical HGB OOF

The default read-only result is resolved through `current_best` and
`current_release.json`. It points to the validated horizon-specific nested HGB
release. Downstream readers should use `current_artifact(...)` from
`release_registry.py` rather than hard-coding an experiment directory. The
`outputs/hgb_oof_canonical` path remains the historical fixed-protocol result;
the training CLI's `--output-dir` is still a writable destination and is not
the published-result selector.

After the protocol audit is `COMPLETE`, the independent canonical runner reads
`training_matrix_manifest.json` and `final_feature_sets.json`; it never imports
the legacy evidence-v3 EF rosters. The manifest supplies the frozen Strict,
Primary, Expanded, and Broad T0 memberships and hashes.

```bash
python3 -m innovation_impact_feature_selection.evidence_derived.run_hgb_oof \
  --output-dir innovation_impact_feature_selection/evidence_derived/outputs/hgb_oof_working

python3 -m innovation_impact_feature_selection.evidence_derived.validate_hgb_oof \
  --output-dir innovation_impact_feature_selection/evidence_derived/outputs/hgb_oof_working
```

The runner fails closed on an incomplete audit, freeze/protocol/hash mismatch,
paper-row misalignment, non-nested sets, or outcome columns. It reuses the
active D3/D5/D8 cohorts, targets, fixed temporal folds, and HGB parameters.
Checkpoints are resumable by horizon, set, and fold. Outputs include OOF
predictions, overall/fold/domain metrics, an outcome-evaluation-only model
comparison, fold-valid paper scores, and hashed run/validation manifests.

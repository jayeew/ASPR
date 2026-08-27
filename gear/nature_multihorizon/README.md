# Nature Multi-Horizon V1

This package is the versioned evidence pipeline for the 42-source Nature
Portfolio corpus. It does not claim all-journal coverage and it uses the
conservative rule `source_max_year < publication_year`, not exact publication
day information.

## Locked scientific contract

- Horizons: 3, 5, and 8 years; 5 years is primary.
- Cohort: each horizon independently requires at least 10 future citers.
- Scope: 12 paper-level natural-science domains; non-natural papers remain in
  the raw inventory but are explicitly out of modeling scope.
- Inputs: eight core indicators and ten bibliographic controls, all derived
  from references and a strictly prior graph.
- Mechanisms: boundary perturbation, community diffusion,
  interdisciplinarity, knowledge recombination, and knowledge brokerage.
- Targets: horizon-global `RGPM-D3/D5/D8`; future-citer count adjustment is
  fitted inside each CV training fold. `RGPM-S3/S5/S8` is a pre-locked
  structural validation subset.
- Scores: an interpretable five-mechanism Simplex score and a separately
  selected GAM/HGB/Rank-Blend performance score.
- Runtime meaning: the primary D5 HGB score is a prospective five-year
  scholarly-diffusion percentile. It must not be interpreted as novelty,
  reviewer sentiment, acceptance probability, or a positive/mixed label.
- Cap handling: 1,000-citer truncation is explicit, horizon-specific, limited
  to at most 2% of each modeled cohort, and backed by a locked τ=5 uncapped OOF
  sensitivity gate.

## Current offline input

The V5 reference closure and the common τ=3/5/8 future layer are now complete.
The latter contains 131,777 papers and 22,871,558 future-citer rows. Five
papers have explicit `missing_checkpoint` outcomes, so the upstream
`overall_pass` deliberately remains `false`; the V1 adapter accepts the other
131,772 papers for training and never converts those five failures to zero.

The 6.7 GB future-citer table remains one read-only source file. The immutable
stage stores its path, size, modification time, row count, and head/tail hash,
then revalidates that identity whenever structural validation or release audit
uses it. This avoids a second multi-GB copy.

## Safe execution order

First run the read-only source audit:

```bash
python3 scripts/run_nature_multihorizon.py audit-source
```

The expected result is `formal_source_ready=true`, while the nested future
audit reports both `overall_pass=false` and `accepted_for_training=true`.
Then run the complete common-cohort pipeline without network access:

```bash
python3 scripts/run_nature_multihorizon.py run \
  --from-stage ingest-v5 --to-stage evaluate --resume
```

The `future-citers` step in this command calls
`import-future-multihorizon`, not OpenAlex. Individual stages remain available,
including the explicit import command:

```bash
python3 scripts/run_nature_multihorizon.py \
  import-future-multihorizon --resume
```

After the run, use the exact printed IDs:

```bash
python3 scripts/run_nature_multihorizon.py publish-release \
  --channel candidate \
  --dataset-id <dataset_id_printed_by_run> \
  --analysis-id <analysis_id_printed_by_run>
python3 scripts/run_nature_multihorizon.py audit-release \
  --release outputs/nature_multihorizon_v1/candidates/<analysis_id>/release.json
```

This first candidate is the locked `publication_year <= 2017` common cohort.
Only after its τ=5 OOF is positive and its data/training gates pass may the
recent short-window requests be unlocked. The expanded run reuses the already
built publication-time ingest, taxonomy, graph, and feature stages:

```bash
python3 scripts/run_nature_multihorizon.py run \
  --from-stage future-citers --to-stage evaluate \
  --future-scope expanded \
  --common-candidate-release outputs/nature_multihorizon_v1/candidates/<common_analysis_id>/release.json \
  --resume --retry-failed --allow-network --unlock-sealed-holdout
```

`--common-candidate-release` automatically and immutably binds the reused
publication-stage dataset. Supplying a different
`--reuse-publication-dataset-id` is rejected. Publish the expanded candidate
with the exact IDs printed by that run and repeat the common-candidate path so
the publication stages can be resolved:

```bash
python3 scripts/run_nature_multihorizon.py publish-release \
  --channel candidate \
  --dataset-id <expanded_dataset_id> \
  --analysis-id <expanded_analysis_id> \
  --common-candidate-release outputs/nature_multihorizon_v1/candidates/<common_analysis_id>/release.json
```

The expanded future layer merges three disjoint request batches: τ=8 for
papers through 2017, τ=5 for 2018--2020, and τ=3 for 2021--2022. Checkpoints
live outside analysis IDs, so downstream code changes do not discard a long
OpenAlex acquisition.

For the common scope, `run` is offline. For the later expanded recent-paper
scope it still refuses network access unless `--allow-network` is explicit.
Use `--dry-run` to inspect every stage path and identifier before starting.

A candidate can be promoted only when all data, OOF, temporal holdout,
performance-uplift, structural-validation, and technical view-integrity gates
pass:

```bash
python3 scripts/run_nature_multihorizon.py publish-release \
  --channel frozen \
  --release outputs/nature_multihorizon_v1/candidates/<analysis_id>/release.json
```

Wave-B/C evidence can be added at publication time through
`--figure-evidence-dir`; its content hashes derive a new `-ev<hash>` release
analysis ID while preserving the training analysis lineage. Scientific
Fig.1--Fig.10 readiness is enforced later by the batch renderer and final
assembler: any missing external run yields only a watermarked `_DRAFT`, never
the paper-level `_SUCCESS` marker.

Every stage is immutable and content-hashed. Changed source, configuration, or
pipeline code produces a different default ID; there is no implicit `latest`.
Frozen promotion is release-driven and does not recompute IDs from current CLI
defaults. Success markers bind the complete stage/release manifest, including
channel, IDs, configuration, code, dirty provenance, and upstream lineage.
The frozen release also contains `run_protocol.json`, assembled from each
stage's immutable protocol, so horizons, future scope, cap, observation year,
smoke/max-paper limits, reuse lineage, and sealed-holdout state remain
auditable after publication.

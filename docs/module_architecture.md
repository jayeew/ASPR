# GEAR module architecture and artifact exchange

The project is split into independently runnable producers. A producer writes to
its own work directory, then publishes an immutable release. Other modules
consume only a pinned release reference; they never read another module's
mutable work directory or guess a `latest` path.

```text
datasets -> indicator_definition -> aspr_scoring
    |                                  |
    +------------> gear_agent          |
    |                 |                |
    +------------> review_reconstruction
                      |                |
                      +----------------+
                        review_evaluation
```

The reconstruction and Agent branches remain information-isolated:
reconstruction may read the full reviewer/author revision history, while the
Agent may read only the submitted paper and its declared data/calibration
releases. This prevents benchmark leakage.

## Stable contracts

| Producer | Primary file | Meaning |
|---|---|---|
| `indicator_definition` | `feature_registry.json` | frozen indicator definitions |
| `aspr_scoring` | `paper_scores.parquet` | paper-level ASPR scores |
| `review_reconstruction` | `human_structured_reviews.jsonl` | revision-aware human labels |
| `gear_agent` | `agent_structured_reviews.jsonl` | Agent outputs |
| `review_evaluation` | `corpus_metrics.json` | Agent-human agreement metrics |

Both review JSONL files contain exactly one
`gear.review_contracts.StructuredReview` per paper and use `paper_id` as the join
key. Publication validates this contract, rejects empty releases and duplicate
paper IDs, so comparison needs no schema conversion.

## Canonical storage

The shared exchange root is:

```text
outputs/gear/artifacts/<producer>/<artifact>/<release>/
```

Every directory contains an `artifact_manifest.json` with hashes and pinned
dependencies. Small references live at:

```text
outputs/gear/artifact_refs/<producer>/<release>.json
```

Set `ASPR_GEAR_ARTIFACT_ROOT` and `ASPR_GEAR_REFERENCE_ROOT` to relocate both
roots. A consumer resolves and hash-verifies a reference JSON before reading the
producer's primary file. Work directories are disposable after publication.

## Independent commands

Inspect all exchange commands:

```bash
python3 -m gear.module_cli --help
```

The existing multi-horizon CLI independently builds the feature/indicator
registry and paper ASPR scores. Its `features`, `train`, and `evaluate`
subcommands are resumable and can be run without reconstruction or review
evaluation:

```bash
python3 scripts/run_nature_multihorizon.py features --dataset-id <id>
python3 scripts/run_nature_multihorizon.py train --dataset-id <id> --analysis-id <id>
python3 scripts/run_nature_multihorizon.py evaluate --dataset-id <id> --analysis-id <id>
```

Publish their stable outputs independently:

```bash
python3 -m gear.module_cli publish --module indicator_definition \
  --release indicators-v1 --source /path/to/indicator-output
python3 -m gear.module_cli publish --module aspr_scoring \
  --release scores-v1 --source /path/to/score-output \
  --dependency outputs/gear/artifact_refs/indicator_definition/indicators-v1.json
```

Reconstruct an explicit list of Markdown papers, then publish it:

```bash
ASPR_GEAR_MODEL_BACKEND=codex_cli \
ASPR_GEAR_RECONSTRUCTION_PAPER_IDS=id1,id2,id3 \
python3 scripts/build_human_review_benchmark.py --dataset-id human-v1
python3 -m gear.module_cli publish --module review_reconstruction \
  --release human-v1 --source outputs/gear/human_review_reconstruction/human-v1
```

Run the Agent, export exact `review.json` files, and publish them:

```bash
python3 -m gear review --paper /path/paper.pdf --output-dir /path/to/run
python3 -m gear.module_cli export-agent \
  --run-dir /path/to/run --output-dir /path/to/agent-export
python3 -m gear.module_cli publish --module gear_agent \
  --release agent-v1 --source /path/to/agent-export
```

Compare two pinned releases. The default matcher is a deterministic lexical and
evidence-overlap baseline recorded in the output; a semantic judge can replace
it without changing either input contract.

```bash
python3 -m gear.module_cli compare \
  --human-reference outputs/gear/artifact_refs/review_reconstruction/human-v1.json \
  --agent-reference outputs/gear/artifact_refs/gear_agent/agent-v1.json \
  --release comparison-v1
```

The comparison release pins both inputs in its manifest, making the evaluation
reproducible.

# Module architecture and artifact exchange

The repository is divided by responsibility, not by a single execution chain.
Each module may reuse another module's released files, but it must not import or
read that module's mutable work directory.

```text
datasets ─────────┬──────────► figures (Fig.1–Fig.10 experiments)
                  ├──────────► GEAR agent reviews
                  └──────────► review reconstruction

review reconstruction ─┐
                       ├────────► consistency evaluation
GEAR agent reviews ────┘
```

## Module ownership

| Module | Code | Owns | May consume |
|---|---|---|---|
| Datasets | `gear/nature_multihorizon/` | corpus, feature, calibration releases | none |
| Review reconstruction | `experiments/gear/review_reconstruction/` | sealed human reference reviews | dataset releases only |
| GEAR agent | `gear/` | evidence-traceable agent review runs | dataset/calibration releases only |
| Figures | `experiments/fig01` … `experiments/fig10`, `experiments/common/new/` | figure tables, renders, audits | dataset/calibration releases only |
| Consistency evaluation | `scripts/run_gear_revision_audit.py` | three-paper revision-aware blinded agent–human audit | pinned human-reference and agent-review inputs |

The reconstruction module remains graph-blind and must never consume a GEAR
review artifact. The agent module must never consume reviewer reports or author
responses. These two constraints prevent benchmark leakage.

## Shared artifact protocol

`artifact_store/` is the sole cross-module exchange layer. A release is stored
under:

```text
artifacts/<producer>/<artifact>/<release>/
├── artifact_manifest.json
└── <payload files>
```

The manifest records every payload path, byte size, SHA-256, pinned dependency
references, and small typed metadata. Publication is write-once: reusing a
release name resolves and verifies the existing content rather than overwriting
it. Consumers receive an `ArtifactReference` containing the producer, artifact,
release, and manifest hash; they must resolve that reference before reading any
payload.

```bash
python3 -m artifact_store publish \
  --producer datasets \
  --artifact dataset_release \
  --release nature-dev100 \
  --source /absolute/path/to/sealed-release
```

The module-specific publication facades are:

- `gear.nature_multihorizon.artifacts.publish_dataset_release`
- `experiments.gear.review_reconstruction.artifacts.publish_reference_dataset`
- `gear.artifacts.publish_review_run`
- `experiments.common.new.adapters.artifacts.publish_figure_result`
- `experiments.gear.consistency_artifacts.publish_consistency_evaluation`

## Migration policy

Existing files remain at their current canonical locations during migration.
New cross-module dependencies must be introduced as artifact references. A
module can then be relocated without changing a consumer, because the consumer
only receives a verified release directory from `ArtifactStore.resolve`.

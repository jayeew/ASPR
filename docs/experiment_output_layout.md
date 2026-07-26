# ASPR experiments and outputs layout

## Canonical structure

The experiment tree uses zero-padded figure identifiers and explicit version
roles:

```text
experiments/
├── common/
│   ├── new/
│   │   ├── base/       # shared Fig.01–Fig.10 builders/renderers
│   │   ├── adapters/   # current five-angle/eight-indicator extensions
│   │   └── run_all.py
│   └── old/            # retained cross-figure legacy/audit code
├── fig01/
│   ├── old/
│   └── new/
…
└── fig10/
    ├── old/
    └── new/
```

Fig.08 has legacy code under `fig08/old`; its current implementation is an AI
illustration handoff documented under `fig08/new`, not a numerical plotting
runner.

```text
outputs/
├── common/
│   ├── new/
│   │   ├── base_suite/          # main current ten-figure result
│   │   ├── baseline_suite_r1/   # retained shared ablation baseline
│   │   ├── model/v6_1_r5/       # current model/OOF source
│   │   ├── cache/               # reusable exact recomputation cache
│   │   └── extension_suite/     # thin-runner suite manifest/contact sheet
│   └── old/
│       ├── model/               # v6 and v6.1 revisions initial–R4
│       ├── final_assembly/
│       └── final_suite/
├── fig01/
│   ├── old/
│   └── new/
…
└── fig10/
    ├── old/
    └── new/
```

The main current numerical results remain in `outputs/common/new/base_suite`
and `outputs/common/new/model/v6_1_r5`. Per-figure `new` runners import the
shared builders and frozen model outputs; they contain only figure-specific
extensions, panel contracts, audits, and exports.

Old figure-specific helper scripts that previously lived in `scripts/` were
also moved next to the experiment they support. Cross-figure packaging tools
are under `experiments/common/old/final_assembly`. The archived output
organizer is now copy-only and cannot delete the canonical source trees.

Frozen scientific manifests are not rewritten merely to change an absolute
path, because that would invalidate their provenance hashes. Readers use
`aspr.path_layout.resolve_artifact_path` to map registered pre-migration roots
to their canonical locations.

## Retention and deletion policy

The following are retained even when superseded:

- old Fig.01–Fig.10 source and final artifacts;
- v6.1 initial/R1–R4 measurement revisions;
- deterministic replay and OOF checkpoints;
- blinded-label and preference-task packages;
- GPT Image source assets and prompt records;
- manifests, source hashes, negative results, and blocked/DRAFT outputs.

Only files that are mechanically reproducible and contain no scientific
evidence are deleted:

- Python bytecode and `__pycache__`;
- `.pytest_cache`, `.mypy_cache`, and `.ruff_cache`;
- empty migration source directories.

The failed `nature_portfolio_v5_gold.nohup` log is retained under
`outputs/common/old/logs` rather than deleted.

## Migration

Preview:

```bash
python3 scripts/reorganize_experiment_layout.py
```

Apply:

```bash
python3 scripts/reorganize_experiment_layout.py --apply
```

The command refuses to overwrite any target and writes
`outputs/common/layout_migration_manifest.json`.

The cleanup and retention decision log is documented in
`docs/experiment_output_cleanup_audit.md`.

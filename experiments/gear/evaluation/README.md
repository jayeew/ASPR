# Unified GEAR evaluation

This package evaluates only the current `gear` runtime. It does not invoke the
EACL novelty or ReviewGrounder repositories and does not produce a cross-track
composite score.

Run the frozen three-paper development pilot with:

```bash
python scripts/run_gear_evaluation.py all \
  --manifest configs/gear/evaluation_nature_pilot3_v1.json \
  --judge-config configs/gear/evaluator_codex_terra.json \
  --output-dir outputs/gear/evaluation/<run-id> \
  --resume
```

Stages are `preflight`, `run-clean`, `run-faults`,
`run-graph-ablations`, `prepare-judges`, `run-judges`, `score`, and
`report`. Checkpoints are content-addressed by the manifest, evaluator config,
tracked diff, untracked source hashes, prompts, and response schemas. A dirty
tree therefore cannot silently reuse a checkpoint created from different code.

The primary KPI family is Analytical Quality, Major Support Precision, and
Novelty Reasoning F1. Driver and guardrail families remain separate. Null means
not measured; judge failures are excluded rather than replaced with zero.

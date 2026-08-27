# Unified GEAR evaluation

This package evaluates only the current `gear` runtime. It does not invoke the
EACL novelty or ReviewGrounder repositories and does not produce a cross-track
composite score.

The gate policy is frozen in
`configs/gear/evaluation_nature_graph_gate_v2.json`. It intentionally contains
no fabricated cases: case rows are added only after reviewer quote, round,
final-paper evidence, point-in-time Graph provenance, and D5 percentile have
all passed validation.

Run an eligible evaluation manifest with:

```bash
python scripts/run_gear_evaluation.py all \
  --manifest /path/to/traceable-nature-gate-manifest.json \
  --judge-config configs/gear/evaluator_codex_terra.json \
  --output-dir outputs/gear/evaluation/<run-id> \
  --resume
```

Stages are `preflight`, `run-clean`, `run-faults`,
`run-graph-ablations`, `prepare-judges`, `run-judges`, `score`, and
`report`. Checkpoints are content-addressed by the manifest, evaluator config,
tracked diff, untracked source hashes, prompts, and response schemas. A dirty
tree therefore cannot silently reuse a checkpoint created from different code.

Graph evaluation has four matched variants: `neutral`, `score`,
`score_topology`, and `placebo_graph`. Every variant reuses the same graph-blind
draft and candidate pools under identical resource caps. The D5 percentile may
only reorder candidates; topology may replace at most two remote queries.

The primary Graph KPIs are claim-relevant verified relation yield, material
correction yield, equal logical resources, and blind review preference.
Accepted-paper novelty polarity is only a compatibility diagnostic. Driver and
guardrail families remain separate. Null means not measured; judge failures are
excluded rather than replaced with zero.

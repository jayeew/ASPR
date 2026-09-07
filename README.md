# ASPR-GEAR — Claim Graph + GEAR innovation analysis

ASPR-GEAR analyzes a paper's concrete contributions, manuscript support,
historical basis and knowledge relationships. The current research focus is the
innovation-analysis system. Historical indicator and diffusion-scoring research
remains available where relevant; it does not automatically validate this system.

```text
Paper full text → PaperIR → shared grounded contribution claims
                              ├─ GEAR: prior art, support, residual increment
                              └─ Claim Graph: historical neighbors and structure
                                   └─ joint Graph analysis across paper claims
GEAR + per-claim Graph analysis → claim-level fusion
Study report writer → whole-paper report using selected branch/joint evidence
```

The runtime is `gear`; `gear/review_pipeline.py` dispatches default `knowledge`
mode to `gear/innovation/pipeline.py`. Branches share contribution identities
but analyze evidence independently. The single-paper `all` command runs GEAR
then Graph before fusion; the study supports separate streaming processes.
`passive` and `active` are explicitly selected compatibility modes.

The output is innovation analysis, not an editorial decision or the old
five-section `StructuredReview`. No numeric novelty score is generated.
Graph structure alone cannot establish firstness, scientific validity or claim
derivation. GEAR distinguishes antecedents from related work and explains
residual increment. Insufficient evidence remains explicit.

## Run one paper

Run from the repository root with an `InnovationPaperInput` JSON:

```bash
python3 -m gear review \
  --input-contract /absolute/path/input.json \
  --output-dir outputs/gear/runs/example
```

Prepare that input from Markdown or PDF with the required metadata:

```bash
python3 -m gear prepare-input \
  --paper /absolute/path/paper.md \
  --paper-id example-paper \
  --title "Example paper title" \
  --doi "10.example/paper" \
  --publication-date 2026-01-10 \
  --cutoff 2026-01-10 \
  --output /absolute/path/input.json
```

Replace example identifiers and dates with actual metadata. `--abstract-file`
can supply an abstract. The input contract requires `paper_id`, `paper_path`,
`title`, `publication_date`, `cutoff_date`, `abstract_text`, and `abstract_source`;
optional fields include DOI, venue, authors, OpenAlex ID and reference work IDs.
See [`InnovationPaperInput`](gear/review_contracts.py).

Use `--stage shared|gear|graph|fusion|all` to address a stage. Shared claims are
prepared/reused first; fusion needs saved GEAR and Graph results. `--targets`
accepts a JSON list of neutral contribution descriptions for specified-target
tasks, without reviewer judgments or reasons. Use a separate run directory when
changing inputs or experimental conditions.

## Outputs and checks

A standard knowledge-mode run writes:

- `innovation_input.json`, `shared/paper_ir.json`, `shared/claims.json`;
- `gear/NN/` evidence, `gear_card.json` and `assessment.json`;
- `graph/NN/` graph facts and assessments, plus `graph/joint/` joint results;
- branch `analysis.json` files and `fusion/innovation_report.md`;
- append-only evidence traces and `model_usage.jsonl`.

```bash
python3 -m gear validate-assets
python3 -m gear validate-run outputs/gear/runs/example
make gear-test
make gear-lint
python3 -m pytest -q tests/innovation_200 tests/gear/test_historical_pdf_config.py
```

`validate-assets` checks native Claim Graph assets, not historical HGB releases.
`validate-run` checks standard v2 claim identity, coverage and evidence keys; it
is not a whole-study or scientific-correctness validator. Study fingerprint
settings differ; see [architecture and validation limits](docs/module_architecture.md).

## Models, retrieval and graph assets

The default backend is `codex_cli`, using fresh read-only CLI sessions. Current
role-specific models and reasoning settings are in
[`gear/model_client.py`](gear/model_client.py); the base `codex_cli.model` does
not determine every role. An `openai_compatible` backend is supported with
`--config`; see [`DeepSeek example`](configs/gear/deepseek.example.json). Keep
API keys only in the configured environment variable.

Native historical graph assets live under `data/claim_graph`; builders are in
`scripts/claim_graph`. Qwen3-Embedding-4B supplies graph embeddings and local
recall, with OpenScholar reranking for GEAR retrieval. These are not the retired
ASPR-Qwen review branch. Runtime insertion leaves the historical graph unchanged
and defaults to at most ten eligible neighbors with cosine strictly above 0.5.
Current thresholded structural values do not use percentiles calibrated under
the older neighbor-selection policy.

Historical literature PDF downloading defaults off. Set
`GEAR_HISTORICAL_PDF_ENABLED=true` for budgeted PDF retrieval and restart workers
after changing it. Target-paper full text is unaffected. Distinguish abstract
and full-text evidence. Resuming after a policy change preserves old results
and creates mixed conditions; use a separate directory for uniform comparisons.

## Current experiments and documentation

- [Staged innovation study](experiments/innovation_200/README.md): initially 200
  papers, with a 1,000-paper claims/reference extension. `papers.jsonl` is the
  active roster; extension alone does not complete eight-system evaluation.
- [Module architecture](docs/module_architecture.md): boundaries, artifacts,
  branch independence and recovery.
- [Implementation and evaluation scope](docs/innovation_v2_implementation.md).
- [Single and joint Graph analysis](docs/graph_joint_analysis.md).
- [Experiment index](experiments/README.md): current studies versus historical
  Fig.1–Fig.10. New figure allocation is still under discussion.
- [Data sources](docs/data_sources.md) and [code availability](docs/code_availability.md).

Documents marked historical preserve earlier designs/results, not current CLI
instructions, runtime requirements or current-system superiority evidence.
The old HGB-to-claim pathway, five-part review contracts and mandatory
immutable-release exchange do not describe the default innovation workflow.

# ASPR-GEAR — Claim Graph + GEAR innovation analysis

[中文项目简介与逐图研究结果](docs/项目简介.md) · [论文文档](docs/paper_summary.md)

ASPR-GEAR analyzes a paper's concrete contributions, manuscript support,
historical basis and knowledge relationships. It produces qualitative innovation
analysis, with explicit limitations where evidence is insufficient.

```text
Paper full text → PaperIR → shared grounded contribution claims
                              ├─ GEAR: prior art, support, residual increment
                              └─ Claim Graph: historical neighbors and structure
                                   └─ joint Graph analysis across paper claims
GEAR + per-claim Graph analysis → claim-level fusion
Study report writer → whole-paper report using selected branch/joint evidence
```

The `gear` CLI starts in `gear/cli.py`. `gear/review_pipeline.py` dispatches
default `knowledge` mode to `gear/innovation/pipeline.py`. Branches share
contribution identities and analyze evidence independently. The single-paper
`all` command runs GEAR then Graph before per-claim fusion. The study supports
separate streaming processes and includes joint Graph in its full fusion report.
`passive` and `active` are explicitly selected compatibility modes.

GEAR distinguishes antecedents from related work and explains the residual
increment. Claim Graph describes historical neighbors and local structure.
Judgments about firstness, scientific validity or derivation require evidence
beyond graph structure.

## Project structure

A fresh clone contains the migrated directory layout, code and entry points.
Use a checkout that preserves symbolic links, such as Git in WSL on Windows.
Populate the documented local paths with any research assets needed for the
task, including graph indexes, model weights, figure masters, study results
and manuscript files. See [AGENTS.md](AGENTS.md) for maintenance rules.

| Directory | Purpose |
| --- | --- |
| `gear/`, `gear/innovation/`, `configs/gear/` | CLI, shared claims, GEAR and Claim Graph analyses, fusion, and runtime configuration |
| `gear/claim_graph/`, `scripts/claim_graph/`, `data/claim_graph/` | Native graph contracts, offline builders and historical graph and embedding assets |
| `data/knowledge_corpus/`, `data/models/`, `data/nature_2026_testset/` | Literature corpus, local models, and paper/reviewer source pairs |
| `experiments/innovation_200/`, `outputs/` | Staged study code, whole-paper reporting and evaluation, plus runtime and study results |
| `experiments/fig8_extension/`, `figures/fig8/data/corpus/` | Fig8 corpus collection, qualification and recomputation, with `outputs/fig8_data/` as a compatibility path |
| [`figures/`](figures/README.md) | Fig1–Fig11 workspaces with `code/`, `scripts/`, `data/`, `final/` and, where present, editable `assets/` |
| `figures/fig3/code/five_dimensions/`, `figures/fig3/data/five_dimensions/` | Five-dimension report evaluation and Fig3 inputs |
| `figures/shared/reviewer_coding/` | Shared reviewer coding, original excerpts and aggregation |
| `figure_pipeline/` | Shared rendering and QA, historical builds, and compatibility imports for relocated figure modules |
| [`manuscript/nature_innovation/`](manuscript/nature_innovation/README.md) | Paths for versioned LaTeX, manuscript figure sets, reading PDFs and standalone Limitation drafts |
| `tests/`, `docs/`, `scripts/maintenance/` | Test suites, methods, migration records and repository maintenance |
| [`archive/`](archive/README.md) | Earlier manuscripts, figures and scoring research, including the three `evidence_derived*` workspaces |
| `experiments/fig01/` … `experiments/fig10/`, `experiments/gear/`, `gear/nature_multihorizon/`, `artifact_store/` | Historical figure workflows, review studies, scoring code and release interfaces |

The data directory guide maps shared inputs, cache paths, figure workspaces and
study outputs. [Data directory guide](data/README.md)

Historical indicator scripts and imports can still access the archived
workspaces through `innovation_impact_feature_selection/`. The root
`fig_reference` link still resolves to `figures/fig5/assets/`.

## Run one paper

Run commands from the repository root. Use a Python environment with the
project dependencies installed. [Python requirements](requirements.txt)

```bash
python3 -m pip install -r requirements.txt
python3 -m gear --help
```

The `review` command also needs a configured model backend. GEAR and Graph
stages may need the local literature, graph and model assets listed above.

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

Use `--stage shared` to prepare grounded claims, `--stage gear` or `--stage graph`
to run each analysis, and `--stage fusion` to combine saved branch results.
The default `--stage all` runs the full workflow and reuses completed stages
when available. `--targets` accepts a JSON list of neutral contribution
descriptions for specified-target tasks. Use a separate run directory when
changing inputs or experimental conditions.

## Outputs and checks

A standard knowledge-mode run writes these artifacts.

- `innovation_input.json`, `shared/paper_ir.json`, `shared/claims.json`
- `gear/NN/` evidence, `gear_card.json` and `assessment.json`
- `graph/NN/` graph facts and assessments, plus `graph/joint/` joint results
- branch `analysis.json` files and `fusion/innovation_report.md`
- append-only evidence traces and `model_usage.jsonl`.

```bash
python3 -m gear validate-assets
python3 -m gear validate-run outputs/gear/runs/example
make gear-test
make gear-lint
make gear-validate
python3 -m pytest -q tests/innovation_200 tests/gear/test_historical_pdf_config.py
```

`validate-assets` and `make gear-validate` check native Claim Graph assets.
`validate-run` checks standard v2 claim-level artifacts, including claim
identity, coverage and evidence keys. Use each study's own evaluation workflow
for whole-paper reports and scientific conclusions. See the
[architecture and validation limits](docs/module_architecture.md).

`make gear-test` includes figure tests. Install the optional plotting packages
before running it. [Figure dependencies](requirements-figures.txt)

Run the five-dimension tests with `python3 -m pytest -q tests/innovation_five_dimensions`.

## Figures and manuscript

The figure index links to each workspace. The editable final master lives at
`figures/figN/final/FigN.svg`. The delivery index `outputs/fig1-11_final/` links
each `FigN/` to the respective `final/` directory. For Fig3, the accepted
five-dimension layout is `figures/fig3/final/Fig3.svg` and the retained-panel
redraw template is `figures/fig3/assets/layout_template.svg`.
[Figure index](figures/README.md)

```bash
python3 -m figures fig3 paths
python3 -m figures fig3 export
```

`paths` works from a fresh clone. For `export`, place the local SVG in `final/`
and install CairoSVG in the active environment so the `cairosvg` command is
available. The export writes a PDF and a PNG next to the SVG. Use the figure's
own workflow to redraw after updating statistics or layout inputs.
[Output and redraw guide](outputs/README.md)

Relocated modules under `figure_pipeline/`, affected paths under `outputs/`
and `scripts/`, and the Fig5 `fig_reference` path use symbolic links in the
reorganized workspace. Large files backed by `/mnt/d/ASPR-data/` use the same
project-relative paths. [Move manifest](docs/project_structure_manifest.json)
· [Local storage guide](docs/HDD_STORAGE.md)

Use the figure references in the manuscript version you are working on.
For example, `manuscript/nature_innovation/manuscript_zh_v10.tex` uses
`figures_v10/`, which includes versioned assets and the accepted Fig3. Use
the respective study's sample and evidence when interpreting each figure.

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

- The staged innovation study uses `experiments/innovation_200/`. Run
  `python3 -m experiments.innovation_200.run_all --help` for the orchestration
  entry point. Read `papers.jsonl` in the chosen run directory for the active
  roster. The 1,000-paper extension covers shared claims and reviewer-reference
  reconstruction only. [Staged study guide](experiments/innovation_200/README.md)
- The separate five-dimension report evaluation uses
  `python3 -m figure_pipeline.innovation_five_dimensions --help`. Its data
  README describes local inputs, report scores, retained diagnostics,
  denominators and reruns. [Five-dimension study guide](figures/fig3/data/five_dimensions/README.md)
- The architecture guide covers module boundaries, artifacts, branch
  independence and recovery. [Module architecture](docs/module_architecture.md)
- [Implementation and evaluation scope](docs/innovation_v2_implementation.md).
- [Single and joint Graph analysis](docs/graph_joint_analysis.md).
- The experiment index includes the current study and earlier Fig.1–Fig.10
  workflows. [Experiment index](experiments/README.md)
- [Data sources](docs/data_sources.md) and [code availability](docs/code_availability.md).

Documents marked historical preserve earlier designs/results, not current CLI
instructions, runtime requirements or current-system superiority evidence.
The old HGB-to-claim pathway, five-part review contracts and mandatory
immutable-release exchange do not describe the default innovation workflow.

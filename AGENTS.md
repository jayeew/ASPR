# AGENTS.md — ASPR-GEAR Project Guide

## Current project scope

ASPR-GEAR analyzes scientific innovation from paper full text. The current
runtime is `gear`, entered through `gear/cli.py` and `gear/review_pipeline.py`.
Default `knowledge` mode in `gear/innovation/pipeline.py` prepares shared
grounded claims, runs independent GEAR prior-art and Claim Graph analyses, then
fuses per-claim results. Joint Graph produces a separate paper-level analysis
that the staged study can use in its whole-paper reports. The runtime makes
qualitative innovation judgments without an editorial decision or numeric
novelty score.

Start with [README.md](README.md), [module architecture](docs/module_architecture.md)
and [implementation scope](docs/innovation_v2_implementation.md). Check current
code for runtime behavior and the relevant study records for empirical results.
Use the [data directory guide](data/README.md) to locate shared inputs, caches,
figure inputs and study outputs.

## Repository map

| Path | Use |
| --- | --- |
| `gear/` and `gear/innovation/` | CLI, input contracts, full-text grounding, retrieval, evidence, claim analysis and fusion |
| `gear/claim_graph/`, `scripts/claim_graph/`, `configs/gear/` | Native graph contracts, offline graph construction and runtime configuration |
| `data/claim_graph/`, `data/knowledge_corpus/`, `data/models/`, `data/nature_2026_testset/` | Shared graph and embedding assets, literature corpus, local models and paper/reviewer source pairs |
| `experiments/innovation_200/` | Staged innovation study, whole-paper reports and evaluation. Run artifacts live under the selected study directory, such as `outputs/innovation_200_20260907/`. |
| `experiments/fig8_extension/` | Fig8 corpus collection, qualification and recomputation. Its data lives in `figures/fig8/data/corpus/`, also accessible via `outputs/fig8_data/`. |
| `figures/fig1/` … `figures/fig11/` | Per-figure code, scripts, evidence, rendering inputs and accepted exports |
| `figures/fig3/code/five_dimensions/` and `figures/fig3/data/five_dimensions/` | Five-dimension report and evaluation workflow, study artifacts and Fig3 inputs |
| `figures/shared/reviewer_coding/` | Shared reviewer coding, source excerpts, aggregation and related analyses |
| `figure_pipeline/` | Shared rendering and QA, historical build workflow, and compatible import paths for relocated figure modules |
| `outputs/` | Runtime and study runs, packages, and compatible paths for relocated figure data and exports |
| `manuscript/nature_innovation/` | Versioned LaTeX, manuscript figure sets, reading PDFs and standalone Limitation drafts |
| `tests/` | Runtime (`innovation_v2/`), staged study (`innovation_200/`), five-dimension analysis (`innovation_five_dimensions/`) and figure tests |
| `docs/`, `scripts/maintenance/` | Architecture, methods, migration records and repository maintenance tools |
| `archive/evidence_derived/`, `archive/evidence_derived_v3/`, `archive/evidence_derived_v4_rebuild/` | Historical indicator research and frozen releases. The old `innovation_impact_feature_selection/` paths are compatibility links. |
| `archive/`, `experiments/fig01/` … `experiments/fig10/`, `experiments/gear/`, `gear/nature_multihorizon/`, `artifact_store/` | Earlier manuscripts, figure workflows, review interfaces, scoring experiments and release artifacts |

## Figures, manuscript and storage

- Start figure work at [figures/README.md](figures/README.md) and the figure's
  own README. Keep figure-specific implementation in `code/`, standalone
  commands in `scripts/`, evidence and rendering inputs in `data/`, editable
  templates in `assets/` where present, and accepted exports in `final/`.
- `figures/figN/final/FigN.svg` is the editable final master. The delivery
  index `outputs/fig1-11_final/` links each `FigN/` to that figure's `final/`.
  For Fig3, use `figures/fig3/final/Fig3.svg` for the accepted five-dimension
  layout and `figures/fig3/assets/layout_template.svg` as an input to redraw
  retained panels. Use [outputs/README.md](outputs/README.md) and the figure
  README to find redraw dependencies.
- Relocated `figure_pipeline.*` modules, the affected paths under `outputs/`
  and `scripts/`, and `fig_reference` are symbolic links. Edit the canonical files
  and preserve links used by Python imports or saved evidence. Use
  `figure_pipeline.paths.PROJECT_ROOT` when moved figure code needs a stable
  repository root. See the [move manifest](docs/project_structure_manifest.json)
  and [cleanup manifest](docs/project_cleanup_manifest.json).
- Use each figure's own cohort, denominators, and evidence. Fig3's five-dimension
  scores evaluate report analysis, while the retained diagnostics evaluate the
  original detailed analyses. See the [five-dimension study README](figures/fig3/data/five_dimensions/README.md)
  and [project overview](docs/项目简介.md) for the respective scopes.
- For manuscript edits or rebuilds, use the requested manuscript version and
  its figure references. Manuscript figure numbering and assets can differ from
  the project delivery index. For example, `manuscript_zh_v10.tex` uses
  `figures_v10/`, which links a mix of versioned assets and the accepted Fig3.
- Selected large files can be backed by links under `/mnt/d/ASPR-data/`. Keep
  using the project-relative data paths and check the mount and per-file
  migration status if a linked file is unavailable. Use only verified files
  for analysis and keep `.partial` copies out of the inputs. See the
  [storage guide](docs/HDD_STORAGE.md). Migration state lives in the local
  `docs/hdd_migration.json`.

## Commands

```bash
python3 -m gear review --input-contract /path/input.json --output-dir /path/run
python3 -m gear review --input-contract /path/input.json --output-dir /path/run --stage shared
python3 -m gear validate-assets
python3 -m gear validate-run /path/run
python3 -m figures fig3 paths
python3 -m figures fig3 export
make gear-test
make gear-lint
make gear-validate
python3 -m pytest -q tests/innovation_200 tests/gear/test_historical_pdf_config.py
python3 -m pytest -q tests/innovation_five_dimensions
```

The figure `paths` command prints the workspace locations. `export` uses
CairoSVG to write PDF and PNG beside the accepted SVG. Use the figure's own
redraw procedure when the statistics or layout inputs change.

`validate-assets` and `make gear-validate` check native Claim Graph files.
`validate-run` checks standard v2 claim-level artifacts, including claim
coverage and evidence keys. Use study-specific evaluations for whole-paper
reports and research results. Legacy Make targets may reference removed
contracts; do not restore those contracts to make historical checks pass.

## Runtime and evidence boundaries

- Keep model clients lazy. Role routing is in `gear/model_client.py`; the study
  overrides it in `experiments/innovation_200/common.py`.
- Establish shared claim identities and manuscript spans before branch
  analysis. GEAR and Graph must not read each other's judgments before fusion.
- GEAR establishes manuscript support, historical comparisons and residual
  increment. Missing prior art is not proof of firstness; a related paper is
  not automatically a direct antecedent. Exclude target-paper versions from
  independent prior-art evidence.
- Use native historical assets in `data/claim_graph`. Runtime insertion is
  temporary and must not mutate the historical graph. Historical abstract claim
  extraction and target full-text claim extraction are distinct processes.
- Default insertion retains at most 10 eligible semantic neighbors with cosine
  strictly above 0.5. Exclude self and ineligible dates; never fabricate neighbors.
- Parent-paper citation paths annotate semantic edges, not claim-level support,
  derivation, causality or antecedence. Communities are clusters, not verified
  disciplines; local connectivity is not global impact.
- Joint Graph uses the union neighborhood. Do not sum single-claim changes or
  invent target-to-target edges because claims share a paper. Separate observed
  graph facts from scientific interpretations.
- Current thresholded metrics must not inherit percentiles calibrated under the
  old neighbor-selection rule. Graph rarity, opportunity/control fields and HGB
  diffusion predictions cannot directly establish novelty.
- Raw evidence belongs in append-only `EvidenceStore`. Every finding requires
  existing evidence keys and matching claim identity/scope. Key validity alone
  does not establish factual or scientific correctness.
- Missing retrieval, verification, neighbors or branch outputs must remain
  explicit limitations/unresolved judgments or failures as appropriate. Never
  turn missing evidence into positive or negative novelty conclusions.
- Do not reintroduce LATS, committee, GraphRAG, an `aspr.*` public runtime,
  paper-level HGB-to-claim allocation or the former ASPR-Qwen reviewer.
  `passive`/`active` are explicit compatibility modes. Qwen embedding is a current
  retrieval/graph component, not that old reviewer.

## Experiments and recovery

- For the staged study, read `papers.jsonl` in the selected run directory for
  the active roster and check which stages actually finished. The 1,000-paper
  extension covers shared claims and reviewer-reference reconstruction only.
- The five-dimension experiment is a separate report and evaluation workflow
  entered via `python3 -m figure_pipeline.innovation_five_dimensions`. Read the
  study README for the main cohort, method conditions, rerun scope and output
  paths. Preserve original reports and evidence, and report missing, unresolved
  and technically unavailable dimensions without filling them with zero scores.
- Distinguish runtime defaults from study settings. The study uses Luna role
  overrides, at most eight claims, and disables response caching, relation
  stability repeats and resume fingerprint checks. Do not describe all study
  artifacts as hash-verified/frozen releases.
- Completed stage artifacts are reused on resume. Keep interrupted evidence
  attempts and report missing denominators. Use separate directories for uniform
  condition comparisons; document intentionally mixed-mode resumes.
- Historical PDF downloading defaults off, controlled by
  `GEAR_HISTORICAL_PDF_ENABLED`. Target-paper full text is unaffected. Preserve
  abstract/full-text/reference-only provenance and evidence limits.
- Reviewer references are source-bound and incomplete. A records contain an
  explicit innovation judgment; B records support identification. Preserve
  reviewer, round, contribution, reasons and source quotes. Silence is not a
  false positive; AI matching/preference is not newly collected human judgment.
- Keep end-to-end baselines distinct from shared-claim controls. Ablation must
  control generation path and relevant budget; substituting raw facts for
  interpreted text is not a clean single-component removal.
- Preserve historical results under their original methods and samples,
  including revisions in directories named `new`. Keep implementation and
  execution checks separate from empirical claims about the current system.

## Style

- Python functions require type hints; Pydantic contracts use `extra="forbid"`.
- Use `pathlib.Path` for I/O; prefer functions under 50 lines where practical.
- Catch specific exceptions; never use bare `except`.
- Order imports: future, standard library, third party, local.
- Store secrets in environment variables, never source files or documentation.

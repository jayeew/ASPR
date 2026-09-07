# AGENTS.md — Claim Graph + GEAR Coding Guidelines

## Current project scope

ASPR-GEAR analyzes scientific innovation from paper full text. The current
runtime is `gear`, entered through `gear/cli.py` and `gear/review_pipeline.py`.
Default `knowledge` mode uses `gear/innovation/`: shared grounded claims,
independent GEAR prior-art and Claim Graph analyses, claim-level fusion, and
separate paper-level joint Graph analysis. The study additionally writes
whole-paper reports. This is not the former five-part peer-review/decision
system and does not emit a numeric novelty score.

The confirmed research direction is the current innovation-analysis system;
historical scoring research is retained where relevant. New Fig.1–Fig.10
allocation is still under discussion, not a frozen protocol. Read `README.md`,
`docs/module_architecture.md`, and `docs/innovation_v2_implementation.md` for
current behavior. Historical documents and directories named `new` do not
override current code.

## Commands

```bash
python3 -m gear review --input-contract /path/input.json --output-dir /path/run
python3 -m gear review --input-contract /path/input.json --output-dir /path/run --stage shared
python3 -m gear validate-assets
python3 -m gear validate-run /path/run
make gear-test
make gear-lint
make gear-validate
python3 -m pytest -q tests/innovation_200 tests/gear/test_historical_pdf_config.py
```

`validate-assets` checks Claim Graph files, not HGB releases. `validate-run`
checks standard v2 claim-level artifacts, not the whole study. Legacy Make
targets may reference removed contracts; do not restore those contracts to
make historical checks pass.

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

- Current staged study code is `experiments/innovation_200/`; the retained name
  does not fix sample size. Read `papers.jsonl` for the active roster. The
  1,000-paper extension runs claims and reference reconstruction only, not all
  branches, reports and evaluations.
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
- Existing Fig.1–Fig.10 outputs support only their historical methods. Do not
  rewrite generated results or treat implementation, schema checks or successful
  execution as current-system empirical success.

## Style

- Python functions require type hints; Pydantic contracts use `extra="forbid"`.
- Use `pathlib.Path` for I/O; prefer functions under 50 lines where practical.
- Catch specific exceptions; never use bare `except`.
- Order imports: future, standard library, third party, local.
- Store secrets in environment variables, never source files or documentation.

## Structure

```text
gear/                         Runtime, retrieval, evidence and contracts
gear/innovation/              Current shared-claim analysis and evaluation
gear/claim_graph/              Native historical graph contracts
scripts/claim_graph/           Offline graph construction
experiments/innovation_200/    Staged study, reports and reference comparison
tests/innovation_v2/           Current runtime tests
tests/innovation_200/          Current study tests
configs/gear/                 Runtime and graph configuration
data/claim_graph/              Historical graph and embedding assets
data/nature_2026_testset/       Paper/reviewer source pairs
outputs/                      Runs, study artifacts and historical figures
gear/nature_multihorizon/      Retained historical scoring research
experiments/fig01/ … fig10/    Historical figure workflows
```

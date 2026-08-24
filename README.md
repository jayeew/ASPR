# ASPR-GEAR

ASPR-GEAR implements the ASPR Evidence-State Reviewer (ASPR-ESR): an
evidence-traceable five-part peer reviewer with independent Agent, optional
ASPR-Qwen, and required Graph-prior branches. The visible review contains a
contribution summary, novelty, strengths, weaknesses, and questions; it never
contains an editorial decision.

```text
ReviewRequest → PaperIR + paper-specific rubric
→ Agent Reviewer ∥ optional ASPR-Qwen ∥ Graph Prior
→ Review Fusion → bounded Evidence Supervisor
→ stability + verification → deterministic review compilation
```

The runtime is in `gear/`. Its review contract is in
`gear/review_contracts.py`; `gear/review_pipeline.py` is the sole execution
path. Paper and prior-art evidence are append-only in `EvidenceStore`. Agent and
Qwen inputs are Graph-blind. The Graph branch exposes one 0–100 ASPR Score only
to Fusion/Supervisor; it cannot create a review point, establish claim novelty,
or override a direct antecedent.

## Model backend

`codex_cli` is the default primary Agent backend: every model request uses a fresh, ephemeral,
read-only Codex CLI session with `gpt-5.6-terra` and `high` reasoning effort.
Alternatively, set `model_backend` to `openai_compatible` in a config file.
The adapter is stateless and supports providers exposing `/chat/completions`,
including DeepSeek. A ready-to-copy non-secret example is
[`configs/gear/deepseek.example.json`](configs/gear/deepseek.example.json).
Keep the provider key only in the environment variable named by `api_key_env`
(for the example, `DEEPSEEK_API_KEY`); never add it to a JSON config or commit it.
The same `--config` selection is used by `python -m gear review`, the one-pass
reconstruction runner, and the consistency-match judge; a run manifest records
the selected backend and model ID.

```bash
# Current isolated Codex CLI mode (default)
python3 -m gear review --paper /absolute/path/paper.md

# DeepSeek or another OpenAI-compatible provider
export DEEPSEEK_API_KEY='...'
python3 -m gear review \
  --config configs/gear/deepseek.example.json \
  --paper /absolute/path/paper.md
```

## Review one paper

```bash
python3 -m gear review \
  --paper /absolute/path/paper.md \
  --cutoff 2025-01-31 \
  --metadata /absolute/path/metadata.json \
  --output-dir outputs/gear/runs/example
```

The input may be Markdown or PDF. A run emits `paper_ir.json`,
`agent_review.json`, optional `qwen_review.json`, `graph_prior.json`, the
internal-only `graph_prior_audit.json`, `fusion_report.json`,
`review_state.json`, `process_diagnostic.json`, review JSON/Markdown, validation
report, immutable traces, and a hash-verified manifest. If the required Agent or
Graph score is unavailable, the system fails closed to `LIMITED` rather than
interpreting missing Graph data as a low score. Revalidate a run with:

```bash
python3 -m gear validate-run outputs/gear/runs/example
```

## Calibration assets

GEAR uses only the current Fig.1–Fig.3 calibration assets. Validate their frozen
release with:

```bash
python3 -m gear validate-assets
python3 -m gear show-calibration --verify
```

Ordinary configuration loading and non-Graph unit tests do not load these large
assets. Strict asset resolution occurs only in the Graph service and the two
explicit validation commands.

## Optional ASPR-Qwen branch

ASPR-Qwen uses a separate OpenAI-compatible endpoint and never receives the
Agent result or Graph payload. It is disabled by default. Configure it under
`aspr_qwen` (`enabled`, `model`, `base_url`, `api_key_env`, `required`). Missing
Qwen does not limit a run unless `required=true`.

## Modules and shared results

Dataset construction, review reconstruction, GEAR agent reviews, Figure 1–10
experiments, and consistency evaluation are separate modules. They exchange only
immutable, hash-verified releases through `artifact_store/`; see
[module architecture](docs/module_architecture.md).

## Review reconstruction and Agent comparison

Human reconstruction synthesizes all reviewer rounds and author replies into the
same `StructuredReview` contract emitted by the Agent. Resolved concerns are not
retained as final weaknesses. Both sides are published as immutable artifacts
and joined directly by `paper_id`; see
[module architecture](docs/module_architecture.md) for runnable reconstruction,
Agent export, publication, and comparison commands.

The legacy dev100 consistency workflow uses blinded one-to-one atomic-point matching and
reports precision, recall, F1, novelty agreement, evidence validity, and
paper-level bootstrap intervals. It is explicitly a revision-aware development
audit, not submission-time AI–Human alignment. The new
`experiments/gear/human_alignment/` framework keeps availability, conditional
quality, end-to-end utility, human–human agreement, and 5,000-iteration
paper-cluster bootstrap reporting separate; empty/empty reviews are never scored
as true matches.

## Checks

```bash
make gear-test
make gear-reconstruction-test
make gear-validate
```

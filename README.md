# ASPR-GEAR

ASPR-GEAR generates evidence-traceable five-part peer reviews. The sole review
backend is an isolated Codex CLI session using `gpt-5.6-terra` with `high`
reasoning effort. The visible review contains a contribution summary, novelty,
strengths, weaknesses, and questions; it never contains an editorial decision.

```text
ReviewRequest → PaperIR → calibration context → Codex review
→ point-level evidence checks → compiled review → verification bundle
```

The runtime is in `gear/`. Its review contract is in
`gear/review_contracts.py`; `gear/review_pipeline.py` is the sole execution
path. Paper evidence is append-only in `EvidenceStore`. Graph fields can set
inspection priority but cannot support scientific claims or novelty judgments.

## Model backend

`codex_cli` is the default: every model request uses a fresh, ephemeral,
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
  --metadata /absolute/path/metadata.json \
  --output-dir outputs/gear/runs/example
```

The input may be Markdown or PDF. Each completed run contains the normalized
paper, `PaperIR`, review JSON and Markdown, validation report, immutable traces,
and a run manifest. Revalidate a run with:

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

## Modules and shared results

Dataset construction, review reconstruction, GEAR agent reviews, Figure 1–10
experiments, and consistency evaluation are separate modules. They exchange only
immutable, hash-verified releases through `artifact_store/`; see
[module architecture](docs/module_architecture.md).

## Nature review reconstruction and consistency evaluation

The unique reconstructed reference dataset is kept in two complementary places:

- `data/gear_review_reconstruction/nature_dev100`: original session packages,
  source provenance, and imported sealed responses.
- `outputs/gear/reconstruction/nature_dev100`: one-pass reconstruction release
  artifacts and the batch manifest.

Each paper is reconstructed in an isolated Codex CLI session. Reviewer reports
provide opinions; author responses only set resolution status. Every retained
point has a reviewer-quote trace and final-paper `P:S-*` evidence. Resolved and
unverifiable items remain only in the trace and revision ledger.

```bash
python3 -m experiments.gear.review_reconstruction build-batch \
  --manifest /path/to/manifest.jsonl \
  --dataset-id nature_dev100 \
  --output-dir outputs/gear/reconstruction/nature_dev100

python3 -m experiments.gear.review_reconstruction validate-response \
  --package /path/to/package.json --response /path/to/response.json
```

The consistency workflow uses blinded one-to-one atomic-point matching and
reports precision, recall, F1, novelty agreement, evidence validity, and
paper-level bootstrap intervals. It is a development-set evaluation and is not
a publication decision signal.

## Checks

```bash
make gear-test
make gear-reconstruction-test
make gear-validate
```

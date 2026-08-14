# GEAR Nature review reconstruction

This package is intentionally outside the production runtime. Deterministic code separates roles, compiles `PaperIR`, builds one session handoff per paper, and verifies hashes. It uses the configured stateless GEAR model backend: the default is an isolated Codex CLI session using `gpt-5.6-terra`; an OpenAI-compatible API backend is also supported.

## Isolation contract

- Exactly one reconstruction conversation per paper.
- Reconstruction packages contain final-paper spans, reviewer reports, and author responses, but no graph fields, GEAR output, or legacy reconstruction.
- Author responses may assign `resolved`, `partially_resolved`, `persists`, or `unverifiable`; they cannot create an opinion.
- `resolved` and `unverifiable` traces have no target point and stay outside the SFT label.
- Every retained point has reviewer-quote trace coverage and final-paper `P:S-*` evidence.

## Commands

```bash
python -m experiments.gear.review_reconstruction build-one \
  --manifest /path/manifest.jsonl --index 0 --output-dir /path/handoff

python -m experiments.gear.review_reconstruction seal-response \
  --response /path/response.json

python -m experiments.gear.review_reconstruction validate-response \
  --package /path/package.json --response /path/response.json

# Use the configured API backend for a batch of independent reconstructions.
python -m scripts.run_gear_reconstruction_sessions \
  --config configs/gear/deepseek.example.json
```

## Freeze gates

- Schema, hash, identity, source role, and paper span integrity: 100%.
- Author-response-as-reviewer and decision leakage: zero.
- `nature_dev100` is development/non-confirmatory.
- Every response must pass schema, hash, trace, resolution-state, and final-paper-evidence validation before release.

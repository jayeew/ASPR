# AGENTS.md - ASPR-GEAR Coding Guidelines

## Project

ASPR-GEAR generates evidence-traceable peer reviews from PDFs. The only runtime package is `gear`; deprecated LATS, committee, GraphRAG, old graph scorer, and Fig.4–Fig.10 evidence must not be reintroduced.

## Commands

```bash
python -m gear review --pdf /path/paper.pdf --cutoff YYYY-MM-DD
python -m gear validate-assets
python -m gear validate-run /path/to/run

make gear-test
make gear-validate

bash scripts/train_gear_qwen_critic.sh
python experiments/gear/train_submission_calibration.py

black gear scripts tests/gear experiments/gear
ruff check gear scripts tests/gear experiments/gear
mypy gear scripts tests/gear experiments/gear --ignore-missing-imports
```

## Runtime boundaries

- GEAR may use only current Fig.1–Fig.3 graph assets.
- No legacy Agent fallback or `aspr.*` public package.
- Model clients must remain lazy; do not create clients at import time.
- Graph opportunity/control fields cannot be used as novelty evidence.
- Missing Qwen, retrieval, graph, or semantic verification must fail closed to an explicit limited result.
- Raw evidence belongs in the append-only `EvidenceStore`; state stores evidence references.
- Every major/critical review statement requires valid evidence keys.

## Style

- Python functions require type hints.
- Pydantic contracts use `extra="forbid"`.
- Use `pathlib.Path` for file I/O.
- Prefer functions under 50 lines where practical.
- Catch specific exceptions; never use bare `except`.
- Keep imports ordered: future, standard library, third party, local.
- Store secrets in environment variables, never source files.

## Structure

```text
gear/                         Runtime package
gear/nature_multihorizon/     Current Fig.1–Fig.3 model/feature code
configs/gear/                 Runtime configuration
tests/gear/                   GEAR tests
experiments/gear/             Submission calibration training
scripts/                      Dataset/training and graph data utilities
data/                         Local data and cache
outputs/                      Generated artifacts
```

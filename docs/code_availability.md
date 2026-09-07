# Code availability and reproducibility

The current innovation runtime is `gear`, with the default analysis in
`gear/innovation/` and native Claim Graph contracts in `gear/claim_graph/`.
The staged study lives in `experiments/innovation_200/`; there is no current
`aspr/` public runtime. Historical figure and scoring code remains in place for
its original research questions.

## Setup and entry points

Install repository Python dependencies in the project's working environment:

```bash
python3 -m pip install -r requirements.txt
python3 -m gear --help
python3 -m gear review --help
```

Dependencies alone do not supply local graph files, embedding/reranker weights,
manuscripts or a configured model backend. Consult the [root README](../README.md)
for input preparation, model settings and evidence boundaries.

```bash
python3 -m gear review --input-contract /absolute/path/input.json --output-dir /absolute/path/new-run
python3 -m gear validate-assets
python3 -m gear validate-run /absolute/path/new-run
make gear-test
make gear-lint
python3 -m pytest -q tests/innovation_200 tests/gear/test_historical_pdf_config.py
```

The standard run validator checks claim-level artifacts, not complete study
reports or scientific correctness. `make gear-validate` invokes native graph
asset checks. Retained old Make targets may depend on removed reviewer contracts;
they are not the acceptance suite for the new system.

## Reproduction records

A reproducible analysis should identify its actual source papers/cutoff, graph
assets and insertion policy, code revision, model/role settings, retrieval policy,
claim identities, raw evidence, stage outputs, missingness and evaluation method.
Provider token fields may be unavailable and must not be invented.

Current stages exchange ordinary files; mandatory immutable-release publication
is not the default runtime protocol. The staged study disables resume fingerprint
checks and model-response caching while reusing completed artifact files. Do not
call an in-progress/mixed-condition run frozen or hash-verified by assumption.
See [architecture](module_architecture.md) and
[study commands](../experiments/innovation_200/README.md).

The study name originates from a 200-paper sample. Its 1,000-paper extension
prepares claims and references only; an extended roster is not a completed
full-system comparison. The new Fig.1–Fig.10 plan remains under discussion.

## Historical code

[The experiment index](../experiments/README.md) identifies retained figure
workflows, sources and limitations. Historical renders do not prove that every
old runner remains compatible with current contracts. Obsolete commands such as
`make figures-current` and `make figures-nature-check` are not current documented
entry points. No old figure output is republished here as a new result.

Data/text redistribution follows [source and sharing boundaries](data_sources.md).

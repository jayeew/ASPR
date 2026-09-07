# Current innovation analysis: implementation and evaluation scope

Updated 2026-09-07. This describes implemented behavior, not completed scientific
validation. The confirmed research focus is Claim Graph + GEAR innovation
analysis; the new ten-figure experimental allocation is still under discussion.

## Default runtime

`python3 -m gear review --input-contract INPUT --output-dir RUN` uses
`knowledge` mode. It prepares shared full-text claims, runs independent GEAR and
native Claim Graph analyses, adds a separate joint Graph interpretation, and
fuses per-claim evidence. `passive` and `active` remain explicit compatibility
modes. Historical paper-level HGB allocation and the old five-part reviewer are
not the default pathway.

Claims preserve source/support spans and a supported scope. GEAR explains
manuscript support, historical antecedents and residual increment. Graph
explains historical neighbors, combinations and local structure. Fusion must
preserve scope and explain disagreements; structural rarity is not proof of
novelty. Findings require actual evidence keys. Missing evidence is a limitation,
not proof of firstness or lack of innovation. Target-paper versions cannot be
counted as independent prior art.

See [architecture](module_architecture.md) for artifacts and stage dependencies,
and [joint Graph](graph_joint_analysis.md) for union-neighborhood behavior.

## Runtime defaults versus staged study

| Setting | Standard runtime | `experiments/innovation_200/common.py` |
| --- | --- | --- |
| Model routing | Role map in `gear/model_client.py` | Luna override, role-specific low/high effort |
| Maximum target claims | 12 | 8 |
| Model-response cache | Enabled | Disabled |
| Relation stability repeat | Enabled | Disabled |
| Resume fingerprint checks | Enabled | Disabled |
| Historical literature PDFs | Off by default; environment/config controlled | Same environment/config policy |
| Saved stage reuse | Completed artifacts reused | Completed artifacts reused |

Model response caching and stage artifact reuse are separate mechanisms. The
study does not rerun successful outputs merely because response caching is off.
Changed inputs or uniform-condition comparisons need separate output directories.
An intentional resume after changing PDF policy retains old evidence/results and
must be described as mixed conditions. The target manuscript remains full text.

## Current staged experiment

The [study README](../experiments/innovation_200/README.md) documents independent
sampling, reference reconstruction, claims, GEAR, Graph, reporting, reference
comparison, pairwise preferences and summarization. The retained `innovation_200`
name does not constrain the roster to 200. The extension to 1,000 prepares claims
and reviewer references; full-system execution is separate.

Eight whole-paper report variants are implemented: `direct_llm`, `gear`, `graph`,
`fusion`, `fusion_no_joint`, `fusion_no_metrics`, `fusion_no_citation_paths`, and
`fusion_text_only`. The direct baseline reads the manuscript independently.
Other variants use shared claims and their selected evidence. The full study
fusion report includes joint Graph; standard per-claim fusion does not.

These are implemented information conditions, not automatically clean causal
ablations. Some variants replace branch interpretations with filtered raw facts,
changing both available information and intermediate processing. A confirmatory
module-benefit claim requires matching the relevant generation path, model and
resource budget. `gear/innovation/experiments.py` additionally implements ordinary
RAG and shared-claim controls; these are not currently among the eight study
report variants.

## Reference and evaluation meaning

References come from existing reviewer quotations, with contribution identity,
reviewer/round, explicit reasons and original source context. Tier A supports
innovation-stance comparison; tier B supports identification. Main evaluation
uses the last explicit opinion selected by the reconstruction stage. Reviewer
silence is not a false-positive label. Final published manuscripts versus earlier
review rounds require version-aware interpretation; these results alone do not
establish submission-time review performance.

Report matching and anonymous pairwise preferences are model judgments. They
must not be labeled newly collected human preferences. Reviewer consistency
provides context rather than an automatic validity gate. Reference agreement,
evidence correctness and usefulness are distinct outcomes. Always report actual
available denominators and distinguish failure/missingness from zero performance.

## Separate frozen-study workflow

`scripts/innovation_experiment.py` and `gear/innovation/` also support the earlier
v2 `screen`, `freeze`, `run`, `diagnostics`, `summarize` workflow with development
and test splits. It includes blind and neutral specified-target tasks,
source/fingerprint checks, direct/RAG controls and diagnostics. Those policies
apply to that workflow, not automatically to the staged `innovation_200` study.
Existing acceptance files record the runs that produced them, not a permanent
current-system scientific pass.

## Validation and legacy boundaries

```bash
make gear-test
make gear-lint
python3 -m pytest -q tests/innovation_200 tests/gear/test_historical_pdf_config.py
python3 -m gear validate-assets
python3 -m gear validate-run /absolute/path/standard-knowledge-run
```

Asset validation checks files/index dimensions, not graph scientific validity.
Run validation expects standard claim-level fusion and fingerprints, so it is
not a whole-study validator when those artifacts/checks are disabled. It does
not certify joint analysis or whole-paper report correctness.

`make gear-legacy-test` retains earlier interface tests for migration diagnostics.
Older reconstruction/module Make targets can reference removed contracts; do
not restore deprecated runtime interfaces merely to pass them. Code availability,
passing tests and completed execution are distinct from evidence that the system
improves innovation analysis. Historical Fig.1–Fig.10 results remain scoped to
their generating methods.

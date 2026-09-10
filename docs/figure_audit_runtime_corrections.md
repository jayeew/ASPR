# Figure audit: runtime corrections, 2026-09-09

These changes repair code paths identified by the figure audit. They do not
constitute new model evaluations, independently verified scientific judgments,
completed fusion results, or a rebuilt historical database.

## Evaluation version and metadata blinding

`experiments/innovation_200/blinding.py` creates a judge-visible report with only
`body` and `references`. It removes the producer's `system` and `paper_id` metadata.
Sources receive neutral `S0001`/`P0001` identifiers based on their evidence content;
identical source content receives the same alias across reports in a comparison.
The report's source-ID citations are replaced consistently. Titles, DOI/URL,
source types and quoted passages are preserved. Native IDs and the report-to-system
mapping remain outside the judge prompt in an adjacent `.mapping.json` file.

The human-reference judge now returns only `matches`; its response schema no longer
contains the system-name enum. Runtime metadata is restored after response validation.
Both evaluators save their exact prompt, visible payload and schema in a separate
`.request.json` file. The source mappings and original report hashes are not sent
to the judge. Report wording and provenance types can still suggest the method;
this corrects metadata exposure and is not a guarantee of semantic anonymity.

New evaluations, status records and logs go under:

```
<study>/evaluations/identity_blind_v2/
    human/<system>/<paper_id>.json
    pairwise/<paper_id>__fusion_vs_<system>.json
    reviewer_consistency/
    status/
    logs/
```

The old `human_evaluation/` and `pairwise/` artifacts are not overwritten, even
when the new evaluator is called with `--overwrite`. That option only overwrites
the selected v2 artifact. Consumers must deliberately select an evaluation version;
legacy summary scripts still reading the old directories have not become v2 results.
The new contradiction rate counts only `contradiction=true AND scope=same`.

Example commands for an explicitly selected completed report cohort (these have
**not** been executed as model runs in this change):

```
python3 experiments/innovation_200/evaluate_human.py --study <frozen-study> --systems direct_llm graph
python3 experiments/innovation_200/compare_reports.py --study <frozen-study>
```

The first comparison remains **Graph + joint versus Direct LLM**. Source-bound AI
reviewer-reference matching is not newly collected independent human correctness.

## Confounded ablations fail closed

The previous `fusion_no_metrics`, `fusion_no_citation_paths` and `fusion_text_only`
paths substituted raw facts for interpreted Graph outputs. Their generation now
raises `UncontrolledAblationError` before reading artifacts or invoking a model.
The report generator defaults to the five remaining supported generation paths;
an explicit request for a blocked variant still raises. Old reports are preserved.
`build_system_context(..., inspect_legacy_variant=True)` exists only to inspect the
old information views; the report generator never uses this bypass.

`ablation_protocol.ablation_plan()` records the uncompleted requirements:

- Freeze the same cohort, claims, history text union, cutoff and model routing.
- Mask facts before interpretation, then recompute dependent branch and joint
  analyses and caches through the same stages.
- Audit the complete exposed payload, including `source_catalog`, and save hashes.
- Preserve the writer task and whole-paper reasoning opportunity, and record
  resource budgets and input/output token counts.
- For joint, use the same claims and historical text union with one whole-paper
  synthesis opportunity in the control.

No implementation of those future controlled variants is claimed here. The
existing no-joint path remains generatable, but the planner marks causal-effect
readiness false until its text/budget/interpretation controls are verified.

## Citation integrity

`02_build_paper_graph.py` now preserves anomalous raw references in the shard
chunks, including original citing/cited IDs and source shard/line. Same-work
citations, invalid IDs and duplicates are excluded from ordinary citation edges.
Normalization also catches old resumed chunks, checks endpoints and deduplicates
citations with an on-disk SQLite set; it writes `paper_edge_anomalies.jsonl` with
source locators. Node generation is independent, so isolated nodes remain.

`06_build_paper_graph_index.py` validates endpoints and records selfloops,
invalid IDs and duplicate rows in its `paper_edge_anomalies` table with input file,
batch and row locators. It refuses to resume an existing index that contains
same-work citations or missing endpoints and asks for a separate output path.

`ClaimGraphRuntime._paper_path` excludes same-work edges and counts distinct
paths when reading a legacy database, without modifying it. New graph policy is
`threshold_parent_path_v2` throughout single-claim, batch and joint paths. Old
policy caches are rejected and require a new run directory, including in study
mode where general fingerprint checks are disabled.

The original historical assets and saved Graph facts have not been changed.
Affected historical path annotations and downstream interpretations must be
recomputed in a new versioned run before reporting new scientific results. Date
inversions are not automatically removed: publication-version/online-first
metadata must first be checked. This correction does not resolve canonical alias
or historical source-record errors that require investigation of the original
database.

## Offline verification

The added tests cover visible schemas/payloads, external mappings, preservation
of source passages and legacy outputs, same-scope contradictions, blocked mixed
paths, raw/shard anomaly provenance, old-chunk cleaning, index resume, isolated
nodes and read-only runtime path correction. They use synthetic fixtures and
mock judges; no model/API call is required.

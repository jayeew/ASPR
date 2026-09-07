# Current Claim Graph + GEAR architecture

Updated against the working-tree implementation on 2026-09-07. The public
runtime package is `gear`; default mode is `knowledge`. This replaces the
previous architecture based on five-part reviews and mandatory release exchange.

## Code boundaries

| Component | Implementation | Responsibility |
| --- | --- | --- |
| CLI/dispatcher | `gear/cli.py`, `gear/review_pipeline.py` | Input and stage selection |
| Shared contributions | `gear/innovation/shared.py`, `gear/grounding.py` | Full-text claims, consolidation and manuscript support |
| GEAR evidence | `gear/evidence_supervisor.py`, `gear/prior_art.py` | Retrieval, comparisons and residual increment |
| Graph facts | `gear/claim_attribution.py` | Native graph lookup/insertion; no paper-score allocation |
| Branch interpretation | `gear/innovation/analysis.py` | Source-bound GEAR, Graph and fusion assessments |
| Joint Graph | `gear/innovation/joint_graph.py` | Union-neighborhood facts and paper-level interpretation |
| Orchestration | `gear/innovation/pipeline.py` | Saved stages and claim-level fusion/report |
| Study reports | `experiments/innovation_200/reporting.py` | Whole-paper reports and ablation views |
| Study evaluation | `experiments/innovation_200/evaluate_human.py`, `compare_reports.py` | Reference matching, reviewer consistency and AI preferences |

Graph and GEAR share neutral contribution identities and scope, without reading
each other's judgments before fusion. Standard `all` executes GEAR then Graph
(including joint Graph), then claim fusion. Logical independence does not mean
this command runs both branches concurrently; the study streams separate stages.

## Contracts and artifacts

`InnovationPaperInput`, `GearClaim`, and `GraphFactCard` live in
`gear/review_contracts.py`; `ClaimSet`, `Assessment`, `Finding`, and
`AnalysisResult` live in `gear/innovation/contracts.py`. `PaperIR` preserves
manuscript spans; `EvidenceStore` preserves append-only source evidence.
Contracts reject extra fields.

| Standard run artifact | Meaning |
| --- | --- |
| `innovation_input.json` | Metadata, manuscript path and cutoff |
| `shared/paper_ir.json`, `shared/claims.json` | Shared manuscript and grounded claims |
| `gear/NN/gear_card.json`, `assessment.json` | Evidence status and claim interpretation |
| `graph/NN/` | Graph facts in evidence storage and claim interpretation |
| `gear/analysis.json`, `graph/analysis.json` | Branch assessments and limitations |
| `graph/joint/facts.json`, `analysis.json`, `report.md`, `status.json` | Joint facts, interpretation and status when available |
| `fusion/NN/`, `fusion/analysis.json` | Source-bound claim fusion |
| `fusion/innovation_report.md` | Standard report organized by claim |
| `model_usage.jsonl` | Model calls, timings and available usage |

Joint Graph is a separate paper-level result. Standard claim fusion does not
automatically consume it. The study's whole-paper `fusion` report consumes GEAR,
per-claim Graph and joint Graph through its report writer; it is distinct from
the runtime's `fusion/analysis.json`.

Findings cite existing evidence keys. Branch status (`complete`, `limited`,
`failed`) differs from stance (`recognized`, `incremental_or_limited`,
`challenged`, `unresolved`). Missing evidence cannot imply negative novelty or
global firstness. Valid evidence keys alone do not establish scientific truth.

## Graph and temporal scope

Historical claim extraction uses the local 2023–2025 Nature abstract corpus;
target claims are mined from full text. Paper citation and native Claim Graph
are separate layers. Five claim types (`METHOD`, `FINDING`, `MECHANISM`,
`RESOURCE`, `THEORY`) are node roles, not graph partitions. Parent-paper paths
annotate semantic relations, not claim support or derivation.

Runtime insertion is temporary, excludes self and date-ineligible neighbors,
and defaults to at most ten neighbors with cosine > 0.5. Sparse neighborhoods
remain explicit. Current thresholded metrics cannot inherit percentiles from
older selection rules. Joint insertion uses the union neighborhood and existing
historical edges; it neither invents within-paper target edges nor sums
individual perturbations. See [Graph details](graph_joint_analysis.md).

The graph is a bounded historical corpus, not all scientific literature.
Community mixing and connectivity are local structural facts, not validated
causality, global novelty or predicted impact.

## Study artifacts and evaluation

`experiments/innovation_200/` retains its original name; `papers.jsonl` defines
the active roster. The 1,000-paper extension updates the roster and runs shared
claims and reviewer-reference reconstruction, not all downstream stages.

The study exchanges ordinary JSON/JSONL/Markdown/CSV within its directory:

- `human_refs/{paper_id}.json`: source-bound reviewer contribution references;
- `papers/{paper_id}/`: shared, GEAR and Graph artifacts;
- `reports/{system}/`: eight report variants and citation appendices;
- `human_evaluation/`, `reviewer_consistency/`, `pairwise/`: comparisons;
- `status/`, `logs/`, `tables/`, `summary.json`, `summary.md`: stage records and
  summaries when generated.

Reference reconstruction does not expose reviewer judgments to system branches.
Tier A contains explicit innovation judgments; tier B supports identification.
The study retains the last explicit opinion for its main comparison and keeps
historical rounds inspectable. Matching/preference calls are AI judgments, not
new human annotations. Reference silence is not a false-positive label.

`gear/innovation/experiments.py` also contains shared-claim direct/RAG controls;
these differ from the study's eight report variants. See
[study instructions](../experiments/innovation_200/README.md).

## Resume and validation limits

Default runtime enables response caching, relation stability checks and resume
fingerprint checks. The study disables all three, limits claims to eight and
uses Luna role overrides with low/high efforts. Completed stage files can still
be reused when model-response caching is disabled. Interrupted evidence attempts
are archived; raw evidence remains append-only.

Use separate directories for uniform new experimental conditions. Explicitly
chosen mixed-policy resumes retain prior results, including historical full text
obtained before PDF downloading was disabled. They cannot be reported as uniform
abstract-only reruns.

`validate-assets` checks required graph files and basic index/embedding shape
consistency, not full semantics, temporal leakage or scientific validity.
`validate-run` checks fingerprints, claim identities, coverage and evidence keys
in three standard claim-level branches. It does not check joint interpretations,
whole-paper reports or scientific truth. The study disables fingerprints and
need not produce standard fusion artifacts, so this is not a drop-in study
validator. Legacy runs without shared claims are only displayed by the CLI.

## Historical interfaces

`gear/module_cli.py`, `gear/module_registry.py`, `artifact_store/`, older
`StructuredReview` consumers and `experiments/gear/` preserve earlier work.
They are not the default v2 exchange contract; check compatibility before use.
Current study artifacts do not universally require old human-agreement gates,
HGB promotion or immutable publication releases. Preserve historical research
without importing its conclusions or runtime assumptions into the new system.

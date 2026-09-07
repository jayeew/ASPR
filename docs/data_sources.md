# Data sources and interpretation boundaries

The current system analyzes target-paper full text with GEAR prior-art evidence
and a native historical Claim Graph. Historical diffusion-scoring datasets are
retained separately; they are not default innovation-runtime inputs.

## Current sources

| Source | Role | Boundary |
| --- | --- | --- |
| Local 2023–2025 Nature abstracts | Historical claim extraction and native Claim Graph | A bounded venue/time corpus, not all literature |
| OpenAlex work metadata and references | Parent paper graph, identity and retrieval | Metadata/citation paths do not prove claim-level derivation |
| Target-paper Markdown or PDF | Shared contribution extraction and internal support | Author claims and manuscript-supported scope must be distinguished |
| Retrieved historical abstracts/full text | GEAR prior-art comparison | Preserve source type, date, identity and actual passages |
| Published transparent peer-review files | Source-bound contribution references | Incomplete, reviewer/round dependent; not exhaustive novelty ground truth |

Native graph artifacts live under `data/claim_graph`; offline construction code
is under `scripts/claim_graph`. `canonical_target_works.parquet` records the
actual historical parent-paper population; do not substitute counts from old
design documents. Runtime uses temporary cutoff-filtered insertion and does not
rewrite historical graph assets.

The [2026 paired corpus](../data/nature_2026_testset/README.md) contains 1,000
paper/review pairs with its own manifest and source validation. This corpus,
the study's active `papers.jsonl`, completed stage outputs and valid evaluation
denominators are different populations. A script or directory named
`innovation_200` can operate on the extended roster.

## Retrieval and time

Historical literature PDF downloading defaults off and is controlled by
`GEAR_HISTORICAL_PDF_ENABLED`. Abstract evidence remains usable within its scope;
reference-only metadata cannot establish an external scientific result. Target
full text is unaffected by this switch. Previously retrieved full text survives
resume after policy changes, which must be reported as mixed conditions.

Preserve publication/cutoff identity and distinguish target versions from
independent prior art. Semantic neighbors, parent citation paths and local
community structure are candidate/context evidence, not a global firstness or
causal claim. Current neighbor thresholds differ from older calibrations, so
old graph percentiles must not be applied to new raw values.

## References and sharing

Reviewer extraction preserves quotes, reviewer/round, contribution identity and
reasons. Tier A contains explicit innovation judgments; B supports identification.
AI extraction, report matching and preference calls must be labeled as such.
Missing reviewer coverage does not make a system contribution false. Published
paper analysis against earlier review rounds is not automatically submission-time
validation.

Local manuscripts, review files and PDFs remain subject to their source licenses.
Do not redistribute third-party full text without permission or a license that
allows it. Share code, compatible configuration, derived tables, source locators,
method descriptions and provenance as permitted; avoid publishing secrets or
restricted cached text. A citation appendix is not itself a redistribution license.

## Historical results

Fig.1–Fig.10 outputs retain their original model, data and protocol scope. Old
HGB prediction, ASPR-Qwen case, Graph-guidance and review-agreement results do
not establish current-system performance. New figure planning is in discussion;
scientific completion must be assessed from the actual new study outputs.

# Primary Codex high-recall discovery-screening protocol v3

## Purpose

This protocol defines the nominal AI screening stream for later evidence-
saturation rounds without using a local language model. It is deliberately a
high-sensitivity routing rule, not the final eligibility judgment.

Use only the named blind AI worksheet. Do not inspect H1/H2 decisions,
downstream outputs, model results, or invalidated local-model files. Do not
call Ollama, Qwen, another local model, an external LLM API, or repository
model-running code.

## Frozen routing rule

- Explicit OpenAlex non-English records receive
  `language_judgment=non_en`, `decision=exclude`, and
  `E_LANGUAGE_NON_ENGLISH`.
- Records with an English or unknown OpenAlex language and any nonblank title
  or abstract receive `language_judgment=en` and `decision=include` for H2
  routing, regardless of apparent topical relevance.
- Records with neither usable title nor abstract receive
  `language_judgment=uncertain`, `decision=exclude`, and
  `E_INSUFFICIENT_METADATA`.

Use an exact title span, or abstract fallback, for both language and
eligibility evidence. The note must state that inclusion is a high-recall
routing decision requiring independent H1/H2 screening. Because every
English/unknown record is an AI include, every such record enters the
mandatory H2 queue after H1; the primary rule therefore cannot silently
exclude an English relevant study.

## Provenance

Use `reviewer_role=AI`, `draft_method=primary_codex_high_recall_routing`,
`independent_ai_review_status=complete`,
`independent_ai_reviewer_id=primary_codex_ai_screening_v3`,
`independent_ai_review_action=high_recall_literature_screening`, a stable run
ID, ISO-8601 review time, the primary Codex task/thread ID in `model_digest`,
and the SHA-256 of this protocol. Create and register an exact-hash sibling
manifest before import.

The rule is deterministic for a frozen worksheet. It does not make or imply
final inclusion decisions and cannot determine `K`, `Q`, `P`, `M`, `D`, or
`F`.

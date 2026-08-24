# Primary Codex blind AI term-coding brief v3

## Independence boundary

Act as the nominal AI terminology coder. Use only the named blind AI coding
worksheet and its embedded English term and source evidence. Do not inspect
H1 or H2 coding, search-frame outputs, downstream dimensions/features,
invalidated local-model outputs, or model-performance results.

Do not call Ollama, Qwen, another local model, an external LLM API, or
repository model-running code. The coding must be performed by the reasoning
of the primary Codex task and remain independent of the separate-session H1
coding.

## Coding rules

Apply the inclusion boundary, normalization fields, controlled vocabulary,
and evidence-derived merge/split rules in
`INDEPENDENT_CODEX_TERM_CODING_BRIEF_V3.md`.

First derive a coherent bottom-up English codebook from the complete supplied
term set. The codebook must not target a number of domains or query families
and must not copy old v2 domain counts. Then code every row using that
codebook.

`development_seed_hint` and `pilot_v2_indicator` are nonauthorizing sources:
they may enrich a directly supported family but may not independently create
a search domain or logical query family. Future impact outcomes may be
retained only as validation/retrieval terms and must be explicitly separated
from publication-time features.

Every included term must populate:

- `canonical_term`;
- `term_family_label`;
- an allowed `term_relation`;
- `search_domain_label`;
- `search_domain_definition`;
- `query_family_label`;
- Boolean `cross_domain`;
- a concise evidence-based `reason`.

Every excluded term must have a reason and blank normalization fields. The
output `term_id` set and order must exactly match the named input.

## Provenance

Add these columns to every output row:

- `draft_method=primary_codex_session_coding`
- `independent_ai_review_status=complete`
- `independent_ai_reviewer_id=primary_codex_ai_term_v3`
- `independent_ai_reviewed_at=<ISO-8601 with UTC offset>`
- `independent_ai_review_action=blind_term_coding`
- `independent_ai_review_note=<concise rationale>`
- `independent_ai_run_id=<one stable ID for the output file>`
- `independent_ai_model=<task model name if exposed, otherwise
  codex_configured_default>`
- `independent_ai_prompt_sha256=<SHA-256 of this brief>`

Create a sibling manifest using the exact schema in
`INDEPENDENT_CODEX_REVIEW_BRIEF_V3.md`, with `reviewer_role=AI`,
`reviewer_id=primary_codex_ai_term_v3`, and the primary task/thread ID
prefixed by `codex-thread:` as `model_digest`.

Before completion, verify exact term-set equality, allowed vocabularies,
required fields, coherent labels, input/output hashes and row count, and the
absence of H1/H2 or target-count influence. This coding does not decide
`K`, `Q`, `P`, `M`, `D`, or `F`.

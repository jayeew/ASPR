# Fig.10 — Same-path ASPR module ablation

**Question.** Which errors arise when graph evidence, ASPR-Qwen, prior-art
retrieval, evidence tracing, fusion, or verification is removed?

**Required design.** Full ASPR and every variant must share the same model,
prompt, retrieval cache, scorer, decoding settings, and 50 cases; exactly one
switch may differ. Human preference requires 750 completed blind decisions.

**Current state.** The existing 400 automatic rows were produced through
different generation paths and are retained only as a comparability
diagnostic. They are not treated as the requested main ablation.

**Gate.** Until the same-path rerun and human labels exist, the output is
`BLOCKED_COMPARABILITY`; it must not publish inferred deltas, preference, or
quality–cost claims.

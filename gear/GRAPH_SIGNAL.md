# Graph signal boundary

GEAR deliberately separates forecast, retrieval, evidence, and review judgment.

1. `nature_multihorizon` predicts five-year scholarly diffusion. The public
   value is `prospective_5y_diffusion_percentile`; it is neither novelty nor an
   acceptance probability.
2. The percentile is a diagnostic forecast. Experimental routing may reorder
   only a fixed local/remote pool; it cannot change the graph-blind draft.
3. Point-in-time Graph topology is a bounded candidate probe. It cannot delete
   claim-level provider queries. The first entrance is an exact manuscript
   citation attached to the claim span. It must have substantive prior-work
   text and pass the candidate comparability gate before replacing the last
   candidate and entering verification at rank one. Generic Graph seeds remain
   lower priority. A topology seed is never review evidence.
4. A review point may change only after full-text retrieval, claim-level
   relation classification, semantic verification, and evidence storage.

The frozen routing formula is:

```text
q = percentile / 100
q_effective = 0.5 + coverage * (q - 0.5)
remote_weight = 0.25 + 0.5 * q_effective
local_weight = 1 - remote_weight
```

The formula is active only when feature coverage is at least 0.8 and
`abs(q_effective - 0.5) >= 0.1`; otherwise routing is exactly neutral. This
prevents weak, central, or partial-feature forecasts from creating arbitrary
candidate permutations. Unavailable forecast also uses `q_effective = 0.5`
and emits an explicit limited Graph packet. Historical cases never call live
OpenAlex. A year-only candidate is eligible only when
`source_year < cutoff_year`.

## Safe topology admission

`claim_linked_citation_rank1_v3` applies four do-no-harm rules:

1. Preserve ordinary claim-level provider queries; Graph direct fetches use a
   separately capped probe and require a matched placebo probe in evaluation.
2. Protect the five highest baseline candidates and classify them before any
   Graph candidate.
3. Require a topology candidate to be `comparable` or `partial`, match at least
   two scientific-frame fields, have exact `claim_alignment >= 0.65`, cover at
   least one essential claim facet, and beat the baseline cutoff score by 0.05
   before admission. Missing exact-alignment output rejects the candidate.
4. Put at most one semantically admitted exact claim citation at rank one,
   replace the tail under the unchanged candidate cap, and require full paired-
   span relation verification before any review correction.

## Review labels

The runtime evidence status is one of `not_assessed`, `manuscript_supported`,
`evidence_qualified`, `evidence_challenged`, or `inconclusive`.
`positive/mixed/negative` is compatibility output only and never controls
Graph routing or evaluation.

Nature reconstruction uses issue traces: reviewer quote, reviewer and round,
optional author response, final-paper evidence, and one of `persists`,
`partially_resolved`, `resolved`, or `unverifiable`. Author responses may
establish resolution state but cannot invent an issue.

## Evaluation status

The implementation supports four equal-budget variants: `neutral`, `score`,
`score_topology`, and `placebo_graph`, plus blinded DIRECT/PARTIAL/PARALLEL/
DISTANT/UNVERIFIABLE judgment and the specified 3-paper and 10-paper gates.

Old three-paper and ten-paper Graph evaluations were retired during the
Primary16 migration because they used the former GEAR score lineage. No current
Graph-benefit conclusion is claimed until a new Primary16 evaluation is run.

The frozen graph context ends in 2022. Any target reviewed after 2022 is
therefore forbidden from using a historical score-table lookup. GEAR recomputes
the target paper's exact Fig.2/3 Primary16 fields at its review cutoff, then
applies the unchanged D5 Primary16 HGB, calibrators, and OOF percentile
reference. The model is never refitted for a recent paper. The former v6.1
19-input model is not a fallback. Feature completeness and frozen-history
coverage are recorded separately; incomplete historical coverage is shrunk
toward neutral before routing.

Claim-linked topology is now enabled by default. Score-only routing remains
disabled: it was non-inferior in 10/10 papers but its median `Delta NDCG@10` was
zero. HGB remains an independent five-year diffusion forecast. Any future score
routing model must be trained against candidate-level relation yield or material
review change, not future citation diffusion.

Paper references are now parsed as individual numbered bibliography entries.
The search frame first exposes references cited in the exact target span (for
example `[27]`), instead of treating a bibliography page as one reference.
Internal `REF-*` identifiers are not OpenAlex work IDs; a candidate is ignored
unless DOI or conservative title resolution succeeds and the publication date is
cutoff-safe. Unresolved references remain explicit coverage gaps.

## Manuscript citation edges

A bibliography edge written in the submitted manuscript is point-in-time safe by
construction. GEAR may resolve that declared reference by DOI or conservative
exact-title matching and retrieve its already-published version-of-record text
after the review cutoff. Only immutable work text is used; current citation
counts, current neighbors, and other post-cutoff graph metadata remain forbidden.
The resolved title, publication year, source URL, retrieval time, and content hash
must be retained. This exception does not apply to shared-reference, co-citation,
or citing-work edges, which still require a cutoff-safe frozen Graph snapshot.

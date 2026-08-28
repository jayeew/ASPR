# Graph signal boundary

GEAR separates novelty evidence from prospective structural impact while
combining them through one registered deterministic contract.

1. `nature_multihorizon` predicts five-year scholarly diffusion. The public
   value is `prospective_5y_diffusion_percentile`; it is neither novelty nor an
   acceptance probability.
2. The percentile is display/ranking metadata and is never treated as a
   probability. Calibrated expected diffusion is reliability-shrunk toward the
   field-year base and may enter only the evidence-gated structural score.
3. Point-in-time Graph topology is a bounded candidate probe. It cannot delete
   claim-level provider queries. The first entrance is an exact manuscript
   citation attached to the claim span. It must have substantive prior-work
   text and pass the candidate comparability gate before replacing the last
   candidate and entering verification at rank one. Generic Graph seeds remain
   lower priority. A topology seed is never review evidence.
4. A novelty review point may change only after full-text retrieval, claim-level
   relation classification, semantic verification, and evidence storage.
5. The Graph scalar can never establish novelty, manuscript validity, or erase
   a direct antecedent. Holding those evidence variables fixed, its marginal
   effect on the separate structural-innovation score is monotone non-negative.

The preserved runtime heads are U (uptake), D (diffusion or fold-local excess
diffusion), P (structural perturbation), A (claim attribution/pathway), and R
(reliability). When a registered excess-diffusion head is present it replaces
total diffusion inside the shrinkage step; otherwise frozen D is used and the
result remains limited. Each fold-local target head carries its training-scope
hash and provenance.

Production admits D-excess/P only through `gear_structural_head_release_v1`.
The sidecar is hash-bound to the canonical Primary16 release and registry. Its
manifest must declare T0-only inputs and outer-training-fold-only target fits;
historical rows must be strict OOF predictions with a fold identifier, while
recent rows must be exact-cutoff predictions from the frozen model. A missing,
corrupt, ambiguous, post-cutoff, or protocol-mismatched sidecar never receives
a numeric substitute: D-excess/P are omitted, frozen HGB-D remains available,
and the Graph signal is explicitly limited.

Claim attribution has two intentionally separate runtime identities. The
default `deterministic_t0` phase-one baseline uses only the frozen claim type,
manuscript centrality, forecast-anatomy roles, and a manuscript-derived pathway
hypothesis. It is never reported as learned. The optional `learned_t0` head is
admitted only through `gear_claim_attribution_release_v1`: a non-executable
linear JSON model, exact T0 feature schema, runtime replay, and distinct
temporal/domain Gate-1 reports must all be hash-bound. Those reports must bind
their result to the candidate model hash and declare that future citing
contexts are labels used during development, never inference features. Missing
or corrupt learned assets produce an empty, explicitly limited attribution;
runtime does not silently substitute the deterministic baseline. The current
Gate-1 OOF estimates are evaluation evidence only and are not a frozen
production model.

Selective retrieval is gated separately. A promoted selector may choose A1–A5
only after its uplift lower confidence bound and wrong-correction,
unsupported-claim, and cost guardrails pass. If action policy execution is
enabled without a promoted selector, runtime records `abstain`; it does not
fall back to an unvalidated score-to-action mapping.

Production action selection requires `gear_graph_action_policy_release_v1`.
The release binds the complete balanced 150-row randomized log, its exact
90-row development subset, paired 60-row Graph/no-Graph policy holdouts, the
portable A0–A5 Q heads and conservative decision rules, a runtime replay, and
the final dual-holdout Gate-2 report by SHA-256. Gate 2 must pass every check
and all three guardrails, and must identify the exact candidate model, T0
feature schema, and data hashes. The self-contained immutable release also
binds the formal frozen-replay manifest, rescue-source audit, and Stage A/B/C
runtime code/config cohort audits. The runtime features are claim count, mean
manuscript centrality, publication year, shrunk diffusion, reliability, the
two forecast-anatomy shares, HGB-P perturbation potential, and forecast
prediction uncertainty (explicit interval width when available, otherwise
`1 - reliability`). P is required and is never replaced with D. Future
outcomes are never inference inputs. The post-retrieval GEAR evidence gap is
explicitly excluded from this phase-one pre-retrieval decision because no
stable T0 field exists; the release manifest records that rationale. Missing,
corrupt, partial, unpaired, or non-replayable assets force `abstain` and a
limited run; the randomized selector remains explicitly experimental.

The frozen routing formula is:

```text
q = percentile / 100
q_effective = 0.5 + coverage * (q - 0.5)
remote_weight = 0.25 + 0.5 * q_effective
local_weight = 1 - remote_weight
```

When the score-routing experiment is enabled, the fixed query budget is
allocated according to these weights (for eight slots, 0 and 100 map to 6/2
and 2/6 local/remote). The production action policy may instead abstain unless
an action's uplift lower confidence bound is positive and its guardrails pass.
Unavailable forecast uses `q_effective = 0.5` and emits an explicit limited
Graph packet. Historical cases never call live OpenAlex. A year-only candidate is eligible only when
`source_year < cutoff_year`.

## Structural innovation fusion

The graph-blind claim inventory is frozen from manuscript spans before reviewer
execution. For each claim, RelationCards and RetrievalCoverageCards determine:

```text
N = manuscript_validity * evidence_coverage
    * (1 - antecedent_risk) * residual_novelty

I = N^alpha * [epsilon + (1-epsilon)D]^beta
    * [epsilon + (1-epsilon)P]^eta * mechanism_validity^gamma
```

All exponents are non-negative and `beta > 0`. The perturbation factor is
omitted while HGB-P is unavailable. Complete direct-antecedent coverage forces
`N=0`, so diffusion cannot compensate. Paper aggregation uses centrality from
the frozen claim inventory and a top-three noisy-OR; Graph never sets claim
centrality. The text compiler only renders frozen numeric fields.

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

Claim-linked topology remains available under matched caps. Scalar routing is a
correctness-tested experimental arm, not the scientific integration mechanism.
The production retrieval policy must be trained against candidate-level
relation yield or material review change, record propensities, and abstain when
its uplift lower bound is non-positive. HGB remains an independent five-year
diffusion forecast and enters the final system through structural fusion.

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

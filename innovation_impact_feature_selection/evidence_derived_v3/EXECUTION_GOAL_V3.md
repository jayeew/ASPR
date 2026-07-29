# Evidence-saturation execution goal

## Objective

Without changing ASPR business code or v2, complete an English-only,
evidence-derived and reproducible workflow that:

1. derives search terminology from source-linked literature;
2. standardizes terms and obtains role-separated AI/H1 coding with frozen
   provenance plus H2 adjudication;
3. lets those decisions, hidden-seed recall, and PRESS determine `K`, `Q`,
   and `P` without a numerical target;
4. retrieves the frozen evidence frame and follows citations to dual-endpoint
   saturation;
5. extracts canonical indicator families before deriving candidate
   dimensions;
6. lets fixed T0, formula, data, bias, validity, redundancy, and
   independent-team gates determine `M`, `D`, and `F`;
7. emits a complete provenance chain and the same result hash when rerun from
   the same frozen inputs.

## Time-box strategy

- Use the local OpenAlex snapshot for frozen historical provenance and
  seed/indexability checks.
- Use free OpenAlex API keys only for deterministic sampled discovery,
  post-snapshot coverage, recall validation, and citation retrieval.
- Precompute the first three deterministic review rounds because at least
  three consecutive dual-zero rounds are required even in the earliest
  stopping case.
- Run local AI screening/coding in advance, while keeping H1 worksheets blind.
- Generate H2 queues only after H1 import and include only mandated
  adjudication/audit records.
- Resume every API and review stage from SQLite checkpoints.
- Acquire only OpenAlex-explicit open-access PDF candidates, hash and freeze
  them for later English full-text and formula verification.

The machine portion is designed to finish within hours. Human decisions may
be entered directly or may adopt an automated draft after review. In the
latter case, the exact file must be explicitly attested and hashed, and the
audit must disclose that provenance rather than describe the draft as
originally manual or independently generated. Unreviewed automated entries
must never be backfilled as H1/H2 decisions.

## Completion definition

This goal is complete only when all of the following are true:

- every active deterministic discovery stratum is completely retrieved;
- all discovery batches through the stopping round are fully reviewed;
- H2 approves at least three consecutive rounds with zero new
  non-redundant English term families and zero new canonical indicator
  families;
- AI/H1 disagreements and all mandatory H2 records are adjudicated;
- at least one independently supplied eligible English hidden H2 validation
  seed is present, and its DOI set exactly reconciles to the completed H2
  review-search and bidirectional citation-tracking logs;
- all eligible indexable development and hidden seeds are recalled;
- PRESS has no unresolved issue;
- the frozen `K/Q/P` search frame is fully retrieved and screened;
- every final indicator has verified English full-text formula evidence and
  passes every hard gate;
- every retained predictive dimension has at least one final indicator and
  support from at least two independent research teams;
- the audit has no blocker and repeated frozen-input runs reproduce
  `K/Q/P/M/D/F` and the deterministic decision hash.

No interim AI proposal, pilot label, quota, or model-performance result may be
reported as a final domain, query, dimension, or indicator decision.

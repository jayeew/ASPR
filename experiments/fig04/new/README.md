# Fig. 4new — layered validation result figure

This directory prepares the source-backed data needed to evaluate GEAR+Graph
and renders the current six-panel Fig. 4new. It does not add a new runtime,
Pydantic contract, schema requirement, or public API.

Run from the repository root:

```bash
python3 -m experiments.fig04.new.prepare_data
python3 -m experiments.fig04.new.run
python3 -m experiments.fig04.new.tests
```

The default output is `outputs/fig04/new/data_20260829/`.

## Data products

- `paper_level_scores.csv`, `cohort_audit.csv`: paper-level overall (241),
  temporal (49), and domain (68) cohorts with source paths.
- `integration_validity.csv`, `integration_contrasts.csv`: three-arm Spearman
  validity, top-decile outcomes, and paired 5,000-replicate bootstrap intervals.
- `graph_predictive_validity.csv`: frozen Graph predictive-validity results.
- `metric_reconciliation.csv`: frozen-report versus row-recomputed checks.
- `claim_level_adoption.csv`, `claim_adoption_validity.csv`: 1,442 claim rows
  and the frozen temporal/domain Gate-1 results.
- `claim_b_enriched_tasks.jsonl`, `claim_b_independent_*`,
  `claim_b_final_validity.csv`: the evidence-complete subset of the existing
  Claim B cohort, with real pre-cutoff antecedent quotations and direct
  independent Terra reviews. The source 30-task pack is preserved; tasks with
  no quoted antecedent remain explicitly excluded rather than relabeled.
- `claim_c_replacement_*`, `claim_c_independent_*`: the original 60
  equal-budget, content-different top-3 tasks plus only the domain-coverage
  additions. The final 78 tasks cover all 12 domains, with at least five in 11
  domains and four in neuroscience. These reuse the existing Claim C
  task/review formats and hide arm identity, graph scores, and future outcomes
  from reviewers.
- `reviewer_alignment_*`: one-to-one GEAR/published-review reconstruction
  audit, blind match tasks, wrong-paper controls, an aspect-shuffle control,
  and a label template using the existing four-way point-match labels.
- `required_data.csv`: the remaining data required before pending claims can be
  calculated.

## Evaluation policy and remaining limitation

1. Missing Claim B relation evidence is never interpreted as a negative label.
   Final Claim B validity is calculated only from independently reviewed,
   evidence-complete tasks.
2. Claim C reviews are generated in fresh Codex CLI sessions. A completed AI
   review is a valid final review source; reviewer identity, cross-session
   agreement, conversation hashes, and session identifiers are not gates.
3. Published-review alignment uses independent labels for both correct
   and wrong-paper packages. Soft matching treats `SAME_POINT` and
   `PARTIAL_POINT` as matches. A paper-specific alignment claim is withheld
   until correct-pair soft F1 exceeds the 95th percentile of the wrong-paper
   control.
4. Panel e's registered primary endpoint is paper-level Recall@3: whether the
   three highest-ranked claims cover at least one frozen independently labeled
   important claim. Candidate k values `{1,2,3,5}` are planning-only and may
   not be selected on strict held-out outcomes; the predeclared k=3 endpoint is
   evaluated from strict fold rows only. Precision@3, NDCG@3, and MRR are
   secondary metrics.

## Incremental commands

All model judgments use independent `codex exec --ephemeral` calls with
`gpt-5.6-terra` and `model_reasoning_effort="medium"`; no OpenAI API client is
used.

```bash
python3 -m experiments.fig04.new.incremental_prepare --enrich-claim-b
python3 -m experiments.fig04.new.run_independent_reviews claim-b
python3 -m experiments.fig04.new.extend_claim_c
python3 -m experiments.fig04.new.run_independent_reviews claim-c
python3 -m experiments.fig04.new.run_independent_reviews matches
python3 -m experiments.fig04.new.finalize_results
```

Historical and incremental rows are treated as one generation. Output audits
retain source paths and model names needed for interpretation, but impose no
hash or session-consistency gate.

Claim C preserves the content-different top-3 criterion. Final coverage is at
least five papers in 11 domains and four in neuroscience; further attempts to
change the neuroscience top-3 were stopped without relaxing the criterion.

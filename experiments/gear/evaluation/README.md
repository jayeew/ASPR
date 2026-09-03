# Unified GEAR evaluation

This package evaluates only the current `gear` runtime. It does not invoke the
EACL novelty or ReviewGrounder repositories and does not produce a cross-track
composite score.

The gate policy is frozen in
`configs/gear/evaluation_nature_graph_gate_v2.json`. It intentionally contains
no fabricated cases: case rows are added only after reviewer quote, round,
final-paper evidence, point-in-time Graph provenance, and D5 percentile have
all passed validation.

Run an eligible evaluation manifest with:

```bash
python scripts/run_gear_evaluation.py all \
  --manifest /path/to/traceable-nature-gate-manifest.json \
  --judge-config configs/gear/evaluator_codex_terra.json \
  --output-dir outputs/gear/evaluation/<run-id> \
  --resume
```

Stages are `preflight`, `run-clean`, `run-faults`,
`run-graph-ablations`, `prepare-judges`, `run-judges`, `score`, and
`report`. Checkpoints are content-addressed by the manifest, evaluator config,
tracked diff, untracked source hashes, prompts, and response schemas. A dirty
tree therefore cannot silently reuse a checkpoint created from different code.

The legacy retrieval comparison retains four matched variants: `neutral`,
`score`, `score_topology`, and `placebo_graph`. Every variant reuses the same
graph-blind draft and candidate pools under identical resource caps. Score
routing now changes local/remote slot geometry under the same total budget; it
is a correctness-tested experimental arm, not the primary integration claim.

The primary Graph KPIs are claim-relevant verified relation yield, material
correction yield, equal logical resources, and independent review preference.
Accepted-paper novelty polarity is only a compatibility diagnostic. Driver and
guardrail families remain separate. Null means not measured; judge failures are
excluded rather than replaced with zero.

The primary integration endpoint is implemented in
`joint_structural_innovation_eval.py`: value of evidence-gated HGB fusion minus
GEAR evidence only against held-out future structural outcomes. Supporting
modules audit claim-attribution conservation, score-range restriction, and
monotone fusion. `graph_action_randomized_runner.py` records matched-budget A0–A5
assignments and propensities; `off_policy_value_eval.py` computes a doubly robust
value estimate. Runtime selection must abstain unless the candidate action has a
positive uplift lower confidence bound and passes correction/claim guardrails.

## Rescue-plan stages

The complete A/B/C implementation is exposed as independent, fail-closed
commands. Implementation availability is not an empirical pass: Gate 1 and Gate
2 remain `not_identifiable` until their registered real datasets are supplied.

```bash
# Stage A: frozen D5 OOF validity, Gate 0, decile cohort, three-arm readiness
python -m experiments.gear.evaluation.run_stage_a_validation \
  --output-dir outputs/gear/stage_a_validation --per-decile 20 \
  --gear-evidence /path/stage_a_paper_evidence.parquet

# Licensed real manuscripts and graph-blind GEAR evidence
python -m experiments.gear.evaluation.acquire_oof_manuscripts \
  --cohort /path/frozen_oof_cohort.csv --output-dir /path/acquisition
python -m experiments.gear.evaluation.collect_stage_a_gear_evidence \
  --runs-dir /path/gear_runs --output-dir /path/gear_evidence

# Real future citation contexts and graph-blind claim-adoption labels
python -m experiments.gear.evaluation.acquire_claim_adoption_contexts \
  --benchmark-manifest /path/benchmark_manifest.json --output-dir /path/contexts
python -m experiments.gear.evaluation.label_claim_adoption \
  --claims /path/stage_a_claim_evidence.parquet \
  --contexts /path/claim_contexts.jsonl \
  --context-papers /path/citation_context_papers.jsonl \
  --config configs/gear/future_adoption_labeler.json \
  --output-dir /path/claim_adoption --workers 4

# Stage B: cross-fitted U/D/P/R heads from fold-local prepared targets
python -m experiments.gear.evaluation.run_structural_heads \
  --input /path/prepared_targets.parquet \
  --feature EF0017 --feature EF0052 \
  --output-dir outputs/gear/structural_heads

# Stage B: claim attribution from aligned citing-context adoption labels
python -m experiments.gear.evaluation.run_claim_attribution \
  --input /path/claim_context_labels.parquet \
  --feature claim_centrality --feature claim_type_code \
  --output-dir outputs/gear/claim_attribution

# Stage C: frozen development rules and independent policy holdout
python -m experiments.gear.evaluation.run_action_policy_evaluation \
  --development /path/action_development.parquet \
  --holdout /path/action_holdout.parquet \
  --feature claim_count --feature mean_claim_centrality \
  --feature publication_year --feature graph_shrunk_diffusion \
  --feature graph_reliability --feature graph_structural_share \
  --feature graph_opportunity_share --feature graph_perturbation_potential \
  --feature graph_prediction_uncertainty \
  --output-dir outputs/gear/action_policy

# Gate 2 requires separate frozen temporal and domain holdouts.
python -m experiments.gear.evaluation.run_rescue_plan \
  --output-dir outputs/gear/rescue_plan \
  --stage-a-gear-evidence /path/stage_a_paper_evidence.parquet \
  --stage-a-per-decile 400 \
  --gate1-data /path/gate1_mechanism_dataset.parquet \
  --gate2-temporal /path/temporal/gate2_integration_frame.parquet \
  --gate2-domain /path/domain/gate2_integration_frame.parquet \
  --gate2-policy /path/graph_policy_holdout_scored.parquet \
  --gate2-no-graph-policy /path/no_graph_policy_holdout_scored.parquet \
  --stage-c-randomized-data /path/randomized_action_log.parquet
```

The Graph-feature evaluation writes a portable linear A0–A5 Q model and a
60-row runtime replay. `run_rescue_plan` hash-binds them to the paired Gate-2
report only when Gate 2 passes. `scripts/promote_graph_action_policy.py` then
revalidates the complete 150/90/60 chain plus the frozen source/config/runtime
audits before atomically writing a self-contained production release; it cannot
publish partial, unfrozen, or failed evidence.

`perturbation_targets.build_perturbation_components` deterministically derives
boundary expansion, community mixing, dependency displacement, and path
shortening/claim adoption from auditable future-graph summaries.
`targets_v6.build_fold_local_structural_targets` then fits diffusion references,
the excess-diffusion null model, and all four perturbation component references
on the outer-training fold only. Every target head emits a content hash and fit
scope. Feature registries containing future/outcome/target columns are rejected
by the U/D/P/R and claim-attribution trainers.

`build_real_perturbation_inputs.py` derives all four graph families from real
future-citer reference lists and writes `complete_graph_cohort.csv` for the
licensed-manuscript join. `run_real_perturbation_validation.py` reports both a
strict latest-block forward-temporal holdout and leave-one-domain-out
predictions plus field/year-shuffled controls. Earlier blocked-time CV is
diagnostic only because its early folds can train on later blocks. The temporal
hybrid uses domain-OOF predictions for development rows and prior-block-only
predictions for the latest holdout. `build_gate1_mechanism_dataset.py` joins
these predictions to real claim-adoption labels and emits explicit
Graph=0/Graph=1 monotonicity
counterfactuals. `build_gate2_integration_frame.py` performs the frozen
claim-to-paper aggregation. The final evaluator requires both registered
holdouts to pass; one cannot substitute for the other, and development rows
fail closed.

The Stage C path separates randomized data collection from policy promotion:
A0–A5 assignments record exact propensity and matched budget, development data
freeze uncertainty margins and guardrails, and an independent holdout is
evaluated with DR and SWITCH-DR. Holdout guardrails use the recorded
propensities. The confirmatory runner requires exactly 90 development and 60
holdout rows, balanced A0–A5 assignments, disjoint unique paper/context IDs,
frozen folds, and matching content hashes for the Graph and no-Graph policy
ablations. Uplift and the Graph-vs-no-Graph contrast use paired DR influence
scores on all 60 rows; any non-finite score fails closed. Enabling a runtime action policy without an
injected promoted selector produces an explicit `abstain` decision.
`collect_randomized_action_outcomes.py` excludes any run whose executed action
does not match its frozen assignment or whose semantic/relation evidence is
unavailable; exclusions remain in a separate audit artifact.

After all automated gates pass, `scripts/run_gear_graph_rescue_full.sh` builds a
label-free Claim-B/Claim-C review pack and writes only a readiness validation.
It never treats an unreviewed pack as completed evidence. Each task needs at
least one evidence-backed review; a separate AI session is a valid reviewer.
Human identity, blind labeling, reviewer calibration, multi-reviewer agreement,
and third-party adjudication are not required. Once the reviews are supplied,
run:

```bash
GEAR_FROZEN_REPLAY_MANIFEST=/absolute/path/frozen_replay_manifest.json \
  bash scripts/finalize_gear_graph_rescue.sh
```

The finalize command first validates completed independent-session reviews with
`--require-completed`, then invokes the strict rescue completion audit. Missing,
partial, or structurally invalid reviews stop the command before an overall
claim can be allowed.

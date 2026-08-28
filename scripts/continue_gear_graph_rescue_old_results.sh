#!/usr/bin/env bash
set -euo pipefail

stage_a_root="${GEAR_STAGE_A_OUTPUT_ROOT:-outputs/gear/stage_a_real_gear_20260827}"
stage_b_root="${GEAR_STAGE_B_OUTPUT_ROOT:-outputs/gear/stage_b_targeted_expansion_20260828}"
stage_c_root="${GEAR_STAGE_C_OUTPUT_ROOT:-outputs/gear/stage_c_randomized_actions_20260828}"
runs_dir="${stage_a_root}/batch_runs_bounded_120"
evidence_dir="${stage_a_root}/batch_evidence"
stage_b_manifest="${stage_b_root}/complete_graph_benchmark_manifest.json"
stage_c_manifest="${stage_c_root}/randomized_manifest_150.json"

wait_for_pid() {
  local pid="$1"
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 20
  done
}

for pid in ${GEAR_WAIT_FOR_PIDS:-}; do
  wait_for_pid "${pid}"
done

python3 -m experiments.gear.evaluation.collect_stage_a_gear_evidence \
  --runs-dir "${runs_dir}" \
  --manifest "${stage_b_manifest}" \
  --output-dir "${evidence_dir}"

python3 -m experiments.gear.evaluation.rescue_postprocess_audit evidence \
  --summary "${evidence_dir}/evidence_collection_summary.json" \
  --claim-evidence "${evidence_dir}/stage_a_claim_evidence.parquet" \
  --paper-evidence "${evidence_dir}/stage_a_paper_evidence.parquet" \
  --manifest "${stage_b_manifest}" \
  --output "${stage_b_root}/postprocess_audits/evidence_coverage.json"

python3 -m experiments.gear.evaluation.collect_randomized_action_outcomes \
  --manifest "${stage_c_manifest}" \
  --runs-dir "${stage_c_root}/runs" \
  --output-dir "${stage_c_root}/outcomes"

python3 -m experiments.gear.evaluation.label_claim_adoption \
  --claims "${evidence_dir}/stage_a_claim_evidence.parquet" \
  --contexts "${stage_b_root}/claim_contexts_241/citation_contexts.jsonl" \
  --context-papers "${stage_b_root}/claim_contexts_241/citation_context_papers.jsonl" \
  --output-dir "${stage_b_root}/claim_adoption_labels" \
  --config configs/gear/future_adoption_labeler.json \
  --workers "${GEAR_CLAIM_LABEL_WORKERS:-4}"

python3 -m experiments.gear.evaluation.build_gate1_mechanism_dataset \
  --claim-adoption "${stage_b_root}/claim_adoption_labels/claim_adoption_labels.parquet" \
  --perturbation-predictions "${stage_b_root}/hgb_p_validation_241/hgb_p_forward_temporal_hybrid_predictions.parquet" \
  --split-manifest "${stage_b_root}/integration_holdout_manifest.json" \
  --output-dir "${stage_b_root}/gate1_temporal"

python3 -m experiments.gear.evaluation.build_gate1_mechanism_dataset \
  --claim-adoption "${stage_b_root}/claim_adoption_labels/claim_adoption_labels.parquet" \
  --perturbation-predictions "${stage_b_root}/hgb_p_validation_241/hgb_p_domain_oof_predictions.parquet" \
  --split-manifest "${stage_b_root}/integration_holdout_manifest.json" \
  --output-dir "${stage_b_root}/gate1_domain"

python3 -m experiments.gear.evaluation.build_gate2_integration_frame \
  --gate1 "${stage_b_root}/gate1_temporal/gate1_mechanism_dataset.parquet" \
  --split temporal_holdout --split joint_time_domain_holdout \
  --output-dir "${stage_b_root}/gate2_temporal"

python3 -m experiments.gear.evaluation.build_gate2_integration_frame \
  --gate1 "${stage_b_root}/gate1_domain/gate1_mechanism_dataset.parquet" \
  --split domain_holdout --split joint_time_domain_holdout \
  --output-dir "${stage_b_root}/gate2_domain"

python3 -m experiments.gear.evaluation.rescue_postprocess_audit claim-gates \
  --summary "${stage_b_root}/claim_adoption_labels/claim_adoption_summary.json" \
  --labels "${stage_b_root}/claim_adoption_labels/claim_adoption_labels.parquet" \
  --context-papers "${stage_b_root}/claim_contexts_241/citation_context_papers.jsonl" \
  --split-manifest "${stage_b_root}/integration_holdout_manifest.json" \
  --temporal-gate1 "${stage_b_root}/gate1_temporal/gate1_mechanism_dataset.parquet" \
  --domain-gate1 "${stage_b_root}/gate1_domain/gate1_mechanism_dataset.parquet" \
  --temporal-gate2 "${stage_b_root}/gate2_temporal/gate2_integration_frame.parquet" \
  --domain-gate2 "${stage_b_root}/gate2_domain/gate2_integration_frame.parquet" \
  --output "${stage_b_root}/postprocess_audits/claim_gate_coverage.json"

python3 -m experiments.gear.evaluation.train_claim_attribution_head \
  --temporal-gate1 "${stage_b_root}/gate1_temporal/gate1_mechanism_dataset.parquet" \
  --domain-gate1 "${stage_b_root}/gate1_domain/gate1_mechanism_dataset.parquet" \
  --output-dir "${stage_b_root}/claim_attribution_release" \
  --release-id "${GEAR_CLAIM_ATTRIBUTION_RELEASE_ID:-claim-attribution-t0-20260828}"

python3 - "${stage_c_root}/outcomes/randomized_action_outcome_report.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("included_cases") != 150 or not report.get("all_actions_observed"):
    raise SystemExit("Stage C needs selective repair: exact valid A0-A5 cohort unavailable")
PY

python3 -m experiments.gear.evaluation.rescue_postprocess_audit randomized \
  --report "${stage_c_root}/outcomes/randomized_action_outcome_report.json" \
  --action-log "${stage_c_root}/outcomes/randomized_action_log.parquet" \
  --manifest "${stage_c_manifest}" \
  --output "${stage_c_root}/postprocess_audits/randomized_outcomes.json"

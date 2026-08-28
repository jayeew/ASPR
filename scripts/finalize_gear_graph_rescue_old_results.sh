#!/usr/bin/env bash
set -euo pipefail

stage_a_root="${GEAR_STAGE_A_OUTPUT_ROOT:-outputs/gear/stage_a_real_gear_20260827}"
stage_b_root="${GEAR_STAGE_B_OUTPUT_ROOT:-outputs/gear/stage_b_targeted_expansion_20260828}"
stage_c_root="${GEAR_STAGE_C_OUTPUT_ROOT:-outputs/gear/stage_c_randomized_actions_20260828}"
final_root="${GEAR_FINAL_OUTPUT_ROOT:-outputs/gear/graph_rescue_final_old_results_20260828}"
expert_pack_root="${GEAR_EXPERT_PACK_ROOT:-${final_root}/expert_annotation_pack}"
evidence_dir="${stage_a_root}/batch_evidence"
stage_c_outcomes="${GEAR_STAGE_C_OUTCOMES_ROOT:-${stage_c_root}/outcomes_consolidated}"
stage_c_policy_graph="${GEAR_STAGE_C_POLICY_GRAPH_ROOT:-${stage_c_root}/policy_graph_consolidated}"
stage_c_policy_no_graph="${GEAR_STAGE_C_POLICY_NO_GRAPH_ROOT:-${stage_c_root}/policy_no_graph_consolidated}"

for pid in ${GEAR_WAIT_FOR_PIDS:-}; do
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 20
  done
done

required=(
  "${evidence_dir}/stage_a_paper_evidence.parquet"
  "${evidence_dir}/stage_a_claim_evidence.parquet"
  "${stage_b_root}/gate1_temporal/gate1_mechanism_dataset.parquet"
  "${stage_b_root}/gate2_temporal/gate2_integration_frame.parquet"
  "${stage_b_root}/gate2_domain/gate2_integration_frame.parquet"
  "${stage_c_outcomes}/randomized_action_log.parquet"
  "${stage_c_policy_graph}/policy_holdout_scored.parquet"
  "${stage_c_policy_no_graph}/policy_holdout_scored.parquet"
)
for path in "${required[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "required finalization artifact missing: ${path}" >&2
    exit 1
  fi
done

python3 -m experiments.gear.evaluation.run_rescue_plan \
  --output-dir "${final_root}" \
  --stage-a-gear-evidence "${evidence_dir}/stage_a_paper_evidence.parquet" \
  --stage-a-per-decile 400 \
  --gate1-data "${stage_b_root}/gate1_temporal/gate1_mechanism_dataset.parquet" \
  --gate2-temporal "${stage_b_root}/gate2_temporal/gate2_integration_frame.parquet" \
  --gate2-domain "${stage_b_root}/gate2_domain/gate2_integration_frame.parquet" \
  --gate2-policy "${stage_c_policy_graph}/policy_holdout_scored.parquet" \
  --gate2-no-graph-policy "${stage_c_policy_no_graph}/policy_holdout_scored.parquet" \
  --stage-c-randomized-data "${stage_c_outcomes}/randomized_action_log.parquet"

python3 -m experiments.gear.evaluation.expert_annotation_pack build \
  --stage-b-claims "${evidence_dir}/stage_a_claim_evidence.parquet" \
  --gate1 "${stage_b_root}/gate1_temporal/gate1_mechanism_dataset.parquet" \
  --stage-c-log "${stage_c_outcomes}/randomized_action_log.parquet" \
  --stage-b-manifest "${stage_b_root}/complete_graph_benchmark_manifest.json" \
  --stage-c-manifest "${stage_c_root}/randomized_manifest_150.json" \
  --runs-dir "${stage_a_root}/batch_runs_bounded_120" \
  --output-dir "${expert_pack_root}"

python3 -m experiments.gear.evaluation.expert_annotation_pack validate \
  --pack-dir "${expert_pack_root}" \
  --output "${final_root}/expert_annotation_pack_ready_validation.json"

python3 - "${final_root}/rescue_plan_status.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not report.get("overall_claim_allowed"):
    raise SystemExit("Rescue plan failed closed: one or more confirmatory gates failed")
PY

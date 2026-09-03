#!/usr/bin/env bash
set -euo pipefail

stage_a_input_root="${GEAR_STAGE_A_INPUT_ROOT:-outputs/gear/stage_a_real_gear_20260827}"
stage_b_input_root="${GEAR_STAGE_B_INPUT_ROOT:-outputs/gear/stage_b_targeted_expansion_20260828}"
stage_c_input_root="${GEAR_STAGE_C_INPUT_ROOT:-outputs/gear/stage_c_randomized_actions_20260828}"
stage_a_root="${GEAR_STAGE_A_OUTPUT_ROOT:-${stage_a_input_root}}"
stage_b_root="${GEAR_STAGE_B_OUTPUT_ROOT:-${stage_b_input_root}}"
stage_c_root="${GEAR_STAGE_C_OUTPUT_ROOT:-${stage_c_input_root}}"
final_root="${GEAR_FINAL_OUTPUT_ROOT:-outputs/gear/graph_rescue_final_20260828}"
action_policy_release_manifest="${GEAR_ACTION_POLICY_RELEASE_MANIFEST:-data/calibration/graph_calibration/gear_graph_action_policy_release_v1/manifest.json}"
action_policy_release_id="${GEAR_ACTION_POLICY_RELEASE_ID:-graph-action-policy-20260828}"
claim_attribution_release_id="${GEAR_CLAIM_ATTRIBUTION_RELEASE_ID:-claim-attribution-t0-20260828}"
structural_release_root="${GEAR_STRUCTURAL_RELEASE_ROOT:-data/calibration/graph_calibration/gear_structural_head_release_v1}"
expert_pack_root="${GEAR_EXPERT_PACK_ROOT:-${final_root}/expert_annotation_pack}"
benchmark_workers="${GEAR_BENCHMARK_WORKERS:-2}"
runs_dir="${stage_a_root}/batch_runs_bounded_120"
evidence_dir="${stage_a_root}/batch_evidence"

mkdir -p "${stage_a_root}" "${stage_b_root}" "${stage_c_root}" "${final_root}"

python3 -m gear validate-assets \
  --config configs/gear/stage_a_evidence_only_bounded.json \
  > "${final_root}/stage_ab_asset_validation.json"

runtime_code_audit_args=()
stage_ab_config_audit_args=()
stage_c_config_audit_args=()
if [[ -n "${GEAR_EXPECTED_RUNTIME_CODE_SHA256:-}" ]]; then
  runtime_code_audit_args=(
    --expected-code-sha256 "${GEAR_EXPECTED_RUNTIME_CODE_SHA256}"
  )
fi
if [[ -n "${GEAR_EXPECTED_STAGE_AB_CONFIG_SHA256:-}" ]]; then
  stage_ab_config_audit_args=(
    --expected-config-sha256 "${GEAR_EXPECTED_STAGE_AB_CONFIG_SHA256}"
  )
fi
if [[ -n "${GEAR_EXPECTED_STAGE_C_CONFIG_SHA256:-}" ]]; then
  stage_c_config_audit_args=(
    --expected-config-sha256 "${GEAR_EXPECTED_STAGE_C_CONFIG_SHA256}"
  )
fi

if [[ -n "${GEAR_WAIT_FOR_PID:-}" ]]; then
  while kill -0 "${GEAR_WAIT_FOR_PID}" 2>/dev/null; do
    sleep 30
  done
fi

python3 -m gear benchmark \
  --manifest "${stage_a_input_root}/benchmark_manifest.json" \
  --output-dir "${runs_dir}" \
  --config configs/gear/stage_a_evidence_only_bounded.json \
  --workers "${benchmark_workers}" --resume --retry-failed --full-artifacts \
  --case-timeout-seconds 3600

python3 -m experiments.gear.evaluation.audit_runtime_cohort \
  --manifest "${stage_a_input_root}/benchmark_manifest.json" \
  --runs-dir "${runs_dir}" \
  --output "${stage_a_root}/postprocess_audits/runtime_cohort.json" \
  "${runtime_code_audit_args[@]}" "${stage_ab_config_audit_args[@]}"

python3 -m experiments.gear.evaluation.collect_stage_a_gear_evidence \
  --runs-dir "${runs_dir}" \
  --manifest "${stage_a_input_root}/benchmark_manifest.json" \
  --output-dir "${evidence_dir}"

python3 -m experiments.gear.evaluation.run_stage_a_validation \
  --output-dir "${stage_a_root}/stage_a_validation_120" \
  --per-decile 400 \
  --gear-evidence "${evidence_dir}/stage_a_paper_evidence.parquet"

python3 - "${stage_a_root}/stage_a_validation_120/stage_a_validation.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not report["conclusion"]["stage_a_established"]:
    raise SystemExit("Stage A failed closed: registered validation is not established")
PY

python3 -m gear benchmark \
  --manifest "${stage_b_input_root}/complete_graph_benchmark_manifest.json" \
  --output-dir "${runs_dir}" \
  --config configs/gear/stage_a_evidence_only_bounded.json \
  --workers "${benchmark_workers}" --resume --retry-failed --full-artifacts \
  --case-timeout-seconds 3600

python3 -m experiments.gear.evaluation.audit_runtime_cohort \
  --manifest "${stage_b_input_root}/complete_graph_benchmark_manifest.json" \
  --runs-dir "${runs_dir}" \
  --output "${stage_b_root}/postprocess_audits/runtime_cohort.json" \
  "${runtime_code_audit_args[@]}" "${stage_ab_config_audit_args[@]}"

python3 -m experiments.gear.evaluation.audit_hgb_p_predictions \
  --temporal "${stage_b_input_root}/hgb_p_validation_241/hgb_p_forward_temporal_hybrid_predictions.parquet" \
  --domain "${stage_b_input_root}/hgb_p_validation_241/hgb_p_domain_oof_predictions.parquet" \
  --output "${stage_b_root}/hgb_p_validation_241/hgb_p_prediction_audit.json"

python3 -m experiments.gear.evaluation.build_claim_a_report \
  --stage-a "${stage_a_root}/stage_a_validation_120/stage_a_validation.json" \
  --structural-validation "${structural_release_root}/validation_report.json" \
  --prediction-audit "${stage_b_root}/hgb_p_validation_241/hgb_p_prediction_audit.json" \
  --coverage-audit "${structural_release_root}/coverage_audit.json" \
  --output "${stage_b_root}/hgb_p_validation_241/claim_a_bounded_validation.json"

python3 -m experiments.gear.evaluation.collect_stage_a_gear_evidence \
  --runs-dir "${runs_dir}" \
  --manifest "${stage_b_input_root}/complete_graph_benchmark_manifest.json" \
  --output-dir "${evidence_dir}"

python3 -m experiments.gear.evaluation.rescue_postprocess_audit evidence \
  --summary "${evidence_dir}/evidence_collection_summary.json" \
  --claim-evidence "${evidence_dir}/stage_a_claim_evidence.parquet" \
  --paper-evidence "${evidence_dir}/stage_a_paper_evidence.parquet" \
  --manifest "${stage_b_input_root}/complete_graph_benchmark_manifest.json" \
  --output "${stage_b_root}/postprocess_audits/evidence_coverage.json"

python3 -m experiments.gear.evaluation.label_claim_adoption \
  --claims "${evidence_dir}/stage_a_claim_evidence.parquet" \
  --contexts "${stage_b_input_root}/claim_contexts_241/citation_contexts.jsonl" \
  --context-papers "${stage_b_input_root}/claim_contexts_241/citation_context_papers.jsonl" \
  --output-dir "${stage_b_root}/claim_adoption_labels" \
  --config configs/gear/future_adoption_labeler.json \
  --workers 4

python3 -m experiments.gear.evaluation.build_gate1_mechanism_dataset \
  --claim-adoption "${stage_b_root}/claim_adoption_labels/claim_adoption_labels.parquet" \
  --perturbation-predictions "${stage_b_input_root}/hgb_p_validation_241/hgb_p_forward_temporal_hybrid_predictions.parquet" \
  --split-manifest "${stage_b_input_root}/integration_holdout_manifest.json" \
  --output-dir "${stage_b_root}/gate1_temporal"

python3 -m experiments.gear.evaluation.build_gate1_mechanism_dataset \
  --claim-adoption "${stage_b_root}/claim_adoption_labels/claim_adoption_labels.parquet" \
  --perturbation-predictions "${stage_b_input_root}/hgb_p_validation_241/hgb_p_domain_oof_predictions.parquet" \
  --split-manifest "${stage_b_input_root}/integration_holdout_manifest.json" \
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
  --context-papers "${stage_b_input_root}/claim_contexts_241/citation_context_papers.jsonl" \
  --split-manifest "${stage_b_input_root}/integration_holdout_manifest.json" \
  --temporal-gate1 "${stage_b_root}/gate1_temporal/gate1_mechanism_dataset.parquet" \
  --domain-gate1 "${stage_b_root}/gate1_domain/gate1_mechanism_dataset.parquet" \
  --temporal-gate2 "${stage_b_root}/gate2_temporal/gate2_integration_frame.parquet" \
  --domain-gate2 "${stage_b_root}/gate2_domain/gate2_integration_frame.parquet" \
  --output "${stage_b_root}/postprocess_audits/claim_gate_coverage.json"

python3 -m experiments.gear.evaluation.train_claim_attribution_head \
  --temporal-gate1 "${stage_b_root}/gate1_temporal/gate1_mechanism_dataset.parquet" \
  --domain-gate1 "${stage_b_root}/gate1_domain/gate1_mechanism_dataset.parquet" \
  --output-dir "${stage_b_root}/claim_attribution_release" \
  --release-id "${claim_attribution_release_id}"

python3 -m gear benchmark \
  --manifest "${stage_c_input_root}/randomized_pilot_manifest_6.json" \
  --output-dir "${stage_c_root}/runs" \
  --config configs/gear/stage_c_randomized_action.json \
  --workers "${benchmark_workers}" --resume --retry-failed --full-artifacts \
  --case-timeout-seconds 3600

python3 -m experiments.gear.evaluation.collect_randomized_action_outcomes \
  --manifest "${stage_c_input_root}/randomized_pilot_manifest_6.json" \
  --runs-dir "${stage_c_root}/runs" \
  --output-dir "${stage_c_root}/pilot_outcomes"

python3 - "${stage_c_root}/pilot_outcomes/randomized_action_outcome_report.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report["included_cases"] != 6 or not report["all_actions_observed"]:
    raise SystemExit("Stage C pilot failed closed: incomplete or mismatched A0-A5")
PY

python3 -m gear benchmark \
  --manifest "${stage_c_input_root}/randomized_manifest_150.json" \
  --output-dir "${stage_c_root}/runs" \
  --config configs/gear/stage_c_randomized_action.json \
  --workers "${benchmark_workers}" --resume --retry-failed --full-artifacts \
  --case-timeout-seconds 3600

python3 -m experiments.gear.evaluation.audit_runtime_cohort \
  --manifest "${stage_c_input_root}/randomized_manifest_150.json" \
  --runs-dir "${stage_c_root}/runs" \
  --output "${stage_c_root}/postprocess_audits/runtime_cohort.json" \
  "${runtime_code_audit_args[@]}" "${stage_c_config_audit_args[@]}"

python3 -m experiments.gear.evaluation.collect_randomized_action_outcomes \
  --manifest "${stage_c_input_root}/randomized_manifest_150.json" \
  --runs-dir "${stage_c_root}/runs" \
  --output-dir "${stage_c_root}/outcomes"

python3 - "${stage_c_root}/outcomes/randomized_action_outcome_report.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report["included_cases"] != 150 or not report["all_actions_observed"]:
    raise SystemExit("Stage C randomized log failed closed")
if report["split_counts"] != {"confirmatory_holdout": 60, "development": 90}:
    raise SystemExit("Stage C randomized split counts changed")
PY

python3 -m experiments.gear.evaluation.rescue_postprocess_audit randomized \
  --report "${stage_c_root}/outcomes/randomized_action_outcome_report.json" \
  --action-log "${stage_c_root}/outcomes/randomized_action_log.parquet" \
  --manifest "${stage_c_input_root}/randomized_manifest_150.json" \
  --output "${stage_c_root}/postprocess_audits/randomized_outcomes.json"

python3 -m experiments.gear.evaluation.run_action_policy_evaluation \
  --development "${stage_c_root}/outcomes/development_action_log.parquet" \
  --holdout "${stage_c_root}/outcomes/holdout_action_log.parquet" \
  --feature claim_count --feature mean_claim_centrality \
  --feature publication_year \
  --output-dir "${stage_c_root}/policy_no_graph"

python3 -m experiments.gear.evaluation.run_action_policy_evaluation \
  --development "${stage_c_root}/outcomes/development_action_log.parquet" \
  --holdout "${stage_c_root}/outcomes/holdout_action_log.parquet" \
  --feature claim_count --feature mean_claim_centrality \
  --feature publication_year --feature graph_shrunk_diffusion \
  --feature graph_reliability --feature graph_structural_share \
  --feature graph_opportunity_share --feature graph_perturbation_potential \
  --feature graph_prediction_uncertainty \
  --output-dir "${stage_c_root}/policy_graph"

python3 -m experiments.gear.evaluation.run_rescue_plan \
  --output-dir "${final_root}" \
  --stage-a-gear-evidence "${evidence_dir}/stage_a_paper_evidence.parquet" \
  --stage-a-per-decile 400 \
  --gate1-data "${stage_b_root}/gate1_temporal/gate1_mechanism_dataset.parquet" \
  --gate2-temporal "${stage_b_root}/gate2_temporal/gate2_integration_frame.parquet" \
  --gate2-domain "${stage_b_root}/gate2_domain/gate2_integration_frame.parquet" \
  --gate2-policy "${stage_c_root}/policy_graph/policy_holdout_scored.parquet" \
  --gate2-no-graph-policy "${stage_c_root}/policy_no_graph/policy_holdout_scored.parquet" \
  --stage-c-randomized-data "${stage_c_root}/outcomes/randomized_action_log.parquet"

python3 - "${final_root}/rescue_plan_status.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not report["overall_claim_allowed"]:
    raise SystemExit("Rescue plan failed closed: one or more confirmatory gates failed")
PY

# Build and validate the independent-session review pack only after all
# automated gates pass. This does not claim that reviews have been completed.
python3 -m experiments.gear.evaluation.expert_annotation_pack build \
  --stage-b-claims "${evidence_dir}/stage_a_claim_evidence.parquet" \
  --gate1 "${stage_b_root}/gate1_temporal/gate1_mechanism_dataset.parquet" \
  --stage-c-log "${stage_c_root}/outcomes/randomized_action_log.parquet" \
  --stage-b-manifest "${stage_b_input_root}/complete_graph_benchmark_manifest.json" \
  --stage-c-manifest "${stage_c_input_root}/randomized_manifest_150.json" \
  --runs-dir "${runs_dir}" \
  --output-dir "${expert_pack_root}"

python3 -m experiments.gear.evaluation.expert_annotation_pack validate \
  --pack-dir "${expert_pack_root}" \
  --output "${final_root}/expert_annotation_pack_ready_validation.json"

if [[ -n "${GEAR_EXPECTED_RESCUE_SOURCE_SHA256:-}" ]]; then
  python3 -m experiments.gear.evaluation.source_fingerprint \
    --expected-sha256 "${GEAR_EXPECTED_RESCUE_SOURCE_SHA256}" \
    --output "${final_root}/source_fingerprint_audit.json"
fi

if [[ -n "${GEAR_FROZEN_REPLAY_MANIFEST:-}" && -f "${GEAR_FROZEN_REPLAY_MANIFEST}" && -f "${final_root}/source_fingerprint_audit.json" ]]; then
  python3 scripts/promote_graph_action_policy.py \
    --model "${stage_c_root}/policy_graph/graph_action_q_model.json" \
    --replay "${stage_c_root}/policy_graph/graph_action_policy_replay.json" \
    --development-data "${stage_c_root}/outcomes/development_action_log.parquet" \
    --randomized-data "${stage_c_root}/outcomes/randomized_action_log.parquet" \
    --graph-policy "${stage_c_root}/policy_graph/policy_holdout_scored.parquet" \
    --no-graph-policy "${stage_c_root}/policy_no_graph/policy_holdout_scored.parquet" \
    --gate2-report "${final_root}/gate2_report.json" \
    --frozen-replay-manifest "${GEAR_FROZEN_REPLAY_MANIFEST}" \
    --source-fingerprint-audit "${final_root}/source_fingerprint_audit.json" \
    --stage-a-runtime-audit "${stage_a_root}/postprocess_audits/runtime_cohort.json" \
    --stage-b-runtime-audit "${stage_b_root}/postprocess_audits/runtime_cohort.json" \
    --stage-c-runtime-audit "${stage_c_root}/postprocess_audits/runtime_cohort.json" \
    --output "${action_policy_release_manifest}" \
    --release-id "${action_policy_release_id}"
else
  python3 - "${final_root}/action_policy_promotion_blocked.json" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "contract": "gear_graph_action_policy_promotion_status_v1",
            "status": "blocked",
            "reason": "formal_frozen_replay_or_source_audit_unavailable",
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
fi

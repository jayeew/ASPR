#!/usr/bin/env bash
set -euo pipefail

# This command is intentionally separate from the automated experiment runner.
# It accepts completed Claim B/C reviews from independent AI sessions or other
# reviewers; human identity, blinding, calibration, and adjudication are not
# completion requirements.

: "${GEAR_FROZEN_REPLAY_MANIFEST:?set GEAR_FROZEN_REPLAY_MANIFEST to the formal frozen replay manifest}"

stage_a_root="${GEAR_STAGE_A_OUTPUT_ROOT:-outputs/gear/stage_a_real_gear_20260827}"
stage_b_root="${GEAR_STAGE_B_OUTPUT_ROOT:-outputs/gear/stage_b_targeted_expansion_20260828}"
stage_c_root="${GEAR_STAGE_C_OUTPUT_ROOT:-outputs/gear/stage_c_randomized_actions_20260828}"
final_root="${GEAR_FINAL_OUTPUT_ROOT:-outputs/gear/graph_rescue_final_20260828}"
structural_manifest="${GEAR_STRUCTURAL_RELEASE_MANIFEST:-data/calibration/graph_calibration/gear_structural_head_release_v1/manifest.json}"
claim_manifest="${GEAR_CLAIM_ATTRIBUTION_RELEASE_MANIFEST:-${stage_b_root}/claim_attribution_release/release.json}"
action_manifest="${GEAR_ACTION_POLICY_RELEASE_MANIFEST:-data/calibration/graph_calibration/gear_graph_action_policy_release_v1/manifest.json}"
expert_pack_root="${GEAR_EXPERT_PACK_ROOT:-${final_root}/expert_annotation_pack}"
completed_validation="${final_root}/expert_annotation_pack_completed_validation.json"
completion_audit="${final_root}/rescue_completion_audit.json"

python3 -m experiments.gear.evaluation.expert_annotation_pack validate \
  --pack-dir "${expert_pack_root}" \
  --require-completed \
  --output "${completed_validation}"

python3 -m experiments.gear.evaluation.audit_rescue_completion \
  --frozen-replay-manifest "${GEAR_FROZEN_REPLAY_MANIFEST}" \
  --source-fingerprint-audit "${final_root}/source_fingerprint_audit.json" \
  --stage-a-runtime-audit "${stage_a_root}/postprocess_audits/runtime_cohort.json" \
  --stage-b-runtime-audit "${stage_b_root}/postprocess_audits/runtime_cohort.json" \
  --stage-c-runtime-audit "${stage_c_root}/postprocess_audits/runtime_cohort.json" \
  --stage-a-validation "${stage_a_root}/stage_a_validation_120/stage_a_validation.json" \
  --stage-a-gate0 "${stage_a_root}/stage_a_validation_120/gate0_report.json" \
  --stage-a-three-arm "${stage_a_root}/stage_a_validation_120/stage_a_three_arm_scores.csv" \
  --hgb-p-validation "${stage_b_root}/hgb_p_validation_241/claim_a_bounded_validation.json" \
  --stage-b-evidence-audit "${stage_b_root}/postprocess_audits/evidence_coverage.json" \
  --claim-adoption-summary "${stage_b_root}/claim_adoption_labels/claim_adoption_summary.json" \
  --claim-gate-coverage-audit "${stage_b_root}/postprocess_audits/claim_gate_coverage.json" \
  --gate1-temporal "${stage_b_root}/claim_attribution_release/gate1_temporal.json" \
  --gate1-domain "${stage_b_root}/claim_attribution_release/gate1_domain.json" \
  --stage-c-randomized-audit "${stage_c_root}/postprocess_audits/randomized_outcomes.json" \
  --stage-c-outcome-report "${stage_c_root}/outcomes/randomized_action_outcome_report.json" \
  --policy-graph-report "${stage_c_root}/policy_graph/policy_holdout_report.json" \
  --policy-no-graph-report "${stage_c_root}/policy_no_graph/policy_holdout_report.json" \
  --gate2-report "${final_root}/gate2_report.json" \
  --structural-head-manifest "${structural_manifest}" \
  --claim-attribution-manifest "${claim_manifest}" \
  --action-policy-manifest "${action_manifest}" \
  --expert-pack-manifest "${expert_pack_root}/manifest.json" \
  --expert-pack-validation "${completed_validation}" \
  --final-rescue-status "${final_root}/rescue_plan_status.json" \
  --output "${completion_audit}"

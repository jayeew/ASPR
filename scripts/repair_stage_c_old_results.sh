#!/usr/bin/env bash
set -euo pipefail

stage_c_root="${GEAR_STAGE_C_OUTPUT_ROOT:-outputs/gear/stage_c_randomized_actions_20260828}"
manifest="${stage_c_root}/randomized_manifest_150.json"
original_runs="${stage_c_root}/runs"
original_outcomes="${stage_c_root}/outcomes_original_complete"
repair_manifest="${stage_c_root}/repair_manifest_execution_invalid.json"
repair_runs="${stage_c_root}/runs_repaired_terra_medium"
consolidated_runs="${stage_c_root}/runs_consolidated"
final_outcomes="${stage_c_root}/outcomes_consolidated"

if [[ -n "${GEAR_WAIT_FOR_PID:-}" ]]; then
  while kill -0 "${GEAR_WAIT_FOR_PID}" 2>/dev/null; do
    sleep 20
  done
fi

python3 -m experiments.gear.evaluation.collect_randomized_action_outcomes \
  --manifest "${manifest}" \
  --runs-dir "${original_runs}" \
  --output-dir "${original_outcomes}"

python3 -m experiments.gear.evaluation.build_randomized_action_repair_manifest \
  --manifest "${manifest}" \
  --audit "${original_outcomes}/randomized_action_audit.parquet" \
  --output "${repair_manifest}" \
  --include-incomplete

repair_count="$(python3 - "${repair_manifest}" <<'PY'
import json
import sys
from pathlib import Path

print(len(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["cases"]))
PY
)"

if [[ "${repair_count}" -gt 0 ]]; then
  python3 -m gear benchmark \
    --manifest "${repair_manifest}" \
    --output-dir "${repair_runs}" \
    --batch-report-dir "${stage_c_root}/repair_batch_report" \
    --config configs/gear/stage_c_randomized_action.json \
    --workers "${GEAR_REPAIR_WORKERS:-2}" --resume --retry-failed --full-artifacts \
    --case-timeout-seconds 3600
fi

python3 -m experiments.gear.evaluation.build_randomized_action_consolidated_runs \
  --manifest "${manifest}" \
  --repair-manifest "${repair_manifest}" \
  --original-runs-dir "${original_runs}" \
  --repaired-runs-dir "${repair_runs}" \
  --output-dir "${consolidated_runs}"

python3 -m experiments.gear.evaluation.collect_randomized_action_outcomes \
  --manifest "${manifest}" \
  --runs-dir "${consolidated_runs}" \
  --output-dir "${final_outcomes}"

python3 - "${final_outcomes}/randomized_action_outcome_report.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("included_cases") != 150 or not report.get("all_actions_observed"):
    raise SystemExit("Stage C repair remains incomplete; no confirmatory claim allowed")
if report.get("split_counts") != {"confirmatory_holdout": 60, "development": 90}:
    raise SystemExit("Stage C repaired split counts changed")
PY

python3 -m experiments.gear.evaluation.rescue_postprocess_audit randomized \
  --report "${final_outcomes}/randomized_action_outcome_report.json" \
  --action-log "${final_outcomes}/randomized_action_log.parquet" \
  --manifest "${manifest}" \
  --output "${stage_c_root}/postprocess_audits/randomized_outcomes_consolidated.json"

python3 -m experiments.gear.evaluation.run_action_policy_evaluation \
  --development "${final_outcomes}/development_action_log.parquet" \
  --holdout "${final_outcomes}/holdout_action_log.parquet" \
  --feature claim_count --feature mean_claim_centrality \
  --feature publication_year \
  --output-dir "${stage_c_root}/policy_no_graph_consolidated"

python3 -m experiments.gear.evaluation.run_action_policy_evaluation \
  --development "${final_outcomes}/development_action_log.parquet" \
  --holdout "${final_outcomes}/holdout_action_log.parquet" \
  --feature claim_count --feature mean_claim_centrality \
  --feature publication_year --feature graph_shrunk_diffusion \
  --feature graph_reliability --feature graph_structural_share \
  --feature graph_opportunity_share --feature graph_perturbation_potential \
  --feature graph_prediction_uncertainty \
  --output-dir "${stage_c_root}/policy_graph_consolidated"

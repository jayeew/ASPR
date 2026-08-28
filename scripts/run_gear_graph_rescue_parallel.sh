#!/usr/bin/env bash
set -euo pipefail

# Pipeline six independent review sessions while allowing at most five
# model-bearing processes on the 16 GiB CUDA device.  Forward and reverse
# queues overlap API work while per-case locks prevent duplicate reviews.
export GEAR_GPU_MAX_PROCESSES="${GEAR_GPU_MAX_PROCESSES:-5}"
export GEAR_GPU_LEASE_TIMEOUT_SECONDS="${GEAR_GPU_LEASE_TIMEOUT_SECONDS:-3600}"
export GEAR_GPU_LEASE_DIR="${GEAR_GPU_LEASE_DIR:-outputs/gear/.gpu_leases}"

stage_a_root="outputs/gear/stage_a_real_gear_20260827"
stage_b_root="outputs/gear/stage_b_targeted_expansion_20260828"
stage_c_root="outputs/gear/stage_c_randomized_actions_20260828"
runs_dir="${stage_a_root}/batch_runs_bounded_120"
shard_root="${stage_b_root}/parallel_shards_20260828"
stage_b_aux_shard_root="${stage_b_root}/parallel_shards_aux_20260828"
stage_c_aux_shard_root="${stage_c_root}/parallel_shards_aux_20260828"
stage_c_middle_shard_root="${stage_c_root}/parallel_shards_middle_20260828"
report_root="${stage_b_root}/parallel_reports_20260828"

python3 scripts/shard_pending_gear_benchmark.py \
  --manifest "${stage_b_root}/complete_graph_benchmark_manifest.json" \
  --runs-dir "${runs_dir}" \
  --output-dir "${shard_root}" \
  --shards 2 \
  --retry-failed

python3 scripts/shard_pending_gear_benchmark.py \
  --manifest "${stage_b_root}/complete_graph_benchmark_manifest.json" \
  --runs-dir "${runs_dir}" \
  --output-dir "${stage_b_aux_shard_root}" \
  --shards 1 \
  --retry-failed \
  --reverse

python3 scripts/shard_pending_gear_benchmark.py \
  --manifest "${stage_c_root}/randomized_manifest_150.json" \
  --runs-dir "${stage_c_root}/runs" \
  --output-dir "${stage_c_aux_shard_root}" \
  --shards 1 \
  --retry-failed \
  --reverse

python3 scripts/shard_pending_gear_benchmark.py \
  --manifest "${stage_c_root}/randomized_manifest_150.json" \
  --runs-dir "${stage_c_root}/runs" \
  --output-dir "${stage_c_middle_shard_root}" \
  --shards 3 \
  --retry-failed

run_stage_b_shard() {
  local shard_index="$1"
  python3 -m gear benchmark \
    --manifest "${shard_root}/shard_${shard_index}.json" \
    --output-dir "${runs_dir}" \
    --batch-report-dir "${report_root}/shard_${shard_index}" \
    --config configs/gear/stage_a_evidence_only_bounded.json \
    --workers 1 --resume --retry-failed --full-artifacts \
    --case-timeout-seconds 3600
}

run_stage_b_aux() {
  python3 -m gear benchmark \
    --manifest "${stage_b_aux_shard_root}/shard_00.json" \
    --output-dir "${runs_dir}" \
    --batch-report-dir "${report_root}/aux_reverse" \
    --config configs/gear/stage_a_evidence_only_bounded.json \
    --workers 1 --resume --retry-failed --full-artifacts \
    --case-timeout-seconds 3600
}

run_stage_c() {
  python3 -m gear benchmark \
    --manifest "${stage_c_root}/randomized_pilot_manifest_6.json" \
    --output-dir "${stage_c_root}/runs" \
    --batch-report-dir "${stage_c_root}/parallel_reports/pilot" \
    --config configs/gear/stage_c_randomized_action.json \
    --workers 1 --resume --retry-failed --full-artifacts \
    --case-timeout-seconds 3600

  python3 -m experiments.gear.evaluation.collect_randomized_action_outcomes \
    --manifest "${stage_c_root}/randomized_pilot_manifest_6.json" \
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
    --manifest "${stage_c_root}/randomized_manifest_150.json" \
    --output-dir "${stage_c_root}/runs" \
    --batch-report-dir "${stage_c_root}/parallel_reports/full150" \
    --config configs/gear/stage_c_randomized_action.json \
    --workers 1 --resume --retry-failed --full-artifacts \
    --case-timeout-seconds 3600
}

run_stage_c_aux() {
  python3 -m gear benchmark \
    --manifest "${stage_c_aux_shard_root}/shard_00.json" \
    --output-dir "${stage_c_root}/runs" \
    --batch-report-dir "${stage_c_root}/parallel_reports/aux_reverse" \
    --config configs/gear/stage_c_randomized_action.json \
    --workers 1 --resume --retry-failed --full-artifacts \
    --case-timeout-seconds 3600
}

run_stage_c_middle() {
  python3 -m gear benchmark \
    --manifest "${stage_c_middle_shard_root}/shard_01.json" \
    --output-dir "${stage_c_root}/runs" \
    --batch-report-dir "${stage_c_root}/parallel_reports/extra_middle" \
    --config configs/gear/stage_c_randomized_action.json \
    --workers 1 --resume --retry-failed --full-artifacts \
    --case-timeout-seconds 3600
}

run_stage_b_shard 00 &
stage_b_00_pid=$!
run_stage_b_shard 01 &
stage_b_01_pid=$!
run_stage_b_aux &
stage_b_aux_pid=$!
run_stage_c &
stage_c_pid=$!
run_stage_c_aux &
stage_c_aux_pid=$!
run_stage_c_middle &
stage_c_middle_pid=$!

status=0
wait "${stage_b_00_pid}" || status=1
wait "${stage_b_01_pid}" || status=1
wait "${stage_b_aux_pid}" || status=1
wait "${stage_c_pid}" || status=1
wait "${stage_c_aux_pid}" || status=1
wait "${stage_c_middle_pid}" || status=1
if [[ "${status}" -ne 0 ]]; then
  echo "Parallel rescue wave failed closed; inspect per-session reports." >&2
  exit 1
fi

# The canonical runner now resumes all completed cases, rewrites whole-cohort
# summaries, builds labels/gates/policies, and enforces the final fail-closed join.
bash scripts/run_gear_graph_rescue_full.sh

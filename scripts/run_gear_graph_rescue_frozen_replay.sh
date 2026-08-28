#!/usr/bin/env bash
set -euo pipefail

# Load the project dotenv with GEAR's parser rather than `source`: comma-separated
# key lists and quoted values are data, not shell syntax.  Re-exec makes those
# values part of the real process environment so resolved configs are frozen and
# subprocesses inherit the selected model backend without exposing secrets.
if [[ "${GEAR_DOTENV_BOOTSTRAPPED:-0}" != "1" ]]; then
  exec python3 - "$0" "$@" <<'PY'
import os
import sys

from gear.env import load_env

load_env()
os.environ["GEAR_DOTENV_BOOTSTRAPPED"] = "1"
os.execvpe("bash", ["bash", sys.argv[1], *sys.argv[2:]], os.environ)
PY
fi

# A formal replay never reads review artifacts from the development batch.  A
# matching freeze manifest permits interruption-safe resume; any source,
# configuration, cohort, label-context, or graph-target drift fails closed.
replay_id="${GEAR_FROZEN_REPLAY_ID:-frozen_v1_20260828}"
replay_root="${GEAR_FROZEN_REPLAY_ROOT:-outputs/gear/frozen_replays/${replay_id}}"
stage_a_input_root="${GEAR_STAGE_A_INPUT_ROOT:-outputs/gear/stage_a_real_gear_20260827}"
stage_b_input_root="${GEAR_STAGE_B_INPUT_ROOT:-outputs/gear/stage_b_targeted_expansion_20260828}"
stage_c_input_root="${GEAR_STAGE_C_INPUT_ROOT:-outputs/gear/stage_c_randomized_actions_20260828}"

read -r runtime_code_sha rescue_source_sha rescue_source_count stage_ab_config_sha stage_c_config_sha < <(
  python3 - <<'PY'
from pathlib import Path

from experiments.gear.evaluation.source_fingerprint import rescue_source_fingerprint
from gear.config import load_config
from gear.review_pipeline import runtime_code_fingerprint
from gear.trace import sha256_value

code_sha, _ = runtime_code_fingerprint()
source_sha, source_count = rescue_source_fingerprint()
stage_ab = sha256_value(load_config(Path("configs/gear/stage_a_evidence_only_bounded.json")))
stage_c = sha256_value(load_config(Path("configs/gear/stage_c_randomized_action.json")))
print(code_sha, source_sha, source_count, stage_ab, stage_c)
PY
)

export GEAR_EXPECTED_RUNTIME_CODE_SHA256="${runtime_code_sha}"
export GEAR_EXPECTED_RESCUE_SOURCE_SHA256="${rescue_source_sha}"
export GEAR_EXPECTED_STAGE_AB_CONFIG_SHA256="${stage_ab_config_sha}"
export GEAR_EXPECTED_STAGE_C_CONFIG_SHA256="${stage_c_config_sha}"
export GEAR_FROZEN_REPLAY_MANIFEST="${replay_root}/freeze_manifest.json"
export GEAR_STAGE_A_INPUT_ROOT="${stage_a_input_root}"
export GEAR_STAGE_B_INPUT_ROOT="${stage_b_input_root}"
export GEAR_STAGE_C_INPUT_ROOT="${stage_c_input_root}"
export GEAR_STAGE_A_OUTPUT_ROOT="${replay_root}/stage_a"
export GEAR_STAGE_B_OUTPUT_ROOT="${replay_root}/stage_b"
export GEAR_STAGE_C_OUTPUT_ROOT="${replay_root}/stage_c"
export GEAR_FINAL_OUTPUT_ROOT="${replay_root}/final"
export GEAR_BENCHMARK_WORKERS="${GEAR_BENCHMARK_WORKERS:-5}"
export GEAR_GPU_MAX_PROCESSES="${GEAR_GPU_MAX_PROCESSES:-5}"
export GEAR_GPU_LEASE_TIMEOUT_SECONDS="${GEAR_GPU_LEASE_TIMEOUT_SECONDS:-3600}"
# The lease namespace is device-global, not replay-local.  A replay-local
# directory would allow a development batch and a frozen replay to each admit
# five resident models onto the same 16 GiB device and can therefore OOM.
export GEAR_GPU_LEASE_DIR="${GEAR_GPU_LEASE_DIR:-outputs/gear/.gpu_leases}"

python3 - \
  "${replay_root}/freeze_manifest.json" \
  "${replay_id}" \
  "${runtime_code_sha}" \
  "${rescue_source_sha}" \
  "${rescue_source_count}" \
  "${stage_ab_config_sha}" \
  "${stage_c_config_sha}" \
  "${stage_a_input_root}" \
  "${stage_b_input_root}" \
  "${stage_c_input_root}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

from gear.config import load_config
from gear.trace import sha256_file

(
    output_raw,
    replay_id,
    code_sha,
    source_sha,
    source_count,
    stage_ab_config_sha,
    stage_c_config_sha,
    stage_a_raw,
    stage_b_raw,
    stage_c_raw,
) = sys.argv[1:]
output = Path(output_raw)
stage_a = Path(stage_a_raw)
stage_b = Path(stage_b_raw)
stage_c = Path(stage_c_raw)
inputs = {
    "stage_a_manifest": stage_a / "benchmark_manifest.json",
    "stage_b_manifest": stage_b / "complete_graph_benchmark_manifest.json",
    "stage_b_contexts": stage_b / "claim_contexts_241/citation_contexts.jsonl",
    "stage_b_context_papers": (
        stage_b / "claim_contexts_241/citation_context_papers.jsonl"
    ),
    "stage_b_split": stage_b / "integration_holdout_manifest.json",
    "hgb_p_temporal": (
        stage_b
        / "hgb_p_validation_241/hgb_p_forward_temporal_hybrid_predictions.parquet"
    ),
    "hgb_p_domain": (
        stage_b / "hgb_p_validation_241/hgb_p_domain_oof_predictions.parquet"
    ),
    "stage_c_pilot_manifest": stage_c / "randomized_pilot_manifest_6.json",
    "stage_c_manifest": stage_c / "randomized_manifest_150.json",
    "stage_ab_config": Path("configs/gear/stage_a_evidence_only_bounded.json"),
    "stage_c_config": Path("configs/gear/stage_c_randomized_action.json"),
}
for prefix, config_path in (
    ("stage_ab", inputs["stage_ab_config"]),
    ("stage_c", inputs["stage_c_config"]),
):
    config = load_config(config_path)
    assets = {
        "forecast_manifest": config.resolved_forecast_release_manifest(),
        "runtime_manifest": config.resolved_forecast_runtime_manifest(),
        "anatomy_manifest": config.resolved_forecast_anatomy_manifest(),
        "structural_manifest": config.resolved_structural_head_manifest(),
        "claim_attribution_manifest": config.resolved_claim_attribution_manifest(),
        "action_policy_manifest": config.resolved_graph_action_policy_manifest(),
    }
    for name, path in assets.items():
        if path is not None:
            inputs[f"{prefix}_{name}"] = path
missing = [str(path) for path in inputs.values() if not path.is_file()]
if missing:
    raise SystemExit(f"frozen replay inputs missing: {missing}")
payload = {
    "contract": "gear_graph_rescue_frozen_replay_v1",
    "replay_id": replay_id,
    "runtime_code_sha256": code_sha,
    "rescue_source_sha256": source_sha,
    "rescue_source_file_count": int(source_count),
    "stage_ab_runtime_config_sha256": stage_ab_config_sha,
    "stage_c_runtime_config_sha256": stage_c_config_sha,
    "inputs": {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in sorted(inputs.items())
    },
}
if output.is_file():
    existing = json.loads(output.read_text(encoding="utf-8"))
    if existing != payload:
        raise SystemExit("frozen replay manifest drifted; choose a new replay id")
else:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
PY

if [[ "${GEAR_FREEZE_ONLY:-0}" == "1" ]]; then
  exit 0
fi

bash scripts/run_gear_graph_rescue_full.sh

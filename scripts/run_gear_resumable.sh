#!/usr/bin/env bash
# Resume existing GEAR artifacts. Cleanup is an explicit, separate invocation.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
if [[ -z "${CODEX_HOME:-}" && -d /mnt/c/Users/jayee/.codex ]]; then
  export CODEX_HOME=/mnt/c/Users/jayee/.codex
fi
export GEAR_EXTERNAL_FULLTEXT_ENABLED="${GEAR_EXTERNAL_FULLTEXT_ENABLED:-true}"
export GEAR_EXTERNAL_FULLTEXT_MAX_WORKS="${GEAR_EXTERNAL_FULLTEXT_MAX_WORKS:-2}"
export GEAR_HISTORICAL_PDF_ENABLED="${GEAR_HISTORICAL_PDF_ENABLED:-false}"
export GEAR_CODEX_SERVICE_TIER="${GEAR_CODEX_SERVICE_TIER:-fast}"
export GEAR_NETWORK_MAX_PROCESSES="${GEAR_NETWORK_MAX_PROCESSES:-2}"
export GEAR_NETWORK_RETRIES="${GEAR_NETWORK_RETRIES:-3}"
export GEAR_CLAIM_WORKERS="${GEAR_CLAIM_WORKERS:-2}"
export GEAR_POSTPROCESS_MIN_AVAILABLE_GIB="${GEAR_POSTPROCESS_MIN_AVAILABLE_GIB:-8}"
export GEAR_RELATION_WORKERS="${GEAR_RELATION_WORKERS:-2}"
exec "${GEAR_PYTHON:-python3}" -u experiments/innovation_200/run_gear.py \
  --study "${GEAR_STUDY:-outputs/innovation_200_20260907}" \
  --workers "${GEAR_PAPER_WORKERS:-4}" --cli-limit "${GEAR_CLI_LIMIT:-12}" \
  --verbose "$@"

#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUT_DIR:-outputs/nature_portfolio_v5}"
OPENALEX_WORKERS="${OPENALEX_WORKERS:-2}"
SLEEP_SECONDS="${SLEEP_SECONDS:-0.1}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-60}"
MAX_RETRIES="${MAX_RETRIES:-5}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100}"

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/checkpoints"
LOG_PATH="$OUT_DIR/logs/03b_reference_closure_online_missing.log"

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

echo "[$(timestamp)] 开始在线补抓 snapshot 缺失参考文献" | tee -a "$LOG_PATH"
echo "[$(timestamp)] 输出目录：$OUT_DIR；workers=$OPENALEX_WORKERS" | tee -a "$LOG_PATH"

cmd=(
    "$PYTHON_BIN" -u scripts/fetch_openalex_reference_missing_v5.py
    --retry-queue "$OUT_DIR/nature_reference_closure_api_retry_queue.csv"
    --reference-works "$OUT_DIR/nature_reference_works.csv"
    --out-dir "$OUT_DIR"
    --checkpoint-csv "$OUT_DIR/checkpoints/reference_missing_online_success.csv"
    --failure-log "$OUT_DIR/checkpoints/reference_missing_online_failures.csv"
    --final-missing "$OUT_DIR/nature_reference_closure_final_missing_ids.csv"
    --manifest "$OUT_DIR/reference_missing_online_manifest.json"
    --workers "$OPENALEX_WORKERS"
    --progress-every "$PROGRESS_EVERY"
    --sleep-seconds "$SLEEP_SECONDS"
    --timeout-seconds "$TIMEOUT_SECONDS"
    --max-retries "$MAX_RETRIES"
)

if [[ -n "${MAX_REFS:-}" ]]; then
    cmd+=(--max-refs "$MAX_REFS")
fi

set +e
"${cmd[@]}" 2>&1 | tee -a "$LOG_PATH"
exit_code=${PIPESTATUS[0]}
set -e

if [[ "$exit_code" -ne 0 ]]; then
    echo "[$(timestamp)] 缺失参考文献补抓失败，退出码=$exit_code；详见 $LOG_PATH" | tee -a "$LOG_PATH"
    exit "$exit_code"
fi

echo "[$(timestamp)] 缺失参考文献补抓完成；日志=$LOG_PATH" | tee -a "$LOG_PATH"

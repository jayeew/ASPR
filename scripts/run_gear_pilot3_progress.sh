#!/usr/bin/env bash
# Run the fixed three-paper Evidence-State Delta pilot with observable progress.
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_ID="${GEAR_PILOT_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${GEAR_PILOT_OUTPUT_DIR:-$PROJECT_ROOT/outputs/gear/revision_audit/pilot3_$RUN_ID}"
MANIFEST="${GEAR_PILOT_MANIFEST:-$PROJECT_ROOT/configs/gear/nature_revision_audit_pilot3.json}"
STEP_TIMEOUT_SECONDS="${GEAR_PILOT_STEP_TIMEOUT_SECONDS:-1800}"
HEARTBEAT_SECONDS="${GEAR_PILOT_HEARTBEAT_SECONDS:-30}"

mkdir -p "$RUN_DIR/logs"
printf '运行编号=%s\n输出目录=%s\n清单文件=%s\n单步骤超时秒数=%s\n' \
  "$RUN_ID" "$RUN_DIR" "$MANIFEST" "$STEP_TIMEOUT_SECONDS" | tee "$RUN_DIR/RUN_INFO.txt"

run_step() {
  local step="$1"
  local log_file="$RUN_DIR/logs/${step}.log"
  local started elapsed pid status
  started="$(date +%s)"
  echo "[$(date -Is)] 开始执行：步骤=$step；日志=$log_file"
  timeout --foreground "$STEP_TIMEOUT_SECONDS" \
    python3 -u scripts/run_gear_revision_audit.py "$step" \
      --manifest "$MANIFEST" \
      --output-dir "$RUN_DIR" \
      --agent-output-root "$RUN_DIR" \
      > >(tee "$log_file") 2>&1 &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$HEARTBEAT_SECONDS"
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    elapsed=$(( $(date +%s) - started ))
    echo "[$(date -Is)] 正在执行：步骤=$step；已用时=${elapsed}秒；已生成产物=$(find "$RUN_DIR" -type f | wc -l)"
  done
  if wait "$pid"; then
    echo "[$(date -Is)] 执行完成：步骤=$step；总用时=$(( $(date +%s) - started ))秒"
    return 0
  fi
  status=$?
  if [[ "$status" -eq 124 ]]; then
    echo "[$(date -Is)] 执行超时：步骤=$step；超过=${STEP_TIMEOUT_SECONDS}秒；请检查 $log_file" >&2
  else
    echo "[$(date -Is)] 执行失败：步骤=$step；退出码=$status；请检查 $log_file" >&2
  fi
  return "$status"
}

for step in preflight run-agent prepare-judges run-judges build-rubrics score-rubrics evaluate report; do
  run_step "$step"
done

echo "[$(date -Is)] 全流程完成：结果报告=$RUN_DIR/RESULTS.md"

#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SOURCE_DIR="${SOURCE_DIR:-outputs/common/new/data/nature_portfolio_v5}"
TARGET_WORKS="${TARGET_WORKS:-$SOURCE_DIR/nature_target_works.csv}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$SOURCE_DIR/checkpoints/future_citers_tau8}"
OUTPUT_DIR="${OUTPUT_DIR:-$SOURCE_DIR/future_multihorizon}"
COMPLETE_END_YEAR="${COMPLETE_END_YEAR:-2025}"
REQUESTED_HORIZON="${REQUESTED_HORIZON:-8}"
MAX_CITERS_PER_WORK="${MAX_CITERS_PER_WORK:-1000}"
BATCH_SIZE="${BATCH_SIZE:-50000}"
PROGRESS_EVERY="${PROGRESS_EVERY:-1000}"
OVERWRITE="${OVERWRITE:-0}"
AUDIT_ONLY="${AUDIT_ONLY:-0}"
ALLOW_MISSING="${ALLOW_MISSING:-0}"
ALLOW_QUALITY_FAILURES="${ALLOW_QUALITY_FAILURES:-0}"

if [[ ! -s "$TARGET_WORKS" ]]; then
    echo "错误：目标论文表不存在或为空：$TARGET_WORKS" >&2
    exit 2
fi
if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "错误：tau8 checkpoint 目录不存在：$CHECKPOINT_DIR" >&2
    exit 2
fi

cmd=(
    "$PYTHON_BIN" -u scripts/materialize_nature_future_multihorizon_v5.py
    --target-works "$TARGET_WORKS"
    --checkpoint-dir "$CHECKPOINT_DIR"
    --output-dir "$OUTPUT_DIR"
    --horizons 3 5 8
    --requested-horizon "$REQUESTED_HORIZON"
    --complete-end-year "$COMPLETE_END_YEAR"
    --max-citers-per-work "$MAX_CITERS_PER_WORK"
    --batch-size "$BATCH_SIZE"
    --progress-every "$PROGRESS_EVERY"
)
if [[ "$OVERWRITE" == "1" ]]; then
    cmd+=(--overwrite)
fi
if [[ "$AUDIT_ONLY" == "1" ]]; then
    cmd+=(--audit-only)
fi
if [[ "$ALLOW_MISSING" == "1" ]]; then
    cmd+=(--allow-missing)
fi
if [[ "$ALLOW_QUALITY_FAILURES" == "1" ]]; then
    cmd+=(--allow-quality-failures)
fi

echo "[$(date +'%Y-%m-%d %H:%M:%S')] 开始从 tau8 checkpoint 离线派生 tau3/tau5/tau8"
echo "目标论文：$TARGET_WORKS"
echo "checkpoint：$CHECKPOINT_DIR"
echo "输出目录：$OUTPUT_DIR"
"${cmd[@]}"

if [[ "$AUDIT_ONLY" == "1" ]]; then
    exit 0
fi

echo "[$(date +'%Y-%m-%d %H:%M:%S')] 多窗口数据构建完成"
echo "质量报告：$OUTPUT_DIR/data_quality_report.json"
echo "下游路径：$OUTPUT_DIR/downstream_paths.env"
echo "使用示例：$OUTPUT_DIR/USAGE.md"

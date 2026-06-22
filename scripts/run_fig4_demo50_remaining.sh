#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${FIG4_OUTPUT_DIR:-outputs/kg_perturbation_fig4_demo50}"
if [[ $# -gt 0 && "$1" != --* ]]; then
  OUTPUT_DIR="$1"
  shift
fi

export FIG4_AGENT_MAX_ITERATIONS="${FIG4_AGENT_MAX_ITERATIONS:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export ASPR_RECALL_MODEL_PATH="${ASPR_RECALL_MODEL_PATH:-/home/jayee/models/bge-m3}"
export ASPR_RERANKER_MODEL_PATH="${ASPR_RERANKER_MODEL_PATH:-/home/jayee/models/OpenScholar_Reranker}"
export ASPR_RECALL_BATCH_SIZE="${ASPR_RECALL_BATCH_SIZE:-8}"
export ASPR_RECALL_RETRY_BATCHES="${ASPR_RECALL_RETRY_BATCHES:-8,4,2,1}"
export ASPR_RERANK_BATCH_SIZE="${ASPR_RERANK_BATCH_SIZE:-16}"
export ASPR_RERANK_RETRY_BATCHES="${ASPR_RERANK_RETRY_BATCHES:-16,8,4,2,1}"

if [[ ! -f "$OUTPUT_DIR/fig4_manifest.csv" ]]; then
  echo "Missing $OUTPUT_DIR/fig4_manifest.csv" >&2
  echo "This script only runs the post-sample Fig.4 stages. Run screen/sample first." >&2
  exit 1
fi

python -m experiments.kg_perturbation_fig4.main_fig4 \
  --stage post-sample \
  --output-dir "$OUTPUT_DIR" \
  --agent-context-mode dossier \
  "$@"

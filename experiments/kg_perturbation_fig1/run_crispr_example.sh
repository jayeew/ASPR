#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

python "$SCRIPT_DIR/fig1_knowledge_perturbation.py" \
  --config "$SCRIPT_DIR/configs/crispr.yaml" \
  --out-dir "$PROJECT_ROOT/outputs/kg_perturbation_fig1"

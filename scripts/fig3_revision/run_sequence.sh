#!/usr/bin/env bash
# Compatibility entry: the consolidated script owns the explicit sequence.
set -euo pipefail
FIG3_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export FIG3_WORKERS="${1:-64}"
if (( $# )); then shift; fi
exec bash "$FIG3_SCRIPT_DIR/fig3.sh" all "$@"

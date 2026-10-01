#!/usr/bin/env bash
# The only recommended task entry point. Does nothing until explicitly invoked.
set -euo pipefail
FIG3_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$FIG3_ROOT"
if [[ -z "${FIG3_PYTHON:-}" ]]; then
  FIG3_PYTHON="python3"
  if [[ -x "$HOME/miniconda3/envs/openrlhf/bin/python" ]]; then
    FIG3_PYTHON="$HOME/miniconda3/envs/openrlhf/bin/python"
  fi
fi
FIG3_WORKERS="${FIG3_WORKERS:-64}"
FIG3_CONFIG="${FIG3_CONFIG:-$FIG3_ROOT/figure_pipeline/fig3_revision/config.json}"
FIG3_COMMAND="${1:-status}"
if (( $# )); then shift; fi
case "$FIG3_COMMAND" in
  all|generate|evaluate|recheck|export)
    exec "$FIG3_PYTHON" -m figure_pipeline.fig3_revision --config "$FIG3_CONFIG" workflow \
      --group "$FIG3_COMMAND" --workers "$FIG3_WORKERS" "$@" ;;
  prepare|status|run|aggregate|render|assemble|workflow)
    exec "$FIG3_PYTHON" -m figure_pipeline.fig3_revision --config "$FIG3_CONFIG" "$FIG3_COMMAND" "$@" ;;
  *) printf 'Unknown Fig.3 command: %s\n' "$FIG3_COMMAND" >&2; exit 2 ;;
esac

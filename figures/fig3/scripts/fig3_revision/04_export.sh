#!/usr/bin/env bash
set -euo pipefail
# Export uses local statistics and rendering only. Honor an explicitly selected Python.
if [[ -z "${FIG3_PYTHON:-}" ]] && ! python3 -c 'import cairosvg, matplotlib, numpy, pydantic' >/dev/null 2>&1; then
  FIG3_EXPORT_PYTHON="$HOME/miniconda3/envs/openrlhf/bin/python"
  if [[ -x "$FIG3_EXPORT_PYTHON" ]] && "$FIG3_EXPORT_PYTHON" -c 'import cairosvg, matplotlib, numpy, pydantic' >/dev/null 2>&1; then
    export FIG3_PYTHON="$FIG3_EXPORT_PYTHON"
    printf '[导出环境] 使用已有绘图依赖环境：%s\n' "$FIG3_PYTHON"
  fi
fi
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/fig3.sh" export "$@"

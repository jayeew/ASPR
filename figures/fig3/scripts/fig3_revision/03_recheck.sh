#!/usr/bin/env bash
# 固定20篇盲法复核：预算内合批、对象级断点、少量原句控制；不重做01/02。
set -euo pipefail
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/fig3.sh" recheck "$@"

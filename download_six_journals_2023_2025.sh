#!/usr/bin/env bash
# 批量下载 6 本 Nature 杂志 2023、2024、2025 年文章
# 使用方式: ./download_six_journals_2023_2025.sh  或  bash download_six_journals_2023_2025.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_SCRIPT="download_nature.py"
YEARS=(2023 2024 2025)
FAILED=()

# 杂志名称与编号（名称|编号）
JOURNALS=(
  "Nature Immunology|41590"
  "Nature Microbiology|41564"
  "Nature Structural & Molecular Biology|41594"
  "Nature Ecology & Evolution|41559"
  "Nature Human Behaviour|41562"
  "Nature Cell Biology|41556"
)

echo "=============================================="
echo "批量下载: 6 本杂志 × 3 年 = 18 个任务"
echo "=============================================="

for entry in "${JOURNALS[@]}"; do
  name="${entry%%|*}"
  id="${entry##*|}"
  for year in "${YEARS[@]}"; do
    echo ""
    echo ">>> 正在处理: $name ($id) - $year 年"
    if python3 "$PYTHON_SCRIPT" --journal-name "$name" --journal-id "$id" --year "$year"; then
      echo ">>> 完成: $name $year"
    else
      echo ">>> 失败: $name $year"
      FAILED+=("$name $year")
    fi
  done
done

echo ""
echo "=============================================="
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "全部 18 个任务已完成"
else
  echo "已完成 $((18 - ${#FAILED[@]})) 个任务，失败 ${#FAILED[@]} 个:"
  printf '  - %s\n' "${FAILED[@]}"
fi
echo "=============================================="

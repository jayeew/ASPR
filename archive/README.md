# ASPR-GEAR archive

This directory contains data excluded from the current runtime and final
evaluation lineage. Nothing under `archive/` may be scanned as a current GEAR
candidate or used as current Fig. 1–Fig. 3 evidence.

Canonical flow names retained by the active system:

1. `GEAR+Graph` — the sole formal topology/HGB integration flow.
2. `GEAR-only` — the evidence-only baseline.
3. `GEAR+Graph-full-calibrated` — the calibration experiment control.
4. `Randomized/Placebo Graph` — the randomized matched-budget control.

Archived groups:

- `fig4_top3_unchanged/`: 11 completed incremental runs excluded because the
  GEAR-only and GEAR+Graph top-3 claim sets had identical content. They are
  retained only as selection/stopping-rule evidence.
- `legacy_figures/`: old Fig. 4–Fig. 10 outputs and the silver review-opinion
  data derived from old Fig. 4/Fig. 10. These are deprecated evidence and must
  not be used for training, algorithm selection, evaluation, or paper claims.

## 历史指标研究

| 工作目录 | 内容 |
| --- | --- |
| [`evidence_derived/`](evidence_derived/) | 历史指标流程、冻结版本和生产发布。 |
| [`evidence_derived_v3/`](evidence_derived_v3/) | v3 指标、恢复记录、训练矩阵与比较实验。 |
| [`evidence_derived_v4_rebuild/`](evidence_derived_v4_rebuild/) | v4 重建、分批处理和比较实验。 |

根目录 `innovation_impact_feature_selection/` 下的三个同名路径链接到这里。历史命令、Python 导入和已保存的来源路径继续使用这些入口。各版本内部保持原有目录深度和结构。大宗运行产物和冻结矩阵由本地工作区提供。

v3 训练矩阵入口及对应测试会读取本地数据集契约和质量报告，具体路径由 `configs/nature_multihorizon/active_dataset.json` 指定。

早期根目录运行日志保存在本地 `archive/logs/`。查看一次运行的状态、日志和证据时，从对应的运行目录进入。 [数据目录指南](../data/README.md) · [迁移清单](../docs/project_structure_manifest.json)

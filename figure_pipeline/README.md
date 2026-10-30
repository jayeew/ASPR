# Current Fig1–8 figure entry points

最终输出口径为 `outputs/fig1-8_reference_final_20261004/`，以其中
`文件清单.csv` 和图片讲解为准。2026-10-08 清理检查核对了清单中的 156 个文件，
哈希全部一致；未重跑模型、实验或绘图。

| 图 | 正式实现 | 必须保留的工作目录／材料 |
| --- | --- | --- |
| Fig1 | `fig1_reference` | `outputs/fig1_reference/`、历史图、旧 study 中的案例 |
| Fig2 | `fig2_reference` | `outputs/fig2_reference/`、`outputs/innovation_200_20260907/` 案例 |
| Fig3 | `fig3_reference` 的 render | `outputs/fig3_reference/` 全目录，含 study、dataset、extensions、assets、layouts |
| Fig4 | `fig4_reference` 转发至 `fig4_explanation_reference` | `outputs/fig4_reference/`，尤其 experiment |
| Fig5 | `fig5_mechanism_study/render_reference.py` | `fig_reference/`、`outputs/fig5_mechanism_study/` 及其复用实验 |
| Fig6 | `fig6_reference` | `outputs/fig6_reference/`、`outputs/fig6_damage_study/`、旧 study 的原报告 |
| Fig7 | `fig7_reference.two_panel` | `outputs/fig7_reference/two_panel/`、frontier_2026 的材料、预测、统计与调用账本 |
| Fig8 | `fig8_reference`，重绘使用 `--render-only` | `outputs/fig8_data/analysis/tables/` 当前统计输入；绘图覆盖 `outputs/fig8_reference/` |

上述目录是导航，不是完整依赖闭包。正式 Fig5 是四 panel 版；
`fig5_reference` 是旧六 panel 入口，不代表当前交付。正式 Fig7 是双 panel 版，
默认 forecast 入口不代表当前交付。不要为整理目录调用 `all`、`prepare` 或预测阶段。

`fig3_revision`、`fig4_explanation_100`、`fig4_explanation_study`、`fig4_rerun`、
`fig4_mechanisms`、`fig5_revision`、`fig5_robustness`、`fig5_four_panel` 与
`fig6_computer` 中仍有被当前流程导入的代码，不能按“旧版本”名称整目录删除。
旧 study 和候选集还是 Fig7 的共享 claims／Graph 来源。

已知重建限制：Fig1 的 prepare 引用 `outputs/GRAPH_STATISTICS.zip`，本次检查时
该文件已缺失；现有统计快照必须保留。完整保护清单、候选归档项和本轮缓存删除记录见
[`docs/fig1_8_cleanup_audit_20261008.json`](../docs/fig1_8_cleanup_audit_20261008.json)。

## Historical Figure Pipeline v2

以下为历史 FROM_WEB_v2 工作流，不是当前 Fig1–8 正式输出入口。

本次修订保留 `outputs/FROM_WEB` 和旧 ZIP，使用冻结 200 篇快照；输出为 `outputs/FROM_WEB_v2`。这是可审计重绘与方法修复，不是新增完整 GEAR/Full、人工标注或前沿预测实验。

运行：

```bash
MPLCONFIGDIR=/tmp/aspr-mpl python3 -m figure_pipeline.build --mode draft
python3 -m figure_pipeline.build --mode publication
python3 -m pytest -q tests/figure_pipeline
```

第二条命令在缺核心证据时应失败，不能把草稿当成已完成的论文主结果。Fig5/6/7 的效果图不由准备度面板替代；运行覆盖集中在 Supplement S1。

模块：`benchmark_revision.py` 负责配对统计及状态分离，`graph_revision.py` 负责图结构清理与诊断，`case_revision.py` 负责机制、基线、期刊与真实案例；`specs/contract.py` 定义证据门槛，`qa/checks.py` 分开检查格式、物理字体和研究证据状态。输入冻结文件通过 SHA-256 检查未被修改。新独立核验/受控实验协议见 `specs/research_protocol.md`。

科学边界：Graph + joint 不等于 Full fusion；已保存 AI 审稿匹配不等于独立人工正确性；n,m 零模型不是创新真值；标题短语热度不等于知识前沿。

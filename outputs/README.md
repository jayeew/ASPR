# 当前入口

图件工作目录按 Fig1–Fig11 组织。这里的旧图路径通过符号链接访问迁移后的目录。Git 提供目录和入口，正式图件、预览、文件清单和实验数据由本地工作区提供。 [逐图入口](../figures/README.md)

# 输出与重绘指南

本地交付从 `fig1-11_final/` 进入。每个 `FigN/` 指向 `../figures/figN/final/` 中的整图和已有 panel。预览在 `fig1-11_final/previews/`，文件和来源见 `fig1-11_final/文件清单.csv`。

研究问题、主要结果及分母见项目简介。各面板的读法见本地 `fig1-11_final/图片讲解.md`。Fig10 的 MOF 事实卡位于 `classic_claims_position_2025_2026/`，同一工作区的 AlphaFold 3 报告属于独立历史案例。 [项目简介](../项目简介.md)

## 直接导出当前正式图

每张图的可编辑母版是 `figures/figN/final/FigN.svg`。最终颜色、标题、线宽和位置以这份母版为准。将本地母版放在该路径，并在当前环境安装 CairoSVG。在仓库根目录运行以下命令可查看目录或重新导出，按需要替换图号。

```bash
python3 -m figures fig7 paths
python3 -m figures fig7 export
```

导出的 PDF 和 PNG 写入母版所在目录。更改数据后重新绘制时，使用下表中的实现与工作区。部分最终配色和标题位置是在导出阶段调整的，原始脚本重跑后需要对照正式 SVG。

| 图 | 实现（figure_pipeline 下） | 保留的工作区和重要输入 |
|---|---|---|
| 1 | fig1_reference | fig1_reference/ 的模板、统计快照和字体配置；历史图资产和 study 案例。原始 prepare 曾引用缺失的 GRAPH_STATISTICS.zip |
| 2 | fig2_reference | fig2_reference/ 和 innovation_200_20260907/ 案例 |
| 3 | fig3_reference | fig3_reference/ 下 study、dataset、extensions、assets、layouts |
| 4 | fig4_reference / fig4_explanation_reference | fig4_reference/data、experiment 与布局 |
| 5 | fig5_mechanism_study/render_reference.py | 根目录 fig_reference/；fig5_mechanism_study/Fig5_formal_statistics.zip 及既有实验 |
| 6 | fig6_reference | fig6_reference/、fig6_damage_study/、历史案例报告和共享绘图模块 |
| 7 | fig7_reference.country_extension render | fig7_reference/country_extension/，以及 two_panel/final/Fig7.svg、frontier_2026 和历史主题输入。两面板 SVG 是四面板上半部分的输入 |
| 8 | fig8_reference --render-only | fig8_data/analysis/tables/，含引用与机构主题指标；fig8_reference/ 存储绘图输出 |
| 9 | fig9_reference | fig9_reference/ 的当前布局与实现 |
| 10 | fig10_reference.render | fig10_reference/、实现目录的 icons/、classic_claims_position_2025_2026/graph_position_v1/ 事实卡 |
| 11 | fig11_reference | fig11_reference/、提示词来源、Fig1 的字体与 SVG 导出工具 |

## 其他输出如何处理

- **研究记录**：innovation_200_20260907、innovation_candidates_20260910、innovation_v2_*、gear、Qwen-ASPR 和专项 study 按原路径保留。它们包含案例、响应、证据或独立实验。
- **历史实现的输入**：fig5_initial、fig5_revision、fig5_robustness、fig5_four_panel、fig5_six_panel、fig6_computer 等常含统计表、评估和来源。旧目录名不足以支持整体删除。
- **备份与交接**：archive/、packages/、根目录代码 ZIP 保留原位置。历史代码包属于备份，其内容与当前工作树的可替代性尚未确认。
- **批量采集与缓存**：fig8_data 中的 snapshot 回执、身份与年份证据、任务账本、机构映射和新贡献结果用于复算与追溯。仅按“画 PNG 用不到”判断，会误删研究依据。

本次删除的旧导出、预览派生表和开发缓存逐项列于 [清理记录](../docs/figure_cleanup_20261106.json)。以前的保护目录与唯一历史材料见 [历史依赖审计](../docs/fig1_8_cleanup_audit_20261008.json)。

## 按十一张正式图复核后的清理（2026-11-06）

移除已停用的 Fig4 交付导出器、Fig7 手选词云及词表、Fig8 八刊旧图的说明生成器。Fig4 历史 rerun 仍导入的绘图模块以及 Fig7 四面板所需的双面板 SVG 继续作为依赖保存。删除的文档包括八图简版图解、Fig5 六面板草案及早期图号待办；保留的十月八图详解已标明历史日期。

旧 dev10 打包脚本、临时 PDF 抽取文本和环境链接已清除；19 份来源 PDF 仍在 docs/ASPR_2026_Agent_Papers/papers。Fig1 设计压缩包的五个成员已逐字节比对，保留解压目录。早期扩散预测简介与论文草稿转入 archive/。正式整图、绘图输入、评估明细、证据记录和独立案例各按其原有研究范围保存。

# Fig.8 定稿

当前定稿：2023–2025 年、8 种期刊、24,919 篇论文；左上 a 为贡献类型，左下 c 为期刊内分布，右侧 b 为年度知识结构散点。画布 18 × 7 英寸。

- `final/`：主图 PNG、600 dpi PNG、PDF、可编辑 SVG、轮廓化 SVG。
- `panels/`：a/b/c 独立 PNG、PDF、SVG。
- `data/plot_data.json.gz`：全部论文指标、汇总和缺失口径，重绘的唯一数据输入。
- `data/journal_year_summary.csv`：可直接查阅的年度汇总。
- `sources/`、`qa/`：原始来源哈希、数据检查及定稿文件哈希。
- `caption_en.txt`、`PANEL_GUIDE_zh.txt`：图注与指标解释。

绘图源码位于仓库 `figure_pipeline/fig8_reference/`；无需复制源码或保留重复压缩包。

仅重绘：

```bash
/home/jayee/miniconda3/envs/openrlhf/bin/python -m figure_pipeline.fig8_reference --render-only
```

从仓库原始数据重建指标并绘图：去掉 `--render-only`。使用 `data/claim_graph/` 的现有资产，不调用语言模型、嵌入模型或 GEAR。

本目录替代旧 200 篇版本及 `fig8_history`。已删除案例网络、候选案例、补充探索图、逐 claim 明细、派生边表及重复中间文件。

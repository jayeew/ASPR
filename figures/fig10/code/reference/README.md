# Fig10｜四条 MOF 主张的局部图位置

`render.py` 读取 `outputs/classic_claims_position_2025_2026/graph_position_v1/` 中 G02 与 C03–C06 的已存图事实，并使用本目录 `icons/` 的示意素材。b 展示 C04 的十个邻居；d 展示四条主张的共同邻域。联合插入涉及 36 个历史邻居及 19 个原有连通分量。

从项目根目录运行 `python -m figure_pipeline.fig10_reference.render`。工作结果位于 `outputs/fig10_reference/`，含图注和 `figure_source_data.json`。排版与发布使用[正式整图](../../outputs/fig1-11_final/Fig10/)。

同目录的 `prepare`、`snapshot`、`run_case` 和报告记录属于[早期 AlphaFold 3 案例](ALPHAFOLD_CASE.md)。重绘 MOF 图直接使用已存事实卡。

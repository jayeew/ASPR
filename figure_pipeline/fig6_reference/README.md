# Fig6 final reference — DAMAGE

用户确认的 Fig.6 最终版本，2026-10-03 固定。当前正式目录：`outputs/fig6_reference/`。

- `final/`：整图 PDF、PNG、600 dpi PNG、可编辑 SVG、文字转曲 SVG。
- `panels/`：a–e 独立 SVG、PDF、PNG、文字转曲 SVG。
- `supplement/`：完整联合图（47 个历史节点）。
- `data/`、`sources/`：原始证据、关系、来源哈希、报告摘要映射。
- `caption_en.md`：英文图注；`source_companion.md`：详细证据说明。
- `PANEL_GUIDE_zh.md`：中文阅读说明；`qa/`：定稿校验结果。
- `reproduce/`：绘图脚本副本；正式入口为 `figure_pipeline/fig6_reference/`。

定稿视觉状态：保留五 panel 和全英文内容；删除 panel / 内部卡片底部灰色说明、整图底部脚注，以及 d、e 最下方总结文字。图例和正文保留。

科学边界：这是基于既有分支结果的回顾性案例。报告原件结尾截断，图中仅总结完整段落，不使用未完成结论。两次重新生成尝试也被拒绝，未冒充成功结果。关系标签未通过独立验证。联合图不能证明因果、绝对首创或系统效果提升。删去的图内说明仍保存在图注和证据说明中。

## 复现

在仓库根目录执行：

```bash
/home/jayee/miniconda3/envs/openrlhf/bin/python -m figure_pipeline.fig6_reference
```

代码依赖仓库内 `fig6_computer` 的共用绘图函数、`fig1_reference` 的 SVG/导出模块以及来源说明中列出的本地研究资料。交付包用于编辑和证据核查，不是脱离仓库的独立运行环境。重新生成会覆盖导出图；`qa/final_manifest.json` 记录本次获批版本的校验值。

早期同名目录已移至相应 `archive/fig6_reference_pre_DAMAGE_*`，不再作为正式 Fig.6 使用。

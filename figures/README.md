# 图件与实验入口

每张图有一个工作目录。按图号查看代码、数据和导出位置。

[Fig1](fig1/) · [Fig2](fig2/) · [Fig3](fig3/) · [Fig4](fig4/) · [Fig5](fig5/) · [Fig6](fig6/) · [Fig7](fig7/) · [Fig8](fig8/) · [Fig9](fig9/) · [Fig10](fig10/) · [Fig11](fig11/)

- `code/`：该图的实现，各模块按职责分组。
- `data/`：原始实验、评价、统计表和绘图输入；相互引用，不复制共享证据。
- `scripts/`：独立脚本。
- `final/`：认可的最终 SVG、PDF、PNG。
- `assets/`：不能由最终图替代的模板或编辑母版。
- `shared/`：多图共用的真人审稿编码等材料。

```bash
python3 -m figures fig3 paths
python3 -m figures fig3 export
```

新克隆即可用 `paths` 查看实际目录。图件母版、模板和实验数据按原路径放在本地。将 `Fig3.svg` 放在 `fig3/final/`，并让当前环境可调用 `cairosvg`，即可用 `export` 从该 SVG 导出 PDF 和 PNG。数据变化后的重绘使用各图原有实现。旧 `figure_pipeline.*` 模块名和 `outputs/*` 路径通过符号链接访问迁移后的目录。 [输出与重绘指南](../outputs/README.md)

Fig3 的最终图已更新为确认的五维来源＋评分布局。`fig3/assets/layout_template.svg` 是重绘其他面板所需的旧布局输入，不是另一份最终图。

论文版本和图件对应关系见[论文工作目录](../manuscript/nature_innovation/README.md)。

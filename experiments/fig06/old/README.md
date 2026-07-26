# Fig. 6 绘制思路指导

## 一句话结论

Fig. 6 的任务是回答 graph-perturbation analysis 是否稳健：它在多领域、数据噪声、文献规模、时间窗口和建模选择变化下是否还能保持稳定，同时明确指出方法在哪些边界条件下会失效。

推荐标题：

```text
Fig. 6 | Robustness and boundary conditions of graph-perturbation analysis
```

## 核心绘图模式

Fig. 6 的核心模式是：

```text
跑完整 robustness 数据 -> Python 直接画统计图
```

总体策略已确认：

```text
优先使用真实数据图。
如果局部数据暂时不全，允许先画 pipeline-ready / schematic version，
但必须在输出文件名、caption 或 panel note 中标明该层是 pipeline-ready。
```

原因：

- 这张图承载可信度证明，所有 panel 都需要精确数值、置信区间、样本量和阈值；
- heatmap、sensitivity curve、forest plot、failure taxonomy 都应由 Python 从 CSV 直接生成；
- `gpt-image-2` 最多用于后期把 Python 生成的 panel 拼成更漂亮的总图，不应重绘或改写数值图。

推荐技术路线：

```text
CSV/JSON audit tables
-> Python pandas/seaborn/matplotlib
-> panel PNG/SVG
-> final figure layout
```

允许的数据来源：

```text
本地 OpenAlex / Fig.1-Fig.5 缓存
本地 outputs 中已有 score table / works / topics
必要时联网补齐领域、venue 或论文元数据
```

统一配色：

```text
Nature / Nature Portfolio: 深红 / 酒红
ASPR graph agent: 蓝色
ASPR-Qwen: 紫色
evidence / verifier / uncertainty: 橙色
neutral context: 灰色
```

## 这张图想回答什么

Fig. 6 不继续展示成功案例，而是补足审稿人会追问的稳定性问题：

- 跨领域是否可重复；
- 加噪、删边、随机边、检索范围变化后排序是否稳定；
- 文献量低到什么程度时开始失效；
- 分析窗口和确认窗口如何影响结论；
- 图构建、社区发现、embedding 和 LLM 后端变化是否改变结果；
- 失败案例能否被系统性归因。

## 推荐输出

```text
outputs/fig06/old/work/kg_perturbation/
  fig6_cross_domain_reproducibility.csv
  fig6_data_quality_perturbation.csv
  fig6_volume_sensitivity.csv
  fig6_temporal_window_sensitivity.csv
  fig6_modeling_choice_reproducibility.csv
  fig6_failure_modes.csv
  fig6_panel_a.png
  fig6_panel_b.png
  fig6_panel_c.png
  fig6_panel_d.png
  fig6_panel_e.png
  fig6_panel_f.png
  fig6_full.png
```

## Panel 设计

### 6a Cross-domain reproducibility

推荐图形：领域 × 指标热力图。

领域可以包括：

```text
CRISPR / biomedical
materials
AI / computer science
neuroscience
clinical translation
```

指标可以包括：

```text
Top-K hit rate
rank stability
OOF Spearman with RGPM
NDCG@10
landmark enrichment
```

画法重点：

- 行是领域，列是验证指标；
- 每格保留数值；
- 右侧显示样本量；
- 样本量不足的格子使用浅灰斜线或脚注。

### 6b Data-quality perturbation

推荐图形：噪声类型 × 噪声水平 heatmap，或多条性能保留曲线。

扰动类型：

```text
加入无关文献噪声
删除引用边
添加随机边
扰动 community label
改变 prior-art 检索范围
```

核心指标：

```text
Jaccard@K
Kendall tau
performance retention
```

画法重点：

- baseline 设为 100% retention；
- 标出 retention >= 0.8 的稳定区；
- 不隐藏明显下降的噪声类型。

### 6c Literature-volume sensitivity

推荐图形：文献保留比例 vs 排序稳定性曲线。

实验设置：

```text
100%, 75%, 50%, 25%, 10% 下采样
低产出 / 中产出 / 高产出领域分组
多随机种子重复
```

画法重点：

- x 轴为保留文献比例；
- y 轴为 rank stability 或 OOF Spearman；
- 阴影表示 seed/bootstrap 不确定性；
- 用竖线标出推荐最小文献量阈值。

### 6d Temporal-window sensitivity

推荐图形：二维 heatmap。

实验网格：

```text
analysis window = 1, 3, 5, 7, 10 years
confirmation horizon = 1, 3, 5, 7, 10 years
```

画法重点：

- x 轴是 analysis window；
- y 轴是 confirmation horizon；
- 颜色是 OOF Spearman、NDCG@10 或 retention；
- 圈出 recommended window；
- 标注短窗口 noisy、长窗口 slow response。

### 6e Modeling-choice reproducibility

推荐图形：forest plot。

比较项：

```text
direct citation only
citation + co-citation
citation + bibliographic coupling
Leiden / Louvain / Infomap
不同 embedding 后端
不同 LLM 后端
```

画法重点：

- x 轴是相对 baseline 的性能变化；
- 0 作为中心参考线；
- 点为平均变化，横线为置信区间；
- 用颜色区分 graph construction、community detection、semantic backend、LLM backend。

### 6f Failure modes

推荐图形：失败类型堆叠条形图 + 2-3 个案例卡片。

失败类型：

| 错误类型 | 含义 |
| --- | --- |
| Hot but not novel | 热门但不新 |
| True but delayed | 真创新但短期不显著 |
| Data-poor frontier | 文献太少，信号不足 |
| Review artifact | 综述造成伪桥接 |
| Noisy prior-art | 检索引入无关文献 |
| Semantic ambiguity | 术语跨领域歧义 |

案例卡片字段：

```text
paper/domain
observed failure
misleading signal
why the method failed
recommended safeguard
```

## 版式建议

推荐 2 × 3：

```text
a  b  c
d  e  f
```

Fig. 6 应该像严谨的统计稳健性图，不要做成概念海报。

## 解释边界

可以说：

```text
方法在多领域、合理数据质量和中等以上文献规模下具有可重复性。
```

不要说：

```text
方法在所有学科、所有时间尺度、所有数据条件下都可靠。
```

caption 必须说明 field、year、window、sample size、metric 和随机种子/重采样设置。

# Fig.3 new：ASPR Score 多尺度预测性能地形解读指南

> 对应图：[`figure_full.png`](../../outputs/fig03/new/figure_full.png)  
> 冻结版本：`fig3-aspr-performance-landscape-v8-blue-to-red`

## 1. 这张图回答什么

Fig.3 检验一个明确的预测问题：仅使用论文发表时可获得的 T0 指标，ASPR
Score 能否对后续科学吸收和跨领域扩散进行时间外排序，并且这种排序能力是否
跨预测窗口、指标证据层级、学科领域和出版年份保持可见。

它与前两张图的分工是：

- Fig.1 展示 landmark 周围知识结构和发表时特征空间的描述性变化；
- Fig.2 冻结指标证据链以及 `7 ⊂ 16 ⊂ 154 ⊂ 221` 四套集合；
- Fig.3 只评价这些冻结集合的时间外预测表现以及正式 ASPR Score 的定义。

图中仅使用正式HGB模型，不把预测关联表述为因果效应或创新性真值。

## 2. 数据时间口径

统一论文和特征语料最晚到 2022 年，但不同预测标签需要等待相应未来窗口成熟：

| 标签 | 成熟论文截止年 | 时间外 OOF 数 |
|---|---:|---:|
| D3 | 2022 | 140,362 |
| D5 | 2020 | 125,070 |
| D8 | 2017 | 101,379 |

因此，Panel c 中 D5 的 2021–2022 和 D8 的 2018–2022 被明确画为
`outcome not yet mature`，而不是普通缺失或模型低性能。

## 3. Panel a：两个分数字段

正式模型是 D5 Full-text 16 HGB。它先分别估计未来吸收概率和在已发生吸收
条件下的扩散程度：

```text
raw_prediction_score
= P(future uptake) × E(diffusion | uptake)
```

随后把原始预测映射到成熟 D5 训练参考群体的经验百分位：

```text
aspr_score = 100 × ECDF_D5(raw_prediction_score)
```

因此：

- `raw_prediction_score` 保留模型的预期扩散尺度；
- `aspr_score` 是便于跨论文阅读的 0–100 相对位置；
- 更高分表示预测的后续吸收与扩散潜力更大，不表示某项指标造成了影响。

当前正式文件为 157,042 篇论文提供这两个字段，论文最晚发表至 2022 年；
百分位参考仍只使用标签已经成熟的 D5 训练群体。

## 4. Panel b：为什么用十分位富集

Panel b 不只报告一个相关系数，而是按 3 行 × 4 列绘制 12 张相互独立的完整
十分位曲线。行是 D3、D5、D8，列是 Strict 7、Full-text 16、Primary 154、
Broad T0 221。对每个 horizon、指标集合和外层时间测试折分别：

1. 按 OOF 预测值分成十个近似等量十分位；
2. 按对应 D3、D5 或 D8 的真实扩散结果定义实际前 10%；
3. 计算每个预测十分位中进入实际前 10% 的比例。

每张小图都显示 D1–D10 的完整单调梯度、10% 随机基线、2,000 次出版年份区块
bootstrap 的 95% 置信区间，以及 D10 的命中率和富集倍数。正式 D5 ×
Full-text 16 单元用琥珀色边框和星号标出，其 D10 为 39.63% 和 3.97 倍。

D5 OOF 中有 105 条记录没有有限的实现扩散标签。这些论文保留在 125,070 条
预测的样本披露中，但不进入实际结果十分位和局部相关计算，避免把缺失结果
错误当作低影响结果。

## 5. Panel c：如何阅读性能地形

Panel c 是三行四列的分层热图：

- 行：D3、D5、D8；
- 列：Strict 7、Full-text 16、Primary 154、Broad T0 221；
- 每张热图横轴：连续的三年后向窗口结束年；
- 每张热图纵轴：12 个固定领域；
- 颜色：该窗口内真实扩散和 OOF 预测的 Spearman 相关。

显示年份 `y` 的单元使用 `y−2`、`y−1`、`y` 三年论文。三年窗口不是图形
插值；它是明确的分析单位，使有效样本不少于 30 的成熟单元比例从单年口径
的约 60.1% 提高至 74.7%。

颜色在全部 12 张热图中共用聚焦的 0.68–0.82 标尺，使总体 OOF ρ=0.734–0.769
之间的差异占完整色谱的 25%，能够清楚分辨；按低表现到高表现依次使用
蓝—青—浅黄—橙—红连续色谱。低于或高于聚焦范围的真实年度窗口分别使用色谱
端点颜色，并由色卡两端的三角形明确披露；底层数值不截断。极浅灰单元表示有效样本少于 30
或数据不可用，另一中性灰表示标签未成熟；两者都不能按连续性能色标解释。
为保证 220 mm 扩展版面下不小于 6.5 pt，12 个领域名只在左侧共享的
`domain order` 键中列出，所有热图均使用同一自上而下顺序。该显示顺序按正式
D5 × Full-text 16 的可靠年度窗口平均 Spearman 从高到低排列，使相近颜色形成
连续带；它只改变领域行的位置，不改变任何单元值。D5 Full-text 16
使用细琥珀色边框和星标，标记其正式模型身份，而不是表示额外统计显著性。

Panel c 不再绘制右侧的 Overall OOF 摘要小热力图。释放出的横向空间全部用于
四列年度—领域主热图，使单元边界、缺失状态和年份结构更容易辨认。全时期
OOF Spearman 仍保留在冻结的审计数据表中，但不在主图重复展示。

## 6. Panel d：正式模型的3D时间—领域地形

Panel d 只使用正式 D5 × Full-text 16 的成熟年份单元，并改为一整片半透明3D
性能地形。先在每个领域内部对成熟年份缺口做线性补齐，再对按 Panel c 顺序排列的
领域—年份网格进行平滑二维插值，因此曲面没有断点或分片边界。该插值只服务于
视觉解释；精确、未插值的年度—领域值仍以 Panel c 为准。曲面不延伸到 D5 尚未
成熟的2021–2022年。z轴使用接近真实数据的0.28–0.90范围，不强制从0开始，
并使用与 Panel c 相同的蓝—青—浅黄—橙—红色谱。图外侧标注正式模型在可靠
成熟窗口内平均 Spearman 最高的三个领域：Chemistry、Earth/climate 和 Astronomy，
并以细箭头指向各领域的实际峰值；这些名称由固定统计规则选择，不是人工挑选。
但在 12 个领域的平均表现范围内归一化，以便清楚区分领域层次；颜色仅承担
相对视觉分组，精确读数仍以 Panel c 和面板表为准。

## 7. Panel e：逐层增益地形

Panel e 固定在正式 D5 任务，逐格计算相邻嵌套集合的相关差：

```text
Full-text 16 − Strict 7
Primary 154 − Full-text 16
Broad T0 221 − Primary 154
```

在有效预测—标签对不少于 30 的三年领域窗口中：

| 扩展 | 可靠单元 | 中位 ΔSpearman | 正增益比例 |
|---|---:|---:|---:|
| Full-text 16 − Strict 7 | 297 | +0.0136 | 79.1% |
| Primary 154 − Full-text 16 | 297 | +0.0010 | 52.5% |
| Broad T0 221 − Primary 154 | 297 | +0.0002 | 51.9% |

这表明从严格 7 扩展到全文 16 的增益更广泛；继续加入大量结构化或词汇代理
后，中位增益接近零，而且正负变化在领域和年份之间交错。该图用于解释性能
饱和，不参与重新选择 Fig.2 的集合成员。

差值色标固定为 `−0.08` 至 `+0.08`：蓝色表示相对前一集合降低，橙红色表示
提高。超出显示范围的单元用上/下三角标记，真实数值仍完整保存在面板数据中。

## 8. 推荐正文表述

可以写：

> Publication-time ASPR scores showed strong out-of-time ranking of subsequent
> scientific diffusion. Papers in the highest predicted decile were enriched
> nearly fourfold for top-decile five-year diffusion, while multi-horizon
> landscapes showed broad but heterogeneous performance across fields and
> publication years.

不要写：

> ASPR proves which papers are innovative or shows that its indicators cause
> later scientific impact.

## 9. 复现与审计

```bash
python3 -m experiments.fig03.new.run --stage all
python3 -m experiments.fig03.new.tests
```

主要审计入口：

- [`decile_enrichment.csv`](../../outputs/fig03/new/panel_data/decile_enrichment.csv)
- [`performance_landscape.csv`](../../outputs/fig03/new/panel_data/performance_landscape.csv)
- [`d5_gain_landscape.csv`](../../outputs/fig03/new/panel_data/d5_gain_landscape.csv)
- [`d5_gain_summary.csv`](../../outputs/fig03/new/panel_data/d5_gain_summary.csv)
- [`audit_report.json`](../../outputs/fig03/new/audit_report.json)
- [`run_manifest.json`](../../outputs/fig03/new/run_manifest.json)

审计会检查正式模型身份、157,042 篇分数、四套指标在 D3/D5/D8 分别使用
140,362／125,070／101,379 条 OOF 预测、12 种组合的十分位
富集、完整 5,040 格性能地形、标签成熟边界、最小样本量、三组增益摘要、
冻结表 SHA-256、时间切分边界，以及 220 × 220 mm 的 SVG/PDF、600 dpi PNG、
灰度和红绿色觉缺陷预览。`--stage all` 只渲染并审计冻结表，不重新训练或重算数值。

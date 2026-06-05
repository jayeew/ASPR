# Fig. 3 README

Fig. 3 现在是一张带诊断保护的实证图：它尝试学习七个 publication-day graph-perturbation indicators 的非负权重，并用严格 out-of-fold 分数检验这个 learned score 是否能预测论文发表后真实发生的知识图谱扰动。

核心问题是：

```text
论文发表当天的结构扰动潜力，能否预测未来真实图谱扰动？
```

如果数据支持，Fig. 3 可以作为主图论证；如果阈值不达标，它会明确标成：

```text
weak empirical association / diagnostic run
```

这意味着当前数据和当前定义还不能强支撑“权重学习有效”的叙事。

## 整体流程

对每篇论文 `p`：

1. 只使用发表当天可见的 `G-` 和 `G0` 计算七个 indicators。
2. 在未来 `tau` 年观察 graph-delta outcomes。
3. 用 matched controls 构造稳定化目标 `RGPM_v2`。
4. 在训练 fold 内学习权重。
5. 在 held-out fold 上得到 `S_w_oof`。
6. 用 `S_w_oof` vs `RGPM_v2` 做最终校准。

加权分数：

```text
S_w(p) = sum_k w_k z_k(p)
w_k >= 0
sum_k w_k = 1
```

## 七个 Publication-Day Indicators

| 指标 | 含义 |
|---|---|
| B | bridge position，论文是否连接原本较远的引用结构 |
| RS | knowledge breadth，引用知识来源是否跨 field 且距离较远 |
| DeltaQ0 | boundary perturbation，发表当天是否扰动社区边界 |
| Uzzi | atypical recombination，引用组合是否非典型 |
| RTD | reference target diversity，引用目标 communities 是否分散 |
| Burt IP | structural holes，论文是否占据结构洞位置 |
| PDE | prospective diffusion entropy，引用 fields 的扩散潜力 |

这些指标会做 rank-normalization，而不是简单填 0 或普通 z-score。

## RGPM-v2 是什么

`RGPM_v2` 是 Realized Graph Perturbation Magnitude 的稳定化版本。它由未来 graph deltas 相对 matched controls 的 z-score 构造：

```text
z_j = (delta_j - median(control_j)) / max(local_MAD, 0.25 * global_MAD, delta_floor)
z_j = clip(z_j, -4, 4)
RGPM_v2 = sqrt(mean(max(z_j, 0)^2))
```

它只使用通过稳定性筛选的 active deltas。旧 Mahalanobis RGPM 只保留为 debug 指标，不再作为主学习目标。

## 主图阈值

Fig. 3 只有同时满足以下条件时，才被视为支持性主图：

```text
active graph deltas >= 5
active delta z-cap hit rate < 5%
OOF Spearman >= 0.30
learned weight beats equal weight by >= 0.03
S_w_oof IQR > 0.35
```

还必须满足数据充分性条件，否则即使 OOF 相关性看起来不错，也只能作为 diagnostic run：

```text
domains >= 4
total papers >= 8,000
papers per domain >= 2,000
landmark or high-RGPM cases per domain >= 20
median matched controls per domain >= 20
relaxed control-tier rate per domain <= 25%
```

这意味着只在 CRISPR 单领域上学习出的权重默认不作为主图证据；它可以用来定位问题，但不能支撑稳定的跨领域 scoring claim。

否则图仍然生成，但必须按 diagnostic run 解读。

如果当前 Fig. 1 selected graph 样本太小，可以用 `--fig1-corpus-source raw` 从 `works_raw.jsonl` 构造更大的 Fig. 3 输入；该模式会按 `primary_topic` 生成较粗 community，适合做 multi-domain 审计和样本扩展测试。

## Panel a: Empirical Learning Framework

Panel a 只讲整体框架：

```text
Publication-day indicators -> Weight learning -> OOF validation against RGPM-v2
```

七指标被归入三个机制组：

- Expansion: RS, PDE, Uzzi
- Bridging: B, RTD, Burt IP
- Reconfiguration: DeltaQ0, Uzzi

Panel a 的意义是说明：权重不是主观指定，而是在训练 fold 内通过未来真实图结构变化反推学习，并在 held-out fold 上验证。

## Panel b: Stabilized RGPM-v2

Panel b 只解释目标变量如何构造。

左侧展示 active graph deltas 的 matched-control z-score，坐标固定在 `[-4, 4]`。如果某个 z-score 触发 clipping，会用三角标记。

右侧展示 `RGPM_v2` 的计算公式，并列出被稳定性筛选排除的 deltas，例如：

- 方差过小
- z-cap hit rate 过高
- control MAD-zero rate 过高
- compression diagnostic 默认不进入主 RGPM

Panel b 的意义是检查：目标变量是否稳定，是否被少数极端 delta 主导。

## Panel c: Mechanism-Level Landscape

Panel c 把七指标权重压缩成三个机制权重：

```text
Expansion / Bridging / Reconfiguration
```

三角图颜色表示：

```text
Delta Spearman rho vs equal-weight baseline
```

它不再画噪声化的原始散点，而是 hexbin/grid-smoothed mean performance。黑色星号是最终 all-data best mechanism point。

如果 top 10% 权重区域不集中，图中会显示：

```text
no stable basin
```

Panel c 的意义是判断机制层面是否存在稳定高性能区域，而不是强行制造 landscape 叙事。

## Panel d: Constrained Two-Weight Profiles

Panel d 展示三组关键指标的局部权重地形：

```text
B vs RTD
DeltaQ0 vs Uzzi
RS vs PDE
```

每个热图是真正的 simplex profile：

- 网格只显示 `w_i + w_j <= 1` 的有效区域。
- 固定两个权重。
- 对剩余指标重新采样。
- cell value 是相对 equal-weight baseline 的 best CV-compatible performance。
- 无效或缺样本区域置灰。

Panel d 的意义是检查关键权重之间是互补、替代，还是没有稳定结构。

## Panel e: Weight Stability

Panel e 不再画大量 spaghetti lines，而是展示：

- top 1% weights 的 median
- top 1% weights 的 IQR ribbon
- final all-data best weight
- fold-level best weights
- 每个指标成为 top-weight 的频率

Panel e 的意义是判断权重结构是否稳定。如果 IQR 很宽、fold 点分散、top-weight frequency 不集中，就说明当前权重学习更像是在适配噪声，而不是找到稳定机制。

## Panel f: OOF Score Calibration

Panel f 是最关键的验证 panel。

左侧 rank-decile calibration 使用：

```text
x = S_w_oof percentile decile
y = mean / median RGPM-v3 percentile
```

不是全样本 best-weight score。左图同时报告完整数据上的 OOF Spearman rho 和 top-decile lift。这个视图更贴近 rank validation，避免线性散点拟合被长尾 RGPM 和异方差主导。

右侧不再使用 radar plot，而是 Low / Mid / High OOF score tertile 的七指标横向条形摘要：

```text
mean rank-normalized indicator +/- bootstrap SE
```

Panel f 的意义是回答：严格 held-out 分数是否真的能预测未来 RGPM-v3。如果 rho 低或 summary fail，就只能说有弱关联或诊断价值。

默认每次运行都会导出核心审计表：

```text
fig3_diagnostics_summary.json
fig3_oof_score_table.csv
fig3_cv_summary.csv
fig3_baseline_comparison.csv
fig3_diagnostics_delta_stability.csv
fig3_diagnostics_domain_adequacy.csv
fig3_indicator_target_correlations.csv
fig3_rgpm_component_correlations.csv
fig3_control_tier_audit.csv
fig3_nonlinear_upper_bound.csv
```

其中 `fig3_nonlinear_upper_bound.csv` 是 quadratic ridge 的 diagnostic-only OOF 上界。如果它明显高于线性 simplex，说明七指标中可能有非线性信号；如果它也低，优先排查数据覆盖、RGPM 构造和指标定义。

## 如何阅读整张图

建议顺序：

1. 看全图标题：是否是 validated association，还是 diagnostic run。
2. 看 Panel b：active deltas 是否足够，z-cap 是否稳定。
3. 看 Panel f：OOF Spearman 是否足够，散点是否真的有排序关系。
4. 看 Panel c/d：是否存在稳定权重 landscape。
5. 看 Panel e：fold weights 和 top weights 是否稳定。
6. 最后再把 Panel a 当作方法总览。

## 与 Fig. 1 / Fig. 2 的关系

Fig. 1 提供本地知识图谱和 OpenAlex/Fig.1 缓存数据。Fig. 3 默认复用 Fig. 1 的本地输出，但会优先从 `works_raw.jsonl` 重建方向性 citation rows。

Fig. 2 定义七个 publication-day indicators。Fig. 3 的任务是检验这些 indicators 的加权组合是否能预测未来真实图谱扰动。

## 重要解读原则

如果 `fig3_diagnostics_summary.json` 中 `overall_pass=false`，不要把 Fig. 3 解读成“权重学习已经有效证明”。这时它的正确定位是：

```text
诊断当前 RGPM、controls、graph deltas 和 feature 标准化哪里还不稳定。
```

这比把弱结果画成强结论更重要。

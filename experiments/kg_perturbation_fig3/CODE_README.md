# Fig. 3 Code README

本文档解释 `fig3_empirical_weight_learning.py` 的本地输入输出适配、核心计算逻辑、诊断导出和常用运行方式。当前版本的重点不是把 Fig. 3 画成“理想示意图”，而是先保证实证链条可检查：RGPM 更稳定、权重学习使用严格 out-of-fold 验证、图中明确区分强结果和 diagnostic run。

## 入口脚本

```bash
python experiments/kg_perturbation_fig3/fig3_empirical_weight_learning.py --panel all
```

默认输入根目录：

```text
outputs/kg_perturbation_fig1/
```

默认输出根目录：

```text
outputs/kg_perturbation_fig3/
```

默认 `--run-mode both`，会尝试运行：

```text
crispr
graphene_2d_materials
ipsc_reprogramming
transformer_foundation_models
multi_domain
```

如果某个 domain 的本地 Fig. 1 输出不可用，该 domain 会被跳过；如果少于两个 domain 可用，则跳过 `multi_domain`。

## 输入适配

Fig. 3 需要真实方向性引用边，因为它要区分：

- 发表当天之前已经存在的 prior references
- 发表后未来窗口内引用该论文的 future citers

因此脚本优先从 Fig. 1 的 `works_raw.jsonl` 中读取每篇论文的 `refs` 字段，重建：

```text
source = citing paper
target = cited paper
```

只有当 `works_raw.jsonl` 不存在或无法产生 citation rows 时，才 fallback 到 Fig. 1 的 `paper_edges.csv`。fallback 时默认仅保留 `direct > 0` 的边；若要包含 bibliographic/cocitation hybrid edges，显式传入：

```bash
--include-hybrid-edges
```

标准化后的 Fig. 3 输入写到：

```text
outputs/kg_perturbation_fig3/fig3_input/<domain>/
```

包含：

```text
works.csv
citations.csv
topics.csv
topic_edges.csv
fig3_input_report.json
```

## 多 Domain 合并

`--run-mode multi_domain` 或 `both` 会把多个 domain 合并后运行。合并时会：

- 给 paper id 加 domain 前缀，避免跨 domain id 冲突。
- 给 community id 加 offset，避免不同 domain 的 community 编号相互污染。
- 保留每个 domain 自己的 `analysis_end_year`，计算 future window 时按论文所属 domain 的可观测截止年判断。

输出结构：

```text
outputs/kg_perturbation_fig3/
  crispr/
  graphene_2d_materials/
  ipsc_reprogramming/
  transformer_foundation_models/
  multi_domain/
  fig3_run_selection.json
```

`fig3_run_selection.json` 记录主图选择逻辑：只要 `multi_domain` 完成，就优先选择它作为证据基准；如果它未通过阈值，则仍标为 diagnostic。只有 multi-domain 不可用时，才回退到 single-domain diagnostic run。

## 主计算流程

主函数：

```python
compute_all(raw, args)
```

分为五步：

1. 计算 publication-day indicators 和 future graph deltas。
2. 用 matched controls 构造稳定化 `RGPM_v2`。
3. 对七个 indicators 做 metric-specific transform 和 rank-normalization。
4. 使用严格 out-of-fold 方式学习非负 simplex 权重。
5. 生成 panels、诊断表和 pass/fail summary。

## Step 1: Indicators And Graph Deltas

核心函数：

```python
compute_indicator_and_delta_tables(...)
```

每篇论文 `p` 只有在满足：

```text
p.year + tau <= p.domain_analysis_end_year
```

时进入候选样本。`--min-refs` 控制 prior references 的最低数量。

七个 publication-day indicators：

```text
B
RS
DeltaQ0
Uzzi
RTD
BurtIP
PDE
```

九个 future graph deltas：

```text
community_reach
field_entropy
cross_community_adoption
path_shortening
modularity_shock
partition_change
boundary_mixing
post_perturbation_concentration
hub_formation
```

方向统一规则：

- `community_reach` higher = stronger perturbation
- `field_entropy` higher = stronger perturbation
- `cross_community_adoption` higher = stronger perturbation
- `path_shortening` higher = stronger perturbation
- `modularity_shock = max(0, -(Q_future - Q_minus))`
- `partition_change = JSD(reference_comms, future_citer_comms)`
- `boundary_mixing` higher = stronger perturbation
- `post_perturbation_concentration` 默认只作为 compression diagnostic
- `hub_formation` 如果稳定性不达标会自动移除

## Step 2: RGPM-v2

核心函数：

```python
compute_rgpm(...)
```

当前主目标不再使用 Mahalanobis distance，而是 `RGPM_v2`：

```text
z_j = (delta_j - median(control_j)) / max(local_MAD, 0.25 * global_MAD, delta_floor)
z_j = clip(z_j, -4, 4)
z_j_pos = max(z_j, 0)
RGPM_v2 = sqrt(mean(z_j_pos^2 over active deltas))
```

默认参数：

```text
--z-cap 4.0
delta_floor = 1e-3
比例类 delta floor = 0.02
matched controls hard floor = 10
```

旧 Mahalanobis 仍保留为：

```text
RGPM_mahalanobis_debug
```

但它不再作为权重学习主目标。

Graph delta 稳定性筛选规则：

```text
drop if nonzero_rate < 0.03
drop if global_MAD < 1e-6
drop if z_cap_hit_rate >= 0.10
drop if control_MAD_zero_rate >= 0.50
```

如果没有任何 delta 通过筛选，代码会选择最少数 fallback deltas 以便生成 diagnostic figure，但 summary 会明确 fail。

## Step 3: Feature Normalization

核心函数：

```python
field_year_standardize(...)
```

新版不是普通 robust z-score，而是 rank-normalized features：

```text
within field-year if n >= 20
else within field if n >= 50
else global
```

metric-specific transform：

```text
B        -> log1p(B)
RS       -> raw
DeltaQ0  -> winsorized 1%-99%
Uzzi     -> invalid pair cases marked missing before normalization
RTD      -> raw
BurtIP   -> raw
PDE      -> raw
```

rank transform：

```text
p = (rank - 0.5) / n
z = NormalDist().inv_cdf(p)
z = clip(z, -3, 3)
```

如果某个指标有效值比例 `< 0.10`，不参与主权重学习；`< 0.30` 会在 diagnostics 中标记 warning。学习阶段对标准化后的缺失值使用中性 `z=0`，这不是原始指标填 0，而是避免 sparse indicator 导致 complete-case 样本塌缩。

## Step 4: Strict OOF Weight Learning

核心函数：

```python
learn_weights(...)
```

权重满足：

```text
w_k >= 0
sum_k w_k = 1
```

严格 OOF 流程：

1. 生成 outer CV folds。
2. 每个 fold 只用 train fold 选择最优权重。
3. 用该 fold 的最优权重预测 held-out fold。
4. 拼接得到 `S_w_oof`。
5. Panel f 和 model diagnostics 只使用 `S_w_oof` vs mechanism-balanced `RGPM-v3`。

全数据 best weight 仍会计算，但只用于解释性 panels，例如 Panel e 的 final best weight。

每次运行都会导出核心审计表。`--audit-only` 只计算并导出这些诊断，不渲染图片。新增表包括：

```text
fig3_indicator_target_correlations.csv
fig3_rgpm_component_correlations.csv
fig3_control_tier_audit.csv
fig3_nonlinear_upper_bound.csv
```

`fig3_nonlinear_upper_bound.csv` 使用 quadratic ridge 做 diagnostic-only OOF 上界，用于判断弱结果是否来自线性 simplex 表达力不足。

固定 baselines：

```text
equal_weights
best_single_indicator
reference_count
cited_by_count
random_dirichlet_median
learned_weight_oof
```

## Diagnostics And Thresholds

使用 `--export-tables` 或 `--diagnostics` 会输出：

```text
fig3_diagnostics_delta_stability.csv
fig3_diagnostics_features.csv
fig3_diagnostics_model.csv
fig3_diagnostics_controls.csv
fig3_diagnostics_summary.json
fig3_oof_score_table.csv
fig3_fold_weights.csv
fig3_baseline_comparison.csv
```

主图阈值：

```text
active graph deltas >= 5
active delta z-cap hit rate < 5%
OOF Spearman >= 0.30
learned weight beats equal weight by >= 0.03
S_w_oof IQR > 0.35
```

如果不达标，图仍生成，但标题和注释会显示：

```text
weak empirical association / diagnostic run
```

## 常用命令

快速 smoke test：

```bash
python experiments/kg_perturbation_fig3/fig3_empirical_weight_learning.py \
  --run-mode single_domain --domains crispr \
  --panel a --max-papers 100 --min-refs 1 --min-controls 5 \
  --n-weight-samples 200 --n-folds 2 --cv-mode random \
  --progress-interval 25 --export-tables --diagnostics
```

单 domain 正式测试：

```bash
python experiments/kg_perturbation_fig3/fig3_empirical_weight_learning.py \
  --run-mode single_domain --domains crispr \
  --panel all --export-tables --diagnostics
```

多 domain 测试：

```bash
python experiments/kg_perturbation_fig3/fig3_empirical_weight_learning.py \
  --domains crispr graphene_2d_materials ipsc_reprogramming transformer_foundation_models \
  --run-mode multi_domain --cv-mode domain --panel all --export-tables --diagnostics
```

主图结论需要至少 4 个 domain、每个 domain 约数千篇论文、每个 domain 足够 landmark 或 high-RGPM cases，并且 matched controls 不能大量退化到宽松 tier。单 CRISPR 或小样本 run 会保留为 diagnostic，不再被提升为主图证据。

如果要用 Fig. 1 的更大 raw corpus 而不是展示图的 `works_selected.csv`，可以加：

```bash
--fig1-corpus-source raw
```

该模式使用 `works_raw.jsonl` 中的全部可用论文，并按 `primary_topic` 构造较粗的 Fig. 3 community。它适合扩大训练/审计样本，但正式主图前仍应检查 `fig3_diagnostics_domain_adequacy.csv` 和 control-tier audit。

调试 Panel d 时可降低 profile 规模：

```bash
--profile-grid-size 12 --profile-n 10
```

## 运行日志

脚本会打印：

- 输入准备来源和 citation source
- raw data 行数
- eligible papers 数量
- 已扫描论文数、已计算论文数、跳过原因
- cache build 年份
- RGPM rows 进度、controls 数量、active deltas
- fold-level weight learning 进度
- OOF Spearman、equal baseline、diagnostic status
- 每个 panel 的保存路径

用 `--quiet` 可以关闭日志。

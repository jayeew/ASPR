# Fig. 7 绘制思路指导

## 一句话结论

Fig. 7 的任务是突出不同 journal / publisher / venue family 在承载 graph-perturbing research 上的系统差异，并把结论组织成 venue-family contribution under field-year controls；Nature / Nature Portfolio 只能写成当前语料中的 aggregate VCI point-estimate 结果，不能写成因果 superiority。

推荐标题：

```text
Fig. 7 | Venue-family contribution under field-year controls
```

备选标题：

```text
Fig. 7 | Venue-level innovation contribution across journals and publishers
```

## 核心绘图模式

Fig. 7 的核心模式是：

```text
跑 venue-level 数据 -> Python 直接画核心统计图 -> 可选 gpt-image-2 做出版级拼图
```

总体策略已确认：

```text
主图使用真实数据图。
如果某些 venue family 的数据暂时不完整，允许先画 pipeline-ready / schematic version，
但 Nature Portfolio 的主结论必须最终由 field-year normalized 数据替换确认。
```

原因：

- Nature Portfolio contribution 的结论必须来自可追溯的 field-year normalized 数据；
- 排名、富集、置信区间、机制热力图、publication-day vs future impact 都应由 Python 精确绘制；
- `gpt-image-2` 可用于把 Nature highlight、publisher family 图例、panel 标题和 visual hierarchy 做得更像 Nature 风格，但不能改动数值、排名和置信区间。

推荐技术路线：

```text
venue metrics CSV
-> Python ranking / forest plot / heatmap / scatter
-> exact panel exports
-> gpt-image-2 or manual layout for final polish
```

## 预设主结论与数据纪律

这张图的投稿安全主结论是：

```text
Nature Portfolio has the top aggregate VCI point estimate in the current
field-year controlled corpus; strict interval separation and per-paper
intensity caveats remain audited.
```

但图稿必须避免给人“硬吹 Nature”的感觉。建议把结论写成数据支持的分层证据：

1. Nature Portfolio 在 field-year normalized innovation contribution index 上最高；
2. Nature Portfolio 对 top 1% / top 5% graph-perturbing papers 显著富集；
3. Nature Portfolio 的 mechanism signature 更均衡，尤其在 bridging、reconfiguration、translation 上突出；
4. publication-day perturbation signal 与 future impact 分开计算，说明不是只测发表后 visibility；
5. 结果被解释为 portfolio-level contribution，而不是“期刊因果制造创新”。

## 必须控制的混杂因素

为了让 Nature 最大的结论站得住，必须控制：

```text
field
year
article type
reference count
team size
journal subject scope
citation delay
open access status
review article vs research article
```

如果数据允许，主图使用 field-year normalized score；supplement 报告更完整的回归或匹配结果。

比较层级已确认：

```text
主图：venue family 层面
supplement：journal 层面
```

主图建议 venue family：

```text
Nature Portfolio
Science family
Cell Press
PNAS
Lancet family
IEEE / ACM
Elsevier flagship journals
Springer / Wiley families
```

允许的数据来源：

```text
本地缓存与 outputs 中已有论文/venue 元数据
OpenAlex / Crossref / publisher 页面等联网补全 venue family
必要时使用手工 venue-family mapping 表
```

统一配色：

```text
Nature Portfolio: 深红 / 酒红，高亮主结论
其他 venue family: 中性灰或低饱和辅助色
ASPR graph evidence: 蓝色
evidence / uncertainty: 橙色
```

## 推荐输出

```text
outputs/kg_perturbation_fig7/
  fig7_venue_portfolio.csv
  fig7_vci_rankings.csv
  fig7_topk_enrichment.csv
  fig7_mechanism_signature.csv
  fig7_pre_post_publication_signal.csv
  fig7_confounder_audit.csv
  fig7_panel_a.png
  fig7_panel_b.png
  fig7_panel_c.png
  fig7_panel_d.png
  fig7_panel_e.png
  fig7_panel_f.png
  fig7_full.png
```

## Panel 设计

### 7a Venue portfolio map

推荐图形：venue portfolio scatter。

坐标：

```text
x: mean field-year normalized graph-perturbation score
y: future realized impact / RGPM / citation-normalized impact
point size: number of papers
point color: publisher / journal family
```

画法重点：

- Nature Portfolio 使用深红/酒红主色并直接标注；
- Science、Cell、PNAS、Lancet、IEEE/ACM、Elsevier、Springer/Wiley 作为对照；
- 用淡色背景区分 high perturbation / high realized impact 象限；
- 样本量过小的 venue 用空心点或灰色点。

这一 panel 的目标读法：

```text
Nature Portfolio 位于高 graph-perturbation、高 future impact 区域。
```

### 7b Field-year normalized venue contribution index

推荐图形：VCI 排名 dot-and-interval chart。

指标：

```text
VCI_j = mean(z_perturbation_paper | journal = j, field, year)
```

画法重点：

- y 轴是 venue / publisher / venue family；
- x 轴是 field-year normalized VCI；
- Nature Portfolio 排第一并高亮；
- 显示置信区间和论文数；
- 其他 venue 使用中性灰或次级颜色。

这一 panel 是 Fig. 7 的主结论 panel。

### 7c High-perturbation paper enrichment

推荐图形：top-K enrichment forest plot。

指标：

```text
Enrichment_j = observed_topK_j / expected_topK_j
```

画法重点：

- x 轴为 enrichment ratio；
- 1 为参考线；
- top 1% 和 top 5% 可用两个点或两个小分面；
- Nature Portfolio 的 enrichment ratio 应该位于最高或显著高于 1；
- 显示 bootstrap 或 binomial confidence interval。

目标读法：

```text
Nature Portfolio 对最高扰动论文显著富集。
```

### 7d Mechanism signature by venue

推荐图形：venue × mechanism heatmap。

机制维度：

```text
Expansion
Bridging
Reconfiguration
Atypical recombination
Translation / compression
```

画法重点：

- Nature Portfolio 放在首行；
- 颜色表示 field-year normalized mechanism score；
- 右侧加 total VCI 或 contribution index；
- 可用小图标标出 Nature Portfolio 的强项机制；
- 不要只展示总分，必须展示机制结构。

目标读法：

```text
Nature Portfolio 不只是总分高，而是在多种创新机制上呈现更均衡和更高强度的承载能力。
```

### 7e Publication-day signal vs future visibility

推荐图形：scatter / binned line。

坐标：

```text
x: publication-day graph-perturbation score
y: future impact / RGPM / citation-normalized impact
color: venue family
```

画法重点：

- x 轴必须是发表当天可见信号；
- y 轴是未来影响；
- Nature Portfolio 单独高亮；
- 加入 field/year adjustment 或分面；
- 用注释说明该 panel 区分 ex ante signal 与 post-publication visibility。

目标读法：

```text
Nature Portfolio 的高贡献不只是发表后曝光度，而是在 publication-day graph signal 上已经可见。
```

### 7f Contribution summary and control audit

推荐图形：证据链 summary panel。

内容：

```text
Nature ranks highest in normalized VCI
Nature enriches top graph-perturbing papers
Nature shows broad mechanism coverage
Signal is measured at publication day
Controls include field/year/article type/reference count
```

画法重点：

- 作为结论板块放在右下或底部；
- 用 check marks 或 evidence cards；
- 可以写 portfolio-level contribution，避免写 causal journal effect。

## 版式建议

推荐 2 × 3：

```text
a  b  c
d  e  f
```

如果想让 Nature 结论更强，可以让 7b 横跨上方两列：

```text
b  b  a
c  d  e
f  f  f
```

## 解释边界

可以说：

```text
Nature Portfolio has the strongest field-year normalized venue contribution in our corpus.
```

不要说：

```text
Nature causally creates innovation.
```

这张图的叙事应是“Nature 对创新型研究的承载和选择贡献最大”，不是“Nature 让论文变创新”。

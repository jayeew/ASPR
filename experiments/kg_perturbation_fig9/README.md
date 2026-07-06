# Fig. 9 绘制思路指导

## 一句话结论

Fig. 9 的任务是展示 ASPR 的一个全流程运行实例：从输入论文开始，分别经过 graph-perturbation agent 和 ASPR-Qwen 两条路径，最终融合成 evidence-grounded 评审意见，并展示关键文字输出、证据来源、流程阶段和不确定性标记。

推荐标题：

```text
Fig. 9 | End-to-end ASPR run generating an evidence-grounded review from agent evidence and ASPR-Qwen
```

## 核心绘图模式

Fig. 9 的核心模式是：

```text
跑轻量案例数据 -> 把结构化文本、指标和流程喂给 gpt-image-2 绘制实例图
```

总体策略已确认：

```text
Fig. 9 接一个真实运行实例。
案例应选择 Nature / Nature 子刊且有 Transparent Peer Review 的论文。
ASPR-Qwen 当前暂不可用时，可以先假设一个 ASPR-Qwen 输出结果用于绘制 pipeline-ready case figure。
```

原因：

- Fig. 9 是 case storyboard，不是大规模统计图；
- 主要内容是流程阶段、关键文本片段、证据 trace、agent 输出、ASPR-Qwen 输出和融合结果；
- 可以先用 Python 生成七指标 profile、小型 trace table 和运行日志，但最终视觉更适合 `gpt-image-2` 画成 publication-style case figure；
- 所有文字必须来自真实运行输出或人工整理后的 `panel_text.json`，不能让模型自行发明评审意见。

推荐技术路线：

```text
case_manifest.csv
agent_output.json
aspr_qwen_output.json
fusion_output.json
trace_table.csv
runtime_log.csv
assumed_aspr_qwen_output.json
-> fig9_panel_text.json
-> gpt-image-2 case storyboard
```

## 这张图想回答什么

Fig. 9 回答：

```text
ASPR 对一篇真实论文完整运行时长什么样？
agent、ASPR-Qwen、fusion 和 verifier 各自输出什么？
最终评审意见如何做到 evidence-grounded、可追踪、可审核？
```

这张图允许文字实例和流程划分偏多，但要控制层级，避免变成长截图。

## 案例选择建议

优先选择：

- 有论文全文、摘要、引用和 reference list；
- 有 Nature 或相近 venue 的 peer review 对照；
- graph-perturbation profile 有强项和弱项；
- agent 与 ASPR-Qwen 的输出能互补；
- fusion 后能生成较完整的 evidence-grounded review；
- verifier 至少触发 1-2 个低置信或证据不足提醒。

允许的数据来源：

```text
本地 Nature markdown / peer-review 语料：
  Windows: D:\aspr_nature_markdown
  WSL/Linux: /mnt/d/aspr_nature_markdown
本地 Fig.1-Fig.7 缓存和 outputs
Nature article / Transparent Peer Review 页面联网补齐 metadata
```

统一配色：

```text
Nature case paper / Nature source: 深红 / 酒红
ASPR graph agent: 蓝色
ASPR-Qwen: 紫色
evidence / verifier / uncertainty: 橙色
final fused review: 深灰或黑色
```

## 推荐输出

```text
outputs/kg_perturbation_fig9/
  fig9_case_manifest.csv
  fig9_agent_output.json
  fig9_aspr_qwen_output.json
  fig9_assumed_aspr_qwen_output.json
  fig9_fusion_output.json
  fig9_claim_evidence_trace.csv
  fig9_runtime_log.csv
  fig9_panel_text.json
  fig9_image2_prompt.md
  fig9_layout_draft.png
  fig9_full.png
```

## Panel 设计

### 9a Input manuscript and run setup

推荐图形：paper card + run setup card。

内容：

```text
title
abstract summary
field
year
reference count
available prior-art corpus
selected model checkpoint
retrieval setting
```

画法重点：

- 左侧是输入论文；
- 右侧是本次 ASPR run 的设置；
- 不要塞太多摘要文字，保留 2-3 行 summary。

### 9b End-to-end execution timeline

推荐图形：横向流程 timeline。

阶段：

```text
parse paper
retrieve prior art
build / query graph
compute perturbation profile
agent innovation evaluation
ASPR-Qwen review generation
fusion
verification
final review
```

画法重点：

- 每一步显示输入和输出；
- 可以标注 runtime seconds；
- agent lane 和 ASPR-Qwen lane 在中间并行出现；
- fusion 之后重新合流。

### 9c Agent evidence and innovation assessment

推荐图形：指标 profile + evidence card。

内容：

```text
B / RS / DeltaQ0 / Uzzi / RTD / Burt IP / PDE percentile
prior-art overlap
graph bridging evidence
innovation claim
risk of overclaim
```

画法重点：

- 指标 profile 可由 Python 先画小图；
- 旁边放 agent 的 2-3 条结构化创新性判断；
- 每条判断带 evidence id。

### 9d ASPR-Qwen reviewer output

推荐图形：review-style text cards。

内容：

```text
summary judgement
novelty and significance
major strengths
major weaknesses
reviewer concerns
recommendation tendency
```

画法重点：

- 明确标注该输出来自 ASPR-Qwen；
- 展示其 review-style 评审语言和关注点；
- 不要把它和 agent evidence 混在同一 panel 中。

### 9e Fusion into final evidence-grounded review

推荐图形：two-source fusion card。

输入：

```text
agent evidence-backed novelty assessment
ASPR-Qwen review-style draft
```

输出：

```text
final structured review
evidence-grounded novelty judgement
limitations
reviewer-style recommendation
confidence
```

画法重点：

- 用颜色标明哪些句子来自 agent evidence，哪些来自 ASPR-Qwen，哪些来自 fusion；
- 最终评审意见可以偏文字，但每块保持短句；
- 重点展示“共同生成”，而不是单模型生成。

### 9f Evidence trace, verifier, and peer-review comparison

推荐图形：trace table + safety flags。

内容：

```text
claim -> evidence source
claim -> prior-art paper
claim -> metric profile
claim -> graph evidence
claim -> verifier status
```

可加 peer-review 对照：

```text
overlap with human peer review
missing human point
agent-only point
ASPR-Qwen-only point
```

画法重点：

- 只放 3-5 条关键 claim；
- 每条 claim 有 evidence status；
- 对 unsupported claim 加 warning；
- 说明 verifier 如何让最终报告更可靠。

## 版式建议

推荐 storyboard 布局：

```text
a  b  b
c  d  e
f  f  f
```

原因：

- 9b 是全流程主线，适合横跨两列；
- 9c 和 9d 分别展示两条生成路径；
- 9e 展示融合；
- 9f 用底部宽 panel 展示 trace 和对照。

## 给 gpt-image-2 的输入准备

建议准备：

```text
fig9_panel_text.json
fig9_trace_table.csv
fig9_metric_profile.csv
fig9_runtime_log.csv
fig9_image2_prompt.md
```

提示词必须强调：

```text
Use only provided text.
Preserve the two-branch ASPR run: agent evidence and ASPR-Qwen reviewer.
Do not invent peer-review claims.
Keep text readable and grouped into cards.
```

## 解释边界

可以说：

```text
This case illustrates how ASPR combines agent evidence and ASPR-Qwen to produce a traceable evidence-grounded review draft.
```

不要说：

```text
One case proves the system is universally reliable.
```

# Fig. 8 绘制思路指导

## 一句话结论

Fig. 8 的任务是绘制 ASPR 的核心算法框架流程图。最终方法不是单独的 agent，也不是单独的 LLM，而是：

```text
ASPR = graph-perturbation agent + ASPR-Qwen
```

两条路径共同生成最终 human-like 论文评审意见：agent 负责证据、检索、图谱扰动指标和创新性判断；ASPR-Qwen 是用 Nature 历年论文及其 peer review 配对语料做 SFT 的垂直评审模型，用来学习人类评审写法、关注点和评价结构；最后通过融合、校验和安全门控生成最终报告。

推荐标题：

```text
Fig. 8 | ASPR algorithm framework integrating graph-perturbation agents with ASPR-Qwen
```

## 核心绘图模式

Fig. 8 的核心模式是：

```text
轻量结构化数据 -> gpt-image-2 绘制算法框架流程图
```

总体策略已确认：

```text
Fig. 8 是算法框架图，不是统计结果图。
绘制时可以把 ASPR-Qwen 按已训练完成的模块呈现，
即使当前 checkpoint 仍在训练中。
```

原因：

- Fig. 8 主要是方法架构和算法流程，不是统计结果图；
- 需要画清楚双路径系统：ASPR graph agent 与 ASPR-Qwen reviewer；
- 最适合准备一个严格的模块清单、流程 JSON、训练数据说明和布局草图，然后喂给 `gpt-image-2` 生成出版级框架图；
- 如果需要完全可控，也可以用 Mermaid / Graphviz / SVG 先生成草图，再交给 `gpt-image-2` 美化。

推荐技术路线：

```text
module_inventory.csv
flow_spec.json
panel_text.json
-> gpt-image-2 framework figure
-> 人工核对所有模块名和箭头方向
```

## 这张图想回答什么

Fig. 8 回答：

```text
ASPR 如何把图谱扰动证据、agentic retrieval/reasoning、Nature peer-review SFT model
组合成一个能生成 human-like review 的端到端系统？
```

必须突出三件事：

1. agent 生成创新性评价；
2. ASPR-Qwen 用 Nature 历年论文及其对应 peer review 训练，生成论文评审意见；
3. 最终输出来自 agent + ASPR-Qwen 的融合，而不是单一路径。

训练语料位置已确认：

```text
Windows: D:\aspr_nature_markdown
WSL/Linux: /mnt/d/aspr_nature_markdown
```

模型设定已确认：

```text
base family: Qwen
training method: SFT
model name in figure: ASPR-Qwen
current state: still training, but figure can depict the completed module
```

最终 human-like review 必须保留：

```text
novelty / significance
strengths / weaknesses
prior-art comparison
graph-perturbation evidence
reviewer-style recommendation
confidence / human-in-loop flag
```

统一配色：

```text
Nature corpus / Nature Portfolio: 深红 / 酒红
ASPR graph agent: 蓝色
ASPR-Qwen: 紫色
evidence / verifier / uncertainty: 橙色
fusion / final report: 深灰或黑色
```

## 推荐输出

```text
outputs/kg_perturbation_fig8/
  fig8_module_inventory.csv
  fig8_flow_spec.json
  fig8_qwen_training_data_schema.json
  fig8_fusion_schema.json
  fig8_image2_prompt.md
  fig8_panel_text.json
  fig8_layout_draft.png
  fig8_full.png
```

## Panel 设计

### 8a Overall ASPR algorithm framework

推荐图形：双路径总流程图。

主路径：

```text
Input manuscript
-> paper parsing / metadata extraction
-> Branch 1: Graph-perturbation agent
-> Branch 2: ASPR-Qwen reviewer
-> fusion and verification
-> final human-like review
```

画法重点：

- 左侧是 input paper；
- 中间分成上下两条 lane；
- 上 lane 是 agent evidence engine；
- 下 lane 是 ASPR-Qwen review model；
- 右侧合流到 fusion/verifier；
- 最右侧输出 final review report。

这是 Fig. 8 的中心 panel。

### 8b Nature peer-review training corpus and ASPR-Qwen

推荐图形：训练数据到垂直模型的流程图。

内容：

```text
Nature historical papers
Transparent Peer Review reports
paper-review paired corpus
instruction / response formatting
SFT fine-tuning
ASPR-Qwen reviewer
```

画法重点：

- 明确训练数据是 Nature 历年论文和对应 peer review；
- 输入对齐为 manuscript -> human review；
- 输出模型命名为 ASPR-Qwen；
- 不要画成 generic LLM prompt。

这一 panel 的核心作用是告诉读者：ASPR 有一个学习 Nature peer-review 风格和判断结构的垂直模型。

### 8c Graph-perturbation agent evidence engine

推荐图形：agent 子流程图。

内容：

```text
prior-art retrieval
citation graph retrieval
knowledge graph construction / update
seven-indicator computation
graph-perturbation profile
innovation assessment
evidence trace
```

画法重点：

- 使用 Fig. 1-Fig. 7 的方法链条作为证据来源；
- 七指标和 RGPM 不要详细展开公式，只显示为 metric module；
- 输出为 structured innovation evaluation，而不是完整 peer review。

### 8d Dual-generation and fusion module

推荐图形：两个输出合流。

两类输入：

```text
Agent output:
  novelty / graph perturbation / prior-art evidence / risk flags

ASPR-Qwen output:
  human-like review language / reviewer concerns / significance / limitations
```

融合输出：

```text
structured review draft
novelty assessment
strengths and weaknesses
evidence-grounded comments
reviewer-style recommendation
```

画法重点：

- 不能让 agent 或 Qwen 单独成为最终答案；
- fusion module 要清楚体现 weighting / rubric / consistency check；
- 可画成 two-input one-output 的 aggregator。

### 8e Verifier, evidence alignment, and safety gates

推荐图形：校验门控流程。

检查项：

```text
claim-evidence alignment
prior-art contradiction
unsupported novelty claim
hallucination / factuality check
metric disagreement
low-confidence flag
human-in-the-loop review
```

画法重点：

- verifier 在 final report 前；
- 对不合格输出有 revise / human review 回路；
- 这一步可以接入 Fig. 9 和 Fig. 10 的证据追踪与消融实验。

### 8f Final human-like review report schema

推荐图形：最终报告结构卡片。

字段：

```text
Summary
Novelty and significance
Graph-perturbation evidence
Comparison with prior art
Strengths
Limitations
Reviewer concerns
Recommendation / confidence
Evidence links
```

画法重点：

- 输出是 human-like review，不只是 innovation score；
- 每个部分标明来自 agent、ASPR-Qwen 或 fusion；
- 保留 evidence links 和 confidence。

## 版式建议

推荐布局：

```text
a  a
b  c
d  e
f  f
```

其中 8a 是总框架，8b 是垂直模型训练，8c 是 agent evidence engine，8d 是融合，8e 是校验，8f 是最终输出。

## 给 gpt-image-2 的输入准备

建议准备：

```text
fig8_panel_text.json       # 每个 panel 的固定文字
fig8_flow_spec.json        # 节点、箭头、lane、颜色
fig8_module_inventory.csv  # 模块名和功能
fig8_image2_prompt.md      # 图形风格和不可改写约束
```

提示词必须强调：

```text
Do not invent module names.
Preserve all arrows and labels.
Keep two-lane architecture: ASPR graph agent and ASPR-Qwen reviewer.
Final method is fusion of both branches.
```

## 解释边界

可以说：

```text
ASPR combines graph-grounded agentic evaluation with ASPR-Qwen, a Nature peer-review SFT model, to generate human-like reviews.
```

不要说：

```text
ASPR is only an LLM reviewer.
ASPR is only a graph metric system.
ASPR replaces human peer reviewers.
```

# Fig. 10 绘制思路指导

## 一句话结论

Fig. 10 的任务是给出 ASPR 模块组合贡献的 pipeline-ready 消融证据，而不是把当前版本写成已完成的因果模块重跑。当前 full ASPR 来自真实 Fig.4 评估；若 ASPR-Qwen checkpoint、真实模块重跑或人工偏好评分暂缺，必须显式标注 LLM-as-judge / pipeline-ready estimate。

推荐标题：

```text
Fig. 10 | Pipeline-ready ablation and reinforcement of ASPR agent-model modules
```

## 核心绘图模式

Fig. 10 是混合模式，但主模式仍是：

```text
跑 ablation 数据 -> Python 画核心统计图
```

总体策略已确认：

```text
主图优先使用真实 ablation / evaluation 数据。
如果 ASPR-Qwen checkpoint 暂不可用，允许先使用假设的 ASPR-Qwen 输出结果绘制 pipeline-ready version。
最终正式图应替换为真实 ASPR-Qwen checkpoint、真实模块重跑和盲评人工偏好；在替换前，caption、CSV 和 panel_text 必须明确标注 LLM-as-judge / assumed-output 层。
```

辅助模式：

```text
轻量错误案例和模块说明 -> gpt-image-2 绘制机制解释卡片
```

原因：

- ablation performance、human preference、error rate、coverage、factuality 等必须由 Python 精确画图；
- module map、degradation example、error taxonomy card 可以用 `gpt-image-2` 做出版级视觉整理；
- 最终图建议由 Python panel + image2 explanation panel 组合，但所有数字仍以 CSV 为准。

推荐技术路线：

```text
ablation_results.csv
human_preference.csv
error_taxonomy.csv
-> Python forest plot / paired preference bar / Pareto
-> panel_text.json for examples
-> gpt-image-2 for module and error cards
```

## 这张图想回答什么

Fig. 10 回答：

```text
ASPR 中哪些模块真的有贡献？
去掉 graph agent、ASPR-Qwen、retrieval、trace、fusion 或 verifier 后会怎样？
增强这些模块后是否提升输出质量？
```

评估方式已确认：

```text
主图使用自动指标 + 少量 human preference。
如果暂无人工评分，先用 LLM-as-judge，并在图注或 audit 表中标注。
```

自动指标建议：

```text
semantic agreement with peer review
novelty coverage
prior-art accuracy
factuality
readability
unsupported claim rate
evidence trace completeness
human-like review structure coverage
```

允许的数据来源：

```text
本地 Nature markdown / peer-review 语料：
  Windows: D:\aspr_nature_markdown
  WSL/Linux: /mnt/d/aspr_nature_markdown
本地 ASPR agent 输出、Fig.9 真实运行实例和 Fig.1-Fig.7 缓存
必要时联网补齐 Nature article / Transparent Peer Review metadata
```

统一配色：

```text
Nature / training corpus: 深红 / 酒红
ASPR graph agent: 蓝色
ASPR-Qwen: 紫色
evidence / verifier / uncertainty: 橙色
full ASPR / fusion: 深灰或黑色
```

它与 Fig. 8/Fig. 9 的关系：

- Fig. 8 画 ASPR 算法框架；
- Fig. 9 展示一次完整运行；
- Fig. 10 用实验说明每个关键模块为何必要。

## 实验版本

建议 ablation variants：

```text
full ASPR
- graph-perturbation agent
- ASPR-Qwen reviewer
- prior-art retrieval
- citation graph retrieval
- seven-indicator computation
- evidence trace
- fusion module
- verifier / self-check
- structured review rubric
generic LLM-only baseline
```

建议 reinforcement variants：

```text
+ larger Nature peer-review corpus
+ domain-specific retriever
+ reviewer-style examples
+ graph evidence chain
+ self-consistency voting
+ expert feedback loop
+ stronger verifier
```

## 推荐输出

```text
outputs/kg_perturbation_fig10/
  fig10_module_inventory.csv
  fig10_ablation_performance.csv
  fig10_quality_degradation_examples.csv
  fig10_reinforcement_results.csv
  fig10_human_preference.csv
  fig10_error_taxonomy.csv
  fig10_panel_text.json
  fig10_panel_a.png
  fig10_panel_b.png
  fig10_panel_c.png
  fig10_panel_d.png
  fig10_panel_e.png
  fig10_panel_f.png
  fig10_full.png
```

## Panel 设计

### 10a ASPR module map

推荐图形：模块图，最好与 Fig. 8 保持同一套颜色。

模块：

```text
paper parsing
prior-art retrieval
citation graph retrieval
seven-indicator computation
graph-perturbation agent
ASPR-Qwen reviewer
fusion module
evidence trace
self-check / verifier
human-in-loop flag
```

画法重点：

- 每个模块带 ablation switch；
- full ASPR 全部打开；
- 后续 panel 的 ablation 名称必须和这里一一对应。

### 10b Ablation performance

推荐图形：forest plot 或 paired delta plot。

指标：

```text
semantic agreement with peer review
novelty coverage
prior-art accuracy
factuality
readability
unsupported claim rate
human preference
```

画法重点：

- y 轴是 ablation variant；
- x 轴是相对 full ASPR 的性能变化；
- 0 为参考线；
- 去掉 graph agent、ASPR-Qwen、retrieval、fusion、verifier 的下降应分别可见；
- generic LLM-only baseline 放在最底部。

### 10c Output quality degradation

推荐图形：degradation matrix + 代表性短句。

示例：

| Ablation | 典型问题 |
| --- | --- |
| no graph agent | 创新性判断空泛，缺少结构证据 |
| no ASPR-Qwen | 评审语言不像人类 reviewer，关注点松散 |
| no prior-art retrieval | 误判 novelty |
| no evidence trace | 不可审查 |
| no fusion | agent 和 Qwen 输出各说各话 |
| no verifier | hallucination 和 unsupported claim 增加 |

画法重点：

- 行是 ablation；
- 列是错误类型；
- 颜色表示频率或严重程度；
- 右侧放短 quote，不放长段落。

### 10d Reinforcement experiment

推荐图形：incremental improvement plot。

增强项：

```text
+ more Nature peer-review pairs
+ domain-specific retriever
+ graph evidence chain
+ reviewer-style examples
+ self-consistency voting
+ expert feedback loop
```

画法重点：

- x 轴或 y 轴显示 quality gain；
- 点大小可表示 runtime / token cost；
- 标出质量提升和成本提升；
- 不隐藏没有明显提升的增强项。

### 10e Human preference study

推荐图形：paired preference bars。

比较：

```text
full ASPR vs generic LLM-only
full ASPR vs no-graph-agent
full ASPR vs no-ASPR-Qwen
full ASPR vs no-fusion
full ASPR vs human review summary
```

评价问题：

```text
which is more useful?
which is more evidence-based?
which better identifies novelty?
which better identifies limitations?
which sounds more like a human reviewer?
```

画法重点：

- 每个问题一条 100% 堆叠条；
- 显示 full ASPR win / tie / comparator win；
- 标注 evaluator count、blind setting 和 sample size。

### 10f Error taxonomy and safeguard mapping

推荐图形：Pareto chart + safeguard mapping。

错误类型：

```text
overclaim novelty
missed prior art
wrong mechanism interpretation
over-reliance on graph score
weak field context
unsupported evidence
non-human-like review tone
fusion inconsistency
```

画法重点：

- 按错误频率排序；
- 比较 full ASPR 与 generic LLM-only 或 no-verifier；
- 右侧显示哪个模块负责缓解该错误；
- 呼应 Fig. 6 的 failure modes，但这里对象是 agent-model system。

## 版式建议

推荐：

```text
a  b  b
c  d  e
f  f  f
```

10b 是主结果，10f 是系统失败归因，适合横跨多列。

## 解释边界

可以说：

```text
ASPR quality degrades when graph evidence, ASPR-Qwen, retrieval, fusion, or verification modules are removed.
```

不要说：

```text
ASPR replaces peer review.
Every module is universally necessary for every paper.
```

Fig. 10 的核心结论应是“模块组合贡献”，不是“LLM 自己足够强”。

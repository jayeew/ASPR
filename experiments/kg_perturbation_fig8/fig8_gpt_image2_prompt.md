# Fig. 8 gpt-image-2 中文提示词

## 主提示词

```text
请绘制一张 Nature / Science 投稿级别的四 panel 主算法机制图。图像主题是：
ASPR: evidence-grounded multi-agent academic review.

总体结构：
使用 4 个 panel labels: a, b, c, d。
panel labels 要小、克制、左上角对齐。
不要放大标题，左上角只放一个很小的图号标签：
"Fig. 8 | ASPR framework"

整张图必须像 Nature Methods / Nature Machine Intelligence 的主算法框架图：
低饱和、大量留白、细线、清楚机制、可缩小阅读。
不要画成软件流程图、系统架构图、运营看板或信息海报。

核心读图目标：
读者应先从 panel a 在 5-10 秒内理解 ASPR 主流程：
Input manuscript -> Reference-graph calibration -> Claim cards -> ASPR reflection-guided review search -> Evidence-linked review.

然后从三个机制 panel 理解：
b. reference graph 如何通过七维指标和加权求和形成 S_w prior / confidence / limitations；
c. reviewer committee 和 ASPR reflection-guided review search 如何合并为一个受 claim cards 约束的候选评审生成机制；
d. Nature transparent peer review 数据如何训练 ASPR-Qwen，并作为 reviewer-style critique 侧支进入 panel c。

可见文字规则：
只渲染短标签，不要把说明性段落画进图中。
如果空间不足，优先保留 panel a 主路径、panel b 的 metric bars -> weighted sum -> S_w prior、panel c 的 Claim-Evidence-Graph-Counterclaim 和 reflect + score、panel d 的 Nature review data -> ASPR-Qwen。

允许出现的主要标签：
"Input manuscript"
"Reference-graph calibration"
"Claim cards"
"ASPR reflection-guided review search"
"Evidence-linked review"
"G0 prior-art graph"
"G+ target inserted"
"metric bars"
"weighted sum"
"S_w prior"
"confidence"
"limitations"
"B"
"RS"
"DeltaQ0"
"Uzzi"
"RTD"
"BurtIP"
"PDE"
"Claim-Evidence-Graph-Counterclaim"
"Claim Decomposer"
"Evidence Mapper"
"Graph Analyst"
"Skeptic Reviewer"
"Meta Reviewer"
"draft"
"revise"
"reflect + score"
"best calibrated draft"
"takes claim cards as constraints"
"ASPR-Qwen"
"Nature review data"
"Transparent peer review"
"SFT records"
"reviewer-style critique"
"calibrated novelty stance"
"citations"

整体版式：
横向 16:10。
推荐布局：
- panel a 占整图上方 42-50%，是最大主框架。
- panel b 位于左下，占下方约 34-38% 宽度。
- panel c 位于中下到右下，占下方最大区域，作为 agent + search 合并机制区。
- panel d 是右下角或 panel c 旁边的紧凑 inset，使用 Nature data 红紫色，作为稳定记忆点。

Panel a: ASPR overview
panel a 是主流程框架，只显示整体路径，不展开复杂细节。
从左到右画：
"Input manuscript"
-> "Reference-graph calibration"
-> "Claim cards"
-> "ASPR reflection-guided review search"
-> "Evidence-linked review"

视觉要求：
- "Reference-graph calibration" 只画成简洁小图谱图标和 "S_w prior / confidence / limitations" 三个小 badge。
- "Claim cards" 只画 2-3 张小卡片。
- "ASPR reflection-guided review search" 只画简洁候选分支图标。
- "Evidence-linked review" 画成 manuscript-style review card。
- panel a 只保留 4 条主箭头，线条干净，不要交叉。
- panel a 不要塞入五个 agent 名称、七维指标、数据训练流程；这些放到 b/c/d。

Panel b: graph-to-S_w calibration
panel b 是图谱校准算法机制，必须比装饰性图谱更像算法。

左侧画两张小图谱：
1. "G0 prior-art graph"
只包含 prior-art nodes 和 2-3 个 field / community clusters，没有 target manuscript。

2. "G+ target inserted"
加入 target manuscript node，显示跨社区连接、弱连接和 bridge edges。
用少量深蓝和琥珀色突出 boundary perturbation 与 atypical recombination。

右侧画一个清晰的算法化计算链：
"metric bars"
七个短小水平指标条依次标注：
"B"
"RS"
"DeltaQ0"
"Uzzi"
"RTD"
"BurtIP"
"PDE"

这些指标条汇入一个小节点：
"weighted sum"

再输出到一个小卡片：
"S_w prior"
"confidence"
"limitations"

从 "S_w prior" 发出两条很细 calibration signal：
- 到 panel c 的 "Graph Analyst" / "Meta Reviewer"
- 到 panel c 的 "reflect + score"

重要：
S_w prior 是 calibration prior，不是论文质量分数、排名或最终判断。
不要画性能柱状图、排行榜或基准数字。

Panel c: constrained reviewer committee + review search
panel c 是最大机制展开区，合并 reviewer committee 和 ASPR reflection-guided review search。
它必须清楚表达：review search takes claim cards as constraints。

左半部分画 reviewer committee：
中央画 claim board，标题：
"Claim-Evidence-Graph-Counterclaim"

claim board 内有 3 张小 claim card，每张只显示：
"claim"
"evidence"
"counterclaim"
"uncertainty"

围绕 claim board 画五个小型 role lens，不画人物头像、不画机器人：
"Claim Decomposer"
"Evidence Mapper"
"Graph Analyst"
"Skeptic Reviewer"
"Meta Reviewer"

用细线和微小编号展示职责链：
"1 decompose"
"2 map evidence"
"3 add graph prior"
"4 challenge claims"
"5 set tone"

"Graph Analyst" 接收来自 panel b 的 S_w prior / confidence / limitations。
"Meta Reviewer" 旁边可以有两个很小标签：
"disagreement"
"recommended tone"

右半部分画 ASPR 自有候选评审搜索机制，标题：
"ASPR reflection-guided review search"

在 claim board 和 search tree 之间放一条明确但细的约束线，标注：
"takes claim cards as constraints"

搜索机制画成小型候选评审搜索树：
"draft"
-> 2-3 个 "revise" candidates
-> "reflect + score"
-> "best calibrated draft"

"reflect + score" 旁边放五个极小 tick labels：
"novelty accuracy"
"prior-art contrast"
"citation"
"graph alignment"
"uncertainty"

进入 search 的输入只保留三类细线：
- claim cards constraint；
- S_w prior / confidence / limitations；
- panel d 的 ASPR-Qwen reviewer-style critique。

质量控制必须体现在 "reflect + score"，不要画独立质量检查闸门。
不要出现任何外部算法名称或借用式算法标签。

Panel d: Nature-trained ASPR-Qwen inset
panel d 是数据贡献和模型侧支，放在右下角或 panel c 旁边。
它要小而清楚，但必须让人看出 Nature transparent peer review 数据训练也是贡献。
使用 Nature data 红紫色作为稳定记忆点。

画一个紧凑 inset，标题：
"Nature review data"

包含四个节点：
"Nature articles"
"Transparent peer review"
"parsing + alignment"
"SFT records"

短箭头：
"SFT records -> ASPR-Qwen"

ASPR-Qwen 节点旁边小字：
"Nature-trained reviewer model"

从 ASPR-Qwen 只画一条细紫色线进入 panel c 的 search tree，标注：
"reviewer-style critique"

不要让 ASPR-Qwen 直接连接最终 review。
不要把 Nature review data 画成底部贯穿主流程的大型长条带。

Final output:
在 panel a 右侧的 "Evidence-linked review" card 中只保留：
"calibrated novelty stance"
"claim-level evidence"
"prior-art comparison"
"limitations"
"recommendation"
"citations"

从 "citations" 用极细 citation links 回连到 panel c 的 claim cards 和 panel b 的 prior-art graph evidence。
这些 citation links 是证据锚点，不是反馈流程，不要画大箭头。

连线规则：
- panel a 只保留主流程箭头。
- panel b -> panel c 只画两条很细 calibration signal。
- panel d -> panel c 只画一条细紫色 critique input。
- final citations -> panel b/c 只画极细 citation anchoring。
- 避免交叉箭头、长距离红色虚线、密集连线。
- 不要让任何 panel 看起来像软件服务模块。

配色：
背景 #FFFFFF 或 #FAFAF7。
Panel a 主流程：中性灰 + evidence blue。
Reference graph: #2F6F9F, #5A9C9A, #A9C7D1。
Metric bars / S_w prior: #1F4E6B with #EAF3F5。
Reviewer committee / claim cards: #D6E8EA, #F8FAFC, #2F6F9F。
ASPR reflection-guided search: #C97A3D with #F7EEE9。
ASPR-Qwen: #7A68A6 with #EFEAF7。
Nature review data inset: #8A2D3B with #F7EEF1。
Final review: #22272E with #F8FAFC。
Weak lines: #D6D9DE and #6B7280。

颜色比例：
70% white / light neutral,
15% graph blue-teal,
6% amber reflection,
5% Qwen purple,
4% Nature data red.

必须避免：
不要画半屏大图谱。
不要把 panel a 画得和小 panels 一样碎；panel a 必须是主概览。
不要让 b/c/d 变成三张独立软件流程图；它们是主图的机制展开。
不要出现任何外部算法名称、借用式算法标签或通用推理框架名称。
不要画独立质量检查闸门。
不要让 ASPR-Qwen 单独输出 final review。
不要把 Nature review data 画成底部主流程。
不要画聊天界面、机器人、服务器、接口网关或数据库机柜。
不要画性能条形图、曲线图、基准数字、排行榜或优越性结论。
不要让 S_w prior 看起来像论文质量最终标签。

最终视觉重点：
panel a 讲清 ASPR 主流程；
panel b 解释 G0/G+ graph -> metric bars -> weighted sum -> S_w prior / confidence / limitations；
panel c 解释 claim cards 如何约束 reviewer committee + ASPR reflection-guided review search；
panel d 解释 Nature transparent peer review data 训练 ASPR-Qwen，并只作为 reviewer-style critique 侧支；
最终 review 是 evidence-linked、uncertainty-calibrated academic review。
```

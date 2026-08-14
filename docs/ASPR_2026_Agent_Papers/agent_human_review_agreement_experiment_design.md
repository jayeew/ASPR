# 人工审稿–Agent 审稿一致性：前沿方法、推荐实验设计与 GEAR 诊断

更新时间：2026-08-11

## 一句话结论

当前 Nature dev100 的结果不是“agent 与人工审稿意见严重不一致”的有效能力测量，而是一次**可用性为零的 fail-closed 部署测量**：100/100 agent run 都进入 `limited`，没有一篇产生可被接受的原子审稿点评。因此，任何看似非零的一致性指标都主要由“双方均无点评”的记分约定产生，不能解释为 agent 识别了人工意见。

在真正比较审稿能力前，必须先将“系统是否产生通过验证的 review”与“产生后与人工是否一致”拆成两个 estimand（被估计对象）。

---

## 1. 前沿研究采用什么一致性比较方法

| 工作 | 参考对象与单位 | 主要比较维度 | 人工/自动验证设计 | 对 ASPR 的可迁移结论 |
|---|---|---|---|---|
| [Beyond Not Novel Enough](papers/01_Beyond_Not_Novel_Enough_EACL2026.pdf) | 182 篇 ICLR 2025 论文、352 份人类 review；聚焦 novelty | 结论一致、推理一致、prior-work engagement、分析深度、sentiment shift | 先独立抽取人类 reference judgment，再作 LLM judge；3 位 PhD 进行盲配对人工评价；75 个重叠比较报告 raw agreement 与 Cohen kappa | 将“结论是否相同”与“理由、证据、深度是否相同”分开；必须有人–人基线和对 LLM judge 的人工校准。 |
| [ReviewGrounder / ReviewBench](papers/02_ReviewGrounder_ACL2026.pdf) | 约 1.3K 篇 ICLR 2024–25，要求每篇至少 3 个完整人类 review | 8 个 rubric：贡献准确、结果解释、比较分析、基于证据的批评、清晰度、覆盖度、建设性语气、虚假/矛盾声明 | 聚合人类 reference；生成端看不到 paper-specific rubrics；固定 evaluator；120 篇专家人工评分与自动分数比较 Pearson/Spearman/MAE | 评分/录用预测不能替代 substantive review；用 paper-specific、可验证的 rubric，并严格隔离生成与评估信息。 |
| [PRAIB](https://arxiv.org/abs/2605.29815) | 1,000 篇 ICLR/NeurIPS 论文、11,000 份由 5 个模型生成的 review，对照原始人类反馈 | 原子 weakness 覆盖、specificity、风格、参与行为、rating/置信度分布、交叉引用模式 | 多模型、多 prompt，对原始人类反馈做行为分布对照 | 不只报 F1；同时报模型是否过度正向、过度自信、低方差，以及是否漏掉人类原子弱点。 |
| [ReviewEval](https://aclanthology.org/2025.findings-emnlp.1120/) | AI review 与人类 assessment | 人类对齐、事实准确性、分析深度、建设性、审稿指南遵从 | 多维 review-quality framework，并用于外部改进循环 | 需要 quality/grounding 轴，而非只与 reference 文本重叠。 |
| [PeerCheck](https://aclanthology.org/2026.findings-acl.1170/) | GPT-4o、Claude-3.7、DeepSeek-V3 review 对人类 review | 主题/术语焦点、人类相似性、prompt/RAG 影响 | 比较不同模型与 CoT、RAG 设置；发现 RAG 效果依模型而异，甚至降质 | 需要 component ablation；不能假设“加 RAG”一定提升人类对齐。 |
| [Dycke & Gurevych, TACL 2026](https://aclanthology.org/2026.tacl-1.22/) | 成对的原论文与逻辑错误反事实版本 | 对研究逻辑缺陷的敏感性 | 控制反事实测试，而非只对历史人类 review 做相关性比较 | 必须补充 capability test：故意植入/修正某类可验证缺陷，检验 agent review 是否随之改变。 |

### 1.1 两个最值得复用的本地设计细节

**Beyond Not Novel Enough** 的评价不是直接算全文相似度。它把 AI 和人工评估拆成：(a) novelty reasoning alignment，(b) novelty decision alignment，(c) claim substantiation，(d) analytical quality；人工盲评为 A/B/Tie/Unclear。其 75 个重叠人工比较的 kappa 仅约 0.287–0.368，说明 novelty 本身存在明显主观性。故其 86.5% “reasoning alignment”不能脱离人工一致性下限来解释。

**ReviewBench** 不把 human review 当成单一的文本参考答案。它以多名人类 review 聚合为 reference，并把 venue guideline、人类关切和论文事实转换为 paper-specific rubric；生成模型看不到该 rubric。其人工验证不是“专家更喜欢哪篇文本”而是检验 rubric evaluator 是否与专家评分一致（120 篇，Pearson 0.8954、Spearman 0.7923、MAE 0.0969）。

---

## 2. 推荐的 ASPR 一致性实验设计

### 2.1 先定义三层目标，而非用一个总分

| 层级 | 要回答的问题 | 分母 | 不能与之混淆的量 |
|---|---|---|---|
| A. 运行可用性（availability） | 在固定输入与预算下，agent 是否产生结构合格、证据可验证的 review？ | 所有投喂论文 | 不能只在成功输出上报告质量。 |
| B. 条件审稿能力（quality conditional on availability） | 对通过 A 的输出，agent 是否发现了人类关切、且判断与证据正确？ | A 通过的论文 | 不要把 limited/空输出记作“质量差但可评分”的普通 review。 |
| C. 系统效用（end-to-end utility） | 人类是否认为该 review 有帮助、节省时间、不会引入实质风险？ | 所有论文或实际使用者 | 不等于与一份历史 review 的文本相似。 |

主报告应同时给出 A、B、C；B 的分母为零时必须写 `not estimable`，而不是生成一个看似精确的 F1。

### 2.2 数据与时间线

1. **冻结论文版本。** 对每篇论文固定 submission version、review period 和 agent 可见的信息。若目标是模拟原始审稿，agent 只能看到投稿版本及当时可获得的外部资料。
2. **保留多名原始 reviewer。** 每篇至少 3 个完整独立 review；同时保存 individual review 和一个独立构建的 consensus reference。不要只拿仲裁后的单一文本当“绝对真值”。
3. **将 rebuttal/final revision 作为独立任务。** 它们可以用于“revision-aware audit”或“issue resolution”评估，但不能与 submission-time reviewer-agreement 混在同一个主指标中。否则目标包含人类审稿当时不可能知道的后验信息。
4. **数据划分。** 以论文为簇做 train/dev/test，按 venue、年份、领域、决策、review 数量、论文长度分层；dev 不得用于提示词或阈值选择后的最终汇报。建议至少保留 200 篇盲测，或明确 100 篇仅为开发集。
5. **防泄漏。** agent 生成时不可见 human review、reference rubric、未来 rebuttal 或最终编辑决定；评估者不可见系统身份。

### 2.3 标注与 reference 构建

对每篇论文建立一个 point ledger，字段至少包括：

- `issue_id`、aspect（novelty/method/experiment/claim/clarity/ethics 等）、direction、severity；
- 原始 reviewer quote、投稿版本中的证据 span、需要外部验证的文献证据；
- 是否被第二名标注者独立确认；
- individual reviewer 支持数，而不是只保存 consensus；
- 若做 revision-aware 任务，再单列 resolution status 和最终稿证据。

推荐两个独立领域专家先盲标，第三人仅裁决冲突。报告 Cohen kappa（两人）、Krippendorff alpha（多标注者）或 Gwet AC1（类别极不平衡时），并同时报告 raw agreement。不能只报告裁决后的“gold”。

### 2.4 指标面板

| 维度 | 推荐主指标 | 必须附带的诊断 |
|---|---|---|
| 运行可用性 | valid structured-review rate；semantic-verification rate；成本/时延 | failure taxonomy、fallback rate、按领域/长度分层的失败率 |
| 原子关切 | 严格一对一 semantic match 的 precision、recall、F1 | 不同 severity/aspect 的 recall；partial/contradictory；空集单独列示 |
| 覆盖 | 人类 weakness/major issue recall；rubric coverage | 每篇 coverage；漏检 top-20 审计 |
| 判断 | novelty/decision 的 macro-F1、balanced accuracy、weighted kappa | 混淆矩阵、positive/negative bias、置信度校准 |
| 评分 | MAE、Spearman、ICC | 与 human–human score correlation 的差距 |
| 事实与证据 | citation/evidence precision、entailment support rate、unsupported-major rate | 无效 span、错误引用、被反事实击穿的比例 |
| 可用性 | 盲评专家 pairwise preference、Likert usefulness/actionability | 评审间一致性、花费时间、原因码 |
| 行为分布 | 长度、具体性、术语焦点、rating 方差、置信度 | 与人类分布的距离，而非只比较均值 |
| 因果能力 | counterfactual sensitivity / minimal-pair accuracy | 对植入逻辑缺陷、夸大主张、遗漏对照的反应差异 |

**空集规则：** `reference_points=0 & candidate_points=0` 不得与真实匹配混合汇总。应单列 `both_empty_rate`，并在“reference 非空”子集重新报 precision/recall/F1。当前 dev100 的 0.06 atomic F1 正是因为 6 个双空样本被标准 F1 记作 1。

### 2.5 匹配与评估流程

```text
冻结投稿版本/外部检索截止日
        │
        ├── 人类原始 reviews ──> 双人原子标注 ──> individual + consensus ledger
        │                                      │
论文 ──> agent review ──> 结构/证据验证 ────────┼──> 盲化 point matching
        │                                      │       (人类优先；LLM judge 需人工校准)
        └── 对照/反事实版本 ──> sensitivity ────┘
                                               │
                           availability / agreement / quality / safety / utility
```

Point matching 可用候选生成（embedding 或 lexical retrieval）降低成本，但最终匹配应为 blinded 的 SAME / PARTIAL / CONTRADICTORY / NO-MATCH，并强制一对一匹配。LLM judge 可以做第一轮，但须在预注册的人工子样本上校准：报告其与专家的 correlation、误差和对系统排序的一致性；不能既用同一模型生成 reference 又独自宣布评估有效。

### 2.6 样本量、不确定性和显著性

- 以**论文**而非 point 为 bootstrap 簇（至少 5,000 次），因为同一论文内 points 高度相关。
- 报告总体、领域、年份、venue、长度、human disagreement、agent availability、reference point count 的分层结果；小于预设最小 n 的组只列探索性描述。
- 所有模型在同一论文、相同检索截止日和相同预算下作 paired comparison；用 paired bootstrap CI 或 permutation test 比较系统。
- 对人工–人工协议给区间；agent 的目标不必超过人类一致性，但必须说明相对 human–human baseline 的位置。
- 预先声明主指标（建议：availability、non-empty reference atomic recall、unsupported-major rate、blind expert usefulness）和停止条件，避免从许多指标中挑最好看的一个。

### 2.7 推荐的两阶段执行计划

| 阶段 | 样本与工作 | 放行条件 |
|---|---|---|
| 0. 运行资格 | 30 篇跨领域烟雾测试；逐 run 保存模型、prompt、检索、证据、validator log | structured-output valid rate 和 semantic support rate 达到预设阈值；否则不启动一致性评测 |
| 1. 标注 pilot | 50 篇、每篇至少 3 份人类 review；双人原子标注 + 盲评 | 人–人协议与 rubric 可操作性达标；修订 codebook，不改 test |
| 2. 盲测 | 至少 200 篇留出论文，或明确 dev100 为非确认性结果 | 报 A/B/C 全面指标、分层 CI、paired baselines、误差审计 |
| 3. 能力压力测试 | 50–100 个 minimal pairs / counterfactuals | agent 对受控缺陷的反应显著强于无差异基线 |

---

## 3. 为什么 ASPR 当前结果“看起来如此差”：根本原因排序

### 根因 1（决定性）：没有可评估的 agent review

当前结果的直接事实是：100 个 run 全部 `limited`、critic source 全部 `unavailable`、candidate atomic points 为 0、semantic verifier 可用数为 0、performance-claim-allowed 为 0。这个条件下不存在“agent 对人类 point 的遗漏模式”可供估计；主要结论只能是**部署配置未提供合格 critic**。

因此 0.06 atomic precision/recall/F1 不是“模型只达到 6% 人类水平”。它来自 6 篇最终 human reference 也为空时的双空记分。相同地，0.31 major recall、0.24 novelty-point F1 与 0.24 novelty accuracy 都部分或全部来自无对应人类目标时的默认记分/恒定 `not_discussed` 输出，而非有效识别。

### 根因 2（实现缺陷）：fallback 文案触发 graph-semantic guard

每个 fallback summary 使用了 `ASPR-Qwen`。现有 `GRAPH_SEMANTIC_TERMS` 把 `ASPR` 视为禁止的图语义术语，导致全部 100 个 run 有 graph-semantic violation。这是**验证器与 fail-closed 文案之间的字符串冲突**，不是论文内容或人工 reference 的不一致。

它需要修复，但修复后也只能让 limited fallback 更准确地被描述；不能把零点评 fallback 当作实质审稿能力。

### 根因 3（目标错位）：你构造的是 revision-aware 最终审计参考，不是同步审稿参考

当前 human reference 经过 reviewer report、author response 和 final-paper span 的一次重建；resolved/unverifiable concern 被移入 ledger。这个设计对于“最终稿还有哪些未解决问题”非常严谨，但与“审稿人面对投稿稿时会提出什么”不是同一个任务。

若 agent 只读投稿稿，reference 包含未来 author response/final revision，会产生后验信息错位；若 agent 也读最终稿，则它是在做 post-revision audit，而不是复现原始 reviewer。应将两者拆为 submission-time peer-review agreement 与 revision-aware audit agreement，分别评估。

### 根因 4（协议风险）：单一重建 reference 掩盖人工分歧

重建后的 review 将多 reviewer 的分歧压缩成一个答案。前沿工作会额外报告 human–human agreement，并保留各 reviewer 的独立 point 集。没有这一基线时，不能判断 agent 与共识的差距是模型失败、人工本来就分歧，还是重建策略改变了目标。

### 根因 5（指标风险）：空集与单一 F1 使结果难读

当前已正确保留 availability=0 与 conditional metric=null，但系统级 F1 仍受空集规则影响。应把：

- `availability`；
- `both_empty`；
- `reference_nonempty` 的 recall/F1；
- `available_agent_outputs` 子集；
- 盲评质量与 evidence safety；

拆开报告。这样既不会掩盖部署失败，也不会误把 fail-closed 行为解释为审稿能力。

### 根因 6（外部有效性有限）：这是单一开发集，且期刊/学科不均衡

dev100 是非确认性开发集；其中 93 篇为 `s41467`，其余期刊每个只有 1 篇。该数据适合 debug 与协议开发，但不适合声称跨学科、跨 venue 的稳定 agent–human agreement。

---

## 4. 对当前结果的正确表述

可使用：

> 在 Nature dev100 的冻结部署配置下，GEAR v2 对 100/100 篇论文进入 fail-closed limited 状态，未产生可供点级一致性评估的合格 agent review。我们据此报告系统可用性为 0，并将条件于合格输出的审稿一致性标记为不可估计；不将空输出导致的表观 F1 解释为审稿能力。

不应使用：

> GEAR 与人工审稿一致性为 6%。

后者把 availability failure、空集记分和 substantive agreement 混为一个数字。

---

## 5. 现有 ASPR 产物与下一次评测的最短路径

现有可复用资产：

- 100 篇重建后的人工 reference、470 个原子点、595 条 resolution ledger；
- reviewer quote 与论文 span trace；
- 100 行样本级指标、bootstrap、分层结果、20 个低一致性案例；
- `limited`、validator issue、critic source 的逐 run 审计。

最短路径不是重做人工 reference，而是：

1. 使 trained critic 或替代 critic 可真正产生经过 schema/semantic verification 的非空 review；
2. 修复 fallback 文案–graph guard 冲突，并把 fallback 归入 availability failure；
3. 以当前 reference 先做一个明确标为 **revision-aware audit** 的 conditional-on-availability 评测；
4. 另建不看 rebuttal/final revision 的 submission-time test split，并保留 individual human reviewer sets；
5. 在两套任务上加入 rubric、盲评、human–human baseline 和 counterfactual sensitivity。

---

## 参考文献

1. Afzal, Nakov, Hope, and Gurevych. 2025/2026. [Beyond “Not Novel Enough”: Enriching Scholarly Critique with LLM-Assisted Feedback](https://arxiv.org/abs/2508.10795). 本地副本：`papers/01_Beyond_Not_Novel_Enough_EACL2026.pdf`。
2. Li et al. 2026. [ReviewGrounder: Improving Review Substantiveness with Rubric-Guided, Tool-Integrated Agents](papers/02_ReviewGrounder_ACL2026.pdf).
3. Żurawicki et al. 2026. [PRAIB: Peer Review AI Benchmark of Behaviour of LLM-Assisted Reviewing](https://arxiv.org/abs/2605.29815).
4. Garg et al. 2025. [ReviewEval: An Evaluation Framework for AI-Generated Reviews](https://aclanthology.org/2025.findings-emnlp.1120/).
5. Chen et al. 2026. [PeerCheck: Enhancing LLM-Generated Academic Reviews Towards Human-Level Quality](https://aclanthology.org/2026.findings-acl.1170/).
6. Dycke and Gurevych. 2026. [Automatic Reviewers Fail to Detect Faulty Reasoning in Research Papers: A New Counterfactual Evaluation Framework](https://aclanthology.org/2026.tacl-1.22/).
7. Wu et al. 2026. [Can AI Be a Good Peer Reviewer? A Survey of Peer Review Process, Evaluation, and the Future](https://aclanthology.org/2026.acl-long.1504/).

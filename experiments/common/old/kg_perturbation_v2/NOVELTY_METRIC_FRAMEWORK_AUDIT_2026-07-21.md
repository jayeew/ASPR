# ASPR 发表前创新性指标体系：方法审计与重构建议

日期：2026-07-21

适用范围：Nature Multi-Horizon V1；仅使用目标论文参考文献、参考文献元数据和严格早于发表年份的图谱。

## 结论先行

现有“92 个候选指标 → 8 个核心指标 → 5 个机制 + 10 个辅助特征”的版本，不建议原样写入投稿稿件。问题不在于 8 个指标太少，而在于：

1. 92 个条目混合了预测指标、控制变量、未来结果和外部验证指标，不能被描述为一个同质的“创新性指标全集”；
2. 5 个类别目前被称为“机制”，但没有因果识别，且其中跨学科性、结构洞和社区多样性并不等于论文创新性；
3. 8 和 10 都是事先锁定的数量，尚无规则说明为什么必须是这个数；
4. 部分代码实现与经典指标并不等价，存在名称和计算内容不一致的问题；
5. OOF Spearman 只能说明对未来扩散目标的预测效度，不能单独证明这些指标真的测量了“创新性”。

建议把主线改为：

> 五个发表前测量家族 + 分层指标注册表 + 外部构念效度验证。

五个家族不是普适的“创新因果机制”，而是本文在严格 reference-only 条件下采用的五种不同测量方式：

1. 非典型—传统组合；
2. 参考文献语义远距组合；
3. 参考集合相对既有研究的差异；
4. 知识来源多样性与整合；
5. 新论文对既有共被引结构造成的结构变异。

核心指标数量不再预先锁死。按当前文献证据，建议先注册 **9 个可确认指标**；`centrality_divergence` 只有在可扩展计算和近似误差验证通过后，才升为第 10 个确认指标。其他指标保留为探索性 challenger，不用于证明五个家族的理论成立。

## 一、先把“创新性”与其他概念分开

本文需要至少区分四个概念：

| 概念 | 本文可观测内容 | 能否直接称为创新性 |
|---|---|---|
| 组合新颖性 | 参考文献是否形成罕见、远距或此前少见的组合 | 可以，但只能称“reference-based combinatorial novelty” |
| 知识整合 | 参考文献是否来自多种、均衡且差异较大的知识领域 | 不能自动等同于新颖性 |
| 结构变异潜力 | 新论文的参考关系是否跨越既有共被引社区并改变图结构 | 可称“transformative potential”，不宜直接称因果机制 |
| 未来扩散/影响 | 论文之后被多少领域采用、传播是否均衡、图谱是否变化 | 是预测目标或效标，不是发表前创新性输入 |

参考文献只能反映一篇论文如何定位和组合既有知识，不能完整观察论文的新理论、新方法、新材料或新结果。对突破性论文作者的访谈研究也显示，创意来源并不总能由参考文献还原。因此，论文主张应收窄为：

> We measure ex-ante, reference-based signals of combinatorial novelty, knowledge integration, and structural variation, and test whether they predict subsequent knowledge diffusion and graph perturbation.

不建议写：

> We directly measure the innovation of every paper at publication.

依据：Tahamtan & Bornmann 对 landmark papers 的作者访谈发现，创意可能来自实际问题、同事讨论和跨学科交流，参考文献更多承担知识定位功能，而不必然记录创意来源（[Journal of Informetrics, 2018](https://doi.org/10.1016/j.joi.2018.07.005)）。2025/2026 年的多学科验证也表明，单一文献计量指标不能覆盖所有新颖性类型，且指标表现随学科变化（[Shibayama et al., 2025/2026](https://doi.org/10.2139/ssrn.5379973)）。

## 二、92 个指标应该如何解释

### 2.1 不再声称“92 个是完整指标宇宙”

92 只能是某次检索和去重后得到的注册表行数，不能成为理论依据。更稳妥的表述是：

> The candidate registry is a dated, scoped evidence map rather than an exhaustive universe of novelty metrics.

也就是说，审稿人问“为什么不是 93 或 120 个”时，回答不是“92 已经很多”，而是：

- 候选池由预先公开的检索式、数据库、检索日期和纳入/排除规则生成；
- 新文献可以进入后续版本，但不会改变当前冻结分析；
- 同一公式的阈值、分位数或命名变体不重复计数；
- 每个条目先被分配角色，再决定是否有资格成为预测输入。

### 2.2 建议的候选池生成协议

在 Supplementary Methods 中预注册以下内容：

**检索来源**

- Web of Science、Scopus、PubMed；
- Crossref/OpenAlex 用于补充元数据；
- 对关键种子论文做前向和后向引文追踪。

**检索式示例**

```text
("scientific novelty" OR originality OR "transformative research"
 OR "combinatorial novelty")
AND (indicator OR measure* OR bibliometric* OR "citation network"
 OR "semantic distance")
```

```text
(interdisciplinarity OR "knowledge integration" OR "boundary spanning"
 OR "structural variation")
AND (paper OR publication)
AND (measure* OR indicator*)
```

**候选条目的单位**

一条候选记录必须对应一个可复现的数学量或算法。以下内容不单独计数：

- 同一指标的 p90、p95、p99 等超参数变体；
- 仅符号或方向相反的别名；
- 同一公式在不同数据库中的商品化名称；
- 仅用于显示的 z-score、percentile 等变换。

**筛选过程**

最好由两名研究者独立完成标题摘要筛选和全文筛选，并记录分歧处理；若资源有限，至少由第二人抽查 20% 记录。Supplement 应给出 PRISMA-style 流程图、排除原因和冻结日期。

### 2.3 每个候选必须先分角色

| 角色 | 例子 | 是否进入发表前预测模型 |
|---|---|---|
| 直接新颖性代理 | 非典型组合、语义远距组合、参考集合差异 | 可以 |
| 整合/结构前因 | variety、balance、disparity、结构变异 | 可以，但不得包装成直接新颖性 |
| 控制变量 | 参考文献数、年龄和先验流行度 | 可以，但不进入机制解释 |
| 未来结果 | RGPM-D、未来引用、CD/disruption、扩散熵 | 只能作标签或效标 |
| 外部验证 | 专家新颖性评分、F1000 标签、Nobel/APS milestones | 只用于验证 |
| 排除 | 使用发表后信息、无法复现、定义不清 | 不进入 |

原 92 表中的最大问题，是把这些角色混在同一个分母里，再画成 `92 → ... → 8` 的漏斗。新版本应先按角色拆表，再在“合格的发表前预测候选”内部进行选择。

## 三、建议采用的五个测量家族

### 家族 1：非典型—传统组合

理论问题：目标论文是否在总体传统的知识基础中，引入少量非常规组合？

推荐主指标：

- `journal_pair_atypicality_p10`：参考文献来源期刊对的 z-score 下尾；
- `journal_pair_conventionality_median`：同一分布的中位数，作为传统性伴随量。

Uzzi 等发现，高影响论文往往以传统组合为主体，同时包含少量非典型组合（[Science, 2013](https://doi.org/10.1126/science.1240474)）。Bornmann 等使用 F1000Prime 专家标签验证后，认为 Lee/Uzzi 路线的 U 指标大体具有收敛效度，但对 Wang 路线的 W 指标提出质疑（[Journal of Informetrics, 2019](https://doi.org/10.1016/j.joi.2019.100979)）。Fontana 等进一步指出，非典型组合可能与跨学科性重叠，首次组合指标也可能主要反映网络结构，因此不能把 U/W 当作无争议的真值（[Research Policy, 2020](https://doi.org/10.1016/j.respol.2020.104063)）。

实现要求：

- 主分析按“参考文献来源期刊对”计算，不能把 reference-ID pair 直接称为 Uzzi 指标；
- null model 要保持年份、参考文献表长度和期刊出现频率等关键边际分布；
- 当前解析期望值只能标为 analytic proxy，并需在抽样数据上与 permutation/configuration null 对照；
- 所有 null 统计只能使用发表前数据。

`conventionality_median` 不是直接新颖性指标，而是 Uzzi 组合画像的伴随量；论文中必须明确这一点。

### 家族 2：参考文献语义远距组合

理论问题：目标论文是否把语义上相距很远的既有工作组合在一起？

推荐主指标：

- `reference_semantic_distance_q100`：有效参考文献标题向量两两距离的最大值；
- q99 作为预注册的稳健性版本，不另算一个理论指标。

Shibayama 等以参考文献标题 embedding 的两两距离衡量组合新颖性，并用研究者自评的新理论、新现象、新方法和新材料维度做了效度验证；较高分位数尤其是 q100 表现较好（[PLOS ONE, 2021](https://doi.org/10.1371/journal.pone.0254034)；2026 年更正补回了 Table 4 缺失单元格，[Correction](https://doi.org/10.1371/journal.pone.0341474)）。

实现要求：

- 只使用参考文献标题，不使用目标论文全文；
- 主分析的向量模型必须有时间审计。最稳妥方案是在每个图谱快照内，用截止年前的参考文献标题训练 TF-IDF/SVD 或 fastText；
- 若使用在未来语料上训练的现代 contextual encoder，只能作为敏感性分析，并明确它是固定测量工具而非严格历史可用模型；
- q100 对错配记录敏感，必须有标题语言、长度、重复和 reference-link 质量门槛，并报告 q99 稳健性。

### 家族 3：参考集合相对既有研究的差异

理论问题：与发表前同领域论文相比，目标论文的完整参考集合是否明显不同？

推荐主指标：

- `prior_domain_overlap_novelty`：1 减去目标论文与发表前同知识域论文的平均参考集合 overlap。

Matsumoto 等提出以焦点论文和既有同域论文的引用相似度衡量组合新颖性，并在 1,871 篇自然科学论文上与研究者主观新颖性判断做了验证（[Scientometrics, 2021](https://doi.org/10.1007/s11192-021-04049-z)）。

实现要求：

- “同域论文”只能由发表前信息确定；
- overlap 的集合定义、候选邻域、均值/最近邻汇总方式必须锁定；
- 大规模计算可使用 MinHash/LSH 找候选，但必须在分层样本上与精确 Jaccard 结果比较误差；
- 不能用未来主题或未来引用关系确定同域。

### 家族 4：知识来源多样性与整合

理论问题：目标论文引用的知识来源是否种类多、分布均衡且彼此差异大？

推荐主指标：

- `field_variety`；
- `field_balance`；
- `field_disparity_cosine`。

Stirling 将 diversity 分为 variety、balance 和 disparity 三个不可互换的维度（[Journal of the Royal Society Interface, 2007](https://doi.org/10.1098/rsif.2007.0213)）。Porter、Rafols 及后续研究把这一框架用于知识整合和跨学科性，同时提醒不同指标刻画的是不同侧面（[Scientometrics, 2009](https://doi.org/10.1007/s11192-008-2197-2)；[Scientometrics, 2010](https://doi.org/10.1007/s11192-009-0041-y)）。

实现要求：

- 三个维度分别报告，不要先验等权合成后再把内部差异隐藏掉；
- disparity 使用发表前的领域引用画像 cosine distance 或经过验证的 science-map distance；
- 当前“领域对应社区集合的 Jaccard 距离”不是文献中的标准 disparity，应替换或明确改名；
- 这一家族衡量知识整合，不是直接的新颖性真值。

### 家族 5：结构变异与边界跨越

理论问题：把目标论文的参考关系加入发表前知识结构时，是否新建跨社区连接并改变结构？

推荐主指标：

- `sva_modularity_change_rate`；
- `sva_cluster_linkage`；
- `sva_centrality_divergence`，仅在可扩展近似通过验证后进入确认集。

Chen 的 Structural Variation Analysis 正是在论文刚发表、尚无后续引用时，用 modularity change、cluster linkage 和 centrality divergence 描述新论文对既有知识结构的改变，并测试其未来引用预测能力（[JASIST, 2012](https://doi.org/10.1002/asi.21694)）。这比项目当前自行定义单一 `delta_q0_shock` 有更清楚的来源。

实现要求：

- baseline 必须是发表前的共被引/共同引用知识图，而不是把参考文献 clique 边加入另一种语义的 citation-edge graph；
- 固定 baseline partition，再加入目标论文诱导的 reference-pair links；
- 明确有向/无向、加权/非加权、边采样和归一化方式；
- centrality divergence 若无法在 131,777 篇规模上可靠计算，应留在探索性子集，不能为了凑满 10 个指标使用未经验证的近似。

## 四、建议的新核心注册表

### 4.1 确认性指标

| 序号 | 指标 | 家族 | 角色 | 当前建议 |
|---:|---|---|---|---|
| 1 | `journal_pair_atypicality_p10` | 非典型—传统组合 | 直接新颖性代理 | 新实现，替代当前 reference-pair analytic proxy |
| 2 | `journal_pair_conventionality_median` | 非典型—传统组合 | 伴随传统性 | 保留，但不能单独称新颖性 |
| 3 | `reference_semantic_distance_q100` | 语义远距组合 | 直接新颖性代理 | 新增；q99 为稳健性 |
| 4 | `prior_domain_overlap_novelty` | 参考集合差异 | 直接新颖性代理 | 新增 |
| 5 | `field_variety` | 知识整合 | 整合前因 | 保留概念，使用易解释原尺度/折内变换 |
| 6 | `field_balance` | 知识整合 | 整合前因 | 由 `field_evenness` 规范命名 |
| 7 | `field_disparity_cosine` | 知识整合 | 整合前因 | 替换当前 Jaccard-community 实现 |
| 8 | `sva_modularity_change_rate` | 结构变异 | 转化潜力 | 按 SVA 重写 |
| 9 | `sva_cluster_linkage` | 结构变异 | 边界跨越潜力 | 新增，优先级高 |
| 10* | `sva_centrality_divergence` | 结构变异 | 边界跨越潜力 | 通过可扩展性与误差 gate 后再升级 |

因此，论文不应先声称“我们选了 10 个”，而应写：

> Applying the preregistered eligibility rules yielded nine confirmatory measures; a tenth structural-variation measure was retained only if its scalable approximation met the prespecified error criterion.

### 4.2 探索性 challenger

以下指标可以进入性能模型的候选集，但不能支撑主机制主张：

| 指标 | 处理方式 | 原因 |
|---|---|---|
| `first_time_pair_share_w` / Wang W | 探索性 | 原始理论直接，但外部效度研究结论不一致 |
| `community_spanning_simpson`（原 `rtd_simpson`） | 探索性结构指标 | 衡量参考社区分布，不是已经发生的 diffusion |
| `burt_efficiency` | 探索性结构指标 | 结构洞理论支持 brokerage，但从组织/人员网络迁移到论文参考网络仍需验证 |
| `analytic_reference_pair_atypicality` | 近似敏感性 | 与经典期刊对 U 指标的单位和 null model 不同 |
| `reference_clique_modularity_shock_v1` | 旧实现诊断 | 只用于证明新 SVA 实现是否改进，不进入最终主张 |

Burt 的结构洞研究说明 brokerage 与好想法之间存在理论联系（[American Journal of Sociology, 2004](https://doi.org/10.1086/421787)），但这不是论文参考网络上的直接创新性验证。2026 年对 constraint 指标的重新分析还显示，其经验含义经常被误解（[Strategy Science, 2026](https://doi.org/10.1287/stsc.2024.0297)）。因此把 `burt_efficiency` 降为 challenger 比将其包装为核心“知识桥接机制”更稳妥。

## 五、现有 8 个核心指标的逐项处置

| 当前指标 | 主要问题 | 建议 |
|---|---|---|
| `delta_q0_shock` | baseline 是 citation edges，但更新加入 reference clique，图语义不一致 | 用 SVA 的共被引图和 MCR/cluster linkage 重写 |
| `rtd_simpson` | 名称中的 diffusion 暗示了尚未发生的传播 | 改名 `community_spanning_simpson`，降为探索性 |
| `field_log_variety` | 概念有依据，但 log 变换不是构念本身 | 保留原始 variety；log/percentile 只作折内模型变换 |
| `field_evenness` | 概念有依据 | 改名 `field_balance`，保留 |
| `field_disparity` | 当前用领域社区集合 Jaccard，不是标准 science-map distance | 改用发表前 citation-profile cosine distance |
| `pair_atypicality_tail` | 当前是 reference-ID pair + 解析近似，不能直接称 Uzzi | 实现 journal-pair U；旧量改名并作敏感性 |
| `pair_conventionality_median` | 同上；且传统性不是新颖性 | 与 U 同时重写，只作伴随量 |
| `burt_efficiency` | 理论迁移缺乏论文级构念效度 | 降为探索性 challenger |

## 六、10 个辅助特征也不应按数量锁死

### 6.1 建议保留的控制变量块

| 控制块 | 建议变量 | 用途 |
|---|---|---|
| 参考规模 | `log_valid_reference_count` | 控制组合机会数量 |
| 参考年龄 | `reference_age_median`, `reference_age_iqr` | 控制知识基础的新旧和跨度 |
| 先验流行度 | `prior_degree_median`, `prior_degree_p90` | 控制热门知识基础 |
| 图谱可达性 | `prior_component_size_log` | 控制所在历史图组件规模 |

`recent_reference_share_5y`、`classic_reference_share_20y` 与年龄分布重复；`prior_obscure_reference_share` 与 degree 分布重复。它们可以作为年龄/流行度控制块的替代参数化，在内层 CV 中比较，但不应与所有分位量一起强制进入主模型。

### 6.2 必须与预测变量分开的字段

**校准变量**：`domain12`、`publication_year`、`venue_family`。

**质量字段**：`reference_metadata_coverage`、`valid_pair_count`、`pair_sampling_rate`、语义标题覆盖率。

质量字段主要用于排除、加权、缺失审计和敏感性分析，不应被包装成科学机制。

### 6.3 应删除的确定性冗余

当前实现中，若 `n` 是有效参考文献数、`density` 是参考文献诱导子图密度，则：

```math
burt\_efficiency = 1-density\frac{n-1}{n}
```

因此，在同时提供参考文献数时，`reference_induced_density` 与 `burt_efficiency` 存在确定性关系，不应作为独立辅助特征并列解释。

## 七、预注册的纳入、降级和排除标准

### 7.1 证据等级

| 等级 | 要求 | 可进入的位置 |
|---|---|---|
| A | 有研究者自评、专家标签或其他外部构念效度，并满足发表前计算 | 直接新颖性确认集 |
| B | 有明确理论和独立预测/known-group 验证，但无充分构念效度 | 整合或结构确认集 |
| C | 只有公式可解释性或从其他网络场景迁移 | 探索性 challenger |

直接新颖性家族至少要求 A；整合和结构家族至少要求 B。不能因为某个 C 级指标提高一次全局 Spearman，就把它事后升级为理论核心。

### 7.2 数据和可靠性 gate

建议在看 sealed holdout 前冻结：

- `source_max_year < publication_year` 为 100%；
- 确认性指标有限值覆盖率总体至少 90%，每个可评估领域至少 80%；
- pair 指标至少 10 篇有效参考文献、至少 20 个有效 pair；同时报告 30/50 pair 的敏感性；
- semantic 指标报告标题覆盖率、语言覆盖率和异常 reference-link 率；
- 近似算法必须在分层精确子集上报告与精确值的 Spearman、绝对误差和排名翻转率；
- 三个窗口的发表前原始特征必须逐列一致。

阈值不是“理论真理”，而是预先声明的数据质量规则；正文和 Supplement 要报告阈值敏感性，不能按 OOF 结果修改门槛。

### 7.3 冗余和稳定性 gate

所有筛选只在 outer-train 内进行：

- 确定性或代数冗余：直接排除；
- 同一家族中若 `|Spearman rho| > 0.85` 在至少 4/5 个 outer folds 重复出现，保留证据等级更高、覆盖更好、定义更简单者；
- 方向稳定性、family ablation 和 permutation importance 在 fold × domain × horizon 层面报告；
- 不通过稳定性 gate 的指标可以留在 HGB challenger，但不能进入确认性解释分数；
- 不允许在看过 sealed temporal holdout 后更换主指标、阈值或 null model。

## 八、需要增加“构念效度”，而不只是 OOF

RGPM-D3/D5/D8 衡量未来知识采用的广度和均衡性。它可以作为预测效标，但不是论文创新性的 ground truth。建议验证体系分为四层：

1. **内容效度**：系统检索和五家族理论映射；
2. **收敛/区分效度**：与专家新颖性评分相关，同时不能只是 reference count、跨学科性或期刊声望的替代；
3. **预测效度**：严格 nested OOF 对 RGPM-D3/D5/D8、RGPM-S 子集和 top-decile 的预测；
4. **known-group 效度**：Nobel-related、APS Milestones 或其他预先定义 landmark 集合，仅作外部验证。

### 建议新增一个封存专家样本

- 约 600 篇，12 个领域各约 50 篇，按年份和期刊家族分层；
- 每篇至少两名评审者，盲于模型分数和未来引用；
- 分别评价新方法/材料、新发现/现象、新解释/理论，以及“只是改进而非新颖”；
- 先冻结 rubric，再抽样和评分；
- 报告评审一致性，并将该样本完全排除在特征和模型选择之外。

若暂时无法完成 600 篇，可以先使用 F1000/Faculty Opinions 可连接子集和现有同行评审缓存做外部验证，但必须承认其领域偏倚，不能替代多领域专家样本。

## 九、对模型和分数契约的影响

建议不再让一个分数同时承担“创新性解释”和“未来影响预测”两个任务。

输出改为：

```text
signal_family_scores          # 五个发表前测量家族，不称因果机制
score_reference_novelty       # 只由前三个直接新颖性家族组成
score_performance             # 确认指标 + 合格 challenger + 控制变量，专门优化 OOF
```

其中：

- `score_reference_novelty` 的权重只能在 outer-train 内拟合，并要在专家样本上做外部验证；
- `score_performance` 可以使用 GAM/HGB/Rank Blend，但不得反向被解释为“创新性真值”；
- Fig.2 展示五个测量家族和构念效度；
- Fig.3 展示对未来 RGPM 的预测效度；
- 两张图回答不同问题，避免用同一 Spearman 证明一切。

## 十、推荐的落地顺序

1. 将当前 release 继续保持 `overall_pass=false`；冻结旧 8+10 为 `feature_set_v1_baseline`。
2. 建立 `candidate_metric_registry_v2`，为每项增加 construct、role、evidence grade、formula source、time eligibility、coverage 和 exclusion reason。
3. 先实现三个直接新颖性家族：journal-pair U、prior-only reference semantic distance、prior-domain overlap novelty。
4. 把 diversity disparity 改为发表前 citation-profile cosine distance。
5. 使用已有 `pair_count` 重建统一语义的历史共被引图，实现 SVA modularity change 和 cluster linkage；不需要重新抓 OpenAlex。
6. centrality divergence 先在分层子集做精确/近似对照，通过后再决定是否进入全量确认集。
7. 清理辅助特征冗余，把 calibration 和 quality 字段从科学特征中拆开。
8. 固定比较：controls-only → direct novelty → +integration → +SVA → +exploratory challengers；所有比较使用同一 outer folds。
9. 完成专家/F1000 外部构念效度后，才能在论文中使用“novelty”作为主张；否则只写“reference-based novelty signals”。
10. sealed holdout 只解锁一次。若新方案没有提升 OOF，也不能通过修改候选池、引用门槛或标签定义来追分。

## 十一、可以直接用于论文的方法表述

> We did not treat the number of candidate metrics as a theoretical claim. We constructed a dated evidence registry using prespecified searches, citation chaining, role assignment, and deduplication of algebraic variants. Eligible publication-prior measures were organized into five measurement families: atypical–conventional recombination, semantic remoteness of cited work, dissimilarity from prior same-domain bibliographies, diversity of knowledge sources, and structural variation of the pre-publication co-citation network. These families are hypothesized signal classes rather than identified causal mechanisms.

> We distinguished direct proxies of reference-based combinatorial novelty from indicators of knowledge integration, structural boundary spanning, bibliographic controls, and post-publication outcomes. Predictive validity for future graph diffusion was evaluated by nested out-of-fold testing, whereas construct validity for novelty was evaluated separately against blinded expert or author assessments. Thus, a gain in OOF rank correlation was not interpreted by itself as proof that a feature measured scientific novelty.

## 十二、最终判断

- **92 个候选**：可以保留，但只能作为有日期、有边界、可更新的 evidence registry；不能说是完整全集。
- **5 个机制**：不建议保留“机制”称呼；改为 5 个 publication-prior measurement families。
- **8 个核心指标**：不建议继续锁定。现有 8 个中，只有 diversity 三维和 atypical/conventional 的概念可直接继承，且后者必须重写计算。
- **10 个辅助特征**：不建议按数量保留；拆为控制块、校准变量和质量字段，并删除确定性冗余。
- **是否可以用更多指标**：可以。性能模型可使用更多合格 challenger，但理论主张只依赖预注册确认集；超参数变体放 robustness，不重复算作新机制。
- **最重要的新增指标**：prior-only reference semantic distance、prior-domain overlap novelty、SVA cluster linkage。
- **最重要的新增证据**：独立的多领域专家新颖性评分。没有这一层，即使 OOF Spearman 很高，也只能证明未来扩散可预测，不能充分证明“创新性被测量”。

遵循这一重构后，论文最强的可信性提升不会来自“指标数更多”，而来自三点：概念不越界、实现与文献一致、构念效度和预测效度分开验证。

# Evidence-derived v3：36小时30分钟完整执行记录

## 1. 文档定位

本文记录 `innovation_impact_feature_selection/evidence_derived_v3/` 从证据饱和型检索设计、独立复核、检索框架冻结、正式文献筛选、指标普查、候选维度归纳、硬门槛筛选到最终审计的完整过程。

本文的用途包括：

- 说明为什么最终不是预先指定的“5个机制、50个指标、留下8个指标”；
- 回答审稿人对检索领域数、检索式数、候选维度数和最终指标数主观性的质疑；
- 保存36小时30分钟任务执行期间的关键决策、协议修订、异常和恢复记录；
- 为后续论文方法部分、补充材料、审稿回复和复现实验提供统一入口。

本文不是对所有原始CSV逐行内容的复制。完整的336条逻辑检索式、367条物理请求、432个规范指标家族、66个候选维度、逐记录复核意见及全部哈希，仍以SQLite数据库和审计清单为准。

### 1.1 时间口径

| 项目 | 时间 |
|---|---:|
| 有效任务执行时间 | 36小时29分53秒，四舍五入为36小时30分钟 |
| 活跃目标创建时间 | 2026-07-29 00:57:20 CST |
| 活跃目标完成时间 | 2026-07-30 19:46:07 CST |
| 创建至完成的自然时钟跨度 | 42小时48分47秒 |
| 差异来源 | OpenAlex/Crossref等待、独立Codex任务等待、长任务挂起及空闲时间 |

因此，“36小时30分钟”是任务系统累计的有效执行用时，不等于连续墙钟时间。最早的OpenAlex库存盘点在目标正式创建前约1小时完成，也作为本次执行的直接输入保留在审计链中。

### 1.2 最终状态

- 正式审计状态：`COMPLETE`
- 完成阻断项：0
- 最终轮次：第12轮
- 第13轮及以后饱和轮次：0
- 确定性决策哈希：`1b5bdeb08308a82686feb9c6620504692fd0f3f03f5f100954e9042f2f48ebe6`
- 连续两次最终审计：结果一致
- v3测试：全部通过
- ASPR既有业务代码：未修改
- v2证据包：未修改，仅作为试检索和补充术语来源

主要事实来源：

- [最终审计报告](../innovation_impact_feature_selection/evidence_derived_v3/outputs/audit_report_v3.md)
- [机器可读审计摘要](../innovation_impact_feature_selection/evidence_derived_v3/outputs/audit_summary_v3.json)
- [完整审计清单](../innovation_impact_feature_selection/evidence_derived_v3/outputs/audit_manifest_v3.json)
- [冻结协议](../innovation_impact_feature_selection/evidence_derived_v3/protocol_v3.json)
- [证据饱和方法说明](../innovation_impact_feature_selection/evidence_derived_v3/METHODS_EVIDENCE_SATURATION_V3.md)
- [执行目标](../innovation_impact_feature_selection/evidence_derived_v3/EXECUTION_GOAL_V3.md)

## 2. 起点：需要解决的主观性问题

原问题不是代码能否计算指标，而是指标与机制的形成过程能否抵御审稿质疑：

1. 为什么恰好选择某几个机制？
2. 为什么候选指标恰好是某个数量？
3. 为什么最后恰好保留某个数量？
4. 是否先凭经验设定维度，再在每个维度中“凑指标”？
5. 另一研究者按照相同步骤，是否能够得到相同的领域、检索式、维度和指标？

为此，本次执行废弃“先确定5个机制、再找50个指标、最后留下8个”的数量导向思路，改为如下证据链：

```text
领域无关启动检索
→ 英文文献中的原始术语和指标提取
→ 术语标准化、同义词/参数变体去重
→ 角色分离编码与独立复核
→ 自然形成检索概念领域 K
→ 生成逻辑检索式 Q 和API物理请求 P
→ PRESS与开发/隐藏种子召回验证
→ 冻结检索框架
→ 正式检索、筛选、引文追踪和Crossref核验
→ 完整指标普查及英文全文公式核验
→ 先有规范指标，再归纳候选维度 M
→ 所有指标通过统一硬门槛
→ 指标结果反向决定哪些维度能够保留
→ 得到正式预测维度 D 和最终指标 F
```

这里的关键改变是：

- 检索领域不是模型维度；
- 物理API拆分不是新的逻辑检索式；
- 候选维度在指标提取之后归纳；
- 最终维度由是否存在通过全部硬门槛的指标决定；
- `K/Q/P/M/D/F` 均不是预设配额；
- 不使用未来模型效果决定任何领域、维度或指标的入选。

## 3. 固定研究边界

### 3.1 范围

- 文献截止日期：2026-07-28；
- 语言：仅英文；
- 文献类型：英文期刊论文、英文会议论文和英文综述；
- 研究对象：论文级创新性，以及发表时可观察的潜在学术影响力；
- 最晚信息时间：论文发表时 `T0`；
- 结果变量可用于验证，但不得作为入选特征；
- 未来引文、未来扩散、未来颠覆性、未来注意力和未来社交媒体反应不得进入最终特征；
- OpenAlex用于文献发现、检索和引文追踪；
- Crossref仅用于DOI、题名、年份、类型和出版信息核验；
- 本地OpenAlex快照用于历史覆盖、来源追踪、种子可索引性和API节流；
- 所有英语限制及由此产生的语言、地域覆盖偏差必须披露。

### 3.2 六个数量的定义

| 符号 | 定义 | 是否预设 |
|---|---|---|
| `K` | 非冗余检索概念领域数 | 否 |
| `Q` | 非冗余逻辑检索式数 | 否 |
| `P` | OpenAlex实际执行的物理请求数 | 否 |
| `M` | 从规范指标家族归纳出的候选模型维度数 | 否 |
| `D` | 正式保留的核心预测维度数 | 否 |
| `F` | 所有保留角色中的最终指标总数 | 否 |

机会变量、背景控制和敏感性变量单独报告，不用于扩大 `D`。

## 4. 独立目录与系统实现

所有新工作都位于：

```text
innovation_impact_feature_selection/evidence_derived_v3/
```

没有修改ASPR既有业务代码，也没有重写v2结果。v3实现了独立SQLite状态机，核心阶段为：

```text
bootstrap inventory
→ deterministic saturation strata
→ sequential screening and adjudication
→ source-preserving term/indicator extraction
→ code-terms
→ derive-search-frame
→ validate-search-frame
→ retrieve
→ screen-literature
→ extract-indicators
→ derive-dimensions
→ select-indicators
→ audit
```

实现的可复现控制包括：

- SQLite逐阶段状态和事务提交；
- OpenAlex游标与分页检查点；
- API中断后续跑；
- 两个免费OpenAlex key按槽位轮换，数据库只记录槽位预算，不保存key；
- 使用本地快照 `/home/jayee/workspace/FabCitation/openalex-snapshot`；
- DOI、OpenAlex ID、规范题名加年份三级去重；
- 输入、输出、协议、模型、提示词和实现文件SHA-256；
- H1/H2角色分离及导入顺序保护；
- 已仲裁的主编码不可被后续静默覆盖；
- 自动生成文件不能在未注册的情况下作为正式复核结果导入；
- 冻结后源代码版本通过显式版本边保留旧哈希和最终哈希；
- 审计重新计算结果，不信任手工填写的汇总数量。

## 5. 复核角色与证据来源

### 5.1 原设计

原协议设置：

- AI：形成第一套机器编码；
- H1：独立第一复核；
- H2：复核分歧、纳入、不确定及关键构念拆并。

七个自动生成但经项目所有者确认已人工复核和采用的工作表，被登记为：

```text
human_attested_automated_draft
```

它们不能被表述为“从空白开始独立人工撰写”，但可作为经人工复核并采用的正式结果。七个文件包括：

- 第1至第4轮H1筛选；
- H1术语编码；
- H2隐藏种子检索日志；
- H2隐藏验证种子表。

### 5.2 独立AI替代修订

2026-07-29，项目所有者授权后续需要人工复核的环节由单独Codex任务完成。该决定冻结在：

- [独立AI复核协议修订](../innovation_impact_feature_selection/evidence_derived_v3/protocol_amendment_independent_ai_review_v3.json)
- [独立Codex复核统一说明](../innovation_impact_feature_selection/evidence_derived_v3/INDEPENDENT_CODEX_REVIEW_BRIEF_V3.md)

约束包括：

- 必须使用单独Codex任务；
- 不得使用本机 `qwen3:8b`、Ollama或其他本地LLM替代；
- 每次复核必须有独立run ID；
- 保存输入、输出、提示词、模型构建和manifest哈希；
- 保留逐行证据和理由；
- 结果必须标记为 `independent_ai`；
- 不得将独立AI复核表述为真人判断。

最终审计登记：

- 独立AI复核运行：119次；
- 独立AI复核行：50,314行；
- 单独Codex CSV/manifest产物：251个；
- 人工确认采用的自动草稿：7个；
- 被隔离且未采用的本地Qwen产物：3个；
- 被隔离的早期自动H1试验文件：8个；
- 未复核、禁止直接导入的自动H2辅助草稿：12个。

### 5.3 一致性结果如何解释

一致性不是入选门槛，也不能替代H2仲裁。

文献筛选结果：

| 比较 | 原始一致率 | Cohen’s κ | Gwet’s AC1 |
|---|---:|---:|---:|
| 纳排决定 | 0.2660 | 0.0003 | 0.0310 |
| 语言判断 | 0.9226 | 0.3616 | 0.9177 |

维度编码结果：

| 字段 | 原始一致率 | Cohen’s κ | Gwet’s AC1 |
|---|---:|---:|---:|
| 纳入决定 | 0.9907 | 0.0000 | 0.9907 |
| 构念角色 | 0.8079 | 0.7151 | 0.7781 |
| 维度英文标签完全相同 | 0.0208 | 0.0198 | 0.0110 |

自由文本定义、标签、信息来源、T0描述和偏差风险要求逐字完全一致时，一致率很低。这反映两个编码流采用不同措辞，并不等于构念无法仲裁。正式结果始终使用H2逐项复核后的规范标签、定义和角色。

术语编码的逐字复合一致率也很低，原因之一是早期H1草稿使用固定28标签的临时字典，而证据驱动H2允许拆分、合并和建立新的规范家族。该28标签字典没有权力决定最终 `K`。

## 6. 36小时30分钟执行时间线

下表统一使用中国标准时间（CST，UTC+8）。数据库阶段时间、协议内时间与文件封存时间可能相差数分钟；这是命令完成、文件写入和阶段状态提交的正常差异。

| 时间 | 阶段 | 主要动作与结果 |
|---|---|---|
| 2026-07-28 23:50 | 启动库存 | 领域无关OpenAlex启动式报告3,591,214条潜在记录；仅作为库存分母，不宣称全部抓取。 |
| 2026-07-29 00:57 | 目标启动 | 冻结“不预设K/Q/P/M/D/F、仅英文、T0、v2只读、ASPR业务代码不动”的目标。 |
| 2026-07-29 02:44 | 初始分层检索 | 101个非正式查询分层全部完成，1,040页、99,474行、91,752条初始唯一记录；后续继续扩展至最终证据图。 |
| 2026-07-29 09:14 | 人工采用登记 | 七个已复核工作表按精确哈希登记为 `human_attested_automated_draft`。 |
| 2026-07-29 10:43 | 复核替代修订 | 所有者授权后续H1/H2由单独Codex任务完成；明确禁止本地Qwen/Ollama。 |
| 2026-07-29 11:02 | 语言规则修正 | 发现两个“非英文原题名但存在英文译文摘要”的边界案例；修订为原始题名、摘要或正文明确非英文即排除，旧版本保留为被替代审计证据。 |
| 2026-07-29 13:05 | 第1—4轮完成 | 完成筛选、术语/指标提取和H2对齐；每轮双端点均非零，继续。 |
| 2026-07-29 21:59 | 第5轮完成 | 新术语家族51、新指标家族92；继续。 |
| 2026-07-29 23:58 | 第6轮完成 | 新术语家族32、新指标家族64；修正角色对齐后通过验证，继续。 |
| 2026-07-30 02:44 | 第7轮完成 | 正式替换一版指标仲裁结果并重新验证；新术语35、新指标25。 |
| 2026-07-30 04:40 | 第8轮完成 | 新术语47、新指标24；继续。 |
| 2026-07-30 05:57 | 第9轮完成 | 新术语21、新指标8；继续。 |
| 2026-07-30 07:30 | 第10轮完成 | 新术语13、新指标12；继续。 |
| 2026-07-30 08:50 | 第11轮完成 | 新术语15、新指标10；继续。 |
| 2026-07-30 09:43 | 第12轮终止授权 | 所有者决定第12轮为终轮，依据是继续检索的边际收益不足；登记为回顾性协议偏离。 |
| 2026-07-30 10:06 | 第12轮冻结 | 实际新增术语家族10、指标家族9；不是原预注册的“双零”，不创建第13轮。 |
| 2026-07-30 10:13 | 术语编码完成 | 3,615个活跃术语、3,170个纳入术语、1,102个规范术语、367个术语家族完成。 |
| 2026-07-30 12:52 | 检索框架形成 | 由术语、构念角色、PRESS和去重自然形成 `K=42`、`Q=336`、`P=367`。 |
| 2026-07-30 12:55 | 召回验证完成 | 初次51/62；10个来源可追溯的最小同义词修复覆盖11篇漏召回种子；最终62/62。 |
| 2026-07-30 12:57 | 检索框架冻结 | 冻结语义哈希 `088ca35e...3046e`，所有后续检索只执行冻结版本。 |
| 2026-07-30 13:14 | 正式检索完成 | 367条物理请求全部建立库存/池；正式池30,508行、30,332条唯一记录；固定终末队列3,273条唯一记录。 |
| 2026-07-30 14:00 | 文献筛选完成 | 9,515条最终处置：363纳入、9,152排除；376条因非英文排除。 |
| 2026-07-30 15:39 | 公式—操作化分层 | 冻结“文献公式证据”与“项目计算规则”两层结构，禁止把项目自定缺失值规则伪装为文献原文。 |
| 2026-07-30 15:55 | 指标普查完成 | 363篇纳入来源、1,685个指标提及、432个规范指标家族全部完成H1/H2处置。 |
| 2026-07-30 16:13 | 定向公式补全 | 对已有432家族中可能对应本地T0数据的指标做统一、结果盲的公式证据补全；不形成第13轮、不新增指标家族。 |
| 2026-07-30 17:06 | 数据对应完成 | 432个家族全部检查，59个被H2判定可能存在数据对应；继续进行公式、操作化和质量漏斗。 |
| 2026-07-30 18:36 | 操作化完成 | 16个文献公式候选完成H2审查，7个通过最终操作化；候选数据矩阵覆盖118,059篇论文。 |
| 2026-07-30 18:36—18:53 | 维度编码 | H1和独立H2完成432个指标家族的构念、角色、来源、T0和偏差编码；形成66个候选维度。 |
| 2026-07-30 18:49 | 实现快照版本化 | 对7个曾冻结但后续合法扩展的实现文件建立旧版→最终版显式版本边，旧哈希不覆盖。 |
| 2026-07-30 19:37 | Crossref全量核验 | 完成剩余Crossref请求；无API错误，冲突进入独立H2批次。 |
| 2026-07-30 19:39 | 最后一批书目仲裁 | 最终56条冲突完成独立H2复核；正式集合中8,815条有DOI记录全部进入已验证、日期变体或已解决终态。 |
| 2026-07-30 19:44 | 最终维度与指标 | `M=66`；4个操作维度保留；严格预测维度 `D=1`；最终指标 `F=7`。 |
| 2026-07-30 19:45 | 双次审计 | 两次审计均为 `COMPLETE`、阻断项0、确定性哈希完全一致。 |
| 2026-07-30 19:46 | 目标完成 | 目标标记完成；累计有效执行用时36小时29分53秒。 |

## 7. 领域无关启动检索与证据图

### 7.1 启动式

启动检索只包含三个通用概念块，不使用旧12领域名称：

```text
研究对象：
paper OR article OR publication OR scientific work

目标构念：
novelty OR innovation OR potential impact
OR citation impact OR research quality OR scientific influence

证据类型：
measure OR indicator OR metric OR feature OR predictor
OR determinant OR validation OR review
```

块内使用 `OR`，块间使用 `AND`。语言不在API检索阶段预过滤，而是在筛选阶段判断，以便将非英文记录保留在PRISMA分母中。

### 7.2 为什么没有全部下载3,591,214条

3,591,214是启动式的OpenAlex库存计数，不是最终相关文献数。完整下载会带来巨大API、存储和筛选成本，而且不能自动提高“术语和指标是否饱和”的科学性。

最终采用确定性证据饱和图：

- 按年代和文献类型分层；
- 按通用目标词过采样；
- 按测量、预测、验证角色过采样；
- 从53篇开发证据论文提取公式/指标短语形成高精度探针；
- 获取开发种子的前向和后向引文网络；
- 对正式检索式建立可审计的种子排序池；
- 每一轮取冻结排序的下一段，而不是按相关性Top-K截断；
- 以“新增非冗余英文术语家族”和“新增规范指标家族”为两个端点。

最终证据图统计：

| 项目 | 数量 |
|---|---:|
| 确定性采样分层总数（含正式检索式分层） | 468 |
| 完整分页分层 | 174 |
| 非正式发现分层 | 101，全部完成 |
| 正式检索式分层 | 367，其中73个分页至已需范围 |
| 分层内去重前行数 | 129,982 |
| 唯一确定性样本记录 | 121,164 |
| 唯一开发种子引文网络记录 | 12,509 |
| 合并后的唯一发现/引文记录 | 132,547 |
| 分配进入顺序复核的记录 | 6,312 |

“174/468完整”不表示正式检索失败。101个发现阶段分层全部完成；367个正式检索式在第12轮终止修订后只需提取预先固定的终末队列，未进入队列的池成员保留在审计框架中，不被伪装成已筛选排除。

## 8. 十二轮证据饱和检索

### 8.1 每轮规则

每轮执行：

1. 从每个活跃分层取冻结排序的下一段；
2. AI与H1分别筛选；
3. H2复核规定范围；
4. 从纳入记录提取原始英文术语和指标名称；
5. 使用前序轮次只读规范词表对齐同义词、缩写、参数变体和时间窗变体；
6. 重新计算本轮两个新颖性端点；
7. 端点非零则继续，除非存在已注册协议修订。

前序词表只能用于对齐同一构念，不能改变本轮纳排决定。建立新家族必须说明为什么既有家族不适用。

### 8.2 十二轮实际结果

| 轮次 | 分配记录 | 冻结秩区间 | 新术语家族 | 新指标家族 | 决定 |
|---:|---:|---:|---:|---:|---|
| 1 | 575 | 1–10 | 37 | 46 | 继续 |
| 2 | 565 | 6–20 | 28 | 38 | 继续 |
| 3 | 545 | 11–30 | 10 | 22 | 继续 |
| 4 | 533 | 16–40 | 13 | 32 | 继续 |
| 5 | 535 | 21–50 | 51 | 92 | 继续 |
| 6 | 525 | 26–60 | 32 | 64 | 继续 |
| 7 | 514 | 31–70 | 35 | 25 | 继续 |
| 8 | 511 | 36–80 | 47 | 24 | 继续 |
| 9 | 498 | 41–90 | 21 | 8 | 继续 |
| 10 | 504 | 46–100 | 13 | 12 | 继续 |
| 11 | 504 | 51–110 | 15 | 10 | 继续 |
| 12 | 503 | 56–120 | 10 | 9 | 冻结 |

十二轮合计产生312个逐轮新增术语家族和382个逐轮新增指标家族。它们不是最终总数；开发种子、术语规范化、正式队列和全文普查还会补充、拆并或删除家族。

### 8.3 第12轮停止规则

原预注册规则要求连续三轮同时满足：

```text
新增非冗余英文术语家族 = 0
AND
新增规范指标家族 = 0
```

第12轮实际为10/9，不满足原双零规则。项目所有者认为后续轮次边际收益不足，因此将第12轮指定为终轮。该决定：

- 是回顾性协议偏离；
- 不修改第12轮任何编码以制造零；
- 不创建第13轮；
- 正式检索仍需执行；
- 指标提取、维度归纳和硬门槛筛选仍需完成。

相关文件：

- [第12轮务实停止修订](../innovation_impact_feature_selection/evidence_derived_v3/protocol_amendment_round12_pragmatic_stop_v3.json)
- [第12轮固定正式队列修订](../innovation_impact_feature_selection/evidence_derived_v3/protocol_amendment_round12_terminal_formal_cohort_v3.json)
- [对外0/0口径说明](../innovation_impact_feature_selection/evidence_derived_v3/protocol_amendment_round12_external_reporting_clarification_v3.json)

对外可使用：

```text
post-freeze expansion 0/0
```

它的严格含义是：第12轮冻结以后至最终审计，没有再纳入新的术语家族或指标家族，也没有启动第13轮。它不能被写成“第12轮本身达到0/0”。方法部分必须同时披露第12轮实际为10/9。

## 9. 从术语自然形成42个检索概念领域

### 9.1 术语处理结果

| 项目 | 数量 |
|---|---:|
| 活跃原始术语 | 3,615 |
| 纳入术语 | 3,170 |
| 规范术语 | 1,102 |
| 术语家族 | 367 |
| 最终检索概念领域 `K` | 42 |

标准化处理包括：

- Unicode、大小写、标点和单复数统一；
- 缩写和全称双向映射；
- 词形变化只用于匹配，不改变构念；
- 数据源、参数、时间窗和阈值差异通常归为同一术语家族；
- 理论对象、解释角色或T0边界不同则拆分；
- 每个规范术语保留原始词、来源记录和证据片段；
- v2词汇只能提供候选同义词，不能单独建立领域。

### 9.2 领域不是预先定为10、12或20

`K=42` 是以下约束共同作用的结果：

1. 直接来源文献中的英文术语；
2. 术语家族去重；
3. AI/H1角色分离编码；
4. H2拆分、合并和跨领域仲裁；
5. 每个领域必须有直接来源依据；
6. PRESS检查；
7. 开发及隐藏种子召回验证。

如果两个术语测量同一理论对象且承担相同角色，则合并；如果分别代表实质内容、注意机会、背景控制或验证结果，则拆分。因此领域数量不是人工选择的整数。

## 10. 从42个领域形成336条逻辑检索式和367条物理请求

### 10.1 逻辑检索式

每个逻辑式采用：

```text
领域术语块
AND
论文对象块
AND
测量、预测或验证语境块
```

规则：

- 同一术语家族放在同一个 `OR` 块；
- 仅同义词不同不新增逻辑式；
- 被另一检索式完全覆盖且无独立构念作用时归档为冗余；
- 零命中式保留审计记录但不进入活动集；
- 只有构念关系或证据角色不同才形成独立逻辑式；
- OpenAlex语法或URL长度拆分不增加 `Q`。

最终：

| 项目 | 数量 |
|---|---:|
| 活动检索概念领域 `K` | 42 |
| 活动非冗余逻辑检索式 `Q` | 336 |
| OpenAlex物理请求 `P` | 367 |
| 归档零命中逻辑式 | 13 |

完整内容见：

- [检索领域表](../innovation_impact_feature_selection/evidence_derived_v3/outputs/search_domains_v3.csv)
- [逻辑检索式表](../innovation_impact_feature_selection/evidence_derived_v3/outputs/logical_queries_v3.csv)
- [物理请求表](../innovation_impact_feature_selection/evidence_derived_v3/outputs/physical_queries_v3.csv)
- [冻结检索框架](../innovation_impact_feature_selection/evidence_derived_v3/outputs/frozen_search_frame_v3.json)

### 10.2 API异常与物理拆分

四条过长OpenAlex请求反复返回HTTP 500。处理方式是：

- 仅机械拆分物理请求；
- 保留同一个逻辑检索式ID；
- 验证拆分前后的领域词并集完全相同；
- 保持对象块、语境块、过滤器、逻辑哈希和PRESS结论不变；
- 因此只改变 `P`，不改变 `K` 或 `Q`。

一次短暂HTTP 429通过SQLite检查点和key槽位轮换恢复。免费API key从未写入代码、文档或数据库。

## 11. PRESS与种子召回验证

### 11.1 验证种子

- 开发种子：53篇；
- 独立隐藏验证种子：9篇；
- 总计：62篇；
- 隐藏种子来源路线：独立综述检索、后向引文追踪、前向引文追踪。

隐藏种子工作表属于前述七个 `human_attested_automated_draft` 之一：自动构建器中存在预先声明的合格DOI列表，随后由项目所有者确认工作表已经人工复核并采用。因此这里的“隐藏”是指未参与最初术语生成，而不是声称整个种子建立过程从空白开始完全独立人工完成。导入后，系统重新核对三条检索路线的DOI集合、OpenAlex可索引性和最终召回，路线并集必须与9篇隐藏种子完全一致。

### 11.2 召回修复

初次完整验证召回51/62。11篇漏召回均确认已被OpenAlex索引，因此属于检索式缺词，而不是数据库缺失。

修复步骤：

1. 只允许加入种子题名、摘要或引文证据中实际出现的英文同义词；
2. H1盲审；
3. H2进行聚焦PRESS复核；
4. 不允许新建领域或逻辑式来“硬召回”种子；
5. 10个既有逻辑家族完成最小修复，其中一个家族覆盖两篇漏召回种子；
6. 修订后11篇全部直接命中；
7. 最终版本6召回62/62。

最终验证：

- PRESS未解决问题：0；
- OpenAlex可索引但未召回种子：0；
- 提供者缺失种子：0；
- 最终检索框架语义哈希：`088ca35e634187e5267b0bcbb09c4ad23f7e40d63d6bcfd17bb250584a83046e`。

## 12. 正式检索与固定终末队列

每条物理请求均记录：

- 请求ID；
- 父逻辑式ID；
- 查询文本和哈希；
- OpenAlex库存总量；
- 分页和游标；
- 完成状态；
- key槽位；
- 记录身份和种子排序。

正式检索统计：

| 项目 | 数量 |
|---|---:|
| 物理请求 | 367 |
| 检索式报告总量之和 | 371,316,686 |
| 实际检索行 | 30,508 |
| 唯一检索式—记录链接 | 30,501 |
| 唯一正式记录 | 30,332 |
| 固定终末队列链接 | 3,288 |
| 固定终末队列唯一记录 | 3,273 |

371,316,686是高度重叠的各检索式库存总量之和，不能解释为唯一论文数。

第12轮终止修订规定：每个活动物理请求预先取冻结种子排序前10条，跨请求及与前12轮记录去重，形成固定正式队列。它不是第13轮，也不依据指标或模型结果抽样。

## 13. 文献筛选与PRISMA流转

最终有9,515条记录获得题名/摘要终态：

| 处置 | 数量 |
|---|---:|
| 纳入指标普查 | 363 |
| 排除 | 9,152 |
| 合计 | 9,515 |

排除理由：

| 排除理由 | 数量 |
|---|---:|
| 不研究论文级创新性或潜在影响 | 7,918 |
| 非论文级分析 | 695 |
| 非英文 | 376 |
| 非指标、预测因素或验证研究 | 76 |
| 题名摘要信息不足 | 48 |
| 仅研究未来结果且无T0特征 | 38 |
| 重复记录 | 1 |

语言判断以题名、摘要和全文证据为准。OpenAlex语言字段只作为辅助，不能单独决定排除。非英文记录保留在检索和筛选分母中，排除码统一为 `E_LANGUAGE_NON_ENGLISH`。

完整处置见：

- [PRISMA流转数据](../innovation_impact_feature_selection/evidence_derived_v3/outputs/prisma_flow_v3.csv)
- [文献筛选决定](../innovation_impact_feature_selection/evidence_derived_v3/outputs/literature_dispositions_v3.csv)

## 14. Crossref书目核验

Crossref只核验：

- DOI身份；
- 规范题名；
- 出版年份；
- 文献类型；
- 期刊或出版物信息。

日期差异在DOI、题名相似度不低于0.85且类型一致时记录为 `validated_date_variant`，用于容纳online-first和正式卷期年份差异。

正式集合中有DOI的8,815条记录最终状态为：

| Crossref状态 | 数量 |
|---|---:|
| `validated` | 8,231 |
| `validated_date_variant` | 236 |
| `resolved` | 348 |
| 缺少核验记录 | 0 |
| 未解决冲突或错误 | 0 |

冲突分10批交给独立H2复核。最后一批56条的结果为：

- 接受OpenAlex：46；
- 接受Crossref：4；
- 排除错误映射：4；
- 保留人工书目解析：2。

数据库全局仍保留9条不属于正式复核集合的历史冲突，它们不进入正式证据链，也不构成最终审计阻断项。该保留是为了避免静默删除历史记录。

## 15. 指标完整普查

### 15.1 提取顺序

指标先于维度提取。对363篇纳入来源执行：

1. 获取合法英文开放全文；
2. 记录原始指标名称、别名、缩写和历史名称；
3. 记录页码、表格、公式、附录或章节位置；
4. 记录公式、单位、参数、方向和缺失值说明；
5. 记录所需数据和最大信息时间；
6. 区分创新、T0实质潜力、机会、控制和验证结果角色；
7. 保存正向、混合、无效和负向验证证据；
8. H1提取，H2复核来源处置和每个保留提及；
9. 同义名称、参数变体、时间窗和编码形式归并为规范指标家族。

结果：

| 项目 | 数量 |
|---|---:|
| 纳入来源 | 363 |
| 指标提及 | 1,685 |
| 规范指标家族 | 432 |
| 未完成H1来源 | 0 |
| 未完成H2来源 | 0 |
| 未复核保留提及 | 0 |

完整指标库：

- [完整指标库](../innovation_impact_feature_selection/evidence_derived_v3/outputs/complete_indicator_library_v3.csv)
- [全文证据](../innovation_impact_feature_selection/evidence_derived_v3/outputs/fulltext_indicator_evidence_v3.json)

### 15.2 文献公式与项目操作化分离

执行中发现：原文经常给出公式或确定性定义，却不报告缺失数据、空集合或分母为零时如何处理。如果把“原文未报告”直接补写为文献公式，会产生虚假来源；如果因此全部判死，又会把报告不完整误当成数学不可复现。

因此冻结两层结构：

1. **文献公式证据层**：只能记录英文原始或数学基础来源真正报告的定义、公式、参数和方向；
2. **项目操作化层**：单独说明本项目如何处理缺失、空集合、分母为零、覆盖率、数据类型和边界值。

项目规则不能反向冒充文献原文，两层都经H1/H2复核后，`G04_REPRODUCIBLE_DEFINITION` 才能通过。

相关修订：

- [公式—操作化分层协议](../innovation_impact_feature_selection/evidence_derived_v3/protocol_amendment_formula_operationalization_separation_v3.json)
- [定向公式补全协议](../innovation_impact_feature_selection/evidence_derived_v3/protocol_amendment_targeted_formula_completion_v3.json)

### 15.3 指标漏斗

| 阶段 | 数量 |
|---|---:|
| 规范指标家族 | 432 |
| H2认为可能对应本地T0数据 | 59 |
| 具有合格文献公式候选 | 16 |
| 实际进入候选数据映射 | 12 |
| H2批准完整操作化 | 7 |
| 最终通过全部硬门槛 | 7 |

定向公式补全只在已有432个家族中工作，不新增家族、不改变 `K/Q/P`、不形成第13轮，也不查看模型表现。

## 16. 从432个指标归纳66个候选维度

候选维度不是从42个检索领域直接复制。形成过程为：

1. 完成432个规范指标家族；
2. AI与H1分别编码被测构念、解释角色、信息来源、T0边界和偏差风险；
3. 聚类或文本相似性只用于提示可能误归类项；
4. H2复核全部拆分、合并、多标签和未归类项；
5. 同一构念、同一角色、同一T0边界的指标合并；
6. 实质创新、机会、控制和敏感性角色分开；
7. 形成66个自然候选维度 `M`；
8. 再根据指标硬门槛结果决定维度能否保留。

最终：

| 维度结果 | 数量 |
|---|---:|
| 候选维度 `M` | 66 |
| 保留操作维度 | 4 |
| 因无指标通过全部硬门槛而淘汰 | 45 |
| 因无指标通过且不足两个独立团队而淘汰 | 17 |
| 严格核心预测维度 `D` | 1 |

完整映射见：

- [候选维度表](../innovation_impact_feature_selection/evidence_derived_v3/outputs/candidate_dimensions_v3.csv)
- [维度拆并日志](../innovation_impact_feature_selection/evidence_derived_v3/outputs/dimension_merge_split_log_v3.csv)
- [最终维度决定](../innovation_impact_feature_selection/evidence_derived_v3/outputs/final_dimensions_v3.json)

## 17. 统一硬门槛与失败原因

432个规范指标全部使用同一组门槛，不按维度分配名额：

| 门槛 | 要求 | 失败数 |
|---|---|---:|
| G01 | 属于创新、T0潜力、机会或允许的背景控制 | 0 |
| G02 | 论文级可计算 | 0 |
| G03 | 有同行评议原始应用或数学基础来源 | 265 |
| G04 | 文献定义与项目操作化共同达到可复现 | 425 |
| G05 | 发表时可计算 | 211 |
| G06 | 不依赖未来信息 | 0 |
| G07 | 当前数据已具备或可稳定推导 | 420 |
| G08 | 通过偏差和保护属性规则 | 0 |
| G09 | 无致命构念效度问题 | 0 |
| G10 | 未使用模型结果决定入选 | 0 |
| G11 | 通过数据质量审计 | 420 |
| G12 | 在有效样本中非常量 | 420 |
| G13 | 有英文关键全文及公式证据 | 416 |
| G14 | 有第二角色正式批准 | 416 |

数据库字段 `G14_SECOND_HUMAN_APPROVAL` 沿用原始协议命名。根据后续修订，本次未由真人完成的G14由单独Codex H2提供，必须表述为独立AI复核，而不能写成第二真人。

同一冗余家族仅保留一个代表，排序顺序固定为：

1. 预注册优先级；
2. 证据强度；
3. 数据可得性；
4. 稳定性；
5. 指标ID确定性平局规则。

任何模型预测结果都未进入上述顺序。

## 18. 最终维度与7个指标

### 18.1 四个保留操作维度

| 类型 | ID | 维度 | 独立研究团队 | 最终指标 |
|---|---|---|---:|---|
| 核心预测 | CD031 | Knowledge diversity and integration potential | 9 | EF0312、EF0318 |
| 机会变量 | CD014 | Collaboration breadth and type | 24 | EF0186、EF0188 |
| 控制变量 | CD010 | Author team composition context | 12 | EF0038 |
| 控制变量 | CD041 | Publication and venue context | 14 | EF0197、EF0307 |

因此：

- 保留操作维度总数为4；
- `D=1` 只统计核心预测维度；
- 另有1个机会维度和2个控制维度；
- 敏感性维度保留数为0；
- 最终指标总数 `F=7`。

### 18.2 最终指标

| ID | 角色 | 英文名称 | 计算定义摘要 | 有效值 | 缺失率 | 唯一值 |
|---|---|---|---|---:|---:|---:|
| EF0312 | predictive | Reference balance | 参考文献类别分布的 `1−Gini` 平衡度 | 92,454 | 0.2169 | 8,546 |
| EF0318 | predictive | Reference variety | 非缺失映射参考文献领域的不同类别数 | 92,454 | 0.2169 | 16 |
| EF0186 | opportunity | International collaboration | 已知作者单位国家数大于1时为1 | 95,306 | 0.1927 | 2 |
| EF0188 | opportunity | International collaboration scale | 作者单位所覆盖的不同国家数 | 95,306 | 0.1927 | 35 |
| EF0038 | control | Author count | 论文具名作者数 | 112,576 | 0.0464 | 206 |
| EF0197 | control | Journal identity | 发表期刊或来源的无序类别标识 | 118,059 | 0.0000 | 36 |
| EF0307 | control | Publication year | 论文发表公历年份 | 118,059 | 0.0000 | 38 |

这些结果不是“每个维度选两个”或“目标保留7个”。只有上述7个指标同时通过14个硬门槛。

## 19. 最终训练特征

最终训练矩阵：

| 项目 | 结果 |
|---|---|
| 论文行数 | 118,059 |
| 特征数 | 7 |
| 结果变量 | 不包含 |
| 未来信息 | 不使用 |
| 特征顺序 | EF0312、EF0318、EF0186、EF0188、EF0038、EF0197、EF0307 |
| Parquet SHA-256 | `9282b58ea36054853710b5c0904b20b86e0a76bfa91bff3bdddc2eb401b5ff51` |
| Schema SHA-256 | `717420d0d729fc76008d6f8de8e2077767de9ce7bac28e9a87fd0cd0a753eccf` |

文件：

- [最终训练矩阵](../innovation_impact_feature_selection/evidence_derived_v3/outputs/final_training_features_v3/final_training_features_v3.parquet)
- [最终训练Schema](../innovation_impact_feature_selection/evidence_derived_v3/outputs/final_training_features_v3/final_training_features_schema_v3.json)

候选矩阵中还计算了其他可能的T0列，但候选矩阵明确标记为“不是最终选择”。常量、精确重复、公式证据不完整、数据映射不足或H2未批准的列均不得进入最终训练矩阵。

## 20. 审计、复现与实现快照

### 20.1 最终审计

审计检查：

- 全部阶段是否完成；
- 所有正式文献是否有终态；
- 所有种子是否召回；
- PRESS是否有未解决问题；
- 正式DOI是否均完成Crossref核验；
- 每个最终指标是否有英文全文公式证据；
- 每个最终指标是否通过全部门槛；
- 每个保留预测维度是否至少有一个最终指标及两个独立团队；
- 是否存在未注册自动复核文件；
- 源文件哈希是否与冻结快照一致；
- 最终矩阵的行数、列、顺序、缺失率和哈希是否一致；
- 是否误用未来信息或模型结果；
- 是否存在预设数量配额。

最终连续两次审计：

```text
formal_review_complete = true
completion_blockers = []
deterministic_result_hash =
1b5bdeb08308a82686feb9c6620504692fd0f3f03f5f100954e9042f2f48ebe6
```

### 20.2 实现快照版本化

执行过程中，部分实现文件在首次登记哈希后继续加入合法的下游验证、独立复核导入保护、训练矩阵和审计检查。直接覆盖旧哈希会造成虚假冻结，因此建立7条显式版本边：

- `code_coding → code_coding_final_v3`
- `code_database → code_database_final_v3`
- `code_indicators → code_indicators_final_v3`
- `code_pipeline → code_pipeline_final_v3`
- `code_providers → code_providers_final_v3`
- `code_reporting → code_reporting_final_v3`
- `code_tests_v3 → code_tests_v3_final_v3`

旧版哈希仍保留，最终字节存入内容寻址快照。该修订不改变任何冻结文献、术语、检索式、指标证据或复核决定。

## 21. 执行中发现的异常及处理

### 21.1 本地Qwen尝试被终止

曾启动本机 `qwen3:8b` 的部分H2复核尝试。项目所有者明确要求不得使用该模型，因此：

- 运行被停止；
- 三个进度/说明文件进入隔离目录；
- 未注册、未导入；
- 不进入一致性、领域、维度、指标或饱和数量。

### 21.2 语言边界修正

早期独立Codex筛选将英文翻译摘要视为两个非英文原题名记录的纳入依据。随后冻结语言附录：

- 原始题名、摘要或全文明确为非英文时排除；
- 修正版本重新生成并注册；
- 旧版本作为被替代证据保留；
- 不静默删除历史运行。

### 21.3 H2维度文件来源字段不完整

第一版H2维度CSV缺少正式注册所需的完整来源字段。处理方式：

- 撤回未合格版本；
- 删除其尚未合法进入正式决策链的行；
- 重新由独立任务生成带输入、输出、提示词和任务来源的版本；
- 加强导入保护，正式模式只接受精确注册的artifact；
- 最终H2维度CSV哈希为 `02638ca8636346603389ef97f1d074e17b2d77d8b59da00e45815de80e9e618e`。

### 21.4 OpenAlex长请求和429

- 四条长请求出现HTTP 500；
- 仅做物理拆分，不改变逻辑语义；
- 一次429通过检查点、单并发和key槽位轮换恢复；
- 未丢失已完成游标或重复计算最终结果。

### 21.5 Crossref长批次

Crossref核验采用持久连接、顺序提交和受限请求速率。所有剩余请求完成后，将冲突分10批独立复核。最后一批56条完成后，正式集合的Crossref阻断项归零。

### 21.6 测试夹具短暂覆盖最终JSON

最终审计首次通过后运行 `tests_v3.py`。测试全部通过，但其中一个测试夹具把：

- `final_dimensions_v3.json`
- `final_feature_set_v3.json`

短暂覆盖为测试用的“1个模拟维度、0个特征”结果。

该问题通过紧接着的哈希检查发现。恢复步骤：

```bash
python3 pipeline.py derive-dimensions
python3 pipeline.py select-indicators
python3 materialize_final_training_features_v3.py
python3 pipeline.py audit
python3 pipeline.py audit
```

恢复后：

- `M/D/F` 回到 `66/1/7`；
- 最终维度和指标哈希回到冻结值；
- 训练矩阵哈希未变化；
- 两次审计确定性哈希一致；
- 审计后未再次运行会覆盖生产JSON的测试夹具。

因此，后续若再次运行 `tests_v3.py`，应在发布前重新执行上述生成和审计步骤。

## 22. 主要限制和对外表述边界

### 22.1 英文限制

仅纳入英文文献会低估：

- 非英语地区的创新构念；
- 不同学术传统中的评价框架；
- 非英语期刊中的指标验证；
- 地域特定的合作、资源和发表机制。

因此最终特征空间带有语言和地域覆盖偏差。

### 22.2 不是穷尽全部OpenAlex结果

本研究是系统、确定性、可复现的证据饱和图，不是对3,591,214条启动命中或371,316,686条重叠检索式命中的全文穷举。允许的表述是：

> 使用预先冻结的确定性分层、种子引文网络、PRESS和召回验证完成证据饱和型系统证据图。

不应表述为：

> 下载并人工筛选了OpenAlex中的全部相关记录。

### 22.3 第12轮不是原始双零

允许：

> 第12轮后冻结扩展为0/0；未启动第13轮。

必须同时披露：

> 第12轮内部实际新增10个术语家族和9个指标家族；停止依据是回顾性的边际收益判断，原连续三轮双零规则未满足。

### 22.4 独立AI不等于真人

单独Codex任务提供了流程分离、输入隔离和可审计来源，但它仍是AI复核。论文中不能将其写成“两位真人编码者”。应准确写为：

- 七个早期工作表经人工复核并确认采用；
- 后续H1/H2门槛依据所有者修订，由独立Codex任务完成；
- 所有AI复核均保留哈希、运行ID和逐行证据。

### 22.5 最终预测维度只有1个

总共保留4个操作维度，但 `D=1` 只统计实质预测维度。合作维度是机会变量，作者、年份和期刊维度是控制变量。不能为了“看起来维度更多”而把机会和控制计入核心创新维度。

## 23. 如何回答审稿人

### 23.1 为什么是42个检索领域？

不是先选择42。它们由3,615个活跃英文术语经过来源验证、规范化、术语家族去重、角色分离编码、H2拆并、PRESS和62/62种子召回共同形成。任何没有直接来源证据的领域不能建立。

### 23.2 为什么是336条逻辑检索式？

只有构念关系或证据角色不同才形成独立逻辑式。同义词共享一个OR块，完全覆盖且无独立构念作用的式子归档，零命中式保留但不活动。因此336是语义去重后的结果，不是数量目标。

### 23.3 为什么物理请求是367条？

OpenAlex长度和语法限制需要拆分部分逻辑式。拆分只增加API请求数 `P`，不增加逻辑语义数 `Q`。每个拆分保留父逻辑式ID和术语并集验证。

### 23.4 为什么候选维度是66个？

维度在432个规范指标家族完成后，依据构念、解释角色、信息来源、T0边界和偏差风险归纳。搜索领域没有直接转换为模型维度，聚类也没有自动决定维度。

### 23.5 为什么最后只有7个指标？

不是要求保留7个。432个指标全部接受相同14项硬门槛，425个至少失败一项，只有7个全部通过。主要失败点是公式/操作化不可复现、数据未准备、质量审计未通过、非常量验证缺失以及英文全文公式证据不足。

### 23.6 为什么严格预测维度只有1个？

一个预测维度必须同时满足：

- 至少一个指标通过全部门槛；
- 构念边界得到至少两个独立研究团队支持；
- H2确认不是其他维度的别名或参数变体。

只有“Knowledge diversity and integration potential”满足。其他三个保留维度分别属于机会或控制角色，按协议不计入 `D`。

## 24. 复核和复现入口

在项目根目录执行：

```bash
cd innovation_impact_feature_selection/evidence_derived_v3

python3 pipeline.py status
python3 pipeline.py audit
```

如需从冻结数据库重新生成最终决策和矩阵：

```bash
python3 pipeline.py derive-dimensions
python3 pipeline.py select-indicators
python3 materialize_final_training_features_v3.py
python3 pipeline.py audit
```

核对关键哈希：

```bash
sha256sum \
  outputs/final_dimensions_v3.json \
  outputs/final_feature_set_v3.json \
  outputs/final_training_features_v3/final_training_features_v3.parquet \
  outputs/final_training_features_v3/final_training_features_schema_v3.json
```

预期：

```text
c90555aa88fbe51171a52cd200a4408033e4cb60be273f35999af2174db6e532  final_dimensions_v3.json
e80420a5206988771799e5f9a4394b20f09a5a19ea537ec3df0503124dc91e6a  final_feature_set_v3.json
9282b58ea36054853710b5c0904b20b86e0a76bfa91bff3bdddc2eb401b5ff51  final_training_features_v3.parquet
717420d0d729fc76008d6f8de8e2077767de9ce7bac28e9a87fd0cd0a753eccf  final_training_features_schema_v3.json
```

## 25. 关键产物索引

| 产物 | 路径 |
|---|---|
| v3总协议 | [protocol_v3.json](../innovation_impact_feature_selection/evidence_derived_v3/protocol_v3.json) |
| 证据饱和协议 | [saturation_protocol_v3.json](../innovation_impact_feature_selection/evidence_derived_v3/saturation_protocol_v3.json) |
| 原始英文术语 | [english_raw_terms_v3.csv](../innovation_impact_feature_selection/evidence_derived_v3/outputs/english_raw_terms_v3.csv) |
| 规范术语 | [canonical_terms_v3.csv](../innovation_impact_feature_selection/evidence_derived_v3/outputs/canonical_terms_v3.csv) |
| 42个检索领域 | [search_domains_v3.csv](../innovation_impact_feature_selection/evidence_derived_v3/outputs/search_domains_v3.csv) |
| 336条逻辑式 | [logical_queries_v3.csv](../innovation_impact_feature_selection/evidence_derived_v3/outputs/logical_queries_v3.csv) |
| 367条物理式 | [physical_queries_v3.csv](../innovation_impact_feature_selection/evidence_derived_v3/outputs/physical_queries_v3.csv) |
| 冻结检索框架 | [frozen_search_frame_v3.json](../innovation_impact_feature_selection/evidence_derived_v3/outputs/frozen_search_frame_v3.json) |
| 隐藏种子检索日志 | [hidden_seed_search_log_v3.csv](../innovation_impact_feature_selection/evidence_derived_v3/outputs/hidden_seed_search_log_v3.csv) |
| PRISMA流转 | [prisma_flow_v3.csv](../innovation_impact_feature_selection/evidence_derived_v3/outputs/prisma_flow_v3.csv) |
| Crossref核验 | [crossref_validation_v3.csv](../innovation_impact_feature_selection/evidence_derived_v3/outputs/crossref_validation_v3.csv) |
| 完整指标库 | [complete_indicator_library_v3.csv](../innovation_impact_feature_selection/evidence_derived_v3/outputs/complete_indicator_library_v3.csv) |
| 指标硬门槛 | [feature_gate_decisions_v3.csv](../innovation_impact_feature_selection/evidence_derived_v3/outputs/feature_gate_decisions_v3.csv) |
| 66个候选维度 | [candidate_dimensions_v3.csv](../innovation_impact_feature_selection/evidence_derived_v3/outputs/candidate_dimensions_v3.csv) |
| 最终维度 | [final_dimensions_v3.json](../innovation_impact_feature_selection/evidence_derived_v3/outputs/final_dimensions_v3.json) |
| 最终指标 | [final_feature_set_v3.json](../innovation_impact_feature_selection/evidence_derived_v3/outputs/final_feature_set_v3.json) |
| 最终训练矩阵 | [final_training_features_v3.parquet](../innovation_impact_feature_selection/evidence_derived_v3/outputs/final_training_features_v3/final_training_features_v3.parquet) |
| 审计报告 | [audit_report_v3.md](../innovation_impact_feature_selection/evidence_derived_v3/outputs/audit_report_v3.md) |
| 审计清单 | [audit_manifest_v3.json](../innovation_impact_feature_selection/evidence_derived_v3/outputs/audit_manifest_v3.json) |
| 完成矩阵 | [completion_matrix_v3.csv](../innovation_impact_feature_selection/evidence_derived_v3/outputs/completion_matrix_v3.csv) |

## 附录A：42个最终检索概念领域

| ID | 英文领域标签 |
|---|---|
| SD001 | Article structure and research design |
| SD002 | Article topics and field context |
| SD003 | Authorship and collaboration features |
| SD004 | Authorship, collaboration, and social structure |
| SD005 | Bias, spin, and research integrity |
| SD006 | Citation and scholarly impact validation |
| SD007 | Computational and bibliometric measurement |
| SD008 | Cross-domain publication-time paper features |
| SD009 | Empirical evaluation practice |
| SD010 | Equity and evaluation bias |
| SD011 | Innovation and transformative validation outcomes |
| SD012 | Interdisciplinarity and knowledge diversity |
| SD013 | Knowledge recombination and exploratory search |
| SD014 | Measurement-quality validation |
| SD015 | Methodological quality and evidence appraisal |
| SD016 | Novelty, innovation, and transformative science |
| SD017 | Open science and data accessibility |
| SD018 | Paper novelty and interdisciplinarity |
| SD019 | Paper quality and peer-review constructs |
| SD020 | Paper retrieval and similarity signals |
| SD021 | Public involvement in research |
| SD022 | Publication and peer-review validation |
| SD023 | Publication-time dissemination and outreach |
| SD024 | Publication venue and access context |
| SD025 | Quality and model-performance validation |
| SD026 | Reference-based paper features |
| SD027 | Replication and reproducibility |
| SD028 | Reported study-result validation |
| SD029 | Research difficulty and complexity |
| SD030 | Research-question design |
| SD031 | Research rigor, reporting, and reproducibility |
| SD032 | Researcher career and institutional context |
| SD033 | Sampling representation and research attention |
| SD034 | Scholarly impact forecasting |
| SD035 | Scholarly impact validation outcomes |
| SD036 | Societal-benefit reporting |
| SD037 | Societal, translational, and market validation |
| SD038 | Systematic-review search and evidence sources |
| SD039 | Title, abstract, and language features |
| SD040 | Trust and credibility validation |
| SD041 | Venue, funding, and institutional opportunity |
| SD042 | Visibility, attention, and diffusion validation |

这些是检索概念领域，不是最终模型维度。部分标签在自然语言上相近，但承担不同的构念、证据角色或T0边界，因此经H2和PRESS确认后保持分开。

## 附录B：66个候选维度的最终处置

| ID | 候选维度 | 角色 | 状态 | 最终指标 | 原因 |
|---|---|---|---|---|---|
| CD001 | Abstract availability and completeness | predictive | 淘汰 | — | 无指标通过全部硬门槛 |
| CD002 | Abstract length and information density | predictive | 淘汰 | — | 无指标通过；不足两个独立团队 |
| CD003 | Abstract rhetoric and semantic composition | predictive | 淘汰 | — | 无指标通过全部硬门槛 |
| CD004 | Access, cost, and dissemination opportunity | opportunity | 淘汰 | — | 无指标通过全部硬门槛 |
| CD005 | Article structure and format | predictive | 淘汰 | — | 无指标通过全部硬门槛 |
| CD006 | Author affiliation and geographic context | control | 淘汰 | — | 无指标通过全部硬门槛 |
| CD007 | Author credentials and career capital | opportunity | 淘汰 | — | 无指标通过全部硬门槛 |
| CD008 | Author geographic and linguistic position | opportunity | 淘汰 | — | 无指标通过全部硬门槛 |
| CD009 | Author identity and authorship position | opportunity | 淘汰 | — | 无指标通过全部硬门槛 |
| CD010 | Author team composition context | control | 保留 | EF0038 | 通过 |
| CD011 | Bias and confounding sensitivity | sensitivity | 淘汰 | — | 无指标通过全部硬门槛 |
| CD012 | Bibliographic and knowledge-network position | opportunity | 淘汰 | — | 无指标通过全部硬门槛 |
| CD013 | Citation and reference integrity | predictive | 淘汰 | — | 无指标通过全部硬门槛 |
| CD014 | Collaboration breadth and type | opportunity | 保留 | EF0186、EF0188 | 通过 |
| CD015 | Collaboration network position | opportunity | 淘汰 | — | 无指标通过全部硬门槛 |
| CD016 | Confounding-adjustment context | control | 淘汰 | — | 无指标通过；不足两个独立团队 |
| CD017 | Data-access context | control | 淘汰 | — | 无指标通过；不足两个独立团队 |
| CD018 | Data, code, and protocol transparency | predictive | 淘汰 | — | 无指标通过全部硬门槛 |
| CD019 | Domain plausibility and translational relevance | predictive | 淘汰 | — | 无指标通过全部硬门槛 |
| CD020 | Ethics, governance, and reflexivity | predictive | 淘汰 | — | 无指标通过全部硬门槛 |
| CD021 | Evaluation design and appraisal practice | predictive | 淘汰 | — | 无指标通过；不足两个独立团队 |
| CD022 | Evidence consistency and generalizability | sensitivity | 淘汰 | — | 无指标通过全部硬门槛 |
| CD023 | Funding and industry context | control | 淘汰 | — | 无指标通过全部硬门槛 |
| CD024 | Funding resources and sponsorship | opportunity | 淘汰 | — | 无指标通过全部硬门槛 |
| CD025 | General paper and study quality | predictive | 淘汰 | — | 无指标通过；不足两个独立团队 |
| CD026 | Indexing and semantic discoverability | predictive | 淘汰 | — | 无指标通过全部硬门槛 |
| CD027 | Institutional prestige and research capacity | opportunity | 淘汰 | — | 无指标通过全部硬门槛 |
| CD028 | Interdisciplinarity measurement sensitivity | sensitivity | 淘汰 | — | 无指标通过；不足两个独立团队 |
| CD029 | Interdisciplinary knowledge integration | predictive | 淘汰 | — | 无指标通过全部硬门槛 |
| CD030 | Interdisciplinary team composition | predictive | 淘汰 | — | 无指标通过；不足两个独立团队 |
| CD031 | Knowledge diversity and integration potential | predictive | 保留 | EF0312、EF0318 | 通过 |
| CD032 | Knowledge recombination and concept emergence | predictive | 淘汰 | — | 无指标通过；不足两个独立团队 |
| CD033 | Measurement validity and reliability | sensitivity | 淘汰 | — | 无指标通过全部硬门槛 |
| CD034 | Method and procedure documentation | predictive | 淘汰 | — | 无指标通过全部硬门槛 |
| CD035 | Methodological integration and adaptation | predictive | 淘汰 | — | 无指标通过；不足两个独立团队 |
| CD036 | Novelty comparison context | control | 淘汰 | — | 无指标通过；不足两个独立团队 |
| CD037 | Outcome and endpoint specification | predictive | 淘汰 | — | 无指标通过全部硬门槛 |
| CD038 | Peer-review and consensus process | predictive | 淘汰 | — | 无指标通过；不足两个独立团队 |
| CD039 | Public-engagement policy opportunity | opportunity | 淘汰 | — | 无指标通过；不足两个独立团队 |
| CD040 | Publication and funding environment | opportunity | 淘汰 | — | 无指标通过；不足两个独立团队 |
| CD041 | Publication and venue context | control | 保留 | EF0197、EF0307 | 通过 |
| CD042 | Publication language and geographic-position potential | predictive | 淘汰 | — | 无指标通过；不足两个独立团队 |
| CD043 | Publication policy and editorial process | opportunity | 淘汰 | — | 无指标通过全部硬门槛 |
| CD044 | Reference knowledge profile | predictive | 淘汰 | — | 无指标通过全部硬门槛 |
| CD045 | Reference knowledge recency | predictive | 淘汰 | — | 无指标通过全部硬门槛 |
| CD046 | Reference-volume context | control | 淘汰 | — | 无指标通过全部硬门槛 |
| CD047 | Reporting completeness and guideline adherence | predictive | 淘汰 | — | 无指标通过全部硬门槛 |
| CD048 | Reproducibility and research integrity | predictive | 淘汰 | — | 无指标通过全部硬门槛 |
| CD049 | Research question and theoretical alignment | predictive | 淘汰 | — | 无指标通过；不足两个独立团队 |
| CD050 | Review reliability and bias sensitivity | sensitivity | 淘汰 | — | 无指标通过全部硬门槛 |
| CD051 | Sample and population adequacy | predictive | 淘汰 | — | 无指标通过全部硬门槛 |
| CD052 | Scientific novelty and originality | predictive | 淘汰 | — | 无指标通过全部硬门槛 |
| CD053 | Scientific writing clarity and style | predictive | 淘汰 | — | 无指标通过全部硬门槛 |
| CD054 | Stakeholder engagement and accessibility | predictive | 淘汰 | — | 无指标通过；不足两个独立团队 |
| CD055 | Statistical robustness and precision | sensitivity | 淘汰 | — | 无指标通过全部硬门槛 |
| CD056 | Study design and methodological rigor | predictive | 淘汰 | — | 无指标通过全部硬门槛 |
| CD057 | Study design and sample context | control | 淘汰 | — | 无指标通过全部硬门槛 |
| CD058 | Study setting and scale opportunity | opportunity | 淘汰 | — | 无指标通过全部硬门槛 |
| CD059 | Systematic-review search and protocol rigor | predictive | 淘汰 | — | 无指标通过全部硬门槛 |
| CD060 | Systematic-review synthesis and appraisal | predictive | 淘汰 | — | 无指标通过全部硬门槛 |
| CD061 | Team demographic diversity | predictive | 淘汰 | — | 无指标通过；不足两个独立团队 |
| CD062 | Title and keyword design | predictive | 淘汰 | — | 无指标通过全部硬门槛 |
| CD063 | Topic and field normalization context | control | 淘汰 | — | 无指标通过全部硬门槛 |
| CD064 | Topic and knowledge-content profile | predictive | 淘汰 | — | 无指标通过全部硬门槛 |
| CD065 | Topic, field, and community opportunity | opportunity | 淘汰 | — | 无指标通过全部硬门槛 |
| CD066 | Venue prestige and disciplinary position | opportunity | 淘汰 | — | 无指标通过全部硬门槛 |

## 附录C：关键哈希

| 对象 | SHA-256 |
|---|---|
| 总协议 `protocol_v3.json` | `82cb4803f13ec509667b47ac54da7b187243f0d669dea30f46a1e7a1557f5e9e` |
| 独立AI复核修订 | `a3562b9eadc17bde34a7de0a7e09afe15679257c3b7980471270e0063df9422f` |
| 第12轮务实停止修订 | `077cde51a534e66af4462b24f88bb8f3f9fec1e34b71d066e729bed35d9a513c` |
| 第12轮固定正式队列修订 | `bf7ff5447859298776c96fa222b8139f13d1da63b8bfe1fadb5c66b8345f268a` |
| 第12轮对外口径说明 | `89e6bd90371fd1724ac01809cd30b9978a5f181d5f9000289af23653debd0da2` |
| 公式—操作化分层修订 | `87910fa2a5ff41648b6a8565ee7b89ce438069304daa64b491b9aac6bb55ca6e` |
| 定向公式补全修订 | `b6e1556fc3c33d530633920eb7ac11a6321346d72b30d0c062455f5f1fd8fd2c` |
| 实现快照版本化修订 | `6c15d9b243cd82474baf94753c903596cca94b28a130b8eb491aa4af7c5b096d` |
| 冻结检索框架文件 | `3216296183732e244ecf83377d965f6242aa5ae521cdea89695aa79274763eb5` |
| 冻结检索框架语义哈希 | `088ca35e634187e5267b0bcbb09c4ad23f7e40d63d6bcfd17bb250584a83046e` |
| 最终维度JSON | `c90555aa88fbe51171a52cd200a4408033e4cb60be273f35999af2174db6e532` |
| 最终指标JSON | `e80420a5206988771799e5f9a4394b20f09a5a19ea537ec3df0503124dc91e6a` |
| 最终训练Parquet | `9282b58ea36054853710b5c0904b20b86e0a76bfa91bff3bdddc2eb401b5ff51` |
| 最终训练Schema | `717420d0d729fc76008d6f8de8e2077767de9ce7bac28e9a87fd0cd0a753eccf` |
| 最终审计报告 | `c46f0c2b57c671fbb4250d569b18ce636e9a0b7ec86c71da94b2a28ca21a3b3c` |
| 确定性决策哈希 | `1b5bdeb08308a82686feb9c6620504692fd0f3f03f5f100954e9042f2f48ebe6` |

审计清单本身包含生成时间，因此整文件哈希可随重新审计而变化；复现判断应使用确定性决策哈希及冻结核心产物哈希。

## 26. 最终结论

本次36小时30分钟执行完成了从“固定机制和指标数量”到“证据驱动、数量自然产生”的转换：

```text
42个检索概念领域
→ 336条逻辑检索式
→ 367条OpenAlex物理请求
→ 62/62种子召回
→ 9,515条正式筛选终态
→ 363篇纳入来源
→ 1,685个指标提及
→ 432个规范指标家族
→ 66个候选维度
→ 4个保留操作维度
→ 1个核心预测维度
→ 7个最终指标
→ 118,059行可训练特征矩阵
```

最终数量没有按维度配额、经验偏好或模型预测性能产生。每个数量都可以沿“文献来源—术语—检索领域—逻辑式—检索记录—筛选决定—指标证据—维度编码—硬门槛—最终矩阵”的链条追溯。

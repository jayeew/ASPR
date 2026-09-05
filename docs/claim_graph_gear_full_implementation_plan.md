# ASPR–GEAR 原生 Claim Graph：完整设计与逐步实施手册

> 本文档综合当前项目数据条件、仓库代码边界以及已经确认的设计决策，给出从零开始实现 Claim Graph + GEAR 创新性评价的完整工程方案。
>
> 当前状态：**设计锁定，尚未按本文档完整实现。**
>
> 本版目标：先把数据、两层图、Claim 抽取、Claim Graph、在线插入、结构指标和 GEAR 融合全部跑通，并证明 Graph 确实产生了 GEAR 单独无法给出的结构信息。**本版不封存测试集，不使用评审意见构造标签，不设计发布版本，不建立 SHA-256 Manifest，不复用旧 paper-level Graph 分数。**

---

## 阅读导航

这份文档按实际实施顺序组织：

1. 第 1–6 节：研究问题、锁定决策、数据边界和整体架构；
2. 第 7–9 节：目录约定、Paper Citation Graph 和摘要抽取；
3. 第 10–17 节：Claim 契约、embedding、Claim edges、社区和历史组合统计；
4. 第 18–28 节：新论文插入、结构扰动指标、旧指标迁移和 Graph profiles；
5. 第 29–33 节：GEAR 双通道融合、自然语言生成、示例和 Graph 有效性论证；
6. 第 34–41 节：逐阶段命令、配置、伪代码、测试、风险与实施检查表；
7. 第 42–44 节：论文方法定位、最终系统输出和成功标准。

若直接开始编码，先阅读第 4 节锁定决策，再按第 34 节 Phase 0→10 顺序实施。

---

## 1. 一句话概括整个方案

本项目不再把 paper-level Graph 分数按权重分配给论文中的 Claims，而是：

1. 用本地 OpenAlex Works 快照，为 24,922 篇 2023–2025 Nature 论文及其两跳历史参考文献构建一个纯引用关系的 **Paper Citation Graph**；
2. 只从这 24,922 篇 Nature 论文的本地 Markdown 摘要中抽取 1–3 条标准化、原子化、可绑定原句的贡献 Claim；
3. 将五类 Claim 放入同一张具有时间方向的 **原生 Claim Graph**；Claim 类型只是节点字段，允许跨类型 Claim 边；
4. 对待评论文提取同样结构的 Claims，将其临时插入 Claim Graph，计算它对历史 Claim 结构造成的局部扰动；
5. 让 GEAR 独立回答“现有文献是否已经做过这件事”，让 Claim Graph 独立回答“这条贡献位于怎样的知识结构中、连接了什么、改变了什么”；
6. 最后将 GEAR 的证据化 prior-art 结论与 Claim Graph 的结构画像合并，形成论文创新性评价。

核心研究主张不是“GEAR 后面再加一个 Graph 分数”，而是：

> **GEAR 提供基于文本证据的局部先例判断，Claim Graph 提供基于历史结构的全局位置、知识重组和插入扰动判断。二者回答不同问题，联合后才能区分直接先例、渐进扩展、局部新颖、跨社区桥接和结构重组型创新。**

---

## 2. 为什么旧方案不成立

旧方案大致是：

```text
paper-level Graph score
        ↓
按照 Claim attribution weight 分配
        ↓
claim-level graph score
```

如果一篇论文的 paper-level Graph 信号对所有 Claims 都是同一个常数 `S_p`，而每条 Claim 的结果是：

\[
S_{p,c}=S_p\times w_{p,c}
\]

那么在同一篇论文内部：

\[
S_{p,c_1}>S_{p,c_2}\iff w_{p,c_1}>w_{p,c_2}
\]

也就是说，Graph 只改变数值尺度，不改变 Claim 排序。最终 Claim 排序本质上仍由 attribution weight 决定，Graph 没有提供 Claim 级差异信息。

因此，本方案明确废弃：

- paper-level Graph 分数向 Claim 下发；
- 用 attribution weight 伪造 Claim 级 Graph 差异；
- 旧 HGB 输出作为 Claim 创新性证据；
- 把 Graph opportunity/control 字段直接解释为 Claim novelty；
- 把旧 `gear/claim_attribution.py` 或旧 structural scorer 重新包装成新 Claim Graph。

新方案的每条 Claim 都拥有自己的邻居、社区位置、历史组合频率和插入扰动，因此同一论文中的 Claims 可以获得真正不同、由 Claim 自身决定的 Graph 结构信号。

---

## 3. 论文的核心研究出发点

### 3.1 GEAR 单独能够做什么

GEAR 对一条目标 Claim 进行 prior-art 检索、候选筛选、证据 Span 对齐和语义关系分类，能够回答：

- 是否存在基本完整覆盖目标 Claim 的直接先例；
- 是否存在部分先例；
- 目标工作是否是已有工作的扩展；
- 两者是平行、支持、冲突还是距离较远；
- 哪些共同维度和差异维度可以由具体文本证据支持。

当前仓库的关系类型包括：

- `DIRECT_ANTECEDENT`
- `PARTIAL_ANTECEDENT`
- `EXTENSION`
- `PARALLEL`
- `SUPPORT`
- `CONFLICT`
- `DISTANT`
- `UNRESOLVED`

这些关系由 `gear/prior_art.py` 中的成对文本证据比较流程产生。

### 3.2 GEAR 单独缺少什么

GEAR 本质上是局部检索和成对比较系统。即使检索质量很好，它仍然缺少：

- 一个持久的历史 Claim 坐标系；
- Claim 位于哪个知识社区的结构信息；
- Claim 是否同时连接多个此前分离的知识社区；
- Claim 是否连接了历史上罕见的社区组合；
- Claim 插入以后是否合并局部分量、缩短路径或跨越结构洞；
- 同一篇论文中哪条 Claim 具有更强的结构重组作用。

GEAR 可以判断“相似或不同”，但不能仅凭若干检索结果可靠地判断“这些知识原来分布在几个历史社区，它们此前是否被同一 Claim 连接过”。

### 3.3 Claim Graph 单独能够做什么

Claim Graph 能够给出：

- 目标 Claim 最近的历史 Claim 邻居；
- 邻居在历史 Claim 社区中的分布；
- 目标 Claim 是否位于单一成熟社区内部；
- 目标 Claim 是否跨越多个社区；
- 被跨接的社区是否距离较远；
- 相同社区组合在严格历史数据中是否常见；
- 加入目标 Claim 前后，局部组件、路径、边界和结构洞发生什么变化；
- 这些 Claim 关系是否同时得到父论文引用路径的支持。

### 3.4 Claim Graph 单独不能做什么

Claim Graph 的边是基于语义相似和论文引用路径生成的候选关系，不是科学关系真值。它不能独立证明：

- 历史 Claim 是目标 Claim 的直接先例；
- 目标 Claim 一定正确；
- 目标 Claim 一定比邻居更优；
- 全世界此前从未出现过该 Claim；
- 一条跨社区边一定构成科学创新。

因此正确结构是双通道，而不是串行污染：

```text
标准化贡献 Claim
        ├── GEAR prior-art 通道
        │     └── 证据化语义关系卡
        │
        └── Claim Graph 插入通道
              └── 历史位置与结构扰动卡

证据化语义关系卡 + 结构扰动卡
                  ↓
             联合创新性评价
```

Graph 构建过程完全独立于 GEAR。Graph 不读取 GEAR 的 relation label、verdict 或 prior-art 判断；GEAR 也不决定 Claim Graph 中有哪些边。

---

## 4. 本版已经锁定的设计决策

以下内容视为当前实现基线，不应在编码过程中悄悄改变。

| 问题 | 锁定决策 |
|---|---|
| 图层数量 | 两层：Paper Citation Graph + Recent Nature Claim Graph |
| 旧 411,490 篇论文图 | 本方案不用；不要为了复用而强行接入 |
| Nature 目标论文 | 本地 2023–2025 年 24,922 篇 Nature 论文 |
| Paper Graph 邻域 | 每篇 Nature 目标论文向后展开两跳：P→R1→R2 |
| Paper Graph 节点 | Nature 目标论文、R1、R2；只保存 OpenAlex 元数据 |
| Paper Graph 边 | 只有真实有向 `CITES` 边 |
| 六跳邻域 | 不采用；small-world 下会快速膨胀且没有 Claim 建图所需的额外语义收益 |
| Claim 来源 | 只使用本地 Nature 论文 Markdown 的摘要 |
| OpenAlex 摘要 | 不作为 Claim 抽取首选，也不要求作为后备；`abstract_inverted_index` 并非每篇都有，不能 100% 恢复 |
| 全文送给 LLM | 禁止；先确定性抽取全部摘要，再逐篇把标题和编号摘要句送给一次受约束 LLM |
| Claim 数量 | 每篇 1–3 条 |
| Claim 绑定 | 每条 Claim 必须绑定原摘要句和精确原文片段 |
| LLM 失败 | 放弃整篇论文的 Claim，写入失败 CSV；不设置 `limited` 等多状态 |
| Claim 类型 | METHOD、FINDING、MECHANISM、RESOURCE、THEORY |
| Claim 类型数量 | 每条 Claim 恰好一个主类型 |
| 跨类型 Claim 边 | 允许；类型只是节点属性，不作为候选边过滤条件 |
| 同论文 Claim 边 | 不建立 |
| Claim→Reference | 不建立 |
| Claim→Paper | 不作为图边；`parent_paper_id` 只是节点字段/外键 |
| Claim Graph 边标签 | 不预标 REPEAT/EXTENSION/COMBINATION；统一为连续证据候选边 |
| Claim 边方向 | 严格从较早 Claim 指向较晚 Claim |
| Claim 边生成来源 | 全图语义近邻 + 父论文引用路径；两者均允许跨类型连接 |
| OpenScholar reranker | 不使用；BGE-M3 完成向量召回和候选论文内 Claim 选择 |
| Paper path 的作用 | 为 Claim 候选边提供独立的书目路径支持，不作为科学关系验证 |
| 社区发现 | 全部 Claim 共同构建一张语义 mutual-kNN backbone，并在整张图上运行 Leiden |
| Paper path 边参与社区 | 不参与主社区构建，只作为 overlay 支持信息 |
| GEAR 与 Graph 构建关系 | 完全独立；Graph 中不出现“经 GEAR 验证的边” |
| 新论文插入 | 临时插入，只服务当前评价，不自动永久写入历史图 |
| 最终输出 | GEAR 关系卡 + Graph 结构画像；第一版不构造复杂加权总分 |
| 测试集 | 当前不封存；先完成全流程和功能验证，正式效果测试后置 |
| 评审意见 | 不作为 Claim Graph 标签，不参与建图/调参；只在正式效果测评中作为保留 reviewer 身份的外部参照 |
| 数据工程 | 使用约定文件名和目录；不建立 release/版本/Manifest/SHA 身份体系 |

---

## 5. 数据资产与真实约束

### 5.1 Nature 论文 Markdown

本地清单：

```text
/mnt/d/aspr_nature_markdown/manifest.jsonl
```

当前已知：

- 24,922 条唯一论文/评审配对；
- 年份范围 2023–2025；
- 每条记录包含 `article_id`、`paper_markdown_path`、`year` 等字段；
- 建图和系统生成只读取 `paper_markdown_path`，不读取 peer-review Markdown；正式效果测评阶段才独立读取 reviewer reports；
- 论文 Markdown 含完整论文内容，但 Claim 抽取阶段只截取摘要。

示例清单字段：

```json
{
  "article_id": "s41467-023-35783-y",
  "paper_markdown_path": "/mnt/d/aspr_nature_markdown/paper/s41467-023-35783-y.md",
  "peer_review_markdown_path": "/mnt/d/aspr_nature_markdown/peer_review/s41467-023-35783-y_r.md",
  "year": 2023
}
```

### 5.2 OpenAlex Works 快照

本地快照：

```text
/mnt/d/FabCitationData/openalex-snapshot
```

当前已知：

- 2,127 个 gzip Works 包；
- 约 595.29 GiB；
- 约 492,361,307 个 Works；
- Work 记录可包含 DOI、标题、发表日期、`referenced_works`、topics/fields、来源信息及 `abstract_inverted_index`。

本方案只依赖本地快照，不需要额外联网下载论文或抓取海量全文。

### 5.3 OpenAlex Work ID SQLite 索引

本地索引：

```text
/home/jayee/workspace/FabCitation/openalex_snapshot_reference_check_results/analysis_state.db
```

当前已知：

- 约 52.57 GB；
- `works_index` 包含 492,361,307 个 Work ID；
- 2,127 个 gzip 包均已完成索引；
- 主要表：

```text
works_index(short_id TEXT PRIMARY KEY, full_id TEXT NOT NULL, indexed_at TEXT NOT NULL)
index_file_state(...)
metadata(...)
```

这个数据库只能高效回答：

- 某个 OpenAlex Work ID 是否存在；
- short ID 对应的 full ID 是什么。

它不能回答：

- DOI 对应哪个 Work ID；
- Work 位于哪个 gzip 文件或字节偏移；
- Work 的标题、摘要、发表日期或参考文献；
- Work→Reference 边。

因此它只能用于 R1/R2 ID 存在性过滤，不能替代 OpenAlex 原始快照扫描。

### 5.4 已有 411,490 篇历史 paper 图

旧数据：

```text
data/knowledge_corpus/nature_multihorizon_v6_1_uncapped_v2
```

它主要覆盖 1980–2022，和 2023–2025 的 24,922 篇 Nature Markdown 在既有 DOI 审计中没有可用交集。该结果是时间范围割裂导致的，不代表任何一侧数据无效。

本方案明确不再依赖这 411,490 篇数据，不尝试把它强行变成 Claim Graph 的历史全文来源。

### 5.5 本地向量模型

可用模型：

```text
/home/jayee/models/bge-m3
```

用途：

- Claim embedding；
- 全部历史 Claim 的语义近邻检索；
- Paper path 候选论文内部的最相似 Claim 选择，不按类型预过滤。

不使用：

```text
/home/jayee/models/OpenScholar_Reranker
```

原因：

- 它是 reranker，不是生成模型；
- Claim 抽取仍需要一次受约束的生成模型；
- Claim 候选边已有 BGE-M3 连续相似度，不再增加第二个 reranker 调用和额外计算链。

### 5.6 生成模型现实条件

当前环境中未确认存在可直接使用的本地生成模型。因此 Claim 抽取实现应复用 GEAR 的惰性 JSON model client 接口，或接入后续配置的本地/远程生成模型。

约束：

- 客户端只能在真正调用时创建，不能 import 时创建；
- 每篇论文最多一次 Claim 抽取调用；
- 不把全文送给生成模型；
- 生成端不可用时，该论文写入失败 CSV 并跳过；
- 不使用启发式 Claim 作为生成模型失败后的伪回退。

---

## 6. 总体架构

```text
                    本地 OpenAlex Works 快照
                              │
             ┌────────────────┴────────────────┐
             │                                 │
     DOI 匹配 24,922 Nature             两跳参考关系扫描
             │                                 │
             └──────────────┬──────────────────┘
                            ▼
                 Layer 1: Paper Citation Graph
           Nature P + R1 + R2；唯一边类型 CITES
                            │
                   提供父论文路径 motif
                            │
                            ▼
本地 Nature Markdown ──► 摘要抽取/切句 ──► 一次受约束 LLM
                                             │
                                             ▼
                               1–3 条原子贡献 Claim/论文
                                             │
                                  BGE-M3 embedding
                                             │
                    ┌────────────────────────┴─────────────────────┐
                    │                                              │
              全图语义近邻                               父论文路径候选
                    │                                              │
                    └──────────────────────┬───────────────────────┘
                                           ▼
                    Layer 2: Type-Attributed Temporal Claim Graph
                    五类 Claim 共存且允许跨类型边的一张图
                                           │
                              Leiden 社区 + 历史统计
                                           │
                                           ▼
                                  新论文 Claim 临时插入
                                           │
                             Claim 结构位置/扰动画像
                                           │
               ┌───────────────────────────┴────────────────────────┐
               │                                                    │
      GEAR 独立 prior-art 关系卡                          Graph 结构画像卡
               │                                                    │
               └───────────────────────────┬────────────────────────┘
                                           ▼
                                Claim 级联合创新性评价
                                           │
                                           ▼
                                 Paper 级创新性综合说明
```

---

## 7. 目录和中间文件约定

所有新数据放在：

```text
data/claim_graph/
```

不创建 release 目录，不区分 candidate/frozen 版本，不生成内容哈希 Manifest。各阶段通过固定文件名衔接。

建议目录：

```text
data/claim_graph/
├── nature_targets.parquet
├── nature_target_failures.csv
├── paper_nodes.parquet
├── paper_edges.parquet
├── paper_graph_stats.json
├── abstracts.parquet
├── abstract_sentences.parquet
├── abstract_failures.csv
├── claim_nodes.parquet
├── claim_failures.csv
├── claim_embeddings.npy
├── claim_embedding_index.parquet
├── semantic_claim_edges.parquet
├── paper_path_claim_edges.parquet
├── claim_edges.parquet
├── claim_edge_type_stats.parquet
├── claim_communities.parquet
├── community_profiles.parquet
├── community_pair_history.parquet
├── historical_insertion_profiles.parquet
├── evaluation/
├── build_state.sqlite
├── logs/
└── chunks/
```

约定：

- Parquet 保存大表；
- CSV 只保存体量较小、需要人工查看的失败记录；
- `.npy` 保存连续向量矩阵；
- `claim_embedding_index.parquet` 保存 `claim_id → embedding_row`；
- `claim_edge_type_stats.parquet` 保存 25 种 `earlier_claim_type → later_claim_type` 组合的边数、占比、平均 cosine 和来源构成；它是审计统计，不是另一张图；
- `build_state.sqlite` 只记录断点状态和已完成文件/论文，不存业务主表；
- 日志使用中文，写入 `data/claim_graph/logs/`；
- 中间 chunk 写入 `data/claim_graph/chunks/`，最终合并后可保留，方便断点恢复。

建议新增代码：

```text
gear/claim_graph/
├── __init__.py
├── contracts.py
├── paper_graph.py
├── abstracts.py
├── extraction.py
├── embeddings.py
├── edges.py
├── communities.py
├── insertion.py
├── metrics.py
└── fusion.py

scripts/claim_graph/
├── 01_prepare_nature_targets.py
├── 02_build_paper_graph.py
├── 03_extract_abstracts.py
├── 04_extract_claims.py
├── 05_embed_claims.py
├── 06a_build_semantic_claim_edges.py
├── 06b_build_paper_path_claim_edges.py
├── 06c_merge_claim_edges.py
├── 07_build_claim_communities.py
├── 08_build_claim_statistics.py
├── 09_demo_insert_paper.py
└── eval/
    ├── 01_prepare_evaluation_papers.py
    ├── 02_extract_reviewer_novelty_judgments.py
    ├── 03_run_system_conditions.py
    ├── 04_prepare_blind_packets.py
    ├── 05_audit_graph_facts.py
    └── 06_compute_metrics.py

configs/gear/claim_graph.yaml

tests/gear/claim_graph/
├── test_contracts.py
├── test_abstracts.py
├── test_claim_extraction.py
├── test_paper_graph.py
├── test_claim_edges.py
├── test_communities.py
├── test_insertion.py
├── test_metrics.py
└── test_fusion.py
```

这些是计划路径，实际实现时应遵守仓库规则：函数带类型标注、Pydantic `extra="forbid"`、使用 `pathlib.Path`、模型客户端惰性创建。

### 7.1 模块职责

| 模块 | 只负责什么 | 明确不负责什么 |
|---|---|---|
| `contracts.py` | Claim、Edge、Community、InsertionProfile 契约 | 数据扫描和模型推理 |
| `paper_graph.py` | Paper nodes/edges、三种 path motif | Claim 语义关系判断 |
| `abstracts.py` | Markdown 摘要定位、清理、切句 | LLM Claim 抽取 |
| `extraction.py` | 一次受约束 LLM、绑定验证 | prior-art 检索和 Graph 指标 |
| `embeddings.py` | BGE 文本模板、批量编码、向量索引 | Claim 类型决定和关系标签 |
| `edges.py` | temporal semantic edges、Paper path 候选、合并 | Leiden 和 GEAR verdict |
| `communities.py` | mutual-kNN、Leiden、centroid、社区距离 | runtime prior-art |
| `metrics.py` | 历史 pair 统计和插入扰动公式 | 自然语言自由推断 |
| `insertion.py` | 新 Claim 临时插入、邻域和 profile | 自动持久化历史节点 |
| `fusion.py` | GEAR 卡与 Graph 卡的显式联合逻辑 | 重做 GEAR 检索或修改 ClaimGraph edges |

### 7.2 模块间通信原则：只通过约定的落盘数据

除同一模块内部的小函数外，前后阶段不通过隐式 Python 对象、全局变量、内存 cache 或直接调用上游模型来传递业务结果。每个阶段都应表现为：

```text
显式输入文件 + 显式配置
             ↓
        单一阶段程序
             ↓
显式输出文件 + 中文统计日志 + 失败表
```

具体规则：

1. 每个脚本只读取命令行列出的输入路径，不自行猜测或扫描其他阶段目录；
2. 每个脚本只写自己的输出文件，不修改上游文件；
3. 下游只依赖落盘 schema，不导入上游脚本中的临时类或内部状态；
4. 大模型、BGE、FAISS、OpenAlex 扫描和 Leiden 都由各自阶段显式执行，下游不得因缺文件而偷偷重新执行上游；
5. 失败记录写到该阶段约定的 CSV，不在主表中发明多种模糊状态；
6. `--resume` 只读取本阶段的 `build_state.sqlite`/chunk，不把断点状态当业务数据；
7. runtime 只消费已经完成的历史 Graph assets，不能在审稿过程中触发 OpenAlex 扫描、历史 Claim 抽取或全图重建；
8. fusion 只读取 GEAR RelationCards 和 Graph FactCards，不直接读取 FAISS、Leiden 或模型内部对象。

这不是 release/version/Manifest 体系。只需固定文件名、字段、主键、空值语义和生产者/消费者。

### 7.3 阶段落盘数据契约

| 阶段 | 只读输入 | 只写输出 | 下游消费者 |
|---|---|---|---|
| Nature target 准备 | 本地 Markdown manifest | `nature_targets.parquet`、`nature_target_failures.csv` | Paper Graph、摘要抽取 |
| Paper Graph | `nature_targets.parquet`、OpenAlex snapshot、ID SQLite | `paper_nodes.parquet`、`paper_edges.parquet`、`paper_graph_stats.json` | Paper-path 候选 |
| 摘要抽取 | `nature_targets.parquet`、Markdown 文件 | `abstracts.parquet`、`abstract_sentences.parquet`、`abstract_failures.csv` | Claim 抽取 |
| Claim 抽取 | `abstracts.parquet`、`abstract_sentences.parquet` | `claim_nodes.parquet`、`claim_failures.csv` | embedding、Graph 构建 |
| Claim embedding | `claim_nodes.parquet` | `claim_embeddings.npy`、`claim_embedding_index.parquet` | 语义边、Paper-path Claim 选择、社区、runtime 检索 |
| 语义候选边 | Claim nodes + embeddings/index | `semantic_claim_edges.parquet` | 边合并 |
| Paper-path 候选边 | Claim nodes + embeddings/index + Paper nodes/edges | `paper_path_claim_edges.parquet` | 边合并 |
| 边合并 | 两张候选边表 + Claim nodes | `claim_edges.parquet`、`claim_edge_type_stats.parquet` | 历史统计、插入指标 |
| 社区构建 | Claim nodes + embeddings/index | `claim_communities.parquet`、`community_profiles.parquet` | 历史统计、runtime 插入 |
| 历史统计 | Claim nodes + merged edges + communities | `community_pair_history.parquet`、`historical_insertion_profiles.parquet` | runtime percentile/rarity |
| 运行态 Claim 准备 | 待评论文 PaperIR/摘要 | `outputs/<run>/claim_graph/contribution_claims.jsonl` | GEAR novelty 通道、Graph 插入通道 |
| GEAR prior-art | contribution Claims + GEAR retrieval evidence | `outputs/<run>/gear_relation_cards.jsonl` | fusion |
| Graph 插入 | contribution Claims + Graph assets + 临时 Paper references | insertion edges/profiles/`graph_fact_cards.jsonl` | fusion |
| 融合 | RelationCards + Graph FactCards | `joint_novelty_cards.jsonl`、最终 review 引用 | 报告生成、测评 |

`semantic_claim_edges.parquet` 与 `paper_path_claim_edges.parquet` 都使用同一主键 `(earlier_claim_id, later_claim_id)`，但只保存自己负责的来源字段。合并阶段负责生成唯一正式边表，两个候选生成模块互不调用。

### 7.4 最小 schema 约束

每张中间表只做必要契约检查：

- 主键不重复；
- 外键能在约定上游表中找到；
- 必填列存在；
- 日期、布尔值和数值列能按约定读取；
- Claim source fragment 能回到 source sentence；
- 向量行号能映射到 Claim ID；
- 边满足不同父论文和严格时间方向。

不做 SHA-256、内容签名或 release identity。每个阶段另外写一段中文日志，记录读取行数、成功行数、失败行数、输出路径和耗时即可。

### 7.5 依赖准备

实现前先检查当前运行环境是否已经提供：

- `pandas`、`numpy`、`pyarrow`：Parquet 和数值表；
- `pydantic`：严格契约；
- `torch` 和可加载 BGE-M3 的模型库；
- `faiss`：全量精确 `IndexFlatIP`，若不可用则使用 torch/NumPy 分块检索；
- `igraph`、`leidenalg`：Leiden 社区；
- `networkx`：小型局部图指标和单元测试；
- GEAR 当前 JSON model client 所需依赖。

不要在 import 阶段加载 BGE、FAISS index、OpenAlex 表或生成模型。所有大对象通过显式 builder/runtime loader 惰性创建。

---

## 8. Layer 1：Paper Citation Graph

### 8.1 Layer 1 的必要性

Paper Citation Graph 不负责直接输出创新分数。它有三个明确用途：

1. 给 Claim 候选关系提供独立于 embedding 的书目路径支持；
2. 找到父论文之间的直接引用、两跳引用和共享参考文献 motif；
3. 让未来待评手稿可以通过其参考文献进入现有历史结构。

如果某条 Claim 只有语义邻居、没有 Paper path 支持，Claim Graph 仍然可以工作；但结果应明确写成 `semantic_only`。Paper Graph 的价值是增加一条独立结构来源，而不是强制每条 Claim 边都必须有引用支持。

### 8.2 节点集合

定义：

- `P`：24,922 篇 2023–2025 Nature 目标论文；
- `R1(P)`：目标论文直接引用的 OpenAlex Works；
- `R2(P)`：R1 Works 直接引用的 OpenAlex Works。

最终节点集合：

\[
V_P=P\cup R1(P)\cup R2(P)
\]

不额外创建 `H` 节点，不创建期刊、作者、机构、Topic 或 Claim 节点。

### 8.3 唯一边类型

Paper Graph 只有：

```text
CITES: citing_work_id → cited_work_id
```

保留：

```text
Nature P → R1
R1       → R2
```

不物化：

- bibliographic coupling 边；
- co-citation 边；
- shared-reference 边；
- similarity 边；
- 六跳可达边；
- Claim→Paper 边。

共享参考文献和两跳关系由 `CITES` 路径动态计算，不作为新的 Paper 边类型。

### 8.4 Paper 节点数据结构

建议 `paper_nodes.parquet`：

| 字段 | 类型 | 含义 |
|---|---|---|
| `work_id` | string | 规范化 OpenAlex Work ID |
| `doi` | string/null | 规范化 DOI |
| `title` | string | OpenAlex 标题 |
| `publication_date` | date/null | 发表日期 |
| `publication_year` | int/null | 发表年份 |
| `work_type` | string/null | OpenAlex Work 类型 |
| `source_id` | string/null | 期刊/来源 OpenAlex ID |
| `source_name` | string/null | 来源名称 |
| `primary_topic_id` | string/null | OpenAlex primary topic |
| `primary_topic_name` | string/null | topic 名称 |
| `field_id` | string/null | 上层 field ID |
| `field_name` | string/null | field 名称 |
| `is_nature_target` | bool | 是否属于 24,922 篇目标论文 |
| `nature_article_id` | string/null | 本地 manifest 的 article ID |
| `hop_min` | int | 相对任意 Nature target 的最小跳数：0/1/2 |

不需要保存全文、参考文献 JSON 大数组或当前引用次数作为创新特征。引用边单独写入 `paper_edges.parquet`。

### 8.5 Paper 边数据结构

建议 `paper_edges.parquet`：

| 字段 | 类型 | 含义 |
|---|---|---|
| `citing_work_id` | string | 引用方 |
| `cited_work_id` | string | 被引方 |
| `citing_hop_min` | int | 引用方最小 hop |
| `cited_hop_min` | int | 被引方最小 hop |

主键逻辑是：

```text
(citing_work_id, cited_work_id)
```

### 8.6 为什么需要三次 OpenAlex 快照扫描

因为本地 SQLite 没有 Work→文件偏移和 DOI→ID 索引，完整两跳元数据闭包通常需要三次扫描：

#### Pass 1：定位 Nature 目标论文

输入：

- 24,922 个规范化 DOI；
- 2,127 个 Works gzip 包。

输出：

- Nature DOI→OpenAlex Work ID；
- Nature 节点元数据；
- P→R1 边；
- R1 ID 集合。

#### Pass 2：定位 R1

输入：

- R1 ID 集合；
- 2,127 个 Works gzip 包。

输出：

- R1 节点元数据；
- R1→R2 边；
- R2 ID 集合。

在扫描前，可以使用 `analysis_state.db` 快速删除快照中不存在的 R1 ID。

#### Pass 3：定位 R2

输入：

- R2 ID 集合；
- 2,127 个 Works gzip 包。

输出：

- R2 节点元数据。

同样先用 `analysis_state.db` 过滤不存在的 R2 ID。

三次扫描的主要成本是 gzip 解压和 JSON 解析，不是 SQLite 查询。除非后续另外构建了 Work ID→gzip 文件/偏移索引，否则无法把三次全量扫描简化为随机查表。

### 8.6.1 每次扫描只解析必要字段

worker 不应把完整 OpenAlex Work JSON 长期保存在内存。对每条记录先读取匹配键：

```text
id
doi
```

只有当前 pass 命中目标集合时，再抽取：

```text
title/display_name
publication_date
publication_year
type
primary_location.source
primary_topic
referenced_works
```

Pass 3 的 R2 只需要节点元数据，不再展开 `referenced_works`。

### 8.6.2 SQLite membership 的简单用法

OpenAlex ID 统一规范为 short ID，例如：

```text
https://openalex.org/W1234567890 → W1234567890
```

对 R1/R2 集合分块查询 `works_index.short_id`，只保留存在项。数据库以只读方式打开。membership 结果写入普通 ID Parquet/文本 chunk，不在这个 52 GB 数据库中新增项目表。

### 8.7 Nature DOI 准备

`01_prepare_nature_targets.py` 应完成：

1. 读取 `/mnt/d/aspr_nature_markdown/manifest.jsonl`；
2. 只保留 `paper_write_status=exists` 且 `paper_markdown_path` 存在的记录；
3. 从 Markdown 前部提取 DOI；
4. 规范化 DOI：去掉 `https://doi.org/`、`http://doi.org/`、`doi:`，小写；
5. 提取本地标题和年份；
6. DOI 缺失或格式不可用时写入 `nature_target_failures.csv`；
7. 输出 `nature_targets.parquet`。

建议结构：

| 字段 | 含义 |
|---|---|
| `nature_article_id` | manifest `article_id` |
| `doi` | 规范化 DOI |
| `local_title` | Markdown 中的标题 |
| `publication_year` | manifest 年份 |
| `paper_markdown_path` | 本地 Markdown 路径 |

不要使用已有 `nature_target_works.parquet` 代替此步骤。此前审计发现该文件 DOI 字段存在严重错配，24,922 个 DOI 中没有可直接接受的年份一致匹配。

### 8.8 快照扫描并发与断点

`02_build_paper_graph.py` 应支持：

- `--workers 0`：0 表示使用本机可用 CPU 核数；
- `--resume`：跳过当前 pass 已完成的 gzip 文件；
- `--pass pass1|pass2|pass3|all`；
- 每个 gzip 独立解压、逐行解析；
- worker 不直接并发写主 Parquet；每个文件输出独立 chunk，主进程合并；
- `build_state.sqlite` 记录：`pass_id`、`file_path`、`status`、`records_seen`、`matches_found`、`finished_at`；
- 不记录文件哈希和复杂身份信息；
- 中断后只重跑 `status != complete` 的文件；
- 详细中文日志同时输出到终端和 `data/claim_graph/logs/paper_graph.log`。

日志示例：

```text
[Pass 1][0312/2127] 开始扫描 works_0312.gz
[Pass 1][0312/2127] 完成：读取 238,114 条，匹配 Nature 17 条，耗时 41.8 秒
[Pass 2] R1 原始 ID 3,482,771；SQLite 确认存在 3,470,225
[Pass 3] 已完成 1840/2127 个文件；累计定位 R2 9,218,334 条
```

### 8.9 Paper Graph 构建验收

完成后只做必要的结构验收：

- `paper_nodes.work_id` 唯一；
- `paper_edges` 不重复；
- 每个 P→R1、R1→R2 方向和 OpenAlex `referenced_works` 一致；
- 所有边的两个端点都存在于 `paper_nodes`；
- 统计 Nature DOI 匹配数、未匹配数、R1 数、R2 数、边数；
- 输出 `paper_graph_stats.json`。

不需要建立数据 release、Manifest、内容哈希链或发布校验协议。

---

## 9. 从本地 Markdown 确定性抽取摘要

### 9.1 为什么先把所有摘要抽出来

必须先完成所有论文的摘要切分，再进行任何 LLM Claim 抽取。这样做的目的不是安全封存，而是工程解耦：

- 摘要解析问题和 LLM 问题能够分别排查；
- 不会在全文 Markdown 中反复定位摘要；
- LLM 永远只能看到已经截取的摘要；
- 抽取任务可以按 paper ID 断点续跑；
- 可以先人工抽查摘要质量，再消耗模型调用。

### 9.2 本地 Markdown 的现实格式

Nature Markdown 并不保证存在 `## Abstract` 标题。常见形式包括：

1. 有显式 `Abstract` heading；
2. 标题、接收日期、作者之后紧跟一个无标题摘要段；
3. 摘要后直接进入 Introduction 正文；
4. 首页存在图片占位符、页眉、DOI、作者单位和页码噪声。

因此摘要提取应采用确定性多规则解析，而不是直接取全文开头固定字符数。

### 9.3 推荐解析顺序

对每篇 Markdown：

1. 读取文本；
2. 只处理文档前部，例如第一个 `Results/Methods/Discussion` heading 之前；
3. 清理图片占位符、页面重复标题、期刊页眉、孤立页码；
4. 识别 DOI；
5. 识别第一篇论文标题；
6. 如果存在 `Abstract` heading，提取到下一个同级 heading；
7. 否则定位作者列表后的第一个完整长段落，作为无标题摘要候选；
8. 遇到作者单位块、页脚或正文开篇时停止；
9. 保留摘要在原 Markdown 中的字符起止位置；
10. 对摘要进行句子切分并编号。

无标题摘要的优先判断：

- 位于标题和作者列表之后；
- 通常是一个完整长段落；
- 常包含 `Here we...`、`We report...`、`Our results...` 等贡献表达，但不能只依赖这些关键词；
- 不以机构地址、邮箱、图注或引用列表开头；
- 后续段落通常转入背景介绍。

不要调用 OpenScholar 判断哪几句是贡献句。LLM 后续直接接收完整的编号摘要句，由一次调用选择并原子化。

### 9.4 摘要表

`abstracts.parquet`：

| 字段 | 含义 |
|---|---|
| `paper_id` | Nature/OpenAlex 论文 ID |
| `nature_article_id` | 本地 article ID |
| `doi` | DOI |
| `title` | 标题 |
| `publication_date` | 发表日期 |
| `abstract_text` | 清理后的完整摘要 |
| `markdown_char_start` | 摘要在原 Markdown 中的起点 |
| `markdown_char_end` | 摘要在原 Markdown 中的终点 |
| `paper_markdown_path` | 原文件路径 |

`abstract_sentences.parquet`：

| 字段 | 含义 |
|---|---|
| `paper_id` | 论文 ID |
| `sentence_id` | 如 `S01`、`S02` |
| `sentence_index` | 从 1 开始 |
| `sentence_text` | 原句文本 |
| `abstract_char_start` | 在摘要中的起点 |
| `abstract_char_end` | 在摘要中的终点 |
| `markdown_char_start` | 在原 Markdown 中的起点 |
| `markdown_char_end` | 在原 Markdown 中的终点 |

### 9.5 摘要失败处理

如果出现以下情况，写入 `abstract_failures.csv` 并跳过：

- Markdown 不存在或无法读取；
- DOI/论文身份无法绑定；
- 找不到可信摘要区域；
- 摘要为空或只有页面噪声；
- 句子切分结果为空。

失败表只需：

```text
paper_id,nature_article_id,paper_markdown_path,reason
```

不设置 `limited`、`degraded` 等多级状态。

---
## 10. 一次受约束 LLM 抽取原子贡献 Claim

### 10.1 为什么只抽取摘要 Claim

Claim Graph 的历史节点数量上限约为：

```text
24,922 papers × 1–3 claims ≈ 24,922–74,766 claims
```

如果把每篇全文交给 LLM：

- 输入 Token 和调用时间会显著上升；
- 全文中方法细节、结果、讨论和背景会产生大量非核心 Claims；
- 不同论文长度差异会污染 Claim 数量；
- 后续 Graph 节点语义粒度很难保持一致。

摘要已经是作者对论文核心贡献的压缩表达。对当前目标——构建 2023–2025 Recent Nature Claim 结构图——摘要提供了更稳定、成本更低、跨论文更可比的 Claim 来源。

因此本版只抽取“摘要贡献 Claim”，不试图覆盖论文所有方法/结果陈述。GEAR 既有全文 `PaperIR` 数据结构保持不变，但本创新性评价只使用其中与 contribution Claim 的 prior-art 检索和证据关系判断有关的内容；它与 Claim Graph inventory 仍是两个不同的数据对象。

### 10.2 Claim 的最小定义

一条 Claim 是：

> 一句能够独立表达论文某个核心贡献、由摘要原句明确支持、只包含一个主要科学命题的陈述。

仅保存以下核心内容即可：

- Claim ID；
- 父论文 ID；
- Claim 类型；
- 标准化 Claim 文本；
- 绑定的摘要句 ID；
- 绑定的原句文本；
- 精确原文片段；
- 论文标题和发表日期等必要元数据。

不要求 Claim 节点存储复杂论证树、Claim→Reference 映射、GEAR verdict、审稿人标签或数十个 LLM 自评字段。

### 10.3 五种 Claim 类型

每条 Claim 恰好选择一个主类型：

#### METHOD

论文提出或显著改变了：

- 方法；
- 算法；
- 系统；
- 技术；
- 材料制备路线；
- 实验或测量手段；
- 干预方案。

示例：

> We introduce a microscopy method that simultaneously measures metabolic flux and chromatin state in living cells.

#### FINDING

论文报告了主要经验观察或结果：

- 新现象；
- 新关联；
- 性能结果；
- 定量差异；
- 观测到的规律。

示例：

> Inhibiting astrocytic lactate transport reduces long-term memory formation.

#### MECHANISM

论文提出或支持一个因果、解释性或过程机制：

- A 如何导致 B；
- 某过程通过何种路径发生；
- 某结构为何产生某结果。

示例：

> Astrocyte-derived lactate regulates memory consolidation through histone lactylation in engram neurons.

#### RESOURCE

论文贡献可复用研究资源：

- 数据集；
- benchmark；
- 工具；
- 图谱；
- catalog；
- 软件平台；
- 实验资源库。

示例：

> We release a spatial atlas of immune-cell states across 42 human tissues.

#### THEORY

论文提出理论性贡献：

- 理论框架；
- 数学模型；
- 定理；
- 一般性预测；
- 新的解释框架。

示例：

> We derive a general scaling law linking network sparsity to phase-transition stability.

### 10.4 Claim 类型为什么只是节点字段

METHOD、FINDING、MECHANISM、RESOURCE、THEORY 描述的是 Claim 在论文中的贡献角色，不是五套彼此隔绝的知识空间。真实创新经常发生在不同角色之间，例如：

- 新 METHOD 使此前无法观察的 FINDING 成为可能；
- RESOURCE 被用于发现新的 MECHANISM；
- THEORY 的预测被实验 FINDING 支持或修正；
- METHOD 与 RESOURCE 的组合促成新的 FINDING。

因此，本方案只构建一张 Claim Graph，并允许任意类型组合的候选边：

```text
METHOD/FINDING/MECHANISM/RESOURCE/THEORY
                    ↓
         one temporal Claim Graph
```

`claim_type` 仍是每个节点必填且单选的字段，用于解释邻域构成、统计跨类型连接和形成自然语言评价；它不参与边的硬过滤，也不决定社区边界。跨类型候选边只表示语义或书目结构上的连续联系，不能直接解释为“方法导致发现”或其他科学关系；这种关系含义仍由 GEAR 的文本证据判断。

### 10.5 不复用现有 `ClaimType`

当前 `gear/contracts.py` 中的 `ClaimType` 服务于全文同行评审，包含：

```text
NOVELTY, METHOD, RESULT, SCOPE, CAUSAL, SIGNIFICANCE
```

不要修改这个枚举来适配 Claim Graph，否则会破坏现有 GEAR `PaperIR` 和 Claim ledger。

应在 `gear/claim_graph/contracts.py` 新建：

```python
class InnovationClaimType(str, Enum):
    METHOD = "METHOD"
    FINDING = "FINDING"
    MECHANISM = "MECHANISM"
    RESOURCE = "RESOURCE"
    THEORY = "THEORY"
```

并建立独立的 `InnovationClaim` / `InnovationClaimInventory`。

### 10.6 最小 Pydantic 契约建议

```python
from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ClaimGraphModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InnovationClaimType(str, Enum):
    METHOD = "METHOD"
    FINDING = "FINDING"
    MECHANISM = "MECHANISM"
    RESOURCE = "RESOURCE"
    THEORY = "THEORY"


class InnovationClaim(ClaimGraphModel):
    claim_id: str
    parent_paper_id: str
    claim_type: InnovationClaimType
    claim_text: str
    source_sentence_ids: list[str] = Field(min_length=1)
    source_sentence_texts: list[str] = Field(min_length=1)
    source_fragments: list[str] = Field(min_length=1)
    title: str
    abstract_text: str
    publication_date: date


class InnovationClaimInventory(ClaimGraphModel):
    parent_paper_id: str
    claims: list[InnovationClaim] = Field(min_length=1, max_length=3)
```

离线 Nature Claim 可以绑定 `abstract_sentences.parquet` 的句子 ID；运行态 GEAR Claim 还应把摘要句映射为正式 `EvidenceSpan`，以满足当前 EvidenceStore 和 major claim evidence 要求。

### 10.7 LLM 输入

每篇论文一次调用，输入只有：

```json
{
  "paper_id": "W...",
  "title": "paper title",
  "abstract_sentences": [
    {"sentence_id": "S01", "text": "..."},
    {"sentence_id": "S02", "text": "..."},
    {"sentence_id": "S03", "text": "..."}
  ]
}
```

不输入：

- 正文；
- 参考文献列表；
- 评审意见；
- GEAR prior-art 结果；
- Paper Graph 特征；
- “这篇论文应有几条创新”等外部判断。

### 10.8 LLM 输出

建议生成端只输出：

```json
{
  "claims": [
    {
      "claim_type": "MECHANISM",
      "claim_text": "Astrocyte-derived lactate regulates memory consolidation through histone lactylation in engram neurons.",
      "source_sentence_ids": ["S04", "S05"],
      "source_fragments": [
        "astrocyte-derived lactate",
        "histone lactylation in engram neurons",
        "memory consolidation"
      ]
    }
  ]
}
```

`claim_id`、`source_sentence_texts`、标题、摘要和发表日期由本地代码补充，不能让 LLM 自由生成。

### 10.9 受约束 Prompt

建议系统 Prompt：

```text
You extract the paper's central contribution claims from only the supplied
title and numbered abstract sentences.

Return 1 to 3 atomic claims. Each claim must:
1. express exactly one scientific contribution;
2. be fully supported by the selected abstract sentence IDs;
3. preserve important scope, population, material, mechanism and comparator
   qualifications from the source;
4. use exactly one type from METHOD, FINDING, MECHANISM, RESOURCE, THEORY;
5. avoid background statements, motivation, generic significance, citations,
   reviewer language and unsupported novelty words;
6. include exact copied source fragments that can be found verbatim in the
   selected sentences.

Do not use outside knowledge. Do not combine independent contributions into
one claim. Return JSON only.
```

### 10.10 原子化要求

应拒绝：

> We propose method X, discover mechanism Y, and release dataset Z.

它包含三种独立贡献。

应拆为：

- METHOD：提出 method X；
- MECHANISM：发现 mechanism Y；
- RESOURCE：发布 dataset Z。

但每篇最多三条，LLM 应优先选择摘要中最核心、最明确的贡献，不追求穷尽。

### 10.11 一次调用后的确定性验证

不再调用第二个模型验证 Claim。只做以下确定性检查：

1. `claims` 数量为 1–3；
2. `claim_type` 属于五种类型；
3. `claim_text` 非空；
4. `source_sentence_ids` 均存在；
5. `source_fragments` 均能在所绑定原句中精确找到；
6. 同一篇 Claim 文本完全重复时去重；
7. 每条 Claim 只有一个类型。

任何一项失败：

- 放弃整篇论文本次所有 Claims；
- 写入 `claim_failures.csv`；
- 不保留“部分成功 Claim”；
- 不进入 fallback heuristic；
- 不设置 `limited`。

这样“可绑定原句”由精确片段验证完成，不需要第二个 reranker 或 entailment 模型。

### 10.12 Claim ID

不使用内容哈希。建议简单、可读：

```text
{parent_paper_id}::C01
{parent_paper_id}::C02
{parent_paper_id}::C03
```

Claim 顺序按照 LLM 输出后，经 `claim_type` 和首次来源句位置稳定排序，再分配编号。

### 10.13 Claim 节点表

`claim_nodes.parquet` 建议字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `claim_id` | string | Claim ID |
| `parent_paper_id` | string | Nature 父论文 OpenAlex ID |
| `nature_article_id` | string | 本地 article ID |
| `doi` | string | 父论文 DOI |
| `title` | string | 父论文标题 |
| `publication_date` | date | 父论文日期 |
| `publication_year` | int | 父论文年份 |
| `claim_type` | enum/string | 五种类型之一 |
| `claim_text` | string | 原子化 Claim |
| `source_sentence_ids` | list[string] | 来源句 |
| `source_sentence_texts` | list[string] | 完整原句 |
| `source_fragments` | list[string] | 精确原文片段 |
| `abstract_text` | string | 完整摘要 |

不在节点中保存：

- GEAR relation label；
- 审评意见；
- Claim 创新分数；
- 参考文献绑定；
- 旧 paper-level Graph 分数；
- HGB 预测值。

### 10.14 Claim 抽取并发与断点

`04_extract_claims.py` 建议支持：

- `--workers` 或 `--concurrency`；
- `--limit`：先跑小批量 pilot；
- `--years 2023 2024 2025`；
- `--resume`：`claim_nodes` 或失败表中已有的 paper 跳过；
- `--retry-failures`：只有显式指定才重试失败项；
- 一篇论文一个请求；
- 每完成一个批次就落盘 chunk；
- 中文日志显示完成数、成功数、失败数、平均 Claims/论文和调用耗时。

建议先跑一个只用于工程验证、不是测试集的分层 pilot：

- 每个年份抽相同数量；
- 尽量覆盖不同 Nature journal/source；
- 总量 300–500 篇；
- 人工查看摘要绑定和 Claim 粒度；
- 调整 Prompt 后再全量运行。

该 pilot 不封存、不用于最终效果声明，也不是监督标签集。

---

## 11. Claim Embedding

### 11.1 Embedding 文本

不建议只编码 LLM 生成的短 Claim，因为过度原子化可能丢失对象和限定条件。

固定拼接：

```text
Title: {paper title}
Claim: {claim text}
Evidence: {bound source sentence text}
```

如果绑定多句，按原摘要顺序拼接。

这三部分分别提供：

- Title：论文主题上下文；
- Claim：标准化贡献命题；
- Evidence：作者原始表述和限定条件。

### 11.2 模型

使用：

```text
/home/jayee/models/bge-m3
```

要求：

- 本地推理；
- 批量编码；
- 输出向量做 L2 normalization；
- 后续用 inner product 等价计算 cosine similarity；
- 不微调；
- 不使用 OpenScholar 二次 rerank。

### 11.3 存储

`claim_embeddings.npy`：

```text
shape = [n_claims, embedding_dim]
dtype = float32
```

`claim_embedding_index.parquet`：

| 字段 | 含义 |
|---|---|
| `claim_id` | Claim ID |
| `embedding_row` | `.npy` 行号 |
| `claim_type` | Claim 类型 |
| `publication_date` | 时间排序 |

### 11.4 运行策略

- 按批次编码即可；`claim_type` 不影响 embedding 空间；
- GPU 可用时用尽合理 batch；OOM 时自动减小 batch；
- 已存在 embedding row 的 Claim 在 `--resume` 时跳过；
- 不为同一 Claim 保存多个模型版本；
- 日志记录吞吐量和剩余数量即可。

---

## 12. Layer 2：原生 Type-Attributed Temporal Claim Graph

### 12.1 图的严格定义

Claim Graph 节点只来自成功抽取的 Nature 摘要 Claims：

\[
V_C=\{c\mid c\text{ extracted from a 2023--2025 Nature abstract}\}
\]

五类 Claim 共同构成一张图：

\[
G_C=(V_C,E_C),\qquad type(c)\in\{METHOD,FINDING,MECHANISM,RESOURCE,THEORY\}
\]

每条边：

- 两端 Claim 类型可以相同，也可以不同；
- 两端父论文不同；
- 来源 Claim 的发表日期严格早于目标 Claim；
- 表示“历史结构中的候选连续证据联系”；
- 不表示已经验证的 antecedent/extension/repeat。

因此 `claim_type` 是节点属性，不是分图键。边上派生保存 `earlier_claim_type`、`later_claim_type` 和 `is_cross_type`，用于描述性统计和审计。

### 12.2 唯一逻辑边类型

统一命名：

```text
CLAIM_CANDIDATE_LINK: earlier_claim → later_claim
```

一对 Claim 只保存一行。不同生成来源通过布尔字段和 motif 计数表达，不创建重复 multiedge。

展示时可派生：

```text
semantic_only
paper_path_only
semantic_and_paper_path
```

但这只是 `from_semantic/from_paper_path` 的显示映射，不是新的科学关系标签。

### 12.3 为什么不预标关系类型

不建立：

- `REPEAT`
- `EXTENSION`
- `COMBINATION`
- `BRIDGE`
- `SUPPORT`
- `CONFLICT`

原因：

- embedding 相似度不能区分重复和扩展；
- 引用路径不能证明科学关系；
- 这些标签会与 GEAR 的证据化关系分类重复；
- 如果先用 GEAR 给 Graph 标边，再用 Graph 评价 GEAR 输出，会形成循环论证。

Claim Graph 只记录连续相似度、图路径和结构位置；关系解释留给 GEAR 与最终融合层。

### 12.4 时间方向

边方向固定为：

```text
earlier Claim → later Claim
```

用 `publication_date` 比较。对同一天发表的论文：

- 不互相建立 temporal Claim 边；
- 按日期分组：先查询这一日期的全部 Claim，再把整组加入索引；
- 避免使用 Work ID 人为制造同日先后顺序。

如果目标论文缺少可用发表日期，应在 Nature 目标准备阶段补齐 OpenAlex 日期；仍缺失则不进入 Claim Graph，并记录失败。

### 12.5 禁止同论文边

同一父论文的 Claims 不连接：

```text
parent_paper_id(c1) == parent_paper_id(c2)  → no edge
```

原因：

- 它们天然共享摘要、标题和术语；
- 同论文语义相似会制造高权重边；
- 本项目要观察跨论文的历史知识结构，而不是论文内部论证结构。

运行态同时插入同一篇测试论文的多个 Claims 时，所有 Claims 都对同一个插入前历史图查询，互相不可作为邻居。

---

## 13. Claim 边生成来源一：全图语义近邻

### 13.1 基本算法

把所有类型的 Claim 放入同一个时间安全索引，按 `publication_date` 升序遍历：

1. 取当天所有 Claims；
2. 对每条 Claim 查询索引中严格更早的全部 Claims，不按类型过滤；
3. 取 cosine similarity 最高的 Top-k；
4. 添加 `earlier → later` 候选边；
5. 查询完成后，才把当天 Claims 加入索引。

推荐初始工程参数：

```yaml
semantic_top_k: 10
```

`k=10` 只是固定图构建参数，不是创新阈值。后续应查看 `k=5/10/20` 下核心结构描述是否稳定，但第一版先统一使用 10。

### 13.2 检索实现

Claim 数量预计不超过约 75k，可使用：

- 小 pilot：分块矩阵 cosine；
- 全量：FAISS `IndexFlatIP`，L2-normalized embedding 下得到精确 cosine 排名；
- 全部 Claim 共用一个索引。

不需要近似 HNSW 即可完成这个规模的精确 Top-k；如果本地依赖不含 FAISS，可以先用分块 NumPy/torch 矩阵乘法实现。

### 13.3 语义候选边字段

语义边至少记录：

- `cosine_similarity`；
- `semantic_rank`；
- `from_semantic=true`；
- 两端日期；
- 两端类型。

不设置“相似度超过 0.7 才是创新关系”之类的语义判断阈值。

---

## 14. Claim 边生成来源二：穿透 Paper Citation Graph

### 14.1 “穿透 Paper”到底是什么

Claim 节点自身不连接 Paper，也不连接 Reference。所谓穿透 Paper，是一个候选生成过程：

```text
目标 Claim
   ↓ parent_paper_id
目标 Nature 论文 P2
   ↓ 在 Paper Citation Graph 中找历史 Nature 论文 P1
历史 Nature 论文 P1
   ↓ 取其中最相似 Claim，不按类型过滤
历史 Claim
```

`parent_paper_id` 是外键，不是 Claim Graph 的一条边。

### 14.2 三种 Paper path motif

对较晚 Nature 论文 `P2` 和较早 Nature 论文 `P1`：

#### 直接引用

\[
P_2\rightarrow P_1
\]

字段：

```text
paper_direct_citation = true
```

#### 有向两跳引用

\[
P_2\rightarrow X\rightarrow P_1
\]

字段：

```text
paper_directed_two_hop_count = number of X paths
```

#### 共享参考文献

\[
P_2\rightarrow X\leftarrow P_1
\]

字段：

```text
paper_shared_reference_count = number of shared X
```

这些 motif 全部由 Layer 1 的真实 `CITES` 边计算，不新增 Paper Graph 边类型。

### 14.2.1 具体候选计算

从 `paper_edges.parquet` 预先建立：

```text
references_by_work[citing_id] -> set(cited_id)
nature_papers_by_reference[reference_id] -> set(nature_paper_id)
```

对较晚 Nature Paper `P2`：

#### Direct

```text
direct_candidates = references_by_work[P2] ∩ older_nature_paper_ids
```

#### Directed two-hop

```text
for X in references_by_work[P2]:
    for P1 in references_by_work[X]:
        if P1 is an older Nature target:
            two_hop_count[P1] += 1
```

#### Shared reference

```text
for X in references_by_work[P2]:
    for P1 in nature_papers_by_reference[X]:
        if P1 is strictly older than P2:
            shared_reference_count[P1] += 1
```

只把具有本地 Claim 节点的较早 Nature target 作为 Claim candidate Paper。R1/R2 非 Nature 节点只承担路径中间点，不会凭空产生 Claim 节点。

### 14.3 为什么两跳已经足够

Claim Graph 只需要一个合理的书目候选来源：

- direct citation 捕获显式先验工作；
- directed two-hop 捕获引用链；
- shared reference 捕获相近知识基础。

六跳邻域在 small-world 图中很可能接近整个学术网络，候选规模巨大、语义含义变弱，而且最终仍没有额外全文用于抽取历史 Claims。因此不采用六跳。

### 14.4 从候选 Paper 选择 Claim

对目标 Claim `c2` 的每个候选历史 Nature Paper `P1`：

1. 找到 `P1` 中的全部 Claims；
2. 用已经计算的 BGE-M3 embedding 比较；
3. 只选择最相似的一条 Claim `c1`；
4. 添加或更新 `c_1 → c_2`；
5. 写入 Paper path motif 字段。

如果候选 Paper 没有任何有效 Claim，则不产生 Claim 边。

不调用 OpenScholar reranker，也不调用 GEAR 验证这条边。

### 14.5 Paper path 候选规模

共享参考文献可能产生大量候选 Paper。建议初始工程限制：

```yaml
paper_path_candidate_papers_max: 50
paper_path_top_k: 10
```

选择顺序：

1. 所有直接引用历史 Nature Paper；
2. 按 directed two-hop count 降序；
3. 按 shared-reference count 降序；
4. 合并去重后最多保留 50 篇候选 Paper；
5. 每篇选出最相似 Claim 后，再按 BGE cosine 保留最多 10 条
   Paper-path Claim 边。

第一个上限控制 Paper motif 枚举成本，第二个上限避免 Paper-path-only 边数量压过固定的语义 Top-10。两个上限都是图构建预算，不是创新阈值。原始 motif 计数保留在边表中。

### 14.6 Paper path 不能证明什么

即使父论文存在直接引用，也不能说明对应的两条 Claims 是直接 antecedent。作者可能引用一篇论文的背景、数据、方法或完全不同结论。

因此 Paper path 只表示：

> 这对语义上相近的 Claims，其父论文之间还存在可观察的书目关系；两端类型可以相同或不同。

最终科学关系仍由 GEAR 的成对文本证据决定。

---

## 15. Claim Edge 合并与数据结构

### 15.1 合并规则

语义候选和 Paper path 候选按以下主键合并：

```text
(earlier_claim_id, later_claim_id)
```

同一对 Claim 最终只有一行：

- 语义 Top-k 命中：`from_semantic=true`；
- Paper path 命中：`from_paper_path=true`；
- 两者都命中：两个字段都为 true；
- cosine similarity 总是由 BGE-M3 计算，即使边只由 Paper path 提名。

结构指标使用的连续边权定义为：

\[
w_{cv}=\max(\operatorname{cosine}(c,v),0)
\]

如果一条 Paper-path-only 候选的 cosine 为负，它仍可以留在候选边表中用于审计，但不贡献连续社区权重；如果目标 Claim 所有保留边权之和为 0，则基于 `p_k` 的社区指标记为不可定义，不使用均匀权重伪造结果。

候选表可派生 `structural_active = cosine_similarity > 0`。非正相似度候选保留来源审计，但不进入组件、密度和路径扰动计算。这里的 0 是 cosine 的自然方向边界，不再额外设置 0.6/0.7 等创新阈值。

### 15.2 `claim_edges.parquet`

建议字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `earlier_claim_id` | string | 较早 Claim |
| `later_claim_id` | string | 较晚 Claim |
| `earlier_paper_id` | string | 较早父论文 |
| `later_paper_id` | string | 较晚父论文 |
| `earlier_claim_type` | string | 较早 Claim 的类型 |
| `later_claim_type` | string | 较晚 Claim 的类型 |
| `is_cross_type` | bool | 两端 Claim 类型是否不同 |
| `earlier_publication_date` | date | 较早日期 |
| `later_publication_date` | date | 较晚日期 |
| `cosine_similarity` | float | BGE cosine |
| `semantic_rank` | int/null | 语义 Top-k 排名 |
| `from_semantic` | bool | 是否由语义通道产生 |
| `from_paper_path` | bool | 是否由 Paper path 产生 |
| `paper_direct_citation` | bool | 父论文是否直接引用 |
| `paper_directed_two_hop_count` | int | 两跳路径数 |
| `paper_shared_reference_count` | int | 共享参考数 |
| `paper_min_path_length` | int/null | 1/2/null |
| `structural_active` | bool | 是否进入结构指标使用的图 |

`edge_kind` 不必落库，可在展示时派生。

### 15.3 基本边验收

必须满足：

- `earlier_publication_date < later_publication_date`；
- 两端类型字段与对应 Claim 节点一致，`is_cross_type` 可由两端类型唯一派生；
- 两端 `parent_paper_id` 不同；
- `from_semantic or from_paper_path` 为 true；
- Claim ID 均存在于 `claim_nodes.parquet`；
- 同一 Claim pair 不重复；
- cosine similarity 有限。

---

## 16. 在单一 Claim Graph 上构建全局社区

### 16.1 为什么社区是必要的

单纯 Top-k 邻居只能回答“最像谁”。社区让系统能够回答：

- 这些邻居是否属于同一个成熟知识区域；
- 目标 Claim 是否同时连接多个历史区域；
- 多个区域之间距离多远；
- 目标 Claim 是否形成跨区域桥接。

### 16.2 社区 backbone

对全部 Claim：

1. 使用所有已经完成的 2023–2025 历史 Claim embedding；
2. 在全体节点中构建 cosine Top-k，不按类型过滤；
3. 排除同一父论文 Claim pair；
4. 只保留 mutual-kNN：A 把 B 放在 Top-k 且 B 也把 A 放在 Top-k；
5. 将保留关系视为无向加权边；
6. 用 Leiden 做社区发现。

推荐：

```yaml
community_knn_k: 10
```

注意：这个 community backbone 是用于得到冻结历史社区划分的派生视图，不是 `claim_edges.parquet` 中的第二种科学边。`claim_edges.parquet` 仍然保持严格时间方向。

### 16.3 为什么社区主要使用语义 mutual-kNN

如果 Paper path 边直接参与社区发现：

- 不同领域引用习惯会主导社区；
- 引用较多的论文会获得不成比例影响；
- 引用路径噪声可能把语义不同的 Claims 拉到一起。

因此：

- 语义 mutual-kNN 决定社区；
- Paper path 作为 overlay，说明哪些 Claim 邻接还有书目结构支持。

### 16.4 社区输出

`claim_communities.parquet`：

| 字段 | 含义 |
|---|---|
| `claim_id` | Claim ID |
| `claim_type` | Claim 类型 |
| `community_id` | 全图社区 ID，如 `K0042` |
| `community_size` | 社区节点数 |

`community_profiles.parquet`：

| 字段 | 含义 |
|---|---|
| `community_id` | 社区 ID |
| `community_size` | 节点数 |
| `claim_type_distribution` | 社区内五类 Claim 的计数/占比 |
| `dominant_claim_type` | 占比最高的 Claim 类型，仅用于描述 |
| `claim_type_entropy` | 社区内类型混合程度 |
| `centroid_embedding_row` | 社区中心向量索引，或单独向量文件行号 |
| `top_terms` | 仅用于人工理解的关键词 |
| `representative_claim_ids` | 最接近 centroid 的若干 Claims |
| `earliest_date` | 最早 Claim 日期 |
| `latest_date` | 最新 Claim 日期 |

社区名称不由 LLM 决定图结构。`top_terms` 和代表 Claim 只用于展示；系统评价可以直接使用稳定的 community ID。

### 16.5 社区距离

任意两个社区之间的距离：

\[
d_{ij}=1-\cos(centroid_i,centroid_j)
\]

在整张 Claim Graph 的统一 embedding 空间中计算。该距离后续用于 `community_disparity` 和 Rao–Stirling integration。

---


## 17. 历史社区组合统计

### 17.1 为什么不能统计 Claim 节点对“第一次出现”

如果把单个 Claim 当作知识组件，那么几乎每个新 Claim 连接的 Claim pair 都是第一次出现，因为节点身份本身是唯一的。这会让 first-time 指标接近恒为 1，完全失去区分能力。

正确做法是把稳定的 Claim 社区当作知识组件，统计：

> 历史上是否已经有一条 Claim 同时连接过社区 A 和社区 B。

### 17.2 历史 Claim 的前驱社区集合

对每条历史 Claim `u`，只查看它发表时能够看到的更早 Claim 邻居：

\[
C(u)=\{z(v): v\rightarrow u\}
\]

这里不能使用后来连接到 `u` 的 Claims，否则会使用未来信息。

前驱集合使用与在线插入相同的 active-neighborhood 规则：semantic Top-10 与最多 Paper-path Top-10 的并集。这样历史 pair commonness 和新 Claim pair 指标的计算口径一致。`from_paper_path` 等来源字段仍单独保留，便于后续比较 semantic-only 与 full neighborhood。

### 17.3 社区与社区对计数

在整张 Claim Graph 上统一计算：

- `N`：具有至少一个历史前驱社区的 Claim 数；
- `N_a`：历史上有多少 Claim 的前驱社区集合包含社区 `a`；
- `O_ab`：历史上有多少 Claim 的前驱社区集合同时包含 `a,b`。

输出 `community_pair_history.parquet`：

| 字段 | 含义 |
|---|---|
| `community_a` | 社区 A |
| `community_b` | 社区 B |
| `pair_connector_count` | `O_ab` |
| `community_a_claim_count` | `N_a` |
| `community_b_claim_count` | `N_b` |
| `historical_claim_count` | `N` |
| `first_observed_date` | 首次由同一历史 Claim 跨接的日期 |

### 17.4 范围命名

由于 Claim Graph 只覆盖 2023–2025 Nature，不能写“历史首次组合”。应使用：

- `first_observed_recent_nature_pair_share`
- `recent_nature_pair_surprisal`
- “在本 Recent Nature Claim Graph 范围内此前未观察到”

GEAR 的更广 prior-art 检索负责补充范围之外的历史文献。

---

## 18. 新论文运行态 Claim 准备

### 18.1 两套 Claim inventory 必须分开

运行态待评价论文可能有两套 Claims：

#### GEAR prior-art evidence Claims

本方案中的用途只限于 prior-art 检索、证据 Span 和 RelationCards。其他 GEAR runtime 功能不进入 Claim Graph 创新性测评。

数据结构：当前 `PaperClaim`、`EvidenceSpan`、`ClaimLedger`。

#### Claim Graph contribution Claims

用途：

- 与历史 Nature Claim 节点保持相同粒度；
- 插入 Claim Graph；
- 计算结构位置和扰动。

数据结构：新的 `InnovationClaimInventory`。

不要直接把全文中所有 GEAR Claims 插图，因为它们数量更多、类型体系不同、粒度也不同，会破坏历史图与待评价论文之间的可比性。

### 18.2 运行态抽取方式

待评论文也使用同一套过程：

1. 从 PaperIR 中定位标题和摘要；
2. 生成正式摘要 `EvidenceSpan`；
3. 对摘要句编号；
4. 使用同一 Prompt 和五类型 taxonomy；
5. 提取 1–3 条 contribution Claims；
6. 绑定摘要 `EvidenceSpan` 和句子 ID；
7. 使用完全相同的 embedding 文本模板。

GEAR 可以复用这些 contribution Claims 作为 novelty verification targets，但 Graph 不读取 GEAR 后续产生的关系标签。

### 18.3 已发表论文和未发表手稿

#### 已发表论文

如果有 DOI/OpenAlex Work ID：

- 可从本地 OpenAlex 数据或配置的检索层获取父论文参考关系；
- 构造临时 P→Reference `CITES` 边；
- 计算 direct/two-hop/shared-reference Paper path 候选。

#### 未发表手稿

没有 OpenAlex ID 时：

- 使用 GEAR/PaperIR 已解析的参考文献；
- 通过 DOI/标题映射到 OpenAlex Work ID；
- 创建一个临时 paper ID，例如 `MANUSCRIPT::{run_id}`；
- 只在当前运行内构造临时引用边。

如果参考文献无法映射，仍可执行 semantic-only Claim 插入。输出中明确 Paper path coverage 为 0，而不是调用旧 Graph scorer 回退。

---

## 19. 新 Claim 如何插入 Claim Graph

### 19.1 插入不是选择一个坐标

图中的“位置”由邻接关系定义，不需要提前指定二维坐标。对每条新 Claim：

1. 确定唯一 Claim 类型；
2. 计算 BGE-M3 embedding；
3. 查询全部类型的历史 Claims；
4. 生成父论文路径候选；
5. 合并候选边；
6. 读取历史邻居的社区；
7. 在内存或临时对象中加入新节点和边；
8. 计算插入前后结构指标；
9. 当前运行结束后丢弃临时节点，不自动写回历史图。

### 19.2 所有测试 Claims 使用同一历史快照

如果一篇测试论文有三条 Claims：

- 三条都查询同一个插入前历史图 `G-`；
- 不先插入 C1 再让 C2 连接 C1；
- 不创建同论文边；
- 三条画像独立计算；
- 最后在 paper-level fusion 中综合。

### 19.3 候选邻居集合

定义：

\[
N(c)=N_{semantic}(c)\cup N_{paper\_path}(c)
\]

每个邻居保留：

- Claim 文本；
- 父论文元数据；
- Claim 类型；
- 社区 ID；
- cosine similarity；
- semantic rank；
- Paper path motif；
- 边来源布尔字段。

### 19.4 插入原始输出

建议每次运行输出：

```text
outputs/<run>/claim_graph/
├── inserted_claims.jsonl
├── insertion_edges.parquet
├── insertion_profiles.jsonl
└── graph_fact_cards.jsonl
```

`insertion_edges.parquet` 是所有汇总指标的原始来源。任何自然语言结构结论都应能回到邻居和指标。

---

## 20. 插入指标从哪里来

对目标 Claim `c`：

- `G-`：插入前的完整历史 Claim 图；
- `N(c)`：本次插入邻居；
- `w_cv`：Claim 到邻居的 cosine 边权；
- `z(v)`：邻居社区；
- `G+`：在 `G-` 上加入 `c` 及其候选边后的图。

所有指标只来自：

1. `insertion_edges`；
2. `claim_communities`；
3. `community_profiles/community_pair_history`；
4. `G-` 和 `G+` 的局部子图；
5. Paper Citation Graph motif。

它们不是模型预测出来的，也不从 paper-level Graph 分数分配而来。

---

## 21. Claim 结构创新的五个测量角度

Graph 分支不把创新压缩成单一概念。对 Graph 而言，一条 Claim 的结构创新可以表现为五类不同现象：

1. 与既有结构的嵌入或偏离；
2. 跨社区知识整合；
3. 非典型社区重组；
4. 局部桥接和路径改变；
5. 社区边界扰动。

第一类是节点插入位置，后四类更接近真正的图扰动。

### 21.1 角度一：既有结构嵌入

回答：

> 这条 Claim 是进入一个成熟、紧密的历史区域，还是离已有 Claims 较远？

#### 最近邻相似度

\[
s_{max}=\max_{v\in N(c)}w_{cv}
\]

字段：

```text
nearest_prior_similarity
nearest_prior_claim_id
```

#### Top-5 平均相似度

\[
s_{top5}=\operatorname{mean}(w_{c,(1)},...,w_{c,(5)})
\]

字段：

```text
mean_top5_similarity
```

#### 语义距离

可派生：

\[
semantic\_gap=1-s_{max}
\]

但它只能表示“与当前图中的最近 Claim 有多远”，不能单独称为 novelty。

#### 邻域重合差异（探索）

把目标 Claim 的邻域社区分布与历史 Claims 的邻域分布比较，可迁移旧 `reference_overlap_novelty` 思路。但 Top-k 邻域是 embedding 生成的，因此该指标与最近邻相似度不独立，第一版只作探索。

### 21.2 角度二：跨社区知识整合

先定义社区权重：

\[
p_k=\frac{\sum_{v\in N(c),z(v)=k}w_{cv}}
{\sum_{v\in N(c)}w_{cv}}
\]

#### 社区数量

```text
community_variety = number of distinct communities in N(c)
```

它容易受 k 影响，因此主要用于展示。

#### 最大社区占比

\[
dominant\_community\_share=\max_k p_k
\]

高值表示边权主要落在一个社区。

#### Simpson 跨社区程度

\[
community\_spanning\_simpson=1-\sum_k p_k^2
\]

它迁移自旧 `rtd_simpson`，但应使用 `community_spanning`，不能叫 diffusion，因为插入时尚未发生传播。

#### 有效社区数

\[
effective\_community\_count=\frac{1}{\sum_k p_k^2}
\]

它与 Simpson 单调等价，更容易解释成“相当于均匀连接了几个社区”。第一版建议保留有效社区数，Simpson 作为派生展示，不把二者当独立证据。

#### 社区均衡度

当社区数至少为 2：

\[
community\_balance=
\frac{-\sum_k p_k\log p_k}{\log K}
\]

#### 社区差异度

设社区中心距离为 `d_ij`：

\[
community\_disparity=\operatorname{mean}_{i<j}(d_{ij})
\]

#### Rao–Stirling 整合

\[
community\_rao\_stirling=
\sum_i\sum_j p_i p_j d_{ij}
\]

它同时考虑：

- 连接了多少社区；
- 连接是否均衡；
- 社区之间距离是否足够远。

仓库已有 variety、evenness、disparity、Rao–Stirling 的通用函数思想，可从 `gear/nature_multihorizon/features_v6.py` 迁移，但输入必须改为 Claim 社区。

### 21.3 角度三：非典型社区重组

目标 Claim 的社区集合：

\[
C(c)=\{z(v):v\in N(c)\}
\]

构造所有无序社区对 `(a,b)`。

#### Recent Nature 首次观察社区对占比

\[
first\_observed\_recent\_nature\_pair\_share=
\frac{\sum_{a<b}\mathbf{1}[O_{ab}=0]}
{\#\{(a,b)\}}
\]

它表示：在当前 2023–2025 Nature Claim 图范围内，这条 Claim 连接的社区组合有多少此前未被同一历史 Claim 跨接过。

#### 社区对平均 surprisal

平滑后 commonness：

\[
commonness_{ab}=\frac{(O_{ab}+0.5)N}{N_aN_b}
\]

\[
community\_pair\_mean\_surprisal=
\operatorname{mean}_{a<b}[-\log(commonness_{ab})]
\]

#### 低频社区对占比

```text
share of community pairs with O_ab <= a fixed count
```

该阈值是工程参数，因此只作补充，不作为主结论。

#### Uzzi-style atypicality/conventionality（探索）

可以把旧的 pair z-score、P10 atypicality 和 median conventionality 迁移到社区对。但多数 Claim 只连接 2–4 个社区，可用 pair 数太少，P10/median 不稳定。

因此第一版优先：

- first-observed share；
- mean surprisal。

Uzzi P10/median 只有在目标 Claim 具有足够社区对时才作为探索结果输出。

### 21.4 角度四：局部桥接和路径改变

令 `G-[N(c)]` 为插入前由邻居诱导的子图。

密度、组件和路径指标统一使用 `structural_active=true` 的 Claim edges，并把严格时间方向临时投影为无向简单图。最短路使用无权 hop distance；不使用 `1/cosine` 等额外距离变换。原始方向和连续相似度仍保存在边表中。

#### 邻居诱导密度

\[
neighbor\_induced\_density=
\frac{2|E(N(c))|}{|N(c)|(|N(c)|-1)}
\]

解释：

- 高密度：邻居本来就互相紧密连接，新 Claim 多为进入现有簇；
- 低密度：邻居彼此关系弱，新 Claim 更可能位于结构洞位置。

#### 组件数和组件合并

\[
components\_before=CC(G^-[N(c)])
\]

\[
component\_merge\_count=
CC(G^-[N(c)])-CC(G^+[N(c)\cup\{c\}])
\]

由于新 Claim 通常连接所有候选邻居，插入后局部组件常变为 1。因此该指标非常依赖 Top-k 边构建规则，不能单独作为强创新证据。

#### 新连通邻居对

对插入前不可达、插入后通过新 Claim 变为可达的邻居对计数：

```text
newly_connected_neighbor_pair_count
```

不要把不可达距离直接设为一个任意大数。不可达→可达单独计数。

#### 局部路径缩短

对插入前已可达的邻居对：

\[
path\_shortening(u,v)=d^-(u,v)-d^+(u,v)
\]

输出：

```text
mean_local_path_shortening
max_local_path_shortening
shortened_neighbor_pair_share
```

只在邻居 ego 范围或固定半径子图计算，避免每次插入做全图最短路。

#### Burt efficiency（探索）

旧体系中的 `burt_efficiency` 可以迁移，但在给定邻居数时，它与邻居诱导密度近似确定性相关。不要把二者作为两条独立创新证据。第一版保留更透明的 `neighbor_induced_density`。

### 21.5 角度五：社区边界扰动

#### 跨边界权重

把目标 Claim 暂时归入连接权重最大的主社区，则：

\[
cross\_boundary\_weight\_share=1-\max_k p_k
\]

它表示有多少连接权重越过主社区边界。

#### 新跨接社区对数

此前没有任何历史 Claim 同时跨接的社区对数量：

```text
first_observed_community_pair_count
```

这比含糊的 `new_community_pair_count` 更明确。它不是说社区之间此前完全没有边，而是说此前没有 Claim 以当前方式同时跨接这两个社区。

#### Claim cluster linkage gain（探索/第二阶段）

统计插入 Claim 带来的跨社区边权增量，并相对于局部历史 linkage 归一化。可以迁移 `sva_cluster_linkage` 的思想，但必须明确这是新节点插入版本，不可直接沿用旧 SVA 名称和数值解释。

#### Claim insertion modularity delta（探索/第二阶段）

在固定历史社区划分下，把新 Claim 分配到主社区，计算：

\[
\Delta Q_{claim}=Q(G^-)-Q(G^+)
\]

需要固定：

- 局部子图半径；
- 边权；
- 新节点社区归属；
- 是否包含 Paper path-only 边。

因为定义选择较多，第一版不把它作为核心主指标。

#### Centrality divergence（探索）

可比较插入前后旧节点 betweenness 分布变化，但：

- 新旧图节点集不同；
- 全局计算昂贵；
- 单节点引起的全局数值通常很小；
- 对局部子图范围敏感。

因此只在离线样例或后续方法研究中保留。

### 21.6 Claim 类型构成：解释维度，不是第六个创新分数

单图设计使目标 Claim 的邻居可能来自不同贡献角色。设目标 Claim 类型为 `type(c)`，邻居类型的归一化权重为：

\[
q_t=\frac{\sum_{v\in N(c),\,type(v)=t}w_{cv}}
{\sum_{v\in N(c)}w_{cv}}
\]

可报告：

\[
cross\_type\_neighbor\_share=1-q_{type(c)}
\]

以及：

\[
effective\_neighbor\_claim\_type\_count=\frac{1}{\sum_t q_t^2}
\]

同时保存 `neighbor_claim_type_distribution` 和主要边的 `earlier_type → later_type` 组合。它们回答“这条 Claim 正在连接哪些贡献角色”，例如 METHOD→FINDING、RESOURCE→MECHANISM，而不是直接回答“创新程度多高”。高跨类型占比既可能表示有意义的知识重组，也可能只是同主题下方法、结果和机制措辞接近，必须结合社区跨度、历史组合统计和 GEAR 文本证据解释。

---

## 22. 从旧创新指标体系迁移什么

旧核心和候选函数可以迁移“数学思想和底层小函数”，不能迁移旧 paper-level 值、旧 HGB 权重或旧 Graph scorer。

| 旧指标/家族 | Claim Graph 对应 | 迁移方式 | 当前角色 |
|---|---|---|---|
| `delta_q0_shock` | `claim_insertion_modularity_delta` | 只迁移边界扰动思想，重新定义图语义 | 探索 |
| `rtd_simpson` | `community_spanning_simpson` | Reference community 改为 Claim-neighbor community | 派生描述 |
| `field_log_variety` | `community_variety` | Field 改为 Claim community；保留原尺度 | 描述 |
| `field_evenness` | `community_balance` | 直接迁移分布均衡思想 | 补充 |
| `field_disparity` | `community_disparity` | 用 Claim community centroid cosine distance | 补充 |
| Rao–Stirling | `community_rao_stirling` | `p_k,d_ij` 改为 Claim 社区 | 核心 |
| `novelty_u` | `recent_nature_pair_surprisal` | source pair 改为历史 Claim 所跨接的 community pair | 核心思想，使用 mean surprisal |
| `first_time_source_pair_share` | `first_observed_recent_nature_pair_share` | source pair 改为 community pair | 核心 |
| Uzzi atypicality P10 | community-pair atypicality | 小样本时不稳定 | 探索 |
| conventionality median | community-pair conventionality | 伴随传统性，不是 novelty 本身 | 探索 |
| `burt_efficiency` | Claim local brokerage | 与 neighbor density 冗余 | 探索 |
| `reference_induced_density` | `neighbor_induced_density` | Reference ego 改为 Claim neighbor ego | 核心 |
| reference overlap novelty | Claim neighborhood-profile difference | 由 embedding 邻域生成，存在同源性 | 探索 |
| SVA cluster linkage | Claim cluster-linkage gain | 新节点图语义需重写 | 第二阶段 |
| SVA centrality divergence | local old-node centrality divergence | 只做离线探索 | 第二阶段 |
| reference age/degree | Claim-neighbor age/degree | 只作控制和描述 | 非创新 |

旧代码参考：

- `gear/nature_multihorizon/contracts.py`：旧 8 核心、10 辅助和五通道划分；
- `gear/nature_multihorizon/features.py`：旧 paper/reference 图指标计算；
- `gear/nature_multihorizon/features_v6.py`：variety、evenness、Rao–Stirling、Novelty U、first-time pairs、SVA；
- `gear/nature_multihorizon/features_v6_1.py`：mean surprisal、pair count、overlap novelty 等。

实现时可以抽取通用数学函数，但新模块命名必须明确属于 Claim Graph，不能让现有 Fig.1–Fig.3 runtime asset 被误认为 Claim novelty evidence。

---

## 23. 第一版推荐的核心结构画像

不要一开始生成几十个高度相关指标。第一版建议固定输出以下十项：

### 23.1 既有嵌入

1. `nearest_prior_similarity`
2. `mean_top5_similarity`

### 23.2 社区整合

3. `effective_community_count`
4. `community_rao_stirling`

### 23.3 非典型重组

5. `first_observed_recent_nature_pair_share`
6. `community_pair_mean_surprisal`

### 23.4 局部桥接

7. `neighbor_induced_density`
8. `component_merge_count`
9. `newly_connected_neighbor_pair_count`

### 23.5 边界跨越

10. `cross_boundary_weight_share`

额外支持字段单独报告：

- `cross_type_neighbor_count`
- `cross_type_neighbor_share`
- `effective_neighbor_claim_type_count`
- `neighbor_claim_type_distribution`
- `paper_path_supported_neighbor_count`
- `paper_path_supported_neighbor_share`
- `semantic_and_paper_path_agreement_count`
- `paper_direct_citation_neighbor_count`
- `paper_two_hop_neighbor_count`
- `paper_shared_reference_neighbor_count`

第二阶段探索：

- `mean_local_path_shortening`
- `claim_cluster_linkage_gain`
- `claim_insertion_modularity_delta`
- `local_centrality_divergence`
- `burt_efficiency`
- community-pair Uzzi P10/median
- neighborhood overlap novelty

### 23.6 为什么不合成总分

这些指标来自若干相关的数据结构：

- Simpson 和 effective community count 单调相关；
- variety/balance/disparity 与 Rao–Stirling 有重叠；
- induced density 与 Burt efficiency 冗余；
- component merge 与 newly-connected pairs 相关；
- nearest similarity 与 neighborhood overlap 相关。

直接把它们加权求和会重复计算同一种现象，并重新引入启发式权重。因此第一版输出“五维结构指纹”，不输出一个伪精确的 Graph novelty score。

---

## 24. 指标数据结构

建议 `InsertionProfile`：

```python
class ClaimInsertionProfile(ClaimGraphModel):
    claim_id: str
    parent_paper_id: str
    claim_type: InnovationClaimType

    neighbor_count: int
    cross_type_neighbor_count: int
    cross_type_neighbor_share: float | None = None
    effective_neighbor_claim_type_count: float | None = None
    neighbor_claim_type_distribution: dict[InnovationClaimType, int]
    nearest_prior_claim_id: str | None = None
    nearest_prior_similarity: float | None = None
    mean_top5_similarity: float | None = None

    community_count: int
    dominant_community_id: str | None = None
    dominant_community_share: float | None = None
    effective_community_count: float | None = None
    community_rao_stirling: float | None = None

    community_pair_count: int
    first_observed_recent_nature_pair_share: float | None = None
    community_pair_mean_surprisal: float | None = None

    neighbor_induced_density: float | None = None
    components_before: int | None = None
    components_after: int | None = None
    component_merge_count: int | None = None
    newly_connected_neighbor_pair_count: int | None = None
    cross_boundary_weight_share: float | None = None

    paper_path_supported_neighbor_count: int
    paper_path_supported_neighbor_share: float | None = None
    direct_citation_neighbor_count: int
    directed_two_hop_neighbor_count: int
    shared_reference_neighbor_count: int
```

这里的 `None` 表示数学上不可定义，例如只有一个社区时没有社区 pair。不要为了保证全字段有值而填 0，因为“没有 pair”与“pair surprisal 为 0”含义不同。

---

## 25. 历史经验基线和 Percentile

### 25.1 为什么需要基线

cosine 0.75、Rao–Stirling 0.31 或 component merge 2 本身很难自然语言解释。可以通过无监督历史经验分布给出相对位置。

### 25.2 历史伪插入

对每条历史 Claim，按其发表日期，只使用严格更早的 Claims 重建其插入画像：

1. 按日期升序处理；
2. 同日 Claims 不互相可见；
3. 使用同样 Top-k 和 Paper path 规则；
4. 计算同样十项指标；
5. 保存 `historical_insertion_profiles.parquet`。

这不是测试集，也不使用标签。它只是为当前指标建立统一的历史经验分布。

### 25.3 Percentile 参考组

第一版的主参考组是全部历史 Claim insertion profiles，不按类型拆成五套图或五套主基线。可额外报告一个按目标 `claim_type` 条件化的辅助 percentile，用于说明同类贡献中的相对位置，但不得代替全图主 percentile。

暂不再按 domain、期刊、季度建立大量小组，避免参考组过小和规则膨胀。

自然语言可以写：

> 该 Claim 的跨社区权重处于全部历史 Claims 的第 92 百分位；在历史 MECHANISM Claims 中的辅助百分位为第 89。

不能写：

> 第 92 百分位证明它具有 92 分创新性。

### 25.4 社区时间问题

当前日常资产的社区划分基于完整 2023–2025 历史图，适用于评估 2025 年之后的新论文。开发阶段的历史伪插入 percentile 可以使用该固定 community ID 作为描述坐标，但所有邻居边和 pair count 必须按当时历史截断；它不能作为严格回顾效果证据。

正式回顾测评必须另外构建按 cutoff 的社区快照。当前实现阶段不提前封存测试集，但 Phase 11 启动后按第 33.8 和 39.11 节执行时间切断。

---

## 26. Paper path 指标的正确角色

以下字段不是创新指标：

- direct citation count；
- two-hop path count；
- shared-reference count；
- paper-path-supported share；
- semantic-and-paper-path agreement。

它们是：

1. Claim 候选边的来源说明；
2. 语义结构是否同时得到书目结构支持的可靠性信息；
3. 解释为什么某个历史 Claim 被放进邻域的可追踪证据。

不能把“引用路径多”直接翻译成“创新更高”。

同样，下列旧辅助指标只能作为控制/描述：

- 邻居年龄中位数/IQR；
- 邻居 degree 中位数/P90；
- obscure neighbor share；
- 历史 component size；
- 候选邻居数量。

---

## 27. Top-k 和二值拓扑的敏感性

### 27.1 关键隐患

如果每条新 Claim 固定连接 Top-10 邻居，那么无论语义是否很强，这十条边都会在二值图中存在。只要十个邻居分布在三个不相连的组件，新 Claim 就会机械地“合并三个组件”。

因此：

- `component_merge_count` 是对既定图构建规则的条件描述；
- 不能单独宣称“合并组件，所以高度创新”；
- 连续权重指标通常比二值组件指标更稳定；
- 结构结论应同时查看 cosine、社区距离、pair rarity 和 Paper path 支持。

### 27.2 第一版处理

- 构图统一使用固定 `k=10`；
- 输出中始终记录 `neighbor_count` 和 k；
- 不设置多个相似度阈值；
- 在完成全流程后，用 k=5/10/20 做一次整体稳定性检查；
- 只有在多个 k 下方向一致的组件/桥接结论，才可用较强措辞。

这不是新增创新判据，而是检查结构结论是否由一个工程参数偶然制造。

---

## 28. Graph 结构画像如何分类

第一版可以生成描述性 profile，但不要把它们当监督真值。

### 28.1 `embedded_in_existing_cluster`

典型事实组合：

- 邻居高度集中在一个社区；
- effective community count 接近 1；
- neighbor induced density 较高；
- component merge 为 0；
- 历史社区组合常见。

含义：

> Claim 主要嵌入一个已有知识区域。

### 28.2 `locally_distinct`

典型事实组合：

- 最近邻和 Top-5 相似度相对较低；
- 邻居仍主要位于一个社区或缺乏明确跨社区桥接；
- Paper path 支持可能较弱。

含义：

> Claim 在当前 Recent Nature 图中局部差异较大，但 Graph 尚不能说明它构成有效创新。

### 28.3 `cross_community_connector`

典型事实组合：

- 连接多个社区，邻居可以来自不同 Claim 类型；
- community Rao–Stirling 较高；
- cross-boundary weight 较高；
- 邻居原本分处多个组件或低密度结构；
- 至少部分邻居有 Paper path 支持。

含义：

> Claim 在历史结构中承担跨社区连接角色。

### 28.4 `structural_recombination_candidate`

除跨社区连接外，还满足：

- first-observed pair share 较高；或
- community-pair surprisal 较高；
- 插入造成组件合并、不可达邻居对连通或明显边界跨越。

含义：

> Claim 是 Recent Nature 范围内的结构重组候选。

### 28.5 `structural_outlier`

典型事实组合：

- 与所有邻居相似度低；
- 邻域稀疏；
- 社区归属不稳定；
- Paper path 支持少。

含义：

> 当前图无法把 Claim 稳定放入历史结构。

不能自动翻译为“最创新”。它也可能是 Claim 抽取失败、术语特殊或历史图覆盖不足。

---


## 29. GEAR 与 Claim Graph 的联合逻辑

### 29.1 两个通道的输入一致、推理独立

共同输入是标准化 contribution Claim：

```text
claim_id
claim_type
claim_text
source span/sentence
paper metadata
```

随后分开：

- GEAR：检索更广 prior art，读取文本证据，输出 RelationCards；
- Claim Graph：只读取 Claim 内容、历史 Claim 图和可选父论文引用信息，输出 InsertionProfile。

Graph 不读取：

- `RelationLabel`；
- essential facet coverage；
- direct antecedent verification；
- GEAR confidence；
- GEAR 检索排名。

GEAR 不把 ClaimGraph cosine 或 community ID 当成 direct antecedent 证据。

### 29.2 联合评价的最小决策矩阵

| GEAR 结论 | Graph 画像 | 联合解释 |
|---|---|---|
| `DIRECT_ANTECEDENT` 已独立确认 | 任意 | 已有直接先例；Graph 不能挽救 firstness，但可描述该 Claim 位于何种结构区域 |
| `EXTENSION` / 多个 `PARTIAL_ANTECEDENT` | 单社区、高密度、常见组合 | 渐进式/簇内扩展 |
| `EXTENSION` / 多个 `PARTIAL_ANTECEDENT` | 跨多个远距社区、罕见组合、明显桥接 | 结构重组型扩展；创新来自重新组织已有组成部分 |
| 未发现 direct antecedent | 单社区嵌入 | 局部新颖但结构常规；不能自动称突破性 |
| 未发现 direct antecedent | 跨社区且历史组合罕见 | 跨社区桥接/结构重组型创新候选 |
| 未发现 direct antecedent | 低相似、无社区、路径支持弱 | 覆盖不足或结构离群；保持谨慎，不直接判高创新 |
| GEAR 未发现先例 | Graph 与历史 Claims 高度接近 | 文本–结构张力；提示检索可能遗漏或 Claim 表述差异，需要谨慎复核 |
| `UNRESOLVED` | 任意 | 不把 Graph profile 提升为确定 novelty 结论 |

### 29.3 Direct antecedent 的优先级

只要 GEAR 按当前规则找到并独立确认 `DIRECT_ANTECEDENT`：

- 该 Claim 的“首次提出”主张被否定或显著限制；
- 即使 Graph 显示它跨社区，也只能称结构传播、整合或新的应用位置；
- Graph 不能覆盖、抵消或平均掉直接文本先例。

### 29.4 为什么 Graph 对最终结果有实质影响

假设两条 Claims 在 GEAR 中都是 `EXTENSION`：

- Claim A 的邻居几乎全部位于一个高密度 METHOD 社区，历史组合常见，插入不改变局部连通；
- Claim B 同时连接三个 MECHANISM 社区，社区间距离较远，组合此前少见，并连接多个局部组件。

如果只有 GEAR，两者都可能被写成“在已有工作基础上的扩展”。加入 Graph 后：

- A 被解释为簇内渐进扩展；
- B 被解释为把已有但分散的知识组成新的机制链条，属于结构重组型扩展。

Graph 改变的是创新的类型、主次和解释，而不只是给 GEAR 分数加一个常数。

---

## 30. 从 Claim 级联合结论到 Paper 级评价

### 30.1 不再按 attribution weight 排序

Paper-level 汇总不使用：

```text
paper graph score × claim attribution weight
```

而是逐条 Claim 生成：

```text
ClaimEvaluation = GEAR relation evidence + Graph structural profile
```

### 30.2 Paper-level 汇总顺序

1. 列出 1–3 条 contribution Claims；
2. 对每条 Claim 判断是否存在 direct antecedent；
3. 标记证据不足或 unresolved；
4. 对剩余 Claims 描述其结构类型；
5. 区分：
   - 直接先例/验证性结果；
   - 渐进式扩展；
   - 局部新颖；
   - 跨社区桥接；
   - 结构重组型创新候选；
6. 说明论文创新主要集中在哪条 Claim；
7. 说明其他 Claims 是方法支持、实证支持还是已有结论重现；
8. 生成 paper-level 综合评价。

### 30.3 主要创新 Claim 的选择

不建议再创造一个复杂排序公式。使用明确的优先逻辑：

1. 已确认 direct antecedent 的 Claim 不作为“首次性”主要创新；
2. unresolved Claim 不以高置信度作为主要创新；
3. 在证据可支持的剩余 Claims 中，比较其 Graph 结构角色；
4. 能够连接远距、历史少见社区并产生稳定局部扰动的 Claim，更适合作为论文的结构性主要创新；
5. 最终文字仍需说明该判断来自哪组事实，而不是只给一个排序编号。

### 30.4 不强迫每篇论文只有一个创新点

一篇论文可能同时具有：

- 一个 METHOD 渐进扩展；
- 一个 MECHANISM 结构重组；
- 一个 FINDING 已有直接先例。

Paper-level 评价应保留这种异质性，不能把三条 Claim 简单平均成一个分数后丢失结构。

---

## 31. Graph 原始事实如何翻译成自然语言

### 31.1 先生成 Graph Fact Card

Graph 计算完成后先生成结构化事实卡，不直接让 LLM自由阅读整个图。

示例：

```json
{
  "claim_id": "MANUSCRIPT::C02",
  "claim_type": "MECHANISM",
  "nearest_prior_similarity": 0.75,
  "mean_top5_similarity": 0.66,
  "community_distribution": [
    {"community_id": "K0012", "weight_share": 0.40},
    {"community_id": "K0031", "weight_share": 0.33},
    {"community_id": "K0008", "weight_share": 0.27}
  ],
  "effective_community_count": 2.91,
  "community_rao_stirling": 0.48,
  "first_observed_recent_nature_pair_share": 0.67,
  "community_pair_mean_surprisal": 1.84,
  "neighbor_induced_density": 0.09,
  "component_merge_count": 2,
  "newly_connected_neighbor_pair_count": 21,
  "cross_boundary_weight_share": 0.60,
  "paper_path_supported_neighbor_count": 5
}
```

### 31.2 结构事实模板

先用确定性模板形成一句可审计陈述：

```text
该 {claim_type} Claim 的历史邻居分布于 {community_count} 个 Claim 社区，
有效社区数为 {effective_community_count}，主社区仅占 {dominant_share}。
插入前邻居形成 {components_before} 个局部分量；插入后合并
{component_merge_count} 个分量。其连接的社区对中有 {first_pair_share}
在本 Recent Nature Claim Graph 范围内此前未被同一 Claim 跨接，且
{paper_path_supported_neighbor_count} 个邻居同时得到父论文引用路径支持。
```

### 31.3 联合自然语言模板

然后加入 GEAR 事实：

```text
GEAR 在截止日期之前未找到完整覆盖该机制链条的直接先例，但分别找到
了覆盖其中三个组成部分的部分先例。Claim Graph 显示，这些组成部分位于
三个类型混合的 Claim 社区，目标 Claim 插入后把原本分离的局部邻域连接
起来，并形成在本 Recent Nature 范围内较少观察到的社区组合。因此，该
Claim 的主要贡献更适合解释为对已有机制组成部分的结构重组和跨社区桥接，
而不是对完全未知单一概念的首次提出。
```

### 31.4 LLM 在语言生成中的边界

LLM 可以：

- 把结构化事实卡改写成流畅文字；
- 合并重复语句；
- 按审稿语气组织段落。

LLM 不可以：

- 自行把低相似度解释成高创新；
- 自行声称“世界首次”；
- 把 Paper path 当成 antecedent 证据；
- 隐去 `Recent Nature 2023–2025` 的范围；
- 在 GEAR `UNRESOLVED` 时生成确定性 novelty 结论；
- 用 Graph profile 推翻 verified direct antecedent。

### 31.5 证据引用

当前 GEAR runtime 要求 major/critical review statement 有 EvidenceStore keys。因此实现时：

- GEAR 科学关系陈述引用 target/prior spans 和 RelationCard evidence keys；
- Graph 结构陈述引用 `graph_fact_card` 对应的 evidence/store record；
- Graph fact 证明的是结构计算，不是 prior 文本内容；
- 最终一句联合解释同时列出两类引用。

---

## 32. 一个完整示例

假设测试论文抽取出：

| Claim | 类型 | 内容 |
|---|---|---|
| C1 | METHOD | 同时测量神经元乳酸通量与染色质变化的方法 |
| C2 | MECHANISM | 星形胶质细胞乳酸通过神经元组蛋白乳酰化调控记忆巩固 |
| C3 | FINDING | 抑制星形胶质细胞乳酸转运降低长期记忆 |

### 32.1 Graph 原始结果

| 指标 | C1 | C2 | C3 |
|---|---:|---:|---:|
| nearest similarity | 0.88 | 0.75 | 0.94 |
| mean Top-5 | 0.82 | 0.66 | 0.89 |
| effective communities | 1.13 | 2.91 | 1.00 |
| Rao–Stirling | 0.07 | 0.48 | 0.00 |
| first-observed pair share | 0.00 | 0.67 | 不可定义 |
| neighbor density | 0.61 | 0.09 | 0.74 |
| component merges | 0 | 2 | 0 |
| cross-boundary weight | 0.10 | 0.60 | 0.00 |
| cross-type neighbor share | 0.18 | 0.71 | 0.24 |

Graph 事实：

- C1 高度嵌入一个以 METHOD Claims 为主的社区，但也连接少量 FINDING Claims；
- C2 连接胶质代谢、表观遗传和记忆印迹三个混合社区，主要邻居同时包含 FINDING、MECHANISM 与 METHOD Claims；
- C3 高度嵌入一个以 FINDING Claims 为主的已有社区。

### 32.2 GEAR 结果

- C1：已有方法分别完成两个测量任务，当前工作属于整合/扩展；
- C2：找到三个部分先例，但未找到完整机制链条直接先例；
- C3：找到基本相同结论的直接先例。

### 32.3 联合评价

- C1：渐进式方法整合；
- C2：结构重组型机制创新候选，是论文最主要创新来源；
- C3：已有发现的验证或重现，不应单独作为主要 novelty。

Paper-level：

> 该论文的创新性主要集中在机制 Claim C2。其方法贡献属于已有技术路线的整合扩展，主要实验发现也存在直接先例；但 C2 将胶质细胞代谢、组蛋白乳酰化和记忆印迹三个此前相对分离的机制社区组织成一个完整链条。GEAR 未发现该完整链条的直接先例，Claim Graph 同时显示其具有跨社区连接和局部结构重组作用。因此，论文的主要创新不是孤立方法或单一结果，而是具有证据支持的机制重组。

---

## 33. Graph 是否真的有用：当前阶段如何证明

### 33.1 先区分两种“证明”

#### 功能性证明

证明系统确实能够：

- 构建原生 Claim Graph；
- 给同一论文不同 Claims 生成不同邻域和结构画像；
- 提供 GEAR 输出中不存在的社区、路径、组合频率和扰动信息；
- 产生可追踪的自然语言结构评价。

这可以在当前不封存测试集的阶段完成。

#### 正确性/效用证明

证明加入 Graph 后，创新性评价比 GEAR-only 更符合专家判断。这最终需要盲评、人工判断或外部效标。当前本版不提前封存测试集，等全流程成功后再设计正式实验。

不能用“Graph 指标能够被算出来”替代“Graph 提高了评价正确性”。

### 33.2 当前阶段的非冗余检查

在所有成功 Claim 上：

1. 按 GEAR relation label 分组；
2. 在同一个 relation label 内查看 Graph 五维画像的分布；
3. 例如在所有 `EXTENSION` Claims 中，是否同时存在：
   - 单社区嵌入；
   - 跨社区桥接；
   - 结构离群；
4. 如果 Graph 只复现 GEAR 关系排序，则没有新增价值；
5. 如果 Graph 在同一 GEAR 语义关系内稳定区分不同结构角色，它提供了非冗余信息。

### 33.3 必须展示的三类案例

至少准备：

#### 案例 A：相同 GEAR，Graph 不同

两条都是 `EXTENSION`，一条簇内、一条跨社区。

#### 案例 B：GEAR 无 direct，Graph 高度嵌入

说明“未检索到先例”不等于“结构突破”，并展示谨慎性。

#### 案例 C：Graph 离群但 GEAR 找到 direct antecedent

说明 Graph 不能覆盖证据化先例，证明融合逻辑不是盲目奖励 Graph。

### 33.4 当前阶段的消融

可以在相同论文上生成：

1. GEAR-only；
2. GEAR + semantic ClaimGraph；
3. GEAR + semantic + Paper-path ClaimGraph；
4. GEAR + 仅保留同类型边的 ClaimGraph（架构消融，不是正式方案）；
5. GEAR + shuffled community IDs（负对照）；
6. GEAR + shuffled Paper path support（负对照）。

当前只比较：

- 输出是否增加了可验证的结构信息；
- 主要创新 Claim 是否发生有解释的变化；
- Graph 结论是否能够回溯到真实邻居和路径；
- 去掉跨类型边后，类型重组、社区桥接和最终解释是否损失有意义信息；
- shuffled 对照是否破坏合理结构解释。

不在当前阶段声称这些消融已经证明专家级准确率提升。

### 33.5 正式测评回答的核心问题

正式测评在全流程稳定后单独启动。当前阶段仍不封存测试集；但一旦进入正式测评，必须先固定样本、截止日期、系统配置、输出模板和评价 rubric，再生成待比较结果。

测评只回答创新性评价问题：

1. **主要方法增益**：Graph+GEAR 是否比 GEAR-only 更正确、具体地识别主要创新 Claim，并解释创新来自直接首次、增量扩展还是跨结构重组？
2. **Graph 不可替代性**：增益是否确实来自真实 Claim Graph，而不是文字变长、增加 Graph 术语或随机社区造成的？
3. **专家外部效度**：Graph+GEAR 的创新性判断是否比 GEAR-only 更接近独立领域专家的判断？
4. **可靠性辅助参照**：不同专家对同一论文创新性的判断本身有多大分歧？该结果只用于说明人工参照的可靠性，不作为论文主结果。

本测评不评价完整审稿能力，不统计实验设计、统计方法、写作质量或可复现性等一般审稿问题。

### 33.6 需要比较的创新性评价系统和人工参照

#### 主要 AI 系统

| 编号 | 系统 | 目的 |
|---|---|---|
| `A0` | 统一 Claim 输入 + 普通 LLM 创新性评价，不提供检索和 Graph | 较弱的无工具基线；只在已有可用模型时运行 |
| `A0R` | 相同 prior-art 候选检索 + LLM 直接总结，不运行 GEAR relation/evidence verification | 区分“有检索”与“有证据化 GEAR”的作用 |
| `A1` | GEAR-only | **主要基线**；隔离 Claim Graph 的净增益 |
| `A2` | ClaimGraph-only + 固定事实模板 | 验证 Graph 能提供什么，同时展示 Graph 单独不能判断 antecedent |
| `A3` | GEAR + semantic-only ClaimGraph | 测量纯 Claim 语义图增益 |
| `A4` | GEAR + semantic + Paper-path ClaimGraph | **完整方法** |

`A1` 与 `A4` 必须使用：

- 完全相同的目标 contribution Claims；
- 完全相同的 GEAR 检索结果、RelationCards 和 evidence spans；
- 完全相同的语言模型、温度、最大输出长度和报告模板骨架；
- 唯一差别是 `A4` 额外获得 Graph FactCards。

这样 `A4-A1` 才能归因于 Graph，而不是 Claim 抽取、检索或生成模型变化。

#### 架构消融和负对照

| 编号 | 条件 | 验证问题 |
|---|---|---|
| `B1` | 完整方法去掉 Paper-path edges | Paper citation motif 是否提供正向补充 |
| `B2` | 完整方法只保留同类型 Claim edges | 跨 Claim 类型边是否带来真实收益 |
| `B3` | 完整方法打乱 community IDs，保持社区大小分布 | 社区组合信息是否只是装饰性语言 |
| `B4` | 完整方法打乱 Paper-path support flags | 书目路径支持是否被正确使用 |
| `B5` | 完整方法用随机等度邻居替换真实邻居 | 局部扰动指标是否真正依赖语义结构 |

`B3–B5` 只用于离线负对照，不进入 runtime，也不能作为论文正式方法。

#### 人工创新性参照

对每篇论文保留独立专家/原 reviewer 身份：

```text
H1, H2, H3, ... = 同一论文的不同人工创新性评价者
```

人工评价只标注创新相关内容。保留三种参照：

1. 每位专家的独立创新性判断；
2. 经裁决的主要创新 Claim、prior-art 关系和创新模式；
3. 多位专家的一致性统计，仅作为人工标注可靠性的辅助信息。

测评中区分两类人工角色：

- `H-expert`：独立判断论文的主要创新 Claim、prior-art 关系、创新模式和创新强弱；
- `J-judge`：在系统身份盲化后评价 AI 创新性结论、核验依据并进行分歧裁决。

同一个人可以在不同论文上承担不同角色，但不应裁决自己对同一论文的原始标注，也不能在完成独立创新性判断前看到任何 AI 输出。

### 33.7 两套正式评价数据

#### E1：同输入专家创新性标注集——主要人工参照

为同一批本地 Nature 论文组织至少两位具有相关背景的领域专家，使 AI 和专家看到：

- 同一版本的论文；
- 同一 prior-art 截止日期；
- 同一创新性评价任务说明；
- 同一组标准化 contribution Claims；
- 同一标注字段：主要创新 Claim、直接/部分先例、增量/重组模式、创新强弱和依据。

至少要保证论文版本和 cutoff 一致。建议专家使用系统输出前冻结的共同 prior-art evidence packet，以隔离创新判断能力；该证据包不能只包含 GEAR 最终选择的文献。专家无需生成完整审稿意见，也不标注与创新性无关的问题。

建议先做 30 篇可行性 pilot，再做约 150–200 篇正式样本；最终数量由 pilot 的差异方差和功效分析决定。每篇至少两位独立专家；主要创新 Claim、直接先例或创新模式不一致时，由第三位专家裁决。

E1 是 AI 与人工参照比较的主要数据，但论文的首要因果比较仍是同输入条件下 `A4 Graph+GEAR` 对 `A1 GEAR-only`。

#### E2：本地 Nature 多 reviewer 创新性陈述集——辅助外部参照

本地 `peer_review_markdown_path` 中一篇论文通常包含 Reviewer #1/#2/#3 等多个报告。这里只抽取其中明确涉及以下内容的句段：

- 论文声称的主要贡献或创新点；
- 是否已有相同/相近工作；
- 贡献是否只是增量改进；
- 哪条 Claim 才是论文真正的创新来源；
- novelty claim 是否过度。

实验、统计、写作、复现和一般方法学批评全部排除。E2 不要求每篇必须有两个 reviewer 都谈创新；没有明确创新性陈述的 reviewer 记为 `NO_EXPLICIT_NOVELTY_JUDGMENT`，不能解释为认可创新。人–人辅助统计只在至少两位 reviewer 都有显式创新性判断的论文子集上计算。

E2 只能作为辅助证据，因为本地 paper Markdown 通常是最终发表版本，而第一轮 reviewer 看到的可能是修改前手稿。作者可能已修改创新表述，因此不能把 E2 当主 gold。

因此：

- A4 vs A1 是主比较；
- E1 提供独立专家外部效度；
- E2 只提供低成本辅助验证和少量人–人可靠性描述；
- 不把 E2 的任一 reviewer report 直接当 Claim Graph 标签或唯一 gold；
- peer review 数据永远不参与 Claim 抽取、建图、社区、指标调参或系统生成。

### 33.8 正式样本选择与时间切断

#### E1 抽样

只按与系统输出无关的字段分层：

- 发表年份；
- Nature 子刊/学科大类；
- 摘要长度；
- 五类 Claim 的基本覆盖。

不要根据完整方法的 Graph 分数、GEAR 成败或案例“好看程度”挑选正式样本，否则会产生选择偏差。Graph profile 分层只能用于额外诊断集，不能替代主随机/分层样本。

#### 时间切断

正式回顾评价必须避免未来信息。最干净的方案是：

1. 选择某一固定日期之后提交的论文作为目标；
2. 历史 Claim Graph 只使用固定日期之前的论文；
3. Leiden communities、community pair history 和 percentile 全部只用该历史截面重建；
4. GEAR 使用完全相同的 prior-art cutoff；
5. 目标论文及同日/之后论文不进入历史图。

优先使用论文的 initial submission/received date 作为 cutoff。只有发表日期而没有投稿日期的样本，不适合用于严格 firstness 测评；可以保留做非时间敏感的创新性描述实验，并明确标注。

如果一个固定 cutoff 样本不足，可按季度建立少量历史快照。禁止用完整 2023–2025 社区划分回评其中较早论文，因为这会把未来 Claim 的结构信息泄漏给目标论文。

### 33.9 统一评价单位

整个正式测评只有一个任务：**论文及其贡献 Claims 的创新性评价**。所有 AI 输出、专家标注和 reviewer 文本中的创新性陈述统一转换为：

```text
InnovationJudgment
├── paper_id
├── evaluator_or_system_id
├── target_claim_id
├── judgment_type
├── stance
├── innovation_mode
├── novelty_level
├── rationale
├── cited_prior_work
├── evidence_refs
└── source_text_span
```

`judgment_type` 只允许创新相关类别：

- `DIRECT_PRIOR_ART`
- `PARTIAL_PRIOR_ART_OR_EXTENSION`
- `NOVELTY_OVERCLAIM`
- `INCREMENTAL_CONTRIBUTION`
- `RECOMBINATIVE_CONTRIBUTION`
- `LOCALLY_DISTINCT_CONTRIBUTION`
- `UNRESOLVED_NOVELTY`

既有 reviewer report 解析规则：

1. 按 Reviewer 编号和评审轮次切分；
2. 第一轮 reviewer comments 与作者 response 分开；
3. 排除作者回复、编辑决定、重复粘贴的后续轮内容；
4. 只保留明确讨论创新、prior art、增量性或主要贡献的句段；
5. 由独立标注者绑定 target Claim 并确认 judgment type；
6. 与创新性无关的审稿意见直接排除，不进入任何指标。

可以使用模型帮助切分候选段落，但最终 `InnovationJudgment` 不能只由被测 AI 自己决定。

### 33.10 有明确主次的三层比较

#### 第一层：AI–AI 方法增益和消融——主要结果

在完全相同论文上做配对比较：

```text
A4 vs A1  = Claim Graph 在 GEAR 上的净增益（主比较）
A4 vs A3  = Paper-path overlay 的增益
A4 vs B2  = 跨类型边的增益
A4 vs B3/B4/B5 = 真实图结构相对随机结构的增益
A4 vs A0  = 完整证据系统相对普通 LLM 的总体增益（可选）
A4 vs A0R = 完整方法相对普通 retrieval-augmented novelty review 的增益（可选）
```

如果后续选择公开的外部论文创新性评价系统作为 baseline，只纳入能够在相同论文版本、相同 cutoff 和本地输入上稳定复现的系统，并单独报告其检索语料覆盖。不能把无法控制截止日期的在线黑盒输出与本地系统直接比较后声称方法优越。

#### 第二层：AI–专家创新性判断——外部效度

将 `A1` 和 `A4` 分别与 E1 裁决后的人工创新性标签比较：

- 是否选中相同的主要创新 Claim；
- direct/partial antecedent 判断是否一致；
- incremental/recombinative/locally-distinct/unresolved 模式是否一致；
- Claim 和 paper 创新强弱排序是否一致；
- 自然语言是否正确说明创新来源和边界；
- 是否出现无证据创新主张。

AI 与人工结果不一致时，由盲化 J-judge 依据论文和共同 evidence packet 裁决，不能把人工单人意见或“reviewer 未提及”直接当成 AI 错误。

#### 第三层：人–人一致性——辅助结果

只计算 E1 独立专家之间，以及 E2 中明确创新性陈述之间的：

- 主要创新 Claim 一致率；
- innovation mode 一致率；
- novelty level/ranking 一致性；
- direct antecedent 一致率。

该结果只说明人工参照的稳定程度，放在附表或补充实验中，不与 `A4 vs A1` 主结果并列，也不把 AI 是否“达到人类水平”作为主要论文主张。

### 33.11 对比指标

#### 主要创新 Claim 识别

| 指标 | 含义 |
|---|---|
| `main_innovation_claim_top1_accuracy` | 系统选出的主要创新 Claim 是否与裁决结果一致 |
| `main_claim_MRR` | 正确主要 Claim 在系统 Claim 排序中的位置 |
| `claim_novelty_judgment_precision/recall/F1` | 对各 Claim 创新性结论与裁决标签的一致性 |
| `valid_AI_only_novelty_point_rate` | AI 提出的额外创新性判断中经盲审确认有效的比例 |
| `unsupported_novelty_assertion_rate` | 缺少论文或 prior-art/Graph facts 支持的创新性断言比例 |

#### 创新性判断

| 指标 | 含义 |
|---|---|
| `direct_antecedent_precision/recall/F1` | 对直接先例判断的正确性 |
| `innovation_mode_macro_F1` | direct/incremental/recombinative/locally-distinct/unresolved 分类 |
| `novelty_overclaim_detection_F1` | 是否识别作者或评价中的 novelty 过度主张 |
| `ordinal_novelty_MAE` | 与人工 4 级创新性判断的绝对误差；仅作辅助，因为创新性主观 |
| `Spearman_rho` / `Kendall_tau` | paper/claim 创新性相对排序与人工排序的一致性 |

#### 解释质量盲评

评价者不知道输出来自哪个系统，对每份输出按 1–5 分评价：

- `correctness`：结论是否符合论文和给定 prior art；
- `specificity`：是否指出具体 Claim、先例和结构位置；
- `structural_insight`：是否提供单纯成对检索没有的有效结构解释；
- `evidence_traceability`：结论是否能回到证据 Span/Graph facts；
- `calibration`：是否正确表达范围和不确定性；
- `decision_usefulness`：是否帮助读者判断主要创新及其边界；
- `redundancy`：是否只是重复或拉长 GEAR 结论；
- `overclaim`：是否把 Graph 结构错误写成世界首次、科学正确或直接先例。

同时做强制二选一/可平局问题：

> 在不考虑文风偏好的情况下，哪份评价更正确且更具体地解释了论文创新来自哪里？

主报告给出 `A4 wins / ties / A1 wins`，避免只报告平均 Likert 分。

#### Graph 事实正确性

这些指标不是请人凭感觉评分，而是从落盘边表重新计算：

- neighbor ID 与 cosine 是否一致；
- community distribution 是否可复算；
- first-observed pair 是否真的在 cutoff 前未出现；
- component merge/path change 是否与局部图一致；
- Paper-path motif 是否能回到真实 `CITES` 路径；
- 自然语言中每个数字是否与 Graph FactCard 一致。

报告 `graph_fact_exact_match_rate` 和 `graph_statement_support_rate`。前者原则上应接近 100%，否则是实现错误，而不是模型效果不足。

#### 工程指标

- 每篇成功率和 `limited` 比例；
- Claim 抽取失败率；
- Graph 插入 coverage；
- GEAR retrieval/verification coverage；
- 每篇运行时延；
- GPU/CPU 峰值内存；
- 模型调用次数和成本；
- 输出长度，用于排除“更长所以更受偏好”的混杂。

### 33.12 人–人一致性指标（辅助）

不同数据类型使用不同指标：

- categorical innovation mode：Fleiss’ κ 或 Krippendorff’s α；
- ordinal novelty level：quadratic weighted κ、Krippendorff’s ordinal α；
- Claim 排名：Kendall’s W 或 pairwise Kendall τ；
- direct antecedent：raw agreement 和 Cohen/Fleiss κ。

该小节只用于说明人工标签可靠性，放在辅助结果中。若一致性低，应报告裁决过程和不确定性，但不把人–人对比扩展成系统主任务。

### 33.13 统计比较

所有系统在同一篇论文上运行，因此使用配对统计：

- precision/recall 错误差异：paired bootstrap；明确二分类可用 McNemar；
- ordinal/Likert 差异：paired bootstrap 或 Wilcoxon signed-rank；
- win/tie/loss：报告 bootstrap 95% CI；非平局样本可做双侧 sign test；
- ranking correlation：按 paper bootstrap 置信区间；
- 多个次要指标使用 Holm 校正；
- expert 和 paper 都重复出现时，补充 mixed-effects ordinal/logistic model，随机效应至少包括 `paper_id` 和 `expert_id`。

抽样单位是 paper，不是单条 Claim。Bootstrap 和数据划分必须按 paper 聚类，避免同一论文的多个 Claims 被当成独立样本夸大显著性。

### 33.14 预先指定的主终点和成功条件

为避免从大量指标中挑好看的结果，正式实验预先指定：

#### 主要效用终点

`A4` 相对 `A1` 在盲评问题“哪份更正确且更具体地解释创新来源”上的非平局胜率及 95% CI。

#### 主要安全终点

`unsupported_novelty_assertion_rate` 和 `overclaim` 不得显著高于 `A1`。Graph 增加结构信息不能以增加无依据创新主张为代价。

#### 关键次要终点

- `main_innovation_claim_top1_accuracy`；
- `innovation_mode_macro_F1`；
- `direct_antecedent_F1`；
- `claim_novelty_judgment_F1`；
- `structural_insight`；
- `graph_statement_support_rate`；
- `Spearman_rho/Kendall_tau` 与专家创新性排序的一致性。

Graph 被认为产生实质正向作用，至少需要同时满足：

1. `A4` 在主要效用终点优于 `A1`；
2. 结构 insight 明显提高；
3. unsupported/overclaim 不恶化；
4. 真实 Graph 优于 shuffled/等度负对照；
5. 至少一个核心增益在去掉跨类型边或 Paper-path 后下降；
6. Graph facts 可以逐条复算和追踪。

如果只提高文字偏好或输出长度，而正确性、结构 insight 和证据支持没有提高，则不能声称 Graph 有效。

### 33.15 测评落盘文件

正式测评也按模块落盘：

```text
data/claim_graph/evaluation/
├── evaluation_papers.parquet
├── human_experts.parquet
├── expert_innovation_judgments.parquet
├── reviewer_novelty_judgments.parquet
├── reviewer_novelty_parse_failures.csv
├── adjudicated_claim_labels.parquet
├── system_outputs/
│   ├── A0.jsonl
│   ├── A0R.jsonl
│   ├── A1.jsonl
│   ├── A2.jsonl
│   ├── A3.jsonl
│   ├── A4.jsonl
│   └── ablations.jsonl
├── blind_pairwise_ratings.parquet
├── blind_judgments.parquet
├── graph_fact_audit.parquet
├── metric_results.parquet
└── evaluation_summary.md
```

reviewer novelty 句段解析、专家标注、盲评和指标计算分别是独立阶段。指标脚本只读取已经落盘的 system outputs 和 judgments，不重新调用 GEAR、Graph 或生成模型。

---

## 34. 分阶段实施顺序

不要同时开发所有模块。按以下顺序，每一步成功后再进入下一步。每个 Phase 都以“约定输入文件可读 → 独立命令执行 → 约定输出文件落盘 → 本阶段验收”作为边界：

- 上游成功不等于自动触发下游；
- 下游不得修改上游产物；
- 缺少输入时明确停止，不在下游内实现隐藏 fallback；
- 每个阶段可用很小的 fixture 数据独立测试；
- 修改某阶段实现时，只要输出 schema 不变，下游代码不应随之修改；
- 运行态 Graph、GEAR 和 fusion 也分别落盘事实卡后再合并。

### Phase 0：目录和契约

实现：

- `gear/claim_graph/contracts.py`；
- `configs/gear/claim_graph.yaml`；
- `data/claim_graph/` 目录；
- 最小 contract tests。

运行命令：

```bash
python3 scripts/claim_graph/00_initialize_claim_graph.py \
  --output-root data/claim_graph \
  --config configs/gear/claim_graph.yaml
```

脚本会依次输出配置校验、目录创建、固定 Claim 类型、契约摘要写入四个步骤；日志同时写到 `data/claim_graph/logs/phase0_initialize.log`。

完成条件：

- 五种 Claim 类型可验证；
- 1–3 Claims inventory 可验证；
- Claim edge 允许跨类型，但禁止同论文和反时间方向；
- InsertionProfile 能序列化。

### Phase 1：Nature 目标表

运行计划命令：

```bash
python3 scripts/claim_graph/01_prepare_nature_targets.py \
  --manifest /mnt/d/aspr_nature_markdown/manifest.jsonl \
  --output data/claim_graph/nature_targets.parquet \
  --failures data/claim_graph/nature_target_failures.csv \
  --progress-every 100
```

脚本会先输出输入/输出路径与年份范围，之后每处理 100 篇成功论文报告累计读取、成功、失败数和当前 `article_id`；完整日志写到 `data/claim_graph/logs/phase1_prepare_targets.log`。首次运行不要加 `--overwrite`；只有明确要重建既有两个输出文件时才加它。

完成条件：

- 能读取 24,922 条 manifest；
- 生成 DOI、标题、年份、Markdown 路径；
- 失败项单独 CSV；
- 随机人工查看 DOI/标题来自正确论文。

### Phase 2：Paper Citation Graph

计划命令：

```bash
python3 scripts/claim_graph/02_build_paper_graph.py \
  --targets data/claim_graph/nature_targets.parquet \
  --snapshot /mnt/d/FabCitationData/openalex-snapshot \
  --id-index /home/jayee/workspace/FabCitation/openalex_snapshot_reference_check_results/analysis_state.db \
  --output-root data/claim_graph \
  --workers 0 \
  --pass all \
  --resume
```

`--workers 0` 会启用全部逻辑 CPU；主进程每完成一个 gzip 分片都会在终端输出其读取量、命中量、边数及累计耗时，并同时写入 `data/claim_graph/logs/paper_graph.log`。不要同时使用 `--resume` 与 `--restart`；明确要从头重建本阶段时才改用 `--restart`，它只清空 Phase 2 自己的分片状态，不会清除其他阶段状态。

完成条件：

- Pass 1/2/3 均可断点；
- Nature DOI→OpenAlex 匹配结果明确；
- 同 DOI 的多个 OpenAlex Work 按 `publication_date` 最新、元数据完整度、参考文献数、Work ID 的顺序确定唯一规范 P 节点；
- P→R1、R1→R2 边完整落表；
- 节点表包含 P/R1/R2 元数据；
- 所有 Paper edges 端点存在。

### Phase 3：全量摘要抽取

计划命令：

```bash
python3 scripts/claim_graph/03_extract_abstracts.py \
  --targets data/claim_graph/nature_targets.parquet \
  --output-root data/claim_graph \
  --workers 0 \
  --progress-every 100 \
  --resume
```

`--workers 0` 会以全部逻辑 CPU 并行处理 Markdown；每完成 100 篇会输出成功数、失败数和篇/秒速度，完整日志在 `data/claim_graph/logs/phase3_extract_abstracts.log`。需要从头重建摘要产物时使用 `--restart`，它只清空 Phase 3 的 chunk 和摘要状态。

完成条件：

- 每篇成功论文得到完整摘要；
- 每个摘要切成编号句；
- 保存原 Markdown 字符位置；
- 失败项单独 CSV；
- 至少人工查看 100 篇跨年份摘要，不包含正文或页眉污染。

### Phase 4：小批量 Claim pilot

计划命令：

```bash
python3 scripts/claim_graph/04_extract_claims.py \
  --abstracts data/claim_graph/abstracts.parquet \
  --sentences data/claim_graph/abstract_sentences.parquet \
  --output-root data/claim_graph \
  --limit 500 \
  --resume
```

检查：

- 1–3 Claims/论文；
- Claim 是否原子；
- 类型是否合理；
- 精确 fragment 是否存在于原句；
- 是否有背景句被误抽；
- 是否存在三类独立贡献挤在一条 Claim。

修订 Prompt 后重新跑 pilot。Prompt 确认后运行全量。

### Phase 5：全量 Claim embedding

```bash
python3 scripts/claim_graph/05_embed_claims.py \
  --claims data/claim_graph/claim_nodes.parquet \
  --model /home/jayee/models/bge-m3 \
  --output-root data/claim_graph \
  --resume
```

完成条件：

- 每条 Claim 有且只有一个 embedding row；
- 向量有限且 L2 normalized；
- embedding index 与 Claim ID 一一对应。

### Phase 6：Claim candidate edges

#### Phase 6A：语义候选边

```bash
python3 scripts/claim_graph/06a_build_semantic_claim_edges.py \
  --claims data/claim_graph/claim_nodes.parquet \
  --embeddings data/claim_graph/claim_embeddings.npy \
  --embedding-index data/claim_graph/claim_embedding_index.parquet \
  --output data/claim_graph/semantic_claim_edges.parquet \
  --resume
```

该步骤不读取 Paper Graph，只生成严格时间安全的全图 semantic Top-k。

#### Phase 6B：Paper-path 候选边

```bash
python3 scripts/claim_graph/06b_build_paper_path_claim_edges.py \
  --claims data/claim_graph/claim_nodes.parquet \
  --embeddings data/claim_graph/claim_embeddings.npy \
  --embedding-index data/claim_graph/claim_embedding_index.parquet \
  --paper-nodes data/claim_graph/paper_nodes.parquet \
  --paper-edges data/claim_graph/paper_edges.parquet \
  --output data/claim_graph/paper_path_claim_edges.parquet \
  --resume
```

该步骤不读取 semantic edge 表，只计算 Paper motifs，并在候选历史论文内部选择最相似 Claim。

#### Phase 6C：候选边合并

```bash
python3 scripts/claim_graph/06c_merge_claim_edges.py \
  --claims data/claim_graph/claim_nodes.parquet \
  --semantic-edges data/claim_graph/semantic_claim_edges.parquet \
  --paper-path-edges data/claim_graph/paper_path_claim_edges.parquet \
  --output data/claim_graph/claim_edges.parquet \
  --type-stats data/claim_graph/claim_edge_type_stats.parquet
```

合并步骤不加载 BGE、OpenAlex 或 GEAR；它只按 Claim pair 主键合并两张候选边表。

完成条件：

- 6A 可以在完全没有 Paper Graph 时独立完成；
- 6B 可以在完全没有 semantic edge 表时独立完成；
- 6C 对相同输入产生确定性的 union/dedup 结果；
- 同类型和跨类型边均可存在，且两端类型字段正确；
- 25 种类型方向组合的统计表可生成，未出现的组合显式记为 0；
- 只有跨论文边；
- 严格 earlier→later；
- 语义和 Paper path flags 正确合并；
- 可以从一条边回溯到父论文 motif。

### Phase 7：Claim communities

```bash
python3 scripts/claim_graph/07_build_claim_communities.py \
  --claims data/claim_graph/claim_nodes.parquet \
  --embeddings data/claim_graph/claim_embeddings.npy \
  --embedding-index data/claim_graph/claim_embedding_index.parquet \
  --output-root data/claim_graph
```

完成条件：

- 全部 Claim 共用一张 mutual-kNN backbone 并运行一次 Leiden；
- 社区 ID 是全图 ID，不带类型前缀；
- 社区允许包含多种 Claim 类型，并保存类型分布；
- 社区代表 Claims 在语义上基本一致；
- Paper path 不参与社区 backbone。

### Phase 8：历史组合与插入基线

```bash
python3 scripts/claim_graph/08_build_claim_graph_statistics.py \
  --claims data/claim_graph/claim_nodes.parquet \
  --edges data/claim_graph/claim_edges.parquet \
  --communities data/claim_graph/claim_communities.parquet \
  --embeddings data/claim_graph/claim_embedding_matrix.npy \
  --embedding-index data/claim_graph/claim_embedding_index.parquet \
  --graph-index data/claim_graph/claim_graph_index.sqlite \
  --output-root data/claim_graph \
  --workers 0 \
  --verbose
```

完成条件：

- community pair history 只用严格历史前驱；
- 能生成十项核心指标；
- 主 percentile 在全部历史 Claim 上计算，可附加同类型条件 percentile；
- 组件、pair 和不可定义字段处理正确。

### Phase 9：独立插入 Demo

```bash
python3 scripts/claim_graph/09_demo_insert_paper.py \
  --paper /path/to/test_paper.pdf \
  --cutoff YYYY-MM-DD \
  --output-root outputs/claim_graph_demo
```

完成条件：

- 抽取 1–3 contribution Claims；
- Claims 不互连；
- 每条 Claim 有 insertion edges/profile/fact card；
- Graph 不调用 GEAR relation classifier；
- 结构自然语言可回到原始数字和邻居。

### Phase 10：GEAR runtime 融合

在 `gear` 内接入：

- `InnovationClaimInventory`；
- `ClaimGraphRuntime` 惰性加载历史资产；
- `ClaimInsertionProfile`；
- `GraphFactCard`；
- fusion 中的显式联合矩阵；
- EvidenceStore references。

完成条件：

- `python -m gear review` 能生成 GEAR-only prior-art 与 Graph profile；
- GEAR RelationCards、Graph FactCards 和 joint cards 分别落盘，fusion 可脱离上游模型单独重跑；
- Graph 不可用时按当前 GEAR runtime 边界输出明确 limited structural result，不回退旧 scorer；
- 离线历史 Claim 失败仍是 drop+CSV，不和 runtime `limited` 混淆；
- direct antecedent 不被 Graph 覆盖；
- final review 的 Graph 陈述带事实引用。

### Phase 11：正式测评（系统跑通后启动）

Phase 11 不回写任何建图文件，所有内容只写入 `data/claim_graph/evaluation/`。

建议拆成六个独立命令：

```bash
python3 scripts/claim_graph/eval/01_prepare_evaluation_papers.py \
  --manifest /mnt/d/aspr_nature_markdown/manifest.jsonl \
  --output-root data/claim_graph/evaluation

python3 scripts/claim_graph/eval/02_extract_reviewer_novelty_judgments.py \
  --papers data/claim_graph/evaluation/evaluation_papers.parquet \
  --output-root data/claim_graph/evaluation

python3 scripts/claim_graph/eval/03_run_system_conditions.py \
  --papers data/claim_graph/evaluation/evaluation_papers.parquet \
  --conditions A0,A0R,A1,A2,A3,A4,B1,B2,B3,B4,B5 \
  --output-root data/claim_graph/evaluation/system_outputs

python3 scripts/claim_graph/eval/04_prepare_blind_packets.py \
  --system-outputs data/claim_graph/evaluation/system_outputs \
  --expert-judgments data/claim_graph/evaluation/expert_innovation_judgments.parquet \
  --reviewer-judgments data/claim_graph/evaluation/reviewer_novelty_judgments.parquet \
  --output-root data/claim_graph/evaluation

python3 scripts/claim_graph/eval/05_audit_graph_facts.py \
  --system-outputs data/claim_graph/evaluation/system_outputs \
  --output data/claim_graph/evaluation/graph_fact_audit.parquet

python3 scripts/claim_graph/eval/06_compute_metrics.py \
  --evaluation-root data/claim_graph/evaluation \
  --output data/claim_graph/evaluation/metric_results.parquet \
  --report data/claim_graph/evaluation/evaluation_summary.md
```

其中 `03_run_system_conditions.py` 只生成匿名创新性评价；`04_prepare_blind_packets.py` 只打乱展示顺序和生成评价包；人工判断完成后才把 `blind_pairwise_ratings.parquet`、`blind_judgments.parquet` 放回目录；`06_compute_metrics.py` 不允许调用任何模型。

完成条件：

- 同一论文的所有 AI 条件使用相同 Claim、cutoff、GEAR evidence 和输出预算；
- reviewer reports 已按 reviewer/round 拆分，只保留显式创新性陈述；
- A4 vs A1 作为主要结果，AI–专家作为外部效度，人–人只作为辅助可靠性结果；
- E1 专家创新标注和 E2 reviewer novelty 陈述分开报告；
- 主终点、安全终点和统计方法在查看正式结果前锁定；
- Graph fact audit 可以脱离自然语言生成单独重算。

---

## 35. `configs/gear/claim_graph.yaml` 建议

```yaml
paths:
  nature_manifest: /mnt/d/aspr_nature_markdown/manifest.jsonl
  openalex_snapshot: /mnt/d/FabCitationData/openalex-snapshot
  openalex_id_index: /home/jayee/workspace/FabCitation/openalex_snapshot_reference_check_results/analysis_state.db
  output_root: data/claim_graph
  embedding_model: /home/jayee/models/bge-m3

paper_graph:
  backward_hops: 2
  workers: 0

claim_extraction:
  min_claims: 1
  max_claims: 3
  allowed_types:
    - METHOD
    - FINDING
    - MECHANISM
    - RESOURCE
    - THEORY

embedding:
  normalize: true
  batch_size: 64

claim_graph:
  semantic_top_k: 10
  community_knn_k: 10
  paper_path_candidate_papers_max: 50
  paper_path_top_k: 10
  forbid_same_paper_edges: true
  require_strictly_earlier_date: true

runtime:
  persist_inserted_claims: false
```

这里只有必要工程参数。不要在第一版加入数十个 novelty thresholds 或复杂权重。

---

## 36. 关键算法伪代码

### 36.1 时间安全语义边

```python
index = EmptyVectorIndex()
for publication_date, daily_claims in group_all_claims_by_date(claims):
    for claim in daily_claims:
        neighbors = index.search(claim.embedding, k=semantic_top_k)
        for neighbor in neighbors:
            if neighbor.parent_paper_id == claim.parent_paper_id:
                continue
            add_semantic_edge(neighbor.claim_id, claim.claim_id)
    index.add_all(daily_claims)
```

### 36.2 Paper path Claim 候选

```python
for later_claim in claims_sorted_by_date:
    later_paper = later_claim.parent_paper_id
    candidate_papers = paper_graph.path_candidates(later_paper)
    candidate_papers = keep_strictly_older_nature_papers(candidate_papers)

    for earlier_paper, motifs in candidate_papers:
        earlier_claims = claims_by_paper[earlier_paper]
        if not earlier_claims:
            continue
        best = max_by_cosine(later_claim, earlier_claims)
        upsert_edge(best, later_claim, motifs=motifs)
```

### 36.3 社区组合历史

```python
for claim in claims_sorted_by_date:
    predecessor_claims = incoming_earlier_neighbors(claim)
    communities = sorted(unique(community[p] for p in predecessor_claims))

    for community in communities:
        community_count[community] += 1
    for left, right in combinations(communities, 2):
        pair_count[left, right] += 1
```

### 36.4 在线插入

```python
history = claim_graph.before(cutoff)
profiles = []

for claim in target_paper_claims:
    semantic = history.semantic_neighbors(claim)
    paper_path = history.paper_path_neighbors(claim, target_paper)
    neighbors = merge_candidates(semantic, paper_path)
    profile = compute_insertion_profile(history, claim, neighbors)
    profiles.append(profile)

return profiles  # do not persist target claims into history
```

---

## 37. 单元测试和小型人工图测试

### 37.1 合同测试

- 跨类型 ClaimEdge 可创建，且 `is_cross_type=true`；
- 同类型 ClaimEdge 可创建，且 `is_cross_type=false`；
- 同论文 ClaimEdge 被拒绝；
- 反时间方向被拒绝；
- Claim inventory 超过 3 条被拒绝；
- 非法 Claim 类型被拒绝；
- source fragment 不在绑定句中被拒绝。

### 37.2 摘要测试

至少覆盖：

- 显式 `Abstract` heading；
- 无 heading 的首页摘要；
- 图片占位符；
- 重复页眉；
- 作者单位块；
- 摘要和 Introduction 相邻；
- 句号缩写、数值和化学式切句。

### 37.3 手工小图

建立 8–12 个 Claim 节点的小图：

- 两个允许类型混合的社区；
- 一个历史跨接 Claim；
- 一个目标 Claim 只连单社区；
- 一个目标 Claim 通过跨类型边连接两个分离社区。

手算验证：

- effective community count；
- Rao–Stirling；
- pair first-observed share；
- neighbor density；
- component merge；
- newly connected pairs；
- cross-boundary weight。
- cross-type neighbor share 和 effective neighbor claim-type count。

### 37.4 Paper path motif 测试

构造：

```text
P2 → P1
P2 → X → P1
P2 → Y ← P1
```

验证 direct、two-hop、shared-reference 三种计数不会混淆，也不会物化为额外 Paper edge type。

### 37.5 融合测试

- direct antecedent + high Graph bridge：firstness 仍被限制；
- extension + embedded：输出 incremental；
- extension + cross-community：输出 recombinative extension；
- unresolved + Graph outlier：输出不确定，不输出高 novelty；
- Graph 不可用：不调用旧 scorer。

### 37.6 实现后的仓库级检查

```bash
python3 -m pytest -q tests/gear/claim_graph
make gear-test
python3 -m gear validate-assets
black gear/claim_graph scripts/claim_graph tests/gear/claim_graph
ruff check gear/claim_graph scripts/claim_graph tests/gear/claim_graph
mypy gear/claim_graph scripts/claim_graph tests/gear/claim_graph --ignore-missing-imports
```

格式化命令会修改文件，应在提交前再以 `--check` 或仓库 `make gear-lint` 检查现有 runtime 范围。ClaimGraph runtime 接入后，还应把新模块纳入 Makefile 的正式 lint/test 列表。

---

## 38. 各阶段最小数据质量检查

用户已经明确不需要复杂 release/Manifest/SHA。本项目只做足以防止明显错误的检查。

### Paper Graph

- Work ID 唯一；
- edge 端点存在；
- 无重复 edge；
- hop 只为 0/1/2；
- 目标 DOI 匹配统计。

### Abstract

- 文本非空；
- 来源 Markdown 存在；
- sentence offsets 可回到 abstract；
- 摘要不含大段图注/参考文献。

### Claim

- 1–3 条；
- 类型合法；
- source IDs 合法；
- fragments 精确存在；
- 同论文 Claim 文本不重复。

### Embedding

- 每 Claim 一行；
- 向量有限；
- norm 约为 1。

### Claim Edges

- 两端类型字段与节点一致，跨类型标记可复算；
- 不同父论文；
- strictly earlier→later；
- 来源 flag 至少一个；
- pair 唯一。

### Community

- 每 Claim 至多一个全图社区；
- 社区可以混合多种 Claim 类型，类型分布可复算；
- centroid 可计算。

### Evaluation

- reviewer comments、author response 和 editor text 分离；
- E2 只保留明确创新性判断，其他审稿意见被排除；
- E1 主要创新 Claim、innovation mode 和 prior-art 标签有人工裁决；
- A1/A4 的 Claim IDs、GEAR evidence、cutoff 和输出预算一致；
- system identity 对 J-judge 隐藏；
- 指标按 paper 聚类，不把同论文 Claims 当独立样本；
- A4 vs A1、E1 外部效度和 E2 辅助结果分开标识。

不新增复杂身份哈希、文件签名和发布流程。

---

## 39. 主要潜在问题与应对

### 39.1 Recent Nature 图不是全球历史创新图

问题：

- 2023 Claim 没有更早的本地图 Claim；
- 2024 只能看到 2023；
- 2025 只能看到 2023–2024；
- 非 Nature 和 2022 以前 Claim 不在图中。

应对：

- 明确命名 Recent Nature Claim Graph；
- Graph 只声称结构位置和范围内组合；
- GEAR 负责更广、时间更长的 prior-art 检索；
- 不使用“绝对历史首次”措辞。

### 39.2 摘要不一定包含所有创新

问题：某些方法细节或理论贡献只在正文中展开。

应对：

- Claim Graph 只建“摘要贡献图”；
- GEAR 全文 PaperIR 仍可提供更完整的论文证据，但本 Claim Graph 测评只评价创新性；
- 最终评价说明 Graph coverage 是标准化摘要贡献，不是假装覆盖全文一切。

### 39.3 LLM Claim 粒度不一致

问题：同一句摘要可能被不同方式拆分。

应对：

- 固定 1–3 条；
- 固定五类型；
- 固定单命题规则；
- 固定 exact fragment binding；
- 先跑 300–500 pilot 人工查看；
- 不使用二次模型链增加成本和新的不稳定性。

### 39.4 类型错分会影响解释，但不会切断邻居

问题：类型现在是描述性节点字段。错误分类不会阻断语义检索和连边，但会污染跨类型邻居占比、社区类型构成和自然语言中的贡献角色解释。

应对：

- Prompt 中给出清晰边界和反例；
- pilot 重点审查类型；
- 节点仍只允许一个主类型，保持统计口径清晰；
- 边生成和 Leiden 不读取类型，从结构上避免一次错分把 Claim 隔离到错误子图。

### 39.5 BGE 相似度不等于科学关系

问题：高相似可能只是同一主题。

应对：

- 边命名为 candidate link；
- 不预标 repeat/extension；
- GEAR 负责文本证据关系；
- Paper path 仅提高结构支持，不当验证标签。

### 39.6 社区和相似边来自同一 embedding

问题：邻居、社区和部分距离共享数据来源，多个指标不是独立证据。

应对：

- 不把十项指标简单加权；
- 使用五维 profile；
- Paper path 作为独立 overlay；
- 报告相关性并删除确定性/高度冗余量。

### 39.7 Top-k 人为制造桥接

问题：固定 k 会强制连边。

应对：

- 连续权重作为主信息；
- component metrics 不单独定性；
- 完成后检查 k=5/10/20 稳定性；
- 结构文字同时引用社区距离、rarity 和 path support。

### 39.8 Paper path 候选可能很多

问题：共享参考文献会产生大量候选。

应对：

- 直接引用优先；
- two-hop/shared 按 motif count 排序；
- 固定最多 50 个候选 Paper；
- 每个候选 Paper 只选择一条最相似 Claim，不按类型预过滤。

### 39.9 三遍 OpenAlex 扫描成本很大

问题：约 595 GiB gzip 快照扫描三次，I/O 和解压成本高。

应对：

- gzip 文件级多进程；
- chunk 落盘；
- `build_state.sqlite` 断点续跑；
- SQLite membership 过滤无效 R1/R2；
- 不重复做 DOI audit 之外的无用全表转换。

### 39.10 Graph 指标多但没有构念效度

问题：能够计算不等于真的测量创新。

应对：

- 主张限于 structural placement/perturbation；
- GEAR 的 evidence-grounded relation 是 novelty 语义证据；
- 当前阶段做功能和非冗余证明；
- 正式效果验证在系统跑通后单独设计。

### 39.11 完整社区划分造成回顾评价的未来信息泄漏

问题：如果用 2023–2025 全部 Claims 构建社区，再评价一篇 2024 Claim，即使边只查询 2024 以前节点，community ID 和 centroid 仍然受 2025 Claims 影响。

应对：

- 日常运行评价晚于历史库截止日期的新论文时，可以使用完整历史社区；
- 正式回顾实验必须按固定 cutoff 重建 Claim nodes、mutual-kNN、Leiden、pair history 和 percentile；
- 多 cutoff 只建立少量季度快照，不为每篇论文重建一套图；
- 论文中把 prospective 使用结果和 retrospective evaluation 结果分开。

### 39.12 单一跨类型图可能被主题相似而非贡献关系主导

问题：BGE 可能因为共享“蛋白、催化、成像”等主题词，把 METHOD、FINDING 和 MECHANISM 连在一起。跨类型边存在并不自动意味着有价值的贡献角色重组。

应对：

- 边仍只叫 candidate link；
- 报告 25 种有向类型组合的数量、cosine 和 Paper-path 支持率；
- 人工检查每种高频跨类型组合的代表边；
- 在正式测评中比较完整图与同类型-only 消融；
- 自然语言必须把跨类型占比与社区跨度、pair rarity、GEAR evidence 联合解释，不能单独奖励跨类型边。

### 39.13 Claim 类型不平衡会使全局 Top-k 和社区偏向多数类

问题：若 FINDING 数量远多于 THEORY/RESOURCE，少数类型 Claim 的全局近邻和社区可能被多数类吞没，`cross_type_neighbor_share` 也会被基率影响。

应对：

- 先报告节点类型比例、每类平均入度/出度、邻居类型条件分布；
- 主图不使用人工配额或禁止跨类型边；
- 同时提供按 target type 条件化的描述基线，而非建立五张图；
- 若少数类型几乎没有稳定邻域，应明确报告 coverage，不把其高跨类型占比解释为创新。

### 39.14 Leiden 社区对随机种子、k 和 resolution 敏感

问题：社区 ID 不是自然真值。不同 k、resolution 或随机种子可能改变社区数量、pair rarity 和桥接结论。

应对：

- 固定正式构建 seed；
- 在 k=5/10/20 和少量 resolution 值上报告 ARI/NMI、社区数量及核心插入指标稳定性；
- 只对多种合理设置下方向一致的结构结论使用强措辞；
- 不根据测评标签反复选择最有利的 resolution。

### 39.15 最终发表论文与首轮 reviewer 所见手稿可能不同

问题：本地 paper Markdown 多半是修订后的最终版本，peer-review Markdown 中的首轮意见针对更早手稿。作者可能已经按意见补实验、改结论或降低 novelty 表述。直接把 AI 对最终稿的结果与首轮 reviewer 比较并不公平。

应对：

- 主要人工外部效度使用 E1 同版本专家创新性标注；
- E2 只作为 reviewer 显式 novelty 陈述的辅助外部参照；
- E2 优先比较较稳定的主要贡献 Claim、prior-art 和 novelty concern，不比较已经明显修正的细节；
- 对能从 review response 确认已经修订的问题单独标记，不计作 AI 漏检。

### 39.16 人工创新性判断不是无噪声 gold

问题：不同专家对“增量”与“结构重组”的边界可能不同，单人判断不能直接作为唯一真值。

应对：

- E1 保留每位专家的独立标签并进行第三人裁决；
- 主要指标使用裁决标签，未裁决分歧保留为不确定；
- 人–人一致性只作为标签可靠性的辅助统计；
- 论文主结论以 A4 vs A1 配对增益为核心，而不是“AI 达到人类水平”。

### 39.17 Claim 抽取失败会产生选择幸存偏差

问题：较短、表达清楚的摘要更容易成功抽取和绑定 Claim；复杂或跨学科论文可能更容易被整体丢弃，最终 Graph 和测评样本会偏向“好抽”的论文。

应对：

- 按年份、期刊/领域、摘要长度报告抽取成功率；
- 对成功与失败论文比较基本元数据；
- 正式测评报告从原始样本到最终可评样本的流失表；
- 不只在成功样本上声称系统覆盖全部 Nature 论文。

### 39.18 待评论文与 Nature 历史语料存在领域分布偏移

问题：未来输入可能不是 Nature 风格摘要，Claim 粒度和语言分布与历史节点不同。低相似/结构离群可能来自写作风格或领域覆盖不足，而不是创新。

应对：

- Graph 结果始终带 Recent Nature scope；
- 输出最近邻相似度和 Paper-path coverage，低 coverage 时降低结构结论强度；
- 后续单独设置 Nature 内和 Nature 外两组 evaluation，不混合汇报；
- Graph 无稳定邻域时输出 structural outlier/limited coverage，不自动输出高度创新。

### 39.19 Semantic 与 Paper-path 并非完全独立证据

问题：作者更容易引用主题相近论文，因此语义边和引用路径相关。`semantic_and_paper_path` 同时命中不能被当作两个独立证据简单相加。

应对：

- Paper-path 只作为 support/coverage 字段；
- 不把两个来源加权成伪概率；
- 通过 A3/A4 和 B1 消融测量 Paper-path 是否真正改善评价；
- 报告两个通道的重合率和条件增益。

### 39.20 盲评可能被输出长度和 Graph 术语泄露系统身份

问题：完整系统输出更长，且“社区、桥接、Rao–Stirling”等词会让评价者猜到系统身份，产生新颖性偏好。

应对：

- A1/A4 使用相同段落结构和近似字数预算；
- 对外展示使用自然语言，不暴露内部系统名和不必要的公式术语；
- 随机化输出顺序；
- 记录评价者对系统身份的猜测，作为盲法检查；
- 同时报告 correctness/overclaim，而不只报告 preference。

### 39.21 Paper-level 汇总可能重复计算高度重叠 Claims

问题：同一论文的 2–3 个 Claims 虽然不互连，但可能表达同一贡献的不同切面，共享大量历史邻居。简单把三条 Graph profile 并列相加会夸大论文创新。

应对：

- paper-level 不求和、不平均成总分；
- 报告主要创新 Claim、次要 Claim 和已有先例 Claim；
- 保存 Claim 间文本相似度和历史邻居重合度作为冗余描述；
- 高度重叠 Claims 在自然语言汇总时合并叙述，不把同一结构贡献计算多次。

### 39.22 评价调参和案例选择可能污染正式结果

问题：在查看 reviewer reports 或正式盲评结果后调整 k、community resolution、fusion 模板或 Claim Prompt，会把正式集变成开发集。

应对：

- 当前实现阶段可自由 pilot；
- 进入正式评价前固定 Prompt、Graph 参数、fusion 逻辑、主终点和样本；
- E1 pilot 与正式样本不重叠；
- E2 reviewer 文本只在系统输出落盘后解封给测评解析模块；
- 参数敏感性作为预先定义分析，不用正式标签挑最优参数。

### 39.23 Reviewer report 没提到 novelty 不等于认为它没有问题

问题：真实 reviewer 可能主要关注实验、统计或表达，也可能默认认可某个 novelty Claim 而不写出来。把“未提及”编码为否定标签会系统性误判 AI。

应对：

- E2 中只有显式表达的 stance 才进入 novelty label；
- 未提及记为 `NOT_MENTIONED`，不是 `SUPPORTED` 或 `NO_PROBLEM`；
- AI 创新性断言的有效性由 J-judge 核验，而不是用 reviewer absence 自动计算；
- 主要创新模式和强弱评分依赖 E1 受控标注；E2 只报告显式 novelty 陈述的一致或互补案例。

### 39.24 只包含最终发表论文会产生选择偏差

问题：本地 Nature 数据都是成功发表的论文，缺少被拒稿、低质量或真正缺乏创新的手稿。系统可能只学会在高质量论文之间区分创新形式，不能据此声称能预测接收决定。

应对：

- 评价目标限定为“已发表论文的创新来源和边界评价”；
- 不把 acceptance prediction 当任务；
- 不使用 reviewer 推荐接收/拒绝作为 Graph 训练标签；
- 若未来获得同分布拒稿数据，再另设外部实验，不影响当前图结构。

### 39.25 共同 evidence packet 可能偏向某个检索系统

问题：如果盲评者只看到 GEAR 找到的 prior art，会高估 GEAR/Graph+GEAR；如果允许无限开放检索，又会把检索能力、时间预算和判断能力混在一起。

应对：

- 主实验的公共 evidence pool 在系统输出前冻结；
- 由统一检索候选、论文参考文献和人工补充的已知关键先例构成，而不是只用 A1/A4 的最终选择；
- 所有系统仍可记录自己实际使用的证据，judge 同时看到公共池和系统引用；
- 专家开放 prior-art 检索作为单独补充实验，单独报告耗时和新增先例。

---

## 40. 明确禁止重新引入的内容

实现过程中不得因为“已有代码”而回退到：

- 旧 paper-level Graph score 下发 Claim；
- 旧 attribution weight Claim 排序；
- 411,490 论文旧图作为当前 Claim 历史图；
- Claim→Reference 绑定；
- Claim→Reference Paper 图边；
- Review opinion 作为 Claim label；
- GEAR 验证后才允许 Claim edge；
- OpenScholar 逐边 rerank；
- 同论文 Claim 边；
- 六跳 OpenAlex 扩张；
- Graph 预标 `REPEAT/EXTENSION`；
- Graph 与 GEAR relation circular fusion；
- 复杂启发式总分；
- 旧 HGB 权重或当前 Fig.1–Fig.3 opportunity/control 字段直接充当 Claim novelty evidence；
- legacy LATS、committee、GraphRAG 或旧 graph scorer；
- 数据 release/版本/内容哈希 Manifest 体系；
- 本版提前封存测试集。

---

## 41. 完整实施检查表

### 数据

- [ ] `nature_targets.parquet` 已生成
- [ ] Nature DOI→OpenAlex 匹配完成
- [ ] Paper P/R1/R2 节点完成
- [ ] P→R1、R1→R2 edges 完成
- [ ] 全部可用摘要已抽取
- [ ] 摘要句已编号并保留 offsets

### Claims

- [ ] Claim Prompt pilot 完成
- [ ] 每篇 1–3 条
- [ ] 五类型单选
- [ ] exact source fragments 可绑定
- [ ] 失败论文已单独 CSV
- [ ] Claim nodes 完成

### Embedding/Edges

- [ ] BGE embeddings 完成
- [ ] 时间安全 semantic Top-k 已独立落盘
- [ ] Paper path motifs/edges 已独立落盘
- [ ] Claim edge union/dedup 完成
- [ ] 25 种有向 Claim 类型组合统计完成
- [ ] 跨类型边允许存在，`is_cross_type` 与两端类型一致
- [ ] 同论文边为 0
- [ ] 时间倒置边为 0

### Communities/Statistics

- [ ] 单一全图 mutual-kNN backbone 完成
- [ ] 单一全图 Leiden communities 完成
- [ ] community Claim 类型分布完成
- [ ] community centroids 完成
- [ ] historical community-pair counts 完成
- [ ] historical insertion profiles 完成
- [ ] 十项核心结构指标完成

### Runtime

- [ ] 测试论文摘要 contribution Claims 完成
- [ ] 临时 Claim 插入完成
- [ ] Graph Fact Cards 完成
- [ ] GEAR RelationCards 独立生成
- [ ] 联合决策矩阵完成
- [ ] natural-language templates 完成
- [ ] EvidenceStore refs 完成
- [ ] Graph 缺失时不回退旧 scorer

### 模块解耦

- [ ] 每阶段脚本的输入/输出路径可显式列出
- [ ] 上游产物一旦生成，下游只读不改
- [ ] semantic edge、Paper-path edge、edge merge 可分别运行
- [ ] community 构建不依赖 merged temporal edge 或 GEAR
- [ ] GEAR RelationCards 与 Graph FactCards 分别落盘
- [ ] fusion 可只读两类事实卡独立重跑
- [ ] runtime 不触发离线历史图重建
- [ ] 每个模块可用 fixture 文件独立测试

### 论证

- [ ] 同一 GEAR label 内 Graph profile 具有差异
- [ ] 至少三类非冗余案例完成
- [ ] semantic-only 与 semantic+paper-path 对比完成
- [ ] shuffled community/path 负对照完成
- [ ] Graph scope 限定为 Recent Nature
- [ ] 没有用 Graph 推翻 direct antecedent

### 正式测评（系统跑通后）

- [ ] E1 同版本、同 cutoff 的专家创新性标注集完成
- [ ] E1 只标注主要创新 Claim、prior art、innovation mode 和 novelty level
- [ ] E2 reviewer reports 已按 reviewer/round 拆分且只保留显式创新性陈述
- [ ] 作者 response、编辑决定、一般审稿问题和重复轮次已排除
- [ ] `InnovationJudgment` 已人工确认
- [ ] A1 GEAR-only 与 A4 Graph+GEAR 使用同一 Claim/GEAR evidence/输出预算
- [ ] A4 vs A1 主配对比较完成
- [ ] A4/A1 与 E1 专家裁决标签的外部效度比较完成
- [ ] 人–人创新性标签一致性作为辅助结果完成
- [ ] shuffled/等度邻居负对照完成
- [ ] 完整图 vs 同类型-only/Paper-path ablation 完成
- [ ] Graph facts 独立复算完成
- [ ] 主效用终点和主要安全终点完成
- [ ] 按 paper 聚类的置信区间和配对统计完成
- [ ] E1 和 E2 结果分开报告
- [ ] 全部测评指标仅涉及创新性评价

---

## 42. 论文中可使用的方法定位

### 42.1 方法名称建议

```text
Type-Attributed Temporal Claim Graph with Online Structural Insertion
```

或：

```text
Evidence-Grounded Prior-Art Verification with Claim-Level Structural Placement
```

### 42.2 核心贡献表述

英文建议：

> We introduce a type-attributed temporal Claim Graph that places standardized
> contribution claims into a recent scientific knowledge structure through
> semantic and citation-path-supported candidate links. Unlike retrieval-only
> novelty assessment, the framework measures a claim's historical neighborhood,
> community position, recombination rarity, brokerage role, and local structural
> perturbation. These graph-derived structural profiles are combined with
> GEAR's evidence-grounded prior-art relations to distinguish direct
> antecedents, incremental extensions, local novelty, and structurally
> recombinative innovation.

中文对应：

> 本研究提出一种带有贡献类型属性的时间 Claim Graph。五类标准化贡献 Claim 共存于一张图中，并通过语义近邻和论文引用路径支持的候选边形成包含同类型与跨类型联系的近期科学知识结构。不同于仅依赖检索的创新性判断，该方法描述 Claim 的历史邻域、社区位置、类型组合、组合罕见性、桥接作用和局部结构扰动，并与 GEAR 的证据化 prior-art 关系联合，从而区分直接先例、渐进扩展、局部新颖和结构重组型创新。

### 42.3 必须保留的范围声明

论文中必须明确：

- Claim Graph 覆盖 2023–2025 Nature 摘要贡献；
- Graph 提供 recent structural context，不代表全历史完整先例库；
- GEAR prior-art 检索承担更广范围的 antecedent verification；
- Graph links 是候选结构联系，不是科学关系标签；
- Graph profile 是结构画像，不能单独证明 Claim 正确或世界首次。

---

## 43. 最终系统应该输出什么

对每条 Claim，输出四块：

### 43.1 Claim 原文卡

- Claim 文本；
- Claim 类型；
- 摘要原句；
- 原文 fragment；
- 父论文元数据。

### 43.2 GEAR prior-art 卡

- 最相关 prior works；
- 目标/先验证据 Span；
- relation labels；
- common/difference dimensions；
- essential facet coverage；
- direct antecedent 独立复核结果；
- coverage limitations。

### 43.3 Claim Graph 结构卡

- 最近历史 Claims；
- 社区分布；
- 十项核心插入指标；
- Paper path support；
- 五维结构画像；
- Recent Nature scope。

### 43.4 联合评价卡

- direct/partial/extension 状态；
- incremental/local/bridge/recombinative/uncertain 解释；
- 该 Claim 是否构成论文主要创新；
- 结论使用的 GEAR evidence refs；
- 结论使用的 Graph fact refs；
- 明确限制。

对整篇论文，输出：

- 各 Claim 创新性差异；
- 主要创新集中在哪里；
- 哪些只是方法支持、实证支持、验证或已有先例；
- 为什么 Graph 改变了 GEAR-only 的解释；
- 一段完整 paper-level innovation assessment。

---

## 44. 最终成功标准

### 44.1 工程与方法成功

本 Idea 在工程上成功，不是指生成了一个 Graph 分数，而是同时满足：

1. 24,922 篇 Nature 数据能够在无需下载额外全文的情况下形成可用摘要 Claim corpus；
2. 每条 Claim 是原生图节点，不继承 paper-level Graph 常数；
3. 五类 Claim 共存于一张时间图，类型是节点字段，同类型和跨类型候选边均可追踪；
4. 同论文 Claims 可以因为自身历史邻域不同而获得不同结构画像；
5. Graph 能输出社区位置、罕见重组、结构洞、组件/路径变化等 GEAR 无法直接产生的信息；
6. GEAR 能用更广 prior-art 文本证据约束 Graph 的过度解释；
7. Graph 能在相同 GEAR relation 下区分渐进扩展和结构重组；
8. 最终自然语言每个主要判断都能回到 Claim 原句、prior-art spans 和 Graph facts；
9. 没有重新引入旧 paper-score 分配、启发式总分或循环验证；
10. Paper Graph、摘要、Claim、embedding、两类候选边、社区、历史统计、GEAR、Graph 插入和 fusion 只通过约定落盘数据通信；
11. 任一模块在输入 schema 不变时可以被独立替换和重跑；
12. GEAR/Graph 任一通道缺失时明确报告 coverage/limited，不执行隐藏旧方案回退。

### 44.2 实证成功

工程跑通只能证明 Graph 可用，不能证明 Graph 有效。论文要声称 Graph 对创新性评价产生实质正向作用，还必须在正式测评中满足：

1. Graph+GEAR 相对 GEAR-only 的配对盲评主要效用终点提高；
2. unsupported novelty assertion 和 overclaim 安全终点不恶化；
3. Graph facts 能从落盘图数据准确复算；
4. 完整真实图优于 shuffled/等度邻居负对照；
5. 去除跨类型边或 Paper-path 后至少有一项预先指定的核心增益下降；
6. A4/A1 与 E1 专家创新性标签的比较支持 A4 的外部效度；
7. 人–人一致性只作为 E1 标签可靠性的辅助说明；
8. E2 reviewer novelty 陈述只提供方向相容的补充案例，不承担主结论。

只有工程标准和实证标准同时成立，才能主张 Graph 不是装饰性附加项，而是提供了 GEAR-only 无法替代的 **Claim-level historical structural placement and perturbation view**。

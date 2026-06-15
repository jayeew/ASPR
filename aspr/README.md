# ASPR Agent System README

ASPR (Academic Scientific Paper Review) 是一个面向学术论文创新性评审的 multi-agent 系统。当前版本的核心目标不是简单生成一段评审文本，而是把检索增强、LATS 式自我反思、七维知识图谱创新指标和最终评审生成连接成一个可解释的评审流水线。

## 1. 系统目标

ASPR 当前主线任务：

1. 输入待评审论文的 `title`、`abstract` 和可选 `keywords`。
2. 从 Semantic Scholar 检索相关论文。
3. 使用 BGE-M3 召回和 OpenScholar reranker 精排相关工作。
4. 基于检索结果构建轻量 reference-neighborhood 图谱证据。
5. 计算七维图谱创新指标，并形成 `graph_metric_evidence`。
6. 使用 LATS 风格的生成、反思、改进循环，生成创新性评价。
7. 在最终报告中同时给出文本对比证据和图谱结构证据。

推荐论文叙事：

```text
Publication-day reference structure contains early signals of future knowledge-graph perturbation.
Graph-grounded multi-agent reviewers can use these signals to produce better calibrated and more interpretable innovation reviews.
```

## 2. 快速运行

### 2.1 环境依赖

核心依赖包括：

```bash
pip install openai langchain langchain-openai langchain-core langgraph pydantic requests pypdf backoff networkx numpy FlagEmbedding
```

可选依赖：

```bash
pip install nano-graphrag ollama
```

本地 LLM 默认使用 Ollama/OpenAI-compatible endpoint：

```text
http://localhost:11434/v1
```

### 2.2 主入口运行

```bash
cp .env.example .env  # fill S2_API_KEY if you use Semantic Scholar
python -m aspr.open_scholar \
  --large_model_port 38011 \
  --top_n 10
```

`aspr/open_scholar.py` 文件底部带有一个默认测试样例，可直接修改 `title`、`abstract` 和 `key_words` 后运行。

### 2.3 单独测试七维图谱 scorer

```bash
python - <<'PY'
from aspr.graph_innovation_scorer import GraphInnovationScorer

papers = [
    {
        "paperId": "p1",
        "title": "Graph neural networks for molecules",
        "abstract": "molecular graph learning",
        "venue": "NeurIPS",
        "fieldsOfStudy": ["Computer Science"],
        "citationCount": 10,
    },
    {
        "paperId": "p2",
        "title": "CRISPR gene editing",
        "abstract": "genome engineering biology",
        "venue": "Nature",
        "fieldsOfStudy": ["Biology"],
        "citationCount": 20,
    },
]

evidence = GraphInnovationScorer().score("test title", "test abstract", papers)
print(evidence.to_prompt_block())
PY
```

### 2.4 语法检查

```bash
python -m py_compile aspr/open_scholar.py aspr/lats.py aspr/prompts.py aspr/graph_innovation_scorer.py
```

## 3. 端到端执行流程

完整链路如下：

```text
User input
  |
  v
Reviewer.__call__()
  |
  +--> keywords_extract()
  |
  +--> OpenScholar.search_semantic_scholar()
  |      |
  |      +--> Semantic Scholar bulk search
  |      +--> paper metadata normalization
  |
  +--> retrieval_recall()
  |      |
  |      +--> BGE-M3 dense/sparse/ColBERT recall
  |
  +--> retrieval_rerank()
  |      |
  |      +--> OpenScholar reranker
  |
  +--> evaluate_paper_innovation()
         |
         +--> build_graph_metric_evidence()
         |      |
         |      +--> GraphInnovationScorer.score()
         |      +--> seven metric evidence block
         |
         +--> build_committee_evidence()
         |      |
         |      +--> ClaimDecomposer
         |      +--> EvidenceMapper
         |      +--> GraphAnalyst
         |      +--> SkepticReviewer
         |      +--> MetaReviewer
         |      +--> reviewer committee evidence block
         |
         +--> run_innovation_evaluation()
                |
                +--> generate_initial_response()
                +--> reflection_chain()
                +--> expand()
                +--> should_loop()
                +--> generate_final_report()
```

## 4. 模块功能说明

### 4.1 `open_scholar.py`

主入口和检索模块。

核心类和函数：

| 名称 | 功能 |
|---|---|
| `Reviewer` | 系统主入口，负责检索、缓存、rerank，并调用 LATS 评审链。 |
| `OpenScholar` | Semantic Scholar API 封装。 |
| `keywords_extract()` | 使用本地 LLM 从摘要中抽取最多 5 个关键词。 |
| `retrieval_recall()` | 使用 BGE-M3 对候选论文做第一阶段召回。 |
| `retrieval_rerank()` | 使用 OpenScholar reranker 对召回结果做精排。 |
| `_query_cache_paths()` | 根据 `title + abstract + keywords` 生成隔离缓存路径。 |

主要输出：

```text
data/retrieval_cache/{hash}_total_related_papers.jsonl
data/retrieval_cache/{hash}_most_related_papers.json
```

当前检索策略：

1. 如果没有传入关键词，先由 `keywords_extract()` 生成关键词。
2. 每个关键词调用 Semantic Scholar bulk search。
3. 合并候选论文。
4. BGE-M3 召回 top candidate pool。
5. reranker 精排到 `--top_n`。
6. 去重后传给 `evaluate_paper_innovation()`。

注意：

- 当前 Semantic Scholar 查询默认限制 `year: 2021-`，如果评审较老领域或历史 landmark，应放宽这个年份条件。
- 缓存已经按输入内容隔离，避免不同论文复用旧检索结果。
- 检索结果会保留 `doi`、`externalIds`、`fieldsOfStudy`、`s2FieldsOfStudy`，供图谱 scorer 使用。

### 4.2 `graph_innovation_scorer.py`

七维图谱创新证据模块。

核心类：

| 名称 | 功能 |
|---|---|
| `GraphInnovationScorer` | 根据当前相关论文集合构建轻量 reference-neighborhood 图，并计算七维指标。 |
| `GraphInnovationEvidence` | 封装七维指标、综合分数、置信度、主导机制和数据限制。 |

当前七维指标：

| 指标 | 解释 | 当前轻量实现 |
|---|---|---|
| `B` | 跨社区桥接位置 | target node 在 reference-neighborhood 图中的 betweenness。 |
| `RS` | 跨学科知识广度 | field 标签上的 Simpson diversity 近似。 |
| `DeltaQ0` | 社区边界扰动 | 加入 target 前后 modularity 的下降近似。 |
| `Uzzi` | 非典型组合 | field/venue 差异和 reference 间弱连接的组合近似。 |
| `RTD` | 参考目标社区多样性 | reference 社区标签上的 Simpson diversity。 |
| `BurtIP` | 结构洞潜力 | reference 子图低密度对应的创新潜力。 |
| `PDE` | 潜在扩散熵 | domain/field 标签分布的 normalized entropy。 |

默认权重来自当前 Fig.3 multi-domain diagnostic run：

```python
{
    "B": 0.01260526,
    "RS": 0.14861328,
    "DeltaQ0": 0.39528357,
    "Uzzi": 0.23449148,
    "RTD": 0.03278302,
    "BurtIP": 0.00431764,
    "PDE": 0.17190574,
}
```

重要限制：

- 当前 `GraphInnovationScorer` 是本地轻量版，只使用已检索相关论文的元数据近似构图。
- Nature 级实证版本应改为 OpenAlex DOI/reference 图谱版，严格使用论文发表当天可见的 reference list 和 `G- / G0`。
- 当前综合分数应该作为 agent 反思约束，而不是直接作为论文质量最终判定。

### 4.3 `review_committee.py`

Graph-grounded reviewer committee 模块。它把单链自我反思升级为多角色审稿委员会，形成 `Claim-Evidence-Graph-Counterclaim` 闭环。

核心结构：

| 名称 | 功能 |
|---|---|
| `ClaimCard` | 单条创新声明的证据契约，包含 claim、相关文献、图谱支持、反方质疑和不确定性。 |
| `AgentReview` | 单个 agent 的结构化意见，包含分数、优点、弱点和 required revisions。 |
| `CommitteeReport` | 委员会最终报告，包含 claim cards、agent reviews、分歧分数、meta-review 和 recommended tone。 |

五个 agent：

| Agent | 功能 |
|---|---|
| `ClaimDecomposer` | 从标题和摘要中抽取 2-5 个候选创新声明。 |
| `EvidenceMapper` | 用相关论文元数据为每个 claim 绑定对比文献。 |
| `GraphAnalyst` | 将七维图谱指标解释成机制证据，例如边界扰动、非典型组合、知识广度、结构洞。 |
| `SkepticReviewer` | 主动找过度声称、缺失对比、弱证据和替代解释。 |
| `MetaReviewer` | 聚合多方意见，输出 `disagreement_score` 和 `recommended_tone`。 |

推荐语气规则：

```text
conservative: 分歧高、图谱置信度低、或存在高不确定 claim。
balanced: 有一定证据，但仍需按 claim 呈现限制。
assertive: 图谱分数和置信度都高，且 SkepticReviewer 反对较弱。
```

当前 v1 不额外调用 LLM 或外部 API，使用规则型 agent 产生稳定结构化输出。后续可以在不改变 schema 的前提下，把某些 agent 替换成 LLM agent。

### 4.4 `lats.py`

LATS 风格的生成、反思、改进与最终报告模块。

核心数据结构：

| 名称 | 功能 |
|---|---|
| `PaperInfo` | 相关论文结构化信息，包含 title、authors、venue、year、abstract、doi、fields。 |
| `Reflection` | 自我反思评分结构，包含文本质量和图谱证据对齐评分。 |
| `Node` | 搜索树节点，保存候选评价、反思结果、价值分数、父子关系。 |
| `TreeState` | LangGraph 状态，保存论文信息、相关论文、图谱证据、committee evidence 和当前最佳评价。 |

核心链路：

| 函数 | 功能 |
|---|---|
| `generate_initial_response()` | 生成初始创新性评价，并立即反思。 |
| `reflection_chain()` | 调用 LLM 对当前候选评价做结构化反思。 |
| `expand()` | 基于反思反馈生成多个改进候选，并保留 top beam。 |
| `should_loop()` | 判断是否继续搜索，默认高度上限为 5。 |
| `generate_final_report()` | 基于最佳候选和图谱证据生成最终报告。 |
| `build_committee_evidence()` | 调用 reviewer committee，生成 claim-level 证据块。 |
| `evaluate_paper_innovation()` | 给 `open_scholar.py` 调用的一体化接口。 |

当前反思维度：

```text
innovation_accuracy          创新性识别准确性
comparison_adequacy          对比充分性
citation_normative           引用规范性
graph_metric_alignment       图谱结构证据对齐
uncertainty_calibration      不确定性校准
readability                  表达清晰度
```

节点奖励函数：

```text
reward =
  innovation_accuracy * 0.30
  + comparison_adequacy * 0.20
  + citation_normative * 0.15
  + graph_metric_alignment * 0.20
  + uncertainty_calibration * 0.10
  + readability * 0.05
```

`found_solution` 的判定会额外要求：

```text
graph_metric_alignment >= 6
uncertainty_calibration >= 5
```

这意味着候选评价即使语言上看起来不错，如果没有正确使用图谱证据，或者在证据弱时夸大创新性，也不会被提前接受为高质量解。

Committee 会进一步影响 reward：

```text
disagreement_score 高 -> 降低不确定性校准和图谱对齐奖励，并强制保守语气。
recommended_tone = conservative -> found_solution 不能提前为 true。
recommended_tone = assertive 且分歧低 -> 允许更积极但仍需证据绑定的创新性表述。
```

### 4.5 `prompts.py`

集中保存 LLM prompt。

核心 prompt：

| Prompt | 用途 |
|---|---|
| `prompts_keywords_extraction` | 从摘要提取关键词。 |
| `INNOVATION_GENERATION_PROMPT` | 初始创新性评价生成。 |
| `INNOVATION_REFLECTION_PROMPT` | 结构化自我反思。 |
| `INNOVATION_IMPROVEMENT_PROMPT` | 根据反思反馈改进评价。 |
| `FINAL_INNOVATION_REPORT_PROMPT` | 生成最终创新性评价报告。 |

所有创新性评价相关 prompt 都接收：

```text
{graph_metric_evidence}
{committee_evidence}
```

这块证据会提醒模型：

- 图谱指标是约束创新性表述的证据，不是自动好坏标签。
- `DeltaQ0 / Uzzi` 高时，可讨论边界扰动和非典型组合。
- `RS / PDE` 高时，可讨论知识广度和潜在扩散。
- `B / RTD / BurtIP` 高时，可讨论桥接、结构洞和低冗余连接。
- 证据置信度低时，必须保守表达并说明不确定性。
- 审稿委员会 evidence 要求最终报告按 claim 逐条包含文本证据、相关工作差异、图谱支持、反方质疑和不确定性。

### 4.6 `graph_rag.py`

GraphRAG 辅助模块，基于 `nano-graphrag` 和 Ollama。

核心函数：

| 名称 | 功能 |
|---|---|
| `insert(message)` | 将文本插入 GraphRAG 工作目录。 |
| `query(query, mode="global")` | 查询 GraphRAG。 |
| `ollama_model_if_cache()` | 带缓存的 Ollama LLM 调用。 |
| `ollama_embedding()` | 使用 Ollama embedding model 生成向量。 |

当前主链路中 `graph_rag.py` 没有默认启用。它适合后续接入全文 PDF 和长文本相关工作分析。

### 4.7 `pdf_downloader.py`

ACL Anthology PDF 下载模块。

核心类：

| 名称 | 功能 |
|---|---|
| `ACLPDFDownloader` | 下载 ACL PDF，支持 HTML 页面解析和重试。 |
| `download_acl_pdf()` | 从 ACL URL 下载 PDF。 |
| `download_pdf_direct()` | 直接下载 PDF URL。 |
| `test_acl_download()` | 简单手动测试函数。 |

当前主链路中 PDF 下载逻辑在 `open_scholar.py` 里保留，但默认注释掉。后续如要从摘要级 RAG 升级到全文级 RAG，可以恢复该链路。

## 5. 图谱证据和审稿委员会如何进入 agent 自我反思

当前系统不是把七维指标和 committee 结果只拼进最终报告，而是在每轮生成和反思中都注入 `graph_metric_evidence` 与 `committee_evidence`：

```text
graph_metric_evidence
committee_evidence
  |
  +--> initial generation prompt
  +--> reflection prompt
  +--> improvement prompt
  +--> final report prompt
```

这会影响三个地方：

1. 初始评价阶段：模型必须把创新性判断和图谱结构证据对应起来。
2. 反思阶段：`graph_metric_alignment` 会检查候选评价是否正确解释七指标，committee 分歧会校准 reward。
3. 改进阶段：如果图谱证据弱或 committee 建议 `conservative`，模型应减少“颠覆性”“开创性”等过度表述。
4. 最终报告阶段：必须按 claim cards 逐条呈现文本证据、相关工作差异、图谱支持、反方质疑和不确定性。

示例：

```text
如果 DeltaQ0 和 Uzzi 高：
  可以写“该工作可能连接了此前较少共同出现的知识组合，并对已有社区边界产生扰动”。

如果 confidence 低：
  应写“基于当前检索到的相关论文，图谱证据只能提供弱支持，仍需更完整 reference graph 验证”。

如果 committee disagreement 高：
  应写“审稿委员会对该 claim 的证据充分性存在分歧，因此当前只能给出保守创新性判断”。
```

## 6. 数据与缓存

### 6.1 检索缓存

缓存目录：

```text
data/retrieval_cache/
```

缓存文件：

```text
{hash}_total_related_papers.jsonl
{hash}_most_related_papers.json
```

`hash` 来源：

```text
title + abstract + normalized keywords
```

这样可以避免不同待评审论文复用同一个旧检索结果。

### 6.2 输出

当前主入口返回：

```python
{
    "innovation_evaluation": "...最终创新性评价报告...",
    "graph_metric_evidence": {
        "metrics": {...},
        "weighted_score": 0.0,
        "confidence": 0.0,
        "top_mechanisms": [...],
        "limitations": [...],
        "diagnostics": {...},
    },
    "committee_report": {
        "claim_cards": [...],
        "agent_reviews": [...],
        "disagreement_score": 0.0,
        "meta_review_summary": "...",
        "recommended_tone": "conservative|balanced|assertive",
    },
    "committee_disagreement_score": 0.0,
    "recommended_tone": "balanced",
    "evaluation_log": [...],
    "success": True,
}
```

## 7. 当前限制

1. 图谱 scorer 是轻量近似版，还没有直接拉取 OpenAlex reference graph。
2. Semantic Scholar 检索依赖 API key 和网络稳定性。
3. LATS 主流程依赖 LangChain、LangGraph、本地 LLM 服务。
4. 反思分数仍由 LLM 生成，但已经被七维图谱证据和 reviewer committee 分歧约束。
5. 当前 Fig.3 权重来自 diagnostic run，不能直接作为强因果结论。
6. 主流程仍以摘要和相关论文元数据为主，全文 PDF RAG 尚未默认启用。

## 8. 推荐的后续开发路线

### 8.1 工程增强

1. 将 `GraphInnovationScorer` 升级为 OpenAlex DOI/reference graph 版本。
2. 为每篇待评审论文构建严格的 `G-` 和 `G0`。
3. 将 `experiments/kg_validator/metrics.py` 的正式七指标计算迁移到 `aspr/` 可复用模块。
4. 增加 JSON schema 输出，强制最终报告包含：
   - innovation claims
   - supporting references
   - graph evidence
   - uncertainty
   - reviewer recommendation
5. 增加 unit tests 和小型 fixture，避免检索、缓存、scorer、committee 回归。

### 8.2 论文级增强

1. 扩大多领域数据规模，目标至少 `10-20` 个领域和 `>100k` papers。
2. 保证 reference closure coverage，目标 `>80%`。
3. 使用 out-of-fold 验证学习七指标权重。
4. 做 ablation：
   - no graph evidence
   - equal weights
   - learned weights
   - no skeptic/reflection
   - no uncertainty calibration
5. 做 human evaluation：
   - LLM baseline
   - RAG reviewer
   - ASPR without graph
   - ASPR with graph evidence
6. 将最终 claim 聚焦为：

```text
Graph-grounded agentic review improves calibration and interpretability of prospective scientific innovation assessment.
```

## 9. 推荐目录演进

后续建议将 `aspr/` 演进为：

```text
aspr/
├── open_scholar.py              # retrieval and main CLI
├── lats.py                      # agentic search and reflection
├── review_committee.py          # graph-grounded reviewer committee
├── graph_innovation_scorer.py   # lightweight graph evidence
├── graph_metrics.py             # future formal OpenAlex seven metrics
├── graph_data.py                # future OpenAlex/S2 graph construction
├── graph_rag.py                 # optional long-context GraphRAG
├── pdf_downloader.py            # PDF download helper
├── prompts.py                   # prompt templates
└── README.md                    # this file
```

## 10. 最小调试清单

修改代码后建议依次运行：

```bash
python -m py_compile aspr/open_scholar.py aspr/lats.py aspr/prompts.py aspr/graph_innovation_scorer.py aspr/review_committee.py
```

```bash
python - <<'PY'
from aspr.graph_innovation_scorer import GraphInnovationScorer
e = GraphInnovationScorer().score("title", "abstract", [])
print(e.to_dict())
PY
```

```bash
python tests/test_review_committee.py
```

如果完整主流程报 `ModuleNotFoundError`，优先检查：

```bash
python -c "import langchain_core, langchain_openai, langgraph, pydantic"
```

如果 Semantic Scholar 检索为空，检查：

```bash
python - <<'PY'
from aspr.env import getenv
print(bool(getenv("S2_API_KEY")))
PY
```

以及 `open_scholar.py` 中的年份过滤：

```python
"year": "2021-"
```

历史领域或 landmark paper 评审通常需要放宽该过滤条件。

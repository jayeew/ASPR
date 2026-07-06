# Fig. 8 gpt-image-2 绘图提示词

## 当前决定

Fig. 8 放弃当前本地 Matplotlib 渲染器形式，不再维护 `render_fig8.py`、renderer test 或已生成的 `outputs/kg_perturbation_fig8/` 结果文件。

Fig. 8 现在只作为一个面向 `gpt-image-2` 的出版级算法框架图设计说明。这个目录应保留 README 作为唯一核心交付物，不再放置绘图代码。

## 基于非实验代码的 ASPR 系统分析

以下分析只来自 `aspr/` 主包与直接支撑系统/训练的数据脚本，不以 `experiments/` 下的 Fig.1-Fig.10 实验绘图代码为依据。

### 1. 主评审入口：`aspr.open_scholar`

ASPR 的在线评审入口是 `Reviewer`。输入是待评审论文的 `title`、`abstract` 和可选 `keywords`。

真实流程：

```text
Input manuscript metadata
-> keywords_extract()
-> OpenScholar retrieval
-> BGE-M3 recall
-> OpenScholar reranker
-> top related papers
-> evaluate_paper_innovation()
```

关键模块含义：

- `keywords_extract()`：用本地 OpenAI-compatible/Ollama LLM 从摘要抽取最多 5 个关键词。
- `OpenScholar.search_semantic_scholar()`：默认调用 Semantic Scholar，也支持 `retrieval_provider=openalex`。
- `retrieval_recall()`：优先用 BGE-M3 做 dense/sparse/ColBERT 召回；模型不可用时回退 TF-IDF。
- `retrieval_rerank()`：优先用 OpenScholar reranker 精排；模型不可用时回退 TF-IDF。
- 检索结果保留 `title`、`abstract`、`year`、`venue`、`citationCount`、`doi`、`externalIds`、`fieldsOfStudy` 等字段，用于后续图谱证据和引用。

Fig. 8 应把这一层画成左侧的 retrieval/prior-art intake，而不是画成一个普通 chatbot。

### 2. 七维图谱创新证据：`aspr.graph_innovation_scorer`

`GraphInnovationScorer` 把已检索到的相关论文近似视为目标论文的 reference neighborhood，构建一个轻量有向图：

```text
target paper
-> retrieved reference-neighborhood papers
-> field / venue / lexical-overlap edges
-> seven graph innovation indicators
```

七个指标必须作为图中 graph evidence engine 的核心输出出现：

```text
B       Bridge position
RS      Rao-Stirling breadth
DeltaQ0 Boundary perturbation
Uzzi    Atypical recombination
RTD     Reference target diversity
BurtIP  Structural-hole potential
PDE     Prospective diffusion entropy
```

输出不是最终结论，而是：

```text
weighted graph prior S_w
confidence
top mechanisms
limitations
diagnostics
```

Fig. 8 必须把图谱分数画成 novelty-language calibration prior，而不是论文质量自动标签。

### 3. Reviewer committee：`aspr.review_committee`

ASPR 主包里已经有一个规则型 graph-grounded reviewer committee。它不是额外实验，而是 LATS 前的结构化证据生成层。

五个 agent：

```text
ClaimDecomposer
EvidenceMapper
GraphAnalyst
SkepticReviewer
MetaReviewer
```

它们形成：

```text
claim cards
supporting references
graph support
counterarguments
uncertainty
disagreement score
recommended tone
```

Fig. 8 应把 committee 画成 graph agent 内部的 claim-evidence-graph-counterclaim loop，用来约束最终评审语气。

### 4. LATS 生成、反思、改进：`aspr.lats`

`lats.py` 是最终创新性评价生成器。它使用 LangGraph 状态图：

```text
generate_initial_response
-> reflection_chain
-> expand / improve
-> should_loop
-> generate_final_report
```

反思评分维度包括：

```text
innovation accuracy
comparison adequacy
citation normativity
graph metric alignment
uncertainty calibration
readability
```

LATS 同时接收：

```text
related papers
graph_metric_evidence
committee_evidence
committee_disagreement_score
recommended_tone
paper dossier
```

Fig. 8 应表现为“生成-反思-改进-最终报告”的 agentic reasoning loop，且这个 loop 受图谱证据和 reviewer committee 约束。

### 5. Prompt schema：`aspr.prompts`

最终报告提示词要求稳定小节：

```text
Calibrated innovation stance
Novelty claims
Prior-art comparison
Evidence and rigor
Limitations and uncertainty
Future work
Claim-level Committee Review
Fig.3-weighted Graph Prior
References
```

Fig. 8 的 final report schema 必须保留这些字段。不要把最终输出画成单一分数。

### 6. Optional GraphRAG：`aspr.graph_rag`

`graph_rag.py` 是一个可选辅助模块，基于 `nano-graphrag`、Ollama `qwen3:8b` 和 `nomic-embed-text` embedding，支持：

```text
insert paper/reference text
query graph-indexed context
```

Fig. 8 可以把 GraphRAG 画成 graph agent 的 optional memory / evidence store，但不要让它抢占主流程。

### 7. Corpus layer：`aspr.corpus`

`aspr.corpus` 管理可复用论文图谱语料层。核心表包括：

```text
works.csv
citations.csv
topics.csv
topic_edges.csv
domains.csv
landmarks.csv
```

它支持：

```text
OpenAlex canonical graph source
domain seeds
manual landmarks
strict anchor policy
fig1 / fig2 / fig3 / fig5 compatible views
quality reports
```

Fig. 8 可以把这个层画成 graph agent 的 canonical scholarly graph substrate。

### 8. ASPR-Qwen / review-style SFT 支撑脚本

非实验脚本中与 ASPR-Qwen 相关的真实支撑包括：

- `scripts/download_nature.py`：通过 PubMed / Nature 页面下载 Nature 系列论文 PDF，并尝试提取 Transparent Peer Review 文件。
- `scripts/download_six_journals_2023_2025.sh`：批量下载 6 本 Nature Portfolio 期刊 2023-2025 年论文与同行评审文件。
- `scripts/hfdata_builder.py`：把 paper / reconstruction markdown 构造成 HuggingFace SFT 数据。
- `scripts/train_sft_qwen.sh`：用 `Qwen/Qwen3-0.6B` 和 `WestlakeNLP/DeepReview-13K` 做全量 SFT。
- `scripts/train_sft_lora_qwen.sh`：用 `Qwen/Qwen3-8B` 和 LoRA / 4-bit 做 review-style SFT。

因此 Fig. 8 的 ASPR-Qwen lane 应表述为：

```text
review-style SFT reviewer model
Qwen base family
Nature / transparent peer-review local corpus when available
DeepReview-style review SFT data in current training scripts
```

不要把 ASPR-Qwen 画成已经证明具有 peer-review performance 的模型；它是系统架构中的 reviewer-style generation branch，真实性能需要另行评估。

## Fig. 8 应表达的核心信息

Fig. 8 的一句话信息：

```text
ASPR combines graph-grounded agentic novelty evaluation with an ASPR-Qwen reviewer-style SFT branch, then fuses and verifies both outputs into an evidence-grounded academic review report.
```

必须画清楚三层：

1. Graph-grounded ASPR agent：负责检索、prior-art 对比、图谱指标、claim cards、LATS 反思和证据化 novelty stance。
2. ASPR-Qwen reviewer lane：负责从 review-style SFT 中学习评审语言、关注点、concerns、limitations 和 recommendation 格式。
3. Fusion + verifier：最终报告必须由两条分支融合，并通过 claim-evidence alignment、overclaim check、uncertainty calibration 和 human-in-the-loop flag。

## 推荐图面结构

使用一个横向 16:10 多 panel scientific workflow figure。推荐排布：

```text
8a  8a  8a
8b  8c  8d
8e  8e  8f
```

其中：

- 8a：全局双路径架构，总览必须最大。
- 8b：retrieval + graph evidence engine。
- 8c：reviewer committee + LATS loop。
- 8d：ASPR-Qwen review-style SFT lane。
- 8e：fusion + verifier + safety gates。
- 8f：final evidence-grounded review schema。

## 统一视觉规范

```text
Background: clean white or very light warm gray
Graph agent lane: blue #2563EB
ASPR-Qwen lane: purple #7C3AED
Nature / review corpus: deep red #8B1E2D
Verifier / uncertainty: orange #F0986E
Fusion and final report: charcoal #111827
Neutral evidence tables: slate #64748B
```

风格要求：

- Nature / Science style publication workflow figure。
- Flat vector-like blocks, crisp arrows, clean typography。
- No 3D, no shadows-heavy UI, no cartoon robots, no chat bubbles as the main metaphor。
- Use exact labels where possible; if text rendering becomes crowded, use numbered callouts with a small legend.
- Every arrow direction must be logical and visible.

## 可直接用于 gpt-image-2 的主提示词

把下面整段作为 `gpt-image-2` prompt：

```text
Create a publication-ready scientific workflow figure for a paper, titled:
"Fig. 8 | ASPR algorithm framework for evidence-grounded academic review"

Canvas and style:
- Wide landscape figure, 16:10 aspect ratio.
- Clean white or very light gray background.
- High-end Nature / Science style methods figure.
- Flat vector-like blocks, crisp arrows, restrained colors, readable typography.
- No 3D rendering, no cartoon robots, no decorative gradients, no dark background.
- Use clear panel labels: 8a, 8b, 8c, 8d, 8e, 8f.
- Preserve exact module names. Do not invent extra system modules.
- This is an architecture / method workflow figure, not a statistical performance plot.

Core message:
ASPR is not a single LLM and not only a graph metric system. ASPR combines:
1. a graph-grounded ASPR agent for retrieval, graph evidence, claim cards, and LATS-style reflection;
2. an ASPR-Qwen reviewer-style SFT branch for review language, concerns, limitations, and recommendation structure;
3. a fusion and verifier layer that produces the final evidence-grounded review report.

Overall layout:
Use six panels in a coherent workflow:

Panel 8a, large top panel, "Overall ASPR architecture":
Draw a left-to-right system diagram.
Left node: "Input manuscript"
Subnode below it: "title / abstract / paper dossier"
Arrow to: "Parsing + keyword extraction"
Then branch into two horizontal lanes:

Upper blue lane named "Graph-grounded ASPR agent":
Nodes in this lane:
"Semantic Scholar / OpenAlex retrieval"
"BGE-M3 recall"
"OpenScholar reranker"
"Reference-neighborhood graph"
"Seven-indicator graph prior S_w"
"Reviewer committee claim cards"
"LATS generation-reflection-improvement"

Lower purple lane named "ASPR-Qwen reviewer":
Nodes in this lane:
"Review-style SFT corpus"
"Qwen base model"
"ASPR-Qwen reviewer"
"Reviewer-style critique"

Both lanes converge into a charcoal node:
"Dual-branch fusion"
Then arrow to orange node:
"Verifier / safety gates"
Then arrow to final node:
"Evidence-grounded review report"

Add a small loop arrow from "Verifier / safety gates" back to "Dual-branch fusion" labeled:
"revise unsupported claims"

Panel 8b, "Retrieval and graph evidence engine":
Show the graph-agent evidence pipeline in detail:
"keywords"
-> "Semantic Scholar / OpenAlex"
-> "related papers"
-> "BGE-M3 recall"
-> "OpenScholar rerank"
-> "reference-neighborhood graph"
-> "GraphInnovationScorer"
Output badge: "S_w + confidence + limitations"
Show the seven graph indicators as small metric chips:
"B", "RS", "DeltaQ0", "Uzzi", "RTD", "BurtIP", "PDE"
Add a note in tiny caption style:
"Graph prior calibrates novelty language; it is not an automatic quality label."

Panel 8c, "Reviewer committee and LATS loop":
Draw a circular or looped reasoning module.
Committee nodes:
"ClaimDecomposer"
"EvidenceMapper"
"GraphAnalyst"
"SkepticReviewer"
"MetaReviewer"
Output:
"claim cards"
"counterarguments"
"disagreement score"
"recommended tone"
Connect this to a LATS loop:
"initial evaluation"
-> "reflection"
-> "improvement"
-> "best candidate"
Use a loop arrow labeled:
"iterate until calibrated or max iterations"

Panel 8d, "ASPR-Qwen review-style SFT branch":
Draw a training-to-inference lane in purple and deep red.
Data sources:
"Nature manuscripts"
"Transparent Peer Review files"
"DeepReview-style review data"
Then:
"paper-review pairing"
-> "instruction / response formatting"
-> "SFT"
-> "ASPR-Qwen reviewer"
Inference output:
"reviewer-style critique"
"concerns"
"limitations"
"recommendation language"
Add a boundary note:
"Reviewer-style generation branch; performance claims require separate evaluation."

Panel 8e, "Fusion, verification, and safety gates":
Draw two inputs into fusion:
Input 1 from graph agent:
"novelty stance + prior-art evidence + graph support + risk flags"
Input 2 from ASPR-Qwen:
"review style + concerns + limitations + recommendation draft"
Fusion node:
"rubric-guided fusion"
Verifier checks as orange checkboxes:
"claim-evidence alignment"
"prior-art contradiction"
"unsupported novelty claim"
"hallucination / factuality"
"S_w and evidence consistency"
"low-confidence human flag"
Bad output loops back to fusion:
"revise"
Very low confidence goes to:
"human-in-the-loop review"

Panel 8f, "Final evidence-grounded review schema":
Draw a structured report card, not a scorecard.
Fields:
"Calibrated innovation stance"
"Novelty claims"
"Prior-art comparison"
"Evidence and rigor"
"Limitations and uncertainty"
"Claim-level committee review"
"Fig.3-weighted graph prior"
"Recommendation / confidence"
"References / evidence links"

Color mapping:
- Graph agent lane: blue.
- ASPR-Qwen lane: purple.
- Nature / review corpus: deep red.
- Verifier / uncertainty / safety gates: orange.
- Fusion and final report: charcoal.
- Neutral data tables and metadata: slate gray.

Strict negative constraints:
- Do not depict ASPR as a single chatbot.
- Do not depict ASPR-Qwen as the only final answer source.
- Do not claim human replacement.
- Do not draw accuracy bars, ROC curves, or performance numbers.
- Do not use anthropomorphic review-performance wording; use "review-style" or "evidence-grounded review".
- Do not invent modules outside the labels listed above.
- Do not show graph metrics as final truth labels; show them as calibration evidence.

The final image should look like a polished methods overview figure suitable for a scientific manuscript, with clean panel alignment, legible labels, and all arrows clearly showing data flow and feedback loops.
```

## 可选精修提示词

如果第一版图中文字太拥挤，用下面提示词要求精修：

```text
Revise the figure to improve readability. Keep the same six-panel layout and the same module names. Increase whitespace, reduce tiny explanatory text, and convert crowded details into numbered callouts. Preserve all arrows, especially the two-lane split, the convergence at Dual-branch fusion, the verifier-to-fusion revision loop, and the final evidence-grounded review schema. Do not add performance plots or new modules.
```

如果第一版错误地把 ASPR 画成单一 LLM，用下面提示词修正：

```text
Correct the architecture: ASPR must have two separate branches. The upper branch is "Graph-grounded ASPR agent" with retrieval, graph prior S_w, reviewer committee, and LATS loop. The lower branch is "ASPR-Qwen reviewer" with review-style SFT and reviewer-style critique. Both branches must merge only at "Dual-branch fusion", then pass through "Verifier / safety gates" before the final report. Do not show ASPR-Qwen alone producing the final answer.
```

## 生成后人工校对清单

生成图片后逐项检查：

- 是否有 8a-8f 六个 panel。
- 是否清楚显示上蓝下紫两条 lane。
- 是否保留 `Semantic Scholar / OpenAlex`、`BGE-M3 recall`、`OpenScholar reranker`。
- 是否显示七指标：`B`、`RS`、`DeltaQ0`、`Uzzi`、`RTD`、`BurtIP`、`PDE`。
- 是否把 `S_w` 画成 graph prior / calibration evidence，而不是 final score。
- 是否显示 reviewer committee 的五个角色。
- 是否显示 LATS 的 generation-reflection-improvement loop。
- 是否显示 ASPR-Qwen 的 SFT branch，但没有声称 checkpoint performance。
- 是否显示 fusion 后还有 verifier / safety gates。
- 是否显示 unsupported claims revise loop。
- 是否最终输出 structured evidence-grounded review report，而不是单一 recommendation。
- 是否没有出现拟人化评审性能表述、`replace human reviewers`、accuracy claim、性能数字或无依据 superiority claim。

## 推荐 caption 边界

可以写：

```text
Fig. 8 summarizes the ASPR application architecture. A graph-grounded agent retrieves prior art, builds reference-neighborhood graph evidence, computes a seven-indicator weighted prior, and produces claim-level evidence through reviewer committee and LATS-style reflection. In parallel, ASPR-Qwen provides a reviewer-style critique branch trained with review-style SFT data. The two branches are fused and verified before emitting an evidence-grounded review report.
```

不要写：

```text
Fig. 8 proves ASPR review accuracy.
Fig. 8 shows ASPR replaces human peer reviewers.
Fig. 8 shows ASPR-Qwen alone generates the final review.
Fig. 8 demonstrates anthropomorphic peer-review performance.
```

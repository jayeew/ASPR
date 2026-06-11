# Fig.4 Agent-based Evaluation README

## 一句话目标

Fig.4 用真实 Nature 子刊论文 PDF、真实 Transparent Peer Review PDF 和 ASPR agent 输出，系统评估自动创新性评价和人工同行评审之间的关系。

这张图不使用示意数据。每个数值都必须能从 manifest、agent 输出、解析文本和指标表追溯回来。

## 这张图想回答什么

Fig.4 回答四个问题：

```text
1. Consistency: agent 的创新性评价和人工 peer review 在语义上是否一致？
2. Efficiency: agent 生成评价相对人工 peer review 基准能节省多少时间？
3. Readability: agent 输出是否比人工评审文本更容易读、错误更少？
4. Coverage: agent 是否覆盖了人工评审关注的关键创新点、方法、证据和局限？
```

最终输出包括 Fig.4 a-e 五个 panel、全图拼接文件、逐论文指标表和输入审计表。

## 数据边界

Nature 子刊论文 PDF 和 Transparent Peer Review PDF 假设已经存在于本地数据集或外部数据盘。Fig.4 不负责重新下载 PDF，也不把原始 PDF 提交到仓库。

所有样本必须通过显式 manifest 对齐：

```text
experiments/fig4/fig4_manifest.csv
```

推荐输出目录：

```text
outputs/fig4/
  fig4_input_audit.csv
  fig4_agent_outputs.jsonl
  fig4_metrics_summary.csv
  fig4_panel_a.png
  fig4_panel_b.png
  fig4_panel_c.png
  fig4_panel_d.png
  fig4_panel_e.png
  fig4_full.png
  cache/
    <paper_id>/
      retrieved_papers.json
      agent_eval.json
      parsed_text.json
      metrics.json
```

`experiments/fig4/` 只放实验规范、入口脚本和轻量配置；大体量 PDF、解析文本、agent 运行结果和图片统一写入 `outputs/fig4/`。

## Manifest 设计

`fig4_manifest.csv` 的每一行对应一篇已发表论文及其同行评审文件。

必需字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `paper_id` | str | 实验内唯一 ID，建议使用 DOI 后缀或稳定 hash |
| `doi` | str | 论文 DOI |
| `title` | str | 论文标题 |
| `year` | int | 出版年份 |
| `journal` | str | Nature 子刊名称 |
| `article_pdf_path` | str | 本地论文 PDF 路径 |
| `peer_review_pdf_path` | str | 本地 Transparent Peer Review PDF 路径 |
| `abstract` | str | 论文摘要；如为空，后续脚本可尝试从 metadata 或 PDF 抽取 |
| `keywords` | str | 逗号分隔关键词；如为空，使用 `aspr.open_scholar.keywords_extract()` |

可选字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `article_url` | str | Nature 文章页面 |
| `peer_review_url` | str | Transparent Peer Review file URL |
| `publication_date` | str | 发表日期 |
| `subject_area` | str | 领域标签 |
| `notes` | str | 人工备注 |

Manifest 审计规则：

- `paper_id` 重复：写入 `fig4_input_audit.csv`，该组重复样本默认排除。
- `article_pdf_path` 不存在或不可读：排除。
- `peer_review_pdf_path` 不存在或不可读：排除。
- `title`、`journal`、`year`、`peer_review_pdf_path` 缺失：排除。
- PDF 可读但抽取文本为空：排除，并记录 `empty_article_text` 或 `empty_peer_review_text`。
- `abstract` 缺失不直接排除，但必须记录抽取来源或 fallback 失败原因。

## 文本解析

PDF 文本抽取默认使用项目已有依赖 `pypdf.PdfReader`，实现时应把解析逻辑放在 Fig.4 独立脚本中，避免修改 `aspr.open_scholar.extract_text_with_pypdf()` 的现有行为。

论文 PDF 解析输出：

```json
{
  "paper_id": "example",
  "article_text": "...",
  "abstract_source": "manifest|pdf|metadata",
  "word_count": 12345
}
```

Peer review PDF 解析输出：

```json
{
  "paper_id": "example",
  "peer_review_text": "...",
  "included_sections": ["reviewer_reports", "reviewer_comments"],
  "excluded_sections": ["author_response", "editor_decision", "references"],
  "word_count": 5432
}
```

默认只保留 reviewer comments / reviewer reports。以下内容默认排除：

- author response / response to reviewers；
- editor decision letter；
- references / bibliography；
- figure legends、supplementary notes、cover page；
- publisher boilerplate。

如果自动切分失败，保留完整 peer review 文本但在 audit 中标记 `review_section_parse_warning`。主图可以包含该样本，但敏感性分析应报告排除 warning 样本后的结果。

## Agent 评价生成

Fig.4 不应在批量运行中直接复用 `aspr.open_scholar.Reviewer.__call__()` 的默认缓存路径，因为现有实现会读写全局文件：

```text
data/total_related_papers.json
data/most_related_paper.json
```

后续实现应新建 Fig.4 wrapper，复用现有函数和类，但把所有中间结果写入逐论文缓存：

```text
outputs/fig4/cache/<paper_id>/
```

推荐复用的项目入口：

- `aspr.open_scholar.keywords_extract`
- `aspr.open_scholar.OpenScholar.search_semantic_scholar`
- `aspr.open_scholar.retrieval_recall`
- `aspr.open_scholar.retrieval_rerank`
- `aspr.lats.evaluate_paper_innovation`

每篇论文的 agent 输出写入 `fig4_agent_outputs.jsonl`，每行至少包含：

```json
{
  "paper_id": "example",
  "title": "Paper title",
  "keywords": ["keyword1", "keyword2"],
  "retrieved_papers_count": 10,
  "retrieved_papers_cache": "outputs/fig4/cache/example/retrieved_papers.json",
  "innovation_evaluation": "...",
  "evaluation_log": ["..."],
  "agent_runtime_seconds": 287.4,
  "success": true,
  "failure_reason": ""
}
```

运行要求：

- 每篇论文单独计时，使用 wall-clock seconds。
- 失败样本不阻塞整批任务，写入 `success=false` 和 `failure_reason`。
- 已存在 `agent_eval.json` 时默认复用，除非显式传入 `--force-agent`。
- 不允许覆盖或依赖 `data/most_related_paper.json` 作为 Fig.4 主缓存。

## 指标定义

所有指标汇总到：

```text
outputs/fig4/fig4_metrics_summary.csv
```

推荐字段：

| 字段 | 说明 |
| --- | --- |
| `paper_id` | 与 manifest 对齐 |
| `journal` | 期刊名称 |
| `year` | 出版年份 |
| `consistency_cosine` | agent 与 peer review 的语义相似度 |
| `agent_runtime_seconds` | agent 运行耗时 |
| `human_baseline_hours` | 人工评审耗时基准 |
| `speedup_vs_human` | 人工基准秒数 / agent 秒数 |
| `time_saved_percent` | `1 - agent_seconds / human_seconds` |
| `agent_errors_per_5000_words` | agent 输出可读性错误率 |
| `peer_errors_per_5000_words` | peer review 错误率 |
| `coverage_score` | 人工关注点被 agent 覆盖比例 |
| `included_in_main` | 是否进入主图 |
| `exclusion_reason` | 排除原因 |

### Consistency

目的：衡量 agent 评价和人工 peer review 在语义上的一致性。

默认方法：

```text
embedding(agent_eval_text) vs embedding(peer_review_text)
metric = cosine similarity
range = [-1, 1], 主图可线性映射到 [0, 1]
```

推荐模型优先级：

1. `BAAI/bge-m3`，与项目已有 `FlagEmbedding` 使用习惯一致；
2. 本地可用的 SentenceTransformer / SciBERT；
3. OpenAI-compatible embedding 服务，前提是记录模型名和服务地址。

同一批 Fig.4 结果必须使用同一个 embedding 模型。模型名、版本、设备和 batch size 写入 `outputs/fig4/fig4_run_config.json`。

### Efficiency

目的：评估 agent 评价生成速度。

默认人工基准：

```text
human_baseline_hours = 5
```

该值来自外部 peer-review 时间基准，用作主图默认值；同时建议报告 `3, 5, 10` 小时的敏感性区间。

计算方式：

```text
human_seconds = human_baseline_hours * 3600
speedup_vs_human = human_seconds / agent_runtime_seconds
time_saved_percent = max(0, 1 - agent_runtime_seconds / human_seconds)
```

如果 agent 运行失败或耗时为 0，该样本不进入 efficiency 主图。

### Readability

目的：比较 agent 输出和 peer review 文本的语言错误率。

推荐工具：

- `language-tool-python`；
- 或等价的 LanguageTool HTTP 服务；
- 如不能运行 LanguageTool，后续脚本必须显式标记 `readability_unavailable`，不能用伪造错误数替代。

错误类别：

```text
spelling
grammar
tense_or_verb_form
other
```

统一归一化：

```text
errors_per_5000_words = error_count / max(word_count, 1) * 5000
```

主图 Panel c 展示 agent 与 peer review 的错误类别堆叠条形图。由于人工 peer review 可能包含审稿人速记、编号列表和 OCR 噪声，应同时导出 OCR/text-quality warning。

### Coverage

目的：衡量 agent 是否覆盖人工评审中的关键关注点。

推荐流程：

1. 从 peer review 中抽取 top keyphrases / aspects。
2. 从 agent evaluation 中抽取 top keyphrases / aspects。
3. 对关键词做小写化、词形还原、标点清理和短语去重。
4. 计算人工关注点被 agent 覆盖的比例。

默认主题类别：

```text
novelty
method
evidence
limitation
comparison
impact
```

Coverage 计算：

```text
coverage_score = covered_peer_aspects / total_peer_aspects
```

其中 `covered_peer_aspects` 可以由短语重叠、embedding similarity 或 LLM judge 判定。若使用 LLM judge，必须保存判定依据到 `outputs/fig4/cache/<paper_id>/coverage_judgement.json`。

## 绘图设计

绘图脚本默认从 `fig4_metrics_summary.csv` 读取数据，不重新计算 agent 输出或指标。

### Panel a: Consistency

展示 semantic consistency 分布。

推荐图形：

- x 轴：journal；
- y 轴：`consistency_cosine` 或映射后的 `[0, 1]` 分数；
- 图形：箱线图 + 每篇论文 jitter scatter；
- 标注：总体均值、bootstrap 95% CI、样本数。

### Panel b: Efficiency

展示 agent runtime 与人工基准的耗时差异。

推荐图形：

- lollipop plot 或 paired bar；
- y 轴使用 log-scale seconds；
- 标出 `human_baseline_hours=5` 的水平线；
- 辅助标注 median speedup 和敏感性区间。

### Panel c: Readability

展示 agent vs peer review 的错误类别分布。

推荐图形：

- 堆叠条形图；
- 两组：agent evaluation、peer review；
- 堆叠类别：spelling、grammar、tense_or_verb_form、other；
- 单位：errors per 5000 words。

### Panel d: Coverage

展示人工关注点被 agent 覆盖的结构。

推荐图形：

- heatmap；
- 行：journal 或 paper；
- 列：novelty、method、evidence、limitation、comparison、impact；
- 值：coverage ratio 或 binary covered；
- 对样本量大的数据，主图按 journal 聚合，补充表保存逐论文结果。

### Panel e: Overall Summary

展示四维综合表现。

推荐图形：

- radar plot、parallel coordinates 或 grouped dot plot；
- 指标：consistency、efficiency、readability、coverage；
- 每个指标归一化到 `[0, 1]`；
- 显示 bootstrap 95% CI；
- readability 归一化方向为错误越少分数越高。

归一化建议：

```text
consistency_norm = clip((consistency_cosine + 1) / 2, 0, 1)
efficiency_norm = clip(log(speedup) / log(speedup_cap), 0, 1)
readability_norm = 1 - normalized_agent_error_rate
coverage_norm = coverage_score
```

`speedup_cap` 和 readability normalization 参数写入 run config，避免图形无法复现。

## 推荐命令

未来入口建议：

```bash
python -m experiments.fig4.main_fig4 \
  --manifest experiments/fig4/fig4_manifest.csv \
  --output-dir outputs/fig4 \
  --panels all \
  --human-hours 5
```

只跑输入审计：

```bash
python -m experiments.fig4.main_fig4 \
  --manifest experiments/fig4/fig4_manifest.csv \
  --output-dir outputs/fig4 \
  --stage audit
```

只复用已有 agent 输出并重算指标：

```bash
python -m experiments.fig4.main_fig4 \
  --manifest experiments/fig4/fig4_manifest.csv \
  --output-dir outputs/fig4 \
  --stage metrics \
  --reuse-agent
```

只重绘 panel：

```bash
python -m experiments.fig4.main_fig4 \
  --output-dir outputs/fig4 \
  --stage draw \
  --panels a,c,e
```

## 质量控制

核心原则：

```text
No mock metric values.
No silent sample dropping.
No global cache overwrite.
Every plotted value must be traceable.
```

每次运行必须生成：

- `fig4_run_config.json`：参数、模型、依赖版本、时间戳；
- `fig4_input_audit.csv`：每个 manifest 样本的纳入/排除状态；
- `fig4_metrics_summary.csv`：每篇论文的最终指标；
- `fig4_diagnostics.json`：样本量、失败数、warning 统计。

主图纳入规则：

- `included_in_main=true`；
- article 和 peer review 文本均非空；
- agent evaluation 生成成功；
- 至少有 consistency、efficiency、coverage 三类核心指标；
- readability 如工具不可用，可以单独标为 missing，但 Panel c 必须改为 diagnostic。

样本排除必须写明原因，不能只在日志中打印。

## 测试计划

### Manifest 测试

- 重复 `paper_id` 应进入 audit fail。
- 不存在的 PDF 路径应进入 audit fail。
- 空 PDF 文本应进入 audit fail。
- 缺失 `peer_review_pdf_path` 的样本不进入主图。
- manifest 中存在额外列时不报错，保留到审计输出。

### Agent 测试

- 使用 mock LATS 输出验证 `outputs/fig4/cache/<paper_id>/` 隔离。
- 批量运行不应修改 `data/most_related_paper.json`。
- 单篇 agent 失败不应中断整批任务。
- `--reuse-agent` 应跳过已存在的 `agent_eval.json`。

### 指标测试

- 相同文本的 consistency 应接近 1。
- 空文本样本应被排除，而不是写入 0 分。
- readability 必须按 word count 归一化到每 5000 words。
- coverage 对大小写、标点和简单复数不敏感。
- LLM judge coverage 必须保存逐项判定证据。

### 绘图测试

- 每个 panel 可独立生成。
- 样本量不足时仍输出 diagnostic 图和 warning。
- `fig4_full.png` 只能从已生成 panel 和 `fig4_metrics_summary.csv` 产生。
- 最终图中的样本数应与 `fig4_input_audit.csv` 一致。

## 与现有项目的关系

Fig.4 使用 ASPR 的现有 agent 能力，但不改变核心包默认行为。

应复用：

- `aspr.open_scholar` 的关键词抽取、Semantic Scholar 检索、召回和重排；
- `aspr.lats` 的反思式创新性评价；
- 项目已有 `pypdf`、`pandas`、`sklearn`、`matplotlib` 工作方式。

应避免：

- 在 Fig.4 批量实验中依赖 `data/most_related_paper.json` 这样的全局单文件缓存；
- 把 PDF、大段解析文本或生成图片提交到 `experiments/fig4/`；
- 用随机数或人工填写的假指标先占位；
- 递归扫描整个 `outputs/` 作为输入。

## 参考资料

- Publons / Clarivate. Global State of Peer Review 2018: https://publons.com/static/Publons-Global-State-Of-Peer-Review-2018.pdf
- Nature News 对 peer-review 时间调查的报道: https://www.nature.com/articles/d41586-018-06602-y


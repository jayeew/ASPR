# Fig7: whole-graph future-direction word clouds

Uses every 2023–2025 claim node and all strictly time-eligible, non-self semantic
edges with cosine > 0.5 from `data/claim_graph`. Existing graph assets are read-only.
This pipeline produces source-grounded prospective hypotheses, not forecast accuracy.

```bash
/home/jayee/miniconda3/envs/openrlhf/bin/python -m figure_pipeline.fig7_reference prepare
/home/jayee/miniconda3/envs/openrlhf/bin/python -m figure_pipeline.fig7_reference predict
/home/jayee/miniconda3/envs/openrlhf/bin/python -m figure_pipeline.fig7_reference render
```

Results are in `outputs/fig7_reference/forecast/`:

- `data/`: complete group statistics, source records, node assignments, coverage,
  accepted and excluded candidates, retained directions, and phrase counts.
- `packets/`: eight size-balanced input packets. Array column definitions are
  embedded in each packet. Every group is assigned exactly once; neighbor evidence
  can occur in more than one packet. Missing-community topic groups remain distinct
  from graph communities.
- `predictions/`: raw structured forecast and consolidation responses.
- `calls.jsonl`, `logs/calls/`: persistent actual-attempt budget and transport records.
- `tables/`: retained directions, exact claim/source links, all normalized phrases.
- `final/`: editable SVG, outlined SVG, PDF, PNG, and 600 dpi PNG.
- `methods_and_caption.md`: definitions, actual results and interpretation limits.

The prediction model is `gpt-6.1-sol`, reasoning `medium`, through the existing
Codex session transport. Forecasts take at most eight calls, consolidation one;
only one additional request is available for failure recovery. Internal transport
retries and automatic formatting retries are disabled. Re-running `predict`
reuses completed predictions and keeps the existing call ledger. Run one prediction
process at a time. Do not delete the ledger to restart the budget.

All graph nodes enter statistical aggregation. The model sees all group profiles
and representative claim texts, not every claim text. Complete claim records and
source fragments remain in the companion data. Cross-group edges involving topic
buckets must not be described as confirmed cross-discipline relations. The fixed
whole-period communities are not historical community discovery snapshots.

Word frequency is the number of distinct retained directions carrying the
normalized phrase. Each category displays up to 20 phrases (seeded selection among frequency
ties). The common font scale is `8 + 10 * sqrt(n / max_displayed_n)` points.
Offline rendering never changes predictions or calls a model. Phrases are mostly horizontal, with four vertical phrases per full cloud. Long
horizontal phrases wrap at word boundaries without changing their size or wording.

Validation:

```bash
python3 -m pytest -q tests/innovation_v2/test_fig7_reference.py
```

## Current unified layout

The default `render` stage now uses `unified.py`: one glyph-packed cloud, category
hues, and a common size/shade scale. It selects 24 phrases per category using
`frequency * log2(1 + distinct cited papers)` as descriptive prominence. This is
not validated scientific importance. `short_labels.json` maps display abbreviations
to the unchanged full prediction phrases; `tables/unified_wordcloud.csv` is the
current display/source mapping. The earlier frequency-only table is retained as
a count record and is not the current display selection. No new model calls occur.

## 2026 论文驱动的前瞻词云（先 pilot，再决定全量）

新增入口 `python -m figure_pipeline.fig7_reference.frontier`。输出隔离在
`outputs/fig7_reference/frontier_2026/`，不覆盖历史版 Fig7 或原实验目录。

流程：原始 1,000 篇名单 → 复用正式集/候选集共享 claims → 缺失论文补抽取 →
补算真实历史邻居和联合邻域事实 → 2026 新发现与历史图共同驱动未来方向 → 归并 → 一张词云。
历史图只读，不创建 2026 论文之间的假想语义边，不补跑 GEAR 或逐 claim 模型创新评价。
内部不支持的 claims 显式排除；部分支持项只使用核验收窄后的 normalized claim。

```bash
# 本轮授权范围：小批量测速，默认只跑 pilot。
python -m figure_pipeline.fig7_reference.frontier inventory
python -m figure_pipeline.fig7_reference.frontier pilot
python -m figure_pipeline.fig7_reference.frontier estimate
```

pilot 固定选 12 篇尚无 Graph 的论文（覆盖不同 claim 数与领域）、1 篇缺失 claims
的论文，并复用 112 篇已有 Graph 的论文，形成接近全量单包大小的 125 篇预测材料。
预测与归并最多 2 次实际请求，gpt-6.1-sol / medium。缺失 claims 的抽取复用现有
Luna/low 流程，一篇包含多次分块、归并和核验调用，日志与预测预算分开记录。

所有目标 claims（除内部不支持者）及全部保留邻接关系进入模型材料。
每条目标 claim 选择相似度最高的一个历史邻居提供原文；所有历史邻居原文、全文支持片段、
完整图事实保存在材料与 EvidenceStore 中。列式编码和短 ID 仅压缩表示，不截断 claim 原文。
预测每条必须同时引用 2026 全文 claim 和实际相连的历史 claim，来自至少两篇论文；
严格检查来源、逐字引文和边端点，但此检查不等于科学预测准确性验证。

正式全量处理须在用户看到 pilot 耗时后另行决定。批准后的命令为：

```bash
python -m figure_pipeline.fig7_reference.frontier all --full
```

亦可分阶段使用 `extract|graph|packets|predict|render --full`。正式预测 8 包 + 1 次归并，
合计至多 10 次实际请求（含失败请求）；没有自动重试，剩余预算用于明确的恢复。
pilot 的预测结果不混入正式全量方向，已生成的 claims 与图事实可复用。

全量包按材料长度均衡；每篇恰好进入一包，跨包方向不能声称被系统穷举。
材料超过 900,000 字符时调用前停止，不静默丢弃论文。预测标签包含完整科学短语及
强调下一步研究任务的短展示标签。单幅词云颜色分 Method/Problem/Combination，
大小和深浅使用 `方向频次 × log₂(1 + 去重来源论文数)`；此权重不是成功概率或科学重要性。

主要输出：`inventory.json`、`pilot_selection.json`、`timings/`、`timing_estimate.json`、
`pilot/coverage.json`、`pilot/tables/`、`pilot/final/Fig7.svg|pdf|png|_600dpi.png`。
全量文件使用独立 `full/` 子目录。完整来源包和图材料保存在 `materials/`、`papers/`。

## 当前双 panel 版本（全量统计修复）

```bash
# 准备阶段的本地语法分析依赖：spaCy 3.8.7 + en_core_web_sm 3.8.0。
/home/jayee/.cache/fig7-topic-runtime/bin/python -m figure_pipeline.fig7_reference.two_panel all
# 重绘读取已保存的统计，无需语法模型或预测调用：
/home/jayee/miniconda3/envs/openrlhf/bin/python -m figure_pipeline.fig7_reference.two_panel render
```

输出为 `outputs/fig7_reference/two_panel/`。当前 panel a 由 `historical_statistics.py`
处理 2023–2025 全部 70,034 条 claims。统一提取 3–6 词完整名词短语，按公开的语言规则
过滤泛化描述，再跨全语料按不同论文数计数。频次至少 5 的候选全局降序排列，同频按
规范短语字母排序；对频次接近（长词组达到短词组的 80%）的嵌套短语，展示更完整表达，
被抑制短语和原计数仍保留在全表。显示筛选后的前 68 个；不读旧的手选 `historical_topics.json`，不设类别配额。
这提供确定、可复现的词面统计，不声称穷尽所有科学主题或完成全部语义同义词归并。

完整候选排名、字号及来源保存在 `tables/historical_wordcloud.csv` 和
`tables/historical_phrase_sources.csv`；未展示候选也保留。规则、数据摘要和源文件哈希
保存于 `data/historical_summary.json`，语法模型版本保存在 `data/historical_parser.json`。
三色按固定表达线索分类，先判定耦合／组合，再判定方法，其余归入问题／对象；分类不影响排序。

b 复用 `frontier_2026/full` 的 96 条方向与 68 个展示主题，读取 `data/topic_labels.json`
中的科学主题标签；未来研究任务保留在完整方向表。本入口不调用预测模型。

卡片采用 fig1_reference 的浅蓝标题栏、细边框和 Times 字体；总标题下无副标题、图底无小字。
a 的论文频次与 b 的证据加权方向权重不是同一单位，字号不能作为跨期增长比较。
b 基于历史 Graph 邻域和 claims，未将 a 的统计词云输入预测模型。
独立复算入口：`python -m figure_pipeline.fig7_reference.historical_audit`。它使用独立的正则匹配
重新核对 68 个展示词的全语料论文数，并验证所有候选的来源与排名。

此前的手选主题图已归档到 `outputs/fig7_reference/archive/Fig7_selected_topics_superseded.zip`，
不能作为全量主题排名使用。`historical_cloud.py` 和 `historical_topics.json` 仅保留旧版记录。

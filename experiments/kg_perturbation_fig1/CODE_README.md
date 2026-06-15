# `fig1_knowledge_perturbation_v3.py` 代码架构说明

这份文档只讲代码。图怎么读、每个面板表达什么，请看同目录下的 `README.md`。

当前脚本是一个单文件 pipeline：从 OpenAlex 获取论文数据，构建论文级混合知识图，做社区发现，生成展示用主题图，计算结构扰动指标，并绘制 Fig. 1 风格的 citation knowledge-graph perturbation 主图。

源码位置：

```text
experiments/kg_perturbation_fig1/fig1_knowledge_perturbation_v3.py
```

## 1. 运行入口

单领域运行：

```bash
cp .env.example .env  # fill OPENALEX_API_KEY or OPENALEX_API_KEYS
python experiments/kg_perturbation_fig1/fig1_knowledge_perturbation_v3.py \
  --config experiments/kg_perturbation_fig1/configs/crispr.yaml
```

多领域运行：

```bash
python experiments/kg_perturbation_fig1/fig1_knowledge_perturbation_v3.py \
  --config experiments/kg_perturbation_fig1/configs/crispr.yaml \
           experiments/kg_perturbation_fig1/configs/graphene.yaml \
           experiments/kg_perturbation_fig1/configs/ipsc.yaml \
           experiments/kg_perturbation_fig1/configs/transformer.yaml
```

主入口是 `main()`：

1. `parse_args()` 解析配置路径、输出目录、API key 和缓存开关。
2. `load_config()` 读取每个 YAML，并递归合并 `DEFAULT_CONFIG`。
3. 初始化 `OpenAlexClient`。
4. 对每个配置调用 `run_domain()`。
5. 如果传入多个配置，额外调用 `draw_multi_domain_figure()`。

## 2. 输入和输出

### 输入

主要输入来自 YAML 配置：

| 配置块 | 作用 |
| --- | --- |
| `domain_name`, `slug` | 领域名称和输出目录名 |
| `search_query` | OpenAlex 检索式 |
| `start_year`, `end_year`, `window_size` | 时间范围和窗口宽度 |
| `anchors` | landmark papers 的 DOI/OpenAlex ID/年份/标签 |
| `api.*` | 请求节流、重试、超时、分页大小 |
| `graph.*` | 建图、剪枝、社区发现参数 |
| `metrics.*` | betweenness 和语义采样参数、曲线模式 |
| `plot.*` | 主图标题、布局、主题展示、参数曲线和注释 |

论文元数据来自 OpenAlex API。若缓存存在且启用 `--use-cache`，脚本会直接读取：

```text
outputs/kg_perturbation_fig1/<slug>/works_raw.jsonl
```

注意：`works_raw.jsonl` 这个名字保留了历史叫法，文件内容已经经过 `normalize_work()` 标准化，不是 OpenAlex 原始 JSON。

### 输出

每个领域输出到：

```text
outputs/kg_perturbation_fig1/<slug>/
```

主要文件：

| 文件 | 内容 |
| --- | --- |
| `works_raw.jsonl` | 标准化后的全部候选论文缓存 |
| `works_selected.csv` | 进入论文图的论文，以及完整社区和展示社区归属 |
| `paper_edges.csv` | 论文级混合图边，含 direct/bibliographic/cocitation 计数 |
| `topic_nodes.csv` | 展示主题图节点，含标签、规模、anchor、布局坐标 |
| `topic_edges.csv` | 展示主题图跨主题边 |
| `perturbation_metrics.csv` | 底层扰动指标、参数代理列和 `*_index` |
| `dominant_parameter_trajectories.csv` | 实际绘制到 panel b 的参数轨迹 |
| `fig1_<slug>_real.png/.svg/.pdf` | 单领域 Fig. 1 输出 |

## 3. 核心数据结构

### 标准化 work

`normalize_work()` 把 OpenAlex work 压成统一字典，核心字段包括：

- `id`, `short_id`, `doi`
- `title`, `year`, `date`, `type`, `language`
- `cited_by_count`, `fwci`, `citation_normalized_percentile`
- `refs`
- `topics`, `primary_topic`
- `text`
- `anchor_label`, `anchor_year`

`text` 由标题、topic 和可选摘要拼接，用于社区标签、TF-IDF 和语义离散度计算。

### 论文级图 `G`

`G` 是 `networkx.Graph`。节点是一篇被选中的论文，边是多证据混合知识关系：

- direct citation
- bibliographic coupling
- co-citation

边属性里会累加：

- `weight`
- `direct`
- `bibliographic`
- `cocitation`

### 两套社区映射

v3 最重要的设计是把分析层和展示层分开：

| 对象 | 来源 | 用途 |
| --- | --- | --- |
| `comm_map` | `detect_communities(G, compact=False)` | 完整社区，用于定量指标 |
| `labels` | `make_community_labels(G, comm_map)` | 完整社区标签 |
| `display_comm_map` | `build_display_comm_map(G, comm_map, cfg)` | 少量展示社区，用于主图 |
| `display_labels` | `make_community_labels(G, display_comm_map)` | 展示社区标签 |

也就是说，panel a 画的是展示层，`perturbation_metrics.csv` 主要来自完整分析层。

### 主题图 `TG`

`make_topic_graph()` 把论文级图压缩成 community-level topic graph：

- 节点是社区。
- 节点属性包含 `label`, `display_label`, `n_papers`, `cited_by_count`, `first_year`, `anchor_labels`, `anchor_year`, `member_ids`, `semantic_text`。
- 边是跨社区论文边的聚合，`weight` 为论文级边权重求和，`n_edges` 为跨社区论文边数量。

脚本会生成两张主题图：

- `TG_full`：完整分析主题图。
- `TG_display`：主图展示主题图。

### `DomainResult`

`DomainResult` 是贯穿导出和绘图的结果容器，包含配置、论文、图、社区、布局、指标和时间窗口。`export_tables()` 和 `draw_single_domain_figure()` 都从这个对象取数据。

## 4. `run_domain()` 主流程

`run_domain()` 是单领域总控函数，执行顺序如下：

1. `make_rolling_windows()` 生成 rolling windows。
2. `make_cumulative_windows()` 生成累计窗口。
3. `fetch_domain_works()` 获取或读取缓存论文。
4. `select_balanced_papers()` 从候选论文中选择建图论文。
5. `build_hybrid_graph()` 构建论文级混合图 `G`。
6. `detect_communities()` 得到完整社区 `comm_map`。
7. `make_community_labels()` 生成完整社区标签。
8. `build_display_comm_map()` 筛选展示社区。
9. `make_community_labels()` 生成展示社区标签。
10. `make_topic_graph()` 生成 `TG_full` 和 `TG_display`。
11. `layout_topic_graph()` 计算展示主题布局。
12. `compute_perturbation_metrics()` 计算底层扰动指标。
13. 打包 `DomainResult`。
14. `export_tables()` 导出 CSV。
15. `draw_single_domain_figure()` 绘制 PNG/SVG/PDF。

## 5. 数据获取逻辑

`OpenAlexClient` 是对 OpenAlex REST API 的轻量封装：

- `get_json()` 负责请求、重试、限速、错误处理。
- `list_works()` 用 cursor 分页拉取 works。
- `get_work_by_doi()` 通过 DOI 获取单篇 anchor paper。
- `get_work_by_openalex_id()` 通过 OpenAlex ID 获取单篇 anchor paper。

`fetch_domain_works()` 的策略：

1. 如果缓存可用，读取 `works_raw.jsonl` 并重新执行 `mark_anchors()`。
2. 否则按 rolling windows 分段检索，避免近期论文淹没早期论文。
3. 强制补抓 `anchors` 中列出的 landmark papers。
4. 若 `fetch_anchor_citers: true`，再抓引用 anchor 的高被引论文，增强后续扰动信号。
5. 对所有 work 执行 `normalize_work()`。
6. 写入缓存。

`mark_anchors()` 通过 DOI 或 OpenAlex ID 匹配 landmark paper，并写入 `anchor_label` 和 `anchor_year`。

## 6. 论文选择逻辑

`select_balanced_papers()` 用于控制图规模。若候选论文数没有超过 `graph.max_papers_for_graph`，直接全选；否则按时间窗口分配 quota。

论文排序分数：

```python
score = log1p(cited_by_count) + (100.0 if anchor else 0.0)
```

实际逻辑等价于：

- 每个 rolling window 内优先选高被引论文。
- anchor papers 永远保留。
- 若总数还没达到上限，再用全局高被引论文补齐。

这样可以避免 2020 年之后的大量论文把早期阶段挤出图谱。

## 7. 论文级混合图计算

`build_hybrid_graph()` 创建论文级图。每篇 selected work 是一个节点。

### Direct citation

如果 selected paper `u` 引用 selected paper `v`，添加 direct edge：

```python
weight += graph.direct_weight
direct += 1
```

默认 direct citation 权重较高，因为它表示明确的知识传递。

### Bibliographic coupling

若两篇 selected papers 共享同一参考文献，则它们形成 bibliographic coupling。共享参考文献数为 `c`，且满足 `c >= graph.min_shared_references` 时添加：

```python
weight += graph.bibliographic_weight * log1p(c)
bibliographic += c
```

`graph.max_reference_fanout` 用来过滤被太多论文共同引用的超高 fanout 参考文献，避免制造过密边。

### Co-citation

若两篇 selected papers 被同一篇候选论文共同引用，则它们形成 co-citation。共同被引次数为 `c`，且满足 `c >= graph.min_cocitations` 时添加：

```python
weight += graph.cocitation_weight * log1p(c)
cocitation += c
```

这里 all fetched works 都可以充当 citing papers，不要求它们本身进入 selected graph。

### 边累加和剪枝

`add_weighted_edge()` 会把同一对论文的不同证据累加到同一条无向边上。

`prune_graph_edges()` 在边数超过 `graph.max_edges` 时，只保留权重最高的边，节点全部保留。

## 8. 社区、标签和展示层

### 社区发现

`detect_communities()` 优先使用：

```python
nx.community.louvain_communities(..., weight="weight", resolution=community_resolution)
```

失败时回退到 `greedy_modularity_communities()`。`compact=True` 是旧逻辑，会把长尾社区合并成 other；v3 主流程使用 `compact=False`。

### 社区标签

`make_community_labels()` 对每个社区收集：

- 论文标题
- `text`
- OpenAlex topics

然后通过 `top_terms_from_texts()` 做 TF-IDF，优先使用代表性词组；如果 TF-IDF 不稳定，再回退到高频 topic；仍没有则使用 `Topic N`。

### 展示社区筛选

`build_display_comm_map()` 从完整社区中筛出少量用于 panel a 的社区。社区得分为：

```python
score =
    3.0 * log1p(size)
  + 0.45 * log1p(citation_score)
  + 0.35 * log1p(degree_score)
```

如果社区包含 anchor，额外加 `1000.0` bonus，确保 landmark 所在社区尽量出现在主图中。

展示层参数主要来自：

- `plot.display_max_topics`
- `plot.display_min_topic_size`

v3 不再把所有长尾社区合并成巨大 other 节点，而是直接从可视化层隐藏长尾。完整论文仍保留在定量分析层。

## 9. 主题布局和主图展示

### 布局

`layout_topic_graph()` 先计算自动图布局：

- 优先 `kamada_kawai_layout`
- 失败时使用 `spring_layout`

如果 YAML 中定义了 `plot.topic_layout_templates`，则进入语义模板布局：

1. 每个模板给出 `name`, `x`, `y`, `keywords`。
2. `score_keyword_match()` 计算主题语义文本和模板关键词的匹配度。
3. 匹配最高的主题被放到模板坐标。
4. 命中模板的主题会把 `display_label` 改成模板 `name`。
5. 剩余主题使用自动布局补位。
6. `repel_positions()` 做轻微排斥，减少圆圈重叠。

### 单个累计快照

`draw_snapshot()` 负责 panel a 中每个小图：

1. `node_set_until_year()` 取当前累计论文。
2. `make_topic_graph()` 用展示社区生成当前累计主题图。
3. `filter_topic_graph_for_display()` 隐藏早期样本过少的非 anchor 主题。
4. `log_scaled_radius()` 根据 `n_papers` 计算主题圆圈半径。
5. `select_backbone_edges()` 筛选可读的主题边。
6. `deterministic_disc_points()` 生成圆圈内部 paper beads。
7. 对 anchor 社区画红色虚线圈、红色星标和注释。
8. 写入主题标签、caption、论文数量和 displayed topics 数。

`select_backbone_edges()` 的边筛选顺序：

1. 每个连通分量的 maximum spanning tree。
2. anchor incident edges。
3. 少量剩余强边。

然后与上一累计窗口比较，标记 `is_new`：

- `is_new=True` 画深色边。
- `is_new=False` 画浅色边。

## 10. 时间窗口

脚本里有两类窗口：

```python
rolling_windows = make_rolling_windows(start_year, end_year, window_size)
cumulative_windows = make_cumulative_windows(start_year, rolling_windows)
```

以 CRISPR 的 `2000-2024`、`window_size=5` 为例：

| 类型 | 窗口 |
| --- | --- |
| rolling | `2000-2004`, `2005-2009`, `2010-2014`, `2015-2019`, `2020-2024` |
| cumulative | `2000-2004`, `2000-2009`, `2000-2014`, `2000-2019`, `2000-2024` |

Panel a 使用 cumulative windows。Panel b 使用 rolling windows。

## 11. 底层扰动指标

`compute_perturbation_metrics()` 用完整论文图 `G` 和完整社区 `comm_map` 计算指标。每个 rolling window 同时构造：

- `roll_nodes`：当前 5 年窗口内的论文。
- `cum_nodes`：从起始年份到当前窗口结束年份的累计论文。
- `Gcum`：累计论文子图。
- `Groom`：当前 rolling window 子图，目前保留但没有在后续计算中直接使用。

### Expansion

衡量当前阶段新增论文、边和 topic 的规模。

```python
Expansion_raw =
    log1p(new_nodes)
  + log1p(new_edges)
  + log1p(new_topics)
```

### Bridging

衡量跨社区桥接程度。

```python
Bridging_raw =
    0.50 * intercommunity_edge_ratio
  + 0.35 * participation_mean
  + 0.15 * bridge_betweenness_top
```

其中：

- `intercommunity_edge_ratio`：累计图中跨社区边比例。
- `participation_mean`：当前窗口论文的 participation coefficient 均值。
- `bridge_betweenness_top`：当前窗口论文中 betweenness centrality 较高部分的均值。

### Reconfiguration

衡量连续累计图之间的社区结构重排。

```python
Reconfiguration_raw =
    partition_change
  + 0.5 * edge_turnover
  + modularity_shift
```

其中：

- `partition_change = 1 - adjusted_rand_score(previous_partition, current_partition)`。
- `edge_turnover` 衡量边集合替换比例。
- `modularity_shift` 是模块度变化幅度。

### Compression

衡量知识结构是否变得更短路径、更集中、更语义收敛。

```python
Compression_raw =
    path_gain
  + hub_gain
  + semantic_gain
```

其中：

- `path_gain`：topic graph 平均最短路径缩短量。
- `hub_gain`：hub concentration 增量。
- `semantic_gain`：TF-IDF 语义离散度下降量。

### `*_index`

四个 raw 指标会根据 `metrics.curve_mode` 转成 `*_index`：

| `curve_mode` | 逻辑 |
| --- | --- |
| `raw` | 直接使用 raw series |
| `cumulative_positive` | `cumsum(max(raw, 0))` |
| 其他值 | `maximum.accumulate(raw)` |

随后统一做：

```python
100.0 * robust_minmax(series)
```

这些列写入 `perturbation_metrics.csv`，但当前固定 Fig. 1 的 panel b 不直接画 `Expansion_index` 等四条旧式扰动曲线。

## 12. Panel b 的 dominant parameter 逻辑

当前图的 panel b 由 `draw_metric_panel()` 绘制。它调用：

```python
dominant_parameter_trajectories(metrics, cfg)
```

参数规格来自：

```yaml
plot:
  dominant_parameters:
    - key: "B"
    - key: "RTD"
    - key: "Uzzi"
    - key: "DeltaQ"
```

内置 `PARAMETER_SPECS` 定义了常见参数和默认来源列：

| 参数 | 默认来源列 | 含义 |
| --- | --- | --- |
| `B` | `B_proxy_raw` | bridge centrality proxy |
| `RTD` | `RTD_proxy_raw` | reference target diversity proxy |
| `Uzzi` | `Uzzi_proxy_raw` | novelty proxy |
| `DeltaQ` | `DeltaQ_directionality_raw` | modularity directionality |
| `RS` | `RS_proxy_raw` | reference span proxy |
| `BurtIP` | `BurtIP_proxy_raw` | structural holes proxy |
| `PDE` | `PDE_proxy_raw` | diffusion potential proxy |

`compute_perturbation_metrics()` 会生成这些 proxy：

```python
B_proxy_raw = Bridging_raw
RTD_proxy_raw = participation_mean
RS_proxy_raw = 0.55 * minmax(semantic_dispersion) + 0.45 * minmax(new_topics)
Uzzi_proxy_raw = 0.45 * minmax(participation_mean)
               + 0.35 * minmax(partition_change)
               + 0.20 * minmax(edge_turnover)
DeltaQ_directionality_raw = modularity_delta
BurtIP_proxy_raw = 0.60 * minmax(participation_mean)
                 + 0.40 * minmax(intercommunity_edge_ratio)
PDE_proxy_raw = 0.55 * minmax(Expansion_raw)
              + 0.45 * minmax(new_edges)
```

如果某个 `dominant_parameters` 条目显式提供 `values`，则优先使用手动值。当前几个领域的 YAML 都为 panel b 提供了手动轨迹，因此最终图中的曲线主要来自配置文件，而不是直接从 proxy 列自动标准化得到。

`standardize_parameter_values()` 的规则：

- 如果是手动 `values`，且没有设置 `standardize_values: true`，直接使用手动值。
- 如果没有手动值，则从 `source` 列读取，并默认做 z-score 标准化。
- 若 `center_zero: true`，使用零中心方向性缩放，`DeltaQ` 默认属于这种类型。
- 可选 `invert`, `clip`, `standardize_mode` 等字段进一步控制曲线。

`dominant_parameter_table()` 会把最终用于绘图的轨迹导出到：

```text
dominant_parameter_trajectories.csv
```

## 13. Landmark 窗口和注释

`landmark_window_index()` 决定 panel b 中哪个 rolling window 被标记为 landmark 窗口：

1. 若配置了 `plot.landmark_focus_year`，优先使用它。
2. 否则取 `anchors` 中最早的年份。
3. 找到包含该年份的 rolling window。

`draw_metric_panel()` 会：

- 给 landmark window 加浅红色背景。
- 在每条参数轨迹的 landmark 点上画星标。
- 根据 `plot.parameter_callouts` 添加箭头注释。
- 根据 `plot.parameter_interpretation_boxes` 绘制底部解释框。

`draw_top_time_axis()` 会根据 anchors 画顶部 innovation event 区域。不过当前函数中的 `"2012-2013\nlandmark papers"` 文案是硬编码，若换成非 CRISPR 领域，需要同步修改代码或改成配置项。

## 14. 绘图导出逻辑

`draw_single_domain_figure()` 创建三行布局：

| 行 | 内容 |
| --- | --- |
| 第 1 行 | 顶部时间轴 |
| 第 2 行 | 五个累计知识图快照 |
| 第 3 行 | dominant parameter trajectories |

它会保存：

```text
fig1_<slug>_real.png
fig1_<slug>_real.svg
fig1_<slug>_real.pdf
```

`draw_multi_domain_figure()` 用于多领域版本。每个领域占一行，前五列绘制累计快照，最后一列绘制该领域对应的参数面板，适合横向比较不同 landmark innovation。

## 15. 常见修改点

### 换一个领域

优先修改 YAML：

- `domain_name`
- `slug`
- `search_query`
- `start_year`, `end_year`
- `anchors`
- `topic_layout_templates`
- `panel_captions`
- `dominant_parameters`
- `parameter_callouts`
- `parameter_interpretation_boxes`

一般不需要改 Python 主逻辑。

### 想让 panel b 完全数据驱动

删除 YAML 中 `plot.dominant_parameters[*].values`，保留 `key` 或指定 `source`。脚本会从 `perturbation_metrics.csv` 中的 proxy 列读取并标准化。

### 重新抓 OpenAlex 数据

删除对应领域的缓存目录，或运行时关闭缓存。缓存位置：

```text
outputs/kg_perturbation_fig1/<slug>/works_raw.jsonl
```

### 主题名字不理想

优先调：

- `plot.topic_layout_templates[].keywords`
- `plot.topic_layout_templates[].name`
- `STOPWORDS_EXTRA`

如果模板命中，`display_label` 会使用模板名；否则使用自动 TF-IDF 标签。

### 图太乱或太稀

优先调：

- `plot.display_max_topics`
- `plot.display_min_topic_size`
- `plot.display_max_backbone_edges`
- `plot.display_extra_edges`
- `plot.min_papers_per_display_topic`
- `graph.community_resolution`

## 16. 当前版本的注意事项

- `plot.metric_scale` 出现在 YAML 中，但当前 `draw_metric_panel()` 没有读取这个字段。
- `Groom` 在 `compute_perturbation_metrics()` 中创建后没有直接参与后续计算。
- 顶部时间轴的 `"2012-2013 landmark papers"` 是硬编码文本。
- Panel a 是展示层社区，Panel b 的 proxy 指标来自完整分析层；两者不是同一套社区对象。
- 当前图适合做结构可视化和机制叙事；若要做强因果结论，需要另外加入 controls、null models 或 bootstrap。

## 17. 最短理解路径

如果只想快速读懂代码，建议按这个顺序看：

1. `run_domain()`
2. `fetch_domain_works()`
3. `build_hybrid_graph()`
4. `build_display_comm_map()`
5. `layout_topic_graph()`
6. `draw_snapshot()`
7. `compute_perturbation_metrics()`
8. `dominant_parameter_trajectories()`
9. `draw_metric_panel()`
10. `export_tables()`

一句话总结：v3 用完整论文图做定量分析，用精简展示主题图做 publication-style panel a，再用配置驱动或 proxy 驱动的 dominant parameters 解释 panel a 中的结构转变。

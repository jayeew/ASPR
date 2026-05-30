# `fig1_knowledge_perturbation_v3.py` 从 0 到 1 代码详解

这份说明已经按最新版本 `fig1_knowledge_perturbation_v3.py` 重写，重点补充了两件事：

1. v3 相比旧版在“图怎么画”上发生了什么变化。
2. 图里的数据、标题、标签、红星、时间轴、曲线，到底各自来自哪里。

对应源码：

- `experiments/kg_perturbation_fig1/fig1_knowledge_perturbation_v3.py`

---

## 1. 先用一句话讲清楚这段代码

这段代码会：

1. 从 OpenAlex 下载某一研究领域的论文元数据。
2. 构建论文级混合知识图。
3. 做社区发现，得到主题模块。
4. 计算 4 类扰动指标。
5. 画出一张更像论文主图的 Fig.1 风格图：
   - 上半部分是 5 个累计知识图快照
   - 下半部分是 4 条扰动曲线

---

## 2. v3 最关键的变化

如果你之前看过旧版，最重要的区别是这句：

**v3 把“用于计算指标的社区划分”和“用于主图展示的社区划分”分开了。**

具体说：

- `comm_map` + `labels`
  - 用于定量分析
  - 用于 `compute_perturbation_metrics()`
  - 尽量保留完整结构

- `display_comm_map` + `display_labels`
  - 用于主图快照展示
  - 只保留少量更可解释的主题模块
  - 隐藏长尾社区，避免出现一个巨大灰色 “other” 节点

这意味着：

**图上看到的主题结构，是“精简后的展示层”；下面曲线的数值，主要仍来自“完整分析层”。**

这是理解 v3 的第一原则。

---

## 3. 主流程怎么跑

总入口还是 `main()`，单领域总控还是 `run_domain()`。

主流程如下：

1. `load_config()` 读取 YAML 并和 `DEFAULT_CONFIG` 合并。
2. `fetch_domain_works()` 从 OpenAlex 抓论文并缓存到 `works_raw.jsonl`。
3. `select_balanced_papers()` 选一批代表性论文用于建图。
4. `build_hybrid_graph()` 构建论文级混合图 `G`。
5. `detect_communities(..., compact=False)` 得到完整社区 `comm_map`。
6. `make_community_labels()` 给完整社区自动起标签。
7. `build_display_comm_map()` 从完整社区中选出少量用于展示的社区。
8. `make_community_labels()` 再给展示社区起标签。
9. `make_topic_graph()` 生成：
   - `TG_full`：完整主题图
   - `TG_display`：展示主题图
10. `layout_topic_graph(TG_display, cfg)` 只对展示图做布局。
11. `compute_perturbation_metrics()` 用完整图算 4 个指标。
12. `export_tables()` 导出 CSV。
13. `draw_single_domain_figure()` 画单领域图。

---

## 4. 这份代码的输入是什么

输入分成两类：

### 4.1 配置输入

主要来自 YAML 和默认配置 `DEFAULT_CONFIG`。

最关键字段：

- `domain_name`
- `slug`
- `search_query`
- `start_year`
- `end_year`
- `window_size`
- `anchors`
- `graph.*`
- `metrics.*`
- `plot.*`

### 4.2 数据输入

主要来自 OpenAlex API：

- 论文基本信息
- DOI
- 年份
- 被引次数
- 参考文献列表
- topics / keywords / mesh
- 可选摘要

如果已经有缓存，就直接读：

- `results/<slug>/works_raw.jsonl`

注意：

虽然名字叫 `works_raw.jsonl`，但内容其实已经是 `normalize_work()` 处理后的标准化结果，不是 OpenAlex 原始 JSON。

---

## 5. 这份代码的输出是什么

每个领域会输出：

- `works_raw.jsonl`
- `works_selected.csv`
- `paper_edges.csv`
- `topic_nodes.csv`
- `topic_edges.csv`
- `perturbation_metrics.csv`
- `fig1_<slug>_real.png/.svg/.pdf`

v3 有个重要变化：

- `topic_nodes.csv` 和 `topic_edges.csv` 导出的不是完整主题图 `TG_full`
- 而是 **展示主题图 `TG_display`**

所以：

- `topic_nodes.csv` 更接近“你在图里看到的主题节点”
- `perturbation_metrics.csv` 更接近“你在下方曲线里看到的数值来源”

---

## 6. 时间窗口怎么定义

### 6.1 rolling windows

`make_rolling_windows()` 生成固定宽度窗口。

例如：

- 2000-2004
- 2005-2009
- 2010-2014
- 2015-2019
- 2020-2024

它主要用于：

- 分段抓数据
- 判断某个时期“新进入”的论文
- 画下方指标曲线时作为横轴区间

### 6.2 cumulative windows

`make_cumulative_windows()` 生成累计窗口：

- 2000-2004
- 2000-2009
- 2000-2014
- 2000-2019
- 2000-2024

它主要用于：

- 画上半部分 5 个累计知识图快照

---

## 7. 论文级混合图 `G` 是什么

图节点是一篇论文。

图边来自 3 类关系：

1. `direct citation`
2. `bibliographic coupling`
3. `co-citation`

所以这不是单纯 citation graph，而是一个混合知识图。

每类边的直觉：

- 直接引用：明确知识传递
- 文献耦合：共享知识基础
- 共引：被后人共同归类

---

## 8. v3 的社区层为什么要拆成两套

这是本版本最重要的设计。

### 8.1 完整社区 `comm_map`

来源：

- `detect_communities(G, cfg, compact=False)`

特点：

- 保留完整分析结构
- 用于指标计算
- 不会为了图好看而强行隐藏长尾

### 8.2 展示社区 `display_comm_map`

来源：

- `build_display_comm_map(G, comm_map, cfg)`

特点：

- 只保留少量更强、更大、或包含 anchor 的社区
- 小而不重要的社区直接不显示
- 不再像旧版那样把长尾并进一个巨大的 “other” 节点

### 8.3 为什么这么做

因为论文主图要兼顾两件事：

- 结构真实
- 视觉清楚

如果把所有社区都显示出来：

- 图会很乱
- 容易出现一个非常大的灰色杂项节点
- landmark paper 的结构作用不明显

所以 v3 把“定量层”和“展示层”分开，是一个很合理的论文制图折中。

---

## 9. 展示社区是怎么选出来的

函数：

- `build_display_comm_map()`

它会给每个完整社区打一个分，分数综合考虑：

- 社区大小
- 社区内论文被引情况
- 社区在图中的加权连接强度
- 是否包含 anchor

如果包含 anchor，会额外加一个非常大的 bonus。

然后：

- 只保留前 `display_max_topics`
- 过滤掉太小的社区 `display_min_topic_size`
- 确保 anchor 所在社区尽量保留

也就是说，图里展示出来的主题不是随机的，而是“结构强 + 语义强 + anchor 重要”的主题。

---

## 10. 主题标签是怎么来的

函数：

- `make_community_labels()`

它会从一个社区内部收集：

- 论文标题
- `text`
- topics

然后做 TF-IDF，选出最能代表这个社区的词。

优先顺序：

1. TF-IDF 代表词
2. 如果 TF-IDF 不稳定，再退回高频 OpenAlex topic
3. 再不行就用 `Topic N`

所以图里的主题标签，例如某个 `display_label`，通常不是手工写的，而是自动生成的。

---

## 11. v3 的布局比旧版更“论文插图化”

函数：

- `layout_topic_graph()`

它的变化很关键：

### 11.1 如果没有模板

就走自动布局：

- `kamada_kawai_layout`
- 失败时 `spring_layout`

### 11.2 如果配置里有 `topic_layout_templates`

就会走“语义模板布局”：

1. 从配置中读出若干模板槽位：
   - `name`
   - `x`
   - `y`
   - `keywords`
2. 对每个展示社区，计算它和模板关键词的匹配程度。
3. 把最匹配的社区放到对应模板位置。
4. 剩余社区再用自动布局补到周围。

这意味着：

**图上某个主题出现在左上、右下，不一定完全是图算法自己决定的，也可能受配置模板引导。**

如果某个节点被模板命中：

- 它的 `display_label` 还会被改成模板名

所以图上的主题标题来源有两层：

1. 默认来自自动社区标签
2. 如果用了 `topic_layout_templates`，则可能改成模板中的 `name`

---

## 12. 图里的每一部分到底从哪里来

这是本说明最重要的一节。

下面按“你在图上看到什么”逐一解释来源。

### 12.1 整张图的大标题

来源：

- `cfg["plot"]["title"]`

默认值在 `DEFAULT_CONFIG["plot"]["title"]`：

- `Landmark papers induce measurable perturbations in citation knowledge graphs`

如果 YAML 里覆盖了 `plot.title`，就用 YAML。

### 12.2 副标题

来源：

- `cfg["plot"]["subtitle"]`

默认值：

- `Expansion, bridging, reconfiguration and compression across cumulative citation-knowledge snapshots`

### 12.3 左上角的 `a`

来源：

- `draw_single_domain_figure()` 里直接写死：
  - `fig.text(0.012, 0.765, "a", ...)`

这不是数据算出来的，是固定面板编号。

### 12.4 `a` 旁边那行标题

来源：

- `draw_single_domain_figure()` 里直接写死：
  - `Cumulative citation knowledge graph snapshots`

也不是自动生成的。

### 12.5 每个小面板顶部的时间标题

来源：

1. `window_label(cfg["start_year"], end, cfg["start_year"])`
2. 然后在 `draw_snapshot()` 中又执行：
   - `.replace("1-", "0-")`

所以图上显示的通常会是：

- `0-5`
- `0-10`
- `0-15`
- `0-20`
- `0-25`

而不是原函数原始返回的 `1-5`、`1-10`。

这是 v3 里一个很容易忽略的小细节。

### 12.6 左侧竖着的领域名

来源：

- `cfg["domain_name"]`

在每一行的第一个图里画出来。

### 12.7 顶部时间轴

来源：

- `draw_top_time_axis()`

它会使用：

- `cfg["start_year"]`
- `cfg["end_year"]`
- `result.cumulative_windows`

来画刻度。

### 12.8 时间轴上的红色 “innovation event” 区域

来源：

- `cfg["anchors"]` 里的年份

代码会取：

- 最小 anchor 年份
- 最大 anchor 年份

然后画一个浅红色区间。

### 12.9 时间轴上 “2012-2013 landmark papers”

这个非常重要。

来源：

- `draw_top_time_axis()` 里写死的文本：
  - `"2012-2013\nlandmark papers"`

也就是说：

**这行字目前不是自动根据 anchor 年份生成的，而是硬编码的。**

如果你以后换领域，这里很可能会误导读者，除非你手动改代码。

这是 v3 目前一个必须特别说明的点。

### 12.10 每个主题的大色块圆圈

来源：

- 展示主题图 `TG`
- 节点大小字段 `n_papers`

半径通过：

- `log_scaled_radius()`

计算，控制参数来自：

- `plot.cluster_radius_min`
- `plot.cluster_radius_max`

颜色来自：

- `community_color_map()`

### 12.11 圆圈内部的小珠子

来源：

- 不是把每篇真实论文逐个画出来
- 而是用 `deterministic_disc_points()` 生成的一组“代表性 paper beads”

小珠子个数取决于：

- 这个主题的 `n_papers`

但位置是一个确定性的示意布局，不是论文真实坐标。

所以：

**圆圈内部的小点是“结构示意”，不是逐篇论文的真实散点图。**

### 12.12 主题之间的弯曲连线

来源：

- 当前累计展示主题图 `TG`
- 上一个累计展示主题图 `TGprev`

再经过：

- `select_backbone_edges()`

这个函数只会挑一小部分边来画：

1. 最大生成树 backbone
2. anchor 附近的重要边
3. 少量其余强边

所以主图中的边不是把所有主题边都画出来，而是一个稀疏 backbone。

### 12.13 深色边和浅色边

来源：

- 是否在上一个累计窗口里已经存在

规则：

- `is_new = True`：更深、更明显
- `is_new = False`：更浅、更透明

所以颜色深浅表达的是：

- 不是边权本身
- 而是“是不是这一阶段新出现的结构关系”

### 12.14 红色星标

来源：

- 当前展示主题节点是否包含 `anchor_labels`
- 且 `anchor_year <= 当前 end_year`

也就是说，红星标注的是：

**包含 landmark paper 的主题**

不是单独把那篇论文作为一个独立节点画出来。

### 12.15 红色注释文字

来源：

- `TG.nodes[n]["anchor_labels"]`

这个字段又来自：

- `mark_anchors()`
- 而 `mark_anchors()` 的来源是 YAML 中的 `anchors`

### 12.16 图上的主题名字

来源优先级：

1. `display_label`
2. 否则 `label`

其中：

- `label` 通常来自自动社区标签
- `display_label` 可能在模板布局中被 `topic_layout_templates[].name` 覆盖

所以你看到的名字，可能来自自动标签，也可能来自人工模板语义槽位。

### 12.17 面板下方的 caption

来源：

- `cfg["plot"]["panel_captions"]`

默认值是：

- `Fragmented prior knowledge`
- `Mechanistic consolidation`
- `Innovation shock: programmable Cas9`
- `Field reconfiguration`
- `Compression into translational hubs`

它们不是自动根据数据生成的，而是配置中的叙事性文案。

也就是说：

**这部分是作者给图加的解释性标题，不是模型算出来的结论。**

### 12.18 每个面板左下角的 `n=... papers / ... displayed topics`

来源：

- `n = len(active)`：当前累计窗口纳入的论文数
- `TG.number_of_nodes()`：当前显示出来的展示主题数

注意这里写的是：

- `displayed topics`

不是完整主题总数。

### 12.19 底部曲线图标题

来源：

- `draw_metric_panel()` 里直接写死：
  - `b  Perturbation fingerprint across rolling 5-year intervals`

所以：

- `b` 是固定面板编号
- 标题文本也是固定字符串

### 12.20 曲线图横轴

默认来源：

- `plot.metric_x_axis = "years"`

此时横轴用的是 rolling window 的中点年份：

- `0.5 * (rolling_start + rolling_end)`

并且 x 轴标题固定成：

- `Publication year / rolling citation-graph window`

如果你改成别的模式，才会退回累计窗口标签。

### 12.21 四条曲线的名字

来源：

- `draw_metric_panel()` 里的固定映射

分别是：

- `Expansion: new nodes / edges`
- `Bridging: cross-community paths`
- `Reconfiguration: community turnover`
- `Compression: path shortening`

这些是展示文案，不是列名直接原样输出。

### 12.22 曲线数值本身来自哪里

来源：

- `perturbation_metrics.csv`
- 更准确地说，来自 `compute_perturbation_metrics()` 产出的 `*_index`

这 4 条线对应：

- `Expansion_index`
- `Bridging_index`
- `Reconfiguration_index`
- `Compression_index`

### 12.23 曲线图中的红色阴影 landmark 区域

来源：

- `cfg["anchors"]` 中所有年份的最小值和最大值

所以阴影区间是自动从 anchor 年份算出来的。

### 12.24 曲线图里的 “landmark innovation”

来源：

- `draw_metric_panel()` 里写死的文字

阴影位置是数据驱动的，但这句解释文字本身是固定文案。

### 12.25 右下角 `Data source: OpenAlex; generated ...`

来源：

- 数据源文本固定写死为 `OpenAlex`
- 日期来自：
  - `dt.date.today().isoformat()`

所以这里显示的是运行当天日期，不是 OpenAlex 数据实际发布时间。

---

## 13. 下方曲线的数据是怎么计算出来的

曲线来自 `compute_perturbation_metrics()`。

它使用的是：

- 完整论文图 `G`
- 完整社区 `comm_map`
- 完整标签 `labels`

而不是展示社区。

所以这里再强调一次：

**下半部分曲线和上半部分快照，在社区层面不是完全同一套对象。**

### 13.1 Expansion

来源列：

- `new_nodes`
- `new_edges`
- `new_topics`

原始值：

```python
log1p(new_nodes) + log1p(new_edges) + log1p(new_topics)
```

### 13.2 Bridging

来源列：

- `intercommunity_edge_ratio`
- `participation_mean`
- `bridge_betweenness_top`

原始值：

```python
0.50 * inter_ratio + 0.35 * part_mean + 0.15 * bridge_bc
```

### 13.3 Reconfiguration

来源列：

- `partition_change`
- `edge_turnover`
- `modularity_shift`

原始值：

```python
partition_change + 0.5 * edge_turnover + modularity_shift
```

### 13.4 Compression

来源列：

- `path_gain`
- `hub_gain`
- `semantic_gain`

原始值：

```python
path_gain + hub_gain + sem_gain
```

### 13.5 为什么图里是 0 到 1 而不是 0 到 100

因为 v3 的 `draw_metric_panel()` 默认会检查：

- `plot.metric_scale`

如果是默认的 `unit`，就会把 `*_index` 再除以 `100.0`。

也就是说：

- CSV 里通常还是 0 到 100
- 图里默认画成 0 到 1 的归一化分数

这是 v3 的一个显示层变化。

---

## 14. 导出的 CSV 在 v3 里怎么读

### 14.1 `works_selected.csv`

v3 比旧版多了两列很重要：

- `display_community`
- `display_label`

这两列告诉你：

- 这篇论文在“展示层”属于哪个主题

同时还有：

- `community`
- `community_label`

它们对应完整分析层。

所以这个表是理解“两套社区系统”的最好入口。

### 14.2 `topic_nodes.csv`

v3 导出的是 `TG_display` 的节点。

字段：

- `community`
- `label`
- `n_papers`
- `cited_by_count`
- `first_year`
- `anchor_labels`
- `x`
- `y`

这里的 `x,y` 是展示布局坐标，不是统计分析特征。

### 14.3 `topic_edges.csv`

v3 导出的是 `TG_display` 的边，不是完整 `TG_full`。

所以这个表更接近“图上可能会被拿来画 backbone 的主题连接池”。

### 14.4 `perturbation_metrics.csv`

这是下半部分曲线最直接的来源。

你可以把它看成三层：

- 规模列：`n_cumulative_papers` 等
- 中间指标列：`new_edges`、`partition_change` 等
- 最终曲线列：`*_index`

---

## 15. 你看到的图并不等于“所有数据的原样投影”

这是 v3 必须理解的一点。

图里至少有 4 层压缩或设计：

1. 论文被采样成 `works_selected`
2. 完整社区又被筛成 `display_comm_map`
3. 展示边又被筛成 backbone 边
4. 主题内部小珠子只是示意，不是逐篇真实坐标

所以这张图的定位更准确地说是：

**“真实数据驱动的结构化论文示意图”**

而不是：

**“所有原始论文关系的无损可视化”**

---

## 16. v3 中几个特别值得注意的“图文来源”问题

### 16.1 `2012-2013 landmark papers` 是硬编码

这行文字不会自动随着别的领域变化。

如果你换成石墨烯、iPSC、Transformer 等领域，这里必须手改，否则会误导。

### 16.2 panel captions 是叙事文案，不是自动推断

例如：

- `Innovation shock: programmable Cas9`

这是作者给图加的解释层，不是代码从指标中“推理出来”的文本。

### 16.3 主题名字可能部分来自模板

如果配置里定义了 `topic_layout_templates`，图上名字可能不是纯自动提词。

### 16.4 下方曲线和上方快照不是同一可视化对象

上方：

- 用 `display_comm_map`

下方：

- 用完整 `comm_map`

因此不能简单理解为“每条曲线只由图上那几个显示主题计算出来”。

---

## 17. 推荐你怎么读源码

如果你现在要真正读懂 v3，我建议按这个顺序看：

1. `run_domain()`
2. `build_display_comm_map()`
3. `layout_topic_graph()`
4. `draw_snapshot()`
5. `draw_metric_panel()`
6. `draw_top_time_axis()`
7. `compute_perturbation_metrics()`

这样你会先吃透“图是什么”，再回来看“数值怎么算”。

---

## 18. 用一句人话总结 v3

如果用最通俗的话概括 v3：

**它先用真实 OpenAlex 数据构建一个完整的论文知识图来做定量分析，然后再从中挑出最能讲故事、最适合论文主图展示的几个主题模块，用一套更像 Nature 主图的示意结构画出来，同时把完整分析得到的 4 类扰动指标放在下方做定量对照。**

---

## 19. 你接下来最值得做的两件事

1. 对着 `run_domain()` 看这份说明，先把“完整分析层”和“展示层”的区别吃透。
2. 再单独精读：
   - `draw_snapshot()`
   - `draw_metric_panel()`

因为你这次特别关心“图中数据和标题来历”，这两个函数就是最核心的答案。

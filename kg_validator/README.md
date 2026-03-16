# 科学创新评价体系：知识图谱实现文档

> **冷启动兼容版** — 所有七个评价维度均只依赖参考文献结构，论文发表当天即可计算，无需等待下游引用积累。

---

## 项目结构

```
kg_validator/
├── fetcher.py        # OpenAlex API 数据拉取
├── graph_builder.py  # NetworkX 图构建 & 时间切片
├── metrics.py        # 七维指标计算（核心算法）
├── comparator.py     # 单篇指标对比 + 领域前后图谱三联图
├── main.py           # 主入口（demo / full / field_contrast）
└── README.md         # 本文档
```

---

## 快速开始

```bash
pip install networkx pandas numpy scipy matplotlib seaborn requests

# 演示模式（合成数据，无需网络，立即验证逻辑）
python main.py --mode demo

# 完整模式（从 OpenAlex 拉取真实数据）
python main.py --mode full --email your@email.com

# 领域图谱时间节点前后对比（三联图）
python main.py --mode field_contrast \
  --filter "concepts.id:C86803240,type:article" \
  --event-year 2024 \
  --event-label "Chemistry Nobel 2024" \
  --email your@email.com

# 目标论文发表前后对比（以论文发表年为分界点）
python main.py --mode paper_contrast \
  --filter "concepts.id:C86803240,type:article" \
  --paper-dois "10.1038/s41586-021-03819-2" \
  --event-label "AlphaFold release" \
  --email your@email.com

# 纯论文邻域模式（不指定领域范围）
python main.py --mode paper_neighborhood_contrast \
  --paper-dois "10.1038/s41586-021-03819-2" \
  --event-label "AlphaFold neighborhood" \
  --neighbor-max-refs 20 \
  --neighbor-citers-per-target 120 \
  --neighbor-citers-per-ref 30 \
  --email your@email.com
```

输出文件保存在 `output/` 目录下：

| 文件 | 内容 |
|------|------|
| `radar_demo.png` | 七维指标雷达图（奖前 vs 奖后） |
| `bars_demo.png` | 七维指标柱状对比图 |
| `ego_demo.png` | 获奖论文 ego 网络（奖前 vs 奖后） |
| `modularity_timeline_demo.png` | 模块度 Q 时间演变曲线 |
| `results_demo.csv` | 所有指标的数值结果 |
| `field_contrast_demo.png` | 演示用领域前后图谱三联图 |
| `community_shift_demo.csv` | 演示用社区变化明细 |
| `field_before_demo.graphml` | 演示用事件前图谱快照 |
| `field_after_demo.graphml` | 演示用事件后图谱快照 |

---

## 新增：领域时间节点前后知识图谱对比

这个模式用于**在引出七个创新性指标之前**，先证明“创新性/重大突破确实会反映在论文引用关系知识图谱上”。

程序会围绕一个给定时间节点（例如诺奖颁布、重大方法发布）构建两张图：

- **Before 图**：`[event_year-before_years, event_year-1]`
- **After 图**：`[event_year-before_years, event_year+after_years]`

然后输出一张三联图：

1. 左图：时间节点前的论文级引用图谱
2. 中图：时间节点后的论文级引用图谱
3. 右图：社区变化摘要（社区数、`ΔQ`、跨社区边比例、最大新社群、Top 3 新社群主导 field/topic）

### 命令模板

```bash
python main.py --mode field_contrast \
  --filter "concepts.id:C86803240,type:article" \
  --event-year 2024 \
  --event-label "Chemistry Nobel 2024" \
  --before-years 10 \
  --after-years 5 \
  --max-plot-nodes 180 \
  --min-community-size 8 \
  --email your@email.com
```

### 参数说明

| 参数 | 含义 |
|------|------|
| `--filter` | 原始 OpenAlex filter，直接定义“领域/方向”范围 |
| `--event-year` | 时间节点年份 |
| `--event-label` | 图标题中的事件名称 |
| `--before-years` | 前窗口长度，默认 `10` |
| `--after-years` | 后窗口长度，默认 `5` |
| `--max-plot-nodes` | 论文级图谱最大绘图节点数，默认 `180` |
| `--min-community-size` | 判定显著新社群的最小社区规模，默认 `8` |
| `--max-records` | OpenAlex 最大拉取记录数，可选 |

### 输出文件

当 `event_label = "Chemistry Nobel 2024"` 时，默认会生成：

| 文件 | 内容 |
|------|------|
| `field_contrast_chemistry_nobel_2024_2024.png` | 领域前后图谱三联图 |
| `community_shift_chemistry_nobel_2024_2024.csv` | 社区变化明细 |
| `field_before_chemistry_nobel_2024_2024.graphml` | 事件前快照 |
| `field_after_chemistry_nobel_2024_2024.graphml` | 事件后快照 |

### 如何解读三联图

- 如果中图出现**新的致密簇/新社群**，且右侧摘要显示：
  - 社区数上升，或
  - `ΔQ` 明显变化，或
  - 跨社区边比例提升，或
  - 出现 `post_event_share` 很高的新社群，
  那就说明这个领域的知识组织方式在事件前后发生了结构变化。
- 如果程序没有识别到显著新社群，也不会强行制造差异；图中和 CSV 会明确写出：
  `未检测到显著新社群 / Difference is weak.`

### Demo 的意义

`python main.py --mode demo` 现在除了原来的七维指标图，还会额外生成一张**保证能看出新社群**的三联图，方便你在论文、汇报或方法章节里先展示“图谱结构变化”这一事实，再自然引出七个评价维度。

---

## 新增：目标论文发表前后知识图谱对比

如果你关心的不是某个抽象时间节点，而是**某一篇或某几篇论文发表后，领域图谱有没有发生显著结构变化**，请使用 `paper_contrast` 模式。

这个模式与 `field_contrast` 的区别是：

- `field_contrast`：时间点由你手工给定 `--event-year`
- `paper_contrast`：时间点自动取自目标论文的 `publication_year`

### 命令模板

```bash
python main.py --mode paper_contrast \
  --filter "concepts.id:C86803240,type:article" \
  --paper-dois "10.1038/s41586-021-03819-2" \
  --before-years 10 \
  --after-years 5 \
  --max-plot-nodes 180 \
  --min-community-size 8 \
  --email your@email.com
```

也可以一次传多篇论文：

```bash
python main.py --mode paper_contrast \
  --filter "concepts.id:C86803240,type:article" \
  --paper-ids "W3177828909,W2038196424" \
  --email your@email.com
```

### 参数说明

| 参数 | 含义 |
|------|------|
| `--paper-dois` | 目标论文 DOI，多个用英文逗号分隔 |
| `--paper-ids` | 目标论文 OpenAlex ID，多个用英文逗号分隔 |
| `--filter` | 定义你要比较的领域/方向范围 |
| `--event-label` | 图标题前缀；若不给，默认用论文标题 |

### 逐参数解释（领域限定版）

下面这条命令：

```bash
python kg_validator/main.py --mode paper_contrast \
  --filter "primary_location.source.id:S137773608,type:article" \
  --paper-dois "10.1038/s41586-021-03819-2" \
  --before-years 10 \
  --after-years 5 \
  --output-dir output \
  --email your@email.com
```

可以拆成下面几个部分理解：

| 参数 | 作用 | 直观解释 |
|------|------|----------|
| `python kg_validator/main.py` | 运行主入口脚本 | 启动整个知识图谱分析流程 |
| `--mode paper_contrast` | 选择“领域限定版的论文前后对比模式” | 在**你指定的领域范围内**，观察目标论文发表前后图谱有没有结构变化 |
| `--filter "primary_location.source.id:S137773608,type:article"` | 限定背景知识图谱的取样范围 | 只抓满足该 OpenAlex filter 的论文；这里的意思是“只取某个 `source.id=S137773608` 下、类型为 `article` 的论文” |
| `--paper-dois "10.1038/s41586-021-03819-2"` | 指定要分析的目标论文 | 程序会先去 OpenAlex 用 DOI 找到这篇论文，再读取它的发表年份作为时间分界点 |
| `--before-years 10` | 事件前窗口长度 | 如果目标论文发表于 `Y` 年，则前图使用 `Y-10` 到 `Y-1` 的论文 |
| `--after-years 5` | 事件后窗口长度 | 后图使用 `Y-10` 到 `Y+5` 的论文；这样可以比较论文发表后 5 年内图谱是否重组 |
| `--output-dir output` | 指定结果保存目录 | 生成的 PNG、CSV、GraphML 都会落在 `output/` 下 |
| `--email your@email.com` | 给 OpenAlex API 传入 `mailto` | 建议换成你自己的邮箱，便于 API 识别请求来源 |

这条命令的真实含义是：

- 先在 `source.id=S137773608` 且 `type=article` 的论文集合里建立一个领域图谱；
- 再围绕 DOI `10.1038/s41586-021-03819-2` 对应论文的发表年份切出前后两张图；
- 最后判断这篇论文发表后，这个**限定领域内部**是否出现了新社群、社区重组或跨社区连接变化。

如果你**没有传 `--event-label`**，程序会自动把目标论文标题用作图标题。

### `paper_contrast` 中最重要的几个参数怎么理解

- `--filter`
  - 它回答的是：**“你打算在哪个领域/方向里看这篇论文的影响？”**
  - 这是最重要的边界条件；同一篇论文，在不同 `filter` 下得到的图谱差异可能完全不同。
- `--before-years`
  - 控制“比较的历史背景有多长”。
  - 取值太小，前图可能太稀；取值太大，可能把不相关的旧结构也混进来。
- `--after-years`
  - 控制“给这篇论文多长时间去显现结构影响”。
  - 太短可能看不到变化，太长则可能混入后续别的事件影响。

### 输出特征

- 每篇目标论文都会单独生成一组三联图和 CSV。
- 三联图中，**目标论文会用星形高亮**显示在事件后图谱中。
- 右侧摘要会额外列出目标论文所在社区及其状态（`new` / `expanded` / `inherited`）。

---

## 新增：纯论文邻域模式

如果你不想先人为指定一个领域，而是想直接问：

**“围绕这篇论文本身及其引用/被引邻域，发表前后图谱结构有没有显著变化？”**

请使用 `paper_neighborhood_contrast`。

### 与 `paper_contrast` 的区别

- `paper_contrast`：需要 `--filter`，表示“在指定领域内观察目标论文发表前后”
- `paper_neighborhood_contrast`：不需要 `--filter`，表示“只围绕目标论文邻域观察发表前后”

### 邻域图如何构建

纯论文邻域模式会自动收集：

- 目标论文本身
- 目标论文的参考文献（默认每篇最多 `20` 篇）
- 引用目标论文的论文
- 引用目标论文参考文献的论文

然后把这些论文合并成一个局部知识图谱，再以目标论文发表年为分界点做前后对比。

### 命令模板

```bash
python main.py --mode paper_neighborhood_contrast \
  --paper-dois "10.1038/s41586-021-03819-2" \
  --before-years 10 \
  --after-years 5 \
  --neighbor-max-refs 20 \
  --neighbor-citers-per-target 120 \
  --neighbor-citers-per-ref 30 \
  --email your@email.com
```

### 邻域参数说明

| 参数 | 含义 |
|------|------|
| `--neighbor-max-refs` | 每篇目标论文纳入的最大参考文献数 |
| `--neighbor-citers-per-target` | 每篇目标论文最多拉取多少篇施引论文 |
| `--neighbor-citers-per-ref` | 每篇参考文献最多拉取多少篇施引论文 |

### 逐参数解释（纯邻域版）

下面这条命令：

```bash
python kg_validator/main.py --mode paper_neighborhood_contrast \
  --paper-dois "10.1038/s41586-021-03819-2" \
  --before-years 10 \
  --after-years 5 \
  --neighbor-max-refs 20 \
  --neighbor-citers-per-target 120 \
  --neighbor-citers-per-ref 30 \
  --output-dir output \
  --email your@email.com
```

含义如下：

| 参数 | 作用 | 直观解释 |
|------|------|----------|
| `--mode paper_neighborhood_contrast` | 选择“纯论文邻域模式” | 不先规定领域，而是直接围绕目标论文的局部引用生态看变化 |
| `--paper-dois "10.1038/s41586-021-03819-2"` | 指定目标论文 | 这篇论文就是局部邻域图的中心 |
| `--before-years 10` | 事件前窗口长度 | 以前 `10` 年作为“论文发表前”的背景图 |
| `--after-years 5` | 事件后窗口长度 | 发表后 `5` 年作为“论文发表后”的观察图 |
| `--neighbor-max-refs 20` | 邻域种子参考文献上限 | 最多从目标论文的参考文献列表中纳入 `20` 篇作为局部图谱的起点 |
| `--neighbor-citers-per-target 120` | 目标论文施引样本上限 | 最多抓取 `120` 篇“引用了目标论文”的论文 |
| `--neighbor-citers-per-ref 30` | 参考文献施引样本上限 | 对每篇参考文献，再最多抓取 `30` 篇“引用了该参考文献”的论文 |
| `--output-dir output` | 指定结果保存目录 | 所有输出文件写到 `output/` |
| `--email your@email.com` | 给 OpenAlex API 传入 `mailto` | 实际运行时建议替换成你的真实邮箱 |

这条命令的真实含义是：

- 不先说“这个论文属于哪个领域”；
- 而是从目标论文本身出发，自动收集：
  - 它的参考文献；
  - 它的施引论文；
  - 它的参考文献的施引论文；
- 然后把这些论文拼成一个**局部知识图谱**；
- 最后再按目标论文发表年切出前后两张图，看局部知识结构是否显著改变。

### 这三个邻域参数的影响非常大

- `--neighbor-max-refs`
  - 越大，邻域图越能覆盖目标论文的知识来源；
  - 但也越容易把邻域扩得太宽、混入更多噪声。
- `--neighbor-citers-per-target`
  - 越大，越能看清“目标论文自己带来的后续扩散”；
  - 但请求量也会明显增加。
- `--neighbor-citers-per-ref`
  - 它控制的是“目标论文所站立的旧知识基础周围，有多少背景结构被纳入”；
  - 这个值越高，越有利于比较“新结构”与“旧结构”的差异，但图会更复杂。

### 一个简洁理解方式

- `paper_contrast` 看的是：**这篇论文在某个既定领域里有没有改变结构**
- `paper_neighborhood_contrast` 看的是：**围绕这篇论文自己长出来的局部知识生态有没有改变结构**

### 什么时候用哪个模式

- 如果你已经知道要在**哪个领域/方向**里证明结构变化，用 `paper_contrast`
- 如果你想先从**目标论文自身扩散出来的局部知识结构**看变化，用 `paper_neighborhood_contrast`

---

## 设计背景：为什么要替换 CD 和 H？

原始体系包含两个依赖**下游引用**的指标：

| 原指标 | 问题 |
|--------|------|
| CD 颠覆指数 | 需要知道后续哪些论文"只引 p 不引 p 的参考文献"，新文章没有后续论文 |
| 知识扩散熵 H | 需要知道下游施引论文的学科分布，新文章 H = 0 毫无意义 |

**核心洞察**：CD 和 H 衡量的是论文**已经产生的影响**；而创新评价更需要的是论文**将要产生影响的潜力**。这个潜力可以通过参考文献的结构来预测。

---

## 七个评价维度

所有指标均只依赖参考文献，✅ = 发表当天可计算。

### 维度一：结构洞桥接中心性 B（Betweenness Centrality）✅

**来源**：Freeman (1977)

**公式**：
```
B(v) = Σ_{s≠v≠t}  σ_st(v) / σ_st
B_norm(v) = 2·B(v) / [(n-1)(n-2)]
```

**解读**：衡量论文在知识图谱中的桥梁作用。桥接中心性越高，说明该论文连接了越多原本不相连的知识群落。

**预期验证信号**：获奖论文在 G⁻ 中 B_norm 显著偏高；进入 G⁺ 后，网络密化，B_norm 略降但绝对值仍高。

---

### 维度二：RS 跨学科多样性指数（Rao-Stirling）✅

**来源**：Rao (1982)；Stirling (2007)

**公式**：
```
RS = Σ_{i≠j}  d_ij · p_i · p_j

p_i  = 参考文献中属于学科 i 的比例
d_ij = 1 − cos(c_i, c_j)   （学科间余弦距离）
```

**解读**：同时考虑学科多样性（variety）、平衡性（balance）和差异性（disparity）三个维度。RS 越高，说明论文整合了来自多个遥远学科的知识。

**预期验证信号**：获奖论文 RS 显著高于领域均值；G⁺ 后该方向研究的 RS 往往收窄（形成新子领域）。

---

### 维度三：社区模块度变化量 ΔQ（Community Modularity Shift）✅（图级）

**来源**：Newman & Girvan (2004)

**公式**：
```
Q = (1/2m) · Σ_{i,j} [A_ij − k_i·k_j/(2m)] · δ(c_i, c_j)
ΔQ = Q(G⁺) − Q(G⁻)
```

**解读**：
- ΔQ < 0 → 原有社区边界被打破（颠覆性创新信号）
- ΔQ > 0 → 社区内部连接加深（巩固型信号）

**注意**：这是图级别的指标，衡量的是整个知识图谱的结构变化，而非单篇论文的属性。

---

### 维度四：Uzzi 非典型组合新颖性（Atypical Combination）✅

**来源**：Uzzi et al. (2013, *Science*)

**公式**：
```
z_{j1,j2} = (O_{j1,j2} − μ_{j1,j2}) / σ_{j1,j2}
Novelty(p) = p10({z_{j1,j2}})   ← 所有期刊对 z-score 的第 10 百分位
```

其中 μ 和 σ 通过对参考文献列表随机置换（蒙特卡洛）得到。

**解读**：最有影响力的论文往往是"以高度传统的知识组合为基底，同时注入少量非典型组合"。**低 p10（极度非典型的期刊组合）+ 高中位 z（高整体传统性）** 是高影响力论文的双峰特征。

**预期验证信号**：获奖论文 p10 < 0（存在非典型期刊对）；G⁺ 后随着该组合被标准化，z 值整体漂移。

---

### 维度五：RTD 引用目标多样性 ⭐ 冷启动替代 CD ✅

**来源**：本体系原创，基于 Simpson 多样性指数

**公式**：
```
RTD(p) = 1 − Σ_c [n_c · (n_c − 1)] / [N · (N − 1)]

n_c = 参考文献中属于社区 c 的数量
N   = 参考文献总数
```

**解读**：如果论文 p 的参考文献均匀分布在多个不同社区（RTD 接近 1），说明 p 正在连接孤立的知识岛，是颠覆性创新的先兆。若参考文献集中在同一社区（RTD 接近 0），则是渐进式巩固研究。

**与 CD 的关系**：CD 衡量的是"后续是否有人绕过你的参考文献"（结果），RTD 衡量的是"你的参考文献是否跨越了多个社区"（先验结构），两者逻辑互为镜像。

**RTD vs RS**：RS 看学科标签的多样性；RTD 看图拓扑社区的多样性。两者互补——RS 依赖 OpenAlex 学科分类的质量，RTD 只依赖引用图结构本身。

---

### 维度六：Burt 结构约束系数 ⭐ 冷启动替代 CD（互补视角）✅

**来源**：Burt (1992, *Structural Holes*)

**公式**：
```
C(p) = Σ_i (p_i + Σ_{j≠i} p_ij · p_j)²
IP(p) = 1 − C_norm(p) ∈ [0, 1]

p_i   = 参考文献中节点 i 的权重（均等时为 1/n）
p_ij  = i 与 j 之间的归一化连接强度
```

**解读**：约束系数 C 越低，说明参考文献之间联系越稀疏（填补的结构洞越多），创新潜力 IP 越高。

**RTD vs Burt 的区别**：
- RTD 关注社区层面的多样性（宏观）
- Burt 关注节点层面的网络嵌入结构（微观）
- 两者结合覆盖不同粒度的结构洞分析

---

### 维度七：PDE 预期扩散熵 ⭐ 冷启动替代知识扩散熵 H ✅

**来源**：Shannon (1948)；本体系将其应用于参考文献的学科分布

**公式**：
```
PDE(p)      = −Σ_k q_k · log₂(q_k)
PDE_norm(p) = PDE(p) / log₂(K)      ← 归一化到 [0, 1]

q_k = 参考文献中属于学科 k 的比例
K   = 学科总数
```

**解读**：参考文献学科分布越均匀（熵越高），论文被多个领域研究者读到和引用的概率就越高，**预期扩散范围越广**。

**PDE vs H（原始扩散熵）的关系**：
- H 是实测值（下游施引论文的学科分布），需要引用积累，反映**实际扩散**
- PDE 是预测值（参考文献的学科分布），发表即可计算，反映**预期扩散潜力**
- 两者在方向上高度一致（Shibayama et al. 2021 验证了类似逻辑）

**PDE vs RS 的区别**：
- RS 加权了学科间的语义距离（多样性 × 差异度）
- PDE 只看纯分布熵，对学科数量 K 更敏感，是 RS 的低成本互补

---

## 指标汇总对照表

| # | 指标 | 类型 | 冷启动 | 原始 | 核心数据来源 | 预期奖后变化 |
|---|------|------|--------|------|-------------|-------------|
| 1 | 桥接中心性 B | 图拓扑 | ✅ | Freeman 1977 | 引用图结构 | ↑ 升高 |
| 2 | RS 跨学科性 | 语义多样性 | ✅ | Stirling 2007 | 参考文献学科标签 | ↑ 升高 |
| 3 | 模块度 ΔQ | 图拓扑（全局） | ✅ | Newman 2004 | 整张知识图谱 | ↓ 降低（社区破壁）|
| 4 | Uzzi 新颖性 | 统计异常性 | ✅ | Uzzi 2013 | 参考文献期刊对 | ↓ p10 更负 |
| 5 | **RTD** 引用目标多样性 | 图拓扑 | ✅ | 本体系 | 参考文献+图社区 | ↑ 升高（后续论文） |
| 6 | **Burt** 结构约束 IP | 图拓扑（微观） | ✅ | Burt 1992 | 参考文献连接结构 | ↑ 升高 |
| 7 | **PDE** 预期扩散熵 | 信息熵 | ✅ | Shannon 1948 | 参考文献学科分布 | ↑ 升高（高影响论文） |

> **说明**：RTD、Burt IP、PDE 三个冷启动指标衡量的是论文**本身的参考文献结构**，因此在奖前/奖后比较中数值不变（种子论文的参考文献不会随时间改变）。它们的验证方式是：**高创新性论文的这三个指标应显著高于同领域同时期的普通论文**（横截面对比）。

---

## 综合创新评分（CIS）

```
CIS(p) = w1·B̂ + w2·RS ̂ + w3·(−δQ̂) + w4·(−Uzzi_p10 ̂) + w5·RTD ̂ + w6·IP ̂ + w7·PDE_norm ̂

其中 X̂ = (X − X̄_field) / σ_field  （相对领域均值的标准化 z-score）
初始建议：均等权重 wi = 1/7
优化建议：以历史诺贝尔论文（正样本）训练 logistic 回归标定各维度权重
```

---

## 使用 API 评估一篇新论文

```python
import networkx as nx
from fetcher import fetch_works_by_doi, normalize_work
from graph_builder import build_graph
from metrics import compute_all_metrics_for_paper, _build_journal_copair_baseline

# 1. 拉取新论文及其参考文献（只需参考文献列表，无需等待引用）
works_raw = fetch_works_by_doi(["10.xxxx/your.new.paper"], email="you@email.com")
works = [normalize_work(w) for w in works_raw]

# 2. 加载背景知识图谱（同领域近 10 年论文）
# （假设 G_background 已通过 build_kg.py 构建好）
G = nx.read_graphml("kg_background.graphml")

# 3. 将新论文插入图中（只加节点和出引边，尚无入引）
for w in works:
    nid = w["id"]
    G.add_node(nid, **{k: v for k, v in w.items() if k not in ("id", "referenced_works")})
    for ref in w["referenced_works"]:
        G.add_edge(nid, ref)

# 4. 构建 Uzzi 基线（可缓存复用）
baseline = _build_journal_copair_baseline(G, n_permutations=100)

# 5. 一键计算全部七维指标
new_paper_id = works[0]["id"]
metrics = compute_all_metrics_for_paper(
    new_paper_id, G,
    uzzi_baseline=baseline,
)
print(metrics)
# 输出示例：
# {
#   'betweenness': 0.312,
#   'rao_stirling': 0.745,
#   'delta_q': None,          <- 图级指标，需要两张图才能计算
#   'uzzi_novelty_p10': -3.2,
#   'rtd_rtd': 0.867,
#   'burt_innovation_potential': 0.923,
#   'pde_pde_norm': 0.891,
#   ...
# }
```

---

## 参考文献

### 新增引用（近年核心）

- **Park, M., Leahey, E., & Funk, R. J. (2023)**. Papers and patents are becoming less disruptive over time. *Nature*, 613, 138–144.
- **Bornmann, L., et al. (2023)**. What do we know about the disruption index in scientometrics? *Scientometrics*, 126, 5221–5249.
- **Uzzi, B., Mukherjee, S., Stringer, M., & Jones, B. F. (2013)**. Atypical combinations and scientific impact. *Science*, 342(6157), 468–472.
- **Shibayama, S., Yin, D., & Matsumoto, K. (2021)**. Measuring novelty in science with word embedding. *PLOS ONE*, 16(7), e0254034.
- **Yin, D., et al. (2023)**. Identify novel elements of knowledge with word embedding. *PLOS ONE*, 18(6), e0284567.

### 基础方法文献

- **Freeman, L. C. (1977)**. A set of measures of centrality based on betweenness. *Sociometry*, 40(1), 35–41.
- **Rao, C. R. (1982)**. Diversity and dissimilarity coefficients. *Theoretical Population Biology*, 21(1), 24–43.
- **Stirling, A. (2007)**. A general framework for analysing diversity in science. *Journal of the Royal Society Interface*, 4(15), 707–719.
- **Newman, M. E. J., & Girvan, M. (2004)**. Finding and evaluating community structure in networks. *Physical Review E*, 69(2), 026113.
- **Shannon, C. E. (1948)**. A mathematical theory of communication. *Bell System Technical Journal*, 27(3), 379–423.
- **Burt, R. S. (1992)**. *Structural Holes: The Social Structure of Competition*. Harvard University Press.

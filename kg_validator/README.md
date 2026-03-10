# 科学创新评价体系：知识图谱实现文档

> **冷启动兼容版** — 所有七个评价维度均只依赖参考文献结构，论文发表当天即可计算，无需等待下游引用积累。

---

## 项目结构

```
kg_validator/
├── fetcher.py        # OpenAlex API 数据拉取
├── graph_builder.py  # NetworkX 图构建 & 时间切片
├── metrics.py        # 七维指标计算（核心算法）
├── comparator.py     # 前后对比逻辑 + 可视化
├── main.py           # 主入口（demo / full 两种模式）
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
```

输出文件保存在 `output/` 目录下：

| 文件 | 内容 |
|------|------|
| `radar_demo.png` | 七维指标雷达图（奖前 vs 奖后） |
| `bars_demo.png` | 七维指标柱状对比图 |
| `ego_demo.png` | 获奖论文 ego 网络（奖前 vs 奖后） |
| `modularity_timeline_demo.png` | 模块度 Q 时间演变曲线 |
| `results_demo.csv` | 所有指标的数值结果 |

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

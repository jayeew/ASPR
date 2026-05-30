# Fig. 1 读图说明

这张图用于展示一个学术领域在 landmark innovation 前后，引用知识图谱如何从分散的知识模块逐步走向连接、重排和收敛。当前固定版本以 **CRISPR-Cas genome editing** 为例，数据来自 OpenAlex，图形经过主题社区压缩和出版级可视化整理。

读图时需要先抓住两层信息：

- **Panel a** 展示累计 citation knowledge graph snapshots，也就是到每个时间点为止已经形成的主题结构。
- **Panel b** 展示 rolling 5-year windows 上的 dominant parameter trajectories，用几个结构参数解释 panel a 中观察到的图谱转变。

这张图不是原始论文引用网络的无损投影，也不是单独用来证明因果关系的统计检验。它的定位是：用真实文献数据构建一个可解释的结构化主图，帮助读者理解 landmark paper 之后知识结构可能发生的扰动模式。

## Panel a: Cumulative citation knowledge graph snapshots

Panel a 的五个小图是累计快照：

| 面板 | 含义 |
| --- | --- |
| `2000-2004` | 起始 5 年内形成的早期知识结构 |
| `2000-2009` | 2000 到 2009 年累计形成的结构 |
| `2000-2014` | landmark papers 出现后的累计结构 |
| `2000-2019` | 后续扩散和重构后的累计结构 |
| `2000-2024` | 全时间段结束时的累计结构 |

这些面板不是彼此独立的 5 年切片。越往右，图中包含的论文越多，表示这个领域到该时间点为止积累出的知识图谱。

### 主题圆圈

每个大圆圈表示一个被压缩后的主题社区，而不是单篇论文。底层首先构建论文级混合图，再通过社区发现把相互接近的论文聚合成主题模块。

圆圈可以这样理解：

- **位置** 表示主题之间的关系布局，既受图结构影响，也受配置里的语义模板约束。
- **大小** 主要反映该主题社区中累计论文数量的相对规模。
- **颜色** 用于区分不同主题社区。
- **内部小点** 是代表性 paper beads，用来表达主题内部包含多篇论文，但不是逐篇论文的真实坐标。
- **标签** 是根据论文标题、OpenAlex topics、关键词等生成或由布局模板覆盖后的主题名称。

左下角的 `n=... papers / ... displayed topics` 表示当前累计窗口中纳入的论文数量，以及最终展示出来的主题数。注意这里写的是 `displayed topics`，不是完整社区总数；长尾社区会被隐藏，以保证主图清晰。

### 主题连线

主题之间的边来自论文级图谱中跨社区的知识联系。底层论文级边综合了三类证据：

- **Direct citation**：论文之间的直接引用。
- **Bibliographic coupling**：两篇论文共享参考文献。
- **Co-citation**：两篇论文被后续论文共同引用。

在主图里，所有跨主题边不会全部画出。图中显示的是筛选后的 backbone：每个连通部分的主要连接、landmark 附近的重要连接，以及少量强边。

边的颜色深浅表示新旧关系：

- **深色边**：相对于上一个累计面板新出现的主题间连接。
- **浅色边**：之前已经存在、在当前累计图中延续的连接。

因此，边不是单纯表达“强弱”，而是在帮助读者看到某一阶段是否出现了新的跨模块知识关系。

### 红色星标和时间轴

红色星标表示包含 landmark paper 的主题社区，而不是把 landmark paper 单独画成一个论文节点。当前 CRISPR 图中标注的关键事件主要是 2012-2013 年的 CRISPR-Cas9 landmark papers。

顶部时间轴中的红色区域对应 innovation event。Panel b 中的浅红色背景则覆盖包含这些 landmark papers 的 rolling 5-year window，也就是 `2010-2014`。

### 五个阶段的读法

这张图最适合从左到右读：

| 阶段 | 图中叙事 | 结构含义 |
| --- | --- | --- |
| `2000-2004` | Fragmented prior knowledge | 早期 CRISPR 相关知识分散在免疫、防御、重复序列、RNA 和 Cas 蛋白等模块中，跨模块连接较弱。 |
| `2000-2009` | Mechanistic consolidation | CRISPR loci、RNA biology、Cas proteins 等模块开始被整理成更稳定的机制性知识基础。 |
| `2000-2014` | Innovation shock: programmable Cas9 | Jinek 2012、Cong/Mali 2013 等 landmark papers 进入图谱，Cas9、RNA、editing、delivery 等模块之间出现更密集连接。 |
| `2000-2019` | Field reconfiguration | 主题模块之间的连接格局被重新组织，genome editing、screens/diagnostics、delivery/therapy 等方向开始扩散。 |
| `2000-2024` | Compression into translational hubs | 领域围绕更稳定的应用和技术中心收敛，部分主题成为连接多个方向的核心 hub。 |

简单说，Panel a 讲的是：landmark innovation 出现前，知识模块相对分散；出现后，新论文和新引用关系把原有模块连起来；之后领域逐步重排，并向少数核心技术路线和应用场景压缩。

## Panel b: Dominant parameter trajectories

Panel b 不是再画一张网络图，而是用 rolling 5-year windows 给 panel a 的结构变化提供参数化解释。横轴是：

```text
2000-2004, 2005-2009, 2010-2014, 2015-2019, 2020-2024
```

纵轴是 standardized parameter value。数值越高，表示该参数在当前窗口相对更强；负值通常表示相对于基线或方向性定义的下降、反向或边界破坏。

当前固定图中画了四条 dominant parameter trajectories：

| 参数 | 图中含义 | 如何解释 |
| --- | --- | --- |
| **B (bridge centrality)** | 桥接中心性 | 值升高表示新知识更强地连接原本分散的主题模块。 |
| **RTD (reference target diversity)** | 参考目标多样性 | 值升高表示论文引用或连接的知识来源更分散，跨主题引用更多。 |
| **Uzzi novelty (-p10)** | 非典型知识重组 | 值升高表示参考文献组合更不寻常，更可能代表跨范式组合或新颖重组。 |
| **DeltaQ directionality** | 模块度变化方向 | 负值通常表示社区边界被打破；回到接近 0 则表示重排后趋于稳定。 |

### 为什么 2010-2014 是重点窗口

CRISPR 的 landmark papers 出现在 2012-2013 年，因此它们落入 `2010-2014` 这个 rolling window。图中该区域用浅红色背景强调。

在当前图里可以看到：

- `B` 在 landmark 窗口明显升高，说明跨社区桥接增强。
- `RTD` 升高，说明参考目标和知识来源变得更分散。
- `Uzzi novelty` 升高，说明非典型知识组合增强。
- `DeltaQ` 明显为负，说明原有社区边界在这个阶段受到冲击。

这些轨迹共同支持一个读图判断：landmark innovation 不只是增加了论文数量，而是改变了知识模块之间的连接方式。

### 底部四个解释框

Panel b 下方的四个解释框把曲线压缩成更直观的机制说明：

| 解释框 | 对应读法 |
| --- | --- |
| **Bridging** | `B + RTD up`，跨社区路径增加，知识模块被连接起来。 |
| **Novelty** | `Uzzi up; DeltaQ < 0`，非典型组合增强，并伴随社区边界扰动。 |
| **Boundary breaking** | `DeltaQ < 0`，原有社区划分被打破或重新组合。 |
| **Convergence** | `B/RTD sustained; DeltaQ near 0`，桥接关系保留，但结构逐步稳定。 |

## Panel a 和 Panel b 怎么合起来读

Panel a 给出结构直觉，Panel b 给出参数证据。两者可以按这样的逻辑合读：

1. 先看 Panel a 中主题是否从分散变得更连通。
2. 再看 Panel b 中 `B` 和 `RTD` 是否上升，判断这种变化是否对应跨模块桥接。
3. 观察 landmark 窗口中 `Uzzi` 是否升高、`DeltaQ` 是否下降，判断是否发生非典型重组和社区边界破坏。
4. 最后看右侧面板是否形成更稳定的核心结构，同时 `DeltaQ` 是否回到接近 0，判断领域是否进入重排后的收敛阶段。

## 解释边界

这张图可以支持“结构扰动可视化”和“机制性解释”，但不应被单独解释为严格因果证明。

需要注意：

- OpenAlex 数据会受到检索词、收录范围和引用延迟影响。
- 主图展示的是筛选后的 display topics，不是所有社区。
- 主题位置和标签经过可视化模板整理，服务于解释和出版呈现。
- Panel b 的参数是标准化轨迹，适合比较同一图中不同时间窗口的相对变化，不适合脱离上下文解释为绝对量。
- 若要做强统计结论，需要额外加入 matched controls、null models 或重采样检验。

一句话总结：这张 Fig. 1 讲的是 CRISPR-Cas9 landmark papers 前后，一个领域如何从分散知识基础，经由桥接和边界破坏，逐步重排并压缩成更稳定的知识结构。

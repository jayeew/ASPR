# Fig. 2 Panel b 候选指标库与七参数筛选标准

> 目的：为 Fig. 2 Panel b 提供一个可写入论文 Methods / Supplementary Table 的系统性候选指标库。  
> 逻辑：先枚举可用于评估论文创新性、重要性、知识图谱扰动潜力的候选指标，再按冷启动可计算性、参考文献依赖性、图谱扰动机制、非冗余性与验证可行性逐步筛选，最终保留七个核心指标：**B、RS、ΔQ0、Uzzi、RTD、Burt IP、PDE**。

---

## 一、总览：为什么要先做候选指标库

Fig. 1 已经展示了 landmark paper 发表后会在知识图谱中诱发四类宏观扰动：

- **Expansion**：新增论文、边、主题或知识模块；
- **Bridging**：原本分离的知识社区被连接；
- **Reconfiguration**：社区结构、模块边界和引用路径发生重排；
- **Compression**：后续研究围绕新范式收敛，路径变短、hub 集中、术语和主题收缩。

Fig. 2 的任务不是再次画知识图谱演化，而是解释：**我们为何选择这七个发表当天即可计算的参考文献侧指标来参数化这些扰动。**

因此，本文先构建一个候选指标宇宙，包括：

1. 未来影响 / 结果型指标；
2. 论文、作者、期刊元数据指标；
3. 参考文献年龄、深度、流行度指标；
4. 非典型组合 / 组合新颖性指标；
5. 跨学科多样性 / 知识分布指标；
6. 图拓扑中心性 / 结构洞指标；
7. 社区重构 / 模块结构指标；
8. 语义 / 文本 / embedding novelty 指标。

候选指标共 **92 个**。其中最终保留的七个核心指标是：

| 核心指标 | 建议正式名称 | 所属机制 | 在 Fig. 1 中主要对应 |
|---|---|---|---|
| **B** | Betweenness bridging centrality | 全局桥接位置 | Bridging / Reconfiguration |
| **RS** | Rao-Stirling disciplinary diversity | 距离加权跨学科多样性 | Expansion / Bridging |
| **ΔQ0** | Publication-day modularity perturbation | 社区边界即时扰动 | Reconfiguration |
| **Uzzi** | Atypical-conventional reference combination profile | 非典型组合 | Reconfiguration / Expansion |
| **RTD** | Reference Target Diversity | 参考文献目标社区多样性 | Bridging |
| **Burt IP** | Structural-hole innovation potential | 微观结构洞 | Bridging |
| **PDE** | Prospective Diffusion Entropy | 预期扩散熵 | Expansion |

---

## 二、92 个候选指标总表

说明：

- “筛选结论”中的“核心七指标之一”表示进入 Fig. 2 最终七参数体系；
- “控制变量”表示建议进入 matched controls、回归模型或 robustness analysis；
- “outcome” 表示适合作为未来影响或未来扰动验证目标，但不能作为冷启动指标；
- “候选替代”表示机制相关，但与七指标存在冗余，适合补充材料。


| 类别                                | 指标                                                                    | 来源出处                                                                   | 实际意义                                                             | 计算方式                                                                          | 筛选结论                                                      |
|:------------------------------------|:------------------------------------------------------------------------|:---------------------------------------------------------------------------|:---------------------------------------------------------------------|:----------------------------------------------------------------------------------|:--------------------------------------------------------------|
| A. 未来影响 / 结果型指标            | Raw citation count（原始被引次数）                                      | Garfield citation analysis；OpenAlex/Scopus/WoS 常规计量                   | 衡量论文发表后获得的学术注意力总量。                                 | C_t(p)=截至时间 t 对论文 p 的施引论文数。                                         | 排除：未来结果型；可作为 outcome。                            |
| A. 未来影响 / 结果型指标            | Field-normalized citation count（领域归一化引用）                       | CWTS/Scopus/WoS 归一化引用传统；Leiden 指标体系                            | 控制领域和年份引用习惯差异后的影响力。                               | FNCI=C_t(p)/E[C_t | field, year, type]。                                          | 排除：未来结果型；作为 validation/outcome。                   |
| A. 未来影响 / 结果型指标            | FWCI（Field-Weighted Citation Impact）                                  | SciVal/Scopus field-weighted impact                                        | 衡量实际引用相对于同类型、同领域、同年份期望引用的倍数。             | FWCI=actual citations / expected citations for matched field-year-document type。 | 排除：未来结果型。                                            |
| A. 未来影响 / 结果型指标            | Top citation percentile（Top 1%/10%）                                   | Scientometrics percentile impact indicators；Leiden Ranking                | 识别极高被引论文。                                                   | 在同领域同年份引用分布中计算百分位；Top x%=1 if percentile≥阈值。                 | 排除：未来结果型；可作为 landmark label。                     |
| A. 未来影响 / 结果型指标            | Citation velocity（引用速度）                                           | Citation dynamics / early citation prediction                              | 反映论文影响扩散速度。                                               | Velocity=(C_{t2}-C_{t1})/(t2-t1)。                                                | 排除：需要发表后引用。                                        |
| A. 未来影响 / 结果型指标            | Citation acceleration（引用加速度）                                     | Citation dynamics / burst-growth studies                                   | 衡量引用增长是否加速。                                               | Acceleration=(V_{t2}-V_{t1})/(t2-t1)，其中 V 为 citation velocity。               | 排除：未来结果型。                                            |
| A. 未来影响 / 结果型指标            | Citation burst strength（引用突发强度）                                 | Kleinberg burst detection；citation burst analysis                         | 识别某段时间内被引异常升高。                                         | 对年度引用序列应用 burst-state 模型或 z-score burst。                             | 排除：需要未来时间序列。                                      |
| A. 未来影响 / 结果型指标            | Citation half-life / persistence（引用半衰期/持久性）                   | Bibliometrics citation aging literature                                    | 衡量影响是否长期持续。                                               | 找到累计引用达到最终引用 50% 的时间，或拟合 aging curve。                         | 排除：未来结果型。                                            |
| A. 未来影响 / 结果型指标            | Beauty coefficient / Sleeping Beauty score（睡美人系数）                | Ke et al. 2015 Sleeping Beauty / beauty coefficient                        | 衡量论文是否经历长期沉睡后爆发。                                     | 根据引用曲线与直线基准的偏离面积计算 beauty coefficient。                         | 排除：未来结果型；用于延迟影响分析。                          |
| A. 未来影响 / 结果型指标            | Downstream diffusion entropy H（实测扩散熵）                            | Shannon entropy applied to future citing-paper fields                      | 衡量论文后续施引者来自多少领域以及是否均衡。                         | H_down(p)=-Σ_k r_k log r_k；r_k 为施引论文学科分布。                              | 排除：需要未来施引论文；PDE 的 outcome 对照。                 |
| A. 未来影响 / 结果型指标            | Disruption/CD index（颠覆性指数）                                       | Funk & Owen-Smith dynamic network measure；Wu/Wang/Evans CD family         | 衡量后续论文是否绕过焦点论文的参考文献而直接引用焦点论文。           | CD=(N_i-N_j)/(N_i+N_j+N_k)，按后续施引模式定义。                                  | 排除：需要未来引用；可作为 validation outcome。               |
| A. 未来影响 / 结果型指标            | Altmetric Attention Score（另类计量关注度）                             | Altmetrics / Altmetric.com / altmetrics manifesto                          | 衡量社交媒体、新闻、政策文件等非学术注意力。                         | 按来源加权汇总 online mentions。                                                  | 排除：非参考文献图谱指标；可作为 external attention outcome。 |
| B. 论文 / 作者 / 期刊元数据控制变量 | Team size（团队规模）                                                   | Team science / Wuchty-Jones-Uzzi；Wu-Wang-Evans                            | 团队规模可能影响产出类型、影响力和颠覆性。                           | N_authors(p)=作者数。                                                             | 控制变量：不进七指标。                                        |
| B. 论文 / 作者 / 期刊元数据控制变量 | Author prior impact（作者既往影响力）                                   | Bibliometric author impact predictors                                      | 反映作者历史声望和资源积累。                                         | mean/log prior citations of authors before publication year。                     | 控制变量：避免声望混淆。                                      |
| B. 论文 / 作者 / 期刊元数据控制变量 | Author h-index / academic age（作者 h 指数/学术年龄）                   | Hirsch h-index；career age bibliometrics                                   | 刻画作者资历和既往影响力。                                           | h-index before t0；academic age=t0-first publication year。                       | 控制变量。                                                    |
| B. 论文 / 作者 / 期刊元数据控制变量 | Author-field diversity（作者学科多样性）                                | Team diversity / interdisciplinarity literature                            | 作者背景越多样，可能带来跨域组合。                                   | 作者历史论文领域分布的 Shannon/Simpson/RS。                                       | 控制变量或补充；非 reference-only。                           |
| B. 论文 / 作者 / 期刊元数据控制变量 | Institution prestige（机构声望）                                        | Research evaluation / institutional ranking metrics                        | 机构资源和声望影响引用与传播。                                       | 机构历史 citation percentile、ranking 或 field-normalized output。                | 控制变量。                                                    |
| B. 论文 / 作者 / 期刊元数据控制变量 | Institutional diversity（机构多样性）                                   | Collaboration diversity literature                                         | 多机构合作可能带来知识来源多样化。                                   | unique institutions 或机构国家/领域分布熵。                                       | 控制变量。                                                    |
| B. 论文 / 作者 / 期刊元数据控制变量 | International collaboration（国际合作）                                 | Science collaboration metrics                                              | 跨国合作可能提高可见度和扩散范围。                                   | 是否跨国；或 unique countries；或 country entropy。                               | 控制变量。                                                    |
| B. 论文 / 作者 / 期刊元数据控制变量 | Funding diversity（资助来源多样性）                                     | Funding acknowledgement analysis                                           | 多资助来源可能对应跨目标或跨机构研究。                               | funding agencies count / entropy。                                                | 控制变量；数据缺失较多。                                      |
| B. 论文 / 作者 / 期刊元数据控制变量 | Journal impact factor / venue prestige（期刊声望）                      | Journal Impact Factor / venue metrics                                      | 发表渠道影响论文被看见和被引。                                       | JIF, SJR, CiteScore, venue mean citation percentile。                             | 控制变量；避免以刊评文。                                      |
| B. 论文 / 作者 / 期刊元数据控制变量 | Venue topical scope（期刊跨学科范围）                                   | Journal interdisciplinarity indicators                                     | 综合期刊更容易跨领域扩散。                                           | venue subject-category entropy / RS / scope count。                               | 控制变量；不进七指标。                                        |
| B. 论文 / 作者 / 期刊元数据控制变量 | Open access status（开放获取状态）                                      | Open access citation advantage literature；OpenAlex OA fields              | OA 可能提高可访问性和引用。                                          | is_oa, oa_status, host venue OA route。                                           | 控制变量。                                                    |
| B. 论文 / 作者 / 期刊元数据控制变量 | Document type / article length（文献类型与文本长度）                    | Document-type normalization in bibliometrics                               | 综述、论文、预印本等类型和篇幅影响引用。                             | type one-hot；title/abstract/全文长度。                                           | 控制变量。                                                    |
| C. 参考文献年龄 / 深度 / 流行度     | Reference count（参考文献数量）                                         | Bibliometrics reference list controls                                      | 参考文献越多，组合空间越大，也可能提高多样性指标。                   | N_ref(p)=|R_p|。                                                                  | 控制变量。                                                    |
| C. 参考文献年龄 / 深度 / 流行度     | Mean reference age（平均参考文献年龄）                                  | Reference aging / disruption literature                                    | 衡量知识基础的新旧程度。                                             | mean(t0-year(r)) for r in R_p。                                                   | 控制变量；不进七指标。                                        |
| C. 参考文献年龄 / 深度 / 流行度     | Median reference age（参考文献年龄中位数）                              | Reference age measures                                                     | 比均值更稳健地描述引用知识年龄。                                     | median(t0-year(r))。                                                              | 控制变量。                                                    |
| C. 参考文献年龄 / 深度 / 流行度     | Reference age variance（参考文献年龄方差）                              | Reference age distribution measures                                        | 衡量是否同时引用新旧知识。                                           | Var(t0-year(r))。                                                                 | 控制变量。                                                    |
| C. 参考文献年龄 / 深度 / 流行度     | Recent-reference share（近期参考文献比例）                              | Reference recency indicators                                               | 衡量论文是否扎根于前沿文献。                                         | share{r: t0-year(r)≤k}，常取 k=3 或 5。                                           | 控制变量。                                                    |
| C. 参考文献年龄 / 深度 / 流行度     | Classic-reference share（经典参考文献比例）                             | Citation classics / reference age                                          | 衡量论文是否调用长期基础知识。                                       | share{r: age≥k 或 prior citations in top x%}。                                    | 控制变量。                                                    |
| C. 参考文献年龄 / 深度 / 流行度     | Mean prior citation of references（参考文献平均先验被引）               | Reference popularity / disruption studies                                  | 参考文献越热门，组合越传统；越冷门，可能更探索。                     | mean(log(1+C_{t0-1}(r)))。                                                        | 控制变量。                                                    |
| C. 参考文献年龄 / 深度 / 流行度     | Top-reference prior citation（最强参考文献先验影响）                    | Reference popularity controls                                              | 衡量是否依赖少数权威基础论文。                                       | max or top-decile mean of prior citations among R_p。                             | 控制变量。                                                    |
| C. 参考文献年龄 / 深度 / 流行度     | Obscure-reference share（冷门参考文献比例）                             | Disruptive science references to less prevalent ideas                      | 衡量是否引用较少被关注的知识。                                       | share{r: prior citation percentile≤x%}。                                          | 控制变量/补充。                                               |
| C. 参考文献年龄 / 深度 / 流行度     | Self-citation share（自引比例）                                         | Citation analysis / self-citation controls                                 | 高自引可能影响图结构与引用信号。                                     | share of references sharing author/institution with p。                           | 控制变量。                                                    |
| D. 非典型组合 / 组合新颖性          | ⭐ Uzzi atypical-conventional combination profile（Uzzi 非典型组合）    | Uzzi et al. 2013 Science                                                   | 识别参考文献期刊组合是否包含少量高度非典型组合，同时整体保持传统性。 | 对参考文献期刊对计算 z=(O-μ)/σ；Novelty_tail=-p10(z)；Conventionality=median(z)。 | 核心七指标之一：保留。                                        |
| D. 非典型组合 / 组合新颖性          | Journal-pair PMI / surprisal（期刊对互信息/惊奇度）                     | Information-theoretic association measures；novel recombination            | 衡量两个期刊在参考文献中共同出现是否罕见。                           | PMI(j1,j2)=log P(j1,j2)/(P(j1)P(j2))；Surprisal=-log P(j1,j2)。                   | 候选替代；被 Uzzi profile 覆盖。                              |
| D. 非典型组合 / 组合新颖性          | Reference-pair co-citation z-score（参考文献对共引 z 分）               | Co-citation analysis；Uzzi-style null models                               | 衡量两篇被引文献组合是否比随机预期罕见。                             | 对参考文献对的共引次数与随机置换 μ,σ 比较。                                       | 候选替代。                                                    |
| D. 非典型组合 / 组合新颖性          | Co-citation rarity average（平均共引稀有度）                            | Co-citation novelty metrics                                                | 参考文献之间越少被共同引用，组合越新颖。                             | mean or lower-tail of -log co-citation probability over all reference pairs。     | 候选替代。                                                    |
| D. 非典型组合 / 组合新颖性          | Topic-pair PMI（主题对非典型性）                                        | Topic-combination novelty literature                                       | 衡量参考文献主题组合是否罕见。                                       | 以 OpenAlex topics/MeSH/concepts 为类别，计算 topic-pair PMI/z-score。            | 候选替代；数据依赖 topic quality。                            |
| D. 非典型组合 / 组合新颖性          | Concept co-occurrence surprise（概念共现惊奇度）                        | Text/knowledge-combination novelty                                         | 衡量标题/摘要/关键词概念组合的罕见性。                               | Surprise=-log P(concept_i,concept_j) 或 observed-vs-expected z。                  | 排除主七：依赖文本/概念抽取。                                 |
| D. 非典型组合 / 组合新颖性          | Keyword-pair novelty（关键词对新颖性）                                  | Keyword co-occurrence novelty metrics                                      | 捕捉论文关键词组合是否少见。                                         | 对关键词对计算 frequency, PMI 或 z-score。                                        | 候选替代；数据不稳定。                                        |
| D. 非典型组合 / 组合新颖性          | Lee-Walsh-Wang U score（组合新颖性 U）                                  | Lee/Walsh/Wang novelty validation literature                               | 文献中用于评价参考组合新颖性的指标族之一。                           | 基于参考组合的 atypicality/rarity 统计，按原文定义计算。                          | 候选替代；不如 Uzzi 解释直观。                                |
| D. 非典型组合 / 组合新颖性          | Wang novelty W score（Wang 新颖性）                                     | Wang et al. novelty score literature                                       | 基于新参考组合或新知识组合定义论文新颖性。                           | 按文献定义识别首次/罕见组合并计算 W。                                             | 候选替代；放 robustness。                                     |
| D. 非典型组合 / 组合新颖性          | Knowledge eccentricity（知识偏心度）                                    | Knowledge-space outlier metrics                                            | 衡量论文参考组合在知识空间中是否偏离常规中心。                       | distance(reference vector, field centroid) 或 local outlier factor。              | 候选替代；部分依赖 embedding。                                |
| D. 非典型组合 / 组合新颖性          | Unusual dataset/method combination（数据/方法非典型组合）               | Research novelty / method recombination                                    | 衡量是否把少见数据、方法和问题结合。                                 | 从文本/元数据抽取 dataset/method labels，再算组合 rarity。                        | 排除主七：抽取难度大。                                        |
| E. 跨学科多样性 / 知识分布          | Disciplinary variety（学科丰富度）                                      | Stirling diversity framework                                               | 参考文献涉及的学科数量。                                             | Variety=|{k: q_k>0}|。                                                            | 候选；被 RS/PDE 覆盖。                                        |
| E. 跨学科多样性 / 知识分布          | Balance/evenness（学科均衡性）                                          | Stirling diversity framework；Pielou evenness                              | 参考文献在不同学科间是否均匀。                                       | Evenness=H/log K 或 1-Gini(q)。                                                   | 候选；PDE 直接体现。                                          |
| E. 跨学科多样性 / 知识分布          | Gini / HHI disciplinary concentration（学科集中度）                     | Gini coefficient；Herfindahl-Hirschman/Simpson                             | 衡量参考文献是否集中在少数学科。                                     | HHI=Σ_k q_k^2；Gini over q_k。                                                    | 候选；作为控制/补充。                                         |
| E. 跨学科多样性 / 知识分布          | Simpson / Gini-Simpson discipline diversity（学科 Simpson 多样性）      | Simpson 1949；Gini-Simpson/Blau diversity                                  | 随机抽两篇参考文献来自不同学科的概率。                               | 1-Σ_k q_k^2 或有限样本 1-Σ n_k(n_k-1)/N(N-1)。                                    | 候选；PDE/RS 的补充。                                         |
| E. 跨学科多样性 / 知识分布          | Generic Shannon entropy of disciplines（学科 Shannon 熵）               | Shannon 1948 entropy applied to field distribution                         | 衡量参考文献学科分布的不确定性和均匀性。                             | H=-Σ_k q_k log q_k。                                                              | 候选；被 PDE 具体化。                                         |
| E. 跨学科多样性 / 知识分布          | ⭐ PDE: Prospective Diffusion Entropy（预期扩散熵）                     | 本文定义的派生指标；数学基础为 Shannon entropy                             | 估计论文发表当天可见的潜在跨学科扩散广度。                           | PDE=-Σ_k q_k log2 q_k；PDE_norm=PDE/log2(K)。                                     | 核心七指标之一：保留；声明原创派生。                          |
| E. 跨学科多样性 / 知识分布          | ⭐ RS: Rao-Stirling disciplinary diversity（Rao-Stirling 跨学科多样性） | Rao 1982；Stirling 2007；Leydesdorff/Rafols interdisciplinarity indicators | 同时考虑学科数量、均衡性和学科间距离。                               | RS=Σ_{i≠j} d_ij p_i p_j，d_ij=1-cos(c_i,c_j)。                                    | 核心七指标之一：保留。                                        |
| E. 跨学科多样性 / 知识分布          | Average field cosine distance（平均学科距离）                           | Science-map distance / cosine-normalized journal-category maps             | 衡量参考文献学科之间平均相距多远。                                   | mean_{i<j}(1-cos(c_i,c_j)) over reference field pairs。                           | 候选；RS 已加权整合。                                         |
| E. 跨学科多样性 / 知识分布          | Interdisciplinarity ratio（跨学科引用比例）                             | Interdisciplinarity / integration indicators                               | 参考文献中来自焦点论文主领域之外的比例。                             | 1 - share of references in focal field。                                          | 候选；作为简单解释指标。                                      |
| E. 跨学科多样性 / 知识分布          | Integration score / Porter interdisciplinarity（整合指数）              | Porter/Rafols interdisciplinarity framework                                | 衡量论文整合多个学科类别的程度。                                     | 常基于 diversity, disparity, balance 的组合函数。                                 | 候选；与 RS 重叠。                                            |
| E. 跨学科多样性 / 知识分布          | DIV / relative variety（DIV/相对多样性）                                | Leydesdorff/Wagner/Bornmann DIV indicator                                  | 改进 RS，把 variety、balance、disparity 分开处理。                   | 按 DIV 文献将三要素分别归一化再组合。                                             | 候选替代；可在 robustness 中比较 RS。                         |
| E. 跨学科多样性 / 知识分布          | Science-map dispersion radius（科学地图扩散半径）                       | Science overlay maps / interdisciplinarity visualization                   | 参考文献在全局科学地图上的空间分散程度。                             | weighted average distance from reference-field centroid。                         | 候选替代；需要科学地图。                                      |
| F. 图拓扑中心性 / 结构洞            | ⭐ B: Betweenness bridging centrality（桥接中心性）                     | Freeman 1977 betweenness centrality                                        | 衡量论文是否位于多个知识社区之间的最短路径上。                       | B(v)=Σ_{s≠v≠t} σ_st(v)/σ_st；B_norm=2B/[(n-1)(n-2)]。                             | 核心七指标之一：保留。                                        |
| F. 图拓扑中心性 / 结构洞            | Degree centrality（度中心性）                                           | Classic network centrality                                                 | 直接连接数量；反映局部连接广度。                                     | deg(v)=|N(v)|；可归一化为 deg/(n-1)。                                             | 候选；过于局部，被 B/RTD/IP 替代。                            |
| F. 图拓扑中心性 / 结构洞            | Weighted degree / strength（加权度/强度）                               | Weighted network analysis                                                  | 考虑边权后的连接总强度。                                             | s(v)=Σ_u w_{vu}。                                                                 | 候选；作为控制。                                              |
| F. 图拓扑中心性 / 结构洞            | Incident edge betweenness（关联边中介性）                               | Girvan-Newman edge betweenness                                             | 焦点论文连接边是否是社区之间关键通道。                               | mean/max edge betweenness over edges incident to p。                              | 候选；被 B 和 ΔQ0 部分覆盖。                                  |
| F. 图拓扑中心性 / 结构洞            | Closeness centrality（接近中心性）                                      | Classic network centrality                                                 | 论文到其他节点的平均距离是否短。                                     | C(v)=(n-1)/Σ_u d(v,u)。                                                           | 候选；引用图可能不连通。                                      |
| F. 图拓扑中心性 / 结构洞            | Harmonic centrality（调和中心性）                                       | Network centrality for disconnected graphs                                 | 适用于不连通网络的接近程度。                                         | H(v)=Σ_{u≠v} 1/d(v,u)。                                                           | 候选。                                                        |
| F. 图拓扑中心性 / 结构洞            | Current-flow / random-walk betweenness（随机游走中介性）                | Newman current-flow betweenness / random-walk centrality                   | 衡量不只走最短路时的桥接作用。                                       | 基于电流流量或随机游走经过 v 的期望次数。                                         | 候选；计算成本高。                                            |
| F. 图拓扑中心性 / 结构洞            | Eigenvector centrality（特征向量中心性）                                | Bonacich/eigenvector centrality                                            | 连接到重要节点的节点更重要。                                         | x=λ^{-1}Ax。                                                                      | 候选；更偏影响力而非创新桥接。                                |
| F. 图拓扑中心性 / 结构洞            | PageRank（网页/引文 PageRank）                                          | Page & Brin 1998；citation-inspired ranking                                | 考虑被重要节点指向的递归重要性。                                     | PR(v)=αΣ_{u→v}PR(u)/out(u)+(1-α)/n。                                              | 候选；更像影响力/声望。                                       |
| F. 图拓扑中心性 / 结构洞            | Katz / alpha centrality（Katz/α 中心性）                                | Katz 1953                                                                  | 考虑所有长度路径，长路径按 α 衰减。                                  | x=(I-αA^T)^{-1}e。                                                                | 候选；与 PageRank/eigenvector 重叠。                          |
| F. 图拓扑中心性 / 结构洞            | HITS hub/authority（HITS 枢纽/权威分数）                                | Kleinberg HITS 1999                                                        | 区分引用很多权威的 hub 与被 hub 指向的 authority。                   | a=A^T h；h=Aa，迭代求解。                                                         | 候选；对 citation graph 解释需谨慎。                          |
| F. 图拓扑中心性 / 结构洞            | Participation coefficient（参与系数）                                   | Guimerà & Amaral functional cartography                                    | 节点连接是否分散到多个模块。                                         | P_i=1-Σ_s(k_is/k_i)^2。                                                           | 候选；与 RTD/B 重叠，放 robustness。                          |
| F. 图拓扑中心性 / 结构洞            | Within-module z-score（模块内 z 分数）                                  | Guimerà & Amaral node roles                                                | 节点在本社区内部是否是局部 hub。                                     | z_i=(k_i^s-mean(k^s))/sd(k^s)。                                                   | 候选；用于角色分类。                                          |
| F. 图拓扑中心性 / 结构洞            | Bridging coefficient（桥接系数）                                        | Bridging centrality literature                                             | 衡量节点是否连接高度数节点之间的低冗余邻域。                         | 通常结合 node degree 与邻居度：BCoeff(v) ∝ 1/deg(v) / Σ 1/deg(neighbor)。         | 候选；公式版本多。                                            |
| F. 图拓扑中心性 / 结构洞            | Local clustering coefficient（局部聚类系数）                            | Watts-Strogatz small-world networks                                        | 邻居之间是否形成闭合三角形；低值可能代表结构洞。                     | C_v=2T_v/[k_v(k_v-1)]。                                                           | 候选；由 Burt IP 更系统表示。                                 |
| F. 图拓扑中心性 / 结构洞            | k-core / coreness（k 核位置）                                           | Network core-periphery / k-shell decomposition                             | 衡量节点处在网络核心还是边缘。                                       | 最大 k 使 v 属于 k-core。                                                         | 候选；偏核心性。                                              |
| F. 图拓扑中心性 / 结构洞            | ⭐ Burt structural-hole innovation potential IP（Burt 结构洞潜力）      | Burt 1992 Structural Holes；constraint/effective size                      | 参考文献彼此越少连接，论文越可能填补结构洞。                         | C=Σ_i(p_i+Σ_j p_ij p_j)^2；IP=1-C_norm。                                          | 核心七指标之一：保留。                                        |
| F. 图拓扑中心性 / 结构洞            | Effective size / efficiency（有效规模/效率）                            | Burt structural holes                                                      | 邻居之间越不冗余，有效规模越大。                                     | EffectiveSize=Σ_j[1-Σ_q p_iq m_jq]；Efficiency=EffectiveSize/degree。             | 候选；与 Burt IP 同簇。                                       |
| G. 社区重构 / 模块结构              | ⭐ ΔQ0: Publication-day modularity perturbation（发表日模块度扰动）     | Newman & Girvan modularity                                                 | 论文 p 加入发表前图谱后，社区边界是否被打破或强化。                  | ΔQ0=Q(G0)-Q(G−)，G0=G−+p+reference edges。                                        | 核心七指标之一：保留。                                        |
| G. 社区重构 / 模块结构              | Q(G−) prior modularity（先验模块度）                                    | Newman-Girvan modularity                                                   | 领域发表前社区分割强度。                                             | Q=(1/2m)Σ_ij[A_ij-k_i k_j/(2m)]δ(c_i,c_j)。                                       | 候选；作为 ΔQ0 的基线。                                       |
| G. 社区重构 / 模块结构              | Q(G0) augmented modularity（发表日增强图模块度）                        | Newman-Girvan modularity                                                   | 加入论文和参考文献边后的社区结构强度。                               | 对 G0 计算 Q。                                                                    | 候选；与 ΔQ0 配套。                                           |
| G. 社区重构 / 模块结构              | ΔQτ realized modularity shift（未来模块度变化）                         | Modularity change over time                                                | 未来图谱是否发生实际重构或巩固。                                     | ΔQτ=Q(G+τ)-Q(G−)。                                                                | 排除七指标：future realized phenotype。                       |
| G. 社区重构 / 模块结构              | ARI partition change（社区划分 ARI 变化）                               | Adjusted Rand Index for cluster similarity                                 | 衡量社区划分前后是否重排。                                           | 1-ARI(C_t,C_{t-1}) 或 1-ARI(C−,C0)。                                              | 候选；更适合 realized reconfiguration。                       |
| G. 社区重构 / 模块结构              | NMI / VI partition distance（NMI/VI 划分距离）                          | Information-theoretic clustering comparison                                | 从信息论角度衡量社区划分差异。                                       | VI(C1,C2)=H(C1)+H(C2)-2I(C1,C2)；或 1-NMI。                                       | 候选；放 Extended Data。                                      |
| G. 社区重构 / 模块结构              | Community turnover / Jaccard change（社区成员周转）                     | Dynamic community analysis                                                 | 社区成员组成是否改变。                                               | 1-Jaccard(V_c^t,V_c^{t-1}) averaged over communities。                            | 候选；需要时间序列。                                          |
| G. 社区重构 / 模块结构              | Conductance / cut ratio（导通率/割边比例）                              | Graph partition quality metrics                                            | 衡量社区边界是否容易被跨越。                                         | conductance(S)=cut(S,¬S)/min(vol(S),vol(¬S))。                                    | 候选；可补充 ΔQ0。                                            |
| G. 社区重构 / 模块结构              | Topic assortativity（主题同配性）                                       | Network assortativity / mixing patterns                                    | 同主题之间是否更倾向连接；低值代表跨主题混合。                       | assortativity coefficient by topic/community labels。                             | 候选。                                                        |
| G. 社区重构 / 模块结构              | Topic average path length / path shortening（主题路径长度/缩短）        | Network distance / compression phenotype                                   | 衡量知识模块间路径是否被缩短。                                       | L=average shortest path in topic graph；Shortening=max(0,L_prev-L_now)。          | 候选；用于 Fig.1 Compression outcome。                        |
| H. 语义 / 文本 / embedding novelty  | Embedding distance to prior centroid（到先验领域中心的嵌入距离）        | Semantic novelty / document embedding literature                           | 论文文本是否远离既有领域语义中心。                                   | 1-cos(embedding(p), centroid(field before t0))。                                  | 排除主七：非 reference-only。                                 |
| H. 语义 / 文本 / embedding novelty  | Local semantic outlier score（局部语义异常值）                          | Outlier detection / semantic novelty                                       | 论文是否是局部语义空间异常点。                                       | LOF 或 distance to k-nearest prior papers。                                       | 排除主七。                                                    |
| H. 语义 / 文本 / embedding novelty  | Semantic dispersion of references（参考文献语义离散度）                 | Semantic diversity metrics                                                 | 参考文献文本语义是否分散。                                           | mean pairwise cosine distance among reference embeddings。                        | 候选；需文本/embedding。                                      |
| H. 语义 / 文本 / embedding novelty  | Topic-model surprise / perplexity（主题模型惊奇度）                     | Topic modeling novelty / LDA                                               | 论文主题组合在既有主题模型下是否意外。                               | -log P(document | prior topic model) 或 perplexity。                              | 排除主七。                                                    |
| H. 语义 / 文本 / embedding novelty  | KL / JS divergence from field distribution（与领域主题分布差异）        | Information divergence measures                                            | 论文主题分布是否偏离领域常态。                                       | D_KL(P_p||P_field) 或 JS(P_p,P_field)。                                           | 排除主七。                                                    |
| H. 语义 / 文本 / embedding novelty  | New keyword / concept rate（新关键词/新概念率）                         | Keyword novelty / concept emergence                                        | 是否引入领域此前少见或未见概念。                                     | share of keywords/concepts unseen or rare before t0。                             | 候选；数据质量依赖强。                                        |
| H. 语义 / 文本 / embedding novelty  | LLM-assessed novelty/originality（LLM 新颖性评分）                      | LLM-assisted research evaluation                                           | 用语言模型判断原创性、重要性或非典型组合。                           | prompt-based score 或 embedding/rubric 评分。                                     | 排除主七：模型版本依赖，后续 Fig.8 可用。                     |

---

## 三、Fig. 2 Panel b 五步筛选标准

建议 Panel b 画成一个漏斗，数字变化为：

```text
92 → 67 → 49 → 29 → 12 → 7
```

### Step 0. Candidate metric universe：92 个候选指标

候选指标来自八类来源：

1. bibliometrics / citation impact；
2. author, venue and metadata controls；
3. reference-age and reference-depth indicators；
4. atypical combination / recombination novelty；
5. interdisciplinarity and diversity measures；
6. network centrality and structural holes；
7. community reconfiguration metrics；
8. semantic / text / embedding novelty。

图上可写：

```text
Candidate metric universe
n = 92
Bibliometrics, network science, diversity, novelty, information theory, semantic metrics
```

---

### Step 1. Publication-day observable / no future leakage：92 → 67

**保留标准：** 指标在论文发表当天，或仅使用发表前知识图谱 `G−` 加目标论文参考文献即可计算。

**排除对象：**

- raw future citation count；
- field-normalized future citation；
- FWCI；
- citation velocity / burst；
- sleeping beauty / beauty coefficient；
- downstream diffusion entropy H；
- disruption / CD index；
- Altmetric score。

**理由：**这些指标适合作为 validation outcome，但它们依赖论文发表后的施引行为，会导致 future leakage。

---

### Step 2. Reference-only / reference-graph computable：67 → 49

**保留标准：** 指标只依赖：

- 目标论文参考文献；
- 参考文献的学科、期刊、主题、年份、先验引用；
- 发表前引用图谱或知识图谱。

**排除对象：**

- 作者声望；
- 机构声望；
- 团队规模；
- 期刊影响因子；
- open access status；
- funding metadata；
- full text / LLM-only 指标。

**理由：**本文核心卖点是 **publication-day, reference-only, graph-grounded**。

---

### Step 3. Mechanistically linked to graph perturbation：49 → 29

**保留标准：** 指标必须能够解释 Fig. 1 的至少一种扰动表型：

| 扰动表型 | 机制问题 | 相关指标簇 |
|---|---|---|
| Expansion | 是否把知识输入扩展到多个学科或领域？ | RS, PDE, Uzzi |
| Bridging | 是否连接原本分离的社区或结构洞？ | B, RTD, Burt IP |
| Reconfiguration | 是否改变社区边界或知识组合规则？ | ΔQ0, Uzzi, B, RTD |
| Compression | 后续是否形成收敛范式？ | realized ΔQτ, path shortening, RS/RTD/PDE decline |

**排除对象：**

- reference count；
- mean reference age；
- prior citation of references；
- self-citation share；
- generic metadata controls。

这些变量有用，但不是扰动机制本身。

---

### Step 4. Non-redundant mechanism clusters：29 → 12

对剩余指标做机制聚类和相关性分析，形成 7 个机制簇：

| 机制簇 | 候选代表 |
|---|---|
| 全局桥梁位置 | B, closeness, PageRank, participation coefficient |
| 距离加权跨学科多样性 | RS, DIV, average field distance |
| 社区边界扰动 | ΔQ0, conductance change, ARI change |
| 非典型组合 | Uzzi, Wang W, Lee-Walsh-Wang U |
| 参考目标社区多样性 | RTD, participation coefficient, Simpson over communities |
| 微观结构洞 | Burt constraint/IP, effective size, local clustering |
| 潜在扩散广度 | PDE, Shannon entropy, Simpson over fields |

这一步保留大约 12 个候选代表，供最终筛选：

```text
B, participation coefficient, RS, DIV, ΔQ0, Uzzi, Wang W,
RTD, Burt IP, effective size, PDE, reference-age depth
```

---

### Step 5. Interpretable, robust, validation-ready basis：12 → 7

最终选择：

```text
B, RS, ΔQ0, Uzzi, RTD, Burt IP, PDE
```

选择理由：

| 选择标准 | 七指标如何满足 |
|---|---|
| 冷启动 | 均可在发表当天由参考文献和 G− / G0 计算 |
| reference-only | 不依赖作者声望、期刊声望或未来引用 |
| 机制互补 | 覆盖桥接、跨学科、社区扰动、非典型组合、结构洞和潜在扩散 |
| 非冗余 | 分别处于不同机制簇 |
| 可解释 | 每个指标均有清晰图谱或信息论含义 |
| 可验证 | 可预测未来 G+τ 中的 expansion、bridging、reconfiguration、compression |

Panel b 图上可以写：

```text
Final seven-parameter basis
B | RS | ΔQ0 | Uzzi | RTD | IP | PDE
```

---

## 四、两个本文派生指标：RTD 与 PDE 的正式表述

### 4.1 RTD：Reference Target Diversity

**推荐表述：**

> RTD and PDE are newly defined reference-only indicators in this study. Their mathematical forms are grounded in established diversity and information-theoretic measures.

中文：

> RTD 和 PDE 是本文定义的仅依赖参考文献的冷启动指标，其数学形式分别建立在 Gini-Simpson 多样性和 Shannon 信息熵之上。

**定义：**

设 `n_c` 为论文 `p` 的参考文献中属于图社区 `c` 的数量，`N` 为具有社区标签的参考文献总数：

```math
RTD(p)=1-\sum_c \frac{n_c(n_c-1)}{N(N-1)}
```

**解释：**

RTD 是参考文献目标社区标签上的 **Gini-Simpson diversity / Blau diversity**。它可以解释为：

> 从目标论文参考文献中随机抽取两篇文献时，二者来自不同图社区的概率。

因此，RTD 高说明论文不是只继承单一知识社区，而是在连接多个知识岛。它对应 Fig. 1 中的 **Bridging** 表型。

**与 participation coefficient 的关系：**

- participation coefficient 看的是节点的邻居社区分布；
- RTD 看的是论文参考文献的社区分布。

所以 RTD 可视为：

```text
reference-side participation diversity
```

或：

```text
cited-reference community diversity
```

---

### 4.2 PDE：Prospective Diffusion Entropy

**定义：**

设 `q_k` 为论文 `p` 的参考文献中属于学科 `k` 的比例，`K` 为出现的学科数量：

```math
PDE(p)=-\sum_k q_k \log_2(q_k)
```

归一化：

```math
PDE_{norm}(p)=\frac{PDE(p)}{\log_2 K}
```

**解释：**

PDE 是 **Shannon entropy** 在论文参考文献学科分布上的应用。它不是已经发生的扩散，而是估计：

> 发表当天可见的潜在跨学科扩散范围。

如果一篇论文的参考文献来自多个学科且分布均匀，则它更可能被多个领域的研究者理解、引用和扩散。

**与 downstream diffusion entropy H 的区别：**

| 指标 | 数据来源 | 时间属性 | 含义 |
|---|---|---|---|
| H | 未来施引论文的学科分布 | 发表后 | 实际扩散 |
| PDE | 目标论文参考文献的学科分布 | 发表当天 | 预期扩散潜力 |

---

### 4.3 RTD / PDE / RS 的区别

| 指标 | 数学基础 | 分类对象 | 是否考虑类别距离 | 解释 |
|---|---|---|---|---|
| **RS** | Rao-Stirling | 学科 | 是，考虑 `d_ij` | 你跨了多远的学科距离 |
| **PDE** | Shannon entropy | 学科 | 否，只看分布均匀性 | 你覆盖了多少且多均匀的学科 |
| **RTD** | Gini-Simpson / Blau diversity | 图社区 | 否，但社区来自图拓扑 | 你连接了多少且多均匀的图社区 |

---

## 五、建议写进论文 Methods 的英文段落

```text
We first constructed a dictionary of 92 candidate metrics from bibliometrics, network science, interdisciplinarity studies, novelty measurement, structural-hole theory and information theory. We then removed metrics that require post-publication citations or non-reference information, retained metrics computable from the publication-day augmented graph G0, grouped the remaining candidates into non-redundant mechanistic clusters, and selected seven interpretable metrics that jointly cover expansion, bridging, reconfiguration and compression phenotypes.
```

```text
Reference Target Diversity (RTD) and Prospective Diffusion Entropy (PDE) are reference-only cold-start metrics introduced in this study. RTD adapts Gini-Simpson diversity to the graph-community labels of cited references, whereas PDE adapts Shannon entropy to the disciplinary distribution of cited references. Their novelty lies not in the diversity formula itself, but in moving established diversity and entropy measures to the publication-day reference side as proxies for future knowledge-graph perturbation potential.
```

---

## 六、参考文献与来源线索

以下为本表中主要指标的来源或理论基础。正式论文中建议替换为 DOI / 原文格式。

1. Garfield, E. Citation indexing and citation analysis.
2. Freeman, L. C. (1977). A set of measures of centrality based on betweenness.
3. Rao, C. R. (1982). Diversity and dissimilarity coefficients.
4. Stirling, A. (2007). A general framework for analysing diversity in science, technology and society.
5. Newman, M. E. J. & Girvan, M. (2004). Finding and evaluating community structure in networks.
6. Uzzi, B. et al. (2013). Atypical combinations and scientific impact.
7. Burt, R. S. (1992). Structural Holes: The Social Structure of Competition.
8. Shannon, C. E. (1948). A Mathematical Theory of Communication.
9. Simpson, E. H. (1949). Measurement of diversity.
10. Guimerà, R. & Amaral, L. A. N. (2005). Functional cartography of complex metabolic networks.
11. Funk, R. J. & Owen-Smith, J. Dynamic network measures of technological change.
12. Ke, Q. et al. (2015). Defining and identifying sleeping beauties in science.
13. Leydesdorff, L. & Rafols, I. Indicators of interdisciplinarity.
14. Leydesdorff, L., Wagner, C. S. & Bornmann, L. Rao-Stirling Diversity, Relative Variety, and the Gini coefficient.
15. Page, L. et al. PageRank.
16. Katz, L. (1953). A new status index derived from sociometric analysis.
17. Kleinberg, J. HITS and burst detection.
18. Hicks, D. et al. The Leiden Manifesto for research metrics.
19. OpenAlex documentation and OpenAlex work-object metadata.
20. User-supplied RTD/PDE methodological note in the present project.

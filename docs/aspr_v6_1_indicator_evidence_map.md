# ASPR v6.1 五角度指标证据地图、数据、模型与全时期 OOF 结果

> 最终协议：`aspr-nature-v6.1-simple-fixed-medium-r5-2026-07-24`  
> 最终指标注册表：`sha256:126202f72519a4eb1657f49b8daa06bdcd9fccb37fe66cd6e4a8ec8471762611`  
> 结果制品：`sha256:1ae922413b008225c42061e7de50eeee10d20ae14233c637b3f3975da5a23cc0`  
> 数据审计：**ready_to_model**  
> 定位：多源系统性范围证据地图；不是穷尽互联网的系统综述，也不是元分析。

## 1. 先给结论

本体系把两个问题严格分开：五个观察角度及其指标描述论文发表时的知识重组证据；D3/D5/D8 标签描述发表后的学术传播与跨领域扩散。未来引用没有参与指标筛选，OOF 也没有决定某个创新指标的去留。

五个角度不是“学界公认的五种互斥创新类型”，而是由组合新颖性、科学计量学和多样性/整合研究支持的五个下位观察维度。它们不能替代同行评审、实验正确性、社会影响或 Nature 录用判断。

最终主创新指标由统一的来源、时间、可计算性、覆盖、稳定性、公式忠实度和非冗余规则产生；不是先规定必须保留八个，也不按 OOF 高低删指标。

## 2. 系统检索与筛选流程

```mermaid
flowchart LR
  A["多源检索<br/>Crossref / OpenAlex / PubMed-PMC / 出版社 / 学术网页"]
  B["DOI与规范化题名去重<br/>逐条记录检索式、日期和决定"]
  C["核心论文、综述和软件目录<br/>前向/后向引文追踪"]
  D["连续两轮未发现<br/>新数学指标家族"]
  E["候选目录<br/>50项、34个来源"]
  F["I1-I10结果盲筛选<br/>来源·时间·本地数据·覆盖·稳定性·忠实度"]
  G["冻结主指标<br/>8项、覆盖五个角度"]
  H["冻结注册表哈希后<br/>才读取标签并运行OOF"]
  A --> B --> C --> D --> E --> F --> G --> H
```

检索截止 **2026-07-24**，共有 **14** 条可审计检索/核验记录。Google Scholar 自动访问失败已原样登记，未杜撰结果数；综述、预印本和 Novelpy 只用于发现候选，主指标证据回到同行评议公式来源和论文级应用。范围声明是“多源系统性范围证据地图”，不宣称穷尽互联网。

## 3. 五个观察角度：意义、来源与最终指标

| 角度 | 观察意义 | 角度来源 | 最终主指标 | 纳入原则 | 排除原则 |
|---|---|---|---|---|---|
| 组合稀有性 | The observed rarity or surprise of co-cited knowledge-source pairs relative to their strictly prior marginal prevalence. | [LEE2015](https://doi.org/10.1016/j.respol.2014.10.007), [BORNMANN2019](https://doi.org/10.1016/j.joi.2019.100979), [FONTANA2020](https://doi.org/10.1016/j.respol.2020.104063), [MATSUMOTO2021](https://doi.org/10.1007/s11192-021-04049-z) | A1.REFERENCE_OVERLAP | Include a paper-level rarity statistic with an explicit observed/expected or prior-overlap formula, T0 history, and independent application or validation. | Exclude threshold-only variants, monotone transforms, and any similarity reference set that is not frozen before publication. |
| 非典型性与常规性 | The unusual left tail and conventional centre of the focal paper's distribution of source-pair null-model z scores. | [UZZI2013](https://doi.org/10.1126/science.1240474), [TEPLITSKIY2022](https://doi.org/10.1073/pnas.2118046119), [FONTANA2020](https://doi.org/10.1016/j.respol.2020.104063), [YAN2016](https://doi.org/10.1140/epjds/s13688-016-0069-1), [ZHOU2022](https://doi.org/10.1016/j.jbi.2022.104047) | A2.HYPERGEOM_MEDIAN | Include the source-defined P10 novelty and median conventionality facets when the frozen null approximation matches its exact reference implementation. | Exclude arbitrary quantiles, duplicate standardizations, and null models without convergence or exact-reference checks. |
| 首次组合 | Whether and how strongly a paper joins knowledge-source pairs absent from the strictly prior literature. | [WANG2017](https://doi.org/10.1016/j.respol.2017.06.006), [BORNMANN2019](https://doi.org/10.1016/j.joi.2019.100979), [FONTANA2020](https://doi.org/10.1016/j.respol.2020.104063) | A3.FIRST_SHARE | Include ex-ante first-occurrence statistics whose denominator and history are explicit and contain no future reuse. | Exclude future-confirmed reuse from T0, reference-count-dominated duplicates, and distance statistics below coverage/stability thresholds. |
| 知识广度与均衡性 | The number, relative spread, and balance of scientific categories represented in a paper's strictly prior references. | [STIRLING2007](https://doi.org/10.1098/rsif.2007.0213), [WANG2015](https://doi.org/10.1371/journal.pone.0127298), [RAFOLS2010](https://doi.org/10.1007/s11192-009-0041-y), [HILL1973](https://doi.org/10.2307/1934352) | A4.VARIETY；A4.OTHER_FIELD_SHARE；A4.GINI_BALANCE | Retain theoretically distinct variety and balance families with paper-level scientometric use; choose one interpretable and stable representative within each family. | Exclude strict monotone rescalings, algebraic complements, and multiple Hill orders as separate primary degrees of freedom. |
| 认知距离与整合 | The cognitive disparity among referenced fields and the degree to which disparate fields are jointly integrated. | [STIRLING2007](https://doi.org/10.1098/rsif.2007.0213), [RAO1982](https://doi.org/10.1016/0040-5809(82)90004-1), [PORTER2007](https://doi.org/10.1007/s11192-007-1700-5), [RAFOLS2010](https://doi.org/10.1007/s11192-009-0041-y), [LEYDESDORFF2019](https://doi.org/10.1016/j.joi.2018.12.006), [ZHANG2016](https://doi.org/10.1002/asi.23487), [LEINSTER2012](https://doi.org/10.1890/10-2402.1), [SHIBAYAMA2021](https://doi.org/10.1371/journal.pone.0254034), [FOSTER2015](https://doi.org/10.1177/0003122415601618) | A5.MEAN_DISTANCE；A5.RAO_STIRLING | Retain an unweighted distance component and at most one share-weighted composite; require frozen pre-publication distance profiles. | Exclude duplicate composites, future-trained semantic spaces, and graph measures lacking scalable formula fidelity or local coverage. |

这些角度的角色是解释论文如何选取、组合和整合既有知识。A4/A5 的跨学科与距离指标是支持性创新上下文，不等于创新本身；A1–A3 更直接观察组合罕见性、非典型性和首次出现。

## 4. 最终主创新指标：公式、来源和测量门槛

| ID | 模型列 | 角度 | 冻结公式 | 原始/应用/验证来源 | 全队列原始覆盖 | 有效分母覆盖 | 最低大类覆盖 | 80%重采样最差ρ | 最大中位相对误差 |
|---|---|---|---|---|---|---|---|---|---|
| A1.REFERENCE_OVERLAP | reference_overlap_novelty_t0 | 组合稀有性 | 1-mean_j(\|R_i intersection R_j\|/\|R_i union R_j\|) | [MATSUMOTO2021](https://doi.org/10.1007/s11192-021-04049-z) | 0.6876 | 0.9357 | 0.5437 | 0.9508 | 0.0025 |
| A2.HYPERGEOM_MEDIAN | hypergeom_conventionality_median_t0 | 非典型性与常规性 | Q0.50((O_ij-N_i*N_j/N)/sqrt(N_j*(N_i/N)*(1-N_i/N)*(N-N_j)/(N-1))) | [YAN2016](https://doi.org/10.1140/epjds/s13688-016-0069-1), [UZZI2013](https://doi.org/10.1126/science.1240474), [TEPLITSKIY2022](https://doi.org/10.1073/pnas.2118046119), [FONTANA2020](https://doi.org/10.1016/j.respol.2020.104063), [ZHOU2022](https://doi.org/10.1016/j.jbi.2022.104047) | 0.7486 | 0.9860 | 0.9433 | 0.9454 | 0.0935 |
| A3.FIRST_SHARE | first_time_source_pair_share | 首次组合 | sum 1[O_ij,<T0=0]/number_valid_pairs | [WANG2017](https://doi.org/10.1016/j.respol.2017.06.006), [BORNMANN2019](https://doi.org/10.1016/j.joi.2019.100979), [FONTANA2020](https://doi.org/10.1016/j.respol.2020.104063) | 0.7634 | 1.0000 | 0.9990 | 0.9576 | 0.0669 |
| A4.VARIETY | field_variety | 知识广度与均衡性 | number of occupied reference-field categories | [STIRLING2007](https://doi.org/10.1098/rsif.2007.0213), [WANG2015](https://doi.org/10.1371/journal.pone.0127298), [RAFOLS2010](https://doi.org/10.1007/s11192-009-0041-y) | 0.7831 | 1.0000 | 1.0000 | 0.9442 | 0.0000 |
| A4.OTHER_FIELD_SHARE | reference_other_field_share | 知识广度与均衡性 | mean(1[field_ref != field_focal]) | [WANG2015](https://doi.org/10.1371/journal.pone.0127298) | 0.7831 | 1.0000 | 1.0000 | 0.9854 | 0.0526 |
| A4.GINI_BALANCE | field_gini_balance | 知识广度与均衡性 | 1-Gini(p_1,...,p_V) over occupied categories | [STIRLING2007](https://doi.org/10.1098/rsif.2007.0213), [WANG2015](https://doi.org/10.1371/journal.pone.0127298), [LEYDESDORFF2019](https://doi.org/10.1016/j.joi.2018.12.006) | 0.7831 | 1.0000 | 1.0000 | 0.9048 | 0.0562 |
| A5.MEAN_DISTANCE | field_disparity_cosine_mean | 认知距离与整合 | mean(1-cosine(profile_k,profile_l)) | [STIRLING2007](https://doi.org/10.1098/rsif.2007.0213), [WANG2015](https://doi.org/10.1371/journal.pone.0127298), [LEYDESDORFF2019](https://doi.org/10.1016/j.joi.2018.12.006) | 0.7060 | 0.9464 | 0.8103 | 0.9573 | 0.0000 |
| A5.RAO_STIRLING | rao_stirling_integration | 认知距离与整合 | sum_k sum_l p_k p_l d_kl | [RAO1982](https://doi.org/10.1016/0040-5809(82)90004-1), [STIRLING2007](https://doi.org/10.1098/rsif.2007.0213), [PORTER2007](https://doi.org/10.1007/s11192-007-1700-5), [WANG2015](https://doi.org/10.1371/journal.pone.0127298), [RAFOLS2010](https://doi.org/10.1007/s11192-009-0041-y) | 0.7583 | 0.9702 | 0.9419 | 0.9743 | 0.0537 |

覆盖门槛使用预先声明的 `eligible_by_metric_family` 有效分母：论文至少有 10 条有效参考文献；来源对指标还要求来源映射不低于 60%，领域指标要求领域映射不低于 60%。参考集合重叠指标只需要参考文献 ID，因此不额外要求来源/领域映射。全队列原始覆盖同时保留，不能用插补伪造为已测量。

## 5. 全部候选范围和逐项决定

共登记 **50** 个候选、**34** 个学术来源。下表包含所有被发现并正式登记的候选，包括因外部文本、未来信息、公式证据不足、覆盖、稳定性或冗余而排除的指标。

| 候选ID | 角度 | 数学家族 | 公式 | 本地等级 | 来源 | 最终角色 | 逐项理由 |
|---|---|---|---|---|---|---|---|
| A1.NOVELTY_U | 组合稀有性 | commonness_lower_tail | -ln(Q0.10((O_ij+zero_rule)*N/(N_i*N_j))) | F1_existing_tables | [LEE2015](https://doi.org/10.1016/j.respol.2014.10.007), [BORNMANN2019](https://doi.org/10.1016/j.joi.2019.100979), [FONTANA2020](https://doi.org/10.1016/j.respol.2020.104063) | sensitivity | Implemented candidate retained for sensitivity; it either failed a primary gate or lost a frozen redundancy-family competition. |
| A1.MEAN_SURPRISAL | 组合稀有性 | commonness_mean | mean(-ln(O_ij*N/(N_i*N_j))) | F1_existing_tables | [FONTANA2020](https://doi.org/10.1016/j.respol.2020.104063) | exploratory | The arithmetic mean of transformed commonness is a transparent project-defined distribution summary, but Lee et al. define the lower-tail Novelty U rather than this mean. It therefore lacks a peer-reviewed original formula and paper-level application as this exact statistic and cannot enter the primary or sensitivity model. |
| A1.LOW_FREQUENCY_SHARE | 组合稀有性 | thresholded_pair_rarity | mean(1[O_ij<=k]) | F1_existing_tables | [LEE2015](https://doi.org/10.1016/j.respol.2014.10.007), [FONTANA2020](https://doi.org/10.1016/j.respol.2020.104063) | exploratory | The count threshold is a project choice rather than a uniquely source-defined paper-level formula. |
| A1.REFERENCE_OVERLAP | 组合稀有性 | prior_reference_overlap_mean | 1-mean_j(\|R_i intersection R_j\|/\|R_i union R_j\|) | F2_local_snapshot | [MATSUMOTO2021](https://doi.org/10.1007/s11192-021-04049-z) | primary | First outcome-blind eligible representative in the frozen A1_REFERENCE_OVERLAP priority order. |
| A2.UZZI_P10 | 非典型性与常规性 | z_distribution_left_tail | -Q0.10((O_ij-E_ij)/SD_ij) | F1_existing_tables | [UZZI2013](https://doi.org/10.1126/science.1240474), [TEPLITSKIY2022](https://doi.org/10.1073/pnas.2118046119), [FONTANA2020](https://doi.org/10.1016/j.respol.2020.104063) | sensitivity | Implemented candidate retained for sensitivity; it either failed a primary gate or lost a frozen redundancy-family competition. |
| A2.UZZI_MEDIAN | 非典型性与常规性 | z_distribution_centre | Q0.50((O_ij-E_ij)/SD_ij) | F1_existing_tables | [UZZI2013](https://doi.org/10.1126/science.1240474), [TEPLITSKIY2022](https://doi.org/10.1073/pnas.2118046119), [FONTANA2020](https://doi.org/10.1016/j.respol.2020.104063) | sensitivity | Implemented candidate retained for sensitivity; it either failed a primary gate or lost a frozen redundancy-family competition. |
| A2.HYPERGEOM_P10 | 非典型性与常规性 | exact_hypergeometric_z_left_tail | -Q0.10((O_ij-N_i*N_j/N)/sqrt(N_j*(N_i/N)*(1-N_i/N)*(N-N_j)/(N-1))) | F1_existing_tables | [YAN2016](https://doi.org/10.1140/epjds/s13688-016-0069-1), [UZZI2013](https://doi.org/10.1126/science.1240474), [TEPLITSKIY2022](https://doi.org/10.1073/pnas.2118046119), [FONTANA2020](https://doi.org/10.1016/j.respol.2020.104063), [ZHOU2022](https://doi.org/10.1016/j.jbi.2022.104047) | sensitivity | Implemented candidate retained for sensitivity; it either failed a primary gate or lost a frozen redundancy-family competition. |
| A2.HYPERGEOM_MEDIAN | 非典型性与常规性 | exact_hypergeometric_z_centre | Q0.50((O_ij-N_i*N_j/N)/sqrt(N_j*(N_i/N)*(1-N_i/N)*(N-N_j)/(N-1))) | F1_existing_tables | [YAN2016](https://doi.org/10.1140/epjds/s13688-016-0069-1), [UZZI2013](https://doi.org/10.1126/science.1240474), [TEPLITSKIY2022](https://doi.org/10.1073/pnas.2118046119), [FONTANA2020](https://doi.org/10.1016/j.respol.2020.104063), [ZHOU2022](https://doi.org/10.1016/j.jbi.2022.104047) | primary | First outcome-blind eligible representative in the frozen A2_CENTRE priority order. |
| A2.UZZI_MEAN | 非典型性与常规性 | z_distribution_centre | mean((O_ij-E_ij)/SD_ij) | F1_existing_tables | [UZZI2013](https://doi.org/10.1126/science.1240474), [FONTANA2020](https://doi.org/10.1016/j.respol.2020.104063) | excluded | The source defines median conventionality, not mean; the mean is a non-source-defined centre variant. |
| A2.UZZI_P01 | 非典型性与常规性 | z_distribution_left_tail | -Q0.01(z_ij) | F1_existing_tables | [UZZI2013](https://doi.org/10.1126/science.1240474), [SHIBAYAMA2021](https://doi.org/10.1371/journal.pone.0254034) | sensitivity | Published quantile variant; cannot add a primary degree of freedom beside source-defined P10. |
| A2.FIELD_YEAR_PERCENTILE | 非典型性与常规性 | z_distribution_left_tail | within-field-year empirical percentile of -P10 z | F1_existing_tables | [TEPLITSKIY2022](https://doi.org/10.1073/pnas.2118046119) | excluded | Same-year percentile is a retrospective standardization and not a distinct mathematical signal. |
| A3.FIRST_ANY | 首次组合 | first_pair_incidence | 1[sum 1(O_ij,<T0=0)>0] | F1_existing_tables | [WANG2017](https://doi.org/10.1016/j.respol.2017.06.006), [BORNMANN2019](https://doi.org/10.1016/j.joi.2019.100979), [FONTANA2020](https://doi.org/10.1016/j.respol.2020.104063) | sensitivity | Implemented candidate retained for sensitivity; it either failed a primary gate or lost a frozen redundancy-family competition. |
| A3.FIRST_COUNT | 首次组合 | first_pair_incidence | sum 1[O_ij,<T0=0] | F1_existing_tables | [WANG2017](https://doi.org/10.1016/j.respol.2017.06.006), [FONTANA2020](https://doi.org/10.1016/j.respol.2020.104063) | sensitivity | Implemented candidate retained for sensitivity; it either failed a primary gate or lost a frozen redundancy-family competition. |
| A3.FIRST_SHARE | 首次组合 | first_pair_incidence | sum 1[O_ij,<T0=0]/number_valid_pairs | F0_existing | [WANG2017](https://doi.org/10.1016/j.respol.2017.06.006), [BORNMANN2019](https://doi.org/10.1016/j.joi.2019.100979), [FONTANA2020](https://doi.org/10.1016/j.respol.2020.104063) | primary | First outcome-blind eligible representative in the frozen A3_FIRST_INCIDENCE priority order. |
| A3.FIRST_DISTANCE_SUM | 首次组合 | distance_weighted_first_pairs | sum d_ij*1[O_ij,<T0=0] | F1_existing_tables | [WANG2017](https://doi.org/10.1016/j.respol.2017.06.006), [FONTANA2020](https://doi.org/10.1016/j.respol.2020.104063) | excluded | Candidate-specific technical-debt rule: the distance-weighted family remains excluded unless the locally derived historical source distances pass the fixed overall and each-domain coverage gates; no missing value may be imputed to obtain eligibility. |
| A3.FIRST_DISTANCE_MEAN | 首次组合 | distance_weighted_first_pairs | mean(d_ij\|O_ij,<T0=0) | F1_existing_tables | [WANG2017](https://doi.org/10.1016/j.respol.2017.06.006), [FONTANA2020](https://doi.org/10.1016/j.respol.2020.104063) | excluded | Known v6 technical debt: v6.1 derives the source-distance values without imputation, but this candidate remains excluded unless it passes the fixed overall and each-domain coverage gates. |
| A3.FUTURE_REUSED | 首次组合 | future_confirmed_first_pairs | share of first-time pairs reused in future years | F1_existing_tables | [WANG2017](https://doi.org/10.1016/j.respol.2017.06.006), [BORNMANN2019](https://doi.org/10.1016/j.joi.2019.100979) | excluded | Uses future reuse and therefore violates the zero-leakage publication-time rule. |
| A4.VARIETY | 知识广度与均衡性 | category_variety | number of occupied reference-field categories | F0_existing | [STIRLING2007](https://doi.org/10.1098/rsif.2007.0213), [WANG2015](https://doi.org/10.1371/journal.pone.0127298), [RAFOLS2010](https://doi.org/10.1007/s11192-009-0041-y) | primary | First outcome-blind eligible representative in the frozen A4_VARIETY priority order. |
| A4.RELATIVE_VARIETY | 知识广度与均衡性 | category_variety | occupied categories / 26 frozen categories | F1_existing_tables | [LEYDESDORFF2019](https://doi.org/10.1016/j.joi.2018.12.006) | sensitivity | Implemented candidate retained for sensitivity; it either failed a primary gate or lost a frozen redundancy-family competition. |
| A4.OTHER_FIELD_SHARE | 知识广度与均衡性 | focal_field_breadth | mean(1[field_ref != field_focal]) | F1_existing_tables | [WANG2015](https://doi.org/10.1371/journal.pone.0127298) | primary | First outcome-blind eligible representative in the frozen A4_FOCAL_BREADTH priority order. |
| A4.GINI_BALANCE | 知识广度与均衡性 | category_balance | 1-Gini(p_1,...,p_V) over occupied categories | F1_existing_tables | [STIRLING2007](https://doi.org/10.1098/rsif.2007.0213), [WANG2015](https://doi.org/10.1371/journal.pone.0127298), [LEYDESDORFF2019](https://doi.org/10.1016/j.joi.2018.12.006) | primary | First outcome-blind eligible representative in the frozen A4_BALANCE priority order. |
| A4.SHANNON | 知识广度与均衡性 | category_balance | -sum p_k ln p_k | F1_existing_tables | [STIRLING2007](https://doi.org/10.1098/rsif.2007.0213), [WANG2015](https://doi.org/10.1371/journal.pone.0127298), [RAFOLS2010](https://doi.org/10.1007/s11192-009-0041-y) | sensitivity | Implemented candidate retained for sensitivity; it either failed a primary gate or lost a frozen redundancy-family competition. |
| A4.PIELOU | 知识广度与均衡性 | category_balance | Shannon entropy / ln(V) | F0_existing | [PIELOU1966](https://doi.org/10.1086/282439), [RAFOLS2010](https://doi.org/10.1007/s11192-009-0041-y), [WANG2015](https://doi.org/10.1371/journal.pone.0127298) | sensitivity | Implemented candidate retained for sensitivity; it either failed a primary gate or lost a frozen redundancy-family competition. |
| A4.GINI_SIMPSON | 知识广度与均衡性 | category_balance | 1-sum p_k^2 | F1_existing_tables | [STIRLING2007](https://doi.org/10.1098/rsif.2007.0213), [WANG2015](https://doi.org/10.1371/journal.pone.0127298) | sensitivity | Implemented candidate retained for sensitivity; it either failed a primary gate or lost a frozen redundancy-family competition. |
| A4.HHI | 知识广度与均衡性 | category_balance | sum p_k^2 | F1_existing_tables | [STIRLING2007](https://doi.org/10.1098/rsif.2007.0213), [WANG2015](https://doi.org/10.1371/journal.pone.0127298) | excluded | Exact affine complement of Gini-Simpson and therefore cannot be an independent feature. |
| A4.HILL_Q0 | 知识广度与均衡性 | hill_effective_categories | V | F1_existing_tables | [HILL1973](https://doi.org/10.2307/1934352), [ZHANG2016](https://doi.org/10.1002/asi.23487) | excluded | Exactly identical to category variety. |
| A4.HILL_Q1 | 知识广度与均衡性 | hill_effective_categories | exp(Shannon) | F1_existing_tables | [HILL1973](https://doi.org/10.2307/1934352), [ZHANG2016](https://doi.org/10.1002/asi.23487) | sensitivity | Implemented candidate retained for sensitivity; it either failed a primary gate or lost a frozen redundancy-family competition. |
| A4.HILL_Q2 | 知识广度与均衡性 | hill_effective_categories | 1/sum p_k^2 | F1_existing_tables | [HILL1973](https://doi.org/10.2307/1934352), [ZHANG2016](https://doi.org/10.1002/asi.23487) | sensitivity | Implemented candidate retained for sensitivity; it either failed a primary gate or lost a frozen redundancy-family competition. |
| A5.MEAN_DISTANCE | 认知距离与整合 | unweighted_field_distance | mean(1-cosine(profile_k,profile_l)) | F0_existing | [STIRLING2007](https://doi.org/10.1098/rsif.2007.0213), [WANG2015](https://doi.org/10.1371/journal.pone.0127298), [LEYDESDORFF2019](https://doi.org/10.1016/j.joi.2018.12.006) | primary | First outcome-blind eligible representative in the frozen A5_UNWEIGHTED_DISTANCE priority order. |
| A5.MAX_DISTANCE | 认知距离与整合 | unweighted_field_distance | max d_kl | F1_existing_tables | [STIRLING2007](https://doi.org/10.1098/rsif.2007.0213), [SHIBAYAMA2021](https://doi.org/10.1371/journal.pone.0254034) | sensitivity | Implemented candidate retained for sensitivity; it either failed a primary gate or lost a frozen redundancy-family competition. |
| A5.P90_DISTANCE | 认知距离与整合 | unweighted_field_distance | Q0.90(d_kl) | F1_existing_tables | [STIRLING2007](https://doi.org/10.1098/rsif.2007.0213), [SHIBAYAMA2021](https://doi.org/10.1371/journal.pone.0254034) | sensitivity | Implemented candidate retained for sensitivity; it either failed a primary gate or lost a frozen redundancy-family competition. |
| A5.RAO_STIRLING | 认知距离与整合 | share_weighted_integration_composite | sum_k sum_l p_k p_l d_kl | F0_existing | [RAO1982](https://doi.org/10.1016/0040-5809(82)90004-1), [STIRLING2007](https://doi.org/10.1098/rsif.2007.0213), [PORTER2007](https://doi.org/10.1007/s11192-007-1700-5), [WANG2015](https://doi.org/10.1371/journal.pone.0127298), [RAFOLS2010](https://doi.org/10.1007/s11192-009-0041-y) | primary | First outcome-blind eligible representative in the frozen A5_COMPREHENSIVE_INDEX priority order. |
| A5.PORTER_INTEGRATION | 认知距离与整合 | share_weighted_integration_composite | sum_i sum_j p_i p_j d_ij | F1_existing_tables | [PORTER2007](https://doi.org/10.1007/s11192-007-1700-5), [STIRLING2007](https://doi.org/10.1098/rsif.2007.0213) | excluded | Algebraically identical to the registered Rao-Stirling implementation. |
| A5.DIV | 认知距离与整合 | multiplicative_div_composite | relative_variety*(1-Gini)*mean_disparity | F1_existing_tables | [LEYDESDORFF2019](https://doi.org/10.1016/j.joi.2018.12.006), [LEYDESDORFF2019B](https://doi.org/10.1016/j.joi.2019.03.016) | sensitivity | Implemented candidate retained for sensitivity; it either failed a primary gate or lost a frozen redundancy-family competition. |
| A5.DIV_STAR | 认知距离与整合 | multiplicative_div_composite | N*DIV | F1_existing_tables | [LEYDESDORFF2019B](https://doi.org/10.1016/j.joi.2019.03.016) | excluded | Constant positive rescaling of DIV under the frozen 26-field taxonomy. |
| A5.TRUE_RS_2DS | 认知距离与整合 | effective_similarity_diversity | 1/(1-Rao-Stirling) | F1_existing_tables | [ZHANG2016](https://doi.org/10.1002/asi.23487), [LEINSTER2012](https://doi.org/10.1890/10-2402.1) | sensitivity | Implemented candidate retained for sensitivity; it either failed a primary gate or lost a frozen redundancy-family competition. |
| A5.LEINSTER_COBBOLD | 认知距离与整合 | effective_similarity_diversity | 1/sum_i p_i*(Zp)_i for q=2 and Z=1-d | F1_existing_tables | [LEINSTER2012](https://doi.org/10.1890/10-2402.1), [ZHANG2016](https://doi.org/10.1002/asi.23487) | sensitivity | At q=2 with Z=1-d it is algebraically equivalent to the registered true-Rao transform. |
| A5.BRIDGE_SHARE | 认知距离与整合 | historical_community_bridging | share of focal relations bridging distinct historical communities | F2_local_snapshot | [FOSTER2015](https://doi.org/10.1177/0003122415601618) | exploratory | The biomedical chemical-network unit and community procedure do not map source-faithfully to the current all-domain compact view. |
| A5.NETWORK_COHERENCE | 认知距离与整合 | network_coherence | mean linkage strength and mean path length among focal knowledge elements | F2_local_snapshot | [RAFOLS2010](https://doi.org/10.1007/s11192-009-0041-y) | exploratory | Published as a two-component network construct; a single source-faithful, scalable paper-level implementation is not yet frozen. |
| A5.SEMANTIC_DISTANCE | 认知距离与整合 | semantic_reference_distance | registered quantile of cosine distances among cited-reference embeddings | F3_external_required | [SHIBAYAMA2021](https://doi.org/10.1371/journal.pone.0254034) | excluded | Would require an unfrozen external text model and the source model is biomedical rather than validated across all 12 natural-science domains. |
| A1.CONCEPT_VOLUME_RARITY | 组合稀有性 | controlled_term_pair_volume_novelty | paper aggregation of inverse prior occurrence volume for concepts and concept pairs | F3_external_required | [MISHRA2016](https://doi.org/10.1045/september2016-mishra), [DOMAINAGNOSTIC2026](https://doi.org/10.1007/s11192-026-05687-x) | excluded | A valid conceptual-novelty family, but it uses controlled textual concepts rather than co-cited knowledge sources and the frozen Nature view has no cross-domain controlled-vocabulary history. |
| A1.CONTENT_CONTEXT_SURPRISE | 组合稀有性 | content_context_surprise | surprise of observed content-context association relative to a historical expectation model | F3_external_required | [SHI2023](https://doi.org/10.1038/s41467-023-36741-4) | excluded | The construct is source-backed but is not the frozen co-cited-source rarity angle and requires external text/context representations. |
| A1.TOPIC_CLOUD_NOVELTY | 组合稀有性 | topic_cloud_novelty | source-defined topic-model features aggregated by a fuzzy cloud novelty model | F3_external_required | [WANG2024](https://doi.org/10.1016/j.joi.2024.101587) | excluded | The topic/cloud family is outside the frozen reference-based operational angle and requires unavailable text plus trained models. |
| A1.LLM_TOKEN_NLL | 组合稀有性 | language_model_token_surprise | mean_t[-log P_model(token_t \| token_<t)] | F3_external_required | [KRUZEN2026](https://doi.org/10.1007/978-3-032-08851-2_24), [DOMAINAGNOSTIC2026](https://doi.org/10.1007/s11192-026-05687-x) | excluded | The source-backed textual-surprise family cannot be derived from the frozen citation/OpenAlex views and is not equivalent to cited-source combination rarity. |
| A1.HUMAN_LLM_NOVELTY | 组合稀有性 | human_llm_novelty_judgment | source-defined aggregation of human knowledge and LLM novelty judgments | F3_external_required | [WU2025](https://doi.org/10.1002/asi.24979) | excluded | This is a dynamic judgment system rather than a frozen mathematical feature and violates the local-data and fully reproducible formula rules. |
| A3.CONCEPT_TIME_NOVELTY | 首次组合 | controlled_term_pair_time_novelty | paper aggregation of the age or first-observed year of concepts and concept pairs | F3_external_required | [MISHRA2016](https://doi.org/10.1045/september2016-mishra), [DOMAINAGNOSTIC2026](https://doi.org/10.1007/s11192-026-05687-x) | excluded | It measures first use of controlled textual concepts rather than first co-citation of frozen knowledge sources, and no cross-domain concept history is locally frozen. |
| A3.NEW_COMPONENT_SHARE | 首次组合 | new_component_incidence | number of components absent from prior literature divided by focal component count | F3_external_required | [YAN2020](https://doi.org/10.1007/s11192-019-03314-6) | excluded | The frozen data contain cited sources and fields but not a source-faithful, cross-domain historical keyword/component view. |
| A3.QUESTION_METHOD_COMBINATION | 首次组合 | question_method_semantic_combination | source-defined semantic atypicality of the focal question-method combination against prior papers | F3_external_required | [LUO2022](https://doi.org/10.1016/j.joi.2022.101282) | excluded | This is a valid semantic combination family but is not computable from the frozen citation/OpenAlex tables and is not the cited-source first-combination construct. |
| A5.FASTTEXT_LOF_SEMANTIC | 认知距离与整合 | semantic_local_outlier | local outlier factor of a paper embedding in the source-defined historical semantic space | F3_external_required | [JEON2023](https://doi.org/10.1016/j.joi.2023.101450) | excluded | Document-level semantic outlier status is not reference-field integration, and the required text/model artifacts are not locally frozen. |
| A5.LLM_SEMANTIC_GAIN | 认知距离与整合 | contextual_embedding_semantic_gain | mean_t[1-cosine(embedding_model1(token_t),embedding_model2(token_t))] | F3_external_required | [KRUZEN2026](https://doi.org/10.1007/978-3-032-08851-2_24), [DOMAINAGNOSTIC2026](https://doi.org/10.1007/s11192-026-05687-x) | excluded | The metric is source-backed but is a language-model representation shift, not reference-field cognitive integration, and requires unfrozen external text models. |

筛选时强制执行：同行评议公式与论文级应用；唯一角度映射；发表前信息；仅冻结本地数据；公式与缺失规则可复现；有效覆盖总体不低于 70%、每大类不低于 50%；80%参考重采样 Spearman不低于 0.90 且中位相对误差不高于 0.10；近似公式与精确实现Spearman 不低于 0.95 且误差不高于 0.05；通过手算、时间和非退化测试。同一数学家族只保留一个主实现。OOF 不在这些规则中。

## 6. 结果盲修订记录

R2 首次实现把参考重叠的背景限制为建模主队列。该版本在有效分母中的总体覆盖为 **0.8754**，最低大类覆盖为 **0.3750**，数学统计类未达到 0.50。R2 制品和哈希已保留。

R3 仅把比较背景扩展到完整的本地冻结 Nature v5 既往记录；总体有效覆盖升至 **0.8800**，但最低大类仍只有 **0.3937**，所以同样未冻结。R3 保留 10 年参考窗口和 3 年共引窗口，其失败制品与哈希也已保留。

R4 回查原始方法与结果表后，只在该论文实际评估的四种窗口方案中选择最后一个尚未核验的 `all references / all prior co-citing papers` 变体。R4 没有改变 Jaccard 公式、共享至少一条参考文献和同领域两项条件，也没有改变焦点论文、阈值、标签、模型或时间折。该版本在任何标签/OOF 读取前通过覆盖后，仍须通过相同的 80% 重采样稳定性门槛。

R5 是完成审计发现的协议角色修订，而不是性能调参：`first_time_source_pair_distance_mean` 和同族 sum 版本虽然已从本地历史来源距离派生出非零值，但有效覆盖仅约 0.22%，最低大类约 0.14%，远低于固定门槛。R4 自动把所有已实现的非主指标标为敏感性，违背了该技术债“未过覆盖则保持排除”的专门规则。R5 将这两项改为排除；覆盖、稳定性、近似忠实度表与 R4 逐值一致，八个主指标、全部模型输入、标签、折、参数和种子均未改变。修订后重新冻结注册表并完整重跑六折 OOF。
R5 的八个最终结果文件与 R4 对应文件 SHA-256 完全相同；总 artifact ID 不同，因为配置、注册表和冻结谱系按 R5 更新。

此外，来源核对发现 `source_pair_mean_surprisal` 是项目自定义分布均值，不是 Lee 等人定义的 Novelty U。它因此被明确降为探索性排除项，不能借用 Lee 来源进入主模型；A1 改由 Matsumoto 等人发表的 `1 − mean Jaccard` 公式竞争。

## 7. 控制特征及其他实际进入模型的变量

Controls represent publication-time citation opportunity, exposure, team scale, or measurement opportunity. They are auxiliary prediction features and must never be interpreted as paper-innovation indicators.

| 特征 | 集合 | 意义 | 公式 | 来源 | 最大信息时点 | 缺失规则 |
|---|---|---|---|---|---|---|
| publication_year | K0/K1 | Controls exposure era and secular changes in citation practices. | calendar publication year | [BORNMANN_DANIEL2008](https://doi.org/10.1108/00220410810844150), [TAHAMTAN2016](https://doi.org/10.1007/s11192-016-1889-2) | publication | Missing year excludes the paper before modeling. |
| domain12 | K0/K1 | Controls broad field differences in citation and diffusion opportunity. | frozen twelve-domain category | [TAHAMTAN2016](https://doi.org/10.1007/s11192-016-1889-2), [TAHAMTAN_BORNMANN2020](https://doi.org/10.1007/s11192-019-03243-4) | publication | Out-of-scope or missing domain is not silently reassigned. |
| openalex_primary_subfield | K1 | Controls finer field heterogeneity remaining inside domain12. | frozen OpenAlex primary subfield label | [TAHAMTAN2016](https://doi.org/10.1007/s11192-016-1889-2), [TAHAMTAN_BORNMANN2020](https://doi.org/10.1007/s11192-019-03243-4) | frozen snapshot metadata for the publication record | Missing label becomes an explicit training-fold unknown category. |
| venue_family | K0/K1 | Controls venue-format and exposure differences without using a current prestige score. | frozen local Nature venue-family identifier | [TAHAMTAN2016](https://doi.org/10.1007/s11192-016-1889-2), [WANG2015](https://doi.org/10.1371/journal.pone.0127298) | publication | Missing venue becomes an explicit training-fold unknown category. |
| log_reference_count | K0/K1 | Controls citation opportunity and the mechanical number of possible reference combinations. | ln(1 + number of distinct declared references) | [TAHAMTAN2016](https://doi.org/10.1007/s11192-016-1889-2), [WANG2015](https://doi.org/10.1371/journal.pone.0127298) | publication | Unretrieved lists are missing; a verified empty list is zero. |
| reference_age_median | K0/K1 | Controls the central recency of the cited knowledge base. | median(publication_year - strictly-prior reference_year) | [TAHAMTAN2016](https://doi.org/10.1007/s11192-016-1889-2), [TAHAMTAN_BORNMANN2020](https://doi.org/10.1007/s11192-019-03243-4) | publication_year-1 | No dated prior reference is null. |
| reference_age_iqr | K1 | Controls dispersion in the ages of cited knowledge. | Q0.75(reference age) - Q0.25(reference age) | [TAHAMTAN2016](https://doi.org/10.1007/s11192-016-1889-2), [TAHAMTAN_BORNMANN2020](https://doi.org/10.1007/s11192-019-03243-4) | publication_year-1 | Insufficient dated prior references are null. |
| title_word_count | K1 | Controls a documented article-presentation correlate of citations. | count of regex word tokens in the frozen title | [TAHAMTAN2016](https://doi.org/10.1007/s11192-016-1889-2) | publication | Empty or unavailable title is null. |
| log_author_count | K1 | Controls team scale and associated visibility/collaboration opportunity. | ln(1 + distinct authors) | [TAHAMTAN2016](https://doi.org/10.1007/s11192-016-1889-2), [WANG2015](https://doi.org/10.1371/journal.pone.0127298) | publication authorship record | Missing frozen OpenAlex authorship metadata is null. |
| log_institution_count | K1 | Controls organizational collaboration scale. | ln(1 + distinct affiliated institutions) | [TAHAMTAN2016](https://doi.org/10.1007/s11192-016-1889-2) | publication authorship record | Missing frozen OpenAlex affiliation metadata is null. |
| log_country_count | K1 | Controls international collaboration and geographic exposure. | ln(1 + distinct affiliation countries) | [TAHAMTAN2016](https://doi.org/10.1007/s11192-016-1889-2), [WANG2015](https://doi.org/10.1371/journal.pone.0127298) | publication authorship record | Missing frozen OpenAlex country metadata is null. |
| log_team_prior_nature_output_max | K2 | Strong sensitivity control for prior team productivity; it is not all-OpenAlex career output. | ln(1 + maximum strictly-prior Nature-cohort paper count across focal authors) | [TAHAMTAN2016](https://doi.org/10.1007/s11192-016-1889-2), [TAHAMTAN_BORNMANN2020](https://doi.org/10.1007/s11192-019-03243-4) | publication_year-1 | No locally matched author IDs is null. |
| log_prior_reference_popularity_median | K2 | Controls prior popularity of the knowledge base selected by the paper. | median over focal references of ln(1 + strictly-prior Nature citation count) | [TAHAMTAN2016](https://doi.org/10.1007/s11192-016-1889-2), [TAHAMTAN_BORNMANN2020](https://doi.org/10.1007/s11192-019-03243-4) | publication_year-1 | No valid prior references is null. |
| bc_degree_per_reference_t0 | K2 | Controls local bibliographic-coupling reach. | number of strictly-prior papers sharing at least one reference with focal / max(1, valid_reference_count) | [BISCARO2014](https://doi.org/10.1371/journal.pone.0099502), [GUAN2017](https://doi.org/10.1016/j.joi.2017.02.007) | publication_year-1 | Unavailable prior coupling view is null; a verified isolate is zero. |
| bc_shared_reference_strength_t0 | K2 | Controls the strength of pre-existing bibliographic similarity. | sum over strictly-prior neighbor papers of references shared with the focal paper | [BISCARO2014](https://doi.org/10.1371/journal.pone.0099502) | publication_year-1 | Unavailable prior coupling view is null; verified no overlap is zero. |
| bc_component_share_t0 | K2 | Controls breadth of the pre-existing connected opportunity neighborhood. | size of focal augmented coupling component / number of eligible strictly-prior papers | [BISCARO2014](https://doi.org/10.1371/journal.pone.0099502) | publication_year-1 | Unavailable prior coupling view is null. |

K0 是原五项历史对照；K1 是十一项主控制集；K2 只作强控制敏感性分析。质量/检索成功标记不作为实质预测特征。当前 JIF、未来引用、可变开放获取状态和发表后作者声望均未进入模型。

实际比较的特征集合为：

- `k0_controls`：`publication_year`, `domain12`, `venue_family`, `log_reference_count`, `reference_age_median`
- `k1_controls`：`publication_year`, `domain12`, `openalex_primary_subfield`, `venue_family`, `log_reference_count`, `reference_age_median`, `reference_age_iqr`, `title_word_count`, `log_author_count`, `log_institution_count`, `log_country_count`
- `b0_v6_primary_plus_k0`：`publication_year`, `domain12`, `venue_family`, `log_reference_count`, `reference_age_median`, `b0_novelty_u_t0_source`, `b0_uzzi_atypicality_p10_t0`, `b0_field_variety`, `b0_field_pielou_evenness`, `b0_field_disparity_cosine_mean`, `b0_rao_stirling_integration`
- `provisional_core8_plus_k1`：`publication_year`, `domain12`, `openalex_primary_subfield`, `venue_family`, `log_reference_count`, `reference_age_median`, `reference_age_iqr`, `title_word_count`, `log_author_count`, `log_institution_count`, `log_country_count`, `novelty_u_t0_source`, `uzzi_atypicality_p10_t0`, `uzzi_conventionality_median_t0`, `first_time_source_pair_share`, `field_variety`, `field_pielou_evenness`, `field_disparity_cosine_mean`, `rao_stirling_integration`
- `final_innovation_plus_k1`：`publication_year`, `domain12`, `openalex_primary_subfield`, `venue_family`, `log_reference_count`, `reference_age_median`, `reference_age_iqr`, `title_word_count`, `log_author_count`, `log_institution_count`, `log_country_count`, `reference_overlap_novelty_t0`, `hypergeom_conventionality_median_t0`, `first_time_source_pair_share`, `field_gini_balance`, `reference_other_field_share`, `field_variety`, `field_disparity_cosine_mean`, `rao_stirling_integration`
- `k2_controls`：`publication_year`, `domain12`, `openalex_primary_subfield`, `venue_family`, `log_reference_count`, `reference_age_median`, `reference_age_iqr`, `title_word_count`, `log_author_count`, `log_institution_count`, `log_country_count`, `log_team_prior_nature_output_max`, `log_prior_reference_popularity_median`, `bc_degree_per_reference_t0`, `bc_shared_reference_strength_t0`, `bc_component_share_t0`
- `final_innovation_plus_k2`：`publication_year`, `domain12`, `openalex_primary_subfield`, `venue_family`, `log_reference_count`, `reference_age_median`, `reference_age_iqr`, `title_word_count`, `log_author_count`, `log_institution_count`, `log_country_count`, `log_team_prior_nature_output_max`, `log_prior_reference_popularity_median`, `bc_degree_per_reference_t0`, `bc_shared_reference_strength_t0`, `bc_component_share_t0`, `reference_overlap_novelty_t0`, `hypergeom_conventionality_median_t0`, `first_time_source_pair_share`, `field_gini_balance`, `reference_other_field_share`, `field_variety`, `field_disparity_cosine_mean`, `rao_stirling_integration`

## 8. 数据及谱系

| 项目 | 数值/说明 |
|---|---|
| 焦点主论文 | 118059 |
| 年份 | 1980–2017 |
| 自然科学大类 | 12 |
| 参考重叠背景论文 | 168378 |
| 扫描的 Nature 引用边 | 7644007 |
| OpenAlex works 分片 | 2127 |
| OpenAlex 完成分片 | 2127 |
| OpenAlex works 记录扫描 | 492361307 |
| OpenAlex 目标元数据覆盖 | 1.0000 |
| 参考重叠窗口 | reference=all_prior; co-citing=all_prior |
| 外部实验数据 | 无；联网只用于文献证据检索 |

论文—参考、目标、队列和机会特征直接复用 v6 冻结视图；审计逐文件验证 v6 与 v6.1 哈希一致。新建的只有候选创新特征、参考重叠历史、OpenAlex 目标元数据和 K1/K2 派生控制视图。OpenAlex 原始快照没有复制、更新或联网补抓。

## 9. 标签是什么

标签不是“创新真值”。对每个 D3/D5/D8 窗口，第一阶段标签为：

```text
future_uptake = 1[n_future_citers > 0]
```

未来请求成功且没有施引者的论文保留为 0；请求失败保持缺失，绝不改成 0。第二阶段只在正 uptake 且未来分类数据完整的训练论文上构造扩散分数：对未来施引者的 field/subfield/topic reach 取`log1p` 后按训练折经验分布转为分位，对 field/topic Simpson均衡度同样转为分位；三个 breadth 分量求均值、两个 evenness分量求均值，最后各占 0.5。

测试论文的最终实现标签为：无 uptake 时等于 0；有 uptake 且分类完整时等于训练折坐标系下的扩散分位；分类不完整时缺失。因此 Spearman 回答的是“模型能否把未参与训练的论文按未来学术传播和跨领域扩散程度正确排序”，不是预测引用次数。

## 10. 模型和 OOF 计算

所有模型使用完全相同的论文、标签和六个扩展时间折。每折只用更早年份训练；数值缺失填补、缺失指示、类别编码、目标分位坐标和校准全部在训练折内完成。

| 折 | 训练 | 预测 |
|---|---|---|
| 1 | ≤1985 | 1986–1999 |
| 2 | ≤1999 | 2000–2004 |
| 3 | ≤2004 | 2005–2009 |
| 4 | ≤2009 | 2010–2012 |
| 5 | ≤2012 | 2013–2013 |
| 6 | ≤2013 | 2014–2017 |

固定 `medium` 两部分模型：第一部分为未来 uptake 的 HistGradientBoostingClassifier；第二部分为正 uptake 条件下扩散得分的 HistGradientBoostingRegressor。最终分数为：

```text
expected_diffusion_score
  = calibrated P(future uptake)
  × calibrated conditional diffusion score
```

固定参数：`max_leaf_nodes=31`、`max_depth=4`、`min_samples_leaf=50`、`learning_rate=0.05`、`max_iter=200`、`l2=10.0`。每个外折内部再用 4 个时间折产生校准预测；没有根据外层 OOF 选择模型容量。

主结果是把 1986–2017 六个互斥测试折拼接后的 D5 Spearman。D3/D8 只检查相对 K1 的方向。相对 K1 和 B0 的 D5 差值使用 2000 次按论文 ID 的配对 bootstrap 计算 95% 区间。未报告条件 Spearman。

## 11. OOF 结果

| 窗口 | 模型 | OOF论文 | Spearman |
|---|---|---|---|
| D3 | final_innovation_plus_k1 | 101379 | 0.7625 |
| D3 | k1_controls | 101379 | 0.6900 |
| D5 | b0_v6_primary_plus_k0 | 101379 | 0.7497 |
| D5 | final_innovation_plus_k1 | 101379 | 0.7670 |
| D5 | final_innovation_plus_k2 | 101379 | 0.7668 |
| D5 | k0_controls | 101379 | 0.6499 |
| D5 | k1_controls | 101379 | 0.6813 |
| D5 | k2_controls | 101379 | 0.6832 |
| D5 | provisional_core8_plus_k1 | 101379 | 0.7631 |
| D8 | final_innovation_plus_k1 | 101379 | 0.7667 |
| D8 | k1_controls | 101379 | 0.6772 |

D5 配对比较：

| 候选模型 | 基线 | Spearman增量 | 95%下界 | 95%上界 | 论文数 |
|---|---|---|---|---|---|
| final_innovation_plus_k1 | k1_controls | 0.0857 | 0.0831 | 0.0884 | 101350 |
| final_innovation_plus_k1 | b0_v6_primary_plus_k0 | 0.0174 | 0.0161 | 0.0186 | 101350 |

补充模型仅使用最终冻结的 8 个创新指标，不含 K0/K1/K2 控制特征，并保持相同 D5 标签、medium 参数和六个时间折。其 OOF Spearman 为 **0.7073**；K1 单独为 **0.6813**，创新指标＋K1 为 **0.7670**。该补充结果描述创新指标的独立预测能力；正式的控制后增量仍应使用 `final_innovation_plus_k1` 相对 `k1_controls` 的配对比较。

成功门槛：

- `d3_direction_positive`：通过
- `d5_hard_minimum_0_74`：通过
- `d5_target_0_75`：通过
- `d8_direction_positive`：通过
- `noninferior_to_b0`：通过
- `positive_increment_over_k1`：通过

总体判定：**全部通过**。

12 大类均保留。领域结果只作异质性报告，不允许因某个领域分数低而删除该领域。完整领域表位于结果制品的 `/home/jayee/workspace/ASPR/outputs/common/new/model/v6_1_r5/oof_d361264b867c/oof_domain_metrics.csv`。

## 12. 能说什么、不能说什么

可以说：这些特征是有来源、发表时可计算、通过预先测量门槛的创新相关证据信号；固定模型对未来学术传播/扩散排序达到本节报告的 OOF 表现；创新信号相对强控制的增量由配对区间量化。

不能说：五角度穷尽创新；某一论文的指标值是创新真值；高 OOF 证明论文更正确、更有社会价值或更应被 Nature 接收；相关性或预测增量构成因果效应。

主要限制包括：组合历史和参考重叠背景是 Nature 本地闭包而不是全球文献全集；Matsumoto 的 WoS 领域步骤被适配为冻结 OpenAlex 主领域；OpenAlex 当前快照中的分类和作者机构元数据无法证明其历史版本从未变化；极高施引论文存在返回上限；早期年份可用历史较短；参考文献缺失会造成结构性不可计算。所有这些限制均保留在结果解释中。

## 13. 可复现性核验、命令与关键制品

复现性核验：**pass**；核对 8 个最终输出哈希，验证 66 个折×模型检查点及其测试论文集合；重复调用结果清单：**一致**。

另外从空检查点目录独立重拟合 66 个模型折；所有 OOF 预测逐字段精确匹配原结果：**通过**。

原计划逐项完成审计：**21/21 项通过**，失败项 **0**。该审计逐项检查检索、来源、五角度、候选家族、筛选门槛、技术债排除、K0/K1/K2、本地数据边界、v6 不变性、时间折、标签公平性、结果门槛、12 大类和确定性复演。

最终回归测试：**138 项通过，0 项失败**。唯一警告是 Python 多进程 `fork` 弃用提示；静态检查只声明已实际完成的 `py_compile`，本环境未安装 Ruff/Black/mypy，故不虚报这些工具的通过结果。

```bash
python3 scripts/run_nature_v6_1_local.py materialize-overlap
python3 scripts/run_nature_v6_1_local.py screen
python3 scripts/run_nature_v6_1_local.py freeze
python3 scripts/run_nature_v6_1_local.py scan-openalex --workers 12
python3 scripts/run_nature_v6_1_local.py materialize
python3 scripts/run_nature_v6_1_local.py audit
python3 scripts/run_nature_v6_1_local.py oof
python3 scripts/run_v6_1_innovation_only.py
python3 scripts/verify_v6_1_reproducibility.py --full-replay
python3 scripts/audit_v6_1_completion.py
python3 scripts/render_v6_1_evidence_map.py
```

| 制品 | 路径/ID | SHA-256或artifact ID |
|---|---|---|
| 候选目录 | /home/jayee/workspace/ASPR/configs/innovation_candidate_catalog_v6_1.json | sha256:543ec2ef9c3a5f77b51c3424274547f9dc0d9e1c639fe566358b2c35f31403e0 |
| 检索日志 | /home/jayee/workspace/ASPR/configs/innovation_search_log_v6_1.json | sha256:c05a5c1a6ddbb362174b7f08dc7d704e493aebb57ae6e0d60530ab59d5866934 |
| 最终指标注册表 | /home/jayee/workspace/ASPR/configs/innovation_candidate_registry_v6_1.json | sha256:126202f72519a4eb1657f49b8daa06bdcd9fccb37fe66cd6e4a8ec8471762611 |
| 控制注册表 | /home/jayee/workspace/ASPR/configs/control_feature_registry_v6_1.json | sha256:06907b954d140ea9b9c119eb5b2c56f5969bfcd8214ad37fe8e0efda503806f6 |
| 筛选 | /home/jayee/workspace/ASPR/outputs/common/new/model/v6_1_r5 | sha256:7d3a30898710101c86b6e1d6cf523c0c96e9f1bf82fedf7966812cfb7d17ec3a |
| 数据物化 | /home/jayee/workspace/ASPR/data/knowledge_corpus/nature_multihorizon_v6_1_local | sha256:88050770c4cbb1b7190175b1b3265ab3e3a439f09349ab8805ada173dd32680e |
| 数据审计 | /home/jayee/workspace/ASPR/outputs/common/new/model/v6_1_r5/data_quality_report.json | sha256:95df1582c8f5fdbfcd1643261d7adcbb912f33680a1bac78411e60a68b6fb6ac |
| 复现性核验 | /home/jayee/workspace/ASPR/outputs/common/new/model/v6_1_r5/reproducibility_report.json | sha256:2a21a86de4ff3c9efbded883274ca3cbdceeed58c6ce9bc28250575e58a1b2ea |
| 原计划完成审计 | /home/jayee/workspace/ASPR/outputs/common/new/model/v6_1_r5/completion_audit.json | sha256:f99621525dee66a136e6cb8c76e52aef697212d2be1aca7d40426fb7566adb02 |
| 最终验证摘要 | /home/jayee/workspace/ASPR/outputs/common/new/model/v6_1_r5/validation_summary.json | sha256:28942c9c8d9c7c77d762d136536e3332610f656880f198d6f4e50978a5341f43 |
| 纯创新指标补充模型 | /home/jayee/workspace/ASPR/outputs/common/new/model/v6_1_r5/supplement_innovation_only_3b387272d53d/innovation_only_manifest.json | sha256:ea96f8c33163ae43f0811e4338e9599bf6819014a208b13f0d219f3c4881031a |
| OOF | /home/jayee/workspace/ASPR/outputs/common/new/model/v6_1_r5 | sha256:1ae922413b008225c42061e7de50eeee10d20ae14233c637b3f3975da5a23cc0 |

## 14. 候选指标来源表

| 来源ID | 引文 | DOI/URL | 证据状态 | 在本项目中的作用 |
|---|---|---|---|---|
| UZZI2013 | Uzzi B, Mukherjee S, Stringer M, Jones B (2013). Atypical Combinations and Scientific Impact. Science 342:468-472. | https://doi.org/10.1126/science.1240474 | 同行评议 | Original paper-level atypicality and conventionality formulas. |
| LEE2015 | Lee Y-N, Walsh JP, Wang J (2015). Creativity in scientific teams: Unpacking novelty and impact. Research Policy 44:684-697. | https://doi.org/10.1016/j.respol.2014.10.007 | 同行评议 | Original paper-level Novelty U/commonness operationalization. |
| WANG2017 | Wang J, Veugelers R, Stephan P (2017). Bias against novelty in science. Research Policy 46:1416-1436. | https://doi.org/10.1016/j.respol.2017.06.006 | 同行评议 | Original new-combination incidence and distance operationalization. |
| BORNMANN2019 | Bornmann L, Tekles A, Zhang HH, Ye FY (2019). Do we measure novelty when we analyze unusual combinations of cited references? Journal of Informetrics 13:100979. | https://doi.org/10.1016/j.joi.2019.100979 | 同行评议 | Independent convergent-validity comparison of Novelty U and W. |
| FONTANA2020 | Fontana M, Iori M, Montobbio F, Sinatra R (2020). New and atypical combinations. Research Policy 49:104063. | https://doi.org/10.1016/j.respol.2020.104063 | 同行评议 | Independent construct and aggregation-level critique. |
| MATSUMOTO2021 | Matsumoto K, Shibayama S, Kang B, Igami M (2021). Introducing a novelty indicator for scientific research. Scientometrics 126:6891-6915. | https://doi.org/10.1007/s11192-021-04049-z | 同行评议 | Paper-level reference-overlap novelty with researcher validation. |
| FOSTER2015 | Foster JG, Rzhetsky A, Evans JA (2015). Tradition and Innovation in Scientists' Research Strategies. American Sociological Review 80:875-908. | https://doi.org/10.1177/0003122415601618 | 同行评议 | Original network community consolidation and bridging strategies. |
| SHIBAYAMA2021 | Shibayama S, Yin D, Matsumoto K (2021). Measuring novelty in science with word embedding. PLOS ONE 16:e0254034. | https://doi.org/10.1371/journal.pone.0254034 | 同行评议 | Original semantic distance among cited references. |
| STIRLING2007 | Stirling A (2007). A general framework for analysing diversity in science, technology and society. Journal of the Royal Society Interface 4:707-719. | https://doi.org/10.1098/rsif.2007.0213 | 同行评议 | Foundational variety-balance-disparity framework. |
| WANG2015 | Wang J, Thijs B, Glänzel W (2015). Interdisciplinarity and Impact: Distinct Effects of Variety, Balance, and Disparity. PLOS ONE 10:e0127298. | https://doi.org/10.1371/journal.pone.0127298 | 同行评议 | Paper-level category count, other-field share, 1-Gini, Shannon, Simpson, mean distance and Rao-Stirling. |
| RAFOLS2010 | Rafols I, Meyer M (2010). Diversity and network coherence as indicators of interdisciplinarity. Scientometrics 82:263-287. | https://doi.org/10.1007/s11192-009-0041-y | 同行评议 | Paper-level diversity, integration, and network coherence. |
| LEYDESDORFF2019 | Leydesdorff L, Wagner CS, Bornmann L (2019). Interdisciplinarity as diversity in citation patterns among journals. Journal of Informetrics 13:255-269. | https://doi.org/10.1016/j.joi.2018.12.006 | 同行评议 | DIV and citation-profile cosine disparity. |
| LEYDESDORFF2019B | Leydesdorff L, Wagner CS, Bornmann L (2019). Diversity measurement: Steps towards the measurement of interdisciplinarity? Journal of Informetrics 13:904-905. | https://doi.org/10.1016/j.joi.2019.03.016 | 同行评议 | DIV* effective-number refinement discussion. |
| PORTER2007 | Porter AL, Cohen AS, Roessner JD, Perreault M (2007). Measuring researcher interdisciplinarity. Scientometrics 72:117-147. | https://doi.org/10.1007/s11192-007-1700-5 | 同行评议 | Paper-level integration score equivalent to Rao-Stirling. |
| ZHANG2016 | Zhang L, Rousseau R, Glänzel W (2016). Diversity of references as an indicator of the interdisciplinarity of journals. JASIST 67:1257-1265. | https://doi.org/10.1002/asi.23487 | 同行评议 | Hill-type true diversity / 2D3 application to reference fields. |
| LEINSTER2012 | Leinster T, Cobbold CA (2012). Measuring diversity: the importance of species similarity. Ecology 93:477-489. | https://doi.org/10.1890/10-2402.1 | 同行评议 | Foundational similarity-sensitive effective diversity family. |
| RAO1982 | Rao CR (1982). Diversity and dissimilarity coefficients: A unified approach. Theoretical Population Biology 21:24-43. | https://doi.org/10.1016/0040-5809(82)90004-1 | 同行评议 | Original quadratic entropy. |
| PIELOU1966 | Pielou EC (1966). Shannon's formula as a measure of specific diversity: its use and misuse. American Naturalist 100:463-465. | https://doi.org/10.1086/282439 | 同行评议 | Established normalized Shannon evenness. |
| TEPLITSKIY2022 | Teplitskiy M, Peng H, Blasco A, Lakhani KR (2022). Is novel research worth doing? PNAS 119:e2118046119. | https://doi.org/10.1073/pnas.2118046119 | 同行评议 | Independent paper-level use of Uzzi novelty and conventionality. |
| RAFOLS2012 | Rafols I, Porter AL, Leydesdorff L (2012). Science overlay maps: a new tool for research policy and library management. Journal of the American Society for Information Science and Technology 61:1871-1887. | https://doi.org/10.1002/asi.21368 | 同行评议 | Paper-level/portfolio integration mapping application. |
| HILL1973 | Hill MO (1973). Diversity and evenness: a unifying notation and its consequences. Ecology 54:427-432. | https://doi.org/10.2307/1934352 | 同行评议 | Original Hill effective-number family. |
| YAN2016 | Yan B, Luo J (2016). Technological novelty profile and invention's future impact. EPJ Data Science 5:2. | https://doi.org/10.1140/epjds/s13688-016-0069-1 | 同行评议 | Peer-reviewed exact fixed-marginal hypergeometric co-occurrence null and z-score equations used to distinguish the source-adapted analytical null from the original Uzzi edge-swap Monte Carlo null. |
| ZHOU2022 | Zhou X, Zhou M, Huang D, Cui L (2022). A probabilistic model for co-occurrence analysis in bibliometrics. Journal of Biomedical Informatics 128:104047. | https://doi.org/10.1016/j.jbi.2022.104047 | 同行评议 | Independent bibliometric application and simulation study of an exact fixed-marginal hypergeometric co-occurrence model. |
| MISHRA2016 | Mishra S, Torvik VI (2016). Quantifying conceptual novelty in the biomedical literature. D-Lib Magazine 22(9/10). | https://doi.org/10.1045/september2016-mishra | 同行评议 | Paper-level time and volume novelty of controlled biomedical concepts and concept pairs. |
| YAN2020 | Yan Y, Tian S, Zhang J (2020). The impact of a paper's new combinations and new components on its citation. Scientometrics 122:1415-1437. | https://doi.org/10.1007/s11192-019-03314-6 | 同行评议 | Paper-level distinction between new knowledge components and new component combinations. |
| LUO2022 | Luo Z, Lu W, He J, Wang Y (2022). Combination of research questions and methods: A new measurement of scientific novelty. Journal of Informetrics 16:101282. | https://doi.org/10.1016/j.joi.2022.101282 | 同行评议 | Paper-level semantic novelty of research-question and method combinations. |
| SHI2023 | Shi F, Evans J (2023). Surprising combinations of research contents and contexts are related to impact and emerge with scientific outsiders from distant disciplines. Nature Communications 14:1641. | https://doi.org/10.1038/s41467-023-36741-4 | 同行评议 | Paper-level surprise of content-context combinations. |
| JEON2023 | Jeon D, Lee J, Ahn JM, Lee C (2023). Measuring the novelty of scientific publications: A fastText and local outlier factor approach. Journal of Informetrics 17:101450. | https://doi.org/10.1016/j.joi.2023.101450 | 同行评议 | Paper-level semantic outlier novelty using text embeddings and local outlier factor. |
| WANG2024 | Wang Z, Zhang H, Chen J, Chen H (2024). An effective framework for measuring the novelty of scientific articles through integrated topic modeling and cloud model. Journal of Informetrics 18:101587. | https://doi.org/10.1016/j.joi.2024.101587 | 同行评议 | Paper-level topic-model and cloud-model novelty framework. |
| WU2025 | Wu W, Zhang C, Zhao Y (2025). Automated novelty evaluation of academic paper: A collaborative approach integrating human and large language model knowledge. JASIST 76:1452-1469. | https://doi.org/10.1002/asi.24979 | 同行评议 | Paper-level human/LLM collaborative novelty evaluation. |
| KRUZEN2026 | Kruzenshtern A, Dodonov V, Chechurin L (2026). AI-based metric for the scientific text novelty. World Conference of AI-Powered Innovation and TRIZ Methodology:367-378. | https://doi.org/10.1007/978-3-032-08851-2_24 | 同行评议 | Original average token negative-log-likelihood and contextual semantic-gain metrics. |
| DOMAINAGNOSTIC2026 | Kruzenshtern A, Dodonov V, Chechurin L (2026). Domain agnostic features for robust novelty assessment of scientific publication. Scientometrics. | https://doi.org/10.1007/s11192-026-05687-x | 同行评议 | Paper-level cross-domain application and stability analysis of token-loss and semantic-gain distributions. |
| ZHAO2025 | Zhao Y, Zhang C (2025). A review on the novelty measurements of academic papers. Scientometrics 130:727-753. | https://doi.org/10.1007/s11192-025-05234-0 | 同行评议 | Discovery and taxonomy review only; not used as the sole formula authority for any admitted metric. |
| NOVELPY2022 | Pelletier P, Wirtz K (2022). Novelpy: A Python package to measure novelty and disruptiveness of bibliometric and patent data. | https://arxiv.org/abs/2211.10346 | 仅发现候选 | Discovery-only software catalog; never final formula evidence. |

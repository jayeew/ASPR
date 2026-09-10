# Fig.1 代码与证据核对记录

仅对用户上传的代码进行静态阅读；没有执行模型、重建图谱或恢复未上传的逐节点数据。

## 当前 figure_pipeline 的逐 panel 分工

该文件实际包含 33 条 panel 记录。历史对话中的 36 个 panel 不作为本次文件实测数量。

| Figure | Panel | 问题 | 证据状态 |
|---|---|---|---|
| 1 | a | 双层历史结构是什么？ | saved_snapshot_descriptive |
| 1 | b | 插入前后发生何种局部变化？ | saved_snapshot_descriptive |
| 1 | c | 真实结构量如何分布？ | saved_snapshot_descriptive |
| 2 | a | 两个分支与joint如何独立衔接？ | saved_snapshot_descriptive |
| 2 | b | 判断如何追到终端来源？ | saved_snapshot_descriptive |
| 2 | c | 实际输出有哪些层次？ | saved_snapshot_descriptive |
| 3 | a | 结构描述之间有何关系？ | saved_snapshot_descriptive |
| 3 | b | 变化是否由局部拓扑机械决定？ | saved_snapshot_descriptive |
| 3 | c | 解释是否经独立原文核验？ | missing_core_evidence |
| 4 | a | 两方法的配对总体效果如何？ | saved_snapshot_descriptive |
| 4 | b | 论文层面差值如何分布？ | saved_snapshot_descriptive |
| 4 | c | 同一参考ID的覆盖状态如何变化？ | saved_snapshot_descriptive |
| 5 | a | 融合改变了什么判断？ | missing_core_evidence |
| 5 | b | 哪些角色出现收益或伤害？ | missing_core_evidence |
| 5 | c | 真实纠正或损害如何溯源？ | missing_core_evidence |
| 6 | a | 受控输入与生成路径是否一致？ | missing_core_evidence |
| 6 | b | 去除组件的配对效应是什么？ | missing_core_evidence |
| 6 | c | joint超出同文本综合的增量是什么？ | missing_core_evidence |
| 7 | a | 可用性条件与真实效果怎样关联？ | missing_core_evidence |
| 7 | b | 扰动后的结论是否稳健？ | missing_core_evidence |
| 7 | c | 不确定性是否恰当？ | missing_core_evidence |
| 8 | a | 历史与未来如何隔离？ | saved_snapshot_descriptive |
| 8 | b | 去重的方向空间如何组织？ | saved_snapshot_descriptive |
| 8 | c | 预测短语如何排序？ | saved_snapshot_descriptive |
| 8 | d | 增长与热度基线表现如何？ | saved_snapshot_descriptive |
| 9 | a | 期刊样本与来源怎样构成？ | saved_snapshot_descriptive |
| 9 | b | 论文归一后贡献角色怎样分布？ | saved_snapshot_descriptive |
| 9 | c | 结构画像及不确定性怎样变化？ | saved_snapshot_descriptive |
| 10 | a | C05原文与抽取的范围差别是什么？ | saved_snapshot_descriptive |
| 10 | b | GEAR支持哪些历史比较？ | saved_snapshot_descriptive |
| 10 | c | 各claim邻域怎样连接？ | saved_snapshot_descriptive |
| 10 | d | union图的局部变化是什么？ | saved_snapshot_descriptive |
| 10 | e | Graph报告如何对应主审稿参考？ | saved_snapshot_descriptive |

## 关键代码定位

### `figure_pipeline/specs/panels.json`
SHA-256: `37e940712d79624fb252470a59cd1176c5fdcb997424d35e4e54cea149e3c2db`


### `docs/module_architecture.md`
SHA-256: `f212f2810368ac0a0f48e06d1b9cd8e3f4377354a9e9c4b143a175105de1fb8b`

```text
# Current Claim Graph + GEAR architecture

Updated against the working-tree implementation on 2026-09-07. The public
runtime package is `gear`; default mode is `knowledge`. This replaces the
previous architecture based on five-part reviews and mandatory release exchange.

## Code boundaries

| Component | Implementation | Responsibility |
| --- | --- | --- |
| CLI/dispatcher | `gear/cli.py`, `gear/review_pipeline.py` | Input and stage selection |
| Shared contributions | `gear/innovation/shared.py`, `gear/grounding.py` | Full-text claims, consolidation and manuscript support |
| GEAR evidence | `gear/evidence_supervisor.py`, `gear/prior_art.py` | Retrieval, comparisons and residual increment |
| Graph facts | `gear/claim_attribution.py` | Native graph lookup/insertion; no paper-score allocation |
| Branch interpretation | `gear/innovation/analysis.py` | Source-bound GEAR, Graph and fusion assessments |
| Joint Graph | `gear/innovation/joint_graph.py` | Union-neighborhood facts and paper-level interpretation |
| Orchestration | `gear/innovation/pipeline.py` | Saved stages and claim-level fusion/report |
| Study reports | `experiments/innovation_200/reporting.py` | Whole-paper reports and ablation views |
| Study evaluation | `experiments/innovation_200/evaluate_human.py`, `compare_reports.py` | Reference matching, reviewer consistency and AI preferences |

Graph and GEAR share neutral contribution identities and scope, without reading
each other's judgments before fusion. Standard `all` executes GEAR then Graph
(including joint Graph), then claim fusion. Logical independence does not mean
this command runs both branches concurrently; the study streams separate stages.

## Contracts and artifacts

`InnovationPaperInput`, `GearClaim`, and `GraphFactCard` live in
`gear/review_contracts.py`; `ClaimSet`, `Assessment`, `Finding`, and
`AnalysisResult` live in `gear/innovation/contracts.py`. `PaperIR` preserves
manuscript spans; `EvidenceStore` preserves append-only source evidence.
Contracts reject extra fields.

| Standard run artifact | Meaning |
| --- | --- |
| `innovation_input.json` | Metadata, manuscript path and cutoff |
| `shared/paper_ir.json`, `shared/claims.json` | Shared manuscript and grounded claims |
| `gear/NN/gear_card.json`, `assessment.json` | Evidence status and claim interpretation |
| `graph/NN/` | Graph facts in evidence storage and claim interpretation |
| `gear/analysis.json`, `graph/analysis.json` | Branch assessments and limitations |
| `graph/joint/facts.json`, `analysis.json`, `report.md`, `status.json` | Joint facts, interpretation and status when available |
| `fusion/NN/`, `fusion/analysis.json` | Source-bound claim fusion |
| `fusion/innovation_report.md` | Standard report organized by claim |
| `model_usage.jsonl` | Model calls, timings and available usage |

Joint Graph is a separate paper-level result. Standard claim fusion does not
automatically consume it. The study's whole-paper `fusion` report consumes GEAR,
per-claim Graph and joint Graph through its report writer; it is distinct from
the runtime's `fusion/analysis.json`.

Findings cite existing evidence keys. Branch status (`complete`, `limited`,
`failed`) differs from stance (`recognized`, `incremental_or_limited`,
`challenged`, `unresolved`). Missing evidence cannot imply negative novelty or
global firstness. Valid evidence keys alone do not establish scientific truth.

## Graph and temporal scope

Historical claim extraction uses the local 2023–2025 Nature abstract corpus;
target claims are mined from full text. Paper citation and native Claim Graph
are separate layers. Five claim types (`METHOD`, `FINDING`, `MECHANISM`,
`RESOURCE`, `THEORY`) are node roles, not graph partitions. Parent-paper paths
annotate semantic relations, not claim support or derivation.

Runtime insertion is temporary, excludes self and date-ineligible neighbors,
and defaults to at most ten neighbors with cosine > 0.5. Sparse neighborhoods
remain explicit. Current thresholded metrics cannot inherit percentiles from
older selection rules. Joint insertion uses the union neighborhood and existing
historical edges; it neither invents within-paper target edges nor sums
individual perturbations. See [Graph details](graph_joint_analysis.md).

The graph is a bounded historical corpus, not all scientific literature.
Community mixing and connectivity are local structural facts, not validated
causality, global novelty or predicted impact.

## Study artifacts and evaluation

`experiments/innovation_200/` retains its original name; `papers.jsonl` defines
the active roster. The 1,000-paper extension updates the roster and runs shared
claims and reviewer-reference reconstruction, not all downstream stages.

The study exchanges ordinary JSON/JSONL/Markdown/CSV within its directory:

- `human_refs/{paper_id}.json`: source-bound reviewer contribution references;
- `papers/{paper_id}/`: shared, GEAR and Graph artifacts;
- `reports/{system}/`: eight report variants and citation appendices;
- `human_evaluation/`, `reviewer_consistency/`, `pairwise/`: comparisons;
- `status/`, `logs/`, `tables/`, `summary.json`, `summary.md`: stage records and
  summaries when generated.

Reference reconstruction does not expose reviewer judgments to system branches.
Tier A contains explicit innovation judgments; tier B supports identification.
The study retains the last explicit opinion for its main comparison and keeps
historical rounds inspectable. Matching/preference calls are AI judgments, not
new human annotations. Reference silence is not a false-positive label.

`gear/innovation/experiments.py` also contains shared-claim direct/RAG controls;
these differ from the study's eight report variants. See
[study instructions](../experiments/innovation_200/README.md).

## Resume and validation limits

Default runtime enables response caching, relation stability checks and resume
fingerprint checks. The study disables all three, limits claims to eight and
uses Luna role overrides with low/high efforts. Completed stage files can still
be reused when model-response caching is disabled. Interrupted evidence attempts
are archived; raw evidence remains append-only.

Use separate directories for uniform new experimental conditions. Explicitly
chosen mixed-policy resumes retain prior results, including historical full text
obtained before PDF downloading was disabled. They cannot be reported as uniform
abstract-only reruns.

`validate-assets` checks required graph files and basic index/embedding shape
consistency, not full semantics, temporal leakage or scientific validity.
`validate-run` checks fingerprints, claim identities, coverage and evidence keys
in three standard claim-level branches. It does not check joint interpretations,
whole-paper reports or scientific truth. The study disables fingerprints and
need not produce standard fusion artifacts, so this is not a drop-in study
validator. Legacy runs without shared claims are only displayed by the CLI.

## Historical interfaces

`gear/module_cli.py`, `gear/module_registry.py`, `artifact_store/`, older
`StructuredReview` consumers and `experiments/gear/` preserve earlier work.
They are not the default v2 exchange contract; check compatibility before use.
Current study artifacts do not universally require old human-agreement gates,
HGB promotion or immutable publication releases. Preserve historical research
without importing its conclusions or runtime assumptions into the new system.

```

### `gear/claim_attribution.py`
SHA-256: `86b4eab7b6d8463af949c1e0357157c8fd289e001ba20fae4bf1308da0e51891`

- `_neighbors`：L318–L386。
- `_paper_path`：L434–L468。
- `_metrics`：L474–L527。
- `_pair_surprisal`：L585–L604。
- `_neighbor_edges`：L606–L622。
- `_cross_boundary_share`：L654–L668。

### `scripts/claim_graph/07_build_claim_graph.py`
SHA-256: `8393e669e6e70f1da1fbbf94f7e8f6e9b5005e9dc73b96362b549c58dae94511`

- `semantic_edges`：L151–L183。
- `communities`：L326–L384。

### `outputs/FROM_WEB/scripts/draw_fig01_fig03.py`
SHA-256: `f8e886e9a8ca47c645d2cbf3831d136702c026ce773dd2c976156590606bf439`

- `select_cases`：L68–L74。
- `atlas_data`：L77–L123。

### `figure_pipeline/graph_revision.py`
SHA-256: `15c3d72ec2304a5143d2c77b49b2b35a25597e74592beae567635e23716b60d3`

- `structural_audit`：L126–L166。
- `draw_atlas`：L259–L302。
- `local_positions`：L305–L325。

### `gear/innovation/joint_graph.py`
SHA-256: `ff4b91742aaadb66a6bdfdfaa81081c03d40dc36c6a800bd868ce80f68cd4afd`


### `PACKAGE_README.md`
SHA-256: `2fd42d00afa3f0e379a5add0b69b0091cb717816830a2acaa34d4fb7faa19d79`

```text
# 本地代码与图谱统计包

包含打包时工作区源码（含未提交改动）、运行配置、绘图程序、测试、必要说明，以及 outputs/GRAPH_STATISTICS/ 中的汇总统计。

不包含 data/ 原始数据库/Parquet/论文/嵌入、逐论文或逐claim实验记录、图稿源快照、日志、模型权重、.env、Git历史。configs中的单论文metadata也未纳入。代码中的接口/常量/测试样例不属于本次导出的研究原始数据。

图谱统计以数据库、保存历史结果、当前运行观察和展示子图分别列示；文件SHA-256证明包内容完整，不证明科学判断正确。缺少原始资产时此包不能独立重绘旧图或重新运行完整GEAR，符合本次仅要代码与统计数据的范围。

统计重算入口：scripts/reporting/summarize_claim_graph.py、summarize_paper_graph.py、summarize_runtime_graph.py；汇总入口assemble_graph_statistics.py；运行这些脚本需要原工作区资产。

```

## 需要改正或明确的绘图口径

1. `atlas_data()` 实际遍历所有传入案例的历史邻居；保存的 selection 文案仍写 C1/C2，因此不能仅按文案断定实际只含 C1/C2。必须统一实际选择集合与说明。
2. `select_cases()` 为三个硬编码 claim ID；这些代码常量不是本包提供的逐条研究证据，不能据此生成真实案例解释。
3. 原 `atlas_data()` 只取所选 claims 的父论文，再取这些论文间的直接引用；共同参考文献和二跳中介很可能在展示子图之外。新增路径见证节点必须来自实际 citation index。
4. `graph_revision.draw_atlas()` 的 claim 颜色与形状都编码 community；本设计把颜色用于 community，形状改用于五种 claim role。
5. `_neighbor_edges()` 明确从 semantic_backbone_adjacency 读取局部历史边。局部连通指标不以全部700,240语义边为计算图。
6. 社区按 mutual-kNN backbone 的 connected nodes 计算；没有 backbone 边的节点 community_id 为 null。社区相互渗透是显示布局，不是软成员关系。
7. 本包保存事实为 threshold_parent_path_v1，代码已采用 v2 路径去自环逻辑。代码修复不是已有事实已重算的证据。
8. 单 claim 星状插入后邻域连通是构造结果；不能把所有局部分量合并量当作独立创新效度。相关诊断属于 Fig.3。
9. 当静态图最大日期不早于目标cutoff时，当前代码拒绝直接使用，要求历史快照。旧时期案例不能沿用2023–2025全图社区。
10. 全文目标claims与摘要历史claims属于不同抽取口径；每篇数量差异不能解释为贡献增长。

# Fig.1-Fig.10 Nature 投稿级持续迭代审计流程

## 总目标

为当前 Fig.1-Fig.10 结果建立一个可以长期反复执行的、带有对抗性思维的 Nature 投稿级审计流程。这个流程的目标不是简单把图做得更好看，而是系统性发现每一组实验在数据、逻辑、可信性、复现性、叙事连贯性、图像排版和投稿说服力上的潜在问题。

最终目标：

> 形成一套 Fig.1-Fig.10 论文图组，其中每个主张、每张图、每个 caption、每个中间数据表、每个图片导出和每个限制条件，都能经得起 Nature 级别审稿人的追问；所有尚未解决的问题都必须被明确标记为 `已修复`、`可带 caveat 保留`、`转入 supplement` 或 `pipeline-ready gap`。

这个流程本身可以作为一个长期 `/goal` 使用。每一轮迭代都必须产出书面证据，而不是只依赖感觉。

---

## 核心原则

### 1. 证据先于说服

- 每个 caption、panel title、图中 annotation、正文段落里的核心 claim，都必须能追溯到中间表、脚本、manifest、quality report 或可复现命令。
- 如果证据来自 proxy、模拟、假设输出、LLM-as-judge、pipeline-ready 数据层，必须在图、caption 和数据表中明确标注。

### 2. 用 Nature 审稿人的恶意视角检查

默认审稿人会问：

- 是否存在 corpus construction artifact？
- 是否有 citation leakage？
- 是否被 field/year imbalance 驱动？
- 是否存在 venue bias？
- 是否过拟合？
- 是否 cherry-pick 了领域或案例？
- 是否用到了 post-publication 信息？
- 是否视觉效果过强但证据不足？

本流程要主动尝试“打碎”这篇论文，而不是帮它找借口。

### 3. 先看图，但不能只看图

每张图都必须同时检查：

- 渲染后的图片；
- 支撑图片的 CSV/JSON；
- 生成图片的脚本或 image prompt；
- 对应 caption 和正文 claim；
- quality report / run manifest；
- 是否能从当前仓库状态复现。

### 4. 不允许无限重做

每个问题最后必须落到四类之一：

- `fixed_in_main`：已在主图或主文中修复；
- `moved_to_supplement`：主图不再承载该 claim，转入补充材料；
- `retained_with_explicit_caveat`：保留，但 caption 和正文明确降级；
- `pipeline_ready_gap`：承认当前只是 pipeline-ready，需要未来真实数据替换。

### 5. 全文叙事阶梯必须清楚

- Fig.1-Fig.5：建立 graph-perturbation measurement 和验证链条；
- Fig.6：测试方法稳健性和边界条件；
- Fig.7：从方法有效性过渡到 venue-level scientific interpretation；
- Fig.8：提出最终方法 ASPR；
- Fig.9：展示 ASPR 在真实论文上的一次可审计运行；
- Fig.10：证明 ASPR 的性能来自模块组合，而不是单一 LLM。

---

## 当前需要纳入审计的输入

### 主图图片路径

除非后续 manifest 指向更新版本，本轮审计默认使用以下图片：

- `outputs/redraw_v6a_best_fig1/fig1_multi_domain_real.png`
- `outputs/redraw_v6a_best_fig2/fig2_empirical_full.png`
- `outputs/redraw_v6a_best_fig3/fig3_selected_weight_learning_full.png`
- `outputs/kg_perturbation_fig4_full50/fig4_full.png`
- `outputs/kg_perturbation_fig5/strict_ai_filtered_image2_handoff/fig5_strict_ai_filtered_image2_generated_preview.png`
- `outputs/kg_perturbation_fig6/fig6_full.png`
- `outputs/kg_perturbation_fig7/fig7_full.png`
- `outputs/kg_perturbation_fig8/fig8_full.png`
- `outputs/kg_perturbation_fig9/fig9_full.png`
- `outputs/kg_perturbation_fig10/fig10_full.png`
- `outputs/kg_perturbation_final_assembly/fig1_fig10_contact_sheet.png`

### 已有总装配材料

- `outputs/kg_perturbation_final_assembly/fig6_fig10_three_round_consistency_report.md`
- `outputs/kg_perturbation_final_assembly/fig1_fig10_caption_drafts.md`
- `outputs/kg_perturbation_final_assembly/fig1_fig10_cross_figure_audit.csv`
- `outputs/kg_perturbation_final_assembly/fig1_fig10_final_checklist.csv`
- `outputs/kg_perturbation_final_assembly/fig1_fig10_pipeline_ready_gaps.csv`
- `outputs/kg_perturbation_final_assembly/fig1_fig10_style_ledger.json`
- `outputs/kg_perturbation_final_assembly/fig1_fig10_terminology_crosswalk.csv`

### 高优先级中间数据

#### Fig.1

- `outputs/redraw_v6a_best_fig1/figure_quality_report.json`
- 各领域子目录下：
  - `works_selected.csv`
  - `topic_nodes.csv`
  - `topic_edges.csv`
  - `paper_edges.csv`
  - `snapshot_delta_metrics.csv`

#### Fig.2

- `outputs/redraw_v6a_best_fig2/fig2_input_audit.csv`
- `fig2_candidate_metrics.csv`
- `fig2_indicator_future_corr.csv`
- `fig2_indicator_future_corr_bootstrap.csv`
- `fig2_reference_closure_report.csv`
- `fig2_matched_controls.csv`
- `fig2_mechanism_evidence_support.csv`
- `figure_quality_report.json`

#### Fig.3

- `outputs/redraw_v6a_best_fig3/figure_quality_report.json`
- `outputs/redraw_v6a_best_fig3/multi_domain/fig3_score_table.csv`
- `fig3_best_weights.csv`
- `fig3_cv_summary.csv`
- `fig3_baseline_comparison.csv`
- `fig3_fold_weights.csv`

#### Fig.4

- `outputs/kg_perturbation_fig4_full50/fig4_metrics_summary.csv`
- `fig4_input_audit.csv`
- `fig4_peer_review_screen.csv`
- `fig4_retrieval_diagnostics.csv`
- `fig4_semantic_claim_matches.jsonl`
- `fig4_structured_consistency_judgements.jsonl`
- `fig4_aspect_relation_summary.csv`

#### Fig.5

- `outputs/kg_perturbation_fig5/*/fig5_panel_text.json`
- `outputs/kg_perturbation_fig5/*/fig5_image2_prompt.md`
- `outputs/kg_perturbation_fig5/plot_data/derived/*.csv`
- `outputs/kg_perturbation_fig5/plot_data/base/*.csv`

#### Fig.6

- `outputs/kg_perturbation_fig6/fig6_cross_domain_reproducibility.csv`
- `fig6_data_quality_perturbation.csv`
- `fig6_volume_sensitivity.csv`
- `fig6_temporal_window_sensitivity.csv`
- `fig6_modeling_choice_reproducibility.csv`
- `fig6_failure_modes.csv`
- `fig6_source_audit.csv`
- `fig6_caption.md`

#### Fig.7

- `outputs/kg_perturbation_fig7/fig7_vci_rankings.csv`
- `fig7_topk_enrichment.csv`
- `fig7_venue_portfolio.csv`
- `fig7_mechanism_signature.csv`
- `fig7_confounder_audit.csv`
- `fig7_metric_sensitivity.csv`
- `fig7_venue_family_mapping_audit.csv`
- `figure_quality_report.json`

#### Fig.8

- `outputs/kg_perturbation_fig8/panel_text.json`
- `flow_spec.json`
- `image2_prompt.md`
- `render_fig8.py`

#### Fig.9

- `outputs/kg_perturbation_fig9/fig9_case_manifest.csv`
- `fig9_claim_evidence_trace.csv`
- `fig9_agent_output.json`
- `fig9_aspr_qwen_output.json`
- `fig9_assumed_aspr_qwen_output.json`
- `fig9_fusion_output.json`
- `fig9_quality_report.json`

#### Fig.10

- `outputs/kg_perturbation_fig10/fig10_ablation_results.csv`
- `fig10_ablation_forest.csv`
- `fig10_error_taxonomy.csv`
- `fig10_human_preference_llm_judge_results.csv`
- `fig10_reinforcement_results.csv`
- `fig10_panel_text.json`
- `figure_quality_report.json`

---

## 一轮完整审计的八个阶段

## Phase 0：冻结本轮审计快照

目的：避免在不断变化的结果上做审计。

### 任务

- 记录 `git status --short`。
- 记录 Fig.1-Fig.10 主图图片路径、文件大小、分辨率、修改时间。
- 记录所有 quality report 和 run manifest。
- 重新生成或检查 contact sheet。
- 创建本轮审计目录：
  - `outputs/nature_submission_audit/iteration_YYYYMMDD_HHMM/`

### 输出

- `snapshot_manifest.json`
- `figure_file_index.csv`
- `intermediate_data_index.csv`
- `contact_sheet.png`
- `git_status.txt`

### 必问问题

- 当前每张图的 canonical image path 是哪个？
- 哪些图是 final candidate？
- 哪些图只是 diagnostic 或 pipeline-ready？
- 哪些输出晚于上一次 consistency report 生成？

### 停止条件

- 每张被审计的图都有唯一主图路径；
- 每张图都有对应中间数据索引；
- 本轮审计对象已经冻结。

---

## Phase 1：主张清单与证据映射

目的：明确论文到底在声称什么。

### 每张图需要抽取的 claim 来源

- panel title；
- axis label；
- 图中 annotation；
- caption draft；
- 对应正文段落；
- legend 和 color explanation；
- supplementary note。

### claim 类型

每条 claim 标记为：

- `descriptive`：描述性；
- `inferential`：推断性；
- `causal`：因果性；
- `predictive`：预测性；
- `methodological`：方法性；
- `comparative`：比较性；
- `algorithmic`：算法结构性。

### 证据等级

每条 claim 必须被归到：

- `directly_supported`：直接由数据支持；
- `supported_with_assumptions`：有假设条件支持；
- `proxy_supported`：由 proxy 实验支持；
- `pipeline_ready`：目前只是 pipeline-ready；
- `unsupported`：无证据；
- `overclaimed`：证据不足以支撑当前措辞。

### 输出

- `claim_inventory.csv`
- `claim_to_evidence_map.csv`
- `unsupported_or_overclaimed_claims.csv`

### 审稿人攻击问题

- claim 是否比数据更强？
- correlation 是否被写成 causation？
- 是否从少数领域外推到普遍规律？
- caption 是否隐藏了 proxy / assumed / LLM-as-judge？
- 图中颜色或箭头是否暗示了不存在的因果关系？

### 停止条件

- 主图和 caption 中没有未映射 claim；
- 所有 `unsupported` 或 `overclaimed` claim 都有处理动作。

---

## Phase 2：逐图数据取证

目的：先审数据，再审美观。

### 每张图统一检查维度

- 数据来源；
- 样本量；
- inclusion / exclusion 标准；
- 缺失值；
- duplicate records；
- leakage risk；
- future information risk；
- field/year imbalance；
- normalization 是否正确；
- bootstrap 或 uncertainty 方法；
- outlier 敏感性；
- provenance 是否清楚；
- 是否存在人工插入行；
- 是否存在未标注的假设数据。

### 每张图输出

- `figX_data_forensics.md`
- `figX_data_risk_table.csv`
- `figX_reproducibility_commands.sh`

### 风险等级

- `P0_blocker`：会推翻中心 claim；
- `P1_major`：投稿前必须修；
- `P2_moderate`：可以带 caveat 或 supplement 保留；
- `P3_minor`：排版、说明或局部 clarity 问题。

### 核心问题

- 这张图能否从当前输入重新生成？
- 派生变量是否可追踪？
- 排除样本是否合理？
- negative/null results 是否可见？
- 是否有手工制作但未标记的部分？

### 停止条件

- 每张图都有 data-risk classification；
- 每个 P0/P1 都有修复、降级或移除路径。

---

## Phase 3：统计与方法可信性审查

目的：判断方法是否能经得起定量审稿人的质疑。

### 全局检查

- performance metric 是否在结果前定义？
- control 是否匹配合理？
- confidence interval 是否有意义？
- 是否存在 multiple-comparison 风险？
- cross-validation 是否按 domain/year 分组？
- field/year effect 是否控制？
- null baseline 是否公平？
- proxy robustness panel 是否明确标注？
- LLM-as-judge 是否与 human labels 分开呈现？

### 分图重点

- Fig.1：图构建、landmark selection、过滤阈值；
- Fig.2：future outcome 定义、matched controls、reference closure；
- Fig.3：leakage、overfitting、model selection、fold stability；
- Fig.4：peer-review parsing、no-leakage、semantic matching、overclaiming；
- Fig.5：forecast humility、机制解释是否过强；
- Fig.6：proxy robustness 和完整 rerun 的边界；
- Fig.7：venue mapping、confounders、field-year normalization、interval separation；
- Fig.8：算法结构是否完整，不把架构图写成结果图；
- Fig.9：真实 evidence 与 assumed ASPR-Qwen lane 的边界；
- Fig.10：ablation 是否真实 rerun，LLM-as-judge 是否被当成人类偏好。

### 输出

- `statistical_credibility_audit.md`
- `methodology_failure_modes.csv`
- `required_sensitivity_reruns.csv`

### 停止条件

- 每个关键方法至少有一个 control / null / sensitivity 论据；
- 缺失的 sensitivity 必须被列为 gap，不能沉默处理。

---

## Phase 4：逻辑与叙事压力测试

目的：判断 Fig.1-Fig.10 是否真的构成一篇 Nature 论文，而不是十张松散图。

### 必须成立的叙事链

1. Fig.1 定义可观察的 graph perturbation；
2. Fig.2 证明这些指标有经验信号；
3. Fig.3 学到更稳健的 composite score；
4. Fig.4 把图证据连接到 peer-review-like innovation judgement；
5. Fig.5 把图信号转为 forecast / mechanism interpretation；
6. Fig.6 测试方法稳健性和失败边界；
7. Fig.7 转向 venue-level scientific contribution；
8. Fig.8 提出 ASPR 系统；
9. Fig.9 展示 ASPR 真实案例运行；
10. Fig.10 做 ASPR 模块消融和增强。

### 必问问题

- 是否有某张图出现在其前提被建立之前？
- Fig.7 是否从方法验证过度跳到 venue ranking？
- Fig.8 是否太晚才引入 ASPR？
- Fig.9 是真实运行还是 storyboard？
- Fig.10 是真实 ablation 还是假设 delta？
- 当前图组是不是其实包含两篇论文：graph perturbation paper 和 ASPR system paper？

### 输出

- `narrative_ladder_audit.md`
- `figure_transition_matrix.csv`
- `claim_scope_downgrades.csv`

### 停止条件

- 每张图都有清楚的“上一张图证明了什么，所以本图要做什么”；
- 任何叙事断裂都有正文桥接段或 caption 修改方案。

---

## Phase 5：视觉与排版审查

目的：保证每张图既高级又诚实。

### 查看尺度

每张图至少在以下尺度检查：

- full resolution；
- 50% zoom；
- A4 打印等效尺度；
- grayscale；
- contact-sheet thumbnail；
- 如果可用，做 color-blind simulation。

### 视觉检查项

- panel label 是否统一；
- title 是否简洁；
- axis label 是否可读；
- metric 和单位是否定义；
- legend 是否贴近数据；
- 是否有文字重叠；
- 是否把大表格当成主视觉；
- uncertainty 是否可见；
- visual hierarchy 是否对应 claim hierarchy；
- 颜色是否遵循 style ledger；
- Nature red 是否被滥用为装饰；
- graph blue 是否专用于 graph evidence；
- ASPR-Qwen purple 是否专用于 reviewer model；
- verifier orange 是否专用于 caveat / evidence / safety；
- 是否有无证据支撑的 AI decorative imagery。

### 图类型预期

- Fig.1-Fig.4：data-first evidence figures；
- Fig.5：visual synthesis，但必须可追溯到 CSV；
- Fig.6-Fig.7：rigorous data figures；
- Fig.8：top-journal method diagram；
- Fig.9：case storyboard，不是长截图；
- Fig.10：ablation data figure，module diagram 只做辅助。

### 输出

- `visual_layout_audit.csv`
- `figure_redline_notes.md`
- `contact_sheet_annotated.png`
- `priority_redraw_queue.csv`

### 停止条件

- 主图不再有 P0/P1 排版问题；
- 每张图不读 Methods 也能从 title、axes、caption 理解基本信息。

---

## Phase 6：复现性与代码审计

目的：确保论文可辩护、可复现。

### 检查项

- 每张图是否有生成脚本？
- 随机性是否有 seed？
- 输出是否有 manifest？
- 输入路径是否 hard-coded 到本地盘？
- API cache 是否记录？
- LLM / image generation 是否可复现或冻结？
- image2 panel 是否有结构化 prompt 和来源文本？
- pipeline-ready assumption 是否 machine-readable？

### 输出

- `reproducibility_matrix.csv`
- `script_to_output_map.csv`
- `external_dependency_audit.csv`
- `manual_intervention_log.csv`

### 每张图复现等级

- `fully_reproducible`
- `reproducible_from_cache`
- `partially_reproducible_with_external_model`
- `pipeline_ready_visual_only`

### 停止条件

- 每张图都有复现等级；
- 每个不可复现元素都有替代方案。

---

## Phase 7：模拟审稿人攻击

目的：提前写出 Nature 审稿人最可能提出的重大质疑。

### 三类 reviewer persona

#### 1. 定量方法审稿人

重点攻击：

- statistics；
- leakage；
- normalization；
- controls；
- uncertainty；
- overfitting；
- field/year confounding。

#### 2. 领域科学审稿人

重点攻击：

- scientific interpretation；
- novelty；
- venue contribution；
- case selection；
- biological/materials/AI domain plausibility；
- 是否过度解释图信号。

#### 3. AI/LLM 系统审稿人

重点攻击：

- ASPR-Qwen checkpoint；
- LLM-as-judge；
- ablation validity；
- evidence trace；
- hallucination control；
- generic LLM baseline 是否公平。

### 每类 reviewer 输出

- 10 条 likely major concerns；
- severity；
- 对应 figure；
- 现有 evidence；
- 处理方式：
  - fix；
  - caveat；
  - supplement；
  - remove；
  - pipeline-ready gap。

### 输出

- `reviewer1_quant_methods_report.md`
- `reviewer2_domain_science_report.md`
- `reviewer3_ai_systems_report.md`
- `reviewer_objection_response_matrix.csv`

### 停止条件

- 每个高概率审稿意见都有预备 response；
- 没有中心 Nature claim 只依赖不可验证的 pipeline-ready 层。

---

## Phase 8：决策板与下一轮迭代

目的：把审计发现转化为具体行动。

### 四个处理通道

- `Fix now for main figure`
- `Move to supplement`
- `Caption caveat only`
- `Defer as pipeline-ready gap`

### 每个 issue 必须记录

- owner；
- figure；
- severity；
- required input；
- fix command 或 edit path；
- verification command；
- expected output；
- stop condition。

### 输出

- `iteration_decision_board.csv`
- `next_iteration_goal.md`
- `resolved_issues.csv`
- `remaining_gaps.csv`

### 停止条件

- 下一轮迭代目标有限、明确、按优先级排序；
- 没有 vague issue。

---

## 逐图审计 Prompt

## Fig.1 审计 Prompt

核心问题：

> Fig.1 是否令人信服地定义了 graph perturbation 是一个可观察、可复现的现象？

潜在问题：

- 领域选择可能 cherry-picked；
- graph snapshot 看起来好看但量化解释不足；
- landmark selection 可能泄漏未来认可；
- node/edge filtering 可能制造人工结构；
- 可视密度可能掩盖 uncertainty；
- 颜色可能暗示不存在的分类。

必须检查：

- domain selection rules；
- works/topic/citation tables；
- missing years 或 sparse graph 是否标注；
- 图是否只是 anecdotal examples；
- Fig.2/Fig.3 的指标是否在 Fig.1 中被自然引出。

建议决策：

- 如果它能建立方法直觉且有 quality report 支撑，保留为主图；
- 如果过度依赖案例视觉，把部分 panel 转 supplement。

---

## Fig.2 审计 Prompt

核心问题：

> Fig.2 是否证明 publication-day perturbation indicators 在 controls 之外具有经验信号？

潜在问题：

- future outcome 可能 citation-biased；
- matched controls 太弱；
- reference closure 不完整；
- correlation 小但视觉强调过强；
- multiple testing 未处理；
- field/year imbalance 驱动结果。

必须检查：

- `fig2_indicator_future_corr_bootstrap.csv`；
- `fig2_matched_controls.csv`；
- `fig2_reference_closure_report.csv`；
- confidence interval 和 sample size；
- null / negative controls 是否可见。

建议决策：

- 主 claim 应是 “有可测信号”，不要写成 “能预测所有未来创新”。

---

## Fig.3 审计 Prompt

核心问题：

> Fig.3 是否合理支持 learned composite perturbation score？

潜在问题：

- weight search 过拟合；
- fold split 不合理；
- performance 被单一领域驱动；
- weights 不稳定；
- baseline comparisons 太弱；
- nonlinear upper bound 被过度解释。

必须检查：

- cross-validation summary；
- fold weights；
- leave-one-domain-out diagnostics；
- learned score vs equal-weight / simple baseline；
- model selection 是否先验固定。

建议决策：

- 如果 weights 不稳定，强调 robust multi-indicator ensemble，而不是精确解释每个 coefficient。

---

## Fig.4 审计 Prompt

核心问题：

> Fig.4 是否可信地把 graph evidence 连接到 human peer-review judgments？

潜在问题：

- peer-review text 泄漏进 agent input；
- semantic matching 过宽；
- LLM judge parse failures；
- review sample 不代表总体；
- stance agreement 掩盖 missing scientific points；
- overclaiming rate 太高。

必须检查：

- peer review 是否从 dossier 中排除；
- semantic claim matches 和 no-match examples；
- strict recall vs soft recall；
- aspect-level coverage；
- overclaiming 和 contradiction metrics；
- “human-like” 是否被过度表述。

建议决策：

- 作为 validation bridge 保留，但必须显式承认 recall 和 overclaiming 的限制。

---

## Fig.5 审计 Prompt

核心问题：

> Fig.5 是否负责任地把 graph evidence 转换成 forecast 和 mechanism interpretation？

潜在问题：

- AI 生成视觉让不确定结果看起来太确定；
- forecast categories 可能手工挑选；
- mechanism cards 可能过度预测；
- 图无法从 CSV 复现；
- image2 美感掩盖证据薄弱。

必须检查：

- `fig5_panel_text.json`；
- derived forecast CSVs；
- 每个视觉 claim 是否有来源行；
- caption 是否说 forecast/handoff，而不是 confirmed discovery。

建议决策：

- 如果可追溯，保留为 synthesis figure；
- 如果不可追溯，降级 claim 或增加 source notes。

---

## Fig.6 审计 Prompt

核心问题：

> Fig.6 是否通过暴露稳健性与边界条件增强方法可信度？

潜在问题：

- proxy perturbation probes 被误解为 full reruns；
- failure modes 是 heuristic；
- panel 与 Fig.2/Fig.3 重复；
- threshold 任意；
- sparse domain 中 robustness 被夸大。

必须检查：

- panel metadata；
- source audit；
- proxy labels 是否写入 caption；
- retention threshold；
- failure modes 是否 evidence-derived；
- Fig.6 是否自然承接 Fig.1-Fig.5。

建议决策：

- 如果 caveats 明确，作为 reliability bridge 保留。

---

## Fig.7 审计 Prompt

核心问题：

> Fig.7 是否负责任地进入 venue-level interpretation，而不暗示因果 superiority？

潜在问题：

- Nature Portfolio claim 被理解为宣传或因果；
- venue family mapping 噪声；
- field-year normalization 稀疏；
- strict interval separation 不成立；
- venue differences 反映 article type / team size / reference count；
- missing metadata 影响 ranking。

必须检查：

- `figure_quality_report.json`；
- VCI ranking；
- top-K enrichment；
- confounder audit；
- venue family mapping audit；
- Nature point estimate vs strict CI separation。

建议决策：

- 如果 strict interval separation 不成立，写成 point-estimate supported with interval caveat。

---

## Fig.8 审计 Prompt

核心问题：

> Fig.8 是否清楚说明 ASPR 是系统，而不是 generic LLM？

潜在问题：

- ASPR-Qwen 还处于 pipeline-ready 时，被画得像完全验证；
- graph agent 和 Qwen 两条路径视觉不平衡；
- fusion/verifier 看起来像装饰；
- evidence trace 不突出；
- 方法图被 caption 写成结果图。

必须检查：

- `panel_text.json`；
- `flow_spec.json`；
- 主图；
- dual-path story；
- final output fields 是否和 Fig.9/Fig.10 对齐；
- 颜色是否符合 style ledger。

建议决策：

- 作为 method framework 保留，但 caption 必须区分 architecture 和 empirical validation。

---

## Fig.9 审计 Prompt

核心问题：

> Fig.9 展示的是 ASPR 真实运行，还是只是 illustrative storyboard？

潜在问题：

- ASPR-Qwen output 是 assumed；
- case 被 cherry-picked；
- trace line numbers 脆弱；
- final review 过度 polished；
- verifier flags 不够突出。

必须检查：

- case manifest；
- evidence trace；
- agent output；
- fusion output；
- Qwen assumption 是否可见；
- final review 与 peer review overlap / missing points。

建议决策：

- 如果真实 trace 清楚且 assumption 明确，可作为 auditable case 保留。

---

## Fig.10 审计 Prompt

核心问题：

> Fig.10 是否证明 ASPR module contribution，而不只是讲了一个合理的模块故事？

潜在问题：

- ablations 是估计而非真实 rerun；
- LLM-as-judge 替代 human preference；
- full ASPR metrics 来自 Fig.4，而 ablation 是 synthetic；
- module deltas 太平滑、太像手工假设；
- error taxonomy 来自阈值而非人工标签。

必须检查：

- ablation CSV source labels；
- human preference / LLM judge table；
- error taxonomy thresholds；
- generic LLM-only baseline 是否视觉分离；
- 每个 ablated module 是否映射到 Fig.8/Fig.9。

建议决策：

- 如果标签明确，可作为 pipeline-ready ablation figure 保留；
- 强 Nature claim 前必须替换为真实 ablation 或 human preference。

---

## 横向一致性检查

### 术语统一

推荐统一使用：

- `graph-perturbation analysis`
- `publication-day perturbation signal`
- `learned perturbation score`
- `peer-review validation`
- `venue-level contribution`
- `ASPR graph agent`
- `ASPR-Qwen`
- `fusion/verifier`
- `evidence trace`
- `pipeline-ready gap`

高风险措辞，除非精确定义，否则避免：

- `proof`
- `causal venue effect`
- `Nature creates innovation`
- `ASPR replaces peer review`
- `human-level reviewer`
- `hallucination-free`
- `fully automated acceptance decision`

### 视觉统一

统一色彩语义：

- Nature corpus / Nature Portfolio：深红 / 酒红；
- ASPR graph agent：蓝色；
- ASPR-Qwen：紫色；
- evidence / verifier / uncertainty：橙色；
- fusion / final output：黑色或深灰；
- neutral context：灰色。

检查：

- 同一颜色是否跨图表达同一含义；
- panel label 是否一致；
- caption 语气是否 evidence-first；
- 是否有某张图显著偏离整套视觉系统。

### claim 强度阶梯

推荐措辞从强到弱：

- `directly observed`
- `validated in cached corpus`
- `supported by controlled association`
- `consistent with`
- `pipeline-ready estimate`
- `illustrative only`

每张图必须停留在它证据允许的强度层级。

---

## Nature 投稿准备度评分

每轮审计结束后，每张图按 0-5 打分。

### 评分维度

1. 数据 provenance；
2. 可复现性；
3. 统计可信性；
4. 视觉清晰度；
5. 叙事必要性；
6. claim/caption 对齐；
7. reviewer objection readiness；
8. supplement readiness。

### 等级解释

- `0-1`：不能投稿；
- `2`：内部探索；
- `3`：pipeline-ready，但必须 caveat；
- `4`：强 main-figure candidate；
- `5`：Nature-level main-figure candidate。

### 输出

- `nature_readiness_scorecard.csv`

### 硬规则

> 如果一张图存在未解决 P0/P1、未标注 pipeline-ready component，或 caption claim 强于证据，则不能评为 Nature-level main-figure candidate。

---

## 单轮审计停止条件

一轮审计只有在以下条件满足后才能结束：

- Fig.1-Fig.10 所有主图图片都已视觉检查；
- 主要中间数据表都已检查或抽样总结；
- 每张图都有 data-risk classification；
- 每张图都有 visual-risk classification；
- 每个 caption 都与证据核对；
- 所有 P0/P1 已修复、降级、转 supplement 或记录为 pipeline-ready gap；
- final checklist 和 caption drafts 已更新；
- verification commands 已运行并记录。

---

## 整个 Nature-prep 周期停止条件

整个长期迭代只有在以下条件满足后才能认为完成：

- 没有主图存在未解决 P0/P1 风险；
- 没有中心 manuscript claim 依赖 pipeline-ready assumption；
- 所有 pipeline-ready gaps 要么解决，要么不再支撑中心 Nature claim；
- 所有图可复现，或有冻结且清楚的生成 provenance；
- 三类 reviewer persona 的 response matrix 已完成；
- 图、caption、正文、supplement 之间 claim scope 一致。

---

## 推荐下一轮 `/goal` 文本

可以直接复制下面这段作为下一轮长期目标：

```text
/goal 运行 Fig.1-Fig.10 Nature 投稿级持续迭代审计流程。
使用 outputs/kg_perturbation_final_assembly/fig1_fig10_nature_submission_iterative_audit_workflow.md 作为协议。
审计当前 Fig.1-Fig.10 的图片结果和所有关键中间数据，重点检查数据有效性、统计逻辑、可信性、复现性、视觉排版、术语统一、caption 对齐和 Nature 审稿风险。
每张图都必须产出 data forensics、visual audit、claim-to-evidence map、reviewer-objection notes 和 readiness score。
不要静默修饰过强 claim；必须修复证据、软化 claim、移到 supplement，或记录为 pipeline-ready gap。
停止条件：完成一轮完整审计，输出 decision board、updated gaps、caption edits、reviewer response matrix 和 verification log。
```


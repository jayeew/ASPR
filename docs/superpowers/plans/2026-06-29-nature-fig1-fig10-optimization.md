# Fig.1-Fig.10 Nature 投稿级持续优化计划

> 本文件是可反复执行的中文任务规划，用于把当前 Fig.1-Fig.10 从“已生成图组”推进为 Nature 投稿级实验包。它不是一次性 TODO，而是可以作为长期 `/goal` 的迭代协议。每一轮都必须留下可复查输出：数据取证、视觉审计、claim-to-evidence、caption 修订、reviewer response matrix、replacement gates 和 verification log。为避免无限循环，默认最多 6 轮主迭代 + 1 轮最终小修；第 6 轮必须收敛为完成、降级、转 supplement 或明确 pipeline-ready gap。中途不暂停让用户选择路线；所有软性取舍按“可信度优先，其次美观，其次 panel 数量”的默认规则自动执行。

## 0. 总目标

把 Fig.1-Fig.10 统一成一条审稿人能读懂、也能追问到底的证据链：

- Fig.1-Fig.3：graph-perturbation measurement 与 learned graph prior；
- Fig.4：peer-review validation；
- Fig.5：forecast / mechanism handoff；
- Fig.6：稳健性与边界条件；
- Fig.7：venue-level scientific interpretation；
- Fig.8：ASPR 系统架构；
- Fig.9：真实 case 的 auditable ASPR run；
- Fig.10：ASPR 模块贡献、metric sensitivity 与 replacement gates。

最终标准不是“图能看”，而是：每个主 claim 都有可追溯证据，每个强 claim 都不依赖 pipeline-ready assumption，每个未完成实验都被降级、转 supplement 或明确记录为 pipeline-ready gap。

## 1. 当前权威输入

### 1.1 审计协议

- `outputs/kg_perturbation_final_assembly/fig1_fig10_nature_submission_iterative_audit_workflow.md`

### 1.2 当前主图与组装输出

- `outputs/kg_perturbation_final_assembly/fig1_fig10_contact_sheet.png`
- `outputs/kg_perturbation_final_assembly/fig1_fig10_caption_drafts.md`
- `outputs/kg_perturbation_final_assembly/fig1_fig10_cross_figure_audit.csv`
- `outputs/kg_perturbation_final_assembly/fig1_fig10_final_checklist.csv`
- `outputs/kg_perturbation_final_assembly/fig1_fig10_pipeline_ready_gaps.csv`
- `outputs/kg_perturbation_final_assembly/fig6_fig10_multi_round_consistency_report.md`

### 1.3 Nature 审计与优化包

- `outputs/nature_submission_audit/iteration_latest/audit_iteration_report.md`
- `outputs/nature_submission_audit/iteration_latest/iteration_decision_board.csv`
- `outputs/nature_submission_audit/iteration_latest/updated_gaps.csv`
- `outputs/nature_submission_audit/iteration_latest/caption_edits.md`
- `outputs/nature_submission_audit/iteration_latest/reviewer_objection_response_matrix.csv`
- `outputs/nature_submission_optimization/iteration_latest/submission_claim_boundaries.md`
- `outputs/nature_submission_optimization/iteration_latest/experiment_upgrade_protocol.md`
- `outputs/nature_submission_optimization/iteration_latest/fig6_fig10_priority_backlog.csv`

### 1.4 Fig.10 关键新增证据

- `outputs/kg_perturbation_fig10/fig10_generic_llm_baseline_results.csv`
- `outputs/kg_perturbation_fig10/fig10_generic_llm_baseline_manifest.json`
- `outputs/kg_perturbation_fig10/fig10_generic_llm_same_rubric_results.csv`
- `outputs/kg_perturbation_fig10/fig10_generic_llm_same_rubric_summary.csv`
- `outputs/kg_perturbation_fig10/fig10_generic_llm_same_rubric_manifest.json`
- `outputs/kg_perturbation_fig10/fig10_evidence_provenance.csv`
- `outputs/kg_perturbation_fig10/fig10_replacement_gates.csv`

当前 Fig.10 的重要边界：qwen3 generic LLM baseline 在 proxy composite 下看起来强，但已完成的 48/50 可评估样本 same-rubric Fig.4 matcher（另有 2 个 zero-peer-point exclusion）显示其几乎不覆盖真实 peer-review 语义点。该结果支持 metric-sensitivity diagnosis，不支持 ASPR superiority。

## 2. 审稿级原则

1. 证据先于叙事：caption、panel title、正文 claim 必须能追溯到 CSV/JSON/manifest/quality report/脚本。
2. proxy 必须显性化：proxy、LLM-as-judge、assumed output、pipeline-ready estimate 不能被写成真实完成实验。
3. 因果 claim 默认禁止：除非有实验设计或强可识别性证据，否则 venue、module、graph signal 都只能写 association/contribution/diagnosis。
4. 不无限重做：每个问题必须落到 fixed、caption caveat、supplement、pipeline-ready gap 四类之一。
5. Fig.6-Fig.10 是投稿风险最高区域，每轮必须优先复查。
6. 迭代轮数默认最多 6 轮主迭代 + 1 轮最终小修；每一轮必须减少未决风险或明确降级，不能重复生成同类报告来代替实验进展。
7. 不向用户索要中途选择：除硬 blocker 外，route、panel 融合、claim 降级、Extended Data 转移、旧图删除和下一轮优先级都由脚本与质量门自动决定。

## 3. 自动迭代轮次

每一轮都执行 `build -> audit -> reflect -> fix-list -> auto-continue`。输出目录使用 `outputs/nature_iter/r0/` 到 `outputs/nature_iter/r6/`；一轮通过后删除上一轮旧 PNG/SVG/PDF，只保留 CSV、JSON、manifest、quality report、review notes 和 fix-list。

- Round 0 baseline audit：冻结当前 Fig.1-Fig.10 问题清单、数据来源、claim scope、style ledger。
- Round 1 data/claim repair：优先修 Fig.1 landmark 时间窗、Fig.2 真实 Fig.1 裁图接线、Fig.5 AI 热点数据门、Fig.4/Fig.10 claim 降级门。
- Round 2 layout redesign：压缩低密度 panel，修重叠越界，统一 Fig.1-Fig.10 色调、字号和 caption 语气。
- Round 3 evidence strengthening：核查 landmark、AI 热点、peer-review claims；无法核查的强表述自动降级。
- Round 4 density pass：以 5-10 秒可读性、panel 负载、信息密度和跨图叙事顺序重排主图。
- Round 5 targeted repair：只修 Round 4 未通过项，不重动已通过图。
- Round 6 final assembly and convergence：生成最终图集、caption draft、claim ledger、strict evidence report、submission readiness checklist；未完成项必须完成、降级、转 supplement 或列为 pipeline-ready gap。
- Final patch：只修 typo、轻微 label overlap、导出路径和 manifest 不一致，不再重构数据或设计。

## 4. 每轮执行顺序

### Round A：证据盘点

- 检查 10 张主图是否存在、是否为最新输出；
- 检查每张图的 source CSV/JSON、quality report、run manifest；
- 记录每个核心 claim 的证据路径；
- 标记缺失、过期或间接证据。

输出：`figure_file_index.csv`、`intermediate_data_index.csv`、每图 data forensics。

### Round B：数据和统计审计

- 样本定义、排除规则、时间窗口、field/year controls；
- leakage、citation bias、venue bias、post-publication signal；
- bootstrap / CI / fold stability / leave-one-domain；
- module ablation 是否真实重跑，还是估计。

输出：`methodology_failure_modes.csv`、每图 data risk table。

### Round C：视觉和版面审计

- 图是否有过度美化或视觉暗示强于证据；
- panel label、颜色、字体、legend、axis、注释是否统一；
- Fig.6/7/10 是否保持严谨数据图；
- Fig.8/9 是否保持顶刊/顶会风格算法与案例图；
- 是否存在表格堆砌、遮挡、字号过小、caption 与图不一致。

输出：`visual_layout_audit.csv`、每图 visual audit。

### Round D：claim-to-evidence 与 reviewer attack

- 为每个 claim 标注支持强度：directly supported、supported with assumptions、proxy supported、pipeline-ready、unsupported；
- 生成 quant methods、domain science、AI systems 三类 reviewer objection；
- 为每个 objection 写出可用 response 或下一步实验。

输出：`claim_to_evidence_map.csv`、`reviewer_objection_response_matrix.csv`。

### Round E：决策与重建

- fixed：直接改图、脚本、caption 或数据；
- caption caveat：保留图，但在 caption/正文显式降调；
- supplement：主文移除强 claim，只在补充展示；
- pipeline-ready gap：保留为未来实验缺口，不能支撑摘要或主结论。

输出：`iteration_decision_board.csv`、`updated_gaps.csv`、`caption_edits.md`、`experiment_upgrade_protocol.md`。

## 5. 逐图优化任务

### Fig.1：graph-perturbation 现象定义

核心问题：Fig.1 是否只是漂亮案例，还是可复现的 graph-perturbation 观察入口？

必须检查：domain selection、landmark 选择时间、topic/citation graph 过滤规则、graph density、Fig.2/Fig.3 指标承接。

可写：代表性结构提供直觉和测量动机。

不能写：这些示例证明普适创新规律。

### Fig.2：publication-day indicators 与未来信号

核心问题：graph indicators 是否在控制项下仍与后续科学信号相关？

必须检查：future outcome 是否 citation-biased、matched controls、reference closure、多重比较、field/year imbalance。

可写：存在可测 association。

不能写：确定性预测未来重要性。

### Fig.3：learned graph-prior score

核心问题：learned score 是否是可靠 ranking proxy，而不是过拟合权重故事？

必须检查：CV summary、fold stability、leave-one-domain、equal-weight baseline、nonlinear upper bound。

可写：multi-indicator graph prior。

不能写：权重是创新机制公式。

### Fig.4：peer-review validation

核心问题：graph-derived evidence 是否与真实 peer-review concerns 部分对齐？

必须检查：peer-review leakage、strict/soft recall、semantic match false positives、no-match examples、overclaiming。

可写：partial structured alignment。

不能写：完整复现 peer review。

### Fig.5：forecast 与 mechanism handoff

核心问题：Fig.5 是否负责任地做 synthesis，而不是把不确定结果画成确定发现？

必须检查：source CSV、image prompt、每个机制 claim 的来源、generated visual 是否引入 unsupported content。

可写：候选机制和机会。

不能写：生成图像证明未来发现。

### Fig.6：robustness 与 boundary conditions

核心问题：Fig.6 是否增强 Fig.1-Fig.5 的方法可信度？

当前边界：部分 panel 是 cached/proxy probes，Panel G 是 cache-level indicator rerun；它们还不是 online OpenAlex retrieval + full graph extraction rerun。

下一步：如需强主文 claim，重跑 OpenAlex retrieval 和完整 graph extraction perturbations。

### Fig.7：venue-level interpretation

核心问题：Fig.7 是否把方法转向 venue-level scientific interpretation，而不暗示因果 superiority？

当前边界：Nature Portfolio 有最高 aggregate VCI 点估计，但 strict interval separation 和 pairwise uncertainty 仍需 caveat。

下一步：补足 matched controls、per-paper intensity sensitivity 和严格区间分离证据。

### Fig.8：ASPR architecture

核心问题：Fig.8 是否清楚提出最终方法 ASPR，而不是把 architecture 写成 performance evidence？

当前边界：graph agent、ASPR-Qwen、fusion、verifier 是系统定义；真实性能由 Fig.9/Fig.10 评估。

下一步：保证每个模块都能在 Fig.9/Fig.10 找到对应证据或 replacement gate。

### Fig.9：single auditable ASPR case

核心问题：Fig.9 展示的是一次真实可审计运行，而不是代表性 checkpoint proof。

当前边界：case trace 真实，但 ASPR-Qwen lane 仍是 assumed pipeline-ready placeholder。

下一步：用 checkpoint-generated ASPR-Qwen output 替换 assumed lane，并增加更多 cases。

### Fig.10：module contribution 与 metric sensitivity

核心问题：Fig.10 是否能证明 ASPR 的性能来自模块组合，而不是单一 LLM 或指标伪影？

当前证据：

- full ASPR 使用 Fig.4 真实 peer-review 样本指标；
- qwen3 generic LLM baseline 已实跑；
- proxy composite 下 qwen3 看起来强；
- 已完成的 48/50 可评估样本 same-rubric Fig.4 matcher 显示 qwen3 几乎不覆盖真实 peer-review 语义点；
- replacement gate 允许 pipeline figure 展示，但不允许 Nature strong claim。

当前边界：Fig.10 是 pipeline-ready ablation + metric-sensitivity figure，不是 ASPR superiority proof。

下一步：冻结 same-rubric manifest 与 zero-peer-point exclusion table，真实重跑 disabled-module ablations，收集 blinded human preference，并保存 checkpoint-generated ASPR-Qwen outputs。

## 5. Fig.6-Fig.10 优先行动队列

1. Fig.10：保持 qwen3 proxy-vs-same-rubric discrepancy 的写法和证据边界，并冻结 same-rubric exclusion table。
2. Fig.10：将 module ablations 从 pipeline estimate 升级为真实 rerun。
3. Fig.10：用 blinded human preference 替换 LLM-as-judge preference bars。
4. Fig.9：替换 assumed ASPR-Qwen lane。
5. Fig.6：用完整 graph extraction rerun 替换 cached/proxy robustness 层。
6. Fig.7：补足 strict interval separation 或把 claim 固定为 point-estimate contribution。
7. Fig.8：确保 architecture-only boundary 在 caption 和图中可见。

## 6. 重建命令

```bash
python3 experiments/kg_perturbation_fig10/build_fig10_same_rubric_baseline.py
python3 experiments/kg_perturbation_fig10/build_fig10_ablation.py
python3 experiments/kg_perturbation_final_assembly/build_final_assembly.py
python3 experiments/nature_submission_audit/build_nature_submission_audit.py --iteration-id latest
python3 experiments/nature_submission_optimization/build_nature_optimization.py --iteration-id latest
```

如本轮涉及 Fig.6/Fig.7/Fig.9，也运行：

```bash
python3 experiments/kg_perturbation_fig6/build_fig6_robustness.py
python3 experiments/kg_perturbation_fig7/build_fig7_venue_contribution.py --max-fetch 0
python3 experiments/kg_perturbation_fig9/build_fig9_case.py
```

## 7. 验证命令

```bash
python3 -m unittest   tests.test_fig10_same_rubric_baseline   tests.test_fig10_generic_baseline   tests.test_fig10_ablation   tests.test_nature_submission_audit   tests.test_nature_submission_optimization   tests.test_final_assembly -v

git diff --check
```

若修改 Fig.6/Fig.7/Fig.9，也运行对应测试：

```bash
python3 -m unittest tests.test_fig6_robustness tests.test_fig7_venue_contribution -v
```

## 8. 每轮停止条件

一轮可以停止，但不能宣称总目标完成，只有当以下条件满足；同时整个长期循环最多 6 轮：

- 10 张图的图片、关键中间数据、claim、caption 都已审计；
- 所有 P0/P1 都进入 fixed、caption caveat、supplement 或 pipeline-ready gap；
- Fig.6-Fig.10 的特殊风险已被显式记录；
- 所有修改后的输出已重建；
- focused tests 和 `git diff --check` 通过；
- final answer 中明确说明哪些 claim 仍不能写入 Nature 摘要或主结论。

整个长期目标只有在以下条件都被真实证据证明时才可完成：

- 没有中心 manuscript claim 依赖 pipeline-ready assumption；
- Fig.6 的 full graph rerun 或相应降级完成；
- Fig.7 的 venue claim 有足够不确定性处理或已降级；
- Fig.9 的 ASPR-Qwen lane 来自真实 checkpoint output；
- Fig.10 的 module ablations、generic baseline、human preference 和 checkpoint evidence 都通过 replacement gates；
- 所有图、caption、正文和 supplement 的 claim scope 一致。

## 9. 推荐长期 `/goal`

```text
/goal 根据最新审计结果，以发表 Nature 论文为最终目标，持续迭代优化 Fig.1-Fig.10，特别是 Fig.6-Fig.10；不要无限循环，最多 6 轮主迭代 + 1 轮最终小修。每轮必须检查图片结果和相关中间数据，定位数据、逻辑、可信性、复现性、claim-to-evidence、caption、视觉排版和审稿风险问题；能修复则修复，不能修复则降级 claim、转 supplement 或记录为 pipeline-ready gap。中途不暂停让用户选择路线，除硬 blocker 外一路按默认动作执行。停止条件：完成本轮审计输出、重建相关图和报告、运行 focused tests 与 git diff --check，并给出下一轮优先级。
```

# ASPR–GEAR Graph Guidance 精简重构计划

本文件冻结 dev100 实测修订后的实施边界。ASPR Score 只控制固定资源内的检索几何，结构画像只生成验证任务，citation topology 只决定检索位置；Graph 不能创建审稿点、不能作为审稿证据，最终实质修正必须由截止合规且已验证的 `RelationCard` 驱动。

## 已确认基线

- `full - score_only` 的 `+1.70 relations/paper` 是 direct-fetch 预算未匹配时的描述性增益，不是因果效率结论。
- `score_only` 与 Neutral 几乎等价，现有 signed tension 错把传播 percentile 当成 novelty 正负轴。
- dev100 的 `feature_coverage=1.0` 与 `EF0197` 全缺失矛盾；coverage 必须由 missing features 推导。
- 旧 `search_terms` 污染严重且 unique-prior yield 为零，退出执行路径。

## 固定数据流

```text
Calibration / precomputed artifact
  -> GraphRuntimePacketV1
  -> Graph-blind Reviewer (+ optional graph-blind Qwen)
  -> text-only Fusion
  -> graph-blind ScientificSearchFrame
  -> GraphGuidancePlanV1
  -> mission-aware retrieval under ResourceLedgerV1
  -> verified RelationCard
  -> ReviewCorrectionEventV1
  -> verification and compilation
```

运行态不区分查表/推理或开发/正式等级。合法 packet 一律使用；profile 或 topology 缧失只跳过对应能力，整个 packet 缺失才是 `LIMITED + graph_unavailable`。V4 仅作只读迁移，旧 `search_terms` 不激活。

## MVP 资源与控制器

每篇固定 cap：provider search 8、direct fetch 8、neighbor expansion 2、full-text candidates 12、relation classifications 12。logical request 消耗预算，network attempt 用于成本报告，cache hit 不返还 logical slot。

令 `q=score/100`，`r=1-|missing|/16`，`q_effective=.5+r(q-.5)`。八个 provider slots 是每篇上限；v5 按可执行 novelty claims 分配 `min(8, 3 × claim_count)` 个槽位：

```text
Q = min(8, 3 × claim_count)
remote_slots = clamp(round(Q * (.25 + .50 * q_effective)), 1, Q-1)
local_slots = Q - remote_slots
```

高 ASPR 增加 remote/cross-field falsification，低 ASPR 增加 local nearest antecedent，同时始终保留另一侧 rescue；不设置 novelty 方向阈值，总资源不随 Score 增加。

结构画像仅允许三组任务：reference-structure diversity（EF0017/0309/0312/0315/0318）、historical depth（EF0052）、terminology emergence（EF0240）。EF0238 仅能调整 seed-pool stop rule，其余 opportunity/control 字段不得生成 novelty mission 或结论。

Topology MVP 仅使用现有能力：bibliographic-coupling seeds、direct fetch、backward references、forward citations和一跳扩展。每个 seed 最多分配两个 claims，每个 claim 最多两个 seeds；分配必须经过 SearchFrame/target-span 与 seed metadata 的确定性 relevance 检查。

## 审计与晋级

检索不足只降低 confidence、设置 `insufficient_coverage` 并软化措辞。只有 verified relation 能改变 direction，并必须产生显式 correction event；无 event 时 final direction 等于 Graph-blind Reviewer direction。ASPR claim assessment 仅使用 `NOT_CHALLENGED/CHALLENGED/REFINED/INCONCLUSIVE/NOT_APPLICABLE`。

正式消融为 Neutral、Score-only、Score+Profile、Full、Shuffled Score、Shuffled Profile、Random Matched Topology。所有 variant 使用相同 resource caps、冻结 Agent branch，shuffle 使用按年份/可用 domain 分层的 derangement。Primary KPI 是每 100 logical requests 的 claim-relevant verified relation yield、relation-to-material-correction rate 和 useful-correction rate；guardrail 包括截止泄漏、Graph-only correction、wrong-paper contamination、raw-score mutation和资源守恒。

dev100 始终是 `development_non_confirmatory=true`。Score controller 通过行为单调性和资源守恒即可上线，但不能据此声称质量提升；profile/topology 需要 matched comparator 的 dev CI 下界大于零才从 shadow 晋级；最终审稿有效性结论必须在新 holdout 上验证。

## 2026-08-24 policy v4 实施校正

- Reviewer direction 与 verification status 在生成边界强制分离。`uncertain`/`not_discussed` 与已给出的 directional points 冲突时，按 supporting/limiting 内容修复为 positive/mixed/negative；显式 positive/mixed/negative 不被覆盖。
- `ScientificSearchFrame` 已前移到 Planner 之前并复用 PriorArt 缓存；Graph seed 不再依据 reviewer 修辞做 claim alignment。
- seed 分配要求至少两个 claim-specific scientific tokens，过滤 generic scientific words；每个 claim 最多一个 seed、每个 seed 最多一个 claim，避免重复 direct fetch 挤占候选多样性。
- Graph direct fetch 替换同一 claim 的一个 provider query slot，不再作为隐藏的追加预算。正常分类预留两个全局 slot 给一跳 topology traversal。
- 高 `q_effective` 的 supporting claim 使用 purpose-first residual-novelty rescue；低分仍以 local antecedent falsification 为主。`q_effective >= 0.60` 时沿历史 coupling seed 的 forward citations 搜索传播桥接，否则沿 references 检查 antecedent chain。该阈值只选择 traversal direction，不产生 novelty polarity。
- 真实 direct antecedent 必须报告 essential-facet coverage，并通过一次独立反证式复核；只有随后生成的 verified `ReviewCorrectionEventV1` 可以修改最终方向。
- 原 novelty limitation 只有在至少两篇独立、截止合规的 paired comparisons 均含 common/difference dimensions，且不存在 direct/partial antecedent 或 conflict 时，才能收敛为 bounded residual support；该 consensus correction 可审计地更新方向。
- dev100 topology 已从冻结 context 重建：100/100 papers、794/794 seeds 均具有非零 shared-reference count、标题和年份；旧 `topology_metadata_incomplete_v4_migration` 已替换为 `topology_coupling_rebuilt`。

## 2026-08-24 policy v5 资源执行校正

- Planner 不再把不可执行的八个槽位全部写入计划。每个 claim 分配三个实际可执行槽位，并受每篇八槽上限约束；local/remote 的整数配额在 claims 间按最大余数法精确守恒。
- EvidenceSupervisor 的 per-claim `RetrievalBudget.normal_max` 直接取 GuidancePlan 的实际分配，不再被旧的固定 `normal_per_claim_max=2` 截断。
- topology direct fetch 与其一跳 neighbor traversal 都替换普通 provider query 槽位，不能再以隐藏追加资源制造 Graph 增益。
- `controller_state.executable_query_slots` 显式记录本篇真正可执行的槽位数，计划分配总数必须与之相等。

## 2026-08-24 policy v6 全链预算闭合

- 确定性的 paper-span 检查不再消耗检索 action budget；`total_actions_max` 只约束模型、检索、拓扑和稳定性工作。
- guided 模式不再隐式执行未规划的 manuscript-citation direct fetch。
- counterfactual mission 预先占用并替换一个 normal slot；没有该 mission 的 claim 不再机械执行反证查询。
- Graph seed anchor 与最终保留候选分开缓存，确保已预留的一跳 traversal 从真实 claim-aligned seed 出发。
- 当 claim 只有两个槽位时 topology 只做 direct fetch、不再追加 traversal，至少保留一个 Score-controlled 检索动作；三个以上槽位才允许一跳扩展。

## 2026-08-24 policy v7 执行修复

- 修复 Graph seed anchor 被错误写入 legacy citation 分支的问题；一跳 traversal 现在稳定使用当前 claim 的真实 Graph seed。
- Planner 已预留的 counterfactual mission 无论点评 severity 都会执行；没有 mission 时不执行，避免预留槽位闲置。
- stability 与 span verification 均为确定性零成本动作；普通 strengths/weaknesses 不再经历无意义的 stability sweep。

## 2026-08-24 policy v8 邻居查询身份修复

- neighbor query ID 现在包含 `citation_neighbor` 与 traversal direction，不再与同一 seed 的 `graph_seed` direct-fetch query 发生 EvidenceStore 键冲突。
- 该冲突过去会把局部 topology action 错误降级为 retrieval gap，并跳过其后已预留的 counterfactual；现在两者均可按计划完成。

## 2026-08-24 policy v9 移除未规划扩展

- 有 GuidancePlan 时，不再运行旧的 `prior_search_unresolved` generic citation expansion；只有显式 topology mission 可以消耗 neighbor slot。
- 因此 neutral/score-only 不会偷偷获得 neighbor 扩展，Full 也不会在已规划 traversal 之外追加搜索。

## 2026-08-24 policy v10 失败槽位记账

- cutoff 或内容规范化后无法形成 anchor 的 topology mission 仍消耗其已预留 logical neighbor slot，但不增加 network attempt。
- 这实现了“失败请求消耗 logical 预算、成本只记录真实网络尝试”，并使各 variant 的计划资源可逐篇精确重建。

## 2026-08-24 policy v11 Coverage 快照

- normal、topology、counterfactual 逐步扩展同一 claim 的 coverage 时，首个快照使用稳定 `COV:<id>`，后续变化快照使用内容哈希后缀。
- EvidenceStore 继续严格 append-only；不再因合法的 coverage 演化触发 overwrite rejection 或双重降级计数。

## 2026-08-24 policy v12 本地去重请求记账

- counterfactual 若在本地被判定与已有 intent 重复，不产生 network attempt，但仍消耗已规划的 logical provider slot。
- 这类任务保留 `contrastive_query_coverage_gap` advisory，同时满足计划槽位守恒，避免把本地去重误计为 Graph 节省或 variant 资源差异。

## 2026-08-24 policy v13 Full-text cap 记账

- 若先前任务已用尽本篇 full-text candidate cap，后续已规划 query 在 provider 调用前停止；它仍消耗 logical provider slot，network attempt 保持 0。
- 由此 cutoff anchor 缺失、本地 intent 去重和 full-text cap 三类本地短路均遵循同一 logical/network 双账规则。

## 2026-08-24 policy v14 对称方向共识

- 单篇 partial/building-block RelationCard 只降低置信度并收窄措辞，不再把 paper-level positive 自动改成 mixed。
- 方向降级要求一条 independently verified、facet coverage ≥0.9 的 direct antecedent，或至少两篇独立、含 common/difference dimensions 的 partial/building-block relations。
- 该门槛与 residual rescue 的双独立来源门槛对称，降低单个分类误差对全篇方向的放大。

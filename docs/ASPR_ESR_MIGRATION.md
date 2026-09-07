# ASPR-ESR in-place migration

> **文档状态（2026-09-07）：历史设计／历史实验记录。** 下文的“当前”、
> “最终”、运行命令、样本数量和验收条件均属于该文档原版本，不代表当前默认系统。
> 最新运行口径为共享全文 claims、独立 GEAR / 原生 Claim Graph 分析及联合创新解释，
> 见[当前架构](module_architecture.md)。保留本页用于方法和结果溯源；旧结果不能直接证明新版系统有效，
> 旧接口或本地资产也不保证仍可运行。新 Fig.1–Fig.10 规划尚未冻结。

The runtime remains the existing `gear` package. No parallel version package or
legacy runtime has been introduced.

## Runtime changes

- `load_config()` is lightweight by default; `validate_assets=True` is reserved
  for explicit asset validation.
- The primary reviewer is now `CodexAgentReviewer` and receives only PaperIR,
  stable paper spans, and the paper-specific rubric.
- ASPR-Qwen is optional and uses the same Graph-blind branch contract.
- `GraphPriorService` lazily adapts the frozen Full-text16 HGB to the public
  `GraphPriorResult`. Detailed features remain in `graph_prior_audit.json`.
- Fusion creates canonical candidate points. Graph data may mark tension but
  cannot create a point.
- `EvidenceSupervisor` runs a bounded normal → counterfactual → citation policy
  and records every action with reason and budget deltas.
- The final compiler reads validated canonical state and rebuilds its summary;
  it does not preserve deleted point text.

## Compatibility

`ReviewBundle` retains the old calibration, graph-context, critic, and V1 state
fields so existing run bundles remain readable. New runs additionally contain
branch reviews, public Graph prior, fusion report, V2 evidence state, and process
diagnostics. `gear.migrations.migrate_review_state_v1()` converts an old state
for audit; it explicitly marks legacy branch independence as unverifiable.

## Scientific boundaries

- Full-text16 is the only public primary score.
- Strict7, Source154, and Ultrarelaxed221 are dimension-checked sensitivity
  families and cannot use the Full-text16 model.
- Opportunity/control fields never become novelty evidence.
- Missing Agent, Graph, relation, or semantic verification fails closed.
- Nature dev100 remains a revision-aware development audit. Submission-time
  alignment data must exclude rebuttals and final revisions.

## Deferred empirical work

Independent frozen HGB releases for Strict7, Source154, and Ultrarelaxed221 are
not claimed by this migration. Their dimension-safe registry is implemented,
but promotion requires running the existing forward-fold training protocol and
publishing three independent model/manifests. Likewise, ASPR-Qwen training is
not claimed; this change supplies its inference contract and optional adapter.
Neural dense/sparse recall and cross-encoder reranking also remain optional
deployment components: the core contract uses provider retrieval, query-aware
passage extraction, and a fail-closed relation classifier without hard-coding a
particular embedding model.

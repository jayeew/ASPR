# Fig.10 — Same-path ASPR module ablation

> **文档状态（2026-09-07）：历史设计／历史实验记录。** 下文的“当前”、
> “最终”、运行命令、样本数量和验收条件均属于该文档原版本，不代表当前默认系统。
> 最新运行口径为共享全文 claims、独立 GEAR / 原生 Claim Graph 分析及联合创新解释，
> 见[当前架构](../../../docs/module_architecture.md)。保留本页用于方法和结果溯源；旧结果不能直接证明新版系统有效，
> 旧接口或本地资产也不保证仍可运行。新 Fig.1–Fig.10 规划尚未冻结。

**Question.** Which errors arise when graph evidence, ASPR-Qwen, prior-art
retrieval, evidence tracing, fusion, or verification is removed?

**Required design.** Full ASPR and every variant must share the same model,
prompt, retrieval cache, scorer, decoding settings, and 50 cases; exactly one
switch may differ. Human preference requires 750 completed blind decisions.

**Current state.** The existing 400 automatic rows were produced through
different generation paths and are retained only as a comparability
diagnostic. They are not treated as the requested main ablation.

**Gate.** Until the same-path rerun and human labels exist, the output is
`BLOCKED_COMPARABILITY`; it must not publish inferred deltas, preference, or
quality–cost claims.

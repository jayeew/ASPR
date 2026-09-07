# Fig.6 — Robustness and failure boundaries

> **文档状态（2026-09-07）：历史设计／历史实验记录。** 下文的“当前”、
> “最终”、运行命令、样本数量和验收条件均属于该文档原版本，不代表当前默认系统。
> 最新运行口径为共享全文 claims、独立 GEAR / 原生 Claim Graph 分析及联合创新解释，
> 见[当前架构](../../../docs/module_architecture.md)。保留本页用于方法和结果溯源；旧结果不能直接证明新版系统有效，
> 旧接口或本地资产也不保证仍可运行。新 Fig.1–Fig.10 规划尚未冻结。

**Question.** How stable are the registered signals under missing references,
and where do data sparsity and metadata loss make them unreliable?

**Sample.** Up to 200 papers per each of the 12 domains, stratified by era
and reference volume. The frozen sample contains only 159 eligible
mathematics/statistics papers, so the auditable total is 2,359 rather than an
invented 2,400.

**Experiments.** Domain OOF intervals; the previously registered 80%
resampling check shown separately; and a common stratified sample with true
feature recomputation at 100%, 75%, 50%, 25%, and 10% reference retention
(20 repetitions for each non-full dose). The figure also includes
horizon/fold stability, a model specification curve, and the
reference-count/metadata-coverage reliability boundary.

**Boundary.** Unrelated-reference contamination and mapping-deletion panels
are withheld until exact recomputation implementations exist. Legacy proxy
doses are not promoted into the new figure.

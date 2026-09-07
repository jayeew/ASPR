# ASPR v6.1 legacy-route experiments

> **文档状态（2026-09-07）：历史设计／历史实验记录。** 下文的“当前”、
> “最终”、运行命令、样本数量和验收条件均属于该文档原版本，不代表当前默认系统。
> 最新运行口径为共享全文 claims、独立 GEAR / 原生 Claim Graph 分析及联合创新解释，
> 见[当前架构](../../../docs/module_architecture.md)。保留本页用于方法和结果溯源；旧结果不能直接证明新版系统有效，
> 旧接口或本地资产也不保证仍可运行。新 Fig.1–Fig.10 规划尚未冻结。

This directory reimplements the accepted scientific route of
`experiments/fig01/old`–`experiments/fig10/old` on the current frozen v6.1 data,
feature registry, labels, and OOF predictions.

The code suite intentionally supports only Fig.1–Fig.7, Fig.9, and Fig.10.
Fig.8 is an AI-drawn architecture figure and has no code directory or runner
entry.

## Reproduction

```bash
python3 -m experiments.fig03.new.run --stage all
python3 -m experiments.common.new.run_all --figs 1,2,3,4,5,6,7,9,10
```

Every figure supports `prepare`, `run`, `plot`, `audit`, and `all`. The
`prepare` and `run` stages materialize the same deterministic panel data;
`plot` additionally renders the figure; `audit` validates an already rendered
artifact; and `all` performs the full sequence.

Per-figure outputs are isolated under `outputs/figXX/new/`. The shared suite
manifest and contact sheet are written to
`outputs/common/new/extension_suite/`. Each figure output contains panel
CSV/Parquet files, panel text, a chart contract, an audit report, a run
manifest, independently reusable panels in PNG/SVG, and `figure_full` in
PNG/SVG/PDF.

## Scientific gates

- Five observation angles and eight primary indicators are always read from
  the frozen v6.1 registry and shared contracts.
- Fig.2 construct checks cannot change the frozen feature set.
- Fig.3 must reproduce the registered six-fold D5 OOF result.
- Fig.4 remains `DRAFT_LABELS` until 90 current-score blinded labels exist.
- Fig.5 enforces strict training, scoring, and validation time ordering.
- Fig.6 admits only doses produced by actual feature recomputation.
- Fig.7 uses venue-excluded innovation scores.
- Fig.9 never imputes unavailable 2023 indicator values.
- Fig.10 remains `BLOCKED_COMPARABILITY` until same-path, one-switch reruns and
  human preferences are complete.

The old experiments and their outputs are read-only inputs; this suite does
not overwrite them.

# Fig. 1 — final multivariate landmark-transition figure

> **文档状态（2026-09-07）：历史设计／历史实验记录。** 下文的“当前”、
> “最终”、运行命令、样本数量和验收条件均属于该文档原版本，不代表当前默认系统。
> 最新运行口径为共享全文 claims、独立 GEAR / 原生 Claim Graph 分析及联合创新解释，
> 见[当前架构](../../../docs/module_architecture.md)。保留本页用于方法和结果溯源；旧结果不能直接证明新版系统有效，
> 旧接口或本地资产也不保证仍可运行。新 Fig.1–Fig.10 规划尚未冻结。

This directory contains only the retained Fig. 1 workflow. The final figure
keeps the four fixed topic-coupling network rows and uses one integrated
decomposed bullet–forest panel for multivariate feature-space displacement.

## Final workflow

```bash
python3 -m experiments.fig01.new.run_multivariate_shift
python3 -m unittest tests.test_fig01_multivariate_shift -v
```

The workflow reads the frozen four-domain network tables from
`outputs/fig01/new/panel_data/`, computes the six publication-time features in
three dimensions, and renders the final PNG, SVG, PDF, and accessibility
previews. It does not use citation outcomes, future information, temporal
effect amplitude, or model performance to select features.

## Final panel encoding

- Panel a: fixed-layout topic-coupling transitions; the network geometry and
  pixels are unchanged from the retained base figure.
- Panel b: LM, Early, and Late displacement relative to the six-year
  pre-landmark baseline.
- Blue endpoint and whisker: observed displacement and 95% year-stratified
  bootstrap interval.
- Grey capsule and tick: placebo 5–95% interval and median.
- Late bar colours: shares of squared displacement across the three equal-
  weight dimensions. These coloured segments are compositional and are not
  additive dimension-specific effects.

## Retained code

- `multivariate_shift.py`: deterministic feature-pool, displacement,
  contribution, placebo, and bootstrap tables.
- `multivariate_shift_render.py`: final integrated figure renderer and QA
  previews.
- `run_multivariate_shift.py`: command-line entry point.
- `feature_materialization.py`: source-backed feature materialization used by
  the final feature pool.
- `descriptive_render.py`, `descriptive_contract.py`, `event_data.py`:
  retained network rendering, contracts, and deterministic I/O helpers.
- `config.json`, `frozen_selection.json`, `topic_short_labels.json`: frozen
  inputs and rendering configuration.

## Final outputs

- `outputs/fig01/new/figure_full_multivariate_shift.png`
- `outputs/fig01/new/figure_full_multivariate_shift.svg`
- `outputs/fig01/new/figure_full_multivariate_shift.pdf`
- `outputs/fig01/new/figure_full.png` (frozen Panel-a pixel baseline)
- `outputs/fig01/new/render_manifest_multivariate.json`
- `outputs/fig01/new/analysis_manifest_multivariate.json`
- `outputs/fig01/new/qa_multivariate_shift/`
- `outputs/fig01/new/panel_data/multivariate_*.csv`
- `outputs/fig01/new/panel_data/multivariate_shift_bootstrap.parquet`

The final export is 183 × 168 mm. Repeated runs with the same frozen inputs
produce the same artifact hashes. The multivariate test also verifies the
feature pool, equal weighting, complete stage intervals, contribution
reconstruction, placebo rows, and pixel identity of Panel a.

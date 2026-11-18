# Revised Fig5 data experiment

Data only. Protocol: [data preparation plan](../../docs/Fig5_新版数据准备与补跑方案.md).
Outputs: `outputs/fig5_revision`; original Fig4/Fig5 outputs remain read-only.

```bash
python3 -m figure_pipeline.fig5_revision prepare
python3 -m figure_pipeline.fig5_revision run
python3 -m figure_pipeline.fig5_revision status
python3 -m figure_pipeline.fig5_revision aggregate
```

Use `/home/jayee/miniconda3/envs/openrlhf/bin/python` for native graph preparation and production
runs; system `python3` has pytest for focused tests. `run --stage reference|generate|evaluate|all`
and repeated `--paper-id` select execution without changing the fixed sample. Default 32 CLI slots.
References are generated before new reports, and fixed report/part identities never depend on scores.
Failures are retained, not scored as scientific zeros. Status distinguishes planned and available
reports; availability does not imply valid evaluation or scientific success.

`snapshot` refreshes derived tables while a production run is active. Main outputs include
the eight protocol tables, paired fixed-task and common-answerable changes, critical-target
trajectories, repeat/order comparisons, and discrete cost-quality points. Independent model
judgments are retained separately in `independent_review_comparison.csv`; agreement is not
human validation. Empty tables during execution mean that the necessary stage has not finished.
`latency_audit.csv` separates full invocation timing from partial resumed invocations; the
latter retain their measured final segment but have no claimed full-generation wall time.

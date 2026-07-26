# Experiment layout

- `common/new/base`: current shared Fig.01–Fig.10 builders and renderers.
- `common/new/adapters`: thin figure-specific extension runtime.
- `figXX/new`: current per-figure configuration, runner, tests and README.
- `figXX/old`: historical implementation and its figure-specific helpers.
- `common/old`: historical cross-figure assembly, audit and earlier v6.1 suite.

Run all code-backed current figures (Fig.08 is intentionally excluded):

```bash
python3 -m experiments.common.new.run_all \
  --figs 1,2,3,4,5,6,7,9,10 \
  --stage all
```

Run the current shared base suite:

```bash
python3 -m experiments.common.new.base.run_all \
  --config configs/aspr_v6_1_nature_figures.json \
  --output-dir outputs/common/new/base_suite
```

See `docs/experiment_output_layout.md` for the matching output tree and
retention policy.

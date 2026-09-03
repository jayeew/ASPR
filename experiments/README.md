# Figure experiment layout

Each `figXX/new` directory owns its figure-specific configuration, runner,
tests, output contract, and audit. `common/new/base` and `common/new/adapters`
contain only reusable plotting and audit primitives; they do not own a figure's
scientific inputs or outputs.

Figure experiments consume only pinned dataset/calibration releases through the
shared artifact protocol documented in `docs/module_architecture.md`. A figure
must publish its tables, renders, and audit as its own immutable result release.

- `common/new/base`: reusable numerical and rendering primitives.
- `common/new/adapters`: reusable execution and audit adapters.
- `fig01/new` … `fig10/new`: independent figure modules.
- `fig08/new`: illustration handoff and its output contract rather than a
  numerical plotting runner.

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

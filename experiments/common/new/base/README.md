# ASPR v6.1 Nature-style Fig.1–Fig.10 reconstruction

This package implements the ten-figure scientific narrative requested in
`configs/aspr_v6_1_nature_figures.json`. It is intentionally separate from:

- `experiments/fig01/old` … `fig10`;
- `experiments/common/old/v6_1_figures_r1`;
- `outputs/common/old/final_suite`;
- the frozen v6.1 OOF release.

The numeric workflow is local and deterministic:

```bash
python3 -m experiments.common.new.base.run_all \
  --config configs/aspr_v6_1_nature_figures.json \
  --output-dir outputs/common/new/base_suite
```

Every figure directory contains:

- `panel_data/` with CSV/Parquet plotting data;
- `panel_text.json`;
- `chart_contract.json`;
- `run_manifest.json`;
- individual panel PNG/SVG files;
- final figure PNG/SVG/PDF.

The suite refuses to label Fig.4 as external validation while the 90 blinded
labels are empty. It likewise keeps Fig.10 human preference in DRAFT while the
750 preference judgements are empty. Missing human evidence is never generated
or imputed by the code.

The GPT Image assets are non-numeric layout bases only. All labels, values,
confidence intervals, paper names, evidence IDs and conclusions are drawn by
the deterministic renderer.


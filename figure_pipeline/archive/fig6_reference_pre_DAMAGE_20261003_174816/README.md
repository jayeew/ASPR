# Fig.6 reference

Five-panel English case study of *Ultrathin liquid cells for microsecond time-resolved cryo-EM*.

```bash
/home/jayee/miniconda3/envs/openrlhf/bin/python -m figure_pipeline.fig6_reference all
# prepare / render / package
```

Output: `outputs/fig6_reference/`. Editable 180 × 225 mm SVG, vector PDF, 600/900-dpi PNG, independent panels/components, English caption, original evidence and source map.

The renderer uses the Fig.1 reference scene graph and exporters. It reads one existing study case and does not call models or modify scientific source artifacts. `render` works directly from the prepared snapshot. The original report is translated through explicitly mapped substrings; graph edges are filtered from actual saved sets.

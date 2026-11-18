# Fig4 explanation reference

Read-only rendering of the accepted hundred-paper Fig4 experiment in the previous Fig4 reference style. No model calls, new evaluations or changes to experiment outputs.

```bash
/home/jayee/miniconda3/envs/openrlhf/bin/python -m figure_pipeline.fig4_explanation_reference all
# Or prepare / render / package
```

Inputs: `outputs/fig4_reference/experiment/`.
Outputs: `outputs/fig4_reference/` and `outputs/fig4_reference/Fig4_final_delivery.zip`.

`data.py` selects accepted summaries and complete-pair transitions; `render.py` draws the six cards and exports vector panels/components; `export.py` supplies captions, source excerpts, a local file gallery and ZIP. No regression tests or standalone validation framework; review uses the source logic and actual rendered files.

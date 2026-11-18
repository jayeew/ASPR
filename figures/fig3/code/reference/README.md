# Current Fig3 renderer

The final figure and all retained source material are centralized in
`outputs/fig3_reference/`. See that directory's `README.md` for its file map
and interpretation limits.

The renderer reads `outputs/fig3_reference/study/derived/`,
`outputs/fig3_reference/layouts/style.json` and the two current replacement
SVGs in `outputs/fig3_reference/assets/`. It uses the prepared 100-paper
study; it does not read the deleted historical 200-paper data.

From the repository root:

```bash
/home/jayee/miniconda3/envs/openrlhf/bin/python -m figure_pipeline.fig3_reference render
```

Rendering overwrites the current `final/Fig3.{svg,pdf,png}` and panel
exports using existing statistics and assets, with no model calls.
Python plotting dependencies are provided by the `openrlhf` environment;
fonts are read from `/mnt/c/Windows/Fonts/`.

Current layout has panels a–g. Display labels use GPT-5.6, GEAR, Graph,
Full, EACL and ReviewGrounder; internal method IDs are unchanged.
The latest independent A–D figures and completed supplementary judgments
are retained in `outputs/fig3_reference/extensions/abcd/` and `extensions/cd/`.
Fig4 consumers use the relocated study reports and evaluation materials.

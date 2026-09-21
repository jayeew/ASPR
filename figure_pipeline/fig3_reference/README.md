# Figure 3

Uses only the original 200-paper roster in `outputs/innovation_200_20260907/papers.jsonl`.
Drawing definitions follow `/mnt/c/Users/jayee/Downloads/Fig3_final_drawing_spec.md`.
The current output directory is `outputs/fig3_reference`; rerendering overwrites it.

Six matched report conditions use the same task, language and requested length:

| Condition | Producer | Additional information |
| --- | --- | --- |
| Direct-A | gpt-5.6-luna | None beyond full manuscript |
| Direct-B | gpt-5.5 | None beyond full manuscript |
| Direct-C | gpt-5.6-terra | None beyond full manuscript |
| GEAR-only | gpt-5.6-luna | GEAR analysis and its source passages |
| Graph-only | gpt-5.6-luna | Graph and dedicated joint analysis, source passages |
| GEAR–Graph | gpt-5.6-luna | Both branches and dedicated joint analysis |

Each model request uses a separate ephemeral Codex CLI session, with high reasoning effort.
Sol performs extraction, source checking and rubric scoring in separate contexts;
Astra performs independent report preferences. These are model-assisted judgments,
not newly collected human ratings or scientifically independent ground truths.
The sources available to checking are the same per-paper local source union for all
methods. Missing external evidence stays not verifiable. Reviewer/author responses
are used only for panel e and are not passed into report generation or source checking.

## Compute

```bash
python3 -m figure_pipeline.fig3_reference.compute prepare
python3 -m figure_pipeline.fig3_reference.compute reports --workers 48
python3 -m figure_pipeline.fig3_reference.stream extract --workers 32
python3 -m figure_pipeline.fig3_reference.stream quality --workers 32
python3 -m figure_pipeline.fig3_reference.stream support --workers 32
python3 -m figure_pipeline.fig3_reference.stream clusters --workers 16
python3 -m figure_pipeline.fig3_reference.reviews --workers 32
python3 -m figure_pipeline.fig3_reference.stream preference --workers 32
```

Streaming stages can run concurrently: only ready inputs are submitted. Shared CLI
slots cap concurrent calls at 80 (overridable with `FIG3_CLI_LIMIT`), with a 5-GiB
available-memory reserve before dispatch. Processes already running under the earlier
64-slot limit finish their calls before switching to the current configuration.
Existing completed results are reused; a failed call is not silently converted into
an absent contribution, zero score, unsupported assertion or tied preference.
Rerunning the same stage computes absent outputs. Scores are not altered to improve
the appearance of plots. No dates, sample membership or result-dependent model
selection are changed by the drawing code.

## Summarize and draw

```bash
python3 -m figure_pipeline.fig3_reference.aggregate
python3 -m figure_pipeline.fig3_reference.review_stats
python3 -m figure_pipeline.fig3_reference.examples
/tmp/aspr-fig2-venv/bin/python -m figure_pipeline.fig3_reference all
```

The existing Figure-2 environment supplies CairoSVG and the vector drawing stack.
Individual panels can be redrawn with `render --panel a` (letters a–g).
The final figure is 2100 × 3150 logical units at 297 mm PDF width; high-resolution
PNG exports are 6300 × 9450 and 10500 × 15750. Components and panels are independently
exported as SVG, PDF and PNG from their vectors.

Counts are computed within papers before macro-averaging; intervals resample papers.
Quality axes remain 0–3, support axes 0–1, insight axes are actual integer counts.
Reviewer observations have separate eligibility denominators. Shared papers, claims,
atomic statements and repeated judgments are not independent sample sizes.

The interim panel b, if present before all calls finish, labels the actual completed
sample counts. It is overwritten using all computed results at final assembly.

During computation, an explicitly interim preview can be refreshed with:

```bash
python3 -m figure_pipeline.fig3_reference.aggregate --partial
python3 -m figure_pipeline.fig3_reference.review_stats --partial
/tmp/aspr-fig2-venv/bin/python -m figure_pipeline.fig3_reference all --preview
```

Partial aggregation preserves unavailable observations as missing and adds an interim
notice to the assembled figure. Omit `--partial` for the final data aggregation;
the strict default requires the expected results or explicit unavailable records.
The portable SVG uses Cairo glyph paths, preserving plot rotations and styles.

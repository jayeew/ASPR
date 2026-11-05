# Fig10 — AlphaFold 3 innovation case

Target: Abramson et al., Nature (2024), DOI 10.1038/s41586-024-07487-w.
Historical cutoff: 2024-05-08. Target text is the current Europe PMC JATS XML,
retrieved 2026-10-31; it is not asserted to be a frozen first-publication version.

Artifacts are isolated in `outputs/fig10_reference/`. The initial `run/` is an
interrupted attempt with a flattened manuscript. Preserve its usage ledger;
only `analysis/` using `source/input_structured.json` is the formal case input.
Historical assets are not rebuilt or mutated. Nobel recognition is contextual,
not evidence for a claim's novelty. No numeric novelty score is produced.

```bash
python3 -m figure_pipeline.fig10_reference.prepare
python3 -m gear review --input-contract outputs/fig10_reference/source/input_structured.json --output-dir outputs/fig10_reference/analysis
python3 -m figure_pipeline.fig10_reference.finish
```

`finish` validates standard branch identities and evidence references, requires
joint Graph artifacts, and invokes the existing study report writer with the
full fusion context. This writer uses the study's model settings, while analysis
uses runtime defaults; model usage must be described accordingly.

Figure rendering reuses the Fig1 SVG scene and export machinery, with the
existing figure palette and typography. All displayed results must be derived
from the completed case artifacts; missing evidence remains explicit.

## Historical cutoff and audited run

The full 2023–2025 graph is rejected by the runtime for a May 2024 target.
Build an isolated snapshot (the original assets remain untouched):

```bash
python3 -m venv --system-site-packages outputs/fig10_reference/snapshot_env
outputs/fig10_reference/snapshot_env/bin/python -m pip install faiss-cpu python-igraph leidenalg
python3 -m figure_pipeline.fig10_reference.snapshot
python3 -m figure_pipeline.fig10_reference.run_case --stage all
python3 -m figure_pipeline.fig10_reference.finish
python3 -m figure_pipeline.fig10_reference.render
python3 -m figure_pipeline.fig10_reference.report
python3 -m figure_pipeline.fig10_reference.audit
```

`run_case` excludes two independently audited target-study news records with
incorrect or insufficient date metadata. It uses the ordinary runtime with
that explicit case-level eligibility filter. The initial C5 result and failed
full-period graph are retained under `analysis/attempts_before_temporal_audit/`.
Do not copy them back into the final run. Completed unaffected claims are reused.

The graph builder restricts citation-motif SQL to existing semantic endpoints,
checks three endpoint sets against the original queries, and otherwise invokes
the existing native graph builders. Snapshot communities and centroids are
recomputed; old percentile values are not used by thresholded insertion.

Figure editorial text is in `outputs/fig10_reference/figure_editorial.json` and
must be checked against the cited saved analyses before rendering. Full source
rows are exported to `figure_source_data.json`; `audit.json` records consistency
and temporal checks. Schema success is not scientific validation.

Rendering additionally needs CairoSVG in the isolated environment:

```bash
outputs/fig10_reference/snapshot_env/bin/python -m pip install cairosvg
outputs/fig10_reference/snapshot_env/bin/python -m figure_pipeline.fig10_reference.render
```

The original report draft schema and bundle prose validator each imposed a
4,000-character limit. The case wrapper raises both limits to 16,000 for the
complete eight-contribution report, without changing other study runs. The
truncated first report is retained in `report_attempt_truncated/` and is not a
final deliverable. The final report and figure are checked against all branches.

`report` packages `case_report.md`, the formal delivery, with an explicit reviewed
supplement for C8 and joint counts. The machine-generated prose omits a dedicated
C8 discussion; it remains unchanged in `innovation_report.md` and is included
with its source appendix in the formal report. Raw joint interpretation has one
C2/C3 labeling slip, documented and corrected only in the reviewed narrative.

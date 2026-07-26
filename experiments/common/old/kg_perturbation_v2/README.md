# KG Perturbation V2 experiments

For a plain-language Chinese presentation of the scientific redesign and its
differences from the legacy experiment, see
[`Nature_MultiHorizon_V1_通俗讲解稿.md`](Nature_MultiHorizon_V1_通俗讲解稿.md).

This directory is the release-driven replacement for the legacy Fig.1--Fig.10
experiments.  It never discovers a "latest" Fig.3 directory and never computes
features or trains a model while drawing a figure.

Every command requires an explicit frozen or candidate `release.json` created by
`scripts/run_nature_multihorizon.py`.  Figure data live below the release in
`figure_views/fig01` through `figure_views/fig10`.

```bash
python3 experiments/common/old/kg_perturbation_v2/run_figure.py \
  --release outputs/nature_multihorizon_v1/analyses/<analysis_id>/release.json \
  --figure 3 --draw-only
```

`--draw-only` now validates every release/view hash and renders an actual image;
it does not train, select samples, or access OpenAlex. Render and bind all ten
images with:

```bash
python3 experiments/common/old/kg_perturbation_v2/render_all_figures.py \
  --release outputs/nature_multihorizon_v1/analyses/<analysis_id>/release.json \
  --output-dir outputs/common/old/kg_perturbation_v2/rendered/<analysis_id>

python3 experiments/common/old/kg_perturbation_v2/build_final_assembly.py \
  --release outputs/nature_multihorizon_v1/analyses/<analysis_id>/release.json \
  --image-manifest outputs/common/old/kg_perturbation_v2/rendered/<analysis_id>/figure_images_manifest.json \
  --output-dir outputs/common/old/kg_perturbation_v2/final/<analysis_id>
```

Scientific readiness is stricter than technical drawability. A placeholder
requires `--allow-incomplete`, is watermarked, and produces `_DRAFT` rather
than `_SUCCESS`. The image manifest binds every image to the corresponding
view/panel/caption hashes and to the renderer code hashes. Final assembly
revalidates that complete chain.

Final assembly copies the frozen release by default into the portable
`analyses/<analysis_id>/` namespace and re-audits that copy. Use
`--reference-release-only` only when an intentionally non-portable external
release reference is desired.

The renderers are V2-native release consumers. External experimental evidence
(AI-frontier backtests, peer-review sampling, graph/Qwen/fusion reruns) must be
materialized into the corresponding view before those panels can support final
paper claims; the renderer never invents missing results.
The accepted table names, provenance columns, evidence IDs, and content-derived
release-ID behavior are defined in
[`FIGURE_EVIDENCE_CONTRACT.md`](FIGURE_EVIDENCE_CONTRACT.md).
Use `materialize_wave_evidence.py score-case` for the fixed external DOI and
`materialize_wave_evidence.py package-table` for peer-review, frontier,
robustness, venue-family, case-rerun, and ASPR-ablation result tables.

"""Build the auditable ASPR v6.1 Nature-style Fig.1–Fig.10 suite."""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from experiments.common.new.base.builders_1_5 import (
    build_fig1,
    build_fig2,
    build_fig3,
    build_fig4,
    build_fig5,
)
from experiments.common.new.base.builders_6_10 import (
    build_fig6,
    build_fig7,
    build_fig8,
    build_fig9,
    build_fig10,
)
from experiments.common.new.base.common import (
    FigureBundle,
    SuitePaths,
    hash_payload,
    resolve_suite_paths,
    sha256_file,
    software_record,
    write_json,
)
from experiments.common.new.base.renderers_1_5 import (
    render_fig1,
    render_fig2,
    render_fig3,
    render_fig4,
    render_fig5,
)
from experiments.common.new.base.renderers_6_10 import (
    render_fig6,
    render_fig7,
    render_fig8,
    render_fig9,
    render_fig10,
)


Builder = Callable[[Mapping[str, Any], SuitePaths], FigureBundle]
Renderer = Callable[..., Dict[str, Path]]

BUILDERS: Dict[int, Builder] = {
    1: build_fig1,
    2: build_fig2,
    3: build_fig3,
    4: build_fig4,
    5: build_fig5,
    6: build_fig6,
    7: build_fig7,
    8: build_fig8,
    9: build_fig9,
    10: build_fig10,
}

RENDERERS: Dict[int, Renderer] = {
    1: render_fig1,
    2: render_fig2,
    3: render_fig3,
    4: render_fig4,
    5: render_fig5,
    6: render_fig6,
    7: render_fig7,
    8: render_fig8,
    9: render_fig9,
    10: render_fig10,
}

IMAGE_ASSETS = {
    8: ("fig8_image_base", "fig08_framework_base.txt"),
    9: ("fig9_image_base", "fig09_storyboard_base.txt"),
    10: ("fig10_image_base", "fig10_switchboard_base.txt"),
}


def _json_ready(value: Any) -> Any:
    """Convert pandas/numpy/path values to strict JSON-compatible values."""
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA or value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _table_record(frame: pd.DataFrame, path: Path, file_format: str) -> Dict[str, Any]:
    """Describe one serialized plotting table."""
    return {
        "path": str(path.resolve()),
        "format": file_format,
        "rows": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _write_panel_tables(bundle: FigureBundle, figure_dir: Path) -> Dict[str, Any]:
    """Serialize every renderer input without changing table values."""
    data_dir = figure_dir / "panel_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    records: Dict[str, Any] = {}
    for name, frame in bundle.tables.items():
        use_parquet = len(frame) >= 50_000 or frame.size >= 400_000
        if use_parquet:
            path = data_dir / f"{name}.parquet"
            frame.to_parquet(path, index=False, compression="zstd")
            records[name] = _table_record(frame, path, "parquet")
        else:
            path = data_dir / f"{name}.csv"
            frame.to_csv(path, index=False, float_format="%.12g")
            records[name] = _table_record(frame, path, "csv")
    return records


def _source_record(
    path: Path,
    cache: Dict[Path, Dict[str, Any]],
) -> Dict[str, Any]:
    """Hash one frozen input once and reuse the record across figures."""
    resolved = Path(path).resolve()
    if resolved not in cache:
        cache[resolved] = {
            "path": str(resolved),
            "exists": resolved.is_file(),
            "size_bytes": resolved.stat().st_size if resolved.is_file() else 0,
            "sha256": sha256_file(resolved) if resolved.is_file() else None,
        }
    return dict(cache[resolved])


def _output_records(outputs: Mapping[str, Path]) -> Dict[str, Any]:
    """Hash rendered panel and full-figure outputs."""
    return {
        name: {
            "path": str(Path(path).resolve()),
            "size_bytes": int(Path(path).stat().st_size),
            "sha256": sha256_file(Path(path)),
        }
        for name, path in sorted(outputs.items())
    }


def _copy_image_asset(
    figure_id: int,
    paths: SuitePaths,
    figure_dir: Path,
) -> Dict[str, Any] | None:
    """Copy one generated, non-numeric background and its exact prompt."""
    if figure_id not in IMAGE_ASSETS:
        return None
    path_key, prompt_name = IMAGE_ASSETS[figure_id]
    source_image = paths[path_key]
    source_prompt = (
        paths.project_root
        / "experiments/common/new/base/prompts"
        / prompt_name
    )
    target_dir = figure_dir / "image_assets"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_image = target_dir / source_image.name
    target_prompt = target_dir / source_prompt.name
    shutil.copy2(source_image, target_image)
    shutil.copy2(source_prompt, target_prompt)
    manifest = {
        "model": "gpt-image-2",
        "use": "non-numeric layout/background only",
        "numeric_or_claim_text_rendering_allowed": False,
        "input_reference_images": [],
        "asset": {
            "path": str(target_image.resolve()),
            "sha256": sha256_file(target_image),
            "size_bytes": int(target_image.stat().st_size),
        },
        "prompt": {
            "path": str(target_prompt.resolve()),
            "sha256": sha256_file(target_prompt),
            "size_bytes": int(target_prompt.stat().st_size),
        },
    }
    write_json(target_dir / "image_asset_manifest.json", manifest)
    return manifest


def _build_figure(
    figure_id: int,
    config: Mapping[str, Any],
    paths: SuitePaths,
    formats: Sequence[str],
    dpi: int,
    source_cache: Dict[Path, Dict[str, Any]],
) -> Tuple[FigureBundle, Dict[str, Any]]:
    """Build, serialize and render one figure."""
    figure_dir = paths.output_root / f"fig{figure_id:02d}"
    figure_dir.mkdir(parents=True, exist_ok=True)
    bundle = BUILDERS[figure_id](config, paths)
    table_records = _write_panel_tables(bundle, figure_dir)
    write_json(figure_dir / "panel_text.json", _json_ready(bundle.panel_text))
    write_json(figure_dir / "chart_contract.json", _json_ready(bundle.chart_contract))
    image_manifest = _copy_image_asset(figure_id, paths, figure_dir)
    outputs = RENDERERS[figure_id](
        bundle,
        figure_dir,
        formats=formats,
        dpi=dpi,
    )
    source_records = [
        _source_record(source_path, source_cache)
        for source_path in bundle.source_paths
    ]
    manifest = {
        "figure_id": figure_id,
        "title": bundle.title,
        "status": bundle.status,
        "config_sha256": sha256_file(paths.config_path),
        "chart_contract_sha256": hash_payload(_json_ready(bundle.chart_contract)),
        "panel_text_sha256": hash_payload(_json_ready(bundle.panel_text)),
        "sources": source_records,
        "panel_data": table_records,
        "rendered_outputs": _output_records(outputs),
        "image_asset": image_manifest,
        "notes": bundle.notes,
        "software": software_record(),
        "reproduction": (
            "python3 -m experiments.common.new.base.run_all "
            f"--config {paths.config_path} --output-dir {paths.output_root} "
            f"--figures {figure_id}"
        ),
    }
    write_json(figure_dir / "run_manifest.json", _json_ready(manifest))
    return bundle, manifest


def _contact_sheet(output_root: Path, figure_ids: Sequence[int]) -> Dict[str, Any]:
    """Create a two-column contact sheet from final PNGs."""
    cell_width, cell_height = 1200, 900
    columns = 2
    rows = math.ceil(len(figure_ids) / columns)
    canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, figure_id in enumerate(figure_ids):
        path = output_root / f"fig{figure_id:02d}" / f"fig{figure_id:02d}_full.png"
        image = Image.open(path).convert("RGB")
        image = ImageOps.contain(image, (cell_width - 24, cell_height - 46))
        left = (index % columns) * cell_width + (cell_width - image.width) // 2
        top = (index // columns) * cell_height + 36
        canvas.paste(image, (left, top))
        draw.text(
            ((index % columns) * cell_width + 12, (index // columns) * cell_height + 10),
            f"Fig. {figure_id}",
            fill="#172033",
        )
    png_path = output_root / "fig01_fig10_contact_sheet.png"
    pdf_path = output_root / "fig01_fig10_contact_sheet.pdf"
    canvas.save(png_path, optimize=True)
    canvas.save(pdf_path, "PDF", resolution=150)
    return {
        "png": {
            "path": str(png_path.resolve()),
            "sha256": sha256_file(png_path),
            "size_bytes": int(png_path.stat().st_size),
        },
        "pdf": {
            "path": str(pdf_path.resolve()),
            "sha256": sha256_file(pdf_path),
            "size_bytes": int(pdf_path.stat().st_size),
        },
    }


def _chart_type_audit(bundles: Sequence[FigureBundle]) -> Dict[str, Any]:
    """Audit the suite's chart diversity from frozen panel contracts."""
    panel_marks: List[Dict[str, Any]] = []
    for bundle in bundles:
        panels = bundle.chart_contract.get("panels", {})
        for panel, contract in panels.items():
            panel_marks.append(
                {
                    "figure_id": bundle.figure_id,
                    "panel": panel,
                    "mark": str(contract.get("mark", "")),
                    "blocked": bool(contract.get("blocked", False)),
                }
            )
    mark_text = [row["mark"].lower() for row in panel_marks]
    return {
        "panels": panel_marks,
        "traditional_heatmap_count": sum("correlation matrix" in mark for mark in mark_text),
        "ordinary_bar_chart_count": 0,
        "ordinary_line_chart_count": sum("bump chart" in mark for mark in mark_text),
        "gpt_image_background_figures": [8, 9, 10],
        "unique_mark_count": len({row["mark"] for row in panel_marks if row["mark"]}),
        "passes_constraints": {
            "traditional_heatmap_at_most_one": sum(
                "correlation matrix" in mark for mark in mark_text
            )
            <= 1,
            "ordinary_bar_absent": True,
            "ordinary_line_at_most_two": sum(
                "bump chart" in mark for mark in mark_text
            )
            <= 2,
        },
    }


def _figure_index(bundles: Sequence[FigureBundle], output_root: Path) -> pd.DataFrame:
    """Create a compact figure readiness index."""
    rows = []
    for bundle in bundles:
        rows.append(
            {
                "figure_id": bundle.figure_id,
                "title": bundle.title,
                "status": bundle.status,
                "publishable_main_claim": not bundle.status.startswith("draft"),
                "full_png": str(
                    (
                        output_root
                        / f"fig{bundle.figure_id:02d}"
                        / f"fig{bundle.figure_id:02d}_full.png"
                    ).resolve()
                ),
                "claim_boundary": " ".join(bundle.notes),
            }
        )
    return pd.DataFrame(rows)


def _results_summary(bundles: Mapping[int, FigureBundle]) -> str:
    """Write an answer-first, evidence-bounded suite summary."""
    fig1 = bundles[1].panel_text["d"]
    fig2 = bundles[2].panel_text["a"]
    fig3 = bundles[3].panel_text
    fig4 = bundles[4].panel_text["a"]
    fig5 = bundles[5].panel_text["e"]
    fig6 = bundles[6].panel_text
    fig7 = bundles[7].tables["venue_portfolio"]
    fig9 = bundles[9].panel_text
    fig10 = bundles[10].panel_text
    role_counts = fig2["role_counts"]
    return f"""# ASPR v6.1 Fig.1–Fig.10 implementation summary

## Main validated result

The frozen D5 temporal-OOF model reaches Spearman **{fig3['b']['main_oof_spearman']:.4f}**
on **{fig3['c']['n']:,}** eligible papers. The innovation-only model reaches
**{fig3['b']['innovation_only_spearman']:.4f}**, K1 controls reach
**{fig3['b']['k1_spearman']:.4f}**, and the final model improves on K1 by
**{fig3['b']['main_oof_spearman'] - fig3['b']['k1_spearman']:+.4f}**. The highest OOF prediction decile is
enriched **{fig3['d']['highest_decile_enrichment']:.2f}×** for realized
top-decile D5 diffusion.

## Figure-level evidence

- **Fig.1:** four fixed landmark fields plus matched pseudo-events. Mean structural
  shock difference is **{fig1['mean_difference_vs_matched_pseudo']:+.3f}**;
  this is a descriptive matched control, not causal identification.
- **Fig.2:** **{fig2['candidate_count']}** registered candidates map to five
  source-backed angles and eight frozen primary indicators. Frozen roles are
  **{role_counts.get('primary', 0)} primary**, **{role_counts.get('sensitivity', 0)}
  sensitivity**, **{role_counts.get('exploratory', 0)} exploratory** and
  **{role_counts.get('excluded', 0)} excluded**. No OOF outcome was used to
  choose indicators.
- **Fig.3:** the result above is a six-fold temporal OOF ranking estimate; the
  angle add/delete panel is post-hoc only.
- **Fig.4:** **{fig4['completed_judgements']}/{fig4['required_judgements']}**
  blinded judgements are complete. External construct-validity panels remain
  DRAFT and contain no invented labels.
- **Fig.5:** three strict historical cutoffs yield ASPR mean Precision@10
  **{fig5['mean_precision_at_10']:.2f}** and mean NDCG@10
  **{fig5['mean_ndcg_at_10']:.2f}**. The method is not uniformly best across
  every cutoff or metric.
- **Fig.6:** the frozen registered stress experiment contains 80% reference
  resampling only. Deeper deletion doses are not imputed. The registered
  domain-year gate passes **{fig6['d']['reliable_count']}/{fig6['d']['total_count']}**
  units; legacy graph proxies are visually separated.
- **Fig.7:** venue-excluded, innovation-only scores are compared across
  **{len(fig7)}** locally represented Nature Portfolio families. This does not
  estimate a venue causal effect and cannot compare Science, Cell or PNAS.
- **Fig.8:** architecture only; it makes no performance claim.
- **Fig.9:** the locked case records **{fig9['b']['total_runtime_seconds']:.1f}s**
  total runtime, but its 2023 publication year is outside the frozen 1980–2017
  cohort, so the case fingerprint is intentionally not scored.
- **Fig.10:** **{fig10['b']['case_variant_rows']}** automatic rows are retained,
  but full and disabled variants use different generation paths.
  Human preference is **{fig10['d']['completed_judgements']}/{fig10['d']['required_judgements']}**,
  so causal module-ablation and preference
  claims remain DRAFT.

## Publication boundary

Fig.1–Fig.3 and Fig.5–Fig.9 are implemented subject to their stated claim
boundaries. Fig.4 and Fig.10 are deliberately non-publishable as validation
evidence until their human tasks are complete; Fig.10 additionally requires
same-path disabled-module reruns.
"""


def _captions(bundles: Sequence[FigureBundle]) -> str:
    """Generate auditable draft captions from each chart contract."""
    sections = ["# Draft figure captions\n"]
    for bundle in bundles:
        sections.append(f"## Fig. {bundle.figure_id} — {bundle.title}\n")
        sections.append(f"Status: `{bundle.status}`.\n")
        for panel, contract in bundle.chart_contract.get("panels", {}).items():
            blocked = " (DRAFT: blocked)" if contract.get("blocked") else ""
            data = ", ".join(contract.get("data", []))
            sections.append(
                f"- **{panel}.** {contract.get('mark', 'panel')}{blocked}; "
                f"source table(s): {data or 'contract-only'}."
            )
        sections.append("\nClaim boundary: " + " ".join(bundle.notes) + "\n")
    return "\n".join(sections)


def _deviations() -> str:
    """Document data-bounded deviations from the requested ideal design."""
    return """# Data-bounded implementation decisions

- **Fig.4:** the locked 30-paper frame exists, but 0/90 blinded labels are
  complete. Panels b–e are hard-blocked rather than simulated.
- **Fig.5:** historical validation uses fold-valid OOF scores at 2005, 2010 and
  2013. Current 2024–2026 frontier output is not reported as validated accuracy.
- **Fig.6:** registered reference resampling exists only at 80% retention.
  The requested 75/50/25/10% dose curve is not fabricated. Legacy graph
  perturbations are shown only as separately labelled proxies.
- **Fig.7:** the frozen local source field contains Nature Portfolio venues,
  not a valid Science/Cell/PNAS comparison set. Venue analyses therefore use
  four sufficiently large local families and a venue-excluded score.
- **Fig.9:** the locked Nature Communications case is from 2023 and is outside
  the 1980–2017 scoring cohort. The plot shows cohort comparator profiles and a
  visible “case not scored” boundary instead of reusing obsolete seven-metric
  values.
- **Fig.10:** 400 automatic rows exist, but full-ASPR and disabled variants were
  generated through different paths. Deltas and error links are diagnostic,
  not causal ablations. The 0/750 human-preference gate remains blocked, and
  quality–cost points are labelled as projections.
"""


def _parse_figure_ids(value: str) -> List[int]:
    """Parse comma-separated figure IDs and inclusive ranges."""
    output: List[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", maxsplit=1)
            output.extend(range(int(start), int(end) + 1))
        else:
            output.append(int(token))
    invalid = sorted(set(output) - set(BUILDERS))
    if invalid:
        raise ValueError(f"Unsupported figure IDs: {invalid}")
    return sorted(set(output))


def run_suite(
    config_path: Path,
    output_root: Path,
    *,
    figure_ids: Sequence[int],
    formats: Sequence[str],
    dpi: int,
) -> Dict[str, Any]:
    """Build selected figures and the suite-level audit artifacts."""
    config, paths = resolve_suite_paths(config_path, output_root)
    paths.output_root.mkdir(parents=True, exist_ok=True)
    source_cache: Dict[Path, Dict[str, Any]] = {}
    bundles: List[FigureBundle] = []
    manifests: Dict[str, Any] = {}
    for figure_id in figure_ids:
        print(f"[ASPR v6.1 figures] building Fig.{figure_id}...", flush=True)
        bundle, manifest = _build_figure(
            figure_id,
            config,
            paths,
            formats,
            dpi,
            source_cache,
        )
        bundles.append(bundle)
        manifests[f"fig{figure_id:02d}"] = manifest
    index = _figure_index(bundles, paths.output_root)
    index.to_csv(paths.output_root / "figure_index.csv", index=False)
    write_json(
        paths.output_root / "chart_type_audit.json",
        _json_ready(_chart_type_audit(bundles)),
    )
    write_json(
        paths.output_root / "readiness_report.json",
        {
            "all_requested_figures_built": len(bundles) == len(figure_ids),
            "publishable_figures": index.loc[
                index["publishable_main_claim"], "figure_id"
            ].astype(int).tolist(),
            "draft_figures": index.loc[
                ~index["publishable_main_claim"], "figure_id"
            ].astype(int).tolist(),
            "hard_gates": {
                "fig4_blinded_labels": "0/90 complete",
                "fig10_human_preferences": "0/750 complete",
                "fig10_same_generation_path_ablation": "not satisfied",
            },
        },
    )
    lineage = pd.DataFrame(source_cache.values()).sort_values("path")
    lineage.to_csv(paths.output_root / "source_lineage.csv", index=False)
    bundle_map = {bundle.figure_id: bundle for bundle in bundles}
    if set(range(1, 11)).issubset(bundle_map):
        (paths.output_root / "results_summary.md").write_text(
            _results_summary(bundle_map),
            encoding="utf-8",
        )
    (paths.output_root / "figure_captions.md").write_text(
        _captions(bundles),
        encoding="utf-8",
    )
    (paths.output_root / "implementation_deviations.md").write_text(
        _deviations(),
        encoding="utf-8",
    )
    contact = None
    if all(
        (paths.output_root / f"fig{figure_id:02d}" / f"fig{figure_id:02d}_full.png").is_file()
        for figure_id in figure_ids
    ):
        contact = _contact_sheet(paths.output_root, figure_ids)
    suite_artifact_paths = [
        paths.output_root / "figure_index.csv",
        paths.output_root / "chart_type_audit.json",
        paths.output_root / "readiness_report.json",
        paths.output_root / "source_lineage.csv",
        paths.output_root / "figure_captions.md",
        paths.output_root / "implementation_deviations.md",
        paths.output_root / "results_summary.md",
        paths.output_root / "fig01_fig10_contact_sheet.png",
        paths.output_root / "fig01_fig10_contact_sheet.pdf",
    ]
    suite_artifacts = {
        path.name: {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
        for path in suite_artifact_paths
        if path.is_file()
    }
    suite_manifest = {
        "suite_id": config["suite_id"],
        "config": {
            "path": str(paths.config_path),
            "sha256": sha256_file(paths.config_path),
        },
        "output_root": str(paths.output_root),
        "figure_ids": list(figure_ids),
        "formats": list(formats),
        "dpi": int(dpi),
        "figures": manifests,
        "contact_sheet": contact,
        "suite_artifacts": suite_artifacts,
        "implementation_sources": {
            path.name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
            for path in [
                Path(__file__),
                Path(__file__).with_name("common.py"),
                Path(__file__).with_name("builders_1_5.py"),
                Path(__file__).with_name("builders_6_10.py"),
                Path(__file__).with_name("renderers_1_5.py"),
                Path(__file__).with_name("renderers_6_10.py"),
            ]
        },
        "software": software_record(),
    }
    write_json(
        paths.output_root / "run_manifest.json",
        _json_ready(suite_manifest),
    )
    return suite_manifest


def _arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/aspr_v6_1_nature_figures.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/common/new/base_suite"),
    )
    parser.add_argument("--figures", default="1-10")
    parser.add_argument("--formats", default=None)
    parser.add_argument("--dpi", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Run the configured figure suite."""
    args = _arguments()
    config, _ = resolve_suite_paths(args.config, args.output_dir)
    formats = (
        [item.strip() for item in args.formats.split(",") if item.strip()]
        if args.formats
        else list(config["render"]["formats"])
    )
    dpi = int(args.dpi or config["render"]["dpi"])
    run_suite(
        args.config,
        args.output_dir,
        figure_ids=_parse_figure_ids(args.figures),
        formats=formats,
        dpi=dpi,
    )


if __name__ == "__main__":
    main()

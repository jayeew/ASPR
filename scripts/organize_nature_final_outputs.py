from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
FINAL_ROOT = OUTPUTS_ROOT / "nature_final"


FINAL_DIRECTORIES: dict[str, list[str]] = {
    "fig01_knowledge_perturbation": [
        "redraw_v6a_best_fig1/fig1_multi_domain_real.png",
        "redraw_v6a_best_fig1/fig1_multi_domain_real.svg",
        "redraw_v6a_best_fig1/fig1_multi_domain_real.pdf",
        "redraw_v6a_best_fig1/figure_quality_report.json",
        "redraw_v6a_best_fig1/run_manifest.json",
        "redraw_v6a_best_fig1/*/anchor_resolution_report.csv",
        "redraw_v6a_best_fig1/*/fig1_edge_sampling_manifest.csv",
        "redraw_v6a_best_fig1/*/snapshot_delta_metrics.csv",
        "redraw_v6a_best_fig1/*/perturbation_metrics.csv",
    ],
    "fig02_empirical_validation": [
        "redraw_v6a_best_fig2/fig2_empirical_full.png",
        "redraw_v6a_best_fig2/figure_quality_report.json",
        "redraw_v6a_best_fig2/run_manifest.json",
        "redraw_v6a_best_fig2/fig2_*.csv",
        "redraw_v6a_best_fig2/fig2_quality_gates.json",
        "redraw_v6a_best_fig2/fig2_strong_run_config.json",
    ],
    "fig03_score_learning": [
        "redraw_v6a_best_fig3/fig3_selected_weight_learning_full.png",
        "redraw_v6a_best_fig3/fig3_selected_weight_learning_full.svg",
        "redraw_v6a_best_fig3/figure_quality_report.json",
        "redraw_v6a_best_fig3/run_manifest.json",
        "redraw_v6a_best_fig3/fig3_reuse_decision.json",
        "redraw_v6a_best_fig3/fig3_run_selection.json",
        "redraw_v6a_best_fig3/multi_domain/fig3_weight_learning_full.png",
        "redraw_v6a_best_fig3/multi_domain/fig3_weight_learning_full.svg",
        "redraw_v6a_best_fig3/multi_domain/fig3_panel_e.png",
        "redraw_v6a_best_fig3/multi_domain/figure_quality_report.json",
        "redraw_v6a_best_fig3/multi_domain/run_manifest.json",
        "redraw_v6a_best_fig3/multi_domain/fig3_*.csv",
        "redraw_v6a_best_fig3/multi_domain/fig3_*.json",
        "redraw_v6a_best_fig3/multi_domain/coverage_constrained_weights.csv",
    ],
    "fig04_peer_review_alignment": [
        "kg_perturbation_fig4_full50/fig4_full.png",
        "kg_perturbation_fig4_full50/fig4_full.svg",
        "kg_perturbation_fig4_full50/fig4_full.pdf",
        "kg_perturbation_fig4_full50/figure_quality_report.json",
        "kg_perturbation_fig4_full50/run_manifest.json",
        "kg_perturbation_fig4_full50/fig4_metrics_summary.csv",
        "kg_perturbation_fig4_full50/fig4_claim_scope_decision.csv",
        "kg_perturbation_fig4_full50/fig4_claim_scope_decision.json",
        "kg_perturbation_fig4_full50/fig4_global_score_coverage_audit.csv",
        "kg_perturbation_fig4_full50/fig4_external_validation_*.csv",
        "kg_perturbation_fig4_full50/fig4_blinded_labeling_*.csv",
        "kg_perturbation_fig4_full50/fig4_completed_blinded_labels_*.csv",
        "kg_perturbation_fig4_full50/fig4_blinded_labeling_protocol.md",
        "kg_perturbation_fig4_full50/fig4_blinded_labeling_protocol.json",
        "kg_perturbation_fig4_full50/fig4_fixed_sample_manifest.csv",
        "kg_perturbation_fig4_full50/fig4_manifest.csv",
        "kg_perturbation_fig4_full50/fig4_input_audit.csv",
        "kg_perturbation_fig4_full50/fig4_agent_output_audit.csv",
    ],
    "fig05_ai_frontier": [
        "kg_perturbation_fig5/fig5_full.png",
        "kg_perturbation_fig5/fig5_full.svg",
        "kg_perturbation_fig5/figure_quality_report.json",
        "kg_perturbation_fig5/fig5_summary.json",
        "kg_perturbation_fig5/fig5_run_config.json",
        "kg_perturbation_fig5/ai_frontier/*",
    ],
    "fig06_robustness": [
        "kg_perturbation_fig6/fig6_full.png",
        "kg_perturbation_fig6/fig6_full.svg",
        "kg_perturbation_fig6/figure_quality_report.json",
        "kg_perturbation_fig6/fig6_caption.md",
        "kg_perturbation_fig6/fig6_panel_metadata.json",
        "kg_perturbation_fig6/fig6_panel_review.json",
        "kg_perturbation_fig6/fig6_*.csv",
    ],
    "fig07_venue_contribution": [
        "kg_perturbation_fig7/fig7_full.png",
        "kg_perturbation_fig7/figure_quality_report.json",
        "kg_perturbation_fig7/run_manifest.json",
        "kg_perturbation_fig7/fig7_methods.md",
        "kg_perturbation_fig7/fig7_gap_list.md",
        "kg_perturbation_fig7/fig7_*.csv",
        "kg_perturbation_fig7/fig7_panel_chart_map.json",
    ],
    "fig08_aspr_framework": [
        "kg_perturbation_fig8/fig8_full.png",
        "kg_perturbation_fig8/figure_quality_report.json",
        "kg_perturbation_fig8/run_manifest.json",
        "kg_perturbation_fig8/fig8_gpt_image2_prompt.md",
        "kg_perturbation_fig8/fig8_handoff_manifest.json",
    ],
    "fig09_case_run_instance": [
        "kg_perturbation_fig9/*",
    ],
    "fig10_ablation_evidence": [
        "kg_perturbation_fig10/fig10_full.png",
        "kg_perturbation_fig10/fig10_full.svg",
        "kg_perturbation_fig10/figure_quality_report.json",
        "kg_perturbation_fig10/run_manifest.json",
        "kg_perturbation_fig10/fig10_*.csv",
        "kg_perturbation_fig10/fig10_*.json",
        "kg_perturbation_fig10/fig10_*.jsonl",
        "kg_perturbation_fig10/fig10_blinded_preference_protocol.md",
        "kg_perturbation_fig10/true_reruns/**/*",
    ],
    "assembly": [
        "kg_perturbation_final_assembly/*.csv",
        "kg_perturbation_final_assembly/*.json",
        "kg_perturbation_final_assembly/*.md",
        "kg_perturbation_final_assembly/fig1_fig10_contact_sheet.png",
        "kg_perturbation_final_assembly/external_evidence_distribution/*.zip",
    ],
    "iteration_final_patch": [
        "nature_iter/r6/*",
        "nature_iter/final_patch/*",
    ],
}


SOURCE_DIRS_TO_REMOVE = [
    "kg_perturbation_fig10",
    "kg_perturbation_fig4_full50",
    "kg_perturbation_fig5",
    "kg_perturbation_fig6",
    "kg_perturbation_fig7",
    "kg_perturbation_fig8",
    "kg_perturbation_fig9",
    "kg_perturbation_final_assembly",
    "nature_iter",
    "redraw_v6a_best_fig1",
    "redraw_v6a_best_fig2",
    "redraw_v6a_best_fig3",
]


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def iter_matches(pattern: str) -> Iterable[Path]:
    yield from OUTPUTS_ROOT.glob(pattern)


def copy_file(src: Path, source_root: Path, dest_root: Path, copied: list[dict[str, str]]) -> None:
    if not src.is_file():
        return
    dest = dest_root / src.relative_to(source_root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    copied.append({"source": relpath(src), "final_path": relpath(dest)})


def copy_pattern(pattern: str, dest_root: Path, copied: list[dict[str, str]]) -> None:
    source_root = OUTPUTS_ROOT / pattern.split("/", 1)[0]
    for src in sorted(iter_matches(pattern)):
        if src.is_dir():
            continue
        if src.is_file():
            copy_file(src, source_root, dest_root, copied)


def write_manifest(copied: Sequence[dict[str, str]], removed_dirs: Sequence[str], dry_run: bool) -> None:
    manifest = {
        "final_root": relpath(FINAL_ROOT),
        "dry_run": dry_run,
        "copied_file_count": len(copied),
        "removed_source_directories": list(removed_dirs),
        "kept_top_level_output_directories": ["nature_final"],
        "directory_scheme": sorted(FINAL_DIRECTORIES),
        "copied_files": list(copied),
    }
    FINAL_ROOT.mkdir(parents=True, exist_ok=True)
    (FINAL_ROOT / "_final_output_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def organize_outputs(*, dry_run: bool = False) -> dict[str, object]:
    copied: list[dict[str, str]] = []
    if FINAL_ROOT.exists() and not dry_run:
        shutil.rmtree(FINAL_ROOT)
    for final_dir, patterns in FINAL_DIRECTORIES.items():
        dest_root = FINAL_ROOT / final_dir
        for pattern in patterns:
            copy_pattern(pattern, dest_root, copied)

    removed_dirs: list[str] = []
    if not dry_run:
        for name in SOURCE_DIRS_TO_REMOVE:
            path = OUTPUTS_ROOT / name
            if path.exists():
                shutil.rmtree(path)
                removed_dirs.append(relpath(path))
        for path in sorted(OUTPUTS_ROOT.iterdir()):
            if path.name != FINAL_ROOT.name:
                if path.is_dir():
                    shutil.rmtree(path)
                    removed_dirs.append(relpath(path))
                elif path.is_file():
                    path.unlink()
                    removed_dirs.append(relpath(path))
    write_manifest(copied, removed_dirs, dry_run)
    return {
        "final_root": relpath(FINAL_ROOT),
        "copied_file_count": len(copied),
        "removed_source_directories": removed_dirs,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidate final Nature outputs and remove intermediate output directories.")
    parser.add_argument("--dry-run", action="store_true", help="Build manifest without deleting old output directories.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = organize_outputs(dry_run=args.dry_run)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

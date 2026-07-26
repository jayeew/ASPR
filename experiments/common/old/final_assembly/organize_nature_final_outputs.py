"""Archived copy-only builder for the pre-v6.1 final figure package.

The original utility deleted every other top-level output after copying the
selected files. That behavior is intentionally disabled in the canonical
layout: historical artifacts may be copied into ``final_suite_rebuild``, but
the source trees are never removed.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
FINAL_ROOT = OUTPUTS_ROOT / "common" / "old" / "final_suite_rebuild"


FINAL_DIRECTORIES: dict[str, list[str]] = {
    "fig01_knowledge_perturbation": [
        "fig01/old/fig1_multi_domain_real.png",
        "fig01/old/fig1_multi_domain_real.svg",
        "fig01/old/fig1_multi_domain_real.pdf",
        "fig01/old/figure_quality_report.json",
        "fig01/old/run_manifest.json",
        "fig01/old/*/anchor_resolution_report.csv",
        "fig01/old/*/fig1_edge_sampling_manifest.csv",
        "fig01/old/*/snapshot_delta_metrics.csv",
        "fig01/old/*/perturbation_metrics.csv",
    ],
    "fig02_empirical_validation": [
        "fig02/old/fig2_empirical_full.png",
        "fig02/old/figure_quality_report.json",
        "fig02/old/run_manifest.json",
        "fig02/old/fig2_*.csv",
        "fig02/old/fig2_quality_gates.json",
        "fig02/old/fig2_strong_run_config.json",
    ],
    "fig03_score_learning": [
        "fig03/old/fig3_selected_weight_learning_full.png",
        "fig03/old/fig3_selected_weight_learning_full.svg",
        "fig03/old/figure_quality_report.json",
        "fig03/old/run_manifest.json",
        "fig03/old/fig3_reuse_decision.json",
        "fig03/old/fig3_run_selection.json",
        "fig03/old/multi_domain/fig3_weight_learning_full.png",
        "fig03/old/multi_domain/fig3_weight_learning_full.svg",
        "fig03/old/multi_domain/fig3_panel_e.png",
        "fig03/old/multi_domain/figure_quality_report.json",
        "fig03/old/multi_domain/run_manifest.json",
        "fig03/old/multi_domain/fig3_*.csv",
        "fig03/old/multi_domain/fig3_*.json",
        "fig03/old/multi_domain/coverage_constrained_weights.csv",
    ],
    "fig04_peer_review_alignment": [
        "fig04/old/fig4_full.png",
        "fig04/old/fig4_full.svg",
        "fig04/old/fig4_full.pdf",
        "fig04/old/figure_quality_report.json",
        "fig04/old/run_manifest.json",
        "fig04/old/fig4_metrics_summary.csv",
        "fig04/old/fig4_claim_scope_decision.csv",
        "fig04/old/fig4_claim_scope_decision.json",
        "fig04/old/fig4_global_score_coverage_audit.csv",
        "fig04/old/fig4_external_validation_*.csv",
        "fig04/old/fig4_blinded_labeling_*.csv",
        "fig04/old/fig4_completed_blinded_labels_*.csv",
        "fig04/old/fig4_blinded_labeling_protocol.md",
        "fig04/old/fig4_blinded_labeling_protocol.json",
        "fig04/old/fig4_fixed_sample_manifest.csv",
        "fig04/old/fig4_manifest.csv",
        "fig04/old/fig4_input_audit.csv",
        "fig04/old/fig4_agent_output_audit.csv",
    ],
    "fig05_ai_frontier": [
        "fig05/old/fig5_full.png",
        "fig05/old/fig5_full.svg",
        "fig05/old/figure_quality_report.json",
        "fig05/old/fig5_summary.json",
        "fig05/old/fig5_run_config.json",
        "fig05/old/ai_frontier/*",
    ],
    "fig06_robustness": [
        "fig06/old/fig6_full.png",
        "fig06/old/fig6_full.svg",
        "fig06/old/figure_quality_report.json",
        "fig06/old/fig6_caption.md",
        "fig06/old/fig6_panel_metadata.json",
        "fig06/old/fig6_panel_review.json",
        "fig06/old/fig6_*.csv",
    ],
    "fig07_venue_contribution": [
        "fig07/old/fig7_full.png",
        "fig07/old/figure_quality_report.json",
        "fig07/old/run_manifest.json",
        "fig07/old/fig7_methods.md",
        "fig07/old/fig7_gap_list.md",
        "fig07/old/fig7_*.csv",
        "fig07/old/fig7_panel_chart_map.json",
    ],
    "fig08_aspr_framework": [
        "fig08/old/fig8_full.png",
        "fig08/old/figure_quality_report.json",
        "fig08/old/run_manifest.json",
        "fig08/old/fig8_gpt_image2_prompt.md",
        "fig08/old/fig8_handoff_manifest.json",
    ],
    "fig09_case_run_instance": [
        "fig09/old/*",
    ],
    "fig10_ablation_evidence": [
        "fig10/old/fig10_full.png",
        "fig10/old/fig10_full.svg",
        "fig10/old/figure_quality_report.json",
        "fig10/old/run_manifest.json",
        "fig10/old/fig10_*.csv",
        "fig10/old/fig10_*.json",
        "fig10/old/fig10_*.jsonl",
        "fig10/old/fig10_blinded_preference_protocol.md",
        "fig10/old/true_reruns/**/*",
    ],
    "assembly": [
        "common/old/final_assembly/*.csv",
        "common/old/final_assembly/*.json",
        "common/old/final_assembly/*.md",
        "common/old/final_assembly/fig1_fig10_contact_sheet.png",
        "common/old/final_assembly/external_evidence_distribution/*.zip",
    ],
    "iteration_final_patch": [
        "nature_iter/r6/*",
        "nature_iter/final_patch/*",
    ],
}


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
        "kept_top_level_output_directories": [
            "common",
            *[f"fig{figure:02d}" for figure in range(1, 11)],
        ],
        "directory_scheme": sorted(FINAL_DIRECTORIES),
        "copied_files": list(copied),
    }
    FINAL_ROOT.mkdir(parents=True, exist_ok=True)
    (FINAL_ROOT / "_final_output_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def organize_outputs(*, dry_run: bool = True) -> dict[str, object]:
    """Copy the archived selection without deleting any source artifact."""
    copied: list[dict[str, str]] = []
    if FINAL_ROOT.exists() and not dry_run:
        shutil.rmtree(FINAL_ROOT)
    for final_dir, patterns in FINAL_DIRECTORIES.items():
        dest_root = FINAL_ROOT / final_dir
        for pattern in patterns:
            if dry_run:
                source_root = OUTPUTS_ROOT / pattern.split("/", 1)[0]
                for source in sorted(iter_matches(pattern)):
                    if source.is_file():
                        copied.append(
                            {
                                "source": relpath(source),
                                "final_path": relpath(
                                    dest_root / source.relative_to(source_root)
                                ),
                            }
                        )
            else:
                copy_pattern(pattern, dest_root, copied)

    removed_dirs: list[str] = []
    if not dry_run:
        write_manifest(copied, removed_dirs, dry_run)
    return {
        "final_root": relpath(FINAL_ROOT),
        "copied_file_count": len(copied),
        "removed_source_directories": removed_dirs,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy the archived Nature output selection without deleting sources."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the copy-only rebuild; the default is a dry run.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = organize_outputs(dry_run=not args.apply)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

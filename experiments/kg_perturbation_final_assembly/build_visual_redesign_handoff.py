#!/usr/bin/env python3
"""Build visual-redesign handoff prompts for remaining Nature reading-pass gaps."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.figure_quality import write_json, write_run_manifest


DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "kg_perturbation_final_assembly" / "visual_redesign_handoff"

FIGURE_HANDOFFS: Mapping[str, Dict[str, Any]] = {
    "Fig.6": {
        "title": "Robustness atlas replacing line-chart-heavy Fig.6",
        "source_artifacts": [
            "outputs/kg_perturbation_fig6/fig6_rank_stability.csv",
            "outputs/kg_perturbation_fig6/fig6_indicator_stability.csv",
            "outputs/kg_perturbation_fig6/fig6_full_rerun_manifest.csv",
            "outputs/kg_perturbation_fig6/figure_quality_report.json",
        ],
        "visual_goal": "Fuse robustness evidence into an atlas/matrix/badge composition instead of many line charts.",
        "must_include": [
            "construction-matched full graph rerun",
            "rank stability",
            "indicator stability",
            "boundary-condition badges",
            "proxy and full-rerun distinction",
        ],
        "must_avoid": [
            "more than two trend-line panels",
            "claiming robustness beyond the quality report",
            "dense caption-like paragraphs inside the figure",
        ],
    },
    "Fig.7": {
        "title": "Venue contribution compact matrix replacing text-heavy Fig.7",
        "source_artifacts": [
            "outputs/kg_perturbation_fig7/fig7_vci_rankings.csv",
            "outputs/kg_perturbation_fig7/fig7_pairwise_contribution_tests.csv",
            "outputs/kg_perturbation_fig7/fig7_metric_sensitivity.csv",
            "outputs/kg_perturbation_fig7/figure_quality_report.json",
        ],
        "visual_goal": "Reduce panel count and convert text-heavy caveats into a compact venue atlas, uncertainty matrix, and caveat badges.",
        "must_include": [
            "Nature Portfolio point estimate",
            "interval caveat",
            "pairwise uncertainty",
            "field-year controls",
            "no causal-superiority wording",
        ],
        "must_avoid": [
            "large paragraph panel",
            "strict dominance claim",
            "crowded repeated line charts",
        ],
    },
    "Fig.9": {
        "title": "Single large auditable ASPR run instance",
        "source_artifacts": [
            "outputs/kg_perturbation_fig9/fig9_case_manifest.json",
            "outputs/kg_perturbation_fig9/fig9_claim_evidence_trace.csv",
            "outputs/kg_perturbation_fig9/fig9_aspr_qwen_output.json",
            "outputs/kg_perturbation_fig9/figure_quality_report.json",
        ],
        "visual_goal": "Replace text-heavy storyboard panels with one large run-instance visual bound to the case manifest.",
        "must_include": [
            "input manuscript",
            "evidence trace",
            "graph-agent assessment",
            "checkpoint ASPR-Qwen lane",
            "fusion and verifier",
            "final review output",
        ],
        "must_avoid": [
            "paragraph-heavy panels",
            "aggregate performance claim",
            "unbound invented review content",
        ],
    },
    "Fig.10": {
        "title": "Ablation evidence and replacement-gate atlas",
        "source_artifacts": [
            "outputs/kg_perturbation_fig10/fig10_evidence_provenance.csv",
            "outputs/kg_perturbation_fig10/fig10_replacement_gates.csv",
            "outputs/kg_perturbation_fig10/fig10_generic_llm_same_rubric_summary.csv",
            "outputs/kg_perturbation_fig10/figure_quality_report.json",
        ],
        "visual_goal": "Unify tone with Fig.1-Fig.9 and reduce panel count around module evidence, same-rubric baseline, and replacement gates.",
        "must_include": [
            "full ASPR observed metrics",
            "generic LLM proxy-vs-same-rubric discrepancy",
            "true rerun gate",
            "blinded human preference gate",
            "checkpoint gate",
        ],
        "must_avoid": [
            "ASPR superiority claim",
            "many small panels",
            "palette disconnected from the shared style ledger",
        ],
    },
}


def relpath(path: Path) -> str:
    """Return a project-relative path when possible."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def artifact_status(rel_paths: Sequence[str]) -> List[Dict[str, Any]]:
    """Return source-artifact status rows."""
    rows: List[Dict[str, Any]] = []
    for rel in rel_paths:
        path = PROJECT_ROOT / rel
        rows.append(
            {
                "artifact_path": rel,
                "exists": int(path.exists()),
                "size_bytes": int(path.stat().st_size) if path.exists() else 0,
            }
        )
    return rows


def render_prompt(figure: str, spec: Mapping[str, Any], artifacts: Sequence[Mapping[str, Any]]) -> str:
    """Render one concise image/design handoff prompt."""
    payload = {
        "figure": figure,
        "title": spec["title"],
        "visual_goal": spec["visual_goal"],
        "source_artifacts": list(artifacts),
        "must_include": spec["must_include"],
        "must_avoid": spec["must_avoid"],
        "style": {
            "journal_target": "Nature-level research figure",
            "palette": "white/gray canvas with graph blue, Nature red, ASPR-Qwen purple, verifier orange, fusion black",
            "text_policy": "short labels only; no paragraph panels; no take-home footer",
            "claim_policy": "visual emphasis must not exceed the claim boundary in source quality reports",
        },
    }
    return (
        f"# {figure} Visual Redesign Handoff\n\n"
        f"{spec['visual_goal']}\n\n"
        "Use the exact source artifacts below. Do not invent new data, claims, module outputs, or performance numbers.\n\n"
        "```json\n"
        + json.dumps(payload, indent=2, ensure_ascii=False)
        + "\n```\n"
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write rows to CSV."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_handoff(out_dir: Path) -> Dict[str, Any]:
    """Build prompts and manifest for all remaining visual redesigns."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: List[Dict[str, Any]] = []
    artifact_rows: List[Dict[str, Any]] = []
    for figure, spec in FIGURE_HANDOFFS.items():
        artifacts = artifact_status(spec["source_artifacts"])
        artifact_rows.extend({"figure": figure, **row} for row in artifacts)
        prompt_path = out_dir / f"{figure.lower().replace('.', '')}_visual_redesign_prompt.md"
        prompt_path.write_text(render_prompt(figure, spec, artifacts), encoding="utf-8")
        manifest_rows.append(
            {
                "figure": figure,
                "prompt_path": relpath(prompt_path),
                "source_artifact_count": len(artifacts),
                "source_artifacts_present": int(all(int(row["exists"]) == 1 for row in artifacts)),
                "visual_goal": spec["visual_goal"],
                "handoff_status": "ready" if all(int(row["exists"]) == 1 for row in artifacts) else "missing_source_artifact",
            }
        )
    write_csv(out_dir / "visual_redesign_handoff_manifest.csv", manifest_rows)
    write_csv(out_dir / "visual_redesign_source_artifacts.csv", artifact_rows)
    quality_gates = {
        "overall_pass": bool(manifest_rows and all(int(row["source_artifacts_present"]) == 1 for row in manifest_rows)),
        "status_label": "visual_redesign_handoff_ready",
        "checks": {
            "all_prompts_written": int(all((PROJECT_ROOT / row["prompt_path"]).exists() for row in manifest_rows)),
            "all_source_artifacts_present": int(all(int(row["source_artifacts_present"]) == 1 for row in manifest_rows)),
            "four_redesign_figures_covered": int({row["figure"] for row in manifest_rows} == {"Fig.6", "Fig.7", "Fig.9", "Fig.10"}),
        },
    }
    write_json(out_dir / "visual_redesign_quality_report.json", quality_gates)
    write_run_manifest(
        out_dir,
        figure="fig6_fig7_fig9_fig10_visual_redesign_handoff",
        argv=sys.argv,
        inputs={"figures": sorted(FIGURE_HANDOFFS.keys())},
        quality_gates=quality_gates,
        extra={"manifest": relpath(out_dir / "visual_redesign_handoff_manifest.csv")},
    )
    return {"quality_gates": quality_gates, "manifest_rows": manifest_rows}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build visual redesign handoff prompts.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Command-line entry point."""
    args = parse_args(argv)
    result = build_handoff(args.out_dir)
    print(f"[visual-redesign] wrote {args.out_dir}")
    print(json.dumps(result["quality_gates"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

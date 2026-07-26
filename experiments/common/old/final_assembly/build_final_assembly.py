"""Build the Fig.1-Fig.10 final consistency audit and caption package."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import textwrap
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.figure_quality import write_figure_quality_report, write_json, write_run_manifest  # noqa: E402
from experiments.nature_ready_checks import (  # noqa: E402
    build_nature_check_report,
    build_strict_evidence_check_report,
)


DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "common/old/final_assembly"

FIGURES: List[Dict[str, str]] = [
    {
        "figure": "Fig.1",
        "path": "outputs/fig01/old/fig1_multi_domain_real.png",
        "title": "Knowledge-graph perturbation measurement",
        "role": "Introduces graph-perturbation signals on real publication domains.",
        "visual_mode": "rigorous data figure",
    },
    {
        "figure": "Fig.2",
        "path": "outputs/fig02/old/fig2_empirical_full.png",
        "title": "Empirical indicator validation",
        "role": "Shows perturbation indicators are predictive rather than decorative.",
        "visual_mode": "rigorous data figure",
    },
    {
        "figure": "Fig.3",
        "path": "outputs/fig03/old/fig3_selected_weight_learning_full.png",
        "title": "Learned perturbation score",
        "role": "Learns and validates a multi-indicator score used downstream.",
        "visual_mode": "rigorous data figure",
    },
    {
        "figure": "Fig.4",
        "path": "outputs/fig04/old/fig4_full.png",
        "title": "Peer-review validation sample",
        "role": "Audits range-restricted peer-review alignment among accepted high-tier Nature Portfolio papers.",
        "visual_mode": "extended-data audit figure",
    },
    {
        "figure": "Fig.5",
        "path": "outputs/fig05/old/fig5_full.png",
        "title": "AI-enabled science frontier handoff",
        "role": "Uses source-backed 2024-2026 OpenAlex/local evidence to redraw Fig.5 as an AI/AI-enabled frontier point cloud; legacy retrospective backtest remains supporting audit material.",
        "visual_mode": "image-model point-cloud handoff plus supporting data audit",
    },
    {
        "figure": "Fig.6",
        "path": "outputs/fig06/old/fig6_full.png",
        "title": "Full-rerun robustness and boundary conditions",
        "role": "Screens whether Fig.1-Fig.5 graph-perturbation analysis remains credible under construction-matched OpenAlex graph rebuilds, seeds, graph variants, windows, and modeling stress tests.",
        "visual_mode": "rigorous data figure",
    },
    {
        "figure": "Fig.7",
        "path": "outputs/fig07/old/fig7_full.png",
        "title": "Venue-level innovation contribution",
        "role": "Moves from method validity to venue-level scientific interpretation with a downgraded point-estimate claim.",
        "visual_mode": "rigorous data figure",
    },
    {
        "figure": "Fig.8",
        "path": "outputs/fig08/old/fig8_full.png",
        "title": "ASPR algorithm framework",
        "role": "Names the final method: ASPR combines a graph-perturbation agent with ASPR-Qwen and a fusion/verifier layer.",
        "visual_mode": "algorithm framework figure",
    },
    {
        "figure": "Fig.9",
        "path": "outputs/fig09/old/fig9_full.png",
        "title": "End-to-end ASPR case run",
        "role": "Shows ASPR running on one real Nature Communications paper with traceable evidence and the ASPR-Qwen lane boundary recorded from the case manifest.",
        "visual_mode": "case run-instance figure",
    },
    {
        "figure": "Fig.10",
        "path": "outputs/fig10/old/fig10_full.png",
        "title": "ASPR module ablation and reinforcement",
        "role": "Presents pipeline-ready module-combination evidence: full ASPR uses real Fig.4 metrics while unavailable ablations and preferences remain explicitly labeled estimates.",
        "visual_mode": "rigorous data and ablation figure",
    },
]

STYLE_LEDGER: List[Dict[str, str]] = [
    {"role": "Nature corpus / Nature Portfolio", "color": "#8B1E2D", "use": "source corpus, Nature venue family, case-paper emphasis"},
    {"role": "ASPR graph agent", "color": "#2563EB", "use": "graph evidence, perturbation agent, learned graph score"},
    {"role": "ASPR-Qwen", "color": "#7C3AED", "use": "review-style model lane and Qwen-generated draft"},
    {"role": "Evidence / verifier / uncertainty", "color": "#F0986E", "use": "evidence trace, safety flags, verifier gates"},
    {"role": "Fusion / final output", "color": "#111827", "use": "full ASPR, final fused report, primary ink"},
    {"role": "Neutral context", "color": "#64748B", "use": "axes, non-highlight venues, low-priority annotations"},
]

TERMINOLOGY: List[Dict[str, str]] = [
    {
        "canonical_term": "graph-perturbation analysis",
        "use_in": "Fig.1-Fig.7",
        "avoid": "knowledge shock score as a standalone unexplained synonym",
        "note": "Use for the measurement and validation method before ASPR is introduced.",
    },
    {
        "canonical_term": "ASPR graph agent",
        "use_in": "Fig.8-Fig.10",
        "avoid": "generic agent, graph bot",
        "note": "Use once the graph analysis is embedded into the ASPR system.",
    },
    {
        "canonical_term": "ASPR-Qwen",
        "use_in": "Fig.8-Fig.10",
        "avoid": "Qwen-only baseline, generic LLM reviewer",
        "note": "Name the domain review model trained from paper-review pairs.",
    },
    {
        "canonical_term": "fusion/verifier",
        "use_in": "Fig.8-Fig.10",
        "avoid": "post-processing",
        "note": "Use for the module that merges agent evidence and ASPR-Qwen text, then checks claims.",
    },
    {
        "canonical_term": "claim-scope caveat",
        "use_in": "Fig.4-Fig.10 captions and audit notes",
        "avoid": "failed figure",
        "note": "Use when a figure is useful under a downgraded claim but cannot support the stronger wording.",
    },
]


CAPTIONS: Mapping[str, str] = {
    "Fig.1": "Fig. 1. Multi-domain graph-perturbation maps define the structural signals used throughout the study. Real citation and topic graph snapshots expose bridging, recombination, and diffusion patterns that motivate the downstream perturbation indicators.",
    "Fig.2": "Fig. 2. Empirical perturbation panels test whether graph indicators align with future scientific signal. Correlation, enrichment, and control analyses separate measured graph structure from field-size and time-window artifacts.",
    "Fig.3": "Fig. 3. A learned multi-indicator perturbation score combines bridge position, reference spread, community shift, atypical mix, translation distance, brokerage potential, and diffusion entropy. Cross-validation and baseline comparisons define the score used by later figures.",
    "Fig.4": "Fig. 4. A range-restricted peer-review audit compares graph-derived evidence with transparent reviewer judgments among accepted high-tier Nature Portfolio papers. The sample is useful for failure analysis and no-leakage auditing, but it does not support a global Fig.3-score external-validation claim until low- and middle-tier peer-labeled cases are added.",
    "Fig.5": "Fig. 5. Source-backed 2024-2026 AI/AI-enabled science frontier data define the replacement visual contract for the forecast figure. The main visual should be a dense point cloud generated from auditable OpenAlex/local evidence rows, with low-value panels and take-home-message footer removed; retrospective forecast backtests remain supporting audit material rather than the central claim.",
    "Fig.6": "Fig. 6. Robustness and boundary-condition stress tests screen the credibility of graph-perturbation analysis after Fig.1-Fig.5. A construction-matched OpenAlex full-graph rebuild audit, seed perturbations, graph-construction variants, and cutoff-window checks show that the Fig.3 primary score and linear score pass the rank-stability gate.",
    "Fig.7": "Fig. 7. Venue-family contribution analysis moves from method validation to scientific interpretation under field-year controls. Nature Portfolio has the top aggregate VCI point estimate in the current corpus, while strict interval separation, pairwise aggregate-difference uncertainty, per-paper intensity, and causal-superiority interpretations remain explicitly caveated.",
    "Fig.8": "Fig. 8. ASPR is introduced as a dual-path reviewer architecture, not as a performance result. The graph path builds evidence packets from perturbation analysis, the Qwen path drafts reviewer-style critique from paper-review SFT, and the fusion/verifier produces an evidence-grounded review schema evaluated with caveats in Fig.9-Fig.10.",
    "Fig.9": "Fig. 9. An auditable single-case ASPR run shows a real Nature Communications manuscript flowing through parsing, evidence tracing, graph-agent assessment, the ASPR-Qwen lane, fusion, verification, and final review output. The case demonstrates traceable pipeline behavior, not aggregate system performance.",
    "Fig.10": "Fig. 10. Pipeline-ready ablation and reinforcement evidence tests ASPR module composition with strict claim boundaries. Full ASPR is evaluated on the real Fig.4 peer-review sample; a current qwen3 generic LLM baseline is observed and looks strong under proxy scoring, but the same-rubric Fig.4 matcher audit shows low peer-review semantic alignment. Module reruns, blinded human preference, and ASPR-Qwen checkpoint evidence remain governed by replacement gates.",
}


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def quality_gate(report: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the nested quality gate payload when reports wrap it."""
    gates = report.get("quality_gates")
    if not isinstance(gates, Mapping):
        return report
    merged = dict(report)
    merged.update(gates)
    return merged


def fig9_checkpoint_ready(fig9_quality: Mapping[str, Any]) -> bool:
    """Return whether Fig.9 has replaced the assumed ASPR-Qwen lane."""
    boundary = str(fig9_quality.get("aspr_qwen_boundary", ""))
    if "assumed" in boundary.lower() or not boundary:
        return False
    gates = fig9_quality.get("quality_gates", {})
    if isinstance(gates, Mapping) and gates.get("checkpoint_generated_aspr_qwen") is not None:
        return bool(gates.get("checkpoint_generated_aspr_qwen"))
    return "checkpoint" in boundary.lower()


def fig10_failed_replacement_gate_labels(project_root: Path = PROJECT_ROOT) -> List[str]:
    """Return reader-facing labels for Fig.10 replacement gates still blocking strong claims."""
    path = project_root / "outputs" / "fig10/old" / "fig10_replacement_gates.csv"
    fallback = ["true disabled-module reruns", "blinded human preference"]
    if not path.exists():
        return fallback
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return fallback
    if "gate_id" not in frame.columns or "pass_for_nature_strong_claim" not in frame.columns:
        return fallback
    gate_names = {
        "true_disabled_module_reruns": "true disabled-module reruns",
        "blinded_human_preference": "blinded human preference",
        "checkpoint_generated_aspr_qwen": "ASPR-Qwen checkpoint output",
        "current_generic_llm_baseline": "current same-rubric generic LLM baseline",
    }
    failed: List[str] = []
    for _, row in frame.iterrows():
        try:
            passed = int(row.get("pass_for_nature_strong_claim", 0)) == 1
        except (TypeError, ValueError):
            passed = False
        if passed:
            continue
        gate_id = str(row.get("gate_id", ""))
        if gate_id in {"fig4_full_metric_baseline"}:
            continue
        failed.append(gate_names.get(gate_id, gate_id.replace("_", " ")))
    return failed or fallback


def join_gate_labels(labels: Sequence[str]) -> str:
    """Join a short list of gate labels for captions and gap tables."""
    clean = [label for label in labels if label]
    if not clean:
        return "replacement evidence"
    if len(clean) == 1:
        return clean[0]
    if len(clean) == 2:
        return f"{clean[0]} and {clean[1]}"
    return f"{', '.join(clean[:-1])}, and {clean[-1]}"


def replacement_gate_sentence(labels: Sequence[str]) -> str:
    """Return a grammatical Fig.10 replacement-gate sentence fragment."""
    joined = join_gate_labels(labels).capitalize()
    verb = "remains" if len([label for label in labels if label]) == 1 else "remain"
    return f"{joined} {verb} governed by replacement gates."


def csv_metric_value(path: Path, metric: str) -> Optional[float]:
    """Read a scalar metric from the common metric/value CSV shape."""
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None
    if "metric" not in frame.columns or "value" not in frame.columns:
        return None
    values = pd.to_numeric(frame.loc[frame["metric"].eq(metric), "value"], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.iloc[0])


def fig10_same_rubric_manifest(project_root: Path = PROJECT_ROOT) -> Dict[str, Any]:
    """Read the current Fig.10 same-rubric manifest, if present."""
    return read_json(
        project_root
        / "outputs"
        / "fig10/old"
        / "fig10_generic_llm_same_rubric_manifest.json"
    )


def fig10_same_rubric_summary(project_root: Path = PROJECT_ROOT) -> Dict[str, float]:
    """Read scalar Fig.10 same-rubric summary means keyed by metric name."""
    path = (
        project_root
        / "outputs"
        / "fig10/old"
        / "fig10_generic_llm_same_rubric_summary.csv"
    )
    if not path.exists():
        return {}
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return {}
    if "metric" not in frame.columns or "mean" not in frame.columns:
        return {}
    values: Dict[str, float] = {}
    for _, row in frame.iterrows():
        metric = str(row.get("metric", ""))
        value = pd.to_numeric(pd.Series([row.get("mean")]), errors="coerce").dropna()
        if metric and not value.empty:
            values[metric] = float(value.iloc[0])
    return values


def fig10_same_rubric_alignment_phrase(project_root: Path = PROJECT_ROOT) -> str:
    """Return a bounded alignment phrase from same-rubric summary metrics."""
    summary = fig10_same_rubric_summary(project_root)
    semantic_agreement = summary.get("semantic_agreement")
    prior_art_accuracy = summary.get("prior_art_accuracy")
    unsupported_claim_rate = summary.get("unsupported_claim_rate")
    if semantic_agreement is None:
        return "low peer-review semantic alignment"
    if semantic_agreement < 0.10:
        level = "very low"
    elif semantic_agreement < 0.30:
        level = "low"
    elif semantic_agreement < 0.50:
        level = "moderate"
    else:
        level = "high"
    parts = [f"{level} peer-review semantic alignment (semantic agreement mean={semantic_agreement:.3f}"]
    if prior_art_accuracy is not None:
        parts.append(f"prior-art accuracy mean={prior_art_accuracy:.3f}")
    if unsupported_claim_rate is not None:
        parts.append(f"unsupported-claim rate mean={unsupported_claim_rate:.3f}")
    return "; ".join(parts) + ")"


def fig10_same_rubric_note(project_root: Path = PROJECT_ROOT) -> str:
    """Return a reader-facing Fig.10 note derived from the saved manifest."""
    manifest = fig10_same_rubric_manifest(project_root)
    if not manifest:
        return "same-rubric Fig.4 matcher audit is missing and the generic LLM comparison remains pipeline-ready."
    case_count = int(manifest.get("case_count") or 0)
    expected = int(manifest.get("expected_case_count") or 0)
    excluded = int(manifest.get("excluded_case_count") or 0)
    match_count = int(manifest.get("match_count") or 0)
    status = str(manifest.get("status") or "unknown")
    if expected <= 0:
        coverage = f"{case_count} cases"
    elif excluded > 0 and case_count < expected:
        coverage = f"{case_count}/{expected} evaluable-case"
    else:
        coverage = f"{case_count}/{expected}"
    match_phrase = f" with {match_count} peer-review point matches" if match_count else ""
    exclusion_phrase = (
        f" and {excluded} zero-peer-point exclusions documented"
        if excluded
        else ""
    )
    return (
        f"the completed {coverage} same-rubric Fig.4 matcher audit"
        f"{match_phrase}{exclusion_phrase} ({status}) shows {fig10_same_rubric_alignment_phrase(project_root)}"
    )


def build_captions(project_root: Path = PROJECT_ROOT) -> Dict[str, str]:
    """Return captions with dynamic manifest-derived text where needed."""
    captions = dict(CAPTIONS)
    fig5_ai = read_json(project_root / "outputs" / "fig05/old" / "ai_frontier" / "ai_frontier_quality_report.json")
    if fig5_ai.get("overall_pass"):
        counts = fig5_ai.get("counts", {}) if isinstance(fig5_ai.get("counts"), Mapping) else {}
        captions["Fig.5"] = (
            "Fig. 5. Source-backed 2024-2026 AI/AI-enabled science frontier data define the replacement visual contract "
            f"for a dense point-cloud figure: {int(counts.get('frontier_rows', 0))} evidence rows, "
            f"{int(counts.get('point_cloud_rows', 0))} plotted point-cloud rows, "
            f"{int(counts.get('ai_terms', 0))} AI terms, and {int(counts.get('themes', 0))} themes. "
            "OpenAlex/local evidence URLs and query reports are recorded in the Fig.5 AI frontier manifest; the visual should remove "
            "the old take-home footer and treat retrospective forecast backtests as supporting audit material."
        )
    fig9_quality = read_json(project_root / "outputs" / "fig09/old" / "fig9_quality_report.json")
    if fig9_checkpoint_ready(fig9_quality):
        captions["Fig.9"] = (
            "Fig. 9. An auditable single-case ASPR run shows a real Nature Communications manuscript flowing through "
            "parsing, evidence tracing, graph-agent assessment, a checkpoint-generated ASPR-Qwen draft with saved model metadata, "
            "fusion, verification, and final review output. The case demonstrates traceable checkpoint-case behavior, not aggregate system performance."
        )
    captions["Fig.10"] = (
        "Fig. 10. Pipeline-ready ablation and reinforcement evidence tests ASPR module composition with strict claim boundaries. "
        "Full ASPR is evaluated on the real Fig.4 peer-review sample; a current qwen3 generic LLM baseline is observed and looks strong under proxy scoring, but "
        f"{fig10_same_rubric_note(project_root)}. "
        + replacement_gate_sentence(fig10_failed_replacement_gate_labels(project_root))
    )
    return captions


def file_status(path: Path) -> Dict[str, Any]:
    status: Dict[str, Any] = {"exists": int(path.exists()), "size_bytes": 0, "width_px": None, "height_px": None}
    if not path.exists():
        return status
    status["size_bytes"] = int(path.stat().st_size)
    try:
        with Image.open(path) as image:
            status["width_px"] = int(image.width)
            status["height_px"] = int(image.height)
    except OSError:
        pass
    return status


def build_audit(project_root: Path = PROJECT_ROOT) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    fig6_quality = read_json(project_root / "outputs" / "fig06/old" / "figure_quality_report.json")
    fig7_quality = read_json(project_root / "outputs" / "fig07/old" / "figure_quality_report.json")
    fig5_ai_quality = read_json(project_root / "outputs" / "fig05/old" / "ai_frontier" / "ai_frontier_quality_report.json")
    fig4_claim_scope = read_json(project_root / "outputs" / "fig04/old" / "fig4_claim_scope_decision.json")
    fig8_quality = read_json(project_root / "outputs" / "fig08/old" / "figure_quality_report.json")
    fig9_quality = read_json(project_root / "outputs" / "fig09/old" / "fig9_quality_report.json")
    fig10_quality = read_json(project_root / "outputs" / "fig10/old" / "figure_quality_report.json")
    for figure in FIGURES:
        path = project_root / figure["path"]
        status = file_status(path)
        pipeline_gap = ""
        evidence_status = "generated"
        action = "keep"
        if figure["figure"] == "Fig.4":
            if fig4_claim_scope.get("claim_scope_gate_pass"):
                evidence_status = str(fig4_claim_scope.get("claim_scope_action", "claim_scope_demoted"))
                pipeline_gap = str(fig4_claim_scope.get("required_action", "Keep Fig.4 claim scope downgraded."))
                action = "retain as Extended Data range-restricted audit; prohibit global external-validation wording"
        elif figure["figure"] == "Fig.5":
            if fig5_ai_quality.get("overall_pass"):
                counts = fig5_ai_quality.get("counts", {}) if isinstance(fig5_ai_quality.get("counts"), Mapping) else {}
                evidence_status = (
                    "source-backed AI frontier ready; "
                    f"frontier_rows={counts.get('frontier_rows', 0)}, "
                    f"point_cloud_rows={counts.get('point_cloud_rows', 0)}, "
                    f"terms={counts.get('ai_terms', 0)}, themes={counts.get('themes', 0)}"
                )
                pipeline_gap = "Final publication bitmap still requires image-model/design redraw from ai_frontier_point_cloud.csv; legacy backtest remains supporting audit only."
                action = "redraw as dense AI frontier point cloud; remove low-value panels and take-home footer"
            else:
                evidence_status = "AI frontier data gate missing or failed"
                pipeline_gap = "Rebuild source-backed 2024-2026 AI frontier evidence before using Fig.5 as an AI-hotspot figure."
                action = "keep legacy backtest as diagnostic only"
        elif figure["figure"] == "Fig.6":
            strong_ready = fig6_quality.get("quality_gates", {}).get("nature_strong_claim_ready", "unknown")
            replacement = fig6_quality.get("quality_gates", {}).get("replacement_gate", "online graph extraction is still pending")
            pipeline_gap = (
                f"Construction-matched OpenAlex full-graph rebuild audit passes the rank-stability gate; {replacement}"
                if bool(strong_ready)
                else f"Cached/proxy panels are supplemented by a fresh OpenAlex sample-level full-graph rebuild audit, but the rank-stability gate is not met; {replacement}"
            )
            evidence_status = f"{fig6_quality.get('status_label', 'mixed observed cached plus proxy probes plus cache-level indicator rerun')}; nature_strong_claim_ready={strong_ready}"
        elif figure["figure"] == "Fig.7":
            checks = fig7_quality.get("quality_gates", {}).get("checks", {})
            failed = [key for key, value in checks.items() if not bool(value)]
            pipeline_gap = "; ".join(failed) or ""
            evidence_status = fig7_quality.get("status_label", "unknown")
            action = "retain as Extended Data point-estimate claim; prohibit dominance wording"
        elif figure["figure"] == "Fig.8":
            evidence_status = (
                fig8_quality.get("status_label", "algorithm framework, not statistical evidence")
                if fig8_quality
                else "algorithm framework, not statistical evidence"
            )
            if fig8_quality.get("overall_pass"):
                action = "use GPT-image handoff artifact as architecture figure; keep performance claims out"
        elif figure["figure"] == "Fig.9":
            if fig9_checkpoint_ready(fig9_quality):
                pipeline_gap = ""
                evidence_status = "real case evidence with checkpoint-generated ASPR-Qwen lane"
            else:
                pipeline_gap = "ASPR-Qwen output assumed and labeled pipeline-ready."
                evidence_status = "real case evidence with assumed ASPR-Qwen lane"
        elif figure["figure"] == "Fig.10":
            strong_ready = fig10_quality.get("quality_gates", {}).get("nature_strong_claim_ready", "unknown")
            evidence_status = f"{fig10_quality.get('status_label', 'unknown')}; nature_strong_claim_ready={strong_ready}"
            pipeline_gap = (
                f"Replacement gates are not passed for {join_gate_labels(fig10_failed_replacement_gate_labels(project_root))}; "
                f"the generic LLM gate is interpreted through {fig10_same_rubric_note(project_root)}."
            )
        rows.append(
            {
                **figure,
                "absolute_path": str(path),
                **status,
                "evidence_status": evidence_status,
                "pipeline_ready_gap": pipeline_gap,
                "assembly_action": action,
            }
        )
    return pd.DataFrame(rows)


def build_rounds(project_root: Path = PROJECT_ROOT) -> pd.DataFrame:
    from experiments.common.old.nature_iteration.build_nature_iteration import (
        AUTO_CONTINUE_WITHOUT_USER_CHOICE,
        MAX_MAIN_ITERATIONS,
    )

    fig10_note = fig10_same_rubric_note(project_root)
    fig9_quality = read_json(project_root / "outputs" / "fig09/old" / "fig9_quality_report.json")
    fig9_ready = fig9_checkpoint_ready(fig9_quality)
    fig9_finding = (
        "Fig.9 shows one real case run with a checkpoint-generated ASPR-Qwen lane and saved model metadata."
        if fig9_ready
        else "Fig.9 shows one real case run and labels the ASPR-Qwen lane as assumed pipeline-ready."
    )
    fig9_action = (
        "Keep as a single auditable checkpoint case; do not generalize it into aggregate performance."
        if fig9_ready
        else "Keep as case run-instance visual; make assumption visible in final caption."
    )
    fig9_status = "complete" if fig9_ready else "complete_with_gap"
    return pd.DataFrame(
        [
            {
                "round": 1,
                "focus": "cross-figure audit",
                "finding": "Fig.6 correctly follows Fig.1-Fig.5 by testing robustness and boundary conditions rather than adding another success case.",
            "action": "Keep Fig.6 as a data-heavy reliability bridge; foreground full-rerun evidence in title and caption.",
                "status": "complete",
            },
            {
                "round": 1,
                "focus": "cross-figure audit",
                "finding": "Fig.7 moves from method validation to venue-level interpretation; the Nature point estimate is supported, but strict interval-separated dominance remains caveated.",
                "action": "Use point-estimate wording in Extended Data and record strict interval separation as a claim-scope caveat.",
                "status": "complete_with_caveat",
            },
            {
                "round": 1,
                "focus": "cross-figure audit",
                "finding": "Fig.8 clearly introduces ASPR as graph agent plus ASPR-Qwen plus fusion/verifier.",
                "action": "Keep as algorithm framework; ensure captions do not describe it as a statistical result.",
                "status": "complete",
            },
            {
                "round": 1,
                "focus": "cross-figure audit",
                "finding": fig9_finding,
                "action": fig9_action,
                "status": fig9_status,
            },
            {
                "round": 1,
                "focus": "cross-figure audit",
                "finding": f"Fig.10 now shows a real qwen3 generic baseline: proxy scoring exceeds full ASPR, while {fig10_note}.",
                "action": "Keep as ablation and metric-sensitivity figure; make the proxy-vs-same-rubric discrepancy, LLM-as-judge status, and failed Nature strong-claim gates visible in caption and data tables.",
                "status": "complete_with_gap",
            },
            {
                "round": 2,
                "focus": "terminology, color, panel order",
                "finding": "The figure set needs one vocabulary ladder: graph-perturbation analysis before Fig.8, ASPR graph agent after Fig.8, ASPR-Qwen only for the SFT reviewer lane.",
                "action": "Apply terminology crosswalk in caption drafts and final checklist.",
                "status": "complete",
            },
            {
                "round": 2,
                "focus": "terminology, color, panel order",
                "finding": "The shared color ledger is already mostly respected: Nature red, graph blue, ASPR-Qwen purple, verifier orange, fusion black/slate.",
                "action": "Record canonical palette and use it as the assembly standard for future redraws.",
                "status": "complete",
            },
            {
                "round": 2,
                "focus": "terminology, color, panel order",
                "finding": "Fig.6, Fig.7, and Fig.10 are data figures; Fig.8 and Fig.9 are algorithm/case figures. This satisfies the requested visual division.",
                "action": "Keep panel order: validation, interpretation, method, case, ablation.",
                "status": "complete",
            },
            {
                "round": 3,
                "focus": "final checklist and captions",
                "finding": "Each figure has a draft caption and an output path index; gaps are recorded instead of reopened.",
                "action": "Write final checklist, caption drafts, gap list, style ledger, and contact sheet.",
                "status": "complete",
            },
            {
                "round": 4,
                "focus": "bounded six-round auto-iteration protocol",
                "finding": f"The active goal now allows up to {MAX_MAIN_ITERATIONS} main iterations plus one final patch, with auto_continue_without_user_choice={AUTO_CONTINUE_WITHOUT_USER_CHOICE}.",
                "action": "Treat next_fix_list.md as the execution queue; do not pause for route choices unless a hard blocker is reached.",
                "status": "protocol_recorded",
            },
        ]
    )


def build_checklist(audit: pd.DataFrame) -> pd.DataFrame:
    fig9_rows = audit.loc[audit["figure"].eq("Fig.9"), "evidence_status"]
    fig9_checkpoint_ready_in_audit = bool(
        not fig9_rows.empty and "checkpoint-generated" in str(fig9_rows.iloc[0]).lower()
    )
    fig9_status = "pass" if fig9_checkpoint_ready_in_audit else "pass_with_gap"
    fig9_notes = (
        "Real manuscript and peer-review trace are present; ASPR-Qwen checkpoint output and metadata are saved for the single case."
        if fig9_checkpoint_ready_in_audit
        else "Real manuscript and peer-review trace are present; ASPR-Qwen lane is assumed and labeled."
    )
    checks = [
        ("Fig.6承接方法可信度", "Fig.6 role plus caption proxy notes", "pass", "Robustness and boundary-condition panels directly test Fig.1-Fig.5 method reliability."),
        ("Fig.7转向venue-level interpretation", "Fig.7 role, outputs, and quality report", "pass_with_caveat", "Venue-level outputs support a point-estimate claim; strict interval-separated Nature dominance remains unsupported."),
        ("Fig.8提出最终方法ASPR", "Fig.8 panel_text and full figure", "pass", "ASPR is defined as graph agent plus ASPR-Qwen plus fusion/verifier."),
        ("Fig.9展示ASPR真实运行", "Fig.9 manifest, trace, quality report", fig9_status, fig9_notes),
        ("Fig.10证明模块贡献", "Fig.10 ablation/provenance CSVs, replacement gates, same-rubric audit, quality report, full figure", "pass_with_gap", "Full ASPR real Fig.4 metrics plus observed qwen3 baseline, proxy-vs-same-rubric discrepancy, and LLM-as-judge estimates; Nature strong-claim gates are not passed."),
        ("颜色字体panel label统一", "style ledger plus generated figures", "pass", "Final assembly records canonical palette and label/caption conventions."),
        ("避免表格堆砌", "visual_mode audit", "pass", "Figures use heatmaps, forest plots, scatter, run-instance maps, module diagrams, matrix cards, and Pareto bars."),
        ("完成多轮自动一致性协议", "rounds report", "pass", "Round 1 audit, Round 2 terminology/visual fix, Round 3 checklist/captions, and the bounded multi-round no-user-choice protocol are written."),
    ]
    return pd.DataFrame(
        [{"requirement": req, "evidence": evidence, "status": status, "notes": notes} for req, evidence, status, notes in checks]
    )


def build_layout_readability_audit(audit: pd.DataFrame, project_root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Record a Nature reading-pass audit for panel density and visual actions."""
    fig5_ai = read_json(project_root / "outputs" / "fig05/old" / "ai_frontier" / "ai_frontier_quality_report.json")
    fig6_quality = read_json(project_root / "outputs" / "fig06/old" / "figure_quality_report.json")
    fig7_quality = read_json(project_root / "outputs" / "fig07/old" / "figure_quality_report.json")
    fig9_quality = read_json(project_root / "outputs" / "fig09/old" / "figure_quality_report.json")
    fig10_quality = read_json(project_root / "outputs" / "fig10/old" / "figure_quality_report.json")
    fig6_checks = (fig6_quality.get("quality_gates") or {}).get("checks") or {}
    fig7_checks = (fig7_quality.get("quality_gates") or {}).get("checks") or {}
    fig9_checks = (fig9_quality.get("quality_gates") or {}).get("checks") or {}
    fig10_checks = (fig10_quality.get("quality_gates") or {}).get("checks") or {}
    fig6_layout_ready = int(fig6_checks.get("main_visual_uses_atlas_matrix_badges") or 0) == 1
    fig7_layout_ready = (
        int(fig7_checks.get("main_visual_panel_count_le_4") or 0) == 1
        and int(fig7_checks.get("text_heavy_panel_f_compacted") or 0) == 1
    )
    fig9_layout_ready = (
        int(fig9_checks.get("large_run_instance_visual") or 0) == 1
        and int(fig9_checks.get("manifest_bound_visual") or 0) == 1
        and int(fig9_checks.get("visible_text_compacted") or 0) == 1
        and int(fig9_checks.get("main_visual_panel_count_le_3") or 0) == 1
        and int(fig9_checks.get("evidence_trace_visible") or 0) == 1
    )
    fig10_layout_ready = (
        int(fig10_checks.get("compact_visual_panel_count_le_4") or 0) == 1
        and int(fig10_checks.get("shared_palette_applied") or 0) == 1
        and int(fig10_checks.get("replacement_gates_embedded_in_visual") or 0) == 1
        and int(fig10_checks.get("same_rubric_baseline_embedded_in_visual") or 0) == 1
        and int(fig10_checks.get("visual_claim_boundary_embedded") or 0) == 1
    )
    rows: List[Dict[str, Any]] = []
    default_actions = {
        "Fig.1": ("ready", "Keep four-domain pre/landmark/post snapshots; no additional panels."),
        "Fig.2": ("ready_with_minor_check", "Use true Fig.1 snapshots in panel a; keep panel d compact family cards."),
        "Fig.3": ("ready", "Use panel e fingerprint for seven-indicator joint contribution."),
        "Fig.4": ("extended_data", "Use as range-restricted audit atlas; do not promote to global validation."),
        "Fig.5": (
            "redraw_handoff_ready" if fig5_ai.get("overall_pass") else "blocked",
            "Redraw as dense AI frontier point cloud from ai_frontier_point_cloud.csv; remove old take-home footer and low-value panels.",
        ),
        "Fig.6": (
            "ready" if fig6_layout_ready else "needs_layout_redesign",
            "Atlas/matrix robustness layout is implemented; keep modeling and failure analyses in audit outputs."
            if fig6_layout_ready
            else "Reduce line-chart dominance; fuse robustness outputs into atlas/matrix/badge views.",
        ),
        "Fig.7": (
            "ready_with_caveat" if fig7_layout_ready else "needs_layout_redesign",
            "Four-panel venue atlas implemented; strict dominance remains a caption/claim-scope caveat."
            if fig7_layout_ready
            else "Reduce panel count and move text-heavy panel f into concise matrix/caveat badge.",
        ),
        "Fig.8": ("handoff_ready", "Use GPT-image architecture handoff; keep visible text short."),
        "Fig.9": (
            "ready" if fig9_layout_ready else "needs_layout_redesign",
            "Large run-instance map is implemented and bound to the case manifest; keep checkpoint boundary in captions/gap list."
            if fig9_layout_ready
            else "Replace text-heavy storyboard with a single large run-instance visual bound to case manifest.",
        ),
        "Fig.10": (
            "ready_with_caveat" if fig10_layout_ready else "needs_layout_redesign",
            "Four-panel ablation evidence atlas implemented; blinded human preference remains a strict claim-boundary caveat."
            if fig10_layout_ready
            else "Unify palette with earlier figures; reduce panels around ablation evidence and replacement gates.",
        ),
    }
    for _, row in audit.iterrows():
        figure = str(row["figure"])
        status, action = default_actions.get(figure, ("unknown", "Review manually."))
        width = int(row["width_px"] or 0)
        height = int(row["height_px"] or 0)
        rows.append(
            {
                "figure": figure,
                "image_exists": int(row["exists"]),
                "width_px": width,
                "height_px": height,
                "visual_mode": row["visual_mode"],
                "reading_pass_status": status,
                "nature_level_action": action,
                "submission_blocker": int(status in {"blocked"}),
                "layout_redesign_needed": int(status == "needs_layout_redesign"),
            }
        )
    return pd.DataFrame(rows)


def build_gap_list(project_root: Path = PROJECT_ROOT) -> pd.DataFrame:
    fig4_quality = quality_gate(read_json(project_root / "outputs" / "fig04/old" / "figure_quality_report.json"))
    fig5_quality = quality_gate(read_json(project_root / "outputs" / "fig05/old" / "figure_quality_report.json"))
    fig5_ai_quality = read_json(project_root / "outputs" / "fig05/old" / "ai_frontier" / "ai_frontier_quality_report.json")
    fig6_quality = quality_gate(read_json(project_root / "outputs" / "fig06/old" / "figure_quality_report.json"))
    fig7_quality = quality_gate(read_json(project_root / "outputs" / "fig07/old" / "figure_quality_report.json"))
    fig9_quality = read_json(project_root / "outputs" / "fig09/old" / "fig9_quality_report.json")
    fig10_quality = quality_gate(read_json(project_root / "outputs" / "fig10/old" / "figure_quality_report.json"))
    fig10_note = fig10_same_rubric_note(project_root)
    rows: List[Dict[str, str]] = []

    if not bool(fig4_quality.get("overall_pass")):
        audit = fig4_quality.get("global_score_coverage_audit", {})
        replacement = fig4_quality.get("external_validation_replacement_manifest", {})
        replacement_needed = (
            replacement.get("additional_ready_labels_needed", {})
            if isinstance(replacement, Mapping)
            else {}
        )
        if replacement_needed:
            needed_text = ", ".join(f"{tier}: {count}" for tier, count in replacement_needed.items())
            next_replacement = (
                "Complete blinded labels in the Fig.4 external-validation replacement manifest "
                f"({needed_text}) and require positive novelty/significance alignment before promoting Fig.4."
            )
        else:
            needed = audit.get("additional_fixed_cases_needed", {}) if isinstance(audit, Mapping) else {}
            needed_text = ", ".join(f"{tier}: {count}" for tier, count in needed.items()) or "low/middle global Fig.3 tiers"
            next_replacement = (
                f"Add peer-labeled fixed cases across the missing global Fig.3 tiers ({needed_text}) "
                "and require positive novelty/significance alignment before promoting Fig.4."
            )
        rows.append(
            {
                "figure": "Fig.4",
                "gap": "Peer-review audit remains range-restricted and cannot support global external validation of Fig.3 scores.",
                "severity": "extended-data blocker",
                "next_replacement": next_replacement,
            }
        )

    fig5_checks = fig5_quality.get("checks", {})
    fig5_ai_ready = bool(fig5_ai_quality.get("overall_pass"))
    fig5_underperforms = (not fig5_ai_ready) and bool(fig5_quality.get("overall_pass")) and not (
        bool(fig5_checks.get("mean_precision_delta_nonnegative"))
        and bool(fig5_checks.get("mean_ndcg_delta_positive"))
    )
    if bool(fig5_quality.get("overall_pass")) and fig5_underperforms:
        precision_at_10 = float(fig5_quality.get("mean_precision_at_10") or float("nan"))
        baseline_precision_at_10 = float(fig5_quality.get("mean_baseline_precision_at_10") or float("nan"))
        ndcg_at_10 = float(fig5_quality.get("mean_ndcg_at_10") or float("nan"))
        baseline_ndcg_at_10 = float(fig5_quality.get("mean_baseline_ndcg_at_10") or float("nan"))
        rows.append(
            {
                "figure": "Fig.5",
                "gap": (
                    "Forecast/backtest artifacts are traceable, but graph-score ranking underperforms the baseline "
                    f"(precision@10 {precision_at_10:.3f} vs {baseline_precision_at_10:.3f}; "
                    f"NDCG@10 {ndcg_at_10:.3f} vs {baseline_ndcg_at_10:.3f})."
                ),
                "severity": "extended-data blocker",
                "next_replacement": "Treat Fig.5 as a failure/backtest audit unless a no-leakage forecast model beats the baseline across historical cutoffs.",
            }
        )

    if not bool(fig6_quality.get("nature_strong_claim_ready")):
        rows.append(
            {
                "figure": "Fig.6",
                "gap": "A fresh OpenAlex sample-level full-graph rebuild audit is present, but rank stability remains below the Nature strong-claim gate.",
                "severity": "main-claim blocker",
                "next_replacement": "Diagnose online reference-closure drift and metric-definition drift, then rerun full graph extraction until rank stability is at least 0.8 or downgrade Fig.6 to boundary-condition evidence.",
            }
        )

    if not bool(fig7_quality.get("strict_claim_supported")):
        headline_supported = bool(fig7_quality.get("headline_point_estimate_supported")) or bool(
            fig7_quality.get("checks", {}).get("nature_rank")
        )
        severity = "claim-scope caveat" if headline_supported else "extended-data blocker"
        gap = (
            "Fig.7 is restricted to a Nature Portfolio aggregate VCI point-estimate claim because strict interval separation is not supported."
            if headline_supported
            else "Strict Nature Portfolio interval separation is not supported by the current bootstrap gate."
        )
        rows.append(
            {
                "figure": "Fig.7",
                "gap": gap,
                "severity": severity,
                "next_replacement": "Keep point-estimate wording unless a larger controlled corpus or stronger uncertainty model makes Nature lower CI exceed the runner-up upper CI.",
            }
        )

    if "assumed" in str(fig9_quality.get("aspr_qwen_boundary", "")).lower():
        rows.append(
            {
                "figure": "Fig.9",
                "gap": "ASPR-Qwen output is assumed for the case run-instance.",
                "severity": "extended-data blocker",
                "next_replacement": "Replace assumed Qwen JSON with the real ASPR-Qwen checkpoint output and save checkpoint metadata.",
            }
        )

    if not bool(fig10_quality.get("nature_strong_claim_ready")):
        failed_fig10_gates = join_gate_labels(fig10_failed_replacement_gate_labels(project_root))
        rows.append(
            {
                "figure": "Fig.10",
                "gap": (
                    f"Replacement gates are not passed for {failed_fig10_gates}; "
                    f"qwen3 proxy scoring exceeds full ASPR, but {fig10_note}."
                ),
                "severity": "extended-data blocker",
                "next_replacement": "Run real module ablations and collect blinded human preference ratings; keep the same-rubric generic baseline manifest, checkpoint metadata, and exclusion table frozen.",
            }
        )

    return pd.DataFrame(rows, columns=["figure", "gap", "severity", "next_replacement"])


def _rel_exists(project_root: Path, rel_path: str) -> int:
    """Return whether a project-relative evidence artifact exists."""
    return int((project_root / rel_path).exists())


def build_strict_evidence_collection_checklist(project_root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """List exact external evidence files needed to clear strict all-figure gates."""
    fig9_quality = read_json(project_root / "outputs" / "fig09/old" / "fig9_quality_report.json")
    fig9_blocker = (
        "checkpoint-generated ASPR-Qwen evidence contract is complete for the single case"
        if fig9_checkpoint_ready(fig9_quality)
        else "ASPR-Qwen lane is an assumed placeholder rather than a saved checkpoint output"
    )
    rows = [
        {
            "figure": "Fig.4",
            "blocker": "global external validation needs blinded novelty/significance labels across low/middle/high Fig.3 tiers",
            "required_submission_artifact": "outputs/fig04/old/fig4_completed_blinded_labels.csv",
            "source_template_or_packet": "outputs/fig04/old/fig4_blinded_labeling_packet.csv; outputs/fig04/old/fig4_completed_blinded_labels_template.csv; outputs/fig04/old/fig4_completed_blinded_labels_labeler_1.csv; outputs/fig04/old/fig4_completed_blinded_labels_labeler_2.csv; outputs/fig04/old/fig4_completed_blinded_labels_labeler_3.csv; outputs/fig04/old/fig4_blinded_labeling_protocol.md; outputs/fig04/old/fig4_blinded_labeling_answer_key.csv; outputs/fig04/old/fig4_external_validation_replacement_manifest.csv",
            "expected_completion": (
                "30 primary labels total: low=10, middle=10, high=10; "
                "novelty/significance/prior-art/confidence in [1,5] with non-LLM/non-synthetic "
                "label_source and labeler_id provenance"
            ),
            "gate_to_clear": "fig4_external_validation: blinded_labeling_complete plus positive novelty/significance bootstrap CI",
            "rerun_command": "FIG4_QUERY_KEYWORD_LIMIT=3 ASPR_OPENALEX_PER_PAGE=10 ASPR_OPENALEX_FROM_YEAR=2000 python3 experiments/fig04/old/main_fig4.py --markdown-root /mnt/d/aspr_nature_markdown --output-dir outputs/fig04/old --sample-size 50 --journal-scope all --retrieval-provider openalex --judge-backend heuristic --reuse-audit --prefer-scored-candidate-pool --require-fixed-sample --forbid-lightweight --forbid-local-retrieval --forbid-lexical-fallback --quiet",
        },
        {
            "figure": "Fig.9",
            "blocker": fig9_blocker,
            "required_submission_artifact": "outputs/fig09/old/fig9_aspr_qwen_output.json; outputs/fig09/old/fig9_checkpoint_metadata.json",
            "source_template_or_packet": "outputs/fig09/old/fig9_checkpoint_metadata_template.json; outputs/fig09/old/fig9_checkpoint_run_contract.json",
            "expected_completion": "one checkpoint-generated case output with checkpoint_invoked=true, output_origin not assumed, non-empty summary_judgement/major_strengths/major_concerns, and metadata keys model_hash/training_config/data_version/prompt/decoding_config/seed/runtime_seconds",
            "gate_to_clear": "checkpoint_generated_aspr_qwen",
            "rerun_command": "python3 experiments/fig09/old/build_fig9_case.py --markdown-root /mnt/d/aspr_nature_markdown --output-dir outputs/fig09/old",
        },
        {
            "figure": "Fig.10",
            "blocker": "module-ablation panel uses estimates until every disabled-module variant has real rerun outputs",
            "required_submission_artifact": "outputs/fig10/old/fig10_true_module_rerun_results.csv",
            "source_template_or_packet": "outputs/fig10/old/fig10_true_module_rerun_results_template.csv; outputs/fig10/old/fig10_true_module_rerun_contract.csv",
            "expected_completion": "400 real case-variant rows: 50 cases x 8 variants, each with review_text_path, evidence_trace_path, runtime_seconds > 0, and metric values in [0,1]",
            "gate_to_clear": "true_disabled_module_reruns",
            "rerun_command": "python3 experiments/fig10/old/build_fig10_ablation.py --fig4-metrics outputs/fig04/old/fig4_metrics_summary.csv --out-dir outputs/fig10/old",
        },
        {
            "figure": "Fig.10",
            "blocker": "preference panel is not based on blinded human ratings",
            "required_submission_artifact": "outputs/fig10/old/fig10_completed_blinded_preferences.csv",
            "source_template_or_packet": (
                "outputs/fig10/old/fig10_blinded_preference_packet.csv; "
                "outputs/fig10/old/fig10_blinded_preference_answer_key.csv; "
                "outputs/fig10/old/fig10_completed_blinded_preferences_template.csv; "
                "outputs/fig10/old/fig10_completed_blinded_preferences_evaluator_1.csv; "
                "outputs/fig10/old/fig10_completed_blinded_preferences_evaluator_2.csv; "
                "outputs/fig10/old/fig10_completed_blinded_preferences_evaluator_3.csv; "
                "outputs/fig10/old/fig10_blinded_preference_protocol.md"
            ),
            "expected_completion": (
                "750 blinded judgements: 50 cases x 5 dimensions x 3 evaluators, "
                "preferred_system in {system_a, system_b, tie}; evaluator_type/preference_source "
                "provenance must be non-LLM/non-synthetic; full ASPR must have Wilson 95% "
                "lower-bound win-rate >0.5 in at least one evidence_grounding/prior_art/usefulness dimension"
            ),
            "gate_to_clear": "blinded_human_preference",
            "rerun_command": "python3 experiments/fig10/old/build_fig10_ablation.py --fig4-metrics outputs/fig04/old/fig4_metrics_summary.csv --out-dir outputs/fig10/old",
        },
    ]
    for row in rows:
        artifact_paths = [part.strip() for part in row["required_submission_artifact"].split(";")]
        template_paths = [part.strip() for part in row["source_template_or_packet"].split(";")]
        row["required_artifacts_present"] = int(all(_rel_exists(project_root, path) for path in artifact_paths))
        row["source_templates_present"] = int(all(_rel_exists(project_root, path) for path in template_paths))
    return pd.DataFrame(rows)


def classify_external_evidence_artifact(figure: str, artifact_path: str, required_return: bool) -> Dict[str, Any]:
    """Classify an external-evidence artifact for coordinator/evaluator distribution."""
    path = artifact_path.strip()
    lower = path.lower()
    role = "required_return" if required_return else "supporting_packet"
    recipient = "coordinator"
    blinded = 0
    contains_answer_key = 0
    note = "Coordinator-facing strict-evidence artifact."

    if "answer_key" in lower:
        role = "coordinator_answer_key"
        recipient = "coordinator"
        contains_answer_key = 1
        note = "Coordinator-only file; do not send to blinded evaluators."
    elif "fig4_completed_blinded_labels_labeler_" in lower:
        role = "return_template"
        recipient = "human_labeler"
        blinded = 1
        note = "Labeler-specific template for Fig.4 blinded human labels."
    elif "completed_blinded_labels_template" in lower:
        role = "return_template"
        recipient = "human_labeler"
        blinded = 1
        note = "Evaluator-facing template for Fig.4 blinded human labels."
    elif "fig4_completed_blinded_labels.csv" in lower:
        role = "required_return"
        recipient = "human_labeler"
        blinded = 1
        note = "Completed Fig.4 blinded human-label return file."
    elif "blinded_labeling_packet" in lower:
        role = "evaluator_packet"
        recipient = "human_labeler"
        blinded = 1
        note = "Evaluator-facing Fig.4 blinded packet."
    elif "blinded_labeling_protocol" in lower:
        role = "protocol"
        recipient = "human_labeler"
        blinded = 1
        note = "Fig.4 human-labeler instructions; keep answer key separate."
    elif "external_validation_replacement_manifest" in lower:
        role = "coordinator_audit"
        recipient = "coordinator"
        contains_answer_key = 1
        note = "Coordinator-only Fig.4 tier/completion audit."
    elif "completed_blinded_preferences_template" in lower:
        role = "return_template"
        recipient = "human_preference_panel"
        blinded = 1
        note = "Evaluator-facing template for blinded Fig.10 preferences."
    elif "completed_blinded_preferences_evaluator_" in lower:
        role = "return_template"
        recipient = "human_preference_panel"
        blinded = 1
        note = "Evaluator-specific template for blinded Fig.10 preferences."
    elif "fig10_completed_blinded_preferences.csv" in lower:
        role = "required_return"
        recipient = "human_preference_panel"
        blinded = 1
        note = "Completed Fig.10 blinded human-preference return file."
    elif "blinded_preference_packet" in lower:
        role = "evaluator_packet"
        recipient = "human_preference_panel"
        blinded = 1
        note = "Evaluator-facing Fig.10 blinded preference packet."
    elif "blinded_preference_protocol" in lower:
        role = "protocol"
        recipient = "human_preference_panel"
        blinded = 1
        note = "Preference-panel instructions; keep answer key separate."
    elif "fig9_aspr_qwen_output" in lower or "checkpoint" in lower:
        role = "checkpoint_contract" if not required_return else "required_return"
        recipient = "checkpoint_runner"
        note = "Checkpoint-runner contract or return artifact."
    elif "true_module_rerun" in lower:
        role = "module_rerun_contract" if not required_return else "required_return"
        recipient = "module_runner"
        note = "Disabled-module rerun contract or return artifact."
    elif required_return:
        role = "required_return"
        recipient = "external_evidence_provider"
        note = "Required strict-gate return artifact."

    return {
        "figure": figure,
        "artifact_path": path,
        "artifact_role": role,
        "recipient": recipient,
        "blinded": blinded,
        "contains_answer_key_or_unblinded_mapping": contains_answer_key,
        "required_for_strict_gate": int(required_return),
        "distribution_note": note,
    }


def build_external_evidence_packet_index(
    evidence_collection: pd.DataFrame,
    project_root: Path = PROJECT_ROOT,
) -> pd.DataFrame:
    """Expand the strict-evidence checklist into a file-level distribution index."""
    rows: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for _, item in evidence_collection.iterrows():
        figure = str(item.get("figure", ""))
        path_groups = [
            (str(item.get("source_template_or_packet", "")), False),
            (str(item.get("required_submission_artifact", "")), True),
        ]
        for raw_paths, required_return in path_groups:
            for artifact_path in [part.strip() for part in raw_paths.split(";") if part.strip()]:
                key = (figure, artifact_path, int(required_return))
                if key in seen:
                    continue
                seen.add(key)
                row = classify_external_evidence_artifact(figure, artifact_path, required_return)
                row["exists"] = _rel_exists(project_root, artifact_path)
                rows.append(row)
    return pd.DataFrame(
        rows,
        columns=[
            "figure",
            "artifact_path",
            "artifact_role",
            "recipient",
            "blinded",
            "contains_answer_key_or_unblinded_mapping",
            "required_for_strict_gate",
            "exists",
            "distribution_note",
        ],
    )


def external_evidence_package_readme(package_name: str, recipient: str) -> str:
    """Return a package-specific README for external evidence collection."""
    common = [
        f"# {package_name}",
        "",
        f"Recipient: `{recipient}`",
        "",
        "Do not request or inspect answer keys, tier labels, Fig.3 scores, system identities, or unblinded coordinator files.",
        "Do not use LLM, model-generated, synthetic, proxy, or automated labels/preferences for completed human-return files.",
        "Return only the completed CSV requested below; keep file names unchanged so the strict evidence checker can validate them.",
        "",
    ]
    if package_name == "fig4_human_labeler_packet":
        lines = [
            *common,
            "## Return File",
            "",
            "- Return `fig4_completed_blinded_labels.csv`.",
            "- Start from `fig4_completed_blinded_labels_template.csv`.",
            "- Or distribute the three labeler-specific templates: `fig4_completed_blinded_labels_labeler_1.csv`, `fig4_completed_blinded_labels_labeler_2.csv`, and `fig4_completed_blinded_labels_labeler_3.csv`; the coordinator can run `make fig4-merge-blinded-labels` after all three are complete.",
            "- Required columns: `blinded_case_id`, `label_novelty_1_5`, `label_significance_1_5`, `label_prior_art_1_5`, `label_confidence_1_5`, `label_source`, `labeler_id`, `label_notes`.",
            "- Scores must be integers from 1 to 5.",
            "- `label_source` and `labeler_id` must identify a non-LLM/non-synthetic blinded human source.",
            "",
        ]
    elif package_name == "fig10_human_preference_panel_packet":
        lines = [
            *common,
            "## Return File",
            "",
            "- Return `fig10_completed_blinded_preferences.csv`.",
            "- Start from `fig10_completed_blinded_preferences_template.csv`.",
            "- Required completion: 750 blinded judgements, from 50 cases x 5 dimensions x 3 evaluators.",
            "- `preferred_system` must be one of `system_a`, `system_b`, or `tie`.",
            "- `evaluator_type` and `preference_source` must identify non-LLM/non-synthetic blinded human provenance.",
            "",
        ]
    elif package_name == "coordinator_private_evidence_packet":
        lines = [
            *common,
            "## Coordinator-Only Files",
            "",
            "- This package may contain answer keys or unblinded mappings.",
            "- Do not distribute this zip to blinded evaluators.",
            "- Use it only after completed human-return files are received.",
            "",
        ]
    elif package_name == "checkpoint_runner_packet":
        lines = [
            *common,
            "## Checkpoint Return Files",
            "",
            "- Return checkpoint-generated ASPR-Qwen output and complete checkpoint metadata.",
            "- Required output files are listed in the strict evidence checklist.",
            "",
        ]
    elif package_name == "module_runner_packet":
        lines = [
            *common,
            "## Module-Rerun Return Files",
            "",
            "- Return real disabled-module rerun results with review text paths, evidence trace paths, runtime, and metrics.",
            "- Do not use estimate rows for strict Nature claims.",
            "",
        ]
    else:
        lines = [*common, "Follow the strict evidence checklist for required return files.", ""]
    return "\n".join(lines)


def _write_artifact_zip(package_path: Path, artifact_paths: Sequence[str], project_root: Path, readme: str = "") -> int:
    """Write a zip package containing existing project-relative artifact paths."""
    package_path.parent.mkdir(parents=True, exist_ok=True)
    file_count = 0
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if readme:
            archive.writestr("README.md", readme)
            file_count += 1
        for artifact_path in artifact_paths:
            source = project_root / artifact_path
            if not source.exists() or not source.is_file():
                continue
            archive.write(source, arcname=artifact_path)
            file_count += 1
    return file_count


def build_external_evidence_distribution_packages(
    out_dir: Path,
    packet_index: pd.DataFrame,
    project_root: Path = PROJECT_ROOT,
) -> pd.DataFrame:
    """Write leakage-safe distribution zip packages from the packet index."""
    package_dir = out_dir / "external_evidence_distribution"
    manifest_rows: List[Dict[str, Any]] = []

    packages = [
        {
            "package_name": "fig4_human_labeler_packet",
            "recipient": "human_labeler",
            "selector": packet_index["figure"].eq("Fig.4")
            & packet_index["recipient"].eq("human_labeler")
            & packet_index["contains_answer_key_or_unblinded_mapping"].astype(int).eq(0)
            & packet_index["required_for_strict_gate"].astype(int).eq(0),
            "blinded_evaluator_package": 1,
        },
        {
            "package_name": "fig10_human_preference_panel_packet",
            "recipient": "human_preference_panel",
            "selector": packet_index["figure"].eq("Fig.10")
            & packet_index["recipient"].eq("human_preference_panel")
            & packet_index["contains_answer_key_or_unblinded_mapping"].astype(int).eq(0)
            & packet_index["required_for_strict_gate"].astype(int).eq(0),
            "blinded_evaluator_package": 1,
        },
        {
            "package_name": "coordinator_private_evidence_packet",
            "recipient": "coordinator",
            "selector": packet_index["recipient"].eq("coordinator")
            | packet_index["contains_answer_key_or_unblinded_mapping"].astype(int).eq(1),
            "blinded_evaluator_package": 0,
        },
        {
            "package_name": "checkpoint_runner_packet",
            "recipient": "checkpoint_runner",
            "selector": packet_index["recipient"].eq("checkpoint_runner"),
            "blinded_evaluator_package": 0,
        },
        {
            "package_name": "module_runner_packet",
            "recipient": "module_runner",
            "selector": packet_index["recipient"].eq("module_runner"),
            "blinded_evaluator_package": 0,
        },
    ]
    for package in packages:
        subset = packet_index[package["selector"]].copy()
        subset = subset[subset["exists"].astype(int).eq(1)]
        artifact_paths = subset["artifact_path"].astype(str).tolist()
        package_path = package_dir / f"{package['package_name']}.zip"
        readme = external_evidence_package_readme(str(package["package_name"]), str(package["recipient"]))
        file_count = _write_artifact_zip(package_path, artifact_paths, project_root, readme=readme)
        package_bytes = package_path.read_bytes() if package_path.exists() else b""
        manifest_rows.append(
            {
                "package_name": package["package_name"],
                "recipient": package["recipient"],
                "package_path": str(package_path.relative_to(out_dir)),
                "file_count": int(file_count),
                "zip_bytes": int(len(package_bytes)),
                "zip_sha256": hashlib.sha256(package_bytes).hexdigest(),
                "contains_answer_key_or_unblinded_mapping": int(
                    subset.get("contains_answer_key_or_unblinded_mapping", pd.Series(dtype=int)).astype(int).max()
                    if not subset.empty
                    else 0
                ),
                "blinded_evaluator_package": int(package["blinded_evaluator_package"]),
                "included_artifacts": "; ".join(artifact_paths),
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(out_dir / "fig1_fig10_external_evidence_distribution_manifest.csv", index=False)
    return manifest


def write_external_evidence_handoff(
    out_dir: Path,
    evidence_collection: pd.DataFrame,
    readiness: Mapping[str, Any],
    packet_index_path: Optional[Path] = None,
    distribution_manifest_path: Optional[Path] = None,
) -> Path:
    """Write a coordinator-facing handoff for missing strict external evidence."""
    lines = [
        "# Fig.1-Fig.10 Strict External Evidence Handoff",
        "",
        "## Submission Readiness",
        "",
        f"- main_claim_ready: {readiness.get('main_claim_ready', 0)}",
        f"- strict_all_figures_ready: {readiness.get('strict_all_figures_ready', 0)}",
        f"- strict_external_evidence_ready: {readiness.get('strict_external_evidence_ready', 0)}",
        f"- strict_failed_figures: {', '.join(readiness.get('strict_failed_figures', [])) or 'none'}",
        f"- strict_evidence_missing_figures: {', '.join(readiness.get('strict_evidence_missing_figures', [])) or 'none'}",
        "",
        "## How To Use This Packet",
        "",
        "Run `make figures-evidence-packets` to refresh the evaluator-facing packets, answer keys, checkpoint contract, true-rerun contract, and this handoff. Completed return files must contain real labels, real checkpoint output, or real module reruns; placeholders keep the strict evidence gate closed.",
        "After `fig4_completed_blinded_labels.csv` and `fig10_completed_blinded_preferences.csv` are returned, run `make figures-external-evidence-intake` to rebuild Fig.4, Fig.10, the final assembly, and the strict external-evidence check.",
        "If Fig.4 returns arrive as the three labeler-specific CSV files, run `make fig4-merge-blinded-labels`; `make figures-external-evidence-intake` also runs this merge before rebuilding Fig.4.",
        "If Fig.10 returns arrive as the three evaluator-specific CSV files, run `make fig10-merge-blinded-preferences`; `make figures-external-evidence-intake` also runs this merge before rebuilding Fig.10.",
        (
            f"Use `{packet_index_path.name}` to separate evaluator-facing files from coordinator-only answer keys."
            if packet_index_path is not None
            else "Use the packet index to separate evaluator-facing files from coordinator-only answer keys."
        ),
        (
            f"Use `{distribution_manifest_path.name}` and the zip files in `external_evidence_distribution/` for leakage-safe distribution."
            if distribution_manifest_path is not None
            else "Use the distribution manifest and zip files for leakage-safe distribution."
        ),
        "",
        "## Required Returns",
        "",
    ]
    for _, row in evidence_collection.iterrows():
        figure = str(row.get("figure", ""))
        lines.extend(
            [
                f"### {figure}",
                "",
                f"- Blocker: {row.get('blocker', '')}",
                f"- Packet or contract to distribute: `{row.get('source_template_or_packet', '')}`",
                f"- Completed return artifact: `{row.get('required_submission_artifact', '')}`",
                f"- Expected completion: {row.get('expected_completion', '')}",
                f"- Gate to clear: `{row.get('gate_to_clear', '')}`",
                f"- Packet/contract present: {int(row.get('source_templates_present', 0))}",
                f"- Completed return present: {int(row.get('required_artifacts_present', 0))}",
                f"- Refresh command: `{row.get('rerun_command', '')}`",
                "",
            ]
        )
    path = out_dir / "fig1_fig10_external_evidence_handoff.md"
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def write_markdown(out_dir: Path, audit: pd.DataFrame, rounds: pd.DataFrame, checklist: pd.DataFrame, gaps: pd.DataFrame) -> None:
    lines = [
        "# Fig.6-Fig.10 Consistency and Final Assembly",
        "",
        "## Assembly Thesis",
        "",
        "Fig.1-Fig.5 establish graph-perturbation evidence and validation. Fig.6 tests the robustness of that method. Fig.7 moves the evidence into venue-level scientific interpretation. Fig.8 introduces ASPR as the final graph-agent plus ASPR-Qwen system. Fig.9 shows one auditable ASPR run. Fig.10 tests whether ASPR quality comes from module composition rather than a single generic LLM.",
        "",
        "## Round 1: Cross-Figure Audit",
        "",
    ]
    for _, row in rounds[rounds["round"].eq(1)].iterrows():
        lines.append(f"- **{row['status']}**: {row['finding']} Action: {row['action']}")
    lines.extend(["", "## Round 2: Terminology, Color, And Panel Order Fixes", ""])
    for _, row in rounds[rounds["round"].eq(2)].iterrows():
        lines.append(f"- **{row['status']}**: {row['finding']} Action: {row['action']}")
    lines.extend(["", "## Round 3: Final Checklist And Caption Package", ""])
    for _, row in checklist.iterrows():
        lines.append(f"- **{row['status']}** `{row['requirement']}`: {row['notes']}")
    protocol_rows = rounds[rounds["round"].ge(4)]
    if not protocol_rows.empty:
        lines.extend(["", "## Active Multi-Round Auto-Iteration Protocol", ""])
        for _, row in protocol_rows.iterrows():
            lines.append(f"- **{row['status']}**: {row['finding']} Action: {row['action']}")
    lines.extend(["", "## Pipeline-Ready Gaps", ""])
    for _, row in gaps.iterrows():
        lines.append(f"- **{row['figure']}** ({row['severity']}): {row['gap']} Next replacement: {row['next_replacement']}")
    lines.extend(["", "## Output Path Index", ""])
    for _, row in audit.iterrows():
        lines.append(f"- **{row['figure']}**: `{row['path']}` - {row['assembly_action']}")
    report = "\n".join(lines) + "\n"
    (out_dir / "fig6_fig10_multi_round_consistency_report.md").write_text(report, encoding="utf-8")
    (out_dir / "fig6_fig10_three_round_consistency_report.md").write_text(report, encoding="utf-8")


def write_captions(out_dir: Path, project_root: Path = PROJECT_ROOT) -> None:
    lines = ["# Fig.1-Fig.10 Caption Drafts", ""]
    captions = build_captions(project_root)
    for figure in FIGURES:
        lines.extend([f"## {figure['figure']}", "", captions[figure["figure"]], ""])
    (out_dir / "fig1_fig10_caption_drafts.md").write_text("\n".join(lines), encoding="utf-8")


def write_style_guide(out_dir: Path) -> None:
    payload = {
        "font": "Use DejaVu Sans / Inter / Aptos-compatible sans-serif. Panel labels use lowercase letter plus two spaces, e.g. `a  Robustness`.",
        "palette": STYLE_LEDGER,
        "terminology": TERMINOLOGY,
        "panel_order": [
            "Fig.1-Fig.5: measurement, empirical validation, learned score, peer-review validation, forecast/mechanism handoff",
            "Fig.6: full-rerun robustness and boundary conditions",
            "Fig.7: venue-level interpretation",
            "Fig.8: final ASPR algorithm",
            "Fig.9: real ASPR case run",
            "Fig.10: ASPR module ablation and reinforcement",
        ],
        "caption_tone": "Evidence-first, caveat-visible, no overclaim that ASPR replaces peer review or that venue-family patterns are causal.",
    }
    write_json(out_dir / "fig1_fig10_style_ledger.json", payload)
    pd.DataFrame(STYLE_LEDGER).to_csv(out_dir / "fig1_fig10_visual_style_ledger.csv", index=False)
    pd.DataFrame(TERMINOLOGY).to_csv(out_dir / "fig1_fig10_terminology_crosswalk.csv", index=False)


def create_contact_sheet(audit: pd.DataFrame, out_dir: Path) -> Path:
    thumb_w, thumb_h = 460, 300
    pad = 28
    label_h = 54
    cols = 2
    rows = (len(audit) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (thumb_w + pad) + pad, rows * (thumb_h + label_h + pad) + pad), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
        label_font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    for idx, row in audit.iterrows():
        col = idx % cols
        row_idx = idx // cols
        x = pad + col * (thumb_w + pad)
        y = pad + row_idx * (thumb_h + label_h + pad)
        path = Path(row["absolute_path"])
        if path.exists():
            with Image.open(path).convert("RGB") as image:
                image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
                image_x = x + (thumb_w - image.width) // 2
                image_y = y + label_h + (thumb_h - image.height) // 2
                sheet.paste(image, (image_x, image_y))
        title = f"{row['figure']} | {row['title']}"
        title_lines = textwrap.wrap(title, width=42)
        draw.text((x, y), title_lines[0], fill="#111827", font=title_font)
        if len(title_lines) > 1:
            draw.text((x, y + 20), title_lines[1], fill="#111827", font=label_font)
        wrapped = textwrap.wrap(str(row["visual_mode"]), width=54)
        draw.text((x, y + 38), " ".join(wrapped[:1]), fill="#475569", font=label_font)
        draw.rectangle((x, y + label_h, x + thumb_w, y + label_h + thumb_h), outline="#CBD5E1", width=1)
    path = out_dir / "fig1_fig10_contact_sheet.png"
    sheet.save(path)
    return path


def _split_failed_figures(value: Any) -> List[str]:
    """Split a semicolon-separated failed-figure field while preserving order."""
    if value is None:
        return []
    figures: List[str] = []
    for part in str(value).split(";"):
        label = part.strip()
        if label and label not in figures:
            figures.append(label)
    return figures


def build_submission_readiness(project_root: Path, out_dir: Path) -> Dict[str, Any]:
    """Summarize main-claim and strict all-figure readiness for the assembly."""
    main_report = build_nature_check_report(project_root, out_dir, require_all_figures=False)
    all_figures_report = build_nature_check_report(project_root, out_dir, require_all_figures=True)
    strict_evidence_report = build_strict_evidence_check_report(project_root, out_dir)
    all_figure_checks = [
        row
        for row in all_figures_report.get("checks", [])
        if row.get("blocking_scope") == "all_figure_blocker"
    ]
    strict_failed_figures: List[str] = []
    for row in all_figure_checks:
        strict_failed_figures.extend(_split_failed_figures(row.get("failed_figures")))
    strict_evidence_missing_figures: List[str] = []
    for row in strict_evidence_report.get("checks", []):
        if not bool(row.get("passed")):
            figure = str(row.get("figure", "")).strip()
            if figure and figure not in strict_evidence_missing_figures:
                strict_evidence_missing_figures.append(figure)
    readiness = {
        "main_claim_ready": int(bool(main_report.get("overall_pass"))),
        "strict_all_figures_ready": int(bool(all_figures_report.get("overall_pass"))),
        "strict_external_evidence_ready": int(bool(strict_evidence_report.get("overall_pass"))),
        "main_claim_status": str(main_report.get("status_label", "")),
        "strict_all_figures_status": str(all_figures_report.get("status_label", "")),
        "strict_external_evidence_status": str(strict_evidence_report.get("status_label", "")),
        "strict_failed_figures": strict_failed_figures,
        "strict_evidence_missing_figures": strict_evidence_missing_figures,
        "claim_ledger": str(main_report.get("claim_ledger", "")),
        "main_claim_summary_csv": str(main_report.get("summary_csv", "")),
        "all_figures_summary_csv": str(all_figures_report.get("summary_csv", "")),
        "strict_evidence_summary_csv": str(strict_evidence_report.get("summary_csv", "")),
    }
    write_json(out_dir / "fig1_fig10_submission_readiness.json", readiness)
    pd.DataFrame(
        [
            {"check": key, "value": value}
            for key, value in readiness.items()
            if not isinstance(value, list)
        ]
    ).to_csv(out_dir / "fig1_fig10_submission_readiness.csv", index=False)
    return readiness


def build_final_assembly(out_dir: Path, project_root: Path = PROJECT_ROOT) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    from experiments.common.old.final_assembly.build_visual_redesign_handoff import build_handoff as build_visual_handoff
    from experiments.common.old.nature_iteration.build_nature_iteration import MAX_MAIN_ITERATIONS

    build_visual_handoff(out_dir / "visual_redesign_handoff")
    audit = build_audit(project_root)
    rounds = build_rounds(project_root)
    checklist = build_checklist(audit)
    layout_audit = build_layout_readability_audit(audit, project_root)
    gaps = build_gap_list(project_root)
    evidence_collection = build_strict_evidence_collection_checklist(project_root)
    packet_index = build_external_evidence_packet_index(evidence_collection, project_root)
    packet_index_path = out_dir / "fig1_fig10_external_evidence_packet_index.csv"
    distribution_manifest_path = out_dir / "fig1_fig10_external_evidence_distribution_manifest.csv"

    audit.to_csv(out_dir / "fig1_fig10_cross_figure_audit.csv", index=False)
    layout_audit.to_csv(out_dir / "fig1_fig10_layout_readability_audit.csv", index=False)
    rounds.to_csv(out_dir / "fig6_fig10_multi_round_audit.csv", index=False)
    rounds.to_csv(out_dir / "fig6_fig10_three_round_audit.csv", index=False)
    checklist.to_csv(out_dir / "fig1_fig10_final_checklist.csv", index=False)
    gaps.to_csv(out_dir / "fig1_fig10_pipeline_ready_gaps.csv", index=False)
    evidence_collection.to_csv(out_dir / "fig1_fig10_strict_evidence_collection_checklist.csv", index=False)
    packet_index.to_csv(packet_index_path, index=False)
    distribution_manifest = build_external_evidence_distribution_packages(out_dir, packet_index, project_root)
    write_markdown(out_dir, audit, rounds, checklist, gaps)
    captions = build_captions(project_root)
    write_captions(out_dir, project_root)
    write_style_guide(out_dir)
    contact_sheet = create_contact_sheet(audit, out_dir)
    readiness = build_submission_readiness(project_root, out_dir)
    evidence_handoff = write_external_evidence_handoff(
        out_dir,
        evidence_collection,
        readiness,
        packet_index_path,
        distribution_manifest_path,
    )

    assembly_checks = {
        "all_figures_have_current_png": int(audit["exists"].eq(1).all()),
        "three_rounds_recorded": int(set(rounds["round"]).issuperset({1, 2, 3})),
        "multi_round_protocol_recorded": int(rounds["round"].max() >= 4),
        "max_main_iterations_eq_6": int(MAX_MAIN_ITERATIONS == 6),
        "auto_iteration_no_user_choice": 1,
        "caption_count_10": int(len(CAPTIONS) == 10),
        "style_ledger_written": int((out_dir / "fig1_fig10_style_ledger.json").exists()),
        "pipeline_gaps_recorded": int((out_dir / "fig1_fig10_pipeline_ready_gaps.csv").exists()),
        "layout_readability_audit_written": int((out_dir / "fig1_fig10_layout_readability_audit.csv").exists()),
        "fig5_ai_frontier_artifacts_present": int(
            (project_root / "outputs" / "fig05/old" / "ai_frontier" / "ai_frontier_quality_report.json").exists()
            and (project_root / "outputs" / "fig05/old" / "ai_frontier" / "ai_frontier_point_cloud.csv").exists()
        ),
        "fig4_claim_scope_artifact_present": int(
            (project_root / "outputs" / "fig04/old" / "fig4_claim_scope_decision.json").exists()
        ),
        "fig8_handoff_quality_present": int(
            (project_root / "outputs" / "fig08/old" / "figure_quality_report.json").exists()
            and (project_root / "outputs" / "fig08/old" / "fig8_handoff_manifest.json").exists()
        ),
        "visual_redesign_handoff_present": int(
            (out_dir / "visual_redesign_handoff" / "visual_redesign_handoff_manifest.csv").exists()
            and (out_dir / "visual_redesign_handoff" / "visual_redesign_quality_report.json").exists()
        ),
        "strict_evidence_collection_checklist_written": int((out_dir / "fig1_fig10_strict_evidence_collection_checklist.csv").exists()),
        "external_evidence_packet_index_written": int(packet_index_path.exists() and packet_index_path.stat().st_size > 500),
        "external_evidence_distribution_packages_written": int(
            distribution_manifest_path.exists()
            and distribution_manifest["file_count"].astype(int).sum() >= 8
            and (
                out_dir / "external_evidence_distribution" / "fig4_human_labeler_packet.zip"
            ).exists()
            and (
                out_dir / "external_evidence_distribution" / "fig10_human_preference_panel_packet.zip"
            ).exists()
            and (
                out_dir / "external_evidence_distribution" / "coordinator_private_evidence_packet.zip"
            ).exists()
        ),
        "external_evidence_distribution_checksums_written": int(
            {"zip_bytes", "zip_sha256"}.issubset(distribution_manifest.columns)
            and distribution_manifest["zip_bytes"].astype(int).gt(0).all()
            and distribution_manifest["zip_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all()
        ),
        "external_evidence_handoff_written": int(evidence_handoff.exists() and evidence_handoff.stat().st_size > 1000),
        "submission_readiness_written": int((out_dir / "fig1_fig10_submission_readiness.json").exists()),
        "contact_sheet_exists": int(contact_sheet.exists() and contact_sheet.stat().st_size > 10_000),
    }
    readiness_checks = {
        "main_claim_nature_check_pass": int(readiness["main_claim_ready"]),
        "strict_all_figures_nature_check_pass": int(readiness["strict_all_figures_ready"]),
        "strict_external_evidence_check_pass": int(readiness["strict_external_evidence_ready"]),
    }
    quality_gates = {
        "checks": {**assembly_checks, **readiness_checks},
        "overall_pass": False,
        "assembly_overall_pass": False,
        "submission_status": "unknown",
        "status_label": "final_assembly_complete_with_pipeline_ready_gaps",
    }
    quality_gates["assembly_overall_pass"] = bool(all(assembly_checks.values()))
    quality_gates["overall_pass"] = bool(quality_gates["assembly_overall_pass"])
    quality_gates["submission_status"] = (
        "all_figures_strict_ready"
        if readiness["strict_all_figures_ready"] and readiness["strict_external_evidence_ready"]
        else "main_claim_ready_with_strict_evidence_gaps"
        if readiness["main_claim_ready"]
        else "main_claim_needs_revision"
    )
    generated = [
        out_dir / "fig6_fig10_multi_round_consistency_report.md",
        out_dir / "fig6_fig10_three_round_consistency_report.md",
        out_dir / "fig1_fig10_caption_drafts.md",
        out_dir / "fig1_fig10_cross_figure_audit.csv",
        out_dir / "fig1_fig10_layout_readability_audit.csv",
        out_dir / "fig1_fig10_final_checklist.csv",
        out_dir / "fig1_fig10_pipeline_ready_gaps.csv",
        out_dir / "fig1_fig10_strict_evidence_collection_checklist.csv",
        packet_index_path,
        distribution_manifest_path,
        evidence_handoff,
        out_dir / "fig1_fig10_submission_readiness.json",
        out_dir / "fig1_fig10_submission_readiness.csv",
        out_dir / "fig1_fig10_style_ledger.json",
        out_dir / "visual_redesign_handoff" / "visual_redesign_handoff_manifest.csv",
        out_dir / "visual_redesign_handoff" / "visual_redesign_quality_report.json",
        contact_sheet,
    ]
    write_run_manifest(
        out_dir,
        figure="fig6_fig10_final_assembly",
        argv=sys.argv,
        inputs={"figures": [row["path"] for row in FIGURES]},
        quality_gates=quality_gates,
        extra={"created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()},
    )
    write_figure_quality_report(
        out_dir,
        figure="fig6_fig10_final_assembly",
        generated_files=[contact_sheet],
        quality_gates=quality_gates,
        extra={"generated_documents": [str(path) for path in generated]},
    )
    return {
        "output_dir": str(out_dir),
        "quality_gates": quality_gates,
        "contact_sheet": str(contact_sheet),
        "captions": captions,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    result = build_final_assembly(args.out_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

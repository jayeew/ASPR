"""Build Nature-readiness claim ledgers and gate summaries for Fig.1-Fig.10."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_CLAIM_FIGURES = {"Fig.1", "Fig.2", "Fig.3", "Fig.6"}
LEAKAGE_COLUMN_PATTERNS = (
    "future",
    "post_cutoff",
    "postcutoff",
    "post_publication",
    "realized",
    "outcome",
    "rgpm",
    "n_future_citers",
    "future_citer",
    "future_graph_delta",
    "cited_by_count",
    "cited_by_count_future",
    "venue_future",
)
ALLOWED_PUBLICATION_DAY_PATTERNS = (
    "future_window",
    "future_tau",
    "future_horizon",
)
NONHUMAN_EXTERNAL_EVIDENCE_PATTERNS = (
    "llm",
    "model",
    "synthetic",
    "heuristic",
    "proxy",
    "automatic",
    "auto",
    "machine",
    "simulated",
    "placeholder",
    "estimated",
    "gpt",
    "qwen",
    "ollama",
)


def read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON object, returning an empty mapping when absent or invalid."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def quality_gate(path: Path) -> Dict[str, Any]:
    """Return the quality-gate mapping from a figure report."""
    payload = read_json(path)
    if not payload:
        return {}
    gates = payload.get("quality_gates")
    if not isinstance(gates, dict):
        return payload
    merged = dict(payload)
    merged.update(gates)
    return merged


def gate_passed(path: Path) -> bool:
    """Return whether a figure quality gate passed."""
    gates = quality_gate(path)
    if "overall_pass" in gates:
        return bool(gates.get("overall_pass"))
    checks = gates.get("checks")
    if isinstance(checks, Mapping):
        return bool(checks) and all(bool(value) for value in checks.values())
    return False


def failed_checks(path: Path) -> str:
    """Return semicolon-separated failed quality checks for a report."""
    checks = quality_gate(path).get("checks", {})
    if not isinstance(checks, Mapping):
        return ""
    failed = [str(key) for key, value in checks.items() if not bool(value)]
    return "; ".join(failed)


def detect_no_leakage_feature_violations(
    frame: pd.DataFrame,
    feature_columns: Optional[Sequence[str]] = None,
) -> List[str]:
    """Return publication-day feature columns that appear to leak future outcomes."""
    columns = list(feature_columns) if feature_columns is not None else list(frame.columns)
    violations: List[str] = []
    for column in columns:
        normalized = str(column).strip().lower()
        if not normalized:
            continue
        if any(pattern in normalized for pattern in ALLOWED_PUBLICATION_DAY_PATTERNS):
            continue
        if any(pattern in normalized for pattern in LEAKAGE_COLUMN_PATTERNS):
            violations.append(str(column))
    return violations


def _nonhuman_or_synthetic_source_rows(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """Return a row mask for external-evidence provenance that cannot support human claims."""
    if frame.empty:
        return pd.Series(dtype=bool)
    present = [column for column in columns if column in frame.columns]
    if not present:
        return pd.Series([True] * len(frame), index=frame.index)
    source_text = frame[present].fillna("").astype(str).agg(" ".join, axis=1).str.strip().str.lower()
    missing_source = source_text.eq("")
    nonhuman = pd.Series([False] * len(frame), index=frame.index)
    for pattern in NONHUMAN_EXTERNAL_EVIDENCE_PATTERNS:
        nonhuman = nonhuman | source_text.str.contains(pattern, regex=False, na=False)
    return missing_source | nonhuman


def ledger_row(
    *,
    figure: str,
    claim_id: str,
    main_text_role: str,
    current_status: str,
    quality_gate_path: Path,
    quality_gate_pass: int,
    allowed_claim: str,
    forbidden_claim: str,
    required_action: str,
    required_gate: str,
    required_artifacts: str,
    main_or_extended_data: str,
) -> Dict[str, Any]:
    """Create one normalized claim-ledger row."""
    return {
        "figure": figure,
        "claim_id": claim_id,
        "main_text_role": main_text_role,
        "required_gate": required_gate,
        "current_status": current_status,
        "quality_gate_path": str(quality_gate_path),
        "quality_gate_pass": int(quality_gate_pass),
        "allowed_claim": allowed_claim,
        "forbidden_claim": forbidden_claim,
        "required_artifacts": required_artifacts,
        "main_or_extended_data": main_or_extended_data,
        "required_action": required_action,
    }


def fig4_peer_review_alignment_status(project_root: Path) -> Dict[str, Any]:
    """Summarize whether current Fig.4 can support peer-review alignment claims."""
    fig4_dir = project_root / "outputs" / "kg_perturbation_fig4_full50"
    report_path = fig4_dir / "figure_quality_report.json"
    report = read_json(report_path)
    if report:
        checks = report.get("checks", {})
        if isinstance(checks, Mapping):
            failed = [str(key) for key, value in checks.items() if not bool(value)]
            claim_ready = bool(report.get("overall_pass")) if "overall_pass" in report else bool(checks) and not failed
            status = str(
                report.get(
                    "status_label",
                    "external_validation_ready" if claim_ready else "external_validation_blocked",
                )
            )
            details = "all figure_quality_report checks pass" if claim_ready else "; ".join(failed)
            return {"status": status, "claim_ready": claim_ready, "details": details}

    path = fig4_dir / "fig4_metrics_summary.csv"
    if not path.exists():
        return {"status": "missing_fig4_metrics", "claim_ready": False, "details": "fig4_metrics_summary.csv missing"}
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        return {"status": "invalid_fig4_metrics", "claim_ready": False, "details": str(exc)}
    metrics: Dict[str, float] = {}
    for column in ["soft_claim_recall", "claim_evidence_coverage", "missing_peer_point_rate", "covered_peer_aspects"]:
        if column in df.columns:
            metrics[column] = float(pd.to_numeric(df[column], errors="coerce").fillna(0.0).mean())
    claim_ready = (
        metrics.get("soft_claim_recall", 0.0) > 0.0
        and metrics.get("claim_evidence_coverage", 0.0) > 0.0
        and metrics.get("covered_peer_aspects", 0.0) > 0.0
        and metrics.get("missing_peer_point_rate", 1.0) < 1.0
    )
    status = "peer_review_alignment_measured" if claim_ready else "peer_review_alignment_failed"
    details = "; ".join(f"{key}={value:.3f}" for key, value in sorted(metrics.items()))
    return {"status": status, "claim_ready": claim_ready, "details": details}


def fig10_replacement_gate_status(project_root: Path) -> Dict[str, Any]:
    """Summarize Fig.10 replacement gates."""
    path = project_root / "outputs" / "kg_perturbation_fig10" / "fig10_replacement_gates.csv"
    if not path.exists():
        return {"status": "missing_replacement_gates", "claim_ready": False, "details": "fig10_replacement_gates.csv missing"}
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        return {"status": "invalid_replacement_gates", "claim_ready": False, "details": str(exc)}
    if "pass_for_nature_strong_claim" not in df.columns:
        return {"status": "invalid_replacement_gates", "claim_ready": False, "details": "missing pass_for_nature_strong_claim"}
    failed = df.loc[~df["pass_for_nature_strong_claim"].astype(bool), "gate_id"].astype(str).tolist()
    claim_ready = not failed
    status = "replacement_gates_passed" if claim_ready else "replacement_gates_block_strong_claim"
    return {"status": status, "claim_ready": claim_ready, "details": "; ".join(failed)}


def build_claim_ledger(project_root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Build a machine-readable ledger of allowed and forbidden Fig.1-Fig.10 claims."""
    reports = {
        "Fig.1": project_root / "outputs" / "redraw_v6a_best_fig1" / "figure_quality_report.json",
        "Fig.2": project_root / "outputs" / "redraw_v6a_best_fig2" / "fig2_quality_gates.json",
        "Fig.3": project_root / "outputs" / "redraw_v6a_best_fig3" / "figure_quality_report.json",
        "Fig.6": project_root / "outputs" / "kg_perturbation_fig6" / "figure_quality_report.json",
        "Fig.7": project_root / "outputs" / "kg_perturbation_fig7" / "figure_quality_report.json",
        "Fig.9": project_root / "outputs" / "kg_perturbation_fig9" / "fig9_quality_report.json",
        "Fig.10": project_root / "outputs" / "kg_perturbation_fig10" / "figure_quality_report.json",
    }
    fig4 = fig4_peer_review_alignment_status(project_root)
    fig10 = fig10_replacement_gate_status(project_root)
    fig9_payload = read_json(reports["Fig.9"])
    fig9_boundary = str(fig9_payload.get("aspr_qwen_boundary", "missing"))
    fig9_checkpoint_ready = "assumed" not in fig9_boundary.lower() and fig9_boundary != "missing"
    fig5_report = project_root / "outputs" / "kg_perturbation_fig5" / "figure_quality_report.json"
    fig5_ai_report = project_root / "outputs" / "kg_perturbation_fig5" / "ai_frontier" / "ai_frontier_quality_report.json"
    fig5_payload = read_json(fig5_report)
    fig5_ai_payload = read_json(fig5_ai_report)
    fig5_gate = quality_gate(fig5_report)
    fig5_checks = fig5_gate.get("checks", {})
    fig5_beats_baseline = bool(
        isinstance(fig5_checks, Mapping)
        and fig5_checks.get("mean_precision_delta_nonnegative")
        and fig5_checks.get("mean_ndcg_delta_positive")
    )
    fig5_ai_ready = bool(fig5_ai_payload.get("overall_pass"))
    fig7_gate = quality_gate(reports["Fig.7"])
    fig7_checks = fig7_gate.get("checks", {})
    fig7_point_estimate_supported = bool(fig7_gate.get("headline_point_estimate_supported")) or bool(
        isinstance(fig7_checks, Mapping) and fig7_checks.get("nature_rank")
    )
    fig7_strict_supported = bool(fig7_gate.get("strict_claim_supported"))
    fig8_report = project_root / "outputs" / "kg_perturbation_fig8" / "figure_quality_report.json"
    fig8_gate = quality_gate(fig8_report)

    rows: List[Dict[str, Any]] = [
        {
            "figure": "Fig.1",
            "claim_id": "fig1_graph_measurement",
            "main_text_role": "diagnostic figure" if not gate_passed(reports["Fig.1"]) else "main evidence",
            "current_status": quality_gate(reports["Fig.1"]).get("status_label", "missing"),
            "quality_gate_path": str(reports["Fig.1"]),
            "quality_gate_pass": int(gate_passed(reports["Fig.1"])),
            "allowed_claim": "Graph-perturbation maps define candidate structural signals.",
            "forbidden_claim": "Do not compare graph density across domains while edge cap or horizon gates fail.",
            "required_action": failed_checks(reports["Fig.1"]) or "Keep edge-cap and horizon gates passing.",
        },
        {
            "figure": "Fig.2",
            "claim_id": "fig2_indicator_validation",
            "main_text_role": "diagnostic association" if not gate_passed(reports["Fig.2"]) else "main empirical validation",
            "current_status": quality_gate(reports["Fig.2"]).get("status_label", "missing"),
            "quality_gate_path": str(reports["Fig.2"]),
            "quality_gate_pass": int(gate_passed(reports["Fig.2"])),
            "allowed_claim": (
                "Indicators provide multi-domain empirical validation against future graph outcomes "
                "when sample-size, matched-control, reference-closure, and mechanism-correlation gates pass."
            ),
            "forbidden_claim": "Do not describe Fig.2 as strong experimental evidence until sample, control, and closure gates pass.",
            "required_action": failed_checks(reports["Fig.2"]) or "Freeze strong-evidence gate outputs.",
        },
        {
            "figure": "Fig.3",
            "claim_id": "fig3_learned_score",
            "main_text_role": "diagnostic association" if not gate_passed(reports["Fig.3"]) else "main evidence",
            "current_status": quality_gate(reports["Fig.3"]).get("status_label", "missing"),
            "quality_gate_path": str(reports["Fig.3"]),
            "quality_gate_pass": int(gate_passed(reports["Fig.3"])),
            "allowed_claim": (
                "The no-leakage learned perturbation score predicts future graph outcomes and improves over "
                "major baselines in random cross-validation, temporal holdout, and leave-domain-out validation when gates pass."
            ),
            "forbidden_claim": "Do not claim a Nature-ready innovation predictor while OOF and enrichment gates fail.",
            "required_action": failed_checks(reports["Fig.3"]) or "Freeze no-leakage baseline, temporal-holdout, and leave-domain-out outputs.",
        },
        {
            "figure": "Fig.4",
            "claim_id": "fig4_external_validation",
            "main_text_role": "extended range-restricted peer-review audit" if not fig4["claim_ready"] else "main validation",
            "current_status": fig4["status"],
            "quality_gate_path": str(project_root / "outputs" / "kg_perturbation_fig4_full50" / "figure_quality_report.json"),
            "quality_gate_pass": int(bool(fig4["claim_ready"])),
            "allowed_claim": "Fig.4 is a range-restricted peer-review audit among accepted high-tier Nature Portfolio papers; it is not the main external validation claim unless alignment gates pass.",
            "forbidden_claim": "Do not claim global external validation, peer-review equivalence, reviewer replacement, or Fig3-score novelty alignment while Fig4 lacks low/middle global Fig3 tiers.",
            "required_action": fig4["details"] if fig4["claim_ready"] else f"{fig4['details']}; keep Fig.4 in Extended Data and use it as a range-restricted audit.",
        },
        {
            "figure": "Fig.5",
            "claim_id": "fig5_ai_frontier" if fig5_ai_ready else "fig5_forecast_backtest",
            "main_text_role": "source-backed AI frontier handoff" if fig5_ai_ready else "extended-data forecast backtest",
            "current_status": fig5_ai_payload.get("status_label")
            if fig5_ai_ready
            else fig5_payload.get("status_label") or quality_gate(fig5_report).get("status_label", "missing_forecast_backtest"),
            "quality_gate_path": str(fig5_ai_report if fig5_ai_ready else fig5_report),
            "quality_gate_pass": int(fig5_ai_ready or gate_passed(fig5_report)),
            "allowed_claim": (
                "Fig.5 provides a source-backed 2024-2026 AI/AI-enabled science frontier data contract for a dense point-cloud visual; OpenAlex/local evidence rows, AI terms, themes, query reports, and manifest are auditable."
                if fig5_ai_ready
                else (
                "Fig.5 supports a no-leakage retrospective forecast/backtest claim in Extended Data: graph-score forecasts beat historical-growth/citation baselines on mean precision@10 and NDCG@10."
                if fig5_beats_baseline
                else "Fig.5 forecast/backtest claims are traceable to CSV tables, but performance claims require beating no-leakage historical baselines."
                )
            ),
            "forbidden_claim": (
                "Do not describe unverified buzzwords as current AI hotspots; do not use the legacy backtest image or an image-model rendering as evidence for forecast accuracy."
                if fig5_ai_ready
                else "Do not use image handoff, layout draft, or future-informed baselines as evidence for forecast accuracy."
            ),
            "required_action": failed_checks(fig5_report)
            or (
                "Redraw Fig.5 from ai_frontier_point_cloud.csv; remove low-value panels and take-home footer; keep retrospective backtest as supporting audit."
                if fig5_ai_ready
                else (
                "Freeze no-leakage backtest metrics and keep Fig.5 in Extended Data unless promoted by manuscript space."
                if fig5_beats_baseline
                else "Use only if graph-score backtests beat baseline ranking methods."
                )
            ),
        },
        {
            "figure": "Fig.6",
            "claim_id": "fig6_robustness",
            "main_text_role": "supplementary robustness" if not bool(quality_gate(reports["Fig.6"]).get("nature_strong_claim_ready")) else "main robustness",
            "current_status": quality_gate(reports["Fig.6"]).get("status_label", "missing"),
            "quality_gate_path": str(reports["Fig.6"]),
            "quality_gate_pass": int(bool(quality_gate(reports["Fig.6"]).get("nature_strong_claim_ready"))),
            "allowed_claim": "Fig.6 supports full graph-rerun robustness when construction-matched OpenAlex rerun artifacts and rank-stability gates pass."
            if bool(quality_gate(reports["Fig.6"]).get("nature_strong_claim_ready"))
            else "Fig.6 supports cached/proxy robustness stress testing and records that the current fresh OpenAlex full-rerun audit is unstable.",
            "forbidden_claim": ""
            if bool(quality_gate(reports["Fig.6"]).get("nature_strong_claim_ready"))
            else "Do not claim full online graph-extraction robustness from cached/proxy panels.",
            "required_action": quality_gate(reports["Fig.6"]).get("replacement_gate", "Rerun online graph extraction under perturbations."),
        },
        {
            "figure": "Fig.7",
            "claim_id": "fig7_venue_contribution",
            "main_text_role": "extended-data venue point-estimate",
            "current_status": fig7_gate.get("status_label", "missing"),
            "quality_gate_path": str(reports["Fig.7"]),
            "quality_gate_pass": int(fig7_point_estimate_supported),
            "allowed_claim": "Nature Portfolio has the top aggregate VCI point estimate in the current controlled corpus.",
            "forbidden_claim": "Do not claim strict Nature Portfolio dominance or superiority while interval and pairwise gates fail.",
            "required_action": "Keep point-estimate wording; strict dominance remains unsupported by interval/pairwise gates."
            if fig7_point_estimate_supported and not fig7_strict_supported
            else failed_checks(reports["Fig.7"]) or "Keep strict interval and pairwise gates passing.",
        },
        {
            "figure": "Fig.8",
            "claim_id": "fig8_architecture",
            "main_text_role": "architecture overview",
            "current_status": fig8_gate.get("status_label", "fig8_handoff_missing"),
            "quality_gate_path": str(fig8_report),
            "quality_gate_pass": int(gate_passed(fig8_report)),
            "allowed_claim": "Fig.8 describes the ASPR application architecture, not statistical performance.",
            "forbidden_claim": "Do not use Fig.8 as performance evidence or imply ASPR-Qwen is validated by the architecture diagram alone.",
            "required_action": failed_checks(fig8_report) or "Keep GPT-image handoff manifest, prompt, and quality report bound to Fig.9/Fig.10 evidence gates.",
        },
        {
            "figure": "Fig.9",
            "claim_id": "fig9_case_storyboard",
            "main_text_role": "supplementary checkpoint case" if fig9_checkpoint_ready else "supplementary prototype case",
            "current_status": fig9_boundary,
            "quality_gate_path": str(reports["Fig.9"]),
            "quality_gate_pass": int(fig9_checkpoint_ready),
            "allowed_claim": (
                "Fig.9 is a single auditable checkpoint-generated ASPR-Qwen case storyboard with saved model metadata."
                if fig9_checkpoint_ready
                else "Fig.9 is a single auditable prototype case storyboard with an explicitly labeled ASPR-Qwen boundary."
            ),
            "forbidden_claim": (
                "Do not cite one Fig.9 case as representative aggregate ASPR checkpoint performance."
                if fig9_checkpoint_ready
                else "Do not cite Fig.9 as representative ASPR checkpoint performance until checkpoint output is saved."
            ),
            "required_action": fig9_payload.get("replacement_gate", "Replace assumed Qwen output with checkpoint output."),
        },
        {
            "figure": "Fig.10",
            "claim_id": "fig10_module_ablation",
            "main_text_role": "pipeline audit" if not fig10["claim_ready"] else "main ablation evidence",
            "current_status": fig10["status"],
            "quality_gate_path": str(project_root / "outputs" / "kg_perturbation_fig10" / "fig10_replacement_gates.csv"),
            "quality_gate_pass": int(bool(fig10["claim_ready"])),
            "allowed_claim": "Fig.10 is a pipeline audit with observed full-ASPR metrics and same-rubric generic baseline.",
            "forbidden_claim": "Do not claim completed causal module reruns, human preference, or ASPR-Qwen checkpoint performance while replacement gates fail.",
            "required_action": fig10["details"],
        },
    ]
    required_gate = {
        "Fig.1": "fig1_sampling_horizon",
        "Fig.2": "fig2_reference_closure_controls",
        "Fig.3": "fig3_holdout_baselines",
        "Fig.4": "fig4_external_validation",
        "Fig.5": "fig5_forecast_backtest",
        "Fig.6": "fig6_full_rerun_robustness",
        "Fig.7": "fig7_strict_or_downgraded_claim",
        "Fig.8": "fig8_source_renderer",
        "Fig.9": "fig9_checkpoint_boundary",
        "Fig.10": "fig10_replacement_gates",
    }
    required_artifacts = {
        "Fig.1": "fig1_edge_sampling_manifest.csv; figure_quality_report.json; run_manifest.json",
        "Fig.2": "fig2_reference_closure_report.csv; fig2_control_tier_audit.csv; fig2_quality_gates.json",
        "Fig.3": "fig3_baseline_comparison.csv; fig3_temporal_holdout.csv; fig3_leave_domain_out.csv; figure_quality_report.json",
        "Fig.4": "fig4_fixed_sample_manifest.csv; fig4_retrieval_audit.jsonl; fig4_metrics_summary.csv; figure_quality_report.json",
        "Fig.5": "fig5_backtest_focus.csv; fig5_alignment_metrics.csv; fig5_failure_cases.csv; figure_quality_report.json",
        "Fig.6": "fig6_full_rerun_manifest.csv; fig6_indicator_stability.csv; fig6_rank_stability.csv; figure_quality_report.json",
        "Fig.7": "fig7_metric_sensitivity.csv; fig7_pairwise_contribution_tests.csv; figure_quality_report.json",
        "Fig.8": "experiments/kg_perturbation_fig8/render_fig8.py; fig8_full.png; fig8_full.svg",
        "Fig.9": "fig9_aspr_qwen_output.json; fig9_checkpoint_metadata.json; fig9_quality_report.json",
        "Fig.10": "fig10_true_module_rerun_results.csv; fig10_human_preference.csv; fig10_replacement_gates.csv",
    }
    for row in rows:
        figure = str(row.get("figure", ""))
        row["required_gate"] = required_gate.get(figure, "")
        row["required_artifacts"] = required_artifacts.get(figure, "")
        row["main_or_extended_data"] = "main" if figure in MAIN_CLAIM_FIGURES else "extended"
    return pd.DataFrame(rows)


def build_nature_check_rows(project_root: Path = PROJECT_ROOT) -> List[Dict[str, Any]]:
    """Build high-level readiness checks from current outputs."""
    reports = {
        "fig1": project_root / "outputs" / "redraw_v6a_best_fig1" / "figure_quality_report.json",
        "fig2": project_root / "outputs" / "redraw_v6a_best_fig2" / "fig2_quality_gates.json",
        "fig3": project_root / "outputs" / "redraw_v6a_best_fig3" / "figure_quality_report.json",
        "fig6": project_root / "outputs" / "kg_perturbation_fig6" / "figure_quality_report.json",
        "fig7": project_root / "outputs" / "kg_perturbation_fig7" / "figure_quality_report.json",
    }
    fig4 = fig4_peer_review_alignment_status(project_root)
    fig10 = fig10_replacement_gate_status(project_root)
    ledger = build_claim_ledger(project_root)
    fig1_fig3_passed = bool(gate_passed(reports["fig1"]) and gate_passed(reports["fig2"]) and gate_passed(reports["fig3"]))
    fig1_fig3_actions: List[str] = []
    for label, key in [("Fig.1", "fig1"), ("Fig.2", "fig2"), ("Fig.3", "fig3")]:
        if not gate_passed(reports[key]):
            details = failed_checks(reports[key])
            fig1_fig3_actions.append(f"{label} failed checks: {details or 'quality gate missing'}")
    fig1_fig3_action = "; ".join(fig1_fig3_actions) if fig1_fig3_actions else "Fig.1-Fig.3 gates pass."
    fig4_is_main_claim = "Fig.4" in MAIN_CLAIM_FIGURES
    fig4_passed = bool(fig4["claim_ready"]) if fig4_is_main_claim else True
    fig4_scope = "main_claim_blocker" if fig4_is_main_claim else "extended_data_nonblocking"
    fig4_action = (
        fig4["details"]
        if fig4_is_main_claim or fig4["claim_ready"]
        else f"{fig4['details']}; Fig.4 is downgraded to Extended Data and must not support the main external-validation claim."
    )
    return [
        {
            "check_id": "claim_ledger_written",
            "passed": bool(not ledger.empty and ledger["figure"].nunique() == 10),
            "status": "ready" if not ledger.empty else "missing",
            "required_action": "Write fig1_fig10_claim_ledger.csv.",
            "blocking_scope": "main_claim_blocker",
        },
        {
            "check_id": "fig1_fig3_strong_gates",
            "passed": fig1_fig3_passed,
            "status": "pass" if fig1_fig3_passed else "diagnostic_or_underpowered",
            "required_action": fig1_fig3_action,
            "blocking_scope": "main_claim_blocker",
        },
        {
            "check_id": "fig4_external_validation",
            "passed": fig4_passed,
            "status": fig4["status"],
            "required_action": fig4_action,
            "blocking_scope": fig4_scope,
        },
        {
            "check_id": "fig6_full_rerun_robustness",
            "passed": bool(quality_gate(reports["fig6"]).get("nature_strong_claim_ready")),
            "status": quality_gate(reports["fig6"]).get("status_label", "missing"),
            "required_action": quality_gate(reports["fig6"]).get("replacement_gate", "Rerun online graph extraction under perturbations."),
            "blocking_scope": "main_claim_blocker",
        },
        {
            "check_id": "fig7_strict_or_downgraded_claim",
            "passed": bool(quality_gate(reports["fig7"]).get("strict_claim_supported"))
            or "superiority" not in " ".join(ledger["allowed_claim"].astype(str)).lower(),
            "status": quality_gate(reports["fig7"]).get("status_label", "missing"),
            "required_action": "Pass strict interval gates or keep only point-estimate wording.",
            "blocking_scope": "extended_data_nonblocking",
        },
        {
            "check_id": "fig10_replacement_gates",
            "passed": bool(fig10["claim_ready"]),
            "status": fig10["status"],
            "required_action": fig10["details"],
            "blocking_scope": "extended_data_nonblocking",
        },
    ]


def build_all_figure_claim_gate_check(ledger: pd.DataFrame) -> Dict[str, Any]:
    """Build a strict gate requiring every figure's current allowed claim to pass."""
    if ledger.empty or "quality_gate_pass" not in ledger.columns:
        return {
            "check_id": "all_figures_claim_gates",
            "passed": False,
            "status": "claim_ledger_missing",
            "required_action": "Build a complete claim ledger before strict all-figure checking.",
            "blocking_scope": "all_figure_blocker",
        }
    failed = ledger.loc[~ledger["quality_gate_pass"].astype(bool)].copy()
    failed_figures = failed["figure"].astype(str).tolist()
    actions = [
        f"{row['figure']}: {row.get('required_action', '')}"
        for _, row in failed.iterrows()
    ]
    return {
        "check_id": "all_figures_claim_gates",
        "passed": not failed_figures,
        "status": "all_figures_ready" if not failed_figures else "extended_data_replacement_gates_block_all_figure_ready",
        "required_action": "All Fig.1-Fig.10 claim gates pass." if not failed_figures else "; ".join(actions),
        "blocking_scope": "all_figure_blocker",
        "failed_figures": ";".join(failed_figures),
    }


def build_nature_check_report(
    project_root: Path = PROJECT_ROOT,
    out_dir: Optional[Path] = None,
    *,
    require_all_figures: bool = False,
) -> Dict[str, Any]:
    """Write and return the Nature-readiness claim ledger and summary."""
    destination = out_dir or project_root / "outputs" / "kg_perturbation_final_assembly"
    destination.mkdir(parents=True, exist_ok=True)
    ledger = build_claim_ledger(project_root)
    checks = build_nature_check_rows(project_root)
    if require_all_figures:
        checks.append(build_all_figure_claim_gate_check(ledger))
    ledger.to_csv(destination / "fig1_fig10_claim_ledger.csv", index=False)
    summary_name = (
        "fig1_fig10_all_figures_nature_check_summary.csv"
        if require_all_figures
        else "fig1_fig10_nature_check_summary.csv"
    )
    report_name = (
        "fig1_fig10_all_figures_nature_check_report.json"
        if require_all_figures
        else "fig1_fig10_nature_check_report.json"
    )
    pd.DataFrame(checks).to_csv(destination / summary_name, index=False)
    main_checks = [row for row in checks if row.get("blocking_scope") == "main_claim_blocker"]
    main_pass = bool(all(row["passed"] for row in main_checks))
    all_figure_pass = True
    if require_all_figures:
        all_figure_checks = [row for row in checks if row.get("blocking_scope") == "all_figure_blocker"]
        all_figure_pass = bool(all(row["passed"] for row in all_figure_checks))
    overall_pass = main_pass and all_figure_pass
    status_label = (
        "nature_ready"
        if overall_pass
        else (
            "all_figures_need_revision_before_nature_submission"
            if require_all_figures
            else "needs_revision_before_nature_submission"
        )
    )
    report = {
        "overall_pass": overall_pass,
        "status_label": status_label,
        "require_all_figures": bool(require_all_figures),
        "checks": checks,
        "claim_ledger": str(destination / "fig1_fig10_claim_ledger.csv"),
        "summary_csv": str(destination / summary_name),
    }
    (destination / report_name).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _strict_fig4_blinded_labels_ready(project_root: Path) -> Dict[str, Any]:
    """Validate returned Fig.4 blinded labels for all primary tier slots."""
    from experiments.kg_perturbation_fig4.main_fig4 import (
        FIG4_BLINDED_LABEL_COLUMNS,
        build_fig4_blinded_external_validation_gates,
        import_completed_fig4_blinded_label_sidecar,
    )

    fig4_dir = project_root / "outputs" / "kg_perturbation_fig4_full50"
    labels_path = fig4_dir / "fig4_completed_blinded_labels.csv"
    key_path = fig4_dir / "fig4_blinded_labeling_answer_key.csv"
    try:
        labels = pd.read_csv(labels_path)
        key = pd.read_csv(key_path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        return {"passed": False, "detail": f"unreadable_fig4_label_artifact: {exc}"}
    required_label_cols = {
        "blinded_case_id",
        "label_novelty_1_5",
        "label_significance_1_5",
        "label_prior_art_1_5",
        "label_confidence_1_5",
        "label_source",
        "labeler_id",
    }
    required_key_cols = {"blinded_case_id", "global_fig3_tier", "assignment_role"}
    if labels.empty or not required_label_cols.issubset(set(labels.columns)):
        missing = sorted(required_label_cols - set(labels.columns))
        return {"passed": False, "detail": f"fig4_completed_labels_missing_columns: {';'.join(missing)}"}
    if key.empty or not required_key_cols.issubset(set(key.columns)):
        missing = sorted(required_key_cols - set(key.columns))
        return {"passed": False, "detail": f"fig4_answer_key_missing_columns: {';'.join(missing)}"}
    primary = key[key["assignment_role"].astype(str).eq("primary_validation_labeling_sample")].copy()
    merged = primary.merge(labels, on="blinded_case_id", how="left")
    numeric_cols = [
        "label_novelty_1_5",
        "label_significance_1_5",
        "label_prior_art_1_5",
        "label_confidence_1_5",
    ]
    numeric = merged[numeric_cols].apply(pd.to_numeric, errors="coerce")
    source_ok = merged["label_source"].astype(str).str.strip().ne("")
    labeler_ok = merged["labeler_id"].astype(str).str.strip().ne("")
    nonhuman_source = _nonhuman_or_synthetic_source_rows(merged, ["label_source", "labeler_id"])
    if bool(nonhuman_source.any()):
        return {
            "passed": False,
            "detail": (
                "fig4_blinded_labels_nonhuman_or_synthetic_source: "
                f"rows={int(nonhuman_source.sum())}"
            ),
        }
    valid = numeric.ge(1).all(axis=1) & numeric.le(5).all(axis=1) & source_ok & labeler_ok
    complete = merged[valid].copy()
    tier_counts = complete["global_fig3_tier"].astype(str).value_counts().to_dict()
    required_tiers = {"low": 10, "middle": 10, "high": 10}
    tiers_ready = all(int(tier_counts.get(tier, 0)) >= count for tier, count in required_tiers.items())
    passed = bool(len(complete) >= 30 and tiers_ready)
    if not passed:
        detail = "fig4_blinded_labels_incomplete: " + ", ".join(
            f"{tier}={int(tier_counts.get(tier, 0))}/10" for tier in ["low", "middle", "high"]
        )
        return {"passed": False, "detail": detail}
    with tempfile.TemporaryDirectory(prefix="aspr_fig4_strict_evidence_") as tmp:
        tmp_fig4_dir = Path(tmp) / "kg_perturbation_fig4_full50"
        shutil.copytree(fig4_dir, tmp_fig4_dir, dirs_exist_ok=True)
        try:
            packet = pd.read_csv(tmp_fig4_dir / "fig4_blinded_labeling_packet.csv")
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            return {"passed": False, "detail": f"unreadable_fig4_blinded_packet: {exc}"}
        for column in FIG4_BLINDED_LABEL_COLUMNS:
            if column not in packet.columns:
                packet[column] = ""
        packet = import_completed_fig4_blinded_label_sidecar(tmp_fig4_dir, packet)
        packet.to_csv(tmp_fig4_dir / "fig4_blinded_labeling_packet.csv", index=False)
        gates = build_fig4_blinded_external_validation_gates(tmp_fig4_dir)
    if bool(gates.get("overall_pass")):
        return {"passed": True, "detail": "fig4_blinded_labels_and_external_validation_ready"}
    failed = [
        str(key)
        for key, value in gates.get("checks", {}).items()
        if not bool(value)
    ]
    return {
        "passed": False,
        "detail": (
            f"{gates.get('status_label', 'blinded_external_validation_incomplete')}: "
            + ";".join(failed)
        ),
    }


def _strict_fig9_checkpoint_ready(project_root: Path) -> Dict[str, Any]:
    """Validate that Fig.9 uses a checkpoint-generated ASPR-Qwen output."""
    from experiments.kg_perturbation_fig9.build_fig9_case import checkpoint_qwen_metadata_complete

    fig9_dir = project_root / "outputs" / "kg_perturbation_fig9"
    output = read_json(fig9_dir / "fig9_aspr_qwen_output.json")
    metadata = read_json(fig9_dir / "fig9_checkpoint_metadata.json")
    if metadata and not isinstance(output.get("checkpoint_metadata"), Mapping):
        output = dict(output)
        output["checkpoint_metadata"] = metadata
    passed = bool(output and checkpoint_qwen_metadata_complete(output))
    return {
        "passed": passed,
        "detail": "fig9_checkpoint_metadata_complete" if passed else "fig9_checkpoint_output_or_metadata_invalid",
    }


def _strict_fig10_true_reruns_ready(project_root: Path) -> Dict[str, Any]:
    """Validate Fig.10 true disabled-module rerun rows and artifact links."""
    from experiments.kg_perturbation_fig10.build_fig10_ablation import (
        VARIANTS,
        expected_case_ids,
        true_module_rerun_status,
    )

    path = project_root / "outputs" / "kg_perturbation_fig10" / "fig10_true_module_rerun_results.csv"
    fig4_metrics = project_root / "outputs" / "kg_perturbation_fig4_full50" / "fig4_metrics_summary.csv"
    try:
        table = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        return {"passed": False, "detail": f"unreadable_true_rerun_results: {exc}"}
    if table.empty or "case_id" not in table.columns or "variant" not in table.columns:
        return {"passed": False, "detail": "true_rerun_results_missing_case_or_variant"}
    required = table[table["variant"].astype(str).isin(VARIANTS)].copy()
    observed_case_ids = sorted(required["case_id"].astype(str).unique())
    expected_ids = expected_case_ids(fig4_metrics) or observed_case_ids
    status = true_module_rerun_status(path, expected_ids=expected_ids)
    expected_rows = 50 * len(VARIANTS)
    passed = bool(len(expected_ids) == 50 and len(required) == expected_rows and status == "observed_true_module_reruns")
    detail = (
        "fig10_true_disabled_module_reruns_ready"
        if passed
        else (
            "fig10_true_disabled_module_reruns_invalid: "
            f"status={status}; expected_cases={len(expected_ids)}; "
            f"observed_cases={len(observed_case_ids)}; rows={len(required)}/{expected_rows}"
        )
    )
    return {"passed": passed, "detail": detail}


def _strict_fig10_blinded_preferences_ready(project_root: Path) -> Dict[str, Any]:
    """Validate Fig.10 completed blinded preference collection."""
    from experiments.kg_perturbation_fig10.build_fig10_ablation import (
        build_fig10_blinded_preference_completion_audit,
        load_human_preference_results,
    )

    fig10_dir = project_root / "outputs" / "kg_perturbation_fig10"
    audit = build_fig10_blinded_preference_completion_audit(fig10_dir, write=False)
    overall = audit[audit["audit_item"].astype(str).eq("overall_blinded_preference_ready")]
    if overall.empty:
        return {"passed": False, "detail": "fig10_blinded_preference_audit_missing_overall_row"}
    row = overall.iloc[0]
    required_judgements = int(row.get("required_judgements") or 0)
    completion_passed = bool(int(row.get("pass") or 0) and required_judgements >= 750)
    if not completion_passed:
        detail = (
            "fig10_blinded_human_preference_incomplete: "
            f"required_judgements={required_judgements}; "
            f"missing_judgements={int(row.get('missing_judgements') or 0)}; "
            f"evaluator_count={int(row.get('evaluator_count') or 0)}"
        )
        return {"passed": False, "detail": detail}
    provenance_path = fig10_dir / "fig10_completed_blinded_preferences.csv"
    if not provenance_path.exists():
        provenance_path = fig10_dir / "fig10_human_preference.csv"
    try:
        provenance = pd.read_csv(provenance_path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        return {"passed": False, "detail": f"unreadable_fig10_preference_provenance: {exc}"}
    source_columns = [
        column
        for column in ["evaluator_type", "preference_source", "label_source", "source"]
        if column in provenance.columns
    ]
    if not source_columns:
        return {
            "passed": False,
            "detail": "fig10_blinded_preferences_nonhuman_or_synthetic_source: missing_human_provenance_source",
        }
    nonhuman_source = _nonhuman_or_synthetic_source_rows(provenance, source_columns)
    if "evaluator_id" in provenance.columns:
        evaluator_identity = provenance[["evaluator_id"]].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
        for pattern in NONHUMAN_EXTERNAL_EVIDENCE_PATTERNS:
            nonhuman_source = nonhuman_source | evaluator_identity.str.contains(pattern, regex=False, na=False)
    if bool(nonhuman_source.any()):
        return {
            "passed": False,
            "detail": (
                "fig10_blinded_preferences_nonhuman_or_synthetic_source: "
                f"rows={int(nonhuman_source.sum())}"
            ),
        }
    preference = load_human_preference_results(fig10_dir / "fig10_human_preference.csv")
    key_dimensions = {"prior_art", "evidence_grounding", "usefulness"}
    support_rows: List[str] = []
    for _, pref_row in preference.iterrows():
        question = str(pref_row.get("question") or "").strip().lower()
        if question not in key_dimensions:
            continue
        full_wins = int(pref_row.get("full_aspr_wins") or 0)
        comparator_wins = int(pref_row.get("comparator_wins") or 0)
        decisive = full_wins + comparator_wins
        lower = _wilson_lower_bound(full_wins, decisive)
        if decisive >= 30:
            support_rows.append(
                f"{question}:full={full_wins},comparator={comparator_wins},wilson_low={lower:.3f}"
            )
        if decisive >= 30 and lower > 0.5:
            return {
                "passed": True,
                "detail": f"fig10_blinded_human_preference_ready: {support_rows[-1]}",
            }
    return {
        "passed": False,
        "detail": (
            "full_aspr_preference_not_supported_on_key_dimensions: "
            + ("; ".join(support_rows) if support_rows else "no_decisive_key_dimension_preferences")
        ),
    }


def _wilson_lower_bound(successes: int, trials: int, z: float = 1.96) -> float:
    """Return the Wilson score lower bound for a binomial proportion."""
    if trials <= 0:
        return 0.0
    p_hat = successes / trials
    z2 = z * z
    denom = 1.0 + z2 / trials
    center = p_hat + z2 / (2 * trials)
    margin = z * math.sqrt((p_hat * (1.0 - p_hat) + z2 / (4 * trials)) / trials)
    return max(0.0, (center - margin) / denom)


def _strict_evidence_content_status(project_root: Path, row: Mapping[str, Any]) -> Dict[str, Any]:
    """Return content-level validation for one strict external-evidence checklist row."""
    figure = str(row.get("figure", ""))
    gate = str(row.get("gate_to_clear", ""))
    if figure == "Fig.4":
        return _strict_fig4_blinded_labels_ready(project_root)
    if figure == "Fig.9":
        return _strict_fig9_checkpoint_ready(project_root)
    if figure == "Fig.10" and gate == "true_disabled_module_reruns":
        return _strict_fig10_true_reruns_ready(project_root)
    if figure == "Fig.10" and gate == "blinded_human_preference":
        return _strict_fig10_blinded_preferences_ready(project_root)
    return {"passed": False, "detail": "strict_evidence_validator_missing"}


def build_strict_evidence_check_report(
    project_root: Path = PROJECT_ROOT,
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Write and return the strict external-evidence artifact check."""
    from experiments.kg_perturbation_final_assembly.build_final_assembly import (
        build_strict_evidence_collection_checklist,
    )

    destination = out_dir or project_root / "outputs" / "kg_perturbation_final_assembly"
    destination.mkdir(parents=True, exist_ok=True)
    checklist = build_strict_evidence_collection_checklist(project_root)
    checks: List[Dict[str, Any]] = []
    for _, row in checklist.iterrows():
        required_present = bool(row.get("required_artifacts_present"))
        templates_present = bool(row.get("source_templates_present"))
        content_status = (
            _strict_evidence_content_status(project_root, row.to_dict())
            if required_present and templates_present
            else {"passed": False, "detail": "strict_evidence_files_missing"}
        )
        content_valid = bool(content_status.get("passed"))
        passed = required_present and templates_present and content_valid
        status = (
            "ready"
            if passed
            else (
                "missing_required_submission_artifact"
                if not required_present and templates_present
                else "invalid_required_submission_artifact"
                if required_present and templates_present
                else "missing_source_template_or_packet"
            )
        )
        checks.append(
            {
                "figure": str(row.get("figure", "")),
                "blocker": str(row.get("blocker", "")),
                "required_submission_artifact": str(row.get("required_submission_artifact", "")),
                "source_template_or_packet": str(row.get("source_template_or_packet", "")),
                "expected_completion": str(row.get("expected_completion", "")),
                "gate_to_clear": str(row.get("gate_to_clear", "")),
                "rerun_command": str(row.get("rerun_command", "")),
                "required_artifacts_present": int(required_present),
                "source_templates_present": int(templates_present),
                "artifact_content_valid": int(content_valid),
                "validation_detail": str(content_status.get("detail", "")),
                "passed": bool(passed),
                "status": status,
            }
        )
    summary = pd.DataFrame(checks)
    summary.to_csv(destination / "fig1_fig10_strict_evidence_check_summary.csv", index=False)
    overall_pass = bool(checks and all(row["passed"] for row in checks))
    report = {
        "overall_pass": overall_pass,
        "status_label": "strict_external_evidence_ready"
        if overall_pass
        else "strict_external_evidence_missing",
        "checks": checks,
        "summary_csv": str(destination / "fig1_fig10_strict_evidence_check_summary.csv"),
    }
    (destination / "fig1_fig10_strict_evidence_check_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "kg_perturbation_final_assembly")
    parser.add_argument("--require-all-figures", action="store_true",
                        help="Require every Fig.1-Fig.10 claim gate, including Extended Data replacement gates, to pass.")
    parser.add_argument("--strict-evidence-check", action="store_true",
                        help="Check whether exact external evidence files needed for strict Fig.1-Fig.10 readiness exist.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.strict_evidence_check:
        report = build_strict_evidence_check_report(args.project_root, args.out_dir)
    else:
        report = build_nature_check_report(args.project_root, args.out_dir, require_all_figures=args.require_all_figures)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])

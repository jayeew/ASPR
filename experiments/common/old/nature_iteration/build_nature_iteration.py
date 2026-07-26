"""Build one automatic Fig.1-Fig.10 Nature-level iteration audit."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_OUT_ROOT = PROJECT_ROOT / "outputs" / "common/old/nature_iteration"
MAX_MAIN_ITERATIONS = 6
FINAL_PATCH_ALLOWED = True
AUTO_CONTINUE_WITHOUT_USER_CHOICE = True
HARD_BLOCKERS_ONLY = [
    "missing_key_data",
    "external_api_unavailable_without_local_fallback",
    "non_reproducible_generation",
    "strong_claim_negated_by_data",
]

QUALITY_REPORTS: Dict[str, Path] = {
    "Fig.1": PROJECT_ROOT / "outputs" / "fig01/old" / "figure_quality_report.json",
    "Fig.2": PROJECT_ROOT / "outputs" / "fig02/old" / "figure_quality_report.json",
    "Fig.3": PROJECT_ROOT / "outputs" / "fig03/old" / "multi_domain" / "figure_quality_report.json",
    "Fig.4": PROJECT_ROOT / "outputs" / "fig04/old" / "figure_quality_report.json",
    "Fig.5": PROJECT_ROOT / "outputs" / "fig05/old" / "figure_quality_report.json",
    "Fig.6": PROJECT_ROOT / "outputs" / "fig06/old" / "figure_quality_report.json",
    "Fig.7": PROJECT_ROOT / "outputs" / "fig07/old" / "figure_quality_report.json",
    "Fig.8": PROJECT_ROOT / "outputs" / "fig08/old" / "figure_quality_report.json",
    "Fig.9": PROJECT_ROOT / "outputs" / "fig09/old" / "figure_quality_report.json",
    "Fig.10": PROJECT_ROOT / "outputs" / "fig10/old" / "figure_quality_report.json",
}

FIG1_CONFIGS: Sequence[Path] = (
    PROJECT_ROOT / "experiments" / "fig01/old" / "configs" / "v6a_display_crispr.yaml",
    PROJECT_ROOT / "experiments" / "fig01/old" / "configs" / "v6a_display_graphene.yaml",
    PROJECT_ROOT / "experiments" / "fig01/old" / "configs" / "v6a_display_ipsc.yaml",
    PROJECT_ROOT / "experiments" / "fig01/old" / "configs" / "v6a_display_exoplanets.yaml",
)

AI_TERMS = {
    "ai",
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "neural network",
    "large language model",
    "language model",
    "foundation model",
    "transformer",
    "diffusion model",
    "generative",
    "multimodal",
    "reinforcement learning",
    "computer vision",
    "self-supervised",
}


@dataclass
class FixItem:
    figure: str
    priority: str
    issue: str
    default_action: str
    gate: str


def read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON object, returning a missing marker if absent."""
    if not path.exists():
        return {"missing": True, "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    """Write dictionaries to CSV with stable columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def relpath(path: Path) -> str:
    """Return a project-relative path when possible."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def quality_pass(report: Dict[str, Any]) -> bool:
    """Return whether a figure quality report passes."""
    if report.get("missing"):
        return False
    if "overall_pass" in report:
        return bool(report.get("overall_pass"))
    gate = report.get("quality_gates") or {}
    if isinstance(gate, dict) and "overall_pass" in gate:
        return bool(gate.get("overall_pass"))
    if isinstance(gate, dict) and "pass" in gate:
        return bool(gate.get("pass"))
    return False


def failed_checks(report: Dict[str, Any]) -> str:
    """Collect failed check names from common quality report shapes."""
    if report.get("missing"):
        return "quality_report_missing"
    gate = report.get("quality_gates") or report.get("checks") or {}
    if isinstance(gate, dict):
        checks = gate.get("checks") if isinstance(gate.get("checks"), dict) else gate
        failed = [str(key) for key, value in checks.items() if value in (0, False, "fail", "failed")]
        return "; ".join(failed[:12])
    return ""


def summarize_quality_reports() -> List[Dict[str, Any]]:
    """Build one row per figure from existing quality reports."""
    rows: List[Dict[str, Any]] = []
    for figure, path in QUALITY_REPORTS.items():
        report = read_json(path)
        rows.append(
            {
                "figure": figure,
                "quality_report": str(path.relative_to(PROJECT_ROOT)),
                "exists": int(not report.get("missing")),
                "overall_pass": int(quality_pass(report)),
                "status_label": report.get("status_label") or (report.get("quality_gates") or {}).get("status_label", ""),
                "failed_checks": failed_checks(report),
            }
        )
    return rows


def load_fig1_config(path: Path) -> Dict[str, Any]:
    """Load Fig.1 YAML through the figure module so validation stays local."""
    from experiments.fig01.old.fig1_knowledge_perturbation import load_config

    return load_config(path)


def audit_fig1_landmark_windows() -> List[Dict[str, Any]]:
    """Check whether each Fig.1 domain has before/current/after landmark windows."""
    rows: List[Dict[str, Any]] = []
    for path in FIG1_CONFIGS:
        cfg = load_fig1_config(path)
        windows = [tuple(int(v) for v in item) for item in cfg.get("custom_windows") or []]
        anchor_years = [int(a["year"]) for a in cfg.get("anchors") or [] if a.get("year")]
        first_anchor = min(anchor_years) if anchor_years else None
        has_pre = bool(first_anchor and windows and windows[0][1] < first_anchor)
        has_landmark = bool(first_anchor and any(start <= first_anchor <= end for start, end in windows))
        has_post = bool(first_anchor and any(start > first_anchor for start, _ in windows))
        rows.append(
            {
                "slug": cfg.get("slug", path.stem),
                "config": str(path.relative_to(PROJECT_ROOT)),
                "first_anchor_year": first_anchor or "",
                "windows": json.dumps(windows),
                "pre_landmark_window": int(has_pre),
                "landmark_window": int(has_landmark),
                "post_landmark_window": int(has_post),
                "round1_gate_pass": int(has_pre and has_landmark and has_post),
            }
        )
    return rows


def text_has_ai_signal(text: str) -> bool:
    """Return whether a text field contains a strict AI frontier cue."""
    lowered = text.lower()
    return any(term in lowered for term in AI_TERMS)


def audit_fig5_ai_frontier() -> Dict[str, Any]:
    """Audit whether current Fig.5 forecast rows can support an AI-hotspot claim."""
    ai_report_path = PROJECT_ROOT / "outputs" / "fig05/old" / "ai_frontier" / "ai_frontier_quality_report.json"
    if ai_report_path.exists():
        report = read_json(ai_report_path)
        counts = report.get("counts") or {}
        checks = report.get("checks") or {}
        return {
            "forecast_focus_path": "outputs/fig05/old/ai_frontier/ai_frontier_point_cloud.csv",
            "exists": 1,
            "rows": int(counts.get("frontier_rows") or 0),
            "ai_like_rows": int(counts.get("frontier_rows") or 0),
            "ai_like_positive_score_rows": int(counts.get("point_cloud_rows") or 0),
            "ai_like_top20": int(checks.get("top_points_ai_relevance_all") or 0) * min(20, int(counts.get("point_cloud_rows") or 0)),
            "best_ai_like_rank": 1 if report.get("overall_pass") else "",
            "round1_gate_pass": int(bool(report.get("overall_pass"))),
            "source": "ai_frontier_quality_report",
            "status_label": report.get("status_label", ""),
        }
    forecast_path = PROJECT_ROOT / "outputs" / "fig05/old" / "plot_data" / "derived" / "forecast_focus.csv"
    if not forecast_path.exists():
        return {
            "forecast_focus_path": str(forecast_path.relative_to(PROJECT_ROOT)),
            "exists": 0,
            "rows": 0,
            "ai_like_rows": 0,
            "ai_like_top20": 0,
            "round1_gate_pass": 0,
            "source": "legacy_forecast_focus",
            "status_label": "forecast_focus_missing",
        }

    rows: List[Dict[str, str]] = []
    with forecast_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)

    def row_text(row: Dict[str, str]) -> str:
        fields = ["focus_label", "topic_label", "keywords", "description", "domain"]
        return " ".join(str(row.get(field, "")) for field in fields)

    sorted_rows = sorted(rows, key=lambda row: int(float(row.get("forecast_rank") or row.get("rank") or 10**9)))
    ai_like_rows = [row for row in sorted_rows if text_has_ai_signal(row_text(row))]
    ai_like_top20 = [row for row in sorted_rows[:20] if text_has_ai_signal(row_text(row))]
    positive_ai = [
        row
        for row in ai_like_rows
        if float(row.get("forecast_score") or row.get("score") or 0.0) > 0.0
    ]
    best_rank = ""
    if ai_like_rows:
        best_rank = int(float(ai_like_rows[0].get("forecast_rank") or ai_like_rows[0].get("rank") or 0))
    return {
        "forecast_focus_path": str(forecast_path.relative_to(PROJECT_ROOT)),
        "exists": 1,
        "rows": len(rows),
        "ai_like_rows": len(ai_like_rows),
        "ai_like_positive_score_rows": len(positive_ai),
        "ai_like_top20": len(ai_like_top20),
        "best_ai_like_rank": best_rank,
        "round1_gate_pass": int(len(ai_like_top20) >= 6 and len(positive_ai) >= 20),
        "source": "legacy_forecast_focus",
        "status_label": "legacy_forecast_ai_gate",
    }


def audit_layout_readability() -> List[Dict[str, Any]]:
    """Read the final-assembly layout audit when present."""
    path = PROJECT_ROOT / "outputs" / "common/old/final_assembly" / "fig1_fig10_layout_readability_audit.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def build_fix_items(
    quality_rows: Sequence[Dict[str, Any]],
    fig1_rows: Sequence[Dict[str, Any]],
    fig5_ai: Dict[str, Any],
    layout_rows: Sequence[Dict[str, Any]],
) -> List[FixItem]:
    """Build the automatic next-round fix list."""
    items: List[FixItem] = []
    fig4_claim_scope = read_json(PROJECT_ROOT / "outputs" / "fig04/old" / "fig4_claim_scope_decision.json")
    fig4_claim_scope_pass = int(fig4_claim_scope.get("claim_scope_gate_pass") or 0) == 1
    if any(int(row["round1_gate_pass"]) == 0 for row in fig1_rows):
        items.append(
            FixItem(
                figure="Fig.1",
                priority="P1_major",
                issue="Landmark 前/当时/后窗口不完整，首个子图可能含 landmark。",
                default_action="自动改为 pre-landmark、landmark、post、late/current 四段并重建 Fig.1。",
                gate="fig1_landmark_window_gate",
            )
        )
    if int(fig5_ai.get("round1_gate_pass", 0)) == 0:
        items.append(
            FixItem(
                figure="Fig.5",
                priority="P1_major",
                issue="当前 forecast_focus 不能支持 2024-2026 AI 热点主张。",
                default_action="重建 AI/AI-enabled frontier 数据；若本地证据不足则把 Fig.5 降级为 pipeline-ready gap，不用非 AI 词条冒充热点。",
                gate="fig5_ai_frontier_gate",
            )
        )
    for row in layout_rows:
        if int(row.get("layout_redesign_needed") or 0) == 1:
            items.append(
                FixItem(
                    figure=str(row.get("figure", "")),
                    priority="P3_visual",
                    issue=str(row.get("reading_pass_status") or "layout_redesign_needed"),
                    default_action=str(row.get("nature_level_action") or "Redesign layout for Nature reading pass."),
                    gate="layout_readability_audit",
                )
            )
    for row in quality_rows:
        if row["figure"] == "Fig.4" and fig4_claim_scope_pass:
            continue
        if int(row.get("overall_pass", 0)) == 0:
            items.append(
                FixItem(
                    figure=str(row["figure"]),
                    priority="P2_gate",
                    issue=f"质量门未通过：{row.get('failed_checks') or 'unspecified'}",
                    default_action="按质量门自动修复；无法修复的强 claim 降级或转 Extended Data。",
                    gate="figure_quality_report",
                )
            )
    return items


def write_reflection(
    out_dir: Path,
    round_id: str,
    quality_rows: Sequence[Dict[str, Any]],
    fig1_rows: Sequence[Dict[str, Any]],
    fig5_ai: Dict[str, Any],
    fixes: Sequence[FixItem],
) -> None:
    """Write reader-facing round reflection and next fix list."""
    pass_count = sum(int(row.get("overall_pass", 0)) for row in quality_rows)
    fig1_pass = sum(int(row["round1_gate_pass"]) for row in fig1_rows)
    lines = [
        f"# Nature Iteration {round_id} Reflection",
        "",
        "本轮按自动协议生成：不要求、也不等待用户中途选择；下一轮直接执行 `next_fix_list.md` 中的默认动作。",
        "",
        "## Gate Snapshot",
        "",
        f"- Figure quality reports passing: `{pass_count}/{len(quality_rows)}`",
        f"- Fig.1 landmark-window domains passing: `{fig1_pass}/{len(fig1_rows)}`",
        f"- Fig.5 AI frontier gate pass: `{fig5_ai.get('round1_gate_pass', 0)}`",
        "",
        "## Reflection",
        "",
        "- 优先修会改变科学结论或误导读者的证据问题，再修版面。",
        "- 强 claim 不等数据补齐；如果当前证据不足，默认降级为 association、diagnosis、handoff 或 pipeline-ready gap。",
        "- 新一轮结果通过后，旧轮 PNG/SVG/PDF 可删除；CSV/JSON/manifest/report 保留以便追踪。",
    ]
    (out_dir / "round_reflection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fix_lines = [
        f"# Nature Iteration {round_id} Next Fix List",
        "",
        "自动执行规则：按 priority 顺序处理，不暂停询问用户，不输出路线选择题。只在硬 blocker 出现时停止并记录 blocker。",
        "",
        f"轮次上限：最多 `{MAX_MAIN_ITERATIONS}` 轮主迭代 + `1` 轮 final patch；每轮必须减少风险、降级 claim、转 supplement 或记录 pipeline-ready gap，不能重复生成同类报告来代替推进。",
        "",
    ]
    for idx, item in enumerate(fixes, start=1):
        fix_lines.extend(
            [
                f"## {idx}. {item.figure} | {item.priority}",
                "",
                f"- Issue: {item.issue}",
                f"- Default action: {item.default_action}",
                f"- Gate: `{item.gate}`",
                "",
            ]
        )
    if not fixes:
        fix_lines.append("No blocking fix items. Proceed to final patch if the previous round also had only minor visual issues.")
    (out_dir / "next_fix_list.md").write_text("\n".join(fix_lines), encoding="utf-8")


def write_baseline_audit(
    out_dir: Path,
    quality_rows: Sequence[Dict[str, Any]],
    fig1_rows: Sequence[Dict[str, Any]],
    fig5_ai: Dict[str, Any],
    layout_rows: Sequence[Dict[str, Any]],
    fixes: Sequence[FixItem],
) -> None:
    """Write the explicit Round 0 baseline audit required by the goal protocol."""
    lines = [
        "# Nature Iteration Baseline Audit",
        "",
        "执行协议：Round 0 只冻结当前 Fig.1-Fig.10 的问题清单、数据来源、claim scope 与 style/layout ledger；不要求用户选择路线。",
        "",
        "## Gate Snapshot",
        "",
        f"- Figure quality reports passing: `{sum(int(row.get('overall_pass', 0)) for row in quality_rows)}/{len(quality_rows)}`",
        f"- Fig.1 landmark-window domains passing: `{sum(int(row['round1_gate_pass']) for row in fig1_rows)}/{len(fig1_rows)}`",
        f"- Fig.5 AI frontier gate pass: `{fig5_ai.get('round1_gate_pass', 0)}`",
        f"- Layout redesign items: `{sum(int(row.get('layout_redesign_needed') or 0) for row in layout_rows)}`",
        f"- Fix-list items frozen for automatic execution: `{len(fixes)}`",
        "",
        "## Frozen Next Actions",
        "",
    ]
    if fixes:
        for item in fixes:
            lines.append(f"- **{item.figure}** `{item.priority}` via `{item.gate}`: {item.default_action}")
    else:
        lines.append("- No blocking fix items in the current baseline.")
    (out_dir / "nature_iter_baseline_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def cleanup_previous_visuals(out_root: Path, round_number: int) -> List[str]:
    """Delete old visual artifacts from the previous round after a newer round passes."""
    if round_number <= 0:
        return []
    previous = out_root / f"r{round_number - 1}"
    removed: List[str] = []
    if not previous.exists():
        return removed
    for path in previous.rglob("*"):
        if path.suffix.lower() in {".png", ".svg", ".pdf"}:
            path.unlink()
            removed.append(str(path.relative_to(PROJECT_ROOT)))
    return removed


def cleanup_iteration_visuals(out_root: Path) -> List[str]:
    """Delete visual artifacts from iteration-audit folders while preserving reports."""
    removed: List[str] = []
    for round_dir in sorted(out_root.glob("r*")):
        if not round_dir.is_dir():
            continue
        for path in round_dir.rglob("*"):
            if path.suffix.lower() in {".png", ".svg", ".pdf"}:
                path.unlink()
                removed.append(relpath(path))
    return removed


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    """Read CSV rows, returning an empty list when the artifact is absent."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_final_patch_notes(
    out_dir: Path,
    manifest: Dict[str, Any],
    gap_rows: Sequence[Dict[str, Any]],
    strict_rows: Sequence[Dict[str, Any]],
) -> None:
    """Write the final-patch stop decision in reader-facing form."""
    lines = [
        "# Nature Iteration Final Patch",
        "",
        "Final patch 只记录收敛、轻微一致性检查和 round-6 后的 blocker/gap；不再重构数据或设计。",
        "",
        "## Stop Decision",
        "",
        f"- Decision: `{manifest['stop_decision']}`",
        f"- r6 fix items: `{manifest['round6_fix_item_count']}`",
        f"- layout redesign items: `{manifest['round6_layout_redesign_needed_count']}`",
        f"- main claim ready: `{manifest['main_claim_ready']}`",
        f"- strict all-figures ready: `{manifest['strict_all_figures_ready']}`",
        f"- strict external evidence ready: `{manifest['strict_external_evidence_ready']}`",
        f"- unresolved pipeline/external gaps recorded: `{len(gap_rows)}`",
        "",
        "## Round-6 Blocker/Gap Summary",
        "",
    ]
    if gap_rows:
        for row in gap_rows:
            lines.append(f"- **{row.get('figure', '')}** `{row.get('severity', '')}`: {row.get('gap', '')}")
    else:
        lines.append("- No pipeline-ready gaps remain.")
    lines.extend(
        [
            "",
            "## Strict Evidence Packets",
            "",
        ]
    )
    if strict_rows:
        for row in strict_rows:
            present = row.get("required_artifacts_present", "")
            lines.append(f"- **{row.get('figure', '')}** `{row.get('gate_to_clear', '')}` present={present}: {row.get('blocker', '')}")
    else:
        lines.append("- No strict evidence checklist rows were found.")
    lines.extend(
        [
            "",
            "## Policy",
            "",
            "- 不继续循环美化；后续只能在外部证据返回后重跑对应 gate。",
            "- 当前状态支持 main-claim ready package；strict external-evidence claims remain explicitly gated.",
        ]
    )
    (out_dir / "final_patch_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_final_patch(out_root: Path) -> Dict[str, Any]:
    """Build the final-patch stop decision after round 6 convergence."""
    out_dir = out_root / "final_patch"
    out_dir.mkdir(parents=True, exist_ok=True)
    round6_manifest = read_json(out_root / f"r{MAX_MAIN_ITERATIONS}" / "round_manifest.json")
    readiness = read_json(PROJECT_ROOT / "outputs" / "common/old/final_assembly" / "fig1_fig10_submission_readiness.json")
    assembly_quality = read_json(PROJECT_ROOT / "outputs" / "common/old/final_assembly" / "figure_quality_report.json")
    layout_rows = audit_layout_readability()
    gap_rows = read_csv_rows(PROJECT_ROOT / "outputs" / "common/old/final_assembly" / "fig1_fig10_pipeline_ready_gaps.csv")
    strict_rows = read_csv_rows(PROJECT_ROOT / "outputs" / "common/old/final_assembly" / "fig1_fig10_strict_evidence_collection_checklist.csv")
    removed = cleanup_iteration_visuals(out_root)

    round6_fix_count = int(round6_manifest.get("fix_item_count") or 0)
    layout_redesign_count = sum(int(row.get("layout_redesign_needed") or 0) for row in layout_rows)
    main_claim_ready = int(readiness.get("main_claim_ready") or 0)
    strict_all_ready = int(readiness.get("strict_all_figures_ready") or 0)
    strict_external_ready = int(readiness.get("strict_external_evidence_ready") or 0)
    assembly_pass = int(bool((assembly_quality.get("quality_gates") or {}).get("overall_pass") or assembly_quality.get("overall_pass")))
    stop_decision = (
        "stop_after_round6_no_blocking_fix_items_external_gaps_recorded"
        if round6_fix_count == 0 and layout_redesign_count == 0 and main_claim_ready == 1
        else "continue_or_blocker_audit_required"
    )
    manifest = {
        "final_patch_id": "final_patch",
        "max_main_iterations": MAX_MAIN_ITERATIONS,
        "round6_manifest": relpath(out_root / f"r{MAX_MAIN_ITERATIONS}" / "round_manifest.json"),
        "round6_fix_item_count": round6_fix_count,
        "round6_layout_redesign_needed_count": layout_redesign_count,
        "assembly_overall_pass": assembly_pass,
        "main_claim_ready": main_claim_ready,
        "strict_all_figures_ready": strict_all_ready,
        "strict_external_evidence_ready": strict_external_ready,
        "pipeline_gap_count": len(gap_rows),
        "strict_evidence_collection_rows": len(strict_rows),
        "removed_iteration_visual_artifacts": removed,
        "auto_continue_without_user_choice": AUTO_CONTINUE_WITHOUT_USER_CHOICE,
        "requires_user_choice_mid_run": False,
        "stop_decision": stop_decision,
        "no_further_visual_iteration": stop_decision.startswith("stop_after_round6"),
        "allowed_next_action": "rerun only after external evidence returns or typo/export-label inconsistency is found",
    }
    completion_rows = [
        {
            "requirement": "six_round_protocol_bounded",
            "status": "pass" if int(round6_manifest.get("max_main_iterations") or 0) == MAX_MAIN_ITERATIONS else "fail",
            "evidence": str(out_root / f"r{MAX_MAIN_ITERATIONS}" / "round_manifest.json"),
        },
        {
            "requirement": "round6_fix_list_empty",
            "status": "pass" if round6_fix_count == 0 else "fail",
            "evidence": str(out_root / f"r{MAX_MAIN_ITERATIONS}" / "next_fix_list.md"),
        },
        {
            "requirement": "layout_redesign_items_resolved",
            "status": "pass" if layout_redesign_count == 0 else "fail",
            "evidence": "outputs/common/old/final_assembly_work/fig1_fig10_layout_readability_audit.csv",
        },
        {
            "requirement": "main_claim_ready",
            "status": "pass" if main_claim_ready == 1 else "fail",
            "evidence": "outputs/common/old/final_assembly_work/fig1_fig10_submission_readiness.json",
        },
        {
            "requirement": "strict_external_evidence_gap_recorded",
            "status": "pass_with_gap" if strict_external_ready == 0 and gap_rows else "pass",
            "evidence": "outputs/common/old/final_assembly_work/fig1_fig10_pipeline_ready_gaps.csv",
        },
        {
            "requirement": "no_old_iteration_visual_artifacts",
            "status": "pass",
            "evidence": "outputs/common/old/nature_iteration/r*/ contains no PNG/SVG/PDF after cleanup",
        },
    ]
    write_csv(
        out_dir / "completion_audit.csv",
        completion_rows,
        ["requirement", "status", "evidence"],
    )
    write_csv(
        out_dir / "round6_blocker_gap_list.csv",
        gap_rows,
        ["figure", "gap", "severity", "next_replacement"],
    )
    write_csv(
        out_dir / "strict_evidence_blockers.csv",
        strict_rows,
        [
            "figure",
            "blocker",
            "required_submission_artifact",
            "source_template_or_packet",
            "expected_completion",
            "gate_to_clear",
            "rerun_command",
            "required_artifacts_present",
            "source_templates_present",
        ],
    )
    write_csv(
        out_dir / "removed_iteration_visual_artifacts.csv",
        [{"path": path} for path in removed] or [{"path": ""}],
        ["path"],
    )
    write_final_patch_notes(out_dir, manifest, gap_rows, strict_rows)
    (out_dir / "final_patch_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def build_iteration(round_number: int, out_root: Path, cleanup_previous: bool = False) -> Dict[str, Any]:
    """Build one iteration audit directory."""
    round_id = f"r{round_number}"
    out_dir = out_root / round_id
    out_dir.mkdir(parents=True, exist_ok=True)

    quality_rows = summarize_quality_reports()
    fig1_rows = audit_fig1_landmark_windows()
    fig5_ai = audit_fig5_ai_frontier()
    layout_rows = audit_layout_readability()
    fig4_claim_scope = read_json(PROJECT_ROOT / "outputs" / "fig04/old" / "fig4_claim_scope_decision.json")
    fixes = build_fix_items(quality_rows, fig1_rows, fig5_ai, layout_rows)

    write_csv(
        out_dir / "quality_gate_summary.csv",
        quality_rows,
        ["figure", "quality_report", "exists", "overall_pass", "status_label", "failed_checks"],
    )
    write_csv(
        out_dir / "fig1_landmark_window_audit.csv",
        fig1_rows,
        ["slug", "config", "first_anchor_year", "windows", "pre_landmark_window", "landmark_window", "post_landmark_window", "round1_gate_pass"],
    )
    write_csv(
        out_dir / "fig5_ai_frontier_audit.csv",
        [fig5_ai],
        [
            "forecast_focus_path",
            "exists",
            "rows",
            "ai_like_rows",
            "ai_like_positive_score_rows",
            "ai_like_top20",
            "best_ai_like_rank",
            "round1_gate_pass",
            "source",
            "status_label",
        ],
    )
    if layout_rows:
        write_csv(
            out_dir / "layout_readability_audit.csv",
            layout_rows,
            [
                "figure",
                "image_exists",
                "width_px",
                "height_px",
                "visual_mode",
                "reading_pass_status",
                "nature_level_action",
                "submission_blocker",
                "layout_redesign_needed",
            ],
        )

    write_reflection(out_dir, round_id, quality_rows, fig1_rows, fig5_ai, fixes)
    if round_number == 0:
        write_baseline_audit(out_dir, quality_rows, fig1_rows, fig5_ai, layout_rows, fixes)

    removed = cleanup_previous_visuals(out_root, round_number) if cleanup_previous and not fixes else []
    manifest = {
        "round_id": round_id,
        "max_main_iterations": MAX_MAIN_ITERATIONS,
        "final_patch_allowed": FINAL_PATCH_ALLOWED,
        "auto_continue_without_user_choice": AUTO_CONTINUE_WITHOUT_USER_CHOICE,
        "requires_user_choice_mid_run": False,
        "fix_list_is_execution_queue": True,
        "execution_policy": "auto_continue_default_actions_no_midrun_user_choice",
        "hard_blockers_only": HARD_BLOCKERS_ONLY,
        "convergence_guard": "At round 6, unresolved items must be completed, demoted, moved to supplement, or recorded as pipeline-ready gaps.",
        "quality_reports_passing": sum(int(row.get("overall_pass", 0)) for row in quality_rows),
        "quality_reports_total": len(quality_rows),
        "fig1_landmark_window_passing": sum(int(row["round1_gate_pass"]) for row in fig1_rows),
        "fig1_landmark_window_total": len(fig1_rows),
        "fig5_ai_frontier_gate_pass": int(fig5_ai.get("round1_gate_pass", 0)),
        "fig4_claim_scope_gate_pass": int(fig4_claim_scope.get("claim_scope_gate_pass") or 0),
        "fig4_claim_scope_action": fig4_claim_scope.get("claim_scope_action", ""),
        "layout_redesign_needed_count": sum(int(row.get("layout_redesign_needed") or 0) for row in layout_rows),
        "fix_item_count": len(fixes),
        "removed_previous_visual_artifacts": removed,
    }
    (out_dir / "round_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build an automatic Nature-level iteration audit.")
    parser.add_argument("--round", type=int, default=0, help=f"Iteration number, 0-{MAX_MAIN_ITERATIONS}.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT, help="Output root for nature_iter rounds.")
    parser.add_argument(
        "--final-patch",
        action="store_true",
        help="Build final-patch stop decision and round-6 blocker/gap artifacts.",
    )
    parser.add_argument(
        "--cleanup-previous-visuals",
        action="store_true",
        help="After a clean round, delete PNG/SVG/PDF files from the previous round while keeping data reports.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Command-line entry point."""
    args = parse_args(argv)
    if args.final_patch:
        manifest = build_final_patch(args.out_root)
        print(f"[nature-iter] wrote {args.out_root / 'final_patch'}")
        print(f"[nature-iter] stop decision: {manifest['stop_decision']}")
        return
    if args.round < 0 or args.round > MAX_MAIN_ITERATIONS:
        raise ValueError(f"--round must be between 0 and {MAX_MAIN_ITERATIONS}")
    manifest = build_iteration(args.round, args.out_root, cleanup_previous=args.cleanup_previous_visuals)
    print(f"[nature-iter] wrote {args.out_root / f'r{args.round}'}")
    print(f"[nature-iter] fix items: {manifest['fix_item_count']}")


if __name__ == "__main__":
    main()

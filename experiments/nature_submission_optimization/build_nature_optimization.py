"""Build a Nature-submission optimization package from the Fig.1-Fig.10 audit."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT_DIR = PROJECT_ROOT / "outputs" / "nature_submission_audit" / "iteration_latest"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "outputs" / "nature_submission_optimization"

FIG_IDS = [f"fig{idx}" for idx in range(1, 11)]
FIG_LABELS = {f"fig{idx}": f"Fig.{idx}" for idx in range(1, 11)}
SEVERITY_RANK = {"P0_blocker": 4, "P1_major": 3, "P2_moderate": 2, "P3_minor": 1}

CAPTION_REPLACEMENTS: Mapping[str, str] = {
    "fig1": "Fig. 1. Representative graph-perturbation examples define observable citation/topic structures that motivate the quantitative indicators tested in Fig.2-Fig.3.",
    "fig2": "Fig. 2. Empirical indicator panels test controlled associations between publication-day graph structure and later scientific signal, with future-outcome and reference-closure caveats retained.",
    "fig3": "Fig. 3. A learned graph-prior score combines seven perturbation indicators and is validated as a predictive ranking proxy, not as a mechanistic ground truth.",
    "fig4": "Fig. 4. Peer-review validation measures partial structured alignment between graph-derived evidence and transparent peer-review concerns, reporting strict and soft matching separately.",
    "fig5": "Fig. 5. Forecast and mechanism handoff converts graph evidence into candidate opportunities and mechanisms; generated imagery is illustrative and not numeric evidence.",
    "fig6": "Fig. 6. Cached robustness and boundary-condition stress tests show where graph-perturbation analysis is stable under deterministic score-table probes plus cache-level indicator reruns; online full graph-extraction reruns remain pipeline-ready.",
    "fig7": "Fig. 7. Venue-family contribution is compared under field-year controls. Nature Portfolio has the top aggregate VCI point estimate in this corpus, while strict interval separation, pairwise aggregate-difference uncertainty, and per-paper-intensity caveats remain audited.",
    "fig8": "Fig. 8. ASPR architecture combines an ASPR graph agent, ASPR-Qwen, fusion, and verifier modules; this panel defines the system rather than providing performance evidence.",
    "fig9": "Fig. 9. One auditable Nature Communications case run traces manuscript evidence, graph-agent output, an explicitly assumed ASPR-Qwen draft, fusion, verification, and final review structure.",
    "fig10": "Fig. 10. Pipeline-ready ablation evidence links ASPR quality to module composition with a major caveat: the real qwen3 generic baseline looks strong under proxy scoring, but the completed 48/50 evaluable-case same-rubric Fig.4 matcher audit, with two zero-peer-point exclusions documented, shows near-zero peer-review semantic coverage; superiority claims remain blocked by module-rerun, human-preference, and checkpoint gates.",
}

CLAIM_BOUNDARIES: Mapping[str, Mapping[str, str]] = {
    "fig1": {
        "safe_claim": "代表性 graph-perturbation 结构可以作为后续定量指标的直觉入口。",
        "forbidden_claim": "这些领域示例已经证明普适的 graph-innovation law。",
        "next_evidence": "补充领域选择规则、landmark 选择时间点和 selection-time leakage 审计。",
    },
    "fig2": {
        "safe_claim": "在控制项下，publication-day graph indicators 与后续科学信号存在可测关联。",
        "forbidden_claim": "publication-day graph 可以确定性预测未来重要性。",
        "next_evidence": "加入非引用 outcome、加强 reference-closure supplement，并保留多重比较/匹配控制说明。",
    },
    "fig3": {
        "safe_claim": "learned multi-indicator score 是 graph prior / ranking proxy。",
        "forbidden_claim": "学习权重就是创新机制的精确公式。",
        "next_evidence": "补充 leave-one-domain、fold stability 和权重稳定性叙事。",
    },
    "fig4": {
        "safe_claim": "ASPR-style evidence 与 transparent peer-review concerns 存在部分结构化对齐。",
        "forbidden_claim": "semantic matching 已经完整复现人类 peer review。",
        "next_evidence": "突出 strict recall、no-match examples、false-positive 风险，并在投稿前做高 DPI 重绘。",
    },
    "fig5": {
        "safe_claim": "graph evidence 可以被转译为候选机制和 forecast handoff。",
        "forbidden_claim": "生成图像证明了未来发现或机制真实性。",
        "next_evidence": "增加 source note、冻结 prompt/seed，并保留每个视觉 claim 对应的来源 CSV 行。",
    },
    "fig6": {
        "safe_claim": "cached/proxy stress tests 与 cache-level indicator reruns 可用于筛查稳健性和边界条件。",
        "forbidden_claim": "Panels B-D 已证明完整 graph-rerun robustness。",
        "next_evidence": "用 OpenAlex retrieval 和完整 graph extraction 在扰动条件下重跑，替换 cache-level/proxy 层。",
    },
    "fig7": {
        "safe_claim": "Nature Portfolio 在当前 field-year controlled corpus 中具有最高 aggregate VCI 点估计。",
        "forbidden_claim": "Nature 因果性地产生创新，或严格支配所有 venue family。",
        "next_evidence": "补足 strict interval separation、pairwise aggregate-difference、matched controls 和 per-paper intensity sensitivity。",
    },
    "fig8": {
        "safe_claim": "ASPR architecture 定义 graph-agent、ASPR-Qwen、fusion 和 verifier 模块。",
        "forbidden_claim": "architecture diagram 本身证明 ASPR 性能。",
        "next_evidence": "把每个模块连接到 Fig.9/Fig.10 的证据，并完成 journal column width 下的可读性 QA。",
    },
    "fig9": {
        "safe_claim": "单个 auditable case 展示可追踪 pipeline behavior。",
        "forbidden_claim": "单个 case 证明 ASPR 代表性性能或 checkpoint performance。",
        "next_evidence": "用真实 checkpoint-generated ASPR-Qwen 输出替换 assumed lane，并增加更多 case。",
    },
    "fig10": {
        "safe_claim": "带 provenance gates 的 pipeline-ready ablation evidence 可支持模块诊断和 metric-sensitivity diagnosis，但不能证明 ASPR 优于 generic LLM。",
        "forbidden_claim": "LLM-as-judge 估计、proxy-scored qwen3 结果或 48/50 evaluable-complete same-rubric audit 单独证明 ASPR superiority、真实模块因果重跑或 checkpoint performance。",
        "next_evidence": "保持 same-rubric manifest 与 exclusion table 冻结，下一步完成真实 module reruns、盲评 human preferences 和 checkpoint-generated ASPR-Qwen outputs。",
    },
}


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def severity_value(severity: str) -> int:
    return SEVERITY_RANK.get(severity, 0)


def classify_action(row: Mapping[str, str]) -> str:
    lane = row.get("lane", "")
    stop_condition = row.get("stop_condition", "")
    if "pipeline-ready" in lane or "pipeline_ready_gap" in stop_condition:
        return "pipeline_ready_gap"
    if "supplement" in lane or "supplement" in stop_condition:
        return "supplement_only"
    if "caption" in lane or "caveat" in lane or "caveat" in stop_condition:
        return "caption_caveat"
    return "fix_now"


def figure_issue_summary(decisions: Sequence[Mapping[str, str]]) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {
        fig_id: {
            "fig_id": fig_id,
            "label": FIG_LABELS[fig_id],
            "issue_count": 0,
            "p1_or_higher_count": 0,
            "pipeline_gap_count": 0,
            "fix_now_count": 0,
            "caption_caveat_count": 0,
            "max_severity": "",
            "primary_action": "keep_with_current_caveats",
        }
        for fig_id in FIG_IDS
    }
    for row in decisions:
        fig_id = row.get("fig_id", "")
        if fig_id not in summary:
            continue
        item = summary[fig_id]
        action_class = classify_action(row)
        item["issue_count"] += 1
        if severity_value(row.get("severity", "")) >= severity_value("P1_major"):
            item["p1_or_higher_count"] += 1
        if action_class == "pipeline_ready_gap":
            item["pipeline_gap_count"] += 1
        if action_class == "fix_now":
            item["fix_now_count"] += 1
        if action_class == "caption_caveat":
            item["caption_caveat_count"] += 1
        if severity_value(row.get("severity", "")) > severity_value(str(item["max_severity"])):
            item["max_severity"] = row.get("severity", "")
        if item["pipeline_gap_count"]:
            item["primary_action"] = "keep_as_pipeline_ready_until_evidence_replaced"
        elif item["fix_now_count"]:
            item["primary_action"] = "fix_before_next_main_figure_export"
        elif item["caption_caveat_count"]:
            item["primary_action"] = "caption_softening_required"
    return summary


def append_readiness(summary: Dict[str, Dict[str, Any]], readiness_rows: Sequence[Mapping[str, str]]) -> None:
    for row in readiness_rows:
        if row.get("dimension") != "overall_mean":
            continue
        fig_id = row.get("fig_id", "")
        if fig_id in summary:
            summary[fig_id]["readiness_score_0_5"] = row.get("score_0_5", "")
            summary[fig_id]["readiness_status"] = row.get("status", "")


def build_backlog(decisions: Sequence[Mapping[str, str]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in decisions:
        fig_id = row.get("fig_id", "")
        if fig_id not in {"fig6", "fig7", "fig8", "fig9", "fig10"}:
            continue
        action_class = classify_action(row)
        rows.append(
            {
                "fig_id": fig_id,
                "label": row.get("label", FIG_LABELS.get(fig_id, "")),
                "issue_id": row.get("issue_id", ""),
                "severity": row.get("severity", ""),
                "action_class": action_class,
                "issue": row.get("issue", ""),
                "required_evidence_or_edit": row.get("action", ""),
                "nature_submission_risk": risk_phrase(action_class),
                "next_experiment": CLAIM_BOUNDARIES.get(fig_id, {}).get("next_evidence", ""),
            }
        )
    rows.sort(key=lambda item: (severity_value(str(item["severity"])), item["action_class"] == "pipeline_ready_gap"), reverse=True)
    return rows


def risk_phrase(action_class: str) -> str:
    phrases = {
        "pipeline_ready_gap": "主文可展示，但强 claim 必须等真实证据替换后才可写入摘要/结果主结论。",
        "caption_caveat": "需要在 caption 和结果段落中显式降调，避免审稿人按强因果或强性能证据解读。",
        "fix_now": "下一次主图导出前应直接修复视觉或元数据问题。",
        "supplement_only": "主文不应承载该 claim，可移至 supplement 或方法限制。",
    }
    return phrases.get(action_class, "保留当前 caveat。")


def write_caption_drafts(path: Path, caption_rows: Sequence[Mapping[str, str]]) -> None:
    edits = {row.get("fig_id", ""): row for row in caption_rows}
    lines = ["# Fig.1-Fig.10 Nature 投稿安全 Caption 替换草稿", ""]
    for fig_id in FIG_IDS:
        edit = edits.get(fig_id, {})
        lines.extend(
            [
                f"## {FIG_LABELS[fig_id]}",
                "",
                f"- 审计建议：{edit.get('recommended_edit', '保留现有 caveat。')}",
                f"- Nature 安全英文 caption 草稿： {CAPTION_REPLACEMENTS[fig_id]}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_claim_boundaries(path: Path) -> None:
    lines = ["# Fig.1-Fig.10 投稿 claim 边界", ""]
    for fig_id in FIG_IDS:
        item = CLAIM_BOUNDARIES[fig_id]
        lines.extend(
            [
                f"## {FIG_LABELS[fig_id]}",
                "",
                f"- 可写 claim：{item['safe_claim']}",
                f"- 不能写：{item['forbidden_claim']}",
                f"- 下一步证据：{item['next_evidence']}",
                "",
            ]
        )
    lines.append("特别提醒：Fig.10 不能写成真实模块因果重跑或 ASPR 优于 generic LLM；当前 qwen3 baseline 已实跑，proxy composite 一度高于 full ASPR，但已完成的 48/50 可评估样本 same-rubric Fig.4 matcher（另有 2 个 zero-peer-point exclusion）显示其几乎不覆盖真实 peer-review 语义点。因此这只能作为指标敏感性和模块诊断证据。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_upgrade_protocol(path: Path, backlog: Sequence[Mapping[str, Any]]) -> None:
    pipeline_items = [row for row in backlog if row.get("action_class") == "pipeline_ready_gap"]
    lines = [
        "# 下一轮实验升级协议",
        "",
        "## 优先级",
        "",
        "1. 先替换 pipeline-ready gap，不把估计结果包装成完成实验。",
        "2. 再处理 caption caveat，确保主文 claim 与证据强度匹配。",
        "3. 最后做视觉 polish、DPI、panel 排序和 supplement 搬运。",
        "4. 整个长期循环最多 6 轮主迭代 + 1 轮最终小修；每轮必须减少风险、替换证据或明确降级，不能无限重做。",
        "",
        "## Pipeline-ready gap 队列",
        "",
    ]
    for row in pipeline_items:
        lines.append(f"- {row['label']} `{row['issue_id']}`：{row['next_experiment']}")
    lines.extend(
        [
            "",
            "## 完成门槛",
            "",
            "- 每个 pipeline-ready gap 必须被真实实验替换、降级到 caption caveat、移至 supplement，或继续保留并写明不能支持强 claim。",
            "- 若接近第 6 轮仍未解决，必须固定为 supplement 或 pipeline-ready limitation，不得继续无限循环。",
            "- 重跑 `experiments/nature_submission_audit/build_nature_submission_audit.py --iteration-id latest`。",
            "- 重跑本优化脚本并确认 `verification_log.md` 总体通过。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_verification_log(out_dir: Path, quality: Mapping[str, Any]) -> None:
    lines = [
        "# Nature 投稿优化验证日志",
        "",
        f"- 输出目录：`{out_dir}`",
        f"- 总体通过：`{quality['overall_pass']}`",
        f"- Fig.6-Fig.10 backlog 数：`{quality['fig6_fig10_backlog_rows']}`",
        f"- Pipeline-ready gap 数：`{quality['pipeline_gap_rows']}`",
        "",
        "## 复现命令",
        "",
        "```bash",
        "python3 experiments/nature_submission_optimization/build_nature_optimization.py --iteration-id latest",
        "python3 -m unittest tests.test_nature_submission_optimization -v",
        "git diff --check",
        "```",
    ]
    path = out_dir / "verification_log.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_optimization(
    out_root: Path = DEFAULT_OUT_ROOT,
    iteration_id: Optional[str] = None,
    audit_dir: Path = DEFAULT_AUDIT_DIR,
) -> Dict[str, Any]:
    if iteration_id is None:
        iteration_id = dt.datetime.now().strftime("iteration_%Y%m%d_%H%M")
    elif iteration_id == "latest":
        iteration_id = "iteration_latest"
    elif not iteration_id.startswith("iteration_"):
        iteration_id = f"iteration_{iteration_id}"
    out_dir = out_root / iteration_id
    out_dir.mkdir(parents=True, exist_ok=True)

    decisions = read_csv_rows(audit_dir / "iteration_decision_board.csv")
    captions = read_csv_rows(audit_dir / "caption_edits.csv")
    readiness = read_csv_rows(audit_dir / "nature_readiness_scorecard.csv")

    summary = figure_issue_summary(decisions)
    append_readiness(summary, readiness)
    status_rows = [summary[fig_id] for fig_id in FIG_IDS]
    backlog = build_backlog(decisions)
    pipeline_gap_count = sum(1 for row in backlog if row["action_class"] == "pipeline_ready_gap")

    write_csv_rows(out_dir / "figure_optimization_status.csv", status_rows)
    write_csv_rows(out_dir / "nature_submission_decision_board.csv", [{**row, "action_class": classify_action(row)} for row in decisions])
    write_csv_rows(out_dir / "fig6_fig10_priority_backlog.csv", backlog)
    write_caption_drafts(out_dir / "caption_replacement_drafts.md", captions)
    write_claim_boundaries(out_dir / "submission_claim_boundaries.md")
    write_upgrade_protocol(out_dir / "experiment_upgrade_protocol.md", backlog)

    quality = {
        "checks": {
            "audit_decision_board_present": int(bool(decisions)),
            "ten_figures_statused": int(len(status_rows) == 10),
            "fig6_fig10_backlog_present": int(len(backlog) >= 5),
            "caption_drafts_present": int((out_dir / "caption_replacement_drafts.md").exists()),
            "claim_boundaries_present": int((out_dir / "submission_claim_boundaries.md").exists()),
            "upgrade_protocol_present": int((out_dir / "experiment_upgrade_protocol.md").exists()),
            "pipeline_gaps_preserved": int(pipeline_gap_count >= 4),
        },
        "fig6_fig10_backlog_rows": len(backlog),
        "pipeline_gap_rows": pipeline_gap_count,
    }
    quality["overall_pass"] = bool(all(quality["checks"].values()))
    write_json(out_dir / "figure_quality_report.json", quality)
    write_json(
        out_dir / "run_manifest.json",
        {
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "audit_dir": str(audit_dir),
            "out_dir": str(out_dir),
            "iteration_id": iteration_id,
            "quality": quality,
        },
    )
    write_verification_log(out_dir, quality)
    return {"out_dir": str(out_dir), "iteration_id": iteration_id, "quality": quality}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--iteration-id", type=str, default=None)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    result = build_optimization(args.out_root, args.iteration_id, args.audit_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])

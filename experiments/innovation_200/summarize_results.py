#!/usr/bin/env python3
"""Summarize already-produced outputs without running new analyses or judges."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.innovation_200.common import (
    read_jsonl,
    setup_stage_logging,
    write_json,
)
from experiments.innovation_200.contracts import SYSTEMS
from experiments.innovation_200.blinding import (
    summary_evaluation_root,
    summary_human_root,
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def human_rows(study: Path, papers: list[dict]) -> list[dict[str, Any]]:
    rows = []
    for paper in papers:
        paper_id = str(paper["paper_id"])
        for system in SYSTEMS:
            path = summary_human_root(study) / system / f"{paper_id}.json"
            if not path.exists():
                continue
            payload = _json(path)
            metrics = payload["metrics"]
            rows.append(
                {
                    "paper_id": paper_id,
                    "field_name": paper["field_name"],
                    "journal_name": paper["journal_name"],
                    "system": system,
                    **{
                        key: value
                        for key, value in metrics.items()
                        if key != "stance_by_dimension"
                    },
                    "stance_by_dimension": json.dumps(
                        metrics["stance_by_dimension"], ensure_ascii=False
                    ),
                }
            )
    return rows


def macro(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "coverage_full_rate",
        "stance_agreement_rate",
        "found_and_agreed_rate",
        "contradiction_rate",
    )
    output = []
    for system in SYSTEMS:
        group = [row for row in rows if row["system"] == system]
        summary: dict[str, Any] = {"system": system, "papers": len(group)}
        for metric in metrics:
            values = [
                float(row[metric]) for row in group if row.get(metric) is not None
            ]
            summary[metric] = statistics.fmean(values) if values else None
            summary[f"{metric}_denominator_papers"] = len(values)
        output.append(summary)
    return output


def dimension_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        payload = json.loads(str(row["stance_by_dimension"]))
        for dimension, counts in payload.items():
            totals[(str(row["system"]), dimension)].update(counts)
    output = []
    for (system, dimension), counts in sorted(totals.items()):
        matched = counts["matched"]
        output.append(
            {
                "system": system,
                "dimension": dimension,
                "references": counts["total"],
                "matched": matched,
                "agreed": counts["agreed"],
                "agreement_rate": counts["agreed"] / matched if matched else None,
            }
        )
    return output


def field_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for field in sorted({str(row["field_name"]) for row in rows}):
        for system in SYSTEMS:
            selected = [
                row
                for row in rows
                if row["field_name"] == field and row["system"] == system
            ]
            values = [
                float(row["coverage_full_rate"])
                for row in selected
                if row.get("coverage_full_rate") is not None
            ]
            output.append(
                {
                    "field_name": field,
                    "system": system,
                    "papers": len(selected),
                    "valid_human_papers": len(values),
                    "coverage_full_rate": statistics.fmean(values) if values else None,
                }
            )
    return output


def consistency_summary(
    study: Path, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    labels: dict[str, str] = {}
    for path in (summary_evaluation_root(study) / "reviewer_consistency").glob(
        "*.json"
    ):
        payload = _json(path)
        relations = {row["relation"] for row in payload["comparisons"]}
        if relations & {"opposite", "intensity_only", "uncertain"}:
            label = "disagreement"
        elif "agree" in relations:
            label = "consistent"
        else:
            label = "not_comparable"
        labels[str(payload["paper_id"])] = label
    output = []
    for label in ("consistent", "disagreement", "not_comparable"):
        for system in SYSTEMS:
            selected = [
                row
                for row in rows
                if labels.get(str(row["paper_id"]), "not_comparable") == label
                and row["system"] == system
            ]
            values = [
                float(row["coverage_full_rate"])
                for row in selected
                if row.get("coverage_full_rate") is not None
            ]
            output.append(
                {
                    "reviewer_relation": label,
                    "system": system,
                    "papers": len(selected),
                    "valid_human_papers": len(values),
                    "coverage_full_rate": statistics.fmean(values) if values else None,
                }
            )
    return output


def pairwise_rows(study: Path) -> list[dict[str, Any]]:
    return [
        _json(path)
        for path in sorted((summary_evaluation_root(study) / "pairwise").glob("*.json"))
        if not path.name.endswith((".mapping.json", ".request.json"))
    ]


def pairwise_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for baseline in (system for system in SYSTEMS if system != "fusion"):
        selected = [row for row in rows if row["baseline_system"] == baseline]
        counts = Counter(row["winners"]["overall"] for row in selected)
        output.append(
            {
                "baseline_system": baseline,
                "comparisons": len(selected),
                "fusion_wins": counts["fusion"],
                "baseline_wins": counts[baseline],
                "ties": counts["tie"],
                "fusion_win_rate_excluding_ties": (
                    counts["fusion"] / (counts["fusion"] + counts[baseline])
                    if counts["fusion"] + counts[baseline]
                    else None
                ),
            }
        )
    return output


def usage_summary(study: Path) -> dict[str, Any]:
    records = []
    current = summary_evaluation_root(study)
    paths = list((study / "status/usage").glob("**/*.jsonl"))
    if current != study:
        paths = [
            p
            for p in paths
            if p.parent.name not in {"evaluate_human", "compare_reports"}
        ]
        paths.extend((current / "status/usage").glob("**/*.jsonl"))
    for path in paths:
        stage = path.parent.name
        for row in read_jsonl(path):
            records.append({"stage": stage, **row})
    per_stage = []
    for stage in sorted({row["stage"] for row in records}):
        selected = [row for row in records if row["stage"] == stage]
        per_stage.append(
            {
                "stage": stage,
                "calls": len(selected),
                "successful_calls": sum(bool(row["success"]) for row in selected),
                "seconds": sum(float(row["seconds"]) for row in selected),
                "cache_hits": sum(bool(row["cached"]) for row in selected),
                "provider_tokens": None,
            }
        )
    return {"total_calls": len(records), "provider_tokens": None, "stages": per_stage}


def stage_summary(study: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    paths = {p.name: p for p in (study / "status").glob("*.json")}
    current = summary_evaluation_root(study)
    if current != study:
        for name in ("evaluate_human.json", "compare_reports.json"):
            paths.pop(name, None)
        paths.update({p.name: p for p in (current / "status").glob("*.json")})
    for path in sorted(paths.values()):
        payload = _json(path)
        if not isinstance(payload, list):
            continue
        counts = Counter(row.get("status") for row in payload)
        output.append(
            {
                "stage": path.stem,
                "complete": counts["complete"],
                "failed": counts["failed"],
                "elapsed_seconds": sum(
                    float(row.get("elapsed_seconds", 0)) for row in payload
                ),
            }
        )
    return output


def output_completeness(study: Path, papers: list[dict]) -> list[dict[str, Any]]:
    rows = []
    for paper in papers:
        paper_id = str(paper["paper_id"])
        row: dict[str, Any] = {
            "paper_id": paper_id,
            "field_name": paper["field_name"],
            "human_reference": (study / "human_refs" / f"{paper_id}.json").exists(),
            "pairwise_complete": len(
                [
                    p
                    for p in (summary_evaluation_root(study) / "pairwise").glob(
                        f"{paper_id}__*.json"
                    )
                    if not p.name.endswith((".mapping.json", ".request.json"))
                ]
            ),
        }
        for system in SYSTEMS:
            row[f"report_{system}"] = (
                study / "reports" / system / f"{paper_id}.json"
            ).exists()
            row[f"human_eval_{system}"] = (
                summary_human_root(study) / system / f"{paper_id}.json"
            ).exists()
        rows.append(row)
    return rows


def write_report(
    study: Path, papers: list[dict], human: list[dict], pairs: list[dict]
) -> None:
    human_summary = macro(human)
    pair_summary = pairwise_summary(pairs)
    stages = stage_summary(study)
    usage = usage_summary(study)
    report_count = sum(
        1
        for system in SYSTEMS
        for paper in papers
        if (study / "reports" / system / f"{paper['paper_id']}.json").exists()
    )
    lines = [
        "# 200篇全文创新分析实验汇总",
        "",
        f"抽样论文：{len(papers)}篇；最终报告：{report_count}/{len(papers) * len(SYSTEMS)}；匿名比较：{len(pairs)}/{len(papers) * 7}。",
        "",
        f"评估数据来源：`{summary_evaluation_root(study).relative_to(study)}`；只汇总该评估目录，不以旧评估填补缺项。",
        "",
        "## 人工具体贡献对比（论文等权）",
        "",
    ]
    lines += [
        "|组别|有效论文|完整覆盖率|创新立场一致率|找到且一致比例|明确矛盾率|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in human_summary:
        fmt = lambda value: "—" if value is None else f"{value:.3f}"
        lines.append(
            f"|{row['system']}|{row['coverage_full_rate_denominator_papers']}|{fmt(row['coverage_full_rate'])}|{fmt(row['stance_agreement_rate'])}|{fmt(row['found_and_agreed_rate'])}|{fmt(row['contradiction_rate'])}|"
        )
    lines += [
        "",
        "空人工参考不记零分；每个指标的有效论文数单独列出。部分匹配保留在逐篇表中。",
        "",
        "## 完整融合与其余组的匿名成对比较",
        "",
        "|对照组|有效比较|融合胜|对照胜|平局|融合胜率（去平局）|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in pair_summary:
        rate = (
            "—"
            if row["fusion_win_rate_excluding_ties"] is None
            else f"{row['fusion_win_rate_excluding_ties']:.3f}"
        )
        lines.append(
            f"|{row['baseline_system']}|{row['comparisons']}|{row['fusion_wins']}|{row['baseline_wins']}|{row['ties']}|{rate}|"
        )
    lines += [
        "",
        "该比较由与生成相同系列的Luna模型完成，只表示同一评判规则下的解释质量偏好，不等同于科学正确性或独立专家认可。",
        "",
        "## 运行记录",
        "",
        f"模型调用：{usage['total_calls']}；provider token/费用：不可用，未估算。",
        "",
    ]
    for row in stages:
        lines.append(
            f"- {row['stage']}：完成 {row['complete']}，失败 {row['failed']}，任务累计耗时 {row['elapsed_seconds']:.1f} 秒。"
        )
    (study / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        study / "summary.json",
        {
            "papers": len(papers),
            "evaluation_source": str(summary_evaluation_root(study).relative_to(study)),
            "human": human_summary,
            "pairwise": pair_summary,
            "stages": stages,
            "usage": usage,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logger = setup_stage_logging(args.study, "summarize_results", args.verbose)
    logger.info("[步骤 1/4] 读取论文、人工对比、成对比较和运行状态：%s", args.study)
    papers = read_jsonl(args.study / "papers.jsonl")
    human = human_rows(args.study, papers)
    pairs = pairwise_rows(args.study)
    logger.info(
        "[步骤 1/4] 输入：论文=%d，人工逐组记录=%d，匿名比较=%d",
        len(papers),
        len(human),
        len(pairs),
    )
    logger.info("[步骤 2/4] 计算论文等权、领域、维度和审稿人分歧汇总")
    _write_csv(args.study / "tables/paper_human_metrics.csv", human)
    _write_csv(args.study / "tables/system_human_metrics.csv", macro(human))
    _write_csv(
        args.study / "tables/stance_dimension_metrics.csv", dimension_summary(human)
    )
    _write_csv(args.study / "tables/field_system_metrics.csv", field_summary(human))
    _write_csv(
        args.study / "tables/reviewer_consistency_system_metrics.csv",
        consistency_summary(args.study, human),
    )
    _write_csv(args.study / "tables/pairwise_results.csv", pairs)
    _write_csv(args.study / "tables/pairwise_summary.csv", pairwise_summary(pairs))
    _write_csv(
        args.study / "tables/field_distribution.csv",
        [
            {"field_name": field, "papers": count}
            for field, count in sorted(
                Counter(str(row["field_name"]) for row in papers).items()
            )
        ],
    )
    _write_csv(args.study / "tables/stage_status.csv", stage_summary(args.study))
    _write_csv(
        args.study / "tables/output_completeness.csv",
        output_completeness(args.study, papers),
    )
    logger.info("[步骤 3/4] CSV表已写入：%s", args.study / "tables")
    write_report(args.study, papers, human, pairs)
    logger.info(
        "[步骤 4/4] 完成：报告=%s，结构化汇总=%s",
        args.study / "summary.md",
        args.study / "summary.json",
    )


if __name__ == "__main__":
    main()

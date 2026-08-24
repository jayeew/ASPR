#!/usr/bin/env python3
"""Run the fixed three-paper Nature revision-aware audit pilot."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.gear.revision_audit import (
    ABSOLUTE_NOVELTY_RE,
    BLIND_JUDGE_PROMPT,
    PILOT_IDS,
    RevisionAuditCase,
    RevisionAuditManifest,
    _load_human_response,
    agent_availability,
    agent_points,
    blind_package_count,
    build_blind_package,
    human_points,
    judge_package,
    load_manifest,
    preflight,
    rubric_set,
    score_pairs,
    score_rubrics,
)
from gear.config import load_config
from gear.contracts import PaperMetadata, ReviewRequest
from gear.review_contracts import StructuredReview
from gear.review_pipeline import review_paper
from gear.trace import sha256_file, sha256_value

DEFAULT_MANIFEST = Path("configs/gear/nature_revision_audit_pilot3.json")
DEFAULT_OUTPUT = Path("outputs/gear/revision_audit/nature_pilot3_esd2_rerun6")


def _progress(message: str) -> None:
    print(f"[GEAR-AUDIT] {message}", flush=True)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _config(path: Path | None):
    config = load_config(path)
    return config.model_copy(
        update={
            "codex_cli": config.codex_cli.model_copy(
                update={"model": "gpt-5.6-terra", "reasoning_effort": "medium"}
            )
        }
    )


def run_agent(
    manifest: RevisionAuditManifest, output: Path, config_path: Path | None
) -> dict[str, Any]:
    config = _config(config_path)
    rows = []
    for index, case in enumerate(manifest.cases, start=1):
        _progress(f"Agent 审稿 {index}/3 开始：{case.paper_id}")
        if case.agent_run_dir.exists() and any(case.agent_run_dir.iterdir()):
            raise RuntimeError(
                f"pilot agent output already exists and cannot be reused: {case.agent_run_dir}"
            )
        metadata = PaperMetadata.model_validate_json(
            case.metadata_path.read_text(encoding="utf-8")
        )
        request = ReviewRequest(paper_path=case.manuscript_path, metadata=metadata)
        bundle = review_paper(request, output_dir=case.agent_run_dir, config=config)
        rows.append(
            {
                "paper_id": case.paper_id,
                "status": bundle.status.value,
                "verification_passed": bundle.verification.passed,
                "output_dir": str(case.agent_run_dir),
            }
        )
        _progress(
            f"Agent 审稿 {index}/3 完成：{case.paper_id}；"
            f"状态={bundle.status.value}，验证通过={bundle.verification.passed}"
        )
    result = {
        "contract": "revision_audit_agent_run",
        "model": "gpt-5.6-terra",
        "reasoning_effort": "medium",
        "rows": rows,
    }
    _write(output / "agent_run_status.json", result)
    return result


def prepare_judges(manifest: RevisionAuditManifest, output: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for case in manifest.cases:
        health = agent_availability(case)
        if not health["available"]:
            entries.append(
                {"paper_id": case.paper_id, "kind": "skipped", "reason": health}
            )
            continue
        human, resolved, unverifiable = human_points(case)
        agent = agent_points(case)
        for kind, left in (("retained", human), ("resolved", resolved)):
            if not left or not agent:
                continue
            chunks = blind_package_count(len(left), len(agent))
            for chunk in range(chunks):
                package = build_blind_package(case, left, agent, kind=kind, chunk=chunk)
                target = (
                    output
                    / "judge_packages"
                    / case.paper_id.replace("/", "_")
                    / kind
                    / f"{chunk:03d}.json"
                )
                _write(target, package)
                entries.append(
                    {
                        "paper_id": case.paper_id,
                        "kind": kind,
                        "chunk": chunk,
                        "path": str(target),
                        "input_sha256": package["input_sha256"],
                    }
                )
        _write(
            output / "reference_counts" / f"{case.paper_id.replace('/', '_')}.json",
            {
                "retained_points": len(human),
                "resolved_points": len(resolved),
                "unverifiable_trace_count": unverifiable,
                "agent_points": len(agent),
            },
        )
    for index, case in enumerate(manifest.cases):
        if not agent_availability(case)["available"]:
            continue
        human, _, _ = human_points(case)
        other = manifest.cases[(index + 1) % len(manifest.cases)]
        if not agent_availability(other)["available"]:
            continue
        agent = agent_points(other)
        chunks = blind_package_count(len(human), len(agent))
        for chunk in range(chunks):
            package = build_blind_package(
                case, human, agent, kind="wrong_paper", chunk=chunk
            )
            target = (
                output
                / "judge_packages"
                / case.paper_id.replace("/", "_")
                / "wrong_paper"
                / f"{chunk:03d}.json"
            )
            _write(target, package)
            entries.append(
                {
                    "paper_id": case.paper_id,
                    "kind": "wrong_paper",
                    "candidate_paper_id": other.paper_id,
                    "chunk": chunk,
                    "path": str(target),
                    "input_sha256": package["input_sha256"],
                }
            )
    result = {"contract": "revision_audit_judge_packages", "packages": entries}
    _write(output / "judge_packages.json", result)
    return result


def run_judges(
    manifest: RevisionAuditManifest, output: Path, config_path: Path | None
) -> dict[str, Any]:
    config = _config(config_path)
    rows = []
    for item in _load(output / "judge_packages.json")["packages"]:
        if "path" not in item:
            rows.append(item)
            continue
        package_path = Path(item["path"])
        target = (
            output
            / "judge_responses"
            / item["paper_id"].replace("/", "_")
            / item["kind"]
            / f"{item['chunk']:03d}.json"
        )
        if (
            target.is_file()
            and _load(target).get("package", {}).get("input_sha256")
            == item["input_sha256"]
            and _load(target).get("prompt_sha256") == sha256_value(BLIND_JUDGE_PROMPT)
        ):
            rows.append({**item, "status": "reused", "path": str(target)})
            continue
        response = judge_package(_load(package_path), config)
        _write(target, response)
        rows.append({**item, "status": "completed", "path": str(target)})
    result = {
        "contract": "revision_audit_judge_run",
        "model": "gpt-5.6-terra",
        "reasoning_effort": "medium",
        "rows": rows,
    }
    _write(output / "judge_run_status.json", result)
    return result


def build_rubrics(
    manifest: RevisionAuditManifest, output: Path, config_path: Path | None
) -> dict[str, Any]:
    config = _config(config_path)
    rows = []
    for case in manifest.cases:
        target = output / "rubrics" / f"{case.paper_id.replace('/', '_')}.json"
        value = rubric_set(case, config)
        _write(target, value)
        rows.append(
            {
                "paper_id": case.paper_id,
                "path": str(target),
                "input_sha256": value["input_sha256"],
            }
        )
    result = {"contract": "revision_audit_rubrics", "rows": rows}
    _write(output / "rubrics_status.json", result)
    return result


def run_rubric_scores(
    manifest: RevisionAuditManifest, output: Path, config_path: Path | None
) -> dict[str, Any]:
    config = _config(config_path)
    rows = []
    for case in manifest.cases:
        rubric_payload = _load(
            output / "rubrics" / f"{case.paper_id.replace('/', '_')}.json"
        )["rubrics"]
        from experiments.gear.revision_audit import RubricSet

        target = output / "rubric_scores" / f"{case.paper_id.replace('/', '_')}.json"
        value = score_rubrics(case, RubricSet.model_validate(rubric_payload), config)
        _write(target, value)
        rows.append(
            {
                "paper_id": case.paper_id,
                "path": str(target),
                "input_sha256": value["input_sha256"],
            }
        )
    result = {"contract": "revision_audit_rubric_scores", "rows": rows}
    _write(output / "rubric_scores_status.json", result)
    return result


def _decisions(output: Path, paper_id: str, kind: str) -> list[dict[str, Any]]:
    root = output / "judge_responses" / paper_id.replace("/", "_") / kind
    rows = []
    for path in sorted(root.glob("*.json")):
        value = _load(path)
        mapping = value["package"]["mapping"]
        for decision in value["response"]["decisions"]:
            rows.append(
                {
                    **decision,
                    "left_id": f"L-{mapping[decision['left_id']]}",
                    "right_id": f"R-{mapping[decision['right_id']]}",
                }
            )
    return rows


def evaluate(manifest: RevisionAuditManifest, output: Path) -> dict[str, Any]:
    availability = [agent_availability(case) for case in manifest.cases]
    _write(
        output / "availability.json",
        {"contract": "revision_audit_availability", "rows": availability},
    )
    papers: list[dict[str, Any]] = []
    all_decisions: list[dict[str, Any]] = []
    negative_rows: list[dict[str, Any]] = []
    for case, health in zip(manifest.cases, availability):
        if not health["available"]:
            papers.append(
                {"paper_id": case.paper_id, "available": False, "reason": health}
            )
            continue
        human, resolved, unverifiable = human_points(case)
        agent = agent_points(case)
        retained = _decisions(output, case.paper_id, "retained")
        negative = _decisions(output, case.paper_id, "wrong_paper")
        resolved_decisions = (
            _decisions(output, case.paper_id, "resolved") if resolved else []
        )
        metrics = score_pairs(human, agent, retained)
        resurrection = (
            score_pairs(resolved, agent, resolved_decisions) if resolved else None
        )
        by_status = {}
        for status in ("persists", "partially_resolved"):
            subset = [point for point in human if point.status == status]
            by_status[status] = (
                score_pairs(subset, agent, retained)["strict"] if subset else None
            )
        major = [
            point
            for point in human
            if point.section in {"weaknesses", "questions"}
            and point.severity == "major"
        ]
        major_weaknesses = [
            point
            for point in human
            if point.section == "weaknesses" and point.severity == "major"
        ]
        major_questions = [
            point
            for point in human
            if point.section == "questions" and point.severity == "major"
        ]
        novelty_limits = [
            point
            for point in human
            if point.section == "novelty" and point.severity != "none"
        ]
        novelty = [point for point in human if point.section == "novelty"]
        human_review = human_points_review(case)
        agent_review = agent_review_object(case)
        evidence = {
            "agent_point_count": len(agent),
            "semantic_verified_rate": (
                sum(point.semantic_verified is True for point in agent) / len(agent)
                if agent
                else None
            ),
            "validated_rate": (
                sum(point.validation_status == "validated" for point in agent)
                / len(agent)
                if agent
                else None
            ),
            "novelty_external_evidence_rate": sum(
                any(key.startswith(("R:", "COV:")) for key in point.evidence_keys)
                for point in agent
                if point.section == "novelty"
            )
            / max(sum(point.section == "novelty" for point in agent), 1),
            "unsupported_absolute_novelty_count": sum(
                bool(ABSOLUTE_NOVELTY_RE.search(point.text))
                and not any(
                    key.startswith(("R:", "COV:")) for key in point.evidence_keys
                )
                for point in agent
                if point.section == "novelty"
            ),
        }
        paper = {
            "paper_id": case.paper_id,
            "available": True,
            "retained": metrics,
            "status_recall": by_status,
            "major_recall": (
                score_pairs(major, agent, retained)["strict"] if major else None
            ),
            "major_weakness_recall": (
                score_pairs(major_weaknesses, agent, retained)["strict"]
                if major_weaknesses
                else None
            ),
            "major_question_recall": (
                score_pairs(major_questions, agent, retained)["strict"]
                if major_questions
                else None
            ),
            "novelty_limitation_recall": (
                score_pairs(novelty_limits, agent, retained)["strict"]
                if novelty_limits
                else None
            ),
            "novelty_reasoning": (
                score_pairs(novelty, agent, retained) if novelty else None
            ),
            "novelty_judgment": {
                "human": human_review.novelty.judgment.value,
                "agent": agent_review.novelty.judgment.value,
                "same": human_review.novelty.judgment == agent_review.novelty.judgment,
            },
            "resolved_issue_resurrection": resurrection,
            "unverifiable_trace_count": unverifiable,
            "evidence": evidence,
        }
        other = manifest.cases[
            (list(manifest.cases).index(case) + 1) % len(manifest.cases)
        ]
        negative_metric = score_pairs(human, agent_points(other), negative)
        negative_rows.append(
            {
                "paper_id": case.paper_id,
                "candidate_paper_id": other.paper_id,
                "strict": negative_metric["strict"],
                "soft": negative_metric["soft"],
                "correct_pair_strict_f1": metrics["strict"]["f1"],
                "judge_discrimination_warning": negative_metric["strict"]["f1"]
                is not None
                and metrics["strict"]["f1"] is not None
                and negative_metric["strict"]["f1"] >= metrics["strict"]["f1"],
            }
        )
        papers.append(paper)
        all_decisions.extend({"paper_id": case.paper_id, **row} for row in retained)
    (output / "point_match_decisions.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in all_decisions
        ),
        encoding="utf-8",
    )
    (output / "paper_metrics.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in papers
        ),
        encoding="utf-8",
    )
    available = [row for row in papers if row["available"]]
    aggregate = {
        "contract": "revision_audit_metrics",
        "paper_count": len(papers),
        "available_count": len(available),
        "descriptive_only": True,
        "strict_f1_mean": _mean([row["retained"]["strict"]["f1"] for row in available]),
        "soft_f1_mean": _mean([row["retained"]["soft"]["f1"] for row in available]),
        "papers": papers,
    }
    _write(output / "aggregate_metrics.json", aggregate)
    _write(
        output / "novelty_metrics.json",
        {
            "contract": "revision_audit_novelty",
            "judgment_confusion": _novelty_confusion(available),
            "papers": [
                {
                    "paper_id": row["paper_id"],
                    "novelty_judgment": row.get("novelty_judgment"),
                    "novelty_reasoning": row.get("novelty_reasoning"),
                    "evidence": row.get("evidence"),
                }
                for row in papers
                if row["available"]
            ],
        },
    )
    _write(
        output / "revision_resolution_metrics.json",
        {
            "contract": "revision_audit_resolution",
            "papers": [
                {
                    "paper_id": row["paper_id"],
                    "status_recall": row.get("status_recall"),
                    "resolved_issue_resurrection": row.get(
                        "resolved_issue_resurrection"
                    ),
                    "unverifiable_trace_count": row.get("unverifiable_trace_count"),
                }
                for row in papers
                if row["available"]
            ],
        },
    )
    _write(
        output / "negative_control_metrics.json",
        {
            "contract": "revision_audit_wrong_paper_control",
            "descriptive_only": True,
            "rows": negative_rows,
        },
    )
    rubric_rows: list[dict[str, Any]] = []
    for case in manifest.cases:
        path = output / "rubric_scores" / f"{case.paper_id.replace('/', '_')}.json"
        if path.is_file():
            score_payload = _load(path)["scores"]
            if not isinstance(score_payload, dict):
                raise ValueError(f"invalid rubric score payload: {path}")
            rubric_rows.append({"paper_id": case.paper_id, **score_payload})
    (output / "rubric_scores.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rubric_rows
        ),
        encoding="utf-8",
    )
    for row in papers:
        _write(
            output / "case_audits" / f"{row['paper_id'].replace('/', '_')}.json", row
        )
    _write(
        output / "failure_report.json",
        {
            "contract": "revision_audit_failures",
            "failures": [row for row in availability if not row["available"]],
        },
    )
    return aggregate


def report(manifest: RevisionAuditManifest, output: Path) -> None:
    aggregate = _load(output / "aggregate_metrics.json")
    lines = [
        "# Nature 三篇 Revision-Aware 一致性 Pilot",
        "",
        "这是终稿审计、投稿日前外部检索、单次 gpt-5.6-terra（中等推理）盲裁判的描述性 pilot；不提供显著性、总体泛化或真人水平结论。",
        "",
        "## 可用性",
        "",
        f"- 合格 AI 审稿：{aggregate['available_count']}/{aggregate['paper_count']}",
        "",
        "## 原子点评一致性",
        "",
        f"- 严格 F1 的论文均值：{aggregate['strict_f1_mean']}",
        f"- 软 F1 的论文均值：{aggregate['soft_f1_mean']}",
        "",
        "## 逐篇真实结果",
        "",
    ]
    for row in aggregate["papers"]:
        lines.append(f"### {row['paper_id']}")
        if not row["available"]:
            lines.extend(
                ["", "该篇未进入条件一致性评分；详见 failure_report.json。", ""]
            )
            continue
        lines.extend(
            [
                "",
                f"- 严格：{row['retained']['strict']}",
                f"- 软：{row['retained']['soft']}",
                f"- 修订状态召回：{row['status_recall']}",
                f"- 重大弱点／重大问题／新颖性限制召回：{row['major_weakness_recall']} / {row['major_question_recall']} / {row['novelty_limitation_recall']}",
                f"- 已解决问题复活：{row['resolved_issue_resurrection']}",
                f"- 新颖性总体判断：{row['novelty_judgment']}",
                f"- 证据：{row['evidence']}",
                "",
            ]
        )
    negative = _load(output / "negative_control_metrics.json")
    lines.extend(["## 错论文负对照", ""])
    for row in negative["rows"]:
        lines.append(
            f"- {row['paper_id']} ← {row['candidate_paper_id']}：严格 F1={row['strict']['f1']}；区分度告警={row['judge_discrimination_warning']}"
        )
    lines.extend(["", "## Rubric 分项结果", ""])
    for path in sorted((output / "rubric_scores").glob("*.json")):
        payload = _load(path)["scores"]
        lines.append(f"- {path.stem}：{payload['scores']}")
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    manifest_payload = {
        "contract": "revision_audit_run_manifest",
        "pilot_ids": list(PILOT_IDS),
        "inputs_sha256": {
            str(path): sha256_file(path)
            for case in manifest.cases
            for path in (
                case.manuscript_path,
                case.metadata_path,
                case.reconstruction_dir / "package.json",
                case.reconstruction_dir / "response.json",
            )
        },
        "outputs": {
            name: sha256_file(output / name)
            for name in (
                "availability.json",
                "aggregate_metrics.json",
                "failure_report.json",
                "RESULTS.md",
            )
            if (output / name).is_file()
        },
    }
    manifest_payload["manifest_sha256"] = sha256_value(manifest_payload)
    _write(output / "run_manifest.json", manifest_payload)


def human_points_review(case: RevisionAuditCase) -> StructuredReview:
    return _load_human_response(case.reconstruction_dir / "response.json").review


def agent_review_object(case: RevisionAuditCase) -> StructuredReview:
    return StructuredReview.model_validate_json(
        (case.agent_run_dir / "review.json").read_text(encoding="utf-8")
    )


def _mean(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def _novelty_confusion(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(
        Counter(
            f"{row['novelty_judgment']['human']}->{row['novelty_judgment']['agent']}"
            for row in rows
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "preflight",
            "run-agent",
            "prepare-judges",
            "run-judges",
            "build-rubrics",
            "score-rubrics",
            "evaluate",
            "report",
            "all",
        ],
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--judge-config", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--agent-output-root",
        type=Path,
        help="Rebase per-paper agent artifacts beneath this run-specific directory.",
    )
    args = parser.parse_args()
    raw_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if raw_manifest.get("contract") == "gear_evaluation_manifest_v1":
        from experiments.gear.evaluation.runner import EvaluationRunner

        if args.judge_config is None:
            raise ValueError("unified evaluation requires --judge-config")
        command_map = {
            "run-agent": "run-clean",
            "build-rubrics": "prepare-judges",
            "score-rubrics": "run-judges",
            "evaluate": "score",
        }
        EvaluationRunner(
            manifest_path=args.manifest,
            judge_config_path=args.judge_config,
            output_dir=args.output_dir,
            resume=args.resume,
        ).run(command_map.get(args.command, args.command))
        return 0
    manifest = load_manifest(args.manifest)
    output = args.output_dir.resolve()
    if args.agent_output_root is not None:
        root = args.agent_output_root.resolve()
        manifest = manifest.model_copy(
            update={
                "cases": [
                    case.model_copy(
                        update={
                            "agent_run_dir": root
                            / "agent_runs"
                            / case.paper_id.replace("/", "_")
                        }
                    )
                    for case in manifest.cases
                ]
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    if args.command in {"preflight", "all"}:
        _progress("输入预检开始")
        _write(output / "input_audit.json", preflight(manifest))
        _progress("输入预检完成")
    if args.command in {"run-agent", "all"}:
        _progress("三篇 Agent 审稿开始")
        run_agent(manifest, output, args.config)
        _progress("三篇 Agent 审稿完成")
    if args.command in {"prepare-judges", "all"}:
        _progress("盲评包准备开始")
        prepare_judges(manifest, output)
        _progress("盲评包准备完成")
    if args.command in {"run-judges", "all"}:
        _progress("盲评匹配开始")
        run_judges(manifest, output, args.config)
        _progress("盲评匹配完成")
    if args.command in {"build-rubrics", "all"}:
        _progress("评测量表构建开始")
        build_rubrics(manifest, output, args.config)
        _progress("评测量表构建完成")
    if args.command in {"score-rubrics", "all"}:
        _progress("量表评分开始")
        run_rubric_scores(manifest, output, args.config)
        _progress("量表评分完成")
    if args.command in {"evaluate", "all"}:
        _progress("三篇结果汇总评测开始")
        evaluate(manifest, output)
        _progress("三篇结果汇总评测完成")
    if args.command in {"report", "all"}:
        _progress("报告生成开始")
        report(manifest, output)
        _progress("报告生成完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

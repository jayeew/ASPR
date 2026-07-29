from __future__ import annotations

import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
OUTPUT_DIR = ROOT / "outputs"


def load_parent_selector() -> ModuleType:
    """Load the unchanged parent selection engine."""
    selector_path = PARENT / "select_features.py"
    specification = importlib.util.spec_from_file_location(
        "aspr_frozen_feature_selector",
        selector_path,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot load selector from {selector_path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def read_json(path: Path) -> Dict[str, Any]:
    """Read one JSON object."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write deterministic JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write a union-schema CSV."""
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def manual_coding_status(path: Path) -> Dict[str, Any]:
    """Audit whether required two-reviewer candidate coding is complete."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required_fields = (
        "primary_source_doi",
        "formula_location",
        "formula_verified",
        "article_level",
        "t0_computable",
        "requires_future",
        "construct_valid",
        "data_ready",
        "decision",
    )
    candidate_ids = sorted({row["feature_id"] for row in rows})
    complete_rows = sum(
        all(str(row.get(field, "")).strip() for field in required_fields)
        for row in rows
    )
    reviewer_counts = Counter(row["feature_id"] for row in rows)
    return {
        "candidate_count": len(candidate_ids),
        "expected_review_rows": len(candidate_ids) * 2,
        "actual_review_rows": len(rows),
        "candidates_with_two_reviewers": sum(
            reviewer_counts[value] == 2 for value in candidate_ids
        ),
        "complete_review_rows": complete_rows,
        "dual_coding_complete": (
            len(rows) == len(candidate_ids) * 2
            and complete_rows == len(rows)
            and all(reviewer_counts[value] == 2 for value in candidate_ids)
        ),
    }


def flattened_decisions(
    selector: ModuleType,
    decisions: Sequence[Mapping[str, Any]],
    gate_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    """Retain parent decision columns plus v2 candidate-stage metadata."""
    rows = selector.decision_csv_rows(decisions, gate_ids)
    by_id = {str(row["feature_id"]): row for row in decisions}
    for row in rows:
        decision = by_id[str(row["feature_id"])]
        row["candidate_stage"] = decision.get(
            "candidate_stage",
            "baseline_screened_family",
        )
        row["promotion_requirements"] = "|".join(
            decision.get("promotion_requirements", [])
        )
    return rows


def report_text(summary: Mapping[str, Any]) -> str:
    """Render the answer-first v2 status report."""
    counts = summary["feature_role_counts"]
    failure_counts = summary["hard_gate_failure_counts_for_new_concepts"]
    lines = [
        "# 扩展版指标普查与固定筛选结果",
        "",
        "## 当前结论",
        "",
        (
            f"指标发现范围已从原有 {summary['baseline_feature_families']} "
            f"个家族扩展为 {summary['combined_registry_records']} 条登记记录："
            f"其中新增 {summary['new_review_mapped_concepts']} 个综述映射候选概念。"
        ),
        "",
        (
            f"固定硬门槛当前留下 {summary['selected_feature_families']} "
            f"个指标家族、{summary['selected_predictor_dimensions']} 个预测维度"
            f"和 {summary['selected_context_dimensions']} 个控制维度。"
        ),
        "",
        (
            "新增候选目前没有被直接送入训练：它们仍需完成原始论文、公式、"
            "发表时点、数据可得性和双人编码核验。这是防止综述中的一个名称"
            "被误当作已经可复现指标。"
        ),
        "",
        "## 新增候选未通过的主要门槛",
        "",
    ]
    for gate_id, count in sorted(failure_counts.items()):
        lines.append(f"- {gate_id}: {count}")
    lines.extend(
        [
            "",
            "## 流程完成状态",
            "",
            (
                f"- OpenAlex 全分页检索完成："
                f"{'是' if summary['openalex_full_retrieval_complete'] else '否'}"
            ),
            (
                f"- Crossref 年度快照全库扫描完成："
                f"{'是' if summary['crossref_snapshot_complete'] else '否'}"
            ),
            (
                f"- 新候选双人全文编码完成："
                f"{'是' if summary['manual_coding']['dual_coding_complete'] else '否'}"
            ),
            (
                f"- 可宣称系统检索与最终指标库已完成："
                f"{'是' if summary['formal_review_complete'] else '否'}"
            ),
            "",
            "最终数量不设目标值；以后新增、合并或排除都必须留下来源和决定原因。",
            "",
            "## 当前角色计数",
            "",
        ]
    )
    for role, count in sorted(counts.items()):
        lines.append(f"- {role}: {count}")
    return "\n".join(lines) + "\n"


def manifest_complete(path: Path) -> bool:
    """Return the all-query completion flag from a retrieval manifest."""
    if not path.exists():
        return False
    payload = read_json(path)
    return bool(payload.get("all_queries_complete", False))


def main() -> None:
    """Apply the unchanged fixed gates to the expanded registry."""
    required = (
        OUTPUT_DIR / "feature_registry_v2.json",
        OUTPUT_DIR / "literature_evidence_v2.json",
        OUTPUT_DIR / "indicator_promotion_template.csv",
    )
    if any(not path.exists() for path in required):
        raise RuntimeError(
            "Run build_indicator_registry_v2.py before screening."
        )
    selector = load_parent_selector()
    protocol = read_json(PARENT / "protocol.json")
    dimensions_payload = read_json(PARENT / "dimensions.json")
    evidence_payload = read_json(
        OUTPUT_DIR / "literature_evidence_v2.json"
    )
    registry = read_json(OUTPUT_DIR / "feature_registry_v2.json")
    rules = read_json(PARENT / "screening_rules.json")
    capabilities = read_json(PARENT / "data_capabilities.json")
    dimensions = dimensions_payload["dimensions"]
    evidence = evidence_payload["records"]
    features = selector.merge_feature_defaults(registry)
    selector.validate_registries(
        protocol,
        dimensions,
        evidence,
        features,
        capabilities,
    )
    decisions = selector.decide_features(features, rules)
    dimension_decisions = selector.decide_dimensions(
        dimensions,
        decisions,
        evidence,
        protocol,
    )
    training_sets = selector.build_training_sets(decisions, rules)
    selected_features = selector.selected_feature_records(decisions)
    baseline_count = 64
    new_decisions = decisions[baseline_count:]
    failure_counts: Counter[str] = Counter()
    for decision in new_decisions:
        failure_counts.update(decision["failed_gates"])
    selected_roles = {"primary", "supporting", "extended", "sensitivity"}
    role_counts = Counter(
        str(decision["final_role"]) for decision in decisions
    )
    selected_dimensions = [
        row for row in dimension_decisions if row["selected"]
    ]
    coding = manual_coding_status(
        OUTPUT_DIR / "indicator_promotion_template.csv"
    )
    openalex_complete = manifest_complete(
        OUTPUT_DIR / "openalex_manifest.json"
    )
    crossref_complete = manifest_complete(
        OUTPUT_DIR / "crossref_snapshot_manifest.json"
    )
    formal_complete = (
        openalex_complete
        and crossref_complete
        and coding["dual_coding_complete"]
        and all(
            decision.get("candidate_stage")
            != "review_mapped_pending_primary_verification"
            for decision in decisions
        )
    )
    summary = {
        "schema_version": "2.0.0",
        "selection_status": (
            "expanded_candidate_census_not_final_systematic_review"
            if not formal_complete
            else "formal_review_complete"
        ),
        "baseline_feature_families": baseline_count,
        "new_review_mapped_concepts": len(new_decisions),
        "combined_registry_records": len(decisions),
        "literature_evidence_records": len(evidence),
        "selected_feature_families": sum(
            role_counts[role] for role in selected_roles
        ),
        "selected_new_feature_families": sum(
            decision["final_role"] in selected_roles
            for decision in new_decisions
        ),
        "feature_role_counts": dict(sorted(role_counts.items())),
        "hard_gate_failure_counts_for_new_concepts": dict(
            sorted(failure_counts.items())
        ),
        "selected_predictor_dimensions": sum(
            row["block"] != "context_control"
            for row in selected_dimensions
        ),
        "selected_context_dimensions": sum(
            row["block"] == "context_control"
            for row in selected_dimensions
        ),
        "selected_dimension_ids": [
            str(row["dimension_id"]) for row in selected_dimensions
        ],
        "training_set_sizes": {
            name: len(values)
            for name, values in training_sets.items()
        },
        "manual_coding": coding,
        "openalex_full_retrieval_complete": openalex_complete,
        "crossref_snapshot_complete": crossref_complete,
        "formal_review_complete": formal_complete,
        "quantity_rule": (
            "No final dimension or feature quota; every count follows from "
            "evidence verification, hard gates, and family deduplication."
        ),
    }
    decision_rows = flattened_decisions(
        selector,
        decisions,
        list(rules["hard_gates"]),
    )
    write_csv(OUTPUT_DIR / "feature_decisions_v2.csv", decision_rows)
    write_json(
        OUTPUT_DIR / "final_dimensions_v2.json",
        {"schema_version": "2.0.0", "dimensions": dimension_decisions},
    )
    write_json(
        OUTPUT_DIR / "final_features_v2.json",
        {"schema_version": "2.0.0", "features": selected_features},
    )
    write_json(
        OUTPUT_DIR / "training_feature_sets_v2.json",
        {"schema_version": "2.0.0", "sets": training_sets},
    )
    write_json(OUTPUT_DIR / "selection_summary_v2.json", summary)
    (OUTPUT_DIR / "selection_report_v2.md").write_text(
        report_text(summary),
        encoding="utf-8",
    )
    print(
        f"Screened {len(decisions)} records; "
        f"{summary['selected_feature_families']} current families retained, "
        f"{summary['selected_new_feature_families']} from unverified additions."
    )


if __name__ == "__main__":
    main()

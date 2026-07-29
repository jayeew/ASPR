from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parent
INPUT_FILES = (
    "protocol.json",
    "search_queries.json",
    "search_snapshot.csv",
    "screening_rules.json",
    "dimensions.json",
    "literature_evidence.json",
    "metric_registry.json",
)


def load_json(path: Path) -> Dict[str, Any]:
    """Load a UTF-8 JSON object.

    Args:
        path: JSON path.

    Returns:
        Parsed JSON object.
    """
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def sha256_path(path: Path) -> str:
    """Return a stable SHA-256 digest for a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_unique(rows: Sequence[Mapping[str, Any]], field: str) -> None:
    """Require unique non-empty identifiers in a row sequence."""
    values = [str(row.get(field, "")).strip() for row in rows]
    if any(not value for value in values):
        raise ValueError(f"Missing required identifier: {field}")
    duplicates = [value for value, count in Counter(values).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate {field}: {duplicates}")


def validate_inputs(
    protocol: Mapping[str, Any],
    rules: Mapping[str, Any],
    dimensions: Sequence[Mapping[str, Any]],
    sources: Sequence[Mapping[str, Any]],
    metrics: Sequence[Mapping[str, Any]],
) -> None:
    """Validate registry referential integrity and anti-quota invariants."""
    require_unique(dimensions, "construct_id")
    require_unique(sources, "source_id")
    require_unique(metrics, "metric_id")
    dimension_ids = {row["construct_id"] for row in dimensions}
    source_ids = {row["source_id"] for row in sources}
    for source in sources:
        unknown = set(source["construct_ids"]) - dimension_ids
        if unknown:
            raise ValueError(f"{source['source_id']} has unknown constructs: {unknown}")
    for metric in metrics:
        unknown_dimensions = set(metric["construct_ids"]) - dimension_ids
        unknown_sources = set(metric["source_ids"]) - source_ids
        if unknown_dimensions or unknown_sources:
            raise ValueError(
                f"{metric['metric_id']} has unknown references: "
                f"dimensions={unknown_dimensions}, sources={unknown_sources}"
            )
    if not protocol["dimension_rules"]["no_required_dimension_count"]:
        raise ValueError("Dimension quotas are forbidden")
    if not rules["family_deduplication"]["no_per_dimension_quota"]:
        raise ValueError("Per-dimension indicator quotas are forbidden")
    provenance = protocol["data_boundary"]["local_audit_provenance"]
    provenance_path = (ROOT / provenance["path"]).resolve()
    if not provenance_path.is_file():
        raise ValueError(f"Missing local audit provenance: {provenance_path}")
    if sha256_path(provenance_path) != provenance["sha256"]:
        raise ValueError(f"Local audit provenance hash changed: {provenance_path}")


def compare_rule(actual: Any, rule: Mapping[str, Any]) -> bool:
    """Apply one declarative equality, membership, or numeric threshold rule."""
    if "allowed" in rule and actual not in rule["allowed"]:
        return False
    if "equals" in rule and actual != rule["equals"]:
        return False
    if "minimum" in rule and (actual is None or actual < rule["minimum"]):
        return False
    if "maximum" in rule and (actual is None or actual > rule["maximum"]):
        return False
    return True


def evaluate_metric_gates(
    metric: Mapping[str, Any], hard_gates: Mapping[str, Mapping[str, Any]]
) -> List[str]:
    """Return all failed hard-gate codes for one metric."""
    failures: List[str] = []
    audit = metric.get("empirical_audit") or {}
    for gate_id, rule in hard_gates.items():
        field = rule.get("field")
        audit_field = rule.get("audit_field")
        actual = metric.get(field) if field else audit.get(audit_field)
        if not compare_rule(actual, rule):
            failures.append(gate_id)
    return failures


def validation_rank(grade: str, order: Sequence[str]) -> int:
    """Return a deterministic ordinal for a validation grade."""
    try:
        return order.index(grade)
    except ValueError:
        return len(order)


def metric_sort_key(
    metric: Mapping[str, Any], validation_order: Sequence[str]
) -> Tuple[Any, ...]:
    """Return the prespecified family-competition sort key."""
    audit = metric.get("empirical_audit") or {}
    return (
        metric["selection_priority"],
        validation_rank(metric["validation_grade"], validation_order),
        -float(audit.get("minimum_domain_coverage") or -1.0),
        -float(audit.get("stability_spearman") or -1.0),
        metric["metric_id"],
    )


def select_metrics(
    metrics: Sequence[Mapping[str, Any]], rules: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    """Apply hard gates and deterministic within-family deduplication."""
    hard_gates = rules["hard_gates"]
    decisions: List[Dict[str, Any]] = []
    by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for raw_metric in metrics:
        metric = dict(raw_metric)
        failures = evaluate_metric_gates(metric, hard_gates)
        metric["failed_gates"] = failures
        metric["passed_hard_gates"] = not failures
        metric["selected"] = False
        metric["final_role"] = "excluded"
        decisions.append(metric)
        if not failures:
            by_family[metric["family_id"]].append(metric)
    policy = rules["family_deduplication"]
    validation_order = policy["validation_grade_order"]
    for family_metrics in by_family.values():
        ranked = sorted(
            family_metrics,
            key=lambda row: metric_sort_key(row, validation_order),
        )
        winner = ranked[0]
        winner["selected"] = True
        primary_grades = rules["role_assignment"]["primary_validation_grades"]
        winner["final_role"] = (
            "primary" if winner["validation_grade"] in primary_grades else "supporting"
        )
        for duplicate in ranked[1:]:
            duplicate["failed_gates"].append("G17_REDUNDANT_WITHIN_FAMILY")
            duplicate["passed_hard_gates"] = False
            duplicate["final_role"] = "excluded_redundant"
    return decisions


def eligible_source(
    source: Mapping[str, Any], construct_id: str, role: str | None = None
) -> bool:
    """Return whether a source contributes evidence to a construct."""
    if not source["peer_reviewed"] or construct_id not in source["construct_ids"]:
        return False
    if str(source["decision"]).startswith("excluded"):
        return False
    return role is None or role in source["evidence_roles"]


def dimension_evidence_counts(
    construct_id: str, sources: Sequence[Mapping[str, Any]]
) -> Tuple[int, int]:
    """Count independent peer-reviewed and operationalization groups."""
    peer_groups = {
        source["research_group"]
        for source in sources
        if eligible_source(source, construct_id)
    }
    operational_roles = {
        "operationalization",
        "mathematical_operationalization",
        "human_measurement",
    }
    operational_groups = {
        source["research_group"]
        for source in sources
        if eligible_source(source, construct_id)
        and operational_roles.intersection(source["evidence_roles"])
    }
    return len(peer_groups), len(operational_groups)


def select_dimensions(
    dimensions: Sequence[Mapping[str, Any]],
    sources: Sequence[Mapping[str, Any]],
    metric_decisions: Sequence[Mapping[str, Any]],
    rules: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Select dimensions only when concept evidence and an indicator both pass."""
    selected_by_dimension: Dict[str, List[str]] = defaultdict(list)
    for metric in metric_decisions:
        if metric["selected"]:
            for construct_id in metric["construct_ids"]:
                selected_by_dimension[construct_id].append(metric["metric_id"])
    results: List[Dict[str, Any]] = []
    for raw_dimension in dimensions:
        dimension = dict(raw_dimension)
        construct_id = dimension["construct_id"]
        peer_count, operational_count = dimension_evidence_counts(
            construct_id, sources
        )
        failures: List[str] = []
        if dimension["scope_role"] != "direct_novelty":
            failures.append("D01_NOT_DIRECT_NOVELTY")
        if peer_count < rules["minimum_independent_peer_reviewed_groups"]:
            failures.append("D02_INSUFFICIENT_INDEPENDENT_SOURCES")
        if operational_count < rules["minimum_operationalization_groups"]:
            failures.append("D03_NO_OPERATIONALIZATION_SOURCE")
        selected_metrics = sorted(selected_by_dimension.get(construct_id, []))
        conceptual_failures = [code for code in failures if code != "D04_NO_METRIC"]
        dimension["conceptually_supported"] = not conceptual_failures
        if not selected_metrics:
            failures.append("D04_NO_SELECTED_INDICATOR")
        dimension["peer_reviewed_group_count"] = peer_count
        dimension["operationalization_group_count"] = operational_count
        dimension["selected_metric_ids"] = selected_metrics
        dimension["failed_gates"] = failures
        dimension["selected"] = not failures
        results.append(dimension)
    return results


def join_values(value: Any) -> str:
    """Flatten list-like values for reviewer-facing CSV files."""
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)


def write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    """Write selected fields from mappings as UTF-8 CSV."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: join_values(row.get(field)) for field in fields})


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write deterministic human-readable JSON."""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def annotate_source_routes(
    sources: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Annotate whether evidence appeared in the frozen ranked API snapshot."""
    with (ROOT / "search_snapshot.csv").open(encoding="utf-8", newline="") as handle:
        snapshot_dois = {
            row["doi"].strip().lower()
            for row in csv.DictReader(handle)
            if row["doi"].strip()
        }
    annotated: List[Dict[str, Any]] = []
    for raw_source in sources:
        source = dict(raw_source)
        doi = str(source.get("doi") or "").lower()
        if source["source_id"] == "ASPR_V61":
            route = "internal_local_audit"
        elif doi in snapshot_dois:
            route = "ranked_api_snapshot"
        else:
            route = "anchor_review_citation_chain_or_targeted_verification"
        source["discovery_route"] = route
        annotated.append(source)
    return annotated


def failure_counts(decisions: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    """Count failed-gate occurrences across metric decisions."""
    counts: Counter[str] = Counter()
    for row in decisions:
        counts.update(row["failed_gates"])
    return dict(sorted(counts.items()))


def build_summary(
    sources: Sequence[Mapping[str, Any]],
    metrics: Sequence[Mapping[str, Any]],
    metric_decisions: Sequence[Mapping[str, Any]],
    dimension_decisions: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Build the machine-readable result summary."""
    selected_metrics = [row for row in metric_decisions if row["selected"]]
    selected_dimensions = [row for row in dimension_decisions if row["selected"]]
    conceptual_dimensions = [
        row for row in dimension_decisions if row["conceptually_supported"]
    ]
    return {
        "literature_source_count": len(sources),
        "peer_reviewed_source_count": sum(
            bool(source["peer_reviewed"]) for source in sources
        ),
        "candidate_metric_family_count": len(metrics),
        "registered_variant_count": sum(len(metric["variants"]) for metric in metrics),
        "conceptually_supported_direct_dimension_count": len(conceptual_dimensions),
        "conceptually_supported_direct_dimension_ids": [
            row["construct_id"] for row in conceptual_dimensions
        ],
        "final_operational_dimension_count": len(selected_dimensions),
        "final_operational_dimension_ids": [
            row["construct_id"] for row in selected_dimensions
        ],
        "final_indicator_count": len(selected_metrics),
        "final_indicator_ids": [row["metric_id"] for row in selected_metrics],
        "final_indicator_roles": {
            row["metric_id"]: row["final_role"] for row in selected_metrics
        },
        "metric_failure_counts": failure_counts(metric_decisions),
        "selection_used_dimension_quota": false_value(),
        "selection_used_indicator_quota": false_value(),
        "selection_used_future_outcomes": false_value(),
        "selection_used_prediction_performance": false_value(),
    }


def false_value() -> bool:
    """Return an explicit false value for audit-friendly summary fields."""
    return False


def report_table_row(metric: Mapping[str, Any]) -> str:
    """Format one selected indicator as a Markdown table row."""
    sources = ", ".join(metric["source_ids"])
    return (
        f"| `{metric['metric_id']}` | {metric['label_zh']} | "
        f"{metric['selected_variant']} | {metric['final_role']} | {sources} |"
    )


def build_report(
    summary: Mapping[str, Any],
    metric_decisions: Sequence[Mapping[str, Any]],
    dimension_decisions: Sequence[Mapping[str, Any]],
) -> str:
    """Build the concise Chinese reviewer-facing selection report."""
    selected_metrics = [row for row in metric_decisions if row["selected"]]
    selected_dimensions = [row for row in dimension_decisions if row["selected"]]
    conceptual_dimensions = [
        row for row in dimension_decisions if row["conceptually_supported"]
    ]
    excluded_highlights = {
        "Novelty U": "本地重采样稳定性未达 ρ≥0.90 且相对误差≤0.10。",
        "异常度 P10": "本地重采样稳定性未达两项固定阈值。",
        "常规性中位数": "它是组合常规性的配套统计量，不是独立的新颖性信号。",
        "元素/语义指标": "需要尚未冻结的完整文本历史、实体体系或历史嵌入模型。",
        "多样性与 Rao–Stirling": "测量跨学科知识基础，属于邻近构念，不直接证明新颖性。",
    }
    lines = [
        "# 独立创新维度与指标遴选结果",
        "",
        "## 结论",
        "",
        (
            f"文献构念层识别出 **{len(conceptual_dimensions)} 个**有充分来源支持的"
            "直接新颖性构念：知识元素新颖性与知识重组新颖性。"
        ),
        (
            f"结合当前冻结数据和全部硬门槛后，最终可操作框架保留 "
            f"**{len(selected_dimensions)} 个维度、{len(selected_metrics)} 个指标**。"
        ),
        "",
    ]
    for dimension in selected_dimensions:
        lines.append(
            f"- **{dimension['label_zh']}**（`{dimension['construct_id']}`）："
            f"{dimension['definition_zh']}"
        )
    lines.extend(
        [
            "",
            "## 最终指标",
            "",
            "| ID | 指标 | 冻结实现 | 角色 | 主要来源 |",
            "|---|---|---|---|---|",
        ]
    )
    lines.extend(report_table_row(metric) for metric in selected_metrics)
    lines.extend(
        [
            "",
            "两个指标分别捕捉“与既有论文参考配置相似程度”和“是否首次组合既有知识源”。"
            "它们属于同一上位构念的不同测量家族，不被包装成两个因果机制。",
            "",
            "## 主要未入选项",
            "",
        ]
    )
    lines.extend(
        f"- **{name}**：{reason}" for name, reason in excluded_highlights.items()
    )
    lines.extend(
        [
            "",
            "## 数量如何产生",
            "",
            (
                f"- 证据表包含 {summary['literature_source_count']} 项来源，其中 "
                f"{summary['peer_reviewed_source_count']} 项为同行评议来源。"
            ),
            (
                f"- 指标库按数学测量家族登记 {summary['candidate_metric_family_count']} 项，"
                f"另记录 {summary['registered_variant_count']} 个参数或实现变体。"
            ),
            "- 程序逐项执行来源、构念、发表时点、本地数据、覆盖率、稳定性和公式测试。",
            "- 同一数学家族最多保留一个代表；没有预设维度数、候选数或最终指标数。",
            "",
            "## 复现",
            "",
            "```bash",
            "python3 innovation_indicator_selection/select_indicators.py",
            "python3 innovation_indicator_selection/tests.py",
            "```",
            "",
            "该结果仅回答“当前数据下可以站得住脚地测量哪些发表时创新证据”，"
            "不等同于完整创新性、论文质量或未来影响力。",
            "",
        ]
    )
    return "\n".join(lines)


def materialize_outputs(
    output_dir: Path,
    sources: Sequence[Mapping[str, Any]],
    metric_decisions: Sequence[Mapping[str, Any]],
    dimension_decisions: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    """Write all reviewer-facing tables, report, and audit manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    source_rows = annotate_source_routes(sources)
    write_csv(
        output_dir / "literature_evidence_table.csv",
        source_rows,
        (
            "source_id",
            "year",
            "citation",
            "doi",
            "url",
            "peer_reviewed",
            "research_group",
            "discovery_route",
            "evidence_roles",
            "construct_ids",
            "contribution_loci",
            "decision",
            "evidence_note",
        ),
    )
    metric_rows: List[Dict[str, Any]] = []
    for metric in metric_decisions:
        audit = metric.get("empirical_audit") or {}
        row = dict(metric)
        row.update(audit)
        metric_rows.append(row)
    write_csv(
        output_dir / "metric_screening_decisions.csv",
        metric_rows,
        (
            "metric_id",
            "label_zh",
            "construct_ids",
            "contribution_locus",
            "family_id",
            "variants",
            "selected_variant",
            "formula",
            "data_modality",
            "source_ids",
            "scope_role",
            "signal_role",
            "source_fidelity",
            "local_data_status",
            "validation_grade",
            "overall_coverage",
            "minimum_domain_coverage",
            "stability_spearman",
            "stability_median_relative_error",
            "failed_gates",
            "selected",
            "final_role",
            "notes",
        ),
    )
    write_csv(
        output_dir / "dimension_screening_decisions.csv",
        dimension_decisions,
        (
            "construct_id",
            "label_zh",
            "label_en",
            "level",
            "scope_role",
            "definition",
            "definition_zh",
            "peer_reviewed_group_count",
            "operationalization_group_count",
            "conceptually_supported",
            "selected_metric_ids",
            "failed_gates",
            "selected",
        ),
    )
    selected_metrics = [row for row in metric_rows if row["selected"]]
    selected_dimensions = [row for row in dimension_decisions if row["selected"]]
    write_csv(
        output_dir / "final_indicators.csv",
        selected_metrics,
        (
            "metric_id",
            "label_zh",
            "construct_ids",
            "selected_variant",
            "formula",
            "source_ids",
            "validation_grade",
            "overall_coverage",
            "minimum_domain_coverage",
            "stability_spearman",
            "stability_median_relative_error",
            "final_role",
        ),
    )
    write_csv(
        output_dir / "final_dimensions.csv",
        selected_dimensions,
        (
            "construct_id",
            "label_zh",
            "label_en",
            "definition",
            "definition_zh",
            "peer_reviewed_group_count",
            "operationalization_group_count",
            "selected_metric_ids",
        ),
    )
    write_json(output_dir / "selection_summary.json", dict(summary))
    report = build_report(summary, metric_decisions, dimension_decisions)
    (output_dir / "selection_report.md").write_text(report, encoding="utf-8")
    output_names = (
        "literature_evidence_table.csv",
        "metric_screening_decisions.csv",
        "dimension_screening_decisions.csv",
        "final_indicators.csv",
        "final_dimensions.csv",
        "selection_summary.json",
        "selection_report.md",
    )
    manifest = {
        "protocol_id": load_json(ROOT / "protocol.json")["protocol_id"],
        "input_sha256": {
            name: sha256_path(ROOT / name) for name in INPUT_FILES
        },
        "script_sha256": sha256_path(Path(__file__).resolve()),
        "output_sha256": {
            name: sha256_path(output_dir / name) for name in output_names
        },
    }
    write_json(output_dir / "audit_manifest.json", manifest)


def run_selection(output_dir: Path | None = None) -> Dict[str, Any]:
    """Run validation, selection, and optional materialization.

    Args:
        output_dir: Output directory. When omitted, no files are written.

    Returns:
        A bundle containing decisions and the result summary.
    """
    protocol = load_json(ROOT / "protocol.json")
    rules = load_json(ROOT / "screening_rules.json")
    dimensions = load_json(ROOT / "dimensions.json")["dimensions"]
    sources = load_json(ROOT / "literature_evidence.json")["records"]
    metrics = load_json(ROOT / "metric_registry.json")["metrics"]
    validate_inputs(protocol, rules, dimensions, sources, metrics)
    metric_decisions = select_metrics(metrics, rules)
    dimension_decisions = select_dimensions(
        dimensions,
        sources,
        metric_decisions,
        protocol["dimension_rules"],
    )
    summary = build_summary(
        sources,
        metrics,
        metric_decisions,
        dimension_decisions,
    )
    if output_dir is not None:
        materialize_outputs(
            output_dir,
            sources,
            metric_decisions,
            dimension_decisions,
            summary,
        )
    return {
        "summary": summary,
        "metric_decisions": metric_decisions,
        "dimension_decisions": dimension_decisions,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Select literature-backed T0 novelty dimensions and indicators."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs",
        help="Directory for tables, report, and audit manifest.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate and select without writing output files.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the standalone indicator-selection workflow."""
    args = parse_args()
    result = run_selection(None if args.check_only else args.output_dir)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

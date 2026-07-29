from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
INPUT_FILES = (
    "protocol.json",
    "dimensions.json",
    "literature_evidence.json",
    "feature_registry.json",
    "screening_rules.json",
    "data_capabilities.json",
    "search_queries.json",
)
GENERATED_OUTPUT_FILES = (
    "literature_evidence_table.csv",
    "evidence_discovery_crosswalk.csv",
    "feature_decisions.csv",
    "final_dimensions.json",
    "final_features.json",
    "training_feature_sets.json",
    "selection_summary.json",
    "selection_report.md",
)


def read_json(path: Path) -> Dict[str, Any]:
    """Read and validate one JSON object."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write deterministic UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    """Write deterministic UTF-8 CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def merge_feature_defaults(registry: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Materialize registry defaults into every feature record."""
    defaults = dict(registry.get("candidate_defaults", {}))
    materialized: List[Dict[str, Any]] = []
    for raw_feature in registry["features"]:
        feature = dict(defaults)
        feature.update(raw_feature)
        materialized.append(feature)
    return materialized


def validate_unique(
    rows: Iterable[Mapping[str, Any]],
    key: str,
    label: str,
) -> None:
    """Require a unique nonempty identifier field."""
    values = [str(row.get(key, "")) for row in rows]
    if any(not value for value in values):
        raise ValueError(f"{label} contains an empty {key}")
    duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate {label} {key}: {duplicates}")


def validate_registries(
    protocol: Mapping[str, Any],
    dimensions: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
    features: Sequence[Mapping[str, Any]],
    capabilities: Mapping[str, Any],
) -> None:
    """Validate cross-file identifiers and frozen protocol invariants."""
    validate_unique(dimensions, "dimension_id", "dimensions")
    validate_unique(evidence, "source_id", "evidence")
    validate_unique(features, "feature_id", "features")
    dimension_ids = {str(row["dimension_id"]) for row in dimensions}
    source_ids = {str(row["source_id"]) for row in evidence}
    missing_dimensions = sorted(
        {
            str(feature["dimension_id"])
            for feature in features
            if str(feature["dimension_id"]) not in dimension_ids
        }
    )
    if missing_dimensions:
        raise ValueError(f"Unknown feature dimensions: {missing_dimensions}")
    missing_sources = sorted(
        {
            str(source_id)
            for feature in features
            for source_id in feature["source_ids"]
            if str(source_id) not in source_ids
        }
    )
    if missing_sources:
        raise ValueError(f"Unknown feature evidence sources: {missing_sources}")
    if protocol["outcome_separation"]["future_outcomes_used_to_select_features"]:
        raise ValueError("Outcome-blind selection protocol has been violated")
    if not protocol["reproducibility"]["selection_has_no_dimension_or_feature_quota"]:
        raise ValueError("A numeric selection quota is not permitted")
    materialized = capabilities["materialized_columns"]
    derivable = capabilities["derivable_features"]
    for feature in features:
        if feature["data_status"] == "materialized_audited":
            data_reference = str(feature.get("data_reference", ""))
            if "." in data_reference:
                column = data_reference.split(".", maxsplit=1)[1]
                if column not in materialized:
                    raise ValueError(
                        f"{feature['feature_id']} references unknown materialized "
                        f"column {column}"
                    )
        if (
            feature["data_status"] == "derivable_from_audited_inputs"
            and feature["name"] in {
                "reference_age_mean",
                "reference_age_cv",
                "topic_publication_volume_prior_3y",
                "topic_growth_slope_prior_5y",
            }
            and feature["name"] not in derivable
        ):
            raise ValueError(
                f"{feature['feature_id']} is absent from derivable_features"
            )


def check_gate(feature: Mapping[str, Any], rule: Mapping[str, Any]) -> bool:
    """Evaluate one declarative hard gate."""
    value = feature.get(str(rule["field"]))
    if "equals" in rule:
        return value == rule["equals"]
    if "allowed" in rule:
        return value in rule["allowed"]
    raise ValueError(f"Gate has neither equals nor allowed: {rule}")


def evaluate_gates(
    feature: Mapping[str, Any],
    hard_gates: Mapping[str, Mapping[str, Any]],
) -> Dict[str, bool]:
    """Evaluate every hard gate without short-circuiting."""
    return {
        gate_id: check_gate(feature, rule)
        for gate_id, rule in hard_gates.items()
    }


def choose_family_winners(
    eligible_features: Sequence[Mapping[str, Any]],
    rules: Mapping[str, Any],
) -> Dict[str, str]:
    """Choose one representative per redundancy family deterministically."""
    evidence_rank = {
        value: index
        for index, value in enumerate(rules["evidence_grade_order"])
    }
    readiness_rank = {
        value: index for index, value in enumerate(rules["readiness_order"])
    }
    by_family: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for feature in eligible_features:
        by_family[str(feature["redundancy_family"])].append(feature)
    winners: Dict[str, str] = {}
    for family, candidates in by_family.items():
        ordered = sorted(
            candidates,
            key=lambda feature: (
                int(feature["selection_priority"]),
                evidence_rank[str(feature["evidence_grade"])],
                readiness_rank[str(feature["data_status"])],
                str(feature["feature_id"]),
            ),
        )
        winners[family] = str(ordered[0]["feature_id"])
    return winners


def selected_role(
    feature: Mapping[str, Any],
    role_rules: Mapping[str, Sequence[str]],
) -> str:
    """Assign primary, supporting, extended, or sensitivity role."""
    if feature["bias_policy"] in role_rules["sensitivity_bias_policy"]:
        return "sensitivity"
    if feature["data_status"] in role_rules["extended_data_status"]:
        return "extended"
    if feature["evidence_grade"] in role_rules["primary_evidence_grades"]:
        return "primary"
    if feature["evidence_grade"] in role_rules["supporting_evidence_grades"]:
        return "supporting"
    raise ValueError(
        f"Eligible feature {feature['feature_id']} has no assignable role"
    )


def decide_features(
    features: Sequence[Mapping[str, Any]],
    rules: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Apply all hard gates, family deduplication, and role assignment."""
    hard_gates = rules["hard_gates"]
    evaluated: List[Dict[str, Any]] = []
    for raw_feature in features:
        feature = dict(raw_feature)
        gate_checks = evaluate_gates(feature, hard_gates)
        feature["gate_checks"] = gate_checks
        feature["failed_gates"] = sorted(
            gate_id for gate_id, passed in gate_checks.items() if not passed
        )
        evaluated.append(feature)
    eligible = [feature for feature in evaluated if not feature["failed_gates"]]
    winners = choose_family_winners(eligible, rules)
    for feature in evaluated:
        family = str(feature["redundancy_family"])
        winner_id = winners.get(family)
        feature["redundancy_winner_id"] = winner_id or ""
        if feature["failed_gates"]:
            feature["final_role"] = "excluded_hard_gate"
            feature["decision_reason"] = "failed " + ", ".join(
                feature["failed_gates"]
            )
        elif winner_id != feature["feature_id"]:
            feature["final_role"] = "excluded_redundant"
            feature["decision_reason"] = (
                f"same family as deterministic representative {winner_id}"
            )
        else:
            role = selected_role(feature, rules["role_assignment"])
            feature["final_role"] = role
            feature["decision_reason"] = (
                f"passes all hard gates; assigned {role} by frozen rules"
            )
    return sorted(evaluated, key=lambda feature: str(feature["feature_id"]))


def evidence_groups_by_dimension(
    evidence: Sequence[Mapping[str, Any]],
) -> Dict[str, List[str]]:
    """Collect independent peer-reviewed research groups per dimension."""
    groups: Dict[str, set[str]] = defaultdict(set)
    for record in evidence:
        if not record["peer_reviewed"]:
            continue
        if not record.get("dimension_admission_evidence", False):
            continue
        for dimension_id in record["dimension_ids"]:
            groups[str(dimension_id)].add(str(record["research_group"]))
    return {
        dimension_id: sorted(values)
        for dimension_id, values in groups.items()
    }


def decide_dimensions(
    dimensions: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Admit dimensions from evidence diversity and selected features."""
    selected_roles = {"primary", "supporting", "extended", "sensitivity"}
    selected_by_dimension: Dict[str, List[str]] = defaultdict(list)
    for feature in decisions:
        if feature["final_role"] in selected_roles:
            selected_by_dimension[str(feature["dimension_id"])].append(
                str(feature["feature_id"])
            )
    groups = evidence_groups_by_dimension(evidence)
    minimum_groups = int(
        protocol["dimension_admission"][
            "minimum_independent_peer_reviewed_groups"
        ]
    )
    minimum_features = int(
        protocol["dimension_admission"]["minimum_selected_features"]
    )
    context_exempt = bool(
        protocol["dimension_admission"][
            "context_controls_exempt_from_predictor_evidence_count"
        ]
    )
    output: List[Dict[str, Any]] = []
    for raw_dimension in dimensions:
        dimension = dict(raw_dimension)
        dimension_id = str(dimension["dimension_id"])
        selected_features = sorted(selected_by_dimension.get(dimension_id, []))
        evidence_groups = groups.get(dimension_id, [])
        enough_evidence = len(evidence_groups) >= minimum_groups
        if context_exempt and dimension["block"] == "context_control":
            enough_evidence = True
        selected = (
            len(selected_features) >= minimum_features and enough_evidence
        )
        dimension["independent_peer_reviewed_groups"] = evidence_groups
        dimension["independent_group_count"] = len(evidence_groups)
        dimension["selected_feature_ids"] = selected_features
        dimension["selected_feature_count"] = len(selected_features)
        dimension["selected"] = selected
        if selected:
            dimension["decision_reason"] = (
                "passes evidence-diversity and selected-feature rules"
            )
        elif not selected_features:
            dimension["decision_reason"] = "no feature passes the frozen rules"
        else:
            dimension["decision_reason"] = (
                f"only {len(evidence_groups)} independent evidence groups; "
                f"{minimum_groups} required"
            )
        output.append(dimension)
    return sorted(output, key=lambda dimension: str(dimension["dimension_id"]))


def build_training_sets(
    decisions: Sequence[Mapping[str, Any]],
    rules: Mapping[str, Any],
) -> Dict[str, List[str]]:
    """Build named feature blocks from roles, never from outcome performance."""
    feature_sets: Dict[str, List[str]] = {}
    for set_name, specification in rules["training_sets"].items():
        allowed_blocks = set(specification["blocks"])
        allowed_roles = set(specification["roles"])
        feature_sets[set_name] = sorted(
            str(feature["name"])
            for feature in decisions
            if feature["block"] in allowed_blocks
            and feature["final_role"] in allowed_roles
        )
    return feature_sets


def evidence_csv_rows(
    evidence: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Flatten evidence records for spreadsheet review."""
    output: List[Dict[str, Any]] = []
    for record in sorted(evidence, key=lambda row: str(row["source_id"])):
        output.append(
            {
                "source_id": record["source_id"],
                "year": record["year"],
                "citation": record["citation"],
                "doi": record["doi"],
                "url": record["url"],
                "study_type": record["study_type"],
                "research_group": record["research_group"],
                "dimension_ids": "|".join(record["dimension_ids"]),
                "evidence_roles": "|".join(record["evidence_roles"]),
                "evidence_direction": record["evidence_direction"],
                "finding": record["finding"],
                "limitations": record["limitations"],
                "formula_authorization": "|".join(
                    record["formula_authorization"]
                ),
            }
        )
    return output


def evidence_discovery_rows(
    evidence: Sequence[Mapping[str, Any]],
    snapshot_path: Path,
) -> List[Dict[str, Any]]:
    """Crosswalk verified evidence sources to frozen database rankings."""
    hits_by_doi: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    if snapshot_path.exists():
        with snapshot_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            for row in csv.DictReader(handle):
                doi = str(row.get("doi", "")).strip().lower()
                if doi:
                    hits_by_doi[doi].append(dict(row))
    output: List[Dict[str, Any]] = []
    for record in sorted(evidence, key=lambda row: str(row["source_id"])):
        hits = sorted(
            hits_by_doi.get(str(record["doi"]).lower(), []),
            key=lambda row: (
                row["query_id"],
                row["provider"],
                int(row["rank"]),
            ),
        )
        output.append(
            {
                "source_id": record["source_id"],
                "doi": record["doi"],
                "database_hit_count": len(hits),
                "query_ids": "|".join(
                    sorted({row["query_id"] for row in hits})
                ),
                "provider_rank_hits": "|".join(
                    f"{row['provider']}:{row['query_id']}:{row['rank']}"
                    for row in hits
                ),
                "discovery_route": (
                    "frozen_database_snapshot_plus_primary_verification"
                    if hits
                    else "citation_chaining_or_direct_primary_verification"
                ),
            }
        )
    return output


def decision_csv_rows(
    decisions: Sequence[Mapping[str, Any]],
    gate_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    """Flatten materialized feature decisions and all gate outcomes."""
    output: List[Dict[str, Any]] = []
    for feature in decisions:
        row: Dict[str, Any] = {
            "feature_id": feature["feature_id"],
            "name": feature["name"],
            "label_zh": feature["label_zh"],
            "dimension_id": feature["dimension_id"],
            "block": feature["block"],
            "scope_role": feature["scope_role"],
            "feature_type": feature["feature_type"],
            "formula": feature["formula"],
            "direction": feature["direction"],
            "maximum_information_time": feature["maximum_information_time"],
            "missing_rule": feature["missing_rule"],
            "source_ids": "|".join(feature["source_ids"]),
            "evidence_grade": feature["evidence_grade"],
            "data_status": feature["data_status"],
            "quality_audit_status": feature["quality_audit_status"],
            "bias_policy": feature["bias_policy"],
            "redundancy_family": feature["redundancy_family"],
            "selection_priority": feature["selection_priority"],
            "redundancy_winner_id": feature["redundancy_winner_id"],
            "failed_gates": "|".join(feature["failed_gates"]),
            "final_role": feature["final_role"],
            "decision_reason": feature["decision_reason"],
        }
        for gate_id in gate_ids:
            row[gate_id] = feature["gate_checks"][gate_id]
        output.append(row)
    return output


def selected_feature_records(
    decisions: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Return compact, fully defined selected feature records."""
    selected_roles = {"primary", "supporting", "extended", "sensitivity"}
    fields = (
        "feature_id",
        "name",
        "label_zh",
        "dimension_id",
        "block",
        "scope_role",
        "feature_type",
        "formula",
        "direction",
        "maximum_information_time",
        "missing_rule",
        "required_data",
        "source_ids",
        "evidence_grade",
        "evidence_summary",
        "data_status",
        "data_reference",
        "bias_policy",
        "final_role",
    )
    output: List[Dict[str, Any]] = []
    for feature in decisions:
        if feature["final_role"] not in selected_roles:
            continue
        output.append(
            {
                field: feature.get(field, "")
                for field in fields
            }
        )
    return output


def count_search_rows(path: Path) -> int:
    """Count snapshot data rows without loading the file."""
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(sum(1 for _ in csv.DictReader(handle)), 0)


def build_summary(
    evidence: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    dimension_decisions: Sequence[Mapping[str, Any]],
    feature_sets: Mapping[str, Sequence[str]],
    search_snapshot_path: Path,
    discovery_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Build an answer-first selection summary."""
    role_counts = Counter(str(row["final_role"]) for row in decisions)
    selected_dimensions = [
        row for row in dimension_decisions if row["selected"]
    ]
    predictor_dimensions = [
        row
        for row in selected_dimensions
        if row["block"] != "context_control"
    ]
    context_dimensions = [
        row
        for row in selected_dimensions
        if row["block"] == "context_control"
    ]
    failure_counts: Counter[str] = Counter()
    for decision in decisions:
        failure_counts.update(decision["failed_gates"])
    return {
        "schema_version": "1.0.0",
        "selection_status": "frozen_outcome_blind",
        "literature_evidence_records": len(evidence),
        "independent_research_groups": len(
            {str(record["research_group"]) for record in evidence}
        ),
        "candidate_feature_families": len(decisions),
        "candidate_dimensions": len(dimension_decisions),
        "selected_predictor_dimensions": len(predictor_dimensions),
        "selected_context_dimensions": len(context_dimensions),
        "selected_dimension_ids": [
            str(row["dimension_id"]) for row in selected_dimensions
        ],
        "selected_feature_families": sum(
            role_counts[role]
            for role in ("primary", "supporting", "extended", "sensitivity")
        ),
        "feature_role_counts": dict(sorted(role_counts.items())),
        "hard_gate_failure_counts": dict(sorted(failure_counts.items())),
        "training_set_sizes": {
            name: len(values) for name, values in feature_sets.items()
        },
        "search_snapshot_records": count_search_rows(search_snapshot_path),
        "search_snapshot_present": search_snapshot_path.exists(),
        "evidence_sources_found_in_snapshot": sum(
            int(row["database_hit_count"]) > 0 for row in discovery_rows
        ),
        "evidence_sources_from_chaining_or_direct_verification": sum(
            int(row["database_hit_count"]) == 0 for row in discovery_rows
        ),
        "quantity_rule": (
            "No dimension or feature count was specified in advance; counts "
            "are consequences of hard gates and deterministic family deduplication."
        ),
        "outcome_boundary": (
            "Future citation, diffusion, and disruption measures were not read "
            "or used for feature admission."
        ),
    }


def report_markdown(
    summary: Mapping[str, Any],
    dimensions: Sequence[Mapping[str, Any]],
    selected_features: Sequence[Mapping[str, Any]],
    feature_sets: Mapping[str, Sequence[str]],
) -> str:
    """Render a concise Chinese selection report."""
    selected_dimensions = [
        dimension for dimension in dimensions if dimension["selected"]
    ]
    lines = [
        "# 论文创新性与发表时潜在影响力特征筛选结果",
        "",
        "## 结论",
        "",
        (
            f"固定规则从 {summary['candidate_feature_families']} 个候选特征家族、"
            f"{summary['literature_evidence_records']} 条文献证据中，自动留下 "
            f"{summary['selected_predictor_dimensions']} 个可操作预测维度、"
            f"{summary['selected_context_dimensions']} 个背景控制维度和 "
            f"{summary['selected_feature_families']} 个特征家族。"
        ),
        "",
        (
            "数量没有预设配额；未来引用、扩散和颠覆结果没有参与定义、"
            "筛选或去重。"
        ),
        "",
        "## 留下的维度",
        "",
        "| 维度 | 构念块 | 特征数 | 独立证据组数 |",
        "|---|---:|---:|---:|",
    ]
    for dimension in selected_dimensions:
        lines.append(
            f"| {dimension['label_zh']} (`{dimension['dimension_id']}`) "
            f"| {dimension['block']} | {dimension['selected_feature_count']} "
            f"| {dimension['independent_group_count']} |"
        )
    lines.extend(
        [
            "",
            "## 留下的特征",
            "",
            "| 特征 | 维度 | 角色 | 就绪状态 |",
            "|---|---|---:|---:|",
        ]
    )
    dimension_labels = {
        str(row["dimension_id"]): str(row["label_zh"])
        for row in dimensions
    }
    for feature in selected_features:
        lines.append(
            f"| `{feature['name']}` | "
            f"{dimension_labels[str(feature['dimension_id'])]} | "
            f"{feature['final_role']} | {feature['data_status']} |"
        )
    lines.extend(
        [
            "",
            "## 可直接用于后续实验的固定集合",
            "",
        ]
    )
    for name, values in feature_sets.items():
        lines.append(f"- `{name}`：{len(values)} 个特征")
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- `innovation_evidence` 是直接的知识重组新颖性证据。",
            "- `substantive_potential` 是发表时科学定位，不等同于直接新颖性。",
            "- `opportunity_visibility` 是传播、受众和资源机会，必须单独报告。",
            "- `context_control` 只用于年份和学科异质性调整。",
            "- 引用潜力不是质量、正确性、社会价值或创新真值。",
            "",
        ]
    )
    return "\n".join(lines)


def verify_declared_provenance(protocol: Mapping[str, Any]) -> Dict[str, Any]:
    """Verify the small local registries reused by this standalone package."""
    results: Dict[str, Any] = {}
    for key, declaration in protocol["local_provenance"].items():
        if not isinstance(declaration, dict) or "path" not in declaration:
            continue
        path = (ROOT / declaration["path"]).resolve()
        actual = sha256_file(path)
        expected = str(declaration["sha256"])
        results[key] = {
            "path": str(path),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matches": actual == expected,
        }
        if actual != expected:
            raise ValueError(
                f"Provenance hash mismatch for {key}: {path}"
            )
    return results


def build_manifest(
    output_dir: Path,
    provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    """Hash all source registries and generated outputs."""
    input_hashes = {
        name: sha256_file(ROOT / name)
        for name in INPUT_FILES
    }
    optional_inputs = (
        "search_snapshot.csv",
        "search_errors.json",
    )
    for name in optional_inputs:
        path = ROOT / name
        if path.exists():
            input_hashes[name] = sha256_file(path)
    output_hashes = {
        name: sha256_file(output_dir / name)
        for name in GENERATED_OUTPUT_FILES
    }
    return {
        "schema_version": "1.0.0",
        "selection_script": {
            "path": str((ROOT / "select_features.py").resolve()),
            "sha256": sha256_file(ROOT / "select_features.py"),
        },
        "python_version": platform.python_version(),
        "input_sha256": input_hashes,
        "output_sha256": output_hashes,
        "verified_reused_provenance": provenance,
        "determinism": {
            "timestamps_omitted": True,
            "stable_sorting": True,
            "future_outcomes_read": False,
            "numeric_quota_used": False,
        },
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Apply frozen, outcome-blind gates to the standalone innovation "
            "and potential-impact feature census."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args()


def main() -> None:
    """Run the complete deterministic feature-selection workflow."""
    args = parse_args()
    output_dir = args.output_dir.resolve()
    protocol = read_json(ROOT / "protocol.json")
    dimensions_payload = read_json(ROOT / "dimensions.json")
    evidence_payload = read_json(ROOT / "literature_evidence.json")
    registry = read_json(ROOT / "feature_registry.json")
    rules = read_json(ROOT / "screening_rules.json")
    capabilities = read_json(ROOT / "data_capabilities.json")
    dimensions = dimensions_payload["dimensions"]
    evidence = evidence_payload["records"]
    features = merge_feature_defaults(registry)
    validate_registries(
        protocol,
        dimensions,
        evidence,
        features,
        capabilities,
    )
    provenance = verify_declared_provenance(protocol)
    decisions = decide_features(features, rules)
    dimension_decisions = decide_dimensions(
        dimensions,
        decisions,
        evidence,
        protocol,
    )
    feature_sets = build_training_sets(decisions, rules)
    selected_features = selected_feature_records(decisions)
    discovery_rows = evidence_discovery_rows(
        evidence,
        ROOT / "search_snapshot.csv",
    )
    summary = build_summary(
        evidence,
        decisions,
        dimension_decisions,
        feature_sets,
        ROOT / "search_snapshot.csv",
        discovery_rows,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    gate_ids = list(rules["hard_gates"])
    evidence_rows = evidence_csv_rows(evidence)
    decision_rows = decision_csv_rows(decisions, gate_ids)
    write_csv(
        output_dir / "literature_evidence_table.csv",
        evidence_rows,
        (
            "source_id",
            "year",
            "citation",
            "doi",
            "url",
            "study_type",
            "research_group",
            "dimension_ids",
            "evidence_roles",
            "evidence_direction",
            "finding",
            "limitations",
            "formula_authorization",
        ),
    )
    write_csv(
        output_dir / "evidence_discovery_crosswalk.csv",
        discovery_rows,
        (
            "source_id",
            "doi",
            "database_hit_count",
            "query_ids",
            "provider_rank_hits",
            "discovery_route",
        ),
    )
    write_csv(
        output_dir / "feature_decisions.csv",
        decision_rows,
        (
            "feature_id",
            "name",
            "label_zh",
            "dimension_id",
            "block",
            "scope_role",
            "feature_type",
            "formula",
            "direction",
            "maximum_information_time",
            "missing_rule",
            "source_ids",
            "evidence_grade",
            "data_status",
            "quality_audit_status",
            "bias_policy",
            "redundancy_family",
            "selection_priority",
            "redundancy_winner_id",
            "failed_gates",
            "final_role",
            "decision_reason",
            *gate_ids,
        ),
    )
    write_json(
        output_dir / "final_dimensions.json",
        {
            "schema_version": "1.0.0",
            "dimensions": dimension_decisions,
        },
    )
    write_json(
        output_dir / "final_features.json",
        {
            "schema_version": "1.0.0",
            "features": selected_features,
        },
    )
    write_json(
        output_dir / "training_feature_sets.json",
        {
            "schema_version": "1.0.0",
            "sets": feature_sets,
        },
    )
    write_json(output_dir / "selection_summary.json", summary)
    (output_dir / "selection_report.md").write_text(
        report_markdown(
            summary,
            dimension_decisions,
            selected_features,
            feature_sets,
        ),
        encoding="utf-8",
    )
    write_json(
        output_dir / "audit_manifest.json",
        build_manifest(output_dir, provenance),
    )
    print(
        f"Selected {summary['selected_predictor_dimensions']} predictor "
        f"dimensions, {summary['selected_context_dimensions']} context "
        f"dimension, and {summary['selected_feature_families']} feature "
        f"families from {summary['candidate_feature_families']} candidates."
    )


if __name__ == "__main__":
    main()

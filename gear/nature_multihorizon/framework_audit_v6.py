"""Reviewer-facing audit of v6 construct, indicator, and outcome selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .evidence_registry import (
    audit_registry_implementations,
    load_evidence_registry,
    registry_sha256,
)
from .evidence_selection_v6 import (
    audit_registry_source_selection,
    evidence_selection_sha256,
    load_evidence_selection_protocol,
)
from .prediction_registry_v6 import (
    audit_prediction_registry_implementations,
    load_prediction_registry,
    prediction_registry_sha256,
)
from .source_audit_v6 import audit_local_sources


def _resolve_config_path(project_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else project_root / path


def audit_v6_framework(
    config_path: Path,
    *,
    project_root: Path,
    include_source_audit: bool = True,
) -> Dict[str, Any]:
    """Build a deterministic definition- and source-readiness audit."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    evidence_path = _resolve_config_path(
        project_root, str(config["evidence_registry_path"])
    )
    prediction_path = _resolve_config_path(
        project_root, str(config["prediction_registry_path"])
    )
    selection_path = _resolve_config_path(
        project_root, str(config["evidence_selection_protocol_path"])
    )
    innovation = load_evidence_registry(evidence_path)
    prediction = load_prediction_registry(prediction_path)
    selection = load_evidence_selection_protocol(selection_path)
    innovation_implementation = audit_registry_implementations(innovation)
    prediction_implementation = audit_prediction_registry_implementations(
        prediction
    )
    selection_audit = audit_registry_source_selection(
        selection, innovation, prediction
    )
    source_audit: Optional[Mapping[str, Any]] = None
    if include_source_audit:
        source_audit = audit_local_sources(
            config, project_root=project_root, deep_hash=False
        )
    definition_pass = bool(
        innovation_implementation["overall_pass"]
        and prediction_implementation["overall_pass"]
        and selection_audit["overall_pass"]
    )
    candidate_dimensions = [
        dimension_id
        for dimension_id, dimension in innovation.dimensions.items()
        if dimension.status.value == "candidate_confirmatory"
    ]
    conditional_dimensions = [
        dimension_id
        for dimension_id, dimension in innovation.dimensions.items()
        if dimension.status.value == "conditional"
    ]
    exploratory_dimensions = [
        dimension_id
        for dimension_id, dimension in innovation.dimensions.items()
        if dimension.status.value == "exploratory"
    ]
    return {
        "audit_version": "aspr-v6-framework-audit-1",
        "protocol_id": config.get("protocol_id"),
        "network_policy": config.get("network_policy"),
        "raw_data_policy": config.get("raw_data_policy"),
        "definition_audit_pass": definition_pass,
        "source_data_audit_pass": (
            bool(source_audit and source_audit.get("overall_pass"))
            if include_source_audit
            else None
        ),
        "release_confirmatory_ready": False,
        "release_confirmatory_blockers": [
            "P1-P8 runtime promotion report has not passed for all claimed confirmatory entities.",
            "D3/D5/D8 nested OOF, temporal, domain-macro, uplift, calibration, and coverage gates have not passed.",
            *(
                []
                if source_audit and source_audit.get("overall_pass")
                else ["One or more required frozen local source assets are unavailable."]
            ),
        ],
        "allowed_current_claim": (
            "The v6 construct, metric, outcome, control, and opportunity "
            "definitions are preregistered and source-audited; candidate, "
            "conditional, and exploratory labels remain explicit."
        ),
        "forbidden_current_claim": (
            "Do not claim that v6 is empirically confirmed, Nature-grade "
            "validated, or predictively accepted until runtime promotion and "
            "all sealed evaluation gates pass."
        ),
        "lineage": {
            "innovation_registry_sha256": registry_sha256(innovation),
            "prediction_registry_sha256": prediction_registry_sha256(prediction),
            "evidence_selection_sha256": evidence_selection_sha256(selection),
            "source_lineage_id": (
                source_audit.get("source_lineage_id") if source_audit else None
            ),
        },
        "innovation_registry": innovation.model_dump(mode="json", by_alias=True),
        "prediction_registry": prediction.model_dump(mode="json", by_alias=True),
        "evidence_selection": selection.model_dump(mode="json"),
        "audits": {
            "innovation_implementation": innovation_implementation,
            "prediction_implementation": prediction_implementation,
            "registry_source_selection": selection_audit,
            "local_sources": source_audit,
        },
        "status_summary": {
            "candidate_confirmatory_dimensions": candidate_dimensions,
            "conditional_dimensions": conditional_dimensions,
            "exploratory_dimensions": exploratory_dimensions,
        },
    }


def _escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _source_links(source_ids: list[str], sources: Mapping[str, Any]) -> str:
    links = []
    for source_id in source_ids:
        source = sources[source_id]
        locator = source.get("doi") or source.get("url")
        url = (
            f"https://doi.org/{locator}"
            if source.get("doi")
            else str(locator)
        )
        links.append(f"[{source_id}]({url})")
    return ", ".join(links)


def render_reviewer_framework_markdown(report: Mapping[str, Any]) -> str:
    """Render a concise, source-linked answer from the machine audit."""
    innovation = report["innovation_registry"]
    prediction = report["prediction_registry"]
    source_pass = report.get("source_data_audit_pass")
    lines = [
        "# ASPR v6 指标体系与来源审计",
        "",
        "## 结论",
        "",
        (
            "- **定义层可以回应审稿人**：观察维度、具体指标、结果变量、"
            "控制变量和机会结构均已登记来源、公式、时间边界、缺失规则及"
            "纳入/排除决定。"
        ),
        (
            "- **目前不能声称体系已经最终确认**："
            f"本地源数据门状态为 `{source_pass}`，且 P1–P8 运行晋升与"
            " D3/D5/D8 封存评估尚未全部通过。"
        ),
        f"- 当前允许表述：{report['allowed_current_claim']}",
        f"- 当前禁止表述：{report['forbidden_current_claim']}",
        "",
        "## 论文创新性／创新证据观察维度",
        "",
        "| 维度 | 角色 | 定义阶段状态 | 为什么纳入 | 来源 |",
        "|---|---|---|---|---|",
    ]
    for dimension in innovation["dimensions"].values():
        source_ids = list(
            dict.fromkeys(
                dimension["foundational_source_ids"]
                + dimension["paper_level_source_ids"]
            )
        )
        lines.append(
            "| {dimension} | {role} | {status} | {reason} | {sources} |".format(
                dimension=_escape(
                    f"{dimension['dimension_id']} — {dimension['label']}"
                ),
                role=_escape(dimension["role"]),
                status=_escape(dimension["status"]),
                reason=_escape(dimension["admission_decision"]),
                sources=_source_links(
                    source_ids, innovation["sources"]
                ),
            )
        )
    lines.extend(
        [
            "",
            "## 具体创新证据指标",
            "",
            "| 指标 | 维度 | 模型角色 | 状态／保真度 | 纳入或降级原因 | 来源 |",
            "|---|---|---|---|---|---|",
        ]
    )
    for metric in innovation["metrics"].values():
        lines.append(
            "| {metric} | {dimension} | {model_use} | {status} / {fidelity} | "
            "{reason} | {sources} |".format(
                metric=_escape(f"{metric['metric_id']} ({metric['code_name']})"),
                dimension=_escape(metric.get("dimension_id") or "—"),
                model_use=_escape(metric["model_use"]),
                status=_escape(metric["status"]),
                fidelity=_escape(metric["fidelity"]),
                reason=_escape(metric["disposition_reason"]),
                sources=_source_links(
                    list(metric["source_ids"]), innovation["sources"]
                ),
            )
        )
    lines.extend(
        [
            "",
            "## 潜在影响力预测观察类别",
            "",
            "| 类别 | 角色 | 状态 | 为什么纳入 | 明确边界 | 来源 |",
            "|---|---|---|---|---|---|",
        ]
    )
    for category in prediction["categories"].values():
        lines.append(
            "| {category} | {role} | {status} | {reason} | {boundary} | "
            "{sources} |".format(
                category=_escape(
                    f"{category['category_id']} — {category['label']}"
                ),
                role=_escape(category["role"]),
                status=_escape(category["status"]),
                reason=_escape(category["rationale"]),
                boundary=_escape(category["boundary"]),
                sources=_source_links(
                    list(category["source_ids"]), prediction["sources"]
                ),
            )
        )
    lines.extend(
        [
            "",
            "## 科学纳入／纳出规则",
            "",
            "- 维度定义门：D1–D7；指标定义门：I1–I10；运行晋升门：P1–P8。",
            "- 预测类别门：K1–K7；预测变量门：V1–V10；评估规则：E1–E7。",
            "- 文献选择门：S1–S8，并同时登记不利或否定性证据。",
            "- 所有出版时预测量执行 `source_max_year < publication_year`。",
            "- 成功抓取且零施引是观测零；失败或截断抓取是缺失。",
            "- future citation、disruption、未来社群和未来复用不能进入创新画像。",
            "- 作者／机构声望、年份、领域、venue、参考文献数量和先验流行度只可作控制变量。",
            "- 项目自定义复合权重、阈值和近似实现必须明确标注，不能冒充文献原定义。",
            "",
            "## 仍需通过的发布门",
            "",
        ]
    )
    lines.extend(
        f"- {blocker}" for blocker in report["release_confirmatory_blockers"]
    )
    lines.extend(
        [
            "",
            "## 血缘",
            "",
            f"- 创新证据注册表：`{report['lineage']['innovation_registry_sha256']}`",
            f"- 影响预测注册表：`{report['lineage']['prediction_registry_sha256']}`",
            f"- 文献选择协议：`{report['lineage']['evidence_selection_sha256']}`",
            f"- 本地数据源：`{report['lineage']['source_lineage_id']}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_framework_audit(
    report: Mapping[str, Any],
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    """Write reviewer-facing JSON and Markdown derived artifacts."""
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).write_text(
        json.dumps(dict(report), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    Path(markdown_path).parent.mkdir(parents=True, exist_ok=True)
    Path(markdown_path).write_text(
        render_reviewer_framework_markdown(report),
        encoding="utf-8",
    )


__all__ = [
    "audit_v6_framework",
    "render_reviewer_framework_markdown",
    "write_framework_audit",
]

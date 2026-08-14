"""Render the final ASPR v6.1 evidence map from frozen artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gear.nature_multihorizon.candidate_registry_v6_1 import (
    load_candidate_registry_v6_1,
)
from gear.nature_multihorizon.modeling_v6_1 import (
    build_v6_1_feature_sets,
    load_simple_config,
)
from gear.nature_multihorizon.source_audit_v6 import sha256_file


DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs" / "nature_multihorizon" / "v6_1_simple.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "aspr_v6_1_indicator_evidence_map.md"


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _escape(value: Any) -> str:
    if value is None:
        return ""
    return (
        str(value)
        .replace("|", "\\|")
        .replace("\n", " ")
        .replace("\r", " ")
    )


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return _escape(value)
    if pd.isna(numeric):
        return "NA"
    return f"{numeric:.{digits}f}"


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    output = [
        "| " + " | ".join(_escape(item) for item in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    output.extend(
        "| " + " | ".join(_escape(item) for item in row) + " |"
        for row in rows
    )
    return "\n".join(output)


def _source_link(
    source_id: str,
    sources: Mapping[str, Any],
) -> str:
    source = sources[source_id]
    if isinstance(source, Mapping):
        doi = source.get("doi")
        url = source.get("url")
    else:
        doi = getattr(source, "doi", None)
        url = getattr(source, "url", None)
    if doi:
        return f"[{source_id}](https://doi.org/{doi})"
    if url:
        return f"[{source_id}]({url})"
    return source_id


def _source_links(
    source_ids: Sequence[str],
    sources: Mapping[str, Any],
) -> str:
    return ", ".join(_source_link(item, sources) for item in source_ids)


def _manifest_output(
    manifest: Mapping[str, Any],
    name: str,
) -> Path:
    return Path(manifest["outputs"][name]["path"]).resolve()


def _single_manifest(root: Path, pattern: str) -> Mapping[str, Any]:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {pattern} under {root}, found {len(matches)}"
        )
    return _load_json(matches[0])


def _candidate_rows(
    catalog: Any,
    decisions: pd.DataFrame,
) -> Iterable[Sequence[Any]]:
    by_id = decisions.set_index("candidate_id").to_dict("index")
    for candidate in catalog.candidates.values():
        decision = by_id[candidate.candidate_id]
        sources = tuple(
            dict.fromkeys(
                (
                    *candidate.original_source_ids,
                    *candidate.paper_application_source_ids,
                    *candidate.validation_source_ids,
                )
            )
        )
        yield (
            candidate.candidate_id,
            catalog.observation_angles[candidate.angle_id].label_zh,
            candidate.mathematical_family,
            candidate.formula,
            candidate.local_computability,
            _source_links(sources, catalog.sources),
            decision["proposed_final_role"],
            decision["proposed_decision_reason"],
        )


def _control_rows(
    control_registry: Mapping[str, Any],
) -> Iterable[Sequence[Any]]:
    sources = control_registry["sources"]
    for name, definition in control_registry["features"].items():
        yield (
            name,
            "/".join(definition["sets"]),
            definition["meaning"],
            definition["formula"],
            ", ".join(
                _source_link(item, sources)
                for item in definition["source_ids"]
            ),
            definition["maximum_information_time"],
            definition["missing_rule"],
        )


def render(config_path: Path, output_path: Path) -> Path:
    """Render the evidence map after screening, audit, and OOF complete."""
    config = load_simple_config(config_path)
    catalog_path = _resolve(config["paths"]["candidate_catalog"])
    registry_path = _resolve(config["paths"]["candidate_registry"])
    control_registry_path = _resolve(
        config["paths"]["control_registry"]
    )
    dataset_root = _resolve(config["paths"]["v6_1_dataset"])
    analysis_root = _resolve(config["paths"]["v6_1_analysis"])
    catalog = load_candidate_registry_v6_1(catalog_path)
    registry = load_candidate_registry_v6_1(registry_path)
    search_log = _load_json(_resolve(catalog.search_log_path))
    control_registry = _load_json(control_registry_path)
    screening = _single_manifest(
        analysis_root, "screening_*/screening_manifest.json"
    )
    oof = _single_manifest(analysis_root, "oof_*/oof_run_manifest.json")
    innovation_only_paths = sorted(
        analysis_root.glob(
            "supplement_innovation_only_*/innovation_only_manifest.json"
        )
    )
    if len(innovation_only_paths) > 1:
        raise ValueError("multiple innovation-only supplement manifests")
    innovation_only = (
        _load_json(innovation_only_paths[0])
        if innovation_only_paths
        else None
    )
    audit = _load_json(analysis_root / "data_quality_report.json")
    reproducibility_path = analysis_root / "reproducibility_report.json"
    reproducibility = (
        _load_json(reproducibility_path)
        if reproducibility_path.is_file()
        else None
    )
    completion_audit_path = analysis_root / "completion_audit.json"
    completion_audit = (
        _load_json(completion_audit_path)
        if completion_audit_path.is_file()
        else None
    )
    validation_path = analysis_root / "validation_summary.json"
    validation = (
        _load_json(validation_path) if validation_path.is_file() else None
    )
    materialization = _load_json(
        dataset_root / "v6_1_materialization_manifest.json"
    )
    openalex = _load_json(
        dataset_root / "target_openalex_metadata_manifest.json"
    )
    overlap = _load_json(
        dataset_root / "reference_overlap_context_manifest.json"
    )
    overlap_extension = _load_json(
        dataset_root / "reference_overlap_extension_manifest.json"
    )
    decisions = pd.read_csv(_manifest_output(screening, "decisions"))
    coverage = pd.read_csv(_manifest_output(screening, "coverage"))
    domain_coverage = pd.read_csv(
        _manifest_output(screening, "domain_coverage")
    )
    stability = pd.read_csv(_manifest_output(screening, "stability"))
    metrics = pd.read_csv(_manifest_output(oof, "metrics"))
    folds = pd.read_csv(_manifest_output(oof, "folds"))
    comparisons = pd.read_csv(_manifest_output(oof, "comparisons"))
    domain_metrics = pd.read_csv(
        _manifest_output(oof, "domain_metrics")
    )
    feature_sets = build_v6_1_feature_sets(registry, config)
    primary_decisions = decisions[
        decisions["proposed_final_role"].eq("primary")
    ].copy()
    coverage_by_id = coverage.set_index("candidate_id").to_dict("index")
    stability_by_code = stability.set_index("code_name").to_dict("index")

    lines: list[str] = []
    lines.extend(
        [
            "# ASPR v6.1 五角度指标证据地图、数据、模型与全时期 OOF 结果",
            "",
            f"> 最终协议：`{config['protocol_id']}`  ",
            f"> 最终指标注册表：`{sha256_file(registry_path)}`  ",
            f"> 结果制品：`{oof['artifact_id']}`  ",
            f"> 数据审计：**{audit['assessment']}**  ",
            "> 定位：多源系统性范围证据地图；不是穷尽互联网的系统综述，也不是元分析。",
            "",
            "## 1. 先给结论",
            "",
            "本体系把两个问题严格分开：五个观察角度及其指标描述论文发表时的"
            "知识重组证据；D3/D5/D8 标签描述发表后的学术传播与跨领域扩散。"
            "未来引用没有参与指标筛选，OOF 也没有决定某个创新指标的去留。",
            "",
            "五个角度不是“学界公认的五种互斥创新类型”，而是由组合新颖性、"
            "科学计量学和多样性/整合研究支持的五个下位观察维度。它们不能替代"
            "同行评审、实验正确性、社会影响或 Nature 录用判断。",
            "",
            "最终主创新指标由统一的来源、时间、可计算性、覆盖、稳定性、"
            "公式忠实度和非冗余规则产生；不是先规定必须保留八个，也不按 OOF"
            " 高低删指标。",
            "",
            "## 2. 系统检索与筛选流程",
            "",
            "```mermaid",
            "flowchart LR",
            '  A["多源检索<br/>Crossref / OpenAlex / PubMed-PMC / 出版社 / 学术网页"]',
            '  B["DOI与规范化题名去重<br/>逐条记录检索式、日期和决定"]',
            '  C["核心论文、综述和软件目录<br/>前向/后向引文追踪"]',
            '  D["连续两轮未发现<br/>新数学指标家族"]',
            f'  E["候选目录<br/>{len(catalog.candidates)}项、{len(catalog.sources)}个来源"]',
            '  F["I1-I10结果盲筛选<br/>来源·时间·本地数据·覆盖·稳定性·忠实度"]',
            f'  G["冻结主指标<br/>{len(primary_decisions)}项、覆盖五个角度"]',
            '  H["冻结注册表哈希后<br/>才读取标签并运行OOF"]',
            "  A --> B --> C --> D --> E --> F --> G --> H",
            "```",
            "",
            f"检索截止 **{search_log['cutoff_date']}**，共有"
            f" **{len(search_log['search_records'])}** 条可审计检索/核验记录。"
            "Google Scholar 自动访问失败已原样登记，未杜撰结果数；"
            "综述、预印本和 Novelpy 只用于发现候选，主指标证据回到同行评议"
            "公式来源和论文级应用。范围声明是“多源系统性范围证据地图”，"
            "不宣称穷尽互联网。",
            "",
            "## 3. 五个观察角度：意义、来源与最终指标",
            "",
        ]
    )
    angle_rows = []
    for angle in catalog.observation_angles.values():
        selected = primary_decisions[
            primary_decisions["angle_id"].eq(angle.angle_id)
        ]["candidate_id"].tolist()
        angle_rows.append(
            (
                angle.label_zh,
                angle.meaning,
                _source_links(angle.source_ids, catalog.sources),
                "；".join(selected),
                angle.inclusion_rule,
                angle.exclusion_rule,
            )
        )
    lines.extend(
        [
            _table(
                (
                    "角度",
                    "观察意义",
                    "角度来源",
                    "最终主指标",
                    "纳入原则",
                    "排除原则",
                ),
                angle_rows,
            ),
            "",
            "这些角度的角色是解释论文如何选取、组合和整合既有知识。"
            "A4/A5 的跨学科与距离指标是支持性创新上下文，不等于创新本身；"
            "A1–A3 更直接观察组合罕见性、非典型性和首次出现。",
            "",
            "## 4. 最终主创新指标：公式、来源和测量门槛",
            "",
        ]
    )
    final_rows = []
    for row in primary_decisions.itertuples(index=False):
        candidate = registry.candidates[str(row.candidate_id)]
        cov = coverage_by_id[candidate.candidate_id]
        stable = stability_by_code[candidate.code_name]
        source_ids = tuple(
            dict.fromkeys(
                (
                    *candidate.original_source_ids,
                    *candidate.paper_application_source_ids,
                    *candidate.validation_source_ids,
                )
            )
        )
        final_rows.append(
            (
                candidate.candidate_id,
                candidate.code_name,
                catalog.observation_angles[candidate.angle_id].label_zh,
                candidate.formula,
                _source_links(source_ids, catalog.sources),
                _fmt(cov["raw_overall_coverage"]),
                _fmt(cov["overall_coverage"]),
                _fmt(cov["minimum_domain_coverage"]),
                _fmt(stable["stability_spearman"]),
                _fmt(stable["stability_median_relative_error"]),
            )
        )
    lines.extend(
        [
            _table(
                (
                    "ID",
                    "模型列",
                    "角度",
                    "冻结公式",
                    "原始/应用/验证来源",
                    "全队列原始覆盖",
                    "有效分母覆盖",
                    "最低大类覆盖",
                    "80%重采样最差ρ",
                    "最大中位相对误差",
                ),
                final_rows,
            ),
            "",
            "覆盖门槛使用预先声明的 `eligible_by_metric_family` 有效分母："
            "论文至少有 10 条有效参考文献；来源对指标还要求来源映射不低于"
            " 60%，领域指标要求领域映射不低于 60%。参考集合重叠指标只需要"
            "参考文献 ID，因此不额外要求来源/领域映射。全队列原始覆盖同时"
            "保留，不能用插补伪造为已测量。",
            "",
            "## 5. 全部候选范围和逐项决定",
            "",
            f"共登记 **{len(catalog.candidates)}** 个候选、"
            f"**{len(catalog.sources)}** 个学术来源。下表包含所有被发现并"
            "正式登记的候选，包括因外部文本、未来信息、公式证据不足、覆盖、"
            "稳定性或冗余而排除的指标。",
            "",
            _table(
                (
                    "候选ID",
                    "角度",
                    "数学家族",
                    "公式",
                    "本地等级",
                    "来源",
                    "最终角色",
                    "逐项理由",
                ),
                _candidate_rows(catalog, decisions),
            ),
            "",
            "筛选时强制执行：同行评议公式与论文级应用；唯一角度映射；"
            "发表前信息；仅冻结本地数据；公式与缺失规则可复现；有效覆盖"
            "总体不低于 70%、每大类不低于 50%；80%参考重采样 Spearman"
            "不低于 0.90 且中位相对误差不高于 0.10；近似公式与精确实现"
            "Spearman 不低于 0.95 且误差不高于 0.05；通过手算、时间和"
            "非退化测试。同一数学家族只保留一个主实现。OOF 不在这些规则中。",
            "",
            "## 6. 结果盲修订记录",
            "",
        ]
    )
    r2_summary_path = (
        PROJECT_ROOT
        / "outputs"
        / "nature_multihorizon_v6_1_r2_local"
        / "measurement_revision_r2_primary_scope"
        / "revision_summary.json"
    )
    r3_summary_path = (
        PROJECT_ROOT
        / "outputs"
        / "nature_multihorizon_v6_1_r3_local"
        / "measurement_revision_r3_nature_background_3y"
        / "revision_summary.json"
    )
    if r2_summary_path.is_file():
        r2 = _load_json(r2_summary_path)
        lines.extend(
            [
                "R2 首次实现把参考重叠的背景限制为建模主队列。该版本在"
                f"有效分母中的总体覆盖为 **{_fmt(r2['overall_eligible_coverage'])}**，"
                f"最低大类覆盖为 **{_fmt(r2['minimum_domain_eligible_coverage'])}**，"
                "数学统计类未达到 0.50。R2 制品和哈希已保留。",
                "",
            ]
        )
    if r3_summary_path.is_file():
        r3 = _load_json(r3_summary_path)
        lines.extend(
            [
                "R3 仅把比较背景扩展到完整的本地冻结 Nature v5 既往记录；"
                f"总体有效覆盖升至 **{_fmt(r3['overall_eligible_coverage'])}**，"
                f"但最低大类仍只有 **{_fmt(r3['minimum_domain_eligible_coverage'])}**，"
                "所以同样未冻结。R3 保留 10 年参考窗口和 3 年共引窗口，"
                "其失败制品与哈希也已保留。",
                "",
                "R4 回查原始方法与结果表后，只在该论文实际评估的四种窗口"
                "方案中选择最后一个尚未核验的 `all references / all prior "
                "co-citing papers` 变体。R4 没有改变 Jaccard 公式、共享至少"
                "一条参考文献和同领域两项条件，也没有改变焦点论文、阈值、"
                "标签、模型或时间折。该版本在任何标签/OOF 读取前通过覆盖后，"
                "仍须通过相同的 80% 重采样稳定性门槛。",
                "",
                "R5 是完成审计发现的协议角色修订，而不是性能调参："
                "`first_time_source_pair_distance_mean` 和同族 sum 版本虽然"
                "已从本地历史来源距离派生出非零值，但有效覆盖仅约 0.22%，"
                "最低大类约 0.14%，远低于固定门槛。R4 自动把所有已实现的"
                "非主指标标为敏感性，违背了该技术债“未过覆盖则保持排除”的"
                "专门规则。R5 将这两项改为排除；覆盖、稳定性、近似忠实度"
                "表与 R4 逐值一致，八个主指标、全部模型输入、标签、折、参数"
                "和种子均未改变。修订后重新冻结注册表并完整重跑六折 OOF。",
                "R5 的八个最终结果文件与 R4 对应文件 SHA-256 完全相同；"
                "总 artifact ID 不同，因为配置、注册表和冻结谱系按 R5 更新。",
                "",
            ]
        )
    lines.extend(
        [
            "此外，来源核对发现 `source_pair_mean_surprisal` 是项目自定义"
            "分布均值，不是 Lee 等人定义的 Novelty U。它因此被明确降为"
            "探索性排除项，不能借用 Lee 来源进入主模型；A1 改由"
            " Matsumoto 等人发表的 `1 − mean Jaccard` 公式竞争。",
            "",
            "## 7. 控制特征及其他实际进入模型的变量",
            "",
            control_registry["role_boundary"],
            "",
            _table(
                (
                    "特征",
                    "集合",
                    "意义",
                    "公式",
                    "来源",
                    "最大信息时点",
                    "缺失规则",
                ),
                _control_rows(control_registry),
            ),
            "",
            "K0 是原五项历史对照；K1 是十一项主控制集；K2 只作强控制"
            "敏感性分析。质量/检索成功标记不作为实质预测特征。当前 JIF、"
            "未来引用、可变开放获取状态和发表后作者声望均未进入模型。",
            "",
            "实际比较的特征集合为：",
            "",
        ]
    )
    for model_id, names in feature_sets.items():
        lines.append(f"- `{model_id}`：{', '.join(f'`{x}`' for x in names)}")
    lines.extend(
        [
            "",
            "## 8. 数据及谱系",
            "",
            _table(
                ("项目", "数值/说明"),
                (
                    ("焦点主论文", audit["n_primary_papers"]),
                    (
                        "年份",
                        f"{audit['publication_year_min']}–"
                        f"{audit['publication_year_max']}",
                    ),
                    ("自然科学大类", audit["n_domains"]),
                    (
                        "参考重叠背景论文",
                        overlap["n_historical_papers"],
                    ),
                    ("扫描的 Nature 引用边", overlap["n_edge_rows"]),
                    (
                        "OpenAlex works 分片",
                        openalex["n_files_registered"],
                    ),
                    (
                        "OpenAlex 完成分片",
                        openalex["n_files_completed"],
                    ),
                    (
                        "OpenAlex works 记录扫描",
                        openalex["n_snapshot_records_scanned"],
                    ),
                    (
                        "OpenAlex 目标元数据覆盖",
                        _fmt(openalex["coverage"]),
                    ),
                    (
                        "参考重叠窗口",
                        "reference="
                        f"{overlap_extension['reference_window_years']}; "
                        "co-citing="
                        f"{overlap_extension['cociting_window_years']}",
                    ),
                    (
                        "外部实验数据",
                        "无；联网只用于文献证据检索",
                    ),
                ),
            ),
            "",
            "论文—参考、目标、队列和机会特征直接复用 v6 冻结视图；审计"
            "逐文件验证 v6 与 v6.1 哈希一致。新建的只有候选创新特征、"
            "参考重叠历史、OpenAlex 目标元数据和 K1/K2 派生控制视图。"
            "OpenAlex 原始快照没有复制、更新或联网补抓。",
            "",
            "## 9. 标签是什么",
            "",
            "标签不是“创新真值”。对每个 D3/D5/D8 窗口，第一阶段标签为：",
            "",
            "```text",
            "future_uptake = 1[n_future_citers > 0]",
            "```",
            "",
            "未来请求成功且没有施引者的论文保留为 0；请求失败保持缺失，"
            "绝不改成 0。第二阶段只在正 uptake 且未来分类数据完整的训练论文"
            "上构造扩散分数：对未来施引者的 field/subfield/topic reach 取"
            "`log1p` 后按训练折经验分布转为分位，对 field/topic Simpson"
            "均衡度同样转为分位；三个 breadth 分量求均值、两个 evenness"
            "分量求均值，最后各占 0.5。",
            "",
            "测试论文的最终实现标签为：无 uptake 时等于 0；有 uptake 且"
            "分类完整时等于训练折坐标系下的扩散分位；分类不完整时缺失。"
            "因此 Spearman 回答的是“模型能否把未参与训练的论文按未来学术"
            "传播和跨领域扩散程度正确排序”，不是预测引用次数。",
            "",
            "## 10. 模型和 OOF 计算",
            "",
            "所有模型使用完全相同的论文、标签和六个扩展时间折。每折只用"
            "更早年份训练；数值缺失填补、缺失指示、类别编码、目标分位坐标"
            "和校准全部在训练折内完成。",
            "",
            _table(
                ("折", "训练", "预测"),
                (
                    (
                        int(row.fold_id),
                        f"≤{int(row.train_year_max)}",
                        f"{int(row.test_year_min)}–{int(row.test_year_max)}",
                    )
                    for row in folds.drop_duplicates("fold_id").itertuples(
                        index=False
                    )
                ),
            ),
            "",
            "固定 `medium` 两部分模型：第一部分为未来 uptake 的"
            " HistGradientBoostingClassifier；第二部分为正 uptake 条件下扩散"
            "得分的 HistGradientBoostingRegressor。最终分数为：",
            "",
            "```text",
            "expected_diffusion_score",
            "  = calibrated P(future uptake)",
            "  × calibrated conditional diffusion score",
            "```",
            "",
            f"固定参数：`max_leaf_nodes={config['model']['max_leaf_nodes']}`、"
            f"`max_depth={config['model']['max_depth']}`、"
            f"`min_samples_leaf={config['model']['min_samples_leaf']}`、"
            f"`learning_rate={config['model']['learning_rate']}`、"
            f"`max_iter={config['model']['max_iter']}`、"
            f"`l2={config['model']['l2_regularization']}`。每个外折内部再用"
            f" {config['model']['inner_temporal_folds']} 个时间折产生校准预测；"
            "没有根据外层 OOF 选择模型容量。",
            "",
            "主结果是把 1986–2017 六个互斥测试折拼接后的 D5 Spearman。"
            "D3/D8 只检查相对 K1 的方向。相对 K1 和 B0 的 D5 差值使用"
            f" {config['evaluation']['bootstrap_iterations']} 次按论文 ID 的"
            "配对 bootstrap 计算 95% 区间。未报告条件 Spearman。",
            "",
            "## 11. OOF 结果",
            "",
            _table(
                ("窗口", "模型", "OOF论文", "Spearman"),
                (
                    (
                        f"D{int(row.horizon)}",
                        row.model_id,
                        int(row.n_oof),
                        _fmt(row.spearman_expected),
                    )
                    for row in metrics.sort_values(
                        ["horizon", "model_id"]
                    ).itertuples(index=False)
                ),
            ),
            "",
            "D5 配对比较：",
            "",
            _table(
                (
                    "候选模型",
                    "基线",
                    "Spearman增量",
                    "95%下界",
                    "95%上界",
                    "论文数",
                ),
                (
                    (
                        row.candidate_model_id,
                        row.baseline_model_id,
                        _fmt(row.spearman_gain),
                        _fmt(row.gain_ci_low),
                        _fmt(row.gain_ci_high),
                        int(row.n_papers),
                    )
                    for row in comparisons.itertuples(index=False)
                ),
            ),
            "",
        ]
    )
    if innovation_only is not None:
        innovation_only_rho = float(innovation_only["spearman_expected"])
        k1_rho = float(
            metrics.set_index(["horizon", "model_id"]).loc[
                (5, "k1_controls"), "spearman_expected"
            ]
        )
        combined_rho = float(
            metrics.set_index(["horizon", "model_id"]).loc[
                (5, "final_innovation_plus_k1"),
                "spearman_expected",
            ]
        )
        lines.extend(
            [
                "补充模型仅使用最终冻结的 8 个创新指标，不含 K0/K1/K2"
                " 控制特征，并保持相同 D5 标签、medium 参数和六个时间折。"
                f"其 OOF Spearman 为 **{_fmt(innovation_only_rho)}**；"
                f"K1 单独为 **{_fmt(k1_rho)}**，创新指标＋K1 为"
                f" **{_fmt(combined_rho)}**。该补充结果描述创新指标的"
                "独立预测能力；正式的控制后增量仍应使用"
                " `final_innovation_plus_k1` 相对 `k1_controls` 的配对比较。",
                "",
            ]
        )
    lines.extend(
        [
            "成功门槛：",
            "",
        ]
    )
    for gate, passed in oof["acceptance"]["gates"].items():
        lines.append(f"- `{gate}`：{'通过' if passed else '未通过'}")
    acceptance_label = (
        "全部通过"
        if oof["acceptance"]["all_required_gates_pass"]
        else "存在未通过门槛"
    )
    lines.extend(
        [
            "",
            f"总体判定：**{acceptance_label}**。",
            "",
            "12 大类均保留。领域结果只作异质性报告，不允许因某个领域分数低"
            "而删除该领域。完整领域表位于结果制品的"
            f" `{_manifest_output(oof, 'domain_metrics')}`。",
            "",
            "## 12. 能说什么、不能说什么",
            "",
            "可以说：这些特征是有来源、发表时可计算、通过预先测量门槛的"
            "创新相关证据信号；固定模型对未来学术传播/扩散排序达到本节报告"
            "的 OOF 表现；创新信号相对强控制的增量由配对区间量化。",
            "",
            "不能说：五角度穷尽创新；某一论文的指标值是创新真值；高 OOF"
            " 证明论文更正确、更有社会价值或更应被 Nature 接收；相关性或预测"
            "增量构成因果效应。",
            "",
            "主要限制包括：组合历史和参考重叠背景是 Nature 本地闭包而不是"
            "全球文献全集；Matsumoto 的 WoS 领域步骤被适配为冻结 OpenAlex"
            " 主领域；OpenAlex 当前快照中的分类和作者机构元数据无法证明其历史"
            "版本从未变化；极高施引论文存在返回上限；早期年份可用历史较短；"
            "参考文献缺失会造成结构性不可计算。所有这些限制均保留在结果解释中。",
            "",
            "## 13. 可复现性核验、命令与关键制品",
            "",
        ]
    )
    if reproducibility is not None:
        idempotent_label = (
            "一致"
            if reproducibility["idempotent_manifest_match"]
            else "不一致"
        )
        lines.extend(
            [
                f"复现性核验：**{reproducibility['assessment']}**；核对 "
                f"{reproducibility['n_manifest_outputs_verified']} 个最终输出"
                "哈希，验证 "
                f"{reproducibility['n_checkpoints_verified']} 个折×模型检查点"
                f"及其测试论文集合；重复调用结果清单：**{idempotent_label}**。",
                "",
            ]
        )
        if reproducibility["full_replay_requested"]:
            replay_label = (
                "通过"
                if reproducibility["full_replay_exact_prediction_match"]
                else "未通过"
            )
            lines.extend(
                [
                    "另外从空检查点目录独立重拟合 "
                    f"{reproducibility['n_full_replay_checkpoints']} 个模型折；"
                    "所有 OOF 预测逐字段精确匹配原结果："
                    f"**{replay_label}**。",
                    "",
                ]
            )
    if completion_audit is not None:
        lines.extend(
            [
                "原计划逐项完成审计："
                f"**{completion_audit['n_passed']}/{completion_audit['n_checks']} "
                "项通过**，"
                f"失败项 **{completion_audit['n_failed']}**。该审计逐项检查"
                "检索、来源、五角度、候选家族、筛选门槛、技术债排除、"
                "K0/K1/K2、本地数据边界、v6 不变性、时间折、标签公平性、"
                "结果门槛、12 大类和确定性复演。",
                "",
            ]
        )
    if validation is not None:
        lines.extend(
            [
                f"最终回归测试：**{validation['passed_tests']} 项通过，"
                f"{validation['failed_tests']} 项失败**。唯一警告是 Python "
                "多进程 `fork` 弃用提示；静态检查只声明已实际完成的"
                " `py_compile`，本环境未安装 Ruff/Black/mypy，故不虚报"
                "这些工具的通过结果。",
                "",
            ]
        )
    lines.extend(
        [
            "```bash",
            "python3 scripts/run_nature_v6_1_local.py materialize-overlap",
            "python3 scripts/run_nature_v6_1_local.py screen",
            "python3 scripts/run_nature_v6_1_local.py freeze",
            "python3 scripts/run_nature_v6_1_local.py scan-openalex --workers 12",
            "python3 scripts/run_nature_v6_1_local.py materialize",
            "python3 scripts/run_nature_v6_1_local.py audit",
            "python3 scripts/run_nature_v6_1_local.py oof",
            "python3 scripts/run_v6_1_innovation_only.py",
            "python3 scripts/verify_v6_1_reproducibility.py --full-replay",
            "python3 scripts/audit_v6_1_completion.py",
            "python3 scripts/render_v6_1_evidence_map.py",
            "```",
            "",
            _table(
                ("制品", "路径/ID", "SHA-256或artifact ID"),
                (
                    (
                        "候选目录",
                        catalog_path,
                        sha256_file(catalog_path),
                    ),
                    (
                        "检索日志",
                        _resolve(catalog.search_log_path),
                        catalog.search_log_sha256,
                    ),
                    (
                        "最终指标注册表",
                        registry_path,
                        sha256_file(registry_path),
                    ),
                    (
                        "控制注册表",
                        control_registry_path,
                        sha256_file(control_registry_path),
                    ),
                    (
                        "筛选",
                        analysis_root,
                        screening["artifact_id"],
                    ),
                    (
                        "数据物化",
                        dataset_root,
                        materialization.get("artifact_id", "manifest hash"),
                    ),
                    (
                        "数据审计",
                        analysis_root / "data_quality_report.json",
                        sha256_file(
                            analysis_root / "data_quality_report.json"
                        ),
                    ),
                    (
                        "复现性核验",
                        reproducibility_path,
                        (
                            reproducibility["artifact_id"]
                            if reproducibility is not None
                            else "not generated"
                        ),
                    ),
                    (
                        "原计划完成审计",
                        completion_audit_path,
                        (
                            completion_audit["artifact_id"]
                            if completion_audit is not None
                            else "not generated"
                        ),
                    ),
                    (
                        "最终验证摘要",
                        validation_path,
                        (
                            sha256_file(validation_path)
                            if validation is not None
                            else "not generated"
                        ),
                    ),
                    (
                        "纯创新指标补充模型",
                        (
                            innovation_only_paths[0]
                            if innovation_only_paths
                            else "not generated"
                        ),
                        (
                            innovation_only["artifact_id"]
                            if innovation_only is not None
                            else "not generated"
                        ),
                    ),
                    ("OOF", analysis_root, oof["artifact_id"]),
                ),
            ),
            "",
            "## 14. 候选指标来源表",
            "",
        ]
    )
    source_rows = []
    for source in catalog.sources.values():
        source_rows.append(
            (
                source.source_id,
                source.citation,
                (
                    f"https://doi.org/{source.doi}"
                    if source.doi
                    else source.url
                ),
                "同行评议" if source.peer_reviewed else "仅发现候选",
                source.source_role,
            )
        )
    lines.append(
        _table(
            ("来源ID", "引文", "DOI/URL", "证据状态", "在本项目中的作用"),
            source_rows,
        )
    )
    lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    path = render(args.config.resolve(), args.output.resolve())
    print(path)


if __name__ == "__main__":
    main()

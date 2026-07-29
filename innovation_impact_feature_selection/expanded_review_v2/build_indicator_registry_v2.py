from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
OUTPUT_DIR = ROOT / "outputs"

Candidate = Tuple[str, str, str]


# These are review-mapped indicator concepts, not pre-admitted final features.
# The third value is a canonical redundancy family so aliases and variants can
# later collapse deterministically.
CATALOG: Dict[str, List[Candidate]] = {
    "D01_RECOMBINATIONAL_NOVELTY": [
        ("concept_pair_first_incidence", "首次概念组合占比", "first_pair_incidence"),
        ("concept_pair_historical_rarity", "概念组合历史稀有度", "concept_pair_rarity"),
        ("keyword_pair_first_incidence", "首次关键词组合占比", "first_pair_incidence"),
        ("topic_pair_first_incidence", "首次主题组合占比", "first_pair_incidence"),
        ("entity_pair_first_incidence", "首次实体组合占比", "first_pair_incidence"),
        ("method_problem_combination_novelty", "方法—问题组合新颖性", "method_problem_recombination"),
        ("cited_source_network_distance", "被引来源网络距离", "knowledge_element_distance"),
        ("semantic_nearest_prior_distance", "距最近既有论文的语义距离", "semantic_element_novelty"),
        ("semantic_prior_density_inverse", "既有语义邻域稀疏度", "semantic_local_density"),
        ("topic_distribution_prior_divergence", "与既有主题分布的偏离度", "topic_distribution_novelty"),
        ("entity_relation_graph_novelty", "实体—关系图谱新颖性", "knowledge_graph_novelty"),
        ("cross_domain_recombination_distance", "跨领域知识重组距离", "cross_domain_recombination")
    ],
    "D02_COMBINATION_PROFILE": [
        ("atypical_pair_share", "异常知识对占比", "combination_profile_tail"),
        ("conventional_pair_share", "常规知识对占比", "combination_profile_core"),
        ("combination_profile_dispersion", "组合异常度分布离散性", "combination_profile_shape"),
        ("combination_profile_skewness", "组合异常度分布偏度", "combination_profile_shape"),
        ("combination_profile_tail_weight", "组合异常度尾部权重", "combination_profile_tail"),
        ("core_tail_joint_continuous_score", "常规核心—异常尾部连续联合分数", "core_tail_joint_profile"),
        ("rarest_pair_surprisal", "最稀有组合信息惊奇度", "rare_pair_tail"),
        ("mean_pair_surprisal", "平均组合信息惊奇度", "pair_surprisal")
    ],
    "D03_KNOWLEDGE_DIVERSITY": [
        ("field_shannon_entropy", "参考知识学科香农熵", "field_entropy"),
        ("effective_field_number", "参考知识有效学科数", "field_entropy"),
        ("field_herfindahl_concentration", "参考知识学科集中度", "field_concentration"),
        ("field_gini_simpson_diversity", "参考知识 Gini–Simpson 多样性", "field_simpson"),
        ("reference_source_variety", "被引来源种类数", "reference_source_variety"),
        ("reference_source_entropy", "被引来源分布熵", "reference_source_entropy"),
        ("topic_entropy", "论文主题分布熵", "topic_entropy"),
        ("effective_topic_number", "论文有效主题数", "topic_entropy"),
        ("team_disciplinary_diversity", "团队学科背景多样性", "team_disciplinary_diversity"),
        ("distance_weighted_field_integration", "距离加权跨学科整合度", "rao_stirling_integration"),
        ("field_participation_coefficient", "学科参与系数", "field_participation"),
        ("knowledge_base_coherence", "知识基础内部连贯度", "knowledge_coherence")
    ],
    "D04_PRIOR_KNOWLEDGE_SEARCH": [
        ("price_index_recent_references", "Price 指数（近期参考文献占比）", "reference_recency"),
        ("old_reference_share", "长期经典参考文献占比", "reference_age_tail"),
        ("reference_age_entropy", "参考文献年龄分布熵", "reference_age_distribution"),
        ("reference_age_skewness", "参考文献年龄分布偏度", "reference_age_distribution"),
        ("reference_age_kurtosis", "参考文献年龄分布峰度", "reference_age_distribution"),
        ("oldest_reference_age", "最老参考文献年龄", "reference_age_tail"),
        ("reference_age_span", "参考文献年龄跨度", "reference_age_span"),
        ("prior_reference_popularity_mean", "既有参考文献受关注度均值", "reference_popularity"),
        ("prior_reference_popularity_max", "既有参考文献受关注度最大值", "reference_popularity"),
        ("highly_cited_reference_share", "高影响参考文献占比", "reference_popularity_tail"),
        ("previously_uncited_reference_share", "此前未被引参考文献占比", "reference_obscurity"),
        ("author_self_citation_share_t0", "作者自引占比（发表时）", "self_citation"),
        ("international_reference_share", "跨国知识来源占比", "reference_geographic_reach"),
        ("field_normalized_reference_impact", "参考文献学科归一化既有影响", "reference_popularity")
    ],
    "D05_TOPIC_MOMENTUM": [
        ("topic_prior_annual_volume", "主题上一年度发文量", "topic_volume"),
        ("topic_compound_growth_rate", "主题历史复合增长率", "topic_growth"),
        ("topic_burst_strength", "主题突现强度", "topic_burst"),
        ("topic_burst_recency", "主题最近突现时间", "topic_burst"),
        ("topic_lifecycle_phase", "主题生命周期阶段", "topic_lifecycle"),
        ("topic_momentum_composite", "主题规模—增长联合动量", "topic_momentum"),
        ("topic_volume_volatility", "主题发文量波动性", "topic_volatility"),
        ("topic_competition_density", "主题内竞争密度", "topic_competition"),
        ("topic_active_author_count", "主题既有活跃作者数", "topic_participant_scale"),
        ("topic_active_institution_count", "主题既有活跃机构数", "topic_participant_scale"),
        ("topic_interdisciplinarity_prior", "主题既有跨学科程度", "topic_interdisciplinarity"),
        ("topic_semantic_drift_rate", "主题语义漂移速度", "topic_semantic_drift")
    ],
    "D06_TEAM_REACH": [
        ("team_prior_output_total", "团队历史发文总量", "team_prior_productivity"),
        ("team_prior_output_mean", "团队成员历史发文均值", "team_prior_productivity"),
        ("team_prior_output_max", "团队成员历史发文最大值", "team_prior_productivity"),
        ("team_recent_output_total", "团队近期发文总量", "team_recent_productivity"),
        ("team_prior_citation_impact_mean", "团队成员历史影响均值", "team_prior_impact"),
        ("team_prior_citation_impact_max", "团队成员历史影响最大值", "team_prior_impact"),
        ("team_career_age_mean", "团队平均学术年龄", "team_career_age"),
        ("team_career_age_max", "团队最大学术年龄", "team_career_age"),
        ("team_prior_unique_coauthor_count", "团队既有独立合作者数", "team_collaboration_reach"),
        ("repeat_collaboration_share", "既有重复合作关系占比", "team_freshness"),
        ("new_collaboration_share", "首次合作关系占比", "team_freshness"),
        ("team_freshness_index", "团队关系新鲜度", "team_freshness"),
        ("institutional_diversity", "团队机构多样性", "institutional_reach"),
        ("collaboration_geographic_distance", "团队合作地理跨度", "geographic_reach"),
        ("author_expertise_diversity", "团队成员专长多样性", "team_disciplinary_diversity"),
        ("coauthor_network_constraint", "团队既有合作网络约束", "coauthor_brokerage"),
        ("coauthor_brokerage_score", "团队既有合作网络中介性", "coauthor_brokerage")
    ],
    "D07_KNOWLEDGE_NETWORK_POSITION": [
        ("bc_weighted_degree_t0", "文献耦合网络加权度", "bc_degree"),
        ("bc_betweenness_t0", "文献耦合网络介数中心性", "bc_betweenness"),
        ("bc_eigenvector_t0", "文献耦合网络特征向量中心性", "bc_spectral_centrality"),
        ("bc_pagerank_t0", "文献耦合网络 PageRank", "bc_spectral_centrality"),
        ("bc_kcore_t0", "文献耦合网络 k-core 层级", "bc_core_position"),
        ("bc_constraint_t0", "文献耦合网络结构洞约束", "bc_structural_holes"),
        ("bc_participation_coefficient_t0", "跨网络社群参与系数", "bc_community_bridge"),
        ("bc_neighbor_community_count_t0", "相邻知识社群数", "bc_community_bridge"),
        ("bc_bridging_centrality_t0", "知识网络桥接中心性", "bc_community_bridge"),
        ("bc_eccentricity_t0", "文献耦合网络离心率", "bc_distance_position"),
        ("bc_local_density_t0", "文献耦合局部网络密度", "bc_local_density"),
        ("bc_component_size_t0", "文献耦合连通分量规模", "bc_component_position"),
        ("bc_novel_edge_share_t0", "与既有知识网络新增边占比", "bc_edge_novelty")
    ],
    "D08_PUBLICATION_VISIBILITY": [
        ("title_character_count", "题名字符数", "title_length"),
        ("title_punctuation_count", "题名标点数量", "title_structure"),
        ("title_question_flag", "疑问式题名", "title_structure"),
        ("title_colon_flag", "冒号式题名", "title_structure"),
        ("title_acronym_count", "题名缩略语数量", "title_lexical_accessibility"),
        ("title_conclusiveness_score", "题名结论性表达强度", "title_claim_style"),
        ("title_specificity_score", "题名具体性", "title_semantics"),
        ("title_keyword_overlap", "题名—关键词重合度", "keyword_discoverability"),
        ("author_keyword_count", "作者关键词数量", "keyword_discoverability"),
        ("abstract_lexical_diversity", "摘要词汇多样性", "abstract_lexical_style"),
        ("abstract_lexical_complexity", "摘要词汇复杂度", "abstract_readability"),
        ("abstract_common_word_share", "摘要常用词占比", "abstract_readability"),
        ("abstract_sentiment_score", "摘要情感倾向", "abstract_claim_style"),
        ("structured_abstract_flag", "结构式摘要", "abstract_structure"),
        ("article_page_count", "论文页数", "article_length"),
        ("figure_count", "图数量", "article_visual_structure"),
        ("table_count", "表数量", "article_visual_structure"),
        ("equation_count", "公式数量", "article_formal_structure"),
        ("publication_language", "论文语言", "language_accessibility"),
        ("publication_month", "发表月份", "publication_timing")
    ],
    "C01_FIELD_TIME_CONTEXT": [
        ("field_year_publication_volume", "学科—年份发文规模", "field_time_scale"),
        ("field_prior_citation_density", "学科既有引文密度", "field_citation_opportunity"),
        ("field_reference_count_norm", "学科参考文献数量基线", "field_reference_practice"),
        ("field_database_coverage_rate", "学科数据库覆盖率", "database_coverage_context"),
        ("subfield_maturity_at_t0", "子领域发表时成熟度", "field_maturity")
    ],
    "X01_OPEN_REPRODUCIBILITY": [
        ("permissive_license_at_publication", "发表时开放许可类型", "open_license"),
        ("materials_shared_at_publication", "发表时研究材料共享", "open_materials"),
        ("protocol_shared_at_publication", "发表时研究方案公开", "open_protocol"),
        ("preregistered_before_publication", "发表前预注册", "preregistration"),
        ("registered_report_flag", "注册报告", "registered_report"),
        ("reporting_checklist_available", "报告规范清单公开", "reporting_transparency"),
        ("repository_persistent_identifier", "数据或代码仓储持久标识", "repository_persistence"),
        ("reproducibility_badge_at_publication", "发表时可复现性徽章", "reproducibility_badge")
    ],
    "X02_METHOD_EVIDENCE_STRENGTH": [
        ("statistical_power_reported", "统计功效或功效分析", "statistical_power"),
        ("effect_size_reported", "效应量报告", "effect_reporting"),
        ("uncertainty_interval_reported", "不确定性区间报告", "uncertainty_reporting"),
        ("randomization_reported", "随机化设计", "design_rigor"),
        ("blinding_reported", "盲法设计", "design_rigor"),
        ("control_group_present", "对照组设置", "design_rigor"),
        ("multisite_study_flag", "多中心研究", "study_scope"),
        ("replication_study_flag", "重复验证研究", "replication"),
        ("sample_representativeness_score", "样本代表性", "sample_validity"),
        ("robustness_check_count", "稳健性检验范围", "robustness"),
        ("limitations_disclosure_score", "局限性披露完整度", "reporting_transparency"),
        ("independent_study_count", "论文内独立研究数量", "evidence_multiplicity")
    ]
}


REVIEW_SOURCES: Dict[str, List[str]] = {
    "D01_RECOMBINATIONAL_NOVELTY": [
        "BAI2025_REVIEW",
        "PARK2026_REVIEW",
        "YIN2023"
    ],
    "D02_COMBINATION_PROFILE": [
        "PARK2026_REVIEW",
        "UZZI2013",
        "BORNMANN2019_NOVELTY_VALIDATION"
    ],
    "D03_KNOWLEDGE_DIVERSITY": [
        "BAI2025_REVIEW",
        "KOUSHA2024_REVIEW",
        "STIRLING2007"
    ],
    "D04_PRIOR_KNOWLEDGE_SEARCH": [
        "KOUSHA2024_REVIEW",
        "XIA2023_REVIEW",
        "TAHAMTAN2016"
    ],
    "D05_TOPIC_MOMENTUM": [
        "BAI2025_REVIEW",
        "XIA2023_REVIEW",
        "YAN2014"
    ],
    "D06_TEAM_REACH": [
        "KOUSHA2024_REVIEW",
        "BAI2025_REVIEW",
        "WUCHTY2007"
    ],
    "D07_KNOWLEDGE_NETWORK_POSITION": [
        "BAI2025_REVIEW",
        "PARK2026_REVIEW",
        "COLLADON2020"
    ],
    "D08_PUBLICATION_VISIBILITY": [
        "KOUSHA2024_REVIEW",
        "BAI2025_REVIEW",
        "LETCHFORD2015_TITLE"
    ],
    "C01_FIELD_TIME_CONTEXT": [
        "KOUSHA2024_REVIEW",
        "XIA2023_REVIEW",
        "BORNMANN_DANIEL2008"
    ],
    "X01_OPEN_REPRODUCIBILITY": [
        "KOUSHA2024_REVIEW",
        "DAVIS2008_OA",
        "PIWOWAR2013"
    ],
    "X02_METHOD_EVIDENCE_STRENGTH": [
        "KOUSHA2024_REVIEW",
        "BAI2025_REVIEW",
        "TAHAMTAN2019"
    ]
}


DATA_REQUIREMENTS: Dict[str, List[str]] = {
    "D01_RECOMBINATIONAL_NOVELTY": [
        "publication-time paper content or references",
        "strictly prior comparison corpus",
        "versioned knowledge units and mapping"
    ],
    "D02_COMBINATION_PROFILE": [
        "focal knowledge-element pairs",
        "strictly prior co-occurrence marginals",
        "frozen null model"
    ],
    "D03_KNOWLEDGE_DIVERSITY": [
        "publication-time references, topics, or author fields",
        "versioned category mapping",
        "frozen category-distance matrix when required"
    ],
    "D04_PRIOR_KNOWLEDGE_SEARCH": [
        "focal reference list",
        "strictly prior reference metadata",
        "frozen author identities when required"
    ],
    "D05_TOPIC_MOMENTUM": [
        "focal topic assignment",
        "strictly prior topic-year corpus",
        "versioned topic model or category mapping"
    ],
    "D06_TEAM_REACH": [
        "publication-time author and affiliation identities",
        "strictly prior author histories",
        "frozen entity disambiguation"
    ],
    "D07_KNOWLEDGE_NETWORK_POSITION": [
        "focal reference list",
        "strictly prior paper-reference graph",
        "frozen network construction rule"
    ],
    "D08_PUBLICATION_VISIBILITY": [
        "publication-time title, abstract, keywords, and document metadata",
        "versioned text parser",
        "frozen document representation"
    ],
    "C01_FIELD_TIME_CONTEXT": [
        "publication year",
        "frozen field assignment",
        "strictly prior field-year corpus"
    ],
    "X01_OPEN_REPRODUCIBILITY": [
        "historically versioned publication-time openness metadata",
        "persistent repository identifiers",
        "publication-time timestamp audit"
    ],
    "X02_METHOD_EVIDENCE_STRENGTH": [
        "complete publication-time full text",
        "domain-specific validated extraction rule",
        "versioned method ontology"
    ]
}


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
    fields: Sequence[str],
) -> None:
    """Write a deterministic UTF-8 CSV table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def ensure_unique(
    rows: Iterable[Mapping[str, Any]],
    key: str,
    label: str,
) -> None:
    """Require unique nonempty identifiers."""
    values = [str(row.get(key, "")) for row in rows]
    duplicates = sorted(
        value
        for value, count in Counter(values).items()
        if value and count > 1
    )
    if any(not value for value in values) or duplicates:
        raise ValueError(
            f"Invalid {label} {key}; duplicates={duplicates}"
        )


def candidate_features(
    dimensions: Mapping[str, Mapping[str, Any]],
    start_index: int,
) -> List[Dict[str, Any]]:
    """Materialize review-mapped concepts as explicitly unverified candidates."""
    rows: List[Dict[str, Any]] = []
    next_index = start_index
    for dimension_id, candidates in CATALOG.items():
        dimension = dimensions[dimension_id]
        if dimension["block"] == "innovation_evidence":
            scope_role = "direct_innovation"
        elif dimension["block"] == "substantive_potential":
            scope_role = "t0_substantive"
        elif dimension["block"] == "opportunity_visibility":
            scope_role = "t0_opportunity"
        elif dimension["block"] == "context_control":
            scope_role = "context_control"
        elif dimension_id == "X01_OPEN_REPRODUCIBILITY":
            scope_role = "t0_opportunity"
        else:
            scope_role = "candidate_only"
        for name, label_zh, redundancy_family in candidates:
            rows.append(
                {
                    "feature_id": f"F{next_index:03d}",
                    "name": name,
                    "label_zh": label_zh,
                    "dimension_id": dimension_id,
                    "block": dimension["block"],
                    "scope_role": scope_role,
                    "feature_type": "candidate_family_pending_definition",
                    "formula": (
                        "Not authorized at discovery stage. A primary source "
                        "must define the exact article-level calculation, "
                        "parameters, units, and temporal boundary."
                    ),
                    "direction": (
                        "Not frozen until primary-source construct validation."
                    ),
                    "maximum_information_time": (
                        "Intended for publication time; exact T0 audit pending."
                    ),
                    "missing_rule": (
                        "Unavailable until a verified formula and all frozen "
                        "inputs are registered."
                    ),
                    "required_data": DATA_REQUIREMENTS[dimension_id],
                    "source_ids": REVIEW_SOURCES[dimension_id],
                    "evidence_grade": "none",
                    "evidence_summary": (
                        "The concept is mapped from peer-reviewed reviews or "
                        "adjacent primary studies, but the exact family has "
                        "not yet passed primary-source formula verification."
                    ),
                    "data_status": "unavailable",
                    "data_reference": "",
                    "bias_policy": (
                        "allowed_context"
                        if dimension["block"] == "context_control"
                        else (
                            "allowed_opportunity"
                            if dimension["block"]
                            in {"opportunity_visibility", "candidate_unready"}
                            and dimension_id != "X02_METHOD_EVIDENCE_STRENGTH"
                            else "allowed_core"
                        )
                    ),
                    "redundancy_family": redundancy_family,
                    "selection_priority": 9,
                    "peer_reviewed_evidence": false_value(),
                    "formula_reproducible": false_value(),
                    "t0_computable": True,
                    "requires_future": False,
                    "fatal_validity_concern": False,
                    "uses_outcome_for_selection": False,
                    "noninvariant": True,
                    "quality_audit_status": "not_applicable",
                    "candidate_stage": (
                        "review_mapped_pending_primary_verification"
                    ),
                    "promotion_requirements": [
                        "one named primary or foundational formula source",
                        "exact formula, units, parameters, and missing rule",
                        "publication-time leakage audit",
                        "local input availability audit",
                        "construct-validity and bias decision",
                        "alias and redundancy adjudication"
                    ]
                }
            )
            next_index += 1
    return rows


def false_value() -> bool:
    """Return False while keeping candidate construction visually explicit."""
    return False


def coding_rows(
    candidates: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Create a two-reviewer promotion template for every new candidate."""
    rows: List[Dict[str, Any]] = []
    for candidate in candidates:
        for reviewer_id in ("reviewer_1", "reviewer_2"):
            rows.append(
                {
                    "feature_id": candidate["feature_id"],
                    "candidate_name": candidate["name"],
                    "label_zh": candidate["label_zh"],
                    "dimension_id": candidate["dimension_id"],
                    "reviewer_id": reviewer_id,
                    "primary_source_doi": "",
                    "formula_location": "",
                    "formula_verified": "",
                    "article_level": "",
                    "t0_computable": "",
                    "requires_future": "",
                    "construct_valid": "",
                    "data_ready": "",
                    "canonical_family": candidate["redundancy_family"],
                    "decision": "",
                    "exclusion_reason": "",
                    "notes": ""
                }
            )
    return rows


def evidence_table_rows(
    records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Flatten evidence records for direct human audit."""
    rows: List[Dict[str, Any]] = []
    for record in sorted(records, key=lambda value: str(value["source_id"])):
        rows.append(
            {
                "source_id": record["source_id"],
                "year": record["year"],
                "citation": record["citation"],
                "doi": record["doi"],
                "url": record["url"],
                "study_type": record["study_type"],
                "peer_reviewed": record["peer_reviewed"],
                "research_group": record["research_group"],
                "dimension_ids": "|".join(record["dimension_ids"]),
                "evidence_roles": "|".join(record["evidence_roles"]),
                "formula_authorization": "|".join(
                    record["formula_authorization"]
                ),
                "evidence_direction": record["evidence_direction"],
                "finding": record["finding"],
                "limitations": record["limitations"],
                "dimension_admission_evidence": record[
                    "dimension_admission_evidence"
                ],
            }
        )
    return rows


def indicator_table_rows(
    records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Flatten all baseline and expanded indicator records."""
    rows: List[Dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "feature_id": record["feature_id"],
                "name": record["name"],
                "label_zh": record["label_zh"],
                "dimension_id": record["dimension_id"],
                "block": record["block"],
                "scope_role": record["scope_role"],
                "candidate_stage": record.get(
                    "candidate_stage",
                    "baseline_screened_family",
                ),
                "feature_type": record["feature_type"],
                "formula": record["formula"],
                "direction": record["direction"],
                "maximum_information_time": record[
                    "maximum_information_time"
                ],
                "required_data": "|".join(record["required_data"]),
                "source_ids": "|".join(record["source_ids"]),
                "evidence_grade": record["evidence_grade"],
                "evidence_summary": record["evidence_summary"],
                "data_status": record["data_status"],
                "bias_policy": record["bias_policy"],
                "redundancy_family": record["redundancy_family"],
                "selection_priority": record["selection_priority"],
            }
        )
    return rows


def main() -> None:
    """Build a versioned expanded evidence table and feature registry."""
    base_registry = read_json(PARENT / "feature_registry.json")
    base_evidence = read_json(PARENT / "literature_evidence.json")
    dimensions_payload = read_json(PARENT / "dimensions.json")
    additional_evidence = read_json(
        ROOT / "additional_literature_evidence.json"
    )
    dimension_lookup = {
        str(row["dimension_id"]): row
        for row in dimensions_payload["dimensions"]
    }
    missing_dimensions = sorted(set(CATALOG) - set(dimension_lookup))
    if missing_dimensions:
        raise ValueError(f"Unknown catalog dimensions: {missing_dimensions}")
    base_features = list(base_registry["features"])
    additions = candidate_features(
        dimension_lookup,
        start_index=len(base_features) + 1,
    )
    if len(additions) != 133:
        raise ValueError(
            f"Expected 133 review-mapped concepts, found {len(additions)}"
        )
    merged_evidence = [
        *base_evidence["records"],
        *additional_evidence["records"],
    ]
    merged_features = [*base_features, *additions]
    ensure_unique(merged_evidence, "source_id", "evidence")
    ensure_unique(merged_features, "feature_id", "features")
    ensure_unique(merged_features, "name", "features")
    known_sources = {
        str(record["source_id"]) for record in merged_evidence
    }
    unknown_sources = sorted(
        {
            str(source_id)
            for feature in merged_features
            for source_id in feature["source_ids"]
            if str(source_id) not in known_sources
        }
    )
    if unknown_sources:
        raise ValueError(f"Unknown feature source IDs: {unknown_sources}")
    evidence_output = {
        "schema_version": "2.0.0",
        "cutoff_date": base_evidence["cutoff_date"],
        "interpretation": {
            **base_evidence["interpretation"],
            "expanded_review_rule": (
                "Review-mapped concepts remain ineligible until their exact "
                "formula is verified in a primary or foundational source."
            ),
        },
        "records": merged_evidence,
    }
    registry_output = {
        "schema_version": "2.0.0",
        "registry_id": (
            "aspr-innovation-impact-t0-expanded-feature-census-2026-07-28"
        ),
        "unit_of_count": base_registry["unit_of_count"],
        "stage_a_unit": (
            "One source-mentioned indicator concept before alias, formula, "
            "parameter, and redundancy adjudication."
        ),
        "candidate_defaults": base_registry["candidate_defaults"],
        "features": merged_features,
    }
    summary = {
        "schema_version": "2.0.0",
        "baseline_verified_or_screened_feature_families": len(base_features),
        "new_review_mapped_indicator_concepts": len(additions),
        "combined_registry_records": len(merged_features),
        "baseline_evidence_records": len(base_evidence["records"]),
        "additional_evidence_records": len(
            additional_evidence["records"]
        ),
        "combined_evidence_records": len(merged_evidence),
        "candidate_counts_by_dimension": dict(
            sorted(
                Counter(
                    str(row["dimension_id"]) for row in additions
                ).items()
            )
        ),
        "admission_warning": (
            "The 133 additions expand the discovery universe; they are not "
            "final training features and deliberately fail primary-evidence "
            "and formula-reproducibility gates until dual review promotes them."
        ),
        "quantity_rule": (
            "No final count is targeted. Full-text coding may add, merge, "
            "rename, split, or exclude concepts with a recorded reason."
        )
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_DIR / "literature_evidence_v2.json", evidence_output)
    write_json(OUTPUT_DIR / "feature_registry_v2.json", registry_output)
    write_json(OUTPUT_DIR / "registry_build_summary.json", summary)
    write_csv(
        OUTPUT_DIR / "literature_evidence_table_v2.csv",
        evidence_table_rows(merged_evidence),
        (
            "source_id",
            "year",
            "citation",
            "doi",
            "url",
            "study_type",
            "peer_reviewed",
            "research_group",
            "dimension_ids",
            "evidence_roles",
            "formula_authorization",
            "evidence_direction",
            "finding",
            "limitations",
            "dimension_admission_evidence",
        ),
    )
    write_csv(
        OUTPUT_DIR / "indicator_catalog_v2.csv",
        indicator_table_rows(merged_features),
        (
            "feature_id",
            "name",
            "label_zh",
            "dimension_id",
            "block",
            "scope_role",
            "candidate_stage",
            "feature_type",
            "formula",
            "direction",
            "maximum_information_time",
            "required_data",
            "source_ids",
            "evidence_grade",
            "evidence_summary",
            "data_status",
            "bias_policy",
            "redundancy_family",
            "selection_priority",
        ),
    )
    write_csv(
        OUTPUT_DIR / "indicator_promotion_template.csv",
        coding_rows(additions),
        (
            "feature_id",
            "candidate_name",
            "label_zh",
            "dimension_id",
            "reviewer_id",
            "primary_source_doi",
            "formula_location",
            "formula_verified",
            "article_level",
            "t0_computable",
            "requires_future",
            "construct_valid",
            "data_ready",
            "canonical_family",
            "decision",
            "exclusion_reason",
            "notes"
        ),
    )
    print(
        f"Built {len(merged_features)} registry rows: {len(base_features)} "
        f"baseline families + {len(additions)} review-mapped concepts."
    )


if __name__ == "__main__":
    main()

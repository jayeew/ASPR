"""
comparator.py — 诺贝尔奖前后指标对比与领域知识图谱前后演化可视化
"""

import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from metrics import (
    _build_journal_copair_baseline,
    compute_all_metrics_for_paper,
    compute_betweenness,
    compute_delta_q,
    compute_modularity,
)

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NobelCase:
    """单篇诺奖案例配置。"""

    name: str
    paper_id: str
    paper_doi: str
    nobel_year: int
    field: str


@dataclass
class ComparisonResult:
    """单篇论文七维指标前后对比结果。"""

    case: NobelCase
    metrics_before: dict
    metrics_after: dict
    delta_q_result: dict
    graph_stats: dict


@dataclass
class FieldContrastSpec:
    """领域图谱时间节点前后对比配置。"""

    filter_query: str
    event_year: int
    event_label: str
    target_paper_ids: list[str] = field(default_factory=list)
    target_paper_titles: list[str] = field(default_factory=list)
    before_years: int = 10
    after_years: int = 5
    max_plot_nodes: int = 180
    min_community_size: int = 8
    slug: Optional[str] = None


@dataclass
class FieldContrastResult:
    """领域图谱前后对比的完整结果。"""

    spec: FieldContrastSpec
    before_graph: nx.DiGraph
    after_graph: nx.DiGraph
    delta_q_result: dict
    graph_stats: dict
    community_rows: list[dict]
    emergent_communities: list[dict]
    selected_nodes: list[str]
    output_paths: dict[str, str] = field(default_factory=dict)


# 指标元数据：(显示名, 期望方向 True=奖后应升 False=奖后应降, 主键)
METRIC_META = [
    ("桥接中心性 B", True, "betweenness"),
    ("RS 跨学科性", True, "rao_stirling"),
    ("模块度 Q", False, "delta_q"),
    ("Uzzi 新颖性 p10", False, "uzzi_novelty_p10"),
    ("RTD 引用多样性", True, "rtd_rtd"),
    ("Burt 创新潜力", True, "burt_innovation_potential"),
    ("PDE 预期扩散熵", True, "pde_pde_norm"),
]

METRIC_PLOT_LABELS_EN = {
    "betweenness": "Bridging Centrality B",
    "rao_stirling": "RS Interdisciplinarity",
    "delta_q": "Modularity Q",
    "uzzi_novelty_p10": "Uzzi Novelty p10",
    "rtd_rtd": "RTD Citation Diversity",
    "burt_innovation_potential": "Burt Innovation Potential",
    "pde_pde_norm": "PDE Diffusion Entropy",
}

_CJK_FONT_CANDIDATES = [
    "Noto Sans CJK SC",
    "Noto Sans SC",
    "Source Han Sans SC",
    "Source Han Sans CN",
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "WenQuanYi Zen Hei",
    "AR PL UKai CN",
    "AR PL UMing CN",
]

_NEW_COMMUNITY_FILL = "#E76F51"
_NEW_COMMUNITY_EDGE = "#111827"
_FALLBACK_COMMUNITY_FILL = "#94A3B8"


def _configure_plot_fonts() -> bool:
    """
    配置 Matplotlib 字体；若缺少中文字体，则回退英文标签。

    Returns:
        bool: 是否找到可用中文字体。
    """
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in _CJK_FONT_CANDIDATES:
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans", "sans-serif"]
            plt.rcParams["axes.unicode_minus"] = False
            return True

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    log.warning("未检测到中文字体，图表标签将自动使用英文以避免缺字警告。")
    return False


HAS_CJK_FONT = _configure_plot_fonts()


def _metric_plot_label(cn_label: str, key: str) -> str:
    """
    返回绘图用指标名；无中文字体时自动回退英文。

    Args:
        cn_label: 中文显示名。
        key: 指标键名。

    Returns:
        str: 实际用于图表的标签。
    """
    if HAS_CJK_FONT:
        return cn_label
    return METRIC_PLOT_LABELS_EN.get(key, key)


def _plot_phase_labels() -> tuple[str, str]:
    """
    返回图表中的前后阶段标签。

    Returns:
        tuple[str, str]: 奖前/奖后或 Before/After。
    """
    if HAS_CJK_FONT:
        return ("奖前", "奖后")
    return ("Before", "After")


def _weak_difference_message() -> str:
    """
    返回“差异较弱”提示文本，自动适配字体环境。

    Returns:
        str: 可直接用于图表的提示文本。
    """
    if HAS_CJK_FONT:
        return "未检测到显著新社群 / Difference is weak."
    return "No significant emergent community detected."


def _slugify(value: str) -> str:
    """
    生成稳定的文件名 slug。

    Args:
        value: 任意字符串。

    Returns:
        str: 仅含小写字母、数字和下划线的 slug。
    """
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip().lower()).strip("_")
    return slug or "field_contrast"


def _safe_growth(before_value: int, after_value: int) -> Optional[float]:
    """
    计算增长率。

    Args:
        before_value: 前期值。
        after_value: 后期值。

    Returns:
        Optional[float]: 增长率；若前值为 0 则返回 None。
    """
    if before_value <= 0:
        return None
    return (after_value - before_value) / before_value


def _community_members(partition: dict) -> dict[int, set[str]]:
    """
    将 node -> community 映射转为 community -> nodes。

    Args:
        partition: 社区划分。

    Returns:
        dict[int, set[str]]: 每个社区的节点集合。
    """
    members: dict[int, set[str]] = defaultdict(set)
    for node_id, community_id in partition.items():
        members[int(community_id)].add(node_id)
    return dict(members)


def _split_topic_names(raw_value: Any) -> list[str]:
    """
    标准化 topic_names 字段。

    Args:
        raw_value: 节点属性中的 topic_names。

    Returns:
        list[str]: 主题名列表。
    """
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        return [item.strip() for item in raw_value.split("|") if item.strip()]
    if isinstance(raw_value, (list, tuple, set)):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    return [str(raw_value).strip()] if str(raw_value).strip() else []


def _top_labels(
    G: nx.DiGraph,
    node_ids: set[str],
    attr_name: str,
    top_k: int = 3,
) -> list[str]:
    """
    统计节点集合在某个属性上的高频标签。

    Args:
        G: 知识图谱。
        node_ids: 节点集合。
        attr_name: 属性名。
        top_k: 返回前 k 项。

    Returns:
        list[str]: 排序后的标签列表。
    """
    counter: Counter = Counter()
    for node_id in node_ids:
        attrs = G.nodes[node_id]
        if attr_name == "topic_names":
            counter.update(_split_topic_names(attrs.get(attr_name)))
        else:
            label = str(attrs.get(attr_name, "")).strip()
            if label:
                counter[label] += 1
    return [label for label, _ in counter.most_common(top_k)]


def _community_brief(G: nx.DiGraph, node_ids: set[str]) -> tuple[str, str]:
    """
    返回社区的主导 field / topic 概述。

    Args:
        G: 知识图谱。
        node_ids: 社区节点集合。

    Returns:
        tuple[str, str]: 主导 field 与主导 topic。
    """
    field_labels = _top_labels(G, node_ids, "field", top_k=2)
    topic_labels = _top_labels(G, node_ids, "topic_names", top_k=3)

    dominant_field = " / ".join(field_labels) if field_labels else "Unknown field"
    dominant_topic = " / ".join(topic_labels) if topic_labels else "Unknown topic"
    return dominant_field, dominant_topic


def _cross_community_edge_ratio(G: nx.DiGraph, partition: dict) -> float:
    """
    计算跨社区边占比。

    Args:
        G: 知识图谱。
        partition: 社区划分。

    Returns:
        float: 跨社区边比例。
    """
    if not partition:
        return 0.0

    total_edges = 0
    cross_edges = 0
    for source, target in G.edges():
        if source not in partition or target not in partition:
            continue
        total_edges += 1
        if partition[source] != partition[target]:
            cross_edges += 1

    if total_edges == 0:
        return 0.0
    return cross_edges / total_edges


def _analyze_community_shifts(
    spec: FieldContrastSpec,
    G_before: nx.DiGraph,
    G_after: nx.DiGraph,
    partition_before: dict,
    partition_after: dict,
) -> tuple[list[dict], list[dict]]:
    """
    分析 after 社区相对 before 的继承与涌现关系。

    Args:
        spec: 对比配置。
        G_before: 事件前图谱。
        G_after: 事件后图谱。
        partition_before: 前期社区划分。
        partition_after: 后期社区划分。

    Returns:
        tuple[list[dict], list[dict]]: 全部社区记录与显著新社群列表。
    """
    before_members = _community_members(partition_before)
    after_members = _community_members(partition_after)
    before_nodes = set(G_before.nodes)
    rows: list[dict] = []

    for after_community_id, after_node_ids in sorted(
        after_members.items(),
        key=lambda item: len(item[1]),
        reverse=True,
    ):
        best_before_id: Optional[int] = None
        best_jaccard = 0.0
        best_overlap = 0
        best_before_size = 0

        for before_community_id, before_node_ids in before_members.items():
            overlap_size = len(after_node_ids & before_node_ids)
            if overlap_size == 0:
                continue
            union_size = len(after_node_ids | before_node_ids)
            jaccard = overlap_size / union_size if union_size else 0.0
            if (jaccard > best_jaccard) or (
                math.isclose(jaccard, best_jaccard) and overlap_size > best_overlap
            ):
                best_before_id = before_community_id
                best_jaccard = jaccard
                best_overlap = overlap_size
                best_before_size = len(before_node_ids)

        post_event_nodes = {
            node_id
            for node_id in after_node_ids
            if (G_after.nodes[node_id].get("year") or 0) >= spec.event_year
        }
        pre_event_nodes = after_node_ids - post_event_nodes
        post_event_share = len(post_event_nodes) / len(after_node_ids) if after_node_ids else 0.0
        dominant_field, dominant_topic = _community_brief(G_after, after_node_ids)
        significant_new = (
            best_jaccard < 0.25
            and post_event_share >= 0.60
            and len(after_node_ids) >= spec.min_community_size
        )

        if significant_new:
            status = "new"
        elif post_event_share >= 0.30:
            status = "expanded"
        else:
            status = "inherited"

        rows.append({
            "after_community_id": after_community_id,
            "status": status,
            "is_new": significant_new,
            "matched_before_community_id": best_before_id,
            "after_size": len(after_node_ids),
            "matched_before_size": best_before_size,
            "overlap_with_before": best_overlap,
            "jaccard_with_before": best_jaccard,
            "post_event_nodes": len(post_event_nodes),
            "pre_event_nodes": len(pre_event_nodes),
            "post_event_share": post_event_share,
            "dominant_field": dominant_field,
            "dominant_topic": dominant_topic,
            "before_only_overlap_nodes": len(pre_event_nodes & before_nodes),
        })

    emergent_rows = [row for row in rows if row["is_new"]]
    emergent_rows.sort(key=lambda item: item["after_size"], reverse=True)
    return rows, emergent_rows


def _choose_communities_for_plot(
    community_rows: list[dict],
    max_plot_nodes: int,
    min_community_nodes: int,
) -> list[dict]:
    """
    选择纳入可视化的社区集合。

    Args:
        community_rows: 社区统计记录。
        max_plot_nodes: 最大绘图节点数。
        min_community_nodes: 每个社区最少保留节点数。

    Returns:
        list[dict]: 需要进入绘图的社区记录。
    """
    if not community_rows:
        return []

    sorted_rows = sorted(
        community_rows,
        key=lambda row: (not row["is_new"], -row["after_size"]),
    )

    chosen_rows: list[dict] = []
    reserved_budget = 0
    for row in sorted_rows:
        if not chosen_rows:
            chosen_rows.append(row)
            reserved_budget += min(min_community_nodes, row["after_size"])
            continue

        next_budget = reserved_budget + min(min_community_nodes, row["after_size"])
        if next_budget <= max_plot_nodes:
            chosen_rows.append(row)
            reserved_budget = next_budget

    return chosen_rows


def _allocate_community_quotas(
    chosen_rows: list[dict],
    max_plot_nodes: int,
    min_community_nodes: int,
) -> dict[int, int]:
    """
    为选中的社区分配绘图节点预算。

    Args:
        chosen_rows: 需要绘图的社区。
        max_plot_nodes: 最大节点数。
        min_community_nodes: 每个社区的最低节点数。

    Returns:
        dict[int, int]: community_id -> quota。
    """
    if not chosen_rows:
        return {}

    quota_floor = max(1, min_community_nodes)
    if sum(min(quota_floor, row["after_size"]) for row in chosen_rows) > max_plot_nodes:
        quota_floor = max(1, max_plot_nodes // max(len(chosen_rows), 1))

    quotas = {
        row["after_community_id"]: min(quota_floor, row["after_size"])
        for row in chosen_rows
    }
    remaining_budget = max_plot_nodes - sum(quotas.values())

    if remaining_budget <= 0:
        return quotas

    weighted_rows = []
    total_weight = 0.0
    for row in chosen_rows:
        weight = float(row["after_size"]) * (1.5 if row["is_new"] else 1.0)
        weighted_rows.append((row["after_community_id"], row["after_size"], weight))
        total_weight += weight

    remainders: list[tuple[float, int, int]] = []
    for community_id, community_size, weight in weighted_rows:
        proportional = remaining_budget * (weight / total_weight) if total_weight else 0.0
        extra = int(math.floor(proportional))
        capacity = max(0, community_size - quotas[community_id])
        extra = min(extra, capacity)
        quotas[community_id] += extra
        remainders.append((proportional - extra, community_id, capacity - extra))

    assigned = sum(quotas.values())
    slots_left = max_plot_nodes - assigned
    for _, community_id, capacity_left in sorted(remainders, reverse=True):
        if slots_left <= 0:
            break
        if capacity_left <= 0:
            continue
        quotas[community_id] += 1
        slots_left -= 1

    return quotas


def _select_plot_nodes(
    G_after: nx.DiGraph,
    partition_after: dict,
    community_rows: list[dict],
    max_plot_nodes: int,
    min_community_nodes: int,
) -> list[str]:
    """
    选择用于三联图渲染的节点。

    Args:
        G_after: 事件后图谱。
        partition_after: 事件后社区划分。
        community_rows: 社区变化记录。
        max_plot_nodes: 绘图节点上限。
        min_community_nodes: 每个社区至少保留的节点数。

    Returns:
        list[str]: 选中的节点 ID。
    """
    if G_after.number_of_nodes() <= max_plot_nodes:
        return list(G_after.nodes)

    after_members = _community_members(partition_after)
    if not after_members:
        top_nodes = sorted(
            G_after.nodes,
            key=lambda node_id: (
                G_after.to_undirected().degree(node_id),
                G_after.nodes[node_id].get("cited_by_count", 0),
            ),
            reverse=True,
        )
        return top_nodes[:max_plot_nodes]

    bridge_budget = min(max(6, int(max_plot_nodes * 0.10)), max_plot_nodes // 4)
    community_budget = max(1, max_plot_nodes - bridge_budget)
    chosen_rows = _choose_communities_for_plot(
        community_rows=community_rows,
        max_plot_nodes=community_budget,
        min_community_nodes=min_community_nodes,
    )
    quotas = _allocate_community_quotas(
        chosen_rows=chosen_rows,
        max_plot_nodes=community_budget,
        min_community_nodes=min_community_nodes,
    )

    selected_nodes: list[str] = []
    selected_set: set[str] = set()
    UG = G_after.to_undirected()

    for row in chosen_rows:
        community_id = row["after_community_id"]
        candidates = sorted(
            after_members.get(community_id, set()),
            key=lambda node_id: (
                UG.degree(node_id),
                G_after.nodes[node_id].get("cited_by_count", 0),
            ),
            reverse=True,
        )
        for node_id in candidates[: quotas.get(community_id, 0)]:
            if node_id in selected_set:
                continue
            selected_nodes.append(node_id)
            selected_set.add(node_id)

    if len(selected_nodes) >= max_plot_nodes:
        return selected_nodes[:max_plot_nodes]

    approx_k = min(120, max(20, int(math.sqrt(max(G_after.number_of_nodes(), 1)) * 6)))
    k_sample = None if G_after.number_of_nodes() <= approx_k else approx_k
    bc_scores = compute_betweenness(G_after, k_sample=k_sample)
    for node_id, _ in sorted(bc_scores.items(), key=lambda item: item[1], reverse=True):
        if node_id in selected_set:
            continue
        selected_nodes.append(node_id)
        selected_set.add(node_id)
        if len(selected_nodes) >= max_plot_nodes:
            return selected_nodes

    for node_id in sorted(
        G_after.nodes,
        key=lambda item: (
            UG.degree(item),
            G_after.nodes[item].get("cited_by_count", 0),
        ),
        reverse=True,
    ):
        if node_id in selected_set:
            continue
        selected_nodes.append(node_id)
        if len(selected_nodes) >= max_plot_nodes:
            break

    return selected_nodes


def _build_color_maps(community_rows: list[dict], partition_before: dict) -> tuple[dict, dict]:
    """
    构建 before/after 社区颜色映射。

    Args:
        community_rows: 社区变化记录。
        partition_before: 前期社区划分。

    Returns:
        tuple[dict, dict]: before_color_map, after_color_map。
    """
    before_communities = sorted(set(partition_before.values()))
    palette = plt.cm.tab20(np.linspace(0, 1, max(len(before_communities), 1)))
    before_color_map = {
        int(community_id): palette[index % len(palette)]
        for index, community_id in enumerate(before_communities)
    }
    after_color_map: dict[int, Any] = {}

    for row in community_rows:
        after_community_id = row["after_community_id"]
        before_community_id = row["matched_before_community_id"]
        if row["is_new"]:
            after_color_map[after_community_id] = _NEW_COMMUNITY_FILL
        elif before_community_id is not None and before_community_id in before_color_map:
            after_color_map[after_community_id] = before_color_map[before_community_id]
        else:
            after_color_map[after_community_id] = _FALLBACK_COMMUNITY_FILL

    return before_color_map, after_color_map


def _build_shared_layout(G_after_plot: nx.DiGraph) -> dict[str, np.ndarray]:
    """
    为 after 子图生成共享布局。

    Args:
        G_after_plot: 采样后的事件后图谱。

    Returns:
        dict[str, np.ndarray]: 节点坐标。
    """
    if G_after_plot.number_of_nodes() == 0:
        return {}
    if G_after_plot.number_of_nodes() == 1:
        node_id = next(iter(G_after_plot.nodes))
        return {node_id: np.array([0.0, 0.0])}

    spacing = 1.2 / math.sqrt(max(G_after_plot.number_of_nodes(), 2))
    return nx.spring_layout(
        G_after_plot.to_undirected(),
        seed=42,
        k=spacing,
        iterations=150,
    )


def _node_sizes(G: nx.DiGraph, node_ids: list[str]) -> list[float]:
    """
    生成绘图节点大小。

    Args:
        G: 图谱。
        node_ids: 节点顺序。

    Returns:
        list[float]: 节点尺寸列表。
    """
    UG = G.to_undirected()
    sizes = []
    for node_id in node_ids:
        degree = UG.degree(node_id)
        cited = G.nodes[node_id].get("cited_by_count", 0) or 0
        size = 70.0 + 18.0 * math.sqrt(max(degree, 1)) + 22.0 * math.log1p(max(cited, 0))
        sizes.append(min(size, 380.0))
    return sizes


def _draw_snapshot_panel(
    ax: plt.Axes,
    G_plot: nx.DiGraph,
    positions: dict[str, np.ndarray],
    node_ids: list[str],
    color_map: dict,
    partition: dict,
    title: str,
    event_year: int,
    target_nodes: Optional[list[str]] = None,
    highlight_new_community: bool = False,
    community_rows: Optional[list[dict]] = None,
) -> None:
    """
    绘制单个图谱快照面板。

    Args:
        ax: Matplotlib 轴。
        G_plot: 子图。
        positions: 共享坐标。
        node_ids: 节点顺序。
        color_map: 社区颜色映射。
        partition: 社区划分。
        title: 面板标题。
        event_year: 事件年份。
        target_nodes: 目标论文节点列表。
        highlight_new_community: 是否强调新社群。
        community_rows: 社区统计记录。
    """
    if G_plot.number_of_nodes() == 0 or not node_ids:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
        return

    new_community_ids = {
        row["after_community_id"] for row in (community_rows or []) if row["is_new"]
    }
    node_fill_colors = []
    node_edge_colors = []
    node_linewidths = []

    for node_id in node_ids:
        community_id = partition.get(node_id)
        node_fill_colors.append(color_map.get(community_id, _FALLBACK_COMMUNITY_FILL))
        node_year = G_plot.nodes[node_id].get("year") or 0
        is_post_event = node_year >= event_year
        is_new_community = highlight_new_community and community_id in new_community_ids

        if is_new_community:
            node_edge_colors.append(_NEW_COMMUNITY_EDGE)
            node_linewidths.append(1.8)
        elif is_post_event:
            node_edge_colors.append(_NEW_COMMUNITY_EDGE)
            node_linewidths.append(0.9)
        else:
            node_edge_colors.append("#F8FAFC")
            node_linewidths.append(0.4)

    nx.draw_networkx_edges(
        G_plot,
        pos=positions,
        ax=ax,
        arrows=False,
        edge_color="#94A3B8",
        alpha=0.20,
        width=0.6,
    )
    nx.draw_networkx_nodes(
        G_plot,
        pos=positions,
        nodelist=node_ids,
        node_size=_node_sizes(G_plot, node_ids),
        node_color=node_fill_colors,
        edgecolors=node_edge_colors,
        linewidths=node_linewidths,
        ax=ax,
        alpha=0.94,
    )

    target_node_ids = [
        node_id for node_id in (target_nodes or [])
        if node_id in G_plot and node_id in positions
    ]
    if target_node_ids:
        nx.draw_networkx_nodes(
            G_plot,
            pos=positions,
            nodelist=target_node_ids,
            node_size=[min(size * 1.8, 520.0) for size in _node_sizes(G_plot, target_node_ids)],
            node_color="#FDE047",
            edgecolors="#7C2D12",
            linewidths=1.8,
            node_shape="*",
            ax=ax,
            alpha=1.0,
        )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")


def _format_pct(value: Optional[float]) -> str:
    """
    将比例格式化为百分比字符串。

    Args:
        value: 比例。

    Returns:
        str: 百分比文本。
    """
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def _target_summary_lines(result: FieldContrastResult) -> list[str]:
    """
    生成目标论文的摘要文本。

    Args:
        result: 对比结果。

    Returns:
        list[str]: 目标论文摘要行。
    """
    target_ids = result.spec.target_paper_ids or []
    target_titles = result.spec.target_paper_titles or []
    if not target_ids:
        return []

    partition_after = result.delta_q_result.get("partition_after", {})
    community_lookup = {
        row["after_community_id"]: row
        for row in result.community_rows
    }

    lines = ["", "Target paper(s):"]
    for index, target_id in enumerate(target_ids):
        title = target_titles[index] if index < len(target_titles) else target_id
        community_id = partition_after.get(target_id)
        community_row = community_lookup.get(community_id)
        status = community_row["status"] if community_row else "unknown"
        community_label = f"C{community_id}" if community_id is not None else "N/A"
        short_title = title if len(title) <= 58 else f"{title[:55]}..."
        lines.append(f"- {community_label} ({status}) {short_title}")
    return lines


def _top_shift_rows(community_rows: list[dict], top_k: int = 6) -> list[dict]:
    """
    选取摘要栏展示的社区。

    Args:
        community_rows: 全部社区记录。
        top_k: 展示数量。

    Returns:
        list[dict]: 截取后的社区记录。
    """
    return sorted(
        community_rows,
        key=lambda row: (not row["is_new"], -row["after_size"]),
    )[:top_k]


def export_community_shift_to_csv(
    result: FieldContrastResult,
    save_path: str,
) -> pd.DataFrame:
    """
    导出社区变化明细表。

    Args:
        result: 对比结果。
        save_path: 保存路径。

    Returns:
        pd.DataFrame: 导出的数据表。
    """
    rows = list(result.community_rows)
    note = "未检测到显著新社群 / No significant emergent community detected"

    if not rows:
        rows = [{
            "status": "insufficient_data",
            "note": note,
        }]
    elif not any(row.get("is_new") for row in rows):
        for row in rows:
            row["note"] = note

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    log.info(f"社区变化明细已导出: {save_path}")
    return df


def plot_field_contrast(
    result: FieldContrastResult,
    save_path: str,
) -> None:
    """
    绘制领域图谱时间节点前后对比三联图。

    Args:
        result: 对比结果。
        save_path: 输出路径。
    """
    partition_before = result.delta_q_result.get("partition_before", {})
    partition_after = result.delta_q_result.get("partition_after", {})
    before_color_map, after_color_map = _build_color_maps(
        community_rows=result.community_rows,
        partition_before=partition_before,
    )

    selected_after_nodes = [node_id for node_id in result.selected_nodes if node_id in result.after_graph]
    after_plot = result.after_graph.subgraph(selected_after_nodes).copy()
    before_plot_nodes = [node_id for node_id in selected_after_nodes if node_id in result.before_graph]
    before_plot = result.before_graph.subgraph(before_plot_nodes).copy()
    positions = _build_shared_layout(after_plot)

    fig = plt.figure(figsize=(22, 7.5))
    outer = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.2, 1.05], wspace=0.10)
    ax_before = fig.add_subplot(outer[0, 0])
    ax_after = fig.add_subplot(outer[0, 1])
    summary_grid = outer[0, 2].subgridspec(2, 1, height_ratios=[0.62, 0.38], hspace=0.15)
    ax_bar = fig.add_subplot(summary_grid[0, 0])
    ax_text = fig.add_subplot(summary_grid[1, 0])

    before_start = result.spec.event_year - result.spec.before_years
    before_end = result.spec.event_year - 1
    after_end = result.spec.event_year + result.spec.after_years

    _draw_snapshot_panel(
        ax=ax_before,
        G_plot=before_plot,
        positions={node_id: positions[node_id] for node_id in before_plot.nodes if node_id in positions},
        node_ids=[node_id for node_id in selected_after_nodes if node_id in before_plot],
        color_map=before_color_map,
        partition=partition_before,
        title=f"Before {result.spec.event_label}\n{before_start}-{before_end}",
        event_year=result.spec.event_year,
        target_nodes=result.spec.target_paper_ids,
        highlight_new_community=False,
    )
    _draw_snapshot_panel(
        ax=ax_after,
        G_plot=after_plot,
        positions=positions,
        node_ids=selected_after_nodes,
        color_map=after_color_map,
        partition=partition_after,
        title=f"After {result.spec.event_label}\n{before_start}-{after_end}",
        event_year=result.spec.event_year,
        target_nodes=result.spec.target_paper_ids,
        highlight_new_community=True,
        community_rows=result.community_rows,
    )

    top_rows = _top_shift_rows(result.community_rows)
    y_positions = np.arange(len(top_rows))
    if top_rows:
        bar_colors = [
            after_color_map.get(row["after_community_id"], _FALLBACK_COMMUNITY_FILL)
            for row in top_rows
        ]
        after_sizes = [row["after_size"] for row in top_rows]
        overlap_sizes = [row["overlap_with_before"] for row in top_rows]
        labels = [
            f"C{row['after_community_id']} · {row['status']}"
            for row in top_rows
        ]

        ax_bar.barh(y_positions, after_sizes, color=bar_colors, alpha=0.9, height=0.62)
        ax_bar.barh(y_positions, overlap_sizes, color="#CBD5E1", alpha=0.95, height=0.28)
        ax_bar.set_yticks(y_positions)
        ax_bar.set_yticklabels(labels, fontsize=9)
        ax_bar.invert_yaxis()
        ax_bar.grid(axis="x", alpha=0.25)
        ax_bar.set_xlabel("Community size", fontsize=10)
    else:
        ax_bar.text(0.5, 0.5, "No community summary available", ha="center", va="center",
                    transform=ax_bar.transAxes)

    ax_bar.set_title("Community Shift Summary", fontsize=12, fontweight="bold")

    largest_new_size = max(
        (row["after_size"] for row in result.emergent_communities),
        default=0,
    )
    summary_lines = [
        f"Event: {result.spec.event_label} ({result.spec.event_year})",
        f"Nodes: {result.graph_stats['before_nodes']} → {result.graph_stats['after_nodes']} ({_format_pct(result.graph_stats['node_growth_rate'])})",
        f"Edges: {result.graph_stats['before_edges']} → {result.graph_stats['after_edges']} ({_format_pct(result.graph_stats['edge_growth_rate'])})",
        f"Communities: {result.graph_stats['before_comm']} → {result.graph_stats['after_comm']}",
        f"ΔQ: {result.delta_q_result.get('delta_Q', 0.0):+.4f}",
        f"Cross-community edges: {_format_pct(result.graph_stats['before_cross_ratio'])} → {_format_pct(result.graph_stats['after_cross_ratio'])}",
        f"Largest new community: {largest_new_size}",
    ]

    if result.emergent_communities:
        summary_lines.append("")
        summary_lines.append("Top emergent communities:")
        for row in result.emergent_communities[:3]:
            summary_lines.append(
                f"- C{row['after_community_id']} ({row['after_size']}) "
                f"{row['dominant_field']} | {row['dominant_topic']}"
            )
    else:
        summary_lines.append("")
        summary_lines.append(_weak_difference_message())

    summary_lines.extend(_target_summary_lines(result))

    ax_text.axis("off")
    ax_text.text(
        0.0,
        1.0,
        "\n".join(summary_lines),
        ha="left",
        va="top",
        fontsize=10.2,
        linespacing=1.4,
        transform=ax_text.transAxes,
    )

    fig.suptitle(
        f"Field Citation Graph Contrast Around {result.spec.event_label}",
        fontsize=14,
        y=0.98,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.03, right=0.98, top=0.88, bottom=0.06, wspace=0.10)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    log.info(f"领域前后图谱对比图已保存: {save_path}")


def run_field_contrast(
    spec: FieldContrastSpec,
    G_full: nx.DiGraph,
    output_dir: Optional[Path] = None,
) -> FieldContrastResult:
    """
    运行领域图谱时间节点前后对比，并可选导出图与 CSV。

    Args:
        spec: 对比配置。
        G_full: 全量图谱。
        output_dir: 输出目录。

    Returns:
        FieldContrastResult: 完整对比结果。
    """
    from graph_builder import slice_graph_by_year, write_graphml_safe

    before_start = spec.event_year - spec.before_years
    before_end = spec.event_year - 1
    after_end = spec.event_year + spec.after_years

    G_before = slice_graph_by_year(G_full, year_end=before_end, year_start=before_start)
    G_after = slice_graph_by_year(G_full, year_end=after_end, year_start=before_start)

    delta_q_result = compute_delta_q(G_before, G_after)
    community_rows, emergent_rows = _analyze_community_shifts(
        spec=spec,
        G_before=G_before,
        G_after=G_after,
        partition_before=delta_q_result.get("partition_before", {}),
        partition_after=delta_q_result.get("partition_after", {}),
    )
    selected_nodes = _select_plot_nodes(
        G_after=G_after,
        partition_after=delta_q_result.get("partition_after", {}),
        community_rows=community_rows,
        max_plot_nodes=spec.max_plot_nodes,
        min_community_nodes=spec.min_community_size,
    )

    graph_stats = {
        "before_nodes": G_before.number_of_nodes(),
        "after_nodes": G_after.number_of_nodes(),
        "before_edges": G_before.number_of_edges(),
        "after_edges": G_after.number_of_edges(),
        "node_growth_rate": _safe_growth(G_before.number_of_nodes(), G_after.number_of_nodes()),
        "edge_growth_rate": _safe_growth(G_before.number_of_edges(), G_after.number_of_edges()),
        "before_comm": delta_q_result.get("n_communities_before", 0),
        "after_comm": delta_q_result.get("n_communities_after", 0),
        "before_cross_ratio": _cross_community_edge_ratio(
            G_before,
            delta_q_result.get("partition_before", {}),
        ),
        "after_cross_ratio": _cross_community_edge_ratio(
            G_after,
            delta_q_result.get("partition_after", {}),
        ),
    }

    result = FieldContrastResult(
        spec=spec,
        before_graph=G_before,
        after_graph=G_after,
        delta_q_result=delta_q_result,
        graph_stats=graph_stats,
        community_rows=community_rows,
        emergent_communities=emergent_rows,
        selected_nodes=selected_nodes,
    )

    if output_dir is not None:
        output_dir.mkdir(exist_ok=True, parents=True)
        slug = spec.slug or _slugify(f"{spec.event_label}_{spec.event_year}")
        before_graph_path = output_dir / f"field_before_{slug}.graphml"
        after_graph_path = output_dir / f"field_after_{slug}.graphml"
        plot_path = output_dir / f"field_contrast_{slug}.png"
        csv_path = output_dir / f"community_shift_{slug}.csv"

        write_graphml_safe(G_before, before_graph_path)
        write_graphml_safe(G_after, after_graph_path)
        export_community_shift_to_csv(result, str(csv_path))
        plot_field_contrast(result, str(plot_path))

        result.output_paths = {
            "before_graphml": str(before_graph_path),
            "after_graphml": str(after_graph_path),
            "field_contrast_png": str(plot_path),
            "community_shift_csv": str(csv_path),
        }

    if result.emergent_communities:
        top_new = result.emergent_communities[0]
        log.info(
            "识别到显著新社群: C%s, size=%s, field=%s",
            top_new["after_community_id"],
            top_new["after_size"],
            top_new["dominant_field"],
        )
    else:
        log.info("未检测到显著新社群，已按“差异较弱”输出结果。")

    return result


# ──────────────────────────────────────────────────────────────────────────────
# 单篇论文七维指标核心对比逻辑
# ──────────────────────────────────────────────────────────────────────────────

def run_comparison(
    case: NobelCase,
    G_full: nx.DiGraph,
    window_before: int = 10,
    window_after: int = 5,
    k_sample_bc: int = 300,
) -> ComparisonResult:
    """
    运行单篇论文在前后两个时间窗中的七维指标对比。

    Args:
        case: 诺奖案例配置。
        G_full: 全量图谱。
        window_before: 前期时间窗长度。
        window_after: 后期时间窗长度。
        k_sample_bc: betweenness 近似采样点数。

    Returns:
        ComparisonResult: 指标对比结果。
    """
    from graph_builder import slice_graph_by_year

    y = case.nobel_year
    log.info(f"\n{'=' * 60}")
    log.info(f"案例: {case.name}  (获奖年: {y})")
    log.info(f"{'=' * 60}")

    G_before = slice_graph_by_year(G_full, year_end=y - 1, year_start=y - window_before)
    G_after = slice_graph_by_year(G_full, year_end=y + window_after, year_start=y - window_before)

    log.info(
        "G_before (%s~%s): %s 节点, %s 边",
        y - window_before,
        y - 1,
        G_before.number_of_nodes(),
        G_before.number_of_edges(),
    )
    log.info(
        "G_after  (%s~%s): %s 节点, %s 边",
        y - window_before,
        y + window_after,
        G_after.number_of_nodes(),
        G_after.number_of_edges(),
    )

    log.info("预计算 G_before 中心性 & 社区划分...")
    bc_before = compute_betweenness(G_before, k_sample=k_sample_bc)
    _, part_before = compute_modularity(G_before)

    log.info("预计算 G_after 中心性 & 社区划分...")
    bc_after = compute_betweenness(G_after, k_sample=k_sample_bc)
    _, part_after = compute_modularity(G_after)

    log.info("构建 Uzzi 基线...")
    uzzi_bl_before = _build_journal_copair_baseline(G_before, n_permutations=50)
    uzzi_bl_after = _build_journal_copair_baseline(G_after, n_permutations=50)

    log.info("计算奖前指标...")
    m_before = compute_all_metrics_for_paper(
        case.paper_id,
        G_before,
        betweenness_cache=bc_before,
        uzzi_baseline=uzzi_bl_before,
        partition_cache=part_before,
    )

    log.info("计算奖后指标...")
    m_after = compute_all_metrics_for_paper(
        case.paper_id,
        G_after,
        betweenness_cache=bc_after,
        uzzi_baseline=uzzi_bl_after,
        partition_cache=part_after,
    )

    log.info("计算 ΔQ...")
    dq_result = compute_delta_q(G_before, G_after)
    m_before["delta_q"] = dq_result["Q_before"]
    m_after["delta_q"] = dq_result["Q_after"]

    graph_stats = {
        "before_nodes": G_before.number_of_nodes(),
        "before_edges": G_before.number_of_edges(),
        "after_nodes": G_after.number_of_nodes(),
        "after_edges": G_after.number_of_edges(),
        "before_density": nx.density(G_before),
        "after_density": nx.density(G_after),
        "before_comm": dq_result["n_communities_before"],
        "after_comm": dq_result["n_communities_after"],
    }

    result = ComparisonResult(
        case=case,
        metrics_before=m_before,
        metrics_after=m_after,
        delta_q_result=dq_result,
        graph_stats=graph_stats,
    )
    _print_comparison_table(result)
    return result


def _print_comparison_table(result: ComparisonResult) -> None:
    """
    在终端打印单篇论文前后指标对比表。

    Args:
        result: 对比结果。
    """
    before_metrics = result.metrics_before
    after_metrics = result.metrics_after

    print(f"\n{'─' * 68}")
    print(f"  {result.case.name}")
    print(f"{'─' * 68}")
    print(f"  {'指标':<26} {'奖前':>12} {'奖后':>12} {'变化':>12}")
    print(f"{'─' * 68}")
    for name, expect_up, key in METRIC_META:
        before_value = before_metrics.get(key)
        after_value = after_metrics.get(key)
        before_text = f"{before_value:.4f}" if before_value is not None else "   N/A  "
        after_text = f"{after_value:.4f}" if after_value is not None else "   N/A  "
        if before_value is not None and after_value is not None:
            delta_value = after_value - before_value
            arrow = "↑" if delta_value > 0 else ("↓" if delta_value < 0 else "—")
            match = "✅" if (delta_value > 0) == expect_up else "⚠️"
            delta_text = f"{match} {arrow}{abs(delta_value):.4f}"
        else:
            delta_text = "   N/A"
        print(f"  {name:<26} {before_text:>12} {after_text:>12} {delta_text:>12}")

    print(f"{'─' * 68}")
    print(
        f"  社区数: {result.graph_stats['before_comm']} → {result.graph_stats['after_comm']}"
        f"   ΔQ = {result.delta_q_result['delta_Q']:+.4f}"
        f"  ({'社区破壁 ✅' if result.delta_q_result['delta_Q'] < 0 else '社区加深'})"
    )
    print(f"{'─' * 68}\n")


# ──────────────────────────────────────────────────────────────────────────────
# 单篇论文可视化
# ──────────────────────────────────────────────────────────────────────────────

def plot_before_after_bars(
    results: list[ComparisonResult],
    save_path: str = "metrics_bar.png",
) -> None:
    """
    七维指标奖前/奖后柱状对比图。

    Args:
        results: 对比结果列表。
        save_path: 输出路径。
    """
    keys = [meta[2] for meta in METRIC_META]
    labels = [_metric_plot_label(meta[0], meta[2]) for meta in METRIC_META]
    n_metrics = len(keys)
    n_cases = len(results)
    before_label, after_label = _plot_phase_labels()

    ncols = 4
    nrows = math.ceil(n_metrics / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for index, (key, label, (_, expect_up, __)) in enumerate(zip(keys, labels, METRIC_META)):
        ax = axes[index]
        case_names = [result.case.name.split("(")[0].strip()[:16] for result in results]
        before_values = [result.metrics_before.get(key) or 0 for result in results]
        after_values = [result.metrics_after.get(key) or 0 for result in results]

        x_positions = range(n_cases)
        width = 0.35
        ax.bar(
            [x - width / 2 for x in x_positions],
            before_values,
            width,
            label=before_label,
            color="#2563A8",
            alpha=0.85,
        )
        ax.bar(
            [x + width / 2 for x in x_positions],
            after_values,
            width,
            label=after_label,
            color="#F0A500",
            alpha=0.85,
        )
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels(case_names, fontsize=7, rotation=20, ha="right")
        ax.legend(fontsize=7)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        direction = "expect UP" if expect_up else "expect DOWN"
        ax.text(
            0.02,
            0.96,
            direction,
            transform=ax.transAxes,
            fontsize=7,
            color="#16a34a" if expect_up else "#dc2626",
            va="top",
        )

    for index in range(n_metrics, len(axes)):
        axes[index].set_visible(False)

    plt.suptitle("Seven-Dimension Innovation Metrics: Before vs After Nobel Prize", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"柱状图已保存: {save_path}")


def plot_radar(results: list[ComparisonResult], save_path: str = "metrics_radar.png") -> None:
    """
    绘制单篇或多篇论文的指标雷达图。

    Args:
        results: 对比结果列表。
        save_path: 输出路径。
    """
    if not results:
        return

    metric_keys = [meta[2] for meta in METRIC_META if meta[2] != "delta_q"]
    metric_labels = [_metric_plot_label(meta[0], meta[2]) for meta in METRIC_META if meta[2] != "delta_q"]
    angles = np.linspace(0, 2 * np.pi, len(metric_keys), endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(
        1,
        len(results),
        subplot_kw={"polar": True},
        figsize=(6 * len(results), 6),
    )
    if len(results) == 1:
        axes = [axes]

    before_label, after_label = _plot_phase_labels()

    for ax, result in zip(axes, results):
        before_values = [result.metrics_before.get(key) or 0.0 for key in metric_keys]
        after_values = [result.metrics_after.get(key) or 0.0 for key in metric_keys]
        before_values += before_values[:1]
        after_values += after_values[:1]

        ax.plot(angles, before_values, color="#2563A8", linewidth=2, label=before_label)
        ax.fill(angles, before_values, color="#2563A8", alpha=0.15)
        ax.plot(angles, after_values, color="#F0A500", linewidth=2, label=after_label)
        ax.fill(angles, after_values, color="#F0A500", alpha=0.15)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, fontsize=9)
        ax.set_title(result.case.name, fontsize=11, y=1.10)
        ax.grid(alpha=0.3)

    axes[0].legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    plt.suptitle("Innovation Metrics Radar: Before vs After Nobel Prize", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"雷达图已保存: {save_path}")


def plot_modularity_timeline(
    results: list[ComparisonResult],
    G_full: nx.DiGraph,
    save_path: str = "modularity_timeline.png",
) -> None:
    """
    绘制模块度时间序列图。

    Args:
        results: 对比结果列表。
        G_full: 全量图谱。
        save_path: 输出路径。
    """
    from graph_builder import slice_graph_by_year

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for result, color in zip(results, colors):
        year = result.case.nobel_year
        years: list[int] = []
        q_values: list[float] = []
        for current_year in range(year - 8, year + 6):
            sliced_graph = slice_graph_by_year(G_full, year_end=current_year, year_start=current_year - 5)
            if sliced_graph.number_of_edges() < 10:
                continue
            q_value, _ = compute_modularity(sliced_graph)
            years.append(current_year)
            q_values.append(q_value)

        if not years:
            continue

        label = result.case.name.split("(")[0].strip()
        ax.plot(years, q_values, marker="o", lw=2, color=color, label=label)
        ax.axvline(x=year, color=color, linestyle="--", alpha=0.5)

    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Modularity Q", fontsize=11)
    ax.set_title("Knowledge Graph Modularity Over Time (dashed = Nobel Year)", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"模块度时间线图已保存: {save_path}")


def plot_ego_network(
    target_id: str,
    G_before: nx.DiGraph,
    G_after: nx.DiGraph,
    save_path: str = "ego_network.png",
    max_nodes: int = 80,
) -> None:
    """
    绘制目标论文的前后 ego 网络。

    Args:
        target_id: 目标论文 ID。
        G_before: 前期图谱。
        G_after: 后期图谱。
        save_path: 输出路径。
        max_nodes: 最多绘制节点数。
    """
    def get_ego_graph(G: nx.DiGraph, center: str, radius: int = 2) -> nx.DiGraph:
        if center not in G:
            return nx.DiGraph()
        ego = nx.ego_graph(G.to_undirected(), center, radius=radius)
        if ego.number_of_nodes() > max_nodes:
            top_nodes = sorted(ego.degree(), key=lambda item: item[1], reverse=True)[:max_nodes]
            ego = ego.subgraph([node_id for node_id, _ in top_nodes]).copy()
        return ego

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    panels = [
        (axes[0], get_ego_graph(G_before, target_id), "Before Nobel"),
        (axes[1], get_ego_graph(G_after, target_id), "After Nobel"),
    ]

    for ax, ego_graph, title in panels:
        if ego_graph.number_of_nodes() == 0:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes, fontsize=14)
            ax.set_title(title)
            ax.axis("off")
            continue

        positions = nx.spring_layout(ego_graph, seed=42, k=0.8)
        domains = list(set(ego_graph.nodes[node_id].get("domain", "Other") for node_id in ego_graph.nodes))
        palette = plt.cm.Set3(np.linspace(0, 1, max(len(domains), 1)))
        domain_color = {domain: palette[index] for index, domain in enumerate(domains)}

        node_sizes = [
            max(30, min(ego_graph.nodes[node_id].get("cited_by_count", 1) * 2, 500))
            for node_id in ego_graph.nodes
        ]
        node_colors = [
            "red" if node_id == target_id else domain_color.get(ego_graph.nodes[node_id].get("domain", "Other"), (.8, .8, .8, 1))
            for node_id in ego_graph.nodes
        ]

        nx.draw_networkx(
            ego_graph,
            positions,
            ax=ax,
            node_size=node_sizes,
            node_color=node_colors,
            with_labels=False,
            edge_color="gray",
            alpha=0.7,
            arrows=False,
            width=0.5,
        )
        ax.set_title(
            f"{title}\n({ego_graph.number_of_nodes()} nodes, {ego_graph.number_of_edges()} edges)",
            fontsize=11,
        )
        ax.axis("off")
        legend_handles = [
            plt.scatter([], [], c=[domain_color.get(domain, (.8, .8, .8, 1))], s=40, label=domain)
            for domain in domains[:6]
        ]
        ax.legend(handles=legend_handles, loc="lower left", fontsize=7, title="Domain")

    plt.suptitle("Ego Network Comparison (Red = Nobel paper)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Ego 网络图已保存: {save_path}")


def export_results_to_csv(
    results: list[ComparisonResult],
    save_path: str = "results.csv",
) -> pd.DataFrame:
    """
    导出单篇论文前后指标结果。

    Args:
        results: 对比结果列表。
        save_path: 保存路径。

    Returns:
        pd.DataFrame: 导出的表格。
    """
    rows = []
    for result in results:
        for phase, metrics in [("before", result.metrics_before), ("after", result.metrics_after)]:
            row = {
                "case": result.case.name,
                "paper_id": result.case.paper_id,
                "nobel_year": result.case.nobel_year,
                "phase": phase,
            }
            for _, _, key in METRIC_META:
                row[key] = metrics.get(key)
            row["delta_Q_overall"] = result.delta_q_result.get("delta_Q")
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    log.info(f"结果已导出: {save_path}")
    return df

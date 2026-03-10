"""
graph_builder.py — 知识图谱构建模块
将论文数据转化为 NetworkX 有向图，并支持按时间切片
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import networkx as nx

log = logging.getLogger(__name__)


def build_node_attributes(work: Dict[str, Any]) -> Dict[str, Any]:
    """
    将标准化论文记录转换为节点属性，并过滤 GraphML 不支持的空值。

    Args:
        work: `normalize_work()` 产出的标准化论文数据。

    Returns:
        Dict[str, Any]: 可直接写入 NetworkX 节点的属性字典。
    """
    attrs = {
        "doi": work.get("doi", ""),
        "title": work.get("title", ""),
        "year": work.get("year"),
        "cited_by_count": work.get("cited_by_count", 0),
        "journal": work.get("journal", ""),
        "domain": work.get("domain", ""),
        "field": work.get("field", ""),
        "subfield": work.get("subfield", ""),
        "topic_names": "|".join(work.get("topic_names") or []),
    }
    return {key: value for key, value in attrs.items() if value is not None}


def _graphml_attr_value(value: Any) -> Optional[Any]:
    """
    将任意属性值转换为 GraphML 支持的标量类型。

    Args:
        value: 原始属性值。

    Returns:
        Optional[Any]: GraphML 支持的值；若为 `None` 则返回 `None` 以便上层剔除。
    """
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item") and callable(value.item):
        return _graphml_attr_value(value.item())
    if isinstance(value, (list, tuple, set)):
        return "|".join(str(item) for item in value if item is not None)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def write_graphml_safe(G: nx.DiGraph, path: Path) -> None:
    """
    以 GraphML 兼容格式导出图，自动清洗不支持的属性类型。

    Args:
        G: 待导出的有向图。
        path: 输出文件路径。
    """
    graph_to_write = nx.DiGraph()
    graph_to_write.graph.update({
        key: value
        for key, raw in G.graph.items()
        if (value := _graphml_attr_value(raw)) is not None
    })

    for node_id, attrs in G.nodes(data=True):
        clean_attrs = {
            key: value
            for key, raw in attrs.items()
            if (value := _graphml_attr_value(raw)) is not None
        }
        graph_to_write.add_node(node_id, **clean_attrs)

    for source, target, attrs in G.edges(data=True):
        clean_attrs = {
            key: value
            for key, raw in attrs.items()
            if (value := _graphml_attr_value(raw)) is not None
        }
        graph_to_write.add_edge(source, target, **clean_attrs)

    nx.write_graphml(graph_to_write, str(path))


def build_graph(works: list[dict]) -> nx.DiGraph:
    """
    从标准化后的论文列表构建有向引用图。
    节点 = 论文 (id)；有向边 = p → ref（p 引用了 ref）。
    节点属性: year, title, doi, journal, domain, field, cited_by_count
    """
    G = nx.DiGraph()
    for w in works:
        nid = w["id"]
        if not nid:
            continue
        G.add_node(nid, **build_node_attributes(w))
        for ref_id in w["referenced_works"]:
            if ref_id:
                G.add_edge(nid, ref_id)   # nid 引用了 ref_id

    log.info(f"图构建完成: {G.number_of_nodes():,} 节点, {G.number_of_edges():,} 边")
    return G


def slice_graph_by_year(G: nx.DiGraph, year_end: int,
                        year_start: Optional[int] = None) -> nx.DiGraph:
    """
    按年份切片：只保留 year_start <= year <= year_end 的节点及其之间的边。
    用于构建"诺贝尔奖前"或"诺贝尔奖后"的快照图。
    """
    def keep(n):
        y = G.nodes[n].get("year")
        if y is None:
            return False
        if year_start and y < year_start:
            return False
        return y <= year_end

    nodes_to_keep = [n for n in G.nodes if keep(n)]
    sub = G.subgraph(nodes_to_keep).copy()
    log.info(f"切片图 (≤{year_end}): {sub.number_of_nodes():,} 节点, {sub.number_of_edges():,} 边")
    return sub


def add_nodes_from_refs(G: nx.DiGraph, works_lookup: dict[str, dict]) -> nx.DiGraph:
    """
    补充节点属性：图中可能存在只作为引用目标、但没有被拉取过属性的节点。
    works_lookup: {openalex_id: normalized_work}
    """
    missing = [n for n in G.nodes if G.nodes[n].get("year") is None]
    filled = 0
    for n in missing:
        if n in works_lookup:
            w = works_lookup[n]
            G.nodes[n].update(build_node_attributes(w))
            filled += 1
    if missing:
        log.info(f"补充节点属性: {filled}/{len(missing)} 个空节点已填充")
    return G


def get_graph_summary(G: nx.DiGraph) -> dict:
    """返回图的基本统计摘要"""
    years = [G.nodes[n].get("year") for n in G.nodes if G.nodes[n].get("year")]
    return {
        "nodes":     G.number_of_nodes(),
        "edges":     G.number_of_edges(),
        "year_min":  min(years) if years else None,
        "year_max":  max(years) if years else None,
        "density":   nx.density(G),
        "weakly_connected_components": nx.number_weakly_connected_components(G),
    }

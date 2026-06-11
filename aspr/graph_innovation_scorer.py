from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import networkx as nx


DEFAULT_GRAPH_METRIC_WEIGHTS = {
    "B": 0.01260526,
    "RS": 0.14861328,
    "DeltaQ0": 0.39528357,
    "Uzzi": 0.23449148,
    "RTD": 0.03278302,
    "BurtIP": 0.00431764,
    "PDE": 0.17190574,
}

METRIC_DESCRIPTIONS = {
    "B": "Bridge position / 跨社区桥接位置",
    "RS": "Rao-Stirling breadth / 跨学科知识广度",
    "DeltaQ0": "Boundary perturbation / 社区边界扰动",
    "Uzzi": "Atypical recombination / 非典型组合",
    "RTD": "Reference target diversity / 参考目标社区多样性",
    "BurtIP": "Structural-hole potential / 结构洞潜力",
    "PDE": "Prospective diffusion entropy / 潜在扩散熵",
}


@dataclass
class GraphInnovationEvidence:
    """Structured graph evidence used to ground innovation self-reflection."""

    metrics: Dict[str, float]
    weighted_score: float
    confidence: float
    top_mechanisms: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_prompt_block(self) -> str:
        """Render a compact evidence block for LLM prompts."""
        metric_lines = [
            f"- {key}: {self.metrics.get(key, 0.0):.3f} ({METRIC_DESCRIPTIONS[key]})"
            for key in DEFAULT_GRAPH_METRIC_WEIGHTS
        ]
        mechanisms = ", ".join(self.top_mechanisms) if self.top_mechanisms else "无明显主导机制"
        limitations = "\n".join(f"- {item}" for item in self.limitations) or "- 未发现明显数据限制"
        return (
            "【图谱结构证据】\n"
            f"综合结构扰动潜力分数: {self.weighted_score:.3f} / 1.000\n"
            f"证据置信度: {self.confidence:.3f} / 1.000\n"
            f"主导机制: {mechanisms}\n"
            "七维指标:\n"
            + "\n".join(metric_lines)
            + "\n数据限制:\n"
            + limitations
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable evidence."""
        return {
            "metrics": self.metrics,
            "weighted_score": self.weighted_score,
            "confidence": self.confidence,
            "top_mechanisms": self.top_mechanisms,
            "limitations": self.limitations,
            "diagnostics": self.diagnostics,
        }


class GraphInnovationScorer:
    """
    Compute publication-day graph evidence from the currently retrieved papers.

    This lightweight scorer is intentionally local and cold-start friendly. It
    approximates the final OpenAlex graph protocol by treating retrieved papers
    as the target paper's reference neighborhood, then builds a small reference
    graph from venue, field, and lexical overlap.
    """

    target_id = "__target_paper__"

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self.weights = self._normalize_weights(weights or DEFAULT_GRAPH_METRIC_WEIGHTS)

    def score(
        self,
        paper_title: str,
        paper_abstract: str,
        retrieved_papers: List[Dict[str, Any]],
    ) -> GraphInnovationEvidence:
        """Compute seven metric values and an interpretable weighted score."""
        papers = [paper for paper in retrieved_papers if paper.get("title") or paper.get("abstract")]
        if not papers:
            return GraphInnovationEvidence(
                metrics={key: 0.0 for key in DEFAULT_GRAPH_METRIC_WEIGHTS},
                weighted_score=0.0,
                confidence=0.0,
                limitations=["没有可用的相关论文，图谱结构证据不可用。"],
                diagnostics={"n_related_papers": 0},
            )

        graph = self._build_reference_graph(paper_title, paper_abstract, papers)
        ref_ids = [node for node in graph.successors(self.target_id)]
        field_labels = [graph.nodes[node].get("field", "") for node in ref_ids]
        domain_labels = [graph.nodes[node].get("domain", "") for node in ref_ids]
        community_labels = self._community_labels(graph, ref_ids)

        metrics = {
            "B": self._bridge_position(graph),
            "RS": self._simpson_diversity(field_labels),
            "DeltaQ0": self._boundary_perturbation(graph),
            "Uzzi": self._atypical_recombination(graph, ref_ids),
            "RTD": self._simpson_diversity([str(community_labels.get(node, "")) for node in ref_ids]),
            "BurtIP": self._burt_innovation_potential(graph, ref_ids),
            "PDE": self._entropy_norm(domain_labels or field_labels),
        }
        metrics = {key: self._clip01(value) for key, value in metrics.items()}
        weighted_score = sum(metrics[key] * self.weights.get(key, 0.0) for key in metrics)
        confidence = self._estimate_confidence(graph, papers, field_labels, domain_labels)
        top_mechanisms = self._top_mechanisms(metrics)
        limitations = self._limitations(papers, field_labels, domain_labels, confidence)

        return GraphInnovationEvidence(
            metrics=metrics,
            weighted_score=self._clip01(weighted_score),
            confidence=confidence,
            top_mechanisms=top_mechanisms,
            limitations=limitations,
            diagnostics={
                "n_related_papers": len(papers),
                "n_graph_nodes": graph.number_of_nodes(),
                "n_graph_edges": graph.number_of_edges(),
                "n_reference_communities": len(set(community_labels.values())),
                "field_coverage": self._coverage(field_labels),
                "domain_coverage": self._coverage(domain_labels),
            },
        )

    def _build_reference_graph(
        self,
        paper_title: str,
        paper_abstract: str,
        papers: List[Dict[str, Any]],
    ) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_node(
            self.target_id,
            title=paper_title,
            abstract=paper_abstract,
            field="target",
            domain="target",
            venue="target",
        )
        normalized = []
        for idx, paper in enumerate(papers):
            node_id = str(paper.get("paperId") or paper.get("id") or f"paper_{idx}")
            fields = self._paper_fields(paper)
            field_label = fields[0] if fields else self._fallback_field(paper)
            domain_label = fields[0].split("/")[0].strip() if fields else field_label
            attrs = {
                "title": str(paper.get("title") or ""),
                "abstract": str(paper.get("abstract") or ""),
                "venue": str(paper.get("venue") or paper.get("journal") or ""),
                "field": field_label,
                "domain": domain_label,
                "citation_count": int(paper.get("citationCount") or paper.get("citation_count") or 0),
                "tokens": self._text_tokens(f"{paper.get('title', '')} {paper.get('abstract', '')}"),
            }
            graph.add_node(node_id, **attrs)
            graph.add_edge(self.target_id, node_id)
            normalized.append((node_id, attrs))

        for i, (left_id, left_attrs) in enumerate(normalized):
            for right_id, right_attrs in normalized[i + 1:]:
                similarity = self._reference_similarity(left_attrs, right_attrs)
                if similarity >= 0.18:
                    graph.add_edge(left_id, right_id, weight=similarity)
                    graph.add_edge(right_id, left_id, weight=similarity)
        return graph

    def _bridge_position(self, graph: nx.DiGraph) -> float:
        undirected = graph.to_undirected()
        if undirected.number_of_nodes() < 3:
            return 0.0
        centrality = nx.betweenness_centrality(undirected, normalized=True)
        return float(centrality.get(self.target_id, 0.0))

    def _boundary_perturbation(self, graph: nx.DiGraph) -> float:
        ref_nodes = [node for node in graph.nodes if node != self.target_id]
        if len(ref_nodes) < 4:
            return 0.0
        before = graph.subgraph(ref_nodes).to_undirected()
        after = graph.to_undirected()
        q_before = self._modularity(before)
        q_after = self._modularity(after)
        return max(0.0, q_before - q_after)

    def _atypical_recombination(self, graph: nx.DiGraph, ref_ids: List[str]) -> float:
        if len(ref_ids) < 2:
            return 0.0
        atypical = 0.0
        total = 0
        undirected = graph.to_undirected()
        for i, left in enumerate(ref_ids):
            for right in ref_ids[i + 1:]:
                total += 1
                left_field = graph.nodes[left].get("field", "")
                right_field = graph.nodes[right].get("field", "")
                left_venue = graph.nodes[left].get("venue", "")
                right_venue = graph.nodes[right].get("venue", "")
                disconnected = 0.35 if not undirected.has_edge(left, right) else 0.0
                field_gap = 0.45 if left_field and right_field and left_field != right_field else 0.0
                venue_gap = 0.20 if left_venue and right_venue and left_venue != right_venue else 0.0
                atypical += disconnected + field_gap + venue_gap
        return atypical / max(total, 1)

    def _burt_innovation_potential(self, graph: nx.DiGraph, ref_ids: List[str]) -> float:
        if len(ref_ids) < 2:
            return 0.0
        possible_edges = len(ref_ids) * (len(ref_ids) - 1)
        existing_edges = 0
        for left in ref_ids:
            for right in ref_ids:
                if left != right and graph.has_edge(left, right):
                    existing_edges += 1
        density = existing_edges / possible_edges if possible_edges else 0.0
        return 1.0 - density

    def _community_labels(self, graph: nx.DiGraph, ref_ids: List[str]) -> Dict[str, int]:
        if not ref_ids:
            return {}
        subgraph = graph.subgraph(ref_ids).to_undirected()
        if subgraph.number_of_edges() == 0:
            return {node: idx for idx, node in enumerate(ref_ids)}
        communities = nx.algorithms.community.greedy_modularity_communities(subgraph)
        return {node: idx for idx, community in enumerate(communities) for node in community}

    def _modularity(self, graph: nx.Graph) -> float:
        graph = graph.copy()
        graph.remove_nodes_from(list(nx.isolates(graph)))
        if graph.number_of_edges() == 0 or graph.number_of_nodes() < 3:
            return 0.0
        communities = nx.algorithms.community.greedy_modularity_communities(graph)
        if len(communities) <= 1:
            return 0.0
        return float(nx.algorithms.community.modularity(graph, communities))

    def _simpson_diversity(self, labels: List[str]) -> float:
        labels = [label for label in labels if label]
        if len(labels) < 2:
            return 0.0
        counts = Counter(labels)
        total = sum(counts.values())
        concentration = sum(count * (count - 1) for count in counts.values()) / (total * (total - 1))
        return 1.0 - concentration

    def _entropy_norm(self, labels: List[str]) -> float:
        labels = [label for label in labels if label]
        if len(labels) < 2:
            return 0.0
        counts = Counter(labels)
        if len(counts) == 1:
            return 0.0
        total = sum(counts.values())
        entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
        return entropy / math.log2(len(counts))

    def _reference_similarity(self, left: Dict[str, Any], right: Dict[str, Any]) -> float:
        score = 0.0
        if left.get("field") and left.get("field") == right.get("field"):
            score += 0.45
        if left.get("domain") and left.get("domain") == right.get("domain"):
            score += 0.25
        if left.get("venue") and left.get("venue") == right.get("venue"):
            score += 0.15
        score += 0.25 * self._jaccard(left.get("tokens", set()), right.get("tokens", set()))
        return min(score, 1.0)

    def _paper_fields(self, paper: Dict[str, Any]) -> List[str]:
        fields: List[str] = []
        for value in paper.get("fieldsOfStudy") or []:
            if value:
                fields.append(str(value))
        for item in paper.get("s2FieldsOfStudy") or []:
            category = item.get("category") if isinstance(item, dict) else item
            if category:
                fields.append(str(category))
        return list(dict.fromkeys(fields))

    def _fallback_field(self, paper: Dict[str, Any]) -> str:
        venue = str(paper.get("venue") or "").strip()
        if venue:
            return f"Venue:{venue}"
        year = paper.get("year")
        return f"Unknown:{year or 'NA'}"

    def _text_tokens(self, text: str) -> set[str]:
        words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text.lower())
        stopwords = {"the", "and", "for", "with", "from", "that", "this", "are", "was", "were", "paper"}
        return {word for word in words if word not in stopwords}

    def _jaccard(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)

    def _estimate_confidence(
        self,
        graph: nx.DiGraph,
        papers: List[Dict[str, Any]],
        field_labels: List[str],
        domain_labels: List[str],
    ) -> float:
        n_factor = min(len(papers) / 10.0, 1.0)
        field_factor = self._coverage(field_labels)
        domain_factor = self._coverage(domain_labels)
        edge_factor = min(graph.number_of_edges() / max(1, len(papers) * 2), 1.0)
        return self._clip01(0.35 * n_factor + 0.30 * field_factor + 0.20 * domain_factor + 0.15 * edge_factor)

    def _coverage(self, labels: List[str]) -> float:
        if not labels:
            return 0.0
        valid = [label for label in labels if label and not label.startswith("Unknown:")]
        return len(valid) / len(labels)

    def _limitations(
        self,
        papers: List[Dict[str, Any]],
        field_labels: List[str],
        domain_labels: List[str],
        confidence: float,
    ) -> List[str]:
        limitations = []
        if len(papers) < 10:
            limitations.append("相关论文少于 10 篇，七维指标只应作为弱证据。")
        if self._coverage(field_labels) < 0.6:
            limitations.append("领域/学科元数据覆盖不足，RS、RTD、PDE 的解释需保守。")
        if self._coverage(domain_labels) < 0.6:
            limitations.append("domain 标签覆盖不足，潜在扩散熵 PDE 可能低估或不稳定。")
        if confidence < 0.5:
            limitations.append("当前图谱证据置信度偏低，最终评审应显式说明不确定性。")
        return limitations

    def _top_mechanisms(self, metrics: Dict[str, float]) -> List[str]:
        ranked = sorted(metrics.items(), key=lambda item: item[1], reverse=True)
        return [
            f"{key}={value:.2f} {METRIC_DESCRIPTIONS[key]}"
            for key, value in ranked[:3]
            if value >= 0.15
        ]

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        cleaned = {key: max(float(weights.get(key, 0.0)), 0.0) for key in DEFAULT_GRAPH_METRIC_WEIGHTS}
        total = sum(cleaned.values())
        if total <= 0:
            return {key: 1.0 / len(cleaned) for key in cleaned}
        return {key: value / total for key, value in cleaned.items()}

    def _clip01(self, value: float) -> float:
        if value is None or not math.isfinite(float(value)):
            return 0.0
        return max(0.0, min(float(value), 1.0))

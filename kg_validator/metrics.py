"""
metrics.py — 科学创新评价指标计算模块（冷启动兼容版）

所有指标均只依赖参考文献结构，发表当天即可计算，无需等待下游引用。

维度一: 结构洞桥接中心性  Betweenness Centrality     (Freeman 1977)
维度二: Rao-Stirling 跨学科多样性指数                (Rao 1982; Stirling 2007)
维度三: 社区模块度变化量  ΔQ                         (Newman & Girvan 2004)
维度四: Uzzi 非典型组合新颖性                        (Uzzi et al. 2013)
维度五: 引用目标多样性    RTD                        (冷启动替代 CD)
维度六: Burt 结构约束系数 Constraint                 (Burt 1992)
维度七: 预期扩散熵        PDE                        (冷启动替代 H; Shannon 1948)
"""

import math
import logging
import random
import warnings
from collections import Counter, defaultdict
from typing import Optional

import networkx as nx
import numpy as np

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 维度一：结构洞桥接中心性
# ══════════════════════════════════════════════════════════════════════════════

def compute_betweenness(
    G: nx.DiGraph,
    target_ids: Optional[list[str]] = None,
    k_sample: Optional[int] = 500,
    normalized: bool = True,
) -> dict[str, float]:
    """
    计算节点的（近似）中介中心性 B_norm(v)。

    B(v) = sum_{s!=v!=t} sigma_st(v) / sigma_st
    B_norm(v) = 2*B(v) / [(n-1)(n-2)]

    转为无向图后计算，更好地反映跨社区桥接作用。
    k_sample: 近似算法随机起点数，None 时精确计算。
    Returns: {node_id: betweenness_score}
    """
    log.info(f"计算 betweenness centrality (k={k_sample})...")
    UG = G.to_undirected()
    if target_ids:
        bc = nx.betweenness_centrality_subset(
            UG,
            sources=list(set(target_ids) & set(UG.nodes)),
            targets=list(UG.nodes),
            normalized=normalized,
        )
    else:
        bc = nx.betweenness_centrality(UG, k=k_sample, normalized=normalized)
    return bc


# ══════════════════════════════════════════════════════════════════════════════
# 维度二：Rao-Stirling 跨学科多样性指数
# ══════════════════════════════════════════════════════════════════════════════

def _field_vector(field_name: str, all_fields: list[str]) -> np.ndarray:
    v = np.zeros(len(all_fields))
    if field_name in all_fields:
        v[all_fields.index(field_name)] = 1.0
    return v


def _cosine_dist(v1: np.ndarray, v2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 1.0
    return 1.0 - float(np.dot(v1, v2) / (n1 * n2))


def compute_rao_stirling(
    target_id: str,
    G: nx.DiGraph,
    level: str = "field",
) -> float:
    """
    Rao-Stirling 跨学科多样性指数。

    RS = sum_{i!=j} d_ij * p_i * p_j
    p_i  : 参考文献中属于学科 i 的比例
    d_ij : 余弦距离（one-hot 时退化为 0/1）
    level: "domain" | "field" | "subfield"
    """
    if target_id not in G:
        return 0.0
    refs = list(G.successors(target_id))
    if not refs:
        return 0.0
    disciplines = [
        G.nodes[r].get(level, "")
        for r in refs if r in G and G.nodes[r].get(level, "")
    ]
    if len(disciplines) < 2:
        return 0.0
    counts     = Counter(disciplines)
    total      = sum(counts.values())
    all_fields = list(counts.keys())
    field_vecs = {f: _field_vector(f, all_fields) for f in all_fields}
    rs = 0.0
    for fi in all_fields:
        for fj in all_fields:
            if fi == fj:
                continue
            rs += _cosine_dist(field_vecs[fi], field_vecs[fj]) \
                  * (counts[fi] / total) * (counts[fj] / total)
    return rs


# ══════════════════════════════════════════════════════════════════════════════
# 维度三：社区模块度变化量 ΔQ
# ══════════════════════════════════════════════════════════════════════════════

def compute_modularity(G: nx.DiGraph) -> tuple[float, dict]:
    """
    greedy 社区检测，返回 (Q, partition)。
    Q > 0.4 通常表明存在显著社区结构。
    """
    UG = G.to_undirected()
    UG.remove_nodes_from(list(nx.isolates(UG)))
    if UG.number_of_edges() == 0:
        return 0.0, {}
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = greedy_modularity_communities(UG)
        partition   = {node: i for i, c in enumerate(communities) for node in c}
        Q           = nx.community.modularity(UG, communities)
    except Exception as e:
        log.warning(f"社区检测失败: {e}")
        return 0.0, {}
    return Q, partition


def compute_delta_q(G_before: nx.DiGraph, G_after: nx.DiGraph) -> dict:
    """
    delta_Q = Q(G_after) - Q(G_before)
    delta_Q < 0: 社区边界被打破（颠覆性创新信号）
    delta_Q > 0: 社区内部连接加深（巩固型信号）
    """
    log.info("计算 Q(G_before)...")
    Q_before, part_before = compute_modularity(G_before)
    log.info(f"  Q_before = {Q_before:.4f}, 社区数 = {len(set(part_before.values()))}")
    log.info("计算 Q(G_after)...")
    Q_after, part_after = compute_modularity(G_after)
    log.info(f"  Q_after  = {Q_after:.4f}, 社区数 = {len(set(part_after.values()))}")
    return {
        "Q_before":             Q_before,
        "Q_after":              Q_after,
        "delta_Q":              Q_after - Q_before,
        "n_communities_before": len(set(part_before.values())),
        "n_communities_after":  len(set(part_after.values())),
        "partition_before":     part_before,
        "partition_after":      part_after,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 维度四：Uzzi 非典型组合新颖性 z-score
# ══════════════════════════════════════════════════════════════════════════════

def _build_journal_copair_baseline(
    G: nx.DiGraph,
    year: Optional[int] = None,
    n_permutations: int = 100,
    sample_size: int = 2000,
) -> dict[tuple, tuple]:
    """构建期刊共被引对零模型基线 {(j1, j2): (mu, sigma)}。"""
    candidates = [
        n for n in G.nodes
        if G.out_degree(n) > 0
        and G.nodes[n].get("journal")
        and (year is None or G.nodes[n].get("year") == year)
    ]
    if len(candidates) > sample_size:
        candidates = random.sample(candidates, sample_size)

    def get_ref_journals(node):
        return [
            G.nodes[r].get("journal", "")
            for r in G.successors(node)
            if r in G and G.nodes[r].get("journal", "")
        ]

    paper_ref_journals = {n: get_ref_journals(n) for n in candidates}
    all_journals       = [j for js in paper_ref_journals.values() for j in js]
    pair_counts_per_perm = defaultdict(list)

    for _ in range(n_permutations):
        shuffled = all_journals.copy()
        random.shuffle(shuffled)
        idx, perm_counts = 0, Counter()
        for node, js in paper_ref_journals.items():
            n_refs  = len(js)
            new_js  = set(shuffled[idx: idx + n_refs])
            idx    += n_refs
            for a in new_js:
                for b in new_js:
                    if a < b:
                        perm_counts[(a, b)] += 1
        for pair, cnt in perm_counts.items():
            pair_counts_per_perm[pair].append(cnt)

    return {
        pair: (np.mean(counts), max(np.std(counts, ddof=1) if len(counts) > 1 else 1.0, 1e-6))
        for pair, counts in pair_counts_per_perm.items()
    }


def compute_uzzi_novelty(
    target_id: str,
    G: nx.DiGraph,
    baseline: Optional[dict] = None,
    percentile: float = 10.0,
) -> dict:
    """
    Uzzi 非典型组合新颖性得分。

    novelty_p10 = p10({z_{j1,j2}})   <- 期刊对 z-score 第 10 百分位
    median_z                          <- 中位 z-score

    "低 p10 + 高中位 z" 是高影响力论文的典型模式。
    """
    if target_id not in G or G.out_degree(target_id) == 0:
        return {"novelty_p10": None, "median_z": None, "n_journal_pairs": 0}
    ref_journals = [
        G.nodes[r].get("journal", "")
        for r in G.successors(target_id)
        if r in G and G.nodes[r].get("journal", "")
    ]
    if len(ref_journals) < 2:
        return {"novelty_p10": None, "median_z": None, "n_journal_pairs": 0}
    if baseline is None:
        warnings.warn("未提供 baseline，Uzzi-z 为近似值")
        baseline = {}
    journal_set = list(set(ref_journals))
    z_scores = []
    for i, ja in enumerate(journal_set):
        for jb in journal_set[i+1:]:
            pair = (min(ja, jb), max(ja, jb))
            O    = ref_journals.count(ja) * ref_journals.count(jb)
            if pair in baseline:
                mu, sigma = baseline[pair]
                z_scores.append((O - mu) / sigma)
            else:
                z_scores.append(-2.0)
    if not z_scores:
        return {"novelty_p10": None, "median_z": None, "n_journal_pairs": 0}
    return {
        "novelty_p10":     float(np.percentile(z_scores, percentile)),
        "median_z":        float(np.median(z_scores)),
        "n_journal_pairs": len(z_scores),
        "z_scores":        z_scores,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 维度五：RTD 引用目标多样性（冷启动替代 CD 颠覆指数）
# ══════════════════════════════════════════════════════════════════════════════

def compute_rtd(
    target_id: str,
    G: nx.DiGraph,
    partition: Optional[dict[str, int]] = None,
) -> dict:
    """
    引用目标多样性 RTD（Reference Target Diversity）。

    RTD = 1 - sum_c [n_c * (n_c-1)] / [N * (N-1)]

    其中 n_c 是参考文献中属于社区 c 的数量，N 是参考文献总数。
    等价于 Simpson 多样性指数在社区分布上的应用。

    RTD = 0: 所有参考文献来自同一社区（巩固型）
    RTD = 1: 参考文献均匀分布在所有社区（颠覆型先兆）

    partition: {node_id: community_id}，若为 None 则自动计算全图社区
    """
    if target_id not in G or G.out_degree(target_id) == 0:
        return {"rtd": None, "n_refs": 0, "n_communities_spanned": 0}

    refs = [r for r in G.successors(target_id) if r in G]
    if len(refs) < 2:
        return {"rtd": None, "n_refs": len(refs), "n_communities_spanned": 0}

    # 获取或计算社区划分
    if partition is None:
        _, partition = compute_modularity(G)
    if not partition:
        # 降级：每个参考文献各为一个社区（RTD = 1）
        partition = {r: i for i, r in enumerate(refs)}

    ref_communities  = [partition.get(r, -(i+1)) for i, r in enumerate(refs)]
    community_counts = Counter(ref_communities)
    N = len(refs)

    # Simpson 多样性
    simpson_concentration = sum(c * (c - 1) for c in community_counts.values()) / (N * (N - 1))
    rtd = 1.0 - simpson_concentration

    return {
        "rtd":                   rtd,
        "n_refs":                N,
        "n_communities_spanned": len(community_counts),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 维度六：Burt 结构约束系数（冷启动替代 CD，互补视角）
# ══════════════════════════════════════════════════════════════════════════════

def compute_burt_constraint(
    target_id: str,
    G: nx.DiGraph,
) -> dict:
    """
    Burt (1992) 结构约束系数 C(p) 与创新潜力 IP(p)。

    C(p) = sum_i (p_i + sum_{j!=i} p_ij * p_j)^2

    p_i   = 参考文献中 i 的权重（均等时为 1/n）
    p_ij  = i 与 j 之间的标准化连接强度

    约束越低 -> 参考文献之间联系越稀疏 -> 结构洞越多 -> 创新潜力越高
    IP(p) = 1 - C_norm(p) in [0, 1]
    """
    if target_id not in G or G.out_degree(target_id) == 0:
        return {"constraint": None, "innovation_potential": None, "n_ego_nodes": 0}

    refs = [r for r in G.successors(target_id) if r in G]
    if len(refs) < 2:
        return {"constraint": None, "innovation_potential": None, "n_ego_nodes": len(refs)}

    n   = len(refs)
    p_i = {r: 1.0 / n for r in refs}

    # 构建 p_ij：参考文献之间的归一化连接强度
    p_ij = defaultdict(float)
    for ri in refs:
        neighbors = set(G.successors(ri)) | set(G.predecessors(ri))
        common    = [rj for rj in refs if rj != ri and rj in neighbors]
        if common:
            w = 1.0 / len(common)
            for rj in common:
                p_ij[(ri, rj)] = w

    # 计算约束系数
    constraint = 0.0
    for ri in refs:
        direct   = p_i.get(ri, 0.0)
        indirect = sum(
            p_ij.get((ri, rj), 0.0) * p_i.get(rj, 0.0)
            for rj in refs if rj != ri
        )
        constraint += (direct + indirect) ** 2

    constraint_norm = min(constraint, 1.0)
    return {
        "constraint":           constraint_norm,
        "innovation_potential": 1.0 - constraint_norm,
        "n_ego_nodes":          n,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 维度七：PDE 预期扩散熵（冷启动替代知识扩散熵 H）
# ══════════════════════════════════════════════════════════════════════════════

def compute_pde(
    target_id: str,
    G: nx.DiGraph,
    level: str = "domain",
) -> dict:
    """
    预期扩散熵 PDE（Projected Diffusion Entropy）。

    PDE(p)      = -sum_k q_k * log2(q_k)   <- Shannon 熵
    PDE_norm(p) = PDE(p) / log2(K)          <- 归一化到 [0, 1]

    q_k = 参考文献中属于学科 k 的比例

    逻辑：参考文献的学科分布越均匀多样，论文被多个领域引用的概率就越高，
    预期的知识扩散范围就越广。

    与 RS 的区别：RS 加权了学科间语义距离；PDE 只看纯分布熵，
    对学科数量 K 更敏感，是 RS 的低成本互补。

    level: "domain"（推荐）| "field" | "subfield"
    """
    if target_id not in G or G.out_degree(target_id) == 0:
        return {"pde": None, "pde_norm": None, "n_disciplines": 0}

    refs = [r for r in G.successors(target_id) if r in G]
    disciplines = [G.nodes[r].get(level, "") for r in refs if G.nodes[r].get(level, "")]

    if not disciplines:
        return {"pde": None, "pde_norm": None, "n_disciplines": 0}

    counts = Counter(disciplines)
    total  = sum(counts.values())
    pde    = -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)
    pde_norm = pde / math.log2(len(counts)) if len(counts) > 1 else 0.0

    return {
        "pde":          pde,
        "pde_norm":     pde_norm,
        "n_disciplines": len(counts),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 综合计算入口
# ══════════════════════════════════════════════════════════════════════════════

def compute_all_metrics_for_paper(
    target_id: str,
    G: nx.DiGraph,
    betweenness_cache: Optional[dict] = None,
    uzzi_baseline: Optional[dict] = None,
    partition_cache: Optional[dict] = None,
) -> dict:
    """
    对单篇论文计算全部七个维度的指标。
    所有指标均只依赖参考文献，发表当天即可计算。

    Parameters
    ----------
    target_id         : 目标论文 OpenAlex ID
    G                 : 知识图谱
    betweenness_cache : 预计算中心性字典（避免重复）
    uzzi_baseline     : 预构建的期刊对基线
    partition_cache   : 预计算社区划分（避免重复）
    """
    results = {"id": target_id}

    # 维度一：桥接中心性
    if betweenness_cache and target_id in betweenness_cache:
        results["betweenness"] = betweenness_cache[target_id]
    else:
        bc = compute_betweenness(G, target_ids=[target_id], k_sample=200)
        results["betweenness"] = bc.get(target_id, 0.0)

    # 维度二：Rao-Stirling
    results["rao_stirling"] = compute_rao_stirling(target_id, G, level="field")

    # 维度三：ΔQ（图级，在 comparator 注入；此处占位）
    results["delta_q"] = None

    # 维度四：Uzzi
    uzzi_res = compute_uzzi_novelty(target_id, G, baseline=uzzi_baseline)
    results.update({f"uzzi_{k}": v for k, v in uzzi_res.items() if k != "z_scores"})

    # 维度五：RTD
    rtd_res = compute_rtd(target_id, G, partition=partition_cache)
    results.update({f"rtd_{k}": v for k, v in rtd_res.items()})

    # 维度六：Burt 约束
    burt_res = compute_burt_constraint(target_id, G)
    results.update({f"burt_{k}": v for k, v in burt_res.items()})

    # 维度七：PDE
    pde_res = compute_pde(target_id, G, level="domain")
    results.update({f"pde_{k}": v for k, v in pde_res.items()})

    return results

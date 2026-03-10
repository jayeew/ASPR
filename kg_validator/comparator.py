"""
comparator.py — 诺贝尔奖前后知识图谱对比与可视化模块（冷启动兼容版）
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import networkx as nx

from metrics import (
    compute_betweenness, compute_rao_stirling,
    compute_delta_q, compute_modularity,
    compute_uzzi_novelty, _build_journal_copair_baseline,
    compute_rtd, compute_burt_constraint, compute_pde,
    compute_all_metrics_for_paper,
)

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NobelCase:
    name:       str
    paper_id:   str
    paper_doi:  str
    nobel_year: int
    field:      str

@dataclass
class ComparisonResult:
    case:           NobelCase
    metrics_before: dict
    metrics_after:  dict
    delta_q_result: dict
    graph_stats:    dict


# 指标元数据：(显示名, 期望方向 True=奖后应升 False=奖后应降, 主键)
METRIC_META = [
    ("桥接中心性 B",      True,  "betweenness"),
    ("RS 跨学科性",       True,  "rao_stirling"),
    ("模块度 Q",          False, "delta_q"),      # ΔQ < 0 为颠覆信号
    ("Uzzi 新颖性 p10",   False, "uzzi_novelty_p10"),
    ("RTD 引用多样性",    True,  "rtd_rtd"),
    ("Burt 创新潜力",     True,  "burt_innovation_potential"),
    ("PDE 预期扩散熵",    True,  "pde_pde_norm"),
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


def _configure_plot_fonts() -> bool:
    """
    配置 Matplotlib 字体；若缺少中文字体，则回退为英文标签。

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


# ──────────────────────────────────────────────────────────────────────────────
# 核心对比逻辑
# ──────────────────────────────────────────────────────────────────────────────

def run_comparison(
    case: NobelCase,
    G_full: nx.DiGraph,
    window_before: int = 10,
    window_after:  int = 5,
    k_sample_bc:   int = 300,
) -> ComparisonResult:
    from graph_builder import slice_graph_by_year

    y = case.nobel_year
    log.info(f"\n{'='*60}")
    log.info(f"案例: {case.name}  (获奖年: {y})")
    log.info(f"{'='*60}")

    G_before = slice_graph_by_year(G_full, year_end=y-1,          year_start=y-window_before)
    G_after  = slice_graph_by_year(G_full, year_end=y+window_after, year_start=y-window_before)

    log.info(f"G_before ({y-window_before}~{y-1}): "
             f"{G_before.number_of_nodes()} 节点, {G_before.number_of_edges()} 边")
    log.info(f"G_after  ({y-window_before}~{y+window_after}): "
             f"{G_after.number_of_nodes()} 节点, {G_after.number_of_edges()} 边")

    # 预计算（重用避免重复开销）
    log.info("预计算 G_before 中心性 & 社区划分...")
    bc_before  = compute_betweenness(G_before, k_sample=k_sample_bc)
    _, part_before = compute_modularity(G_before)

    log.info("预计算 G_after 中心性 & 社区划分...")
    bc_after   = compute_betweenness(G_after,  k_sample=k_sample_bc)
    _, part_after  = compute_modularity(G_after)

    log.info("构建 Uzzi 基线...")
    uzzi_bl_before = _build_journal_copair_baseline(G_before, n_permutations=50)
    uzzi_bl_after  = _build_journal_copair_baseline(G_after,  n_permutations=50)

    log.info("计算奖前指标...")
    m_before = compute_all_metrics_for_paper(
        case.paper_id, G_before,
        betweenness_cache=bc_before,
        uzzi_baseline=uzzi_bl_before,
        partition_cache=part_before,
    )

    log.info("计算奖后指标...")
    m_after = compute_all_metrics_for_paper(
        case.paper_id, G_after,
        betweenness_cache=bc_after,
        uzzi_baseline=uzzi_bl_after,
        partition_cache=part_after,
    )

    log.info("计算 ΔQ...")
    dq_result = compute_delta_q(G_before, G_after)
    m_before["delta_q"] = dq_result["Q_before"]
    m_after["delta_q"]  = dq_result["Q_after"]

    graph_stats = {
        "before_nodes":   G_before.number_of_nodes(),
        "before_edges":   G_before.number_of_edges(),
        "after_nodes":    G_after.number_of_nodes(),
        "after_edges":    G_after.number_of_edges(),
        "before_density": nx.density(G_before),
        "after_density":  nx.density(G_after),
        "before_comm":    dq_result["n_communities_before"],
        "after_comm":     dq_result["n_communities_after"],
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


def _print_comparison_table(r: ComparisonResult):
    mb, ma = r.metrics_before, r.metrics_after
    print(f"\n{'─'*68}")
    print(f"  {r.case.name}")
    print(f"{'─'*68}")
    print(f"  {'指标':<26} {'奖前':>12} {'奖后':>12} {'变化':>12}")
    print(f"{'─'*68}")
    for name, expect_up, key in METRIC_META:
        bv = mb.get(key)
        av = ma.get(key)
        b_str = f"{bv:.4f}" if bv is not None else "   N/A  "
        a_str = f"{av:.4f}" if av is not None else "   N/A  "
        if bv is not None and av is not None:
            delta = av - bv
            arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "—")
            match = "✅" if (delta > 0) == expect_up else "⚠️"
            d_str = f"{match} {arrow}{abs(delta):.4f}"
        else:
            d_str = "   N/A"
        print(f"  {name:<26} {b_str:>12} {a_str:>12} {d_str:>12}")
    print(f"{'─'*68}")
    print(f"  社区数: {r.graph_stats['before_comm']} → {r.graph_stats['after_comm']}"
          f"   ΔQ = {r.delta_q_result['delta_Q']:+.4f}"
          f"  ({'社区破壁 ✅' if r.delta_q_result['delta_Q'] < 0 else '社区加深'})")
    print(f"{'─'*68}\n")


# ──────────────────────────────────────────────────────────────────────────────
# 可视化
# ──────────────────────────────────────────────────────────────────────────────

def plot_before_after_bars(results: list[ComparisonResult],
                           save_path: str = "metrics_bar.png"):
    """
    七维指标奖前/奖后柱状对比图（每个指标一个子图）。
    """
    keys   = [m[2] for m in METRIC_META]
    labels = [_metric_plot_label(m[0], m[2]) for m in METRIC_META]
    n_metrics = len(keys)
    n_cases   = len(results)
    before_label, after_label = _plot_phase_labels()

    ncols = 4
    nrows = math.ceil(n_metrics / ncols)

    import math as _math
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for idx, (key, label, (_, expect_up, __)) in enumerate(
            zip(keys, labels, METRIC_META)):
        ax = axes[idx]
        case_names  = [r.case.name.split("(")[0].strip()[:16] for r in results]
        before_vals = [r.metrics_before.get(key) or 0 for r in results]
        after_vals  = [r.metrics_after.get(key)  or 0 for r in results]

        x     = range(n_cases)
        width = 0.35
        ax.bar([xi - width/2 for xi in x], before_vals, width,
               label=before_label, color="#2563A8", alpha=0.85)
        ax.bar([xi + width/2 for xi in x], after_vals,  width,
               label=after_label, color="#F0A500", alpha=0.85)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xticks(list(x))
        ax.set_xticklabels(case_names, fontsize=7, rotation=20, ha="right")
        ax.legend(fontsize=7)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        direction = "expect UP" if expect_up else "expect DOWN"
        ax.text(0.02, 0.96, direction, transform=ax.transAxes,
                fontsize=7, color="#16a34a" if expect_up else "#dc2626", va="top")

    # 隐藏多余子图
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Seven-Dimension Innovation Metrics: Before vs After Nobel Prize",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"柱状图已保存: {save_path}")


def plot_radar(results: list[ComparisonResult],
               save_path: str = "radar.png"):
    """雷达图：七维指标奖前/奖后对比。"""
    import math as _math
    keys   = [m[2] for m in METRIC_META]
    labels = [_metric_plot_label(m[0], m[2]) for m in METRIC_META]
    N      = len(keys)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    before_label, after_label = _plot_phase_labels()

    fig, axes = plt.subplots(
        1, len(results), figsize=(6 * len(results), 6),
        subplot_kw={"polar": True}
    )
    if len(results) == 1:
        axes = [axes]

    def norm_val(v):
        if v is None: return 0.5
        return min(max(float(v), -1), 2) / 2 + 0.5

    for ax, r in zip(axes, results):
        bv = [norm_val(r.metrics_before.get(k)) for k in keys] 
        av = [norm_val(r.metrics_after.get(k))  for k in keys]
        bv += bv[:1]; av += av[:1]

        ax.plot(angles, bv, "o-", lw=2, color="#2563A8", label=before_label, alpha=0.85)
        ax.fill(angles, bv, alpha=0.12, color="#2563A8")
        ax.plot(angles, av, "s-", lw=2, color="#F0A500", label=after_label,  alpha=0.85)
        ax.fill(angles, av, alpha=0.12, color="#F0A500")

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=8)
        ax.set_ylim(0, 1)
        ax.set_title(r.case.name, size=10, pad=14, fontweight="bold")
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

    plt.suptitle("Innovation Metrics Radar: Before vs After Nobel Prize",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"雷达图已保存: {save_path}")


def plot_modularity_timeline(
    results: list[ComparisonResult],
    G_full: nx.DiGraph,
    save_path: str = "modularity_timeline.png",
):
    """模块度 Q 时间序列图，在获奖年份画垂直标注线。"""
    from graph_builder import slice_graph_by_year

    fig, ax = plt.subplots(figsize=(12, 5))
    colors  = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for r, color in zip(results, colors):
        y     = r.case.nobel_year
        years, q_vals = [], []
        for yr in range(y - 8, y + 6):
            G_sl = slice_graph_by_year(G_full, year_end=yr, year_start=yr - 5)
            if G_sl.number_of_edges() < 10:
                continue
            Q, _ = compute_modularity(G_sl)
            years.append(yr); q_vals.append(Q)
        if not years:
            continue
        label = r.case.name.split("(")[0].strip()
        ax.plot(years, q_vals, marker="o", lw=2, color=color, label=label)
        ax.axvline(x=y, color=color, linestyle="--", alpha=0.5)

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
    G_before:  nx.DiGraph,
    G_after:   nx.DiGraph,
    save_path: str = "ego_network.png",
    max_nodes: int = 80,
):
    """获奖论文 ego 网络：奖前 vs 奖后。"""
    def get_ego(G, center, radius=2):
        if center not in G:
            return nx.DiGraph()
        ego = nx.ego_graph(G.to_undirected(), center, radius=radius)
        if ego.number_of_nodes() > max_nodes:
            top = sorted(ego.degree(), key=lambda x: x[1], reverse=True)[:max_nodes]
            ego = ego.subgraph([n for n, _ in top]).copy()
        return ego

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, ego, title in [
        (axes[0], get_ego(G_before, target_id), "Before Nobel"),
        (axes[1], get_ego(G_after,  target_id), "After Nobel"),
    ]:
        if ego.number_of_nodes() == 0:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)
            ax.set_title(title); continue

        pos     = nx.spring_layout(ego, seed=42, k=0.8)
        domains = list(set(ego.nodes[n].get("domain", "Other") for n in ego.nodes))
        palette = plt.cm.Set3(np.linspace(0, 1, max(len(domains), 1)))
        d_color = {d: palette[i] for i, d in enumerate(domains)}

        node_sizes  = [max(30, min(ego.nodes[n].get("cited_by_count", 1) * 2, 500))
                       for n in ego.nodes]
        node_colors = ["red" if n == target_id
                       else d_color.get(ego.nodes[n].get("domain", "Other"), (.8,.8,.8,1))
                       for n in ego.nodes]

        nx.draw_networkx(ego, pos, ax=ax, node_size=node_sizes,
                         node_color=node_colors, with_labels=False,
                         edge_color="gray", alpha=0.7, arrows=False, width=0.5)
        ax.set_title(f"{title}\n({ego.number_of_nodes()} nodes, "
                     f"{ego.number_of_edges()} edges)", fontsize=11)
        ax.axis("off")
        legend_els = [plt.scatter([], [], c=[d_color.get(d, (.8,.8,.8,1))],
                                  s=40, label=d) for d in domains[:6]]
        ax.legend(handles=legend_els, loc="lower left", fontsize=7, title="Domain")

    plt.suptitle("Ego Network Comparison (Red = Nobel paper)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Ego 网络图已保存: {save_path}")


def export_results_to_csv(results: list[ComparisonResult],
                          save_path: str = "results.csv") -> pd.DataFrame:
    rows = []
    for r in results:
        for phase, metrics in [("before", r.metrics_before), ("after", r.metrics_after)]:
            row = {
                "case":       r.case.name,
                "paper_id":   r.case.paper_id,
                "nobel_year": r.case.nobel_year,
                "phase":      phase,
            }
            for _, _, key in METRIC_META:
                row[key] = metrics.get(key)
            row["delta_Q_overall"] = r.delta_q_result.get("delta_Q")
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    log.info(f"结果已导出: {save_path}")
    return df


# 补充：math import for plot function
import math

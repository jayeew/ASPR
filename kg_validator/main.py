"""
main.py — 主程序入口

使用方式:
    # 演示模式（合成数据，无需网络）
    python main.py --mode demo

    # 完整模式（从 OpenAlex 拉取真实数据，需要网络）
    python main.py --mode full --email your@email.com
"""

import sys
import json
import random
import logging
import argparse
from pathlib import Path

import networkx as nx
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("kg_validator.log", encoding="utf-8"),
    ]
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 预置诺贝尔奖案例
# ──────────────────────────────────────────────────────────────────────────────

NOBEL_CASES_CONFIG = [
    # {
    #     "name":       "CRISPR (Chemistry 2020)",
    #     "paper_doi":  "10.1126/science.1225829",
    #     "paper_id":   "W2038196424",
    #     "nobel_year": 2020,
    #     "field":      "Biochemistry",
    # },
    {
        "name":       "AlphaFold (Chemistry 2024)",
        "paper_doi":  "10.1038/s41586-021-03819-2",
        "paper_id":   "W3177828909",
        "nobel_year": 2024,
        "field":      "Computational Biology",
    },
    # {
    #     "name":       "Gravitational Waves (Physics 2017)",
    #     "paper_doi":  "10.1103/PhysRevLett.116.061102",
    #     "paper_id":   "W2284278991",
    #     "nobel_year": 2017,
    #     "field":      "Physics",
    # },
]

# ──────────────────────────────────────────────────────────────────────────────
# 合成图（Demo 模式）
# ──────────────────────────────────────────────────────────────────────────────

def build_synthetic_graph(
    seed_id: str = "W_SEED",
    nobel_year: int = 2015,
    n_before: int = 300,
    n_after_extra: int = 150,
    seed: int = 42,
) -> nx.DiGraph:
    """
    构造合成知识图谱，模拟颠覆性创新的发生：
    - 奖前：三个高模块度的独立学科社区
    - 获奖论文：跨三社区引用（高 RS / RTD / PDE / Burt IP）
    - 奖后：新增论文大量引用获奖论文，且跨社区混引（模块度下降）
    """
    rng = random.Random(seed)
    np.random.seed(seed)
    G   = nx.DiGraph()

    domains = ["Biology", "Physics", "Chemistry"]
    fields  = {
        "Biology":   ["Molecular Biology", "Genetics", "Cell Biology"],
        "Physics":   ["Quantum Physics", "Astrophysics", "Optics"],
        "Chemistry": ["Organic Chemistry", "Biochemistry", "Materials"],
    }

    community_nodes: dict[str, list[str]] = {d: [] for d in domains}

    # 奖前基础节点
    for i in range(n_before):
        domain  = domains[i % 3]
        node_id = f"W_{domain[:3].upper()}_{i:04d}"
        G.add_node(node_id,
                   year=rng.randint(nobel_year - 10, nobel_year - 1),
                   domain=domain, field=rng.choice(fields[domain]),
                   subfield="", journal=f"Journal of {domain}",
                   cited_by_count=rng.randint(1, 200),
                   title=f"Paper on {domain} #{i}",
                   doi=f"10.0000/fake.{i:04d}")
        community_nodes[domain].append(node_id)

    # 社区内部引用（高密度）
    for domain, nodes in community_nodes.items():
        for n in nodes:
            refs = rng.sample([x for x in nodes if x != n], min(rng.randint(3, 8), len(nodes)-1))
            for r in refs:
                G.add_edge(n, r)

    # 少量跨社区引用
    all_nodes = list(G.nodes)
    for _ in range(int(n_before * 0.05)):
        a, b = rng.choice(all_nodes), rng.choice(all_nodes)
        if G.nodes[a].get("domain") != G.nodes[b].get("domain") and a != b:
            G.add_edge(a, b)

    # 获奖论文：跨三社区引用（所有冷启动指标应高）
    G.add_node(seed_id,
               year=nobel_year - 3, domain="Biology", field="Molecular Biology",
               subfield="CRISPR", journal="Science", cited_by_count=5000,
               title="The Nobel Prize Winning Paper", doi="10.0000/seed.0001")

    seed_refs = []
    for domain, nodes in community_nodes.items():
        for r in rng.sample(nodes, min(6, len(nodes))):
            G.add_edge(seed_id, r)
            seed_refs.append(r)

    # 奖后新增论文（80% 颠覆型引用模式）
    for i in range(n_after_extra):
        node_id  = f"W_NEW_{i:04d}"
        domain   = rng.choice(domains)
        cd_type  = "disrupting" if rng.random() < 0.80 else "consolidating"
        G.add_node(node_id,
                   year=rng.randint(nobel_year, nobel_year + 5),
                   domain=domain, field=rng.choice(fields[domain]),
                   subfield="", cited_by_count=rng.randint(1, 100),
                   journal=rng.choice(["Nature", "Science", "Cell", "PNAS"]),
                   title=f"Post-award paper #{i} ({cd_type})",
                   doi=f"10.0000/new.{i:04d}")
        G.add_edge(node_id, seed_id)
        if cd_type == "disrupting":
            others = [n for n in all_nodes if n not in seed_refs]
            for r in rng.sample(others, min(3, len(others))):
                G.add_edge(node_id, r)
        else:
            for r in rng.sample(seed_refs, min(3, len(seed_refs))):
                G.add_edge(node_id, r)
        for r in rng.sample(all_nodes, min(2, len(all_nodes))):
            G.add_edge(node_id, r)

    log.info(f"合成图构建完成: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    return G


# ──────────────────────────────────────────────────────────────────────────────
# Demo 模式
# ──────────────────────────────────────────────────────────────────────────────

def run_demo_mode(output_dir: Path):
    from graph_builder import slice_graph_by_year
    from comparator import (
        NobelCase, run_comparison,
        plot_radar, plot_before_after_bars,
        plot_ego_network, plot_modularity_timeline,
        export_results_to_csv, METRIC_META,
    )

    SEED_ID    = "W_SEED"
    NOBEL_YEAR = 2015

    log.info("【Demo】构建合成知识图谱...")
    G_full = build_synthetic_graph(seed_id=SEED_ID, nobel_year=NOBEL_YEAR,
                                   n_before=300, n_after_extra=150)

    case   = NobelCase(
        name="Synthetic Disruptive Innovation (Demo 2015)",
        paper_id=SEED_ID, paper_doi="10.0000/seed.0001",
        nobel_year=NOBEL_YEAR, field="Biology",
    )
    result = run_comparison(case, G_full, window_before=10, window_after=5)

    log.info("\n生成可视化图表...")
    plot_radar([result],              save_path=str(output_dir / "radar_demo.png"))
    plot_before_after_bars([result],  save_path=str(output_dir / "bars_demo.png"))

    G_before = slice_graph_by_year(G_full, year_end=NOBEL_YEAR-1, year_start=NOBEL_YEAR-10)
    G_after  = slice_graph_by_year(G_full, year_end=NOBEL_YEAR+5, year_start=NOBEL_YEAR-10)
    plot_ego_network(SEED_ID, G_before, G_after,
                     save_path=str(output_dir / "ego_demo.png"))
    plot_modularity_timeline([result], G_full,
                             save_path=str(output_dir / "modularity_timeline_demo.png"))

    df = export_results_to_csv([result], save_path=str(output_dir / "results_demo.csv"))

    # 验证预期方向
    log.info("\n【指标变化方向验证】")
    mb, ma  = result.metrics_before, result.metrics_after
    all_ok  = True
    for name, expect_up, key in METRIC_META:
        if key == "delta_q":
            dq     = result.delta_q_result.get("delta_Q", 0)
            passed = dq < 0
            info   = f"ΔQ={dq:+.4f}"
        else:
            bv, av = mb.get(key), ma.get(key)
            if bv is None or av is None:
                log.info(f"  ⚠️  {name:<26} 数据不足，跳过")
                continue
            passed = (av > bv) if expect_up else (av < bv)
            info   = f"{bv:.4f} → {av:.4f}"
        sym  = "✅" if passed else "❌"
        log.info(f"  {sym}  {name:<26} {info}")
        if not passed:
            all_ok = False

    verdict = ("✅ 全部指标变化方向符合预期！体系设计可行。"
               if all_ok else
               "⚠️  部分指标未达预期，请检查数据完整性。")
    log.info(f"\n{verdict}")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Full 模式
# ──────────────────────────────────────────────────────────────────────────────

def run_full_mode(args):
    from fetcher import fetch_works_cursor, fetch_works_batch_ids, normalize_work
    from graph_builder import build_graph, build_node_attributes, write_graphml_safe
    from comparator import (NobelCase, run_comparison, plot_radar,
                             plot_before_after_bars, export_results_to_csv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    all_results = []

    for cfg in NOBEL_CASES_CONFIG:
        log.info(f"\n拉取案例: {cfg['name']}")
        y = cfg["nobel_year"]
        raw_works = fetch_works_cursor(
            f"primary_location.source.id:S137773608,"
            f"publication_year:{y-12}-{y+6},type:article",
            email=args.email,
            max_records=args.max_records,
        )
        works = [normalize_work(w) for w in raw_works]
        G_full = build_graph(works)

        # 确保目标论文在图中
        if cfg["paper_id"] not in G_full:
            extra = fetch_works_batch_ids([cfg["paper_id"]], email=args.email)
            for w in [normalize_work(e) for e in extra]:
                nid = w["id"]
                G_full.add_node(nid, **build_node_attributes(w))
                for ref in w["referenced_works"]:
                    G_full.add_edge(nid, ref)

        write_graphml_safe(G_full, output_dir / f"kg_{cfg['paper_id']}.graphml")

        case   = NobelCase(name=cfg["name"], paper_id=cfg["paper_id"],
                           paper_doi=cfg["paper_doi"], nobel_year=cfg["nobel_year"],
                           field=cfg["field"])
        result = run_comparison(case, G_full, window_before=10, window_after=5)
        all_results.append(result)

    plot_radar(all_results,             save_path=str(output_dir / "radar.png"))
    plot_before_after_bars(all_results, save_path=str(output_dir / "bars.png"))
    export_results_to_csv(all_results,  save_path=str(output_dir / "results.csv"))
    log.info(f"\n所有结果已保存至 {output_dir}/")


# ──────────────────────────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",        choices=["full", "demo"], default="full")
    p.add_argument("--email",       default="jayeew@qq.com")
    p.add_argument("--output_dir",  default="output")
    p.add_argument("--max_records", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args       = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if args.mode == "demo":
        run_demo_mode(output_dir)
    else:
        run_full_mode(args)

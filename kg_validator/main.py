"""
main.py — 主程序入口

使用方式:
    # 演示模式（合成数据，无需网络）
    python main.py --mode demo

    # 完整模式（从 OpenAlex 拉取真实数据，需要网络）
    python main.py --mode full --email your@email.com

    # 领域图谱时间节点前后对比
    python main.py --mode field_contrast --filter "concepts.id:C123,type:article" \
        --event-year 2024 --event-label "Nobel Prize 2024"
"""

import argparse
import logging
import random
import re
import sys
from pathlib import Path

import networkx as nx
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("kg_validator.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


def _split_cli_list(raw_value: str) -> list[str]:
    """
    将逗号分隔的 CLI 参数拆成列表。

    Args:
        raw_value: 原始字符串。

    Returns:
        list[str]: 去空白后的条目列表。
    """
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _slugify(value: str) -> str:
    """
    生成稳定文件名 slug。

    Args:
        value: 原始文本。

    Returns:
        str: 仅含字母、数字和下划线的 slug。
    """
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip().lower()).strip("_")
    return slug or "paper_contrast"


def _deduplicate_works_by_id(works: list[dict]) -> list[dict]:
    """
    依据论文 ID 去重，并保留最后一次出现的记录。

    Args:
        works: 标准化后的论文列表。

    Returns:
        list[dict]: 去重后的论文列表。
    """
    deduped: dict[str, dict] = {}
    for work in works:
        work_id = work.get("id")
        if work_id:
            deduped[work_id] = work
    return list(deduped.values())


def _filter_works_by_year(works: list[dict], year_start: int, year_end: int) -> list[dict]:
    """
    过滤落在指定年份区间内的论文。

    Args:
        works: 标准化后的论文列表。
        year_start: 起始年份。
        year_end: 结束年份。

    Returns:
        list[dict]: 过滤后的论文列表。
    """
    filtered = []
    for work in works:
        year = work.get("year")
        if year is None:
            continue
        if year_start <= year <= year_end:
            filtered.append(work)
    return filtered


def _load_target_works(args: argparse.Namespace) -> list[dict]:
    """
    根据 DOI 或 OpenAlex ID 拉取目标论文。

    Args:
        args: 命令行参数。

    Returns:
        list[dict]: 标准化后的目标论文列表。
    """
    from fetcher import fetch_works_batch_ids, fetch_works_by_doi, normalize_work

    raw_target_works: list[dict] = []
    paper_dois = _split_cli_list(args.paper_dois)
    paper_ids = _split_cli_list(args.paper_ids)

    if paper_dois:
        raw_target_works.extend(fetch_works_by_doi(paper_dois, email=args.email))
    if paper_ids:
        raw_target_works.extend(fetch_works_batch_ids(paper_ids, email=args.email))

    target_works = _deduplicate_works_by_id([normalize_work(item) for item in raw_target_works])
    if not target_works:
        raise RuntimeError("未能根据 --paper-ids / --paper-dois 拉取到目标论文。")
    return target_works


def build_paper_neighborhood_graph(
    target_works: list[dict],
    email: str = "",
    before_years: int = 10,
    after_years: int = 5,
    max_refs: int = 20,
    citers_per_target: int = 120,
    citers_per_ref: int = 30,
) -> nx.DiGraph:
    """
    构建纯论文邻域图：目标论文 + 参考文献 + 施引论文 + 参考文献的施引邻域。

    Args:
        target_works: 目标论文列表。
        email: OpenAlex mailto。
        before_years: 向前观察窗口。
        after_years: 向后观察窗口。
        max_refs: 每篇目标论文纳入的最大参考文献数。
        citers_per_target: 每篇目标论文拉取的施引论文上限。
        citers_per_ref: 每篇参考文献拉取的施引论文上限。

    Returns:
        nx.DiGraph: 邻域知识图谱。
    """
    from fetcher import fetch_citing_works, fetch_works_batch_ids, normalize_work
    from graph_builder import build_graph, build_node_attributes

    target_years = [work["year"] for work in target_works if work.get("year") is not None]
    if not target_years:
        raise RuntimeError("目标论文缺少 publication_year，无法构建邻域图。")

    year_start = min(target_years) - before_years
    year_end = max(target_years) + after_years

    target_ids = [work["id"] for work in target_works]
    reference_ids: list[str] = []
    for work in target_works:
        reference_ids.extend(work.get("referenced_works", [])[:max_refs])
    reference_ids = list(dict.fromkeys([ref_id for ref_id in reference_ids if ref_id]))

    normalized_refs = _deduplicate_works_by_id(
        [normalize_work(item) for item in fetch_works_batch_ids(reference_ids, email=email)]
    )

    citing_target_works: list[dict] = []
    for target_id in target_ids:
        citing_target_works.extend(
            normalize_work(item)
            for item in fetch_citing_works(
                target_id,
                email=email,
                max_records=citers_per_target,
            )
        )

    citing_reference_works: list[dict] = []
    for ref_id in reference_ids:
        citing_reference_works.extend(
            normalize_work(item)
            for item in fetch_citing_works(
                ref_id,
                email=email,
                max_records=citers_per_ref,
            )
        )

    candidate_works = _deduplicate_works_by_id(
        target_works + normalized_refs + citing_target_works + citing_reference_works
    )
    candidate_works = _filter_works_by_year(candidate_works, year_start=year_start, year_end=year_end)
    G = build_graph(candidate_works)

    for work in target_works:
        node_id = work["id"]
        if node_id not in G:
            G.add_node(node_id, **build_node_attributes(work))
        else:
            G.nodes[node_id].update(build_node_attributes(work))
        for ref_id in work["referenced_works"][:max_refs]:
            G.add_edge(node_id, ref_id)

    log.info(
        "论文邻域图构建完成: %s 节点, %s 边, refs=%s, citing_target=%s, citing_refs=%s",
        G.number_of_nodes(),
        G.number_of_edges(),
        len(reference_ids),
        len(citing_target_works),
        len(citing_reference_works),
    )
    return G


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
        "name": "AlphaFold (Chemistry 2024)",
        "paper_doi": "10.1038/s41586-021-03819-2",
        "paper_id": "W3177828909",
        "nobel_year": 2024,
        "field": "Computational Biology",
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
    构造合成知识图谱，模拟颠覆性创新发生后的显著结构变化。

    - 奖前：三个高模块度独立社区
    - 种子论文：跨三社区引用
    - 奖后：形成一个围绕种子论文的新紧密社群，同时保留一般扩散论文

    Args:
        seed_id: 颠覆性创新论文 ID。
        nobel_year: 事件年份。
        n_before: 事件前论文数。
        n_after_extra: 事件后新增论文数。
        seed: 随机种子。

    Returns:
        nx.DiGraph: 合成知识图谱。
    """
    rng = random.Random(seed)
    np.random.seed(seed)
    G = nx.DiGraph()

    domains = ["Biology", "Physics", "Chemistry"]
    fields = {
        "Biology": ["Molecular Biology", "Genetics", "Cell Biology"],
        "Physics": ["Quantum Physics", "Astrophysics", "Optics"],
        "Chemistry": ["Organic Chemistry", "Biochemistry", "Materials"],
    }
    topics = {
        "Biology": "Gene Regulation|Protein Interaction|Cell Signaling",
        "Physics": "Wave Function|Detector Design|Optical Measurement",
        "Chemistry": "Catalysis|Reaction Design|Molecular Assembly",
    }

    community_nodes: dict[str, list[str]] = {domain: [] for domain in domains}

    for index in range(n_before):
        domain = domains[index % 3]
        node_id = f"W_{domain[:3].upper()}_{index:04d}"
        field_name = rng.choice(fields[domain])
        G.add_node(
            node_id,
            year=rng.randint(nobel_year - 10, nobel_year - 1),
            domain=domain,
            field=field_name,
            subfield="",
            journal=f"Journal of {domain}",
            cited_by_count=rng.randint(1, 200),
            title=f"Paper on {domain} #{index}",
            doi=f"10.0000/fake.{index:04d}",
            topic_names=topics[domain],
        )
        community_nodes[domain].append(node_id)

    for domain, node_ids in community_nodes.items():
        for node_id in node_ids:
            sampled_refs = rng.sample(
                [candidate for candidate in node_ids if candidate != node_id],
                min(rng.randint(3, 8), len(node_ids) - 1),
            )
            for ref_id in sampled_refs:
                G.add_edge(node_id, ref_id)

    all_before_nodes = list(G.nodes)
    for _ in range(int(n_before * 0.05)):
        source = rng.choice(all_before_nodes)
        target = rng.choice(all_before_nodes)
        if source != target and G.nodes[source].get("domain") != G.nodes[target].get("domain"):
            G.add_edge(source, target)

    G.add_node(
        seed_id,
        year=nobel_year - 3,
        domain="Biology",
        field="Molecular Biology",
        subfield="Breakthrough Integration",
        journal="Science",
        cited_by_count=5000,
        title="The Nobel Prize Winning Paper",
        doi="10.0000/seed.0001",
        topic_names="Protein Folding|Representation Learning|Structural Biology",
    )

    seed_refs: list[str] = []
    for domain, node_ids in community_nodes.items():
        for ref_id in rng.sample(node_ids, min(6, len(node_ids))):
            G.add_edge(seed_id, ref_id)
            seed_refs.append(ref_id)

    n_emergent = max(42, n_after_extra // 3)
    n_general = max(0, n_after_extra - n_emergent)
    emergent_nodes: list[str] = []

    for index in range(n_emergent):
        node_id = f"W_EMG_{index:04d}"
        emergent_nodes.append(node_id)
        G.add_node(
            node_id,
            year=rng.randint(nobel_year, nobel_year + 2),
            domain="Computational Biology",
            field="AI for Structural Biology",
            subfield="Protein Folding Foundation Models",
            journal=rng.choice(["Nature", "Science", "Cell", "Nature Methods"]),
            cited_by_count=rng.randint(20, 600),
            title=f"Emergent post-event paper #{index}",
            doi=f"10.0000/emergent.{index:04d}",
            topic_names="Protein Folding|Deep Learning|Structure Prediction",
        )

    for node_id in emergent_nodes:
        peer_refs = rng.sample(
            [candidate for candidate in emergent_nodes if candidate != node_id],
            min(7, len(emergent_nodes) - 1),
        )
        for ref_id in peer_refs:
            G.add_edge(node_id, ref_id)
        G.add_edge(node_id, seed_id)
        for ref_id in rng.sample(seed_refs, min(2, len(seed_refs))):
            G.add_edge(node_id, ref_id)

    for _ in range(max(6, len(emergent_nodes) // 5)):
        source = rng.choice(emergent_nodes)
        target = rng.choice(emergent_nodes)
        if source != target:
            G.add_edge(source, target)

    for index in range(n_general):
        node_id = f"W_NEW_{index:04d}"
        domain = rng.choice(domains)
        field_name = rng.choice(fields[domain])
        citation_mode = "disrupting" if rng.random() < 0.75 else "consolidating"

        G.add_node(
            node_id,
            year=rng.randint(nobel_year, nobel_year + 5),
            domain=domain,
            field=field_name,
            subfield="",
            cited_by_count=rng.randint(1, 120),
            journal=rng.choice(["Nature", "Science", "Cell", "PNAS"]),
            title=f"Post-award paper #{index} ({citation_mode})",
            doi=f"10.0000/new.{index:04d}",
            topic_names=topics[domain],
        )
        G.add_edge(node_id, seed_id)

        if citation_mode == "disrupting":
            historical_nodes = [node for node in all_before_nodes if node not in seed_refs]
            for ref_id in rng.sample(historical_nodes, min(3, len(historical_nodes))):
                G.add_edge(node_id, ref_id)
            for ref_id in rng.sample(emergent_nodes, min(2, len(emergent_nodes))):
                G.add_edge(node_id, ref_id)
        else:
            for ref_id in rng.sample(seed_refs, min(3, len(seed_refs))):
                G.add_edge(node_id, ref_id)
            if rng.random() < 0.35:
                G.add_edge(node_id, rng.choice(emergent_nodes))

        for ref_id in rng.sample(all_before_nodes, min(2, len(all_before_nodes))):
            G.add_edge(node_id, ref_id)

    log.info(
        "合成图构建完成: %s 节点, %s 边, 新社群节点=%s",
        G.number_of_nodes(),
        G.number_of_edges(),
        len(emergent_nodes),
    )
    return G


# ──────────────────────────────────────────────────────────────────────────────
# Demo 模式
# ──────────────────────────────────────────────────────────────────────────────

def run_demo_mode(output_dir: Path):
    """
    运行离线 demo，生成七维指标图和领域前后图谱三联图。

    Args:
        output_dir: 输出目录。

    Returns:
        tuple: ComparisonResult 与 FieldContrastResult。
    """
    from graph_builder import slice_graph_by_year
    from comparator import (
        ComparisonResult,
        FieldContrastResult,
        FieldContrastSpec,
        METRIC_META,
        NobelCase,
        export_results_to_csv,
        plot_before_after_bars,
        plot_ego_network,
        plot_modularity_timeline,
        plot_radar,
        run_comparison,
        run_field_contrast,
    )

    seed_id = "W_SEED"
    nobel_year = 2015

    log.info("【Demo】构建合成知识图谱...")
    G_full = build_synthetic_graph(seed_id=seed_id, nobel_year=nobel_year, n_before=300, n_after_extra=150)

    case = NobelCase(
        name="Synthetic Disruptive Innovation (Demo 2015)",
        paper_id=seed_id,
        paper_doi="10.0000/seed.0001",
        nobel_year=nobel_year,
        field="Biology",
    )
    result: ComparisonResult = run_comparison(case, G_full, window_before=10, window_after=5)

    log.info("\n生成七维指标图表...")
    plot_radar([result], save_path=str(output_dir / "radar_demo.png"))
    plot_before_after_bars([result], save_path=str(output_dir / "bars_demo.png"))

    G_before = slice_graph_by_year(G_full, year_end=nobel_year - 1, year_start=nobel_year - 10)
    G_after = slice_graph_by_year(G_full, year_end=nobel_year + 5, year_start=nobel_year - 10)
    plot_ego_network(seed_id, G_before, G_after, save_path=str(output_dir / "ego_demo.png"))
    plot_modularity_timeline([result], G_full, save_path=str(output_dir / "modularity_timeline_demo.png"))
    export_results_to_csv([result], save_path=str(output_dir / "results_demo.csv"))

    log.info("\n生成领域前后图谱三联图...")
    contrast_spec = FieldContrastSpec(
        filter_query="synthetic_demo",
        event_year=nobel_year,
        event_label="Synthetic Breakthrough",
        before_years=10,
        after_years=5,
        max_plot_nodes=180,
        min_community_size=8,
        slug="demo",
    )
    field_result: FieldContrastResult = run_field_contrast(contrast_spec, G_full, output_dir=output_dir)

    if not field_result.emergent_communities:
        raise RuntimeError("Demo synthetic graph should detect at least one emergent community.")
    if field_result.graph_stats["after_nodes"] <= field_result.graph_stats["before_nodes"]:
        raise RuntimeError("Demo synthetic graph should grow after the event year.")
    if field_result.graph_stats["after_edges"] <= field_result.graph_stats["before_edges"]:
        raise RuntimeError("Demo synthetic graph should add edges after the event year.")

    log.info("\n【指标变化方向验证】")
    metrics_before = result.metrics_before
    metrics_after = result.metrics_after
    all_ok = True
    for name, expect_up, key in METRIC_META:
        if key == "delta_q":
            delta_q = result.delta_q_result.get("delta_Q", 0)
            passed = delta_q < 0
            info = f"ΔQ={delta_q:+.4f}"
        else:
            before_value = metrics_before.get(key)
            after_value = metrics_after.get(key)
            if before_value is None or after_value is None:
                log.info("  ⚠️  %-26s 数据不足，跳过", name)
                continue
            passed = (after_value > before_value) if expect_up else (after_value < before_value)
            info = f"{before_value:.4f} → {after_value:.4f}"
        symbol = "✅" if passed else "❌"
        log.info("  %s  %-26s %s", symbol, name, info)
        if not passed:
            all_ok = False

    verdict = (
        "✅ 全部指标变化方向符合预期！体系设计可行。"
        if all_ok else
        "⚠️  部分指标未达预期，请检查数据完整性。"
    )
    log.info("\n%s", verdict)
    return result, field_result


# ──────────────────────────────────────────────────────────────────────────────
# Full 模式
# ──────────────────────────────────────────────────────────────────────────────

def run_full_mode(args: argparse.Namespace) -> None:
    """
    运行既有 full 模式。

    Args:
        args: 命令行参数。
    """
    from comparator import (
        NobelCase,
        export_results_to_csv,
        plot_before_after_bars,
        plot_radar,
        run_comparison,
    )
    from fetcher import fetch_works_batch_ids, fetch_works_cursor, normalize_work
    from graph_builder import build_graph, build_node_attributes, write_graphml_safe

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    all_results = []

    for cfg in NOBEL_CASES_CONFIG:
        log.info("\n拉取案例: %s", cfg["name"])
        nobel_year = cfg["nobel_year"]
        raw_works = fetch_works_cursor(
            f"primary_location.source.id:S137773608,publication_year:{nobel_year-12}-{nobel_year+6},type:article",
            email=args.email,
            max_records=args.max_records,
        )
        works = [normalize_work(work) for work in raw_works]
        G_full = build_graph(works)

        if cfg["paper_id"] not in G_full:
            extra_works = fetch_works_batch_ids([cfg["paper_id"]], email=args.email)
            for work in [normalize_work(item) for item in extra_works]:
                node_id = work["id"]
                G_full.add_node(node_id, **build_node_attributes(work))
                for ref_id in work["referenced_works"]:
                    G_full.add_edge(node_id, ref_id)

        write_graphml_safe(G_full, output_dir / f"kg_{cfg['paper_id']}.graphml")

        case = NobelCase(
            name=cfg["name"],
            paper_id=cfg["paper_id"],
            paper_doi=cfg["paper_doi"],
            nobel_year=cfg["nobel_year"],
            field=cfg["field"],
        )
        all_results.append(run_comparison(case, G_full, window_before=10, window_after=5))

    plot_radar(all_results, save_path=str(output_dir / "radar.png"))
    plot_before_after_bars(all_results, save_path=str(output_dir / "bars.png"))
    export_results_to_csv(all_results, save_path=str(output_dir / "results.csv"))
    log.info("\n所有结果已保存至 %s/", output_dir)


# ──────────────────────────────────────────────────────────────────────────────
# 领域图谱时间节点前后对比模式
# ──────────────────────────────────────────────────────────────────────────────

def run_field_contrast_mode(args: argparse.Namespace) -> None:
    """
    运行领域图谱时间节点前后对比模式。

    Args:
        args: 命令行参数。
    """
    from comparator import FieldContrastSpec, run_field_contrast
    from fetcher import fetch_works_cursor, normalize_work
    from graph_builder import build_graph

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    year_start = args.event_year - args.before_years
    year_end = args.event_year + args.after_years
    combined_filter = f"{args.filter_query},publication_year:{year_start}-{year_end}"

    log.info("拉取领域图谱数据: %s", combined_filter)
    raw_works = fetch_works_cursor(
        combined_filter,
        email=args.email,
        max_records=args.max_records,
    )
    works = [normalize_work(work) for work in raw_works]
    G_full = build_graph(works)

    contrast_spec = FieldContrastSpec(
        filter_query=args.filter_query,
        event_year=args.event_year,
        event_label=args.event_label or f"Event {args.event_year}",
        before_years=args.before_years,
        after_years=args.after_years,
        max_plot_nodes=args.max_plot_nodes,
        min_community_size=args.min_community_size,
    )
    result = run_field_contrast(contrast_spec, G_full, output_dir=output_dir)

    for name, path in result.output_paths.items():
        log.info("%s: %s", name, path)


def run_paper_contrast_mode(args: argparse.Namespace) -> None:
    """
    运行“目标论文发表前后”的领域图谱对比模式。

    Args:
        args: 命令行参数。
    """
    from comparator import FieldContrastSpec, run_field_contrast
    from fetcher import fetch_works_cursor, normalize_work
    from graph_builder import build_graph, build_node_attributes

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    target_works = _load_target_works(args)

    valid_years = [work["year"] for work in target_works if work.get("year")]
    if not valid_years:
        raise RuntimeError("目标论文缺少 publication_year，无法做前后时间切片。")

    year_start = min(valid_years) - args.before_years
    year_end = max(valid_years) + args.after_years
    combined_filter = f"{args.filter_query},publication_year:{year_start}-{year_end}"

    log.info("拉取论文驱动对比所需领域图谱数据: %s", combined_filter)
    raw_works = fetch_works_cursor(
        combined_filter,
        email=args.email,
        max_records=args.max_records,
    )
    works = [normalize_work(work) for work in raw_works]
    G_full = build_graph(works)

    for work in target_works:
        node_id = work["id"]
        if node_id not in G_full:
            G_full.add_node(node_id, **build_node_attributes(work))
        else:
            G_full.nodes[node_id].update(build_node_attributes(work))
        for ref_id in work["referenced_works"]:
            G_full.add_edge(node_id, ref_id)

    for work in sorted(target_works, key=lambda item: (item["year"], item["id"])):
        paper_id = work["id"]
        paper_year = work["year"]
        paper_title = work.get("title") or paper_id

        if args.event_label:
            if len(target_works) == 1:
                event_label = args.event_label
            else:
                event_label = f"{args.event_label} · {paper_id}"
        else:
            event_label = paper_title

        contrast_spec = FieldContrastSpec(
            filter_query=args.filter_query,
            event_year=paper_year,
            event_label=event_label,
            target_paper_ids=[paper_id],
            target_paper_titles=[paper_title],
            before_years=args.before_years,
            after_years=args.after_years,
            max_plot_nodes=args.max_plot_nodes,
            min_community_size=args.min_community_size,
            slug=_slugify(f"paper_{paper_id}_{paper_year}"),
        )
        result = run_field_contrast(contrast_spec, G_full, output_dir=output_dir)
        log.info(
            "完成论文驱动对比: %s (%s) -> ΔQ=%+.4f",
            paper_title,
            paper_year,
            result.delta_q_result.get("delta_Q", 0.0),
        )


def run_paper_neighborhood_contrast_mode(args: argparse.Namespace) -> None:
    """
    运行纯论文邻域模式：不依赖领域过滤，只围绕目标论文构建邻域图谱。

    Args:
        args: 命令行参数。
    """
    from comparator import FieldContrastSpec, run_field_contrast

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    target_works = _load_target_works(args)
    G_full = build_paper_neighborhood_graph(
        target_works=target_works,
        email=args.email,
        before_years=args.before_years,
        after_years=args.after_years,
        max_refs=args.neighbor_max_refs,
        citers_per_target=args.neighbor_citers_per_target,
        citers_per_ref=args.neighbor_citers_per_ref,
    )

    for work in sorted(target_works, key=lambda item: (item["year"], item["id"])):
        paper_id = work["id"]
        paper_year = work["year"]
        paper_title = work.get("title") or paper_id

        if args.event_label:
            if len(target_works) == 1:
                event_label = args.event_label
            else:
                event_label = f"{args.event_label} · {paper_id}"
        else:
            event_label = paper_title

        contrast_spec = FieldContrastSpec(
            filter_query="paper_neighborhood",
            event_year=paper_year,
            event_label=event_label,
            target_paper_ids=[paper_id],
            target_paper_titles=[paper_title],
            before_years=args.before_years,
            after_years=args.after_years,
            max_plot_nodes=args.max_plot_nodes,
            min_community_size=args.min_community_size,
            slug=_slugify(f"neighborhood_{paper_id}_{paper_year}"),
        )
        result = run_field_contrast(contrast_spec, G_full, output_dir=output_dir)
        log.info(
            "完成纯论文邻域对比: %s (%s) -> ΔQ=%+.4f",
            paper_title,
            paper_year,
            result.delta_q_result.get("delta_Q", 0.0),
        )


# ──────────────────────────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    Returns:
        argparse.Namespace: 参数对象。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["full", "demo", "field_contrast", "paper_contrast", "paper_neighborhood_contrast"],
        default="full",
    )
    parser.add_argument("--email", default="jayeew@qq.com")
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", default="output")
    parser.add_argument("--max_records", "--max-records", dest="max_records", type=int, default=None)
    parser.add_argument("--filter", dest="filter_query", default="")
    parser.add_argument("--event-year", dest="event_year", type=int, default=None)
    parser.add_argument("--event-label", dest="event_label", default="")
    parser.add_argument("--paper-ids", dest="paper_ids", default="")
    parser.add_argument("--paper-dois", dest="paper_dois", default="")
    parser.add_argument("--before-years", dest="before_years", type=int, default=10)
    parser.add_argument("--after-years", dest="after_years", type=int, default=5)
    parser.add_argument("--max-plot-nodes", dest="max_plot_nodes", type=int, default=180)
    parser.add_argument("--min-community-size", dest="min_community_size", type=int, default=8)
    parser.add_argument("--neighbor-max-refs", dest="neighbor_max_refs", type=int, default=20)
    parser.add_argument("--neighbor-citers-per-target", dest="neighbor_citers_per_target", type=int, default=120)
    parser.add_argument("--neighbor-citers-per-ref", dest="neighbor_citers_per_ref", type=int, default=30)
    args = parser.parse_args()

    if args.mode == "field_contrast":
        if not args.filter_query:
            parser.error("--mode field_contrast 需要提供 --filter")
        if args.event_year is None:
            parser.error("--mode field_contrast 需要提供 --event-year")
    elif args.mode == "paper_contrast":
        if not args.filter_query:
            parser.error("--mode paper_contrast 需要提供 --filter")
        if not args.paper_ids and not args.paper_dois:
            parser.error("--mode paper_contrast 需要提供 --paper-ids 或 --paper-dois")
    elif args.mode == "paper_neighborhood_contrast":
        if not args.paper_ids and not args.paper_dois:
            parser.error("--mode paper_neighborhood_contrast 需要提供 --paper-ids 或 --paper-dois")

    return args


if __name__ == "__main__":
    cli_args = parse_args()
    output_dir = Path(cli_args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if cli_args.mode == "demo":
        run_demo_mode(output_dir)
    elif cli_args.mode == "field_contrast":
        run_field_contrast_mode(cli_args)
    elif cli_args.mode == "paper_contrast":
        run_paper_contrast_mode(cli_args)
    elif cli_args.mode == "paper_neighborhood_contrast":
        run_paper_neighborhood_contrast_mode(cli_args)
    else:
        run_full_mode(cli_args)

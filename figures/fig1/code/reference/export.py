"""SVG/PDF/PNG delivery, font outlines, inspection and scientific QA."""

from __future__ import annotations

import copy
import hashlib
import html
import importlib.metadata
import json
import os
import sys
import xml.etree.ElementTree as ET
import zipfile
from itertools import pairwise
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image

from .data import (
    ASSETS,
    ROOT,
    bare,
    community_ccdf,
    connect,
    local_metrics,
    read,
    source,
    write,
)
from .svg import NS, el, font_path, measure


def fonts(out: Path) -> None:
    config = out / "layouts/fonts.conf"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        '<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "fonts.dtd"><fontconfig><include>/etc/fonts/fonts.conf</include><dir>/mnt/c/Windows/Fonts</dir></fontconfig>'
    )
    os.environ["FONTCONFIG_FILE"] = str(config.resolve())


def convert(path: Path, scale: float = 3, pdf: bool = True) -> None:
    import cairosvg

    root = ET.parse(path).getroot()
    _, _, width, height = map(float, root.attrib["viewBox"].split())
    if pdf:
        cairosvg.svg2pdf(url=str(path), write_to=str(path.with_suffix(".pdf")))
    cairosvg.svg2png(
        url=str(path),
        write_to=str(path.with_suffix(".png")),
        output_width=round(width * scale),
        output_height=round(height * scale),
    )


def outline(path: Path, destination: Path) -> None:
    from matplotlib.font_manager import FontProperties
    from matplotlib.path import Path as MplPath
    from matplotlib.textpath import TextPath

    root = ET.parse(path).getroot()
    for parent in root.iter():
        for node in list(parent):
            if node.tag != f"{{{NS}}}text":
                continue
            a = node.attrib
            value = node.text or ""
            size = float(a.get("font-size", 12))
            bold = a.get("font-weight") == "bold"
            italic = a.get("font-style") == "italic"
            sans = a.get("font-family") == "Arial"
            x, y = float(a.get("x", 0)), float(a.get("y", 0))
            width = measure(value, size, bold, italic, sans)
            anchor = a.get("text-anchor", "start")
            x -= width if anchor == "end" else width / 2 if anchor == "middle" else 0
            textpath = TextPath(
                (0, 0),
                value,
                size=size,
                prop=FontProperties(fname=str(font_path(bold, italic, sans))),
            )
            parts = []
            for points, code in textpath.iter_segments(curves=True, simplify=False):
                cmd = {
                    MplPath.MOVETO: "M",
                    MplPath.LINETO: "L",
                    MplPath.CURVE3: "Q",
                    MplPath.CURVE4: "C",
                    MplPath.CLOSEPOLY: "Z",
                }[code]
                parts.append(
                    cmd + (" ".join(f"{v:.5f}" for v in points) if cmd != "Z" else "")
                )
            group = el(
                "g", transform=f"translate({x} {y}) scale(1 -1)", aria_label=value
            )
            group.append(el("path", d=" ".join(parts), fill=a.get("fill", "#000000")))
            at = list(parent).index(node)
            parent.remove(node)
            parent.insert(at, group)
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)


def reassemble_components(out: Path) -> None:
    """Compose from editable layer files, preserving template placement and scale."""
    for template in sorted((out / "layouts/templates").glob("*.svg")):
        root = ET.parse(template).getroot()
        for parent in root.iter():
            for child in list(parent):
                ident = child.attrib.get("data-component")
                if ident:
                    layer = ET.parse(
                        out / "components/layers" / f"{ident}.svg"
                    ).getroot()
                    group = next(
                        n for n in layer if n.attrib.get("data-component") == ident
                    )
                    index = list(parent).index(child)
                    parent.remove(child)
                    parent.insert(index, copy.deepcopy(group))
        destination = (
            out / "final/Fig1.svg"
            if template.stem == "Fig1"
            else out / "panels" / template.name
        )
        ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)


def assemble(out: Path, high_res: bool = True, only: str | None = None) -> None:
    reassemble_components(out)
    fonts(out)
    for folder in ["components", "panels"]:
        paths = sorted((out / folder).glob("*.svg"))
        for i, path in enumerate(paths):
            if only and not (
                path.stem == only
                or path.stem.startswith(only + "_")
                or path.stem == "case_" + only
            ):
                continue
            convert(
                path,
                max(
                    3,
                    min(
                        8,
                        1200
                        / float(ET.parse(path).getroot().attrib["viewBox"].split()[2]),
                    ),
                ),
            )
        print(f"Exported {folder}: {len(paths)} SVG sources.", flush=True)
    main = out / "final/Fig1.svg"
    convert(main, 2)
    outline(main, out / "final/Fig1_outlined.svg")
    if high_res:
        import cairosvg

        for factor in [6, 10]:
            cairosvg.svg2png(
                url=str(main),
                write_to=str(out / f"final/Fig1_{factor}x.png"),
                output_width=1055 * factor,
                output_height=1491 * factor,
            )
            print(f"Exported {1055 * factor} × {1491 * factor} PNG.", flush=True)
    # Compare the actual rendered SVG and PDF using a second rasterizer.
    import pymupdf

    doc = pymupdf.open(out / "final/Fig1.pdf")
    doc[0].get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False).save(
        out / "qa/pdf_preview.png"
    )
    doc.close()
    reference = Image.open(
        "/mnt/c/Users/jayee/Downloads/ChatGPT Image 2026年9月9日 20_25_04.png"
    ).convert("RGB")
    current = (
        Image.open(out / "final/Fig1.png")
        .convert("RGB")
        .resize(reference.size, Image.Resampling.LANCZOS)
    )
    comparison = Image.new("RGB", (2110, 1491), "white")
    comparison.paste(reference, (0, 0))
    comparison.paste(current, (1055, 0))
    comparison.save(out / "qa/reference_comparison.png")
    Image.blend(reference, current, 0.5).save(out / "qa/reference_overlay.png")


def validate(out: Path, check_exports: bool = True) -> dict[str, Any]:
    snapshot = read(out / "data/snapshot.json")
    checks: dict[str, bool] = {}
    cases = snapshot["cases"]
    checks["four_distinct_papers"] = len({c["input"]["paper_id"] for c in cases}) == 4
    for c in cases:
        label = c["label"]
        checks[label + "_same_historical_coordinates"] = (
            c["layout"]["before"] == c["layout"]["after_historical"]
        )
        checks[label + "_all_neighbors_displayed"] = set(c["layout"]["before"]) == {
            n["claim_id"] for n in c["neighbors"]
        }
        checks[label + "_eligibility"] = all(
            n["publication_date"] < c["input"]["cutoff_date"]
            and n["cosine_similarity"] > 0.5
            and n["parent_paper_id"] != c["input"]["paper_id"]
            for n in c["neighbors"]
        )
        checks[label + "_snapshot_precedes_cutoff"] = (
            snapshot["statistics"]["historical_max_date"] < c["input"]["cutoff_date"]
        )
        metrics = local_metrics(c["neighbors"], c["edges"], c["claim"]["claim_type"])
        checks[label + "_metric_recalculation"] = all(
            metrics[k] == c["metrics"][k]
            for k in [
                "components_before",
                "components_after",
                "component_merge_count",
                "newly_connected_neighbor_pair_count",
                "connected_pair_share",
                "role_counts",
            ]
        )
        checks[label + "_source_span_coverage"] = set(
            c["shared_claim"]["source_span_ids"]
        ) == {s["span_id"] for s in c["source_spans"]}
        checks[label + "_explicit_path_availability"] = all(
            n["path_recomputed"]["status"]
            in [
                "reference_ids_unavailable",
                "parent_mapping_unavailable",
                "observed_in_saved_references",
            ]
            for n in c["neighbors"]
        )
    stats = snapshot["statistics"]
    checks["role_counts_sum"] = sum(stats["roles"].values()) == stats["claims"]
    checks["role_matrix_sum"] = (
        int(np.array(stats["role_matrix_counts"]).sum()) == stats["semantic_edges"]
    )
    checks["column_normalization"] = bool(
        np.allclose(np.array(stats["role_matrix_column_normalized"]).sum(axis=0), 1)
    )
    checks["community_histogram_sum"] = (
        sum(stats["community_size_histogram"].values()) == stats["community_count"]
    )
    checks["community_ccdf_exact"] = stats["community_ccdf"] == community_ccdf(
        stats["community_size_histogram"]
    )
    shares = [row["share_at_least_size"] for row in stats["community_ccdf"]]
    checks["community_ccdf_monotone"] = shares[0] == 1 and all(
        a >= b > 0 for a, b in pairwise(shares)
    )
    checks["community_claims_sum"] = (
        sum(int(k) * v for k, v in stats["community_size_histogram"].items())
        + stats["unassigned"]
        == stats["claims"]
    )
    checks["paper_count_sum"] = (
        sum(stats["claims_per_paper"].values()) == stats["target_papers"]
    )
    checks["claims_per_paper_sum"] = (
        sum(int(k) * v for k, v in stats["claims_per_paper"].items()) == stats["claims"]
    )
    layer = snapshot["atlas"]["paper_layer"]
    checks["paper_no_selfloops"] = all(u != v for u, v in layer["edges"])
    checks["paper_no_duplicates"] = len({tuple(e) for e in layer["edges"]}) == len(
        layer["edges"]
    )
    checks["paper_endpoints"] = all(
        x in {p["work_id"] for p in layer["nodes"]} for e in layer["edges"] for x in e
    )
    paper_graph = nx.DiGraph()
    paper_graph.add_nodes_from(p["work_id"] for p in layer["nodes"])
    paper_graph.add_edges_from(layer["edges"])
    checks["paper_no_isolated_nodes"] = not list(nx.isolates(paper_graph))
    checks["paper_single_weak_component"] = nx.is_weakly_connected(paper_graph)
    checks["paper_all_targets_connected"] = all(
        paper_graph.degree(bare(c["input"]["openalex_work_id"])) > 0
        for c in snapshot["cases"]
    )
    checks.update(source_checks(snapshot))
    component_qa = out / "qa/component_bounds.json"
    if component_qa.exists():
        checks["component_text_bounds"] = not read(component_qa)
    if check_exports:
        for folder in ["components", "panels"]:
            files = list((out / folder).glob("*.svg"))
            checks[folder + "_all_formats"] = bool(files) and all(
                p.with_suffix(ext).exists() for p in files for ext in [".pdf", ".png"]
            )
        for factor in [6, 10]:
            path = out / f"final/Fig1_{factor}x.png"
            checks[f"png_{factor}x_dimensions"] = path.exists() and Image.open(
                path
            ).size == (1055 * factor, 1491 * factor)
        root = ET.parse(out / "final/Fig1.svg").getroot()
        ids = [n.attrib["id"] for n in root.iter() if "id" in n.attrib]
        checks["svg_unique_ids"] = len(ids) == len(set(ids))
        checks["svg_editable_text"] = bool(root.findall(f".//{{{NS}}}text"))
        checks["outlined_no_text"] = (
            not ET.parse(out / "final/Fig1_outlined.svg")
            .getroot()
            .findall(f".//{{{NS}}}text")
        )
    boxes = (
        read(out / "qa/text_bounds.json")
        if (out / "qa/text_bounds.json").exists()
        else []
    )
    overflow = [
        b
        for b in boxes
        if b["x"] < 0
        or b["y"] < 0
        or b["x"] + b["width"] > 1055.5
        or b["y"] + b["height"] > 1491.5
    ]
    checks["page_text_bounds"] = not overflow
    report = {
        "checks": checks,
        "all_passed": all(checks.values()),
        "text_overflow": overflow,
        "scope": "Figure geometry, provenance availability and numerical consistency, not scientific validity or current-system empirical success.",
        "limitations": [
            "Paper nodes are selected for connected citation context, not as an unbiased population sample. Saved reference lists may be incomplete.",
            "Saved cosine neighbors were retained, not re-retrieved. Path recomputation is a figure derivative.",
            "Historical path-annotation count is the saved snapshot statistic; no global path rebuild.",
            "Paper metadata date anomalies are not automatically treated as false citations.",
        ],
    }
    write(out / "qa/validation.json", report)
    print(
        json.dumps(
            {
                "all_passed": report["all_passed"],
                "failed": [k for k, v in checks.items() if not v],
            }
        ),
        flush=True,
    )
    return report


def source_checks(snapshot: dict[str, Any]) -> dict[str, bool]:
    """Validate the scientific display against read-only native records."""
    checks = {
        "historical_source_hashes_unchanged": all(
            source(Path(record["path"]))["sha256"] == record["sha256"]
            for record in snapshot["sources"]
        )
    }
    frame = pd.read_parquet(
        ASSETS / "claim_backbone_edges.parquet", columns=["claim_id_a", "claim_id_b"]
    )
    native = {tuple(sorted(edge)) for edge in frame.itertuples(index=False, name=None)}
    displayed = {tuple(sorted(edge)) for edge in snapshot["atlas"]["claim_edges"]}
    checks["atlas_edges_exist_in_native_backbone"] = displayed <= native
    for case in snapshot["cases"]:
        neighbors = {n["claim_id"] for n in case["neighbors"]}
        expected = {e for e in native if set(e) <= neighbors}
        checks[case["label"] + "_exact_native_induced_edges"] = {
            tuple(e) for e in case["edges"]
        } == expected
    target_refs = {
        bare(c["input"].get("openalex_work_id") or c["input"]["paper_id"]): {
            bare(r) for r in c["input"]["reference_work_ids"]
        }
        for c in snapshot["cases"]
    }
    pairs = {tuple(e) for e in snapshot["atlas"]["paper_layer"]["edges"]}
    for c in snapshot["cases"]:
        for n in c["neighbors"]:
            pairs.update(
                tuple(e) for w in n["path_recomputed"]["witnesses"] for e in w["edges"]
            )
    with connect("paper_graph_index.sqlite") as db:
        checks["all_citations_and_witnesses_have_sources"] = all(
            v in target_refs[u]
            if u in target_refs
            else bool(
                db.execute(
                    "SELECT 1 FROM paper_edges WHERE citing_work_id=? AND cited_work_id=?",
                    (u, v),
                ).fetchone()
            )
            for u, v in pairs
        )
    return checks


def handoff(out: Path) -> None:
    snapshot = read(out / "data/snapshot.json")
    rows = []
    for c in snapshot["cases"]:
        rows.extend(
            [
                f"## Case {c['label']} — {c['claim']['claim_id']}",
                c["claim"]["claim_text"],
                f"Source: {c['input']['title']}; DOI {c['input']['doi']}; cutoff {c['input']['cutoff_date']}.",
                "Selection: " + c["selection_reason"],
                "",
                "### Manuscript source spans",
            ]
        )
        rows.extend(
            f"**{s['span_id']}** (characters {s['char_start']}–{s['char_end']})\n\n{s['text']}\n"
            for s in c["source_spans"]
        )
    (out / "data/case_source_companion.md").write_text(
        "\n\n".join(rows), encoding="utf-8"
    )
    case_metrics = {c["label"]: c["metrics"] for c in snapshot["cases"]}
    b_communities = len(case_metrics["B"]["community_probabilities"])
    c_cross = round(
        case_metrics["C"]["cross_type_neighbor_share"]
        * case_metrics["C"]["neighbor_count"]
    )
    caption = f"""Fig. 1 | Two-layer knowledge graph and local claim-insertion profiles.

a, A deterministic display sample of the historical claim backbone and actual paper citations. The paper layer uses connected citation context around the four target papers; it is selected for readability, not as an unbiased population sample. Connector paths are traversed without direction solely for display selection, while arrows retain actual citation direction. Claim color denotes selected historical community; shape denotes contribution role. Grey combines other communities and unassigned claims in this display only. K labels are display aliases, not disciplines. Four full-text target claims and their parent papers are independent overlays, not a joint insertion. Layout coordinates have no metric interpretation.

b, Four claims from different papers, each with all ten retained historical neighbors. Before and after share historical coordinates and backbone edges; after adds only the target and its retained semantic links. Claim text is a grounded normalized-claim excerpt, not a verbatim manuscript quotation. A: high similarity without component merging. B: VSIG10L and esophageal epithelial integrity, with {b_communities} historical communities. C: Sensight imaging-probe design, with {c_cross} of ten neighbors having a different role from the target. D: three components of sizes 4, 3 and 3 merge through the target, making 33 of 45 historical pairs newly reachable. These are local structural descriptions, not firstness or correctness judgments. Paper-path coverage is computed against saved references and the historical citation index; saved reference lists may be incomplete.

c, Distinct historical target and citation-extended paper populations. Historical claims are abstract-derived; target examples are full-text-derived. Counts include the five targets with zero saved claims. Citation-edge counts exclude duplicate and same-work records using the supplied aggregate audit, without asserting scientific validity of each citation.

d, Exact empirical complementary cumulative distribution of community size: the fraction of all 2,422 assigned communities with size at least the horizontal-axis threshold. The step curve uses observed counts without smoothing or fitting. Full-population role composition is shown for six selected communities, alongside backbone edge counts between those communities. Off-diagonal matrix entries are mirrored for display and counted once in the data; color uses log(1 + count). Unassigned claims are not communities. No power-law fit is claimed.

e, Historical semantic-edge role pairs (earlier rows, later columns), normalized within later-role columns. Color uses a square-root transform of share. Paper-path diagrams show relation syntax; actual witnesses are separately exported. Shared reference means bibliographic coupling. The 297,063 paper-path annotations are a saved historical snapshot statistic. Semantic, path-annotation and backbone counts overlap and must not be summed.

In panel a, outgoing one-hop citations from PA–PD match their parent-paper node colors. Subsequent edges of outward two-hop paths retain the original grey; remaining context edges use a slightly lighter grey.

Data preparation reuses saved thresholded cosine neighbors, checks eligibility, recomputes induced topology and deduplicates actual paper-path witnesses in a separate figure derivative. It is not a new end-to-end study run. Full identities, source spans, candidate list, community mapping, layout coordinates and metric definitions accompany the figure.
"""
    (out / "caption_en.md").write_text(caption, encoding="utf-8")
    paper_display = snapshot["atlas"]["paper_layer"]
    note = f"""# Fig.1 使用与复现

最终图：final/Fig1.svg（可编辑文字）、final/Fig1_outlined.svg（文字转路径）、final/Fig1.pdf（297 mm 宽矢量）、final/Fig1_6x.png（6330 × 8946）、final/Fig1_10x.png（10550 × 14910）。final/Fig1.png 是快速预览。

components/ 下每个小图、图例和指标卡独立保存 SVG/PDF/PNG；components/layers/ 是保持原始定位比例的无包装组件。panels/ 保存 a–e 和四张完整案例卡。layouts/components.json 给出组件输入、尺寸及组装位置。data/snapshot.json 是绘图输入快照，保留完整记录和稳定身份。

数据口径：局部图只计算保留邻居的历史骨架诱导图；前后历史节点坐标不变；社区是固定聚类而非学科；图结构不证明科学新颖性。四例路径均基于已有参考 ID 与历史引用索引计算，已保存参考列表不保证完整。297,063 是历史已保存路径注释数量，不是本次重新生成。Paper 原始边 63,984,971 与清洁边 63,966,133 分开。

当前图：B 为 VSIG10L 食管上皮研究，C 为 Sensight 成像探针设计。Paper Graph 显示 {len(paper_display["nodes"])} 个节点、{len(paper_display["edges"])} 条有向引用边，无孤立节点，PA–PD 通过真实引用记录连接。选择连通背景属于展示筛选，不是无偏抽样，也不意味着因果或科学承接关系。连接路径仅用于筛选，箭头保留真实引用方向。Claim Graph 标签为 Semantic relations and citation context；引用背景来自父论文路径，实际显示的是历史骨架。四个 case 的日期、邻居数与来源保存在源记录和图注中。Panel d 使用经验互补累积分布，分母是全部 2,422 个已分配社区；未做拟合或平滑。逐阈值数据见 data/community_ccdf.csv。

输出策略：默认直接覆盖 outputs/fig1_reference 中的同名结果，不创建版本目录或历史备份。assemble、validate 和 all 完成时同步覆盖 outputs/Fig1_reference_delivery.zip。全图核对仅依赖当前快照和科学源数据，不依赖旧结果目录。

Paper Graph 边配色：PA–PD 的一跳出边采用对应节点颜色；向外二跳路径的后续边保留原灰色；其余背景引用边向白色混合 18%，略微调淡。数据、坐标和边宽不变，逐边记录见 qa/paper_edge_colors.json。

真实案例替换了参考图中的占位文字和数字。A 的模糊连接变化卡改为分量合并数；D 的平均最短路径卡改为新可达历史节点对占比。原图布局与配色保留，真实拓扑与角色形状不能逐像素复制。

图的默认物理宽度为 297 mm；压缩为 183 mm 时需重新审查小字，不能把高清像素等同于小尺寸印刷可读性。

复现（在项目根目录；Python 3.12）：

```bash
python3 -m venv --system-site-packages /tmp/aspr-fig1-venv
/tmp/aspr-fig1-venv/bin/pip install -r figure_pipeline/fig1_reference/requirements.txt
/tmp/aspr-fig1-venv/bin/python -m figure_pipeline.fig1_reference all
/tmp/aspr-fig1-venv/bin/python -m figure_pipeline.fig1_reference render --only b_A_before
/tmp/aspr-fig1-venv/bin/python -m figure_pipeline.fig1_reference assemble --only b_A_before --preview-only
/tmp/aspr-fig1-venv/bin/python -m figure_pipeline.fig1_reference validate
/tmp/aspr-fig1-venv/bin/python -m figure_pipeline.fig1_reference.audit
```

prepare 只读本地科学资产，另写绘图快照；render 只读取快照生成 SVG；assemble 导出其他格式；validate 检查图形与数值一致性。audit 从源表独立复算并全量扫描 Paper Graph，详细结果见 qa/data_audit.json 和 qa/data_audit_zh.md。不调用模型、不修改历史图数据库或旧实验产物。字体取自 /mnt/c/Windows/Fonts；在其他机器需配置同名字体或修改 svg.font_path。
"""
    (out / "README_zh.md").write_text(note, encoding="utf-8")
    gallery(out)
    write(
        out / "layouts/runtime.json",
        {
            "python": sys.version,
            "libraries": {
                name: importlib.metadata.version(name)
                for name in [
                    "numpy",
                    "pandas",
                    "pyarrow",
                    "networkx",
                    "scipy",
                    "matplotlib",
                    "pillow",
                    "cairosvg",
                    "pymupdf",
                    "duckdb",
                ]
            },
        },
    )
    records = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "file_manifest.json":
            records.append(
                {
                    "path": str(path.relative_to(out)),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    write(out / "file_manifest.json", records)
    package_delivery(out)


def package_delivery(out: Path) -> None:
    """Replace the current delivery archive without retaining previous archives."""
    name = "Fig1_reference" if out.name == "fig1_reference" else out.name
    destination = out.parent / f"{name}_delivery.zip"
    temporary = destination.with_suffix(".tmp.zip")
    with zipfile.ZipFile(
        temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                archive.write(path, Path("Fig1") / path.relative_to(out))
        for path in sorted(Path(__file__).parent.iterdir()):
            if path.is_file():
                archive.write(path, Path("source") / path.relative_to(ROOT))
        test = ROOT / "tests/figure_pipeline/test_fig1_reference.py"
        if test.exists():
            archive.write(test, Path("source") / test.relative_to(ROOT))
    temporary.replace(destination)


def gallery(out: Path) -> None:
    """A local file index, not a separately authored analytical report."""
    cards = []
    for entry in read(out / "layouts/components.json"):
        ident = html.escape(entry["id"])
        title = html.escape(entry["title"])
        cards.append(
            f'<article><a href="components/{ident}.svg"><img loading="lazy" src="components/{ident}.png" alt="{title}"></a><b>{title}</b><small>{ident}</small><p><a href="components/{ident}.svg">SVG</a> · <a href="components/{ident}.pdf">PDF</a> · <a href="components/{ident}.png">PNG</a> · <a href="components/layers/{ident}.svg">组装图层</a></p></article>'
        )
    panels = []
    for key in ["a", "b", "c", "d", "e", "case_A", "case_B", "case_C", "case_D"]:
        panels.append(
            f'<article><a href="panels/{key}.svg"><img loading="lazy" src="panels/{key}.png" alt="{key}"></a><b>{key}</b><p><a href="panels/{key}.svg">SVG</a> · <a href="panels/{key}.pdf">PDF</a> · <a href="panels/{key}.png">PNG</a></p></article>'
        )
    page = (
        """<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Fig.1 · 图稿与独立组件</title><style>body{font-family:Arial,sans-serif;background:#f5f8fb;color:#172b45;max-width:1280px;margin:36px auto;padding:0 24px}a{color:#176da9}h1{font-size:28px}p{line-height:1.7}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:18px}article{background:white;border:1px solid #dce6ee;border-radius:8px;padding:14px;min-width:0}article img{width:100%;height:210px;object-fit:contain}b,small{display:block;margin-top:10px}small{color:#64758b;overflow-wrap:anywhere}.full{display:block;max-width:850px;width:100%;margin:24px auto}h2{margin-top:40px}</style><h1>Fig.1 · 图稿与独立组件</h1><p>5 个主 panel，4 张案例卡，84 个独立组件。SVG 保留可编辑对象；数据与图注随图提供。</p><p><a href="final/Fig1_6x.png">6330 × 8946 PNG</a> · <a href="final/Fig1_10x.png">10550 × 14910 PNG</a> · <a href="final/Fig1.pdf">矢量 PDF</a> · <a href="final/Fig1.svg">SVG 母版</a> · <a href="final/Fig1_outlined.svg">文字转路径 SVG</a></p><p><a href="README_zh.md">使用说明</a> · <a href="caption_en.md">英文图注</a> · <a href="data/case_source_companion.md">案例来源</a> · <a href="qa/validation.json">核验结果</a> · <a href="qa/reference_comparison.png">参考图对照</a></p><img class="full" src="final/Fig1.png" alt="Fig.1"><h2>完整 panel 与案例卡</h2><div class="grid">"""
        + "".join(panels)
        + '</div><h2>独立小图与指标组件</h2><div class="grid">'
        + "".join(cards)
        + "</div></html>"
    )
    (out / "index.html").write_text(page, encoding="utf-8")

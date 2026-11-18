"""Latest-only delivery, vector reassembly, raster exports and scientific checks."""

from __future__ import annotations

import copy
import html
import math
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
from PIL import Image

from figure_pipeline.fig1_reference.export import convert, fonts, outline
from figure_pipeline.fig1_reference.svg import NS

from .data import METRICS, REFERENCE, ROOT, read, topology, write


def reassemble(out: Path) -> None:
    """Reinsert edited component layers into panel and final templates."""
    for path in sorted((out / "layouts/templates").glob("*.svg")):
        root = ET.parse(path).getroot()
        for parent in root.iter():
            for child in list(parent):
                ident = child.get("data-component")
                if ident:
                    layer = ET.parse(
                        out / "components/layers" / f"{ident}.svg"
                    ).getroot()
                    replacement = next(
                        n for n in layer if n.get("data-component") == ident
                    )
                    at = list(parent).index(child)
                    parent.remove(child)
                    parent.insert(at, copy.deepcopy(replacement))
        folder = "final" if path.stem == "Fig2" else "panels"
        ET.ElementTree(root).write(
            out / folder / path.name, encoding="utf-8", xml_declaration=True
        )
    # Standalone SVG is the same component source; it is never a crop of a raster.
    for path in (out / "components/layers").glob("*.svg"):
        (out / "components" / path.name).write_bytes(path.read_bytes())


def assemble(out: Path, high_res: bool = True, only: str | None = None) -> None:
    import cairosvg
    import pymupdf

    reassemble(out)
    fonts(out)
    manifest = read(out / "layouts/components.json")
    selected = [
        r for r in manifest if not only or r["id"] == only or r["panel"] == only
    ]
    if only and not selected:
        raise ValueError(f"Unknown panel/component: {only}")
    for row in selected:
        convert(out / row["svg"], max(3, min(6, 800 / row["width"])))
    panels = {r["panel"] for r in selected}
    for ident in sorted(panels):
        convert(out / "panels" / f"{ident}.svg", 3)
    main = out / "final/Fig2.svg"
    convert(main, 2)
    outline(main, out / "final/Fig2_outlined.svg")
    if high_res:
        for factor in [6, 10]:
            cairosvg.svg2png(
                url=str(main),
                write_to=str(out / f"final/Fig2_{factor}x.png"),
                output_width=1055 * factor,
                output_height=1491 * factor,
            )
            print(f"Exported {1055 * factor} × {1491 * factor} PNG.", flush=True)
    else:
        # Never leave stale high-resolution images alongside a newly assembled master.
        for factor in [6, 10]:
            (out / f"final/Fig2_{factor}x.png").unlink(missing_ok=True)
    with pymupdf.open(out / "final/Fig2.pdf") as doc:
        doc[0].get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False).save(
            str(out / "qa/pdf_preview.png")
        )
        write(
            out / "qa/pdf_fonts.json",
            {"fonts": doc[0].get_fonts(), "page_points": list(doc[0].rect)},
        )
    with Image.open(REFERENCE) as ref, Image.open(out / "final/Fig2.png") as current:
        reference = ref.convert("RGB")
        rendered = current.convert("RGB").resize(
            reference.size, Image.Resampling.LANCZOS
        )
        comparison = Image.new("RGB", (2110, 1491), "white")
        comparison.paste(reference, (0, 0))
        comparison.paste(rendered, (1055, 0))
        comparison.save(out / "qa/reference_comparison.png")
        Image.blend(reference, rendered, 0.5).save(out / "qa/reference_overlay.png")
    print(f"Exported {len(selected)} components and {len(panels)} panels.", flush=True)


def scientific_checks(data: dict[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    claims, facts, j = data["claims"], data["facts"], data["joint"]
    targets = {c["claim_id"] for c in claims}
    checks["four_grounded_claims_one_paper"] = len(targets) == 4 and all(
        c.startswith(data["paper"]["paper_id"] + "::") for c in targets
    )
    union = {n["claim_id"] for f in facts for n in f["neighbors"]}
    checks["joint_uses_all_claims"] = (
        j["target_claim_count"] == len(claims) and not j["missing_fact_claim_ids"]
    )
    checks["union_node_count"] = len(union) == j["historical_neighbor_count"]
    native_nodes = pd.read_parquet(
        ROOT / "data/claim_graph/claim_nodes.parquet",
        columns=[
            "claim_id",
            "parent_paper_id",
            "claim_type",
            "claim_text",
            "publication_date",
        ],
    ).set_index("claim_id")
    checks["historical_nodes_match_native_fields"] = all(
        str(native_nodes.loc[cid, field]) == str(neighbor[field])
        for cid, neighbor in data["historical"].items()
        for field in ["parent_paper_id", "claim_type", "claim_text", "publication_date"]
    )
    checks["historical_snapshot_precedes_cutoff"] = (
        str(native_nodes["publication_date"].max()) < data["paper"]["cutoff_date"]
    )
    checks["unique_insertion_edges"] = len(
        {tuple(e) for e in j["insertion_edges"]}
    ) == len(j["insertion_edges"])
    checks["no_target_to_target_edges"] = all(
        u in targets and v in union for u, v in j["insertion_edges"]
    )
    checks["joint_insertion_matches_single_claims"] = {
        tuple(e) for e in j["insertion_edges"]
    } == {
        (f["claim"]["claim_id"], n["claim_id"]) for f in facts for n in f["neighbors"]
    }
    counts = {
        n: sum(any(v["claim_id"] == n for v in f["neighbors"]) for f in facts)
        for n in union
    }
    checks["shared_neighbors_counted_once"] = {
        n: v for n, v in counts.items() if v > 1
    } == j["shared_historical_neighbors"]
    before = topology(sorted(union), j["historical_edges"])
    graph = nx.Graph()
    graph.add_nodes_from(union | targets)
    graph.add_edges_from(j["historical_edges"] + j["insertion_edges"])
    after = nx.number_connected_components(graph)
    checks["joint_components_recomputed"] = (before["components"], after) == (
        j["historical_components_before"],
        j["historical_components_after"],
    )
    checks["joint_new_pairs_recomputed"] = (
        before["new_pairs"] == j["joint_newly_connected_historical_pairs"]
        and after == 1
    )
    checks["four_independent_branch_identities"] = all(
        [a["claim_id"] for a in data["assessments"][branch]]
        == [c["claim_id"] for c in claims]
        for branch in ["gear", "graph"]
    )
    historical_edges = pd.read_parquet(
        ROOT / "data/claim_graph/claim_backbone_edges.parquet",
        columns=["claim_id_a", "claim_id_b"],
    )
    edge_set = {
        tuple(sorted(e)) for e in historical_edges.itertuples(index=False, name=None)
    }
    checks["joint_edges_exist_in_native_backbone"] = all(
        tuple(sorted(e)) in edge_set for e in j["historical_edges"]
    )
    checks["joint_contains_all_native_union_edges"] = {
        tuple(sorted(e)) for e in j["historical_edges"]
    } == {e for e in edge_set if e[0] in union and e[1] in union}
    for i, f in enumerate(facts):
        nodes = [n["claim_id"] for n in f["neighbors"]]
        calc, metrics = (
            topology(nodes, f["neighbor_edges"]),
            {m["name"]: m["value"] for m in f["metrics"]},
        )
        checks[f"C{i + 1}_policy"] = (
            len(nodes) <= 10
            and len(nodes) == len(set(nodes))
            and all(
                n["cosine_similarity"] > 0.5
                and n["publication_date"] < data["paper"]["cutoff_date"]
                and n["parent_paper_id"] != data["paper"]["paper_id"]
                for n in f["neighbors"]
            )
        )
        checks[f"C{i + 1}_historical_topology"] = (
            abs(calc["density"] - metrics["neighbor_induced_density"]) < 1e-10
            and calc["merges"] == metrics["component_merge_count"]
            and calc["new_pairs"] == metrics["newly_connected_neighbor_pair_count"]
        )
        checks[f"C{i + 1}_native_edges"] = {
            tuple(sorted(e)) for e in f["neighbor_edges"]
        } == {e for e in edge_set if e[0] in nodes and e[1] in nodes}
    metrics = data["metrics"]
    checks["fourteen_metrics_by_name"] = set(metrics) == {m[0] for m in METRICS}
    checks["C1_similarity_recomputed"] = (
        abs(
            metrics["nearest_prior_similarity"]
            - max(n["cosine_similarity"] for n in facts[0]["neighbors"])
        )
        < 1e-10
    )
    checks["C1_top5_recomputed"] = (
        abs(
            metrics["mean_top5_similarity"]
            - sum(n["cosine_similarity"] for n in facts[0]["neighbors"][:5]) / 5
        )
        < 1e-10
    )
    checks["C1_cross_role_recomputed"] = metrics["cross_type_neighbor_count"] == sum(
        n["claim_type"] != claims[0]["claim_type"] for n in facts[0]["neighbors"]
    )
    checks["C1_single_community_metrics"] = (
        {n["community_id"] for n in facts[0]["neighbors"]} == {7}
        and metrics["effective_community_count"] == 1
        and metrics["community_rao_stirling"] == 0
        and metrics["cross_boundary_weight_share"] == 0
    )
    checks["unavailable_pair_metrics_stay_null"] = (
        metrics["first_observed_recent_nature_pair_share"] is None
        and metrics["community_pair_mean_surprisal"] is None
    )
    for name, kind in [
        ("direct_citation_neighbor_count", "direct"),
        ("two_hop_neighbor_count", "two_hop"),
        ("co_citation_neighbor_count", "shared_reference"),
    ]:
        checks[name + "_witnesses"] = metrics[name] == sum(
            any(w["kind"] == kind for w in r["witnesses"])
            for r in data["path_witnesses"].values()
        )
    checks["path_arrows_no_loops"] = all(
        u != v
        for r in data["path_witnesses"].values()
        for w in r["witnesses"]
        for u, v in w["edges"]
    )
    checks["retrieved_works_before_cutoff"] = all(
        w["record"]["payload"]["publication_date"] < data["paper"]["cutoff_date"]
        for w in data["works"]
    )
    checks["manuscript_aliases_exist"] = all(
        s["span_id"] in claims[0]["source_span_ids"] for s in data["spans"]
    )
    checks["no_fabricated_fusion_output"] = data["fusion_available"] is False
    checks["null_residual_not_imputed"] = (
        data["gear_card"]["residual_contribution"] is None
    )
    checks["report_is_actual_whole_paper_fusion"] = (
        data["report"]["system"] == "fusion"
        and data["report"]["paper_id"] == data["paper"]["paper_id"]
    )
    checks.update(contrast_checks(data, native_nodes, edge_set))
    return checks


def contrast_checks(
    data: dict[str, Any], native: pd.DataFrame, edge_set: set[tuple[str, str]]
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    cases = data["contrasts"]
    checks["two_distinct_example_papers"] = (
        len({data["paper"]["paper_id"], *(c["paper"]["paper_id"] for c in cases)}) == 2
    )
    for case in cases:
        alias, fact = case["alias"], case["fact"]
        neighbors, metrics = fact["neighbors"], case["metrics"]
        ids = {n["claim_id"] for n in neighbors}
        checks[f"{alias}_claim_identity"] = (
            fact["claim"]["claim_id"] == case["claim"]["claim_id"]
        )
        checks[f"{alias}_insertion_policy"] = (
            fact["insertion_policy"] == data["facts"][0]["insertion_policy"]
        )
        checks[f"{alias}_eligible_unique_neighbors"] = len(ids) == len(
            neighbors
        ) <= 10 and all(
            n["cosine_similarity"] > 0.5
            and n["publication_date"] < case["paper"]["cutoff_date"]
            and n["parent_paper_id"] != case["paper"]["paper_id"]
            for n in neighbors
        )
        checks[f"{alias}_native_nodes"] = all(
            str(native.loc[n["claim_id"], f]) == str(n[f])
            for n in neighbors
            for f in ["claim_text", "parent_paper_id", "publication_date", "claim_type"]
        )
        checks[f"{alias}_complete_native_edges"] = {
            tuple(sorted(e)) for e in fact["neighbor_edges"]
        } == {e for e in edge_set if set(e) <= ids}
        checks[f"{alias}_fourteen_metrics"] = set(metrics) == {m[0] for m in METRICS}
        if neighbors:
            weights: dict[int, float] = {}
            for n in neighbors:
                if n["community_id"] is not None:
                    weights[n["community_id"]] = (
                        weights.get(n["community_id"], 0) + n["cosine_similarity"]
                    )
            total = sum(weights.values())
            effective = total**2 / sum(w**2 for w in weights.values())
            checks[f"{alias}_displayed_communities"] = (
                len(weights) == 5
                and abs(effective - metrics["effective_community_count"]) < 1e-10
            )
            calc = topology(sorted(ids), fact["neighbor_edges"])
            checks[f"{alias}_topology_recomputed"] = (
                abs(calc["density"] - metrics["neighbor_induced_density"]) < 1e-10
                and calc["merges"] == metrics["component_merge_count"]
                and calc["new_pairs"] == metrics["newly_connected_neighbor_pair_count"]
            )
        else:
            checks[f"{alias}_empty_not_imputed"] = (
                all(v is None for v in metrics.values()) and not fact["neighbor_edges"]
            )
            checks[f"{alias}_missing_assessment_explicit"] = (
                case["assessments"]["graph"] is None
                and "graph" in case["missing_assessments"]
            )
        if case["prior"]:
            relation, work = (
                case["prior"]["relation"]["payload"],
                case["prior"]["work"]["payload"],
            )
            checks[f"{alias}_partial_antecedent_source"] = (
                relation["target_claim_id"] == case["claim"]["claim_id"]
                and relation["prior_work_id"] == work["work_id"]
                and relation["relation_label"] == "PARTIAL_ANTECEDENT"
            )
            checks[f"{alias}_prior_date_and_level"] = (
                relation["temporal_valid"]
                and work["publication_date"] < case["paper"]["cutoff_date"]
                and relation["evidence_level"] == "abstract_evidence"
            )
            checks[f"{alias}_saved_card_status"] = (
                case["gear_card"]["status"] == "residual_extension"
            )
    audit = data["status_audit"]
    checks["main_status_reconciled_not_coverage_failure"] = (
        audit["coverage_sufficient"]
        and len(audit["unresolved_relations"]) == 1
        and audit["card_status"] == "inconclusive"
    )
    checks["shown_branch_stances_are_saved"] = (
        data["assessments"]["gear"][0]["overall_stance"] == "unresolved"
        and data["assessments"]["graph"][0]["overall_stance"] == "recognized"
    )
    profile = data["detailed_profile"]
    x_case = next(c for c in cases if c["alias"] == "X")
    checks["detailed_profile_is_X_not_C1"] = (
        profile["claim_id"] == x_case["claim"]["claim_id"]
        and profile["paper_id"] != data["paper"]["paper_id"]
    )
    checks["X_all_14_recomputed_match_saved"] = set(profile["metrics"]) == set(
        x_case["metrics"]
    ) and all(
        (value is None and x_case["metrics"][key] is None)
        or (
            value is not None
            and x_case["metrics"][key] is not None
            and abs(value - x_case["metrics"][key]) < 1e-9
        )
        for key, value in profile["metrics"].items()
    )
    assignments = pd.read_parquet(
        ROOT / "data/claim_graph/claim_communities.parquet"
    ).set_index("claim_id")["community_id"]
    checks["X_complete_native_community_assignments"] = all(
        n["community_id"] is not None
        and assignments.loc[n["claim_id"]] == n["community_id"]
        for n in x_case["fact"]["neighbors"]
    )
    checks["all_displayed_metrics_finite"] = len(profile["metrics"]) == 14 and all(
        v is not None and math.isfinite(v) for v in profile["metrics"].values()
    )
    pairs = profile["community_pairs"]
    checks["X_complete_pair_history"] = (
        len(pairs) == 10
        and all(r["observed"] for r in pairs)
        and profile["metrics"]["community_pair_mean_surprisal"] is not None
    )
    for name, kind in [
        ("direct_citation_neighbor_count", "direct"),
        ("two_hop_neighbor_count", "two_hop"),
        ("co_citation_neighbor_count", "shared_reference"),
    ]:
        checks["X_" + name + "_witnesses"] = profile["metrics"][name] == sum(
            any(w["kind"] == kind for w in r["witnesses"])
            for r in profile["path_witnesses"].values()
        )
    checks["X_path_coverage_and_no_loops"] = all(
        r["counts"] is not None
        and all(u != v for w in r["witnesses"] for u, v in w["edges"])
        for r in profile["path_witnesses"].values()
    )
    return checks


def validate(out: Path, exports: bool = True) -> dict[str, Any]:
    data, manifest = (
        read(out / "data/snapshot.json"),
        read(out / "layouts/components.json"),
    )
    checks = scientific_checks(data)
    style = read(out / "layouts/style.json")
    checks["reference_panel_anchors"] = style["panel_boxes"] == {
        "a": [10, 49, 1035, 584],
        "b": [12, 644, 500, 442],
        "c": [524, 644, 520, 442],
        "d": [12, 1098, 500, 344],
        "e": [524, 1098, 520, 344],
    }
    root = ET.parse(out / "final/Fig2.svg").getroot()
    ids = [n.get("id") for n in root.iter() if n.get("id")]
    checks["unique_svg_ids"] = len(ids) == len(set(ids))
    checks["all_components_assembled"] = {
        n.get("data-component") for n in root.iter() if n.get("data-component")
    } == {r["id"] for r in manifest}
    metric_groups = {
        "c_semantic": [1, 2],
        "c_roles": [8],
        "c_connectivity": [9, 10, 11],
        "c_communities": [3, 4, 5, 6, 7],
        "c_paths": [12, 13, 14],
    }
    for ident, metric_ids in metric_groups.items():
        group = next(n for n in root.iter() if n.get("data-component") == ident)
        checks[ident + "_bound_to_X"] = (
            group.get("data-claim-id") == data["detailed_profile"]["claim_id"]
        )
        rendered = [
            t.text for t in group.iter(f"{{{NS}}}text") if t.get("text-anchor") == "end"
        ]
        expected = []
        for index in metric_ids:
            v = data["detailed_profile"]["metrics"][METRICS[index - 1][0]]
            expected.append(
                "N/A"
                if v is None
                else str(v)
                if isinstance(v, int)
                else f"{v:.3f}".rstrip("0").rstrip(".")
            )
        checks[ident + "_displayed_values_match_X"] = rendered == expected
    checks["no_NA_in_figure_text"] = all(
        "N/A" not in (t.text or "") for t in root.iter(f"{{{NS}}}text")
    )
    checks["retired_case_absent"] = all(
        "AGN" not in (t.text or "") and "micromachines" not in (t.text or "")
        for t in root.iter(f"{{{NS}}}text")
    )
    checks["editable_text"] = bool(list(root.iter(f"{{{NS}}}text")))
    checks["no_raster_images_in_master"] = not list(root.iter(f"{{{NS}}}image"))
    boxes = read(out / "qa/component_text_boxes.json")
    for ident, item in boxes.items():
        checks[ident + "_text_inside_component"] = all(
            b["x"] >= 0
            and b["y"] >= 0
            and b["x"] + b["width"] <= item["width"] + 0.5
            and b["y"] + b["height"] <= item["height"] + 0.5
            for b in item["boxes"]
        )
    for row in manifest:
        _, _, width, height = style["panel_boxes"][row["panel"]]
        checks[row["id"] + "_inside_panel"] = (
            row["x"] >= 0
            and row["y"] >= 0
            and row["x"] + row["width"] <= width
            and row["y"] + row["height"] <= height
        )
    if exports:
        checks["all_component_exports"] = all(
            (out / "components" / f"{r['id']}.{ext}").exists()
            for r in manifest
            for ext in ["svg", "pdf", "png"]
        )
        checks["all_panel_exports"] = all(
            (out / "panels" / f"{p}.{ext}").exists()
            for p in "abcde"
            for ext in ["svg", "pdf", "png"]
        )
        checks["outlined_copy_has_no_text"] = not list(
            ET.parse(out / "final/Fig2_outlined.svg").getroot().iter(f"{{{NS}}}text")
        )
        for factor in [6, 10]:
            path = out / f"final/Fig2_{factor}x.png"
            if path.exists():
                # Reading the PNG header does not allocate a full-resolution raster.
                raw = path.read_bytes()[:24]
                size = (
                    int.from_bytes(raw[16:20], "big"),
                    int.from_bytes(raw[20:24], "big"),
                )
                checks[f"PNG_{factor}x_dimensions"] = size == (
                    1055 * factor,
                    1491 * factor,
                )
    result = {
        "passed": all(checks.values()),
        "check_count": len(checks),
        "checks": checks,
        "manual_review": "Reference comparison, PNG and PDF preview reviewed at full figure and panel scale; this is not a claim of pixel-identical scientific content.",
    }
    write(out / "qa/validation.json", result)
    if not result["passed"]:
        raise ValueError(
            "Validation failed: " + ", ".join(k for k, v in checks.items() if not v)
        )
    print(f"Passed {len(checks)} data and artifact checks.", flush=True)
    return result


def handoff(out: Path) -> None:
    manifest, snapshot = (
        read(out / "layouts/components.json"),
        read(out / "data/snapshot.json"),
    )
    caption = """Fig. 2 | GEAR–Graph framework for evidence-based innovation analysis.

a, Full-text claims are mined, consolidated and checked against manuscript spans. Shared grounded claims enter independent historical-evidence and Graph reasoning branches. The latter uses prebuilt, cutoff-compatible historical assets and temporary insertion, with separate per-claim and union-neighborhood interpretations. Standard claim synthesis and study whole-paper reporting are distinct output paths.
b, Main C1 illustrates selected historical comparisons. W1–W3 are retrieved sources, not automatically direct antecedents. Sufficient retrieval coverage coexists with an inconclusive card because one relation remains unresolved. Contrast X illustrates a saved partial-antecedent judgment: an earlier study describes polymer-confined gold nanoparticles, whereas the target reports gold–PLGA nanoparticles with a distinct outer ionic-liquid coating. This comparison uses the prior abstract and is a system judgment, not an independently confirmed direct antecedent.
c, Complete retained neighborhoods for C1 (10 neighbors, one community) and X (10 neighbors, five communities). All 14 displayed descriptors belong to X, highlighted in blue. Effective community count is the inverse sum of squared cosine-weighted community proportions. All ten X neighbors have native community assignments. The five communities form ten unordered pairs, all present in the historical pair table. Consequently, unseen-pair share is an observed zero, and mean pair surprisal is calculable (0.301). Surprisal follows the saved log commonness definition, not a calibrated novelty score. All displayed X metrics are independently recomputed from retained neighbors, native community data and cleaned citation witnesses, and agree with saved values. Grey lines denote historical backbone edges; dashed lines denote temporary semantic insertion. Colors encode communities, not verified disciplines; shapes encode roles. Parent-paper citation descriptors provide context, not claim-level support. No missing values are imputed.
d, All four claims are inserted into the union of their historical neighborhoods. Shared neighbors are counted once. No target-to-target edges are invented. Connectivity and scientific interpretation remain distinct.
e, Saved C1 branch assessments differ (GEAR: unresolved; Graph: recognized within the provided context). These categorical judgments describe their respective branches and are not directly interchangeable. The adjacent synthesis block is explicitly a generic method: no standard per-claim fused assessment was saved. The lower whole-paper report is an actual study output. Reviewer references are used for evaluation only.

Example: Du et al., Communications Biology (2026), doi:10.1038/s42003-026-09574-2. Target publication and analysis cutoff are both 23 January 2026 in this record. Historical graph claims derive from a different, bounded corpus. Contrast X: Ionic liquid-coated gold core polymeric nanoparticles for selective neutrophil hitchhiking towards endometriosis treatment, doi:10.1038/s42004-026-01909-8 (cutoff 5 February 2026). X1: AuCl4−-responsive self-assembly of ionic liquid block copolymers for obtaining composite gold nanoparticles and polymeric micelles with controlled morphologies (2014), OpenAlex W2030340194. These two papers were selected to illustrate the method with complete displayed metrics, not to estimate performance, missingness rates or cross-domain generalization.
"""
    (out / "caption_en.md").write_text(caption, encoding="utf-8")
    readme = f"""# Fig.2 reference 重绘交付

按 1055 × 1491 参考图坐标绘制；五个 panel、{len(manifest)} 个独立组件。文字与网络为矢量，最终 PDF 宽 297 mm。所有结果默认覆盖，不生成版本目录。

- `final/Fig2.svg`：可编辑母版；`Fig2_outlined.svg`：字体转路径便携版。
- `final/Fig2.pdf`：矢量 PDF；`Fig2.png`：2× 预览；`Fig2_6x.png`：6330 × 8946；`Fig2_10x.png`：10550 × 14910。
- `panels/`：a–e 独立 SVG/PDF/PNG；`components/`：独立小图；`components/layers/`：拼版源图层。
- `data/snapshot.json`：真实来源快照；`layouts/components.json`：组件坐标与接口；`qa/validation.json`：核验结果。

## 复现

```bash
python3 -m venv --system-site-packages /tmp/aspr-fig2-venv
/tmp/aspr-fig2-venv/bin/pip install -r figure_pipeline/fig2_reference/requirements.txt
/tmp/aspr-fig2-venv/bin/python -m figure_pipeline.fig2_reference all
/tmp/aspr-fig2-venv/bin/python -m figure_pipeline.fig2_reference render --only c
/tmp/aspr-fig2-venv/bin/python -m figure_pipeline.fig2_reference assemble --only c
/tmp/aspr-fig2-venv/bin/python -m figure_pipeline.fig2_reference validate
```

`prepare` 从已有运行读取并覆盖绘图快照；之后的 `render/assemble` 只使用快照。修改 Python 后运行 render；直接修改 components/layers 的 SVG 后运行 assemble。`--only` 支持 panel 或组件 ID，限制栅格导出范围，整图始终同步更新。`--no-high-res` 用于快速预览，会移除过期高清图，避免新旧混放。all/assemble 均更新同一个交付 ZIP。

## 数据口径和与参考图的必要差异

采用一个主论文和一个指标完整的机制对照。主论文 s42003-026-09574-2 的四条共享贡献贯穿 a/d/e，在 b 展开 C1，在 c 作为单社区对照；c 的完整 14 项指标改为 X。C1 有 10 个邻居、4 条历史骨架边，局部密度 4/45；局部分量 6→1，新增可达历史节点对 38。联合图有 27 个历史节点、8 个共享节点，分量 8→1，新增可达历史节点对 235。这些是局部连通描述，不是创新分数。

保留 GEAR 原始 inconclusive 状态、缺失 residual_contribution 和独立 assessment 的范围内增量解释，三者不混用。已按当前 _card 逻辑核对：coverage_sufficient=True 仅说明检索条件满足；仍有 1 条 UNRESOLVED 历史关系，因此 inconclusive 可以与之并存，并非必然数据冲突。residual_contribution 仅由 PARTIAL_ANTECEDENT 差异填充，不能据其空值抹去独立 assessment 中的限定范围解释。e 上部直接显示保存的 GEAR unresolved / Graph recognized，并将通用融合机制单独标注；没有标准逐 claim fusion 文件，因此不展示虚构的融合结果。e 下部仍为真实整篇报告摘要。

X 为 s42004-026-01909-8 的第 02 条 FINDING：金核–PLGA–离子液体外层的纳米颗粒结构及包覆后的尺寸、均匀性变化。10 个邻居均有社区归属，涉及 5 个社区，有效社区数 4.570，Rao–Stirling 差异度 0.145，非主导社区权重占比 0.701。全部 10 个无序社区对均见于历史配对表，未见配对占比为真实的 0，平均 surprisal 为 0.301。c 的 14 项显示指标全部有真实有限值，未用 0 替代缺失值。

b 的 X 对照使用 2014 年聚合物胶束内金纳米颗粒研究摘要。partial antecedent 为保存的系统判断，独立复核标志为 False；不将摘要证据表述为全文复核，也不将该判断当作人工真值。

主案例和 X 的引用路径均按去重、去自环规则核验。X 的 14 项复算值、全部社区对及路径见证保存在 data/detailed_profile_X.json。C1 仍是 a/d/e 的主论文和 c 的单社区对照，未展示的原始缺失字段保留于快照，不为消除显示缺失而改写原始结果。两篇论文定向选取，仅用于机制讲解；不能据此推断整体数据完整率或系统效果。

与参考图相同的 panel 边界与色彩层级；为确保可读性，14 指标改为两列分组，网络使用真实节点数量、拓扑和角色，删除示意图的虚构科学文字与数值。未更改 Fig.1 或运行 GEAR/全文抽取/模型评价。
"""
    (out / "README_zh.md").write_text(readme, encoding="utf-8")
    companion = [
        "# Figure 2 source companion",
        "",
        "All summaries below link to saved artifacts; aliases are figure-local, stable identifiers.",
        "",
    ]
    for work in snapshot["works"]:
        w = work["record"]["payload"]
        companion += [
            f"## {work['alias']}: {w['title']}",
            f"Year: {w['publication_year']}; evidence: {work['relation']['payload']['evidence_level']}",
            f"Evidence key: `{work['record']['evidence_id']}`",
            w.get("abstract") or "Abstract unavailable",
            "",
            work["relation"]["payload"]["rationale"],
            "",
        ]
    for span in snapshot["spans"]:
        companion += [
            f"## {snapshot['aliases'][span['span_id']]}: {span['span_id']}",
            span["text"],
            "",
        ]
    for claim in snapshot["claims"]:
        companion += [
            f"## {snapshot['aliases'][claim['claim_id']]}: {claim['claim_id']}",
            f"Role: {claim['claim_type']}",
            "Author claim: " + claim["author_claim_text"],
            "Grounded claim: " + claim["normalized_claim_text"],
            "Scope verification: " + claim["narrowing_reason"],
            "",
        ]
    for case in snapshot["contrasts"]:
        companion += [
            f"## Contrast {case['alias']}: {case['paper']['paper_id']}",
            case["selection_reason"],
            f"Cutoff: {case['paper']['cutoff_date']}",
            f"Claim ID: {case['claim']['claim_id']}",
            case["claim"]["normalized_claim_text"],
            "Scope verification: " + case["claim"]["narrowing_reason"],
            "Missing assessments: " + str(case["missing_assessments"]),
            "Saved descriptor values (all 14 displayed values belong to X):",
            *[
                f"- {name}: {value if value is not None else 'N/A'}"
                for name, value in case["metrics"].items()
            ],
        ]
        if case["prior"]:
            prior = case["prior"]
            work, relation = prior["work"]["payload"], prior["relation"]["payload"]
            companion += [
                f"### {prior['alias']}: {work['title']}",
                f"Evidence keys: {prior['work']['evidence_id']}; {prior['relation']['evidence_id']}",
                f"Relation: {relation['relation_label']}; evidence: {relation['evidence_level']}; independently verified: {relation['independent_verification_passed']}",
                relation["rationale"],
                "Common dimensions: " + "; ".join(relation["common_dimensions"]),
                "Differences: " + "; ".join(relation["difference_dimensions"]),
            ]
    companion += [
        "## Panel c detailed profile: X",
        "All 14 displayed descriptors belong to X, not C1. Recomputed values and source pair records: detailed_profile_X.json. Citation witnesses: path_witnesses_X.json.",
        "All ten neighbors are assigned to five communities, implying ten unordered pairs, all observed in the historical table. Unseen share = 0/10; mean pair surprisal = 0.3005252655. No missing value was imputed.",
    ]
    companion += [
        "## Main status reconciliation",
        snapshot["status_audit"]["interpretation"],
        snapshot["status_audit"]["residual_field"],
    ]
    companion += [
        "## Actual whole-paper report (unaltered)",
        snapshot["report"]["body"],
    ]
    (out / "data/source_companion.md").write_text(
        "\n\n".join(companion), encoding="utf-8"
    )
    cards = []
    for row in manifest:
        ident = html.escape(row["id"])
        cards.append(
            f'<article><a href="components/{ident}.svg"><img src="components/{ident}.png" alt="{html.escape(row["title"])}"></a><h3>{ident}</h3><p>{html.escape(row["title"])}</p><a href="components/{ident}.pdf">PDF</a> · <a href="components/{ident}.svg">SVG</a> · <a href="components/{ident}.png">PNG</a></article>'
        )
    gallery = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>Fig.2 reference</title><style>body{font:16px Arial;background:#f3f6fa;color:#203254;margin:32px auto;max-width:1300px}a{color:#285cb3}h1{font:32px Georgia}.hero{display:block;max-width:850px;width:100%;margin:24px auto;background:white}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}article{background:white;padding:15px;border:1px solid #dce6ef;border-radius:8px}article img{width:100%;height:230px;object-fit:contain}h3{font-size:15px}p{font-size:13px}</style><h1>Fig.2 · GEAR–Graph framework</h1><p>5 panels · 42 vector components · Source-backed example · Latest outputs overwrite previous files.</p><p><a href="final/Fig2.svg">Editable SVG</a> · <a href="final/Fig2.pdf">Vector PDF</a> · <a href="final/Fig2_6x.png">6× PNG</a> · <a href="final/Fig2_10x.png">10× PNG</a> · <a href="README_zh.md">说明与复现</a> · <a href="qa/reference_comparison.png">参考图对照</a></p><a href="final/Fig2_6x.png"><img class="hero" src="final/Fig2.png" alt="Figure 2"></a><h2>Independent panels</h2>"""
    gallery = gallery.replace(
        "42 vector components · Source-backed example",
        f"{len(manifest)} vector components · One main paper + one complete-metric contrast",
    )
    gallery += (
        "<p>"
        + " · ".join(f'<a href="panels/{p}.pdf">Panel {p} PDF</a>' for p in "abcde")
        + '</p><h2>Components</h2><section class="grid">'
        + "".join(cards)
        + "</section></html>"
    )
    (out / "index.html").write_text(gallery, encoding="utf-8")
    files = [
        {"path": str(p.relative_to(out)), "bytes": p.stat().st_size}
        for p in sorted(out.rglob("*"))
        if p.is_file() and p.name != "file_manifest.json"
    ]
    write(out / "file_manifest.json", files)


def package(out: Path) -> Path:
    destination = out.parent / "Fig2_reference_delivery.zip"
    temp = destination.with_suffix(".tmp.zip")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                archive.write(path, str(Path(out.name) / path.relative_to(out)))
        for folder in [
            ROOT / "figure_pipeline/fig2_reference",
            ROOT / "figure_pipeline/fig1_reference",
        ]:
            for path in folder.glob("*.py"):
                archive.write(path, str(path.relative_to(ROOT)))
        req = ROOT / "figure_pipeline/fig2_reference/requirements.txt"
        archive.write(req, str(req.relative_to(ROOT)))
        test = ROOT / "tests/figure_pipeline/test_fig2_reference.py"
        archive.write(test, str(test.relative_to(ROOT)))
    temp.replace(destination)
    return destination

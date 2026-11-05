"""Export inspectable graph tables and render verified findings."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict

from gear.innovation.joint_graph import components

from .graph_position import DEST, read, save


def main() -> None:
    group_meta = {g["group_id"]: g for g in read("groups")}
    tables = defaultdict(list)
    overview = ["# 经典成果的局部知识位置", ""]
    for path in sorted(DEST.glob("*/facts.json")):
        i = path.parent.name
        f = json.loads(path.read_text())
        members = f.get("members", [f])
        s = f["structure"]
        nodes = {n["claim_id"]: n for c in members for n in c["neighbors"]}
        before = components(set(nodes), s["historical_edges"])
        after = components(
            set(nodes) | {c["claim_id"] for c in members},
            s["historical_edges"] + s["insertion_edges"],
        )
        assert (
            len(set(before.values())) - len({after[n] for n in nodes})
            == s["joint_component_merge_count"]
        )
        row = {
            "id": i,
            "cutoff": members[0]["cutoff_exclusive"],
            "coverage": members[0]["graph_coverage_end"],
            "neighbors": len(nodes),
            "members": len(members),
            "components_before": s["historical_components_before"],
            "components_after": s["historical_components_after"],
            "new_pairs": s["joint_newly_connected_historical_pairs"],
        }
        tables["groups_summary" if i.startswith("G") else "claims_summary"].append(row)
        for c in members:
            tables["nodes"].append(
                {"task": i, "id": c["claim_id"], "kind": "target", "text": c["text_en"]}
            )
            if i.startswith("C"):
                for n in nodes.values():
                    assert (
                        n["publication_date"] < row["cutoff"]
                        and n["cosine_similarity"] > 0.5
                    )
                    tables["neighbors"].append(dict(target=i, **n))
                assert len(nodes) <= 10
                reported = dict(c["metrics"])
                if len(nodes) < 2:
                    reported["neighbor_induced_density"] = None
                tables["metrics"].append(dict(target=i, **reported))
                for k, v in Counter(n["community_id"] for n in nodes.values()).items():
                    tables["communities"].append(
                        {"target": i, "community": k, "count": v}
                    )
        if i.startswith("G"):
            for node in sorted(nodes):
                owners = [
                    c["claim_id"]
                    for c in members
                    if node in {n["claim_id"] for n in c["neighbors"]}
                ]
                tables["group_membership"].append(
                    {
                        "group_id": i,
                        "historical_node": node,
                        "members": owners,
                        "shared": len(owners) > 1,
                    }
                )
        for k, n in nodes.items():
            tables["nodes"].append(
                {"task": i, "id": k, "kind": "historical", "text": n["claim_text"]}
            )
        for key in ["historical_edges", "insertion_edges"]:
            tables["edges"].extend(
                {"task": i, "kind": key, "source": a, "target": b} for a, b in s[key]
            )
        if i.startswith("G"):
            tables["pairs"].extend(dict(group=i, **p) for p in s["claim_pairs"])
        lines = (
            [
                f"# {i} 局部知识位置",
                "",
                f"评估截止日 {row['cutoff']}（不含当天），图谱覆盖至 {row['coverage']}。",
                "",
            ]
            + [c["claim_id"] + " " + c["text_zh"] for c in members]
            + [
                "",
                f"历史邻居 {len(nodes)} 个。接入前后历史节点所属连通部分为 {row['components_before']} 和 {row['components_after']}。",
                "",
            ]
        )
        group = group_meta[members[0]["group_id"]]
        row["topic"] = group["topic"]
        row["no_neighbor_claims"] = s["claims_without_neighbors"]
        row["missing_members"] = (
            sorted(set(group["claim_ids"]) - {c["claim_id"] for c in members})
            if i.startswith("G")
            else []
        )
        row["status"] = (
            "limited"
            if row["missing_members"]
            or row["no_neighbor_claims"]
            or (path.parent / "excluded_findings.json").exists()
            else "complete"
        )
        row["claim_ids"] = [c["claim_id"] for c in members]
        row["claim_texts_zh"] = [c["text_zh"] for c in members]
        row["interpretation"] = None
        row["limits"] = None
        ap = path.parent / "analysis.json"
        if ap.exists():
            analysis = json.loads(ap.read_text())
            row["interpretation"] = analysis["findings"][0]["interpretation"]
            row["limits"] = analysis["limits"]
            for a in analysis["findings"]:
                lines.extend(
                    [
                        a["observation"],
                        a["interpretation"],
                        "依据 " + "、".join(a["evidence_keys"]),
                        "",
                    ]
                )
            lines.append(analysis["limits"])
            for c in members:
                if len(c["neighbors"]) == 1:
                    lines.append(
                        f"指标审阅：{c['claim_id']} 仅有一个历史近邻，可配对数为零。原生密度字段的 0 是实现默认值，正式指标表记为空值。由此无法判断邻域集中程度。"
                    )
            if i.startswith("G"):
                lines.append("成员重叠与分离")
                for pair in s["claim_pairs"]:
                    left, right = pair["claim_ids"]
                    lines.append(
                        f"{left} 与 {right} 共享 {pair['shared_neighbor_count']} 个近邻，Jaccard 为 {pair['jaccard']}，联合接入后两者"
                        + (
                            "连通。"
                            if pair["connected_via_historical_graph"]
                            else "分离。"
                        )
                    )
        else:
            lines.append("模型解释尚未完成。")
        if (path.parent / "excluded_findings.json").exists():
            lines.append(
                "部分解释因证据键未通过校验而排除，记录保存在 excluded_findings.json。"
            )
        lines.append("科学近邻原文")
        lines.extend(
            f"- {n['claim_id']} · {n['publication_date']} · cosine {n['cosine_similarity']:.4f} · {n['claim_text']}"
            for n in nodes.values()
        )
        (path.parent / "report.md").write_text("\n\n".join(lines))
        overview.append(
            f"- [{i} 逐项结果]({i}/report.md)，历史邻居 {len(nodes)} 个，新增可达历史节点对 {row['new_pairs']} 对。"
        )
        if ap.exists():
            overview.append(analysis["findings"][0]["interpretation"])
    for name, rows in tables.items():
        (DEST / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        )
        with (DEST / f"{name}.csv").open("w", encoding="utf-8-sig", newline="") as h:
            w = csv.DictWriter(h, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (DEST / "report.md").write_text("\n\n".join(overview))
    save(
        "validation.json",
        {
            "facts": len(list(DEST.glob("*/facts.json"))),
            "interpretations": len(list(DEST.glob("*/analysis.json"))),
            "edge_reconstruction": True,
        },
    )


if __name__ == "__main__":
    main()

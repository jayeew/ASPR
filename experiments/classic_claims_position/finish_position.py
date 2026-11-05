"""Assemble the accepted reports and traceable tables."""

from __future__ import annotations

import json
from zipfile import ZIP_DEFLATED, ZipFile

from gear.config import load_config
from gear.model_client import LazyRoleClient

from .audit_position import main as audit
from .export_position import main as export
from .graph_position import DEST, save
from .interpret_position import PROMPT, Position


def main() -> None:
    export()
    audit()
    val = json.loads((DEST / "validation.json").read_text())
    assert val["facts"] == val["interpretations"] == 24
    attempts = []
    groups = []
    for path in sorted(DEST.glob("*/analysis.json")):
        i = path.parent.name
        a = json.loads(path.read_text())
        f = json.loads(path.with_name("facts.json").read_text())
        members = f.get("members", [f])
        keys = (
            {"structure"}
            | {c["claim_id"] for c in members}
            | {
                c["claim_id"] + "/" + n["claim_id"]
                for c in members
                for n in c["neighbors"]
            }
        )
        assert all(
            x["evidence_keys"] and set(x["evidence_keys"]) <= keys
            for x in a["findings"]
        )
        evidence = {"structure": f["structure"]}
        for c in members:
            evidence[c["claim_id"]] = {
                k: c[k]
                for k in (
                    "text_en",
                    "cutoff_exclusive",
                    "graph_coverage_end",
                    "metrics",
                )
            }
            evidence.update(
                {c["claim_id"] + "/" + n["claim_id"]: n for n in c["neighbors"]}
            )
        client = LazyRoleClient(load_config(), "graph_analysis")._get()
        cache_path = client._cache_path(
            client._prompt(PROMPT, json.dumps(evidence, ensure_ascii=False)),
            Position.model_json_schema(),
        )
        cached = None
        if (
            cache_path
            and cache_path.exists()
            and json.loads(cache_path.read_text())
            == {k: a[k] for k in ("findings", "limits")}
        ):
            cached = cache_path.stat().st_mtime_ns < max(
                p.stat().st_mtime_ns for p in DEST.glob("*/facts.json")
            )
        attempts.append(
            {
                "task": i,
                "attempts": a["attempt"],
                "first_attempt_cache_hit_inferred_from_timestamp": cached,
                "failures": [
                    json.loads(p.read_text())
                    for p in path.parent.glob("attempt_*.json")
                ],
            }
        )
        if i.startswith("G"):
            groups.append((i, f, a))
    save("task_audit.json", attempts)
    lines = [
        "# 18 条经典成果的局部图谱位置",
        "",
        "18 条单条分析与 6 组联合分析均已完成。17 条取得 10 个合格近邻。C12 取得 1 个，故其局部结构判断受邻居数限制。2026 三组的 claim 图谱覆盖至 2025 年末。",
        "",
        "|组|成员|联合邻居|共享邻居|历史连通部分（前→后）|新增可达历史节点对|",
        "|---|---|---:|---:|---:|---:|",
    ]
    for i, f, a in groups:
        s = f["structure"]
        lines.append(
            f"|[{i}]({i}/report.md)|{', '.join(c['claim_id'] for c in f['members'])}|{s['historical_neighbor_count']}|{len(s['shared_historical_neighbors'])}|{s['historical_components_before']} → {s['historical_components_after']}|{s['joint_newly_connected_historical_pairs']}|"
        )
    for i, f, a in groups:
        lines += [
            "",
            f"## {i}",
            "",
            a["findings"][0]["interpretation"],
            "",
            a["limits"],
            "",
            " · ".join(
                "[" + c["claim_id"] + "](" + c["claim_id"] + "/report.md)"
                for c in f["members"]
            ),
        ]
    (DEST / "report.md").write_text("\n".join(lines))
    (DEST / "acceptance.md").write_text(
        "# 第二阶段验收\n\n18 个单条和 6 个联合任务均有结构事实和中文解释。全部证据键已核对，逐项检查日期与严格相似度阈值，并独立复算局部连通变化。原始资产和输入的保护检查见 run_manifest.json。\n\n调用次数和失败记录见 task_audit.json。首次请求的缓存状态依据请求键、缓存内容及文件时间重建；缺少逐请求遥测的项目保留为未知。\n"
    )
    with ZipFile(DEST / "graph_position.zip", "w", ZIP_DEFLATED) as z:
        for p in sorted(DEST.rglob("*")):
            if p.is_file() and p.suffix != ".zip":
                z.write(p, p.relative_to(DEST))


if __name__ == "__main__":
    main()

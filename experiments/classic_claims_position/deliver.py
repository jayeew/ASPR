"""Generate reviewable Chinese report, manifests and portable data package."""

from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

from .package import write_json
from .preflight import OUT, ROOT
from .validate import data_check, digest, protected_check, rows


def inventory() -> list[dict]:
    cached = OUT / "graph_file_inventory.json"
    if cached.exists():
        records = json.loads(cached.read_text())
        if all(
            Path(r["path"]).stat().st_size == r["bytes"]
            and Path(r["path"]).stat().st_mtime_ns == r["mtime_ns"]
            for r in records
        ):
            return records
    assets = []
    for root, kind in [
        (ROOT / "data/claim_graph", "reused_2026"),
        *[(p, "snapshot_2025") for p in sorted((OUT / "snapshots").iterdir())],
    ]:
        for p in sorted(root.iterdir()):
            if p.is_file() and p.suffix in {
                ".parquet",
                ".sqlite",
                ".npy",
                ".faiss",
                ".json",
            }:
                assets.append(
                    {
                        "kind": kind,
                        "path": str(p),
                        "bytes": p.stat().st_size,
                        "mtime_ns": p.stat().st_mtime_ns,
                        "sha256": digest(p),
                    }
                )
    write_json(cached, assets)
    return assets


def table(path: Path, data: list[dict]) -> None:
    if not data:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(
            {
                k: (
                    json.dumps(v, ensure_ascii=False)
                    if isinstance(v, (dict, list))
                    else v
                )
                for k, v in r.items()
            }
            for r in data
        )


def report(validation: dict) -> str:
    claims = rows("claims")
    sources = rows("sources")
    evidence = rows("evidence")
    source_by_id = {s["source_id"]: s for s in sources}
    failures = json.loads((OUT / "acquisition_failures.json").read_text())
    lines = [
        "# 经典成果在2025／2026年知识体系中的位置：第一阶段数据交付",
        "",
        "本阶段完成数据与图谱准备，不产生创新性、知识位置或联合插入分析结论。18条目标claim未插入任何图谱，没有调用创新分析模型，也没有修改Fig10。",
        "",
        "## 数据与验收概览",
        "",
        f"固定清单包含18条claim、6组。当前有{len(sources)}条科学文献记录、{len(evidence)}条可定位片段；18条claim均有核心支持片段。这里的“有支持”表示已定位到相应科学材料，不代表完整实验细节已核实，更不是系统评估结论。",
        "",
        f"程序验收：{'通过' if validation['valid'] else '未通过'}。原始图谱及受保护的AlphaFold 3资产检查：{'通过' if validation['protected']['valid'] else '未通过'}。详见validation.json。",
        "",
        "中文原文和英文评估文本逐条保存；不静默改写主张、增删成员或改换实验对象。英文保留原文的限定范围。物理及生物材料中的作者姓名、引用属于正常科学书目信息，未添加任何荣誉身份。",
        "",
        "## 研究口径与日期规则",
        "",
        "研究问题固定为：这些经典成果在2025／2026年的知识体系中处于什么位置，与已有研究及后续发展存在什么关系？历史年代仅为背景，不是本轮文献截止条件，也不是经过系统优先权调查核定的首次发现日。",
        "",
        "文献的最早可靠发表日期作为发表时间，保留日期精度。只有日期区间上界严格早于组截止日，材料才进入评估证据；截止日当天整日排除，不按公告时刻截断。没有足够日期信息的材料保留为待定书目。IceCube仪器正文实际读取arXiv v3（2024-02-06），与2017年期刊日期分别保存，并按较晚版本日检查资格。",
        "",
        "|组|内容|条数|评估截止日（不含）|图谱实际覆盖|图谱claim数|社区数|",
        "|---|---|---:|---|---|---:|---:|",
    ]
    for g in rows("groups"):
        asset = next(a for a in validation["graphs"] if a["root"] == g["graph_root"])
        lines.append(
            f"|{g['group_id']}|{g['topic']}|{len(g['claim_ids'])}|{g['cutoff_exclusive']}|{asset['coverage_start']}—{asset['coverage_end']}|{asset['claim_count']:,}|{asset['community_count']:,}|"
        )
    lines += [
        "",
        "六个日期的官方依据保存在external_provenance/cutoff_verification.json，包含URL、原始日期片段、获取日期及SHA256。2025医学公告正文抓取受限，采用官方搜索返回的公告日期；其余采用官方页面提取。外部材料不进入evaluation目录。",
        "",
        "## 图谱构建与适用边界",
        "",
        "三个2025快照独立过滤目标论文、claim、向量和论文引用索引；删除未来或无日期的论文节点及其关联边，再调用原生构建流程重建语义边、论文路径标注、Leiden社区、向量索引、质心和运行统计。没有沿用原图谱的社区标签或社区统计。论文路径计算使用原生三类关系的等价集合运算以缩小查询开销，执行前与原生SQL结果做一致性检查；中断日志与已完成分块均保留。",
        "",
        "三个2026组复用同一份现有资产，没有扩建2026文献。图谱的claim覆盖从2023年开始，来自有限的Nature系语料，不覆盖这些成果的完整历史与跨期刊发展。论文引用索引可能含更早的引用文献，这不等于存在那些年份的历史claim。后续“位置”只能解释为在该语料中的局部结构关系，不能宣称代表全科学知识。",
        "",
        "引用索引另有元数据异常：原始复用资产含1条日期为2031-01-01的非目标论文节点和4202条无日期节点。核查发现该未来节点没有被任何Nature目标论文引用，因而不会进入这些目标语义边的直接、两跳或共享参考文献路径标注；详情与SQL保存在reused_reference_anomalies.json。无日期参考记录仍属日期待定元数据，不能作为合格科学支持来源。三个2025快照已删除未来与无日期节点及关联边。主表的覆盖日期专指claim语料，不能用来概括整个原始引用索引。",
        "",
        "历史语义结构按当前仓库离线原生构建器生成。后续运行的邻居筛选应沿用运行时最多10个、cosine严格大于0.5及日期资格规则。本阶段不计算目标邻域、不插入目标。原生统计脚本产生的百分位文件作为构建产物留存，但清单明确禁用，不可将旧规则百分位用于后续阈值化指标的校准。",
        "",
        "## 逐条科学材料与支持范围",
        "",
    ]
    for c in claims:
        bound = [e for e in evidence if e["evidence_id"] in c["evidence_ids"]]
        lines += [
            f"### {c['claim_id']} · {c['group_id']} · {c['historical_period']}",
            "",
            "**中文原文：** " + c["text_zh"],
            "",
            "**英文评估文本：** " + c["text_en"],
            "",
            f"类型：{c['claim_type']}；截止：{c['cutoff_exclusive']}（不含）；支持状态：{c['evidence_status']}。",
            "",
        ]
        for e in bound:
            s = source_by_id[e["source_id"]]
            lines.append(
                f"- [{s['title']}]({s['url']}) · {s['date_lower'] or '日期待定'}（{s['date_precision']}）· {e['role']} / {e['relationship']} / {e['material_type']}。{e['support_scope']} 定位：`{e['locator']}`；来源`{s['source_id']}`，证据`{e['evidence_id']}`。"
            )
        lines += ["", "局限：" + "；".join(c["limitations"]), ""]
    lines += [
        "## 获取缺口、建议修订与失败记录",
        "",
        "C03的1989年原始论文、C13的1986年原始论文暂未取得可用正文或摘要，分别以2013年综述和2020年后续实验定位一般科学主张，不能据此核定原始优先权。C01专门隧穿论文未下载成功，改由同年能级论文中的原始隧穿测量段落支持。C17原始摘要支持光控放电，蓝光和培养条件仍标为细节级待补。",
        "",
        "C15的原文保持不变：2000年材料已经报告大于99.5% ee，因此不能把“接近单一对映体”这个宽泛claim唯一归于2003年。建议下一阶段把历史年份解释为研究发展区间，或另行核查最早实验；本阶段未静默改写。C11的1988—2011区间亦保留为用户背景，不把2017年仪器总结当作1988年的原始论证。",
        "",
        f"共记录{len(failures)}条缓存获取失败/受限元数据，详见acquisition_failures.json；它们包含HTTP错误、超时和材料权限限制，不当作论文不存在。无摘要记录如下：",
        "",
        "|DOI|标题|日期状态|",
        "|---|---|---|",
    ]
    for s in sources:
        if s["material_status"] == "reference_only":
            lines.append(f"|{s['doi']}|{s['title']}|{s['date_lower'] or '待定'}|")
    lines += [
        "",
        "候选DOI 10.1021/ja00198a049实际指向无关论文，已排除，改用10.1021/ja00197a079；其他未解析候选保留在excluded_sources.json。全部初始抓取文件仍在sources目录，不进入有效证据列表。多来源不等于独立复现；后续综述或同一研究组总结均按实际材料身份标记。检索是围绕18个主张的证据准备，不是穷尽至2025／2026年的系统综述。",
        "",
        "## 数据契约、脱敏和下一阶段接口",
        "",
        "`evaluation/claims.jsonl`是固定科学主张，`sources.jsonl`是书目，`evidence.jsonl`是可定位片段及支持范围，`groups.jsonl`和`groups/G01.json`至`G06.json`定义六组联合输入包。每条片段绑定来源和claim，附原始文件路径、SHA256、获取信息、材料层级和版本日期。独立Pydantic契约禁止多余字段，schema.json可审阅。原始片段同时写入追加式EvidenceStore；早期整理版本保留，当前有效集合以evaluation/evidence.jsonl为准。",
        "",
        "评估输入只使用evaluation目录及其科学来源；external_provenance保存外部奖项来源、原编号映射与用户文档。程序扫描评估JSON的显式奖项词和原编号。原始出版商页面可能带站点导航，因此不能将sources原始HTML或整个交付目录直接塞给模型；必须采用已筛选的片段。脱敏不消除模型已有知识。",
        "",
        "现有gear review接口接收论文全文契约，本数据是多来源案例包，没有伪造论文全文、作者、DOI或论文发表日。后续需要单独适配固定claim与来源绑定、知识位置研究问题、GEAR与Graph各自的结果语义、时间角色、六组联合插入及输出校验；不能将既有“新颖性认可／未决”直接改名。本阶段没有实施这些运行适配。",
        "",
        "## 交付目录与复现",
        "",
        "- `claims.csv`：18条中英文总表、年代、日期、状态、来源和证据编号。",
        "- `sources.csv`、`evidence.csv`、`groups.csv`：科学来源、片段与组映射。",
        "- `evaluation/`：中性JSONL、18个逐条证据包及6组联合数据包。",
        "- `sources/`、`web_scientific/`：已获取材料、原始片段及下载状态。",
        "- `snapshots/`：三个独立历史图谱；`graph_assets.json`：包括2026复用资产的文件哈希与版本。",
        "- `validation.json`及`acceptance_report.md`：验收结果、覆盖边界和待补事项。",
        "- `source_files.json`：科学材料与外部溯源文件的校验清单。",
        "- `data_bundle.zip`：便于审阅的数据、证据和报告，排除大型图谱与虚拟环境；图谱仍完整保存在上述快照目录。",
        "",
        "复现命令见experiments/classic_claims_position/README.md。已保存的web工具提取作为输入材料一并保留，可在离线状态重建数据表和校验；网络重新抓取可能因访问权限变化得到不同结果，应保留新的获取记录。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    validation = json.loads((OUT / "validation.json").read_text())
    equivalence = json.loads((OUT / "native_motif_equivalence.json").read_text())
    assert len(equivalence) == 21 and all(r["native_equivalent"] for r in equivalence)
    validation["reused_reference_metadata"] = json.loads(
        (OUT / "reused_reference_anomalies.json").read_text()
    )
    validation["native_motif_audit"] = {
        "valid": True,
        "checked_nonempty_annotations": 21,
        "tolerance": "float32: rtol=1e-6, atol=1e-7; counts and flags exact",
    }
    statistics = []
    for graph in validation["graphs"]:
        stats = json.loads((Path(graph["root"]) / "phase8_stats.json").read_text())
        valid = (
            stats["claim_count"] == graph["claim_count"]
            and stats["community_count"] == graph["community_count"]
        )
        assert valid, graph["root"]
        statistics.append(
            {
                "root": graph["root"],
                "valid": valid,
                "claim_count": stats["claim_count"],
                "community_count": stats["community_count"],
            }
        )
    validation["statistics_alignment"] = statistics
    for group in rows("groups"):
        asset = next(
            g for g in validation["graphs"] if g["root"] == group["graph_root"]
        )
        assert asset["coverage_end"] == group["graph_coverage_end"]
    assert (
        validation["reused_reference_metadata"]["future_nodes_cited_by_nature_targets"]
        == 0
    )
    validation["data"] = data_check()
    validation["protected"] = protected_check()
    validation["valid"] = all(
        [
            validation["data"]["valid"],
            validation["protected"]["valid"],
            *[g["valid"] for g in validation["graphs"]],
        ]
    )
    write_json(OUT / "validation.json", validation)
    if not validation["valid"]:
        raise RuntimeError("Acceptance failed; do not publish completion")
    for name in ["claims", "sources", "evidence", "groups"]:
        if name != "claims":
            table(OUT / f"{name}.csv", rows(name))
    assets = inventory()
    write_json(
        OUT / "graph_assets.json",
        {
            "graphs": validation["graphs"],
            "files": assets,
            "percentile_use": "disabled",
            "original_graph_modified": False,
        },
    )
    files = []
    for directory in ["sources", "web_scientific", "external_provenance"]:
        for p in sorted((OUT / directory).rglob("*")):
            if p.is_file():
                files.append(
                    {
                        "path": str(p.relative_to(OUT)),
                        "bytes": p.stat().st_size,
                        "sha256": digest(p),
                    }
                )
    write_json(OUT / "source_files.json", files)
    (OUT / "review_report_zh.md").write_text(report(validation))
    (OUT / "acceptance_report.md").write_text(
        "# 第一阶段验收报告\n\n通过全部程序验收；详细记录见validation.json。\n\n```json\n"
        + json.dumps(validation, ensure_ascii=False, indent=2)
        + "\n```\n\n18条均有可定位核心支持；原始全文获取不足及C17条件级缺口不隐藏，详见中文审阅报告。未运行分析、未插入claim、未修改Fig10。\n"
    )
    with zipfile.ZipFile(OUT / "data_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob("*")):
            rel = p.relative_to(OUT)
            if (
                p.is_file()
                and rel.parts[0] not in {".venv", "snapshots"}
                and p.suffix not in {".zip", ".log"}
            ):
                z.write(p, str(rel))
    print("Delivered", OUT, flush=True)


if __name__ == "__main__":
    main()

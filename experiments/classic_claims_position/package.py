"""Build neutral evidence packets and human review tables; never assess claims."""

from __future__ import annotations

import calendar
import csv
import hashlib
import json
import re
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

from gear.trace import EvidenceStore

from .catalog import CLAIMS, SOURCES
from .contracts import Claim, Evidence, Group, Source
from .preflight import OUT, ROOT
from .scope_notes import CONTEXT_DOIS, LIMITS, SCOPE

GROUPS = [
    ("G01", "超导电路", 1, 2, "2025-10-07"),
    ("G02", "MOF", 3, 6, "2025-10-08"),
    ("G03", "免疫耐受", 7, 10, "2025-10-06"),
    ("G04", "IceCube", 11, 12, "2026-10-06"),
    ("G05", "手性化学", 13, 15, "2026-10-07"),
    ("G06", "光遗传学", 16, 18, "2026-10-05"),
]
EXCLUDE = {
    "10.1002/anie.200705241": "补充候选未取得可解析书目或文本；不纳入有效来源",
    "10.1021/ja00198a049": "候选 DOI 指向无关的四苯乙烯研究；正确 DOI 为 10.1021/ja00197a079",
    "10.1038/83805": "未能解析候选 DOI；不纳入证据",
    "10.1038/nmeth.1036": "未能解析候选 DOI；不纳入证据",
    "10.1038/nmeth.2": "未能解析候选 DOI；不纳入证据",
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def date_bounds(candidates: list[dict]) -> tuple:
    # Print date wins over much later digitisation of old papers; all candidates retained in raw records.
    values = []
    for c in candidates:
        p = c["parts"]
        y = p[0]
        m = p[1] if len(p) > 1 else 1
        d = p[2] if len(p) > 2 else 1
        lo = f"{y:04}-{m:02}-{d:02}"
        hi = f"{y:04}-{(p[1] if len(p)>1 else 12):02}-{(p[2] if len(p)>2 else calendar.monthrange(y,p[1] if len(p)>1 else 12)[1]):02}"
        values.append(
            (lo, hi, ["year", "month", "day"][min(len(p), 3) - 1], c["field"])
        )
    return min(values) if values else (None, None, "unknown", "unavailable")


def publication_candidates(record: dict) -> list[dict]:
    if record["date_candidates"]:
        return record["date_candidates"]
    path = OUT / record["raw_directory"] / "europepmc.json"
    metadata = path.with_suffix(".json.meta.json")
    if not metadata.exists() or json.loads(metadata.read_text()).get("status") != 200:
        return []
    records = json.loads(path.read_text()).get("resultList", {}).get("result", [])
    for item in records:
        if item.get("doi", "").lower() != record["doi"].lower():
            continue
        value = item.get("firstPublicationDate")
        if value:
            parts = [int(p) for p in value.split("-")]
            # EuropePMC may impute day 1 for a month-only record. Keep the conservative interval.
            if parts[-1] == 1:
                parts = parts[:2]
            return [
                {
                    "field": "europepmc.json:firstPublicationDate (day 1 treated as month precision)",
                    "parts": parts,
                }
            ]
    return []


def excerpts(record: dict) -> list[dict]:
    directory = OUT / record["raw_directory"]
    pieces = []
    if record["abstracts"]:
        item = record["abstracts"][-1]
        filename = item["locator"].split(":")[0].split(" CSS ")[0]
        pieces.append(
            {
                "text": item["text"],
                "locator": item["locator"],
                "raw_file": str((directory / filename).relative_to(OUT)),
                "material_type": "abstract",
            }
        )
    for filename, pdf in [
        ("aps_pages.json", "aps_fulltext.pdf"),
        ("pdf_pages.json", "supplement.pdf"),
    ]:
        p = directory / filename
        if p.exists():
            pages = json.loads(p.read_text())
            # Primary physics papers: first page contains experimental abstract. IceCube front matter spans pages.
            selected = (
                range(min(8, len(pages)))
                if "P03012" in record["doi"]
                else range(min(1, len(pages)))
            )
            for i in selected:
                text = pages[i].strip()
                if len(text) > 150:
                    pieces.append(
                        {
                            "text": text,
                            "locator": f"{pdf} page {i+1}; {filename}[{i}]",
                            "raw_file": str(p.relative_to(OUT)),
                            "material_type": "fulltext_excerpt",
                        }
                    )
    web = directory / "curated_excerpt.json"
    if web.exists():
        pieces.append(json.loads(web.read_text()))
    return pieces


def main() -> None:
    public = OUT / "evaluation"
    public.mkdir(exist_ok=True)
    source_rows = []
    evidence_rows = []
    failures = []
    excluded = []
    for path in sorted((OUT / "sources").glob("*/record.json")):
        r = json.loads(path.read_text())
        failures.extend(
            {"source_id": r["source_id"], **a}
            for a in r["attempts"]
            if a["status"] != 200
        )
        if r["doi"] in EXCLUDE:
            excluded.append({"doi": r["doi"], "reason": EXCLUDE[r["doi"]]})
            continue
        r["claim_numbers"] = {d: ns for d, ns, _ in SOURCES}.get(
            r["doi"], r["claim_numbers"]
        )
        lo, hi, precision, basis = date_bounds(publication_candidates(r))
        pieces = excerpts(r)
        sr = Source(
            source_id=r["source_id"],
            claim_ids=[f"C{n:02}" for n in r["claim_numbers"]],
            role=r["role"],
            doi=r["doi"],
            url=r["url"],
            title=re.sub("<[^>]+>", "", r["title"]),
            date_lower=lo,
            date_upper=hi,
            date_precision=precision,
            date_basis=basis,
            acquired_at=r["attempts"][0]["acquired_at"],
            authors=r["authors"],
            content_version_date="2024-02-06" if "P03012" in r["doi"] else None,
            material_status=(
                "fulltext"
                if any(x["material_type"] == "fulltext_excerpt" for x in pieces)
                or r["fulltext_available"]
                else ("abstract" if pieces else "reference_only")
            ),
            raw_directory=r["raw_directory"],
        )
        source_rows.append(sr.model_dump(mode="json"))
        for item in pieces:
            raw = OUT / item["raw_file"]
            ident = (
                "E-"
                + hashlib.sha256(
                    ("v2" + r["source_id"] + item["locator"] + item["text"]).encode()
                ).hexdigest()[:16]
            )
            ids = []
            for n in r["claim_numbers"]:
                cutoff = next(g[4] for g in GROUPS if g[2] <= n <= g[3])
                if (
                    hi
                    and max(hi, "2024-02-06" if "P03012" in r["doi"] else hi) < cutoff
                ):
                    ids.append(f"C{n:02}")
            if not ids:
                continue
            if re.search(
                r"nobel|laureate|诺贝尔|获奖|授奖", item["text"], re.IGNORECASE
            ):
                excluded.append(
                    {
                        "source_id": r["source_id"],
                        "locator": item["locator"],
                        "reason": "Explicit award wording in passage",
                    }
                )
                continue
            e = Evidence(
                evidence_id=ident,
                source_id=r["source_id"],
                claim_ids=ids,
                role=r["role"],
                support_scope=SCOPE.get(
                    r["doi"],
                    "仅作为所标 claim 的后续发展背景；不证明原始首创或完整实验条件。",
                ),
                relationship=(
                    "development_context"
                    if r["doi"] in CONTEXT_DOIS
                    else "core_support"
                ),
                raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
                **item,
            )
            evidence_rows.append(e.model_dump(mode="json"))
    claim_rows = []
    mapping = []
    groups = []
    for gid, topic, start, end, cutoff in GROUPS:
        for n in range(start, end + 1):
            original, zh, en, kind, period = CLAIMS[n - 1]
            evid = [
                e["evidence_id"] for e in evidence_rows if f"C{n:02}" in e["claim_ids"]
            ]
            core = [
                e
                for e in evidence_rows
                if f"C{n:02}" in e["claim_ids"] and e["relationship"] == "core_support"
            ]
            limitations = ["多来源整理材料，不是待审论文；片段支持不等于独立实验复现。"]
            limitations.extend(LIMITS.get(n, []))
            if not core:
                limitations.append("未取得可定位的科学支持片段，保持待补。")
            claim = Claim(
                claim_id=f"C{n:02}",
                group_id=gid,
                text_zh=zh,
                text_en=en,
                claim_type=kind,
                historical_period=period,
                cutoff_exclusive=cutoff,
                evidence_ids=evid,
                source_ids=sorted(
                    {
                        e["source_id"]
                        for e in evidence_rows
                        if f"C{n:02}" in e["claim_ids"]
                    }
                ),
                evidence_status="located" if core else "pending",
                limitations=limitations,
            )
            claim_rows.append(claim.model_dump(mode="json"))
            mapping.append(
                {"claim_id": f"C{n:02}", "original_id": original, "group_id": gid}
            )
        graph = (
            OUT / "snapshots" / cutoff
            if cutoff.startswith("2025")
            else ROOT / "data/claim_graph"
        )
        groups.append(
            Group(
                group_id=gid,
                topic=topic,
                claim_ids=[f"C{n:02}" for n in range(start, end + 1)],
                cutoff_exclusive=cutoff,
                graph_root=str(graph),
                graph_coverage_end=pc.max(
                    pq.read_table(
                        graph / "claim_nodes.parquet", columns=["publication_date"]
                    )["publication_date"]
                ).as_py(),
                graph_limitations=[
                    "Nature 系有限语料，起于 2023 年；知识位置仅指该语料的局部关系。"
                ]
                + (["不包含 2026 年图谱文献。"] if cutoff.startswith("2026") else []),
            ).model_dump(mode="json")
        )
    for name, rows in [
        ("claims", claim_rows),
        ("sources", source_rows),
        ("evidence", evidence_rows),
        ("groups", groups),
    ]:
        write_rows(public / f"{name}.jsonl", rows)
    for c in claim_rows:
        selected = [e for e in evidence_rows if c["claim_id"] in e["claim_ids"]]
        source_ids = {e["source_id"] for e in selected}
        write_json(
            public / "claims" / f"{c['claim_id']}.json",
            {
                "claim": c,
                "evidence": selected,
                "sources": [r for r in source_rows if r["source_id"] in source_ids],
            },
        )
    for g in groups:
        ids = set(g["claim_ids"])
        e = [r for r in evidence_rows if ids.intersection(r["claim_ids"])]
        sids = {r["source_id"] for r in e}
        write_json(
            public / "groups" / f"{g['group_id']}.json",
            {
                "group": g,
                "claims": [r for r in claim_rows if r["claim_id"] in ids],
                "evidence": e,
                "sources": [r for r in source_rows if r["source_id"] in sids],
            },
        )
    write_json(OUT / "external_provenance/claim_mapping.json", mapping)
    failures = [
        dict(metadata_file=str(p.relative_to(OUT)), **json.loads(p.read_text()))
        for p in (OUT / "sources").rglob("*.meta.json")
        if json.loads(p.read_text()).get("status") != 200
    ]
    write_rows(
        OUT / "sources.jsonl",
        [
            json.loads(p.read_text())
            for p in sorted((OUT / "sources").glob("*/record.json"))
        ],
    )
    write_json(OUT / "acquisition_failures.json", failures)
    write_json(OUT / "excluded_sources.json", excluded)
    write_json(
        OUT / "schema.json",
        {m.__name__: m.model_json_schema() for m in (Claim, Evidence, Source, Group)},
    )
    store = EvidenceStore(OUT / "evidence_store")
    for r in evidence_rows:
        store.add_evidence(r["evidence_id"], "scientific_source_excerpt", r)
    with (OUT / "claims.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        rows = [
            {
                **c,
                "evidence_ids": ";".join(c["evidence_ids"]),
                "source_ids": ";".join(c["source_ids"]),
                "limitations": ";".join(c["limitations"]),
            }
            for c in claim_rows
        ]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        "claims",
        len(claim_rows),
        "sources",
        len(source_rows),
        "evidence",
        len(evidence_rows),
        "pending",
        [r["claim_id"] for r in claim_rows if not r["evidence_ids"]],
    )


if __name__ == "__main__":
    main()

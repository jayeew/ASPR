"""Offline acceptance checks for the preparation bundle and native graph assets."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from gear.innovation.validation import validate_assets

from .catalog import CLAIMS
from .contracts import Claim, Evidence, Group, Source
from .preflight import OUT, ROOT


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def rows(name: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (OUT / "evaluation" / f"{name}.jsonl").read_text().splitlines()
    ]


def graph_check(root: Path, cutoff: str | None) -> dict[str, Any]:
    result = validate_assets(root)
    if not result["valid"]:
        return result
    with sqlite3.connect(
        f"file:{root}/claim_graph_index.sqlite?mode=ro", uri=True
    ) as db:
        lo, hi, count = db.execute(
            "SELECT MIN(publication_date),MAX(publication_date),COUNT(*) FROM claim_nodes"
        ).fetchone()
        bad_edges = db.execute(
            "SELECT COUNT(*) FROM claim_edges e LEFT JOIN claim_nodes a ON a.claim_row=e.earlier_row LEFT JOIN claim_nodes b ON b.claim_row=e.later_row WHERE a.claim_id IS NULL OR b.claim_id IS NULL OR a.publication_date>=b.publication_date OR a.parent_paper_id=b.parent_paper_id"
        ).fetchone()[0]
        edge_count = db.execute("SELECT COUNT(*) FROM claim_edges").fetchone()[0]
        communities = db.execute(
            "SELECT COUNT(DISTINCT community_id) FROM claim_nodes"
        ).fetchone()[0]
        sql_ids = db.execute(
            "SELECT claim_id FROM claim_nodes ORDER BY claim_row"
        ).fetchall()
    index = pq.read_table(root / "claim_embedding_index.parquet").to_pylist()
    ordered = [r["claim_id"] for r in sorted(index, key=lambda r: r["claim_row"])]
    claim_ids = set(
        pq.read_table(root / "claim_nodes.parquet", columns=["claim_id"])[
            "claim_id"
        ].to_pylist()
    )
    aligned = ordered == [r[0] for r in sql_ids] and set(ordered) == claim_ids
    with sqlite3.connect(
        f"file:{root}/paper_graph_index.sqlite?mode=ro", uri=True
    ) as db:
        paper_lo, paper_hi, paper_count = db.execute(
            "SELECT MIN(publication_date),MAX(publication_date),COUNT(*) FROM paper_nodes"
        ).fetchone()
        future = (
            db.execute(
                "SELECT COUNT(*) FROM paper_nodes WHERE publication_date IS NULL OR publication_date>=?",
                (cutoff,),
            ).fetchone()[0]
            if cutoff
            else None
        )
        # Indexed anti-join checks both endpoints, including any deleted future nodes.
        orphan = db.execute(
            "SELECT COUNT(*) FROM paper_edges e WHERE NOT EXISTS (SELECT 1 FROM paper_nodes p WHERE p.work_id=e.citing_work_id) OR NOT EXISTS (SELECT 1 FROM paper_nodes p WHERE p.work_id=e.cited_work_id)"
        ).fetchone()[0]
    centroids = np.load(root / "community_centroid_matrix.npy", mmap_mode="r")
    centroid_index = pq.read_table(root / "community_centroid_index.parquet")
    centroid_ok = len(centroid_index) == centroids.shape[0] == communities
    result.update(
        root=str(root),
        coverage_start=lo,
        coverage_end=hi,
        claim_count=count,
        edge_count=edge_count,
        community_count=communities,
        invalid_edges=bad_edges,
        embedding_alignment=aligned,
        centroid_alignment=centroid_ok,
        paper_count=paper_count,
        paper_coverage_start=paper_lo,
        paper_coverage_end=paper_hi,
        future_or_undated_papers=future,
        orphan_paper_edges=orphan,
        cutoff_exclusive=cutoff,
        percentile_use="disabled",
        target_insertion=False,
    )
    result["valid"] = bool(
        result["valid"]
        and aligned
        and centroid_ok
        and bad_edges == 0
        and orphan == 0
        and (not cutoff or (hi < cutoff and future == 0))
    )
    return result


def data_check() -> dict:
    data = {name: rows(name) for name in ("claims", "groups", "sources", "evidence")}
    errors = []
    for name, model in [
        ("claims", Claim),
        ("groups", Group),
        ("sources", Source),
        ("evidence", Evidence),
    ]:
        for row in data[name]:
            model.model_validate(row)
    claims = {r["claim_id"]: r for r in data["claims"]}
    sources = {r["source_id"]: r for r in data["sources"]}
    evidence = {r["evidence_id"]: r for r in data["evidence"]}
    if len(claims) != 18 or len(data["groups"]) != 6:
        errors.append("wrong counts")
    if {g["group_id"] for g in data["groups"]} != {f"G{i:02}" for i in range(1, 7)}:
        errors.append("group identities")
    if len(sources) != len(data["sources"]) or len(evidence) != len(data["evidence"]):
        errors.append("duplicate source or evidence identities")
    for g in data["groups"]:
        for cid in g["claim_ids"]:
            if (
                cid not in claims
                or claims[cid]["group_id"] != g["group_id"]
                or claims[cid]["cutoff_exclusive"] != g["cutoff_exclusive"]
            ):
                errors.append("claim group/cutoff mismatch " + cid)
    if (
        len(list((OUT / "evaluation/claims").glob("C*.json"))) != 18
        or len(list((OUT / "evaluation/groups").glob("G*.json"))) != 6
    ):
        errors.append("missing individual/group packet")
    members = [c for g in data["groups"] for c in g["claim_ids"]]
    if len(set(members)) != 18 or sorted(members) != sorted(claims):
        errors.append("group mapping")
    if [len(g["claim_ids"]) for g in data["groups"]] != [2, 4, 4, 2, 3, 3]:
        errors.append("group sizes")
    for n, c in enumerate(data["claims"]):
        if (c["text_zh"], c["text_en"]) != tuple(CLAIMS[n][1:3]):
            errors.append("changed claim " + c["claim_id"])
        for eid in c["evidence_ids"]:
            if eid not in evidence or c["claim_id"] not in evidence[eid]["claim_ids"]:
                errors.append("unbound " + eid)
    for e in data["evidence"]:
        if digest(OUT / e["raw_file"]) != e["raw_sha256"]:
            errors.append("hash " + e["evidence_id"])
        s = sources[e["source_id"]]
        for cid in e["claim_ids"]:
            if (
                not s["date_upper"]
                or max(s["date_upper"], s["content_version_date"] or s["date_upper"])
                >= claims[cid]["cutoff_exclusive"]
            ):
                errors.append("date " + e["evidence_id"])
    for p in (OUT / "evaluation").rglob("*"):
        if p.is_file() and re.search(
            r"nobel|laureate|诺贝尔|获奖|授奖|P25-C|P26-C|M25-C|M26-C|C25-C|C26-C",
            p.read_text(),
            re.IGNORECASE,
        ):
            errors.append("award leakage " + str(p))
    for r in json.loads(
        (OUT / "external_provenance/cutoff_verification.json").read_text()
    ):
        raw = OUT / "external_provenance" / r["raw_file"]
        if digest(raw) != r["raw_sha256"] or r["date_quote"] not in raw.read_text():
            errors.append("cutoff provenance " + r["group_id"])
    return {
        "valid": not errors,
        "errors": errors,
        "claim_count": len(claims),
        "group_count": 6,
        "source_count": len(sources),
        "evidence_count": len(evidence),
        "located_core_claims": sum(
            c["evidence_status"] == "located" for c in claims.values()
        ),
        "scope": "schema,identity,date,hash,explicit-label checks; scientific scope manually recorded",
    }


def protected_check() -> dict:
    initial = json.loads((OUT / "protected_before.json").read_text())
    changed = []
    for name, old in initial.items():
        p = ROOT / name
        if (
            (
                not p.exists()
                or p.stat().st_size != old["size"]
                or p.stat().st_mtime_ns != old["mtime_ns"]
            )
            or old.get("sha256")
            and digest(p) != old["sha256"].removeprefix("sha256:")
        ):
            changed.append(name)
    return {
        "valid": not changed,
        "checked_files": len(initial),
        "changed": changed,
        "method": "size+mtime_ns for source assets; additional SHA256 for protected figure outputs where recorded",
    }


def main() -> None:
    result = {"data": data_check(), "graphs": [], "protected": protected_check()}
    for cutoff in ["2025-10-06", "2025-10-07", "2025-10-08", None]:
        root = OUT / "snapshots" / cutoff if cutoff else ROOT / "data/claim_graph"
        print("Validating", root, flush=True)
        result["graphs"].append(graph_check(root, cutoff))
    result["valid"] = all(
        [
            result["data"]["valid"],
            result["protected"]["valid"],
            *[g["valid"] for g in result["graphs"]],
        ]
    )
    (OUT / "validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()

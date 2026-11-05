"""Read-only graph placement of fixed contributions."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from gear.claim_attribution import ClaimGraphRuntime
from gear.innovation.joint_graph import joint_structure

from .preflight import OUT, ROOT

DEST = OUT.parent / "graph_position_v1"


def read(name: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (OUT / "evaluation" / f"{name}.jsonl").read_text().splitlines()
    ]


def save(name: str, value: object) -> None:
    p = DEST / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def edges(rt: ClaimGraphRuntime, neighbors: list) -> list:
    mapping = {
        rt._claim_db.execute(
            "SELECT claim_row FROM claim_nodes WHERE claim_id=?", (n.claim_id,)
        ).fetchone()[0]: n.claim_id
        for n in neighbors
    }
    return [[mapping[a], mapping[b]] for a, b in sorted(rt._neighbor_edges(neighbors))]


def main() -> None:
    claims = read("claims")
    model = ROOT / "data/models/Qwen3-Embedding-4B"
    encoder = ClaimGraphRuntime(ROOT / "data/claim_graph", model)
    vectors = encoder.encode_batch([c["text_en"] for c in claims], batch_size=4)
    encoder.close()
    for group in read("groups"):
        rt = ClaimGraphRuntime(Path(group["graph_root"]), model)
        cards = []
        facts = []

        for c, v in zip(claims, vectors, strict=True):
            if c["group_id"] != group["group_id"]:
                continue
            ident = c["claim_id"]
            ns = rt._neighbors(
                v,
                SimpleNamespace(
                    paper_id=None,
                    reference_work_ids=[],
                    cutoff_date=date.fromisoformat(c["cutoff_exclusive"]),
                ),
            )
            card = SimpleNamespace(claim=SimpleNamespace(claim_id=ident), neighbors=ns)
            cards.append(card)
            metrics = {m.name: m.value for m in rt._metrics(ns, c["claim_type"])}
            for k in [
                "direct_citation_neighbor_count",
                "two_hop_neighbor_count",
                "co_citation_neighbor_count",
            ]:
                metrics[k] = None
            fact = {
                k: c[k]
                for k in [
                    "claim_id",
                    "group_id",
                    "text_en",
                    "text_zh",
                    "cutoff_exclusive",
                    "claim_type",
                ]
            }
            fact.update(
                graph_coverage_end=group["graph_coverage_end"],
                neighbors=[
                    n.model_dump(
                        mode="json",
                        exclude={
                            "direct_citation",
                            "two_hop_path_count",
                            "shared_reference_count",
                            "shared_reference_salton",
                        },
                    )
                    for n in ns
                ],
                citation_path=None,
                citation_reason="无目标论文参考文献表",
                metrics=metrics,
                structure=joint_structure([card], edges(rt, ns)),
            )
            save(f"{ident}/facts.json", fact)
            facts.append(fact)
        union = {n.claim_id: n for card in cards for n in card.neighbors}
        structure = joint_structure(cards, edges(rt, list(union.values())))
        sets = {c.claim.claim_id: {n.claim_id for n in c.neighbors} for c in cards}
        for pair in structure["claim_pairs"]:
            a, b = map(sets.get, pair["claim_ids"])
            pair["jaccard"] = len(a & b) / len(a | b) if a | b else None
        save(
            group["group_id"] + "/facts.json",
            {"group_id": group["group_id"], "members": facts, "structure": structure},
        )
        rt.close()
        print(group["group_id"], len(union), flush=True)


if __name__ == "__main__":
    main()

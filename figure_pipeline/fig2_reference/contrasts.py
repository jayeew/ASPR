"""A complete-data mechanism contrast, with no new model-generated judgments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .data import ROOT, STUDY, file_source, network_layout, read, records, write
from .profile import prepare_profile

SELECTION = [
    (
        "X",
        "s42004-026-01909-8",
        "02",
        "IL-coated nanoparticles",
        "Complete 14-metric finding example: ten eligible, community-assigned neighbors in five communities with all ten community pairs observed; selected for explanatory coverage, not performance.",
    ),
]


def load_contrast(
    alias: str, paper_id: str, number: str, title: str, reason: str
) -> dict[str, Any]:
    root = STUDY / "papers" / paper_id
    sources: list[Path] = []

    def load(relative: str) -> Any:
        path = root / relative
        sources.append(path)
        return read(path)

    item = load("innovation_input.json")
    shared = load("shared/claims.json")
    claim = next(c for c in shared["claims"] if c["claim_id"].endswith("::" + number))
    traces = {}
    for branch in ["gear", "graph"]:
        path = root / branch / number / "evidence_trace.jsonl"
        sources.append(path)
        traces[branch] = records(path)
    fact = next(
        r["payload"] for r in reversed(traces["graph"]) if r["kind"] == "graph_fact"
    )
    card = load(f"gear/{number}/gear_card.json")
    assessments = {
        b: load(f"{b}/{number}/assessment.json")
        if (root / b / number / "assessment.json").exists()
        else None
        for b in ["gear", "graph"]
    }
    prior = None
    if alias == "X":
        relation = next(
            r
            for r in traces["gear"]
            if r["kind"] == "relation_card"
            and r["payload"]["relation_label"] == "PARTIAL_ANTECEDENT"
            and r["payload"]["prior_work_id"].endswith("W2030340194")
        )
        work_id = relation["payload"]["prior_work_id"]
        work = next(
            r
            for r in traces["gear"]
            if r["kind"] == "retrieved_work" and r["payload"]["work_id"] == work_id
        )
        prior = {"alias": "X1", "relation": relation, "work": work}
    return {
        "alias": alias,
        "title": title,
        "selection_reason": reason,
        "paper": item,
        "claim": claim,
        "fact": fact,
        "gear_card": card,
        "assessments": assessments,
        "prior": prior,
        "missing_assessments": [b for b, value in assessments.items() if value is None],
        "metrics": {m["name"]: m["value"] for m in fact["metrics"]},
        "records": traces,
        "sources": [file_source(p) for p in sources],
    }


def add_contrasts(out: Path) -> None:
    data = read(out / "data/snapshot.json")
    cases = [load_contrast(*row) for row in SELECTION]
    data["contrasts"] = cases
    for case in cases:
        data["aliases"][case["claim"]["claim_id"]] = case["alias"]
        data["sources"].extend(case["sources"])
    data["sources"] = list({s["path"]: s for s in data["sources"]}.values())
    # Preserve the main paper's existing color identity and add colors for X.
    ids = sorted(
        {
            n["community_id"]
            for n in data["historical"].values()
            if n["community_id"] is not None
        }
    )
    additional = {
        n["community_id"]
        for c in cases
        for n in c["fact"]["neighbors"]
        if n["community_id"] is not None
    }
    data["display_community_order"] = ids + sorted(additional - set(ids))
    coverage = next(
        r["payload"]
        for r in reversed(data["gear_records"])
        if r["kind"] == "retrieval_coverage"
    )
    relations = [r for r in data["gear_records"] if r["kind"] == "relation_card"]
    data["status_audit"] = {
        "coverage_sufficient": coverage["coverage_sufficient"],
        "unresolved_relations": [
            r["evidence_id"]
            for r in relations
            if r["payload"]["relation_label"] == "UNRESOLVED"
        ],
        "card_status": data["gear_card"]["status"],
        "interpretation": "Coverage sufficiency is a retrieval condition. An unresolved relation prevents bounded_no_antecedent under the current _card rule. These fields can coexist without a record conflict.",
        "residual_field": "The current _card populates residual_contribution from PARTIAL_ANTECEDENT differences only; a null field does not erase a separately scoped assessment finding.",
        "code_source": file_source(ROOT / "gear/evidence_supervisor.py"),
    }
    data["notes"] = [n for n in data["notes"] if "shown as a schema" not in n]
    data["notes"] += [
        "One primary paper and one contrast selected for complete displayed metrics; not a representative performance sample.",
        "Panel e contrasts saved GEAR and Graph assessments. The generic synthesis method is explicitly not a saved per-claim fused output.",
        "Auxiliary cutoff dates differ; all examples use the same saved insertion policy and the dated historical corpus. No between-paper performance inference is made.",
    ]
    data["notes"] = list(dict.fromkeys(data["notes"]))
    prepare_profile(data, out)
    write(out / "data/snapshot.json", data)
    write(out / "data/source_manifest.json", data["sources"])
    write(out / "data/aliases.json", data["aliases"])
    write(out / "qa/status_audit.json", data["status_audit"])
    write(
        out / "data/contrast_selection.json",
        [
            {
                "alias": c["alias"],
                "paper_id": c["paper"]["paper_id"],
                "claim_id": c["claim"]["claim_id"],
                "reason": c["selection_reason"],
                "cutoff": c["paper"]["cutoff_date"],
                "policy": c["fact"]["insertion_policy"],
            }
            for c in cases
        ],
    )
    layouts = read(out / "layouts/networks.json")
    for case in cases:
        f = case["fact"]
        target = case["claim"]["claim_id"]
        nodes = [n["claim_id"] for n in f["neighbors"]] + [target]
        edges = f["neighbor_edges"] + [[target, n] for n in nodes[:-1]]
        layouts[case["alias"]] = network_layout(nodes, edges, 731)
    write(out / "layouts/networks.json", layouts)
    print(
        "Added complete-metric nanoparticle example X; audited main GEAR status.",
        flush=True,
    )

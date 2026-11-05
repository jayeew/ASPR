"""Audit the case's temporal graph boundary and source-bound output artifacts."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from gear.innovation.validation import validate_run
from gear.trace import EvidenceStore

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/fig10_reference"


def main() -> None:
    run = OUT / "analysis"
    result = validate_run(run)
    errors = list(result["errors"])
    db = sqlite3.connect(
        f'file:{OUT / "snapshot/claim_graph_index.sqlite"}?mode=ro', uri=True
    )
    minimum, maximum, count = db.execute(
        "SELECT MIN(publication_date),MAX(publication_date),COUNT(*) FROM claim_nodes"
    ).fetchone()
    if maximum >= "2024-05-08":
        errors.append("Future snapshot claims")
    db.close()
    edges = 0
    for directory in sorted((run / "graph").glob("[0-9][0-9]")):
        for record in EvidenceStore(directory)._evidence.values():
            if record.kind != "graph_fact":
                continue
            fact = record.payload
            if len(fact["neighbors"]) > 10:
                errors.append("Too many neighbors")
            for n in fact["neighbors"]:
                edges += 1
                if (
                    n["publication_date"] >= "2024-05-08"
                    or n["cosine_similarity"] <= 0.5
                ):
                    errors.append("Ineligible neighbor")
                if n["parent_paper_id"] == "s41586-024-07487-w":
                    errors.append("Target used as prior art")
    joint = json.loads((run / "graph/joint/facts.json").read_text())
    targets = {
        c["claim_id"]
        for c in json.loads((run / "shared/claims.json").read_text())["claims"]
    }
    if any(a in targets and b in targets for a, b in joint["insertion_edges"]):
        errors.append("Invented target edge")
    execution = {
        p.parent.name: json.loads(p.read_text())
        for p in (run / "gear").glob("*/execution.json")
    }
    result.update(
        {
            "valid": not errors,
            "errors": errors,
            "snapshot_min_date": minimum,
            "snapshot_max_date": maximum,
            "snapshot_claim_count": count,
            "insertion_edge_count": edges,
            "gear_execution": execution,
            "scientific_correctness": "Not established by schema and evidence-key checks",
        }
    )
    (OUT / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise ValueError(errors)
    files = [
        "Fig10.svg",
        "Fig10_outlined.svg",
        "Fig10_outlined.pdf",
        "Fig10_outlined.png",
        "innovation_report.md",
        "case_report.md",
        "case_methods.md",
        "report.json",
        "figure_source_data.json",
        "audit.json",
    ]
    manifest = {
        name: hashlib.sha256((OUT / name).read_bytes()).hexdigest() for name in files
    }
    (OUT / "delivery_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

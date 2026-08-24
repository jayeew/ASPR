from __future__ import annotations

from gear.grounding import GroundingWorkflow
from gear.trace import EvidenceStore


def test_grounding_workflow_writes_traceable_cards(tmp_path, paper_ir) -> None:
    store = EvidenceStore(tmp_path)
    report = GroundingWorkflow().run(paper_ir, store)

    assert report.contributions
    assert report.stage_hashes
    assert store.has("GW:REPORT")
    assert all(
        store.has(f"GW:C:{card.contribution_id}") for card in report.contributions
    )
    assert all(
        point.paper_evidence_keys[0].startswith("P:") for point in report.findings
    )

from datetime import date
from pathlib import Path

import pytest

from gear.artifacts import read_model
from gear.config import GearConfig
from gear.contracts import PaperIR
from gear.innovation import pipeline
from gear.innovation.contracts import Assessment, ClaimSet, Finding
from gear.innovation.validation import validate_run
from gear.review_contracts import (
    GearClaimCard,
    GearEvidenceStatus,
    GraphFactCard,
    GraphNeighbor,
    InnovationPaperInput,
)


@pytest.mark.parametrize("claim_count", [1, 2])
def test_complete_pipeline_shares_identity_and_resolves_sources(
    tmp_path, monkeypatch, claim_count
):
    fixture = Path("outputs/nature_2026_random3/s41467-026-68293-8")
    if not fixture.exists():
        pytest.skip("Local saved manuscript fixture unavailable")
    paper = read_model(fixture / "gear" / "paper_ir.json", PaperIR)
    from gear.review_contracts import GearBranchResult

    old = read_model(fixture / "gear" / "gear_branch_result.json", GearBranchResult)
    item = read_model(fixture / "innovation_input.json", InnovationPaperInput)
    shared = ClaimSet(
        paper_id=item.paper_id,
        input_fingerprint="test",
        claims=old.claims[:claim_count],
    )
    from gear.artifacts import write_model

    write_model(tmp_path / "shared" / "claims.json", shared)
    monkeypatch.setattr(pipeline, "prepare_shared", lambda *a: (paper, shared))
    monkeypatch.setattr(
        pipeline, "run_joint", lambda *a: tmp_path / "graph" / "joint" / "analysis.json"
    )

    def insert(self, claim, item):
        neighbor = GraphNeighbor(
            claim_id="prior",
            parent_paper_id="prior-paper",
            claim_type=claim.claim_type,
            claim_text="historical basis",
            publication_date=date(2025, 1, 1),
            cosine_similarity=0.8,
            semantic_rank=1,
            community_id=1,
        )
        return GraphFactCard(claim=claim, neighbors=[neighbor], metrics=[])

    monkeypatch.setattr(pipeline.ClaimGraphRuntime, "insert", insert)
    monkeypatch.setattr(
        pipeline.EvidenceSupervisor,
        "evaluate",
        lambda self, c, *a: GearClaimCard(
            claim=c, status=GearEvidenceStatus.INCONCLUSIVE, summary="limited"
        ),
    )
    seen = []

    def assess(config, claim_id, text, sources, mode):
        seen.append((mode, claim_id, text))
        return Assessment(
            claim_id=claim_id,
            claim_text=text,
            supported_scope=text,
            findings=[
                Finding(
                    dimension="increment",
                    text="bounded",
                    evidence_keys=[next(iter(sources))],
                )
            ],
            overall_stance="unresolved",
            overall_reason="limited",
            limitations=["test"],
        )

    monkeypatch.setattr(pipeline, "assess", assess)
    paths = pipeline.run(item, tmp_path, GearConfig(), "all", Path("."), Path("."))
    assert set(paths) == {"shared", "gear", "graph", "graph_joint", "fusion"}
    assert len({row[1:] for row in seen}) == claim_count
    assert validate_run(tmp_path)["valid"]
    # A rerun must not call the model or replace evidence.
    pipeline.run(item, tmp_path, GearConfig(), "all", Path("."), Path("."))
    assert len(seen) == 3 * claim_count

    # A repaired branch must invalidate fusion but preserve the old source records.
    from gear.innovation.contracts import AnalysisResult

    branch_path = tmp_path / "gear" / "analysis.json"
    branch = read_model(branch_path, AnalysisResult)
    branch.assessments[0].overall_reason = "repaired interpretation"
    write_model(branch_path, branch)
    pipeline.fuse(item, tmp_path, GearConfig(), shared)
    assert len(seen) == 4 * claim_count
    archives = list((tmp_path / "fusion_attempts").iterdir())
    assert len(archives) == 1
    assert (archives[0] / "analysis.json").exists()
    assert validate_run(tmp_path)["valid"]

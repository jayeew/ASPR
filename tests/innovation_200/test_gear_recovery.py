from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from experiments.innovation_200 import run_gear
from experiments.innovation_200.common import write_json
from experiments.innovation_200.recovery import (
    clean_limited,
    execution_summary,
    study_gear_lock,
)
from gear.artifacts import write_model
from gear.innovation.contracts import AnalysisResult, Assessment, ClaimSet, Finding
from gear.review_contracts import GearClaim, GearClaimCard
from gear.trace import EvidenceStore


def _claim(paper_id: str, suffix: str) -> GearClaim:
    return GearClaim(
        claim_id=f"{paper_id}::CLAIM::{suffix}",
        claim_type="FINDING",
        author_claim_text="test",
        normalized_claim_text="test",
        source_span_ids=[],
        support_span_ids=[],
        internal_support="supported",
        narrowing_reason="",
    )


def _saved_claim(root: Path, claim: GearClaim, limited: bool = False) -> Assessment:
    directory = root / "gear" / claim.claim_id.rsplit("::", 1)[-1]
    store = EvidenceStore(directory)
    store.add_evidence("P:1", "manuscript_span", {"text": "test"})
    card = GearClaimCard(
        claim=claim, status="residual_extension", summary="test", evidence_keys=["P:1"]
    )
    store.add_evidence(f"GEAR:{claim.claim_id}", "gear_claim_card", card)
    write_model(directory / "gear_card.json", card)
    result = Assessment(
        claim_id=claim.claim_id,
        claim_text="test",
        supported_scope="test",
        findings=[Finding(dimension="increment", text="test", evidence_keys=["P:1"])],
        overall_stance="recognized",
        overall_reason="test",
        limitations=["abstract only"] if limited else [],
    )
    write_model(directory / "assessment.json", result)
    return result


def _paper(study: Path, paper_id: str, limited: bool = True) -> Path:
    root = study / "papers" / paper_id
    claims = [_claim(paper_id, "01"), _claim(paper_id, "02")]
    write_model(
        root / "shared/claims.json",
        ClaimSet(paper_id=paper_id, input_fingerprint="", claims=claims),
    )
    assessments = [
        _saved_claim(root, claims[0]),
        _saved_claim(root, claims[1], limited),
    ]
    write_model(
        root / "gear/analysis.json",
        AnalysisResult(
            paper_id=paper_id,
            system="gear",
            claim_fingerprint="",
            status="limited" if limited else "complete",
            assessments=assessments,
        ),
    )
    return root


def test_clean_preserves_success_and_moves_limited_evidence_and_dependencies(
    tmp_path: Path,
) -> None:
    root = _paper(tmp_path, "paper")
    success = (root / "gear/01/evidence_trace.jsonl").read_bytes()
    failed = (root / "gear/02/evidence_trace.jsonl").read_bytes()
    for relative in (
        "reports/gear/paper.json",
        "reports/fusion/paper.md",
        "human_evaluation/fusion/paper.json",
        "pairwise/paper__fusion_vs_graph.json",
        "reports/direct_llm/paper.json",
        "reports/graph/paper.json",
        "human_evaluation/graph/paper.json",
        "papers/paper/graph/analysis.json",
        "papers/paper/fusion/analysis.json",
    ):
        write_json(tmp_path / relative, {"saved": True})
    write_json(
        tmp_path / "status/run_gear.json", [{"paper_id": "paper", "status": "complete"}]
    )
    write_json(
        tmp_path / "status/generate_reports.json",
        [
            {"paper_id": "paper__gear", "status": "complete"},
            {"paper_id": "paper__graph", "status": "complete"},
        ],
    )
    with study_gear_lock(tmp_path):
        result = clean_limited(tmp_path, ["paper"])
    archive = Path(result["archive"])
    assert result["papers"][0]["kept_claims"] == ["paper::CLAIM::01"]
    assert (root / "gear/01/evidence_trace.jsonl").read_bytes() == success
    assert not (root / "gear/02").exists()
    assert not (root / "gear/analysis.json").exists()
    assert (
        archive / "artifacts/papers/paper/gear/02/evidence_trace.jsonl"
    ).read_bytes() == failed
    assert not (root / "fusion").exists()
    assert not (tmp_path / "reports/gear/paper.json").exists()
    assert not (tmp_path / "pairwise/paper__fusion_vs_graph.json").exists()
    assert (tmp_path / "reports/direct_llm/paper.json").exists()
    assert (tmp_path / "reports/graph/paper.json").exists()
    assert (tmp_path / "human_evaluation/graph/paper.json").exists()
    assert (root / "graph/analysis.json").exists()
    statuses = json.loads((tmp_path / "status/generate_reports.json").read_text())
    assert [row["status"] for row in statuses] == ["pending", "complete"]


def test_complete_paper_and_repeat_cleanup_are_preserved(tmp_path: Path) -> None:
    complete = _paper(tmp_path, "complete", limited=False)
    limited = _paper(tmp_path, "limited")
    before = {str(p): p.read_bytes() for p in complete.rglob("*") if p.is_file()}
    with study_gear_lock(tmp_path):
        first = clean_limited(tmp_path, ["complete", "limited"])
        second = clean_limited(tmp_path, ["complete", "limited"])
    assert first["papers"][0]["action"] == "kept_complete_paper"
    assert second["moved"] == []
    assert all(Path(path).read_bytes() == content for path, content in before.items())
    assert (limited / "gear/01/assessment.json").exists()


@pytest.mark.parametrize(
    "problem", ["identity", "trace", "execution", "missing", "branch_error"]
)
def test_cleanup_rejects_invalid_or_failed_claim_even_without_limitations(
    tmp_path: Path, problem: str
) -> None:
    root = _paper(tmp_path, "paper")
    directory = root / "gear/01"
    if problem == "identity":
        path = directory / "assessment.json"
        row = json.loads(path.read_text())
        row["claim_id"] = "wrong"
        write_json(path, row)
    elif problem == "trace":
        path = directory / "evidence_trace.jsonl"
        path.write_text(path.read_text().replace('"text":"test"', '"text":"tampered"'))
        # Formatting of EvidenceStore is stable JSON, but corrupt hashes explicitly too.
        path.write_text(path.read_text().replace("sha256:", "invalid:"))
    elif problem == "execution":
        write_json(
            directory / "execution.json", {"status": "failed", "errors": ["retrieval"]}
        )
    elif problem == "missing":
        (directory / "assessment.json").unlink()
    else:
        path = root / "gear/analysis.json"
        row = json.loads(path.read_text())
        row["limitations"] = ["paper::CLAIM::01:RuntimeError:retrieval"]
        write_json(path, row)
    with study_gear_lock(tmp_path):
        result = clean_limited(tmp_path, ["paper"])
    assert not directory.exists()
    assert result["papers"][0]["kept_claims"] == []


def test_study_lock_rejects_second_writer_and_releases(tmp_path: Path) -> None:
    with (
        study_gear_lock(tmp_path),
        pytest.raises(RuntimeError, match="Another GEAR"),
        study_gear_lock(tmp_path),
    ):
        pytest.fail("second writer entered")
    with study_gear_lock(tmp_path):
        pass


def test_execution_status_distinguishes_scientific_limits_and_service_errors(
    tmp_path: Path,
) -> None:
    root = _paper(tmp_path, "paper")
    result = AnalysisResult.model_validate_json(
        (root / "gear/analysis.json").read_text()
    )
    assert execution_summary(root, result, 2)["execution_status"] == "legacy_unknown"
    for suffix in ("01", "02"):
        write_json(
            root / "gear" / suffix / "execution.json",
            {"status": "complete", "errors": []},
        )
    summary = execution_summary(root, result, 2)
    assert summary["execution_status"] == "completed"
    assert summary["scientific_limited"] is True
    write_json(root / "gear/01/execution.json", {"status": "failed", "errors": ["429"]})
    assert (
        execution_summary(root, result, 2)["execution_status"]
        == "completed_with_errors"
    )


def test_selected_run_has_separate_status_and_preserves_study_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_json(
        tmp_path / "status/run_gear.json",
        [{"paper_id": "existing", "status": "complete"}],
    )
    original = (tmp_path / "status/run_gear.json").read_bytes()
    captured: dict[str, object] = {}

    def fake_stage(*args: object, **kwargs: object) -> list[dict]:
        captured.update(kwargs)
        return []

    monkeypatch.setattr(run_gear, "run_stage", fake_stage)
    monkeypatch.setattr(run_gear, "configure_limits", lambda *args: None)
    args = run_gear._parser().parse_args(
        ["--study", str(tmp_path), "--paper-id", "paper"]
    )
    assert run_gear._run(args, [{"paper_id": "paper"}]) == 0
    assert Path(str(captured["status_path"])).name.startswith("run_gear_selected_")
    assert (tmp_path / "status/run_gear.json").read_bytes() == original


def test_normal_resume_skips_limited_without_model_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _paper(tmp_path, "paper")

    class Item:
        paper_id = "paper"

    monkeypatch.setattr(run_gear, "paper_input", lambda _: Item())
    monkeypatch.setattr(
        run_gear, "run_branch", lambda *args: pytest.fail("should reuse saved paper")
    )
    result = run_gear.run_one(
        {}, tmp_path, tmp_path, tmp_path, False, logging.getLogger("test")
    )
    assert result["skipped"] is True
    assert result["branch_status"] == "limited"
    assert (root / "gear/02/assessment.json").exists()


def test_retry_failed_preserves_scientific_limits_and_only_archives_execution_failure(
    tmp_path: Path,
) -> None:
    from experiments.innovation_200.recovery import retry_failed

    root = _paper(tmp_path, "paper")
    before = (root / "gear/02/assessment.json").read_bytes()
    write_json(
        root / "gear/01/execution.json",
        {"status": "failed", "errors": ["OpenAlex 429"]},
    )
    write_json(root / "gear/02/execution.json", {"status": "complete", "errors": []})
    with study_gear_lock(tmp_path):
        record = retry_failed(tmp_path, ["paper"])
    assert not (root / "gear/01").exists()
    assert (root / "gear/02/assessment.json").read_bytes() == before
    assert not (root / "gear/analysis.json").exists()
    assert record["papers"][0]["kept_claims"] == ["paper::CLAIM::02"]
    assert record["papers"][0]["retry_claims"][0]["reason"] == "execution_failure"


def test_retry_failed_keeps_healthy_card_when_only_summary_failed(
    tmp_path: Path,
) -> None:
    from experiments.innovation_200.recovery import retry_failed

    root = _paper(tmp_path, "paper")
    (root / "gear/01/assessment.json").unlink()
    write_json(
        root / "gear/01/execution.json", {"status": "evidence_complete", "errors": []}
    )
    row = json.loads((root / "gear/analysis.json").read_text())
    row["limitations"] = ["paper::CLAIM::01:RuntimeError:summary model unavailable"]
    write_json(root / "gear/analysis.json", row)
    card = (root / "gear/01/gear_card.json").read_bytes()
    trace = (root / "gear/01/evidence_trace.jsonl").read_bytes()
    with study_gear_lock(tmp_path):
        result = retry_failed(tmp_path, ["paper"])
    assert (root / "gear/01/gear_card.json").read_bytes() == card
    assert (root / "gear/01/evidence_trace.jsonl").read_bytes() == trace
    assert result["papers"][0]["retry_claims"][0]["action"] == "reassess_saved_card"
    assert not (root / "gear/analysis.json").exists()


def test_retry_failed_noop_for_healthy_limited_paper_including_legacy(
    tmp_path: Path,
) -> None:
    from experiments.innovation_200.recovery import retry_failed

    root = _paper(tmp_path, "paper")
    before = {
        str(path): path.read_bytes()
        for path in (root / "gear").rglob("*")
        if path.is_file()
    }
    with study_gear_lock(tmp_path):
        result = retry_failed(tmp_path, ["paper"])
    assert result["moved"] == []
    assert result["papers"][0]["action"] == "kept_existing_paper"
    assert all(Path(path).read_bytes() == data for path, data in before.items())


def test_retry_failed_archives_interrupted_claim_without_card(tmp_path: Path) -> None:
    from experiments.innovation_200.recovery import retry_failed

    root = _paper(tmp_path, "paper")
    (root / "gear/01/gear_card.json").unlink()
    (root / "gear/01/assessment.json").unlink()
    original = (root / "gear/01/evidence_trace.jsonl").read_bytes()
    with study_gear_lock(tmp_path):
        result = retry_failed(tmp_path, ["paper"])
    assert not (root / "gear/01").exists()
    archived = (
        Path(result["archive"]) / "artifacts/papers/paper/gear/01/evidence_trace.jsonl"
    )
    assert archived.read_bytes() == original
    assert (root / "gear/02/assessment.json").exists()


def test_pinned_roster_survives_study_extension(tmp_path: Path) -> None:
    (tmp_path / "papers.jsonl").write_text(
        "\n".join(
            json.dumps({"paper_id": p}) for p in ["original_a", "original_b", "added"]
        )
    )
    pinned = tmp_path / "gear_papers.jsonl"
    pinned.write_text(
        "\n".join(json.dumps({"paper_id": p}) for p in ["original_a", "original_b"])
    )
    assert [r["paper_id"] for r in run_gear._selected_rows(tmp_path, None)] == [
        "original_a",
        "original_b",
    ]
    assert [
        r["paper_id"] for r in run_gear._selected_rows(tmp_path, ["original_b"])
    ] == ["original_b"]
    with pytest.raises(ValueError, match="absent"):
        run_gear._selected_rows(tmp_path, ["added"])
    pinned.write_text(json.dumps({"paper_id": "unknown"}))
    with pytest.raises(ValueError, match="absent"):
        run_gear._selected_rows(tmp_path, None)


def test_coverage_repair_archives_and_reuses_healthy_evidence(tmp_path: Path) -> None:
    from datetime import date

    from experiments.innovation_200.recovery import repair_coverage
    from gear.config import GearConfig
    from gear.contracts import RetrievalBudget, RetrievalCoverageCard
    from gear.evidence_supervisor import EvidenceSupervisor
    from gear.innovation.gear_resume import restore_evidence

    root = _paper(tmp_path, "paper")
    directory = root / "gear/01"
    claim = _claim("paper", "01")
    coverage = RetrievalCoverageCard(
        coverage_id="cov",
        target_claim_id=claim.claim_id,
        cutoff_date=date(2026, 1, 1),
        required_query_roles=[
            "author_terminology",
            "object_problem",
            "mechanism_outcome",
            "purpose_semantic",
            "legacy_contrastive",
        ],
        completed_query_roles=[
            "author_terminology",
            "object_problem",
            "mechanism_outcome",
            "purpose_semantic",
        ],
        unique_eligible_count=42,
        compared_work_ids=["selected_without_a_relation"],
        whole_paper_ranking_completed=True,
        purpose_ranking_completed=True,
    )
    EvidenceStore(directory).add_evidence(
        f"COVERAGE:{claim.claim_id}", "retrieval_coverage", coverage
    )
    before = (directory / "evidence_trace.jsonl").read_bytes()
    healthy_other = (root / "gear/02/assessment.json").read_bytes()
    manifest = repair_coverage(tmp_path, ["paper"])
    marker = json.loads((directory / "recovery_source.json").read_text())
    source = Path(marker["source"])
    assert (source / "evidence_trace.jsonl").read_bytes() == before
    assert (root / "gear/02/assessment.json").read_bytes() == healthy_other
    assert not (root / "gear/analysis.json").exists()
    assert len(manifest["papers"][0]["retry_claims"]) == 1
    supervisor = EvidenceSupervisor(GearConfig(), EvidenceStore(directory))
    assert restore_evidence(
        supervisor, source, claim, date(2026, 1, 1), {}, {}, RetrievalBudget()
    ) == (True, False)
    assert supervisor.store.has("P:1")
    assert not supervisor.store.has(f"GEAR:{claim.claim_id}")
    assert not supervisor.store.has(f"COVERAGE:{claim.claim_id}")
    card = supervisor.prior_art.coverage_card(
        claim.claim_id,
        date(2026, 1, 1),
        require_contrastive=True,
        direct_or_partial_found=False,
    )
    assert card.unique_eligible_count == 42
    assert card.compared_work_ids == []
    assert not card.coverage_sufficient
    assert not repair_coverage(tmp_path, ["paper"])["moved"]
    with pytest.raises(ValueError, match="cutoff"):
        restore_evidence(
            supervisor, source, claim, date(2025, 1, 1), {}, {}, RetrievalBudget()
        )


def test_supplement_only_allowlisted_claim_and_replay_is_safe(tmp_path: Path) -> None:
    from datetime import date

    from experiments.innovation_200.recovery import supplement_claims
    from gear.contracts import QuerySpec, RetrievalCoverageCard

    root = _paper(tmp_path, "paper")
    claim = _claim("paper", "01")
    directory = root / "gear/01"
    EvidenceStore(directory).add_evidence(
        f"COVERAGE:{claim.claim_id}",
        "retrieval_coverage",
        RetrievalCoverageCard(
            coverage_id="cov",
            target_claim_id=claim.claim_id,
            cutoff_date=date(2026, 1, 1),
        ),
    )
    before = (directory / "evidence_trace.jsonl").read_bytes()
    other = (root / "gear/02/assessment.json").read_bytes()
    plan = [
        {
            "claim_id": claim.claim_id,
            "queries": [
                QuerySpec(
                    query_id="supplement",
                    claim_id=claim.claim_id,
                    family="lexical",
                    query="grounded broader terms",
                ).model_dump(mode="json")
            ],
        }
    ]
    # A bad later entry must fail before the earlier valid entry moves.
    with pytest.raises(ValueError, match="Duplicate"):
        supplement_claims(tmp_path, plan + plan)
    assert (directory / "evidence_trace.jsonl").read_bytes() == before
    # A completed earlier recovery may receive a new targeted supplement.
    write_json(directory / "recovery_source.json", {"source": "older-archive"})
    result = supplement_claims(tmp_path, plan)
    marker = json.loads((directory / "recovery_source.json").read_text())
    assert (Path(marker["source"]) / "evidence_trace.jsonl").read_bytes() == before
    assert (root / "gear/02/assessment.json").read_bytes() == other
    assert result["claims"] == [claim.claim_id]
    assert not (root / "gear/analysis.json").exists()
    assert not supplement_claims(tmp_path, plan)["moved"]

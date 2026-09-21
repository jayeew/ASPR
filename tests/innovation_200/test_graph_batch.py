from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from gear.artifacts import write_model
from gear.claim_attribution import ClaimGraphRuntime
from gear.config import GearConfig
from gear.innovation import graph_batch
from gear.innovation.contracts import Assessment, ClaimSet
from gear.innovation.joint_graph import analyze_prepared_joint
from gear.review_contracts import (
    GearClaim,
    GraphClaim,
    GraphFactCard,
    GraphNeighbor,
    InnovationPaperInput,
)
from gear.trace import EvidenceStore


def inputs(
    tmp_path: Path, paper_id: str
) -> tuple[InnovationPaperInput, Path, ClaimSet, GraphFactCard]:
    root = tmp_path / paper_id
    claim = GearClaim(
        claim_id=f"{paper_id}::c1",
        claim_type="FINDING",
        author_claim_text="test",
        normalized_claim_text="test",
        source_span_ids=[],
        support_span_ids=[],
        internal_support="supported",
        narrowing_reason="",
    )
    shared = ClaimSet(paper_id=paper_id, input_fingerprint="", claims=[claim])
    write_model(root / "shared/claims.json", shared)
    item = InnovationPaperInput(
        paper_id=paper_id,
        title="test",
        paper_path=Path("none"),
        publication_date=date(2026, 1, 1),
        cutoff_date=date(2026, 1, 1),
        abstract_text="test",
        abstract_source="test",
    )
    card = GraphFactCard(
        insertion_policy="threshold_parent_path:k=10:cosine>0.5",
        claim=GraphClaim(
            claim_id=claim.claim_id,
            paper_id=paper_id,
            claim_type="FINDING",
            claim_text="test",
            source_sentence_ids=[],
            source_sentence_texts=[],
        ),
        neighbors=[
            GraphNeighbor(
                claim_id="old",
                parent_paper_id="historical",
                claim_type="FINDING",
                claim_text="old",
                publication_date=date(2025, 1, 1),
                cosine_similarity=0.8,
                semantic_rank=1,
            )
        ],
        metrics=[],
    )
    return item, root, shared, card


def test_embedding_batch_reuses_encoder_and_preserves_order(tmp_path: Path) -> None:
    class Encoder:
        def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
            assert kwargs["batch_size"] == 2
            return np.array([[len(t), 1] for t in texts])

    runtime = ClaimGraphRuntime(tmp_path, tmp_path)
    runtime._model = Encoder()
    assert runtime.encode_batch(["a", "bbb"], 2).tolist() == [[1, 1], [3, 1]]
    assert runtime.encode_batch([], 2).shape == (0, 0)
    with pytest.raises(ValueError):
        runtime.encode_batch(["x"], 0)


def test_prepare_batches_across_papers_and_closes_before_joint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = [inputs(tmp_path, p) for p in ("a", "b", "c")]
    events = []

    class Runtime:
        def __init__(self, *args: object) -> None:
            pass

        def encode_batch(self, texts: list[str], batch_size: int) -> np.ndarray:
            events.append(("encode", len(texts)))
            return np.ones((len(texts), 2))

        def insert_vector(
            self, claim: GraphClaim, item: InnovationPaperInput, vector: np.ndarray
        ) -> GraphFactCard:
            return next(card for _, _, _, card in data if card.claim == claim)

        def close(self) -> None:
            events.append(("close", 0))

    def pending(item: InnovationPaperInput, root: Path, *args: object) -> list:
        card = next(
            card for _, _, _, card in data if card.claim.paper_id == item.paper_id
        )
        return [(card.claim, item, root / "graph/c1")]

    def joint(*args: object, **kwargs: object) -> None:
        assert events[-1] == ("close", 0)
        assert kwargs["prepare_only"] is True

    monkeypatch.setattr(graph_batch, "ClaimGraphRuntime", Runtime)
    monkeypatch.setattr(graph_batch, "pending_claims", pending)
    monkeypatch.setattr(graph_batch, "run_joint", joint)
    assert (
        graph_batch.prepare_graph(
            [(i, r) for i, r, _, _ in data], GearConfig(), tmp_path, tmp_path, 2
        )
        == []
    )
    assert events == [("encode", 2), ("encode", 1), ("close", 0)]


def test_analysis_resumes_without_encoder_and_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    item, root, shared, card = inputs(tmp_path, "a")
    task = graph_batch.model_tasks([(item, root)])[0]
    calls = []

    def assess(*args: object) -> Assessment:
        calls.append(1)
        return Assessment(
            claim_id=shared.claims[0].claim_id,
            claim_text="test",
            supported_scope="test",
            findings=[
                {
                    "dimension": "historical_basis",
                    "text": "test",
                    "evidence_keys": [f"GRAPH:{shared.claims[0].claim_id}"],
                }
            ],
            overall_stance="unresolved",
            overall_reason="limited",
            limitations=["test"],
        )

    monkeypatch.setattr(graph_batch, "assess", assess)
    with pytest.raises(KeyError):
        graph_batch.analyze_task(task, GearConfig())
    assert not calls
    EvidenceStore(root / "graph/c1").add_evidence(
        f"GRAPH:{card.claim.claim_id}", "graph_fact", card
    )
    graph_batch.analyze_task(task, GearConfig())
    assert graph_batch.analyze_task(task, GearConfig())["skipped"]
    assert calls == [1]
    assert (
        graph_batch.finalize_graph(item, root, GearConfig()).status.value == "limited"
    )


def test_prepared_joint_rejects_missing_evidence_without_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, root, shared, _ = inputs(tmp_path, "a")
    from gear.artifacts import write_json

    write_json(root / "graph/joint/facts.json", {"input_claims": []})
    monkeypatch.setattr(
        "gear.innovation.joint_graph.analyze_joint",
        lambda *args: pytest.fail("Must not invoke a model"),
    )
    with pytest.raises(ValueError, match="evidence"):
        analyze_prepared_joint(GearConfig(), root, shared)


def test_joint_model_runs_from_saved_facts_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gear.artifacts import write_json
    from gear.innovation.joint_graph import JointAnalysis, joint_structure

    _, root, shared, card = inputs(tmp_path, "a")
    target = root / "graph/joint"
    fact = joint_structure([card], [])
    fact["missing_fact_claim_ids"] = []
    fact["input_claims"] = [{"claim_id": card.claim.claim_id, "claim_text": "test"}]
    store = EvidenceStore(target)
    store.add_evidence("JOINT_GRAPH:a", "joint_graph_fact", fact)
    store.add_evidence("GRAPH:a::c1", "graph_fact", card)
    write_json(target / "facts.json", fact)
    calls = []

    def analyze(*args: object) -> JointAnalysis:
        calls.append(1)
        return JointAnalysis(
            paper_id="a",
            knowledge_summary="test",
            findings=[
                {
                    "question": "test",
                    "claim_ids": ["a::c1"],
                    "observation": "test",
                    "interpretation": "test",
                    "evidence_keys": ["JOINT_GRAPH:a"],
                    "limitations": [],
                }
            ],
            limitations=[],
        )

    monkeypatch.setattr("gear.innovation.joint_graph.analyze_joint", analyze)
    monkeypatch.setattr(
        "gear.innovation.joint_graph.ClaimGraphRuntime",
        lambda *args: pytest.fail("Model phase must not open graph assets"),
    )
    assert (
        analyze_prepared_joint(GearConfig(), root, shared) == target / "analysis.json"
    )
    assert (
        analyze_prepared_joint(GearConfig(), root, shared) == target / "analysis.json"
    )
    assert calls == [1]
    assert (target / "report.md").is_file()
    changed = shared.model_copy(deep=True)
    changed.claims[0].normalized_claim_text = "changed"
    with pytest.raises(ValueError, match="changed"):
        analyze_prepared_joint(GearConfig(), root, changed)


def test_missing_neighbors_and_stale_assessment_stay_limited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    item, root, _shared, card = inputs(tmp_path, "a")
    card.neighbors = []
    EvidenceStore(root / "graph/c1").add_evidence("GRAPH:a::c1", "graph_fact", card)
    monkeypatch.setattr(
        graph_batch, "assess", lambda *args: pytest.fail("No neighbors")
    )
    task = graph_batch.model_tasks([(item, root)])[0]
    assert graph_batch.analyze_task(task, GearConfig())["status"] == "limited"
    result = graph_batch.finalize_graph(item, root, GearConfig())
    assert result.status.value == "limited"
    assert not result.assessments
    assert "No eligible" in result.limitations[0]


def test_encoder_rejects_graph_dimension_mismatch(tmp_path: Path) -> None:
    class Encoder:
        def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
            return np.ones((len(texts), 2))

    np.save(tmp_path / "claim_embedding_matrix.npy", np.ones((3, 4)))
    runtime = ClaimGraphRuntime(tmp_path, tmp_path)
    runtime._model = Encoder()
    with pytest.raises(ValueError, match="graph expects dimension 4"):
        runtime.encode_batch(["test"])

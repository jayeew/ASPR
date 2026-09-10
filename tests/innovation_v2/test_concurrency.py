from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import pytest

from gear.config import GearConfig
from gear.contracts import EvidenceSpan, RelationCard, RetrievedWork
from gear.evidence_supervisor import EvidenceSupervisor
from gear.local_ranking import LocalScientificRanker
from gear.review_contracts import GearClaim
from gear.trace import EvidenceStore


def test_supervisors_can_share_one_lazy_local_ranker(tmp_path: Path) -> None:
    ranker = LocalScientificRanker(Path("recall"), Path("reranker"))
    first = EvidenceSupervisor(
        GearConfig(), EvidenceStore(tmp_path / "first"), local_ranker=ranker
    )
    second = EvidenceSupervisor(
        GearConfig(), EvidenceStore(tmp_path / "second"), local_ranker=ranker
    )

    assert first.prior_art._local_ranker is ranker
    assert second.prior_art._local_ranker is ranker


def test_batched_relations_preserve_ordered_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batches: list[list[str]] = []

    class Classifier:
        def __init__(self, config: GearConfig) -> None:
            pass

        def classify_many(
            self, span: EvidenceSpan, works: list[RetrievedWork], **kwargs: object
        ) -> list[RelationCard]:
            batches.append([work.work_id for work in works])
            return [
                RelationCard(
                    relation_id="r" + work.work_id,
                    target_claim_id="c",
                    target_span_id="s",
                    prior_work_id=work.work_id,
                    relation_label="DISTANT",
                    evidence_level="abstract_evidence",
                    difference_dimensions=["different"],
                    retrieval_query_id="q",
                    temporal_valid=True,
                )
                for work in works
            ]

    monkeypatch.setattr("gear.evidence_supervisor.RelationClassifier", Classifier)
    span = EvidenceSpan(
        span_id="s",
        source_id="p",
        page=1,
        section_path=["Results"],
        char_start=0,
        char_end=1,
        text="x",
        text_sha256="sha256:" + hashlib.sha256(b"x").hexdigest(),
    )
    claim = GearClaim(
        claim_id="c",
        claim_type="FINDING",
        author_claim_text="x",
        normalized_claim_text="x",
        source_span_ids=["s"],
        support_span_ids=["s"],
        internal_support="supported",
        narrowing_reason="",
    )
    works = [
        RetrievedWork(
            work_id=f"w{i}",
            target_claim_id="c",
            title="t",
            retrieval_query_id="q",
            retrieval_source="test",
        )
        for i in range(12)
    ]
    store = EvidenceStore(tmp_path)
    supervisor = EvidenceSupervisor(GearConfig(), store)
    relations = {}
    actions = []
    supervisor._classify(span, claim, works, date(2026, 1, 1), relations, actions)
    assert batches == [[f"w{i}" for i in range(10)]]
    assert list(relations) == [f"w{i}" for i in range(10)]
    assert actions[0].output_ids == [f"RELATION:c:w{i}" for i in range(10)]


def test_repeated_identity_exclusion_preserves_changed_provenance(
    tmp_path: Path,
) -> None:
    claim = GearClaim(
        claim_id="c",
        claim_type="FINDING",
        author_claim_text="x",
        normalized_claim_text="x",
        source_span_ids=["s"],
        support_span_ids=["s"],
        internal_support="supported",
        narrowing_reason="",
    )
    work = RetrievedWork(
        work_id="w",
        target_claim_id="c",
        title="t",
        retrieval_query_id="q1",
        retrieval_source="test",
    )
    supervisor = EvidenceSupervisor(GearConfig(), EvidenceStore(tmp_path))
    supervisor._record_identity(claim, work, "same_identifier")
    supervisor._record_identity(claim, work, "same_identifier")
    supervisor._record_identity(
        claim, work.model_copy(update={"retrieval_query_id": "q2"}), "same_identifier"
    )
    records = EvidenceStore(tmp_path)._evidence
    assert len(records) == 2
    assert {r.payload["work"]["retrieval_query_id"] for r in records.values()} == {
        "q1",
        "q2",
    }

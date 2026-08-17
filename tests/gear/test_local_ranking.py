from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from gear.contracts import EvidenceLevel, RetrievedSpan, RetrievedWork
from gear.local_ranking import LocalScientificRanker


def _work(work_id: str, title: str) -> RetrievedWork:
    text = f"{title} abstract evidence."
    return RetrievedWork(
        work_id=work_id,
        target_claim_id="C-one",
        title=title,
        abstract=text,
        spans=[
            RetrievedSpan(
                span_id=f"RS-{work_id}",
                text=text,
                text_sha256="sha256:"
                + hashlib.sha256(text.encode("utf-8")).hexdigest(),
                source=EvidenceLevel.ABSTRACT,
            )
        ],
        retrieval_query_id="Q-one",
        retrieval_source="fake",
    )


def test_local_ranker_is_lazy_and_dual_view_union_is_ranked():
    ranker = LocalScientificRanker(Path("missing-recall"), Path("missing-reranker"))
    assert ranker._recall is None
    assert ranker._reranker is None

    class Recall:
        @staticmethod
        def encode(texts, return_dense=True):
            vectors = []
            for text in texts:
                lowered = text.casefold()
                vectors.append(
                    np.asarray(
                        [
                            float("mechanism" in lowered),
                            float("purpose" in lowered),
                        ]
                    )
                )
            return {"dense_vecs": np.asarray(vectors)}

    class Reranker:
        @staticmethod
        def compute_score(pairs, normalize=True):
            return [
                (
                    0.9
                    if any(term in document.casefold() for term in query.split())
                    else 0.1
                )
                for query, document in pairs
            ]

    ranker._recall = Recall()
    ranker._reranker = Reranker()
    works = [_work("W-a", "mechanism"), _work("W-b", "purpose")]
    ranked, scores = ranker.rank(
        works,
        whole_paper_view="mechanism",
        purpose_view="purpose",
        recall_limit=1,
        rerank_top_k=1,
        output_limit=2,
    )
    assert {work.work_id for work in ranked} == {"W-a", "W-b"}
    assert set(scores) == {"W-a", "W-b"}

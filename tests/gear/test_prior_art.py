from __future__ import annotations

import hashlib
from datetime import date

from gear.contracts import (
    EvidenceLevel,
    RelationLabel,
    RetrievalBudget,
    RetrievedSpan,
    RetrievedWork,
)
from gear.prior_art import PriorArtService, RelationClassifier


class SearchFake:
    def __init__(self):
        self.search_calls = 0
        self.neighbor_calls = 0

    def search_query(self, query, *, provider=None, date_to=None, limit=None):
        self.search_calls += 1
        rows = [
            {
                "paperId": "future",
                "title": "Future work",
                "publication_date": "2030-01-01",
                "year": 2030,
                "abstract": "Future evidence that must be filtered.",
            }
        ]
        rows.extend(
            {
                "paperId": f"W-{index}",
                "title": f"Prior evidence {index}",
                "publication_date": "2000-01-01",
                "year": 2000,
                "abstract": "A verifiable abstract about bounded evidence controllers.",
            }
            for index in range(20)
        )
        return rows[: int(limit or len(rows))]

    def fetch_neighbors(self, work_id, direction="references", *, limit=None):
        self.neighbor_calls += 1
        return []


def test_retrieval_budget_cache_and_cutoff_are_enforced(gear_config, paper_ir):
    config = gear_config.model_copy(update={"allow_external_retrieval": True})
    client = SearchFake()
    service = PriorArtService(config, search_client=client)
    budget = RetrievalBudget()
    works = service.retrieve(paper_ir.claims[0], date(2010, 1, 1), budget)
    assert len(works) <= 12
    assert all(work.work_id != "future" for work in works)
    assert budget.normal_used <= budget.normal_max
    assert budget.fulltext_kept <= budget.fulltext_max
    first_call_count = client.search_calls

    cached_budget = RetrievalBudget()
    cached = service.retrieve(paper_ir.claims[0], date(2010, 1, 1), cached_budget)
    assert cached
    assert service.last_cache_hit is True
    assert client.search_calls == first_call_count

    changed_cutoff_budget = RetrievalBudget()
    service.retrieve(
        paper_ir.claims[0],
        date(2011, 1, 1),
        changed_cutoff_budget,
    )
    assert client.search_calls > first_call_count

    service.retrieve(
        paper_ir.claims[0],
        date(2010, 1, 1),
        budget,
        family="contrastive",
    )
    assert budget.fulltext_kept <= 12
    assert budget.contrastive_used <= budget.contrastive_max


def test_same_year_unknown_date_cannot_be_antecedent(gear_config, paper_ir):
    claim = paper_ir.claims[0]
    span = paper_ir.span_map()[claim.span_id]
    prior_span = RetrievedSpan(
        span_id="RS-one",
        text="The prior abstract describes the same mechanism.",
        text_sha256="sha256:"
        + hashlib.sha256(
            b"The prior abstract describes the same mechanism."
        ).hexdigest(),
        source=EvidenceLevel.ABSTRACT,
    )
    prior = RetrievedWork(
        work_id="W-same-year",
        target_claim_id=claim.claim_id,
        title="Same-year work",
        publication_year=2010,
        spans=[prior_span],
        retrieval_query_id="QRY-one",
        retrieval_source="fake",
    )
    classifier = RelationClassifier(
        gear_config,
        generator=lambda system, user: {
            "relation_label": "DIRECT_ANTECEDENT",
            "difference_dimensions": [],
            "rationale": "same",
        },
    )
    card = classifier.classify(
        span,
        prior,
        target_claim_id=claim.claim_id,
        cutoff=date(2010, 6, 1),
    )
    assert card.relation_label == RelationLabel.PARALLEL
    assert card.temporal_valid is False
    assert card.temporal_order_unresolved is True


def test_missing_prior_text_is_unresolved(gear_config, paper_ir):
    claim = paper_ir.claims[0]
    span = paper_ir.span_map()[claim.span_id]
    prior = RetrievedWork(
        work_id="W-metadata",
        target_claim_id=claim.claim_id,
        title="Metadata only",
        publication_date=date(2000, 1, 1),
        retrieval_query_id="QRY-meta",
        retrieval_source="fake",
    )
    card = RelationClassifier(gear_config).classify(
        span,
        prior,
        target_claim_id=claim.claim_id,
        cutoff=date(2010, 1, 1),
    )
    assert card.relation_label == RelationLabel.UNRESOLVED
    assert card.evidence_level == EvidenceLevel.METADATA_ONLY


def test_relation_without_difference_dimensions_fails_closed(gear_config, paper_ir):
    claim = paper_ir.claims[0]
    span = paper_ir.span_map()[claim.span_id]
    text = "An earlier abstract describes a bounded evidence controller."
    prior = RetrievedWork(
        work_id="W-incomplete-relation",
        target_claim_id=claim.claim_id,
        title="Earlier work",
        publication_date=date(2000, 1, 1),
        spans=[
            RetrievedSpan(
                span_id="RS-incomplete",
                text=text,
                text_sha256="sha256:"
                + hashlib.sha256(text.encode("utf-8")).hexdigest(),
                source=EvidenceLevel.ABSTRACT,
            )
        ],
        retrieval_query_id="QRY-incomplete",
        retrieval_source="fake",
    )
    classifier = RelationClassifier(
        gear_config,
        generator=lambda system, user: {
            "relation_label": "DIRECT_ANTECEDENT",
            "difference_dimensions": [],
            "rationale": "incomplete",
        },
    )

    card = classifier.classify(
        span,
        prior,
        target_claim_id=claim.claim_id,
        cutoff=date(2010, 1, 1),
    )

    assert card.relation_label == RelationLabel.UNRESOLVED
    assert card.difference_dimensions == ["classifier_output_incomplete"]
    assert (
        classifier.last_failure == "relation_classifier_missing_difference_dimensions"
    )

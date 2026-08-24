from __future__ import annotations

import hashlib
import json
from datetime import date

from gear.contracts import (
    EvidenceLevel,
    RelationLabel,
    RetrievalBudget,
    RetrievedSpan,
    RetrievedWork,
    ScientificSearchFrame,
)
from gear.prior_art import PriorArtService, RelationClassifier


class SearchFake:
    def __init__(self):
        self.search_calls = 0
        self.neighbor_calls = 0

    def search_query(
        self,
        query,
        *,
        provider=None,
        date_to=None,
        limit=None,
        search_mode="text",
    ):
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


class MetadataOnlySearchFake:
    retrieval_provider = "openalex"

    def __init__(self):
        self.pdf_calls = 0

    def search_query(
        self,
        query,
        *,
        provider=None,
        date_to=None,
        limit=None,
        search_mode="text",
    ):
        return [
            {
                "paperId": "https://openalex.org/W1",
                "title": "Metadata-only prior work",
                "publication_date": "2000-01-01",
                "year": 2000,
                "abstract": "",
                "retrieval_source": "openalex",
            }
        ]

    def fetch_neighbors(self, work_id, direction="references", *, limit=None):
        return []

    def fetch_pdf_text(self, work_id, *, max_bytes, max_pages, max_characters):
        self.pdf_calls += 1
        return "Full-text evidence recovered from an OpenAlex Content PDF."


def scientific_frame(_system, user):
    request = json.loads(user)
    return {
        "target_object": ["evidence controller"],
        "task_problem": ["scientific evidence assessment"],
        "mechanism": ["claim-level evidence comparison"],
        "population_input": ["manuscripts"],
        "outcome_observable": ["trace accuracy"],
        "comparator": ["baseline assessment system"],
        "author_terms": ["evidence controller"],
        "brand_terms": [],
        "legacy_terms": ["evidence-grounded review"],
        "claimed_delta": "improves trace accuracy",
        "citation_seed_ids": [],
        "source_span_ids": [request["target_span_id"]],
    }


def comparable_candidates(_system, user):
    request = json.loads(user)
    return {
        "decisions": [
            {
                "work_id": item["work_id"],
                "verdict": "comparable",
                "matched_fields": ["target_object", "mechanism"],
                "score": 0.9,
                "reason": "The abstract describes the same object and mechanism.",
            }
            for item in request["candidates"]
        ]
    }


def service_with_model(config, client):
    return PriorArtService(
        config,
        search_client=client,
        query_generator=scientific_frame,
        rerank_generator=comparable_candidates,
    )


def test_retrieval_budget_cache_and_cutoff_are_enforced(gear_config, paper_ir):
    config = gear_config.model_copy(update={"allow_external_retrieval": True})
    client = SearchFake()
    service = service_with_model(config, client)
    budget = RetrievalBudget()
    claim = paper_ir.claims[0]
    works = service.retrieve(
        claim,
        date(2010, 1, 1),
        budget,
        target_span=paper_ir.span_map()[claim.span_id],
        paper_ir=paper_ir,
    )
    assert len(works) <= 12
    assert all(work.work_id != "future" for work in works)
    assert budget.normal_used <= budget.normal_max
    assert budget.fulltext_kept <= budget.fulltext_max
    assert {query.family for query in service.last_query_specs} == {
        "lexical",
        "semantic",
    }
    assert {query.query_role for query in service.last_query_specs} == {
        "author_terminology",
        "object_problem",
        "mechanism_outcome",
        "purpose_semantic",
    }
    lexical = next(
        query.query for query in service.last_query_specs if query.family == "lexical"
    )
    semantic = next(
        query.query for query in service.last_query_specs if query.family == "semantic"
    )
    assert " AND " not in lexical
    assert len(lexical.split()) <= 12
    assert len(semantic) <= 420
    assert all(
        hit.gate_label is None
        for hit in service.last_hits
        if hit.selection_stage != "compared"
    )
    first_call_count = client.search_calls

    cached_budget = RetrievalBudget()
    cached = service.retrieve(
        claim,
        date(2010, 1, 1),
        cached_budget,
        target_span=paper_ir.span_map()[claim.span_id],
        paper_ir=paper_ir,
    )
    assert cached
    assert service.last_cache_hit is True
    assert client.search_calls == first_call_count

    changed_cutoff_budget = RetrievalBudget()
    service.retrieve(
        claim,
        date(2011, 1, 1),
        changed_cutoff_budget,
        target_span=paper_ir.span_map()[claim.span_id],
        paper_ir=paper_ir,
    )
    assert client.search_calls > first_call_count

    service.retrieve(
        claim,
        date(2010, 1, 1),
        budget,
        family="contrastive",
        target_span=paper_ir.span_map()[claim.span_id],
        paper_ir=paper_ir,
    )
    assert budget.fulltext_kept <= 12
    assert budget.contrastive_used <= budget.contrastive_max


def test_redundant_contrastive_query_is_coverage_gap_not_service_failure(
    gear_config, paper_ir
):
    config = gear_config.model_copy(update={"allow_external_retrieval": True})

    def redundant_contrastive_frame(system, user):
        frame = dict(scientific_frame(system, user))
        frame["legacy_terms"] = list(frame["author_terms"])
        return frame

    service = PriorArtService(
        config,
        search_client=SearchFake(),
        query_generator=redundant_contrastive_frame,
        rerank_generator=comparable_candidates,
    )
    claim = paper_ir.claims[0]
    span = paper_ir.span_map()[claim.span_id]
    budget = RetrievalBudget()

    assert service.retrieve(
        claim,
        date(2010, 1, 1),
        budget,
        target_span=span,
        paper_ir=paper_ir,
    )
    works = service.retrieve(
        claim,
        date(2010, 1, 1),
        budget,
        family="contrastive",
        target_span=span,
        paper_ir=paper_ir,
    )

    assert works == []
    assert service.last_service_failed is False
    assert service.last_failures == []
    assert service.last_advisories == [
        "contrastive_query_coverage_gap:contrastive query did not change the search intent"
    ]
    coverage = service.coverage_card(
        claim.claim_id,
        date(2010, 1, 1),
        require_contrastive=True,
        direct_or_partial_found=False,
    )
    assert coverage.coverage_sufficient is False


def test_metadata_only_candidates_are_ignored_without_pdf_fallback(
    gear_config, paper_ir
):
    config = gear_config.model_copy(update={"allow_external_retrieval": True})
    client = MetadataOnlySearchFake()
    service = service_with_model(config, client)
    claim = paper_ir.claims[0]
    works = service.retrieve(
        claim,
        date(2010, 1, 1),
        RetrievalBudget(),
        target_span=paper_ir.span_map()[claim.span_id],
        paper_ir=paper_ir,
    )
    assert works == []
    assert client.pdf_calls == 0


def test_enabled_pdf_fallback_recovers_fulltext_span(gear_config, paper_ir):
    retrieval = gear_config.retrieval.model_copy(
        update={"openalex_pdf_enabled": True, "openalex_pdf_max_downloads": 1}
    )
    config = gear_config.model_copy(
        update={"allow_external_retrieval": True, "retrieval": retrieval}
    )
    client = MetadataOnlySearchFake()
    service = service_with_model(config, client)
    claim = paper_ir.claims[0]
    works = service.retrieve(
        claim,
        date(2010, 1, 1),
        RetrievalBudget(),
        target_span=paper_ir.span_map()[claim.span_id],
        paper_ir=paper_ir,
    )
    assert len(works) == 1
    assert client.pdf_calls == 1
    assert works[0].spans[0].source == EvidenceLevel.FULLTEXT


def test_candidate_gate_repairs_one_malformed_batch(gear_config, paper_ir):
    retrieval = gear_config.retrieval.model_copy(update={"rerank_candidate_limit": 8})
    config = gear_config.model_copy(
        update={"allow_external_retrieval": True, "retrieval": retrieval}
    )
    calls = 0

    def repairable_gate(system, user):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"decisions": []}
        assert "Repair it" in system
        return comparable_candidates(system, user)

    service = PriorArtService(
        config,
        search_client=SearchFake(),
        query_generator=scientific_frame,
        rerank_generator=repairable_gate,
    )
    claim = paper_ir.claims[0]
    works = service.retrieve(
        claim,
        date(2010, 1, 1),
        RetrievalBudget(),
        target_span=paper_ir.span_map()[claim.span_id],
        paper_ir=paper_ir,
    )
    assert works
    assert calls == 2


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


def test_relation_classifier_selects_most_relevant_prior_span(gear_config, paper_ir):
    claim = paper_ir.claims[0]
    target = paper_ir.span_map()[claim.span_id]

    def prior_span(span_id, text):
        return RetrievedSpan(
            span_id=span_id,
            text=text,
            text_sha256="sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest(),
            source=EvidenceLevel.ABSTRACT,
        )

    prior = RetrievedWork(
        work_id="W-multi-span",
        target_claim_id=claim.claim_id,
        title="Earlier evidence controller",
        publication_date=date(2000, 1, 1),
        spans=[
            prior_span(
                "RS-off-topic",
                "Astronomical spectroscopy measures distant stellar atmospheres.",
            ),
            prior_span(
                "RS-relevant",
                "A novel evidence controller supports bounded scientific paper review.",
            ),
        ],
        retrieval_query_id="QRY-one",
        source_query_ids=["QRY-one", "QRY-two"],
        retrieval_source="fake",
    )
    classifier = RelationClassifier(
        gear_config,
        generator=lambda _system, _user: {
            "relation_label": "PARTIAL_ANTECEDENT",
            "difference_dimensions": ["mechanism"],
            "rationale": "The selected passage describes the same mechanism.",
        },
    )
    card = classifier.classify(
        target,
        prior,
        target_claim_id=claim.claim_id,
        cutoff=date(2010, 1, 1),
    )
    assert card.prior_span_id == "RS-relevant"
    assert card.source_query_ids == ["QRY-one", "QRY-two"]


def test_scientific_search_frame_normalizes_list_delta() -> None:
    frame = ScientificSearchFrame(
        target_object=["tumor"],
        task_problem=["response"],
        claimed_delta=["first mechanism", "second result"],
        source_span_ids=["S-1"],
    )
    assert frame.claimed_delta == "first mechanism; second result"

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
from gear.graph_prior_contracts import GraphResourceCapsV1, ResourceLedgerV1
from gear.prior_art import PriorArtService, RelationClassifier


def test_graph_direct_fetches_replace_provider_slots() -> None:
    ledger = ResourceLedgerV1(
        paper_id="p",
        caps=GraphResourceCapsV1(provider_searches=8, direct_fetches=2),
    )

    remaining = PriorArtService._reserve_graph_seed_slots(
        4, ["W1", "W1", "W2", "W3"], ledger
    )

    assert remaining == 2

    with_traversal = PriorArtService._reserve_graph_seed_slots(
        4, ["W1", "W2"], ledger, neighbor_slots=1
    )

    assert with_traversal == 1


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


class GraphSeedSearchFake(SearchFake):
    def __init__(self):
        super().__init__()
        self.fetch_calls = 0

    def fetch_work(self, work_id):
        self.fetch_calls += 1
        return {
            "paperId": work_id,
            "title": "Exact topology seed",
            "publication_date": "2000-01-01",
            "year": 2000,
            "abstract": "A prior topology anchor for evidence controllers.",
        }

    def search_query(
        self,
        query,
        *,
        provider=None,
        date_to=None,
        limit=None,
        search_mode="text",
    ):
        rows = super().search_query(
            query,
            provider=provider,
            date_to=date_to,
            limit=limit,
            search_mode=search_mode,
        )
        if query == "Topology guided evidence controller antecedents":
            rows = [
                {
                    "paperId": "W-seed",
                    "title": "Topology guided evidence controller antecedents",
                    "publication_date": "2000-01-01",
                    "year": 2000,
                    "abstract": "A claim-aligned topology anchor.",
                },
                *rows,
            ]
        return rows[: int(limit or len(rows))]


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
    repeated = service.retrieve(
        claim,
        date(2010, 1, 1),
        cached_budget,
        target_span=paper_ir.span_map()[claim.span_id],
        paper_ir=paper_ir,
    )
    assert repeated == []
    assert service.last_cache_hit is False
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


def test_titled_graph_seed_replaces_query_without_direct_fetch(
    gear_config, paper_ir
) -> None:
    client = GraphSeedSearchFake()
    service = service_with_model(
        gear_config.model_copy(update={"allow_external_retrieval": True}), client
    )
    ledger = ResourceLedgerV1(
        paper_id=paper_ir.paper_id,
        caps=GraphResourceCapsV1(provider_searches=4, direct_fetches=2),
    )
    claim = paper_ir.claims[0]

    service.retrieve(
        claim,
        date(2010, 1, 1),
        RetrievalBudget(normal_max=4),
        target_span=paper_ir.span_map()[claim.span_id],
        paper_ir=paper_ir,
        graph_seed_work_ids=["W-seed"],
        graph_seed_searches=[
            ("W-seed", "Topology guided evidence controller antecedents")
        ],
        allowed_query_roles=[
            "author_terminology",
            "object_problem",
            "mechanism_outcome",
            "purpose_semantic",
        ],
        resource_ledger=ledger,
    )

    graph_queries = [
        query
        for query in service.last_query_specs
        if query.transformation.startswith("graph_claim_aligned_topology_search:")
    ]
    assert len(graph_queries) == 1
    assert graph_queries[0].search_mode == "semantic"
    assert ":W-seed:" in graph_queries[0].transformation
    assert graph_queries[0].query_role == "purpose_semantic"
    assert "Topology guided evidence controller antecedents" in graph_queries[0].query
    assert any(
        term.casefold() in graph_queries[0].query.casefold()
        for term in service.last_frame.target_object
    )
    assert ledger.logical_provider_searches == 4
    assert ledger.logical_direct_fetches == 0
    assert client.search_calls == 4
    assert client.fetch_calls == 0
    assert service.last_graph_seed_works == []


def test_titleless_graph_seed_keeps_legacy_direct_fetch(gear_config, paper_ir) -> None:
    client = GraphSeedSearchFake()
    service = service_with_model(
        gear_config.model_copy(update={"allow_external_retrieval": True}), client
    )
    ledger = ResourceLedgerV1(
        paper_id=paper_ir.paper_id,
        caps=GraphResourceCapsV1(provider_searches=4, direct_fetches=2),
    )
    claim = paper_ir.claims[0]

    service.retrieve(
        claim,
        date(2010, 1, 1),
        RetrievalBudget(normal_max=4),
        target_span=paper_ir.span_map()[claim.span_id],
        paper_ir=paper_ir,
        graph_seed_work_ids=["W-seed"],
        allowed_query_roles=[
            "author_terminology",
            "object_problem",
            "mechanism_outcome",
            "purpose_semantic",
        ],
        resource_ledger=ledger,
    )

    assert ledger.logical_provider_searches == 3
    assert ledger.logical_direct_fetches == 1
    assert client.fetch_calls == 1


def test_resource_ledger_caps_real_provider_calls_and_remote_role_order(
    gear_config, paper_ir
) -> None:
    client = SearchFake()
    service = service_with_model(
        gear_config.model_copy(update={"allow_external_retrieval": True}), client
    )
    ledger = ResourceLedgerV1(
        paper_id=paper_ir.paper_id,
        caps=GraphResourceCapsV1(provider_searches=2),
    )
    claim = paper_ir.claims[0]
    service.retrieve(
        claim,
        date(2010, 1, 1),
        RetrievalBudget(normal_max=4),
        target_span=paper_ir.span_map()[claim.span_id],
        paper_ir=paper_ir,
        allowed_query_roles=[
            "mechanism_outcome",
            "purpose_semantic",
            "author_terminology",
            "object_problem",
        ],
        resource_ledger=ledger,
    )
    assert [query.query_role for query in service.last_query_specs] == [
        "mechanism_outcome",
        "purpose_semantic",
    ]
    assert ledger.logical_provider_searches == 2
    assert ledger.network_provider_attempts == 2
    assert client.search_calls == 2


def test_fulltext_cap_consumes_planned_logical_slots_without_network(
    gear_config, paper_ir
) -> None:
    client = SearchFake()
    service = service_with_model(
        gear_config.model_copy(update={"allow_external_retrieval": True}), client
    )
    ledger = ResourceLedgerV1(
        paper_id=paper_ir.paper_id,
        caps=GraphResourceCapsV1(provider_searches=2),
    )
    claim = paper_ir.claims[0]

    works = service.retrieve(
        claim,
        date(2010, 1, 1),
        RetrievalBudget(normal_max=2, fulltext_max=1, fulltext_kept=1),
        target_span=paper_ir.span_map()[claim.span_id],
        paper_ir=paper_ir,
        allowed_query_roles=["author_terminology", "object_problem"],
        resource_ledger=ledger,
    )

    assert works == []
    assert ledger.logical_provider_searches == 2
    assert ledger.network_provider_attempts == 0
    assert client.search_calls == 0


def test_neighbor_query_id_includes_traversal_direction(gear_config, paper_ir) -> None:
    service = service_with_model(
        gear_config.model_copy(update={"allow_external_retrieval": True}),
        SearchFake(),
    )
    claim = paper_ir.claims[0]
    seed = service.retrieve(
        claim,
        date(2010, 1, 1),
        RetrievalBudget(),
        target_span=paper_ir.span_map()[claim.span_id],
        paper_ir=paper_ir,
    )[0]
    expansion_budget = RetrievalBudget(citation_expansion_max=2)

    service.expand_neighbors(
        seed,
        claim,
        date(2010, 1, 1),
        expansion_budget,
        direction="references",
    )
    references_id = service.last_query_specs[0].query_id
    service.expand_neighbors(
        seed,
        claim,
        date(2010, 1, 1),
        expansion_budget,
        direction="citations",
    )
    citations_id = service.last_query_specs[0].query_id

    assert references_id != citations_id


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
    ledger = ResourceLedgerV1(paper_id=paper_ir.paper_id)

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
        resource_ledger=ledger,
    )

    assert works == []
    assert service.last_service_failed is False
    assert service.last_failures == []
    assert service.last_advisories == [
        "contrastive_query_coverage_gap:contrastive query did not change the search intent"
    ]
    assert ledger.logical_provider_searches == 1
    assert ledger.network_provider_attempts == 0
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


def test_direct_antecedent_requires_facet_coverage_and_independent_pass(
    gear_config, paper_ir
):
    claim = paper_ir.claims[0]
    target = paper_ir.span_map()[claim.span_id]
    text = "An earlier controller implements every essential target facet."
    prior = RetrievedWork(
        work_id="W-confirmed",
        target_claim_id=claim.claim_id,
        title="Earlier complete controller",
        publication_date=date(2000, 1, 1),
        spans=[
            RetrievedSpan(
                span_id="RS-confirmed",
                text=text,
                text_sha256="sha256:"
                + hashlib.sha256(text.encode("utf-8")).hexdigest(),
                source=EvidenceLevel.FULLTEXT,
            )
        ],
        retrieval_query_id="QRY-confirmed",
        retrieval_source="fake",
    )

    def generate(system, user):
        if "Independently try to falsify" in system:
            return {"confirmed": True, "missing_facets": [], "rationale": "complete"}
        return {
            "relation_label": "DIRECT_ANTECEDENT",
            "common_dimensions": ["mechanism", "outcome"],
            "difference_dimensions": ["implementation detail"],
            "essential_facet_coverage": 1.0,
            "rationale": "All essential facets precede the target.",
        }

    card = RelationClassifier(gear_config, generator=generate).classify(
        target,
        prior,
        target_claim_id=claim.claim_id,
        cutoff=date(2010, 1, 1),
    )

    assert card.relation_label == RelationLabel.DIRECT_ANTECEDENT
    assert card.essential_facet_coverage == 1.0
    assert card.independent_verification_passed is True


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

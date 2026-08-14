"""Claim-level prior-art retrieval and paired-span relation classification."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any, Protocol

from .config import GearConfig, load_config
from .contracts import (
    EvidenceLevel,
    EvidenceSpan,
    PaperClaim,
    QuerySpec,
    RelationCard,
    RelationLabel,
    RetrievalBudget,
    RetrievedSpan,
    RetrievedWork,
)
from .model_client import (
    JsonModelClient,
    ModelClientUnavailableError,
    build_json_model_client,
)

RELATION_CLASSIFICATION_PROMPT = """Compare one target-paper claim span with one
prior-work span. Return exactly one relation from DIRECT_ANTECEDENT,
PARTIAL_ANTECEDENT, EXTENSION, PARALLEL, SUPPORT, CONFLICT, DISTANT, UNRESOLVED.
Similarity alone is not antecedence. Cite both supplied span IDs."""

TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
STOPWORDS = {
    "about",
    "after",
    "also",
    "among",
    "based",
    "first",
    "from",
    "introduce",
    "method",
    "novel",
    "paper",
    "propose",
    "show",
    "study",
    "that",
    "their",
    "these",
    "this",
    "using",
    "with",
    "work",
}


class SearchClient(Protocol):
    def search_query(
        self,
        query: str,
        *,
        provider: str | None = None,
        date_to: date | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]: ...

    def fetch_neighbors(
        self,
        work_id: str,
        direction: str = "references",
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]: ...


def _stable_id(prefix: str, payload: str) -> str:
    return prefix + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:18]


def _tokens(text: str) -> list[str]:
    return [
        token.casefold()
        for token in TOKEN_PATTERN.findall(str(text or ""))
        if token.casefold() not in STOPWORDS
    ]


def _parse_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _parse_year(value: Any) -> int | None:
    try:
        year = int(value)
    except (TypeError, ValueError):
        return None
    return year if 1500 <= year <= 3000 else None


class QueryPlanner:
    """Produce bounded, reproducible lexical/mechanism query families."""

    def plan(self, claim: PaperClaim) -> list[QuerySpec]:
        terms = list(dict.fromkeys(_tokens(claim.text)))
        lexical = " ".join(terms[:10]) or claim.text[:180]
        mechanism_terms = [
            term
            for term in terms
            if term not in {"framework", "approach", "system", "model"}
        ]
        mechanism = " ".join(mechanism_terms[:8]) or lexical
        return [
            QuerySpec(
                query_id=_stable_id("QRY-", f"{claim.claim_id}|lexical|{lexical}"),
                claim_id=claim.claim_id,
                family="lexical",
                query=lexical,
            ),
            QuerySpec(
                query_id=_stable_id("QRY-", f"{claim.claim_id}|mechanism|{mechanism}"),
                claim_id=claim.claim_id,
                family="mechanism",
                query=mechanism,
            ),
        ]

    def contrastive(self, claim: PaperClaim) -> QuerySpec:
        terms = [
            term
            for term in dict.fromkeys(_tokens(claim.text))
            if not re.search(
                r"(?:net|bert|gpt|former|framework)$", term, flags=re.IGNORECASE
            )
        ]
        query = " ".join(terms[:8]) or claim.text[:180]
        return QuerySpec(
            query_id=_stable_id("QRY-", f"{claim.claim_id}|contrastive|{query}"),
            claim_id=claim.claim_id,
            family="contrastive",
            query=query,
        )


class PriorPassageExtractor:
    """Select one to three query-relevant passages instead of whole prior papers."""

    def extract(
        self,
        text: str,
        *,
        query: str,
        source: EvidenceLevel,
        maximum: int = 3,
    ) -> list[RetrievedSpan]:
        chunks = [
            chunk.strip()
            for chunk in re.split(r"\n\s*\n|(?<=[.!?。！？])\s+", text)
            if len(chunk.strip()) >= 40
        ]
        if not chunks and text.strip():
            chunks = [text.strip()]
        query_tokens = set(_tokens(query))
        ranked = sorted(
            chunks,
            key=lambda chunk: (
                len(query_tokens & set(_tokens(chunk)))
                / max(len(query_tokens | set(_tokens(chunk))), 1),
                len(chunk),
            ),
            reverse=True,
        )[:maximum]
        spans: list[RetrievedSpan] = []
        for chunk in ranked:
            passage = chunk[:4_000]
            digest = hashlib.sha256(passage.encode("utf-8")).hexdigest()
            spans.append(
                RetrievedSpan(
                    span_id="RS-" + digest[:18],
                    text=passage,
                    text_sha256=f"sha256:{digest}",
                    source=source,
                )
            )
        return spans


class PriorArtService:
    """Retrieve candidates under a hard budget and pre-classification cutoff."""

    def __init__(
        self,
        config: GearConfig | None = None,
        *,
        search_client: SearchClient | None = None,
    ) -> None:
        self.config = config or load_config()
        self.search_client = search_client
        self.query_planner = QueryPlanner()
        self.passage_extractor = PriorPassageExtractor()
        self.last_failures: list[str] = []
        self.last_queries: list[str] = []
        self.last_cache_hit = False
        self._pdf_downloads_used = 0
        self._pdf_text_cache: dict[str, str] = {}

    def _client(self) -> SearchClient:
        if self.search_client is not None:
            return self.search_client
        from gear.scholar import OpenScholar

        class _Args:
            s2_api_key = ""
            openalex_api_key = ""
            and_search = False
            retrieval_provider = "openalex"

        self.search_client = OpenScholar(_Args())
        return self.search_client

    def retrieve(
        self,
        claim: PaperClaim,
        cutoff: date,
        budget: RetrievalBudget,
        *,
        family: LiteralQueryFamily = "normal",
    ) -> list[RetrievedWork]:
        self.last_failures = []
        self.last_queries = []
        self.last_cache_hit = False
        cache_hits: list[bool] = []
        if not self.config.allow_external_retrieval:
            self.last_failures.append("external_retrieval_disabled")
            if family == "contrastive":
                budget.contrastive_used = budget.contrastive_max
            else:
                budget.normal_used = budget.normal_max
            return []
        if family == "contrastive":
            if budget.contrastive_used >= budget.contrastive_max:
                return []
            queries = [self.query_planner.contrastive(claim)]
            budget.contrastive_used += 1
        else:
            remaining = budget.normal_max - budget.normal_used
            if remaining <= 0:
                return []
            queries = self.query_planner.plan(claim)[:remaining]
            budget.normal_used += len(queries)
        remaining_slots = budget.fulltext_max - budget.fulltext_kept
        if remaining_slots <= 0:
            return []
        works: dict[str, RetrievedWork] = {}
        for query in queries:
            self.last_queries.append(f"{query.query_id}:{query.query}")
            try:
                rows, cache_hit = self._cached_search(
                    query.query,
                    cutoff=cutoff,
                    limit=min(self.config.retrieval.provider_limit, remaining_slots),
                )
                cache_hits.append(cache_hit)
            except Exception as exc:
                self.last_failures.append(f"{query.query_id}:{exc}")
                continue
            for row in rows:
                work = self._normalize_work(row, query)
                if work is None or not self._eligible_before_cutoff(work, cutoff):
                    continue
                works.setdefault(work.work_id, work)
                if len(works) >= remaining_slots:
                    break
            if len(works) >= remaining_slots:
                break
        budget.fulltext_kept += len(works)
        self.last_cache_hit = bool(cache_hits) and all(cache_hits)
        return list(works.values())

    def expand_neighbors(
        self,
        seed: RetrievedWork,
        claim: PaperClaim,
        cutoff: date,
        budget: RetrievalBudget,
    ) -> list[RetrievedWork]:
        self.last_failures = []
        self.last_queries = []
        self.last_cache_hit = False
        if budget.citation_expansion_used >= budget.citation_expansion_max:
            return []
        budget.citation_expansion_used += 1
        remaining_slots = budget.fulltext_max - budget.fulltext_kept
        if remaining_slots <= 0:
            return []
        query = QuerySpec(
            query_id=_stable_id("QRY-", f"{claim.claim_id}|citation|{seed.work_id}"),
            claim_id=claim.claim_id,
            family="citation",
            query=seed.work_id,
        )
        self.last_queries = [f"{query.query_id}:{query.query}"]
        try:
            rows, self.last_cache_hit = self._cached_neighbors(
                seed.work_id,
                cutoff=cutoff,
                limit=remaining_slots,
            )
        except Exception as exc:
            self.last_failures.append(f"citation_expansion:{exc}")
            return []
        works: list[RetrievedWork] = []
        for row in rows:
            work = self._normalize_work(row, query)
            if work is not None and self._eligible_before_cutoff(work, cutoff):
                works.append(work)
            if len(works) >= remaining_slots:
                break
        budget.fulltext_kept += len(works)
        return works

    def _cache_path(self, payload: Mapping[str, Any]) -> Path:
        identity = json.dumps(
            {
                "config_version": self.config.config_version,
                **dict(payload),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        root = self.config.resolve_path(self.config.cache_dir) / "retrieval"
        root.mkdir(parents=True, exist_ok=True)
        return root / f"{digest}.json"

    def _cached_search(
        self,
        query: str,
        *,
        cutoff: date,
        limit: int,
    ) -> tuple[list[dict[str, Any]], bool]:
        client = self._client()
        provider = str(getattr(client, "retrieval_provider", type(client).__name__))
        path = self._cache_path(
            {
                "operation": "search_query",
                "query": query,
                "cutoff": cutoff.isoformat(),
                "provider": provider,
                "limit": int(limit),
            }
        )
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, list):
                    return [
                        dict(item) for item in payload if isinstance(item, dict)
                    ], True
            except (OSError, json.JSONDecodeError):
                pass
        rows = client.search_query(
            query,
            provider=None,
            date_to=cutoff,
            limit=limit,
        )
        path.write_text(
            json.dumps(rows, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return rows, False

    def _cached_neighbors(
        self,
        work_id: str,
        *,
        cutoff: date,
        limit: int,
    ) -> tuple[list[dict[str, Any]], bool]:
        client = self._client()
        provider = str(getattr(client, "retrieval_provider", type(client).__name__))
        path = self._cache_path(
            {
                "operation": "fetch_neighbors",
                "query": work_id,
                "cutoff": cutoff.isoformat(),
                "provider": provider,
                "direction": "references",
                "limit": int(limit),
            }
        )
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, list):
                    return [
                        dict(item) for item in payload if isinstance(item, dict)
                    ], True
            except (OSError, json.JSONDecodeError):
                pass
        rows = client.fetch_neighbors(
            work_id,
            "references",
            limit=limit,
        )
        eligible = []
        for row in rows:
            raw_date = _parse_date(
                row.get("publication_date") or row.get("publicationDate")
            )
            raw_year = _parse_year(row.get("year") or row.get("publication_year"))
            if raw_date is not None and raw_date >= cutoff:
                continue
            if raw_date is None and raw_year is not None and raw_year > cutoff.year:
                continue
            eligible.append(row)
        path.write_text(
            json.dumps(eligible, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return eligible, False

    @staticmethod
    def _eligible_before_cutoff(work: RetrievedWork, cutoff: date) -> bool:
        if work.publication_date is not None:
            return work.publication_date < cutoff
        if work.publication_year is not None:
            return work.publication_year <= cutoff.year
        return True

    def _normalize_work(
        self,
        row: Mapping[str, Any],
        query: QuerySpec,
    ) -> RetrievedWork | None:
        work_id = str(row.get("paperId") or row.get("id") or "").strip()
        title = str(row.get("title") or row.get("display_name") or "").strip()
        if not work_id or not title:
            return None
        abstract = str(row.get("abstract") or "").strip()
        full_text = str(row.get("full_text") or row.get("fulltext") or "").strip()
        citation_context = str(row.get("citation_context") or "").strip()
        if not abstract and not full_text and not citation_context:
            full_text = self._openalex_pdf_text(work_id)
        authors_raw = row.get("authors") or []
        if isinstance(authors_raw, str):
            authors = [part.strip() for part in authors_raw.split(",") if part.strip()]
        else:
            authors = [
                (
                    str(item.get("name") or item.get("display_name") or "").strip()
                    if isinstance(item, dict)
                    else str(item).strip()
                )
                for item in authors_raw
            ]
            authors = [item for item in authors if item]
        publication_date = _parse_date(
            row.get("publication_date") or row.get("publicationDate")
        )
        publication_year = _parse_year(row.get("year") or row.get("publication_year"))
        raw_external = row.get("externalIds")
        external: Mapping[str, Any] = (
            raw_external if isinstance(raw_external, dict) else {}
        )
        doi = str(row.get("doi") or external.get("DOI") or "").strip() or None
        spans: list[RetrievedSpan] = []
        evidence_text = full_text[:30_000] or abstract or citation_context
        evidence_level = (
            EvidenceLevel.FULLTEXT
            if full_text
            else (
                EvidenceLevel.ABSTRACT if abstract else EvidenceLevel.CITATION_CONTEXT
            )
        )
        if evidence_text:
            spans = self.passage_extractor.extract(
                evidence_text,
                query=query.query,
                source=evidence_level,
            )
        if not spans:
            return None
        return RetrievedWork(
            work_id=work_id,
            target_claim_id=query.claim_id,
            title=title,
            abstract=abstract,
            authors=authors,
            venue=str(row.get("venue") or ""),
            publication_date=publication_date,
            publication_year=publication_year,
            doi=doi,
            cited_work_ids=[str(item) for item in row.get("referenced_works") or []],
            spans=spans,
            retrieval_query_id=query.query_id,
            retrieval_source=str(row.get("retrieval_source") or "unknown"),
        )

    def _openalex_pdf_text(self, work_id: str) -> str:
        limits = self.config.retrieval
        if not limits.openalex_pdf_enabled or "openalex.org/" not in work_id.casefold():
            return ""
        if work_id in self._pdf_text_cache:
            return self._pdf_text_cache[work_id]
        if self._pdf_downloads_used >= limits.openalex_pdf_max_downloads:
            return ""
        fetch_pdf_text = getattr(self._client(), "fetch_pdf_text", None)
        if not callable(fetch_pdf_text):
            return ""
        self._pdf_downloads_used += 1
        try:
            text = str(
                fetch_pdf_text(
                    work_id,
                    max_bytes=limits.openalex_pdf_max_bytes,
                    max_pages=limits.openalex_pdf_max_pages,
                    max_characters=limits.openalex_pdf_max_characters,
                )
                or ""
            ).strip()
        except (OSError, RuntimeError, TypeError, ValueError):
            text = ""
        self._pdf_text_cache[work_id] = text
        return text


LiteralQueryFamily = str


class RelationClassifier:
    """Classify paired evidence; lexical similarity never creates antecedence."""

    def __init__(
        self,
        config: GearConfig | None = None,
        *,
        generator: Callable[[str, str], Mapping[str, Any]] | None = None,
    ) -> None:
        self.config = config or load_config()
        self.client: JsonModelClient = build_json_model_client(self.config)
        self.generator = generator
        self.last_failure: str | None = None

    def classify(
        self,
        claim_span: EvidenceSpan,
        prior: RetrievedWork,
        *,
        target_claim_id: str,
        cutoff: date,
    ) -> RelationCard:
        self.last_failure = None
        prior_span = prior.spans[0] if prior.spans else None
        temporal_valid, temporal_unresolved = self._temporal_status(prior, cutoff)
        if prior_span is None:
            return self._card(
                claim_span,
                prior,
                target_claim_id,
                RelationLabel.UNRESOLVED,
                EvidenceLevel.METADATA_ONLY,
                temporal_valid,
                temporal_unresolved,
                "Prior work has no verifiable abstract or full-text span.",
            )
        if not temporal_valid:
            label = (
                RelationLabel.PARALLEL
                if temporal_unresolved
                else RelationLabel.UNRESOLVED
            )
            return self._card(
                claim_span,
                prior,
                target_claim_id,
                label,
                prior_span.source,
                temporal_valid,
                temporal_unresolved,
                "Temporal precedence is not established.",
            )
        user = json.dumps(
            {
                "target_claim_id": target_claim_id,
                "target_span": claim_span.model_dump(mode="json"),
                "prior_work": prior.model_dump(mode="json"),
                "output": {
                    "relation_label": "one allowed label",
                    "difference_dimensions": [],
                    "rationale": "paired-span rationale",
                },
            },
            ensure_ascii=False,
        )
        try:
            payload = (
                self.generator(RELATION_CLASSIFICATION_PROMPT, user)
                if self.generator is not None
                else self.client.generate_json(
                    system=RELATION_CLASSIFICATION_PROMPT,
                    user=user,
                )
            )
            label = RelationLabel(str(payload.get("relation_label")))
            dimensions = [
                str(item) for item in payload.get("difference_dimensions") or []
            ]
            rationale = str(payload.get("rationale") or "")
            if label != RelationLabel.UNRESOLVED and not dimensions:
                self.last_failure = "relation_classifier_missing_difference_dimensions"
                label = RelationLabel.UNRESOLVED
                dimensions = ["classifier_output_incomplete"]
                rationale = (
                    "The classifier omitted the required difference dimensions; "
                    "the relation is unresolved."
                )
        except (ModelClientUnavailableError, ValueError, TypeError) as exc:
            self.last_failure = str(exc)
            target_tokens = set(_tokens(claim_span.text))
            prior_tokens = set(_tokens(prior_span.text))
            overlap = len(target_tokens & prior_tokens) / max(
                len(target_tokens | prior_tokens), 1
            )
            label = RelationLabel.DISTANT if overlap < 0.1 else RelationLabel.UNRESOLVED
            dimensions = ["lexical_scope"]
            rationale = "Relation model unavailable; lexical overlap is not treated as antecedence."
        return self._card(
            claim_span,
            prior,
            target_claim_id,
            label,
            prior_span.source,
            temporal_valid,
            temporal_unresolved,
            rationale,
            dimensions,
        )

    @staticmethod
    def _temporal_status(prior: RetrievedWork, cutoff: date) -> tuple[bool, bool]:
        if prior.publication_date is not None:
            return prior.publication_date < cutoff, prior.publication_date == cutoff
        if prior.publication_year is None:
            return False, True
        if prior.publication_year < cutoff.year:
            return True, False
        if prior.publication_year == cutoff.year:
            return False, True
        return False, False

    @staticmethod
    def _card(
        claim_span: EvidenceSpan,
        prior: RetrievedWork,
        target_claim_id: str,
        label: RelationLabel,
        level: EvidenceLevel,
        temporal_valid: bool,
        temporal_unresolved: bool,
        rationale: str,
        dimensions: Sequence[str] | None = None,
    ) -> RelationCard:
        prior_span_id = prior.spans[0].span_id if prior.spans else None
        identity = (
            f"{target_claim_id}|{claim_span.span_id}|{prior.work_id}|"
            f"{prior_span_id}|{label.value}|{prior.retrieval_query_id}"
        )
        normalized_dimensions = list(dimensions or [])
        if not normalized_dimensions:
            normalized_dimensions = [
                (
                    "temporal_order"
                    if temporal_unresolved
                    else "insufficient_paired_evidence"
                )
            ]
        return RelationCard(
            relation_id=_stable_id("REL-", identity),
            target_claim_id=target_claim_id,
            target_span_id=claim_span.span_id,
            prior_work_id=prior.work_id,
            prior_span_id=prior_span_id,
            prior_work_date=prior.publication_date,
            prior_work_year=prior.publication_year,
            relation_label=label,
            evidence_level=level,
            difference_dimensions=normalized_dimensions,
            retrieval_query_id=prior.retrieval_query_id,
            rationale=rationale,
            temporal_valid=temporal_valid,
            temporal_order_unresolved=temporal_unresolved,
        )


__all__ = [
    "PriorArtService",
    "PriorPassageExtractor",
    "QueryPlanner",
    "RelationClassifier",
    "SearchClient",
]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build large Fig. 3 input datasets from external landmark and topic sources.

The builder is intentionally independent from Fig. 1. It creates the standard
Fig. 3 files consumed by fig3_empirical_weight_learning.py:

    works.csv
    citations.csv
    topics.csv
    topic_edges.csv
    landmark_registry.csv
    dataset_report.json

Primary data sources:
    Nobel Prize API v2.1, OpenAlex Works/Topics, optional Semantic Scholar Graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import requests


NOBEL_API_BASE = "https://api.nobelprize.org/2.1"
OPENALEX_API_BASE = "https://api.openalex.org"
S2_API_BASE = "https://api.semanticscholar.org/graph/v1"

NOBEL_SCIENCE_CATEGORIES = {
    "physics",
    "chemistry",
    "physiology or medicine",
    "economic sciences",
}

OPENALEX_WORK_SELECT = ",".join(
    [
        "id",
        "doi",
        "display_name",
        "publication_year",
        "publication_date",
        "type",
        "language",
        "cited_by_count",
        "referenced_works",
        "primary_topic",
        "topics",
        "keywords",
        "authorships",
        "cited_by_api_url",
        "fwci",
        "citation_normalized_percentile",
    ]
)

S2_SEARCH_FIELDS = "paperId,title,year,authors,citationCount,externalIds,fieldsOfStudy"

STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "because",
    "been",
    "between",
    "both",
    "from",
    "have",
    "into",
    "that",
    "their",
    "this",
    "with",
    "were",
    "which",
    "whose",
    "work",
    "works",
    "study",
    "studies",
    "research",
    "discovery",
    "discoveries",
}


@dataclass
class DomainSeed:
    slug: str
    display_name: str
    topic_id: str
    query: str
    works_count: int = 0
    field_name: str = ""
    subfield_name: str = ""


@dataclass
class NobelSeed:
    prize_id: str
    award_year: int
    category: str
    motivation: str
    laureates: List[str]


def progress_log(message: str, enabled: bool = True) -> None:
    if enabled:
        print(message, flush=True)


def slugify(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(text).lower()).strip("_")
    return text or "domain"


def short_openalex_id(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.rstrip("/").split("/")[-1]


def normalize_openalex_id(value: object) -> str:
    sid = short_openalex_id(value)
    if not sid:
        return ""
    return f"https://openalex.org/{sid}"


def normalize_doi(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text, flags=re.I)
    return text.lower()


def stable_int_id(value: str, modulo: int = 1_000_000_000) -> int:
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo


def tokens(text: str) -> List[str]:
    raw = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", str(text).lower())
    return [t for t in raw if t not in STOPWORDS and len(t) >= 3]


def token_overlap(a: Iterable[str], b: Iterable[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, min(len(sa), len(sb)))


def domain_seed_from_openalex_topic(topic: Mapping[str, Any], fallback_name: str = "") -> DomainSeed:
    tid = normalize_openalex_id(topic.get("id"))
    name = str(topic.get("display_name") or fallback_name or short_openalex_id(tid))
    field = topic.get("field") or {}
    subfield = topic.get("subfield") or {}
    return DomainSeed(
        slug=slugify(name),
        display_name=name,
        topic_id=tid,
        query=name,
        works_count=int(topic.get("works_count") or 0),
        field_name=str(field.get("display_name") or ""),
        subfield_name=str(subfield.get("display_name") or ""),
    )


class RestClient:
    """Small retrying REST client for public scholarly APIs."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        email: Optional[str] = None,
        sleep_seconds: float = 0.1,
        timeout_seconds: int = 60,
        max_retries: int = 5,
        key_header: Optional[str] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.email = email
        self.sleep_seconds = float(sleep_seconds)
        self.timeout_seconds = int(timeout_seconds)
        self.max_retries = int(max_retries)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "aspr-fig3-dataset-builder/1.0"})
        if api_key and key_header:
            self.session.headers.update({key_header: api_key})

    def get_json(self, path_or_url: str, params: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        params_dict = dict(params or {})
        if self.api_key and self.base_url == OPENALEX_API_BASE:
            params_dict["api_key"] = self.api_key
        if self.email and self.base_url == OPENALEX_API_BASE:
            params_dict["mailto"] = self.email
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            url = path_or_url
        else:
            url = f"{self.base_url}/{path_or_url.lstrip('/')}"
        for attempt in range(self.max_retries + 1):
            if self.sleep_seconds > 0:
                time.sleep(self.sleep_seconds)
            try:
                resp = self.session.get(url, params=params_dict, timeout=self.timeout_seconds)
            except requests.RequestException as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(f"Request failed: {url}: {exc}") from exc
                time.sleep(min(60.0, 2.0**attempt))
                continue
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                delay = float(resp.headers.get("Retry-After") or min(60.0, 2.0**attempt))
                time.sleep(delay)
                continue
            raise RuntimeError(f"HTTP {resp.status_code} for {resp.url}: {resp.text[:600]}")
        raise RuntimeError(f"Retries exhausted: {url}")


class OpenAlexClient(RestClient):
    def __init__(
        self,
        api_key: Optional[str],
        email: Optional[str],
        sleep_seconds: float,
        timeout_seconds: int,
        max_retries: int,
    ) -> None:
        super().__init__(
            OPENALEX_API_BASE,
            api_key=api_key,
            email=email,
            sleep_seconds=sleep_seconds,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )

    def list_cursor(
        self,
        endpoint: str,
        max_records: int,
        params: Optional[Mapping[str, Any]] = None,
        per_page: int = 100,
    ) -> List[Dict[str, Any]]:
        params_dict = dict(params or {})
        params_dict["per_page"] = int(per_page)
        params_dict["cursor"] = "*"
        out: List[Dict[str, Any]] = []
        while len(out) < int(max_records):
            data = self.get_json(endpoint, params=params_dict)
            results = data.get("results", []) or []
            if not results:
                break
            remaining = int(max_records) - len(out)
            out.extend(results[:remaining])
            next_cursor = (data.get("meta") or {}).get("next_cursor")
            if not next_cursor or len(results) < per_page:
                break
            params_dict["cursor"] = next_cursor
        return out

    def discover_topics(self, max_topics: int, min_works_count: int) -> List[DomainSeed]:
        params = {
            "filter": f"works_count:>{int(min_works_count)}",
            "sort": "works_count:desc",
            "select": "id,display_name,description,works_count,field,subfield,domain",
        }
        topics = self.list_cursor("/topics", max_records=max_topics, params=params)
        seeds: List[DomainSeed] = []
        for topic in topics:
            seeds.append(domain_seed_from_openalex_topic(topic))
        return seeds

    def get_topic(self, topic_id: str) -> Optional[Dict[str, Any]]:
        tid = short_openalex_id(topic_id)
        if not tid:
            return None
        try:
            return self.get_json(
                "/topics/" + tid,
                params={"select": "id,display_name,description,works_count,field,subfield,domain"},
            )
        except Exception:
            return None

    def list_works(
        self,
        max_records: int,
        search: Optional[str] = None,
        filters: Optional[Sequence[str]] = None,
        sort: Optional[str] = None,
        per_page: int = 100,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"select": OPENALEX_WORK_SELECT}
        filt = [f for f in (filters or []) if f]
        if filt:
            params["filter"] = ",".join(filt)
        if search:
            params["search"] = search
        if sort:
            params["sort"] = sort
        return self.list_cursor("/works", max_records=max_records, params=params, per_page=per_page)

    def get_work(self, openalex_id_or_doi: str) -> Optional[Dict[str, Any]]:
        ident = str(openalex_id_or_doi or "").strip()
        if not ident:
            return None
        if "/" in ident and "doi.org" not in ident and not ident.startswith("http"):
            ident = normalize_openalex_id(ident)
        if ident.lower().startswith("10.") or "doi.org" in ident:
            doi = normalize_doi(ident)
            endpoint = "/works/" + urllib.parse.quote("https://doi.org/" + doi, safe="")
        else:
            endpoint = "/works/" + short_openalex_id(ident)
        try:
            return self.get_json(endpoint, params={"select": OPENALEX_WORK_SELECT})
        except Exception:
            return None


class SemanticScholarClient(RestClient):
    def __init__(
        self,
        api_key: Optional[str],
        sleep_seconds: float,
        timeout_seconds: int,
        max_retries: int,
    ) -> None:
        super().__init__(
            S2_API_BASE,
            api_key=api_key,
            sleep_seconds=sleep_seconds,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            key_header="x-api-key",
        )

    def search_papers(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        try:
            data = self.get_json(
                "/paper/search",
                params={"query": query, "limit": int(limit), "fields": S2_SEARCH_FIELDS},
            )
        except Exception:
            return []
        return data.get("data", []) or []


def fetch_nobel_seeds(client: RestClient, from_year: int, to_year: int, progress: bool) -> List[NobelSeed]:
    seeds: List[NobelSeed] = []
    offset = 0
    limit = 100
    while True:
        data = client.get_json(
            "/nobelPrizes",
            params={"offset": offset, "limit": limit, "sort": "asc"},
        )
        prizes = data.get("nobelPrizes", []) or []
        if not prizes:
            break
        for prize in prizes:
            try:
                year = int(prize.get("awardYear"))
            except Exception:
                continue
            if year < from_year or year > to_year:
                continue
            category_obj = prize.get("category") or {}
            category = str(category_obj.get("en") or "").strip().lower()
            if category not in NOBEL_SCIENCE_CATEGORIES:
                continue
            laureate_names: List[str] = []
            motivation_parts: List[str] = []
            for laureate in prize.get("laureates") or []:
                known = laureate.get("knownName") or laureate.get("fullName") or {}
                name = known.get("en") if isinstance(known, Mapping) else ""
                if not name:
                    org = laureate.get("orgName") or {}
                    name = org.get("en") if isinstance(org, Mapping) else ""
                if name:
                    laureate_names.append(str(name))
                motivation = laureate.get("motivation") or {}
                if isinstance(motivation, Mapping) and motivation.get("en"):
                    motivation_parts.append(str(motivation["en"]))
            seeds.append(
                NobelSeed(
                    prize_id=str(prize.get("id") or f"{category}-{year}"),
                    award_year=year,
                    category=category,
                    motivation=" ".join(dict.fromkeys(motivation_parts)),
                    laureates=list(dict.fromkeys(laureate_names)),
                )
            )
        offset += len(prizes)
        if len(prizes) < limit:
            break
    progress_log(f"Fetched {len(seeds):,} Nobel science prize seeds.", progress)
    return seeds


def work_text(work: Mapping[str, Any]) -> str:
    parts = [str(work.get("display_name") or work.get("title") or "")]
    primary = work.get("primary_topic") or {}
    if isinstance(primary, Mapping):
        parts.append(str(primary.get("display_name") or ""))
    for topic in work.get("topics") or []:
        if isinstance(topic, Mapping):
            parts.append(str(topic.get("display_name") or ""))
    for keyword in work.get("keywords") or []:
        if isinstance(keyword, Mapping):
            parts.append(str(keyword.get("display_name") or ""))
    return " ".join(parts)


def work_author_tokens(work: Mapping[str, Any]) -> List[str]:
    names: List[str] = []
    for auth in work.get("authorships") or []:
        author = auth.get("author") or {}
        if isinstance(author, Mapping) and author.get("display_name"):
            names.append(str(author["display_name"]))
    return tokens(" ".join(names))


def score_landmark_match(seed: NobelSeed, work: Mapping[str, Any]) -> float:
    title_topic_tokens = tokens(work_text(work))
    motivation_tokens = tokens(seed.motivation)
    author_tokens = set(work_author_tokens(work))
    laureate_tokens = tokens(" ".join(seed.laureates))
    laureate_surnames = [tokens(name)[-1] for name in seed.laureates if tokens(name)]
    name_score = 0.0
    if laureate_surnames:
        surname_hits = sum(1 for surname in laureate_surnames if surname in author_tokens)
        name_fraction = surname_hits / len(laureate_surnames)
        name_score = 0.65 * name_fraction + 0.35 * float(surname_hits > 0)
    motivation_score = token_overlap(motivation_tokens, title_topic_tokens)
    year = int(work.get("publication_year") or 0)
    if year <= 0 or year > seed.award_year:
        year_score = 0.0
    else:
        lag = seed.award_year - year
        year_score = math.exp(-abs(lag - 10.0) / 18.0)
    citation_score = min(1.0, math.log1p(float(work.get("cited_by_count") or 0)) / math.log1p(5000.0))
    doi_score = 1.0 if normalize_doi(work.get("doi")) else 0.4
    return float(
        0.42 * name_score
        + 0.20 * motivation_score
        + 0.18 * citation_score
        + 0.12 * year_score
        + 0.08 * doi_score
    )


def nobel_query_variants(seed: NobelSeed) -> List[str]:
    motivation_terms = tokens(seed.motivation)
    variants = [
        " ".join(seed.laureates[:2] + motivation_terms[:8]),
        " ".join(seed.laureates[:3]),
        " ".join(motivation_terms[:10]),
    ]
    variants.extend(" ".join([name] + motivation_terms[:6]) for name in seed.laureates[:3])
    seen: set[str] = set()
    out: List[str] = []
    for query in variants:
        query = re.sub(r"\s+", " ", query).strip()
        if len(query) < 4 or query.lower() in seen:
            continue
        seen.add(query.lower())
        out.append(query)
    return out


def find_nobel_landmarks(
    seeds: Sequence[NobelSeed],
    openalex: OpenAlexClient,
    s2: Optional[SemanticScholarClient],
    max_candidates_per_seed: int,
    min_confidence: float,
    progress: bool,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str]] = set()
    for i, seed in enumerate(seeds, start=1):
        if i == 1 or i % 25 == 0 or i == len(seeds):
            progress_log(f"Matching Nobel seed {i:,}/{len(seeds):,}: {seed.category} {seed.award_year}", progress)
        queries = nobel_query_variants(seed)
        if not queries:
            continue
        from_year = max(1900, seed.award_year - 45)
        to_year = seed.award_year - 1
        filters = [
            f"from_publication_date:{from_year}-01-01",
            f"to_publication_date:{to_year}-12-31",
            "type:article|preprint|review|book-chapter|book",
            "is_retracted:false",
            "is_paratext:false",
        ]
        candidates: List[Dict[str, Any]] = []
        for query in queries:
            try:
                candidates.extend(
                    openalex.list_works(
                        max_records=max_candidates_per_seed,
                        search=query,
                        filters=filters,
                        sort="relevance_score:desc",
                    )
                )
            except Exception:
                continue
        if s2 is not None:
            s2_query = queries[0]
            for s2_work in s2.search_papers(s2_query, limit=min(20, max_candidates_per_seed)):
                doi = normalize_doi((s2_work.get("externalIds") or {}).get("DOI"))
                if doi:
                    work = openalex.get_work(doi)
                    if work:
                        candidates.append(work)
        best_by_work: Dict[str, Tuple[float, Mapping[str, Any]]] = {}
        for work in candidates:
            wid = normalize_openalex_id(work.get("id"))
            if not wid:
                continue
            score = score_landmark_match(seed, work)
            if wid not in best_by_work or score > best_by_work[wid][0]:
                best_by_work[wid] = (score, work)
        ranked = sorted(best_by_work.values(), key=lambda item: item[0], reverse=True)
        for rank, (confidence, work) in enumerate(ranked[:5], start=1):
            wid = normalize_openalex_id(work.get("id"))
            key = (seed.prize_id, wid)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "landmark_source": "nobel",
                    "source_prize_id": seed.prize_id,
                    "award_year": seed.award_year,
                    "category": seed.category,
                    "motivation": seed.motivation,
                    "laureates": "; ".join(seed.laureates),
                    "candidate_rank": rank,
                    "match_confidence": confidence,
                    "include_main": int(confidence >= min_confidence),
                    "id": wid,
                    "short_id": short_openalex_id(wid),
                    "doi": normalize_doi(work.get("doi")),
                    "title": work.get("display_name") or "",
                    "year": int(work.get("publication_year") or 0),
                    "cited_by_count": int(work.get("cited_by_count") or 0),
                    "primary_topic_id": normalize_openalex_id((work.get("primary_topic") or {}).get("id")),
                    "primary_topic": (work.get("primary_topic") or {}).get("display_name", ""),
                }
            )
    registry = pd.DataFrame(rows)
    if not registry.empty:
        registry = registry.sort_values(["include_main", "match_confidence"], ascending=[False, False]).reset_index(drop=True)
    return registry


def read_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = raw.get("landmarks", raw.get("records", raw)) if isinstance(raw, Mapping) else raw
        return list(pd.DataFrame(rows).to_dict("records"))
    return list(pd.read_csv(path).to_dict("records"))


def split_names(value: object) -> List[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value or "")
    return [part.strip() for part in re.split(r";|\|", text) if part.strip()]


def find_external_landmarks(
    path: Optional[Path],
    openalex: OpenAlexClient,
    max_candidates_per_seed: int,
    min_confidence: float,
    progress: bool,
) -> pd.DataFrame:
    """Match optional non-Nobel landmark seed rows into the shared registry schema."""
    if path is None:
        return pd.DataFrame()
    if not path.exists():
        raise FileNotFoundError(path)
    rows_in = read_records(path)
    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str]] = set()
    for idx, row in enumerate(rows_in, start=1):
        source = str(row.get("landmark_source") or row.get("source") or row.get("award") or "external").strip()
        source_id = str(row.get("source_prize_id") or row.get("source_id") or row.get("id") or f"{source}-{idx}")
        try:
            award_year = int(row.get("award_year") or row.get("year") or row.get("prize_year") or 0)
        except Exception:
            award_year = 0
        seed = NobelSeed(
            prize_id=source_id,
            award_year=award_year if award_year > 0 else 9999,
            category=str(row.get("category") or source),
            motivation=str(row.get("motivation") or row.get("description") or row.get("citation") or ""),
            laureates=split_names(row.get("laureates") or row.get("authors") or row.get("people") or ""),
        )
        progress_log(f"Matching external landmark seed {idx:,}/{len(rows_in):,}: {source_id}", progress)
        candidates: List[Dict[str, Any]] = []
        for ident_col in ("openalex_id", "work_id", "doi"):
            ident = str(row.get(ident_col) or "").strip()
            if ident:
                work = openalex.get_work(ident)
                if work:
                    candidates.append(work)
        query_base = str(row.get("title") or row.get("query") or "").strip()
        queries = [query_base] if query_base else []
        queries.extend(nobel_query_variants(seed))
        seen_queries: set[str] = set()
        filters = ["type:article|preprint|review|book-chapter|book", "is_retracted:false", "is_paratext:false"]
        if award_year > 0:
            filters.extend(
                [
                    f"from_publication_date:{max(1900, award_year - 60)}-01-01",
                    f"to_publication_date:{award_year - 1}-12-31",
                ]
            )
        for query in queries:
            query = re.sub(r"\s+", " ", query).strip()
            if not query or query.lower() in seen_queries:
                continue
            seen_queries.add(query.lower())
            try:
                candidates.extend(
                    openalex.list_works(
                        max_records=max_candidates_per_seed,
                        search=query,
                        filters=filters,
                        sort="relevance_score:desc",
                    )
                )
            except Exception:
                continue
        best_by_work: Dict[str, Tuple[float, Mapping[str, Any]]] = {}
        provided_conf = row.get("match_confidence")
        for work in candidates:
            wid = normalize_openalex_id(work.get("id"))
            if not wid:
                continue
            score = score_landmark_match(seed, work)
            if provided_conf not in (None, ""):
                try:
                    score = max(score, float(provided_conf))
                except Exception:
                    pass
            if wid not in best_by_work or score > best_by_work[wid][0]:
                best_by_work[wid] = (score, work)
        ranked = sorted(best_by_work.values(), key=lambda item: item[0], reverse=True)
        for rank, (confidence, work) in enumerate(ranked[:5], start=1):
            wid = normalize_openalex_id(work.get("id"))
            key = (source_id, wid)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "landmark_source": source,
                    "source_prize_id": source_id,
                    "award_year": award_year,
                    "category": seed.category,
                    "motivation": seed.motivation,
                    "laureates": "; ".join(seed.laureates),
                    "candidate_rank": rank,
                    "match_confidence": confidence,
                    "include_main": int(confidence >= min_confidence),
                    "id": wid,
                    "short_id": short_openalex_id(wid),
                    "doi": normalize_doi(work.get("doi")),
                    "title": work.get("display_name") or "",
                    "year": int(work.get("publication_year") or 0),
                    "cited_by_count": int(work.get("cited_by_count") or 0),
                    "primary_topic_id": normalize_openalex_id((work.get("primary_topic") or {}).get("id")),
                    "primary_topic": (work.get("primary_topic") or {}).get("display_name", ""),
                }
            )
    registry = pd.DataFrame(rows)
    if not registry.empty:
        registry = registry.sort_values(["include_main", "match_confidence"], ascending=[False, False]).reset_index(drop=True)
    return registry


def read_domain_seeds(path: Optional[Path]) -> List[DomainSeed]:
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = raw.get("domains", raw) if isinstance(raw, Mapping) else raw
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(path)
    out: List[DomainSeed] = []
    for row in df.to_dict("records"):
        name = str(row.get("display_name") or row.get("name") or row.get("query") or row.get("slug"))
        topic_id = normalize_openalex_id(row.get("topic_id") or row.get("id") or "")
        out.append(
            DomainSeed(
                slug=slugify(row.get("slug") or name),
                display_name=name,
                topic_id=topic_id,
                query=str(row.get("query") or name),
                works_count=int(row.get("works_count") or 0),
                field_name=str(row.get("field_name") or row.get("field") or ""),
                subfield_name=str(row.get("subfield_name") or row.get("subfield") or ""),
            )
        )
    return out


def landmark_priority_domains(
    landmark_registry: pd.DataFrame,
    openalex: OpenAlexClient,
    max_domains: int,
    min_works_count: int,
    progress: bool,
) -> List[DomainSeed]:
    """Build domain seeds from matched landmark primary OpenAlex topics."""
    if landmark_registry.empty or "primary_topic_id" not in landmark_registry.columns:
        return []
    lm = landmark_registry.copy()
    if "include_main" in lm.columns:
        lm = lm[pd.to_numeric(lm["include_main"], errors="coerce").fillna(0).astype(int) == 1]
    lm["primary_topic_id"] = lm["primary_topic_id"].map(normalize_openalex_id)
    lm = lm[lm["primary_topic_id"].astype(str) != ""]
    if lm.empty:
        return []
    grouped = (
        lm.groupby("primary_topic_id", as_index=False)
        .agg(
            landmark_count=("id", "nunique"),
            mean_confidence=("match_confidence", "mean"),
            max_cited_by_count=("cited_by_count", "max"),
            primary_topic=("primary_topic", "first"),
        )
        .sort_values(
            ["landmark_count", "mean_confidence", "max_cited_by_count"],
            ascending=[False, False, False],
        )
    )
    seeds: List[DomainSeed] = []
    for row in grouped.itertuples(index=False):
        if len(seeds) >= int(max_domains):
            break
        topic_id = str(getattr(row, "primary_topic_id"))
        topic = openalex.get_topic(topic_id)
        if topic:
            seed = domain_seed_from_openalex_topic(topic, fallback_name=str(getattr(row, "primary_topic", "")))
        else:
            seed = DomainSeed(
                slug=slugify(getattr(row, "primary_topic", "") or topic_id),
                display_name=str(getattr(row, "primary_topic", "") or short_openalex_id(topic_id)),
                topic_id=topic_id,
                query=str(getattr(row, "primary_topic", "") or short_openalex_id(topic_id)),
                works_count=0,
            )
        if seed.works_count and seed.works_count < int(min_works_count):
            continue
        seeds.append(seed)
    progress_log(f"Discovered {len(seeds):,} landmark-priority OpenAlex topic domains.", progress)
    return seeds


def merge_domain_seeds(primary: Sequence[DomainSeed], fallback: Sequence[DomainSeed], max_domains: int) -> List[DomainSeed]:
    out: List[DomainSeed] = []
    seen: set[str] = set()
    for seed in list(primary) + list(fallback):
        key = seed.topic_id or seed.slug
        if not key or key in seen:
            continue
        out.append(seed)
        seen.add(key)
        if len(out) >= int(max_domains):
            break
    return out


def normalize_work_for_fig3(work: Mapping[str, Any], domain: DomainSeed, landmark_ids: set[str]) -> Dict[str, Any]:
    wid = normalize_openalex_id(work.get("id"))
    primary = work.get("primary_topic") or {}
    topic_id = normalize_openalex_id(primary.get("id"))
    topic_name = str(primary.get("display_name") or domain.display_name)
    field_obj = primary.get("field") or {}
    subfield_obj = primary.get("subfield") or {}
    primary_field = str(
        subfield_obj.get("display_name")
        or field_obj.get("display_name")
        or domain.subfield_name
        or domain.field_name
        or domain.display_name
    )
    return {
        "id": wid,
        "short_id": short_openalex_id(wid),
        "doi": normalize_doi(work.get("doi")),
        "title": work.get("display_name") or "",
        "year": int(work.get("publication_year") or 0),
        "domain": domain.slug,
        "primary_field": primary_field,
        "display_community": stable_int_id(topic_id or topic_name, modulo=100_000_000),
        "display_topic_id": topic_id,
        "display_topic_label": topic_name,
        "is_landmark": int(wid in landmark_ids),
        "document_type": work.get("type") or "",
        "cited_by_count": int(work.get("cited_by_count") or 0),
        "reference_count": len(work.get("referenced_works") or []),
    }


def collect_domain_works(
    domain: DomainSeed,
    landmarks: pd.DataFrame,
    openalex: OpenAlexClient,
    max_papers_per_domain: int,
    max_anchor_citers: int,
    start_year: int,
    end_year: int,
    work_types: Sequence[str],
    progress: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    type_filter = "type:" + "|".join(work_types)
    filters = [
        f"from_publication_date:{int(start_year)}-01-01",
        f"to_publication_date:{int(end_year)}-12-31",
        "language:en",
        type_filter,
        "is_retracted:false",
        "is_paratext:false",
    ]
    if domain.topic_id:
        filters.append(f"primary_topic.id:{short_openalex_id(domain.topic_id)}")
    topic_quota = max(100, int(max_papers_per_domain) * 70 // 100)
    works = openalex.list_works(
        max_records=topic_quota,
        search=None if domain.topic_id else domain.query,
        filters=filters,
        sort="cited_by_count:desc",
    )
    by_id: Dict[str, Dict[str, Any]] = {
        normalize_openalex_id(work.get("id")): work
        for work in works
        if normalize_openalex_id(work.get("id"))
    }

    landmark_rows = landmarks[
        (landmarks["include_main"].astype(int) == 1)
        & (
            (landmarks["primary_topic_id"].astype(str) == domain.topic_id)
            | (landmarks["primary_topic"].astype(str).str.lower() == domain.display_name.lower())
        )
    ].copy() if not landmarks.empty else pd.DataFrame()
    landmark_work_ids: List[str] = []
    for row in landmark_rows.to_dict("records"):
        work = openalex.get_work(str(row["id"]))
        if work:
            wid = normalize_openalex_id(work.get("id"))
            by_id[wid] = work
            if wid:
                landmark_work_ids.append(wid)
            if max_anchor_citers > 0:
                sid = short_openalex_id(work.get("id"))
                citer_filters = [
                    f"from_publication_date:{int(row.get('year') or start_year)}-01-01",
                    f"to_publication_date:{int(end_year)}-12-31",
                    "language:en",
                    type_filter,
                    "is_retracted:false",
                    "is_paratext:false",
                ]
                # OpenAlex has used citation-neighborhood filters in a few forms;
                # try the current common spelling first and fall back gracefully.
                citer_candidates: List[Dict[str, Any]] = []
                for citation_filter in (f"cites:{sid}", f"cited_by:{sid}"):
                    try:
                        citer_candidates = openalex.list_works(
                            max_records=max_anchor_citers,
                            filters=citer_filters + [citation_filter],
                            sort="cited_by_count:desc",
                        )
                        if citer_candidates:
                            break
                    except Exception:
                        continue
                for citer in citer_candidates:
                    cid = normalize_openalex_id(citer.get("id"))
                    if cid:
                        by_id[cid] = citer

    ordered_ids: List[str] = []
    seen_ordered: set[str] = set()
    for wid in landmark_work_ids:
        if wid and wid in by_id and wid not in seen_ordered:
            ordered_ids.append(wid)
            seen_ordered.add(wid)
    remaining_ids = sorted(
        [wid for wid in by_id if wid not in seen_ordered],
        key=lambda wid: int(by_id[wid].get("cited_by_count") or 0),
        reverse=True,
    )
    ordered_ids.extend(remaining_ids)
    selected = [by_id[wid] for wid in ordered_ids[: int(max_papers_per_domain)]]
    selected_ids = {normalize_openalex_id(work.get("id")) for work in selected}
    landmark_ids = set(landmark_rows["id"].astype(str).tolist()) & selected_ids
    works_rows = [normalize_work_for_fig3(work, domain, landmark_ids) for work in selected]
    refs_rows: List[Dict[str, str]] = []
    for work in selected:
        source = normalize_openalex_id(work.get("id"))
        for ref in work.get("referenced_works") or []:
            target = normalize_openalex_id(ref)
            if source and target and target in selected_ids:
                refs_rows.append({"source": source, "target": target})
    report = {
        "domain": domain.slug,
        "display_name": domain.display_name,
        "topic_id": domain.topic_id,
        "works_rows": len(works_rows),
        "citation_rows": len(refs_rows),
        "landmark_rows": len(landmark_ids),
    }
    return pd.DataFrame(works_rows), pd.DataFrame(refs_rows), report


def build_topics_and_edges(works: pd.DataFrame, citations: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if works.empty:
        return pd.DataFrame(columns=["community", "label", "x", "y"]), pd.DataFrame(columns=["source_community", "target_community", "weight"])
    topic_cols = ["display_community", "display_topic_label", "domain", "display_topic_id"]
    topics = (
        works[topic_cols]
        .drop_duplicates("display_community")
        .rename(columns={"display_community": "community", "display_topic_label": "label", "display_topic_id": "topic_id"})
        .copy()
    )
    n = len(topics)
    if n:
        angles = [2.0 * math.pi * i / n for i in range(n)]
        topics["x"] = [math.cos(a) for a in angles]
        topics["y"] = [math.sin(a) for a in angles]
    if citations.empty:
        edges = pd.DataFrame(columns=["source_community", "target_community", "weight"])
    else:
        comm = works.set_index("id")["display_community"].to_dict()
        tmp = citations.copy()
        tmp["source_community"] = tmp["source"].map(comm)
        tmp["target_community"] = tmp["target"].map(comm)
        tmp = tmp.dropna(subset=["source_community", "target_community"])
        tmp = tmp[tmp["source_community"] != tmp["target_community"]]
        edges = (
            tmp.groupby(["source_community", "target_community"], as_index=False)
            .size()
            .rename(columns={"size": "weight"})
        )
    return topics[["community", "label", "x", "y", "domain", "topic_id"]], edges


def dataset_summary(works: pd.DataFrame, citations: pd.DataFrame, reports: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for domain, sub in works.groupby("domain", sort=True):
        csub = citations[citations["source"].isin(set(sub["id"].astype(str)))] if not citations.empty else pd.DataFrame()
        rows.append(
            {
                "domain": domain,
                "n_papers": int(len(sub)),
                "n_landmarks": int(pd.to_numeric(sub["is_landmark"], errors="coerce").fillna(0).sum()),
                "year_min": int(sub["year"].min()) if len(sub) else 0,
                "year_max": int(sub["year"].max()) if len(sub) else 0,
                "citation_rows": int(len(csub)),
                "citation_coverage": float(len(csub) / max(1, len(sub))),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build strong-evidence Fig. 3 datasets from Nobel/OpenAlex/S2 sources.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/kg_perturbation_fig3_auto"), help="Output directory.")
    parser.add_argument("--domain-seeds", type=Path, default=None, help="Optional CSV/JSON domain seed file.")
    parser.add_argument("--max-domains", type=int, default=20, help="Number of OpenAlex topic domains to build.")
    parser.add_argument("--min-topic-works", type=int, default=50000, help="Minimum OpenAlex works_count for discovered topics.")
    parser.add_argument("--papers-per-domain", type=int, default=3000, help="Target works per domain.")
    parser.add_argument("--max-anchor-citers", type=int, default=500, help="Maximum citing works to add per landmark.")
    parser.add_argument("--start-year", type=int, default=1980)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--nobel-from-year", type=int, default=1980)
    parser.add_argument("--nobel-to-year", type=int, default=2025)
    parser.add_argument("--min-landmark-confidence", type=float, default=0.75)
    parser.add_argument("--max-candidates-per-nobel", type=int, default=50)
    parser.add_argument("--extra-landmarks", type=Path, default=None, help="Optional CSV/JSON non-Nobel landmark seed file.")
    parser.add_argument("--work-types", nargs="+", default=["article", "preprint", "review", "book-chapter", "book"])
    parser.add_argument("--openalex-api-key", default=os.getenv("OPENALEX_API_KEY"))
    parser.add_argument("--openalex-email", default=os.getenv("OPENALEX_EMAIL"))
    parser.add_argument("--s2-api-key", default=os.getenv("S2_API_KEY"))
    parser.add_argument("--use-semantic-scholar", action="store_true", help="Use Semantic Scholar as optional landmark-match enhancer.")
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    progress = not args.quiet
    args.out_dir.mkdir(parents=True, exist_ok=True)
    openalex = OpenAlexClient(
        api_key=args.openalex_api_key,
        email=args.openalex_email,
        sleep_seconds=args.sleep_seconds,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
    )
    nobel_client = RestClient(
        NOBEL_API_BASE,
        sleep_seconds=args.sleep_seconds,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
    )
    s2 = None
    if args.use_semantic_scholar:
        s2 = SemanticScholarClient(
            api_key=args.s2_api_key,
            sleep_seconds=args.sleep_seconds,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
        )

    nobel_seeds = fetch_nobel_seeds(nobel_client, args.nobel_from_year, args.nobel_to_year, progress)
    landmark_registry = find_nobel_landmarks(
        nobel_seeds,
        openalex=openalex,
        s2=s2,
        max_candidates_per_seed=args.max_candidates_per_nobel,
        min_confidence=args.min_landmark_confidence,
        progress=progress,
    )
    extra_landmarks = find_external_landmarks(
        args.extra_landmarks,
        openalex=openalex,
        max_candidates_per_seed=args.max_candidates_per_nobel,
        min_confidence=args.min_landmark_confidence,
        progress=progress,
    )
    if not extra_landmarks.empty:
        landmark_registry = pd.concat([landmark_registry, extra_landmarks], ignore_index=True, sort=False)
        landmark_registry = landmark_registry.drop_duplicates(["source_prize_id", "id"]).reset_index(drop=True)
    landmark_registry.to_csv(args.out_dir / "landmark_registry.csv", index=False)

    seeds = read_domain_seeds(args.domain_seeds)
    landmark_domain_count = 0
    if not seeds:
        progress_log("Discovering landmark-priority OpenAlex topic domains...", progress)
        landmark_domains = landmark_priority_domains(
            landmark_registry,
            openalex=openalex,
            max_domains=args.max_domains,
            min_works_count=args.min_topic_works,
            progress=progress,
        )
        landmark_domain_count = len(landmark_domains)
        fallback_domains: List[DomainSeed] = []
        if len(landmark_domains) < int(args.max_domains):
            fallback_limit = max(int(args.max_domains) * 3, int(args.max_domains) + 10)
            fallback_domains = openalex.discover_topics(
                max_topics=fallback_limit,
                min_works_count=args.min_topic_works,
            )
        seeds = merge_domain_seeds(landmark_domains, fallback_domains, args.max_domains)
    seeds = seeds[: int(args.max_domains)]
    pd.DataFrame([seed.__dict__ for seed in seeds]).to_csv(args.out_dir / "domain_seeds.csv", index=False)

    all_works: List[pd.DataFrame] = []
    all_citations: List[pd.DataFrame] = []
    reports: List[Dict[str, Any]] = []
    for i, seed in enumerate(seeds, start=1):
        progress_log(f"[{i}/{len(seeds)}] Collecting domain {seed.slug} ({seed.display_name})", progress)
        works, citations, report = collect_domain_works(
            seed,
            landmark_registry,
            openalex=openalex,
            max_papers_per_domain=args.papers_per_domain,
            max_anchor_citers=args.max_anchor_citers,
            start_year=args.start_year,
            end_year=args.end_year,
            work_types=args.work_types,
            progress=progress,
        )
        all_works.append(works)
        all_citations.append(citations)
        reports.append(report)
    works_df = pd.concat(all_works, ignore_index=True) if all_works else pd.DataFrame()
    works_df = works_df.drop_duplicates("id").reset_index(drop=True)
    selected_ids = set(works_df["id"].astype(str)) if not works_df.empty else set()
    citations_df = pd.concat(all_citations, ignore_index=True) if all_citations else pd.DataFrame(columns=["source", "target"])
    if not citations_df.empty:
        citations_df = citations_df[
            citations_df["source"].astype(str).isin(selected_ids)
            & citations_df["target"].astype(str).isin(selected_ids)
        ].drop_duplicates().reset_index(drop=True)
    topics_df, topic_edges_df = build_topics_and_edges(works_df, citations_df)

    works_df.to_csv(args.out_dir / "works.csv", index=False)
    citations_df.to_csv(args.out_dir / "citations.csv", index=False)
    topics_df.to_csv(args.out_dir / "topics.csv", index=False)
    topic_edges_df.to_csv(args.out_dir / "topic_edges.csv", index=False)
    summary_df = dataset_summary(works_df, citations_df, reports)
    summary_df.to_csv(args.out_dir / "fig3_dataset_summary.csv", index=False)
    report = {
        "source_kind": "automatic_landmark_topic_dataset",
        "n_domains": int(summary_df["domain"].nunique()) if not summary_df.empty else 0,
        "works_rows": int(len(works_df)),
        "citation_rows": int(len(citations_df)),
        "topic_rows": int(len(topics_df)),
        "topic_edge_rows": int(len(topic_edges_df)),
        "landmark_registry_rows": int(len(landmark_registry)),
        "main_landmark_rows": int(landmark_registry["include_main"].sum()) if not landmark_registry.empty else 0,
        "landmark_priority_domain_rows": int(landmark_domain_count),
        "parameters": {
            "max_domains": args.max_domains,
            "min_topic_works": args.min_topic_works,
            "papers_per_domain": args.papers_per_domain,
            "max_anchor_citers": args.max_anchor_citers,
            "start_year": args.start_year,
            "end_year": args.end_year,
            "min_landmark_confidence": args.min_landmark_confidence,
        },
        "domain_reports": reports,
    }
    (args.out_dir / "dataset_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    progress_log(f"Done. Wrote Fig. 3 dataset to {args.out_dir}", progress)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gear.corpus import (  # noqa: E402
    DEFAULT_COMPLETE_END_YEAR,
    apply_strict_anchor_policy,
    audit_corpus,
    build_topics_and_edges,
    make_views,
    normalize_doi,
    normalize_openalex_id,
    root_work_columns,
    short_openalex_id,
    slugify,
    stable_int_id,
)
from gear.env import getenv  # noqa: E402
from scripts.publication_corpus_v2 import (  # noqa: E402
    FIGURE_LOGIC_POLICY,
    deduplicate_domain_dois,
    nonempty,
    openalex_search_query_from_domain_query,
    read_csv,
    safe_numeric,
    utc_now,
    write_json,
)


DEFAULT_DOMAIN_SEED_CSV = (
    PROJECT_ROOT
    / "data"
    / "knowledge_corpus"
    / "v1_strict_landmark_external_topup_dedup_protein2_meta_topup_ready_refsupport_candidates_v3e_magnetic_manual_topup"
    / "domains.csv"
)
DEFAULT_REGISTRY_CSV = PROJECT_ROOT / "data" / "knowledge_corpus" / "landmark_registry_v3.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "knowledge_corpus" / "v3_openalex_graph"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "outputs" / "openalex_v3_citation_graph"
OPENALEX_API_BASE = "https://api.openalex.org"
OPENALEX_WORK_SELECT = ",".join(
    [
        "id",
        "doi",
        "display_name",
        "publication_year",
        "type",
        "language",
        "cited_by_count",
        "referenced_works",
        "primary_topic",
    ]
)
OPENALEX_WORK_TYPES = ["article", "preprint", "review", "book-chapter", "book"]
SOURCE_PRIORITY = {"landmark_exact": 0, "anchor_citer": 1, "query_core": 2}


def split_api_keys(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for item in value:
            out.extend(split_api_keys(item))
        return list(dict.fromkeys(out))
    text = str(value or "").strip()
    if not text:
        return []
    return list(dict.fromkeys(part.strip() for part in re.split(r"[,;\s]+", text) if part.strip()))


class OpenAlexClient:
    """Tiny standard-library OpenAlex client for reproducible corpus fetches."""

    def __init__(
        self,
        api_key: Optional[str],
        api_keys: Optional[Sequence[str]],
        email: Optional[str],
        sleep_seconds: float,
        timeout_seconds: int,
        max_retries: int,
    ) -> None:
        self.api_keys = split_api_keys([api_key, api_keys])
        self.email = email
        self.sleep_seconds = float(sleep_seconds)
        self.timeout_seconds = int(timeout_seconds)
        self.max_retries = int(max_retries)
        self._api_key_index = 0
        self._api_key_lock = threading.Lock()

    def next_api_key(self) -> Optional[str]:
        if not self.api_keys:
            return None
        with self._api_key_lock:
            key = self.api_keys[self._api_key_index % len(self.api_keys)]
            self._api_key_index += 1
        return key

    def get_json(self, endpoint: str, params: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        url = endpoint if endpoint.startswith("http") else f"{OPENALEX_API_BASE}/{endpoint.lstrip('/')}"
        base_params = dict(params or {})
        for attempt in range(self.max_retries + 1):
            params_dict = dict(base_params)
            key = self.next_api_key()
            if key:
                params_dict["api_key"] = key
            if self.email:
                params_dict["mailto"] = self.email
            query = urllib.parse.urlencode(params_dict)
            full_url = f"{url}?{query}" if query else url
            if self.sleep_seconds > 0:
                time.sleep(self.sleep_seconds)
            request = urllib.request.Request(full_url, headers={"User-Agent": "ASPR OpenAlex v3 graph builder"})
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                    retry_after = exc.headers.get("Retry-After")
                    time.sleep(float(retry_after) if retry_after else min(60.0, 2.0**attempt))
                    continue
                raise
            except urllib.error.URLError:
                if attempt >= self.max_retries:
                    raise
                time.sleep(min(60.0, 2.0**attempt))
        raise RuntimeError(f"Retries exhausted: {url}")

    def list_works(
        self,
        max_records: int,
        search: Optional[str] = None,
        filters: Optional[Sequence[str]] = None,
        sort: Optional[str] = None,
        per_page: int = 200,
        progress: bool = False,
        progress_label: str = "",
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {
            "select": OPENALEX_WORK_SELECT,
            "per-page": min(200, max(10, int(per_page))),
            "cursor": "*",
        }
        filt = [item for item in (filters or []) if item]
        if filt:
            params["filter"] = ",".join(filt)
        if search:
            params["search"] = search
        if sort:
            params["sort"] = sort

        rows: List[Dict[str, Any]] = []
        page = 0
        while len(rows) < int(max_records):
            payload = self.get_json("/works", params=params)
            page += 1
            results = payload.get("results") or []
            if not results:
                break
            rows.extend(results[: int(max_records) - len(rows)])
            if progress and (page == 1 or page % 5 == 0 or len(rows) >= int(max_records)):
                label = progress_label or "OpenAlex works"
                print(f"{label}：已拉取 {len(rows):,}/{int(max_records):,} 行", flush=True)
            next_cursor = (payload.get("meta") or {}).get("next_cursor")
            if not next_cursor or len(results) < int(params["per-page"]):
                break
            params["cursor"] = next_cursor
        return rows

    def get_work(self, identifier: str) -> Optional[Dict[str, Any]]:
        ident = str(identifier or "").strip()
        if not ident:
            return None
        if ident.lower().startswith("10.") or "doi.org" in ident.lower():
            doi = normalize_doi(ident)
            key = urllib.parse.quote(f"https://doi.org/{doi}", safe="")
        else:
            key = short_openalex_id(ident)
        try:
            return self.get_json(f"/works/{key}", params={"select": OPENALEX_WORK_SELECT})
        except Exception:
            return None


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
            n_rows += 1
    return n_rows


def load_domain_metadata(path: Path) -> Dict[str, Dict[str, Any]]:
    frame = read_csv(path)
    out: Dict[str, Dict[str, Any]] = {}
    if frame.empty:
        return out
    for row in frame.to_dict("records"):
        domain = slugify(row.get("slug") or row.get("domain") or row.get("display_name"))
        if not domain:
            continue
        out[domain] = dict(row)
    return out


def domain_meta_for(domain: str, metadata: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    meta = dict(metadata.get(domain, {}))
    meta.setdefault("slug", domain)
    meta.setdefault("display_name", domain.replace("_", " "))
    meta.setdefault("query", meta.get("display_name", domain.replace("_", " ")))
    meta.setdefault("field_name", "")
    meta.setdefault("subfield_name", "")
    meta.setdefault("topic_id", "")
    return meta


def filters_for_domain(start_year: int, end_year: int, work_types: Sequence[str]) -> List[str]:
    return [
        f"from_publication_date:{int(start_year)}-01-01",
        f"to_publication_date:{int(end_year)}-12-31",
        "language:en",
        "type:" + "|".join(work_types),
        "is_retracted:false",
        "is_paratext:false",
    ]


def add_record(
    records: List[Dict[str, Any]],
    domain: str,
    work: Optional[Mapping[str, Any]],
    source_kind: str,
    anchor_label: str = "",
) -> None:
    if not work:
        return
    work_id = normalize_openalex_id(work.get("id"))
    if not work_id:
        return
    records.append(
        {
            "domain": slugify(domain),
            "source_kind": source_kind,
            "anchor_label": nonempty(anchor_label),
            "work": dict(work),
        }
    )


def fetch_domain_records(
    domain: str,
    landmarks: pd.DataFrame,
    meta: Mapping[str, Any],
    openalex: OpenAlexClient,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    base_filters = filters_for_domain(args.start_year, args.end_year, args.work_types)

    for row in landmarks.to_dict("records"):
        work = None
        for key in ["doi", "openalex_id", "id"]:
            value = row.get(key)
            if nonempty(value):
                work = openalex.get_work(str(value))
                if work:
                    break
        add_record(records, domain, work, "landmark_exact", anchor_label=row.get("label", ""))

    query = openalex_search_query_from_domain_query(meta.get("query") or meta.get("display_name") or domain)
    if query:
        query_records = openalex.list_works(
            max_records=args.papers_per_domain,
            search=query,
            filters=base_filters,
            sort="cited_by_count:desc",
            progress=not args.quiet,
            progress_label=f"[{domain}] query core",
        )
        for work in query_records:
            add_record(records, domain, work, "query_core")

    if args.max_anchor_citers > 0:
        landmark_records = [record for record in records if record.get("source_kind") == "landmark_exact"]
        for idx, record in enumerate(landmark_records, start=1):
            work = record.get("work") or {}
            sid = short_openalex_id(work.get("id"))
            if not sid:
                continue
            year = int(safe_numeric(work.get("publication_year"), default=args.start_year))
            citer_filters = filters_for_domain(max(args.start_year, year), args.end_year, args.work_types)
            for citation_filter in (f"cites:{sid}", f"cited_by:{sid}"):
                try:
                    citer_records = openalex.list_works(
                        max_records=args.max_anchor_citers,
                        filters=citer_filters + [citation_filter],
                        sort="cited_by_count:desc",
                        progress=not args.quiet,
                        progress_label=f"[{domain}] landmark {idx} citers",
                    )
                except Exception:
                    citer_records = []
                if citer_records:
                    for citer in citer_records:
                        add_record(records, domain, citer, "anchor_citer")
                    break
    return records


def deduplicate_records(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for record in records:
        work = record.get("work") or {}
        work_id = normalize_openalex_id(work.get("id"))
        if not work_id:
            continue
        current = by_id.get(work_id)
        priority = SOURCE_PRIORITY.get(str(record.get("source_kind")), 99)
        current_priority = SOURCE_PRIORITY.get(str((current or {}).get("source_kind")), 99)
        if current is None or priority < current_priority:
            by_id[work_id] = dict(record)
        elif current is not None and nonempty(record.get("anchor_label")) and not nonempty(current.get("anchor_label")):
            current["anchor_label"] = nonempty(record.get("anchor_label"))
    return list(by_id.values())


def select_records(records: Sequence[Mapping[str, Any]], max_records: int) -> List[Dict[str, Any]]:
    unique = deduplicate_records(records)
    landmarks = [record for record in unique if record.get("source_kind") == "landmark_exact"]
    landmark_ids = {normalize_openalex_id((record.get("work") or {}).get("id")) for record in landmarks}
    others = [record for record in unique if normalize_openalex_id((record.get("work") or {}).get("id")) not in landmark_ids]
    others = sorted(
        others,
        key=lambda record: (
            SOURCE_PRIORITY.get(str(record.get("source_kind")), 99),
            -int(safe_numeric((record.get("work") or {}).get("cited_by_count"), default=0)),
            int(safe_numeric((record.get("work") or {}).get("publication_year"), default=9999)),
        ),
    )
    selected = landmarks + others
    return [dict(record) for record in selected[: int(max_records)]]


def work_to_row(
    record: Mapping[str, Any],
    meta: Mapping[str, Any],
    landmark_labels: Mapping[str, str],
    fetched_at: str,
    landmark_source_label: str = "landmark_registry_v3",
    source_dataset_prefix: str = "openalex_v3",
) -> Dict[str, Any]:
    work = record.get("work") or {}
    work_id = normalize_openalex_id(work.get("id"))
    primary = work.get("primary_topic") or {}
    topic_id = normalize_openalex_id(primary.get("id"))
    topic_name = nonempty(primary.get("display_name")) or nonempty(meta.get("display_name")) or slugify(record.get("domain"))
    field_obj = primary.get("field") or {}
    subfield_obj = primary.get("subfield") or {}
    primary_field = (
        nonempty(subfield_obj.get("display_name"))
        or nonempty(field_obj.get("display_name"))
        or nonempty(meta.get("subfield_name"))
        or nonempty(meta.get("field_name"))
    )
    refs = [normalize_openalex_id(ref) for ref in (work.get("referenced_works") or []) if normalize_openalex_id(ref)]
    year = int(safe_numeric(work.get("publication_year"), default=0))
    label = landmark_labels.get(work_id, "")
    is_landmark = int(bool(label))
    source_kind = nonempty(record.get("source_kind")) or "query_core"
    return {
        "id": work_id,
        "short_id": short_openalex_id(work_id),
        "doi": normalize_doi(work.get("doi")),
        "title": nonempty(work.get("display_name")) or work_id,
        "year": year,
        "domain": slugify(record.get("domain")),
        "primary_field": primary_field,
        "display_community": stable_int_id(f"{record.get('domain')}:{topic_id or topic_name}", modulo=100_000_000),
        "display_topic_id": topic_id,
        "display_topic_label": topic_name,
        "legacy_is_landmark": is_landmark,
        "is_landmark": is_landmark,
        "anchor_label": label,
        "reliable_anchor_source": landmark_source_label if is_landmark else "",
        "anchor_policy": "strict",
        "document_type": nonempty(work.get("type")),
        "cited_by_count": int(safe_numeric(work.get("cited_by_count"), default=0)),
        "reference_count": int(len(refs)),
        "source_provider": "openalex",
        "source_dataset": f"{source_dataset_prefix}_{source_kind}",
        "fetched_at": fetched_at,
        "referenced_works": json.dumps(refs, ensure_ascii=False),
        "partial_2026": int(year >= 2026),
    }


def citation_rows_from_records(
    records: Sequence[Mapping[str, Any]],
    selected_ids: set[str],
    local_references_only: bool,
    source_dataset_prefix: str = "openalex_v3",
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for record in records:
        work = record.get("work") or {}
        source = normalize_openalex_id(work.get("id"))
        if not source:
            continue
        for ref in work.get("referenced_works") or []:
            target = normalize_openalex_id(ref)
            if not target:
                continue
            if local_references_only and target not in selected_ids:
                continue
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "relation": "reference",
                    "source_dataset": f"{source_dataset_prefix}_{nonempty(record.get('source_kind')) or 'query_core'}",
                }
            )
    return rows


def trim_domain_tables(
    works: pd.DataFrame,
    citations: pd.DataFrame,
    max_works: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep a fixed-size domain table after DOI dedupe, preserving landmarks first."""
    if works.empty or len(works) <= int(max_works):
        return works.copy(), citations.copy()
    out = works.copy()
    out["_is_landmark_sort"] = pd.to_numeric(out.get("is_landmark", 0), errors="coerce").fillna(0).astype(int)
    out["_cited_sort"] = pd.to_numeric(out.get("cited_by_count", 0), errors="coerce").fillna(0)
    out["_year_sort"] = pd.to_numeric(out.get("year", 9999), errors="coerce").fillna(9999)
    out = out.sort_values(
        ["_is_landmark_sort", "_cited_sort", "_year_sort", "id"],
        ascending=[False, False, True, True],
    ).head(int(max_works))
    keep_ids = set(out["id"].astype(str))
    out = out.drop(columns=["_is_landmark_sort", "_cited_sort", "_year_sort"], errors="ignore").reset_index(drop=True)
    kept_citations = citations.copy()
    if not kept_citations.empty and "source" in kept_citations.columns:
        kept_citations = kept_citations[kept_citations["source"].astype(str).isin(keep_ids)].copy()
    return out, kept_citations.reset_index(drop=True)


def global_deduplicate_works(
    works: pd.DataFrame,
    citations: pd.DataFrame,
    domains: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Assign each OpenAlex work id to one domain, preserving landmarks first."""
    if works.empty or "id" not in works.columns or not works["id"].duplicated().any():
        return works.copy(), citations.copy(), {"dropped_duplicate_works": 0}
    domain_order = {slugify(domain): idx for idx, domain in enumerate(domains)}
    out = works.copy()
    out["_domain_order"] = out.get("domain", "").astype(str).map(slugify).map(domain_order).fillna(len(domain_order))
    out["_is_landmark_sort"] = pd.to_numeric(out.get("is_landmark", 0), errors="coerce").fillna(0).astype(int)
    out["_cited_sort"] = pd.to_numeric(out.get("cited_by_count", 0), errors="coerce").fillna(0)
    out["_year_sort"] = pd.to_numeric(out.get("year", 9999), errors="coerce").fillna(9999)
    out = out.sort_values(
        ["_is_landmark_sort", "_cited_sort", "_domain_order", "_year_sort", "id"],
        ascending=[False, False, True, True, True],
    )
    before = int(len(out))
    out = out.drop_duplicates("id", keep="first")
    dropped = before - int(len(out))
    out = out.drop(columns=["_domain_order", "_is_landmark_sort", "_cited_sort", "_year_sort"], errors="ignore")
    keep_ids = set(out["id"].astype(str))
    kept_citations = citations.copy()
    if not kept_citations.empty and "source" in kept_citations.columns:
        kept_citations = kept_citations[kept_citations["source"].astype(str).isin(keep_ids)].copy()
    return out.reset_index(drop=True), kept_citations.reset_index(drop=True), {"dropped_duplicate_works": int(dropped)}


def standardize_landmarks(registry: pd.DataFrame, landmark_source_label: str = "strict_manual_v3") -> pd.DataFrame:
    columns = [
        "domain",
        "landmark_source",
        "source_id",
        "label",
        "id",
        "doi",
        "title",
        "year",
        "match_confidence",
        "include_main",
        "accepted_landmark_source",
        "needs_manual_confirmation",
        "evidence_key",
        "authority_basis",
        "authority_note",
        "doi_url",
    ]
    out = registry.copy()
    out["domain"] = out["domain"].map(slugify)
    out["id"] = out.get("openalex_id", out.get("id", "")).map(normalize_openalex_id)
    out["doi"] = out.get("doi", "").map(normalize_doi)
    out["source_id"] = out.get("source_id", out["doi"])
    out["match_confidence"] = pd.to_numeric(out.get("title_similarity", 1.0), errors="coerce").fillna(1.0)
    out["include_main"] = 1
    out["landmark_source"] = landmark_source_label
    out["accepted_landmark_source"] = landmark_source_label
    out["needs_manual_confirmation"] = 0
    out["evidence_key"] = out["doi"].where(out["doi"].astype(str) != "", out["id"])
    for col in columns:
        if col not in out.columns:
            out[col] = ""
    return out[columns].copy()


def build_graph(args: argparse.Namespace) -> Dict[str, Any]:
    registry = read_csv(args.registry_csv)
    if registry.empty:
        raise ValueError(f"No landmark registry rows found: {args.registry_csv}")
    registry["domain"] = registry["domain"].map(slugify)
    registry = registry[pd.to_numeric(registry.get("include_main", 1), errors="coerce").fillna(1).astype(int) == 1].copy()
    registry_counts = registry.groupby("domain").size()
    eligible_domains = sorted(domain for domain, count in registry_counts.items() if 1 <= int(count) <= 3)
    domains = [slugify(domain) for domain in (args.domains or eligible_domains)]
    if args.max_domains:
        domains = domains[: int(args.max_domains)]
    if not domains:
        raise ValueError("No domains selected for OpenAlex v3 graph fetch")

    metadata = load_domain_metadata(args.domain_seed_csv)
    openalex = OpenAlexClient(
        api_key=args.openalex_api_key,
        api_keys=split_api_keys(args.openalex_api_keys),
        email=args.openalex_email,
        sleep_seconds=args.sleep_seconds,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.checkpoint_dir or (args.report_dir / "checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    fetched_at = utc_now()
    work_parts: List[pd.DataFrame] = []
    citation_parts: List[pd.DataFrame] = []
    domain_rows: List[Dict[str, Any]] = []
    report_rows: List[Dict[str, Any]] = []
    selected_landmark_rows: List[pd.DataFrame] = []

    for index, domain in enumerate(domains, start=1):
        domain_landmarks = registry[registry["domain"].astype(str) == domain].copy()
        meta = domain_meta_for(domain, metadata)
        checkpoint = checkpoint_dir / f"{domain}.jsonl"
        if checkpoint.exists() and not args.refresh:
            records = read_jsonl(checkpoint)
            fetch_status = "checkpoint"
        else:
            print(f"[openalex-v3] fetching {index}/{len(domains)} {domain}", flush=True)
            try:
                records = fetch_domain_records(domain, domain_landmarks, meta, openalex, args)
                write_jsonl(checkpoint, records)
                fetch_status = "fetched"
            except Exception as exc:
                report_rows.append(
                    {
                        "domain": domain,
                        "status": "fetch_failed",
                        "error": str(exc),
                        "n_registry_landmarks": int(len(domain_landmarks)),
                    }
                )
                if args.fail_on_domain_error:
                    raise
                continue

        selected = select_records(records, max_records=args.papers_per_domain + args.duplicate_buffer_per_domain)
        landmark_label_lookup: Dict[str, str] = {}
        for record in selected:
            if record.get("source_kind") == "landmark_exact":
                work_id = normalize_openalex_id((record.get("work") or {}).get("id"))
                if work_id:
                    landmark_label_lookup[work_id] = nonempty(record.get("anchor_label")) or work_id
        work_rows = [
            work_to_row(
                record,
                meta,
                landmark_label_lookup,
                fetched_at,
                landmark_source_label=args.landmark_source_label,
                source_dataset_prefix=args.source_dataset_prefix,
            )
            for record in selected
        ]
        works = pd.DataFrame(work_rows)
        for col in root_work_columns():
            if col not in works.columns:
                works[col] = ""
        works = works[root_work_columns()].copy()
        selected_ids = set(works["id"].astype(str))
        citation_rows = citation_rows_from_records(
            selected,
            selected_ids,
            args.local_references_only,
            source_dataset_prefix=args.source_dataset_prefix,
        )
        citations = pd.DataFrame(citation_rows)
        if not citations.empty:
            citations = citations.drop_duplicates(["source", "target"]).reset_index(drop=True)
        works, citations, dedupe_report = deduplicate_domain_dois(works, citations)
        works, citations = trim_domain_tables(works, citations, max_works=args.papers_per_domain)
        selected_ids = set(works["id"].astype(str))
        for col in ["source", "target", "relation", "source_dataset"]:
            if col not in citations.columns:
                citations[col] = ""
        local_rows = int(citations["target"].astype(str).isin(selected_ids).sum()) if not citations.empty else 0
        local_closure = float(local_rows / max(1, len(citations))) if len(citations) else 0.0

        domain_rows.append(
            {
                "slug": domain,
                "display_name": nonempty(meta.get("display_name")) or domain.replace("_", " "),
                "query": meta.get("query", ""),
                "field_name": meta.get("field_name", ""),
                "subfield_name": meta.get("subfield_name", ""),
                "topic_id": meta.get("topic_id", ""),
                "seed_source": args.domain_seed_source,
                "n_works": int(len(works)),
            }
        )
        report_rows.append(
            {
                "domain": domain,
                "status": fetch_status,
                "query": openalex_search_query_from_domain_query(meta.get("query") or meta.get("display_name") or domain),
                "n_registry_landmarks": int(len(domain_landmarks)),
                "n_raw_records": int(len(records)),
                "n_selected_works": int(len(works)),
                "n_selected_landmarks": int(works["is_landmark"].astype(int).sum()) if not works.empty else 0,
                "n_dropped_duplicate_dois": int(dedupe_report.get("dropped_duplicate_works", 0)),
                "n_citation_rows": int(len(citations)),
                "local_reference_closure": local_closure,
                "year_min": int(pd.to_numeric(works["year"], errors="coerce").min()) if not works.empty else 0,
                "year_max": int(pd.to_numeric(works["year"], errors="coerce").max()) if not works.empty else 0,
                "checkpoint": str(checkpoint),
            }
        )
        work_parts.append(works)
        citation_parts.append(citations)
        selected_landmark_rows.append(domain_landmarks)

    all_works = pd.concat(work_parts, ignore_index=True, sort=False) if work_parts else pd.DataFrame(columns=root_work_columns())
    all_citations = (
        pd.concat(citation_parts, ignore_index=True, sort=False)
        if citation_parts
        else pd.DataFrame(columns=["source", "target", "relation", "source_dataset"])
    )
    for col in ["source", "target", "relation", "source_dataset"]:
        if col not in all_citations.columns:
            all_citations[col] = ""
    if not all_citations.empty:
        all_citations = all_citations.drop_duplicates(["source", "target"]).reset_index(drop=True)
    global_dedupe_report = {"dropped_duplicate_works": 0}
    if args.global_dedupe_work_ids:
        all_works, all_citations, global_dedupe_report = global_deduplicate_works(all_works, all_citations, domains)
        counts_by_domain = all_works["domain"].astype(str).value_counts().to_dict() if not all_works.empty else {}
        for row in domain_rows:
            row["n_works"] = int(counts_by_domain.get(str(row.get("slug")), 0))
    selected_registry = pd.concat(selected_landmark_rows, ignore_index=True, sort=False) if selected_landmark_rows else pd.DataFrame()
    landmarks = standardize_landmarks(selected_registry, landmark_source_label=args.landmark_source_label)
    all_works = apply_strict_anchor_policy(all_works, landmarks, complete_end_year=DEFAULT_COMPLETE_END_YEAR)
    topics, topic_edges = build_topics_and_edges(
        all_works,
        all_citations[all_citations["target"].astype(str).isin(set(all_works["id"].astype(str)))] if not all_citations.empty else all_citations,
    )
    domains_frame = pd.DataFrame(domain_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_works.to_csv(args.out_dir / "works.csv", index=False)
    all_citations.to_csv(args.out_dir / "citations.csv", index=False)
    domains_frame.to_csv(args.out_dir / "domains.csv", index=False)
    landmarks.to_csv(args.out_dir / "landmarks.csv", index=False)
    topics.to_csv(args.out_dir / "topics.csv", index=False)
    topic_edges.to_csv(args.out_dir / "topic_edges.csv", index=False)
    report = pd.DataFrame(report_rows)
    report.to_csv(args.report_dir / "domain_fetch_report.csv", index=False)

    quality_report = audit_corpus(args.out_dir, min_papers_per_domain=args.min_papers_per_domain)
    make_views(args.out_dir, anchor_policy="strict")
    manifest = {
        "artifact_kind": args.artifact_kind,
        "created_at": fetched_at,
        "figure_logic_policy": FIGURE_LOGIC_POLICY,
        "registry_csv": str(args.registry_csv),
        "domain_seed_csv": str(args.domain_seed_csv),
        "landmark_source_label": args.landmark_source_label,
        "source_dataset_prefix": args.source_dataset_prefix,
        "domain_seed_source": args.domain_seed_source,
        "out_dir": str(args.out_dir),
        "report_dir": str(args.report_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "domains": domains,
        "n_domains": int(all_works["domain"].nunique()) if not all_works.empty else 0,
        "n_works": int(len(all_works)),
        "n_citation_rows": int(len(all_citations)),
        "n_landmarks": int(len(landmarks)),
        "global_dedupe_work_ids": bool(args.global_dedupe_work_ids),
        "global_dedupe_report": global_dedupe_report,
        "papers_per_domain": int(args.papers_per_domain),
        "max_anchor_citers": int(args.max_anchor_citers),
        "local_references_only": bool(args.local_references_only),
        "quality_overall_pass": bool(quality_report.get("overall_pass", False)),
        "domain_fetch_report": str(args.report_dir / "domain_fetch_report.csv"),
    }
    write_json(args.out_dir / "manifest.json", manifest)
    write_json(args.report_dir / "openalex_v3_citation_graph_manifest.json", manifest)
    if args.artifact_kind != "openalex_v3_citation_graph":
        write_json(args.report_dir / f"{args.artifact_kind}_manifest.json", manifest)
    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a fresh OpenAlex citation graph from landmark registry v3.")
    parser.add_argument("--registry-csv", type=Path, default=DEFAULT_REGISTRY_CSV)
    parser.add_argument("--domain-seed-csv", type=Path, default=DEFAULT_DOMAIN_SEED_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--artifact-kind", default="openalex_v3_citation_graph")
    parser.add_argument("--landmark-source-label", default="strict_manual_v3")
    parser.add_argument("--source-dataset-prefix", default="openalex_v3")
    parser.add_argument("--domain-seed-source", default="landmark_registry_v3_openalex")
    parser.add_argument("--domains", nargs="+", default=None)
    parser.add_argument("--max-domains", type=int, default=None)
    parser.add_argument("--papers-per-domain", type=int, default=2500)
    parser.add_argument("--min-papers-per-domain", type=int, default=2500)
    parser.add_argument("--duplicate-buffer-per-domain", type=int, default=300)
    parser.add_argument("--max-anchor-citers", type=int, default=250)
    parser.add_argument("--start-year", type=int, default=1980)
    parser.add_argument("--end-year", type=int, default=DEFAULT_COMPLETE_END_YEAR)
    parser.add_argument("--work-types", nargs="+", default=OPENALEX_WORK_TYPES)
    parser.add_argument("--local-references-only", action="store_true")
    parser.add_argument("--global-dedupe-work-ids", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--fail-on-domain-error", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--openalex-api-key", default=getenv("OPENALEX_API_KEY"))
    parser.add_argument("--openalex-api-keys", default=getenv("OPENALEX_API_KEYS"))
    parser.add_argument("--openalex-email", default=getenv("OPENALEX_EMAIL"))
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    manifest = build_graph(args)
    print(
        f"[openalex-v3] wrote {manifest['n_works']} works / {manifest['n_citation_rows']} citations "
        f"across {manifest['n_domains']} domains to {args.out_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

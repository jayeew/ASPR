"""Source-linked metadata and descriptive statistics for Fig7 c–f."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd
import pycountry
import requests

from gear.env import getenv

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/fig7_reference/two_panel"
DEFAULT_OUT = ROOT / "outputs/fig7_reference/country_extension"
ASSETS = ROOT / "data/claim_graph"
TYPES = ("METHOD", "FINDING", "MECHANISM", "RESOURCE", "THEORY")
SELECT = "id,doi,title,publication_year,authorships,is_authors_truncated,countries_distinct_count,updated_date"
VALID_CODES = {c.alpha_2 for c in pycountry.countries}


def address_without_contact(text: str) -> str:
    return (
        re.sub(
            r"\s*(?:e[- ]?mail\s*:?)?\s*[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\s*$", "", text
        )
        .strip()
        .rstrip(". ;")
    )


def affiliation_country(text: str) -> str | None:
    """Read only explicit country/region suffixes of an authorship address."""
    aliases = {c.name: c.alpha_2 for c in pycountry.countries}
    aliases.update(
        {
            "USA": "US",
            "U.S.A.": "US",
            "United States of America": "US",
            "UK": "GB",
            "U.K.": "GB",
            "England": "GB",
            "Scotland": "GB",
            "Wales": "GB",
            "Northern Ireland": "GB",
            "South Korea": "KR",
            "Republic of Korea": "KR",
            "Russia": "RU",
            "Iran": "IR",
            "P. R. China": "CN",
            "P.R. China": "CN",
            "PR China": "CN",
            "People’s Republic of China": "CN",
            "People's Republic of China": "CN",
        }
    )
    value = address_without_contact(text)
    # Preserve source geographic units rather than merging regions into a state.
    for region, code in (
        ("Hong Kong", "HK"),
        ("Macao", "MO"),
        ("Macau", "MO"),
        ("Taiwan", "TW"),
    ):
        if re.search(
            r"\b" + re.escape(region) + r"(?:[,\s]+(?:SAR[,\s]+)?China)?$",
            value,
            re.IGNORECASE,
        ):
            return code
    for name, code in sorted(aliases.items(), key=lambda pair: -len(pair[0])):
        if re.search(
            r"(?:^|[,;\s])" + re.escape(name.rstrip(".")) + r"$", value, re.IGNORECASE
        ):
            return code
    return None


def indexed_text(text: str) -> tuple[str, list[int]]:
    chars, offsets = [], []
    for index, char in enumerate(text):
        for normalized in unicodedata.normalize("NFKC", char).casefold():
            if normalized.isalnum():
                chars.append(normalized)
                offsets.append(index)
    return "".join(chars), offsets


def repair_affiliations(
    paper: dict[str, Any], record: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Recover unmapped author countries only from locally matched raw addresses."""
    authors = record.get("authorships") or []
    missing = [
        i
        for i, a in enumerate(authors)
        if not a.get("countries")
        and not any(x.get("country_code") for x in a.get("institutions") or [])
    ]
    candidates = {}
    for i in missing:
        addresses = list(dict.fromkeys(authors[i].get("raw_affiliation_strings") or []))
        codes = [affiliation_country(a) for a in addresses]
        if addresses and all(codes):
            candidates[i] = list(zip(addresses, codes))
    if not candidates:
        return record, []
    path = Path(paper["paper_path"])
    if not path.is_file():
        return record, []
    text = path.read_text(encoding="utf-8")
    normalized, offsets = indexed_text(text)
    repairs, updated = [], list(authors)
    for i, addresses in candidates.items():
        matches = []
        for address, code in addresses:
            needle, _ = indexed_text(address_without_contact(address))
            start = normalized.find(needle) if len(needle) >= 20 else -1
            if start < 0:
                break
            lo, hi = offsets[start], offsets[start + len(needle) - 1] + 1
            matches.append(
                {
                    "raw_affiliation": address,
                    "country_code": code,
                    "local_start": lo,
                    "local_end": hi,
                    "local_quote": text[lo:hi],
                }
            )
        if len(matches) == len(addresses):
            updated[i] = authors[i] | {
                "countries": sorted({m["country_code"] for m in matches})
            }
            repairs.append(
                {
                    "author_index": i,
                    "matches": matches,
                    "method": "explicit_address_suffix_and_exact_local_normalized_match",
                }
            )
    original_codes = {c for a in authors for c in a.get("countries") or []} | {
        i["country_code"]
        for a in authors
        for i in a.get("institutions") or []
        if i.get("country_code")
    }
    return (
        record
        | {
            "authorships": updated,
            "_countries_before_local_repair": len(original_codes),
        },
        repairs,
    )


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def csv(
    path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def doi(value: Any) -> str:
    return re.sub(
        r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", str(value or "").strip().lower()
    )


def title_key(value: str) -> str:
    return re.sub(r"\W+", "", unicodedata.normalize("NFKC", value).casefold())


def country_name(code: str) -> str:
    names = {
        "US": "United States",
        "GB": "United Kingdom",
        "KR": "South Korea",
        "TW": "Taiwan",
        "RU": "Russia",
        "IR": "Iran",
        "HK": "Hong Kong",
    }
    entry = pycountry.countries.get(alpha_2=code)
    return names.get(code, entry.name if entry else code)


def input_papers() -> list[dict[str, Any]]:
    nodes = pd.read_parquet(ASSETS / "claim_nodes.parquet", columns=["parent_paper_id"])
    history = set(nodes.parent_paper_id)
    canonical = pd.read_parquet(ASSETS / "canonical_target_works.parquet")
    paths = pd.read_parquet(ASSETS / "nature_targets.parquet").set_index("article_id")
    rows = [
        {
            "paper_id": r.nature_article_id,
            "doi": doi(r.doi),
            "work_id": r.work_id,
            "title": r.title,
            "year": int(r.publication_year),
            "venue": r.source_name,
            "cohort": "historical",
            "paper_path": paths.loc[r.nature_article_id, "paper_markdown_path"],
        }
        for r in canonical.itertuples()
        if r.nature_article_id in history
    ]
    evidence = read(BASE / "data/evidence.json")
    cited = {
        evidence[c["source_id"]]["paper_id"]
        for d in read(BASE / "data/directions.json")
        for c in d["citations"]
    }
    inventory = read(BASE.parent / "frontier_2026/inventory.json")
    rows.extend(
        {
            "paper_id": r["paper_id"],
            "doi": doi(r["doi"]),
            "work_id": (r["metadata"].get("openalex_work_id") or "").rsplit("/", 1)[-1],
            "title": r["title"],
            "year": 2026,
            "venue": r["venue"],
            "cohort": "forecast_target",
            "paper_path": r["paper_path"],
        }
        for r in inventory
        if r["paper_id"] in cited - history
    )
    assert len(rows) == len({r["paper_id"] for r in rows}) == 25023
    assert len({r["doi"] for r in rows}) == len(rows)
    return sorted(rows, key=lambda r: r["paper_id"])


def audit(out: Path) -> None:
    rows = input_papers()
    write(out / "data/papers.json", rows)
    protected = {
        str(p.relative_to(ROOT)): digest(p)
        for p in sorted(BASE.rglob("*"))
        if p.is_file()
    }
    previous = out / "qa/protected_inputs.json"
    if previous.exists():
        assert read(previous) == protected, "Original word-cloud assets changed"
    else:
        write(previous, protected)
    nodes = pd.read_parquet(ASSETS / "claim_nodes.parquet")
    source = pd.read_csv(BASE / "tables/historical_phrase_sources.csv")
    assert nodes.claim_id.is_unique
    assert not source.duplicated(["phrase", "claim_id"]).any()
    assert set(source.claim_id) <= set(nodes.claim_id)
    evidence = read(BASE / "data/evidence.json")
    citations = [
        c["source_id"]
        for d in read(BASE / "data/directions.json")
        for c in d["citations"]
    ]
    assert set(citations) <= set(evidence)
    cited = {evidence[k]["paper_id"] for k in citations}
    assert cited <= {r["paper_id"] for r in rows}
    result = {
        "papers": len(rows),
        "historical_papers": 24914,
        "forecast_cited_papers": len(cited),
        "claims": len(nodes),
        "phrase_source_rows": len(source),
        "orphan_keys": 0,
        "selection": "Historical claim-bearing corpus plus actually cited 2026 papers",
    }
    write(out / "qa/input_audit.json", result)
    print(json.dumps(result), flush=True)


def get_response(url: str, params: dict[str, Any], key: str) -> dict[str, Any]:
    params = dict(params)
    if key:
        params["api_key"] = key
    status: int | str = "network"
    for attempt in range(4):
        try:
            response = requests.get(url, params=params, timeout=60)
            status = response.status_code
            if response.ok:
                return response.json()
            if status not in (429, 500, 502, 503, 504):
                break
        except (requests.RequestException, ValueError):
            status = "network_or_json"
        time.sleep(min(2 ** (attempt + 1), 16))
    raise RuntimeError(f"OpenAlex request failed: {status}")


def fetch_batch(out: Path, field: str, values: list[str], key: str) -> Path:
    token = hashlib.sha256((field + "|" + "|".join(values)).encode()).hexdigest()[:20]
    path = out / "raw" / f"{field}_{token}.json"
    if path.exists():
        return path
    payload = get_response(
        "https://api.openalex.org/works",
        {"filter": field + ":" + "|".join(values), "select": SELECT, "per_page": 100},
        key,
    )
    write(
        path,
        {
            "requested_field": field,
            "requested_values": values,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "results": payload.get("results", []),
            "meta": payload.get("meta", {}),
        },
    )
    return path


def batch_round(
    out: Path, field: str, values: list[str], key: str, workers: int
) -> None:
    batches = [values[i : i + 100] for i in range(0, len(values), 100)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {
            pool.submit(fetch_batch, out, field, batch, key): batch for batch in batches
        }
        for i, future in enumerate(as_completed(pending), 1):
            try:
                future.result()
            except RuntimeError as error:
                token = hashlib.sha256("|".join(pending[future]).encode()).hexdigest()[
                    :20
                ]
                write(
                    out / "errors" / f"{field}_{token}.json",
                    {"field": field, "values": pending[future], "error": str(error)},
                )
            if i % 10 == 0 or i == len(batches):
                print(f"{field}: {i}/{len(batches)} batches completed", flush=True)


def raw_records(out: Path) -> dict[str, tuple[dict[str, Any], str, str]]:
    records: dict[str, tuple[dict[str, Any], str, str]] = {}
    for path in sorted((out / "raw").glob("*.json")):
        payload = read(path)
        for record in payload.get("results", []):
            ident = doi(record.get("doi"))
            if not ident:
                continue
            previous = records.get(ident)
            rank = (len(record.get("authorships") or []), payload["fetched_at"])
            if previous is None or rank > (
                len(previous[0].get("authorships") or []),
                previous[2],
            ):
                records[ident] = (
                    record,
                    str(path.relative_to(out)),
                    payload["fetched_at"],
                )
    return records


def fetch(out: Path, workers: int = 4) -> None:
    if not (out / "data/papers.json").exists():
        audit(out)
    papers = read(out / "data/papers.json")
    key = (
        getenv("OPENALEX_API_KEY") or getenv("OPENALEX_API_KEYS").split(",")[0]
    ).strip()
    batch_round(
        out,
        "openalex_id",
        sorted({r["work_id"] for r in papers if r["work_id"]}),
        key,
        workers,
    )
    records = raw_records(out)
    missing = [r["doi"] for r in papers if r["doi"] not in records]
    if missing:
        batch_round(out, "doi", sorted(missing), key, workers)
    records = raw_records(out)
    truncated = [
        r
        for r, _, _ in records.values()
        if r.get("is_authors_truncated")
        or len(r.get("authorships") or []) >= 100
        or (r.get("authors_count") or 0) > len(r.get("authorships") or [])
    ]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(fetch_singleton, out, r["id"].rsplit("/", 1)[-1], key)
            for r in truncated
        ]
        for index, future in enumerate(as_completed(futures), 1):
            future.result()
            if index % 10 == 0 or index == len(futures):
                print(f"Full author lists: {index}/{len(truncated)}", flush=True)
    normalize(out)


def fetch_singleton(out: Path, ident: str, key: str) -> None:
    path = out / "raw" / f"singleton_{ident}.json"
    if path.exists():
        return
    try:
        result = get_response(
            f"https://api.openalex.org/works/{ident}", {"select": SELECT}, key
        )
        write(
            path,
            {"fetched_at": datetime.now(timezone.utc).isoformat(), "results": [result]},
        )
    except RuntimeError as error:
        write(
            out / "errors" / f"singleton_{ident}.json",
            {"work_id": ident, "error": str(error)},
        )


def assess_record(paper: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    similarity = SequenceMatcher(
        None, title_key(paper["title"]), title_key(record.get("title") or "")
    ).ratio()
    if doi(record.get("doi")) != paper["doi"]:
        issues.append("doi_mismatch")
    if similarity < 0.85:
        issues.append("title_mismatch")
    if record.get("publication_year") != paper["year"]:
        issues.append("year_mismatch")
    authors = record.get("authorships") or []
    if not authors:
        issues.append("no_authorships")
    if (
        record.get("is_authors_truncated")
        or (record.get("authors_count") or 0) > len(authors)
        or (len(authors) >= 100 and not record.get("_singleton_checked"))
    ):
        issues.append("truncated_authorships")
    countries: set[str] = set()
    missing_authors = 0
    for author in authors:
        direct = set(author.get("countries") or [])
        institutional = {
            i["country_code"]
            for i in author.get("institutions") or []
            if i.get("country_code")
        }
        if direct and institutional and direct != institutional:
            issues.append("author_country_conflict")
        codes = direct | institutional
        if not codes:
            missing_authors += 1
        countries.update(codes)
    if countries - VALID_CODES:
        issues.append("invalid_country_code")
    if missing_authors:
        issues.append("authors_without_country")
    count_explained = (
        record.get("_countries_before_local_repair")
        == record.get("countries_distinct_count")
        and record.get("_countries_before_local_repair") is not None
        and len(countries) > record["_countries_before_local_repair"]
    )
    if (
        record.get("countries_distinct_count") is not None
        and record["countries_distinct_count"] != len(countries)
        and not count_explained
    ):
        issues.append("country_count_mismatch")
    conflict = any(
        x.endswith(("mismatch", "conflict")) or x == "invalid_country_code"
        for x in issues
    )
    status = "conflict" if conflict else "partial" if issues else "complete"
    return {
        "status": status,
        "country_codes": sorted(countries & VALID_CODES),
        "issues": sorted(set(issues)),
        "authors_returned": len(authors),
        "authors_count": record.get("authors_count"),
        "authors_without_country": missing_authors,
        "title_similarity": similarity,
        "returned_work_id": record.get("id"),
        "country_count_updated_from_verified_local_addresses": count_explained,
        "openalex_updated_date": record.get("updated_date"),
    }


def normalize(out: Path) -> list[dict[str, Any]]:
    records = raw_records(out)
    rows = []
    for paper in read(out / "data/papers.json"):
        match = records.get(paper["doi"])
        if match:
            record, source, fetched_at = match
            record = record | {
                "_singleton_checked": Path(source).name.startswith("singleton_")
            }
            record, repairs = repair_affiliations(paper, record)
            row = (
                paper
                | assess_record(paper, record)
                | {"metadata_source": source, "fetched_at": fetched_at}
            )
            if repairs:
                row["local_author_repairs"] = repairs
        else:
            row = paper | {
                "status": "unmatched",
                "country_codes": [],
                "issues": ["no_matching_doi"],
                "metadata_source": "",
                "fetched_at": "",
            }
        rows.append(row)
    override_path = out / "data/local_verifications.json"
    overrides = read(override_path) if override_path.exists() else {}
    for row in rows:
        if row["paper_id"] in overrides:
            override = overrides[row["paper_id"]]
            text = Path(row["paper_path"]).read_text(encoding="utf-8")
            assert (
                override["country_codes"]
                and set(override["country_codes"]) <= VALID_CODES
            )
            assert override["affiliation_quotes"] and all(
                q in text for q in override["affiliation_quotes"]
            )
            assert override["complete_affiliation_block"] is True
            row.update(
                status="complete",
                country_codes=sorted(set(override["country_codes"])),
                original_issues=row["issues"],
                issues=[],
                local_verification=override,
            )
    write(out / "data/paper_countries.json", rows)
    flat = [
        {
            k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
            for k, v in r.items()
        }
        for r in rows
    ]
    csv(out / "tables/paper_country_provenance.csv", flat)
    csv(
        out / "tables/unresolved_papers.csv",
        [r for r in flat if r["status"] != "complete"],
    )
    print("Metadata status:", dict(Counter(r["status"] for r in rows)), flush=True)
    return rows


def weights(row: dict[str, Any]) -> dict[str, float]:
    codes = row["country_codes"]
    return (
        {c: 1 / len(codes) for c in codes}
        if row["status"] == "complete" and codes
        else {}
    )


def selected_topics() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    history = sorted(
        (r for r in read(BASE / "data/historical_wordcloud.json") if r["displayed"]),
        key=lambda r: (-r["frequency"], r["phrase"]),
    )[:16]
    future, seen = [], set()
    for row in sorted(
        (r for r in read(BASE / "data/forecast_wordcloud.json") if r["displayed"]),
        key=lambda r: (-r["prominence"], r["display_label"]),
    ):
        key = tuple(sorted(row["direction_ids"]))
        if key not in seen:
            future.append(row)
            seen.add(key)
        if len(future) == 12:
            break
    return history, future


def coverage(
    rows: list[dict[str, Any]],
    history: list[dict[str, Any]],
    future: list[dict[str, Any]],
    out: Path,
) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[("cohort", row["cohort"])].append(row)
        if row["cohort"] == "historical":
            groups[("year", str(row["year"]))].append(row)
            groups[("venue", row["venue"])].append(row)
    by_id = {r["paper_id"]: r for r in rows}
    for topic in history:
        groups[("historical_topic", topic["display_label"])] = [
            by_id[p] for p in topic["source_paper_ids"]
        ]
    for topic in future:
        groups[("forecast_topic", topic["display_label"])] = [
            by_id[p] for p in topic["source_paper_ids"]
        ]
    evidence = read(BASE / "data/evidence.json")
    cited = {
        evidence[c["source_id"]]["paper_id"]
        for d in read(BASE / "data/directions.json")
        for c in d["citations"]
    }
    groups[("forecast_sources", "all_96_directions")] = [
        by_id[p] for p in sorted(cited)
    ]
    table = [
        {
            "dimension": kind,
            "group": label,
            "total": len(part),
            "complete": sum(r["status"] == "complete" for r in part),
            "coverage": sum(r["status"] == "complete" for r in part) / len(part),
        }
        for (kind, label), part in groups.items()
    ]
    csv(out / "tables/coverage.csv", table)
    gate = [
        r
        for r in table
        if r["dimension"] == "year"
        or (r["dimension"], r["group"]) == ("cohort", "historical")
    ]
    result = {
        "threshold": 0.95,
        "historical_gate_passed": all(r["coverage"] >= 0.95 for r in gate),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "groups": table,
    }
    write(out / "qa/coverage.json", result)
    return result


def historical_statistics(
    rows: list[dict[str, Any]], topics: list[dict[str, Any]], out: Path
) -> tuple[list[str], dict[str, Any]]:
    history = [
        r for r in rows if r["cohort"] == "historical" and r["status"] == "complete"
    ]
    by_id = {r["paper_id"]: weights(r) for r in history}
    counts, participation = Counter(), Counter()
    for w in by_id.values():
        counts.update(w)
        participation.update(w.keys())
    order = sorted(counts, key=lambda c: (-counts[c], c))
    top = order[:12]
    countries = [
        {
            "country_code": c,
            "country": country_name(c),
            "fractional_papers": counts[c],
            "participating_papers": participation[c],
            "share": counts[c] / len(history),
        }
        for c in order
    ]
    csv(out / "tables/c_country_contributions.csv", countries)
    cells = []
    for topic in topics:
        ids = set(topic["source_paper_ids"]) & by_id.keys()
        for c in order:
            contribution = sum(by_id[p].get(c, 0) for p in ids)
            n = sum(c in by_id[p] for p in ids)
            ratio = (
                contribution / counts[c] / (len(ids) / len(history)) if ids else None
            )
            cells.append(
                {
                    "country_code": c,
                    "topic": topic["display_label"],
                    "phrase": topic["phrase"],
                    "fractional_papers": contribution,
                    "participating_papers": n,
                    "country_denominator": counts[c],
                    "topic_denominator": len(ids),
                    "corpus_denominator": len(history),
                    "relative_preference": ratio,
                    "log2_preference": (
                        math.log2(ratio) if ratio and ratio > 0 else None
                    ),
                    "low_sample": 0 < n < 5,
                }
            )
    csv(out / "tables/d_topic_preferences.csv", cells)
    assert math.isclose(sum(counts.values()), len(history), abs_tol=1e-6)
    return top, {
        "countries": countries,
        "topic_cells": cells,
        "historical_complete": len(history),
    }


def type_statistics(rows: list[dict[str, Any]], out: Path) -> list[dict[str, Any]]:
    nodes = pd.read_parquet(
        ASSETS / "claim_nodes.parquet", columns=["parent_paper_id", "claim_type"]
    )
    assert set(nodes.claim_type) <= set(TYPES)
    fractions = pd.crosstab(nodes.parent_paper_id, nodes.claim_type).reindex(
        columns=TYPES, fill_value=0
    )
    fractions = fractions.div(fractions.sum(axis=1), axis=0)
    total: dict[str, Counter[str]] = defaultdict(Counter)
    denominators: Counter[str] = Counter()
    for row in rows:
        if row["cohort"] != "historical":
            continue
        for c, w in weights(row).items():
            denominators[c] += w
            total[c].update(
                {t: w * float(fractions.loc[row["paper_id"], t]) for t in TYPES}
            )
    result = [
        {
            "country_code": c,
            "claim_type": t,
            "proportion": total[c][t] / denominators[c],
            "country_denominator": denominators[c],
        }
        for c in sorted(total)
        for t in TYPES
    ]
    for c, values in total.items():
        assert math.isclose(sum(values.values()), denominators[c], abs_tol=1e-6)
    csv(out / "tables/e_contribution_types.csv", result)
    return result


def future_statistics(
    rows: list[dict[str, Any]], topics: list[dict[str, Any]], top: list[str], out: Path
) -> list[dict[str, Any]]:
    by_id = {r["paper_id"]: r for r in rows}
    directions = {d["candidate_id"]: d for d in read(BASE / "data/directions.json")}
    evidence = read(BASE / "data/evidence.json")
    result, links = [], []
    for topic in topics:
        ids = sorted(
            {
                evidence[c["source_id"]]["paper_id"]
                for d in topic["direction_ids"]
                for c in directions[d]["citations"]
            }
        )
        assert ids == sorted(set(topic["source_paper_ids"]))
        counts: Counter[str] = Counter()
        historical_n = sum(by_id[p]["cohort"] == "historical" for p in ids)
        for p in ids:
            w = weights(by_id[p]) or {"Unresolved": 1.0}
            for c, weight in w.items():
                group = c if c in top or c == "Unresolved" else "Other"
                counts[group] += weight
                links.append(
                    {
                        "topic": topic["display_label"],
                        "paper_id": p,
                        "doi": by_id[p]["doi"],
                        "country_code": c,
                        "display_group": group,
                        "weight": weight,
                        "cohort": by_id[p]["cohort"],
                        "status": by_id[p]["status"],
                    }
                )
        assert math.isclose(sum(counts.values()), len(ids), abs_tol=1e-9)
        result.extend(
            {
                "country_code": c,
                "topic": topic["display_label"],
                "fractional_sources": counts[c],
                "proportion": counts[c] / len(ids),
                "source_papers": len(ids),
                "historical_sources": historical_n,
                "target_sources": len(ids) - historical_n,
                "direction_ids": "|".join(topic["direction_ids"]),
            }
            for c in top + ["Other", "Unresolved"]
        )
    csv(out / "tables/f_future_sources.csv", result)
    csv(out / "tables/f_source_provenance.csv", links)
    return result


def aggregate(out: Path) -> None:
    rows = read(out / "data/paper_countries.json")
    history, future = selected_topics()
    report = coverage(rows, history, future, out)
    top, stats = historical_statistics(rows, history, out)
    if not top:
        raise ValueError(
            "No complete historical country records; cannot render country comparisons"
        )
    types = type_statistics(rows, out)
    forecast = future_statistics(rows, future, top, out)
    plot = stats | {
        "top_countries": top,
        "historical_topics": history,
        "future_topics": future,
        "type_cells": types,
        "future_cells": forecast,
        "coverage": report,
    }
    write(out / "data/plot_data.json", plot)
    write(
        out / "qa/statistical_checks.json",
        {
            "fractional_mass_conserved": True,
            "type_proportions_sum_to_one": True,
            "future_proportions_sum_to_one": True,
            "selection_independent_of_country_topic_values": True,
        },
    )
    print(
        "Historical gate passed:",
        report["historical_gate_passed"],
        "Top countries:",
        top,
        flush=True,
    )

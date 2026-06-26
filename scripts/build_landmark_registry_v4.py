from __future__ import annotations

import argparse
import difflib
import json
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aspr.corpus import normalize_doi, normalize_openalex_id, normalize_title, slugify  # noqa: E402


DEFAULT_BASE_CORPUS_DIR = (
    PROJECT_ROOT
    / "data"
    / "knowledge_corpus"
    / "v1_strict_landmark_external_topup_dedup_protein2_meta_topup_ready_refsupport_candidates_v3e_magnetic_manual_topup"
)
DEFAULT_V3_REGISTRY = PROJECT_ROOT / "data" / "knowledge_corpus" / "landmark_registry_v3.csv"
DEFAULT_V4_SEED = PROJECT_ROOT / "data" / "knowledge_corpus" / "landmark_registry_v4_seed.csv"
DEFAULT_OUT_CSV = PROJECT_ROOT / "data" / "knowledge_corpus" / "landmark_registry_v4.csv"
DEFAULT_DOMAIN_SEED_CSV = PROJECT_ROOT / "data" / "knowledge_corpus" / "publication_candidate_domains_v4.csv"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "outputs" / "landmark_registry_v4"
TRUSTED_BASE_SOURCES = {
    "manual",
    "nobel",
    "award",
    "awards",
    "curated",
    "expert",
    "external_authority",
    "strict_manual",
    "strict_manual_v3",
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def nonempty(value: object) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"", "nan", "none", "null", "<na>"} else text


def safe_int(value: object, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return int(default)
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def doi_url(doi: object) -> str:
    doi_norm = normalize_doi(doi)
    return f"https://doi.org/{doi_norm}" if doi_norm else ""


def title_similarity(left: object, right: object) -> float:
    a = normalize_title(left)
    b = normalize_title(right)
    if not a or not b:
        return 0.0
    return float(difflib.SequenceMatcher(None, a, b).ratio())


def standard_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "domain",
        "display_name",
        "field_name",
        "query",
        "label",
        "title",
        "year",
        "doi",
        "openalex_id",
        "evidence_type",
        "evidence_url",
        "evidence_note",
        "include_main",
        "source_registry",
    ]
    out = frame.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = ""
    out["domain"] = out["domain"].map(slugify)
    out["display_name"] = out["display_name"].map(nonempty)
    out["field_name"] = out["field_name"].map(nonempty)
    out["query"] = out["query"].map(nonempty)
    out["label"] = out["label"].map(nonempty)
    out["title"] = out["title"].map(nonempty)
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["doi"] = out["doi"].map(normalize_doi)
    out["openalex_id"] = out["openalex_id"].map(normalize_openalex_id)
    out["evidence_type"] = out["evidence_type"].map(nonempty)
    out["evidence_url"] = out["evidence_url"].map(nonempty)
    out["evidence_note"] = out["evidence_note"].map(nonempty)
    out["include_main"] = pd.to_numeric(out["include_main"], errors="coerce").fillna(1).astype(int)
    out["source_registry"] = out["source_registry"].map(nonempty)
    return out[columns].copy()


def load_domain_metadata(corpus_dir: Path) -> Dict[str, Dict[str, str]]:
    domains = read_csv(corpus_dir / "domains.csv")
    out: Dict[str, Dict[str, str]] = {}
    if domains.empty:
        return out
    for row in domains.to_dict("records"):
        domain = slugify(row.get("slug") or row.get("domain") or row.get("display_name"))
        if not domain:
            continue
        out[domain] = {
            "display_name": nonempty(row.get("display_name")) or domain.replace("_", " "),
            "field_name": nonempty(row.get("field_name")),
            "query": nonempty(row.get("query")) or nonempty(row.get("display_name")) or domain.replace("_", " "),
        }
    return out


def from_v3_registry(path: Path, domain_meta: Mapping[str, Mapping[str, str]]) -> pd.DataFrame:
    frame = read_csv(path)
    rows: List[Dict[str, Any]] = []
    for row in frame.to_dict("records"):
        domain = slugify(row.get("domain"))
        meta = domain_meta.get(domain, {})
        rows.append(
            {
                "domain": domain,
                "display_name": meta.get("display_name", domain.replace("_", " ")),
                "field_name": meta.get("field_name", ""),
                "query": meta.get("query", domain.replace("_", " ")),
                "label": nonempty(row.get("label")) or f"{domain} {safe_int(row.get('year'))}",
                "title": nonempty(row.get("title")) or nonempty(row.get("openalex_title")),
                "year": safe_int(row.get("year")),
                "doi": normalize_doi(row.get("doi")),
                "openalex_id": normalize_openalex_id(row.get("openalex_id")),
                "evidence_type": nonempty(row.get("authority_basis")) or "strict_manual_v3",
                "evidence_url": nonempty(row.get("doi_url")) or doi_url(row.get("doi")),
                "evidence_note": nonempty(row.get("authority_note")) or "carried forward from strict v3 registry",
                "include_main": int(row.get("include_main", 1) or 1),
                "source_registry": "landmark_registry_v3",
            }
        )
    return standard_columns(pd.DataFrame(rows))


def from_base_landmarks(corpus_dir: Path, domain_meta: Mapping[str, Mapping[str, str]]) -> pd.DataFrame:
    frame = read_csv(corpus_dir / "landmarks.csv")
    rows: List[Dict[str, Any]] = []
    for row in frame.to_dict("records"):
        source = nonempty(row.get("landmark_source")).lower()
        if source not in TRUSTED_BASE_SOURCES:
            continue
        domain = slugify(row.get("domain"))
        meta = domain_meta.get(domain, {})
        rows.append(
            {
                "domain": domain,
                "display_name": meta.get("display_name", domain.replace("_", " ")),
                "field_name": meta.get("field_name", ""),
                "query": meta.get("query", domain.replace("_", " ")),
                "label": nonempty(row.get("label")) or nonempty(row.get("title")),
                "title": nonempty(row.get("title")) or nonempty(row.get("label")),
                "year": safe_int(row.get("year")),
                "doi": normalize_doi(row.get("doi")),
                "openalex_id": normalize_openalex_id(row.get("id")),
                "evidence_type": source,
                "evidence_url": doi_url(row.get("doi")),
                "evidence_note": nonempty(row.get("accepted_landmark_source")) or "trusted non-legacy landmark seed",
                "include_main": int(row.get("include_main", 1) or 1),
                "source_registry": "retained_candidate_landmarks",
            }
        )
    return standard_columns(pd.DataFrame(rows))


def from_seed_csv(path: Path) -> pd.DataFrame:
    frame = read_csv(path)
    if frame.empty:
        return standard_columns(pd.DataFrame())
    frame = frame.rename(columns={"domain_id": "domain"})
    frame["source_registry"] = "landmark_registry_v4_seed"
    return standard_columns(frame)


def openalex_work_for_doi(doi: str, email: str = "", timeout: int = 30) -> Optional[Dict[str, Any]]:
    doi_norm = normalize_doi(doi)
    if not doi_norm:
        return None
    key = urllib.parse.quote(f"https://doi.org/{doi_norm}", safe="")
    params = {
        "select": "id,doi,display_name,publication_year,cited_by_count,referenced_works",
    }
    if email:
        params["mailto"] = email
    url = f"https://api.openalex.org/works/{key}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "ASPR landmark registry v4 validator"})
    with urllib.request.urlopen(req, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _validation_template(row: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    out["validation_status"] = "not_checked"
    out["failure_reason"] = ""
    out["openalex_title"] = ""
    out["openalex_year"] = ""
    out["openalex_cited_by_count"] = ""
    out["openalex_reference_count"] = ""
    out["title_similarity"] = ""
    return out


def validate_openalex_row(row: Mapping[str, Any], email: str, timeout: int, sleep_seconds: float) -> Dict[str, Any]:
    out = _validation_template(row)
    if sleep_seconds > 0:
        time.sleep(float(sleep_seconds))
    try:
        work = openalex_work_for_doi(out.get("doi", ""), email=email, timeout=timeout)
    except Exception as exc:
        out["validation_status"] = "failed"
        out["failure_reason"] = f"openalex_lookup_failed:{type(exc).__name__}"
        return out
    if not work:
        out["validation_status"] = "failed"
        out["failure_reason"] = "openalex_lookup_empty"
        return out
    fetched_id = normalize_openalex_id(work.get("id"))
    fetched_doi = normalize_doi(work.get("doi"))
    fetched_title = nonempty(work.get("display_name"))
    fetched_year = safe_int(work.get("publication_year"))
    sim = title_similarity(out.get("title"), fetched_title)
    out["openalex_id"] = normalize_openalex_id(out.get("openalex_id")) or fetched_id
    out["openalex_title"] = fetched_title
    out["openalex_year"] = fetched_year
    out["openalex_cited_by_count"] = safe_int(work.get("cited_by_count"))
    out["openalex_reference_count"] = len(work.get("referenced_works") or [])
    out["title_similarity"] = sim
    if fetched_doi != normalize_doi(out.get("doi")):
        out["validation_status"] = "failed"
        out["failure_reason"] = "doi_mismatch"
    elif fetched_year and abs(fetched_year - safe_int(out.get("year"))) > 1:
        out["validation_status"] = "failed"
        out["failure_reason"] = "year_mismatch"
    elif sim < 0.55:
        out["validation_status"] = "failed"
        out["failure_reason"] = "title_similarity_low"
    else:
        out["validation_status"] = "passed"
    return out


def validate_openalex(
    frame: pd.DataFrame,
    email: str,
    timeout: int,
    sleep_seconds: float,
    workers: int = 1,
    max_rows: int = 0,
) -> pd.DataFrame:
    records = frame.to_dict("records")
    if max_rows > 0:
        checked = records[: int(max_rows)]
        unchecked = [_validation_template(row) for row in records[int(max_rows) :]]
    else:
        checked = records
        unchecked = []
    if int(workers) <= 1:
        rows = [validate_openalex_row(row, email, timeout, sleep_seconds) for row in checked]
    else:
        with ThreadPoolExecutor(max_workers=int(workers)) as pool:
            rows = list(pool.map(lambda row: validate_openalex_row(row, email, timeout, sleep_seconds), checked))
    return pd.DataFrame(rows + unchecked)


def build_registry(args: argparse.Namespace) -> Dict[str, Any]:
    domain_meta = load_domain_metadata(args.base_corpus_dir)
    frames = [
        from_v3_registry(args.v3_registry_csv, domain_meta),
        from_base_landmarks(args.base_corpus_dir, domain_meta),
        from_seed_csv(args.seed_csv),
    ]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined[combined["include_main"].astype(int) == 1].copy()
    combined = combined[combined["domain"].astype(str) != ""].copy()
    combined = combined[combined["doi"].astype(str) != ""].copy()
    combined = combined[pd.to_numeric(combined["year"], errors="coerce") <= int(args.max_main_year)].copy()
    combined["evidence_url"] = combined["evidence_url"].where(combined["evidence_url"].astype(str) != "", combined["doi"].map(doi_url))
    combined["evidence_type"] = combined["evidence_type"].where(combined["evidence_type"].astype(str) != "", "manual_v4")
    combined["evidence_note"] = combined["evidence_note"].where(combined["evidence_note"].astype(str) != "", "strict DOI-backed v4 seed")
    combined = combined.sort_values(["domain", "year", "source_registry", "doi"]).drop_duplicates(["domain", "doi"], keep="first")
    combined = combined.groupby("domain", group_keys=False).head(int(args.max_landmarks_per_domain)).reset_index(drop=True)

    if args.validate_openalex:
        validated = validate_openalex(
            combined,
            email=args.openalex_email,
            timeout=args.timeout_seconds,
            sleep_seconds=args.sleep_seconds,
            workers=getattr(args, "validation_workers", 1),
            max_rows=getattr(args, "max_validation_rows", 0),
        )
    else:
        validated = combined.copy()
        validated["validation_status"] = "passed"
        validated["failure_reason"] = ""
        validated["openalex_title"] = ""
        validated["openalex_year"] = ""
        validated["openalex_cited_by_count"] = ""
        validated["openalex_reference_count"] = ""
        validated["title_similarity"] = ""

    args.report_dir.mkdir(parents=True, exist_ok=True)
    validated.to_csv(args.report_dir / "landmark_registry_v4_validation.csv", index=False)
    passed = validated[validated["validation_status"].astype(str) == "passed"].copy()
    counts = passed.groupby("domain").size() if not passed.empty else pd.Series(dtype=int)
    eligible_domains = sorted(domain for domain, count in counts.items() if 1 <= int(count) <= int(args.max_landmarks_per_domain))
    passed = passed[passed["domain"].astype(str).isin(eligible_domains)].copy()
    passed = passed.sort_values(["domain", "year", "doi"]).reset_index(drop=True)

    output_cols = [
        "domain",
        "display_name",
        "field_name",
        "query",
        "label",
        "doi",
        "title",
        "year",
        "evidence_type",
        "evidence_url",
        "evidence_note",
        "include_main",
        "openalex_id",
        "openalex_title",
        "openalex_year",
        "openalex_cited_by_count",
        "openalex_reference_count",
        "title_similarity",
        "validation_status",
        "failure_reason",
        "source_registry",
    ]
    for col in output_cols:
        if col not in passed.columns:
            passed[col] = ""
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    passed[output_cols].to_csv(args.out_csv, index=False)
    passed[output_cols].to_json(args.out_json, orient="records", force_ascii=False, indent=2)

    domains = (
        passed.groupby("domain", sort=True)
        .agg(
            display_name=("display_name", "first"),
            field_name=("field_name", "first"),
            query=("query", "first"),
            n_landmarks=("doi", "count"),
            first_landmark_year=("year", "min"),
        )
        .reset_index()
        .rename(columns={"domain": "slug"})
    )
    domains.to_csv(args.domain_seed_csv, index=False)
    coverage = domains.sort_values(["field_name", "slug"]).reset_index(drop=True)
    coverage.to_csv(args.report_dir / "landmark_registry_v4_domain_coverage.csv", index=False)

    failed = validated[validated["validation_status"].astype(str) != "passed"].copy()
    manifest = {
        "created_at": utc_now(),
        "artifact_kind": "landmark_registry_v4",
        "policy": "strict_doi_or_openalex_v4_no_legacy_fig1_anchor",
        "base_corpus_dir": str(args.base_corpus_dir),
        "v3_registry_csv": str(args.v3_registry_csv),
        "seed_csv": str(args.seed_csv),
        "out_csv": str(args.out_csv),
        "domain_seed_csv": str(args.domain_seed_csv),
        "validate_openalex": bool(args.validate_openalex),
        "validation_workers": int(getattr(args, "validation_workers", 1)),
        "max_validation_rows": int(getattr(args, "max_validation_rows", 0)),
        "max_main_year": int(args.max_main_year),
        "max_landmarks_per_domain": int(args.max_landmarks_per_domain),
        "n_input_rows": int(len(combined)),
        "n_passed_rows": int(len(passed)),
        "n_failed_rows": int(len(failed)),
        "n_domains": int(passed["domain"].nunique()) if not passed.empty else 0,
        "failed_domains": sorted(failed["domain"].dropna().astype(str).unique().tolist()),
        "domains": sorted(passed["domain"].dropna().astype(str).unique().tolist()),
    }
    write_json(args.report_dir / "landmark_registry_v4_manifest.json", manifest)
    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build strict DOI-backed landmark registry v4.")
    parser.add_argument("--base-corpus-dir", type=Path, default=DEFAULT_BASE_CORPUS_DIR)
    parser.add_argument("--v3-registry-csv", type=Path, default=DEFAULT_V3_REGISTRY)
    parser.add_argument("--seed-csv", type=Path, default=DEFAULT_V4_SEED)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-json", type=Path, default=PROJECT_ROOT / "data" / "knowledge_corpus" / "landmark_registry_v4.json")
    parser.add_argument("--domain-seed-csv", type=Path, default=DEFAULT_DOMAIN_SEED_CSV)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--max-main-year", type=int, default=2015)
    parser.add_argument("--max-landmarks-per-domain", type=int, default=3)
    parser.add_argument("--validate-openalex", action="store_true")
    parser.add_argument("--openalex-email", default="")
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--validation-workers", type=int, default=1)
    parser.add_argument("--max-validation-rows", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    manifest = build_registry(args)
    print(
        f"[landmark-v4] wrote {manifest['n_passed_rows']} strict landmarks "
        f"across {manifest['n_domains']} domains to {args.out_csv}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

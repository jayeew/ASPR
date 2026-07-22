from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_openalex_v3_citation_graph import OpenAlexClient, split_api_keys  # noqa: E402
from scripts.nature_portfolio_v5 import (  # noqa: E402
    BROAD_CATEGORIES,
    DEFAULT_V4_SCREEN_DOMAINS,
    DEFAULT_V5_OUTPUT_DIR,
    NATURE_SOURCE_SEEDS,
    OPENALEX_SOURCE_SELECT_V5,
    SUPPLEMENTAL_DOMAIN_SEEDS,
    classify_broad_category,
    journal_family_from_name,
    nonempty,
    read_csv,
    slugify,
    utc_now,
    write_json,
)
from aspr.env import getenv  # noqa: E402


def source_seed_rows() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for family, display_name, broad_category, url in NATURE_SOURCE_SEEDS:
        rows.append(
            {
                "source_display_name": display_name,
                "journal_family": family,
                "broad_category": broad_category,
                "nature_source_url": url,
                "source_seed_kind": "curated_nature_portfolio_seed",
                "source_query": display_name,
            }
        )
    return pd.DataFrame(rows)


def taxonomy_rows(existing_domain_seed: Path = DEFAULT_V4_SCREEN_DOMAINS) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for row in BROAD_CATEGORIES:
        rows.append(
            {
                "row_kind": "broad_category",
                "broad_category": row["broad_category"],
                "broad_category_label": row["label"],
                "domain": "",
                "domain_display_name": "",
                "query": "",
                "source": "nature_index_scientific_reports_openalex_mapping",
                "nature_index_field": row["nature_index_field"],
                "openalex_hints": row["openalex_hints"],
            }
        )

    existing = read_csv(existing_domain_seed)
    if not existing.empty:
        for domain in existing.to_dict("records"):
            field = nonempty(domain.get("field_name"))
            subfield = nonempty(domain.get("subfield_name"))
            display = nonempty(domain.get("display_name")) or nonempty(domain.get("slug"))
            broad = classify_broad_category(field, subfield, display)
            if broad == "multidisciplinary":
                broad = classify_broad_category(nonempty(domain.get("query")), default="biology_life_sciences")
            rows.append(
                {
                    "row_kind": "fine_domain",
                    "broad_category": broad,
                    "broad_category_label": "",
                    "domain": slugify(domain.get("slug") or display),
                    "domain_display_name": display,
                    "query": nonempty(domain.get("query")) or display,
                    "source": f"existing_seed:{existing_domain_seed}",
                    "nature_index_field": "",
                    "openalex_hints": f"{field}; {subfield}".strip("; "),
                }
            )

    for broad, domain, display, query in SUPPLEMENTAL_DOMAIN_SEEDS:
        rows.append(
            {
                "row_kind": "fine_domain",
                "broad_category": broad,
                "broad_category_label": "",
                "domain": slugify(domain),
                "domain_display_name": display,
                "query": query,
                "source": "supplemental_nature_gap_fill_seed",
                "nature_index_field": "",
                "openalex_hints": "",
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.drop_duplicates(["row_kind", "broad_category", "domain"], keep="first").reset_index(drop=True)
    return out


def _source_row_from_openalex(seed: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    issn = source.get("issn") or []
    return {
        **seed,
        "source_id": source.get("id", ""),
        "openalex_source_display_name": source.get("display_name", ""),
        "issn_l": source.get("issn_l", ""),
        "issn": ";".join(str(item) for item in issn) if isinstance(issn, list) else str(issn or ""),
        "source_type": source.get("type", ""),
        "host_organization_name": source.get("host_organization_name", ""),
        "works_count": int(source.get("works_count") or 0),
        "cited_by_count": int(source.get("cited_by_count") or 0),
        "homepage_url": source.get("homepage_url", ""),
        "openalex_match_status": "matched",
    }


def enrich_sources_with_openalex(
    seeds: pd.DataFrame,
    *,
    openalex: OpenAlexClient,
    per_query: int = 5,
    quiet: bool = False,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for idx, seed in enumerate(seeds.to_dict("records"), start=1):
        query = nonempty(seed.get("source_query")) or nonempty(seed.get("source_display_name"))
        if not query:
            continue
        if not quiet:
            print(f"[Nature source roster] 正在匹配来源 {idx}/{len(seeds)}：{query}", flush=True)
        try:
            payload = openalex.get_json(
                "/sources",
                params={
                    "search": query,
                    "filter": "type:journal",
                    "per-page": int(per_query),
                    "select": OPENALEX_SOURCE_SELECT_V5,
                },
            )
        except Exception as exc:
            row = dict(seed)
            row.update({"source_id": "", "openalex_match_status": f"fetch_failed:{exc}"})
            rows.append(row)
            continue
        results = payload.get("results") or []
        if not results:
            row = dict(seed)
            row.update({"source_id": "", "openalex_match_status": "not_found"})
            rows.append(row)
            continue
        query_norm = query.lower().strip()
        exact = [item for item in results if str(item.get("display_name", "")).lower().strip() == query_norm]
        chosen = exact[0] if exact else results[0]
        rows.append(_source_row_from_openalex(dict(seed), chosen))
    out = pd.DataFrame(rows)
    if "journal_family" in out.columns:
        out["journal_family"] = [
            family if nonempty(family) else journal_family_from_name(name)
            for family, name in zip(out["journal_family"], out.get("source_display_name", pd.Series("", index=out.index)))
        ]
    return out


def build_roster(args: argparse.Namespace) -> Dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    taxonomy = taxonomy_rows(args.existing_domain_seed)
    seeds = source_seed_rows()
    if args.offline:
        roster = seeds.copy()
        roster["source_id"] = ""
        roster["openalex_match_status"] = "offline_seed_only"
    else:
        openalex = OpenAlexClient(
            api_key=args.openalex_api_key,
            api_keys=split_api_keys(args.openalex_api_keys),
            email=args.openalex_email,
            sleep_seconds=args.sleep_seconds,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
        )
        roster = enrich_sources_with_openalex(seeds, openalex=openalex, per_query=args.per_query, quiet=args.quiet)
    if not roster.empty:
        roster["broad_category"] = [
            broad if broad != "multidisciplinary" else classify_broad_category(name, default="multidisciplinary")
            for broad, name in zip(roster.get("broad_category", ""), roster.get("source_display_name", ""))
        ]
    roster.to_csv(args.out_dir / "nature_source_roster.csv", index=False)
    taxonomy.to_csv(args.out_dir / "nature_subject_taxonomy.csv", index=False)
    fine = taxonomy[taxonomy["row_kind"].astype(str) == "fine_domain"].copy()
    domain_coverage = (
        fine.groupby("broad_category", as_index=False)
        .agg(n_fine_domains=("domain", "nunique"), domains=("domain", lambda values: ";".join(sorted(set(map(str, values))))))
        .sort_values("broad_category")
    )
    domain_coverage.to_csv(args.out_dir / "domain_coverage_report.csv", index=False)
    n_sources_with_openalex_id = (
        int(roster.get("source_id", pd.Series(dtype=str)).fillna("").astype(str).str.strip().ne("").sum())
        if not roster.empty
        else 0
    )
    manifest = {
        "artifact_kind": "nature_portfolio_v5_source_roster",
        "created_at": utc_now(),
        "offline": bool(args.offline),
        "source_roster": str(args.out_dir / "nature_source_roster.csv"),
        "subject_taxonomy": str(args.out_dir / "nature_subject_taxonomy.csv"),
        "domain_coverage_report": str(args.out_dir / "domain_coverage_report.csv"),
        "n_sources": int(len(roster)),
        "n_sources_with_openalex_id": n_sources_with_openalex_id,
        "n_broad_categories": int(taxonomy.loc[taxonomy["row_kind"] == "broad_category", "broad_category"].nunique()),
        "n_fine_domains": int(fine["domain"].nunique()) if not fine.empty else 0,
        "existing_domain_seed": str(args.existing_domain_seed),
    }
    write_json(args.out_dir / "v5_nature_portfolio_full_manifest.json", manifest)
    if not args.offline and n_sources_with_openalex_id == 0:
        raise RuntimeError("没有匹配到任何 OpenAlex source_id；请检查 OpenAlex 网络/API 参数或 source select 字段。")
    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Nature Portfolio v5 source roster and subject taxonomy.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_V5_OUTPUT_DIR)
    parser.add_argument("--existing-domain-seed", type=Path, default=DEFAULT_V4_SCREEN_DOMAINS)
    parser.add_argument("--offline", action="store_true", help="Write curated seeds without OpenAlex source lookup.")
    parser.add_argument("--per-query", type=int, default=5)
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
    manifest = build_roster(args)
    if not args.quiet:
        print(
            f"[Nature source roster] 已写入 {manifest['n_sources']} 个来源、"
            f"{manifest['n_broad_categories']} 个大类、{manifest['n_fine_domains']} 个细领域",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

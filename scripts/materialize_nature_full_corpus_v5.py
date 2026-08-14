from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gear.corpus import audit_corpus, build_topics_and_edges, make_views, root_work_columns  # noqa: E402
from scripts.nature_portfolio_v5 import (  # noqa: E402
    DEFAULT_V5_CORPUS_DIR,
    DEFAULT_V5_OUTPUT_DIR,
    coverage_quality_summary,
    normalize_openalex_id,
    read_csv,
    target_reference_edges,
    utc_now,
    write_json,
    write_partitioned_table,
)


EXTRA_WORK_COLUMNS = [
    "broad_category",
    "journal_family",
    "source_id",
    "source_display_name",
    "source_issn_l",
    "openalex_primary_field",
    "openalex_primary_subfield",
    "primary_topic",
    "is_target_work",
]


def ensure_root_works(targets: pd.DataFrame) -> pd.DataFrame:
    out = targets.copy()
    for col in root_work_columns() + EXTRA_WORK_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    for col in ["legacy_is_landmark", "is_landmark", "cited_by_count", "reference_count", "partial_2026", "is_target_work"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    out["id"] = out["id"].map(normalize_openalex_id)
    out["short_id"] = out["short_id"].where(out["short_id"].astype(str).str.strip().ne(""), out["id"].astype(str).str.rsplit("/", n=1).str[-1])
    out["year"] = pd.to_numeric(out["year"], errors="coerce").fillna(0).astype(int)
    return out[root_work_columns() + EXTRA_WORK_COLUMNS].drop_duplicates("id").reset_index(drop=True)


def build_domain_tables(works: pd.DataFrame, taxonomy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not works.empty:
        grouped = works.groupby("domain", sort=True)
        for domain, sub in grouped:
            rows.append(
                {
                    "slug": domain,
                    "display_name": str(sub["display_topic_label"].mode().iloc[0]) if "display_topic_label" in sub.columns and not sub["display_topic_label"].mode().empty else str(domain),
                    "query": "",
                    "field_name": str(sub["openalex_primary_field"].mode().iloc[0]) if "openalex_primary_field" in sub.columns and not sub["openalex_primary_field"].mode().empty else "",
                    "subfield_name": str(sub["openalex_primary_subfield"].mode().iloc[0]) if "openalex_primary_subfield" in sub.columns and not sub["openalex_primary_subfield"].mode().empty else "",
                    "broad_category": str(sub["broad_category"].mode().iloc[0]) if "broad_category" in sub.columns and not sub["broad_category"].mode().empty else "",
                    "seed_source": "nature_portfolio_v5_venue_driven",
                    "n_works": int(len(sub)),
                }
            )
    domains = pd.DataFrame(rows)
    if not taxonomy.empty and "row_kind" in taxonomy.columns:
        fine = taxonomy[taxonomy["row_kind"].astype(str) == "fine_domain"].copy()
        missing = sorted(set(fine.get("domain", pd.Series(dtype=str)).astype(str)) - set(domains.get("slug", pd.Series(dtype=str)).astype(str)))
        if missing:
            domains = pd.concat(
                [
                    domains,
                    fine[fine["domain"].astype(str).isin(missing)].rename(
                        columns={"domain": "slug", "domain_display_name": "display_name"}
                    )[["slug", "display_name", "query", "broad_category"]].assign(
                        field_name="",
                        subfield_name="",
                        seed_source="nature_portfolio_v5_taxonomy_seed_no_rows",
                        n_works=0,
                    ),
                ],
                ignore_index=True,
                sort=False,
            )
    return domains.sort_values(["broad_category", "slug"]).reset_index(drop=True) if not domains.empty else domains


def write_methods_doc(path: Path, manifest: Dict[str, Any], quality: Dict[str, Any]) -> None:
    body = f"""# Nature Portfolio v5 Full Corpus Methods

This corpus is a venue-driven Nature Portfolio / Nature-style knowledge-graph
dataset. Target papers are OpenAlex works published in Nature-family sources,
including Nature, Nature research journals, Communications journals, Scientific
Reports and npj journals. Papers are mapped to broad categories and fine domains
using source metadata plus OpenAlex primary topics, fields and subfields.

Prediction features must be computable at publication day from the target paper,
its references, reference metadata and the pre-publication graph. Future citers
and future graph deltas are label-only artifacts and are excluded from
publication-day feature sets.

## Corpus Summary

- Target works: {manifest.get("n_target_works", 0):,}
- Reference works: {manifest.get("n_reference_works", 0):,}
- Reference edges: {manifest.get("n_reference_edges", 0):,}
- Broad categories: {quality.get("n_broad_categories", 0)}
- Fine domains: {quality.get("n_fine_domains", 0)}
- Tau horizon: {quality.get("tau", 8)}

## Quality Gate Status

- Overall pass: {quality.get("overall_pass", False)}
- Checks: `{json.dumps(quality.get("checks", {}), ensure_ascii=False, sort_keys=True)}`

## Files

- `works.csv`: target Nature-family works for legacy figure readers.
- `citations.csv`: target-to-reference edges.
- `reference_works.csv`: reference closure metadata when fetched.
- `future_citers.csv` and `future_graph_deltas.csv`: label-only future graph artifacts.
- `canonical_tables/`: partitioned parquet when available, otherwise CSV fallback.
- `nature_full_v5.duckdb`: optional local DuckDB database when the dependency is installed.
"""
    path.write_text(body, encoding="utf-8")


def write_duckdb_database(corpus_dir: Path, tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    db_path = corpus_dir / "nature_full_v5.duckdb"
    try:
        import duckdb  # type: ignore[import-not-found]
    except ImportError:
        return {"status": "skipped", "reason": "duckdb_not_installed", "path": str(db_path)}
    con = duckdb.connect(str(db_path))
    try:
        written = []
        for name, table in tables.items():
            temp_name = f"_{name}_df"
            con.register(temp_name, table)
            con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM {temp_name}")
            con.unregister(temp_name)
            written.append({"table": name, "rows": int(len(table))})
    finally:
        con.close()
    return {"status": "written", "path": str(db_path), "tables": written}


def materialize(args: argparse.Namespace) -> Dict[str, Any]:
    args.corpus_dir.mkdir(parents=True, exist_ok=True)
    targets = read_csv(args.target_works)
    if targets.empty:
        raise FileNotFoundError(f"No target works found: {args.target_works}")
    works = ensure_root_works(targets)
    refs = read_csv(args.reference_works)
    edges = read_csv(args.reference_edges)
    if edges.empty:
        edges = target_reference_edges(works)
    future_citers = read_csv(args.future_citers)
    future_deltas = read_csv(args.future_graph_deltas)
    source_roster = read_csv(args.source_roster)
    taxonomy = read_csv(args.subject_taxonomy)
    domains = build_domain_tables(works, taxonomy)
    topics, topic_edges = build_topics_and_edges(works, edges[edges["target"].astype(str).isin(set(works["id"].astype(str)))] if not edges.empty else edges)
    landmarks = pd.DataFrame(
        columns=[
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
        ]
    )

    works.to_csv(args.corpus_dir / "works.csv", index=False)
    edges.to_csv(args.corpus_dir / "citations.csv", index=False)
    refs.to_csv(args.corpus_dir / "reference_works.csv", index=False)
    future_citers.to_csv(args.corpus_dir / "future_citers.csv", index=False)
    future_deltas.to_csv(args.corpus_dir / "future_graph_deltas.csv", index=False)
    domains.to_csv(args.corpus_dir / "domains.csv", index=False)
    topics.to_csv(args.corpus_dir / "topics.csv", index=False)
    topic_edges.to_csv(args.corpus_dir / "topic_edges.csv", index=False)
    landmarks.to_csv(args.corpus_dir / "landmarks.csv", index=False)
    source_roster.to_csv(args.corpus_dir / "nature_source_roster.csv", index=False)
    taxonomy.to_csv(args.corpus_dir / "nature_subject_taxonomy.csv", index=False)

    canonical_root = args.corpus_dir / "canonical_tables"
    partition_reports = [
        write_partitioned_table(works, canonical_root, "target_works"),
        write_partitioned_table(refs, canonical_root, "reference_works", partition_cols=("broad_category", "domain")),
        write_partitioned_table(edges, canonical_root, "reference_edges", partition_cols=("source_dataset",)),
    ]
    if not future_deltas.empty:
        partition_reports.append(write_partitioned_table(future_deltas, canonical_root, "future_graph_deltas", partition_cols=("tau",)))
    duckdb_report = write_duckdb_database(
        args.corpus_dir,
        {
            "works": works,
            "reference_works": refs,
            "reference_edges": edges,
            "future_citers": future_citers,
            "future_graph_deltas": future_deltas,
            "domains": domains,
            "topics": topics,
            "topic_edges": topic_edges,
            "source_roster": source_roster,
            "subject_taxonomy": taxonomy,
        },
    )

    legacy_quality = audit_corpus(args.corpus_dir, min_papers_per_domain=args.min_papers_per_domain)
    make_views(args.corpus_dir, anchor_policy="strict")
    quality = coverage_quality_summary(
        works,
        future=future_deltas,
        tau=args.tau,
        min_broad_categories=args.min_broad_categories,
        min_fine_domains=args.min_fine_domains,
        min_broad_eligible=args.min_broad_eligible,
        min_domain_eligible=args.min_domain_eligible,
    )
    quality["legacy_audit_overall_pass"] = bool(legacy_quality.get("overall_pass", False))
    write_json(args.corpus_dir / "data_quality_report.json", quality)

    coverage = (
        works.groupby(["broad_category", "domain"], as_index=False)
        .agg(n_works=("id", "size"), year_min=("year", "min"), year_max=("year", "max"))
        .sort_values(["broad_category", "domain"])
    )
    coverage.to_csv(args.corpus_dir / "domain_coverage_report.csv", index=False)

    manifest = {
        "artifact_kind": "v5_nature_portfolio_full",
        "created_at": utc_now(),
        "target_works": str(args.target_works),
        "reference_works": str(args.reference_works),
        "reference_edges": str(args.reference_edges),
        "future_citers": str(args.future_citers),
        "future_graph_deltas": str(args.future_graph_deltas),
        "corpus_dir": str(args.corpus_dir),
        "n_target_works": int(len(works)),
        "n_reference_works": int(len(refs)),
        "n_reference_edges": int(len(edges)),
        "n_future_citer_rows": int(len(future_citers)),
        "n_future_delta_rows": int(len(future_deltas)),
        "n_broad_categories": int(works["broad_category"].nunique()) if "broad_category" in works.columns else 0,
        "n_fine_domains": int(works["domain"].nunique()),
        "tau": int(args.tau),
        "quality_overall_pass": bool(quality.get("overall_pass", False)),
        "canonical_partition_reports": partition_reports,
        "duckdb_report": duckdb_report,
        "figure_contract": "Root CSV tables and views remain compatible with existing Fig1-Fig5 readers; v5-specific reference closure is stored separately.",
        "no_leakage_contract": "future_citers and future_graph_deltas are label-only artifacts.",
    }
    write_json(args.corpus_dir / "v5_nature_portfolio_full_manifest.json", manifest)
    write_methods_doc(args.corpus_dir / "methods_nature_full_corpus.md", manifest, quality)
    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize Nature Portfolio v5 full corpus.")
    parser.add_argument("--target-works", type=Path, default=DEFAULT_V5_OUTPUT_DIR / "nature_target_works.csv")
    parser.add_argument("--reference-works", type=Path, default=DEFAULT_V5_OUTPUT_DIR / "nature_reference_works.csv")
    parser.add_argument("--reference-edges", type=Path, default=DEFAULT_V5_OUTPUT_DIR / "nature_reference_edges.csv")
    parser.add_argument("--future-citers", type=Path, default=DEFAULT_V5_OUTPUT_DIR / "nature_future_citers.csv")
    parser.add_argument("--future-graph-deltas", type=Path, default=DEFAULT_V5_OUTPUT_DIR / "nature_future_graph_deltas.csv")
    parser.add_argument("--source-roster", type=Path, default=DEFAULT_V5_OUTPUT_DIR / "nature_source_roster.csv")
    parser.add_argument("--subject-taxonomy", type=Path, default=DEFAULT_V5_OUTPUT_DIR / "nature_subject_taxonomy.csv")
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_V5_CORPUS_DIR)
    parser.add_argument("--tau", type=int, default=8)
    parser.add_argument("--min-papers-per-domain", type=int, default=200)
    parser.add_argument("--min-broad-categories", type=int, default=10)
    parser.add_argument("--min-fine-domains", type=int, default=80)
    parser.add_argument("--min-broad-eligible", type=int, default=2000)
    parser.add_argument("--min-domain-eligible", type=int, default=200)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    manifest = materialize(args)
    if not args.quiet:
        print(
            f"[Materialize v5] 已将 {manifest['n_target_works']} 篇目标论文、"
            f"{manifest['n_reference_edges']} 条引用边物化到 {args.corpus_dir}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

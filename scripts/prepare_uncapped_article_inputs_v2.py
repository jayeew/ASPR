#!/usr/bin/env python3
"""Freeze the uncapped mature article universe and its reference edges."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.nature_portfolio_v5 import (
    normalize_openalex_id,
    parse_referenced_works,
    utc_now,
)


def sha256_file(path: Path) -> str:
    """Return the prefixed SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write deterministic JSON through an atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def filter_mature_articles(
    input_path: Path, *, publication_year_max: int
) -> pd.DataFrame:
    """Keep unique English article targets through the latest mature D3 year."""
    frame = pd.read_parquet(input_path)
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    selected = frame[
        frame["document_type"].fillna("").astype(str).eq("article")
        & frame["year"].notna()
        & frame["year"].le(int(publication_year_max))
    ].copy()
    selected["year"] = selected["year"].astype(int)
    selected["id"] = selected["id"].map(normalize_openalex_id)
    selected = selected[selected["id"].astype(str).str.strip().ne("")]
    selected = selected.sort_values(
        ["year", "source_display_name", "id"], kind="stable"
    ).drop_duplicates("id", keep="first")
    return selected.reset_index(drop=True)


def write_reference_edges(targets: pd.DataFrame, path: Path) -> dict[str, int]:
    """Stream target-to-reference edges without constructing a giant DataFrame."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    unique_references: set[str] = set()
    edge_count = 0
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source", "target", "relation", "source_dataset"],
        )
        writer.writeheader()
        for row in targets[["id", "referenced_works"]].itertuples(index=False):
            paper_id = normalize_openalex_id(row.id)
            references = set(parse_referenced_works(row.referenced_works))
            for reference_id in sorted(references):
                writer.writerow(
                    {
                        "source": paper_id,
                        "target": reference_id,
                        "relation": "reference",
                        "source_dataset": "nature_portfolio_uncapped_v2_reference",
                    }
                )
                unique_references.add(reference_id)
                edge_count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {
        "n_reference_edges": edge_count,
        "n_unique_reference_ids": len(unique_references),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    """Create the final article target and reference-edge inputs."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = filter_mature_articles(
        args.uncapped_target_parquet,
        publication_year_max=args.publication_year_max,
    )
    target_csv = args.output_dir / "nature_target_works.csv"
    target_parquet = args.output_dir / "nature_target_works.parquet"
    targets.to_csv(target_csv, index=False)
    targets.to_parquet(target_parquet, index=False, compression="zstd")
    edge_path = args.output_dir / "nature_reference_edges.csv"
    edge_counts = write_reference_edges(targets, edge_path)
    shutil.copy2(args.source_roster, args.output_dir / "nature_source_roster.csv")
    if args.subject_taxonomy.is_file():
        shutil.copy2(
            args.subject_taxonomy,
            args.output_dir / "nature_subject_taxonomy.csv",
        )
    source_year_counts = (
        targets.groupby(["source_display_name", "year"], as_index=False)
        .size()
        .rename(columns={"size": "n_articles"})
    )
    source_year_counts.to_csv(
        args.output_dir / "article_source_year_counts.csv", index=False
    )
    year_counts = targets.groupby("year").size()
    checks = {
        "target_id_unique": bool(not targets["id"].duplicated().any()),
        "articles_only": bool(targets["document_type"].eq("article").all()),
        "publication_year_cap_exact": bool(
            int(targets["year"].max()) == int(args.publication_year_max)
        ),
        "all_targets_have_source": bool(
            targets["source_display_name"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .all()
        ),
        "reference_edge_count_positive": edge_counts["n_reference_edges"] > 0,
        "all_years_present_1980_to_cap": bool(
            set(range(1980, int(args.publication_year_max) + 1)).issubset(
                set(year_counts.index.astype(int))
            )
        ),
    }
    manifest: dict[str, Any] = {
        "artifact_kind": "nature_portfolio_uncapped_v2_mature_articles",
        "created_at": utc_now(),
        "publication_year_max": int(args.publication_year_max),
        "n_target_articles": len(targets),
        "n_sources": int(targets["source_display_name"].nunique()),
        "n_domains": int(targets["domain"].nunique()),
        **edge_counts,
        "quality_checks": checks,
        "overall_pass": bool(all(checks.values())),
        "outputs": {
            "target_csv": str(target_csv.resolve()),
            "target_csv_sha256": sha256_file(target_csv),
            "target_parquet": str(target_parquet.resolve()),
            "reference_edges": str(edge_path.resolve()),
            "reference_edges_sha256": sha256_file(edge_path),
        },
    }
    write_json(args.output_dir / "nature_target_works_manifest.json", manifest)
    if not manifest["overall_pass"]:
        raise RuntimeError(f"Mature article input audit failed: {manifest}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uncapped-target-parquet", type=Path, required=True)
    parser.add_argument("--source-roster", type=Path, required=True)
    parser.add_argument("--subject-taxonomy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--publication-year-max", type=int, default=2022)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the mature article preparation."""
    args = build_parser().parse_args(argv)
    print(json.dumps(build(args), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Read-only, scoped inventory supporting the proposed Fig.8 experiment."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "existing_asset_audit.json"


def parquet_info(path: Path) -> dict[str, Any]:
    """Inspect schema and row count without materializing large data files."""
    file = pq.ParquetFile(path)
    return {
        "path": str(path.relative_to(ROOT)),
        "rows": file.metadata.num_rows,
        "columns": file.schema_arrow.names,
        "bytes": path.stat().st_size,
    }


def counts(values: list[Any]) -> dict[str, int]:
    return dict(Counter(str(value) if value is not None else "<null>" for value in values))


def inspect_targets() -> dict[str, Any]:
    path = ROOT / "data/claim_graph/canonical_target_works.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    names = ["source_name", "publication_year", "work_type", "field_name"]
    result: dict[str, Any] = {name: counts(table[name].to_pylist()) for name in names}
    result["missing"] = {
        name: sum(row[name] is None or row[name] == "" for row in rows)
        for name in table.column_names
    }
    result["duplicate_keys"] = {
        name: len(rows) - len({row[name] for row in rows})
        for name in ["work_id", "doi", "nature_article_id"]
    }
    result["journal_field"] = dict(Counter(
        f"{row['source_name']} | {row['field_name']}" for row in rows
    ))
    claims = pq.read_table(
        ROOT / "data/claim_graph/claim_nodes.parquet",
        columns=["claim_id", "parent_paper_id", "claim_type", "publication_date"],
    ).to_pylist()
    by_id = {row["nature_article_id"]: row for row in rows}
    result["claim_count"] = len(claims)
    result["claim_bearing_papers"] = len({row["parent_paper_id"] for row in claims})
    result["claim_types"] = counts([row["claim_type"] for row in claims])
    result["orphan_claims"] = sum(row["parent_paper_id"] not in by_id for row in claims)
    result["claim_date_differs_from_canonical"] = sum(
        str(row["publication_date"]) != by_id[row["parent_paper_id"]]["publication_date"]
        for row in claims if row["parent_paper_id"] in by_id
    )
    return result


def inspect_legacy_corpus() -> dict[str, Any]:
    root = ROOT / "data/knowledge_corpus/nature_multihorizon_v6_1_uncapped_v2"
    names = ["papers_common_all.parquet", "target_openalex_metadata.parquet", "reference_metadata.parquet"]
    result: dict[str, Any] = {name: parquet_info(root / name) for name in names}
    table = pq.read_table(root / names[0], columns=[
        "paper_id", "publication_year", "source_display_name", "journal_family",
    ])
    for name in ["publication_year", "journal_family"]:
        result[name] = counts(table[name].to_pylist())
    result["nature_main_rows"] = sum(value == "Nature" for value in table["source_display_name"].to_pylist())
    result["unique_paper_ids"] = len(set(table["paper_id"].to_pylist()))
    result["top_sources"] = dict(Counter(table["source_display_name"].to_pylist()).most_common(15))
    return result


def main() -> None:
    graph = ROOT / "data/claim_graph"
    names = [
        "canonical_target_works", "nature_targets", "abstracts", "abstract_sentences",
        "claim_nodes", "claim_embedding_index", "claim_communities",
        "semantic_claim_edges", "paper_nodes", "paper_edges",
    ]
    result: dict[str, Any] = {
        "audit_date": "2026-10-21",
        "scope": "Local inventory and selected integrity checks; not a census or semantic validation",
        "graph_files": {name: parquet_info(graph / f"{name}.parquet") for name in names},
        "targets": inspect_targets(),
        "legacy_corpus": inspect_legacy_corpus(),
        "external_paths_exist": {
            path: Path(path).exists() for path in [
                "/mnt/d/FabCitationData/openalex-snapshot",
                "/home/jayee/workspace/FabCitation/openalex_snapshot_reference_check_results/analysis_state.db",
                "/mnt/d/aspr_nature_markdown/manifest.jsonl",
            ]
        },
    }
    roster = ROOT / "outputs/innovation_200_20260907/papers.jsonl"
    papers = [json.loads(line) for line in roster.read_text().splitlines() if line.strip()]
    result["study_roster"] = {
        "path": str(roster.relative_to(ROOT)), "rows": len(papers),
        "journals": counts([row["journal_name"] for row in papers]),
    }
    for name, path in {
        "fig8_qa": ROOT / "outputs/fig8_reference/qa/data_validation.json",
        "testset_2026": ROOT / "data/nature_2026_testset/summary.json",
    }.items():
        result[name] = json.loads(path.read_text())
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(OUT)
    print(json.dumps({
        "target_rows": result["graph_files"]["canonical_target_works"]["rows"],
        "target_missing_ids": {key: result["targets"]["missing"][key]
                               for key in ["source_id", "field_id", "primary_topic_id"]},
        "legacy_nature_main_rows": result["legacy_corpus"]["nature_main_rows"],
        "legacy_years": result["legacy_corpus"]["publication_year"],
        "fields": result["targets"]["field_name"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
DATABASE = OUTPUT_DIR / "expanded_search.sqlite3"


def read_json(path: Path) -> Dict[str, Any]:
    """Read one JSON object."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write deterministic JSON."""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    """Write deterministic UTF-8 CSV."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    """Return the streaming SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def query_rows(
    connection: sqlite3.Connection,
) -> List[Dict[str, Any]]:
    """Return one audit row per frozen query."""
    rows: List[Dict[str, Any]] = []
    for raw in connection.execute(
        """
        SELECT query_id, expression, reported_total, retrieved_rows,
               unique_hits, pages, complete, stopped_reason, error
        FROM query_runs
        WHERE provider = 'OpenAlex'
        ORDER BY query_id
        """
    ):
        row = dict(raw)
        row["domain_id"] = str(row["query_id"]).split("__", maxsplit=1)[0]
        row["reported_minus_retrieved"] = (
            int(row["reported_total"] or 0) - int(row["retrieved_rows"])
        )
        rows.append(row)
    return rows


def seed_rows(
    connection: sqlite3.Connection,
    evidence: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Audit exact DOI recall for all prior evidence sources."""
    rows: List[Dict[str, Any]] = []
    for record in sorted(evidence, key=lambda value: str(value["source_id"])):
        doi = str(record["doi"]).casefold()
        work = connection.execute(
            """
            SELECT record_key, title
            FROM works
            WHERE provider = 'OpenAlex' AND doi = ?
            """,
            (doi,),
        ).fetchone()
        query_ids: List[str] = []
        if work is not None:
            query_ids = [
                str(row[0])
                for row in connection.execute(
                    """
                    SELECT query_id
                    FROM query_hits
                    WHERE provider = 'OpenAlex' AND record_key = ?
                    ORDER BY query_id
                    """,
                    (work["record_key"],),
                )
            ]
        rows.append(
            {
                "source_id": record["source_id"],
                "doi": doi,
                "year": record["year"],
                "dimension_ids": "|".join(record["dimension_ids"]),
                "found_by_frozen_queries": bool(work),
                "query_count": len(query_ids),
                "query_ids": "|".join(query_ids),
                "retrieved_title": work["title"] if work else "",
                "citation": record["citation"],
            }
        )
    return rows


def title_sample_rows(
    connection: sqlite3.Connection,
    per_query: int = 5,
) -> List[Dict[str, Any]]:
    """Return the top deterministic title sample for every nonempty query."""
    rows: List[Dict[str, Any]] = []
    query_ids = [
        str(row[0])
        for row in connection.execute(
            """
            SELECT query_id
            FROM query_runs
            WHERE provider = 'OpenAlex' AND unique_hits > 0
            ORDER BY query_id
            """
        )
    ]
    for query_id in query_ids:
        for row in connection.execute(
            """
            SELECT h.rank, w.doi, w.title, w.publication_year,
                   w.work_type, (w.abstract <> '') AS has_abstract
            FROM query_hits AS h
            JOIN works AS w
              ON h.provider = w.provider
             AND h.record_key = w.record_key
            WHERE h.provider = 'OpenAlex' AND h.query_id = ?
            ORDER BY h.rank, w.record_key
            LIMIT ?
            """,
            (query_id, per_query),
        ):
            rows.append(
                {
                    "query_id": query_id,
                    "domain_id": query_id.split("__", maxsplit=1)[0],
                    **dict(row),
                }
            )
    return rows


def domain_counts(
    connection: sqlite3.Connection,
) -> Dict[str, Dict[str, int]]:
    """Count links and unique works within each search domain."""
    links: Counter[str] = Counter()
    works: Dict[str, set[str]] = defaultdict(set)
    for row in connection.execute(
        """
        SELECT query_id, record_key
        FROM query_hits
        WHERE provider = 'OpenAlex'
        """
    ):
        domain_id = str(row["query_id"]).split("__", maxsplit=1)[0]
        links[domain_id] += 1
        works[domain_id].add(str(row["record_key"]))
    return {
        domain_id: {
            "query_record_links": links[domain_id],
            "unique_works": len(works[domain_id]),
        }
        for domain_id in sorted(links)
    }


def report_markdown(summary: Mapping[str, Any]) -> str:
    """Render the retrieval audit in Chinese."""
    lines = [
        "# OpenAlex 扩展检索质量审计",
        "",
        "## 结论",
        "",
        (
            f"72 条冻结检索式已全部分页完成，共返回 "
            f"{summary['query_record_links']:,} 个检索式—论文链接，去重后为 "
            f"{summary['unique_works']:,} 篇唯一论文。"
        ),
        "",
        (
            f"其中 {summary['unique_dois']:,} 篇有 DOI，"
            f"{summary['works_with_abstract']:,} 篇有 OpenAlex 摘要。"
        ),
        "",
        (
            f"53 篇已知证据种子仅召回 {summary['seed_sources_recalled']} 篇"
            f"（{summary['seed_recall_rate']:.1%}）。因此检索的机械完整性已经"
            "通过，但敏感性门槛未通过，不能把当前集合称为最终完备检索。"
        ),
        "",
        "## 下一步",
        "",
        "1. 冻结并保留当前 v2 数据库及哈希，不覆盖。",
        "2. 对漏召回种子做 OpenAlex DOI 可索引性核验。",
        "3. 建立 v2.1 高召回补充检索式，并执行前后向引文追踪。",
        "4. 合并去重后重新计算种子召回，再进入双人题录筛选。",
        "",
        "## 各领域规模",
        "",
    ]
    for domain_id, counts in summary["domain_counts"].items():
        lines.append(
            f"- {domain_id}: {counts['unique_works']:,} 篇唯一论文，"
            f"{counts['query_record_links']:,} 个链接"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Audit completeness, duplication, metadata coverage, and seed recall."""
    if not DATABASE.exists():
        raise RuntimeError(f"Missing retrieval database: {DATABASE}")
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    queries = query_rows(connection)
    evidence = read_json(
        OUTPUT_DIR / "literature_evidence_v2.json"
    )["records"]
    seeds = seed_rows(connection, evidence)
    work_counts = connection.execute(
        """
        SELECT COUNT(*) AS works,
               SUM(doi <> '') AS dois,
               SUM(abstract <> '') AS abstracts
        FROM works
        WHERE provider = 'OpenAlex'
        """
    ).fetchone()
    links = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM query_hits
            WHERE provider = 'OpenAlex'
            """
        ).fetchone()[0]
    )
    multiplicity = {
        str(row["query_count"]): int(row["work_count"])
        for row in connection.execute(
            """
            SELECT query_count, COUNT(*) AS work_count
            FROM (
                SELECT record_key, COUNT(*) AS query_count
                FROM query_hits
                WHERE provider = 'OpenAlex'
                GROUP BY record_key
            )
            GROUP BY query_count
            ORDER BY query_count
            """
        )
    }
    samples = title_sample_rows(connection)
    domains = domain_counts(connection)
    connection.close()
    recalled = sum(bool(row["found_by_frozen_queries"]) for row in seeds)
    summary = {
        "schema_version": "2.0.0",
        "retrieval_mechanically_complete": (
            len(queries) == 72
            and all(bool(row["complete"]) for row in queries)
            and all(not row["error"] for row in queries)
            and all(
                int(row["reported_minus_retrieved"]) == 0
                for row in queries
            )
        ),
        "retrieval_sensitivity_gate_passed": recalled == len(seeds),
        "formal_retrieval_accepted": False,
        "formal_retrieval_status": (
            "complete_cursor_snapshot_but_seed_recall_requires_v2_1"
        ),
        "query_count": len(queries),
        "queries_complete": sum(bool(row["complete"]) for row in queries),
        "queries_with_errors": sum(bool(row["error"]) for row in queries),
        "queries_with_reported_retrieved_difference": sum(
            int(row["reported_minus_retrieved"]) != 0 for row in queries
        ),
        "reported_total_before_cross_query_deduplication": sum(
            int(row["reported_total"] or 0) for row in queries
        ),
        "query_record_links": links,
        "unique_works": int(work_counts["works"]),
        "unique_dois": int(work_counts["dois"]),
        "works_with_abstract": int(work_counts["abstracts"]),
        "cross_query_duplicate_links": links - int(work_counts["works"]),
        "work_query_multiplicity": multiplicity,
        "seed_sources_total": len(seeds),
        "seed_sources_recalled": recalled,
        "seed_sources_missed": len(seeds) - recalled,
        "seed_recall_rate": recalled / len(seeds),
        "domain_counts": domains,
        "database_sha256": sha256_file(DATABASE),
        "next_action": (
            "Preserve v2; run direct DOI indexability checks, v2.1 "
            "high-recall supplemental searches, and citation chaining."
        ),
    }
    write_json(OUTPUT_DIR / "retrieval_audit_summary_v2.json", summary)
    write_csv(
        OUTPUT_DIR / "query_retrieval_audit_v2.csv",
        queries,
        (
            "query_id",
            "domain_id",
            "expression",
            "reported_total",
            "retrieved_rows",
            "unique_hits",
            "pages",
            "reported_minus_retrieved",
            "complete",
            "stopped_reason",
            "error",
        ),
    )
    write_csv(
        OUTPUT_DIR / "seed_recall_audit_v2.csv",
        seeds,
        (
            "source_id",
            "doi",
            "year",
            "dimension_ids",
            "found_by_frozen_queries",
            "query_count",
            "query_ids",
            "retrieved_title",
            "citation",
        ),
    )
    write_csv(
        OUTPUT_DIR / "query_top_title_sample_v2.csv",
        samples,
        (
            "query_id",
            "domain_id",
            "rank",
            "doi",
            "title",
            "publication_year",
            "work_type",
            "has_abstract",
        ),
    )
    (OUTPUT_DIR / "retrieval_audit_report_v2.md").write_text(
        report_markdown(summary),
        encoding="utf-8",
    )
    print(
        f"Audited {summary['unique_works']} unique works; "
        f"seed recall {recalled}/{len(seeds)}."
    )


if __name__ == "__main__":
    main()

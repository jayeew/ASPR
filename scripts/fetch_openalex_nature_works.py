from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gear.env import getenv  # noqa: E402
from scripts.build_openalex_v3_citation_graph import OpenAlexClient, split_api_keys  # noqa: E402
from scripts.nature_portfolio_v5 import (  # noqa: E402
    DEFAULT_COMPLETE_END_YEAR,
    DEFAULT_V5_OUTPUT_DIR,
    GRAPH_WORK_TYPES,
    OPENALEX_WORK_SELECT_V5,
    TARGET_WORK_TYPES,
    append_jsonl,
    nonempty,
    openalex_source_filter,
    read_csv,
    read_jsonl,
    target_work_row,
    utc_now,
    write_json,
)


def fetch_source_works(
    source_row: Dict[str, Any],
    *,
    openalex: OpenAlexClient,
    start_year: int,
    end_year: int,
    max_works: int,
    work_types: Sequence[str],
    per_page: int,
    quiet: bool,
) -> List[Dict[str, Any]]:
    source_id = nonempty(source_row.get("source_id"))
    if not source_id:
        return []
    filters = [
        openalex_source_filter(source_id),
        f"from_publication_date:{int(start_year)}-01-01",
        f"to_publication_date:{int(end_year)}-12-31",
        "language:en",
        "type:" + "|".join(work_types),
        "is_retracted:false",
        "is_paratext:false",
    ]
    works = openalex.list_works(
        max_records=int(max_works),
        filters=filters,
        sort="publication_date:asc",
        per_page=int(per_page),
        progress=not quiet,
        progress_label=f"[{source_row.get('source_display_name')}] 目标论文",
    )
    return works


def load_source_roster(path: Path, max_sources: Optional[int] = None) -> pd.DataFrame:
    roster = read_csv(path)
    if roster.empty:
        raise FileNotFoundError(f"No source roster rows found: {path}")
    if "source_id" not in roster.columns:
        roster["source_id"] = ""
    roster = roster[roster["source_id"].fillna("").astype(str).str.strip().ne("")].copy()
    if max_sources:
        roster = roster.head(int(max_sources)).copy()
    if roster.empty:
        raise ValueError(f"Source roster has no OpenAlex source_id values: {path}")
    return roster.reset_index(drop=True)


def fetch_one_source(
    idx: int,
    total: int,
    source: Dict[str, Any],
    *,
    args: argparse.Namespace,
    openalex: OpenAlexClient,
    checkpoint_dir: Path,
    fetched_at: str,
) -> Dict[str, Any]:
    source_name = nonempty(source.get("source_display_name")) or nonempty(source.get("openalex_source_display_name"))
    checkpoint = checkpoint_dir / f"{idx:04d}_{source_name.replace('/', '_').replace(' ', '_')}.jsonl"
    if checkpoint.exists() and not args.refresh:
        raw_records = [item.get("work", item) for item in read_jsonl(checkpoint)]
        status = "checkpoint"
    else:
        if not args.quiet:
            print(f"[Nature works] 正在拉取来源 {idx}/{total}：{source_name}", flush=True)
        try:
            raw_records = fetch_source_works(
                source,
                openalex=openalex,
                start_year=args.start_year,
                end_year=args.end_year,
                max_works=args.max_works_per_source,
                work_types=args.work_types,
                per_page=args.per_page,
                quiet=args.quiet or int(args.workers) > 1,
            )
            checkpoint.unlink(missing_ok=True)
            append_jsonl(checkpoint, [{"source": source_name, "work": work} for work in raw_records])
            status = "fetched"
        except Exception as exc:
            raw_records = []
            status = f"fetch_failed:{exc}"
    source_rows = [target_work_row(work, source_row=source, fetched_at=fetched_at) for work in raw_records]
    report_row = {
        "source_display_name": source_name,
        "source_id": source.get("source_id", ""),
        "journal_family": source.get("journal_family", ""),
        "broad_category": source.get("broad_category", ""),
        "status": status,
        "n_raw_records": int(len(raw_records)),
        "n_target_rows": int(len(source_rows)),
        "checkpoint": str(checkpoint),
    }
    return {"idx": idx, "rows": source_rows, "report": report_row}


def fetch_nature_works(args: argparse.Namespace) -> Dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.checkpoint_dir or (args.out_dir / "checkpoints" / "target_works")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    roster = load_source_roster(args.source_roster, max_sources=args.max_sources)
    openalex = OpenAlexClient(
        api_key=args.openalex_api_key,
        api_keys=split_api_keys(args.openalex_api_keys),
        email=args.openalex_email,
        sleep_seconds=args.sleep_seconds,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
    )

    fetched_at = utc_now()
    rows: List[Dict[str, Any]] = []
    report_rows: List[Dict[str, Any]] = []
    source_records = roster.to_dict("records")
    workers = max(1, int(args.workers))
    if workers == 1:
        results = [
            fetch_one_source(
                idx,
                len(source_records),
                source,
                args=args,
                openalex=openalex,
                checkpoint_dir=checkpoint_dir,
                fetched_at=fetched_at,
            )
            for idx, source in enumerate(source_records, start=1)
        ]
    else:
        if not args.quiet:
            print(f"[Nature works] 启用并发拉取：workers={workers}，来源数={len(source_records)}", flush=True)
        results = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    fetch_one_source,
                    idx,
                    len(source_records),
                    source,
                    args=args,
                    openalex=openalex,
                    checkpoint_dir=checkpoint_dir,
                    fetched_at=fetched_at,
                )
                for idx, source in enumerate(source_records, start=1)
            ]
            for done, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                if not args.quiet:
                    report = result["report"]
                    print(
                        f"[Nature works] 来源完成 {done}/{len(futures)}："
                        f"{report['source_display_name']}，rows={report['n_target_rows']:,}，status={report['status']}",
                        flush=True,
                    )
    for result in sorted(results, key=lambda item: int(item["idx"])):
        rows.extend(result["rows"])
        report_rows.append(result["report"])

    works = pd.DataFrame(rows)
    if not works.empty:
        works = (
            works.sort_values(["year", "source_display_name", "id"])
            .drop_duplicates("id", keep="first")
            .reset_index(drop=True)
        )
    works_path = args.out_dir / "nature_target_works.csv"
    works.to_csv(works_path, index=False)
    report = pd.DataFrame(report_rows)
    report.to_csv(args.out_dir / "nature_target_fetch_report.csv", index=False)
    manifest = {
        "artifact_kind": "nature_portfolio_v5_target_works_fetch",
        "created_at": fetched_at,
        "source_roster": str(args.source_roster),
        "out_dir": str(args.out_dir),
        "target_works": str(works_path),
        "fetch_report": str(args.out_dir / "nature_target_fetch_report.csv"),
        "checkpoint_dir": str(checkpoint_dir),
        "start_year": int(args.start_year),
        "end_year": int(args.end_year),
        "work_types": list(args.work_types),
        "max_works_per_source": int(args.max_works_per_source),
        "workers": int(args.workers),
        "n_sources": int(len(roster)),
        "n_target_works": int(len(works)),
        "n_broad_categories": int(works["broad_category"].nunique()) if not works.empty and "broad_category" in works.columns else 0,
        "n_fine_domains": int(works["domain"].nunique()) if not works.empty and "domain" in works.columns else 0,
    }
    write_json(args.out_dir / "nature_target_works_manifest.json", manifest)
    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Nature Portfolio target works from OpenAlex sources.")
    parser.add_argument("--source-roster", type=Path, default=DEFAULT_V5_OUTPUT_DIR / "nature_source_roster.csv")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_V5_OUTPUT_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--max-sources", type=int, default=None)
    parser.add_argument("--max-works-per-source", type=int, default=5000)
    parser.add_argument("--start-year", type=int, default=1980)
    parser.add_argument("--end-year", type=int, default=DEFAULT_COMPLETE_END_YEAR)
    parser.add_argument("--work-types", nargs="+", default=list(TARGET_WORK_TYPES))
    parser.add_argument("--per-page", type=int, default=200)
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--openalex-api-key", default=getenv("OPENALEX_API_KEY"))
    parser.add_argument("--openalex-api-keys", default=getenv("OPENALEX_API_KEYS"))
    parser.add_argument("--openalex-email", default=getenv("OPENALEX_EMAIL"))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    # Keep this import-side constant visible for users who choose a broader graph pull.
    _ = GRAPH_WORK_TYPES
    manifest = fetch_nature_works(args)
    if not args.quiet:
        print(
            f"[Nature works] 已写入 {manifest['n_target_works']} 篇目标论文，"
            f"覆盖 {manifest['n_sources']} 个来源，输出到 {manifest['target_works']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

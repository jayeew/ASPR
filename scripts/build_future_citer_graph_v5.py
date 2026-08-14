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
    OPENALEX_WORK_SELECT_V5,
    entropy_from_counts,
    normalize_openalex_id,
    read_csv,
    read_jsonl,
    short_openalex_id,
    simpson_from_counts,
    topic_metadata,
    utc_now,
    write_json,
    write_jsonl,
)


def _future_checkpoint_path(checkpoint_dir: Path, paper_id: str) -> Path:
    return checkpoint_dir / f"{short_openalex_id(paper_id)}.jsonl"


def fetch_future_citers_for_work(
    row: Dict[str, Any],
    *,
    openalex: OpenAlexClient,
    tau: int,
    max_citers_per_work: int,
    per_page: int,
) -> List[Dict[str, Any]]:
    paper_id = normalize_openalex_id(row.get("id"))
    year_value = pd.to_numeric(row.get("year"), errors="coerce")
    if not paper_id or pd.isna(year_value):
        return []
    year = int(year_value)
    if year <= 0:
        return []
    start = year + 1
    end = min(DEFAULT_COMPLETE_END_YEAR, year + int(tau))
    if start > end:
        return []
    filters = [
        f"cites:{short_openalex_id(paper_id)}",
        f"from_publication_date:{start}-01-01",
        f"to_publication_date:{end}-12-31",
        "language:en",
        "is_retracted:false",
        "is_paratext:false",
    ]
    return openalex.list_works(
        max_records=int(max_citers_per_work),
        filters=filters,
        sort="publication_date:asc",
        per_page=int(per_page),
        progress=False,
    )


def future_delta_row(paper: Dict[str, Any], citers: Sequence[Dict[str, Any]], tau: int) -> Dict[str, Any]:
    fields: List[str] = []
    subfields: List[str] = []
    topics: List[str] = []
    years: List[int] = []
    for work in citers:
        meta = topic_metadata(work)
        if meta.get("primary_field"):
            fields.append(meta["primary_field"])
        if meta.get("primary_subfield"):
            subfields.append(meta["primary_subfield"])
        if meta.get("primary_topic"):
            topics.append(meta["primary_topic"])
        year = pd.to_numeric(work.get("publication_year"), errors="coerce")
        if pd.notna(year):
            years.append(int(year))
    field_counts = pd.Series(fields).value_counts() if fields else pd.Series(dtype=int)
    subfield_counts = pd.Series(subfields).value_counts() if subfields else pd.Series(dtype=int)
    topic_counts = pd.Series(topics).value_counts() if topics else pd.Series(dtype=int)
    n = int(len(citers))
    return {
        "paper_id": normalize_openalex_id(paper.get("id")),
        "year": int(pd.to_numeric(paper.get("year"), errors="coerce")),
        "tau": int(tau),
        "n_future_citers": n,
        "future_community_reach": int(topic_counts.size),
        "future_field_reach": int(field_counts.size),
        "future_subfield_reach": int(subfield_counts.size),
        "future_field_entropy": entropy_from_counts(field_counts.to_numpy(dtype=float)) if len(field_counts) else 0.0,
        "future_topic_entropy": entropy_from_counts(topic_counts.to_numpy(dtype=float)) if len(topic_counts) else 0.0,
        "future_field_simpson": simpson_from_counts(field_counts.to_numpy(dtype=float)) if len(field_counts) else 0.0,
        "future_topic_simpson": simpson_from_counts(topic_counts.to_numpy(dtype=float)) if len(topic_counts) else 0.0,
        "future_first_year": min(years) if years else "",
        "future_last_year": max(years) if years else "",
    }


def build_future_graph(args: argparse.Namespace) -> Dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.checkpoint_dir or (args.out_dir / "checkpoints" / f"future_citers_tau{args.tau}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    targets = read_csv(args.target_works)
    if targets.empty:
        raise FileNotFoundError(f"No target works found: {args.target_works}")
    targets["year"] = pd.to_numeric(targets["year"], errors="coerce")
    eligible = targets[targets["year"].notna() & (targets["year"].astype(int) <= DEFAULT_COMPLETE_END_YEAR - int(args.tau))].copy()
    if args.max_papers is not None:
        eligible = eligible.head(int(args.max_papers)).copy()
    openalex = OpenAlexClient(
        api_key=args.openalex_api_key,
        api_keys=split_api_keys(args.openalex_api_keys),
        email=args.openalex_email,
        sleep_seconds=args.sleep_seconds,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
    )

    def process_one(idx: int, paper: Dict[str, Any]) -> Dict[str, Any]:
        paper_id = normalize_openalex_id(paper.get("id"))
        checkpoint = _future_checkpoint_path(checkpoint_dir, paper_id)
        if checkpoint.exists() and not args.refresh:
            citers = [item.get("work", item) for item in read_jsonl(checkpoint)]
            status = "checkpoint"
        else:
            if not args.quiet and int(args.workers) == 1 and (idx == 1 or idx % 100 == 0 or idx == len(eligible)):
                print(f"[Future citers v5] 正在拉取 future citers：{idx:,}/{len(eligible):,}", flush=True)
            try:
                citers = fetch_future_citers_for_work(
                    paper,
                    openalex=openalex,
                    tau=args.tau,
                    max_citers_per_work=args.max_citers_per_work,
                    per_page=args.per_page,
                )
                write_jsonl(checkpoint, [{"paper_id": paper_id, "work": work} for work in citers])
                status = "fetched"
            except Exception as exc:
                citers = []
                status = f"fetch_failed:{exc}"
        local_citer_rows: List[Dict[str, Any]] = []
        for citer in citers:
            meta = topic_metadata(citer)
            citer_year_value = pd.to_numeric(citer.get("publication_year"), errors="coerce")
            citer_year = int(citer_year_value) if pd.notna(citer_year_value) else 0
            local_citer_rows.append(
                {
                    "paper_id": paper_id,
                    "citer_id": normalize_openalex_id(citer.get("id")),
                    "citer_year": citer_year,
                    "citer_primary_field": meta.get("primary_field", ""),
                    "citer_primary_subfield": meta.get("primary_subfield", ""),
                    "citer_primary_topic": meta.get("primary_topic", ""),
                    "fetch_status": status,
                }
            )
        return {
            "idx": idx,
            "paper_id": paper_id,
            "citer_rows": local_citer_rows,
            "delta_row": future_delta_row(paper, citers, tau=args.tau),
            "status": status,
        }

    citer_rows: List[Dict[str, Any]] = []
    delta_rows: List[Dict[str, Any]] = []
    papers = eligible.to_dict("records")
    workers = max(1, int(args.workers))
    if workers == 1:
        results = [process_one(idx, paper) for idx, paper in enumerate(papers, start=1)]
    else:
        if not args.quiet:
            print(f"[Future citers v5] 启用并发拉取：workers={workers}，论文数={len(papers):,}", flush=True)
        results = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_one, idx, paper) for idx, paper in enumerate(papers, start=1)]
            for done, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                if not args.quiet and (done == 1 or done % 100 == 0 or done == len(futures)):
                    print(
                        f"[Future citers v5] 已完成 future-citer 请求 {done:,}/{len(futures):,}，"
                        f"paper={result['paper_id']}，citers={len(result['citer_rows']):,}",
                        flush=True,
                    )
    for result in sorted(results, key=lambda item: int(item["idx"])):
        citer_rows.extend(result["citer_rows"])
        delta_rows.append(result["delta_row"])

    future_citers = pd.DataFrame(citer_rows)
    future_deltas = pd.DataFrame(delta_rows)
    citers_path = args.out_dir / "nature_future_citers.csv"
    deltas_path = args.out_dir / "nature_future_graph_deltas.csv"
    future_citers.to_csv(citers_path, index=False)
    future_deltas.to_csv(deltas_path, index=False)
    coverage = float((future_deltas["n_future_citers"] > 0).mean()) if not future_deltas.empty else 0.0
    manifest = {
        "artifact_kind": "nature_portfolio_v5_future_citer_graph",
        "created_at": utc_now(),
        "target_works": str(args.target_works),
        "future_citers": str(citers_path),
        "future_graph_deltas": str(deltas_path),
        "checkpoint_dir": str(checkpoint_dir),
        "tau": int(args.tau),
        "max_citers_per_work": int(args.max_citers_per_work),
        "workers": int(args.workers),
        "n_target_works": int(len(targets)),
        "n_tau_eligible_works": int(len(eligible)),
        "n_future_citer_rows": int(len(future_citers)),
        "n_future_delta_rows": int(len(future_deltas)),
        "future_citer_coverage": coverage,
        "no_leakage_contract": "future_citers_and_future_graph_deltas_are_label_only_not_publication_day_features",
    }
    write_json(args.out_dir / "future_citer_graph_manifest.json", manifest)
    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build future-citer graph deltas for Nature Portfolio v5 labels.")
    parser.add_argument("--target-works", type=Path, default=DEFAULT_V5_OUTPUT_DIR / "nature_target_works.csv")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_V5_OUTPUT_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--tau", type=int, default=8)
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument("--max-citers-per-work", type=int, default=500)
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
    _ = OPENALEX_WORK_SELECT_V5
    manifest = build_future_graph(args)
    if not args.quiet:
        print(
            f"[Future citers v5] 已写入 {manifest['n_future_delta_rows']} 行 future-delta；"
            f"覆盖率={manifest['future_citer_coverage']:.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

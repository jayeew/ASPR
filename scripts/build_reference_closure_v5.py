from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gear.env import getenv  # noqa: E402
from scripts.build_openalex_v3_citation_graph import OpenAlexClient, split_api_keys  # noqa: E402
from scripts.nature_portfolio_v5 import (  # noqa: E402
    DEFAULT_V5_OUTPUT_DIR,
    normalize_openalex_id,
    read_csv,
    read_jsonl,
    reference_work_row,
    target_reference_edges,
    utc_now,
    write_json,
)


def _existing_reference_rows(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for item in read_jsonl(path):
        work = item.get("work", item)
        wid = normalize_openalex_id(work.get("id"))
        if wid:
            rows[wid] = work
    return rows


def fetch_reference_works(
    reference_ids: Sequence[str],
    *,
    openalex: OpenAlexClient,
    checkpoint_jsonl: Path,
    max_refs: Optional[int],
    quiet: bool,
    workers: int = 1,
) -> pd.DataFrame:
    cache = _existing_reference_rows(checkpoint_jsonl)
    wanted = [rid for rid in reference_ids if rid and rid not in cache]
    if max_refs is not None:
        wanted = wanted[: max(0, int(max_refs) - len(cache))]
    checkpoint_jsonl.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def fetch_one(rid: str) -> Optional[Dict[str, Any]]:
        try:
            return openalex.get_work(rid)
        except Exception:
            return None

    def record_work(work: Dict[str, Any]) -> None:
        if work:
            wid = normalize_openalex_id(work.get("id"))
            if not wid:
                return
            with lock:
                if wid in cache:
                    return
                cache[wid] = work
                with checkpoint_jsonl.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"work": work}, ensure_ascii=False, sort_keys=True) + "\n")

    workers = max(1, int(workers))
    if workers == 1:
        for idx, rid in enumerate(wanted, start=1):
            if not quiet and (idx == 1 or idx % 250 == 0 or idx == len(wanted)):
                print(f"[Reference closure v5] 已拉取参考文献 {idx:,}/{len(wanted):,}", flush=True)
            work = fetch_one(rid)
            if work:
                record_work(work)
    elif wanted:
        if not quiet:
            print(f"[Reference closure v5] 启用并发拉取：workers={workers}，待拉取={len(wanted):,}", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(fetch_one, rid) for rid in wanted]
            for idx, future in enumerate(as_completed(futures), start=1):
                work = future.result()
                if work:
                    record_work(work)
                if not quiet and (idx == 1 or idx % 500 == 0 or idx == len(futures)):
                    print(f"[Reference closure v5] 已完成参考文献请求 {idx:,}/{len(futures):,}", flush=True)
    rows = [reference_work_row(work) for work in cache.values()]
    return pd.DataFrame(rows)


def build_reference_closure(args: argparse.Namespace) -> Dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    targets = read_csv(args.target_works)
    if targets.empty:
        raise FileNotFoundError(f"No target works found: {args.target_works}")
    edges = target_reference_edges(targets)
    if args.max_edges is not None:
        edges = edges.head(int(args.max_edges)).copy()
    edges_path = args.out_dir / "nature_reference_edges.csv"
    edges.to_csv(edges_path, index=False)
    reference_ids = sorted(edges["target"].dropna().astype(str).map(normalize_openalex_id).unique().tolist()) if not edges.empty else []
    checkpoint = args.checkpoint_jsonl or (args.out_dir / "checkpoints" / "reference_works.jsonl")
    if args.skip_fetch:
        refs = pd.DataFrame({"id": reference_ids})
    else:
        openalex = OpenAlexClient(
            api_key=args.openalex_api_key,
            api_keys=split_api_keys(args.openalex_api_keys),
            email=args.openalex_email,
            sleep_seconds=args.sleep_seconds,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
        )
        refs = fetch_reference_works(
            reference_ids,
            openalex=openalex,
            checkpoint_jsonl=checkpoint,
            max_refs=args.max_refs,
            quiet=args.quiet,
            workers=args.workers,
        )
    refs_path = args.out_dir / "nature_reference_works.csv"
    refs.to_csv(refs_path, index=False)
    manifest = {
        "artifact_kind": "nature_portfolio_v5_reference_closure",
        "created_at": utc_now(),
        "target_works": str(args.target_works),
        "reference_edges": str(edges_path),
        "reference_works": str(refs_path),
        "checkpoint_jsonl": str(checkpoint),
        "skip_fetch": bool(args.skip_fetch),
        "n_target_works": int(len(targets)),
        "n_reference_edges": int(len(edges)),
        "n_unique_reference_ids": int(len(reference_ids)),
        "n_reference_works": int(len(refs)),
        "max_refs": args.max_refs,
        "max_edges": args.max_edges,
        "workers": int(args.workers),
    }
    write_json(args.out_dir / "reference_closure_manifest.json", manifest)
    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Nature Portfolio v5 reference closure.")
    parser.add_argument("--target-works", type=Path, default=DEFAULT_V5_OUTPUT_DIR / "nature_target_works.csv")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_V5_OUTPUT_DIR)
    parser.add_argument("--checkpoint-jsonl", type=Path, default=None)
    parser.add_argument("--skip-fetch", action="store_true", help="Only materialize reference edges and IDs.")
    parser.add_argument("--max-refs", type=int, default=None)
    parser.add_argument("--max-edges", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--openalex-api-key", default=getenv("OPENALEX_API_KEY"))
    parser.add_argument("--openalex-api-keys", default=getenv("OPENALEX_API_KEYS"))
    parser.add_argument("--openalex-email", default=getenv("OPENALEX_EMAIL"))
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    manifest = build_reference_closure(args)
    if not args.quiet:
        print(
            f"[Reference closure v5] 已写入 {manifest['n_reference_edges']} 条引用边和 "
            f"{manifest['n_reference_works']} 条参考文献元数据",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

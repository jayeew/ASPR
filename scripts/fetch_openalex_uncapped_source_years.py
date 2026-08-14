#!/usr/bin/env python3
"""Replace capped Nature source pulls with complete source-by-year partitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gear.env import getenv
from scripts.build_openalex_v3_citation_graph import (
    OpenAlexClient,
    split_api_keys,
)
from scripts.nature_portfolio_v5 import (
    OPENALEX_WORK_SELECT_V5,
    TARGET_WORK_TYPES,
    nonempty,
    openalex_source_filter,
    target_work_row,
    utc_now,
)

DEFAULT_SOURCE_NAMES = (
    "Nature",
    "Nature Communications",
    "Scientific Reports",
)
LEGACY_CAP = 25_000
SUSTAINED_ACTIVITY_THRESHOLD = 100


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one local artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write deterministic JSON via an atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """Write one partition atomically and return its row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one source-year checkpoint."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def source_slug(value: object) -> str:
    """Return a stable filesystem-safe source label."""
    return "_".join(str(value).strip().lower().replace("&", "and").split())


def load_selected_roster(path: Path, source_names: Sequence[str]) -> pd.DataFrame:
    """Load the exact source rows requested for uncapped replacement."""
    roster = pd.read_csv(path, low_memory=False)
    required = {"source_display_name", "source_id"}
    missing = sorted(required - set(roster.columns))
    if missing:
        raise ValueError(f"Source roster is missing columns: {missing}")
    selected = roster[roster["source_display_name"].isin(source_names)].copy()
    observed = set(selected["source_display_name"].astype(str))
    absent = sorted(set(source_names) - observed)
    if absent:
        raise ValueError(f"Requested source names are absent from roster: {absent}")
    if selected["source_id"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("Selected source roster contains empty OpenAlex IDs")
    return selected.sort_values("source_display_name").reset_index(drop=True)


def partition_filters(
    source_id: object, year: int, work_types: Sequence[str]
) -> list[str]:
    """Return the frozen OpenAlex filter contract for one source-year."""
    return [
        openalex_source_filter(source_id),
        f"from_publication_date:{int(year)}-01-01",
        f"to_publication_date:{int(year)}-12-31",
        "language:en",
        "type:" + "|".join(work_types),
        "is_retracted:false",
        "is_paratext:false",
    ]


def fetch_complete_partition(
    openalex: OpenAlexClient,
    *,
    filters: Sequence[str],
    per_page: int,
) -> tuple[list[dict[str, Any]], int, int]:
    """Exhaust a cursor and require fetched rows to equal ``meta.count``."""
    params: dict[str, Any] = {
        "select": OPENALEX_WORK_SELECT_V5,
        "per-page": min(200, max(10, int(per_page))),
        "cursor": "*",
        "filter": ",".join(filters),
        "sort": "publication_date:asc",
    }
    rows: list[dict[str, Any]] = []
    expected: int | None = None
    pages = 0
    while True:
        payload = openalex.get_json("/works", params=params)
        pages += 1
        meta = payload.get("meta") or {}
        if expected is None:
            expected = int(meta.get("count", 0))
        results = payload.get("results") or []
        rows.extend(dict(item) for item in results)
        next_cursor = meta.get("next_cursor")
        if not results or not next_cursor:
            break
        params["cursor"] = next_cursor
    expected = int(expected or 0)
    unique_ids = {nonempty(row.get("id")) for row in rows if nonempty(row.get("id"))}
    if len(rows) != expected or len(unique_ids) != expected:
        raise RuntimeError(
            "Incomplete source-year cursor: "
            f"meta.count={expected}, rows={len(rows)}, unique={len(unique_ids)}"
        )
    return rows, expected, pages


def partition_paths(
    checkpoint_dir: Path, source_name: str, year: int
) -> tuple[Path, Path]:
    """Return data and manifest paths for one partition."""
    root = checkpoint_dir / f"source={source_slug(source_name)}"
    return root / f"year={year}.jsonl", root / f"year={year}.manifest.json"


def valid_checkpoint(data_path: Path, manifest_path: Path) -> bool:
    """Return whether a checkpoint has a complete, hash-verified manifest."""
    if not data_path.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        manifest.get("status") == "complete"
        and int(manifest.get("expected_count", -1))
        == int(manifest.get("fetched_count", -2))
        and manifest.get("sha256") == sha256_file(data_path)
    )


def fetch_partition_job(
    source: Mapping[str, Any],
    year: int,
    *,
    args: argparse.Namespace,
    openalex: OpenAlexClient,
) -> dict[str, Any]:
    """Fetch or resume one complete source-year partition."""
    source_name = str(source["source_display_name"])
    data_path, manifest_path = partition_paths(args.checkpoint_dir, source_name, year)
    if not args.refresh and valid_checkpoint(data_path, manifest_path):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return dict(payload, checkpoint_status="resumed")
    filters = partition_filters(source["source_id"], year, args.work_types)
    rows, expected, pages = fetch_complete_partition(
        openalex, filters=filters, per_page=args.per_page
    )
    fetched = write_jsonl(data_path, rows)
    manifest: dict[str, Any] = {
        "artifact_kind": "openalex_uncapped_source_year_partition",
        "source_display_name": source_name,
        "source_id": source["source_id"],
        "year": int(year),
        "filters": filters,
        "expected_count": expected,
        "fetched_count": fetched,
        "unique_count": len({row.get("id") for row in rows}),
        "pages": pages,
        "status": "complete",
        "checkpoint": str(data_path.resolve()),
        "sha256": sha256_file(data_path),
        "fetched_at": utc_now(),
    }
    write_json(manifest_path, manifest)
    return dict(manifest, checkpoint_status="fetched")


def fetch_all_partitions(
    roster: pd.DataFrame, args: argparse.Namespace
) -> pd.DataFrame:
    """Fetch all requested source-year partitions concurrently."""
    openalex = OpenAlexClient(
        api_key=args.openalex_api_key,
        api_keys=split_api_keys(args.openalex_api_keys),
        email=args.openalex_email,
        sleep_seconds=args.sleep_seconds,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
    )
    jobs = [
        (source, year)
        for year in range(args.start_year, args.end_year + 1)
        for source in roster.to_dict("records")
    ]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = {
            executor.submit(
                fetch_partition_job,
                source,
                year,
                args=args,
                openalex=openalex,
            ): (source["source_display_name"], year)
            for source, year in jobs
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if not args.quiet and (
                completed == 1 or completed % 10 == 0 or completed == len(futures)
            ):
                print(
                    "[Uncapped source-year] "
                    f"partitions={completed}/{len(futures)}, "
                    f"latest={result['source_display_name']} {result['year']} "
                    f"rows={result['fetched_count']}",
                    flush=True,
                )
    return pd.DataFrame(results).sort_values(
        ["source_display_name", "year"], kind="stable"
    )


def audit_annual_continuity(partitions: pd.DataFrame) -> dict[str, Any]:
    """Require complete counts and no internal zero year per active source."""
    count_match = partitions["expected_count"].eq(partitions["fetched_count"])
    unique_match = partitions["expected_count"].eq(partitions["unique_count"])
    source_audits: list[dict[str, Any]] = []
    for source_name, group in partitions.groupby("source_display_name", sort=True):
        active = group[
            group["expected_count"].ge(SUSTAINED_ACTIVITY_THRESHOLD)
        ].sort_values("year")
        internal_zero_years: list[int] = []
        if not active.empty:
            first_year = int(active["year"].min())
            last_year = int(active["year"].max())
            internal_zero_years = (
                group[
                    group["year"].between(first_year, last_year)
                    & group["expected_count"].eq(0)
                ]["year"]
                .astype(int)
                .tolist()
            )
        else:
            first_year = 0
            last_year = 0
        source_audits.append(
            {
                "source_display_name": source_name,
                "first_sustained_year": first_year,
                "last_sustained_year": last_year,
                "sustained_activity_threshold": SUSTAINED_ACTIVITY_THRESHOLD,
                "internal_zero_years": internal_zero_years,
                "expected_total": int(group["expected_count"].sum()),
                "fetched_total": int(group["fetched_count"].sum()),
                "exactly_legacy_cap": bool(
                    int(group["fetched_count"].sum()) == LEGACY_CAP
                ),
            }
        )
    source_frame = pd.DataFrame(source_audits)
    checks = {
        "all_partition_counts_match": bool(count_match.all()),
        "all_partition_ids_unique": bool(unique_match.all()),
        "no_internal_zero_years": bool(
            source_frame["internal_zero_years"].map(len).eq(0).all()
        ),
        "no_source_equals_legacy_cap": bool(~source_frame["exactly_legacy_cap"].any()),
    }
    return {
        "checks": checks,
        "overall_pass": bool(all(checks.values())),
        "sources": source_audits,
    }


def materialize_uncapped_targets(
    roster: pd.DataFrame,
    partitions: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Replace only the three capped sources in the baseline target table."""
    baseline = pd.read_csv(args.baseline_target_works, low_memory=False)
    selected_names = set(roster["source_display_name"].astype(str))
    retained = baseline[
        ~baseline["source_display_name"].astype(str).isin(selected_names)
    ].copy()
    source_lookup = {
        str(row["source_display_name"]): row for row in roster.to_dict("records")
    }
    replacement_rows: list[dict[str, Any]] = []
    for partition in partitions.to_dict("records"):
        source_name = str(partition["source_display_name"])
        for work in read_jsonl(Path(partition["checkpoint"])):
            replacement_rows.append(
                target_work_row(
                    work,
                    source_row=source_lookup[source_name],
                    fetched_at=str(partition["fetched_at"]),
                )
            )
    replacement = pd.DataFrame(replacement_rows)
    combined = pd.concat([retained, replacement], ignore_index=True, sort=False)
    combined = combined.sort_values(
        ["year", "source_display_name", "id"], kind="stable"
    ).drop_duplicates("id", keep="first")
    combined = combined.reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_dir / "nature_target_works.csv", index=False)
    combined.to_parquet(
        args.output_dir / "nature_target_works.parquet",
        index=False,
        compression="zstd",
    )
    return combined


def source_count_audit(
    baseline: pd.DataFrame, combined: pd.DataFrame, selected_names: set[str]
) -> pd.DataFrame:
    """Return before/after counts for every selected source."""
    old_counts = baseline.groupby("source_display_name").size()
    new_counts = combined.groupby("source_display_name").size()
    rows = []
    for source_name in sorted(selected_names):
        old_count = int(old_counts.get(source_name, 0))
        new_count = int(new_counts.get(source_name, 0))
        rows.append(
            {
                "source_display_name": source_name,
                "old_count": old_count,
                "new_count": new_count,
                "added_count": new_count - old_count,
                "old_exactly_legacy_cap": old_count == LEGACY_CAP,
                "new_exactly_legacy_cap": new_count == LEGACY_CAP,
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Fetch, audit, merge, and freeze the uncapped target universe."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    roster = load_selected_roster(args.source_roster, args.source_name)
    partitions = fetch_all_partitions(roster, args)
    continuity = audit_annual_continuity(partitions)
    partitions.to_csv(args.output_dir / "source_year_audit.csv", index=False)
    write_json(args.output_dir / "source_year_continuity_audit.json", continuity)
    if not continuity["overall_pass"]:
        raise RuntimeError(f"Source-year continuity audit failed: {continuity}")
    baseline = pd.read_csv(args.baseline_target_works, low_memory=False)
    combined = materialize_uncapped_targets(roster, partitions, args)
    selected_names = set(roster["source_display_name"].astype(str))
    counts = source_count_audit(baseline, combined, selected_names)
    counts.to_csv(args.output_dir / "source_count_audit.csv", index=False)
    if counts["new_exactly_legacy_cap"].any():
        raise RuntimeError("A replacement source still equals the legacy 25,000 cap")
    target_csv = args.output_dir / "nature_target_works.csv"
    manifest: dict[str, Any] = {
        "artifact_kind": "nature_portfolio_uncapped_source_year_targets",
        "created_at": utc_now(),
        "source_names": sorted(selected_names),
        "start_year": int(args.start_year),
        "end_year": int(args.end_year),
        "legacy_cap_removed": LEGACY_CAP,
        "n_baseline_targets": len(baseline),
        "n_uncapped_targets": len(combined),
        "n_added_targets": int(len(combined) - len(baseline)),
        "n_unique_targets": int(combined["id"].nunique()),
        "annual_continuity": continuity,
        "source_count_audit": counts.to_dict("records"),
        "outputs": {
            "target_csv": str(target_csv.resolve()),
            "target_csv_sha256": sha256_file(target_csv),
            "target_parquet": str(
                (args.output_dir / "nature_target_works.parquet").resolve()
            ),
            "source_year_audit": str(
                (args.output_dir / "source_year_audit.csv").resolve()
            ),
        },
        "quality_checks": {
            **continuity["checks"],
            "target_primary_key_unique": bool(not combined["id"].duplicated().any()),
            "all_selected_sources_exceed_legacy_cap": bool(
                counts["new_count"].gt(LEGACY_CAP).all()
            ),
        },
    }
    manifest["overall_pass"] = bool(all(manifest["quality_checks"].values()))
    write_json(args.output_dir / "uncapped_target_manifest.json", manifest)
    if not manifest["overall_pass"]:
        raise RuntimeError(f"Uncapped target audit failed: {manifest}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-roster", type=Path, required=True)
    parser.add_argument("--baseline-target-works", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument(
        "--source-name", action="append", default=list(DEFAULT_SOURCE_NAMES)
    )
    parser.add_argument("--start-year", type=int, default=1980)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--work-types", nargs="+", default=list(TARGET_WORK_TYPES))
    parser.add_argument("--per-page", type=int, default=200)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--openalex-api-key", default=getenv("OPENALEX_API_KEY"))
    parser.add_argument("--openalex-api-keys", default=getenv("OPENALEX_API_KEYS"))
    parser.add_argument("--openalex-email", default=getenv("OPENALEX_EMAIL"))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the uncapped source-year acquisition."""
    args = build_parser().parse_args(argv)
    args.checkpoint_dir = args.checkpoint_dir or (
        args.output_dir / "checkpoints" / "source_year"
    )
    if args.start_year > args.end_year:
        raise ValueError("start-year must not exceed end-year")
    manifest = run(args)
    if not args.quiet:
        print(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

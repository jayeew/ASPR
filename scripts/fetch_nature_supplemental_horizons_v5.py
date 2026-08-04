from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aspr.env import getenv  # noqa: E402
from aspr.nature_multihorizon.future_citers import _normalize_citer  # noqa: E402
from scripts.build_openalex_v3_citation_graph import (  # noqa: E402
    OpenAlexClient,
    split_api_keys,
)
from scripts.materialize_nature_future_multihorizon_v5 import (  # noqa: E402
    DELTA_COLUMNS,
    FutureCiterParquetWriter,
    _delta_row,
    _missing_delta_row,
    _normalized_checkpoint_rows,
)
from scripts.nature_portfolio_v5 import (  # noqa: E402
    entropy_from_counts,
    normalize_openalex_id,
    short_openalex_id,
    simpson_from_counts,
    utc_now,
)


DEFAULT_COMPLETE_END_YEAR = 2025
DEFAULT_SNAPSHOT_DIR = Path(
    "/mnt/d/FabCitationData/openalex-snapshot"
)
DEFAULT_SOURCE_DIR = (
    PROJECT_ROOT / "outputs" / "common" / "new" / "data" / "nature_portfolio_v5"
)
DEFAULT_COHORTS = (
    "2018:3,5",
    "2019:3,5",
    "2020:3,5",
    "2021:3",
    "2022:3",
    "2023:1,2",
    "2024:1",
)
DEFAULT_OUTPUT_DIR = DEFAULT_SOURCE_DIR / "future_supplemental_2018_2024"

_SNAPSHOT_TARGET_ALIASES: Dict[str, str] = {}
_SNAPSHOT_TARGET_YEARS: Dict[str, int] = {}
_SNAPSHOT_TARGET_MAX_HORIZONS: Dict[str, int] = {}
_SNAPSHOT_COMPLETE_END_YEAR = DEFAULT_COMPLETE_END_YEAR

SNAPSHOT_EDGE_SCHEMA = pa.schema(
    [
        pa.field("paper_id", pa.string(), nullable=False),
        pa.field("publication_year", pa.int16(), nullable=False),
        pa.field("requested_horizon", pa.int16(), nullable=False),
        pa.field("citer_id", pa.string(), nullable=False),
        pa.field("citer_year", pa.int16(), nullable=False),
        pa.field("citer_primary_field", pa.string()),
        pa.field("citer_primary_subfield", pa.string()),
        pa.field("citer_primary_topic", pa.string()),
        pa.field("referenced_works", pa.list_(pa.string())),
    ]
)


def parse_cohort_specs(
    values: Sequence[str],
    *,
    complete_end_year: int,
) -> Dict[int, Tuple[int, ...]]:
    """Parse YEAR:H1,H2 cohort specifications.

    Args:
        values: Repeated cohort specifications such as ``2023:1,2``.
        complete_end_year: Last calendar year with complete outcome data.

    Returns:
        Sorted mapping from publication year to requested horizons.
    """

    parsed: Dict[int, set[int]] = {}
    for value in values:
        year_text, separator, horizons_text = str(value).partition(":")
        if not separator or not year_text.strip() or not horizons_text.strip():
            raise ValueError(f"Invalid cohort specification: {value!r}")
        year = int(year_text)
        horizons = {
            int(item.strip())
            for item in horizons_text.split(",")
            if item.strip()
        }
        if not horizons or min(horizons) <= 0:
            raise ValueError(f"Cohort horizons must be positive: {value!r}")
        if year + max(horizons) > int(complete_end_year):
            raise ValueError(
                f"Incomplete cohort {value!r}: "
                f"{year}+{max(horizons)}>{complete_end_year}"
            )
        parsed.setdefault(year, set()).update(horizons)
    if not parsed:
        raise ValueError("At least one cohort specification is required")
    return {
        year: tuple(sorted(horizons))
        for year, horizons in sorted(parsed.items())
    }


def read_cohort_targets(
    path: Path,
    cohort_specs: Mapping[int, Sequence[int]],
) -> pd.DataFrame:
    """Read only the target identifiers needed by the supplemental cohorts."""

    header = set(pd.read_csv(path, nrows=0).columns)
    id_column = "id" if "id" in header else "paper_id"
    year_column = "year" if "year" in header else "publication_year"
    required = {id_column, year_column}
    if not required.issubset(header):
        raise ValueError(f"Target works is missing columns: {sorted(required - header)}")
    usecols = [id_column, year_column]
    if "short_id" in header:
        usecols.append("short_id")
    targets = pd.read_csv(path, usecols=usecols, low_memory=False)
    targets = targets.rename(
        columns={id_column: "paper_id", year_column: "publication_year"}
    )
    targets["paper_id"] = targets["paper_id"].map(normalize_openalex_id)
    targets["publication_year"] = pd.to_numeric(
        targets["publication_year"], errors="coerce"
    )
    targets = targets[
        targets["paper_id"].astype(str).str.strip().ne("")
        & targets["publication_year"].isin(cohort_specs)
    ].copy()
    targets["publication_year"] = targets["publication_year"].astype(int)
    if "short_id" not in targets:
        targets["short_id"] = targets["paper_id"].map(short_openalex_id)
    else:
        targets["short_id"] = targets["short_id"].fillna("").astype(str).str.strip()
        empty = targets["short_id"].eq("")
        targets.loc[empty, "short_id"] = targets.loc[empty, "paper_id"].map(
            short_openalex_id
        )
    if targets["paper_id"].duplicated().any():
        examples = targets.loc[
            targets["paper_id"].duplicated(keep=False), "paper_id"
        ].head(5)
        raise ValueError(f"Supplemental target IDs are not unique: {examples.tolist()}")
    observed_years = set(targets["publication_year"].unique())
    missing_years = sorted(set(cohort_specs) - observed_years)
    if missing_years:
        raise ValueError(f"No target works found for cohort years: {missing_years}")
    return targets.sort_values(
        ["publication_year", "paper_id"], kind="stable"
    ).reset_index(drop=True)


def _checkpoint_dir(root: Path, year: int, requested_horizon: int) -> Path:
    return root / f"year_{year}_tau{requested_horizon}"


def _checkpoint_path(root: Path, paper: Mapping[str, Any], horizon: int) -> Path:
    directory = _checkpoint_dir(root, int(paper["publication_year"]), horizon)
    return directory / f"{paper['short_id']}.jsonl"


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            )
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return count


def _batches(rows: Sequence[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    for start in range(0, len(rows), int(size)):
        yield list(rows[start : start + int(size)])


def _snapshot_files(snapshot_dir: Path, minimum_outcome_year: int) -> List[Path]:
    works_root = snapshot_dir / "data" / "works"
    if not works_root.is_dir():
        raise FileNotFoundError(f"OpenAlex snapshot works directory not found: {works_root}")
    minimum_partition = f"updated_date={int(minimum_outcome_year)}-01-01"
    return sorted(
        path
        for path in works_root.glob("updated_date=*/part_*.gz")
        if path.parent.name >= minimum_partition
    )


def _snapshot_file_key(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:24]


def _read_json_object(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _snapshot_state_valid(
    source_path: Path,
    spool_path: Path,
    state_path: Path,
) -> bool:
    if not spool_path.is_file() or not state_path.is_file():
        return False
    try:
        state = _read_json_object(state_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    stat = source_path.stat()
    return (
        state.get("source_path") == str(source_path.resolve())
        and int(state.get("source_size_bytes", -1)) == stat.st_size
        and int(state.get("source_mtime_ns", -1)) == stat.st_mtime_ns
        and int(state.get("bad_json_records", -1)) == 0
    )


def _scan_snapshot_file(
    source_path: Path,
    *,
    spool_path: Path,
    state_path: Path,
    target_aliases: Mapping[str, str],
    target_years: Mapping[str, int],
    target_max_horizons: Mapping[str, int],
    complete_end_year: int,
) -> Dict[str, Any]:
    """Scan one OpenAlex snapshot shard into an atomic matched-work spool."""

    temporary = spool_path.with_name(f".{spool_path.name}.tmp-{os.getpid()}")
    spool_path.parent.mkdir(parents=True, exist_ok=True)
    records_seen = 0
    eligible_works = 0
    matched_works = 0
    matched_edges = 0
    bad_json_records = 0
    with gzip.open(source_path, "rt", encoding="utf-8") as source, temporary.open(
        "w", encoding="utf-8"
    ) as output:
        for line in source:
            records_seen += 1
            try:
                work = json.loads(line)
            except json.JSONDecodeError:
                bad_json_records += 1
                continue
            year_value = work.get("publication_year")
            try:
                citer_year = int(year_value)
            except (TypeError, ValueError):
                continue
            if (
                citer_year > int(complete_end_year)
                or str(work.get("language") or "") != "en"
                or bool(work.get("is_retracted"))
                or bool(work.get("is_paratext"))
            ):
                continue
            references = work.get("referenced_works")
            if not isinstance(references, list) or not references:
                continue
            eligible_works += 1
            matched_ids: set[str] = set()
            for reference in references:
                paper_id = target_aliases.get(str(reference))
                if paper_id is None:
                    continue
                publication_year = target_years[paper_id]
                if (
                    publication_year < citer_year
                    <= publication_year + target_max_horizons[paper_id]
                ):
                    matched_ids.add(paper_id)
            if not matched_ids:
                continue
            selected_work = {
                "id": work.get("id"),
                "publication_year": citer_year,
                "primary_topic": work.get("primary_topic"),
                "referenced_works": references,
            }
            output.write(
                json.dumps(
                    {
                        "paper_ids": sorted(matched_ids),
                        "work": selected_work,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            matched_works += 1
            matched_edges += len(matched_ids)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, spool_path)
    stat = source_path.stat()
    state = {
        "artifact_kind": "nature_supplemental_snapshot_shard",
        "created_at": utc_now(),
        "source_path": str(source_path.resolve()),
        "source_size_bytes": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "records_seen": records_seen,
        "eligible_works": eligible_works,
        "matched_works": matched_works,
        "matched_edges": matched_edges,
        "bad_json_records": bad_json_records,
        "spool_path": str(spool_path.resolve()),
        "spool_size_bytes": spool_path.stat().st_size,
    }
    _atomic_json(state_path, state)
    return state


def _initialize_snapshot_worker(
    target_aliases: Dict[str, str],
    target_years: Dict[str, int],
    target_max_horizons: Dict[str, int],
    complete_end_year: int,
) -> None:
    global _SNAPSHOT_COMPLETE_END_YEAR
    global _SNAPSHOT_TARGET_ALIASES
    global _SNAPSHOT_TARGET_MAX_HORIZONS
    global _SNAPSHOT_TARGET_YEARS

    _SNAPSHOT_TARGET_ALIASES = target_aliases
    _SNAPSHOT_TARGET_YEARS = target_years
    _SNAPSHOT_TARGET_MAX_HORIZONS = target_max_horizons
    _SNAPSHOT_COMPLETE_END_YEAR = int(complete_end_year)


def _scan_snapshot_file_worker(
    source_path: Path,
    spool_path: Path,
    state_path: Path,
) -> Dict[str, Any]:
    return _scan_snapshot_file(
        source_path,
        spool_path=spool_path,
        state_path=state_path,
        target_aliases=_SNAPSHOT_TARGET_ALIASES,
        target_years=_SNAPSHOT_TARGET_YEARS,
        target_max_horizons=_SNAPSHOT_TARGET_MAX_HORIZONS,
        complete_end_year=_SNAPSHOT_COMPLETE_END_YEAR,
    )


def _scan_snapshot(
    targets: pd.DataFrame,
    cohort_specs: Mapping[int, Sequence[int]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    minimum_outcome_year = min(cohort_specs) + 1
    source_files = _snapshot_files(args.snapshot_dir, minimum_outcome_year)
    if args.snapshot_max_files is not None:
        source_files = source_files[: max(0, int(args.snapshot_max_files))]
    state_root = args.checkpoint_dir / "_snapshot_state"
    spool_root = state_root / "spool"
    shard_state_root = state_root / "shards"
    target_aliases: Dict[str, str] = {}
    target_years: Dict[str, int] = {}
    target_max_horizons: Dict[str, int] = {}
    for paper in targets.to_dict("records"):
        paper_id = str(paper["paper_id"])
        target_aliases[paper_id] = paper_id
        target_aliases[str(paper["short_id"])] = paper_id
        publication_year = int(paper["publication_year"])
        target_years[paper_id] = publication_year
        target_max_horizons[paper_id] = max(cohort_specs[publication_year])
    pending: List[Tuple[Path, Path, Path]] = []
    completed_states: List[Dict[str, Any]] = []
    for source_path in source_files:
        key = _snapshot_file_key(source_path)
        spool_path = spool_root / f"{key}.jsonl"
        state_path = shard_state_root / f"{key}.json"
        if _snapshot_state_valid(source_path, spool_path, state_path):
            completed_states.append(_read_json_object(state_path))
        else:
            pending.append((source_path, spool_path, state_path))
    if not args.quiet:
        print(
            f"[Supplemental v5 snapshot] files={len(source_files):,}, "
            f"cached={len(completed_states):,}, pending={len(pending):,}, "
            f"compressed_gib={sum(path.stat().st_size for path in source_files) / 2**30:.1f}",
            flush=True,
        )
    with ProcessPoolExecutor(
        max_workers=max(1, int(args.snapshot_workers)),
        initializer=_initialize_snapshot_worker,
        initargs=(
            target_aliases,
            target_years,
            target_max_horizons,
            args.complete_end_year,
        ),
    ) as executor:
        pending_iterator = iter(pending)
        futures: Dict[Any, Tuple[Path, Path, Path]] = {}

        def submit_next() -> bool:
            try:
                source_path, spool_path, state_path = next(pending_iterator)
            except StopIteration:
                return False
            future = executor.submit(
                _scan_snapshot_file_worker,
                source_path,
                spool_path,
                state_path,
            )
            futures[future] = (source_path, spool_path, state_path)
            return True

        for _ in range(max(1, int(args.snapshot_workers)) * 2):
            if not submit_next():
                break
        scanned = 0
        while futures:
            finished, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
            for future in finished:
                futures.pop(future)
                completed_states.append(future.result())
                scanned += 1
                if not args.quiet and (
                    scanned == 1
                    or scanned % int(args.snapshot_progress_every) == 0
                    or scanned == len(pending)
                ):
                    print(
                        f"[Supplemental v5 snapshot] scanned={scanned:,}/{len(pending):,}",
                        flush=True,
                    )
                submit_next()
    bad_json_records = sum(
        int(state.get("bad_json_records", 0)) for state in completed_states
    )
    if bad_json_records:
        raise RuntimeError(
            f"Snapshot scan found {bad_json_records} malformed JSON records"
        )
    return {
        "artifact_kind": "nature_supplemental_snapshot_scan",
        "snapshot_dir": str(args.snapshot_dir.resolve()),
        "minimum_outcome_year": minimum_outcome_year,
        "n_source_files": len(source_files),
        "n_cached_files": len(source_files) - len(pending),
        "n_scanned_files": len(pending),
        "records_seen": sum(
            int(state.get("records_seen", 0)) for state in completed_states
        ),
        "eligible_works": sum(
            int(state.get("eligible_works", 0)) for state in completed_states
        ),
        "matched_works": sum(
            int(state.get("matched_works", 0)) for state in completed_states
        ),
        "matched_edges": sum(
            int(state.get("matched_edges", 0)) for state in completed_states
        ),
        "bad_json_records": bad_json_records,
        "complete": args.snapshot_max_files is None,
        "state_root": str(state_root.resolve()),
    }


def _consolidate_snapshot_checkpoints(
    targets: pd.DataFrame,
    cohort_specs: Mapping[int, Sequence[int]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    state_root = args.checkpoint_dir / "_snapshot_state"
    spool_root = state_root / "spool"
    shard_state_root = state_root / "shards"
    consolidated_root = state_root / "consolidated"
    consolidated_root.mkdir(parents=True, exist_ok=True)
    paper_lookup = {
        str(paper["paper_id"]): paper for paper in targets.to_dict("records")
    }
    open_handles: OrderedDict[Path, Any] = OrderedDict()
    consolidated_spools = 0
    appended_edges = 0
    pending_markers: List[Tuple[Path, Dict[str, Any]]] = []

    def append_handle(path: Path) -> Any:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = open_handles.pop(path, None)
        if handle is None:
            handle = path.open("a", encoding="utf-8")
        open_handles[path] = handle
        while len(open_handles) > int(args.snapshot_open_files):
            _, old_handle = open_handles.popitem(last=False)
            old_handle.close()
        return handle

    def commit_pending_markers() -> None:
        if not pending_markers:
            return
        for handle in open_handles.values():
            handle.flush()
            os.fsync(handle.fileno())
        for marker, payload in pending_markers:
            _atomic_json(marker, payload)
        pending_markers.clear()

    state_paths = sorted(shard_state_root.glob("*.json"))
    try:
        for index, state_path in enumerate(state_paths, start=1):
            key = state_path.stem
            marker = consolidated_root / f"{key}.json"
            if marker.is_file():
                continue
            state = _read_json_object(state_path)
            spool_path = spool_root / f"{key}.jsonl"
            touched: set[Path] = set()
            local_edges = 0
            with spool_path.open("r", encoding="utf-8") as spool:
                for line in spool:
                    row = json.loads(line)
                    work = row["work"]
                    for paper_id in row["paper_ids"]:
                        paper = paper_lookup.get(str(paper_id))
                        if paper is None:
                            continue
                        year = int(paper["publication_year"])
                        requested_horizon = max(cohort_specs[year])
                        checkpoint = _checkpoint_path(
                            args.checkpoint_dir, paper, requested_horizon
                        )
                        handle = append_handle(checkpoint)
                        handle.write(
                            json.dumps(
                                {"paper_id": paper_id, "work": work},
                                ensure_ascii=False,
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        touched.add(checkpoint)
                        local_edges += 1
            pending_markers.append(
                (
                    marker,
                    {
                    "artifact_kind": "nature_supplemental_snapshot_consolidation",
                    "created_at": utc_now(),
                    "source_state": str(state_path.resolve()),
                    "source_spool": str(spool_path.resolve()),
                    "appended_edges": local_edges,
                    },
                )
            )
            consolidated_spools += 1
            appended_edges += local_edges
            if (
                len(pending_markers)
                >= int(args.snapshot_consolidation_batch_size)
            ):
                commit_pending_markers()
            if not args.quiet and (
                consolidated_spools == 1
                or consolidated_spools % int(args.snapshot_progress_every) == 0
                or index == len(state_paths)
            ):
                print(
                    "[Supplemental v5 snapshot] "
                    f"consolidated={consolidated_spools:,}, "
                    f"appended_edges={appended_edges:,}",
                    flush=True,
                )
        commit_pending_markers()
    finally:
        for handle in open_handles.values():
            handle.close()
    return {
        "n_shard_states": len(state_paths),
        "n_newly_consolidated_spools": consolidated_spools,
        "n_newly_appended_edges": appended_edges,
    }


def _materialize_snapshot_edge_cache(
    targets: pd.DataFrame,
    cohort_specs: Mapping[int, Sequence[int]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Sequentially compact snapshot spools into one canonical Parquet table."""

    state_root = args.checkpoint_dir / "_snapshot_state"
    spool_paths = sorted((state_root / "spool").glob("*.jsonl"))
    edge_path = state_root / "snapshot_citer_edges.parquet"
    coverage_path = state_root / "snapshot_covered_paper_ids.parquet"
    manifest_path = state_root / "snapshot_edge_cache_manifest.json"
    if edge_path.is_file() and coverage_path.is_file() and manifest_path.is_file():
        manifest = _read_json_object(manifest_path)
        if (
            int(manifest.get("n_source_spools", -1)) == len(spool_paths)
            and int(manifest.get("n_source_spools", -1)) == 2049
        ):
            return manifest
    paper_lookup = {
        str(paper["paper_id"]): paper for paper in targets.to_dict("records")
    }
    temporary_edge = edge_path.with_name(f".{edge_path.name}.tmp-{os.getpid()}")
    writer = pq.ParquetWriter(
        temporary_edge,
        SNAPSHOT_EDGE_SCHEMA,
        compression="zstd",
    )
    buffer: List[Dict[str, Any]] = []
    covered_ids: set[str] = set()
    row_count = 0

    def flush() -> None:
        nonlocal buffer
        nonlocal row_count
        if not buffer:
            return
        writer.write_table(pa.Table.from_pylist(buffer, schema=SNAPSHOT_EDGE_SCHEMA))
        row_count += len(buffer)
        buffer = []

    try:
        for index, spool_path in enumerate(spool_paths, start=1):
            with spool_path.open("r", encoding="utf-8") as spool:
                for line in spool:
                    row = json.loads(line)
                    work = row["work"]
                    for paper_id in row["paper_ids"]:
                        paper = paper_lookup.get(str(paper_id))
                        if paper is None:
                            continue
                        publication_year = int(paper["publication_year"])
                        requested_horizon = max(cohort_specs[publication_year])
                        normalized = _normalize_citer(
                            str(paper_id),
                            requested_horizon,
                            publication_year,
                            work,
                        )
                        if normalized is None:
                            continue
                        buffer.append(
                            {
                                "paper_id": str(paper_id),
                                "publication_year": publication_year,
                                "requested_horizon": requested_horizon,
                                "citer_id": normalize_openalex_id(
                                    normalized["citer_id"]
                                ),
                                "citer_year": int(normalized["citer_year"]),
                                "citer_primary_field": normalized.get(
                                    "citer_primary_field"
                                ),
                                "citer_primary_subfield": normalized.get(
                                    "citer_primary_subfield"
                                ),
                                "citer_primary_topic": normalized.get(
                                    "citer_primary_topic"
                                ),
                                "referenced_works": [
                                    normalize_openalex_id(item)
                                    for item in (
                                        normalized.get("referenced_works") or []
                                    )
                                    if normalize_openalex_id(item)
                                ],
                            }
                        )
                        covered_ids.add(str(paper_id))
                        if len(buffer) >= int(args.parquet_batch_size):
                            flush()
            if not args.quiet and (
                index == 1
                or index % int(args.snapshot_progress_every) == 0
                or index == len(spool_paths)
            ):
                print(
                    "[Supplemental v5 snapshot] edge_cache_spools="
                    f"{index:,}/{len(spool_paths):,}, rows={row_count + len(buffer):,}",
                    flush=True,
                )
        flush()
    finally:
        writer.close()
    os.replace(temporary_edge, edge_path)
    pd.DataFrame(
        {"paper_id": sorted(covered_ids)}
    ).to_parquet(coverage_path, index=False, compression="zstd")
    manifest = {
        "artifact_kind": "nature_supplemental_snapshot_edge_cache",
        "created_at": utc_now(),
        "n_source_spools": len(spool_paths),
        "n_edge_rows": row_count,
        "n_covered_papers": len(covered_ids),
        "edge_path": str(edge_path.resolve()),
        "coverage_path": str(coverage_path.resolve()),
        "primary_key": ["paper_id", "citer_id"],
        "source_contract": "local_openalex_snapshot_first",
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def populate_snapshot_checkpoints(
    targets: pd.DataFrame,
    cohort_specs: Mapping[int, Sequence[int]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Populate supplemental checkpoints from the local snapshot first."""

    scan = _scan_snapshot(targets, cohort_specs, args)
    edge_cache = _materialize_snapshot_edge_cache(
        targets, cohort_specs, args
    )
    result = {**scan, "edge_cache": edge_cache}
    manifest_path = args.checkpoint_dir / "_snapshot_state" / "manifest.json"
    _atomic_json(manifest_path, result)
    return result


def _fetch_batch(
    papers: Sequence[Dict[str, Any]],
    *,
    requested_horizon: int,
    complete_end_year: int,
    checkpoint_root: Path,
    openalex: OpenAlexClient,
    max_citers_per_work: int,
    max_records_per_batch: int,
    per_page: int,
) -> Dict[str, Any]:
    year = int(papers[0]["publication_year"])
    target_by_id = {
        normalize_openalex_id(paper["paper_id"]): dict(paper) for paper in papers
    }
    filters = [
        "cites:" + "|".join(short_openalex_id(item) for item in target_by_id),
        f"from_publication_date:{year + 1}-01-01",
        f"to_publication_date:{min(complete_end_year, year + requested_horizon)}-12-31",
        "language:en",
        "is_retracted:false",
        "is_paratext:false",
    ]
    works = openalex.list_works(
        max_records=int(max_records_per_batch),
        filters=filters,
        sort="publication_date:asc",
        per_page=int(per_page),
        progress=False,
    )
    if len(works) >= int(max_records_per_batch):
        raise RuntimeError(
            "Batch result reached max_records_per_batch; rerun with a smaller "
            "batch or larger record cap"
        )
    matched: Dict[str, List[Dict[str, Any]]] = {
        paper_id: [] for paper_id in target_by_id
    }
    target_ids = set(target_by_id)
    for work in works:
        references = {
            normalize_openalex_id(item)
            for item in (work.get("referenced_works") or [])
        }
        for paper_id in target_ids.intersection(references):
            if len(matched[paper_id]) < int(max_citers_per_work):
                matched[paper_id].append(work)
    for paper_id, paper in target_by_id.items():
        checkpoint = _checkpoint_path(
            checkpoint_root, paper, requested_horizon
        )
        _atomic_jsonl(
            checkpoint,
            ({"paper_id": paper_id, "work": work} for work in matched[paper_id]),
        )
    return {
        "publication_year": year,
        "requested_horizon": int(requested_horizon),
        "n_targets": len(papers),
        "n_query_works": len(works),
        "n_checkpoint_rows": sum(len(items) for items in matched.values()),
        "n_zero_targets": sum(not items for items in matched.values()),
    }


def fetch_missing_checkpoints(
    targets: pd.DataFrame,
    cohort_specs: Mapping[int, Sequence[int]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Fetch missing outcome checkpoints with resumable batched OpenAlex queries."""

    coverage_path = (
        args.checkpoint_dir
        / "_snapshot_state"
        / "snapshot_covered_paper_ids.parquet"
    )
    snapshot_covered_ids = (
        set(pd.read_parquet(coverage_path, columns=["paper_id"])["paper_id"].astype(str))
        if coverage_path.is_file()
        else set()
    )
    openalex = OpenAlexClient(
        api_key=args.openalex_api_key,
        api_keys=split_api_keys(args.openalex_api_keys),
        email=args.openalex_email,
        sleep_seconds=args.sleep_seconds,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
    )
    diagnostics: Counter[str] = Counter()
    failed_batches: List[Dict[str, Any]] = []
    for round_index in range(1, int(args.retry_rounds) + 1):
        pending_batches: List[Tuple[int, List[Dict[str, Any]]]] = []
        for year, horizons in cohort_specs.items():
            requested_horizon = max(int(value) for value in horizons)
            cohort = targets[targets["publication_year"].eq(year)]
            missing = [
                paper
                for paper in cohort.to_dict("records")
                if str(paper["paper_id"]) not in snapshot_covered_ids
                and not _checkpoint_path(
                    args.checkpoint_dir, paper, requested_horizon
                ).is_file()
            ]
            pending_batches.extend(
                (requested_horizon, batch)
                for batch in _batches(missing, args.batch_size)
            )
        if not pending_batches:
            break
        if not args.quiet:
            print(
                f"[Supplemental v5] retry_round={round_index}, "
                f"pending_batches={len(pending_batches):,}, "
                f"pending_targets={sum(len(batch) for _, batch in pending_batches):,}",
                flush=True,
            )
        failed_batches = []
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
            batch_iterator = iter(pending_batches)
            futures: Dict[Any, Tuple[int, List[Dict[str, Any]]]] = {}

            def submit_next() -> bool:
                try:
                    requested_horizon, batch = next(batch_iterator)
                except StopIteration:
                    return False
                future = executor.submit(
                    _fetch_batch,
                    batch,
                    requested_horizon=requested_horizon,
                    complete_end_year=args.complete_end_year,
                    checkpoint_root=args.checkpoint_dir,
                    openalex=openalex,
                    max_citers_per_work=args.max_citers_per_work,
                    max_records_per_batch=args.max_records_per_batch,
                    per_page=args.per_page,
                )
                futures[future] = (requested_horizon, batch)
                return True

            for _ in range(max(1, int(args.workers)) * 2):
                if not submit_next():
                    break
            completed = 0
            total_batches = len(pending_batches)
            while futures:
                finished, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
                for future in finished:
                    requested_horizon, batch = futures.pop(future)
                    completed += 1
                    try:
                        result = future.result()
                    except Exception as exc:
                        failed_batches.append(
                            {
                                "publication_year": int(
                                    batch[0]["publication_year"]
                                ),
                                "requested_horizon": int(requested_horizon),
                                "n_targets": len(batch),
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                        )
                        diagnostics["failed_batches"] += 1
                    else:
                        diagnostics["successful_batches"] += 1
                        diagnostics["query_works"] += int(
                            result["n_query_works"]
                        )
                        diagnostics["checkpoint_rows"] += int(
                            result["n_checkpoint_rows"]
                        )
                    if not args.quiet and (
                        completed == 1
                        or completed % int(args.progress_every) == 0
                        or completed == total_batches
                    ):
                        print(
                            f"[Supplemental v5] round={round_index}, "
                            f"batches={completed:,}/{total_batches:,}, "
                            f"failed={len(failed_batches):,}",
                            flush=True,
                        )
                    submit_next()
    return {
        "diagnostics": dict(diagnostics),
        "failed_batches": failed_batches,
    }


def materialize_supplement(
    targets: pd.DataFrame,
    cohort_specs: Mapping[int, Sequence[int]],
    args: argparse.Namespace,
    fetch_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    """Materialize canonical non-rectangular supplemental horizon tables."""

    final_output = args.output_dir
    temporary_output = final_output.with_name(f".{final_output.name}.tmp-{os.getpid()}")
    if temporary_output.exists():
        shutil.rmtree(temporary_output)
    if final_output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {final_output}; pass --overwrite to rebuild"
        )
    temporary_output.mkdir(parents=True, exist_ok=False)
    statuses: List[Dict[str, Any]] = []
    requests: List[Dict[str, Any]] = []
    deltas: List[Dict[str, Any]] = []
    missing: List[str] = []
    writer = FutureCiterParquetWriter(
        temporary_output / "future_citers.parquet", args.parquet_batch_size
    )
    try:
        for paper in targets.to_dict("records"):
            paper_id = str(paper["paper_id"])
            publication_year = int(paper["publication_year"])
            horizons = tuple(int(value) for value in cohort_specs[publication_year])
            requested_horizon = max(horizons)
            checkpoint = _checkpoint_path(
                args.checkpoint_dir, paper, requested_horizon
            )
            requests.append(
                {
                    "paper_id": paper_id,
                    "publication_year": publication_year,
                    "requested_horizon": requested_horizon,
                    "requested_horizons": list(horizons),
                    "request_batch": "supplemental_2018_2024_batched_openalex",
                }
            )
            if not checkpoint.is_file():
                missing.append(paper_id)
                statuses.append(
                    {
                        "paper_id": paper_id,
                        "requested_horizon": requested_horizon,
                        "fetch_status": "failed",
                        "n_returned": None,
                        "is_zero_success": 0,
                        "cap_hit": 0,
                        "last_citer_year": None,
                        "error_type": "missing_checkpoint",
                        "attempt_count": int(args.retry_rounds),
                        "publication_year": publication_year,
                        "checkpoint_file": None,
                        "request_batch": "supplemental_2018_2024_batched_openalex",
                    }
                )
                deltas.extend(
                    _missing_delta_row(paper_id, publication_year, horizon)
                    for horizon in horizons
                )
                continue
            normalized, raw_count, invalid_count = _normalized_checkpoint_rows(
                checkpoint,
                paper_id,
                publication_year,
                requested_horizon,
            )
            if invalid_count:
                raise ValueError(
                    f"Invalid citer rows in {checkpoint}: {invalid_count}"
                )
            last_year = (
                max(int(row["citer_year"]) for row in normalized)
                if normalized
                else None
            )
            cap_hit = raw_count >= int(args.max_citers_per_work)
            statuses.append(
                {
                    "paper_id": paper_id,
                    "requested_horizon": requested_horizon,
                    "fetch_status": "success",
                    "n_returned": len(normalized),
                    "is_zero_success": int(not normalized),
                    "cap_hit": int(cap_hit),
                    "last_citer_year": last_year,
                    "error_type": "",
                    "attempt_count": 1,
                    "publication_year": publication_year,
                    "checkpoint_file": str(checkpoint.resolve()),
                    "request_batch": "supplemental_2018_2024_batched_openalex",
                }
            )
            for horizon in horizons:
                in_window = [
                    row
                    for row in normalized
                    if publication_year
                    < int(row["citer_year"])
                    <= publication_year + horizon
                ]
                deltas.append(
                    _delta_row(
                        paper_id,
                        publication_year,
                        horizon,
                        in_window,
                        cap_hit,
                        last_year,
                    )
                )
                for row in in_window:
                    writer.append(
                        {
                            "paper_id": paper_id,
                            "horizon": horizon,
                            "citer_id": row["citer_id"],
                            "citer_year": row["citer_year"],
                            "citer_primary_field": row.get(
                                "citer_primary_field"
                            ),
                            "citer_primary_subfield": row.get(
                                "citer_primary_subfield"
                            ),
                            "citer_primary_topic": row.get(
                                "citer_primary_topic"
                            ),
                            "referenced_works": row.get("referenced_works"),
                        }
                    )
        n_citer_rows = writer.close()
        status_frame = pd.DataFrame(statuses).sort_values(
            ["publication_year", "paper_id"], kind="stable"
        )
        request_frame = pd.DataFrame(requests).sort_values(
            ["publication_year", "paper_id"], kind="stable"
        )
        delta_frame = pd.DataFrame(deltas)[DELTA_COLUMNS].sort_values(
            ["publication_year", "paper_id", "horizon"], kind="stable"
        )
        status_frame.to_parquet(
            temporary_output / "future_fetch_status.parquet",
            index=False,
            compression="zstd",
        )
        request_frame.to_parquet(
            temporary_output / "future_request_manifest.parquet",
            index=False,
            compression="zstd",
        )
        delta_frame.to_parquet(
            temporary_output / "future_graph_deltas_multihorizon.parquet",
            index=False,
            compression="zstd",
        )
        delta_frame.to_csv(
            temporary_output / "future_graph_deltas_multihorizon.csv",
            index=False,
        )
        expected_delta_rows = sum(
            int(targets["publication_year"].eq(year).sum()) * len(horizons)
            for year, horizons in cohort_specs.items()
        )
        duplicate_keys = int(
            delta_frame.duplicated(["paper_id", "horizon"]).sum()
        )
        success_rate = float(status_frame["fetch_status"].eq("success").mean())
        quality = {
            "artifact_kind": "nature_portfolio_v5_supplemental_quality",
            "created_at": utc_now(),
            "cohort_horizons": {
                str(year): list(horizons)
                for year, horizons in cohort_specs.items()
            },
            "n_target_papers": int(len(targets)),
            "expected_delta_rows": int(expected_delta_rows),
            "actual_delta_rows": int(len(delta_frame)),
            "delta_key_duplicates": duplicate_keys,
            "missing_checkpoint_count": len(missing),
            "success_rate": success_rate,
            "label_only_no_leakage": True,
            "overall_pass": (
                len(delta_frame) == expected_delta_rows
                and duplicate_keys == 0
                and not missing
            ),
        }
        manifest = {
            "artifact_kind": "nature_portfolio_v5_future_supplemental",
            "schema_version": "1.0.0",
            "created_at": utc_now(),
            "target_works": str(args.target_works.resolve()),
            "checkpoint_dir": str(args.checkpoint_dir.resolve()),
            "output_dir": str(final_output.resolve()),
            "complete_end_year": int(args.complete_end_year),
            "cohort_horizons": quality["cohort_horizons"],
            "n_target_papers": int(len(targets)),
            "n_future_citer_rows": int(n_citer_rows),
            "n_future_delta_rows": int(len(delta_frame)),
            "n_fetch_status_rows": int(len(status_frame)),
            "fetch_summary": dict(fetch_summary),
            "quality_overall_pass": quality["overall_pass"],
            "primary_keys": {
                "future_citers": ["paper_id", "horizon", "citer_id"],
                "future_fetch_status": ["paper_id", "requested_horizon"],
                "future_request_manifest": ["paper_id", "requested_horizon"],
                "future_graph_deltas": ["paper_id", "horizon"],
            },
            "no_leakage_contract": (
                "Future citers and graph deltas are label-only outcomes and "
                "must not be used as publication-day features."
            ),
        }
        for name, payload in (
            ("data_quality_report.json", quality),
            ("future_supplemental_manifest.json", manifest),
        ):
            (temporary_output / name).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
        if missing and not args.allow_missing:
            raise RuntimeError(
                f"{len(missing)} supplemental checkpoints remain missing"
            )
        if final_output.exists():
            shutil.rmtree(final_output)
        os.replace(temporary_output, final_output)
        return manifest
    except Exception:
        try:
            writer.close()
        except Exception:
            pass
        raise


def _new_delta_aggregate() -> Dict[str, Any]:
    return {
        "n": 0,
        "fields": Counter(),
        "subfields": Counter(),
        "topics": Counter(),
        "first_year": None,
        "last_year": None,
    }


def _update_delta_aggregate(
    aggregate: Dict[str, Any],
    citer: Mapping[str, Any],
) -> None:
    aggregate["n"] += 1
    for key, counter_name in (
        ("citer_primary_field", "fields"),
        ("citer_primary_subfield", "subfields"),
        ("citer_primary_topic", "topics"),
    ):
        value = str(citer.get(key) or "")
        if value:
            aggregate[counter_name][value] += 1
    year = int(citer["citer_year"])
    aggregate["first_year"] = (
        year
        if aggregate["first_year"] is None
        else min(int(aggregate["first_year"]), year)
    )
    aggregate["last_year"] = (
        year
        if aggregate["last_year"] is None
        else max(int(aggregate["last_year"]), year)
    )


def _delta_from_aggregate(
    paper_id: str,
    publication_year: int,
    horizon: int,
    aggregate: Mapping[str, Any],
    requested_cap_hit: bool,
    last_citer_year: Optional[int],
) -> Dict[str, Any]:
    n = int(aggregate["n"])
    fields = aggregate["fields"]
    subfields = aggregate["subfields"]
    topics = aggregate["topics"]
    field_values = [float(value) for value in fields.values()]
    topic_values = [float(value) for value in topics.values()]

    def coverage(counter: Counter[str]) -> float:
        valid = int(sum(counter.values()))
        return 1.0 if n == 0 else float(valid / n)

    horizon_cap_hit = bool(
        requested_cap_hit
        and (
            last_citer_year is None
            or last_citer_year <= publication_year + horizon
        )
    )
    return {
        "paper_id": paper_id,
        "year": publication_year,
        "publication_year": publication_year,
        "tau": horizon,
        "horizon": horizon,
        "fetch_status": "success",
        "fetch_valid": 1,
        "cap_hit": int(horizon_cap_hit),
        "requested_horizon_cap_hit": int(requested_cap_hit),
        "n_future_citers": n,
        "future_community_reach": len(topics),
        "future_topic_reach": len(topics),
        "future_field_reach": len(fields),
        "future_subfield_reach": len(subfields),
        "future_field_entropy": (
            entropy_from_counts(field_values) if field_values else 0.0
        ),
        "future_topic_entropy": (
            entropy_from_counts(topic_values) if topic_values else 0.0
        ),
        "future_field_simpson": (
            simpson_from_counts(field_values) if field_values else 0.0
        ),
        "future_topic_simpson": (
            simpson_from_counts(topic_values) if topic_values else 0.0
        ),
        "future_field_valid_n": int(sum(fields.values())),
        "future_subfield_valid_n": int(sum(subfields.values())),
        "future_topic_valid_n": int(sum(topics.values())),
        "future_field_coverage": coverage(fields),
        "future_subfield_coverage": coverage(subfields),
        "future_topic_coverage": coverage(topics),
        "future_first_year": aggregate["first_year"],
        "future_last_year": aggregate["last_year"],
    }


def materialize_supplement_from_sources(
    targets: pd.DataFrame,
    cohort_specs: Mapping[int, Sequence[int]],
    args: argparse.Namespace,
    acquisition_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    """Merge snapshot edge cache and online checkpoints into final tables."""

    final_output = args.output_dir
    temporary_output = final_output.with_name(f".{final_output.name}.tmp-{os.getpid()}")
    if temporary_output.exists():
        shutil.rmtree(temporary_output)
    if final_output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {final_output}; pass --overwrite to rebuild"
        )
    temporary_output.mkdir(parents=True, exist_ok=False)
    edge_path = (
        args.checkpoint_dir / "_snapshot_state" / "snapshot_citer_edges.parquet"
    )
    coverage_path = (
        args.checkpoint_dir
        / "_snapshot_state"
        / "snapshot_covered_paper_ids.parquet"
    )
    covered_ids = (
        set(pd.read_parquet(coverage_path)["paper_id"].astype(str))
        if coverage_path.is_file()
        else set()
    )
    target_records = {
        str(paper["paper_id"]): paper for paper in targets.to_dict("records")
    }
    aggregates: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for paper_id, paper in target_records.items():
        for horizon in cohort_specs[int(paper["publication_year"])]:
            aggregates[(paper_id, int(horizon))] = _new_delta_aggregate()
    returned_counts: Counter[str] = Counter()
    cap_hits: Dict[str, bool] = {}
    last_years: Dict[str, Optional[int]] = {}
    missing: List[str] = []
    duplicate_snapshot_edges = 0
    writer = FutureCiterParquetWriter(
        temporary_output / "future_citers.parquet", args.parquet_batch_size
    )
    seen_snapshot: Dict[str, set[str]] = {}

    def add_citer(
        paper_id: str,
        publication_year: int,
        citer: Mapping[str, Any],
    ) -> None:
        citer_year = int(citer["citer_year"])
        for horizon in cohort_specs[publication_year]:
            horizon = int(horizon)
            if citer_year > publication_year + horizon:
                continue
            writer.append(
                {
                    "paper_id": paper_id,
                    "horizon": horizon,
                    "citer_id": citer["citer_id"],
                    "citer_year": citer_year,
                    "citer_primary_field": citer.get("citer_primary_field"),
                    "citer_primary_subfield": citer.get(
                        "citer_primary_subfield"
                    ),
                    "citer_primary_topic": citer.get("citer_primary_topic"),
                    "referenced_works": citer.get("referenced_works"),
                }
            )
            _update_delta_aggregate(aggregates[(paper_id, horizon)], citer)

    try:
        if edge_path.is_file():
            parquet = pq.ParquetFile(edge_path)
            for batch in parquet.iter_batches(batch_size=args.parquet_batch_size):
                for citer in batch.to_pylist():
                    paper_id = str(citer["paper_id"])
                    citer_id = str(citer["citer_id"])
                    paper_seen = seen_snapshot.setdefault(paper_id, set())
                    if citer_id in paper_seen:
                        duplicate_snapshot_edges += 1
                        continue
                    paper_seen.add(citer_id)
                    publication_year = int(citer["publication_year"])
                    add_citer(paper_id, publication_year, citer)
                    returned_counts[paper_id] += 1
                    year = int(citer["citer_year"])
                    last_years[paper_id] = max(
                        year, int(last_years.get(paper_id) or year)
                    )
        for paper_id, paper in target_records.items():
            if paper_id in covered_ids:
                cap_hits[paper_id] = False
                continue
            publication_year = int(paper["publication_year"])
            horizons = tuple(int(value) for value in cohort_specs[publication_year])
            requested_horizon = max(horizons)
            checkpoint = _checkpoint_path(
                args.checkpoint_dir, paper, requested_horizon
            )
            if not checkpoint.is_file():
                missing.append(paper_id)
                continue
            normalized, raw_count, invalid_count = _normalized_checkpoint_rows(
                checkpoint,
                paper_id,
                publication_year,
                requested_horizon,
            )
            if invalid_count:
                raise ValueError(
                    f"Invalid citer rows in {checkpoint}: {invalid_count}"
                )
            cap_hits[paper_id] = raw_count >= int(args.max_citers_per_work)
            returned_counts[paper_id] = len(normalized)
            last_years[paper_id] = (
                max(int(row["citer_year"]) for row in normalized)
                if normalized
                else None
            )
            for citer in normalized:
                add_citer(paper_id, publication_year, citer)
        n_citer_rows = writer.close()
        statuses: List[Dict[str, Any]] = []
        requests: List[Dict[str, Any]] = []
        deltas: List[Dict[str, Any]] = []
        for paper_id, paper in target_records.items():
            publication_year = int(paper["publication_year"])
            horizons = tuple(int(value) for value in cohort_specs[publication_year])
            requested_horizon = max(horizons)
            failed = paper_id in missing
            source = "snapshot" if paper_id in covered_ids else "openalex_api"
            requests.append(
                {
                    "paper_id": paper_id,
                    "publication_year": publication_year,
                    "requested_horizon": requested_horizon,
                    "request_batch": "supplemental_2018_2024_snapshot_then_online",
                }
            )
            statuses.append(
                {
                    "paper_id": paper_id,
                    "requested_horizon": requested_horizon,
                    "fetch_status": "failed" if failed else "success",
                    "n_returned": None if failed else int(returned_counts[paper_id]),
                    "is_zero_success": int(
                        not failed and int(returned_counts[paper_id]) == 0
                    ),
                    "cap_hit": int(cap_hits.get(paper_id, False)),
                    "last_citer_year": last_years.get(paper_id),
                    "error_type": "missing_checkpoint" if failed else "",
                    "attempt_count": int(args.retry_rounds) if failed else 1,
                    "publication_year": publication_year,
                    "checkpoint_file": (
                        str(edge_path.resolve())
                        if source == "snapshot"
                        else str(
                            _checkpoint_path(
                                args.checkpoint_dir,
                                paper,
                                requested_horizon,
                            ).resolve()
                        )
                    ),
                    "request_batch": "supplemental_2018_2024_snapshot_then_online",
                }
            )
            for horizon in horizons:
                if failed:
                    deltas.append(
                        _missing_delta_row(paper_id, publication_year, horizon)
                    )
                else:
                    deltas.append(
                        _delta_from_aggregate(
                            paper_id,
                            publication_year,
                            horizon,
                            aggregates[(paper_id, horizon)],
                            cap_hits.get(paper_id, False),
                            last_years.get(paper_id),
                        )
                    )
        status_frame = pd.DataFrame(statuses).sort_values(
            ["publication_year", "paper_id"], kind="stable"
        )
        request_frame = pd.DataFrame(requests).sort_values(
            ["publication_year", "paper_id"], kind="stable"
        )
        delta_frame = pd.DataFrame(deltas)[DELTA_COLUMNS].sort_values(
            ["publication_year", "paper_id", "horizon"], kind="stable"
        )
        compatibility_float_columns = [
            "n_future_citers",
            "future_community_reach",
            "future_topic_reach",
            "future_field_reach",
            "future_subfield_reach",
            "future_field_valid_n",
            "future_subfield_valid_n",
            "future_topic_valid_n",
        ]
        delta_frame[compatibility_float_columns] = delta_frame[
            compatibility_float_columns
        ].astype(float)
        status_frame["n_returned"] = status_frame["n_returned"].astype(float)
        for frame, name in (
            (status_frame, "future_fetch_status.parquet"),
            (request_frame, "future_request_manifest.parquet"),
            (delta_frame, "future_graph_deltas_multihorizon.parquet"),
        ):
            frame.to_parquet(
                temporary_output / name, index=False, compression="zstd"
            )
        delta_frame.to_csv(
            temporary_output / "future_graph_deltas_multihorizon.csv",
            index=False,
        )
        expected_delta_rows = sum(
            int(targets["publication_year"].eq(year).sum()) * len(horizons)
            for year, horizons in cohort_specs.items()
        )
        delta_duplicates = int(
            delta_frame.duplicated(["paper_id", "horizon"]).sum()
        )
        quality = {
            "artifact_kind": "nature_portfolio_v5_supplemental_quality",
            "created_at": utc_now(),
            "cohort_horizons": {
                str(year): list(horizons)
                for year, horizons in cohort_specs.items()
            },
            "n_target_papers": len(targets),
            "expected_delta_rows": expected_delta_rows,
            "actual_delta_rows": len(delta_frame),
            "delta_key_duplicates": delta_duplicates,
            "snapshot_edge_duplicates_removed": duplicate_snapshot_edges,
            "missing_checkpoint_count": len(missing),
            "success_rate": float(status_frame["fetch_status"].eq("success").mean()),
            "label_only_no_leakage": True,
            "overall_pass": (
                len(delta_frame) == expected_delta_rows
                and delta_duplicates == 0
                and not missing
            ),
        }
        manifest = {
            "artifact_kind": "nature_portfolio_v5_future_supplemental",
            "schema_version": "1.0.0",
            "created_at": utc_now(),
            "target_works": str(args.target_works.resolve()),
            "checkpoint_dir": str(args.checkpoint_dir.resolve()),
            "output_dir": str(final_output.resolve()),
            "complete_end_year": int(args.complete_end_year),
            "cohort_horizons": quality["cohort_horizons"],
            "n_target_papers": len(targets),
            "n_future_citer_rows": n_citer_rows,
            "n_future_delta_rows": len(delta_frame),
            "n_fetch_status_rows": len(status_frame),
            "acquisition_summary": dict(acquisition_summary),
            "quality_overall_pass": quality["overall_pass"],
            "primary_keys": {
                "future_citers": ["paper_id", "horizon", "citer_id"],
                "future_fetch_status": ["paper_id", "requested_horizon"],
                "future_request_manifest": ["paper_id", "requested_horizon"],
                "future_graph_deltas": ["paper_id", "horizon"],
            },
            "no_leakage_contract": (
                "Future citers and graph deltas are label-only outcomes and "
                "must not be used as publication-day features."
            ),
        }
        for name, payload in (
            ("data_quality_report.json", quality),
            ("future_supplemental_manifest.json", manifest),
        ):
            (temporary_output / name).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
        if missing and not args.allow_missing:
            raise RuntimeError(
                f"{len(missing)} supplemental checkpoints remain missing"
            )
        if final_output.exists():
            shutil.rmtree(final_output)
        os.replace(temporary_output, final_output)
        return manifest
    except Exception:
        try:
            writer.close()
        except Exception:
            pass
        raise


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch and materialize complete short-horizon Nature Portfolio v5 "
            "supplements using batched, resumable OpenAlex requests."
        )
    )
    parser.add_argument(
        "--target-works",
        type=Path,
        default=DEFAULT_SOURCE_DIR / "nature_target_works.csv",
    )
    parser.add_argument(
        "--cohort",
        action="append",
        default=None,
        help="YEAR:H1,H2; repeat for multiple publication years",
    )
    parser.add_argument(
        "--complete-end-year", type=int, default=DEFAULT_COMPLETE_END_YEAR
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument(
        "--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR
    )
    parser.add_argument("--snapshot-workers", type=int, default=8)
    parser.add_argument("--snapshot-progress-every", type=int, default=10)
    parser.add_argument("--snapshot-open-files", type=int, default=512)
    parser.add_argument(
        "--snapshot-consolidation-batch-size", type=int, default=100
    )
    parser.add_argument("--snapshot-max-files", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-citers-per-work", type=int, default=1000)
    parser.add_argument("--max-records-per-batch", type=int, default=25_000)
    parser.add_argument("--per-page", type=int, default=200)
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-rounds", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--parquet-batch-size", type=int, default=50_000)
    parser.add_argument("--openalex-api-key", default=getenv("OPENALEX_API_KEY"))
    parser.add_argument(
        "--openalex-api-keys", default=getenv("OPENALEX_API_KEYS")
    )
    parser.add_argument("--openalex-email", default=getenv("OPENALEX_EMAIL"))
    parser.add_argument("--fetch-only", action="store_true")
    parser.add_argument("--snapshot-only", action="store_true")
    parser.add_argument("--skip-snapshot", action="store_true")
    parser.add_argument("--materialize-only", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    modes = sum(
        bool(value)
        for value in (
            args.fetch_only,
            args.snapshot_only,
            args.materialize_only,
        )
    )
    if modes > 1:
        parser.error(
            "--fetch-only, --snapshot-only and --materialize-only are "
            "mutually exclusive"
        )
    if args.snapshot_max_files is not None and not args.snapshot_only:
        parser.error("--snapshot-max-files requires --snapshot-only")
    if (
        args.batch_size <= 0
        or args.workers <= 0
        or args.snapshot_workers <= 0
        or args.snapshot_open_files <= 0
        or args.snapshot_consolidation_batch_size <= 0
    ):
        parser.error("Batch, worker and open-file limits must be positive")
    args.checkpoint_dir = args.checkpoint_dir or (
        args.output_dir.parent / f"{args.output_dir.name}_checkpoints"
    )
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cohort_specs = parse_cohort_specs(
        args.cohort or DEFAULT_COHORTS,
        complete_end_year=args.complete_end_year,
    )
    targets = read_cohort_targets(args.target_works, cohort_specs)
    if not args.quiet:
        counts = targets["publication_year"].value_counts().sort_index().to_dict()
        print(
            f"[Supplemental v5] cohorts={dict(cohort_specs)}, targets={counts}",
            flush=True,
        )
    acquisition_summary: Dict[str, Any] = {
        "snapshot": {},
        "diagnostics": {},
        "failed_batches": [],
        "mode": "materialize_only",
    }
    if not args.materialize_only and not args.skip_snapshot:
        acquisition_summary["snapshot"] = populate_snapshot_checkpoints(
            targets, cohort_specs, args
        )
        acquisition_summary["mode"] = "snapshot"
    if args.snapshot_only:
        print(
            json.dumps(
                acquisition_summary, ensure_ascii=False, indent=2, sort_keys=True
            )
        )
        return 0
    if not args.materialize_only:
        online_summary = fetch_missing_checkpoints(
            targets, cohort_specs, args
        )
        acquisition_summary.update(online_summary)
        acquisition_summary["mode"] = (
            "online_only" if args.skip_snapshot else "snapshot_then_online"
        )
    if args.fetch_only:
        print(
            json.dumps(
                acquisition_summary, ensure_ascii=False, indent=2, sort_keys=True
            )
        )
        return 0 if not acquisition_summary["failed_batches"] else 2
    manifest = materialize_supplement_from_sources(
        targets, cohort_specs, args, acquisition_summary
    )
    if not args.quiet:
        print(
            "[Supplemental v5] complete: "
            f"papers={manifest['n_target_papers']:,}, "
            f"citer_rows={manifest['n_future_citer_rows']:,}, "
            f"delta_rows={manifest['n_future_delta_rows']:,}, "
            f"output={args.output_dir}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

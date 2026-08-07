"""Local-only OpenAlex target metadata extraction for v6.1 controls."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import pandas as pd

from aspr.corpus import normalize_openalex_id

OPENALEX_CONTROL_VERSION = "aspr-openalex-controls-v6.1-1"
OPENALEX_EXPANDED_API_VERSION = "aspr-openalex-controls-expanded-api-1"
_WORK_ID_PREFIX = b'{"id":"'
_TARGET_IDS: Set[str] = set()


def _worker_initialize(target_ids: Sequence[str]) -> None:
    global _TARGET_IDS
    _TARGET_IDS = set(target_ids)


def _record_work_id(line: bytes) -> str:
    if not line.startswith(_WORK_ID_PREFIX):
        return ""
    end = line.find(b'"', len(_WORK_ID_PREFIX))
    if end < 0:
        return ""
    return normalize_openalex_id(
        line[len(_WORK_ID_PREFIX) : end].decode("ascii", errors="ignore")
    )


def _target_metadata(record: Mapping[str, Any], source_file: Path) -> Dict[str, Any]:
    authorships = record.get("authorships") or []
    author_ids = {
        normalize_openalex_id((item.get("author") or {}).get("id"))
        for item in authorships
        if isinstance(item, dict)
    }
    author_ids.discard("")
    institution_ids: Set[str] = set()
    countries: Set[str] = set()
    for authorship in authorships:
        if not isinstance(authorship, dict):
            continue
        countries.update(
            str(value).strip()
            for value in authorship.get("countries") or []
            if str(value).strip()
        )
        for institution in authorship.get("institutions") or []:
            if isinstance(institution, dict):
                institution_id = normalize_openalex_id(institution.get("id"))
                if institution_id:
                    institution_ids.add(institution_id)
                country = str(institution.get("country_code") or "").strip()
                if country:
                    countries.add(country)
        for affiliation in authorship.get("affiliations") or []:
            if not isinstance(affiliation, dict):
                continue
            institution_ids.update(
                normalized
                for value in affiliation.get("institution_ids") or []
                if (normalized := normalize_openalex_id(value))
            )
    authors_count = record.get("authors_count")
    institutions_count = record.get("institutions_distinct_count")
    countries_count = record.get("countries_distinct_count")
    canonical_record = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "paper_id": normalize_openalex_id(record.get("id")),
        "openalex_updated_date": str(record.get("updated_date") or ""),
        "openalex_author_count": (
            int(authors_count)
            if isinstance(authors_count, (int, float)) and authors_count >= 0
            else len(author_ids)
        ),
        "openalex_institution_count": (
            int(institutions_count)
            if isinstance(institutions_count, (int, float)) and institutions_count >= 0
            else len(institution_ids)
        ),
        "openalex_country_count": (
            int(countries_count)
            if isinstance(countries_count, (int, float)) and countries_count >= 0
            else len(countries)
        ),
        "openalex_author_ids": sorted(author_ids),
        "metadata_source_file": str(source_file),
        "raw_record_sha256": (f"sha256:{hashlib.sha256(canonical_record).hexdigest()}"),
    }


def _scan_one_file(path_string: str) -> Tuple[str, int, List[Dict[str, Any]]]:
    path = Path(path_string)
    rows: List[Dict[str, Any]] = []
    record_count = 0
    with gzip.open(path, "rb") as stream:
        for line in stream:
            record_count += 1
            work_id = _record_work_id(line)
            if work_id and work_id in _TARGET_IDS:
                record = json.loads(line)
                rows.append(_target_metadata(record, path))
    return str(path), record_count, rows


def snapshot_work_files(snapshot_root: Path) -> Tuple[Path, ...]:
    """Return work parts newest-first for deterministic resumable scanning."""
    root = Path(snapshot_root) / "data" / "works"
    return tuple(
        sorted(
            root.glob("updated_date=*/part_*.gz"),
            key=lambda path: (
                path.parent.name,
                path.name,
            ),
            reverse=True,
        )
    )


def extract_target_metadata(
    target_ids: Iterable[str],
    snapshot_root: Path,
    output_dir: Path,
    *,
    workers: int = 4,
    max_files: int | None = None,
    seed_metadata_path: Path | None = None,
    bulk_partition_first: bool = False,
    stop_when_complete: bool = True,
    resume: bool = True,
) -> Mapping[str, Any]:
    """Scan only the frozen snapshot and materialize target team-size controls.

    Each completed gzip part receives a small JSON checkpoint. The scan can be
    interrupted and resumed without rereading completed parts.
    """
    ids = sorted(
        {
            normalized
            for value in target_ids
            if (normalized := normalize_openalex_id(value))
        }
    )
    if not ids:
        raise ValueError("target_ids is empty")
    root = Path(output_dir)
    checkpoint_dir = root / "openalex_scan_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    files = list(snapshot_work_files(snapshot_root))
    if bulk_partition_first:
        partition_sizes: Dict[Path, int] = {}
        for path in files:
            partition_sizes[path.parent] = partition_sizes.get(path.parent, 0) + 1
        files.sort(
            key=lambda path: (
                partition_sizes[path.parent],
                path.parent.name,
                path.name,
            ),
            reverse=True,
        )
    if max_files is not None:
        files = files[: int(max_files)]
    pending = []
    collected: List[Dict[str, Any]] = []
    seeded: List[Dict[str, Any]] = []
    seed_candidate_count = 0
    rejected_seed_ids: Set[str] = set()
    if seed_metadata_path and Path(seed_metadata_path).is_file():
        seed = pd.read_parquet(seed_metadata_path)
        seed = seed[seed["paper_id"].astype(str).isin(ids)].copy()
        seed_candidate_count = len(seed)
        source_files = seed.get(
            "metadata_source_file",
            pd.Series("", index=seed.index, dtype="string"),
        ).astype("string")
        incomplete_projection = source_files.str.contains(
            "uncapped_source_year_v2/checkpoints/source_year",
            regex=False,
            na=False,
        ) | source_files.str.contains(
            "/openalex_outputs/checkpoints/target_works/",
            regex=False,
            na=False,
        )
        rejected_seed_ids = set(seed.loc[incomplete_projection, "paper_id"].astype(str))
        seed = seed.loc[~incomplete_projection].copy()
        seeded = seed.to_dict("records")
    seeded_ids = {str(row["paper_id"]) for row in seeded}
    scan_ids = sorted(set(ids) - seeded_ids)
    scan_id_set = set(scan_ids)
    scanned_records = 0
    for path in files:
        token = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:20]
        checkpoint = checkpoint_dir / f"{token}.json"
        if resume and checkpoint.is_file():
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            if payload.get("source_file") == str(path):
                collected.extend(
                    row
                    for row in payload.get("rows") or []
                    if str(row.get("paper_id")) in scan_id_set
                )
                scanned_records += int(payload.get("record_count") or 0)
                continue
        pending.append((path, checkpoint))

    found_ids = {str(row["paper_id"]) for row in collected}
    batch_size = max(1, int(workers)) * 4
    files_scanned_this_run = 0
    for start in range(0, len(pending), batch_size):
        if stop_when_complete and scan_id_set.issubset(found_ids):
            break
        batch = pending[start : start + batch_size]
        with ProcessPoolExecutor(
            max_workers=max(1, int(workers)),
            initializer=_worker_initialize,
            initargs=(scan_ids,),
        ) as executor:
            future_lookup = {
                executor.submit(_scan_one_file, str(path)): (path, checkpoint)
                for path, checkpoint in batch
            }
            for future in as_completed(future_lookup):
                path, checkpoint = future_lookup[future]
                source_file, record_count, rows = future.result()
                payload = {
                    "source_file": source_file,
                    "source_size_bytes": path.stat().st_size,
                    "record_count": int(record_count),
                    "rows": rows,
                }
                checkpoint.write_text(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                collected.extend(rows)
                found_ids.update(str(row["paper_id"]) for row in rows)
                scanned_records += int(record_count)
                files_scanned_this_run += 1

    frame = pd.DataFrame([*seeded, *collected])
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "paper_id",
                "openalex_updated_date",
                "openalex_author_count",
                "openalex_institution_count",
                "openalex_country_count",
                "openalex_author_ids",
                "metadata_source_file",
                "raw_record_sha256",
            ]
        )
    frame = frame.sort_values(
        ["paper_id", "openalex_updated_date", "metadata_source_file"],
        kind="stable",
    ).drop_duplicates("paper_id", keep="last")
    frame = frame[frame["paper_id"].astype(str).isin(ids)].copy()
    root.mkdir(parents=True, exist_ok=True)
    output_path = root / "target_openalex_metadata.parquet"
    frame.to_parquet(output_path, index=False)
    manifest = {
        "artifact_kind": "aspr_v6_1_openalex_control_metadata",
        "version": OPENALEX_CONTROL_VERSION,
        "snapshot_root": str(Path(snapshot_root).resolve()),
        "snapshot_manifest_sha256": "sha256:"
        + hashlib.sha256(
            (Path(snapshot_root) / "data" / "works" / "manifest").read_bytes()
        ).hexdigest(),
        "n_target_ids": len(ids),
        "n_seed_candidate_ids": seed_candidate_count,
        "n_target_ids_seeded": len(seeded_ids),
        "n_seed_ids_rejected_incomplete_projection": len(rejected_seed_ids),
        "n_target_ids_scanned": len(scan_ids),
        "n_files_scanned_this_run": files_scanned_this_run,
        "n_files_registered": len(files),
        "n_files_completed": len(files)
        - len(
            [
                item
                for item in files
                if not (
                    checkpoint_dir
                    / (
                        hashlib.sha256(str(item).encode("utf-8")).hexdigest()[:20]
                        + ".json"
                    )
                ).is_file()
            ]
        ),
        "n_snapshot_records_scanned": int(scanned_records),
        "n_target_records_found": int(len(frame)),
        "coverage": float(len(frame) / len(ids)),
        "output_path": str(output_path),
        "output_sha256": (
            f"sha256:{hashlib.sha256(output_path.read_bytes()).hexdigest()}"
        ),
        "network_used": False,
    }
    manifest_path = root / "target_openalex_metadata_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _fetch_api_batch(
    ids: Sequence[str],
    checkpoint_path: Path,
    api_key: str | None,
) -> Mapping[str, Any]:
    """Fetch and checkpoint one reduced OpenAlex works batch."""
    if checkpoint_path.is_file():
        return json.loads(checkpoint_path.read_text(encoding="utf-8"))
    short_ids = [value.rsplit("/", 1)[-1] for value in ids]
    parameters = {
        "filter": "openalex_id:" + "|".join(short_ids),
        "select": (
            "id,updated_date,authorships,institutions_distinct_count,"
            "countries_distinct_count"
        ),
        "per_page": 100,
    }
    if api_key:
        parameters["api_key"] = api_key
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(parameters)
    error: Exception | None = None
    for attempt in range(5):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "ASPR-expanded-data-builder/1.0"},
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                response_payload = json.load(response)
            payload = {
                "requested_ids": list(ids),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "records": response_payload.get("results") or [],
                "meta": response_payload.get("meta") or {},
            }
            checkpoint_path.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return payload
        except (urllib.error.URLError, TimeoutError) as exc:
            error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"OpenAlex API batch failed: {error}")


def fetch_expanded_target_metadata_api(
    target_ids: Iterable[str],
    seed_metadata_path: Path,
    output_dir: Path,
    *,
    workers: int = 4,
    api_key: str | None = None,
) -> Mapping[str, Any]:
    """Freeze expanded target authorship metadata from batched API reads."""
    ids = sorted(
        {
            normalized
            for value in target_ids
            if (normalized := normalize_openalex_id(value))
        }
    )
    seed = pd.read_parquet(seed_metadata_path)
    seed = seed[seed["paper_id"].astype(str).isin(ids)].copy()
    seed_candidate_count = len(seed)
    source_files = seed.get(
        "metadata_source_file", pd.Series("", index=seed.index, dtype="string")
    ).astype("string")
    incomplete_projection = source_files.str.contains(
        "uncapped_source_year_v2/checkpoints/source_year",
        regex=False,
        na=False,
    ) | source_files.str.contains(
        "/openalex_outputs/checkpoints/target_works/",
        regex=False,
        na=False,
    )
    rejected_seed_ids = set(seed.loc[incomplete_projection, "paper_id"].astype(str))
    seed = seed.loc[~incomplete_projection].copy()
    seed_ids = set(seed["paper_id"].astype(str))
    missing_ids = sorted(set(ids) - seed_ids)
    root = Path(output_dir)
    checkpoint_dir = root / "openalex_api_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    batches = [
        missing_ids[start : start + 100] for start in range(0, len(missing_ids), 100)
    ]
    rows: List[Dict[str, Any]] = []
    total_cost = 0.0
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = {}
        for batch_index, batch in enumerate(batches):
            token = hashlib.sha256("|".join(batch).encode("utf-8")).hexdigest()[:20]
            checkpoint = checkpoint_dir / f"batch_{batch_index:04d}_{token}.json"
            futures[executor.submit(_fetch_api_batch, batch, checkpoint, api_key)] = (
                checkpoint
            )
        for future in as_completed(futures):
            payload = future.result()
            total_cost += float((payload.get("meta") or {}).get("cost_usd") or 0.0)
            rows.extend(
                _target_metadata(record, Path("openalex_api_checkpoint"))
                for record in payload.get("records") or []
            )
    fetched = pd.DataFrame(rows)
    if fetched.empty and missing_ids:
        raise ValueError("OpenAlex API returned no expanded target metadata")
    frame = pd.concat([seed, fetched], ignore_index=True, sort=False)
    frame = frame.sort_values(
        ["paper_id", "openalex_updated_date", "metadata_source_file"],
        kind="stable",
    ).drop_duplicates("paper_id", keep="last")
    frame = frame[frame["paper_id"].astype(str).isin(ids)].copy()
    if len(frame) != len(ids):
        missing = sorted(set(ids) - set(frame["paper_id"].astype(str)))
        raise ValueError(f"expanded OpenAlex metadata is missing {len(missing)} IDs")
    output_path = root / "target_openalex_metadata.parquet"
    frame.to_parquet(output_path, index=False)
    manifest = {
        "artifact_kind": "aspr_expanded_openalex_control_metadata",
        "version": OPENALEX_EXPANDED_API_VERSION,
        "network_used_during_data_build": True,
        "network_forbidden_during_feature_build_and_training": True,
        "seed_metadata_path": str(Path(seed_metadata_path).resolve()),
        "seed_metadata_sha256": "sha256:"
        + hashlib.sha256(Path(seed_metadata_path).read_bytes()).hexdigest(),
        "n_target_ids": len(ids),
        "n_seed_candidate_ids": seed_candidate_count,
        "n_seeded_ids": len(seed_ids),
        "n_seed_ids_rejected_incomplete_projection": len(rejected_seed_ids),
        "incomplete_projection_policy": (
            "Source-year and legacy target-work checkpoints omitted authorships by "
            "construction and must be refreshed before team controls are built."
        ),
        "n_api_requested_ids": len(missing_ids),
        "n_target_records_found": len(frame),
        "coverage": float(len(frame) / len(ids)),
        "api_batch_count": len(batches),
        "api_reported_cost_usd": total_cost,
        "output_path": str(output_path.resolve()),
        "output_sha256": "sha256:"
        + hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }
    manifest_path = root / "target_openalex_metadata_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_k1_team_controls(metadata: pd.DataFrame) -> pd.DataFrame:
    """Convert raw frozen counts to registered log1p K1 controls."""
    required = {
        "paper_id",
        "openalex_author_count",
        "openalex_institution_count",
        "openalex_country_count",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"OpenAlex metadata is missing columns: {missing}")
    output = metadata[sorted(required)].copy()
    mappings = {
        "openalex_author_count": "log_author_count",
        "openalex_institution_count": "log_institution_count",
        "openalex_country_count": "log_country_count",
    }
    for source, target in mappings.items():
        values = pd.to_numeric(output[source], errors="coerce")
        values = values.where(values.ge(0))
        output[target] = values.map(
            lambda value: math.log1p(value) if pd.notna(value) else float("nan")
        )
    return output[
        [
            "paper_id",
            "log_author_count",
            "log_institution_count",
            "log_country_count",
        ]
    ]


__all__ = [
    "OPENALEX_CONTROL_VERSION",
    "build_k1_team_controls",
    "extract_target_metadata",
    "fetch_expanded_target_metadata_api",
    "snapshot_work_files",
]

"""Local-only OpenAlex target metadata extraction for v6.1 controls."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import pandas as pd

from aspr.corpus import normalize_openalex_id


OPENALEX_CONTROL_VERSION = "aspr-openalex-controls-v6.1-1"
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
        "openalex_author_count": int(authors_count)
        if isinstance(authors_count, (int, float)) and authors_count >= 0
        else len(author_ids),
        "openalex_institution_count": int(institutions_count)
        if isinstance(institutions_count, (int, float))
        and institutions_count >= 0
        else len(institution_ids),
        "openalex_country_count": int(countries_count)
        if isinstance(countries_count, (int, float)) and countries_count >= 0
        else len(countries),
        "openalex_author_ids": sorted(author_ids),
        "metadata_source_file": str(source_file),
        "raw_record_sha256": (
            f"sha256:{hashlib.sha256(canonical_record).hexdigest()}"
        ),
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
    if max_files is not None:
        files = files[: int(max_files)]
    pending = []
    collected: List[Dict[str, Any]] = []
    scanned_records = 0
    for path in files:
        token = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:20]
        checkpoint = checkpoint_dir / f"{token}.json"
        if resume and checkpoint.is_file():
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            if payload.get("source_file") == str(path):
                collected.extend(payload.get("rows") or [])
                scanned_records += int(payload.get("record_count") or 0)
                continue
        pending.append((path, checkpoint))

    with ProcessPoolExecutor(
        max_workers=max(1, int(workers)),
        initializer=_worker_initialize,
        initargs=(ids,),
    ) as executor:
        future_lookup = {
            executor.submit(_scan_one_file, str(path)): (path, checkpoint)
            for path, checkpoint in pending
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
            scanned_records += int(record_count)

    frame = pd.DataFrame(collected)
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
    root.mkdir(parents=True, exist_ok=True)
    output_path = root / "target_openalex_metadata.parquet"
    frame.to_parquet(output_path, index=False)
    manifest = {
        "artifact_kind": "aspr_v6_1_openalex_control_metadata",
        "version": OPENALEX_CONTROL_VERSION,
        "snapshot_root": str(Path(snapshot_root).resolve()),
        "snapshot_manifest_sha256": "sha256:"
        + hashlib.sha256(
            (
                Path(snapshot_root)
                / "data"
                / "works"
                / "manifest"
            ).read_bytes()
        ).hexdigest(),
        "n_target_ids": len(ids),
        "n_files_registered": len(files),
        "n_files_completed": len(files) - len(
            [
                item
                for item in files
                if not (
                    checkpoint_dir
                    / (
                        hashlib.sha256(str(item).encode("utf-8")).hexdigest()[
                            :20
                        ]
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
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
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
    "snapshot_work_files",
]

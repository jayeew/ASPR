"""Derive a compact work-metadata view from existing local raw checkpoints."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from aspr.corpus import normalize_openalex_id

from .source_audit_v6 import sha256_file
from .v5_adapter import iter_jsonl_records


VIEW_VERSION = "aspr-local-work-view-v6-1"
WORK_VIEW_SCHEMA = pa.schema(
    [
        ("work_id", pa.string()),
        ("publication_year", pa.int32()),
        ("title", pa.string()),
        ("abstract", pa.string()),
        ("work_type", pa.string()),
        ("source_id", pa.string()),
        ("source_name", pa.string()),
        ("source_type", pa.string()),
        ("topic_id", pa.string()),
        ("topic_name", pa.string()),
        ("subfield_id", pa.string()),
        ("subfield_name", pa.string()),
        ("field_id", pa.string()),
        ("field_name", pa.string()),
        ("domain_id", pa.string()),
        ("domain_name", pa.string()),
        ("referenced_works", pa.list_(pa.string())),
        ("record_sha256", pa.string()),
        ("origin_path", pa.string()),
        ("origin_line", pa.int64()),
        ("view_version", pa.string()),
    ]
)


def reconstruct_abstract(inverted_index: Any) -> Optional[str]:
    """Reconstruct OpenAlex abstract text from its inverted index."""
    if not isinstance(inverted_index, Mapping) or not inverted_index:
        return None
    positioned: List[tuple[int, str]] = []
    for token, positions in inverted_index.items():
        if not isinstance(positions, Sequence) or isinstance(positions, (str, bytes)):
            continue
        for position in positions:
            try:
                positioned.append((int(position), str(token)))
            except (TypeError, ValueError):
                continue
    if not positioned:
        return None
    positioned.sort(key=lambda item: item[0])
    return " ".join(token for _, token in positioned)


def _source_record(work: Mapping[str, Any]) -> Mapping[str, Any]:
    primary = work.get("primary_location")
    if isinstance(primary, Mapping) and isinstance(primary.get("source"), Mapping):
        return primary["source"]
    locations = work.get("locations")
    if isinstance(locations, Sequence) and not isinstance(locations, (str, bytes)):
        for location in locations:
            if isinstance(location, Mapping) and isinstance(
                location.get("source"), Mapping
            ):
                return location["source"]
    return {}


def _topic_parts(work: Mapping[str, Any]) -> Dict[str, str]:
    topic = work.get("primary_topic")
    topic = topic if isinstance(topic, Mapping) else {}
    subfield = topic.get("subfield")
    subfield = subfield if isinstance(subfield, Mapping) else {}
    field = topic.get("field")
    field = field if isinstance(field, Mapping) else {}
    domain = topic.get("domain")
    domain = domain if isinstance(domain, Mapping) else {}
    return {
        "topic_id": normalize_openalex_id(topic.get("id")),
        "topic_name": str(topic.get("display_name") or ""),
        "subfield_id": normalize_openalex_id(subfield.get("id")),
        "subfield_name": str(subfield.get("display_name") or ""),
        "field_id": normalize_openalex_id(field.get("id")),
        "field_name": str(field.get("display_name") or ""),
        "domain_id": normalize_openalex_id(domain.get("id")),
        "domain_name": str(domain.get("display_name") or ""),
    }


def compact_work_record(
    work: Mapping[str, Any], *, origin_path: str = "", origin_line: int = 0
) -> Dict[str, Any]:
    """Select v6 fields and attach a hash of the complete source record."""
    work_id = normalize_openalex_id(work.get("id"))
    source = _source_record(work)
    topics = _topic_parts(work)
    referenced = work.get("referenced_works")
    referenced = (
        [normalize_openalex_id(item) for item in referenced]
        if isinstance(referenced, list)
        else []
    )
    referenced = [item for item in referenced if item]
    raw_payload = json.dumps(
        dict(work),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    year_value = work.get("publication_year") or work.get("year")
    try:
        publication_year = int(year_value) if year_value is not None else None
    except (TypeError, ValueError):
        publication_year = None
    return {
        "work_id": work_id,
        "publication_year": publication_year,
        "title": str(work.get("display_name") or work.get("title") or ""),
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
        "work_type": str(work.get("type") or ""),
        "source_id": normalize_openalex_id(source.get("id")),
        "source_name": str(source.get("display_name") or ""),
        "source_type": str(source.get("type") or ""),
        **topics,
        "referenced_works": referenced,
        "record_sha256": f"sha256:{hashlib.sha256(raw_payload).hexdigest()}",
        "origin_path": origin_path,
        "origin_line": int(origin_line),
        "view_version": VIEW_VERSION,
    }


def iter_checkpoint_records(path: Path) -> Iterator[tuple[int, Dict[str, Any]]]:
    """Yield line numbers and unwrapped work objects from a local checkpoint."""
    for line_number, work in enumerate(
        iter_jsonl_records(Path(path), unwrap_key="work", strict=True), start=1
    ):
        yield line_number, work


def _batch_table(rows: List[Dict[str, Any]]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=WORK_VIEW_SCHEMA)


def materialize_local_work_view(
    checkpoint_paths: Sequence[Path],
    output_path: Path,
    *,
    required_ids: Optional[Iterable[str]] = None,
    input_hashes: bool = True,
    batch_size: int = 25_000,
) -> Dict[str, Any]:
    """Materialize a deduplicated compact view without contacting OpenAlex."""
    paths = [Path(path) for path in checkpoint_paths]
    if not paths or any(not path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise FileNotFoundError(f"local checkpoint files are missing: {missing}")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    database_path = output.with_suffix(output.suffix + ".dedupe.sqlite")
    if output.exists() or database_path.exists():
        raise FileExistsError(
            "local work view or dedupe state already exists; publish a new versioned path"
        )

    wanted = (
        {normalize_openalex_id(value) for value in required_ids}
        if required_ids is not None
        else None
    )
    if wanted is not None:
        wanted.discard("")
    connection = sqlite3.connect(database_path)
    connection.execute("CREATE TABLE emitted (work_id TEXT PRIMARY KEY)")
    writer: Optional[pq.ParquetWriter] = None
    batch: List[Dict[str, Any]] = []
    records_seen = 0
    records_selected = 0
    duplicates = 0
    abstract_count = 0
    source_count = 0
    try:
        for path in paths:
            for line_number, work in iter_checkpoint_records(path):
                records_seen += 1
                work_id = normalize_openalex_id(work.get("id"))
                if not work_id or (wanted is not None and work_id not in wanted):
                    continue
                inserted = connection.execute(
                    "INSERT OR IGNORE INTO emitted(work_id) VALUES (?)", (work_id,)
                ).rowcount
                if not inserted:
                    duplicates += 1
                    continue
                row = compact_work_record(
                    work, origin_path=str(path), origin_line=line_number
                )
                abstract_count += int(bool(row["abstract"]))
                source_count += int(bool(row["source_id"]))
                batch.append(row)
                records_selected += 1
                if len(batch) >= batch_size:
                    table = _batch_table(batch)
                    if writer is None:
                        writer = pq.ParquetWriter(output, table.schema)
                    writer.write_table(table)
                    batch.clear()
                    connection.commit()
        if batch:
            table = _batch_table(batch)
            if writer is None:
                writer = pq.ParquetWriter(output, table.schema)
            writer.write_table(table)
            batch.clear()
        if writer is None:
            raise ValueError("no checkpoint records matched the requested work IDs")
    finally:
        if writer is not None:
            writer.close()
        connection.commit()
        connection.close()

    input_records = [
        {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path) if input_hashes else None,
        }
        for path in paths
    ]
    manifest = {
        "artifact_kind": "aspr_v6_local_work_view",
        "view_version": VIEW_VERSION,
        "network_policy": "forbidden",
        "input_checkpoints": input_records,
        "required_id_count": len(wanted) if wanted is not None else None,
        "records_seen": records_seen,
        "records_selected": records_selected,
        "duplicates_removed": duplicates,
        "required_id_coverage": (
            records_selected / len(wanted) if wanted else None
        ),
        "abstract_coverage": abstract_count / max(1, records_selected),
        "source_coverage": source_count / max(1, records_selected),
        "output_path": str(output),
        "output_sha256": sha256_file(output),
        "dedupe_database": str(database_path),
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "VIEW_VERSION",
    "WORK_VIEW_SCHEMA",
    "compact_work_record",
    "iter_checkpoint_records",
    "materialize_local_work_view",
    "reconstruct_abstract",
]

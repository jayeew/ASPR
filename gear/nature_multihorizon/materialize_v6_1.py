"""Independent local-only materialization for ASPR v6.1."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import shutil
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from gear.corpus import normalize_openalex_id

from .feature_materializer_v6 import annual_field_distances
from .features_v6 import (
    canonical_pair,
    first_time_source_pair_distance_mean,
    first_time_source_pair_share,
    marginal_pair_z_scores,
    novelty_u,
    rao_stirling_integration,
    uzzi_atypicality_p10,
    uzzi_conventionality_median,
)
from .features_v6_1 import (
    div_index,
    field_distance_quantile,
    field_gini_balance,
    field_gini_simpson,
    field_hhi,
    field_hill_number,
    field_other_field_share,
    field_relative_variety,
    field_shannon_entropy,
    first_time_pair_any,
    first_time_pair_count,
    first_time_pair_distance_sum,
    hypergeometric_pair_z_scores,
    low_frequency_pair_share,
    reference_overlap_novelty,
    source_pair_mean_surprisal,
    true_diversity_from_rao,
)
from .modeling_v6 import safe_spearman
from .openalex_controls_v6_1 import build_k1_team_controls
from .source_audit_v6 import sha256_file


MATERIALIZATION_VERSION_V6_1 = "aspr-v6.1-local-materialization-4"
REFERENCE_OVERLAP_CONTEXT_VERSION = (
    "aspr-v6.1-reference-overlap-context-4"
)
REFERENCE_OVERLAP_REFERENCE_WINDOW_YEARS: Optional[int] = None
REFERENCE_OVERLAP_COCITING_WINDOW_YEARS: Optional[int] = None
FIELD_TAXONOMY_SIZE = 26
B0_INNOVATION_FEATURES: Tuple[str, ...] = (
    "novelty_u_t0_source",
    "uzzi_atypicality_p10_t0",
    "field_variety",
    "field_pielou_evenness",
    "field_disparity_cosine_mean",
    "rao_stirling_integration",
)
PROVISIONAL_CORE8: Tuple[str, ...] = (
    "novelty_u_t0_source",
    "uzzi_atypicality_p10_t0",
    "uzzi_conventionality_median_t0",
    "first_time_source_pair_share",
    "field_variety",
    "field_pielou_evenness",
    "field_disparity_cosine_mean",
    "rao_stirling_integration",
)


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _pair_set(values: Sequence[str]) -> Set[Tuple[Any, Any]]:
    unique = sorted({value for value in values if value})
    return {
        canonical_pair(left, right)
        for left, right in itertools.combinations(unique, 2)
    }


def _as_list(value: Any) -> List[str]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = []
    elif isinstance(value, (list, tuple, set, np.ndarray)):
        parsed = list(value)
    else:
        parsed = []
    return [str(item) for item in parsed if str(item).strip()]


def _normalize_bibliographies(
    papers: pd.DataFrame,
    paper_references: pd.DataFrame,
    reference_metadata: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, List[Any]]]]:
    paper_frame = papers.copy()
    paper_frame["paper_id"] = paper_frame["paper_id"].map(normalize_openalex_id)
    paper_frame["publication_year"] = pd.to_numeric(
        paper_frame["publication_year"], errors="coerce"
    )
    paper_frame = paper_frame[
        paper_frame["paper_id"].ne("")
        & paper_frame["publication_year"].notna()
    ].drop_duplicates("paper_id", keep="last")
    paper_frame["publication_year"] = paper_frame[
        "publication_year"
    ].astype(int)
    metadata = reference_metadata.copy()
    metadata["reference_id"] = metadata["reference_id"].map(
        normalize_openalex_id
    )
    metadata["reference_year"] = pd.to_numeric(
        metadata["reference_year"], errors="coerce"
    )
    metadata["source_id"] = metadata["source_id"].fillna("").astype(str)
    metadata["field_id"] = metadata["field_id"].fillna("").astype(str)
    metadata = metadata.drop_duplicates("reference_id", keep="last")
    bibliography = paper_references[["paper_id", "reference_id"]].copy()
    bibliography["paper_id"] = bibliography["paper_id"].map(
        normalize_openalex_id
    )
    bibliography["reference_id"] = bibliography["reference_id"].map(
        normalize_openalex_id
    )
    bibliography = (
        bibliography.drop_duplicates()
        .merge(
            metadata[
                [
                    "reference_id",
                    "reference_year",
                    "source_id",
                    "field_id",
                ]
            ],
            on="reference_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            paper_frame[["paper_id", "publication_year"]],
            on="paper_id",
            how="inner",
            validate="many_to_one",
        )
    )
    prior = bibliography[
        bibliography["reference_year"].notna()
        & bibliography["reference_year"].lt(bibliography["publication_year"])
    ].copy()
    grouped: Dict[str, Dict[str, List[Any]]] = {}
    declared = (
        bibliography.groupby("paper_id", sort=False)["reference_id"]
        .agg(list)
        .to_dict()
    )
    for paper_id, group in prior.groupby("paper_id", sort=False):
        grouped[str(paper_id)] = {
            "reference_ids": group["reference_id"].astype(str).tolist(),
            "reference_years": group["reference_year"].astype(int).tolist(),
            "source_ids": [
                value for value in group["source_id"].astype(str) if value
            ],
            "field_ids": [
                value for value in group["field_id"].astype(str) if value
            ],
            "declared_reference_ids": declared.get(str(paper_id), []),
        }
    for paper_id in paper_frame["paper_id"].astype(str):
        grouped.setdefault(
            paper_id,
            {
                "reference_ids": [],
                "reference_years": [],
                "source_ids": [],
                "field_ids": [],
                "declared_reference_ids": declared.get(paper_id, []),
            },
        )
    return paper_frame, grouped


def _historical_group_rows(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    """Collapse a contiguous edge batch to one row per citing work."""
    rows: List[Dict[str, Any]] = []
    for work_id, group in frame.groupby("source_work_id", sort=False):
        source_year = pd.to_numeric(
            group["source_year"].iloc[0], errors="coerce"
        )
        if pd.isna(source_year):
            continue
        target_sources = sorted(
            {
                value
                for value in group["target_source_id"].astype(str)
                if value
            }
        )
        if not target_sources:
            continue
        rows.append(
            {
                "work_id": str(work_id),
                "publication_year": int(source_year),
                "citing_source_id": str(
                    group["citing_source_id"].iloc[0] or ""
                ),
                "cited_source_ids": target_sources,
            }
        )
    return rows


def _append_parquet_rows(
    writer: Optional[pq.ParquetWriter],
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    schema: pa.Schema,
) -> Optional[pq.ParquetWriter]:
    """Append a bounded Python batch to a deterministic parquet artifact."""
    if not rows:
        return writer
    table = pa.Table.from_pylist(list(rows), schema=schema)
    if writer is None:
        writer = pq.ParquetWriter(path, schema, compression="zstd")
    writer.write_table(table)
    return writer


def _write_source_field_profiles(
    counts: Mapping[Tuple[int, str, str], int],
    path: Path,
    *,
    batch_size: int = 100_000,
) -> None:
    """Write aggregated source-field events without a second large table."""
    schema = pa.schema(
        [
            ("source_year", pa.int64()),
            ("source_id", pa.string()),
            ("target_field_id", pa.string()),
            ("citation_count", pa.int64()),
        ]
    )
    writer: Optional[pq.ParquetWriter] = None
    rows: List[Dict[str, Any]] = []
    for (year, source_id, field_id), value in counts.items():
        rows.append(
            {
                "source_year": int(year),
                "source_id": str(source_id),
                "target_field_id": str(field_id),
                "citation_count": int(value),
            }
        )
        if len(rows) >= int(batch_size):
            writer = _append_parquet_rows(writer, path, rows, schema)
            rows = []
    writer = _append_parquet_rows(writer, path, rows, schema)
    if writer is not None:
        writer.close()
    else:
        pq.write_table(
            pa.Table.from_pylist([], schema=schema),
            path,
            compression="zstd",
        )


def materialize_historical_source_context(
    reference_edges_path: Path,
    reference_metadata: pd.DataFrame,
    output_dir: Path,
    *,
    chunksize: int = 500_000,
    resume: bool = True,
) -> Mapping[str, Any]:
    """Build compact all-closure source histories and prior popularity data."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    history_path = root / "historical_paper_sources.parquet"
    profile_path = root / "source_field_citation_events.parquet"
    popularity_db_path = root / "reference_prior_popularity.sqlite"
    manifest_path = root / "historical_source_context_manifest.json"
    if (
        resume
        and history_path.is_file()
        and profile_path.is_file()
        and popularity_db_path.is_file()
        and manifest_path.is_file()
    ):
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    for incomplete in (
        history_path,
        profile_path,
        popularity_db_path,
        popularity_db_path.with_name(popularity_db_path.name + "-wal"),
        popularity_db_path.with_name(popularity_db_path.name + "-shm"),
    ):
        incomplete.unlink(missing_ok=True)

    metadata = reference_metadata[
        [
            "reference_id",
            "reference_year",
            "source_id",
            "field_id",
        ]
    ].copy()
    metadata["reference_id"] = metadata["reference_id"].map(
        normalize_openalex_id
    )
    metadata["reference_year"] = pd.to_numeric(
        metadata["reference_year"], errors="coerce"
    )
    metadata["source_id"] = metadata["source_id"].fillna("").astype(str)
    metadata["field_id"] = metadata["field_id"].fillna("").astype(str)
    metadata = metadata.drop_duplicates("reference_id", keep="last")
    year_lookup = metadata.set_index("reference_id")[
        "reference_year"
    ].to_dict()
    source_lookup = metadata.set_index("reference_id")["source_id"].to_dict()
    field_lookup = metadata.set_index("reference_id")["field_id"].to_dict()

    connection = sqlite3.connect(popularity_db_path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS citation_counts "
        "(reference_id TEXT NOT NULL, source_year INTEGER NOT NULL, "
        "citation_count INTEGER NOT NULL, "
        "PRIMARY KEY(reference_id, source_year))"
    )
    history_schema = pa.schema(
        [
            ("work_id", pa.string()),
            ("publication_year", pa.int64()),
            ("citing_source_id", pa.string()),
            ("cited_source_ids", pa.list_(pa.string())),
        ]
    )
    history_writer: Optional[pq.ParquetWriter] = None
    history_row_count = 0
    source_field_counts: Counter[Tuple[int, str, str]] = Counter()
    carry = pd.DataFrame()
    edge_rows = 0
    for chunk in pd.read_csv(
        reference_edges_path,
        usecols=["source", "target"],
        dtype={"source": "string", "target": "string"},
        chunksize=int(chunksize),
    ):
        chunk["source_work_id"] = chunk["source"].map(normalize_openalex_id)
        chunk["target_work_id"] = chunk["target"].map(normalize_openalex_id)
        chunk["source_year"] = chunk["source_work_id"].map(year_lookup)
        chunk["citing_source_id"] = (
            chunk["source_work_id"].map(source_lookup).fillna("").astype(str)
        )
        chunk["target_source_id"] = (
            chunk["target_work_id"].map(source_lookup).fillna("").astype(str)
        )
        chunk["target_field_id"] = (
            chunk["target_work_id"].map(field_lookup).fillna("").astype(str)
        )
        edge_rows += len(chunk)
        valid_popularity = chunk[
            chunk["source_year"].notna()
            & chunk["target_work_id"].ne("")
        ][["target_work_id", "source_year"]].copy()
        valid_popularity["source_year"] = valid_popularity[
            "source_year"
        ].astype(int)
        popularity = (
            valid_popularity.value_counts(sort=False)
            .rename("citation_count")
            .reset_index()
        )
        connection.executemany(
            "INSERT INTO citation_counts(reference_id,source_year,citation_count) "
            "VALUES(?,?,?) ON CONFLICT(reference_id,source_year) DO UPDATE SET "
            "citation_count=citation_count+excluded.citation_count",
            [
                (str(row.target_work_id), int(row.source_year), int(row.citation_count))
                for row in popularity.itertuples(index=False)
            ],
        )
        valid_profiles = chunk[
            chunk["source_year"].notna()
            & chunk["citing_source_id"].ne("")
            & chunk["target_field_id"].ne("")
        ]
        profile_counts = (
            valid_profiles.assign(
                source_year=valid_profiles["source_year"].astype(int)
            )[
                ["source_year", "citing_source_id", "target_field_id"]
            ]
            .value_counts(sort=False)
        )
        source_field_counts.update(
            {
                (int(key[0]), str(key[1]), str(key[2])): int(value)
                for key, value in profile_counts.items()
            }
        )
        group_frame = chunk[
            [
                "source_work_id",
                "source_year",
                "citing_source_id",
                "target_source_id",
            ]
        ]
        if not carry.empty:
            group_frame = pd.concat([carry, group_frame], ignore_index=True)
        last_source = str(group_frame["source_work_id"].iloc[-1])
        carry_mask = group_frame["source_work_id"].astype(str).eq(last_source)
        ready = group_frame[~carry_mask]
        carry = group_frame[carry_mask].copy()
        ready_rows = _historical_group_rows(ready)
        history_writer = _append_parquet_rows(
            history_writer,
            history_path,
            ready_rows,
            history_schema,
        )
        history_row_count += len(ready_rows)
        connection.commit()
    if not carry.empty:
        carry_rows = _historical_group_rows(carry)
        history_writer = _append_parquet_rows(
            history_writer,
            history_path,
            carry_rows,
            history_schema,
        )
        history_row_count += len(carry_rows)
    if history_writer is not None:
        history_writer.close()
    else:
        pq.write_table(
            pa.Table.from_pylist([], schema=history_schema),
            history_path,
            compression="zstd",
        )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_citation_reference_year "
        "ON citation_counts(reference_id, source_year)"
    )
    connection.commit()
    connection.close()
    _write_source_field_profiles(source_field_counts, profile_path)
    manifest = {
        "artifact_kind": "aspr_v6_1_historical_source_context",
        "materialization_version": MATERIALIZATION_VERSION_V6_1,
        "network_used": False,
        "reference_edges_path": str(Path(reference_edges_path).resolve()),
        "reference_edges_sha256": sha256_file(reference_edges_path),
        "n_edge_rows": int(edge_rows),
        "n_historical_papers": int(history_row_count),
        "n_source_field_profile_rows": int(len(source_field_counts)),
        "outputs": {
            "historical_paper_sources": {
                "path": str(history_path),
                "sha256": sha256_file(history_path),
            },
            "source_field_events": {
                "path": str(profile_path),
                "sha256": sha256_file(profile_path),
            },
            "reference_prior_popularity": {
                "path": str(popularity_db_path),
                "sha256": sha256_file(popularity_db_path),
            },
        },
    }
    manifest["artifact_id"] = _canonical_hash(manifest)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return manifest


def _historical_reference_group_rows(
    frame: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Collapse a contiguous edge batch to source-paper reference lists."""
    rows: List[Dict[str, Any]] = []
    for work_id, group in frame.groupby("source_work_id", sort=False):
        source_year = pd.to_numeric(
            group["source_year"].iloc[0], errors="coerce"
        )
        source_field = str(group["source_field_id"].iloc[0] or "")
        if pd.isna(source_year) or not source_field:
            continue
        references: Dict[str, int] = {}
        for item in group.itertuples(index=False):
            target_id = str(item.target_work_id or "")
            target_year = pd.to_numeric(item.target_year, errors="coerce")
            if (
                target_id
                and pd.notna(target_year)
                and int(target_year) < int(source_year)
            ):
                references[target_id] = int(target_year)
        if not references:
            continue
        ordered = sorted(references.items())
        rows.append(
            {
                "work_id": str(work_id),
                "publication_year": int(source_year),
                "openalex_primary_field": source_field,
                "reference_ids": [item[0] for item in ordered],
                "reference_years": [item[1] for item in ordered],
            }
        )
    return rows


def materialize_reference_overlap_context(
    reference_edges_path: Path,
    target_works_path: Path,
    eligible_source_ids: Optional[Iterable[str]],
    reference_metadata: pd.DataFrame,
    output_dir: Path,
    *,
    chunksize: int = 500_000,
    resume: bool = True,
) -> Mapping[str, Any]:
    """Build the local history needed for source-faithful overlap novelty."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    output_path = root / "historical_paper_references.parquet"
    manifest_path = root / "reference_overlap_context_manifest.json"
    if resume and output_path.is_file() and manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("context_version")
            == REFERENCE_OVERLAP_CONTEXT_VERSION
            and existing["outputs"]["historical_paper_references"][
                "sha256"
            ]
            == sha256_file(output_path)
        ):
            return existing

    target_metadata = pd.read_csv(
        target_works_path,
        usecols=["id", "year", "openalex_primary_field"],
        dtype={
            "id": "string",
            "year": "Int64",
            "openalex_primary_field": "string",
        },
    )
    target_metadata["id"] = target_metadata["id"].map(
        normalize_openalex_id
    )
    target_metadata["year"] = pd.to_numeric(
        target_metadata["year"], errors="coerce"
    )
    target_metadata["openalex_primary_field"] = (
        target_metadata["openalex_primary_field"]
        .fillna("")
        .astype(str)
    )
    if eligible_source_ids is None:
        eligible_ids = set(target_metadata["id"].dropna().astype(str))
        source_scope = (
            "all frozen Nature v5 target records with a valid year, "
            "primary field, and prior bibliography"
        )
    else:
        eligible_ids = {
            normalized
            for value in eligible_source_ids
            if (normalized := normalize_openalex_id(value))
        }
        source_scope = (
            "explicit eligible-source list supplied by the caller"
        )
    if not eligible_ids:
        raise ValueError("eligible overlap-history source IDs are empty")
    target_metadata = target_metadata[
        target_metadata["id"].isin(eligible_ids)
    ].drop_duplicates("id", keep="last")
    source_year_lookup = target_metadata.set_index("id")["year"].to_dict()
    source_field_lookup = target_metadata.set_index("id")[
        "openalex_primary_field"
    ].to_dict()

    reference_frame = reference_metadata[
        ["reference_id", "reference_year"]
    ].copy()
    reference_frame["reference_id"] = reference_frame["reference_id"].map(
        normalize_openalex_id
    )
    reference_frame["reference_year"] = pd.to_numeric(
        reference_frame["reference_year"], errors="coerce"
    )
    reference_frame = reference_frame.drop_duplicates(
        "reference_id", keep="last"
    )
    reference_year_lookup = reference_frame.set_index("reference_id")[
        "reference_year"
    ].to_dict()
    schema = pa.schema(
        [
            ("work_id", pa.string()),
            ("publication_year", pa.int64()),
            ("openalex_primary_field", pa.string()),
            ("reference_ids", pa.list_(pa.string())),
            ("reference_years", pa.list_(pa.int64())),
        ]
    )
    incomplete = output_path.with_suffix(".incomplete.parquet")
    incomplete.unlink(missing_ok=True)
    writer: Optional[pq.ParquetWriter] = None
    carry = pd.DataFrame()
    edge_rows = 0
    history_rows = 0
    for chunk in pd.read_csv(
        reference_edges_path,
        usecols=["source", "target"],
        dtype={"source": "string", "target": "string"},
        chunksize=int(chunksize),
    ):
        chunk["source_work_id"] = chunk["source"].map(normalize_openalex_id)
        chunk["target_work_id"] = chunk["target"].map(normalize_openalex_id)
        chunk["source_year"] = chunk["source_work_id"].map(
            source_year_lookup
        )
        chunk["source_field_id"] = (
            chunk["source_work_id"]
            .map(source_field_lookup)
            .fillna("")
            .astype(str)
        )
        chunk["target_year"] = chunk["target_work_id"].map(
            reference_year_lookup
        )
        edge_rows += len(chunk)
        group_frame = chunk[
            [
                "source_work_id",
                "source_year",
                "source_field_id",
                "target_work_id",
                "target_year",
            ]
        ]
        if not carry.empty:
            group_frame = pd.concat([carry, group_frame], ignore_index=True)
        last_source = str(group_frame["source_work_id"].iloc[-1])
        carry_mask = group_frame["source_work_id"].astype(str).eq(last_source)
        ready = group_frame[~carry_mask]
        carry = group_frame[carry_mask].copy()
        ready_rows = _historical_reference_group_rows(ready)
        writer = _append_parquet_rows(
            writer, incomplete, ready_rows, schema
        )
        history_rows += len(ready_rows)
    if not carry.empty:
        carry_rows = _historical_reference_group_rows(carry)
        writer = _append_parquet_rows(
            writer, incomplete, carry_rows, schema
        )
        history_rows += len(carry_rows)
    if writer is not None:
        writer.close()
    else:
        pq.write_table(
            pa.Table.from_pylist([], schema=schema),
            incomplete,
            compression="zstd",
        )
    incomplete.replace(output_path)
    identity = {
        "artifact_kind": "aspr_v6_1_reference_overlap_context",
        "context_version": REFERENCE_OVERLAP_CONTEXT_VERSION,
        "network_used": False,
        "reference_edges_path": str(Path(reference_edges_path).resolve()),
        "reference_edges_sha256": sha256_file(reference_edges_path),
        "target_works_path": str(Path(target_works_path).resolve()),
        "target_works_sha256": sha256_file(target_works_path),
        "source_scope": source_scope,
        "n_eligible_source_ids": len(eligible_ids),
        "eligible_source_ids_sha256": _canonical_hash(
            {"paper_ids": sorted(eligible_ids)}
        ),
        "target_metadata_key_profile_sha256": _canonical_hash(
            {
                "n_rows": len(target_metadata),
                "id_min": str(target_metadata["id"].min()),
                "id_max": str(target_metadata["id"].max()),
            }
        ),
        "reference_metadata_key_profile_sha256": _canonical_hash(
            {
                "n_rows": len(reference_frame),
                "id_min": str(reference_frame["reference_id"].min()),
                "id_max": str(reference_frame["reference_id"].max()),
            }
        ),
        "n_edge_rows": int(edge_rows),
        "n_historical_papers": int(history_rows),
        "outputs": {
            "historical_paper_references": {
                "path": str(output_path),
                "sha256": sha256_file(output_path),
            }
        },
    }
    manifest = {**identity, "artifact_id": _canonical_hash(identity)}
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return manifest


def build_reference_overlap_features(
    papers: pd.DataFrame,
    paper_references: pd.DataFrame,
    reference_metadata: pd.DataFrame,
    historical_paper_references: pd.DataFrame,
    *,
    reference_window_years: Optional[int] = 10,
    cociting_window_years: Optional[int] = 3,
) -> pd.DataFrame:
    """Compute Matsumoto-style 1-minus-mean-Jaccard paper novelty."""
    paper_frame, bibliographies = _normalize_bibliographies(
        papers, paper_references, reference_metadata
    )
    history = historical_paper_references.copy()
    history["publication_year"] = pd.to_numeric(
        history["publication_year"], errors="coerce"
    )
    history = history[history["publication_year"].notna()].copy()
    history["publication_year"] = history["publication_year"].astype(int)
    rows: List[Dict[str, Any]] = []
    for year, year_papers in paper_frame.groupby(
        "publication_year", sort=True
    ):
        year = int(year)
        selected = history[history["publication_year"].lt(year)]
        if cociting_window_years is not None:
            selected = selected[
                selected["publication_year"].ge(
                    year - int(cociting_window_years)
                )
            ]
        comparison_sets: Dict[str, Set[str]] = {}
        comparison_fields: Dict[str, str] = {}
        postings: Dict[Tuple[str, str], Set[str]] = {}
        for item in selected.itertuples(index=False):
            references = {
                str(reference_id)
                for reference_id, reference_year in zip(
                    _as_list(item.reference_ids),
                    list(item.reference_years),
                )
                if int(reference_year) < year
                and (
                    reference_window_years is None
                    or int(reference_year)
                    >= year - int(reference_window_years)
                )
            }
            field = str(item.openalex_primary_field or "")
            if not field or not references:
                continue
            work_id = str(item.work_id)
            comparison_sets[work_id] = references
            comparison_fields[work_id] = field
            for reference_id in references:
                postings.setdefault((field, reference_id), set()).add(
                    work_id
                )
        for paper in year_papers.to_dict("records"):
            paper_id = str(paper["paper_id"])
            bibliography = bibliographies[paper_id]
            focal = {
                str(reference_id)
                for reference_id, reference_year in zip(
                    bibliography["reference_ids"],
                    bibliography["reference_years"],
                )
                if int(reference_year) < year
                and (
                    reference_window_years is None
                    or int(reference_year)
                    >= year - int(reference_window_years)
                )
            }
            field = str(paper.get("openalex_primary_field") or "")
            comparison_ids: Set[str] = set()
            for reference_id in focal:
                comparison_ids.update(
                    postings.get((field, reference_id), set())
                )
            comparisons = [
                comparison_sets[work_id]
                for work_id in sorted(comparison_ids)
                if comparison_fields[work_id] == field
            ]
            rows.append(
                {
                    "paper_id": paper_id,
                    "reference_overlap_novelty_t0": (
                        reference_overlap_novelty(focal, comparisons)
                    ),
                    "reference_overlap_comparison_count": len(comparisons),
                    "reference_overlap_reference_count": len(focal),
                }
            )
    return pd.DataFrame(rows)


def _annual_source_profiles(
    events: pd.DataFrame,
    focal_year: int,
    *,
    window_years: int,
) -> Dict[str, np.ndarray]:
    selected = events[
        events["source_year"].between(
            int(focal_year) - int(window_years),
            int(focal_year) - 1,
            inclusive="both",
        )
    ]
    if selected.empty:
        return {}
    matrix = selected.pivot_table(
        index="source_id",
        columns="target_field_id",
        values="citation_count",
        aggfunc="sum",
        fill_value=0.0,
    ).astype(float)
    return {
        str(source): matrix.loc[source].to_numpy(dtype=float)
        for source in matrix.index
    }


def _source_distance(
    pair: Tuple[Any, Any],
    profiles: Mapping[str, np.ndarray],
) -> Optional[float]:
    left = profiles.get(str(pair[0]))
    right = profiles.get(str(pair[1]))
    if left is None or right is None or left.shape != right.shape:
        return None
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0:
        return None
    similarity = float(np.dot(left, right) / denominator)
    return float(np.clip(1.0 - similarity, 0.0, 1.0))


def build_candidate_innovation_features(
    papers: pd.DataFrame,
    paper_references: pd.DataFrame,
    reference_metadata: pd.DataFrame,
    field_events: pd.DataFrame,
    historical_paper_sources: pd.DataFrame,
    source_field_events: pd.DataFrame,
    *,
    field_profile_window_years: int = 5,
) -> pd.DataFrame:
    """Materialize all feasible candidate families without outcomes."""
    paper_frame, bibliographies = _normalize_bibliographies(
        papers, paper_references, reference_metadata
    )
    focal_years = sorted(paper_frame["publication_year"].unique())
    field_distances = annual_field_distances(
        field_events,
        focal_years,
        window_years=int(field_profile_window_years),
    )
    historical = historical_paper_sources.copy()
    historical["publication_year"] = pd.to_numeric(
        historical["publication_year"], errors="coerce"
    )
    historical = historical[
        historical["publication_year"].notna()
    ].sort_values(["publication_year", "work_id"], kind="stable")
    historical["publication_year"] = historical[
        "publication_year"
    ].astype(int)
    historical_records = historical.itertuples(index=False)
    historical_record = next(historical_records, None)
    n_historical_papers = 0
    source_counts: Counter[Any] = Counter()
    pair_counts: Counter[Tuple[Any, Any]] = Counter()
    rows: List[Dict[str, Any]] = []
    for year, year_papers in paper_frame.groupby(
        "publication_year", sort=True
    ):
        year = int(year)
        while (
            historical_record is not None
            and int(historical_record.publication_year) < year
        ):
            values = _as_list(historical_record.cited_source_ids)
            source_counts.update(set(values))
            pair_counts.update(_pair_set(values))
            n_historical_papers += 1
            historical_record = next(historical_records, None)
        source_profiles = _annual_source_profiles(
            source_field_events,
            year,
            window_years=int(field_profile_window_years),
        )
        year_field_distances = field_distances.get(year, {})
        source_distance_cache: Dict[Tuple[Any, Any], float] = {}
        for paper in year_papers.to_dict("records"):
            paper_id = str(paper["paper_id"])
            values = bibliographies[paper_id]
            sources = values["source_ids"]
            fields = values["field_ids"]
            z_scores = marginal_pair_z_scores(
                sources,
                pair_counts,
                source_counts,
                n_historical_papers,
            )
            hypergeometric_z_scores = hypergeometric_pair_z_scores(
                sources,
                pair_counts,
                source_counts,
                n_historical_papers,
            )
            novel_pairs = [
                pair
                for pair in _pair_set(sources)
                if int(pair_counts.get(pair, 0)) == 0
            ]
            for pair in novel_pairs:
                if pair not in source_distance_cache:
                    distance = _source_distance(pair, source_profiles)
                    if distance is not None:
                        source_distance_cache[pair] = distance
            try:
                field_mean = field_distance_quantile(
                    fields, year_field_distances, quantile=0.5
                )
                occupied_distances = [
                    float(year_field_distances[pair])
                    for pair in _pair_set(fields)
                ]
                field_mean = (
                    float(np.mean(occupied_distances))
                    if occupied_distances
                    else float("nan")
                )
                field_max = field_distance_quantile(
                    fields, year_field_distances, quantile=1.0
                )
                field_p90 = field_distance_quantile(
                    fields, year_field_distances, quantile=0.9
                )
                rao = rao_stirling_integration(
                    fields, year_field_distances
                )
                div = div_index(
                    fields,
                    year_field_distances,
                    total_categories=FIELD_TAXONOMY_SIZE,
                )
            except KeyError:
                field_mean = field_max = field_p90 = rao = div = float("nan")
            variety = float(len(set(fields))) if fields else float("nan")
            shannon = field_shannon_entropy(fields)
            gini_simpson = field_gini_simpson(fields)
            row = {
                "paper_id": paper_id,
                "publication_year": year,
                "domain12": str(paper.get("domain12") or "unmapped"),
                "novelty_u_t0_source": novelty_u(
                    sources,
                    pair_counts,
                    source_counts,
                    n_historical_papers,
                ),
                "source_pair_mean_surprisal": source_pair_mean_surprisal(
                    sources,
                    pair_counts,
                    source_counts,
                    n_historical_papers,
                ),
                "low_frequency_source_pair_share": low_frequency_pair_share(
                    sources, pair_counts, maximum_prior_count=1
                ),
                "uzzi_atypicality_p10_t0": uzzi_atypicality_p10(z_scores),
                "uzzi_conventionality_median_t0": (
                    uzzi_conventionality_median(z_scores)
                ),
                "hypergeom_atypicality_p10_t0": uzzi_atypicality_p10(
                    hypergeometric_z_scores
                ),
                "hypergeom_conventionality_median_t0": (
                    uzzi_conventionality_median(hypergeometric_z_scores)
                ),
                "first_time_source_pair_any": first_time_pair_any(
                    sources, pair_counts
                ),
                "first_time_source_pair_count": first_time_pair_count(
                    sources, pair_counts
                ),
                "first_time_source_pair_share": first_time_source_pair_share(
                    sources, pair_counts
                ),
                "first_time_source_pair_distance_sum": (
                    first_time_pair_distance_sum(
                        sources,
                        pair_counts,
                        source_distance_cache,
                    )
                ),
                "first_time_source_pair_distance_mean": (
                    first_time_source_pair_distance_mean(
                        sources,
                        pair_counts,
                        source_distance_cache,
                    )
                ),
                "field_variety": variety,
                "field_relative_variety": (
                    field_relative_variety(
                        fields, total_categories=FIELD_TAXONOMY_SIZE
                    )
                    if fields
                    else float("nan")
                ),
                "reference_other_field_share": field_other_field_share(
                    fields,
                    focal_field=paper.get("openalex_primary_field"),
                ),
                "field_gini_balance": field_gini_balance(fields),
                "field_shannon_entropy": shannon,
                "field_pielou_evenness": (
                    shannon / math.log(len(set(fields)))
                    if len(set(fields)) >= 2
                    else float("nan")
                ),
                "field_gini_simpson": gini_simpson,
                "field_hhi": field_hhi(fields),
                "field_hill_q0": (
                    field_hill_number(fields, order=0)
                    if fields
                    else float("nan")
                ),
                "field_hill_q1": (
                    field_hill_number(fields, order=1)
                    if fields
                    else float("nan")
                ),
                "field_hill_q2": (
                    field_hill_number(fields, order=2)
                    if fields
                    else float("nan")
                ),
                "field_disparity_cosine_mean": field_mean,
                "field_disparity_cosine_max": field_max,
                "field_disparity_cosine_p90": field_p90,
                "rao_stirling_integration": rao,
                "field_div_index": div,
                "rao_true_diversity_q2": true_diversity_from_rao(rao),
                "valid_reference_count": len(values["reference_ids"]),
                "valid_source_pair_count": len(_pair_set(sources)),
                "source_mapping_coverage": len(sources)
                / max(1, len(values["reference_ids"])),
                "field_mapping_coverage": len(fields)
                / max(1, len(values["reference_ids"])),
                "novel_source_pair_distance_coverage": len(
                    set(novel_pairs) & set(source_distance_cache)
                )
                / max(1, len(novel_pairs)),
                "n_historical_source_papers": n_historical_papers,
                "source_max_year": year - 1,
                "definition_version": MATERIALIZATION_VERSION_V6_1,
            }
            rows.append(row)
    output = pd.DataFrame(rows)
    if output["source_max_year"].ge(output["publication_year"]).any():
        raise ValueError("v6.1 innovation features contain time leakage")
    return output


def _title_word_counts(
    target_works_path: Path,
) -> pd.DataFrame:
    frame = pd.read_csv(
        target_works_path,
        usecols=["id", "title"],
        low_memory=False,
    )
    frame["paper_id"] = frame["id"].map(normalize_openalex_id)
    frame["title_word_count"] = (
        frame["title"]
        .fillna("")
        .astype(str)
        .str.findall(r"\b[\w'-]+\b")
        .str.len()
        .astype(float)
    )
    frame.loc[frame["title_word_count"].eq(0), "title_word_count"] = np.nan
    return frame[["paper_id", "title_word_count"]].drop_duplicates(
        "paper_id", keep="last"
    )


def _team_prior_output(metadata: pd.DataFrame, papers: pd.DataFrame) -> pd.DataFrame:
    """Return prior Nature-target output, not an all-OpenAlex career count."""
    frame = papers[["paper_id", "publication_year"]].merge(
        metadata[["paper_id", "openalex_author_ids"]],
        on="paper_id",
        how="left",
        validate="one_to_one",
    )
    author_counts: Counter[str] = Counter()
    rows = []
    for year, group in frame.groupby("publication_year", sort=True):
        for row in group.itertuples(index=False):
            authors = _as_list(row.openalex_author_ids)
            prior = [author_counts[author] for author in authors]
            rows.append(
                {
                    "paper_id": str(row.paper_id),
                    "log_team_prior_nature_output_max": (
                        float(math.log1p(max(prior))) if prior else np.nan
                    ),
                }
            )
        for row in group.itertuples(index=False):
            author_counts.update(set(_as_list(row.openalex_author_ids)))
    return pd.DataFrame(rows)


def _reference_prior_popularity(
    papers: pd.DataFrame,
    bibliographies: Mapping[str, Mapping[str, Sequence[Any]]],
    sqlite_path: Path,
) -> pd.DataFrame:
    """Compute reference prior popularity with one indexed SQLite query."""
    connection = sqlite3.connect(sqlite_path)
    connection.execute("DROP TABLE IF EXISTS temp.focal_references")
    connection.execute(
        "CREATE TEMP TABLE focal_references "
        "(paper_id TEXT NOT NULL, publication_year INTEGER NOT NULL, "
        "reference_order INTEGER NOT NULL, reference_id TEXT NOT NULL)"
    )
    insert_rows: List[Tuple[str, int, int, str]] = []
    for paper in papers[
        ["paper_id", "publication_year"]
    ].itertuples(index=False):
        for order, reference_id in enumerate(
            bibliographies[str(paper.paper_id)]["reference_ids"]
        ):
            insert_rows.append(
                (
                    str(paper.paper_id),
                    int(paper.publication_year),
                    int(order),
                    str(reference_id),
                )
            )
            if len(insert_rows) >= 100_000:
                connection.executemany(
                    "INSERT INTO focal_references VALUES(?,?,?,?)",
                    insert_rows,
                )
                insert_rows = []
    if insert_rows:
        connection.executemany(
            "INSERT INTO focal_references VALUES(?,?,?,?)",
            insert_rows,
        )
    connection.execute(
        "CREATE INDEX temp.idx_focal_reference "
        "ON focal_references(reference_id,publication_year)"
    )
    connection.execute("DROP TABLE IF EXISTS temp.cumulative_citations")
    connection.execute(
        "CREATE TEMP TABLE cumulative_citations AS "
        "SELECT reference_id, source_year, "
        "SUM(citation_count) OVER("
        "PARTITION BY reference_id ORDER BY source_year "
        "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
        ") AS cumulative_count FROM citation_counts"
    )
    connection.execute(
        "CREATE INDEX temp.idx_cumulative_reference_year "
        "ON cumulative_citations(reference_id,source_year)"
    )
    cursor = connection.execute(
        "SELECT f.paper_id, f.reference_order, "
        "COALESCE(("
        "SELECT c.cumulative_count FROM cumulative_citations c "
        "WHERE c.reference_id=f.reference_id "
        "AND c.source_year<f.publication_year "
        "ORDER BY c.source_year DESC LIMIT 1"
        "),0) AS prior_count "
        "FROM focal_references f "
        "ORDER BY f.paper_id, f.reference_order"
    )
    rows: List[Dict[str, Any]] = []
    current_paper = ""
    values: List[float] = []
    for paper_id, _, count in cursor:
        paper_id = str(paper_id)
        if current_paper and paper_id != current_paper:
            rows.append(
                {
                    "paper_id": current_paper,
                    "log_prior_reference_popularity_median": float(
                        np.median(values)
                    ),
                }
            )
            values = []
        current_paper = paper_id
        values.append(math.log1p(int(count)))
    if current_paper:
        rows.append(
            {
                "paper_id": current_paper,
                "log_prior_reference_popularity_median": float(
                    np.median(values)
                ),
            }
        )
    connection.close()
    measured = pd.DataFrame(rows)
    return papers[["paper_id"]].merge(
        measured,
        on="paper_id",
        how="left",
        validate="one_to_one",
    )


def build_v6_1_controls(
    papers: pd.DataFrame,
    paper_references: pd.DataFrame,
    reference_metadata: pd.DataFrame,
    v6_controls: pd.DataFrame,
    *,
    target_works_path: Path,
    openalex_metadata_path: Optional[Path],
    popularity_sqlite_path: Path,
) -> pd.DataFrame:
    """Build K0, K1, and K2 columns from frozen local views."""
    paper_frame, bibliographies = _normalize_bibliographies(
        papers, paper_references, reference_metadata
    )
    base = paper_frame[
        [
            "paper_id",
            "publication_year",
            "domain12",
            "openalex_primary_subfield",
            "venue_family",
        ]
    ].merge(
        v6_controls[
            [
                "paper_id",
                "log_reference_count",
                "reference_age_median",
                "reference_age_iqr",
            ]
        ],
        on="paper_id",
        how="left",
        validate="one_to_one",
    )
    base = base.merge(
        _title_word_counts(target_works_path),
        on="paper_id",
        how="left",
        validate="one_to_one",
    )
    if openalex_metadata_path and Path(openalex_metadata_path).is_file():
        metadata = pd.read_parquet(openalex_metadata_path)
        base = base.merge(
            build_k1_team_controls(metadata),
            on="paper_id",
            how="left",
            validate="one_to_one",
        )
        base = base.merge(
            _team_prior_output(metadata, paper_frame),
            on="paper_id",
            how="left",
            validate="one_to_one",
        )
        base["openalex_metadata_found"] = base["paper_id"].isin(
            set(metadata["paper_id"].astype(str))
        ).astype(int)
    else:
        for column in (
            "log_author_count",
            "log_institution_count",
            "log_country_count",
            "log_team_prior_nature_output_max",
        ):
            base[column] = np.nan
        base["openalex_metadata_found"] = 0
    base = base.merge(
        _reference_prior_popularity(
            paper_frame, bibliographies, popularity_sqlite_path
        ),
        on="paper_id",
        how="left",
        validate="one_to_one",
    )
    base["source_max_year"] = base["publication_year"].astype(int) - 1
    base["definition_version"] = MATERIALIZATION_VERSION_V6_1
    return base


def _copy_v6_artifacts(v6_dir: Path, output_dir: Path) -> None:
    for name in (
        "papers_common_all.parquet",
        "papers_primary_articles.parquet",
        "paper_references.parquet",
        "reference_metadata.parquet",
        "field_citation_events_aggregated.parquet",
        "targets_zero_inclusive.parquet",
        "cohort_membership.parquet",
        "opportunity_features.parquet",
    ):
        source = Path(v6_dir) / name
        target = Path(output_dir) / name
        if target.exists():
            continue
        try:
            target.hardlink_to(source.resolve())
        except OSError:
            shutil.copy2(source, target)
    for name in (
        "expanded_dataset_contract.json",
        "materialized_data_quality_report.json",
    ):
        source = Path(v6_dir) / name
        target = Path(output_dir) / name
        if not source.is_file() or target.exists():
            continue
        shutil.copy2(source, target)


def materialize_reference_overlap_extension(
    *,
    project_root: Path,
    v6_dataset_dir: Path,
    output_dir: Path,
    nature_v5_root: Optional[Path] = None,
    resume: bool = True,
) -> Mapping[str, Any]:
    """Add the source-faithful overlap candidate without building controls."""
    project_root = Path(project_root).resolve()
    v6_root = Path(v6_dataset_dir).resolve()
    root = Path(output_dir).resolve()
    source_root = (
        Path(nature_v5_root).resolve()
        if nature_v5_root is not None
        else (
            project_root
            / "outputs"
            / "common"
            / "new"
            / "data"
            / "nature_portfolio_v5"
        ).resolve()
    )
    root.mkdir(parents=True, exist_ok=True)
    _copy_v6_artifacts(v6_root, root)
    candidate_path = root / "innovation_candidate_features.parquet"
    if not candidate_path.is_file():
        raise ValueError(
            "base candidate features must be materialized before overlap"
        )
    papers = pd.read_parquet(root / "papers_primary_articles.parquet")
    references = pd.read_parquet(root / "paper_references.parquet")
    metadata = pd.read_parquet(root / "reference_metadata.parquet")
    context = materialize_reference_overlap_context(
        source_root / "nature_reference_edges.csv",
        source_root / "nature_target_works.csv",
        None,
        metadata,
        root,
        resume=resume,
    )
    candidates = pd.read_parquet(candidate_path)
    manifest_path = root / "reference_overlap_extension_manifest.json"
    prior_extension: Mapping[str, Any] = {}
    if manifest_path.is_file():
        prior_extension = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
    needs_recompute = (
        not resume
        or "reference_overlap_novelty_t0" not in candidates
        or prior_extension.get("context_artifact_id")
        != context["artifact_id"]
        or prior_extension.get("materialization_version")
        != MATERIALIZATION_VERSION_V6_1
        or prior_extension.get("reference_window_years") != "all_prior"
        or prior_extension.get("cociting_window_years") != "all_prior"
    )
    if needs_recompute:
        overlap = build_reference_overlap_features(
            papers,
            references,
            metadata,
            pd.read_parquet(root / "historical_paper_references.parquet"),
            reference_window_years=(
                REFERENCE_OVERLAP_REFERENCE_WINDOW_YEARS
            ),
            cociting_window_years=REFERENCE_OVERLAP_COCITING_WINDOW_YEARS,
        )
        candidates = candidates.drop(
            columns=[
                "reference_overlap_novelty_t0",
                "reference_overlap_comparison_count",
                "reference_overlap_reference_count",
            ],
            errors="ignore",
        )
        candidates = candidates.merge(
            overlap, on="paper_id", how="left", validate="one_to_one"
        )
        candidates["definition_version"] = MATERIALIZATION_VERSION_V6_1
        candidates.to_parquet(candidate_path, index=False)
    identity = {
        "artifact_kind": "aspr_v6_1_reference_overlap_extension",
        "materialization_version": MATERIALIZATION_VERSION_V6_1,
        "context_artifact_id": context["artifact_id"],
        "candidate_features_path": str(candidate_path),
        "candidate_features_sha256": sha256_file(candidate_path),
        "n_papers": int(len(candidates)),
        "formula": (
            "1-mean Jaccard overlap over prior same-primary-field papers"
        ),
        "reference_window_years": "all_prior",
        "cociting_window_years": "all_prior",
        "window_variant_source": (
            "Matsumoto et al. (2021), published all-reference/"
            "all-co-citing variant"
        ),
        "network_used": False,
    }
    manifest = {**identity, "artifact_id": _canonical_hash(identity)}
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return manifest


def materialize_v6_1_dataset(
    *,
    project_root: Path,
    v6_dataset_dir: Path,
    output_dir: Path,
    openalex_metadata_path: Optional[Path] = None,
    nature_v5_root: Optional[Path] = None,
    resume: bool = True,
) -> Mapping[str, Any]:
    """Create the independent v6.1 dataset without altering v6 artifacts."""
    project_root = Path(project_root).resolve()
    v6_root = Path(v6_dataset_dir).resolve()
    root = Path(output_dir).resolve()
    source_root = (
        Path(nature_v5_root).resolve()
        if nature_v5_root is not None
        else (
            project_root
            / "outputs"
            / "common"
            / "new"
            / "data"
            / "nature_portfolio_v5"
        ).resolve()
    )
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "v6_1_materialization_manifest.json"
    if resume and manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    _copy_v6_artifacts(v6_root, root)
    papers = pd.read_parquet(root / "papers_primary_articles.parquet")
    references = pd.read_parquet(root / "paper_references.parquet")
    metadata = pd.read_parquet(root / "reference_metadata.parquet")
    field_events = pd.read_parquet(
        root / "field_citation_events_aggregated.parquet"
    )
    v6_features = pd.read_parquet(v6_root / "innovation_features.parquet")
    v6_controls = pd.read_parquet(v6_root / "control_features.parquet")
    context_manifest = materialize_historical_source_context(
        source_root / "nature_reference_edges.csv",
        metadata,
        root,
        resume=resume,
    )
    candidate_path = root / "innovation_candidate_features.parquet"
    if resume and candidate_path.is_file():
        candidate_features = pd.read_parquet(candidate_path)
    else:
        candidate_features = build_candidate_innovation_features(
            papers,
            references,
            metadata,
            field_events,
            pd.read_parquet(root / "historical_paper_sources.parquet"),
            pd.read_parquet(root / "source_field_citation_events.parquet"),
        )
        b0 = v6_features[["paper_id", *B0_INNOVATION_FEATURES]].rename(
            columns={
                name: f"b0_{name}" for name in B0_INNOVATION_FEATURES
            }
        )
        candidate_features = candidate_features.merge(
            b0, on="paper_id", how="left", validate="one_to_one"
        )
        candidate_features.to_parquet(candidate_path, index=False)
    overlap_manifest = materialize_reference_overlap_extension(
        project_root=project_root,
        v6_dataset_dir=v6_root,
        output_dir=root,
        nature_v5_root=source_root,
        resume=resume,
    )
    candidate_features = pd.read_parquet(candidate_path)
    controls_path = root / "control_features_v6_1.parquet"
    controls = build_v6_1_controls(
        papers,
        references,
        metadata,
        v6_controls,
        target_works_path=source_root / "nature_target_works.csv",
        openalex_metadata_path=openalex_metadata_path,
        popularity_sqlite_path=root / "reference_prior_popularity.sqlite",
    )
    controls.to_parquet(controls_path, index=False)
    metadata_manifest_path = root / "target_openalex_metadata_manifest.json"
    metadata_manifest: Mapping[str, Any] = {}
    if metadata_manifest_path.is_file():
        metadata_manifest = json.loads(
            metadata_manifest_path.read_text(encoding="utf-8")
        )
    manifest = {
        "artifact_kind": "aspr_v6_1_local_dataset",
        "materialization_version": MATERIALIZATION_VERSION_V6_1,
        "network_used": False,
        "network_used_during_materialization": False,
        "upstream_metadata_network_used": bool(
            metadata_manifest.get("network_used_during_data_build", False)
        ),
        "target_metadata_manifest": str(metadata_manifest_path)
        if metadata_manifest_path.is_file()
        else None,
        "v6_dataset_unchanged": True,
        "v6_dataset_dir": str(v6_root),
        "historical_source_context_artifact_id": context_manifest[
            "artifact_id"
        ],
        "reference_overlap_extension_artifact_id": overlap_manifest[
            "artifact_id"
        ],
        "n_papers": int(len(papers)),
        "outputs": {
            "innovation_candidate_features": {
                "path": str(candidate_path),
                "sha256": sha256_file(candidate_path),
            },
            "control_features_v6_1": {
                "path": str(controls_path),
                "sha256": sha256_file(controls_path),
            },
        },
        "coverage": {
            name: float(
                pd.to_numeric(candidate_features[name], errors="coerce")
                .notna()
                .mean()
            )
            for name in PROVISIONAL_CORE8
        },
        "k1_metadata_coverage": {
            name: float(
                pd.to_numeric(controls[name], errors="coerce").notna().mean()
            )
            for name in (
                "title_word_count",
                "log_author_count",
                "log_institution_count",
                "log_country_count",
            )
        },
    }
    manifest["artifact_id"] = _canonical_hash(manifest)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return manifest


def strict_monotone_audit(features: pd.DataFrame) -> pd.DataFrame:
    """Report registered algebraic/monotone redundancy relationships."""
    pairs = (
        ("field_variety", "field_relative_variety"),
        ("field_variety", "field_hill_q0"),
        ("field_shannon_entropy", "field_hill_q1"),
        ("field_gini_simpson", "field_hhi"),
        ("field_gini_simpson", "field_hill_q2"),
        ("rao_stirling_integration", "rao_true_diversity_q2"),
    )
    rows = []
    for left, right in pairs:
        paired = features[[left, right]].dropna()
        rows.append(
            {
                "left": left,
                "right": right,
                "n_paired": len(paired),
                "spearman": safe_spearman(paired[left], paired[right]),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "B0_INNOVATION_FEATURES",
    "FIELD_TAXONOMY_SIZE",
    "MATERIALIZATION_VERSION_V6_1",
    "PROVISIONAL_CORE8",
    "build_candidate_innovation_features",
    "build_reference_overlap_features",
    "REFERENCE_OVERLAP_COCITING_WINDOW_YEARS",
    "REFERENCE_OVERLAP_REFERENCE_WINDOW_YEARS",
    "build_v6_1_controls",
    "materialize_historical_source_context",
    "materialize_reference_overlap_context",
    "materialize_reference_overlap_extension",
    "materialize_v6_1_dataset",
    "strict_monotone_audit",
]

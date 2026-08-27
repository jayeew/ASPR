"""Real source-to-runtime replay for the frozen Fig.3 Full-text-16 matrix."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations, pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .active_dataset import load_active_dataset
from .feature_materializer_v6 import annual_field_distances
from .t0_runtime_v3 import (
    ContextSnapshot,
    ReferenceT0,
    TargetT0Record,
    _title_words,
    coerce_fulltext16_storage_schema,
    materialize_fulltext16,
)


@dataclass(frozen=True)
class RuntimeReplayPaths:
    papers: Path
    target_metadata: Path
    reference_metadata: Path
    field_citation_events: Path
    target_works: Path

    @classmethod
    def from_active_dataset(cls, project_root: Path) -> RuntimeReplayPaths:
        active = load_active_dataset(project_root)
        feature_root = Path(active["feature_dataset_dir"])
        return cls(
            papers=feature_root / "papers_primary_articles.parquet",
            target_metadata=feature_root / "target_openalex_metadata.parquet",
            reference_metadata=feature_root / "reference_metadata.parquet",
            field_citation_events=feature_root
            / "field_citation_events_aggregated.parquet",
            target_works=Path(active["target_works"]),
        )

    def values(self) -> tuple[Path, ...]:
        return (
            self.papers,
            self.target_metadata,
            self.reference_metadata,
            self.field_citation_events,
            self.target_works,
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _as_strings(value: object) -> tuple[str, ...]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    try:
        items = iter(value)  # type: ignore[call-overload]
    except TypeError:
        text = str(value)
        return (text,) if text else ()
    return tuple(str(item) for item in items if str(item))


def _country_codes(count: object) -> tuple[str, ...]:
    if count is None or pd.isna(count):
        return ()
    numeric = float(str(count))
    if numeric <= 0:
        return ()
    return tuple(f"COUNTRY_{index}" for index in range(int(numeric)))


def _update_title_and_author_context(
    context: ContextSnapshot,
    rows: Iterable[Any],
) -> None:
    for row in rows:
        context.seen_title_bigrams.update(pairwise(_title_words(str(row.title))))
        authors = sorted(set(_as_strings(row.openalex_author_ids)))[:100]
        for left, right in combinations(authors, 2):
            key = (left, right) if left < right else (right, left)
            context.prior_author_adjacency.setdefault(left, set()).add(right)
            context.prior_author_adjacency.setdefault(right, set()).add(left)
            context.prior_coauthor_weights[key] = (
                context.prior_coauthor_weights.get(key, 0) + 1
            )


def _add_primary_work_to_coupling_context(
    context: ContextSnapshot,
    work_id: str,
    publication_year: int,
    reference_ids: object,
    reference_lookup: Mapping[str, tuple[int | None, str | None]],
) -> None:
    """Mirror the frozen v6 opportunity graph's strictly-prior postings."""
    for reference_id in set(_as_strings(reference_ids)):
        reference_year = reference_lookup.get(reference_id, (None, None))[0]
        if reference_year is None:
            continue
        if int(reference_year) >= int(publication_year):
            continue
        context.bibliographic_coupling_index.setdefault(reference_id, set()).add(
            work_id
        )


def _reference_lookup(path: Path) -> dict[str, tuple[int | None, str | None]]:
    frame = pd.read_parquet(
        path, columns=["reference_id", "reference_year", "field_id"]
    ).drop_duplicates("reference_id", keep="last")
    output: dict[str, tuple[int | None, str | None]] = {}
    for row in frame.itertuples(index=False):
        year = int(row.reference_year) if pd.notna(row.reference_year) else None
        field = (
            str(row.field_id)
            if pd.notna(row.field_id) and str(row.field_id).strip()
            else None
        )
        output[str(row.reference_id)] = (year, field)
    return output


def _target_references(
    reference_ids: object,
    lookup: Mapping[str, tuple[int | None, str | None]],
) -> tuple[ReferenceT0, ...]:
    output: list[ReferenceT0] = []
    for reference_id in _as_strings(reference_ids):
        year, field = lookup.get(reference_id, (None, None))
        output.append(ReferenceT0(reference_id, year, field))
    return tuple(output)


def _load_targets(
    paths: RuntimeReplayPaths,
    official_ids: set[str],
) -> pd.DataFrame:
    papers = pd.read_parquet(
        paths.papers,
        columns=[
            "paper_id",
            "publication_year",
            "source_id",
            "referenced_works",
        ],
    )
    papers["paper_id"] = papers["paper_id"].astype(str)
    papers = papers[papers["paper_id"].isin(official_ids)].copy()
    work_parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        paths.target_works,
        usecols=["id", "title"],
        dtype={"id": "string", "title": "string"},
        chunksize=100_000,
    ):
        selected = chunk[chunk["id"].isin(official_ids)]
        if not selected.empty:
            work_parts.append(selected)
    if not work_parts:
        raise ValueError("runtime replay target titles are unavailable")
    works = pd.concat(work_parts, ignore_index=True).rename(columns={"id": "paper_id"})
    metadata = pd.read_parquet(
        paths.target_metadata,
        columns=[
            "paper_id",
            "openalex_author_count",
            "openalex_author_ids",
            "openalex_country_count",
        ],
    )
    metadata["metadata_observed"] = True
    targets = papers.merge(works, on="paper_id", how="left", validate="one_to_one")
    targets = targets.merge(
        metadata,
        on="paper_id",
        how="left",
        validate="one_to_one",
    )
    targets["metadata_observed"] = targets["metadata_observed"].fillna(False)
    if len(targets) != len(official_ids) or targets["title"].isna().any():
        raise ValueError(
            "runtime replay target inputs do not cover the official cohort"
        )
    return targets.sort_values(
        ["publication_year", "paper_id"], kind="stable"
    ).reset_index(drop=True)


def build_runtime_fulltext16_matrix(
    *,
    project_root: Path,
    official_matrix_path: Path,
    paths: RuntimeReplayPaths | None = None,
    years: Sequence[int] | None = None,
    limit_per_year: int | None = None,
) -> tuple[pd.DataFrame, ContextSnapshot, dict[str, Any]]:
    """Recompute Full-text-16 without reading official feature values.

    The official matrix supplies only the exact cohort paper IDs.  Every feature
    value is rebuilt from frozen publication-time source views.
    """
    source_paths = paths or RuntimeReplayPaths.from_active_dataset(project_root)
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    official_ids = set(
        pd.read_parquet(official_matrix_path, columns=["paper_id"])["paper_id"].astype(
            str
        )
    )
    targets = _load_targets(source_paths, official_ids)
    focal_years = sorted(int(value) for value in targets["publication_year"].unique())
    selected_years = {int(value) for value in years} if years else set(focal_years)
    reference_lookup = _reference_lookup(source_paths.reference_metadata)
    online_context_year = max(focal_years) + 1
    distance_by_year = annual_field_distances(
        pd.read_parquet(source_paths.field_citation_events),
        [*focal_years, online_context_year],
    )
    context = ContextSnapshot(source_max_year=min(focal_years) - 1)
    output: list[dict[str, object]] = []
    rows_by_year = {
        int(year): frame
        for year, frame in targets.groupby("publication_year", sort=True)
    }
    for year in focal_years:
        context.source_max_year = year - 1
        context.field_distances = dict(distance_by_year.get(year, {}))
        year_targets = rows_by_year[year]
        if year in selected_years:
            materialized = year_targets
            if limit_per_year is not None:
                materialized = materialized.head(int(limit_per_year))
            for row in materialized.itertuples(index=False):
                author_count = (
                    int(row.openalex_author_count)
                    if pd.notna(row.openalex_author_count)
                    and float(row.openalex_author_count) > 0
                    else None
                )
                target = TargetT0Record(
                    paper_id=str(row.paper_id),
                    publication_year=year,
                    title=str(row.title),
                    author_ids=_as_strings(row.openalex_author_ids),
                    author_count=author_count,
                    country_codes=_country_codes(row.openalex_country_count),
                    metadata_observed=bool(row.metadata_observed),
                    source_id=(
                        str(row.source_id)
                        if pd.notna(row.source_id) and str(row.source_id)
                        else None
                    ),
                    references=_target_references(
                        row.referenced_works, reference_lookup
                    ),
                )
                output.append(
                    {
                        "paper_id": target.paper_id,
                        **materialize_fulltext16(target, context),
                    }
                )
        # The frozen batch opportunity graph scores a complete publication year
        # before adding that year's focal Nature papers to the reference postings.
        # This is deliberately independent from the v6.1 historical-reference
        # context used by a different reference-overlap novelty feature.
        for row in year_targets.itertuples(index=False):
            _add_primary_work_to_coupling_context(
                context,
                str(row.paper_id),
                year,
                row.referenced_works,
                reference_lookup,
            )
        _update_title_and_author_context(context, year_targets.itertuples(index=False))
    context.source_max_year = max(focal_years)
    context.field_distances = dict(distance_by_year.get(online_context_year, {}))
    runtime = coerce_fulltext16_storage_schema(pd.DataFrame(output))
    manifest = {
        "contract": "aspr_fulltext16_runtime_replay_inputs_v1",
        "official_cohort_count": len(official_ids),
        "runtime_row_count": len(runtime),
        "selected_years": sorted(selected_years),
        "online_context_year": online_context_year,
        "source_hashes": {
            str(path.resolve()): _sha256_file(path) for path in source_paths.values()
        },
    }
    encoded = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    manifest["manifest_sha256"] = "sha256:" + hashlib.sha256(encoded).hexdigest()
    return runtime, context, manifest


def build_runtime_context_for_year(
    *,
    project_root: Path,
    official_matrix_path: Path,
    target_year: int,
    paths: RuntimeReplayPaths | None = None,
) -> tuple[ContextSnapshot, dict[str, Any]]:
    """Replay frozen source views only up to the year before a target."""
    source_paths = paths or RuntimeReplayPaths.from_active_dataset(project_root)
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    official_ids = set(
        pd.read_parquet(official_matrix_path, columns=["paper_id"])["paper_id"].astype(
            str
        )
    )
    targets = _load_targets(source_paths, official_ids)
    eligible = targets[targets["publication_year"].astype(int) < int(target_year)]
    if eligible.empty:
        raise ValueError("historical runtime context has no strictly prior papers")
    reference_lookup = _reference_lookup(source_paths.reference_metadata)
    distances = annual_field_distances(
        pd.read_parquet(source_paths.field_citation_events), [int(target_year)]
    )
    context = ContextSnapshot(
        source_max_year=int(eligible["publication_year"].min()) - 1
    )
    for year, frame in eligible.groupby("publication_year", sort=True):
        numeric_year = int(year)
        for row in frame.itertuples(index=False):
            _add_primary_work_to_coupling_context(
                context,
                str(row.paper_id),
                numeric_year,
                row.referenced_works,
                reference_lookup,
            )
        _update_title_and_author_context(context, frame.itertuples(index=False))
    context.source_max_year = int(eligible["publication_year"].astype(int).max())
    context.field_distances = dict(distances.get(int(target_year), {}))
    source_hashes = {
        str(path.resolve()): _sha256_file(path) for path in source_paths.values()
    }
    manifest = {
        "target_year": int(target_year),
        "source_max_year": context.source_max_year,
        "source_hashes": source_hashes,
        "builder": "runtime_context_before_target_year_v1",
    }
    return context, manifest


__all__ = [
    "RuntimeReplayPaths",
    "build_runtime_context_for_year",
    "build_runtime_fulltext16_matrix",
]

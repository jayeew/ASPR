"""Annual strictly-prior materialization of feasible v6 reference features."""

from __future__ import annotations

import itertools
import json
import math
from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gear.corpus import normalize_openalex_id

from .features_v6 import (
    canonical_pair,
    cosine_distance_profiles,
    field_disparity_mean,
    field_pielou_evenness,
    field_variety,
    first_time_source_pair_share,
    marginal_pair_z_scores,
    novelty_u,
    rao_stirling_integration,
    uzzi_atypicality_p10,
    uzzi_conventionality_median,
)

FEATURE_VIEW_VERSION = "aspr-reference-features-v6-1"


def build_field_citation_events(work_view: pd.DataFrame) -> pd.DataFrame:
    """Map local work citations to year and source/target OpenAlex fields."""
    required = {"work_id", "publication_year", "field_id", "referenced_works"}
    missing = sorted(required - set(work_view.columns))
    if missing:
        raise ValueError(f"work_view is missing columns: {missing}")
    metadata = work_view[["work_id", "field_id"]].copy()
    metadata["work_id"] = metadata["work_id"].map(normalize_openalex_id)
    metadata["field_id"] = metadata["field_id"].fillna("").astype(str)
    field_lookup = metadata.drop_duplicates("work_id").set_index("work_id")["field_id"]
    events = work_view[
        ["work_id", "publication_year", "field_id", "referenced_works"]
    ].copy()
    events["source_work_id"] = events["work_id"].map(normalize_openalex_id)
    events["source_year"] = pd.to_numeric(events["publication_year"], errors="coerce")
    events["source_field_id"] = events["field_id"].fillna("").astype(str)
    events = events.explode("referenced_works")
    events["target_work_id"] = events["referenced_works"].map(normalize_openalex_id)
    events["target_field_id"] = events["target_work_id"].map(field_lookup).fillna("")
    events = events[
        events["source_year"].notna()
        & events["source_field_id"].ne("")
        & events["target_field_id"].ne("")
        & events["source_work_id"].ne(events["target_work_id"])
    ].copy()
    events["source_year"] = events["source_year"].astype(int)
    return events[
        [
            "source_work_id",
            "target_work_id",
            "source_year",
            "source_field_id",
            "target_field_id",
        ]
    ].drop_duplicates()


def annual_field_distances(
    events: pd.DataFrame,
    focal_years: Sequence[int],
    *,
    window_years: int = 5,
) -> dict[int, Mapping[tuple[Any, Any], float]]:
    """Create frozen prior-window cosine distances for each focal year."""
    if window_years <= 0:
        raise ValueError("window_years must be positive")
    required = {"source_year", "source_field_id", "target_field_id"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"field citation events are missing columns: {missing}")
    frame = events.copy()
    frame["source_year"] = pd.to_numeric(frame["source_year"], errors="coerce")
    frame = frame[frame["source_year"].notna()].copy()
    frame["source_year"] = frame["source_year"].astype(int)
    group_columns = ["source_year", "source_field_id", "target_field_id"]
    if "citation_count" in frame:
        frame["citation_count"] = pd.to_numeric(
            frame["citation_count"], errors="coerce"
        ).fillna(0.0)
        aggregated = (
            frame.groupby(group_columns, observed=True)["citation_count"]
            .sum()
            .reset_index()
        )
    else:
        aggregated = (
            frame.groupby(group_columns, observed=True)
            .size()
            .rename("citation_count")
            .reset_index()
        )
    outputs: dict[int, Mapping[tuple[Any, Any], float]] = {}
    for focal_year in sorted({int(value) for value in focal_years}):
        selected = aggregated[
            aggregated["source_year"].between(
                focal_year - window_years, focal_year - 1, inclusive="both"
            )
        ]
        if selected.empty:
            outputs[focal_year] = {}
            continue
        matrix = selected.pivot_table(
            index="source_field_id",
            columns="target_field_id",
            values="citation_count",
            aggfunc="sum",
            fill_value=0,
        ).astype(float)
        profiles: dict[Hashable, Sequence[float]] = {
            str(field): matrix.loc[field].to_numpy(dtype=float).tolist()
            for field in matrix.index
        }
        outputs[focal_year] = cosine_distance_profiles(profiles)
    return outputs


def aggregate_field_citation_events_from_edges(
    reference_edges_path: Path,
    reference_metadata: pd.DataFrame,
    *,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Stream v5 closure edges into compact year/field citation counts."""
    required = {"reference_id", "reference_year", "field_id"}
    missing = sorted(required - set(reference_metadata))
    if missing:
        raise ValueError(f"reference metadata is missing columns: {missing}")
    metadata = reference_metadata[["reference_id", "reference_year", "field_id"]].copy()
    metadata["reference_id"] = metadata["reference_id"].map(normalize_openalex_id)
    metadata["reference_year"] = pd.to_numeric(
        metadata["reference_year"], errors="coerce"
    )
    metadata["field_id"] = metadata["field_id"].fillna("").astype(str)
    metadata = metadata[
        metadata["reference_id"].ne("")
        & metadata["reference_year"].notna()
        & metadata["field_id"].ne("")
    ].drop_duplicates("reference_id", keep="last")
    year_lookup = metadata.set_index("reference_id")["reference_year"].to_dict()
    field_lookup = metadata.set_index("reference_id")["field_id"].to_dict()
    counts: Counter[tuple[int, str, str]] = Counter()
    path = Path(reference_edges_path)
    for chunk in pd.read_csv(
        path,
        usecols=["source", "target"],
        chunksize=int(chunksize),
        dtype={"source": "string", "target": "string"},
    ):
        source_ids = chunk["source"].map(normalize_openalex_id)
        target_ids = chunk["target"].map(normalize_openalex_id)
        years = source_ids.map(year_lookup)
        source_fields = source_ids.map(field_lookup)
        target_fields = target_ids.map(field_lookup)
        valid = (
            years.notna()
            & source_fields.notna()
            & target_fields.notna()
            & source_fields.ne("")
            & target_fields.ne("")
        )
        compact = pd.DataFrame(
            {
                "source_year": years[valid].astype(int),
                "source_field_id": source_fields[valid].astype(str),
                "target_field_id": target_fields[valid].astype(str),
            }
        )
        grouped = compact.value_counts(sort=False)
        counts.update(
            {
                (int(key[0]), str(key[1]), str(key[2])): int(value)
                for key, value in grouped.items()
            }
        )
    return pd.DataFrame(
        [
            {
                "source_year": key[0],
                "source_field_id": key[1],
                "target_field_id": key[2],
                "citation_count": value,
            }
            for key, value in sorted(counts.items())
        ]
    )


def _normalize_inputs(
    papers: pd.DataFrame,
    paper_references: pd.DataFrame,
    work_view: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, list[Any]]]]:
    required_papers = {"paper_id", "publication_year"}
    required_references = {"paper_id", "reference_id"}
    required_metadata = {
        "work_id",
        "publication_year",
        "source_id",
        "field_id",
    }
    for name, required, frame in (
        ("papers", required_papers, papers),
        ("paper_references", required_references, paper_references),
        ("work_view", required_metadata, work_view),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} is missing columns: {missing}")

    paper_frame = papers.copy()
    paper_frame["paper_id"] = paper_frame["paper_id"].map(normalize_openalex_id)
    paper_frame["publication_year"] = pd.to_numeric(
        paper_frame["publication_year"], errors="coerce"
    )
    paper_frame = paper_frame[
        paper_frame["paper_id"].ne("") & paper_frame["publication_year"].notna()
    ].drop_duplicates("paper_id")
    paper_frame["publication_year"] = paper_frame["publication_year"].astype(int)

    metadata = work_view[
        ["work_id", "publication_year", "source_id", "field_id"]
    ].copy()
    metadata["reference_id"] = metadata["work_id"].map(normalize_openalex_id)
    metadata["reference_year"] = pd.to_numeric(
        metadata["publication_year"], errors="coerce"
    )
    metadata["source_id"] = metadata["source_id"].fillna("").astype(str)
    metadata["field_id"] = metadata["field_id"].fillna("").astype(str)
    metadata = metadata.drop_duplicates("reference_id")

    bibliography = paper_references[["paper_id", "reference_id"]].copy()
    bibliography["paper_id"] = bibliography["paper_id"].map(normalize_openalex_id)
    bibliography["reference_id"] = bibliography["reference_id"].map(
        normalize_openalex_id
    )
    bibliography = bibliography.drop_duplicates().merge(
        metadata[["reference_id", "reference_year", "source_id", "field_id"]],
        on="reference_id",
        how="left",
        validate="many_to_one",
    )
    bibliography = bibliography.merge(
        paper_frame[["paper_id", "publication_year"]],
        on="paper_id",
        how="inner",
        validate="many_to_one",
    )
    bibliography["strictly_prior_reference"] = bibliography[
        "reference_year"
    ].notna() & bibliography["reference_year"].lt(bibliography["publication_year"])

    grouped: dict[str, dict[str, list[Any]]] = {}
    for paper_id, group in bibliography.groupby("paper_id", sort=False):
        prior = group[group["strictly_prior_reference"]]
        grouped[str(paper_id)] = {
            "declared_reference_ids": group["reference_id"].tolist(),
            "valid_reference_ids": prior["reference_id"].tolist(),
            "reference_years": prior["reference_year"].astype(int).tolist(),
            "source_ids": [value for value in prior["source_id"].astype(str) if value],
            "field_ids": [value for value in prior["field_id"].astype(str) if value],
        }
    return paper_frame, grouped


def _paper_source_pairs(source_ids: Sequence[str]) -> set[tuple[Any, Any]]:
    unique_sources = sorted({value for value in source_ids if value})
    return {
        canonical_pair(left, right)
        for left, right in itertools.combinations(unique_sources, 2)
    }


def _update_historical_source_counts(
    source_ids: Sequence[str],
    source_counts: Counter[Any],
    pair_counts: Counter[tuple[Any, Any]],
) -> None:
    source_counts.update(set(source_ids))
    pair_counts.update(_paper_source_pairs(source_ids))


def build_v6_reference_feature_table(
    papers: pd.DataFrame,
    paper_references: pd.DataFrame,
    work_view: pd.DataFrame,
    *,
    field_citation_events: pd.DataFrame | None = None,
    field_profile_window_years: int = 5,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Build annual T0 reference features from frozen local tables.

    Historical source-pair counts use prior Nature Portfolio papers only.  The
    registry therefore retains the recombination metrics as adaptations.  No
    current-year paper is added until all papers in that year are scored.
    """
    paper_frame, bibliographies = _normalize_inputs(papers, paper_references, work_view)
    focal_years = sorted(paper_frame["publication_year"].unique().tolist())
    if field_citation_events is None:
        field_citation_events = build_field_citation_events(work_view)
    distance_by_year = annual_field_distances(
        field_citation_events,
        focal_years,
        window_years=field_profile_window_years,
    )
    source_counts: Counter[Any] = Counter()
    pair_counts: Counter[tuple[Any, Any]] = Counter()
    n_historical_papers = 0
    rows: list[dict[str, Any]] = []
    for year, year_papers in paper_frame.groupby("publication_year", sort=True):
        year = int(year)
        distances = distance_by_year.get(year, {})
        for paper in year_papers.to_dict("records"):
            paper_id = str(paper["paper_id"])
            values = bibliographies.get(
                paper_id,
                {
                    "declared_reference_ids": [],
                    "valid_reference_ids": [],
                    "reference_years": [],
                    "source_ids": [],
                    "field_ids": [],
                },
            )
            declared = values["declared_reference_ids"]
            valid = values["valid_reference_ids"]
            sources = values["source_ids"]
            fields = values["field_ids"]
            ages = [year - int(value) for value in values["reference_years"]]
            pair_z_scores = marginal_pair_z_scores(
                sources, pair_counts, source_counts, n_historical_papers
            )
            quality_flags: list[str] = []
            if not sources:
                quality_flags.append("no_mapped_reference_sources")
            if not fields:
                quality_flags.append("no_mapped_reference_fields")
            if not distances:
                quality_flags.append("no_prior_field_profile")
            if len(pair_z_scores) < 20:
                quality_flags.append("fewer_than_20_valid_uzzi_pairs")
            try:
                disparity = field_disparity_mean(fields, distances)
                rao = rao_stirling_integration(fields, distances)
            except KeyError:
                disparity = float("nan")
                rao = float("nan")
                quality_flags.append("incomplete_field_distance_coverage")
            rows.append(
                {
                    "paper_id": paper_id,
                    "publication_year": year,
                    "domain12": paper.get("domain12", "unmapped"),
                    "novelty_u_t0_source": novelty_u(
                        sources,
                        pair_counts,
                        source_counts,
                        n_historical_papers,
                    ),
                    "first_time_source_pair_share": first_time_source_pair_share(
                        sources, pair_counts
                    ),
                    "first_time_source_pair_distance_mean": np.nan,
                    "uzzi_atypicality_p10_t0": uzzi_atypicality_p10(pair_z_scores),
                    "uzzi_conventionality_median_t0": (
                        uzzi_conventionality_median(pair_z_scores)
                    ),
                    "field_variety": field_variety(fields),
                    "field_pielou_evenness": field_pielou_evenness(fields),
                    "field_disparity_cosine_mean": disparity,
                    "rao_stirling_integration": rao,
                    "log_reference_count": float(math.log1p(len(declared))),
                    "reference_age_median": (
                        float(np.median(ages)) if ages else np.nan
                    ),
                    "reference_age_iqr": (
                        float(np.quantile(ages, 0.75) - np.quantile(ages, 0.25))
                        if ages
                        else np.nan
                    ),
                    "valid_reference_count": len(valid),
                    "reference_metadata_coverage": len(valid) / max(1, len(declared)),
                    "source_mapping_coverage": len(sources) / max(1, len(valid)),
                    "field_mapping_coverage": len(fields) / max(1, len(valid)),
                    "valid_source_pair_count": len(_paper_source_pairs(sources)),
                    "n_historical_source_papers": n_historical_papers,
                    "source_max_year": year - 1,
                    "historical_pair_scope": "prior_nature_portfolio_targets_only",
                    "definition_version": FEATURE_VIEW_VERSION,
                    "quality_flags": json.dumps(
                        sorted(set(quality_flags)), ensure_ascii=False
                    ),
                }
            )
        # Same-year papers become historical only after every same-year focal
        # paper has been scored, which enforces source_max_year < publication.
        for paper_id in year_papers["paper_id"].astype(str):
            sources = bibliographies.get(paper_id, {}).get("source_ids", [])
            if sources:
                _update_historical_source_counts(sources, source_counts, pair_counts)
        n_historical_papers += len(year_papers)

    output = pd.DataFrame(rows)
    violations = output[
        pd.to_numeric(output["source_max_year"], errors="coerce")
        >= pd.to_numeric(output["publication_year"], errors="coerce")
    ]
    if not violations.empty:
        raise ValueError("v6 feature materialization contains temporal leakage")
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        output.to_parquet(path, index=False)
    return output


__all__ = [
    "FEATURE_VIEW_VERSION",
    "aggregate_field_citation_events_from_edges",
    "annual_field_distances",
    "build_field_citation_events",
    "build_v6_reference_feature_table",
]

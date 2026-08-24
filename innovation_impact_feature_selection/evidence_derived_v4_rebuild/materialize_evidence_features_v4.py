"""Materialize literature-derived reference-discipline features for v4.

This is a standalone, outcome-blind implementation.  It intentionally reads
the frozen raw input tables rather than accepting historical feature values as
evidence for the rebuilt v4 definitions.  All distance profiles for a paper
published in year ``y`` use only field-citation events from ``y-5`` through
``y-1``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_ROOT = (
    ROOT.parent.parent / "data" / "knowledge_corpus"
    / "nature_multihorizon_v6_1_uncapped_v2"
)
DEFAULT_OUTPUT = ROOT / "outputs" / "evidence_features_v4.parquet"
DEFAULT_REPORT = ROOT / "outputs" / "evidence_features_v4_report.json"
WINDOW_YEARS = 5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_pair(left: str, right: str) -> Tuple[str, str]:
    return (left, right) if left <= right else (right, left)


def _profile_distances(
    events: pd.DataFrame, years: Sequence[int]
) -> Dict[int, Dict[Tuple[str, str], float]]:
    """Build strictly-prior five-year cosine distances for each focal year."""
    outputs: Dict[int, Dict[Tuple[str, str], float]] = {}
    grouped = events.groupby(
        ["source_year", "source_field_id", "target_field_id"],
        observed=True,
    )["citation_count"].sum().reset_index()
    for year in sorted({int(value) for value in years}):
        prior = grouped.loc[
            grouped["source_year"].between(year - WINDOW_YEARS, year - 1)
        ]
        if prior.empty:
            outputs[year] = {}
            continue
        matrix = prior.pivot_table(
            index="source_field_id",
            columns="target_field_id",
            values="citation_count",
            aggfunc="sum",
            fill_value=0.0,
        ).astype(float)
        vectors = matrix.to_numpy(dtype=float)
        norms = np.linalg.norm(vectors, axis=1)
        distances: Dict[Tuple[str, str], float] = {}
        labels = [str(value) for value in matrix.index]
        for i, left in enumerate(labels):
            if norms[i] <= 0:
                continue
            for j in range(i + 1, len(labels)):
                if norms[j] <= 0:
                    continue
                similarity = float(np.dot(vectors[i], vectors[j]) / (norms[i] * norms[j]))
                distances[(left, labels[j])] = float(1.0 - similarity)
        outputs[year] = distances
    return outputs


def _distribution_values(
    fields: Sequence[str],
) -> tuple[float, float, float, float]:
    """Return 1-Gini, Gini-Simpson, Shannon, and distinct-category count."""
    values = [str(item) for item in fields if str(item).strip()]
    if not values:
        return (math.nan, math.nan, math.nan, math.nan)
    counts = np.asarray(list(Counter(values).values()), dtype=float)
    probabilities = counts / counts.sum()
    gini = float(
        np.abs(counts[:, None] - counts[None, :]).sum()
        / (2.0 * len(counts) ** 2 * float(counts.mean()))
    )
    return (
        float(1.0 - gini),
        float(1.0 - np.square(probabilities).sum()),
        float(-np.sum(probabilities * np.log(probabilities))),
        float(len(counts)),
    )


def _distance_values(
    fields: Sequence[str], distances: Mapping[Tuple[str, str], float]
) -> tuple[float, float]:
    """Compute the two documented ordered-reference-pair disparity forms."""
    values = [str(item) for item in fields if str(item).strip()]
    if len(values) < 2:
        return (math.nan, math.nan)
    counts = Counter(values)
    categories = sorted(counts)
    required = [
        _canonical_pair(left, right)
        for left, right in combinations(categories, 2)
    ]
    if any(pair not in distances for pair in required):
        return (math.nan, math.nan)
    numerator = sum(
        2.0 * counts[left] * counts[right] * distances[_canonical_pair(left, right)]
        for left, right in combinations(categories, 2)
    )
    total = float(len(values))
    average_ordered_without_self = numerator / (total * (total - 1.0))
    rao_stirling = numerator / (total * total)
    return (float(average_ordered_without_self), float(rao_stirling))


def _cross_disciplinary_ratio(
    focal_field: str, reference_fields: Sequence[str], reference_total: int
) -> float:
    """Return references outside the focal field among all observable refs."""
    if reference_total <= 0 or not focal_field:
        return math.nan
    # A reference with no source category cannot establish disjointness, so it
    # remains in the denominator but is not assumed cross-disciplinary.
    return float(sum(field != focal_field for field in reference_fields) / reference_total)


def materialize(data_root: Path, output_path: Path, report_path: Path) -> Mapping[str, object]:
    """Create the v4 feature matrix from frozen outcome-blind inputs."""
    files = {
        "papers": data_root / "papers_common_all.parquet",
        "references": data_root / "paper_references.parquet",
        "reference_metadata": data_root / "reference_metadata.parquet",
        "field_events": data_root / "field_citation_events_aggregated.parquet",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen input(s): " + ", ".join(missing))
    papers = pd.read_parquet(
        files["papers"],
        columns=["paper_id", "publication_year", "openalex_primary_field", "natural_science_eligible"],
    )
    papers = papers.loc[papers["natural_science_eligible"].eq(1)].copy()
    papers["publication_year"] = pd.to_numeric(papers["publication_year"], errors="coerce")
    papers = papers.loc[papers["publication_year"].notna()].copy()
    papers["publication_year"] = papers["publication_year"].astype(int)
    papers["openalex_primary_field"] = papers["openalex_primary_field"].fillna("").astype(str)
    focal_ids = set(papers["paper_id"].astype(str))

    references = pd.read_parquet(files["references"], columns=["paper_id", "reference_id"])
    references = references.loc[references["paper_id"].astype(str).isin(focal_ids)].copy()
    metadata = pd.read_parquet(
        files["reference_metadata"],
        columns=["reference_id", "reference_year", "field_id"],
    )
    metadata = metadata.drop_duplicates("reference_id", keep="last")
    metadata["field_id"] = metadata["field_id"].fillna("").astype(str)
    bibliography = references.merge(
        papers[["paper_id", "publication_year"]],
        how="inner",
        on="paper_id",
        validate="many_to_one",
    ).merge(metadata, how="left", on="reference_id", validate="many_to_one")
    bibliography["reference_year"] = pd.to_numeric(
        bibliography["reference_year"], errors="coerce"
    )
    bibliography = bibliography.loc[
        bibliography["reference_year"].notna()
        & bibliography["reference_year"].lt(bibliography["publication_year"])
    ].copy()
    bibliography["field_id"] = bibliography["field_id"].fillna("").astype(str)
    fields_by_paper = bibliography.groupby("paper_id", sort=False)["field_id"].agg(list).to_dict()
    reference_counts = bibliography.groupby("paper_id", sort=False).size().to_dict()

    events = pd.read_parquet(files["field_events"])
    events["source_year"] = pd.to_numeric(events["source_year"], errors="coerce")
    events["citation_count"] = pd.to_numeric(events["citation_count"], errors="coerce").fillna(0.0)
    events = events.loc[events["source_year"].notna()].copy()
    events["source_year"] = events["source_year"].astype(int)
    events["source_field_id"] = events["source_field_id"].fillna("").astype(str)
    events["target_field_id"] = events["target_field_id"].fillna("").astype(str)
    events = events.loc[events["source_field_id"].ne("") & events["target_field_id"].ne("")]
    distance_by_year = _profile_distances(events, papers["publication_year"].tolist())

    rows = []
    for paper in papers.itertuples(index=False):
        paper_id = str(paper.paper_id)
        raw_fields = fields_by_paper.get(paper_id, [])
        fields = [value for value in raw_fields if value]
        balance, simpson, entropy, variety = _distribution_values(fields)
        dissimilarity, rao = _distance_values(
            fields, distance_by_year.get(int(paper.publication_year), {})
        )
        rows.append(
            {
                "paper_id": paper_id,
                "publication_year": int(paper.publication_year),
                "EF0001_average_reference_category_dissimilarity": dissimilarity,
                "EF0002_complemented_gini_interdisciplinarity": balance,
                "EF0003_cross_disciplinary_reference_ratio": _cross_disciplinary_ratio(
                    str(paper.openalex_primary_field), fields, int(reference_counts.get(paper_id, 0))
                ),
                "EF0004_gini_simpson_interdisciplinarity": simpson,
                "EF0007_rao_stirling_reference_diversity": rao,
                "EF0008_reference_discipline_shannon_entropy": entropy,
                "EF0009_referenced_subject_category_variety": variety,
                "mapped_reference_count": int(len(fields)),
                "total_reference_count": int(reference_counts.get(paper_id, 0)),
                "field_profile_window_years": WINDOW_YEARS,
                "definition_version": "evidence_derived_v4_raw_formula_20260819",
            }
        )
    frame = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    feature_columns = [column for column in frame if column.startswith("EF")]
    report = {
        "schema_version": "evidence_derived_v4_raw_formula_matrix",
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": _sha256(Path(__file__).resolve()),
        "inputs": {name: {"path": str(path.resolve()), "sha256": _sha256(path)} for name, path in files.items()},
        "output": str(output_path.resolve()),
        "output_sha256": _sha256(output_path),
        "row_count": int(len(frame)),
        "outcome_columns_used": False,
        "field_profile_window_years": WINDOW_YEARS,
        "feature_quality": {
            column: {
                "valid_count": int(frame[column].notna().sum()),
                "unique_count": int(frame[column].nunique(dropna=True)),
            }
            for column in feature_columns
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result = materialize(args.data_root.resolve(), args.output.resolve(), args.report.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Publication-prior core and bibliographic features for multi-horizon models."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from aspr.corpus import normalize_openalex_id

from .contracts import AUXILIARY_FEATURES, CORE_FEATURES
from .graph_snapshots import PriorGraph, SnapshotRepository, sample_reference_pairs


FEATURE_DEFINITION_VERSION = "nature-multihorizon-feature-v1"


def _entropy_evenness(values: Sequence[str]) -> float:
    if not values:
        return 0.0
    counts = pd.Series(values).value_counts().to_numpy(dtype=float)
    if len(counts) <= 1:
        return 0.0
    probabilities = counts / counts.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return float(entropy / math.log(len(counts)))


def _simpson(values: Sequence[Any]) -> float:
    if not values:
        return 0.0
    counts = pd.Series(list(values)).value_counts().to_numpy(dtype=float)
    probabilities = counts / counts.sum()
    return float(1.0 - np.sum(probabilities**2))


def _field_disparity(fields: Sequence[str], graph: PriorGraph) -> float:
    valid = [field for field in fields if field]
    if len(valid) < 2:
        return 0.0
    counts = pd.Series(valid).value_counts().to_dict()
    total_pairs = len(valid) * (len(valid) - 1) / 2.0
    weighted_distance = 0.0
    for left, right in itertools.combinations(sorted(counts), 2):
        left_groups = graph.field_communities.get(left, frozenset())
        right_groups = graph.field_communities.get(right, frozenset())
        union = left_groups | right_groups
        if not union:
            distance = 1.0
        else:
            distance = 1.0 - len(left_groups & right_groups) / len(union)
        weighted_distance += float(counts[left] * counts[right]) * distance
    return float(weighted_distance / total_pairs) if total_pairs else 0.0


def _pair_z_scores(
    pairs: Sequence[Tuple[str, str]], graph: PriorGraph
) -> List[float]:
    scores: List[float] = []
    denominator = max(1, graph.n_prior_papers)
    for left, right in pairs:
        left_count = int(graph.prior_paper_count.get(left, 0))
        right_count = int(graph.prior_paper_count.get(right, 0))
        if left_count <= 0 or right_count <= 0:
            continue
        observed = float(graph.pair_count.get((left, right), 0.0))
        expected = float(left_count * right_count / denominator)
        variance = max(expected * max(1e-9, 1.0 - min(1.0, expected / denominator)), 1e-9)
        scores.append(float((observed - expected) / math.sqrt(variance)))
    return scores


def _reference_metadata(reference_works: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    frame = reference_works.copy()
    frame["reference_id"] = frame["reference_id"].map(normalize_openalex_id)
    frame = frame.drop_duplicates("reference_id", keep="last")
    return {str(row["reference_id"]): row for row in frame.to_dict("records")}


def _paper_reference_map(paper_references: pd.DataFrame) -> Dict[str, List[str]]:
    frame = paper_references[["paper_id", "reference_id"]].copy()
    frame["paper_id"] = frame["paper_id"].map(normalize_openalex_id)
    frame["reference_id"] = frame["reference_id"].map(normalize_openalex_id)
    frame = frame[frame["paper_id"].ne("") & frame["reference_id"].ne("")].drop_duplicates()
    return frame.groupby("paper_id")["reference_id"].agg(list).to_dict()


def _paper_feature_row(
    paper: Mapping[str, Any],
    reference_ids: Sequence[str],
    lookup: Mapping[str, Mapping[str, Any]],
    graph: PriorGraph,
    *,
    max_pairs: int,
    seed: int,
) -> Dict[str, Any]:
    paper_id = normalize_openalex_id(paper.get("paper_id"))
    publication_year = int(paper["publication_year"])
    if graph.source_max_year >= publication_year:
        raise ValueError(
            f"Graph leakage for {paper_id}: source_max_year={graph.source_max_year}, publication_year={publication_year}"
        )

    declared_ids = sorted(set(normalize_openalex_id(item) for item in reference_ids if item))
    metadata_rows = []
    for reference_id in declared_ids:
        row = lookup.get(reference_id)
        if not row:
            continue
        reference_year = pd.to_numeric(
            row.get("publication_year", row.get("year")), errors="coerce"
        )
        # A same-year or future-dated reference cannot enter a publication-prior feature.
        if pd.notna(reference_year) and int(reference_year) < publication_year:
            metadata_rows.append((reference_id, row, int(reference_year)))
    valid_ids = [item[0] for item in metadata_rows]
    ages = [publication_year - item[2] for item in metadata_rows]
    fields = [
        str(
            item[1].get("openalex_primary_field")
            or item[1].get("primary_field")
            or item[1].get("field")
            or ""
        )
        for item in metadata_rows
    ]
    fields = [field for field in fields if field]

    communities = [
        graph.community[reference_id]
        for reference_id in valid_ids
        if reference_id in graph.community
    ]
    pairs, pair_sampling_rate = sample_reference_pairs(
        valid_ids,
        max_pairs=max_pairs,
        seed_key=f"{seed}:{paper_id}",
    )
    z_scores = _pair_z_scores(pairs, graph)
    possible_edges = len(set(valid_ids)) * (len(set(valid_ids)) - 1) // 2
    sampled_induced_edges = sum(pair in graph.edges for pair in pairs)
    density = sampled_induced_edges / len(pairs) if pairs else 0.0
    induced_edges = density * possible_edges
    degrees = [float(graph.degree.get(reference_id, 0)) for reference_id in valid_ids]
    component_sizes = [float(graph.component_size.get(reference_id, 1)) for reference_id in valid_ids]
    obscure_threshold = graph.obscure_degree_threshold

    n_ego_neighbors = len(set(valid_ids))
    effective_size = (
        n_ego_neighbors - (2.0 * induced_edges / n_ego_neighbors)
        if n_ego_neighbors > 0
        else 0.0
    )
    burt_efficiency = effective_size / n_ego_neighbors if n_ego_neighbors > 1 else 0.0
    modularity_before = graph.modularity()
    modularity_after = graph.modularity_after_sampled_pairs(
        pairs,
        sampling_rate=pair_sampling_rate,
    )

    quality_flags: List[str] = []
    metadata_coverage = len(valid_ids) / max(1, len(declared_ids))
    if len(valid_ids) < 10:
        quality_flags.append("fewer_than_10_valid_references")
    if metadata_coverage < 0.60:
        quality_flags.append("reference_metadata_coverage_below_60pct")
    if len(z_scores) < 20:
        quality_flags.append("fewer_than_20_valid_reference_pairs")
    if not communities:
        quality_flags.append("no_prior_graph_community_coverage")

    return {
        "paper_id": paper_id,
        "publication_year": publication_year,
        "domain12": paper.get("domain12", "unmapped"),
        "venue_family": paper.get("venue_family", ""),
        # Core eight.
        "delta_q0_shock": float(modularity_before - modularity_after),
        "rtd_simpson": _simpson(communities),
        "field_log_variety": float(math.log1p(len(set(fields)))),
        "field_evenness": _entropy_evenness(fields),
        "field_disparity": _field_disparity(fields, graph),
        "pair_atypicality_tail": float(-np.quantile(z_scores, 0.10)) if z_scores else np.nan,
        "pair_conventionality_median": float(np.median(z_scores)) if z_scores else np.nan,
        "burt_efficiency": float(burt_efficiency),
        # Auxiliary ten.
        "log_reference_count": float(math.log1p(len(declared_ids))),
        "reference_age_median": float(np.median(ages)) if ages else np.nan,
        "reference_age_iqr": (
            float(np.quantile(ages, 0.75) - np.quantile(ages, 0.25)) if ages else np.nan
        ),
        "recent_reference_share_5y": float(np.mean(np.asarray(ages) <= 5)) if ages else np.nan,
        "classic_reference_share_20y": float(np.mean(np.asarray(ages) >= 20)) if ages else np.nan,
        "prior_graph_degree_median": float(np.median(degrees)) if degrees else np.nan,
        "prior_graph_degree_p90": float(np.quantile(degrees, 0.90)) if degrees else np.nan,
        "prior_obscure_reference_share": (
            float(np.mean(np.asarray(degrees) <= obscure_threshold)) if degrees else np.nan
        ),
        "prior_component_size_log": (
            float(np.median(np.log1p(component_sizes))) if component_sizes else np.nan
        ),
        "reference_induced_density": float(density),
        # Provenance and reliability.
        "definition_version": FEATURE_DEFINITION_VERSION,
        "graph_id": graph.graph_id,
        "source_max_year": int(graph.source_max_year),
        "reference_count_declared": int(len(declared_ids)),
        "valid_reference_count": int(len(valid_ids)),
        "reference_metadata_coverage": float(metadata_coverage),
        "valid_pair_count": int(len(z_scores)),
        "pair_sampling_rate": float(pair_sampling_rate),
        "quality_flags": json.dumps(sorted(quality_flags), ensure_ascii=False),
    }


def build_feature_table(
    papers: pd.DataFrame,
    paper_references: pd.DataFrame,
    reference_works: pd.DataFrame,
    snapshot_catalog: Union[pd.DataFrame, Path],
    *,
    max_pairs: int = 10_000,
    seed: int = 20260710,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Compute the locked 8 core and 10 auxiliary publication-prior features."""

    required = {"paper_id", "publication_year"}
    missing = required - set(papers)
    if missing:
        raise ValueError(f"papers is missing required columns: {sorted(missing)}")
    lookup = _reference_metadata(reference_works)
    references = _paper_reference_map(paper_references)
    repository = SnapshotRepository(snapshot_catalog)
    rows: List[Dict[str, Any]] = []
    for paper in papers.to_dict("records"):
        year = pd.to_numeric(paper.get("publication_year"), errors="coerce")
        if pd.isna(year):
            continue
        paper_id = normalize_openalex_id(paper.get("paper_id"))
        graph = repository.for_year(int(year))
        rows.append(
            _paper_feature_row(
                paper,
                references.get(paper_id, []),
                lookup,
                graph,
                max_pairs=max_pairs,
                seed=seed,
            )
        )
    output = pd.DataFrame(rows)
    expected = list(CORE_FEATURES + AUXILIARY_FEATURES)
    missing_features = [column for column in expected if column not in output]
    if missing_features:
        raise RuntimeError(f"Feature builder did not emit locked features: {missing_features}")
    violations = output[
        pd.to_numeric(output["source_max_year"], errors="coerce")
        >= pd.to_numeric(output["publication_year"], errors="coerce")
    ]
    if not violations.empty:
        raise ValueError("One or more features use same-year or future graph data")
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        output.to_parquet(output_path, index=False)
    return output


def feature_quality_summary(features: pd.DataFrame) -> Dict[str, Any]:
    """Return GO-gate coverage diagnostics for the locked feature registry."""

    scientific = list(CORE_FEATURES + AUXILIARY_FEATURES)
    finite = pd.DataFrame(
        {
            column: np.isfinite(pd.to_numeric(features[column], errors="coerce"))
            for column in scientific
        }
    )
    core_finite = finite[list(CORE_FEATURES)].all(axis=1)
    return {
        "n_papers": int(len(features)),
        "core8_all_finite_rate": float(core_finite.mean()) if len(features) else 0.0,
        "core8_gate_95pct": bool(core_finite.mean() >= 0.95) if len(features) else False,
        "feature_finite_rates": {column: float(finite[column].mean()) for column in scientific},
        "strict_prior_year": bool(
            (
                pd.to_numeric(features["source_max_year"], errors="coerce")
                < pd.to_numeric(features["publication_year"], errors="coerce")
            ).all()
        ),
    }

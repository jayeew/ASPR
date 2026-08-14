"""Publication-time controls and opportunity features for v6 prediction."""

from __future__ import annotations

import json
import math
from collections import deque
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

import numpy as np
import pandas as pd

from gear.corpus import normalize_openalex_id


PREDICTION_FEATURE_VERSION = "aspr-prediction-features-v6-1"


def _normalized_reference_years(
    reference_works: pd.DataFrame,
) -> Dict[str, int]:
    id_column = "work_id" if "work_id" in reference_works else "reference_id"
    year_column = (
        "publication_year"
        if "publication_year" in reference_works
        else "reference_year"
    )
    if id_column not in reference_works or year_column not in reference_works:
        raise ValueError(
            "reference_works requires work_id/reference_id and "
            "publication_year/reference_year"
        )
    metadata = reference_works[[id_column, year_column]].copy()
    metadata = metadata.rename(columns={year_column: "publication_year"})
    metadata["reference_id"] = metadata[id_column].map(normalize_openalex_id)
    metadata["publication_year"] = pd.to_numeric(
        metadata["publication_year"], errors="coerce"
    )
    metadata = metadata[
        metadata["reference_id"].ne("")
        & metadata["publication_year"].notna()
    ].drop_duplicates("reference_id", keep="last")
    return {
        str(row.reference_id): int(row.publication_year)
        for row in metadata.itertuples(index=False)
    }


def _reference_lists(
    paper_references: pd.DataFrame,
) -> Dict[str, List[str]]:
    required = {"paper_id", "reference_id"}
    missing = sorted(required - set(paper_references))
    if missing:
        raise ValueError(f"paper_references is missing columns: {missing}")
    frame = paper_references[["paper_id", "reference_id"]].copy()
    frame["paper_id"] = frame["paper_id"].map(normalize_openalex_id)
    frame["reference_id"] = frame["reference_id"].map(normalize_openalex_id)
    frame = frame[
        frame["paper_id"].ne("") & frame["reference_id"].ne("")
    ].drop_duplicates()
    return (
        frame.groupby("paper_id", sort=False)["reference_id"]
        .agg(list)
        .to_dict()
    )


def build_registered_control_features(
    papers: pd.DataFrame,
    paper_references: pd.DataFrame,
    reference_works: pd.DataFrame,
    *,
    prior_graph_features: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build the preregistered strong baseline without future information."""
    required = {"paper_id", "publication_year", "domain12"}
    missing = sorted(required - set(papers))
    if missing:
        raise ValueError(f"papers is missing columns: {missing}")
    years = _normalized_reference_years(reference_works)
    references = _reference_lists(paper_references)
    rows: List[Dict[str, Any]] = []
    for paper in papers.to_dict("records"):
        paper_id = normalize_openalex_id(paper.get("paper_id"))
        year_value = pd.to_numeric(
            paper.get("publication_year"), errors="coerce"
        )
        if not paper_id or pd.isna(year_value):
            continue
        year = int(year_value)
        declared = sorted(set(references.get(paper_id, [])))
        valid_years = [
            years[reference_id]
            for reference_id in declared
            if reference_id in years and years[reference_id] < year
        ]
        ages = [year - reference_year for reference_year in valid_years]
        rows.append(
            {
                "paper_id": paper_id,
                "publication_year": year,
                "domain12": str(paper.get("domain12") or "unmapped"),
                "venue_family": str(paper.get("venue_family") or "unknown"),
                "log_reference_count": float(math.log1p(len(declared))),
                "reference_age_median": (
                    float(np.median(ages)) if ages else np.nan
                ),
                "reference_age_iqr": (
                    float(np.quantile(ages, 0.75) - np.quantile(ages, 0.25))
                    if ages
                    else np.nan
                ),
                "reference_list_retrieved": int(paper_id in references),
                "reference_year_coverage": (
                    len(valid_years) / len(declared) if declared else np.nan
                ),
                "source_max_year": year - 1,
                "definition_version": PREDICTION_FEATURE_VERSION,
            }
        )
    output = pd.DataFrame(rows)
    if prior_graph_features is not None:
        required_graph = {"paper_id", "prior_graph_degree_median"}
        missing_graph = sorted(required_graph - set(prior_graph_features))
        if missing_graph:
            raise ValueError(
                f"prior_graph_features is missing columns: {missing_graph}"
            )
        graph = prior_graph_features[
            ["paper_id", "prior_graph_degree_median"]
        ].copy()
        graph["paper_id"] = graph["paper_id"].map(normalize_openalex_id)
        graph = graph.drop_duplicates("paper_id", keep="last")
        output = output.merge(
            graph, on="paper_id", how="left", validate="one_to_one"
        )
    else:
        output["prior_graph_degree_median"] = np.nan
    violations = output["source_max_year"].ge(output["publication_year"])
    if violations.any():
        raise ValueError("control features contain temporal leakage")
    return output


def _as_reference_list(value: Any) -> List[str]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = []
    elif isinstance(value, (list, tuple, set, np.ndarray)):
        parsed = list(value)
    else:
        parsed = []
    return sorted(
        {
            normalized
            for item in parsed
            if (normalized := normalize_openalex_id(item))
        }
    )


class _DisjointSet:
    """Small deterministic union-find used for annual component sizes."""

    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}
        self.size: Dict[str, int] = {}

    def add(self, item: str) -> None:
        if item not in self.parent:
            self.parent[item] = item
            self.size[item] = 1

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size.pop(right_root)

    def merged_size(self, roots: Iterable[str]) -> int:
        unique = {self.find(root) for root in roots}
        return sum(self.size[root] for root in unique)


def _neighbor_induced_edges(
    neighbors: Set[str],
    adjacency: Mapping[str, Set[str]],
) -> int:
    return (
        sum(len(adjacency.get(node, set()) & neighbors) for node in neighbors)
        // 2
    )


def _harmonic_from_new_focal(
    neighbors: Set[str],
    adjacency: Mapping[str, Set[str]],
    eligible_count: int,
) -> float:
    if not neighbors or eligible_count <= 0:
        return 0.0
    distances: Dict[str, int] = {node: 1 for node in neighbors}
    queue = deque(sorted(neighbors))
    while queue:
        node = queue.popleft()
        next_distance = distances[node] + 1
        for adjacent in adjacency.get(node, set()):
            if adjacent not in distances:
                distances[adjacent] = next_distance
                queue.append(adjacent)
    return float(
        sum(1.0 / distance for distance in distances.values())
        / eligible_count
    )


def _prepare_historical_works(
    historical_work_view: pd.DataFrame,
    reference_years: Mapping[str, int],
) -> List[tuple[int, str, List[str]]]:
    required = {"work_id", "publication_year", "referenced_works"}
    missing = sorted(required - set(historical_work_view))
    if missing:
        raise ValueError(f"historical_work_view is missing columns: {missing}")
    rows: List[tuple[int, str, List[str]]] = []
    for row in historical_work_view[
        ["work_id", "publication_year", "referenced_works"]
    ].itertuples(index=False):
        work_id = normalize_openalex_id(row.work_id)
        year_value = pd.to_numeric(row.publication_year, errors="coerce")
        if not work_id or pd.isna(year_value):
            continue
        year = int(year_value)
        references = [
            reference_id
            for reference_id in _as_reference_list(row.referenced_works)
            if reference_years.get(reference_id, year) < year
        ]
        rows.append((year, work_id, references))
    return sorted(rows, key=lambda item: (item[0], item[1]))


def build_bibliographic_opportunity_features(
    papers: pd.DataFrame,
    paper_references: pd.DataFrame,
    historical_work_view: pd.DataFrame,
    *,
    reference_metadata: Optional[pd.DataFrame] = None,
    compute_exact_clustering: bool = False,
    compute_exact_closeness: bool = False,
) -> pd.DataFrame:
    """Build focal positions in a strictly prior shared-reference graph.

    The graph is updated only after all focal papers in a publication year are
    scored.  Harmonic closeness is optional because exact all-component BFS is
    not a primary registered opportunity input.
    """
    required = {"paper_id", "publication_year"}
    missing = sorted(required - set(papers))
    if missing:
        raise ValueError(f"papers is missing columns: {missing}")
    reference_years = _normalized_reference_years(
        reference_metadata
        if reference_metadata is not None
        else historical_work_view
    )
    focal_references = _reference_lists(paper_references)
    historical = _prepare_historical_works(
        historical_work_view, reference_years
    )
    focal = papers[["paper_id", "publication_year"]].copy()
    focal["paper_id"] = focal["paper_id"].map(normalize_openalex_id)
    focal["publication_year"] = pd.to_numeric(
        focal["publication_year"], errors="coerce"
    )
    focal = focal[
        focal["paper_id"].ne("") & focal["publication_year"].notna()
    ].drop_duplicates("paper_id")
    focal["publication_year"] = focal["publication_year"].astype(int)

    postings: Dict[str, Set[str]] = {}
    adjacency: Dict[str, Set[str]] = {}
    seen_work_ids: Set[str] = set()
    dsu = _DisjointSet()
    eligible_count = 0
    history_index = 0

    def add_historical(work_id: str, references: Sequence[str]) -> None:
        nonlocal eligible_count
        if work_id in seen_work_ids or not references:
            return
        seen_work_ids.add(work_id)
        neighbors: Set[str] = set()
        for reference_id in references:
            neighbors.update(postings.get(reference_id, set()))
        dsu.add(work_id)
        for neighbor in neighbors:
            dsu.union(work_id, neighbor)
        if compute_exact_clustering or compute_exact_closeness:
            adjacency[work_id] = set(neighbors)
            for neighbor in neighbors:
                adjacency.setdefault(neighbor, set()).add(work_id)
        for reference_id in references:
            postings.setdefault(reference_id, set()).add(work_id)
        eligible_count += 1

    rows: List[Dict[str, Any]] = []
    for year, year_papers in focal.groupby("publication_year", sort=True):
        year = int(year)
        while (
            history_index < len(historical)
            and historical[history_index][0] < year
        ):
            _, work_id, references = historical[history_index]
            add_historical(work_id, references)
            history_index += 1
        for paper in year_papers.itertuples(index=False):
            paper_id = str(paper.paper_id)
            declared = sorted(set(focal_references.get(paper_id, [])))
            valid = [
                reference_id
                for reference_id in declared
                if reference_years.get(reference_id, year) < year
            ]
            shared_counts: Dict[str, int] = {}
            for reference_id in valid:
                for neighbor in postings.get(reference_id, set()):
                    shared_counts[neighbor] = shared_counts.get(neighbor, 0) + 1
            neighbors = set(shared_counts)
            degree = len(neighbors)
            possible_neighbor_edges = degree * (degree - 1) // 2
            observed_neighbor_edges = (
                _neighbor_induced_edges(neighbors, adjacency)
                if compute_exact_clustering
                else 0
            )
            component_size = 1 + dsu.merged_size(neighbors)
            quality_flags: List[str] = []
            if len(valid) < 2:
                quality_flags.append("fewer_than_two_valid_references")
            if not compute_exact_closeness:
                quality_flags.append("exact_harmonic_closeness_not_requested")
            if not compute_exact_clustering:
                quality_flags.append("exact_local_clustering_not_requested")
            rows.append(
                {
                    "paper_id": paper_id,
                    "publication_year": year,
                    "bc_degree_per_reference_t0": (
                        degree / max(1, len(valid))
                        if len(valid) >= 2
                        else np.nan
                    ),
                    "bc_shared_reference_strength_t0": (
                        float(sum(shared_counts.values()))
                        if len(valid) >= 2
                        else np.nan
                    ),
                    "bc_component_share_t0": (
                        component_size / (eligible_count + 1)
                        if eligible_count
                        else np.nan
                    ),
                    "bc_local_clustering_t0": (
                        (
                            observed_neighbor_edges / possible_neighbor_edges
                            if possible_neighbor_edges
                            else 0.0
                        )
                        if compute_exact_clustering
                        else np.nan
                    ),
                    "bc_harmonic_closeness_t0": (
                        _harmonic_from_new_focal(
                            neighbors, adjacency, eligible_count
                        )
                        if compute_exact_closeness
                        else np.nan
                    ),
                    "bc_reference_coverage": (
                        len(valid) / len(declared) if declared else np.nan
                    ),
                    "eligible_prior_paper_count": eligible_count,
                    "source_max_year": year - 1,
                    "definition_version": PREDICTION_FEATURE_VERSION,
                    "quality_flags": json.dumps(
                        sorted(quality_flags), ensure_ascii=False
                    ),
                }
            )
    output = pd.DataFrame(rows)
    violations = output["source_max_year"].ge(output["publication_year"])
    if violations.any():
        raise ValueError("opportunity features contain temporal leakage")
    return output


__all__ = [
    "PREDICTION_FEATURE_VERSION",
    "build_bibliographic_opportunity_features",
    "build_registered_control_features",
]

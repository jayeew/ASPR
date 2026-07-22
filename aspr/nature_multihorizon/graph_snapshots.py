"""Conservative pre-publication-year graph snapshots and pair statistics."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd

from aspr.corpus import normalize_openalex_id


def _normalize_pair(left: str, right: str) -> Tuple[str, str]:
    return (left, right) if left <= right else (right, left)


def sample_reference_pairs(
    reference_ids: Sequence[str],
    *,
    max_pairs: int = 10_000,
    seed_key: str = "",
) -> Tuple[List[Tuple[str, str]], float]:
    """Return deterministic reference pairs and the retained pair fraction."""

    unique = sorted(set(item for item in reference_ids if item))
    total = len(unique) * (len(unique) - 1) // 2
    if total == 0:
        return [], 0.0
    if total <= int(max_pairs):
        return list(itertools.combinations(unique, 2)), 1.0
    seed = int.from_bytes(
        hashlib.blake2b(seed_key.encode("utf-8"), digest_size=8).digest(),
        "big",
    )
    sampled_indices = np.sort(
        np.random.default_rng(seed).choice(total, size=int(max_pairs), replace=False)
    )
    starts = np.asarray(
        [index * (2 * len(unique) - index - 1) // 2 for index in range(len(unique))],
        dtype=np.int64,
    )
    pairs: List[Tuple[str, str]] = []
    for sampled in sampled_indices:
        left_index = int(np.searchsorted(starts, sampled, side="right") - 1)
        right_index = left_index + 1 + int(sampled - starts[left_index])
        pairs.append((unique[left_index], unique[right_index]))
    return pairs, float(max_pairs / total)


def _community_partition(graph: nx.Graph, seed: int) -> Dict[str, int]:
    if graph.number_of_nodes() == 0:
        return {}
    if graph.number_of_edges() == 0:
        return {str(node): index for index, node in enumerate(sorted(graph.nodes()))}
    if graph.number_of_nodes() <= 20_000:
        communities = nx.algorithms.community.greedy_modularity_communities(graph)
    else:
        communities = nx.algorithms.community.asyn_lpa_communities(graph, seed=seed)
    ordered = sorted((sorted(map(str, group)) for group in communities), key=lambda group: group[0])
    return {
        node: community_id
        for community_id, group in enumerate(ordered)
        for node in group
    }


def _component_partition(graph: nx.Graph) -> Tuple[Dict[str, int], Dict[str, int]]:
    component_id: Dict[str, int] = {}
    component_size: Dict[str, int] = {}
    ordered = sorted(
        (sorted(map(str, group)) for group in nx.connected_components(graph)),
        key=lambda group: group[0],
    )
    for index, group in enumerate(ordered):
        for node in group:
            component_id[node] = index
            component_size[node] = len(group)
    return component_id, component_size


def _cutoff_years(papers: pd.DataFrame, interval: int) -> List[int]:
    years = pd.to_numeric(papers["publication_year"], errors="coerce").dropna().astype(int)
    if years.empty:
        return []
    return sorted({int(year // interval * interval) for year in years})


def build_graph_snapshots(
    papers: pd.DataFrame,
    paper_references: pd.DataFrame,
    reference_works: pd.DataFrame,
    reference_edges: pd.DataFrame,
    output_dir: Path,
    *,
    interval: int = 5,
    max_pairs_per_paper: int = 10_000,
    seed: int = 20260710,
) -> pd.DataFrame:
    """Build immutable snapshots whose every source edge predates the cutoff.

    ``cutoff_year=y`` means the snapshot includes only edges and historical
    citing papers with year strictly less than ``y``.  A focal paper published
    in year ``y`` may therefore safely use the snapshot at ``y`` or earlier.
    """

    if interval <= 0:
        raise ValueError("interval must be positive")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    works = reference_works.copy()
    works["reference_id"] = works["reference_id"].map(normalize_openalex_id)
    works["publication_year"] = pd.to_numeric(works["publication_year"], errors="coerce")
    field_column = next(
        (
            column
            for column in ("openalex_primary_field", "primary_field", "field")
            if column in works
        ),
        None,
    )
    field_lookup = (
        works.set_index("reference_id")[field_column].fillna("").astype(str).to_dict()
        if field_column
        else {}
    )

    paper_years = papers[["paper_id", "publication_year"]].copy()
    paper_years["paper_id"] = paper_years["paper_id"].map(normalize_openalex_id)
    paper_years["publication_year"] = pd.to_numeric(
        paper_years["publication_year"], errors="coerce"
    )
    bibliographies = paper_references[["paper_id", "reference_id"]].copy()
    bibliographies["paper_id"] = bibliographies["paper_id"].map(normalize_openalex_id)
    bibliographies["reference_id"] = bibliographies["reference_id"].map(normalize_openalex_id)
    bibliographies = bibliographies.merge(paper_years, on="paper_id", how="left")

    citation_edges = reference_edges.copy()
    for column in ("source_reference_id", "target_reference_id"):
        citation_edges[column] = citation_edges[column].map(normalize_openalex_id)
    citation_edges["edge_year"] = pd.to_numeric(citation_edges["edge_year"], errors="coerce")
    citation_edges = citation_edges[
        citation_edges["source_reference_id"].ne("")
        & citation_edges["target_reference_id"].ne("")
        & citation_edges["source_reference_id"].ne(citation_edges["target_reference_id"])
    ].copy()

    citation_edges = citation_edges.sort_values("edge_year", kind="stable")
    bibliographies = bibliographies.sort_values("publication_year", kind="stable")
    graph = nx.Graph()
    occurrence: Dict[str, int] = {}
    pair_counts: Dict[Tuple[str, str], float] = {}
    n_prior_papers = 0
    previous_cutoff: Optional[int] = None
    catalog_rows: List[Dict[str, object]] = []
    for cutoff in _cutoff_years(paper_years, interval):
        lower_bound = -np.inf if previous_cutoff is None else previous_cutoff
        new_edges = citation_edges[
            (citation_edges["edge_year"] >= lower_bound)
            & (citation_edges["edge_year"] < cutoff)
        ]
        graph.add_edges_from(
            _normalize_pair(str(row.source_reference_id), str(row.target_reference_id))
            for row in new_edges.itertuples(index=False)
        )

        new_references = bibliographies[
            (bibliographies["publication_year"] >= lower_bound)
            & (bibliographies["publication_year"] < cutoff)
        ]
        for paper_id, group in new_references.groupby("paper_id", sort=False):
            reference_ids = sorted(set(group["reference_id"].dropna().astype(str)))
            if not reference_ids:
                continue
            n_prior_papers += 1
            graph.add_nodes_from(reference_ids)
            for reference_id in reference_ids:
                occurrence[reference_id] = occurrence.get(reference_id, 0) + 1
            pairs, retained_fraction = sample_reference_pairs(
                reference_ids,
                max_pairs=max_pairs_per_paper,
                seed_key=f"{seed}:{paper_id}",
            )
            inverse_probability = 1.0 / max(retained_fraction, 1e-12)
            for pair in pairs:
                pair_counts[pair] = pair_counts.get(pair, 0.0) + inverse_probability

        community = _community_partition(graph, seed=seed)
        component, component_size = _component_partition(graph)
        node_rows = [
            {
                "node_id": str(node),
                "degree": int(graph.degree(node)),
                "community_id": int(community.get(str(node), -1)),
                "component_id": int(component.get(str(node), -1)),
                "component_size": int(component_size.get(str(node), 1)),
                "prior_paper_count": int(occurrence.get(str(node), 0)),
                "field": field_lookup.get(str(node), ""),
            }
            for node in sorted(graph.nodes())
        ]
        pair_rows = [
            {"left_id": pair[0], "right_id": pair[1], "pair_count": float(count)}
            for pair, count in sorted(pair_counts.items())
        ]
        edge_rows = [
            {"left_id": pair[0], "right_id": pair[1]}
            for pair in sorted(_normalize_pair(str(left), str(right)) for left, right in graph.edges())
        ]

        graph_id = f"prior-{cutoff}-v1"
        node_path = root / f"{graph_id}.nodes.parquet"
        edge_path = root / f"{graph_id}.edges.parquet"
        pair_path = root / f"{graph_id}.pairs.parquet"
        pd.DataFrame(
            node_rows,
            columns=[
                "node_id",
                "degree",
                "community_id",
                "component_id",
                "component_size",
                "prior_paper_count",
                "field",
            ],
        ).to_parquet(node_path, index=False)
        pd.DataFrame(edge_rows, columns=["left_id", "right_id"]).to_parquet(edge_path, index=False)
        pd.DataFrame(pair_rows, columns=["left_id", "right_id", "pair_count"]).to_parquet(
            pair_path, index=False
        )
        catalog_rows.append(
            {
                "cutoff_year": int(cutoff),
                "source_max_year": int(cutoff - 1),
                "graph_id": graph_id,
                # Relative paths survive ArtifactStore's staging-directory rename.
                "node_path": node_path.name,
                "edge_path": edge_path.name,
                "pair_path": pair_path.name,
                "n_nodes": graph.number_of_nodes(),
                "n_edges": graph.number_of_edges(),
                "n_prior_papers": int(n_prior_papers),
                "n_pair_counts": len(pair_rows),
                "community_algorithm": (
                    "greedy_modularity" if graph.number_of_nodes() <= 20_000 else "label_propagation"
                ),
            }
        )
        previous_cutoff = int(cutoff)

    catalog = pd.DataFrame(catalog_rows)
    catalog_path = root / "graph_snapshots.parquet"
    catalog.to_parquet(catalog_path, index=False)
    catalog.attrs["base_dir"] = str(root)
    manifest = {
        "artifact_kind": "nature_multihorizon_graph_snapshots",
        "strict_cutoff": "edge_year < cutoff_year",
        "interval_years": int(interval),
        "n_snapshots": int(len(catalog)),
        "cutoff_years": catalog["cutoff_year"].tolist() if not catalog.empty else [],
    }
    (root / "graph_snapshots_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return catalog


@dataclass(frozen=True)
class PriorGraph:
    cutoff_year: int
    source_max_year: int
    graph_id: str
    edges: frozenset[Tuple[str, str]]
    degree: Mapping[str, int]
    community: Mapping[str, int]
    component_size: Mapping[str, int]
    prior_paper_count: Mapping[str, int]
    field: Mapping[str, str]
    pair_count: Mapping[Tuple[str, str], float]
    n_prior_papers: int
    base_modularity: float
    community_degree_sum: Mapping[int, float]
    community_internal_edges: Mapping[int, float]
    field_communities: Mapping[str, frozenset[int]]
    obscure_degree_threshold: float

    def induced_edge_count(self, reference_ids: Sequence[str]) -> int:
        nodes = sorted(set(reference_ids))
        return sum(_normalize_pair(left, right) in self.edges for left, right in itertools.combinations(nodes, 2))

    def modularity(self) -> float:
        return self.base_modularity

    def modularity_after_reference_clique(self, reference_ids: Sequence[str]) -> float:
        nodes = sorted(set(reference_ids))
        return self.modularity_after_sampled_pairs(
            list(itertools.combinations(nodes, 2)),
            sampling_rate=1.0,
        )

    def modularity_after_sampled_pairs(
        self,
        pairs: Sequence[Tuple[str, str]],
        *,
        sampling_rate: float,
    ) -> float:
        """Estimate clique modularity after adding a bounded uniform pair sample."""
        if sampling_rate <= 0:
            return self.base_modularity
        additions = [pair for pair in pairs if _normalize_pair(*pair) not in self.edges]
        inverse_probability = 1.0 / sampling_rate
        m_before = len(self.edges)
        m_after = m_before + len(additions) * inverse_probability
        if m_after == 0:
            return 0.0
        degree_sum = dict(self.community_degree_sum)
        internal = dict(self.community_internal_edges)
        for left, right in additions:
            left_group = int(self.community.get(left, -1))
            right_group = int(self.community.get(right, -1))
            degree_sum[left_group] = degree_sum.get(left_group, 0) + inverse_probability
            degree_sum[right_group] = degree_sum.get(right_group, 0) + inverse_probability
            if left_group == right_group:
                internal[left_group] = internal.get(left_group, 0) + inverse_probability
        return float(
            sum(
                internal.get(group, 0) / m_after
                - (degree_sum.get(group, 0) / (2 * m_after)) ** 2
                for group in degree_sum
            )
        )


def load_snapshot_catalog(path: Path) -> pd.DataFrame:
    """Load a catalog and retain its relocation-safe artifact base directory."""

    source = Path(path)
    catalog = pd.read_parquet(source)
    catalog.attrs["base_dir"] = str(source.parent)
    return catalog


def _resolve_snapshot_path(catalog: pd.DataFrame, value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    base = catalog.attrs.get("base_dir")
    if not base:
        raise ValueError(
            "Relative snapshot paths require catalog.attrs['base_dir']; "
            "load persisted catalogs with load_snapshot_catalog(path)"
        )
    return Path(str(base)) / path


def load_prior_graph(catalog: Union[pd.DataFrame, Path], focal_year: int) -> PriorGraph:
    """Load the most recent snapshot whose cutoff does not exceed focal year."""

    catalog_frame = load_snapshot_catalog(catalog) if isinstance(catalog, Path) else catalog
    eligible = catalog_frame[
        pd.to_numeric(catalog_frame["cutoff_year"], errors="coerce") <= int(focal_year)
    ]
    if eligible.empty:
        raise ValueError(f"No strictly prior graph snapshot for publication year {focal_year}")
    row = eligible.sort_values("cutoff_year").iloc[-1]
    nodes = pd.read_parquet(_resolve_snapshot_path(catalog_frame, row["node_path"]))
    edges = pd.read_parquet(_resolve_snapshot_path(catalog_frame, row["edge_path"]))
    pairs = pd.read_parquet(_resolve_snapshot_path(catalog_frame, row["pair_path"]))
    edge_values = frozenset(
        _normalize_pair(str(item.left_id), str(item.right_id))
        for item in edges.itertuples(index=False)
    )
    degree = dict(zip(nodes["node_id"].astype(str), nodes["degree"].astype(int)))
    community = dict(
        zip(nodes["node_id"].astype(str), nodes["community_id"].astype(int))
    )
    fields = dict(
        zip(nodes["node_id"].astype(str), nodes["field"].fillna("").astype(str))
    )
    field_communities: Dict[str, set[int]] = {}
    for node, field in fields.items():
        if field:
            field_communities.setdefault(field, set()).add(int(community.get(node, -1)))
    degree_sum: Dict[int, int] = {}
    internal: Dict[int, int] = {}
    for node, value in degree.items():
        group = int(community.get(node, -1))
        degree_sum[group] = degree_sum.get(group, 0) + int(value)
    for left, right in edge_values:
        group = int(community.get(left, -1))
        if group == int(community.get(right, -2)):
            internal[group] = internal.get(group, 0) + 1
    m = len(edge_values)
    base_modularity = float(
        sum(
            internal.get(group, 0) / m - (degree_sum.get(group, 0) / (2 * m)) ** 2
            for group in degree_sum
        )
    ) if m else 0.0
    return PriorGraph(
        cutoff_year=int(row["cutoff_year"]),
        source_max_year=int(row["source_max_year"]),
        graph_id=str(row["graph_id"]),
        edges=edge_values,
        degree=degree,
        community=community,
        component_size=dict(
            zip(nodes["node_id"].astype(str), nodes["component_size"].astype(int))
        ),
        prior_paper_count=dict(
            zip(nodes["node_id"].astype(str), nodes["prior_paper_count"].astype(int))
        ),
        field=fields,
        pair_count={
            _normalize_pair(str(item.left_id), str(item.right_id)): float(item.pair_count)
            for item in pairs.itertuples(index=False)
        },
        n_prior_papers=int(row["n_prior_papers"]),
        base_modularity=base_modularity,
        community_degree_sum=degree_sum,
        community_internal_edges=internal,
        field_communities={
            field: frozenset(groups) for field, groups in field_communities.items()
        },
        obscure_degree_threshold=(
            float(np.quantile(np.asarray(list(degree.values()), dtype=float), 0.10))
            if degree
            else 0.0
        ),
    )


class SnapshotRepository:
    """Small LRU-free cache used while building paper features by year."""

    def __init__(self, catalog: Union[pd.DataFrame, Path]) -> None:
        self.catalog = load_snapshot_catalog(catalog) if isinstance(catalog, Path) else catalog.copy()
        if not isinstance(catalog, Path) and catalog.attrs.get("base_dir"):
            self.catalog.attrs["base_dir"] = catalog.attrs["base_dir"]
        self._cache: Dict[int, PriorGraph] = {}

    def for_year(self, focal_year: int) -> PriorGraph:
        eligible = self.catalog[
            pd.to_numeric(self.catalog["cutoff_year"], errors="coerce") <= int(focal_year)
        ]
        if eligible.empty:
            raise ValueError(f"No graph snapshot for {focal_year}")
        cutoff = int(eligible["cutoff_year"].max())
        if cutoff not in self._cache:
            self._cache[cutoff] = load_prior_graph(self.catalog, focal_year)
        return self._cache[cutoff]

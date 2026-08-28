"""Build auditable perturbation inputs from real future-citer reference graphs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import deque
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import requests

from .acquire_oof_manuscripts import _openalex_api_keys

OPENALEX_FIELDS_API = "https://api.openalex.org/fields"


def build_inputs(
    benchmark_manifest: Path,
    future_citers_path: Path,
    reference_edges_path: Path,
    reference_works_path: Path,
    output_dir: Path,
    *,
    horizon: int = 5,
    claim_adoption_path: Path | None = None,
) -> dict[str, Any]:
    """Materialize four future-graph families at frozen paper grain."""
    cases = json.loads(benchmark_manifest.read_text(encoding="utf-8"))["cases"]
    paper_ids = {str(case["paper_id"]) for case in cases}
    output_dir.mkdir(parents=True, exist_ok=True)
    field_taxonomy = _field_taxonomy(output_dir)
    cache_key = hashlib.sha256("\n".join(sorted(paper_ids)).encode()).hexdigest()[:16]
    future = _load_future_citers(
        future_citers_path,
        paper_ids,
        horizon,
        field_taxonomy,
        cache_path=output_dir / f"filtered_future_citers_{cache_key}.parquet",
    )
    focal_edges = _scan_edges(reference_edges_path, paper_ids)
    predecessors = set(focal_edges["target"].astype(str))
    predecessor_edges = _scan_edges(reference_edges_path, predecessors)
    work_ids = _needed_work_ids(paper_ids, focal_edges, predecessor_edges, future)
    fields = _scan_work_fields(reference_works_path, work_ids)
    adoption = _load_adoption(claim_adoption_path)
    rows = [
        _paper_row(
            paper_id,
            future[future["paper_id"].eq(paper_id)],
            focal_edges[focal_edges["source"].eq(paper_id)],
            predecessor_edges,
            fields,
            adoption.get(paper_id),
            horizon,
        )
        for paper_id in sorted(paper_ids)
    ]
    output = pd.DataFrame(rows)
    metadata = pd.DataFrame(cases)[
        ["paper_id", "score_decile", "domain12", "publication_year"]
    ]
    output = output.merge(metadata, on="paper_id", how="left", validate="one_to_one")
    target = output_dir / "real_perturbation_inputs.parquet"
    output.to_parquet(target, index=False)
    _complete_graph_rows(output).to_csv(
        output_dir / "complete_graph_cohort.csv", index=False
    )
    summary = _summary(output, target, future_citers_path, horizon)
    (output_dir / "real_perturbation_inputs_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _load_future_citers(
    path: Path,
    paper_ids: set[str],
    horizon: int,
    field_taxonomy: dict[str, str],
    *,
    cache_path: Path,
) -> pd.DataFrame:
    if cache_path.is_file():
        return pd.read_parquet(cache_path)
    dataset = ds.dataset(str(path), format="parquet")
    predicate = ds.field("paper_id").isin(sorted(paper_ids)) & (
        ds.field("horizon") == horizon
    )
    frame = dataset.to_table(
        columns=["paper_id", "citer_primary_field", "referenced_works"],
        filter=predicate,
    ).to_pandas()
    frame["citer_primary_field"] = frame["citer_primary_field"].map(field_taxonomy)
    frame.to_parquet(cache_path, index=False)
    return frame


def _field_taxonomy(output_dir: Path) -> dict[str, str]:
    path = output_dir / "openalex_fields.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {str(row["id"]): str(row["display_name"]) for row in payload}
    params: dict[str, str | int] = {"per_page": 200}
    keys = _openalex_api_keys()
    if keys:
        params["api_key"] = keys[0]
    response = requests.get(OPENALEX_FIELDS_API, params=params, timeout=45)
    response.raise_for_status()
    payload = response.json().get("results", [])
    rows = [{"id": row["id"], "display_name": row["display_name"]} for row in payload]
    path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {str(row["id"]): str(row["display_name"]) for row in rows}


def _scan_edges(path: Path, sources: set[str]) -> pd.DataFrame:
    if not sources:
        return pd.DataFrame(columns=["source", "target"])
    selected: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path, usecols=["source", "target"], dtype=str, chunksize=500_000
    ):
        matched = chunk[chunk["source"].isin(sources)]
        if not matched.empty:
            selected.append(matched)
    if not selected:
        return pd.DataFrame(columns=["source", "target"])
    return pd.concat(selected, ignore_index=True).drop_duplicates()


def _needed_work_ids(
    paper_ids: set[str],
    focal_edges: pd.DataFrame,
    predecessor_edges: pd.DataFrame,
    future: pd.DataFrame,
) -> set[str]:
    output = set(paper_ids)
    for frame in (focal_edges, predecessor_edges):
        output.update(frame["source"].dropna().astype(str))
        output.update(frame["target"].dropna().astype(str))
    for values in future["referenced_works"]:
        output.update(str(value) for value in _references(values))
    return output


def _scan_work_fields(path: Path, work_ids: set[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for chunk in pd.read_csv(
        path, usecols=["id", "openalex_primary_field"], dtype=str, chunksize=500_000
    ):
        matched = chunk[chunk["id"].isin(work_ids)].dropna(
            subset=["openalex_primary_field"]
        )
        output.update(
            zip(matched["id"], matched["openalex_primary_field"], strict=False)
        )
    return output


def _load_adoption(path: Path | None) -> dict[str, float]:
    if path is None or not path.is_file():
        return {}
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    required = {"paper_id", "claim_adoption_breadth"}
    if not required.issubset(frame):
        raise ValueError(
            f"claim adoption file missing: {sorted(required - set(frame))}"
        )
    return dict(
        zip(
            frame["paper_id"].astype(str),
            pd.to_numeric(frame["claim_adoption_breadth"]),
            strict=False,
        )
    )


def _paper_row(
    paper_id: str,
    future: pd.DataFrame,
    focal_edges: pd.DataFrame,
    predecessor_edges: pd.DataFrame,
    fields: dict[str, str],
    adoption: float | None,
    horizon: int,
) -> dict[str, Any]:
    predecessors = set(focal_edges["target"].astype(str))
    pre_edges = predecessor_edges[predecessor_edges["source"].isin(predecessors)]
    future_fields = {str(value) for value in future["citer_primary_field"].dropna()}
    predecessor_fields = {fields[value] for value in predecessors if value in fields}
    focal_field = fields.get(paper_id)
    focal_only, focal_with_predecessor = _dependency_counts(future, predecessors)
    pre_field_edges = _field_edges(pre_edges, fields)
    post_field_edges = [*pre_field_edges]
    post_field_edges.extend(
        (focal_field, fields[target])
        for target in predecessors
        if focal_field is not None and target in fields
    )
    post_field_edges.extend(_future_field_edges(future, fields))
    pre_path = _mean_pair_distance(predecessor_fields, pre_field_edges)
    post_path = _mean_pair_distance(predecessor_fields, post_field_edges)
    return {
        "paper_id": paper_id,
        "horizon": horizon,
        "future_new_community_count": len(future_fields - predecessor_fields),
        "total_future_community_count": len(future_fields),
        "outsider_citer_share": _outsider_share(future, focal_field),
        "pre_cross_community_edge_rate": _cross_rate(pre_field_edges),
        "post_cross_community_edge_rate": _cross_rate(post_field_edges),
        "focal_only_citers": focal_only,
        "focal_and_predecessor_citers": focal_with_predecessor,
        "pre_shortest_path": pre_path,
        "post_shortest_path": post_path,
        "claim_adoption_breadth": adoption,
        "future_citer_count": len(future),
        "operational_definition": "local_primary_field_reference_graph_v1",
    }


def _dependency_counts(future: pd.DataFrame, predecessors: set[str]) -> tuple[int, int]:
    with_predecessor = 0
    for references in future["referenced_works"]:
        if predecessors.intersection(str(value) for value in _references(references)):
            with_predecessor += 1
    return len(future) - with_predecessor, with_predecessor


def _field_edges(frame: pd.DataFrame, fields: dict[str, str]) -> list[tuple[str, str]]:
    return [
        (fields[str(row.source)], fields[str(row.target)])
        for row in frame.itertuples(index=False)
        if str(row.source) in fields and str(row.target) in fields
    ]


def _future_field_edges(
    future: pd.DataFrame, fields: dict[str, str]
) -> list[tuple[str, str]]:
    output: list[tuple[str, str]] = []
    for row in future.itertuples(index=False):
        source = row.citer_primary_field
        if source is None or pd.isna(source):
            continue
        output.extend(
            (str(source), fields[str(target)])
            for target in _references(row.referenced_works)
            if str(target) in fields
        )
    return output


def _references(value: Any) -> list[Any]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)


def _outsider_share(future: pd.DataFrame, focal_field: str | None) -> float:
    values = future["citer_primary_field"].dropna().astype(str)
    if focal_field is None or values.empty:
        return float("nan")
    return float(values.ne(focal_field).mean())


def _cross_rate(edges: list[tuple[str, str]]) -> float:
    if not edges:
        return float("nan")
    return sum(source != target for source, target in edges) / len(edges)


def _mean_pair_distance(communities: set[str], edges: list[tuple[str, str]]) -> float:
    nodes = sorted(communities)
    if len(nodes) < 2:
        return float("nan")
    adjacency: dict[str, set[str]] = {node: set() for node in nodes}
    for source, target in edges:
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)
    penalty = len(adjacency) + 1
    values = [
        _distance(adjacency, source, target, penalty)
        for source, target in combinations(nodes, 2)
    ]
    return float(np.mean(values))


def _distance(
    adjacency: dict[str, set[str]], source: str, target: str, penalty: int
) -> int:
    queue = deque([(source, 0)])
    visited = {source}
    while queue:
        node, distance = queue.popleft()
        if node == target:
            return distance
        for neighbor in adjacency.get(node, set()) - visited:
            visited.add(neighbor)
            queue.append((neighbor, distance + 1))
    return penalty


def _summary(
    frame: pd.DataFrame, target: Path, future_path: Path, horizon: int
) -> dict[str, Any]:
    valid = _complete_graph_rows(frame)
    return {
        "contract": "gear_real_perturbation_inputs_v1",
        "horizon": horizon,
        "papers": len(frame),
        "papers_with_future_citers": int(frame["future_citer_count"].gt(0).sum()),
        "papers_with_complete_graph_inputs": len(valid),
        "papers_with_claim_adoption": int(
            frame["claim_adoption_breadth"].notna().sum()
        ),
        "source_future_citers": str(future_path.resolve()),
        "output_sha256": "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest(),
    }


def _complete_graph_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.dropna(
        subset=[
            "outsider_citer_share",
            "pre_cross_community_edge_rate",
            "post_cross_community_edge_rate",
            "pre_shortest_path",
            "post_shortest_path",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--future-citers", type=Path, required=True)
    parser.add_argument("--reference-edges", type=Path, required=True)
    parser.add_argument("--reference-works", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--claim-adoption", type=Path)
    args = parser.parse_args()
    result = build_inputs(
        args.benchmark_manifest,
        args.future_citers,
        args.reference_edges,
        args.reference_works,
        args.output_dir,
        horizon=args.horizon,
        claim_adoption_path=args.claim_adoption,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_inputs"]

"""Pure publication-time feature functions shared by batch and GEAR runtime.

This module never reads outcomes and requires ``context.source_max_year`` to be
strictly earlier than the target publication year.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations, pairwise
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?")
FULLTEXT16_NUMERIC_FIELDS: Tuple[str, ...] = (
    "EF0017",
    "EF0038",
    "EF0052",
    "EF0083",
    "EF0186",
    "EF0188",
    "EF0238",
    "EF0240",
    "EF0307",
    "EF0309",
    "EF0312",
    "EF0314",
    "EF0315",
    "EF0318",
    "EF0319",
)


@dataclass(frozen=True)
class ReferenceT0:
    reference_id: str
    publication_year: Optional[int] = None
    field_id: Optional[str] = None


@dataclass(frozen=True)
class TargetT0Record:
    paper_id: str
    publication_year: int
    title: str
    author_ids: Tuple[str, ...] = ()
    author_count: Optional[int] = None
    country_codes: Tuple[str, ...] = ()
    metadata_observed: bool = True
    source_id: Optional[str] = None
    references: Tuple[ReferenceT0, ...] = ()


@dataclass
class ContextSnapshot:
    source_max_year: int
    prior_titles: List[str] = field(default_factory=list)
    seen_title_bigrams: Set[Tuple[str, str]] = field(default_factory=set)
    prior_author_adjacency: Dict[str, Set[str]] = field(default_factory=dict)
    prior_coauthor_weights: Dict[Tuple[str, str], int] = field(default_factory=dict)
    prior_paper_reference_ids: List[Set[str]] = field(default_factory=list)
    reference_metadata: Dict[str, ReferenceT0] = field(default_factory=dict)
    bibliographic_coupling_index: Dict[str, Set[str]] = field(default_factory=dict)
    field_distances: Dict[Tuple[str, str], float] = field(default_factory=dict)


@dataclass(frozen=True)
class MaterializationReplayReport:
    row_count: int
    missingness_identical: bool
    categorical_identical: bool
    numeric_within_tolerance: bool
    numeric_max_absolute_error: float
    raw_prediction_within_tolerance: bool
    raw_prediction_max_absolute_error: float
    rtol: float = 1e-7
    atol: float = 1e-9

    @property
    def eligible_inference(self) -> bool:
        return bool(
            self.row_count > 0
            and self.missingness_identical
            and self.categorical_identical
            and self.numeric_within_tolerance
            and self.raw_prediction_within_tolerance
        )


def compute_backward_citation_age(
    publication_year: int,
    reference_years: Sequence[Optional[int]],
) -> Optional[float]:
    eligible = [
        int(year)
        for year in reference_years
        if year is not None and int(year) <= int(publication_year)
    ]
    if not eligible:
        return None
    return float(np.mean([int(publication_year) - year for year in eligible]))


def _gini_balance(counts: Sequence[int]) -> Optional[float]:
    values: np.ndarray = np.asarray(
        [int(value) for value in counts if int(value) > 0], dtype=float
    )
    if not len(values):
        return None
    if len(values) == 1:
        return 1.0
    difference = np.abs(values[:, None] - values[None, :]).sum()
    gini = difference / (2.0 * len(values) * values.sum())
    return float(1.0 - gini)


def _distance(
    left: str,
    right: str,
    distances: Mapping[Tuple[str, str], float],
) -> Optional[float]:
    if left == right:
        return 0.0
    key = (left, right) if left < right else (right, left)
    value = distances.get(key)
    return float(value) if value is not None and math.isfinite(float(value)) else None


def compute_reference_diversity(
    field_ids: Sequence[Optional[str]],
    field_distances: Mapping[Tuple[str, str], float],
) -> Dict[str, Optional[float]]:
    observed = [str(value) for value in field_ids if value]
    if not observed:
        return {
            "reference_variety": None,
            "reference_balance": None,
            "reference_disparity": None,
            "rao_stirling_diversity": None,
            "field_shannon_entropy": None,
            "field_pielou_evenness": None,
        }
    counts = Counter(observed)
    fields = sorted(counts)
    variety = float(len(fields))
    balance = _gini_balance(list(counts.values()))
    probabilities = {field: counts[field] / len(observed) for field in fields}
    entropy = float(
        -sum(value * math.log(value) for value in probabilities.values() if value > 0)
    )
    evenness = float(entropy / math.log(len(fields))) if len(fields) > 1 else None
    if len(fields) < 2:
        disparity = None
        rao = None
    else:
        pair_distances: List[float] = []
        rao_value = 0.0
        complete = True
        for left, right in combinations(fields, 2):
            value = _distance(left, right, field_distances)
            if value is None:
                complete = False
                break
            pair_distances.append(value)
            rao_value += 2.0 * probabilities[left] * probabilities[right] * value
        disparity = float(np.mean(pair_distances)) if complete else None
        rao = float(rao_value) if complete else None
    return {
        "reference_variety": variety,
        "reference_balance": balance,
        "reference_disparity": disparity,
        "rao_stirling_diversity": rao,
        "field_shannon_entropy": entropy,
        "field_pielou_evenness": evenness,
    }


def compute_additive_entropy_diversity(
    entropy: Optional[float],
    evenness: Optional[float],
    disparity: Optional[float],
) -> Optional[float]:
    if entropy is None or evenness is None or disparity is None:
        return None
    return float(entropy + evenness - disparity)


def _title_words(title: str) -> List[str]:
    tokens = TOKEN_PATTERN.findall(str(title or "").casefold())
    return [token for token in tokens if token and not token[0].isdigit()]


def build_context_snapshot(
    records: Sequence[TargetT0Record],
    *,
    target_year: int,
    field_distances: Optional[Mapping[Tuple[str, str], float]] = None,
) -> ContextSnapshot:
    """Build one immutable-by-convention snapshot from strictly earlier papers."""
    prior = sorted(
        (
            record
            for record in records
            if int(record.publication_year) < int(target_year)
        ),
        key=lambda item: (int(item.publication_year), item.paper_id),
    )
    if not prior:
        raise ValueError("ContextSnapshot requires at least one strictly prior record")
    seen_bigrams: Set[Tuple[str, str]] = set()
    adjacency: Dict[str, Set[str]] = {}
    weights: Dict[Tuple[str, str], int] = {}
    paper_references: List[Set[str]] = []
    reference_metadata: Dict[str, ReferenceT0] = {}
    coupling_index: Dict[str, Set[str]] = {}
    for record in prior:
        words = _title_words(record.title)
        seen_bigrams.update(pairwise(words))
        authors = sorted(set(record.author_ids))[:100]
        for left, right in combinations(authors, 2):
            key = (left, right) if left < right else (right, left)
            adjacency.setdefault(left, set()).add(right)
            adjacency.setdefault(right, set()).add(left)
            weights[key] = weights.get(key, 0) + 1
        valid_reference_ids: Set[str] = set()
        for reference in record.references:
            if not reference.reference_id:
                continue
            existing = reference_metadata.get(reference.reference_id)
            reference_metadata[reference.reference_id] = _merge_reference_metadata(
                existing,
                reference,
            )
            if reference.publication_year is not None and int(
                reference.publication_year
            ) < int(record.publication_year):
                valid_reference_ids.add(reference.reference_id)
                coupling_index.setdefault(reference.reference_id, set()).add(
                    record.paper_id
                )
        paper_references.append(valid_reference_ids)
    normalized_distances = {
        ((left, right) if left < right else (right, left)): float(value)
        for (left, right), value in (field_distances or {}).items()
    }
    return ContextSnapshot(
        source_max_year=max(int(record.publication_year) for record in prior),
        prior_titles=[record.title for record in prior],
        seen_title_bigrams=seen_bigrams,
        prior_author_adjacency=adjacency,
        prior_coauthor_weights=weights,
        prior_paper_reference_ids=paper_references,
        reference_metadata=reference_metadata,
        bibliographic_coupling_index=coupling_index,
        field_distances=normalized_distances,
    )


def _merge_reference_metadata(
    existing: Optional[ReferenceT0],
    incoming: ReferenceT0,
) -> ReferenceT0:
    if existing is None:
        return incoming
    years = {
        int(value)
        for value in (existing.publication_year, incoming.publication_year)
        if value is not None
    }
    fields = {
        str(value)
        for value in (existing.field_id, incoming.field_id)
        if value is not None and str(value)
    }
    if len(years) > 1 or len(fields) > 1:
        raise ValueError(f"conflicting reference metadata for {incoming.reference_id}")
    return ReferenceT0(
        reference_id=incoming.reference_id,
        publication_year=next(iter(years), None),
        field_id=next(iter(fields), None),
    )


def context_snapshot_sha256(snapshot: ContextSnapshot) -> str:
    payload = _snapshot_payload(snapshot)
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def save_context_snapshot(snapshot: ContextSnapshot, path: Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _snapshot_payload(snapshot)
    payload["snapshot_sha256"] = context_snapshot_sha256(snapshot)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def load_context_snapshot(path: Path) -> ContextSnapshot:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = str(payload.pop("snapshot_sha256", ""))
    snapshot = _snapshot_from_payload(payload)
    if expected != context_snapshot_sha256(snapshot):
        raise ValueError("ContextSnapshot hash mismatch")
    return snapshot


def _snapshot_payload(snapshot: ContextSnapshot) -> Dict[str, Any]:
    return {
        "schema_version": "aspr_context_snapshot_v1",
        "source_max_year": int(snapshot.source_max_year),
        "prior_titles": list(snapshot.prior_titles),
        "seen_title_bigrams": [
            list(item) for item in sorted(snapshot.seen_title_bigrams)
        ],
        "prior_author_adjacency": {
            key: sorted(values)
            for key, values in sorted(snapshot.prior_author_adjacency.items())
        },
        "prior_coauthor_weights": [
            [left, right, int(value)]
            for (left, right), value in sorted(snapshot.prior_coauthor_weights.items())
        ],
        "prior_paper_reference_ids": [
            sorted(values) for values in snapshot.prior_paper_reference_ids
        ],
        "reference_metadata": {
            key: {
                "publication_year": value.publication_year,
                "field_id": value.field_id,
            }
            for key, value in sorted(snapshot.reference_metadata.items())
        },
        "bibliographic_coupling_index": {
            key: sorted(values)
            for key, values in sorted(snapshot.bibliographic_coupling_index.items())
        },
        "field_distances": [
            [left, right, float(value)]
            for (left, right), value in sorted(snapshot.field_distances.items())
        ],
    }


def _snapshot_from_payload(payload: Mapping[str, Any]) -> ContextSnapshot:
    if payload.get("schema_version") != "aspr_context_snapshot_v1":
        raise ValueError("unsupported ContextSnapshot schema")
    return ContextSnapshot(
        source_max_year=int(payload["source_max_year"]),
        prior_titles=[str(value) for value in payload.get("prior_titles", [])],
        seen_title_bigrams={
            (str(item[0]), str(item[1]))
            for item in payload.get("seen_title_bigrams", [])
        },
        prior_author_adjacency={
            str(key): {str(value) for value in values}
            for key, values in payload.get("prior_author_adjacency", {}).items()
        },
        prior_coauthor_weights={
            (str(item[0]), str(item[1])): int(item[2])
            for item in payload.get("prior_coauthor_weights", [])
        },
        prior_paper_reference_ids=[
            {str(value) for value in values}
            for values in payload.get("prior_paper_reference_ids", [])
        ],
        reference_metadata={
            str(key): ReferenceT0(
                reference_id=str(key),
                publication_year=(
                    int(value["publication_year"])
                    if value.get("publication_year") is not None
                    else None
                ),
                field_id=(
                    str(value["field_id"])
                    if value.get("field_id") is not None
                    else None
                ),
            )
            for key, value in payload.get("reference_metadata", {}).items()
        },
        bibliographic_coupling_index={
            str(key): {str(value) for value in values}
            for key, values in payload.get("bibliographic_coupling_index", {}).items()
        },
        field_distances={
            (str(item[0]), str(item[1])): float(item[2])
            for item in payload.get("field_distances", [])
        },
    )


def compute_title_novelty(
    title: str,
    seen_bigrams: Set[Tuple[str, str]],
) -> Dict[str, float]:
    words = _title_words(title)
    bigrams = list(pairwise(words))
    return {
        "title_new_bigram_share": (
            float(sum(item not in seen_bigrams for item in bigrams) / len(bigrams))
            if bigrams
            else 0.0
        )
    }


def _component_sizes(binary: np.ndarray) -> List[int]:
    remaining = set(range(len(binary)))
    sizes: List[int] = []
    while remaining:
        stack = [remaining.pop()]
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            neighbors = set(np.flatnonzero(binary[node]).tolist()) & remaining
            remaining.difference_update(neighbors)
            stack.extend(neighbors)
        sizes.append(size)
    return sizes or [0]


def compute_prior_team_graph(
    authors: Sequence[str],
    adjacency: Mapping[str, Set[str]],
    weights: Mapping[Tuple[str, str], int],
) -> Dict[str, float]:
    """Exact pure implementation used by the frozen batch surrogate."""
    selected = sorted({str(author) for author in authors if str(author)})[:100]
    n_authors = len(selected)
    if n_authors < 2:
        return {
            "prior_team_edge_count": 0.0,
            "prior_team_edge_density": 0.0,
            "prior_team_edge_strength": 0.0,
            "prior_team_giant_component_share": 1.0 if n_authors else np.nan,
            "prior_team_mean_clustering": 0.0 if n_authors else np.nan,
            "prior_team_relative_algebraic_connectivity": 0.0 if n_authors else np.nan,
            "prior_author_degree_mean": (
                float(
                    np.mean([len(adjacency.get(author, set())) for author in selected])
                )
                if selected
                else np.nan
            ),
            "prior_team_graph_truncated": float(len(set(authors)) > 100),
        }
    matrix: np.ndarray = np.zeros((n_authors, n_authors), dtype=np.float64)
    edge_strength = 0.0
    for left_index, right_index in combinations(range(n_authors), 2):
        left, right = selected[left_index], selected[right_index]
        key = (left, right) if left < right else (right, left)
        weight = float(weights.get(key, 0))
        if weight > 0:
            matrix[left_index, right_index] = weight
            matrix[right_index, left_index] = weight
            edge_strength += weight
    binary = (matrix > 0).astype(np.float64)
    edge_count = float(binary.sum() / 2.0)
    possible = n_authors * (n_authors - 1) / 2.0
    degrees = binary.sum(axis=1)
    triangles = np.diag(binary @ binary @ binary) / 2.0
    local = np.divide(
        2.0 * triangles,
        degrees * (degrees - 1.0),
        out=np.zeros_like(degrees),
        where=degrees >= 2,
    )
    components = _component_sizes(binary)
    laplacian = np.diag(matrix.sum(axis=1)) - matrix
    eigenvalues = np.linalg.eigvalsh(laplacian)
    denominator = float(eigenvalues[-1]) if len(eigenvalues) else 0.0
    relative = (
        float(max(eigenvalues[1], 0.0) / denominator)
        if len(eigenvalues) > 1 and denominator > 0
        else 0.0
    )
    return {
        "prior_team_edge_count": edge_count,
        "prior_team_edge_density": float(edge_count / possible),
        "prior_team_edge_strength": float(edge_strength / possible),
        "prior_team_giant_component_share": float(max(components) / n_authors),
        "prior_team_mean_clustering": float(local.mean()),
        "prior_team_relative_algebraic_connectivity": relative,
        "prior_author_degree_mean": float(
            np.mean([len(adjacency.get(author, set())) for author in selected])
        ),
        "prior_team_graph_truncated": float(len(set(authors)) > 100),
    }


def compute_bibliographic_coupling(
    focal_reference_ids: Sequence[str],
    prior_paper_reference_ids: Sequence[Set[str]] = (),
    *,
    bibliographic_coupling_index: Optional[Mapping[str, Set[str]]] = None,
) -> Optional[float]:
    focal = {str(value) for value in focal_reference_ids if str(value)}
    if len(focal) < 2:
        return None
    if bibliographic_coupling_index is not None:
        neighbor_ids: Set[str] = set()
        for reference_id in focal:
            neighbor_ids.update(bibliographic_coupling_index.get(reference_id, set()))
        neighbors = len(neighbor_ids)
    else:
        neighbors = sum(
            bool(focal & set(references)) for references in prior_paper_reference_ids
        )
    return float(neighbors / len(focal))


def _resolve_target_references(
    target: TargetT0Record,
    context: ContextSnapshot,
) -> List[ReferenceT0]:
    resolved: List[ReferenceT0] = []
    for reference in target.references:
        stored = context.reference_metadata.get(reference.reference_id)
        resolved.append(
            ReferenceT0(
                reference_id=reference.reference_id,
                publication_year=(
                    reference.publication_year
                    if reference.publication_year is not None
                    else (stored.publication_year if stored is not None else None)
                ),
                field_id=(
                    reference.field_id
                    if reference.field_id is not None
                    else (stored.field_id if stored is not None else None)
                ),
            )
        )
    return resolved


def materialize_fulltext16(
    target: TargetT0Record,
    context: ContextSnapshot,
    *,
    include_journal_identity: bool = True,
) -> Dict[str, object]:
    if int(context.source_max_year) >= int(target.publication_year):
        raise ValueError("ContextSnapshot must end before the target publication year")
    resolved_references = _resolve_target_references(target, context)
    valid_references = [
        reference
        for reference in resolved_references
        if reference.reference_id
        and reference.publication_year is not None
        and int(reference.publication_year) < int(target.publication_year)
    ]
    reference_fields = [reference.field_id for reference in valid_references]
    diversity = compute_reference_diversity(reference_fields, context.field_distances)
    graph = compute_prior_team_graph(
        target.author_ids,
        context.prior_author_adjacency,
        context.prior_coauthor_weights,
    )
    title = compute_title_novelty(target.title, context.seen_title_bigrams)
    country_codes = sorted({value for value in target.country_codes if value})
    additive = compute_additive_entropy_diversity(
        diversity["field_shannon_entropy"],
        diversity["field_pielou_evenness"],
        diversity["reference_disparity"],
    )
    return {
        "EF0017": additive,
        "EF0038": (
            float(target.author_count)
            if target.author_count is not None and int(target.author_count) > 0
            else float(len(set(target.author_ids))) if target.author_ids else None
        ),
        "EF0052": compute_backward_citation_age(
            target.publication_year,
            [reference.publication_year for reference in resolved_references],
        ),
        "EF0083": graph["prior_team_mean_clustering"],
        # The frozen Fig.3 matrix encodes missing/empty country lists as the
        # negative class for the binary international-collaboration flag while
        # retaining missingness for the numeric country count (EF0188).
        "EF0186": (float(len(country_codes) > 1) if target.metadata_observed else None),
        "EF0188": float(len(country_codes)) if country_codes else None,
        "EF0197": target.source_id if include_journal_identity else None,
        "EF0238": compute_bibliographic_coupling(
            [reference.reference_id for reference in valid_references],
            context.prior_paper_reference_ids,
            bibliographic_coupling_index=(context.bibliographic_coupling_index or None),
        ),
        "EF0240": title["title_new_bigram_share"],
        "EF0307": float(target.publication_year),
        "EF0309": diversity["rao_stirling_diversity"],
        "EF0312": diversity["reference_balance"],
        "EF0314": float(len(valid_references)),
        "EF0315": diversity["reference_disparity"],
        "EF0318": diversity["reference_variety"],
        "EF0319": graph["prior_team_relative_algebraic_connectivity"],
    }


def coerce_fulltext16_storage_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen Fig.3 matrix storage schema before HGB inference.

    The official matrix persists numeric features as float32.  Replaying the
    mathematically identical values as float64 can place a tiny number of rows
    on the opposite side of a fitted tree threshold, so schema coercion is part
    of the model contract rather than a cosmetic serialization choice.
    """
    output = frame.copy()
    missing = sorted(set(FULLTEXT16_NUMERIC_FIELDS) - set(output.columns))
    if missing:
        raise ValueError(f"Full-text 16 frame is missing numeric fields: {missing}")
    for name in FULLTEXT16_NUMERIC_FIELDS:
        output[name] = pd.to_numeric(output[name], errors="coerce").astype("float32")
    if "EF0197" in output:
        output["EF0197"] = output["EF0197"].astype("string")
    return output


def validate_materialization_replay(
    runtime_matrix: pd.DataFrame,
    frozen_matrix: pd.DataFrame,
    *,
    prediction_func: Optional[Callable[[pd.DataFrame], Any]] = None,
    rtol: float = 1e-7,
    atol: float = 1e-9,
) -> MaterializationReplayReport:
    """Validate the predeclared online-materialization promotion gate."""
    feature_names = tuple(
        f"EF{value:04d}"
        for value in (
            17,
            38,
            52,
            83,
            186,
            188,
            197,
            238,
            240,
            307,
            309,
            312,
            314,
            315,
            318,
            319,
        )
    )
    required = {"paper_id", *feature_names}
    if not required.issubset(runtime_matrix.columns) or not required.issubset(
        frozen_matrix.columns
    ):
        raise ValueError("replay matrices must contain paper_id and Full-text 16")
    if (
        runtime_matrix["paper_id"].duplicated().any()
        or frozen_matrix["paper_id"].duplicated().any()
    ):
        raise ValueError("replay matrices require unique paper IDs")
    runtime = runtime_matrix.set_index("paper_id").sort_index()
    frozen = frozen_matrix.set_index("paper_id").sort_index()
    if not runtime.index.equals(frozen.index):
        raise ValueError("runtime and frozen replay paper IDs differ")
    runtime = runtime.loc[:, feature_names]
    frozen = frozen.loc[:, feature_names]
    missingness_identical = bool(runtime.isna().equals(frozen.isna()))
    runtime_category = runtime["EF0197"].astype("string").fillna("<missing>")
    frozen_category = frozen["EF0197"].astype("string").fillna("<missing>")
    categorical_identical = bool(runtime_category.equals(frozen_category))
    numeric_names = [name for name in feature_names if name != "EF0197"]
    runtime_numeric = runtime[numeric_names].apply(pd.to_numeric, errors="coerce")
    frozen_numeric = frozen[numeric_names].apply(pd.to_numeric, errors="coerce")
    numeric_close = np.isclose(
        runtime_numeric.to_numpy(dtype=float),
        frozen_numeric.to_numpy(dtype=float),
        rtol=float(rtol),
        atol=float(atol),
        equal_nan=True,
    )
    numeric_within = bool(numeric_close.all())
    differences = np.abs(
        runtime_numeric.to_numpy(dtype=float) - frozen_numeric.to_numpy(dtype=float)
    )
    finite_differences = differences[np.isfinite(differences)]
    numeric_maximum = (
        float(finite_differences.max()) if len(finite_differences) else 0.0
    )
    prediction_within = False
    prediction_maximum = float("inf")
    if prediction_func is not None:
        runtime_prediction = np.asarray(prediction_func(runtime), dtype=float)
        frozen_prediction = np.asarray(prediction_func(frozen), dtype=float)
        if runtime_prediction.shape != frozen_prediction.shape:
            raise ValueError("replay prediction shapes differ")
        prediction_differences = np.abs(runtime_prediction - frozen_prediction)
        finite_prediction = prediction_differences[np.isfinite(prediction_differences)]
        prediction_maximum = (
            float(finite_prediction.max()) if len(finite_prediction) else 0.0
        )
        prediction_within = bool(
            np.isclose(
                runtime_prediction,
                frozen_prediction,
                rtol=float(rtol),
                atol=float(atol),
                equal_nan=True,
            ).all()
        )
    return MaterializationReplayReport(
        row_count=len(runtime),
        missingness_identical=missingness_identical,
        categorical_identical=categorical_identical,
        numeric_within_tolerance=numeric_within,
        numeric_max_absolute_error=numeric_maximum,
        raw_prediction_within_tolerance=prediction_within,
        raw_prediction_max_absolute_error=prediction_maximum,
        rtol=float(rtol),
        atol=float(atol),
    )


__all__ = [
    "ContextSnapshot",
    "MaterializationReplayReport",
    "ReferenceT0",
    "TargetT0Record",
    "build_context_snapshot",
    "compute_additive_entropy_diversity",
    "compute_backward_citation_age",
    "compute_bibliographic_coupling",
    "compute_prior_team_graph",
    "compute_reference_diversity",
    "compute_title_novelty",
    "coerce_fulltext16_storage_schema",
    "context_snapshot_sha256",
    "load_context_snapshot",
    "materialize_fulltext16",
    "save_context_snapshot",
    "validate_materialization_replay",
]

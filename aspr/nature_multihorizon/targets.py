"""Multi-horizon diffusion targets and optional structural validation targets."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from aspr.corpus import normalize_openalex_id


TARGET_DEFINITION_VERSION = "nature-multihorizon-target-v1"
SUCCESS_STATUSES = frozenset({"success", "fetched", "checkpoint", "zero_success"})
DIFFUSION_TARGET_COMPONENTS = (
    "future_field_reach",
    "future_subfield_reach",
    "future_topic_reach",
    "future_field_simpson",
    "future_topic_simpson",
)


class FoldLocalDiffusionTarget:
    """Fit RGPM-D component percentile references on training labels only."""

    def fit(self, frame: pd.DataFrame) -> "FoldLocalDiffusionTarget":
        missing = sorted(set(DIFFUSION_TARGET_COMPONENTS) - set(frame.columns))
        if missing:
            raise ValueError(f"Diffusion target components are missing: {missing}")
        self.references_: Dict[str, np.ndarray] = {}
        for name in DIFFUSION_TARGET_COMPONENTS:
            values = pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
            if name.endswith("_reach"):
                values = np.log1p(values)
            reference = np.sort(values[np.isfinite(values)])
            if len(reference) < 2:
                raise ValueError(f"Need at least two finite training values for {name}")
            self.references_[name] = reference
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "references_"):
            raise RuntimeError("FoldLocalDiffusionTarget must be fitted first")
        ranked: List[np.ndarray] = []
        for name in DIFFUSION_TARGET_COMPONENTS:
            values = pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
            if name.endswith("_reach"):
                values = np.log1p(values)
            reference = self.references_[name]
            left = np.searchsorted(reference, values, side="left")
            right = np.searchsorted(reference, values, side="right")
            ranks = right.astype(float) / len(reference)
            tied = right > left
            # Match pandas ``rank(method='average', pct=True)`` for values
            # present in the training reference; discrete reach/evenness
            # components contain many ties.
            ranks[tied] = (left[tied] + right[tied] + 1.0) / (
                2.0 * len(reference)
            )
            ranks[~np.isfinite(values)] = np.nan
            ranked.append(ranks)
        matrix = np.column_stack(ranked)
        breadth = np.mean(matrix[:, :3], axis=1)
        evenness = np.mean(matrix[:, 3:], axis=1)
        return 0.5 * breadth + 0.5 * evenness

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)


def _simpson(values: Sequence[str]) -> float:
    if not values:
        return 0.0
    counts = pd.Series(values).value_counts().to_numpy(dtype=float)
    probabilities = counts / counts.sum()
    return float(1.0 - np.sum(probabilities**2))


def _rank_within_group(
    frame: pd.DataFrame, column: str, group_columns: Sequence[str]
) -> pd.Series:
    return frame.groupby(list(group_columns), dropna=False)[column].rank(
        method="average", pct=True
    )


def _descriptive_adjustment(
    frame: pd.DataFrame, *, include_domain_year: bool = False
) -> pd.Series:
    """Residual rank for description only; OOF adjustment remains model-side."""

    valid = frame["rgpm_d_raw"].notna() & frame["n_future_citers"].notna()
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    if valid.sum() < 5:
        return result
    selected = frame.loc[valid]
    numeric = pd.DataFrame(
        {
            "intercept": 1.0,
            "log_future_citers": np.log1p(selected["n_future_citers"].astype(float)),
        },
        index=selected.index,
    )
    parts = [numeric]
    if include_domain_year:
        parts.append(
            pd.get_dummies(
                selected[["domain12", "publication_year_bin"]].astype(str),
                drop_first=True,
                dtype=float,
            )
        )
    design = pd.concat(parts, axis=1).to_numpy(dtype=float)
    target = selected["rgpm_d_raw"].to_numpy(dtype=float)
    penalty = 1e-6 * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    beta = np.linalg.pinv(design.T @ design + penalty) @ design.T @ target
    residual = pd.Series(target - design @ beta, index=selected.index)
    result.loc[selected.index] = residual.rank(method="average", pct=True)
    return result


def build_future_fetch_status(
    papers: pd.DataFrame,
    future_citers: pd.DataFrame,
    *,
    requested_horizon: int = 8,
    explicit_status: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Create an explicit status row for every requested paper.

    Empty successful requests must be supplied through ``explicit_status`` as
    ``zero_success`` (or success with ``n_returned=0``).  Missing status is
    conservatively marked ``not_requested_or_failed`` and never converted to a
    true zero.
    """

    paper_ids = papers["paper_id"].map(normalize_openalex_id).drop_duplicates()
    counts: Dict[str, int] = {}
    observed_status: Dict[str, str] = {}
    if not future_citers.empty:
        citers = future_citers.copy()
        citers["paper_id"] = citers["paper_id"].map(normalize_openalex_id)
        counts = citers.groupby("paper_id")["citer_id"].nunique().astype(int).to_dict()
        if "fetch_status" in citers:
            observed_status = (
                citers.groupby("paper_id")["fetch_status"].first().fillna("").astype(str).to_dict()
            )
    explicit_lookup: Dict[str, Mapping[str, Any]] = {}
    if explicit_status is not None and not explicit_status.empty:
        status_frame = explicit_status.copy()
        status_frame["paper_id"] = status_frame["paper_id"].map(normalize_openalex_id)
        if "requested_horizon" in status_frame:
            status_frame = status_frame[
                pd.to_numeric(status_frame["requested_horizon"], errors="coerce")
                == int(requested_horizon)
            ]
        explicit_lookup = {
            str(row["paper_id"]): row for row in status_frame.to_dict("records")
        }

    rows: List[Dict[str, Any]] = []
    for paper_id in paper_ids:
        explicit = explicit_lookup.get(str(paper_id), {})
        status = str(explicit.get("fetch_status") or observed_status.get(str(paper_id), ""))
        n_returned_value = explicit.get("n_returned", counts.get(str(paper_id)))
        if status.startswith("fetch_failed") or status in {"failed", "error"}:
            normalized_status = "failed"
            n_returned = np.nan
        elif status in SUCCESS_STATUSES or str(paper_id) in counts:
            n_returned = int(n_returned_value or 0)
            normalized_status = "zero_success" if n_returned == 0 else "success"
        else:
            normalized_status = "not_requested_or_failed"
            n_returned = np.nan
        rows.append(
            {
                "paper_id": paper_id,
                "requested_horizon": int(requested_horizon),
                "fetch_status": normalized_status,
                "n_returned": n_returned,
                "is_zero_success": int(normalized_status == "zero_success"),
                "cap_hit": int(bool(explicit.get("cap_hit", False))),
                "last_citer_year": explicit.get("last_citer_year", np.nan),
                "error_type": explicit.get("error_type", ""),
                "attempt_count": int(explicit.get("attempt_count", 1 if status else 0)),
            }
        )
    return pd.DataFrame(rows)


def build_diffusion_targets(
    papers: pd.DataFrame,
    future_citers: pd.DataFrame,
    future_fetch_status: pd.DataFrame,
    *,
    horizons: Sequence[int] = (3, 5, 8),
    min_future_citers: int = 10,
    min_taxonomy_coverage: float = 0.80,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Build RGPM-D3/D5/D8 on the locked future-adoption cohort.

    Component ranks are fitted only among papers meeting the outcome-side
    ``future_citers >= min_future_citers`` rule. Missing OpenAlex taxonomy is
    represented as missing target evidence, never as zero reach/evenness.
    """

    if min_future_citers < 1:
        raise ValueError("min_future_citers must be positive")
    if not 0.0 <= float(min_taxonomy_coverage) <= 1.0:
        raise ValueError("min_taxonomy_coverage must be in [0, 1]")

    paper_columns = ["paper_id", "publication_year", "domain12"]
    missing = set(paper_columns) - set(papers)
    if missing:
        raise ValueError(f"papers is missing target columns: {sorted(missing)}")
    paper_frame = papers[paper_columns].copy()
    paper_frame["paper_id"] = paper_frame["paper_id"].map(normalize_openalex_id)
    paper_frame["publication_year"] = pd.to_numeric(
        paper_frame["publication_year"], errors="coerce"
    )
    citer_frame = future_citers.copy()
    if not citer_frame.empty:
        citer_frame["paper_id"] = citer_frame["paper_id"].map(normalize_openalex_id)
        citer_frame["citer_year"] = pd.to_numeric(citer_frame["citer_year"], errors="coerce")
        if "requested_horizon" not in citer_frame:
            citer_frame["requested_horizon"] = max(map(int, horizons))
    status_frame = future_fetch_status.copy()
    status_frame["paper_id"] = status_frame["paper_id"].map(normalize_openalex_id)
    status_frame["requested_horizon"] = pd.to_numeric(
        status_frame["requested_horizon"], errors="coerce"
    )

    rows: List[Dict[str, Any]] = []
    grouped_citers = {
        paper_id: group for paper_id, group in citer_frame.groupby("paper_id", sort=False)
    } if not citer_frame.empty else {}
    for paper in paper_frame.to_dict("records"):
        paper_id = str(paper["paper_id"])
        year_value = pd.to_numeric(paper["publication_year"], errors="coerce")
        if pd.isna(year_value):
            continue
        publication_year = int(year_value)
        paper_status = status_frame[status_frame["paper_id"] == paper_id]
        paper_citers = grouped_citers.get(paper_id, pd.DataFrame())
        for horizon in sorted(set(map(int, horizons))):
            covering = paper_status[paper_status["requested_horizon"] >= horizon]
            if covering.empty:
                fetch_status = "not_requested_or_failed"
                cap_hit = 0
                requested_cap_hit = 0
                valid_fetch = False
            else:
                chosen = covering.sort_values("requested_horizon").iloc[0]
                fetch_status = str(chosen.get("fetch_status", "not_requested_or_failed"))
                requested_cap_hit = int(chosen.get("cap_hit", 0) or 0)
                last_citer_year = pd.to_numeric(
                    chosen.get("last_citer_year"), errors="coerce"
                )
                # Fetches are sorted chronologically. A capped tau8 request is
                # still complete for tau3/tau5 when its 1,000th returned citer
                # occurs strictly after that shorter cutoff.
                cap_hit = int(
                    requested_cap_hit
                    and (
                        pd.isna(last_citer_year)
                        or int(last_citer_year) <= publication_year + horizon
                    )
                )
                valid_fetch = fetch_status in SUCCESS_STATUSES

            if valid_fetch and not paper_citers.empty:
                in_window = paper_citers[
                    (paper_citers["citer_year"] >= publication_year + 1)
                    & (paper_citers["citer_year"] <= publication_year + horizon)
                ].drop_duplicates("citer_id")
            else:
                in_window = pd.DataFrame(columns=paper_citers.columns)
            n_future = int(len(in_window)) if valid_fetch else np.nan

            def values(column: str) -> List[str]:
                if column not in in_window:
                    return []
                return [
                    str(value)
                    for value in in_window[column].dropna().astype(str)
                    if str(value).strip()
                ]

            fields = values("citer_primary_field")
            subfields = values("citer_primary_subfield")
            topics = values("citer_primary_topic")
            def taxonomy_coverage(items: Sequence[str]) -> float:
                if not valid_fetch:
                    return float("nan")
                if n_future == 0:
                    return 1.0
                return float(len(items) / n_future)

            field_coverage = taxonomy_coverage(fields)
            subfield_coverage = taxonomy_coverage(subfields)
            topic_coverage = taxonomy_coverage(topics)
            taxonomy_ok = bool(
                valid_fetch
                and min(field_coverage, subfield_coverage, topic_coverage)
                >= float(min_taxonomy_coverage)
            )

            def reach(items: Sequence[str]) -> float:
                if not valid_fetch or (n_future > 0 and not items):
                    return float("nan")
                return float(len(set(items)))

            def evenness(items: Sequence[str]) -> float:
                if not valid_fetch or (n_future > 0 and not items):
                    return float("nan")
                return _simpson(items)

            rows.append(
                {
                    "paper_id": paper_id,
                    "publication_year": publication_year,
                    "publication_year_bin": publication_year // 5 * 5,
                    "domain12": paper["domain12"],
                    "horizon": horizon,
                    "fetch_status": fetch_status,
                    "fetch_valid": int(valid_fetch),
                    "target_valid": int(taxonomy_ok),
                    "target_rank_eligible": int(
                        taxonomy_ok and n_future >= int(min_future_citers)
                    ),
                    "cap_hit": cap_hit,
                    "requested_horizon_cap_hit": requested_cap_hit,
                    "n_future_citers": n_future,
                    "future_field_valid_n": len(fields) if valid_fetch else np.nan,
                    "future_subfield_valid_n": len(subfields) if valid_fetch else np.nan,
                    "future_topic_valid_n": len(topics) if valid_fetch else np.nan,
                    "future_field_coverage": field_coverage,
                    "future_subfield_coverage": subfield_coverage,
                    "future_topic_coverage": topic_coverage,
                    "future_field_reach": reach(fields),
                    "future_subfield_reach": reach(subfields),
                    "future_topic_reach": reach(topics),
                    "future_field_simpson": evenness(fields),
                    "future_topic_simpson": evenness(topics),
                }
            )

    targets = pd.DataFrame(rows)
    valid = targets["target_rank_eligible"] == 1
    # The scientific target is ranked once over the complete horizon cohort.
    # Domain/year conditioning belongs to evaluation, not target construction.
    group_columns = ["horizon"]
    breadth_components: List[str] = []
    for column in ("future_field_reach", "future_subfield_reach", "future_topic_reach"):
        logged = f"__log_{column}"
        targets[logged] = np.log1p(pd.to_numeric(targets[column], errors="coerce"))
        ranked = f"__rank_{column}"
        targets[ranked] = np.nan
        targets.loc[valid, ranked] = _rank_within_group(
            targets.loc[valid], logged, group_columns
        )
        breadth_components.append(ranked)
    evenness_components: List[str] = []
    for column in ("future_field_simpson", "future_topic_simpson"):
        ranked = f"__rank_{column}"
        targets[ranked] = np.nan
        targets.loc[valid, ranked] = _rank_within_group(
            targets.loc[valid], column, group_columns
        )
        evenness_components.append(ranked)
    targets["rgpm_d_breadth"] = targets[breadth_components].mean(axis=1)
    targets["rgpm_d_evenness"] = targets[evenness_components].mean(axis=1)
    targets["rgpm_d_raw"] = 0.5 * targets["rgpm_d_breadth"] + 0.5 * targets["rgpm_d_evenness"]
    targets.loc[~valid, ["rgpm_d_breadth", "rgpm_d_evenness", "rgpm_d_raw"]] = np.nan
    targets["rgpm_d_adjusted_full_descriptive"] = np.nan
    targets["rgpm_d_adjusted_domain_year_sensitivity"] = np.nan
    for _, horizon_rows in targets.groupby("horizon", sort=True):
        indices = horizon_rows.index
        targets.loc[indices, "rgpm_d_adjusted_full_descriptive"] = (
            _descriptive_adjustment(horizon_rows).to_numpy(float)
        )
        targets.loc[indices, "rgpm_d_adjusted_domain_year_sensitivity"] = (
            _descriptive_adjustment(
                horizon_rows, include_domain_year=True
            ).to_numpy(float)
        )
    targets["target_name"] = targets["horizon"].map(lambda value: f"RGPM-D{int(value)}")
    targets["definition_version"] = TARGET_DEFINITION_VERSION
    targets["minimum_future_citers_definition"] = int(min_future_citers)
    targets["minimum_taxonomy_coverage_definition"] = float(min_taxonomy_coverage)
    targets = targets.drop(
        columns=[
            column
            for column in targets
            if column.startswith("__log_") or column.startswith("__rank_")
        ]
    )
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        targets.to_parquet(output_path, index=False)
    return targets


def build_diffusion_targets_from_deltas(
    papers: pd.DataFrame,
    future_deltas: pd.DataFrame,
    *,
    horizons: Sequence[int] = (3, 5, 8),
    min_future_citers: int = 10,
    min_taxonomy_coverage: float = 0.80,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Build RGPM-D directly from audited precomputed horizon components.

    The offline V5 materializer has already streamed 22M+ future-citer rows
    into exact field/subfield/topic reach and Simpson components.  Reusing
    those components avoids loading the large citer bibliography table into
    memory while retaining the same horizon-global target definition.
    """

    required = {
        "paper_id",
        "publication_year",
        "horizon",
        "fetch_status",
        "fetch_valid",
        "cap_hit",
        "requested_horizon_cap_hit",
        "n_future_citers",
        "future_field_coverage",
        "future_subfield_coverage",
        "future_topic_coverage",
        *DIFFUSION_TARGET_COMPONENTS,
    }
    missing = sorted(required - set(future_deltas.columns))
    if missing:
        raise ValueError(f"Precomputed future deltas are missing columns: {missing}")
    if "paper_id" not in papers or "domain12" not in papers:
        raise ValueError("papers must contain paper_id and domain12")
    selected_horizons = tuple(sorted(set(int(value) for value in horizons)))
    targets = future_deltas[
        pd.to_numeric(future_deltas["horizon"], errors="coerce").isin(
            selected_horizons
        )
    ].copy()
    targets["paper_id"] = targets["paper_id"].map(normalize_openalex_id)
    targets["horizon"] = pd.to_numeric(
        targets["horizon"], errors="raise"
    ).astype(int)
    if targets.duplicated(["paper_id", "horizon"]).any():
        raise ValueError("Precomputed future deltas contain duplicate paper/horizon keys")
    paper_domains = papers[["paper_id", "domain12"]].copy()
    paper_domains["paper_id"] = paper_domains["paper_id"].map(
        normalize_openalex_id
    )
    if paper_domains["paper_id"].duplicated().any():
        raise ValueError("papers contains duplicate paper_id rows")
    targets = targets.drop(columns=["domain12"], errors="ignore").merge(
        paper_domains,
        on="paper_id",
        how="left",
        validate="many_to_one",
    )
    targets["publication_year"] = pd.to_numeric(
        targets["publication_year"], errors="coerce"
    )
    targets["publication_year_bin"] = (
        targets["publication_year"].floordiv(5).mul(5).astype("Int64")
    )
    fetch_valid = (
        pd.to_numeric(targets["fetch_valid"], errors="coerce")
        .fillna(0)
        .astype(bool)
    )
    coverages = targets[
        [
            "future_field_coverage",
            "future_subfield_coverage",
            "future_topic_coverage",
        ]
    ].apply(pd.to_numeric, errors="coerce")
    taxonomy_ok = fetch_valid & coverages.min(axis=1).ge(
        float(min_taxonomy_coverage)
    )
    n_future = pd.to_numeric(targets["n_future_citers"], errors="coerce")
    targets["target_valid"] = taxonomy_ok.astype(int)
    targets["target_rank_eligible"] = (
        taxonomy_ok & n_future.ge(int(min_future_citers))
    ).astype(int)
    valid = targets["target_rank_eligible"].eq(1)

    breadth_components: List[str] = []
    for column in (
        "future_field_reach",
        "future_subfield_reach",
        "future_topic_reach",
    ):
        logged = f"__log_{column}"
        targets[logged] = np.log1p(pd.to_numeric(targets[column], errors="coerce"))
        ranked = f"__rank_{column}"
        targets[ranked] = np.nan
        targets.loc[valid, ranked] = _rank_within_group(
            targets.loc[valid], logged, ["horizon"]
        )
        breadth_components.append(ranked)
    evenness_components: List[str] = []
    for column in ("future_field_simpson", "future_topic_simpson"):
        ranked = f"__rank_{column}"
        targets[ranked] = np.nan
        targets.loc[valid, ranked] = _rank_within_group(
            targets.loc[valid], column, ["horizon"]
        )
        evenness_components.append(ranked)
    targets["rgpm_d_breadth"] = targets[breadth_components].mean(axis=1)
    targets["rgpm_d_evenness"] = targets[evenness_components].mean(axis=1)
    targets["rgpm_d_raw"] = (
        0.5 * targets["rgpm_d_breadth"]
        + 0.5 * targets["rgpm_d_evenness"]
    )
    targets.loc[
        ~valid, ["rgpm_d_breadth", "rgpm_d_evenness", "rgpm_d_raw"]
    ] = np.nan
    targets["rgpm_d_adjusted_full_descriptive"] = np.nan
    targets["rgpm_d_adjusted_domain_year_sensitivity"] = np.nan
    for _, horizon_rows in targets.groupby("horizon", sort=True):
        indices = horizon_rows.index
        targets.loc[indices, "rgpm_d_adjusted_full_descriptive"] = (
            _descriptive_adjustment(horizon_rows).to_numpy(float)
        )
        targets.loc[indices, "rgpm_d_adjusted_domain_year_sensitivity"] = (
            _descriptive_adjustment(
                horizon_rows, include_domain_year=True
            ).to_numpy(float)
        )
    targets["target_name"] = targets["horizon"].map(
        lambda value: f"RGPM-D{int(value)}"
    )
    targets["definition_version"] = TARGET_DEFINITION_VERSION
    targets["minimum_future_citers_definition"] = int(min_future_citers)
    targets["minimum_taxonomy_coverage_definition"] = float(
        min_taxonomy_coverage
    )
    targets["target_component_source"] = "audited_precomputed_future_deltas"
    targets = targets.drop(
        columns=[
            column
            for column in targets
            if column.startswith("__log_") or column.startswith("__rank_")
        ]
    )
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        targets.to_parquet(output_path, index=False)
    return targets


def build_structural_targets(
    structural_deltas: pd.DataFrame,
    *,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Aggregate precomputed future structural deltas into RGPM-S targets."""

    components = (
        "modularity_shock",
        "boundary_mixing_change",
        "partition_change",
        "path_shortening",
    )
    required = {"paper_id", "horizon", *components}
    missing = required - set(structural_deltas)
    if missing:
        raise ValueError(f"structural_deltas is missing columns: {sorted(missing)}")
    output = structural_deltas.copy()
    output["paper_id"] = output["paper_id"].map(normalize_openalex_id)
    # Keep RGPM-S comparable to RGPM-D: target ranks are horizon-global;
    # domain/year conditioning is reported only in evaluation.
    group_columns = ["horizon"]
    ranked_columns = []
    for column in components:
        ranked = f"__rank_{column}"
        output[ranked] = _rank_within_group(output, column, group_columns)
        ranked_columns.append(ranked)
    output["rgpm_s"] = output[ranked_columns].mean(axis=1)
    output["target_name"] = output["horizon"].map(lambda value: f"RGPM-S{int(value)}")
    output["definition_version"] = TARGET_DEFINITION_VERSION
    output = output.drop(columns=ranked_columns)
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        output.to_parquet(output_path, index=False)
    return output

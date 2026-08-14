from __future__ import annotations

import hashlib
import heapq
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


FIGURE_IDS: Tuple[str, ...] = tuple(f"fig{index:02d}" for index in range(1, 11))
CORE_FEATURES: Tuple[str, ...] = (
    "delta_q0_shock",
    "rtd_simpson",
    "field_log_variety",
    "field_evenness",
    "field_disparity",
    "pair_atypicality_tail",
    "pair_conventionality_median",
    "burt_efficiency",
)
AUX_FEATURES: Tuple[str, ...] = (
    "log_reference_count",
    "reference_age_median",
    "reference_age_iqr",
    "recent_reference_share_5y",
    "classic_reference_share_20y",
    "prior_graph_degree_median",
    "prior_graph_degree_p90",
    "prior_obscure_reference_share",
    "prior_component_size_log",
    "reference_induced_density",
)
MECHANISM_COLUMNS: Tuple[str, ...] = (
    "boundary_perturbation",
    "community_diffusion",
    "interdisciplinarity",
    "knowledge_recombination",
    "knowledge_brokerage",
)
OPTIONAL_FIGURE_EVIDENCE: Mapping[str, str] = {
    "fig04_peer_review_validation": "peer_review_validation",
    "fig05_frontier_backtest": "frontier_backtest",
    "fig06_registered_robustness": "robustness_evidence",
    "fig07_venue_family_inference": "venue_family_inference",
    "fig09_case_profile": "external_case_profile",
    "fig09_case_evidence": "case_evidence",
    "fig10_registered_ablations": "ablation_evidence",
}
CONDITIONAL_CLAIM_SCOPE = (
    "42 Nature Portfolio sources; pre-publication-year graph; "
    "validated conditionally among papers with at least 10 future citers; "
    "cap-hit rows flagged and uncapped sensitivity gated"
)
MIN_CONDITIONAL_CELL_N = 30


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_artifact(release_path: Path, release: Mapping[str, Any], name: str) -> Optional[Path]:
    artifacts = release.get("artifacts", {})
    if not isinstance(artifacts, Mapping) or name not in artifacts:
        return None
    value = artifacts[name]
    if isinstance(value, Mapping):
        value = value.get("path")
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else (release_path.parent / path).resolve()


def _declared_release_artifact_path(
    release: Mapping[str, Any],
    name: str,
    source_path: Path,
) -> str:
    artifacts = release.get("artifacts", {})
    value = artifacts.get(name) if isinstance(artifacts, Mapping) else None
    if isinstance(value, Mapping):
        declared = value.get("release_path")
        if isinstance(declared, str) and declared:
            candidate = Path(declared)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError(
                    f"Invalid release_path for source artifact {name}: {declared}"
                )
            return candidate.as_posix()
    return source_path.name


def _read_table(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    if path.name.endswith(".tmp") or ".tmp-" in path.name:
        raise ValueError(f"Temporary artifacts cannot be figure inputs: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported table format: {path}")


def _optional_evidence_table(
    release_path: Path,
    release: Mapping[str, Any],
    artifact_name: str,
) -> pd.DataFrame:
    path = _resolve_artifact(release_path, release, artifact_name)
    if path is None or not path.is_file():
        return pd.DataFrame(
            [
                {
                    "availability_status": f"{artifact_name}_not_in_release",
                }
            ]
        )
    frame = _read_table(path)
    if frame.empty:
        return pd.DataFrame(
            [{"availability_status": f"{artifact_name}_is_empty"}]
        )
    return frame


def _finite_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _feature_quality(features: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for group, names in (("core8", CORE_FEATURES), ("aux10", AUX_FEATURES)):
        for name in names:
            values = _finite_series(features, name)
            finite = values.dropna()
            rows.append(
                {
                    "feature_group": group,
                    "feature": name,
                    "n_rows": int(len(features)),
                    "n_finite": int(len(finite)),
                    "finite_coverage": float(len(finite) / len(features)) if len(features) else np.nan,
                    "q05": float(finite.quantile(0.05)) if len(finite) else np.nan,
                    "median": float(finite.median()) if len(finite) else np.nan,
                    "q95": float(finite.quantile(0.95)) if len(finite) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _group_score_summary(papers: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()
    join_columns = [
        name
        for name in ("domain12", "publication_year", "venue_family")
        if name in papers.columns and name not in scores.columns
    ]
    base = scores.copy()
    if "paper_id" in base.columns and "paper_id" in papers.columns and join_columns:
        base = base.merge(papers[["paper_id", *join_columns]].drop_duplicates("paper_id"), on="paper_id", how="left")
    group_columns = [name for name in ("horizon", "domain12", "publication_year", "venue_family") if name in base.columns]
    value_columns = [
        name
        for name in (
            "score_mechanism",
            "score_performance_raw",
            "score_performance_calibrated",
            "score_performance_percentile",
            *(f"mechanism__{mechanism}" for mechanism in MECHANISM_COLUMNS),
        )
        if name in base.columns
    ]
    if not group_columns or not value_columns:
        return pd.DataFrame()
    for name in value_columns:
        base[name] = pd.to_numeric(base[name], errors="coerce")
    grouped = base.groupby(group_columns, dropna=False)[value_columns].agg(["count", "mean", "median"])
    grouped.columns = [f"{name}_{stat}" for name, stat in grouped.columns]
    return grouped.reset_index()


def _score_strata(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()
    value = "score_performance_calibrated" if "score_performance_calibrated" in scores.columns else "score_performance_raw"
    if value not in scores.columns:
        return pd.DataFrame()
    frame = scores.copy()
    frame[value] = pd.to_numeric(frame[value], errors="coerce")
    group_columns = [name for name in ("horizon", "domain12") if name in frame.columns]
    if not group_columns:
        group_columns = ["horizon"] if "horizon" in frame.columns else []
    if not group_columns:
        frame["_all"] = "all"
        group_columns = ["_all"]

    assigned = frame.copy()
    percentiles = assigned.groupby(group_columns, dropna=False)[value].rank(method="average", pct=True)
    assigned["score_stratum"] = pd.cut(
        percentiles,
        bins=[0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
        labels=["low", "middle", "high"],
        include_lowest=True,
    ).astype("string")
    summary = (
        assigned.groupby([*group_columns, "score_stratum"], dropna=False)[value]
        .agg(n="count", mean="mean", median="median", minimum="min", maximum="max")
        .reset_index()
    )
    return summary.drop(columns=["_all"], errors="ignore")


def _score_strata_membership(scores: pd.DataFrame) -> pd.DataFrame:
    """Return auditable OOF paper assignments, not only aggregate bins."""
    if scores.empty:
        return pd.DataFrame()
    value = (
        "score_performance_calibrated"
        if "score_performance_calibrated" in scores.columns
        else "score_performance_raw"
    )
    if value not in scores:
        return pd.DataFrame()
    frame = scores.copy()
    frame[value] = pd.to_numeric(frame[value], errors="coerce")
    groups = [name for name in ("horizon", "domain12") if name in frame]
    if not groups:
        groups = ["horizon"] if "horizon" in frame else []
    if not groups:
        frame["_all"] = "all"
        groups = ["_all"]
    percentile = frame.groupby(groups, dropna=False)[value].rank(
        method="average", pct=True
    )
    frame["score_stratum"] = pd.cut(
        percentile,
        bins=[0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
        labels=["low", "middle", "high"],
        include_lowest=True,
    ).astype("string")
    keep = [
        name
        for name in (
            "paper_id",
            "horizon",
            "domain12",
            value,
            "score_stratum",
            "model_version",
            "quality_flags",
            "claim_scope",
        )
        if name in frame
    ]
    return frame[keep].drop(columns=["_all"], errors="ignore")


def _venue_family_summary(
    papers: pd.DataFrame,
    scores: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Compare venue families after within-domain/period ranking."""
    if scores.empty:
        return pd.DataFrame()
    metadata = [
        name
        for name in ("paper_id", "domain12", "publication_year", "venue_family")
        if name in papers
    ]
    frame = scores.copy()
    missing = [name for name in metadata if name not in frame and name != "paper_id"]
    if missing:
        frame = frame.merge(
            papers[["paper_id", *missing]].drop_duplicates("paper_id"),
            on="paper_id",
            how="left",
            validate="many_to_one",
        )
    selected = predictions[
        predictions.get("is_selected", pd.Series(False, index=predictions.index))
        .fillna(False)
        .astype(bool)
    ]
    if {"paper_id", "horizon", "target_adjusted_oof"}.issubset(selected.columns):
        frame = frame.merge(
            selected[["paper_id", "horizon", "target_adjusted_oof"]],
            on=["paper_id", "horizon"],
            how="left",
            validate="one_to_one",
        )
    frame["publication_period"] = (
        pd.to_numeric(frame["publication_year"], errors="coerce").floordiv(5).mul(5)
    )
    cells = ["horizon", "domain12", "publication_period"]
    score_column = "score_performance_calibrated"
    frame["conditional_cell_n"] = frame.groupby(cells, dropna=False)[
        score_column
    ].transform("count")
    frame = frame[
        frame["conditional_cell_n"].ge(MIN_CONDITIONAL_CELL_N)
    ].copy()
    if frame.empty:
        return pd.DataFrame(
            [
                {
                    "availability_status": "no_domain_period_cells_ge_30",
                    "minimum_conditional_cell_n": MIN_CONDITIONAL_CELL_N,
                }
            ]
        )
    frame["conditional_score_rank"] = frame.groupby(cells, dropna=False)[
        score_column
    ].rank(method="average", pct=True)
    if "target_adjusted_oof" in frame:
        frame["conditional_target_rank"] = frame.groupby(cells, dropna=False)[
            "target_adjusted_oof"
        ].rank(method="average", pct=True)
        frame["predicted_top_decile"] = frame["conditional_score_rank"].ge(0.90)
        frame["true_top_decile"] = frame["conditional_target_rank"].ge(0.90)
        frame["top_decile_hit"] = (
            frame["predicted_top_decile"] & frame["true_top_decile"]
        ).astype(float)
    group_columns = ["horizon", "venue_family", "publication_period"]
    aggregate: Dict[str, tuple[str, str]] = {
        "n": ("paper_id", "count"),
        "eligible_domain_cells": ("domain12", "nunique"),
        "minimum_source_cell_n": ("conditional_cell_n", "min"),
        "conditional_score_mean": ("conditional_score_rank", "mean"),
    }
    if "conditional_target_rank" in frame:
        aggregate.update(
            {
                "future_diffusion_mean": ("conditional_target_rank", "mean"),
                "predicted_top_share": ("predicted_top_decile", "mean"),
                "top_decile_hit_share": ("top_decile_hit", "mean"),
            }
        )
    for mechanism in MECHANISM_COLUMNS:
        column = f"mechanism__{mechanism}"
        if column in frame:
            aggregate[f"mechanism__{mechanism}_mean"] = (column, "mean")
    return (
        frame.groupby(group_columns, dropna=False)
        .agg(**aggregate)
        .reset_index()
    )


def _safe_select(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    return frame[[name for name in columns if name in frame.columns]].copy()


def _holdout_view(frame: pd.DataFrame, protocol: str) -> pd.DataFrame:
    if not frame.empty:
        output = frame.copy()
        output["availability_status"] = "unlocked_evaluated"
        output["evaluation_protocol"] = protocol
        return output
    return pd.DataFrame(
        [
            {
                "evaluation_protocol": protocol,
                "availability_status": "sealed_locked_or_not_evaluated",
                "paper_id": "",
            }
        ]
    )


def _forecast_score_view(
    papers: pd.DataFrame,
    full_scores: pd.DataFrame,
    oof_scores: pd.DataFrame,
    sealed_predictions: pd.DataFrame,
    *,
    horizon: int = 5,
) -> pd.DataFrame:
    """Build a leakage-explicit forecast table with OOF-first precedence."""

    metadata_columns = [
        column
        for column in (
            "paper_id",
            "publication_year",
            "domain12",
            "venue_family",
        )
        if column in papers
    ]
    metadata = (
        papers[metadata_columns].drop_duplicates("paper_id")
        if "paper_id" in metadata_columns
        else pd.DataFrame(columns=["paper_id"])
    )

    def attach_metadata(frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        if output.empty or metadata.empty or "paper_id" not in output:
            return output
        missing = [name for name in metadata.columns if name not in output]
        if missing:
            output = output.merge(
                metadata[["paper_id", *[name for name in missing if name != "paper_id"]]],
                on="paper_id",
                how="left",
                validate="many_to_one",
            )
        return output

    development = oof_scores[
        pd.to_numeric(
            oof_scores.get("horizon", pd.Series(np.nan, index=oof_scores.index)),
            errors="coerce",
        ).eq(horizon)
    ].copy()
    development = attach_metadata(development)
    development["score_scope"] = "development_oof"
    development["score_is_out_of_sample"] = 1
    development["outcome_observable"] = 1

    sealed = sealed_predictions[
        pd.to_numeric(
            sealed_predictions.get(
                "horizon", pd.Series(np.nan, index=sealed_predictions.index)
            ),
            errors="coerce",
        ).eq(horizon)
    ].copy()
    if "is_selected" in sealed:
        sealed = sealed[sealed["is_selected"].fillna(False).astype(bool)].copy()
    sealed = sealed.rename(
        columns={
            "prediction_raw": "score_performance_raw",
            "prediction_calibrated": "score_performance_calibrated",
            "prediction_percentile": "score_performance_percentile",
            "model_id": "model_version",
        }
    )
    if sealed.duplicated(["paper_id", "horizon"]).any():
        raise ValueError("Selected sealed scores contain duplicate paper/horizon rows")
    sealed = attach_metadata(sealed)
    sealed["score_scope"] = "sealed_temporal_holdout"
    sealed["score_is_out_of_sample"] = 1
    sealed["outcome_observable"] = 1
    if "quality_flags" not in sealed:
        sealed["quality_flags"] = ""
    sealed["quality_flags"] = sealed["quality_flags"].fillna("").map(
        lambda value: ";".join(
            item
            for item in (
                str(value).strip(";"),
                "outcome_conditioned_future_citers_ge_10",
            )
            if item
        )
    )
    if "cap_hit" in sealed:
        cap_mask = pd.to_numeric(sealed["cap_hit"], errors="coerce").fillna(0).astype(bool)
        sealed.loc[cap_mask, "quality_flags"] = sealed.loc[
            cap_mask, "quality_flags"
        ].map(
            lambda value: ";".join(
                item
                for item in (
                    str(value).strip(";"),
                    "future_citer_cap_hit_1000",
                )
                if item
            )
        )
    if "claim_scope" not in sealed:
        sealed["claim_scope"] = CONDITIONAL_CLAIM_SCOPE
    else:
        sealed["claim_scope"] = sealed["claim_scope"].fillna(
            CONDITIONAL_CLAIM_SCOPE
        )

    descriptive = full_scores[
        pd.to_numeric(
            full_scores.get(
                "horizon", pd.Series(np.nan, index=full_scores.index)
            ),
            errors="coerce",
        ).eq(horizon)
    ].copy()
    descriptive = attach_metadata(descriptive)
    descriptive["score_scope"] = "full_fit_descriptive"
    descriptive["score_is_out_of_sample"] = 0
    if "outcome_observable" not in descriptive:
        flags = descriptive.get(
            "quality_flags", pd.Series("", index=descriptive.index)
        ).fillna("").astype(str)
        descriptive["outcome_observable"] = (
            ~flags.str.contains("recent_paper_outcome_not_observed", regex=False)
        ).astype(int)

    combined = pd.concat(
        [development, sealed, descriptive], ignore_index=True, sort=False
    )
    if combined.empty:
        return combined
    combined["__scope_priority"] = combined["score_scope"].map(
        {
            "development_oof": 0,
            "sealed_temporal_holdout": 1,
            "full_fit_descriptive": 2,
        }
    )
    combined = (
        combined.sort_values(
            ["__scope_priority", "paper_id"], kind="stable"
        )
        .drop_duplicates(["paper_id", "horizon"], keep="first")
        .drop(columns="__scope_priority")
        .reset_index(drop=True)
    )
    return combined


def _filter_metrics(metrics: pd.DataFrame, patterns: Sequence[str]) -> pd.DataFrame:
    if metrics.empty or "metric" not in metrics.columns:
        return metrics.copy()
    matcher = "|".join(patterns)
    searchable = pd.Series("", index=metrics.index, dtype="string")
    for column in ("metric", "model_id", "scope", "horizon", "sensitivity"):
        if column in metrics:
            searchable = searchable + "|" + metrics[column].astype("string").fillna("")
    return metrics[searchable.str.contains(matcher, case=False, regex=True, na=False)].copy()


def _robustness_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return only pre-registered robustness rows, never value-selected rows."""

    if metrics.empty:
        return pd.DataFrame(
            [{"availability_status": "robustness_metrics_not_run"}]
        )
    scope = metrics.get("scope", pd.Series("", index=metrics.index)).astype(str)
    model = metrics.get("model_id", pd.Series("", index=metrics.index)).astype(str)
    sensitivity = metrics.get(
        "sensitivity", pd.Series("main", index=metrics.index)
    ).fillna("main").astype(str)
    horizon = pd.to_numeric(metrics.get("horizon"), errors="coerce")
    metric = metrics.get("metric", pd.Series("", index=metrics.index)).astype(str)
    locked_metrics = {
        "rho_global_calibrated",
        "rho_global_uncalibrated",
        "rho_domain_macro",
        "rho_conditional",
        "top_decile_enrichment",
        "positive_domain_ratio",
        "n_finite_oof",
    }
    main_horizon = (
        model.eq("nested_selector")
        & scope.eq("development_oof")
        & sensitivity.eq("main")
    )
    primary_sensitivity = (
        horizon.eq(5)
        & model.eq("nested_selector")
        & (
            scope.str.startswith("sensitivity_")
            | scope.eq("fold_stability")
        )
    )
    temporal = scope.str.contains("sealed_temporal_holdout", regex=False)
    selected = metrics[
        metric.isin(locked_metrics)
        & (main_horizon | primary_sensitivity | temporal)
    ].copy()
    if selected.empty:
        return pd.DataFrame(
            [{"availability_status": "robustness_metrics_not_run"}]
        )
    return selected


def _ablation_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Expose only explicit ablation runs; ordinary model rows never qualify."""

    if metrics.empty:
        return pd.DataFrame([{"availability_status": "ablation_not_run"}])
    scope = metrics.get("scope", pd.Series("", index=metrics.index)).astype(str)
    sensitivity = metrics.get(
        "sensitivity", pd.Series("", index=metrics.index)
    ).fillna("").astype(str)
    explicit = scope.str.contains("ablation", case=False, regex=False) | sensitivity.str.match(
        r"^(remove_|without_|drop_|no_calibration|no_graph_agent|no_qwen|no_fusion)"
    )
    selected = metrics[explicit].copy()
    if selected.empty:
        return pd.DataFrame([{"availability_status": "ablation_not_run"}])
    return selected


def _mechanism_registry(release_path: Path, release: Mapping[str, Any]) -> pd.DataFrame:
    path = _resolve_artifact(release_path, release, "mechanism_registry")
    if path and path.exists():
        payload = _read_json(path)
        entries = payload.get("mechanisms", payload)
        rows: List[Dict[str, Any]] = []
        if isinstance(entries, Mapping):
            for mechanism, specification in entries.items():
                feature_names = specification.get("features", []) if isinstance(specification, Mapping) else specification
                if not isinstance(feature_names, list):
                    feature_names = [feature_names]
                for feature in feature_names:
                    rows.append({"mechanism": mechanism, "feature": feature})
        return pd.DataFrame(rows)
    fallback = {
        "boundary_perturbation": ["delta_q0_shock"],
        "community_diffusion": ["rtd_simpson"],
        "interdisciplinarity": ["field_log_variety", "field_evenness", "field_disparity"],
        "knowledge_recombination": ["pair_atypicality_tail", "pair_conventionality_median"],
        "knowledge_brokerage": ["burt_efficiency"],
    }
    return pd.DataFrame(
        [{"mechanism": mechanism, "feature": feature} for mechanism, names in fallback.items() for feature in names]
    )


def _feature_registry(release_path: Path, release: Mapping[str, Any]) -> pd.DataFrame:
    path = _resolve_artifact(release_path, release, "feature_registry")
    if path is None or not path.is_file():
        return pd.DataFrame()
    payload = _read_json(path)
    rows = payload.get("features", [])
    return pd.DataFrame(rows) if isinstance(rows, list) else pd.DataFrame()


def _feature_redundancy(features: pd.DataFrame) -> pd.DataFrame:
    available = [name for name in CORE_FEATURES if name in features]
    if not available:
        return pd.DataFrame()
    correlation = features[available].apply(
        pd.to_numeric, errors="coerce"
    ).corr(method="spearman")
    rows: List[Dict[str, Any]] = []
    for left_index, left in enumerate(available):
        for right in available[left_index + 1 :]:
            value = float(correlation.loc[left, right])
            rows.append(
                {
                    "feature_left": left,
                    "feature_right": right,
                    "spearman": value,
                    "absolute_redundancy": abs(value),
                    "complementarity": 1.0 - abs(value),
                }
            )
    return pd.DataFrame(rows)


def _graph_snapshot_sample(
    release_path: Path,
    release: Mapping[str, Any],
    *,
    max_nodes: int = 180,
    max_edges: int = 900,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    catalog_path = _resolve_artifact(release_path, release, "graph_snapshots")
    if catalog_path is None or not catalog_path.is_file():
        status = pd.DataFrame(
            [{"availability_status": "graph_snapshot_not_in_release"}]
        )
        return status, status.copy()
    catalog = pd.read_parquet(catalog_path)
    if catalog.empty:
        status = pd.DataFrame(
            [{"availability_status": "graph_snapshot_catalog_empty"}]
        )
        return status, status.copy()
    row = catalog.sort_values("cutoff_year").iloc[-1]
    node_path = catalog_path.parent / str(row["node_path"])
    edge_path = catalog_path.parent / str(row["edge_path"])
    if not node_path.is_file() or not edge_path.is_file():
        status = pd.DataFrame(
            [{"availability_status": "graph_snapshot_assets_missing"}]
        )
        return status, status.copy()
    heap: List[Tuple[float, str, int]] = []
    for batch in pq.ParquetFile(node_path).iter_batches(
        batch_size=100_000,
        columns=["node_id", "degree", "community_id"],
    ):
        frame = batch.to_pandas()
        for item in frame.itertuples(index=False):
            candidate = (
                float(item.degree),
                str(item.node_id),
                int(item.community_id),
            )
            if len(heap) < max_nodes:
                heapq.heappush(heap, candidate)
            elif candidate > heap[0]:
                heapq.heapreplace(heap, candidate)
    selected_nodes = sorted(heap, reverse=True)
    node_ids = {item[1] for item in selected_nodes}
    node_frame = pd.DataFrame(
        [
            {
                "node_id": node_id,
                "degree": degree,
                "community_id": community,
                "cutoff_year": int(row["cutoff_year"]),
                "graph_id": str(row["graph_id"]),
            }
            for degree, node_id, community in selected_nodes
        ]
    )
    edge_rows: List[Dict[str, Any]] = []
    for batch in pq.ParquetFile(edge_path).iter_batches(
        batch_size=200_000,
        columns=["left_id", "right_id"],
    ):
        frame = batch.to_pandas()
        selected = frame[
            frame["left_id"].astype(str).isin(node_ids)
            & frame["right_id"].astype(str).isin(node_ids)
        ]
        for item in selected.itertuples(index=False):
            edge_rows.append(
                {
                    "left_id": str(item.left_id),
                    "right_id": str(item.right_id),
                    "cutoff_year": int(row["cutoff_year"]),
                    "graph_id": str(row["graph_id"]),
                }
            )
            if len(edge_rows) >= max_edges:
                break
        if len(edge_rows) >= max_edges:
            break
    edge_frame = pd.DataFrame(
        edge_rows,
        columns=["left_id", "right_id", "cutoff_year", "graph_id"],
    )
    if edge_frame.empty:
        edge_frame = pd.DataFrame(
            [
                {
                    "left_id": "",
                    "right_id": "",
                    "cutoff_year": int(row["cutoff_year"]),
                    "graph_id": str(row["graph_id"]),
                    "availability_status": "no_edges_among_top_degree_nodes",
                }
            ]
        )
    return node_frame, edge_frame


def _target_relationships(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    structural_targets: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    targets5 = targets[targets.get("horizon", pd.Series(dtype=int)).eq(5)].copy()
    if not targets5.empty and "rgpm_d_raw" in targets5:
        merged = features.merge(
            targets5[["paper_id", "rgpm_d_raw"]],
            on="paper_id",
            how="inner",
            validate="one_to_one",
        )
        for feature in CORE_FEATURES:
            valid = merged[[feature, "rgpm_d_raw"]].apply(pd.to_numeric, errors="coerce").dropna()
            rows.append(
                {
                    "feature": feature,
                    "target": "RGPM-D5",
                    "spearman": float(valid[feature].corr(valid["rgpm_d_raw"], method="spearman")) if len(valid) >= 3 else np.nan,
                    "n": int(len(valid)),
                }
            )
    structural5 = structural_targets[
        structural_targets.get("horizon", pd.Series(dtype=int)).eq(5)
    ].copy()
    if not structural5.empty and "rgpm_s" in structural5:
        merged = features.merge(
            structural5[["paper_id", "rgpm_s"]],
            on="paper_id",
            how="inner",
            validate="one_to_one",
        )
        for feature in CORE_FEATURES:
            valid = merged[[feature, "rgpm_s"]].apply(pd.to_numeric, errors="coerce").dropna()
            rows.append(
                {
                    "feature": feature,
                    "target": "RGPM-S5",
                    "spearman": float(valid[feature].corr(valid["rgpm_s"], method="spearman")) if len(valid) >= 3 else np.nan,
                    "n": int(len(valid)),
                }
            )
    return pd.DataFrame(rows)


def _case_profiles(
    release_path: Path,
    release: Mapping[str, Any],
    papers: pd.DataFrame,
    scores: pd.DataFrame,
) -> pd.DataFrame:
    registry_path = _resolve_artifact(release_path, release, "case_registry")
    if registry_path is None or not registry_path.is_file():
        return pd.DataFrame()
    payload = _read_json(registry_path)
    cases = payload.get("cases", [])
    if not isinstance(cases, list) or not cases:
        return pd.DataFrame()
    case_frame = pd.DataFrame(cases)

    def normalize_doi(value: Any) -> str:
        return str(value or "").lower().replace("https://doi.org/", "").strip()

    if "doi" not in case_frame:
        case_frame["doi"] = ""
    case_frame["_doi"] = case_frame["doi"].map(normalize_doi)
    if "doi" in papers and "paper_id" in papers:
        paper_lookup = papers[["paper_id", "doi"]].copy()
        paper_lookup["_doi"] = paper_lookup["doi"].map(normalize_doi)
        paper_lookup = paper_lookup.drop_duplicates("_doi", keep="first")
        selected = case_frame.merge(
            paper_lookup,
            on="_doi",
            how="left",
            suffixes=("", "_paper"),
            validate="many_to_one",
        )
    else:
        selected = case_frame.copy()
        selected["paper_id"] = pd.NA
    score_frame = scores.copy()
    if "horizon" in score_frame:
        score_frame = score_frame[score_frame["horizon"].eq(5)].copy()
    if "paper_id" in score_frame:
        score_frame = score_frame.drop_duplicates("paper_id", keep="first")
        selected = selected.merge(
            score_frame,
            on="paper_id",
            how="left",
            validate="many_to_one",
        )
    selected["horizon"] = 5
    percentile = selected.get("score_performance_percentile")
    if percentile is None:
        percentile = pd.Series(np.nan, index=selected.index, dtype=float)
    selected["case_status"] = np.select(
        [
            selected["paper_id"].isna(),
            pd.to_numeric(percentile, errors="coerce").isna(),
        ],
        ["not_in_source_corpus", "paper_found_but_not_scored"],
        default="scored",
    )
    keep = [
        "paper_id",
        "case_id",
        "doi",
        "selection_policy",
        "outcome_eligibility",
        "case_status",
        "score_scope",
        "outcome_observable",
        "claim_scope",
        "horizon",
        "score_mechanism",
        "score_performance_raw",
        "score_performance_calibrated",
        "score_performance_percentile",
        *(f"mechanism__{name}" for name in MECHANISM_COLUMNS),
        "quality_flags",
    ]
    return _safe_select(selected, keep)


def _evidence_contract() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"order": 1, "component": "publication_year_prior_reference_graph", "output": "reference evidence"},
            {"order": 2, "component": "core8_features", "output": "eight publication-prior indicators"},
            {"order": 3, "component": "mechanism5_simplex", "output": "score_mechanism"},
            {"order": 4, "component": "performance18_model", "output": "score_performance"},
            {"order": 5, "component": "score_packet", "output": "dual-score evidence packet"},
            {"order": 6, "component": "aspr_graph_agent", "output": "graph-grounded assessment"},
            {"order": 7, "component": "aspr_qwen", "output": "language assessment"},
            {"order": 8, "component": "fusion_verifier", "output": "verified review"},
        ]
    )


def _build_view_tables(
    figure_id: str,
    release_path: Path,
    release: Mapping[str, Any],
    tables: Mapping[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    papers = tables["papers"]
    features = tables["features_raw"]
    targets = tables["targets"]
    scores = tables["paper_scores"]
    oof_scores = tables["oof_paper_scores"]
    predictions = tables["oof_predictions"]
    sealed_predictions = tables["sealed_holdout_predictions"]
    strict_predictions = tables["strict_label_holdout_predictions"]
    metrics = tables["evaluation_metrics"]
    ledger = tables["model_ledger"]
    structural_targets = tables["structural_targets"]
    if figure_id == "fig01":
        graph_nodes, graph_edges = _graph_snapshot_sample(
            release_path, release
        )
        return {
            "graph_nodes": graph_nodes,
            "graph_edges": graph_edges,
            "mechanism_trajectories": _group_score_summary(papers, oof_scores),
        }
    if figure_id == "fig02":
        return {
            "feature_quality": _feature_quality(features),
            "feature_definitions": _feature_registry(release_path, release),
            "feature_redundancy": _feature_redundancy(features),
            "mechanism_mapping": _mechanism_registry(release_path, release),
            "target_relationships": _target_relationships(features, targets, structural_targets),
        }
    if figure_id == "fig03":
        return {
            "oof_metrics": metrics.copy(),
            "model_ledger": ledger.copy(),
            "oof_predictions": _safe_select(
                predictions,
                (
                    "paper_id",
                    "horizon",
                    "outer_fold",
                    "model_id",
                    "domain12",
                    "publication_year",
                    "target_adjusted_oof",
                    "prediction_raw",
                    "prediction_uncalibrated",
                    "prediction_calibrated",
                    "prediction_percentile",
                    "is_selected",
                    "cap_hit",
                    "is_sealed_holdout",
                ),
            ),
            "sealed_holdout_predictions": _holdout_view(
                sealed_predictions, "sealed_temporal_holdout"
            ),
            "strict_label_holdout_predictions": _holdout_view(
                strict_predictions, "strict_label_availability"
            ),
        }
    if figure_id == "fig04":
        return {
            "score_strata": _score_strata(oof_scores),
            "score_strata_membership": _score_strata_membership(oof_scores),
            "peer_review_validation": tables["peer_review_validation"].copy(),
        }
    if figure_id == "fig05":
        primary_scores = _forecast_score_view(
            papers,
            scores,
            oof_scores,
            sealed_predictions,
            horizon=5,
        )
        return {
            "forecast_scores": _safe_select(
                primary_scores,
                (
                    "paper_id",
                    "horizon",
                    "domain12",
                    "score_performance_raw",
                    "score_performance_calibrated",
                    "score_performance_percentile",
                    "model_version",
                    "quality_flags",
                    "claim_scope",
                    "score_scope",
                    "score_is_out_of_sample",
                    "outcome_observable",
                ),
            ),
            "frontier_backtest": tables["frontier_backtest"].copy(),
        }
    if figure_id == "fig06":
        registered = tables["robustness_evidence"].copy()
        computed = _robustness_metrics(metrics)
        display = (
            computed
            if "availability_status" in registered
            else registered.copy()
        )
        return {
            "robustness_metrics": display,
            "robustness_evidence": registered,
            "candidate_ledger": ledger.copy(),
        }
    if figure_id == "fig07":
        return {
            "venue_family_summary": _venue_family_summary(
                papers, oof_scores, predictions
            ),
            "venue_family_inference": tables[
                "venue_family_inference"
            ].copy(),
        }
    if figure_id == "fig08":
        return {"architecture_contract": _evidence_contract()}
    if figure_id == "fig09":
        release_case = tables["external_case_profile"].copy()
        if "availability_status" in release_case:
            case_profiles = _case_profiles(
                release_path,
                release,
                papers,
                scores,
            )
        else:
            case_profiles = release_case.copy()
        return {
            "case_profiles": case_profiles,
            "external_case_profile": release_case,
            "case_evidence": tables["case_evidence"].copy(),
        }
    registered_ablation = tables["ablation_evidence"].copy()
    display_ablation = (
        _ablation_metrics(metrics)
        if "availability_status" in registered_ablation
        else registered_ablation.copy()
    )
    return {
        "ablation_metrics": display_ablation,
        "ablation_evidence": registered_ablation,
        "model_ledger": ledger.copy(),
    }


def _caption_stats(
    tables: Mapping[str, pd.DataFrame],
    *,
    analysis_id: str,
    figure_id: str,
) -> Dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "analysis_id": analysis_id,
        "figure_id": figure_id,
        "tables": {
            name: {
                "rows": int(len(frame)),
                "columns": list(frame.columns),
                "numeric": {
                    column: {
                        "n_finite": int(
                            np.isfinite(
                                pd.to_numeric(frame[column], errors="coerce")
                            ).sum()
                        ),
                        "mean": (
                            float(
                                pd.to_numeric(frame[column], errors="coerce").mean()
                            )
                            if pd.to_numeric(
                                frame[column], errors="coerce"
                            ).notna().any()
                            else None
                        ),
                        "minimum": (
                            float(
                                pd.to_numeric(frame[column], errors="coerce").min()
                            )
                            if pd.to_numeric(
                                frame[column], errors="coerce"
                            ).notna().any()
                            else None
                        ),
                        "maximum": (
                            float(
                                pd.to_numeric(frame[column], errors="coerce").max()
                            )
                            if pd.to_numeric(
                                frame[column], errors="coerce"
                            ).notna().any()
                            else None
                        ),
                    }
                    for column in frame.columns
                    if pd.to_numeric(frame[column], errors="coerce").notna().any()
                },
            }
            for name, frame in tables.items()
        }
    }


def _evidence_contract_reasons(
    frame: pd.DataFrame,
    expected_ids: Sequence[str],
    *,
    minimum_n: int,
    require_interval: bool = True,
) -> List[str]:
    """Validate a release-bound external-evidence table."""

    if frame.empty or "availability_status" in frame:
        return ["release-bound evidence table is unavailable"]
    required_columns = {
        "evidence_id",
        "metric",
        "value",
        "n",
        "source_artifact_sha256",
        "protocol_hash",
    }
    if require_interval:
        required_columns.update({"ci_low", "ci_high"})
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        return [f"evidence table is missing columns: {missing_columns}"]
    reasons: List[str] = []
    evidence_ids = frame["evidence_id"].fillna("").astype(str)
    observed_ids = set(evidence_ids)
    expected_id_set = set(expected_ids)
    unexpected_ids = sorted(observed_ids - expected_id_set)
    if unexpected_ids:
        reasons.append(f"unexpected evidence IDs are not allowed: {unexpected_ids}")
    for evidence_id in expected_ids:
        rows = frame[evidence_ids.eq(evidence_id)].copy()
        if rows.empty:
            reasons.append(f"missing evidence_id={evidence_id}")
            continue
        valid = np.isfinite(pd.to_numeric(rows["value"], errors="coerce"))
        valid &= pd.to_numeric(rows["n"], errors="coerce").ge(minimum_n)
        valid &= rows["source_artifact_sha256"].astype(str).str.fullmatch(
            r"sha256:[0-9a-f]{64}"
        )
        valid &= rows["protocol_hash"].astype(str).str.fullmatch(
            r"sha256:[0-9a-f]{64}"
        )
        if require_interval:
            low = pd.to_numeric(rows["ci_low"], errors="coerce")
            high = pd.to_numeric(rows["ci_high"], errors="coerce")
            value = pd.to_numeric(rows["value"], errors="coerce")
            valid &= np.isfinite(low) & np.isfinite(high)
            valid &= low.le(value) & value.le(high)
        if not valid.any():
            reasons.append(
                f"evidence_id={evidence_id} has no provenance-bound valid row"
            )
    return reasons


def _metric_bar_supported(frame: pd.DataFrame) -> pd.Series:
    metric = frame.get(
        "metric", pd.Series("", index=frame.index)
    ).fillna("").astype(str)
    return (
        metric.str.startswith("rho_")
        | metric.eq("top_decile_enrichment")
        | metric.str.contains("ratio|share", regex=True)
        | metric.str.startswith("n_")
    )


def _claim_readiness(
    figure_id: str,
    tables: Mapping[str, pd.DataFrame],
) -> Dict[str, Any]:
    """Recompute whether plot data support the pre-registered paper claim."""

    robustness_ids = [
        "horizon_3_5_8",
        "citation_threshold_sensitivity",
        "graph_snapshot_frequency",
        "community_algorithm",
        *(f"remove_{name}" for name in MECHANISM_COLUMNS),
        "remove_all_auxiliary",
        "remove_calibration",
        "model_family_comparison",
        "seed_stability",
        "fold_stability",
    ]
    ablation_ids = [
        *(f"remove_{name}" for name in MECHANISM_COLUMNS),
        "remove_all_auxiliary",
        "no_calibration",
        "model_family_comparison",
        "no_graph_agent",
        "no_qwen",
        "no_fusion_verifier",
    ]
    required: Dict[str, List[str]] = {
        "fig01": ["prior_graph_snapshot", "tau5_oof_mechanism_trajectories"],
        "fig02": ["core8_quality", "mechanism_mapping", "rgpm_d5_s5_relationships"],
        "fig03": ["nested_oof", "sealed_temporal", "strict_label_temporal"],
        "fig04": ["peer_review_resample_v2", "new_score_external_validity"],
        "fig05": ["ai_frontier_tau5_join", "forecast_backtest_v2"],
        "fig06": robustness_ids,
        "fig07": ["venue_family_diffusion_enrichment_mechanism_time_panels"],
        "fig08": ["dual_score_architecture_contract"],
        "fig09": ["fixed_case_score", "graph_qwen_fusion_rerun"],
        "fig10": ablation_ids,
    }
    reasons: List[str] = []
    if figure_id == "fig01":
        nodes = tables["graph_nodes"]
        edges = tables["graph_edges"]
        valid_nodes = nodes.get(
            "node_id", pd.Series("", index=nodes.index)
        ).fillna("").astype(str).str.len().gt(0).sum()
        valid_edges = (
            edges.get("left_id", pd.Series("", index=edges.index))
            .fillna("")
            .astype(str)
            .str.len()
            .gt(0)
            & edges.get("right_id", pd.Series("", index=edges.index))
            .fillna("")
            .astype(str)
            .str.len()
            .gt(0)
        ).sum()
        if valid_nodes < 20 or valid_edges < 1:
            reasons.append("prior graph sample needs at least 20 nodes and one edge")
        trajectory = tables["mechanism_trajectories"]
        tau5 = trajectory[
            pd.to_numeric(
                trajectory.get(
                    "horizon", pd.Series(np.nan, index=trajectory.index)
                ),
                errors="coerce",
            ).eq(5)
        ]
        for mechanism in MECHANISM_COLUMNS:
            mean_column = f"mechanism__{mechanism}_mean"
            count_column = f"mechanism__{mechanism}_count"
            if mean_column not in tau5 or count_column not in tau5:
                reasons.append(f"tau5 trajectory is missing {mechanism}")
                continue
            valid = np.isfinite(pd.to_numeric(tau5[mean_column], errors="coerce"))
            valid &= pd.to_numeric(tau5[count_column], errors="coerce").gt(0)
            years = pd.to_numeric(
                tau5.loc[valid, "publication_year"], errors="coerce"
            ).nunique()
            if years < 3:
                reasons.append(f"tau5 {mechanism} has fewer than three years")
    elif figure_id == "fig02":
        quality = tables["feature_quality"]
        core_quality = quality[
            quality.get(
                "feature_group", pd.Series("", index=quality.index)
            ).eq("core8")
        ]
        covered = set(
            core_quality.loc[
                pd.to_numeric(
                    core_quality.get(
                        "finite_coverage",
                        pd.Series(np.nan, index=core_quality.index),
                    ),
                    errors="coerce",
                ).ge(0.95),
                "feature",
            ].astype(str)
        ) if "feature" in core_quality else set()
        if covered != set(CORE_FEATURES):
            reasons.append("all eight core indicators need at least 95% finite coverage")
        mapping = tables["mechanism_mapping"]
        observed_pairs = set(
            zip(
                mapping.get("mechanism", pd.Series(dtype=str)).astype(str),
                mapping.get("feature", pd.Series(dtype=str)).astype(str),
            )
        )
        expected_pairs = {
            ("boundary_perturbation", "delta_q0_shock"),
            ("community_diffusion", "rtd_simpson"),
            *(('interdisciplinarity', name) for name in CORE_FEATURES[2:5]),
            *(('knowledge_recombination', name) for name in CORE_FEATURES[5:7]),
            ("knowledge_brokerage", "burt_efficiency"),
        }
        if observed_pairs != expected_pairs:
            reasons.append("8-to-5 mechanism mapping is incomplete or changed")
        targets = tables["target_relationships"]
        for target, minimum_n in (("RGPM-D5", 5_000), ("RGPM-S5", 2_000)):
            rows = targets[
                targets.get(
                    "target", pd.Series("", index=targets.index)
                ).eq(target)
            ]
            valid = rows[
                np.isfinite(
                    pd.to_numeric(
                        rows.get(
                            "spearman", pd.Series(np.nan, index=rows.index)
                        ),
                        errors="coerce",
                    )
                )
                & pd.to_numeric(
                    rows.get("n", pd.Series(np.nan, index=rows.index)),
                    errors="coerce",
                ).ge(minimum_n)
            ]
            if set(valid.get("feature", pd.Series(dtype=str)).astype(str)) != set(
                CORE_FEATURES
            ):
                reasons.append(
                    f"{target} requires finite relationships for all core indicators"
                )
    elif figure_id == "fig03":
        metrics = tables["oof_metrics"]
        sensitivity = metrics.get(
            "sensitivity", pd.Series("main", index=metrics.index)
        ).fillna("main").astype(str)
        main_metrics = metrics[sensitivity.eq("main")].copy()
        metric_names = {
            "rho_global_calibrated",
            "rho_global_uncalibrated",
            "rho_domain_macro",
            "rho_conditional",
        }
        locked = main_metrics[
            main_metrics.get(
                "model_id", pd.Series("", index=main_metrics.index)
            ).eq(
                "nested_selector"
            )
            & main_metrics.get(
                "scope", pd.Series("", index=main_metrics.index)
            ).eq("development_oof")
            & main_metrics.get(
                "metric", pd.Series("", index=main_metrics.index)
            ).isin(metric_names | {"n_finite_oof", "top_decile_enrichment"})
        ].copy()
        expected_models = {
            "domain_year_only",
            "bibliographic_aux10_ridge",
            "mechanism5_equal_weight",
            "mechanism5_simplex",
            "gam18",
            "hgb18",
            "rank_blend",
        }
        panel_a = main_metrics[
            _finite_series(main_metrics, "horizon").eq(5)
            & main_metrics.get(
                "scope", pd.Series("", index=main_metrics.index)
            ).eq("development_oof_all_models")
            & main_metrics.get(
                "metric", pd.Series("", index=main_metrics.index)
            ).eq("rho_global_calibrated")
            & np.isfinite(_finite_series(main_metrics, "value"))
        ]
        if set(panel_a.get("model_id", pd.Series(dtype=str)).astype(str)) != expected_models:
            reasons.append("tau5 Panel A requires all seven locked model families")
        for horizon in (3, 5, 8):
            horizon_rows = locked[
                _finite_series(locked, "horizon").eq(horizon)
            ]
            summary = horizon_rows[horizon_rows["metric"].isin(metric_names)]
            finite = np.isfinite(_finite_series(summary, "value"))
            finite &= np.isfinite(_finite_series(summary, "ci_low"))
            finite &= np.isfinite(_finite_series(summary, "ci_high"))
            if set(summary.loc[finite, "metric"].astype(str)) != metric_names:
                reasons.append(f"tau{horizon} is missing locked OOF metrics with CIs")
            n_rows = horizon_rows[horizon_rows["metric"].eq("n_finite_oof")]
            if not _finite_series(n_rows, "value").ge(5_000).any():
                reasons.append(f"tau{horizon} has fewer than 5,000 finite OOF rows")
            enrichment = horizon_rows[
                horizon_rows["metric"].eq("top_decile_enrichment")
            ]
            enrichment_valid = np.isfinite(_finite_series(enrichment, "value"))
            enrichment_valid &= np.isfinite(
                _finite_series(enrichment, "ci_low")
            ) & np.isfinite(_finite_series(enrichment, "ci_high"))
            if not enrichment_valid.any():
                reasons.append(
                    f"tau{horizon} Panel D top-decile enrichment is missing"
                )
            for scope_name in (
                "sealed_temporal_holdout",
                "strict_label_availability__sealed_temporal_holdout",
            ):
                holdout_metric = main_metrics[
                    _finite_series(main_metrics, "horizon").eq(horizon)
                    & main_metrics.get(
                        "scope", pd.Series("", index=main_metrics.index)
                    ).eq(scope_name)
                    & main_metrics.get(
                        "metric", pd.Series("", index=main_metrics.index)
                    ).eq("rho_global_calibrated")
                ]
                holdout_valid = np.isfinite(
                    _finite_series(holdout_metric, "value")
                )
                holdout_valid &= np.isfinite(
                    _finite_series(holdout_metric, "ci_low")
                ) & np.isfinite(_finite_series(holdout_metric, "ci_high"))
                if not holdout_valid.any():
                    reasons.append(
                        f"tau{horizon} Panel C is missing {scope_name} with CI"
                    )
        predictions = tables["oof_predictions"].copy()
        if "is_selected" in predictions:
            predictions = predictions[
                predictions["is_selected"].fillna(False).astype(bool)
            ]
        for horizon in (3, 5, 8):
            rows = predictions[
                pd.to_numeric(
                    predictions.get(
                        "horizon", pd.Series(np.nan, index=predictions.index)
                    ),
                    errors="coerce",
                ).eq(horizon)
            ]
            valid = np.isfinite(_finite_series(rows, "prediction_calibrated"))
            valid &= np.isfinite(_finite_series(rows, "target_adjusted_oof"))
            if int(valid.sum()) < 5_000:
                reasons.append(f"tau{horizon} selected OOF predictions are incomplete")
        for name in (
            "sealed_holdout_predictions",
            "strict_label_holdout_predictions",
        ):
            frame = tables[name]
            unlocked = frame.get(
                "availability_status", pd.Series("", index=frame.index)
            ).eq("unlocked_evaluated")
            finite = np.isfinite(
                _finite_series(frame, "prediction_calibrated")
            ) & np.isfinite(_finite_series(frame, "target_adjusted_oof"))
            for horizon in (3, 5, 8):
                horizon_mask = _finite_series(frame, "horizon").eq(horizon)
                if int((unlocked & finite & horizon_mask).sum()) < 30:
                    reasons.append(
                        f"{name} tau{horizon} has fewer than 30 finite unlocked rows"
                    )
    elif figure_id == "fig04":
        strata = tables["score_strata_membership"]
        tau5 = strata[
            pd.to_numeric(
                strata.get("horizon", pd.Series(np.nan, index=strata.index)),
                errors="coerce",
            ).eq(5)
        ]
        if len(tau5) < 5_000 or set(tau5.get("score_stratum", ())) != {
            "low",
            "middle",
            "high",
        }:
            reasons.append("tau5 OOF score strata are incomplete")
        summary = tables["score_strata"]
        summary5 = summary[
            _finite_series(summary, "horizon").eq(5)
        ]
        valid_summary = np.isfinite(_finite_series(summary5, "mean"))
        valid_summary &= _finite_series(summary5, "n").gt(0)
        observed_summary = (
            set(summary5.loc[valid_summary, "score_stratum"].astype(str))
            if "score_stratum" in summary5
            else set()
        )
        if observed_summary != {"low", "middle", "high"}:
            reasons.append("tau5 score-stratum summary contains missing values")
        reasons.extend(
            _evidence_contract_reasons(
                tables["peer_review_validation"],
                required[figure_id],
                minimum_n=30,
            )
        )
    elif figure_id == "fig05":
        scores = tables["forecast_scores"]
        out_of_sample = scores[
            pd.to_numeric(
                scores.get(
                    "score_is_out_of_sample",
                    pd.Series(0, index=scores.index),
                ),
                errors="coerce",
            ).eq(1)
        ]
        if np.isfinite(
            _finite_series(out_of_sample, "score_performance_percentile")
        ).sum() < 5_000:
            reasons.append("tau5 out-of-sample forecast scores are incomplete")
        reasons.extend(
            _evidence_contract_reasons(
                tables["frontier_backtest"],
                required[figure_id],
                minimum_n=30,
            )
        )
    elif figure_id == "fig06":
        evidence = tables["robustness_evidence"]
        reasons.extend(
            _evidence_contract_reasons(
                evidence,
                required[figure_id],
                minimum_n=30,
            )
        )
        if "evidence_id" in evidence:
            displayed_ids = set(
                evidence.loc[
                    _metric_bar_supported(evidence), "evidence_id"
                ].astype(str)
            )
            missing_display = sorted(
                set(required[figure_id]) - displayed_ids
            )
            if missing_display:
                reasons.append(
                    "registered robustness rows use unsupported display metrics: "
                    f"{missing_display}"
                )
    elif figure_id == "fig07":
        summary = tables["venue_family_summary"]
        expected_families = {
            "nature_flagship",
            "nature_specialist_research",
            "nature_communications",
            "scientific_reports",
            "communications_series",
            "npj_series",
        }
        required_columns = {
            "conditional_score_mean",
            "future_diffusion_mean",
            "predicted_top_share",
            *(f"mechanism__{name}_mean" for name in MECHANISM_COLUMNS),
        }
        if not required_columns.issubset(summary.columns):
            reasons.append("venue-family summary is missing registered panels")
        else:
            tau5 = summary[
                _finite_series(summary, "horizon").eq(5)
            ]
            panel_columns = [
                "n",
                "minimum_source_cell_n",
                *sorted(required_columns),
            ]
            finite_panel = tau5[panel_columns].apply(
                pd.to_numeric, errors="coerce"
            ).notna().all(axis=1)
            finite_panel &= _finite_series(tau5, "n").gt(0)
            valid_tau5 = tau5[finite_panel]
            if set(valid_tau5.get("venue_family", ())) != expected_families:
                reasons.append("all six Nature Portfolio venue families are required")
            if _finite_series(valid_tau5, "publication_period").nunique() < 2:
                reasons.append("venue-family time migration needs at least two periods")
            elif "venue_family" in valid_tau5:
                period_counts = valid_tau5.groupby(
                    "venue_family", dropna=False
                )["publication_period"].nunique()
                if any(
                    int(period_counts.get(family, 0)) < 2
                    for family in expected_families
                ):
                    reasons.append(
                        "each venue family needs at least two finite time periods"
                    )
            if not _finite_series(valid_tau5, "minimum_source_cell_n").ge(
                MIN_CONDITIONAL_CELL_N
            ).all():
                reasons.append("venue-family cells violate the n>=30 gate")
        reasons.extend(
            _evidence_contract_reasons(
                tables["venue_family_inference"],
                required[figure_id],
                minimum_n=30,
            )
        )
    elif figure_id == "fig08":
        architecture = tables["architecture_contract"]
        expected_components = [
            "publication_year_prior_reference_graph",
            "core8_features",
            "mechanism5_simplex",
            "performance18_model",
            "score_packet",
            "aspr_graph_agent",
            "aspr_qwen",
            "fusion_verifier",
        ]
        observed = architecture.sort_values("order").get(
            "component", pd.Series(dtype=str)
        ).astype(str).tolist()
        if observed != expected_components:
            reasons.append("dual-score architecture contract is incomplete")
    elif figure_id == "fig09":
        external_profile = tables["external_case_profile"]
        reasons.extend(
            _evidence_contract_reasons(
                external_profile,
                ["fixed_case_score"],
                minimum_n=1,
                require_interval=False,
            )
        )
        profiles = tables["case_profiles"]
        scored = profiles[
            profiles.get(
                "case_status", pd.Series("", index=profiles.index)
            ).eq("scored")
        ]
        mechanism_columns = [
            f"mechanism__{name}" for name in MECHANISM_COLUMNS
        ]
        if len(profiles) != 1 or len(scored) != 1:
            reasons.append("Fig.9 requires exactly one pre-locked scored case")
        if scored.empty or not set(mechanism_columns).issubset(scored.columns):
            reasons.append("fixed tau5 case score is unavailable")
        elif not np.isfinite(
            scored[mechanism_columns].apply(pd.to_numeric, errors="coerce")
        ).all(axis=1).any():
            reasons.append("fixed case mechanism profile is incomplete")
        quality_valid = (
            _finite_series(scored, "valid_reference_count").ge(10)
            & _finite_series(scored, "reference_metadata_coverage").ge(0.60)
        )
        if not quality_valid.any():
            reasons.append(
                "fixed case needs at least 10 valid references and 60% metadata coverage"
            )
        reasons.extend(
            _evidence_contract_reasons(
                tables["case_evidence"],
                ["graph_qwen_fusion_rerun"],
                minimum_n=1,
                require_interval=False,
            )
        )
    elif figure_id == "fig10":
        ablation = tables["ablation_evidence"]
        reasons.extend(
            _evidence_contract_reasons(
                ablation,
                required[figure_id],
                minimum_n=30,
            )
        )
        if "evidence_id" in ablation:
            displayed_ids = set(
                ablation.loc[
                    _metric_bar_supported(ablation), "evidence_id"
                ].astype(str)
            )
            missing_display = sorted(
                set(required[figure_id]) - displayed_ids
            )
            if missing_display:
                reasons.append(
                    "registered ablation rows use unsupported display metrics: "
                    f"{missing_display}"
                )
    return {
        "claim_readiness": "ready" if not reasons else "placeholder",
        "readiness_reasons": reasons,
        "required_evidence_ids": required[figure_id],
    }


def export_figure_views(
    release_path: Path,
    output_dir: Optional[Path] = None,
    figures: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Export deterministic, release-bound plot data for Fig.1--Fig.10.

    The exporter never accesses raw OpenAlex data and never trains a model. Empty
    inputs remain explicit empty CSV files so downstream drawing cannot silently
    fall back to a legacy output directory.
    """
    release_path = release_path.resolve()
    release = _read_json(release_path)
    analysis_id = str(release.get("analysis_id") or "")
    if not analysis_id:
        raise ValueError("release.json is missing analysis_id")
    selected = tuple(figures or FIGURE_IDS)
    invalid = sorted(set(selected) - set(FIGURE_IDS))
    if invalid:
        raise ValueError(f"Unknown figure IDs: {invalid}")
    view_root = output_dir.resolve() if output_dir else release_path.parent / "figure_views"
    if (release_path.parent / "_SUCCESS").exists():
        try:
            view_root.relative_to(release_path.parent)
        except ValueError:
            pass
        else:
            raise ValueError(
                "Published releases are immutable; build figure views before publish or use an external bundle"
            )
    view_root.mkdir(parents=True, exist_ok=True)

    artifact_names = (
        "papers",
        "features_raw",
        "targets",
        "paper_scores",
        "oof_paper_scores",
        "oof_predictions",
        "sealed_holdout_predictions",
        "strict_label_holdout_predictions",
        "evaluation_metrics",
        "model_ledger",
        "structural_targets",
    )
    artifact_paths = {name: _resolve_artifact(release_path, release, name) for name in artifact_names}
    dependency_paths = dict(artifact_paths)
    for name in (
        "feature_registry",
        "mechanism_registry",
        "case_registry",
        "graph_snapshots",
        *OPTIONAL_FIGURE_EVIDENCE.keys(),
    ):
        path = _resolve_artifact(release_path, release, name)
        if path is not None and path.is_file():
            dependency_paths[name] = path
    release_artifacts = release.get("artifacts", {})
    if isinstance(release_artifacts, Mapping):
        for name in sorted(
            str(value)
            for value in release_artifacts
            if str(value).startswith("figure_evidence_asset__")
        ):
            path = _resolve_artifact(release_path, release, name)
            if path is not None and path.is_file():
                dependency_paths[name] = path
    missing_artifacts = [
        name
        for name, path in artifact_paths.items()
        if path is None or not path.is_file()
    ]
    if missing_artifacts:
        raise FileNotFoundError(f"Release is missing required figure artifacts: {missing_artifacts}")
    tables = {name: _read_table(path) for name, path in artifact_paths.items()}
    for artifact_name, table_name in OPTIONAL_FIGURE_EVIDENCE.items():
        tables[table_name] = _optional_evidence_table(
            release_path, release, artifact_name
        )
    empty_artifacts = [
        name
        for name, frame in tables.items()
        if frame.empty
        and name
        not in {
            "structural_targets",
            "sealed_holdout_predictions",
            "strict_label_holdout_predictions",
        }
    ]
    if empty_artifacts:
        raise ValueError(f"Required figure artifacts are empty: {empty_artifacts}")
    results: Dict[str, Any] = {}
    for figure_id in selected:
        final_dir = view_root / figure_id
        if final_dir.exists():
            raise FileExistsError(f"Refusing to overwrite figure view: {final_dir}")
        temporary_dir = view_root / f".{figure_id}.building"
        if temporary_dir.exists():
            raise FileExistsError(f"Stale temporary figure view exists: {temporary_dir}")
        data_dir = temporary_dir / "data"
        data_dir.mkdir(parents=True)
        view_tables = _build_view_tables(figure_id, release_path, release, tables)
        readiness = _claim_readiness(figure_id, view_tables)
        empty_views = [name for name, frame in view_tables.items() if frame.empty]
        if empty_views:
            raise ValueError(f"{figure_id} produced empty plot-data tables: {empty_views}")
        output_files: List[Dict[str, Any]] = []
        for name, frame in view_tables.items():
            path = data_dir / f"{name}.csv"
            frame.to_csv(path, index=False)
            output_files.append({"path": str(path.relative_to(temporary_dir)), "rows": int(len(frame)), "sha256": _sha256(path)})
        _write_json(
            temporary_dir / "panel_spec.json",
            {
                "schema_version": "1.0.0",
                "figure_id": figure_id,
                "analysis_id": analysis_id,
                "data_tables": [item["path"] for item in output_files],
                "draw_only": True,
                **readiness,
            },
        )
        caption_stats = _caption_stats(
            view_tables,
            analysis_id=analysis_id,
            figure_id=figure_id,
        )
        caption_stats.update(readiness)
        _write_json(temporary_dir / "caption_stats.json", caption_stats)
        source_inputs = {
            name: {
                "artifact_name": name,
                "release_path": _declared_release_artifact_path(
                    release, name, path
                ),
                "sha256": _sha256(path),
            }
            for name, path in dependency_paths.items()
            if path is not None and path.exists()
        }
        _write_json(
            temporary_dir / "view_manifest.json",
            {
                "schema_version": "1.0.0",
                "analysis_id": analysis_id,
                "figure_id": figure_id,
                "source_release": "release.json",
                "source_inputs": source_inputs,
                "outputs": output_files,
                **readiness,
            },
        )
        (temporary_dir / "_SUCCESS").write_text("ok\n", encoding="utf-8")
        temporary_dir.rename(final_dir)
        results[figure_id] = {
            "path": str(final_dir),
            "tables": caption_stats["tables"],
            **readiness,
        }
    return {"analysis_id": analysis_id, "figure_views": results, "root": str(view_root)}

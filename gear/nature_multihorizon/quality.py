"""Quality gates for datasets, out-of-fold evidence, and frozen releases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .contracts import AUXILIARY_FEATURES, CORE_FEATURES
from .taxonomy import DOMAIN_IDS


LEGACY_COLUMN_NAMES: Tuple[str, ...] = (
    "b_z",
    "rs_z",
    "deltaq0_z",
    "uzzi_z",
    "rtd_z",
    "burtip_z",
    "pde_z",
    "s_w",
)


def _check(name: str, passed: bool, value: Any, threshold: Any, *, blocking: bool = True) -> Dict[str, Any]:
    return {
        "name": name,
        "status": "pass" if passed else ("fail" if blocking else "warn"),
        "blocking": bool(blocking),
        "value": value,
        "threshold": threshold,
    }


def _finite_coverage(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns or frame.empty:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    return float(np.isfinite(values).mean())


def _legacy_columns(tables: Mapping[str, pd.DataFrame]) -> List[str]:
    forbidden = set(LEGACY_COLUMN_NAMES)
    collisions: List[str] = []
    for table_name, frame in tables.items():
        for column in frame.columns:
            normalized = str(column).strip().casefold()
            if normalized in forbidden:
                collisions.append(f"{table_name}.{column}")
    return sorted(collisions)


def _primary_key_check(frame: pd.DataFrame, columns: Sequence[str]) -> Tuple[bool, int]:
    if frame.empty or any(column not in frame.columns for column in columns):
        return False, 0
    duplicate_count = int(frame.duplicated(list(columns), keep=False).sum())
    null_count = int(frame[list(columns)].isna().any(axis=1).sum())
    return duplicate_count == 0 and null_count == 0, duplicate_count + null_count


def _orphan_count(
    child: pd.DataFrame,
    parent: pd.DataFrame,
    columns: Sequence[str],
) -> int:
    if child.empty or parent.empty or any(
        column not in child.columns or column not in parent.columns for column in columns
    ):
        return int(len(child)) if not child.empty else 0
    keys = list(columns)
    parent_keys = parent[keys].drop_duplicates()
    joined = child[keys].drop_duplicates().merge(
        parent_keys,
        on=keys,
        how="left",
        indicator=True,
    )
    return int(joined["_merge"].ne("both").sum())


def _publication_prior_coverage(features: pd.DataFrame, papers: pd.DataFrame) -> Tuple[float, str]:
    if features.empty or papers.empty or "paper_id" not in features.columns or "paper_id" not in papers.columns:
        return 0.0, "missing features/papers or paper_id"
    publication_column = "publication_year" if "publication_year" in papers.columns else "year" if "year" in papers.columns else ""
    if not publication_column:
        return 0.0, "missing publication year"
    provenance_columns = [name for name in features.columns if name == "source_max_year" or name.endswith("__source_max_year")]
    if not provenance_columns:
        return 0.0, "missing source_max_year provenance"
    joined = features[["paper_id", *provenance_columns]].merge(
        papers[["paper_id", publication_column]].drop_duplicates("paper_id"),
        on="paper_id",
        how="left",
    )
    year = pd.to_numeric(joined[publication_column], errors="coerce")
    valid_rows = year.notna()
    for name in provenance_columns:
        source_year = pd.to_numeric(joined[name], errors="coerce")
        valid_rows &= source_year.notna() & source_year.lt(year)
    coverage = float(valid_rows.mean()) if len(joined) else 0.0
    return coverage, ",".join(provenance_columns)


def audit_pipeline_tables(
    tables: Mapping[str, pd.DataFrame],
    *,
    quality_config: Optional[Mapping[str, Any]] = None,
    prevalidated: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Apply the locked V1 data and OOF gates to canonical tables.

    Missing artifacts fail the relevant gate instead of being interpreted as
    empty or zero-valued evidence. The resulting report can be written for a
    candidate release; all blocking checks must pass before freezing.
    """
    config = dict(quality_config or {})
    trusted = dict(prevalidated or {})
    checks: List[Dict[str, Any]] = []
    papers = tables.get("papers", pd.DataFrame())
    references = tables.get("paper_references", pd.DataFrame())
    reference_works = tables.get("reference_works", pd.DataFrame())
    reference_edges = tables.get("reference_edges", pd.DataFrame())
    future_citers = tables.get("future_citers", pd.DataFrame())
    future_status = tables.get("future_fetch_status", pd.DataFrame())
    future_requests = tables.get("future_request_manifest", pd.DataFrame())
    features = tables.get("features_raw", pd.DataFrame())
    targets = tables.get("targets", pd.DataFrame())
    cohorts = tables.get("cohort_membership", pd.DataFrame())
    splits = tables.get("split_membership", pd.DataFrame())
    predictions = tables.get("oof_predictions", pd.DataFrame())
    sealed_predictions = tables.get("sealed_holdout_predictions", pd.DataFrame())
    strict_predictions = tables.get(
        "strict_label_holdout_predictions", pd.DataFrame()
    )
    scores = tables.get("paper_scores", pd.DataFrame())
    ledger = tables.get("model_ledger", pd.DataFrame())
    metrics = tables.get("evaluation_metrics", pd.DataFrame())
    structural_subset = tables.get("structural_subset", pd.DataFrame())
    structural_targets = tables.get("structural_targets", pd.DataFrame())

    primary_keys = {
        "papers": (papers, ("paper_id",)),
        "paper_references": (references, ("paper_id", "reference_id")),
        "reference_works": (reference_works, ("reference_id",)),
        "reference_edges": (reference_edges, ("source_reference_id", "target_reference_id")),
        "future_citers": (future_citers, ("paper_id", "horizon", "citer_id")),
        "future_fetch_status": (future_status, ("paper_id", "requested_horizon")),
        "future_request_manifest": (future_requests, ("paper_id", "requested_horizon")),
        "features_raw": (features, ("paper_id",)),
        "targets": (targets, ("paper_id", "horizon")),
        "cohort_membership": (cohorts, ("paper_id", "horizon")),
        "split_membership": (splits, ("paper_id", "horizon", "split_id")),
        "oof_predictions": (predictions, ("paper_id", "horizon", "model_id")),
        "sealed_holdout_predictions": (
            sealed_predictions,
            ("paper_id", "horizon", "model_id"),
        ),
        "strict_label_holdout_predictions": (
            strict_predictions,
            ("paper_id", "horizon", "model_id"),
        ),
        "paper_scores": (scores, ("paper_id", "horizon")),
        "model_ledger": (ledger, ("horizon", "outer_fold", "candidate_id")),
        "evaluation_metrics": (
            metrics,
            ("horizon", "model_id", "scope", "metric", "sensitivity"),
        ),
        "structural_subset": (
            structural_subset,
            ("paper_id", "horizon"),
        ),
        "structural_targets": (
            structural_targets,
            ("paper_id", "horizon"),
        ),
    }
    for name, (frame, key) in primary_keys.items():
        if name in set(trusted.get("primary_key_tables", ())):
            passed, issue_count = True, 0
        else:
            passed, issue_count = _primary_key_check(frame, key)
        checks.append(_check(f"primary_key:{name}", passed, issue_count, 0))

    relationships = (
        ("features_to_papers", features, papers, ("paper_id",)),
        ("targets_to_papers", targets, papers, ("paper_id",)),
        ("cohorts_to_targets", cohorts, targets, ("paper_id", "horizon")),
        ("splits_to_cohorts", splits, cohorts, ("paper_id", "horizon")),
        ("oof_to_splits", predictions, splits, ("paper_id", "horizon")),
        ("scores_to_papers", scores, papers, ("paper_id",)),
    )
    for name, child, parent, keys in relationships:
        orphan_count = _orphan_count(child, parent, keys)
        checks.append(_check(f"referential_integrity:{name}", orphan_count == 0, orphan_count, 0))

    collisions = sorted(
        set(_legacy_columns(tables))
        | set(str(value) for value in trusted.get("legacy_columns", ()))
    )
    checks.append(_check("no_legacy_columns", not collisions, collisions, []))

    domain_column = "domain12" if "domain12" in papers.columns else ""
    if domain_column and len(papers):
        domain_values = papers[domain_column].fillna("").astype(str)
        out_of_scope = domain_values.eq("out_of_scope_nonnatural")
        denominator = int((~out_of_scope).sum())
        mapped = domain_values.isin(DOMAIN_IDS)
        domain_coverage = float(mapped.sum() / max(1, denominator))
    else:
        domain_coverage = 0.0
    domain_threshold = float(config.get("min_domain_mapping_coverage", 0.95))
    checks.append(_check("domain12_mapping_coverage", domain_coverage >= domain_threshold, domain_coverage, domain_threshold))
    hierarchy_coverage = 0.0
    if not papers.empty and {"domain12", "domain12_reason"}.issubset(papers):
        natural_rows = papers[papers["domain12"].isin(DOMAIN_IDS)]
        if len(natural_rows):
            hierarchy_coverage = float(
                natural_rows["domain12_reason"]
                .fillna("")
                .astype(str)
                .str.startswith("field_hierarchy:")
                .mean()
            )
    hierarchy_threshold = float(
        config.get("min_field_hierarchy_domain_coverage", 0.90)
    )
    checks.append(
        _check(
            "domain12_official_hierarchy_coverage",
            hierarchy_coverage >= hierarchy_threshold,
            hierarchy_coverage,
            hierarchy_threshold,
        )
    )

    reference_coverage_value = trusted.get("global_reference_metadata_coverage")
    reference_coverage = (
        float(reference_coverage_value)
        if reference_coverage_value is not None
        else 0.0
    )
    if reference_coverage_value is None and not references.empty and "reference_id" in references.columns and not reference_works.empty and "reference_id" in reference_works.columns:
        known = set(reference_works["reference_id"].dropna().astype(str))
        reference_coverage = float(references["reference_id"].astype(str).isin(known).mean())
    reference_threshold = float(config.get("min_global_reference_coverage", 0.70))
    checks.append(_check("global_reference_metadata_coverage", reference_coverage >= reference_threshold, reference_coverage, reference_threshold))

    provenance_coverage, provenance_detail = _publication_prior_coverage(features, papers)
    checks.append(
        _check(
            "publication_prior_feature_provenance",
            provenance_coverage == 1.0,
            {"coverage": provenance_coverage, "columns": provenance_detail},
            "source_max_year < publication_year for every modeled row",
        )
    )

    feature_threshold = float(config.get("min_feature_finite_coverage", 0.95))
    feature_coverages = {name: _finite_coverage(features, name) for name in (*CORE_FEATURES, *AUXILIARY_FEATURES)}
    checks.append(
        _check(
            "feature_finite_coverage",
            bool(feature_coverages)
            and min(feature_coverages[name] for name in CORE_FEATURES) >= feature_threshold,
            feature_coverages,
            {"core8_minimum": feature_threshold, "aux10": "reported and fold-locally imputed"},
        )
    )

    status_coverage = 0.0
    fetch_success = 0.0
    batch_status: Dict[str, Dict[str, float]] = {}
    failure_na = False
    if (
        not future_status.empty
        and not future_requests.empty
        and {"paper_id", "requested_horizon", "fetch_status"}.issubset(future_status.columns)
        and {"paper_id", "requested_horizon"}.issubset(future_requests.columns)
    ):
        expected_columns = ["paper_id", "requested_horizon"]
        if "request_batch" in future_requests:
            expected_columns.append("request_batch")
        expected_keys = future_requests[expected_columns].drop_duplicates(
            ["paper_id", "requested_horizon"]
        )
        if "request_batch" not in expected_keys:
            expected_keys["request_batch"] = "unspecified"
        expected_keys["request_batch"] = (
            expected_keys["request_batch"].fillna("unspecified").astype(str)
        )
        actual = future_status[["paper_id", "requested_horizon", "fetch_status"]].drop_duplicates(
            ["paper_id", "requested_horizon"]
        )
        joined_status = expected_keys.merge(
            actual,
            on=["paper_id", "requested_horizon"],
            how="left",
            validate="one_to_one",
        )
        status_coverage = float(joined_status["fetch_status"].isin(["success", "failed"]).mean())
        fetch_success = float(joined_status["fetch_status"].eq("success").mean())
        for batch_name, batch_rows in joined_status.groupby(
            "request_batch", dropna=False, sort=True
        ):
            batch_status[str(batch_name)] = {
                "status_coverage": float(
                    batch_rows["fetch_status"].isin(["success", "failed"]).mean()
                ),
                "success_rate": float(
                    batch_rows["fetch_status"].eq("success").mean()
                ),
                "n_requested": int(len(batch_rows)),
            }
        failed = future_status["fetch_status"].eq("failed")
        failure_na = "n_returned" in future_status.columns and bool(future_status.loc[failed, "n_returned"].isna().all())
    success_threshold = float(config.get("min_future_fetch_success", 0.99))
    checks.append(_check("future_status_coverage", status_coverage == 1.0, status_coverage, 1.0))
    checks.append(_check("future_fetch_success", fetch_success >= success_threshold, fetch_success, success_threshold))
    for batch_name, values in batch_status.items():
        safe_name = "".join(
            character if character.isalnum() or character in {"_", "-"} else "_"
            for character in batch_name
        )
        checks.append(
            _check(
                f"future_status_coverage_batch:{safe_name}",
                values["status_coverage"] == 1.0,
                values,
                {"status_coverage": 1.0},
            )
        )
        checks.append(
            _check(
                f"future_fetch_success_batch:{safe_name}",
                values["success_rate"] >= success_threshold,
                values,
                {"success_rate_minimum": success_threshold},
            )
        )
    checks.append(_check("future_failures_are_na", failure_na, failure_na, True))

    expected_complete_years = {3: 2022, 5: 2020, 8: 2017}
    future_year_coverage: Dict[str, Any] = {}
    expanded_complete = True
    if not future_requests.empty and {
        "publication_year",
        "requested_horizon",
    }.issubset(future_requests.columns):
        request_year = pd.to_numeric(
            future_requests["publication_year"], errors="coerce"
        )
        request_horizon = pd.to_numeric(
            future_requests["requested_horizon"], errors="coerce"
        )
        for horizon, expected_end in expected_complete_years.items():
            values = request_year[request_horizon.ge(horizon)].dropna()
            observed_end = int(values.max()) if len(values) else None
            passed = observed_end is not None and observed_end >= expected_end
            expanded_complete &= passed
            future_year_coverage[f"tau{horizon}"] = {
                "observed_end_year": observed_end,
                "required_end_year": expected_end,
                "passed": passed,
            }
    else:
        expanded_complete = False
    checks.append(
        _check(
            "expanded_future_horizon_coverage",
            expanded_complete,
            future_year_coverage,
            {"tau3": 2022, "tau5": 2020, "tau8": 2017},
        )
    )

    trusted_year_valid = trusted.get("future_citer_year_window")
    year_valid = bool(trusted_year_valid) if trusted_year_valid is not None else False
    if trusted_year_valid is None and not future_citers.empty and {"paper_id", "citer_year", "horizon"}.issubset(future_citers.columns):
        publication_column = "publication_year" if "publication_year" in papers.columns else "year" if "year" in papers.columns else ""
        if publication_column:
            joined = future_citers.merge(papers[["paper_id", publication_column]], on="paper_id", how="left")
            citing_year = pd.to_numeric(joined["citer_year"], errors="coerce")
            publication_year = pd.to_numeric(joined[publication_column], errors="coerce")
            horizon = pd.to_numeric(joined["horizon"], errors="coerce")
            year_valid = bool((citing_year.gt(publication_year) & citing_year.le(publication_year + horizon)).all())
    checks.append(_check("future_citer_year_window", year_valid, year_valid, True))

    target_quality_coverage = 0.0
    modeled_target_rows = pd.DataFrame()
    if not targets.empty and {
        "fetch_valid",
        "target_valid",
        "n_future_citers",
    }.issubset(targets.columns):
        modeled_target_rows = targets[
            targets["fetch_valid"].fillna(False).astype(bool)
            & pd.to_numeric(targets["n_future_citers"], errors="coerce").ge(10)
        ]
        if len(modeled_target_rows):
            target_quality_coverage = float(
                modeled_target_rows["target_valid"].fillna(False).astype(bool).mean()
            )
    target_quality_threshold = float(
        config.get("min_target_valid_coverage", 0.95)
    )
    checks.append(
        _check(
            "future_taxonomy_target_valid_coverage",
            target_quality_coverage >= target_quality_threshold,
            target_quality_coverage,
            target_quality_threshold,
        )
    )
    for horizon in (3, 5, 8):
        horizon_rows = (
            modeled_target_rows[
                pd.to_numeric(
                    modeled_target_rows.get("horizon"), errors="coerce"
                ).eq(horizon)
            ]
            if not modeled_target_rows.empty and "horizon" in modeled_target_rows
            else pd.DataFrame()
        )
        coverage = (
            float(horizon_rows["target_valid"].fillna(False).astype(bool).mean())
            if len(horizon_rows)
            else 0.0
        )
        checks.append(
            _check(
                f"future_taxonomy_target_valid_coverage_tau{horizon}",
                coverage >= target_quality_threshold,
                {"coverage": coverage, "n": int(len(horizon_rows))},
                target_quality_threshold,
            )
        )

    sealed_target_years = {
        3: (2019, 2022),
        5: (2017, 2020),
        8: (2014, 2017),
    }
    for horizon, (start_year, end_year) in sealed_target_years.items():
        if (
            not modeled_target_rows.empty
            and {"horizon", "publication_year"}.issubset(modeled_target_rows)
        ):
            target_horizon = pd.to_numeric(
                modeled_target_rows["horizon"], errors="coerce"
            )
            target_year = pd.to_numeric(
                modeled_target_rows["publication_year"], errors="coerce"
            )
            sealed_rows = modeled_target_rows[
                target_horizon.eq(horizon)
                & target_year.between(start_year, end_year, inclusive="both")
            ]
        else:
            sealed_rows = pd.DataFrame()
        coverage = (
            float(sealed_rows["target_valid"].fillna(False).astype(bool).mean())
            if len(sealed_rows)
            else 0.0
        )
        checks.append(
            _check(
                f"sealed_target_valid_coverage_tau{horizon}",
                coverage >= target_quality_threshold,
                {"coverage": coverage, "n": int(len(sealed_rows))},
                {
                    "minimum": target_quality_threshold,
                    "publication_years": [start_year, end_year],
                },
            )
        )

    min_cohort = int(config.get("min_cohort_rows", 5_000))
    min_domain_rows = int(config.get("min_reportable_domain_rows", 200))
    max_cap_hit_rate = float(config.get("max_cap_hit_rate_in_cohort", 0.02))
    for horizon in (3, 5, 8):
        subset = cohorts[cohorts.get("horizon", pd.Series(dtype=int)).eq(horizon)] if not cohorts.empty else pd.DataFrame()
        member_column = "cohort_member" if "cohort_member" in subset.columns else "is_eligible"
        if member_column in subset.columns:
            subset = subset[subset[member_column].fillna(False).astype(bool)]
        checks.append(_check(f"cohort_rows_tau{horizon}", len(subset) >= min_cohort, int(len(subset)), min_cohort))
        cap_rate = (
            float(
                pd.to_numeric(subset.get("cap_hit"), errors="coerce")
                .fillna(0)
                .astype(bool)
                .mean()
            )
            if len(subset) and "cap_hit" in subset
            else 0.0
        )
        checks.append(
            _check(
                f"cap_hit_rate_tau{horizon}",
                cap_rate <= max_cap_hit_rate,
                cap_rate,
                max_cap_hit_rate,
            )
        )
        if "domain12" in subset.columns:
            evaluable_domains = int((subset["domain12"].value_counts() >= min_domain_rows).sum())
        else:
            evaluable_domains = 0
        checks.append(_check(f"evaluable_domains_tau{horizon}", evaluable_domains >= 8, evaluable_domains, 8))

    oof_coverage = 0.0
    if not predictions.empty:
        prediction_column = "prediction_calibrated" if "prediction_calibrated" in predictions.columns else "score_performance_calibrated"
        if prediction_column in predictions.columns:
            oof_coverage = _finite_coverage(predictions, prediction_column)
    checks.append(_check("finite_oof_coverage", oof_coverage >= 0.95, oof_coverage, 0.95))

    def one_metric(scope: str) -> float:
        if metrics.empty or not {
            "horizon",
            "model_id",
            "scope",
            "metric",
            "value",
        }.issubset(metrics):
            return float("nan")
        sensitivity = (
            metrics["sensitivity"].fillna("main").astype(str)
            if "sensitivity" in metrics
            else pd.Series("main", index=metrics.index)
        )
        expected_sensitivity = (
            "uncapped_cohort_member"
            if scope == "sensitivity_uncapped_future_citers"
            else "main"
        )
        rows = metrics[
            pd.to_numeric(metrics["horizon"], errors="coerce").eq(5)
            & metrics["model_id"].eq("nested_selector")
            & metrics["scope"].eq(scope)
            & metrics["metric"].eq("rho_global_calibrated")
            & sensitivity.eq(expected_sensitivity)
        ]
        return (
            float(pd.to_numeric(rows.iloc[0]["value"], errors="coerce"))
            if len(rows) == 1
            else float("nan")
        )

    tau5_main_rho = one_metric("development_oof")
    tau5_uncapped_rho = one_metric("sensitivity_uncapped_future_citers")
    max_cap_sensitivity_delta = float(
        config.get("max_uncapped_oof_rho_delta", 0.02)
    )
    cap_sensitivity_delta = abs(tau5_main_rho - tau5_uncapped_rho)
    checks.append(
        _check(
            "tau5_uncapped_future_citer_sensitivity",
            np.isfinite(tau5_uncapped_rho)
            and tau5_uncapped_rho > 0
            and np.isfinite(cap_sensitivity_delta)
            and cap_sensitivity_delta <= max_cap_sensitivity_delta,
            {
                "main_rho": tau5_main_rho,
                "uncapped_rho": tau5_uncapped_rho,
                "absolute_delta": cap_sensitivity_delta,
            },
            {
                "uncapped_rho": "> 0",
                "absolute_delta_max": max_cap_sensitivity_delta,
            },
        )
    )

    sealed_years = {3: set(range(2019, 2023)), 5: set(range(2017, 2021)), 8: set(range(2014, 2018))}
    for horizon, required_years in sealed_years.items():
        observed: set[int] = set()
        if not sealed_predictions.empty and {
            "horizon",
            "publication_year",
        }.issubset(sealed_predictions.columns):
            values = pd.to_numeric(
                sealed_predictions.loc[
                    pd.to_numeric(
                        sealed_predictions["horizon"], errors="coerce"
                    ).eq(horizon),
                    "publication_year",
                ],
                errors="coerce",
            ).dropna()
            observed = set(values.astype(int))
        checks.append(
            _check(
                f"sealed_holdout_year_coverage_tau{horizon}",
                required_years.issubset(observed),
                sorted(observed),
                sorted(required_years),
            )
        )
        strict_observed: set[int] = set()
        if not strict_predictions.empty and {
            "horizon",
            "publication_year",
        }.issubset(strict_predictions.columns):
            values = pd.to_numeric(
                strict_predictions.loc[
                    pd.to_numeric(
                        strict_predictions["horizon"], errors="coerce"
                    ).eq(horizon),
                    "publication_year",
                ],
                errors="coerce",
            ).dropna()
            strict_observed = set(values.astype(int))
        checks.append(
            _check(
                f"strict_holdout_year_coverage_tau{horizon}",
                required_years.issubset(strict_observed),
                sorted(strict_observed),
                sorted(required_years),
            )
        )

    score_columns = {
        "score_mechanism",
        "score_performance_raw",
        "score_performance_calibrated",
        "score_performance_percentile",
    }
    scores_complete = bool(not scores.empty and score_columns.issubset(scores.columns))
    checks.append(_check("dual_score_contract", scores_complete, sorted(set(scores.columns) & score_columns), sorted(score_columns)))

    tau5_upgrade = pd.DataFrame()
    tau5_holdout = pd.DataFrame()
    tau5_headline = pd.DataFrame()
    tau5_strict = pd.DataFrame()
    if not metrics.empty and {"horizon", "scope", "metric", "value"}.issubset(metrics.columns):
        tau5 = metrics[pd.to_numeric(metrics["horizon"], errors="coerce").eq(5)]
        tau5_upgrade = tau5[
            tau5["scope"].eq("upgrade_gate")
            & tau5["metric"].eq("pass_delta_0_03_ci_low_positive")
        ]
        tau5_holdout = tau5[
            tau5["scope"].eq("sealed_temporal_holdout")
            & tau5["metric"].eq("rho_global_calibrated")
        ]
        tau5_headline = tau5[
            tau5["scope"].eq("development_oof")
            & tau5["metric"].eq("rho_global_calibrated")
        ]
        tau5_strict = tau5[
            tau5["scope"].eq(
                "strict_label_availability__sealed_temporal_holdout"
            )
            & tau5["metric"].eq("rho_global_calibrated")
        ]
    upgrade_pass = bool(len(tau5_upgrade) and float(tau5_upgrade["value"].max()) >= 1.0)
    checks.append(_check("tau5_performance_upgrade_gate", upgrade_pass, upgrade_pass, True))
    holdout_ci_low = (
        float(pd.to_numeric(tau5_holdout.get("ci_low"), errors="coerce").max())
        if len(tau5_holdout) and "ci_low" in tau5_holdout
        else float("nan")
    )
    checks.append(_check("tau5_sealed_holdout_positive", np.isfinite(holdout_ci_low) and holdout_ci_low > 0, holdout_ci_low, "> 0"))
    strict_ci_low = (
        float(pd.to_numeric(tau5_strict.get("ci_low"), errors="coerce").max())
        if len(tau5_strict) and "ci_low" in tau5_strict
        else float("nan")
    )
    checks.append(
        _check(
            "tau5_strict_label_availability_positive",
            np.isfinite(strict_ci_low) and strict_ci_low > 0,
            strict_ci_low,
            "> 0",
        )
    )
    headline_rho = float(pd.to_numeric(tau5_headline["value"], errors="coerce").max()) if len(tau5_headline) else float("nan")
    checks.append(_check("tau5_global_oof_expectation", np.isfinite(headline_rho) and headline_rho >= 0.60, headline_rho, 0.60, blocking=False))

    count_diagnostics = pd.DataFrame()
    if not metrics.empty and {"scope", "metric", "value"}.issubset(metrics.columns):
        count_diagnostics = metrics[
            metrics["scope"].eq("target_adjustment_diagnostic")
            & metrics["metric"].eq(
                "rho_adjusted_target_vs_log_future_citers"
            )
        ]
    max_abs_count_rho = (
        float(
            pd.to_numeric(count_diagnostics["value"], errors="coerce")
            .abs()
            .max()
        )
        if len(count_diagnostics)
        else float("nan")
    )
    checks.append(
        _check(
            "target_adjustment_removes_count_rank_signal",
            np.isfinite(max_abs_count_rho) and max_abs_count_rho <= 0.10,
            max_abs_count_rho,
            "absolute Spearman <= 0.10 in every horizon",
        )
    )

    structural = pd.DataFrame()
    if not metrics.empty and {
        "horizon",
        "model_id",
        "scope",
        "metric",
        "value",
        "ci_low",
    }.issubset(metrics):
        sensitivity = (
            metrics["sensitivity"].fillna("main").astype(str)
            if "sensitivity" in metrics
            else pd.Series("main", index=metrics.index)
        )
        structural = metrics[
            pd.to_numeric(metrics["horizon"], errors="coerce").eq(5)
            & metrics["model_id"].eq("nested_selector")
            & metrics["scope"].eq("structural_validation_subset")
            & metrics["metric"].eq("rho_rgpm_s5")
            & sensitivity.eq("main")
        ]
    structural_value = (
        float(pd.to_numeric(structural.iloc[0]["value"], errors="coerce"))
        if len(structural) == 1
        else float("nan")
    )
    structural_ci_low = (
        float(pd.to_numeric(structural.iloc[0]["ci_low"], errors="coerce"))
        if len(structural) == 1
        else float("nan")
    )
    structural_pass = bool(
        len(structural) == 1
        and np.isfinite(structural_value)
        and structural_value > 0
        and np.isfinite(structural_ci_low)
        and structural_ci_low > 0
    )
    checks.append(
        _check(
            "tau5_structural_validation",
            structural_pass,
            {
                "matching_rows": int(len(structural)),
                "rho": structural_value,
                "ci_low": structural_ci_low,
            },
            {"matching_rows": 1, "rho": "> 0", "ci_low": "> 0"},
        )
    )
    tau5_structural = (
        structural_subset[
            pd.to_numeric(
                structural_subset.get("horizon"), errors="coerce"
            ).eq(5)
        ]
        if not structural_subset.empty
        else pd.DataFrame()
    )
    tau5_structural_domains = (
        int(tau5_structural["domain12"].nunique())
        if "domain12" in tau5_structural
        else 0
    )
    minimum_structural_rows = int(config.get("min_structural_rows", 2_000))
    minimum_structural_domains = int(config.get("min_structural_domains", 6))
    tau5_structural_size_pass = bool(
        len(tau5_structural) >= minimum_structural_rows
        and tau5_structural_domains >= minimum_structural_domains
    )
    checks.append(
        _check(
            "tau5_structural_subset_size",
            tau5_structural_size_pass,
            {
                "n": int(len(tau5_structural)),
                "domains": tau5_structural_domains,
            },
            {
                "min_n": minimum_structural_rows,
                "min_domains": minimum_structural_domains,
            },
        )
    )

    failed = [item for item in checks if item["status"] == "fail"]
    post_training_checks = {
        "expanded_future_horizon_coverage",
        "primary_key:sealed_holdout_predictions",
        "primary_key:strict_label_holdout_predictions",
        "primary_key:structural_subset",
        "primary_key:structural_targets",
        "finite_oof_coverage",
        "dual_score_contract",
        "tau5_performance_upgrade_gate",
        "tau5_sealed_holdout_positive",
        "tau5_strict_label_availability_positive",
        "tau5_structural_validation",
        "tau5_structural_subset_size",
        "tau5_uncapped_future_citer_sensitivity",
        *(f"sealed_target_valid_coverage_tau{horizon}" for horizon in (3, 5, 8)),
        *(f"sealed_holdout_year_coverage_tau{horizon}" for horizon in (3, 5, 8)),
        *(f"strict_holdout_year_coverage_tau{horizon}" for horizon in (3, 5, 8)),
    }
    training_failures = [item for item in failed if item["name"] not in post_training_checks]
    return {
        "schema_version": "1.0.0",
        "checks": checks,
        "n_checks": len(checks),
        "n_failed": len(failed),
        "go_for_training": not training_failures,
        "go_for_frozen_release": not failed,
    }


def write_quality_report(path: Path, report: Mapping[str, Any]) -> None:
    """Write a deterministic quality report JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(dict(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

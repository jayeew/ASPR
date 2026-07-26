"""Analysis builders for the redesigned ASPR v6.1 Fig.1--Fig.10 suite."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from aspr.nature_multihorizon.candidate_registry_v6_1 import (
    CandidateRegistryV61,
    load_candidate_registry_v6_1,
)
from aspr.nature_multihorizon.modeling_v6 import safe_spearman
from aspr.nature_multihorizon.modeling_v6_1 import (
    assemble_all_period_frame,
    evaluate_oof_points,
    paired_bootstrap_gain_intervals,
    run_fixed_medium_oof,
)
from aspr.nature_multihorizon.source_audit_v6 import sha256_file
from aspr.path_layout import resolve_artifact_path


ANGLE_ORDER = (
    "A1_COMBINATION_RARITY",
    "A2_ATYPICALITY_CONVENTIONALITY",
    "A3_FIRST_TIME_COMBINATION",
    "A4_KNOWLEDGE_BREADTH_BALANCE",
    "A5_COGNITIVE_DISTANCE_INTEGRATION",
)

ANGLE_SHORT = {
    "A1_COMBINATION_RARITY": "A1 组合稀有性",
    "A2_ATYPICALITY_CONVENTIONALITY": "A2 非典型性与常规性",
    "A3_FIRST_TIME_COMBINATION": "A3 首次组合",
    "A4_KNOWLEDGE_BREADTH_BALANCE": "A4 知识广度与均衡性",
    "A5_COGNITIVE_DISTANCE_INTEGRATION": "A5 认知距离与整合",
}

FEATURE_SHORT = {
    "reference_overlap_novelty_t0": "参考重叠新颖度",
    "hypergeom_conventionality_median_t0": "组合常规性中位数",
    "first_time_source_pair_share": "首次来源组合占比",
    "field_gini_balance": "领域均衡度",
    "reference_other_field_share": "跨本领域参考占比",
    "field_variety": "领域类别数",
    "field_disparity_cosine_mean": "平均认知距离",
    "rao_stirling_integration": "Rao–Stirling整合度",
}

MODEL_SHORT = {
    "k0_controls": "K0 控制",
    "k1_controls": "K1 控制",
    "k2_controls": "K2 控制",
    "b0_v6_primary_plus_k0": "B0（v6指标+K0）",
    "provisional_core8_plus_k1": "暂定8指标+K1",
    "final_innovation_plus_k1": "最终8指标+K1",
    "final_innovation_plus_k2": "最终8指标+K2",
    "innovation_only": "纯创新指标",
}


@dataclass(frozen=True)
class SuiteInputs:
    """All frozen inputs used by the ten redesigned experiments."""

    project_root: Path
    suite_config_path: Path
    suite_config: Mapping[str, Any]
    source_config_path: Path
    source_config: Mapping[str, Any]
    analysis_root: Path
    dataset_root: Path
    registry_path: Path
    registry_json: Mapping[str, Any]
    registry: CandidateRegistryV61
    completion_audit: Mapping[str, Any]
    reproducibility: Mapping[str, Any]
    data_quality: Mapping[str, Any]
    screening_manifest: Mapping[str, Any]
    oof_manifest: Mapping[str, Any]
    pure_manifest: Mapping[str, Any]
    candidates: pd.DataFrame
    candidate_coverage: pd.DataFrame
    candidate_domain_coverage: pd.DataFrame
    subsampling: pd.DataFrame
    approximation: pd.DataFrame
    papers: pd.DataFrame
    membership: pd.DataFrame
    features: pd.DataFrame
    oof_predictions: pd.DataFrame
    oof_metrics: pd.DataFrame
    fold_metrics: pd.DataFrame
    domain_metrics: pd.DataFrame
    comparisons: pd.DataFrame
    folds: pd.DataFrame
    pure_predictions: pd.DataFrame
    pure_metrics: pd.DataFrame
    pure_fold_metrics: pd.DataFrame
    source_paths: Tuple[Path, ...]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _single_path(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {pattern} below {root}; found {len(matches)}")
    return matches[0].resolve()


def _manifest_output_path(record: Mapping[str, Any], project_root: Path) -> Path:
    """Resolve one frozen output while preserving its original manifest path."""
    return resolve_artifact_path(str(record["path"]), project_root=project_root)


def _validate_manifest_outputs(
    manifest: Mapping[str, Any],
    project_root: Path,
) -> Tuple[Path, ...]:
    verified = []
    for record in (manifest.get("outputs") or {}).values():
        path = _manifest_output_path(record, project_root)
        if not path.is_file():
            raise FileNotFoundError(path)
        expected = str(record["sha256"])
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Frozen artifact hash mismatch: {path}")
        verified.append(path)
    return tuple(verified)


def load_inputs(project_root: Path, suite_config_path: Path) -> SuiteInputs:
    """Load and hash-validate every frozen input used by the figure suite."""
    project_root = Path(project_root).resolve()
    suite_config_path = Path(suite_config_path).resolve()
    suite_config = _load_json(suite_config_path)
    source_config_path = _resolve(project_root, suite_config["source_config"])
    source_config = _load_json(source_config_path)
    analysis_root = _resolve(project_root, suite_config["analysis_root"])
    dataset_root = _resolve(project_root, suite_config["dataset_root"])
    registry_path = _resolve(project_root, suite_config["candidate_registry"])

    screening_path = _single_path(analysis_root, "screening_*/screening_manifest.json")
    oof_path = _single_path(analysis_root, "oof_*/oof_run_manifest.json")
    pure_path = _single_path(
        analysis_root,
        "supplement_innovation_only_*/innovation_only_manifest.json",
    )
    screening_manifest = _load_json(screening_path)
    oof_manifest = _load_json(oof_path)
    pure_manifest = _load_json(pure_path)
    verified = [
        suite_config_path,
        source_config_path,
        registry_path,
        screening_path,
        oof_path,
        pure_path,
    ]
    verified.extend(_validate_manifest_outputs(oof_manifest, project_root))
    verified.extend(_validate_manifest_outputs(pure_manifest, project_root))

    screening_root = screening_path.parent
    candidates = pd.read_csv(screening_root / "candidate_decisions.csv")
    candidate_coverage = pd.read_csv(screening_root / "candidate_coverage.csv")
    candidate_domain_coverage = pd.read_csv(
        screening_root / "candidate_domain_coverage.csv"
    )
    subsampling = pd.read_csv(
        screening_root / "reference_subsampling_summary.csv"
    )
    approximation = pd.read_csv(screening_root / "approximation_fidelity.csv")
    papers = pd.read_parquet(dataset_root / "papers_primary_articles.parquet")
    membership = pd.read_parquet(dataset_root / "cohort_membership.parquet")
    features = pd.read_parquet(dataset_root / "innovation_candidate_features.parquet")
    oof_predictions = pd.read_parquet(
        _manifest_output_path(oof_manifest["outputs"]["predictions"], project_root)
    )
    oof_metrics = pd.read_csv(
        _manifest_output_path(oof_manifest["outputs"]["metrics"], project_root)
    )
    fold_metrics = pd.read_csv(
        _manifest_output_path(oof_manifest["outputs"]["fold_metrics"], project_root)
    )
    domain_metrics = pd.read_csv(
        _manifest_output_path(oof_manifest["outputs"]["domain_metrics"], project_root)
    )
    comparisons = pd.read_csv(
        _manifest_output_path(oof_manifest["outputs"]["comparisons"], project_root)
    )
    folds = pd.read_csv(
        _manifest_output_path(oof_manifest["outputs"]["folds"], project_root)
    )
    pure_predictions = pd.read_parquet(
        _manifest_output_path(pure_manifest["outputs"]["predictions"], project_root)
    )
    pure_metrics = pd.read_csv(
        _manifest_output_path(pure_manifest["outputs"]["metrics"], project_root)
    )
    pure_fold_metrics = pd.read_csv(
        _manifest_output_path(
            pure_manifest["outputs"]["fold_metrics"],
            project_root,
        )
    )
    registry_json = _load_json(registry_path)

    local_files = (
        analysis_root / "completion_audit.json",
        analysis_root / "reproducibility_report.json",
        analysis_root / "data_quality_report.json",
        screening_root / "candidate_decisions.csv",
        screening_root / "candidate_coverage.csv",
        screening_root / "candidate_domain_coverage.csv",
        screening_root / "reference_subsampling_summary.csv",
        screening_root / "approximation_fidelity.csv",
        dataset_root / "papers_primary_articles.parquet",
        dataset_root / "cohort_membership.parquet",
        dataset_root / "innovation_candidate_features.parquet",
    )
    for path in local_files:
        if not path.is_file():
            raise FileNotFoundError(path)
        verified.append(path.resolve())

    return SuiteInputs(
        project_root=project_root,
        suite_config_path=suite_config_path,
        suite_config=suite_config,
        source_config_path=source_config_path,
        source_config=source_config,
        analysis_root=analysis_root,
        dataset_root=dataset_root,
        registry_path=registry_path,
        registry_json=registry_json,
        registry=load_candidate_registry_v6_1(registry_path),
        completion_audit=_load_json(analysis_root / "completion_audit.json"),
        reproducibility=_load_json(analysis_root / "reproducibility_report.json"),
        data_quality=_load_json(analysis_root / "data_quality_report.json"),
        screening_manifest=screening_manifest,
        oof_manifest=oof_manifest,
        pure_manifest=pure_manifest,
        candidates=candidates,
        candidate_coverage=candidate_coverage,
        candidate_domain_coverage=candidate_domain_coverage,
        subsampling=subsampling,
        approximation=approximation,
        papers=papers,
        membership=membership,
        features=features,
        oof_predictions=oof_predictions,
        oof_metrics=oof_metrics,
        fold_metrics=fold_metrics,
        domain_metrics=domain_metrics,
        comparisons=comparisons,
        folds=folds,
        pure_predictions=pure_predictions,
        pure_metrics=pure_metrics,
        pure_fold_metrics=pure_fold_metrics,
        source_paths=tuple(dict.fromkeys(verified)),
    )


def primary_records(inputs: SuiteInputs) -> pd.DataFrame:
    """Return the eight primary metrics with frozen angle/source metadata."""
    rows = []
    for candidate_id, candidate in inputs.registry_json["candidates"].items():
        if candidate["final_role"] != "primary":
            continue
        rows.append(
            {
                "candidate_id": candidate_id,
                "feature": candidate["code_name"],
                "feature_label": FEATURE_SHORT[candidate["code_name"]],
                "angle_id": candidate["angle_id"],
                "angle_label": ANGLE_SHORT[candidate["angle_id"]],
                "mathematical_family": candidate["mathematical_family"],
                "formula": candidate["formula"],
                "source_ids": ";".join(candidate["original_source_ids"]),
                "n_original_sources": len(candidate["original_source_ids"]),
                "n_application_sources": len(candidate["paper_application_source_ids"]),
            }
        )
    order = {angle: index for index, angle in enumerate(ANGLE_ORDER)}
    output = pd.DataFrame(rows)
    output["angle_order"] = output["angle_id"].map(order)
    return output.sort_values(["angle_order", "candidate_id"], kind="stable")


def _balanced_percentiles(values: pd.Series, ids: pd.Series) -> pd.Series:
    frame = pd.DataFrame({"value": values, "paper_id": ids.astype(str)})
    valid = np.isfinite(pd.to_numeric(frame["value"], errors="coerce"))
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    ordered = frame.loc[valid].sort_values(
        ["value", "paper_id"], kind="mergesort"
    )
    n_rows = len(ordered)
    output.loc[ordered.index] = (np.arange(n_rows, dtype=float) + 0.5) / n_rows
    return output


def _wilson_interval(successes: int, total: int) -> Tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    z_value = 1.959963984540054
    rate = successes / total
    denominator = 1.0 + z_value**2 / total
    center = (rate + z_value**2 / (2.0 * total)) / denominator
    radius = (
        z_value
        * np.sqrt(rate * (1.0 - rate) / total + z_value**2 / (4.0 * total**2))
        / denominator
    )
    return float(center - radius), float(center + radius)


def prediction_deciles(
    predictions: pd.DataFrame,
    model_ids: Sequence[str],
    *,
    horizon: int = 5,
) -> pd.DataFrame:
    """Build balanced prediction deciles and observed high-impact rates."""
    selected = predictions[
        predictions["horizon"].eq(int(horizon))
        & predictions["model_id"].isin(model_ids)
    ].copy()
    truth = selected.drop_duplicates("paper_id")[
        ["paper_id", "realized_diffusion_target"]
    ].copy()
    truth["target_percentile"] = _balanced_percentiles(
        truth["realized_diffusion_target"], truth["paper_id"]
    )
    truth["high_impact"] = truth["target_percentile"].ge(0.9).astype(int)
    target_annotations = truth[
        ["paper_id", "target_percentile", "high_impact"]
    ]
    rows = []
    for model_id, group in selected.groupby("model_id", sort=False):
        merged = group.merge(
            target_annotations,
            on="paper_id",
            validate="one_to_one",
        )
        merged["prediction_percentile"] = _balanced_percentiles(
            merged["expected_diffusion_score"], merged["paper_id"]
        )
        merged["prediction_decile"] = np.minimum(
            10,
            np.floor(merged["prediction_percentile"] * 10).astype(int) + 1,
        )
        for decile, subset in merged.groupby("prediction_decile", sort=True):
            successes = int(subset["high_impact"].sum())
            low, high = _wilson_interval(successes, len(subset))
            rows.append(
                {
                    "model_id": str(model_id),
                    "prediction_decile": int(decile),
                    "n": len(subset),
                    "high_impact_count": successes,
                    "observed_high_impact_rate": successes / len(subset),
                    "rate_ci_low": low,
                    "rate_ci_high": high,
                    "mean_realized_diffusion": float(
                        subset["realized_diffusion_target"].mean()
                    ),
                    "median_realized_diffusion": float(
                        subset["realized_diffusion_target"].median()
                    ),
                    "high_impact_definition": (
                        "deterministic global top 10% of OOF realized diffusion; "
                        "ties broken by paper_id"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _wide_pair(
    predictions: pd.DataFrame,
    candidate_model: str,
    baseline_model: str,
) -> pd.DataFrame:
    selected = predictions[
        predictions["model_id"].isin([candidate_model, baseline_model])
    ]
    truth = selected.drop_duplicates("paper_id").set_index("paper_id")[
        "realized_diffusion_target"
    ]
    scores = selected.pivot(
        index="paper_id",
        columns="model_id",
        values="expected_diffusion_score",
    )
    return scores.join(truth.rename("truth"), how="inner").dropna()


def paired_gain_interval(
    predictions: pd.DataFrame,
    candidate_model: str,
    baseline_model: str,
    *,
    iterations: int,
    seed: int,
) -> Dict[str, Any]:
    """Compute one paired paper-bootstrap Spearman-gain interval."""
    wide = _wide_pair(predictions, candidate_model, baseline_model)
    result = paired_bootstrap_gain_intervals(
        wide["truth"],
        wide[candidate_model],
        {"baseline": wide[baseline_model]},
        iterations=int(iterations),
        seed=int(seed),
    ).iloc[0]
    return {
        "candidate_model_id": candidate_model,
        "baseline_model_id": baseline_model,
        "n_papers": int(result["n_papers"]),
        "candidate_spearman": float(result["candidate_spearman"]),
        "baseline_spearman": float(result["baseline_spearman"]),
        "spearman_gain": float(result["spearman_gain"]),
        "gain_ci_low": float(result["gain_ci_low"]),
        "gain_ci_high": float(result["gain_ci_high"]),
        "bootstrap_iterations": int(iterations),
        "bootstrap_unit": "paper_id",
    }


def experiment01_tables(inputs: SuiteInputs) -> Dict[str, pd.DataFrame]:
    """Corpus, temporal coverage and future-label definition."""
    d5 = inputs.membership[inputs.membership["horizon"].eq(5)]
    flow = pd.DataFrame(
        [
            {"stage": "Nature论文（1980–2017）", "n": len(d5), "order": 1},
            {
                "stage": "共同有效队列",
                "n": int(d5["common_cohort_member"].sum()),
                "order": 2,
            },
            {
                "stage": "初始训练期（1980–1985）",
                "n": int(d5["publication_year"].le(1985).sum()),
                "order": 3,
            },
            {
                "stage": "时间OOF论文（1986–2017）",
                "n": int(
                    inputs.oof_predictions[
                        inputs.oof_predictions["model_id"].eq("k1_controls")
                        & inputs.oof_predictions["horizon"].eq(5)
                    ]["paper_id"].nunique()
                ),
                "order": 4,
            },
        ]
    )
    domain = (
        inputs.papers.groupby("domain12", as_index=False)
        .size()
        .rename(columns={"size": "n_papers"})
        .sort_values("n_papers")
    )
    yearly = (
        inputs.papers.groupby("publication_year", as_index=False)
        .size()
        .rename(columns={"size": "n_papers"})
    )
    target_rows = []
    final = inputs.oof_predictions[
        inputs.oof_predictions["model_id"].eq("final_innovation_plus_k1")
    ]
    for horizon, group in final.groupby("horizon", sort=True):
        valid = group["realized_diffusion_target"].dropna()
        target_rows.append(
            {
                "horizon": int(horizon),
                "n_oof": len(group),
                "uptake_rate": float(group["future_uptake"].mean()),
                "conditional_member_rate": float(
                    group["conditional_diffusion_member"].mean()
                ),
                "realized_diffusion_mean": float(valid.mean()),
                "realized_diffusion_median": float(valid.median()),
                "realized_diffusion_p90": float(valid.quantile(0.9)),
                "label_formula": (
                    "future_uptake × fold-local conditional diffusion target"
                ),
            }
        )
    return {
        "corpus_flow": flow,
        "domain_counts": domain,
        "year_counts": yearly,
        "target_summary": pd.DataFrame(target_rows),
    }


def experiment02_tables(inputs: SuiteInputs) -> Dict[str, pd.DataFrame]:
    """Outcome-blind candidate census, gates and final five-angle map."""
    candidates = inputs.candidates.copy()
    candidates["angle_label"] = candidates["angle_id"].map(ANGLE_SHORT)
    role_counts = (
        candidates.groupby(["angle_id", "angle_label", "proposed_final_role"])
        .size()
        .rename("n_candidates")
        .reset_index()
    )
    flow = pd.DataFrame(
        [
            {"stage": "检索候选", "n": len(candidates), "order": 1},
            {
                "stage": "本地已实现",
                "n": int(inputs.candidate_coverage["implemented_in_candidate_view"].sum()),
                "order": 2,
            },
            {
                "stage": "通过全部运行门",
                "n": int(candidates["eligible_all_runtime_gates"].sum()),
                "order": 3,
            },
            {
                "stage": "数学家族竞争后主指标",
                "n": int(candidates["proposed_final_role"].eq("primary").sum()),
                "order": 4,
            },
        ]
    )
    gate_columns = (
        "coverage_pass",
        "stability_pass",
        "approximation_pass",
        "toy_test_pass",
        "temporal_test_pass",
        "nondegenerate_test_pass",
    )
    gate_matrix = candidates[
        candidates["eligible_all_runtime_gates"].eq(1)
        | candidates["proposed_final_role"].isin(["primary", "sensitivity"])
    ][["candidate_id", "angle_id", "proposed_final_role", *gate_columns]].copy()
    gate_matrix["angle_order"] = gate_matrix["angle_id"].map(
        {value: index for index, value in enumerate(ANGLE_ORDER)}
    )
    gate_matrix = gate_matrix.sort_values(
        ["angle_order", "proposed_final_role", "candidate_id"], kind="stable"
    )
    return {
        "selection_flow": flow,
        "role_counts": role_counts,
        "gate_matrix": gate_matrix,
        "primary_map": primary_records(inputs),
    }


def experiment03_tables(inputs: SuiteInputs) -> Dict[str, pd.DataFrame]:
    """Coverage, stability, approximation fidelity and construct redundancy."""
    primary = primary_records(inputs)
    quality = primary.merge(
        inputs.candidates[
            [
                "candidate_id",
                "overall_coverage",
                "minimum_domain_coverage",
                "stability_spearman",
                "stability_median_relative_error",
                "approximation_spearman",
                "approximation_median_relative_error",
            ]
        ],
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    domain_coverage = inputs.candidate_domain_coverage[
        inputs.candidate_domain_coverage["candidate_id"].isin(primary["candidate_id"])
    ].merge(
        primary[["candidate_id", "feature_label", "angle_id", "angle_label"]],
        on="candidate_id",
        how="left",
        validate="many_to_one",
    )
    feature_names = primary["feature"].tolist()
    correlations = inputs.features[feature_names].corr(method="spearman")
    correlation_long = (
        correlations.rename_axis("feature_left")
        .reset_index()
        .melt(
            id_vars="feature_left",
            var_name="feature_right",
            value_name="spearman",
        )
    )
    feature_angle = primary.set_index("feature")["angle_id"].to_dict()
    pair_rows = []
    for left_index, left in enumerate(feature_names):
        for right in feature_names[left_index + 1 :]:
            pair_rows.append(
                {
                    "feature_left": left,
                    "feature_right": right,
                    "spearman": float(correlations.loc[left, right]),
                    "pair_type": (
                        "同一角度"
                        if feature_angle[left] == feature_angle[right]
                        else "跨角度"
                    ),
                }
            )
    approximation = inputs.approximation[
        inputs.approximation["code_name"].isin(feature_names)
    ].copy()
    return {
        "primary_quality": quality,
        "primary_domain_coverage": domain_coverage,
        "feature_correlations": correlation_long,
        "correlation_pairs": pd.DataFrame(pair_rows),
        "approximation_fidelity": approximation,
    }


def _combined_main_and_pure(inputs: SuiteInputs) -> pd.DataFrame:
    return pd.concat(
        [inputs.oof_predictions, inputs.pure_predictions],
        ignore_index=True,
    )


def experiment04_tables(inputs: SuiteInputs) -> Dict[str, pd.DataFrame]:
    """Main D5 predictive value and interpretable score-strata behavior."""
    metrics = pd.concat([inputs.oof_metrics, inputs.pure_metrics], ignore_index=True)
    model_order = (
        "k0_controls",
        "k1_controls",
        "innovation_only",
        "b0_v6_primary_plus_k0",
        "provisional_core8_plus_k1",
        "final_innovation_plus_k1",
        "k2_controls",
        "final_innovation_plus_k2",
    )
    points = metrics[
        metrics["horizon"].eq(5) & metrics["model_id"].isin(model_order)
    ].copy()
    points["model_order"] = points["model_id"].map(
        {name: index for index, name in enumerate(model_order)}
    )
    points["model_label"] = points["model_id"].map(MODEL_SHORT)
    gains = inputs.comparisons.copy()
    gains["candidate_label"] = gains["candidate_model_id"].map(MODEL_SHORT)
    gains["baseline_label"] = gains["baseline_model_id"].map(MODEL_SHORT)
    deciles = prediction_deciles(
        inputs.oof_predictions,
        ["k1_controls", "final_innovation_plus_k1"],
    )
    deciles["model_label"] = deciles["model_id"].map(MODEL_SHORT)
    return {"model_points": points, "paired_gains": gains, "prediction_deciles": deciles}


def experiment05_tables(inputs: SuiteInputs) -> Dict[str, pd.DataFrame]:
    """D3/D5/D8 horizon consistency and cross-horizon rank agreement."""
    metrics = inputs.oof_metrics[
        inputs.oof_metrics["model_id"].isin(
            ["k1_controls", "final_innovation_plus_k1"]
        )
    ].copy()
    pivot = metrics.pivot(
        index="horizon", columns="model_id", values="spearman_expected"
    ).reset_index()
    pivot["spearman_gain"] = (
        pivot["final_innovation_plus_k1"] - pivot["k1_controls"]
    )
    folds = inputs.fold_metrics[
        inputs.fold_metrics["model_id"].isin(
            ["k1_controls", "final_innovation_plus_k1"]
        )
    ].pivot_table(
        index=["horizon", "outer_fold_id", "test_year_min", "test_year_max"],
        columns="model_id",
        values="spearman_expected",
    ).reset_index()
    folds["spearman_gain"] = (
        folds["final_innovation_plus_k1"] - folds["k1_controls"]
    )
    selected = inputs.oof_predictions[
        inputs.oof_predictions["model_id"].eq("final_innovation_plus_k1")
    ]
    score_wide = selected.pivot(
        index="paper_id", columns="horizon", values="expected_diffusion_score"
    )
    agreement = score_wide.corr(method="spearman")
    agreement_long = (
        agreement.rename_axis("horizon_left")
        .reset_index()
        .melt(
            id_vars="horizon_left",
            var_name="horizon_right",
            value_name="spearman",
        )
    )
    return {
        "horizon_metrics": metrics,
        "horizon_gains": pivot,
        "fold_horizon_gains": folds,
        "prediction_rank_agreement": agreement_long,
    }


def experiment06_tables(inputs: SuiteInputs) -> Dict[str, pd.DataFrame]:
    """Six-fold expanding-time generalization and year-specific drift."""
    iterations = int(inputs.suite_config["bootstrap"]["iterations"])
    seed = int(inputs.suite_config["bootstrap"]["seed"])
    d5 = inputs.oof_predictions[inputs.oof_predictions["horizon"].eq(5)]
    ci_rows = []
    for fold_id, group in d5.groupby("outer_fold_id", sort=True):
        row = paired_gain_interval(
            group,
            "final_innovation_plus_k1",
            "k1_controls",
            iterations=iterations,
            seed=seed + int(fold_id),
        )
        row["outer_fold_id"] = int(fold_id)
        row["test_year_min"] = int(group["publication_year"].min())
        row["test_year_max"] = int(group["publication_year"].max())
        ci_rows.append(row)
    yearly_rows = []
    selected = d5[
        d5["model_id"].isin(["k1_controls", "final_innovation_plus_k1"])
    ]
    for (year, model_id), group in selected.groupby(
        ["publication_year", "model_id"], sort=True
    ):
        yearly_rows.append(
            {
                "publication_year": int(year),
                "model_id": str(model_id),
                "n": len(group),
                "spearman_expected": safe_spearman(
                    group["realized_diffusion_target"],
                    group["expected_diffusion_score"],
                ),
            }
        )
    fold_metrics = pd.concat(
        [
            inputs.fold_metrics[
                inputs.fold_metrics["horizon"].eq(5)
                & inputs.fold_metrics["model_id"].isin(
                    ["k1_controls", "final_innovation_plus_k1"]
                )
            ],
            inputs.pure_fold_metrics,
        ],
        ignore_index=True,
    )
    return {
        "temporal_folds": inputs.folds[inputs.folds["horizon"].eq(5)].copy(),
        "fold_metrics": fold_metrics,
        "fold_gain_intervals": pd.DataFrame(ci_rows),
        "yearly_metrics": pd.DataFrame(yearly_rows),
    }


def experiment07_tables(inputs: SuiteInputs) -> Dict[str, pd.DataFrame]:
    """Twelve-domain generalization with paired uncertainty."""
    iterations = int(inputs.suite_config["bootstrap"]["iterations"])
    seed = int(inputs.suite_config["bootstrap"]["seed"])
    d5 = inputs.oof_predictions[inputs.oof_predictions["horizon"].eq(5)]
    ci_rows = []
    for index, (domain, group) in enumerate(d5.groupby("domain12", sort=True)):
        row = paired_gain_interval(
            group,
            "final_innovation_plus_k1",
            "k1_controls",
            iterations=iterations,
            seed=seed + 100 + index,
        )
        row["domain12"] = str(domain)
        ci_rows.append(row)
    pure_rows = []
    for domain, group in inputs.pure_predictions.groupby("domain12", sort=True):
        pure_rows.append(
            {
                "domain12": str(domain),
                "model_id": "innovation_only",
                "n_oof": len(group),
                "spearman_expected": safe_spearman(
                    group["realized_diffusion_target"],
                    group["expected_diffusion_score"],
                ),
            }
        )
    metrics = inputs.domain_metrics[
        inputs.domain_metrics["horizon"].eq(5)
        & inputs.domain_metrics["model_id"].isin(
            ["k1_controls", "final_innovation_plus_k1"]
        )
    ].copy()
    return {
        "domain_metrics": metrics,
        "domain_gain_intervals": pd.DataFrame(ci_rows),
        "pure_domain_metrics": pd.DataFrame(pure_rows),
    }


def experiment08_tables(inputs: SuiteInputs) -> Dict[str, pd.DataFrame]:
    """Innovation-only signal relative to control-only and full models."""
    combined = _combined_main_and_pure(inputs)
    model_ids = ["k1_controls", "innovation_only", "final_innovation_plus_k1"]
    metrics = pd.concat([inputs.oof_metrics, inputs.pure_metrics], ignore_index=True)
    metrics = metrics[
        metrics["horizon"].eq(5) & metrics["model_id"].isin(model_ids)
    ].copy()
    folds = pd.concat(
        [
            inputs.fold_metrics[
                inputs.fold_metrics["horizon"].eq(5)
                & inputs.fold_metrics["model_id"].isin(
                    ["k1_controls", "final_innovation_plus_k1"]
                )
            ],
            inputs.pure_fold_metrics,
        ],
        ignore_index=True,
    )
    deciles = prediction_deciles(combined, model_ids)
    score_wide = combined[
        combined["horizon"].eq(5) & combined["model_id"].isin(model_ids)
    ].pivot(index="paper_id", columns="model_id", values="expected_diffusion_score")
    correlation = score_wide.corr(method="spearman")
    correlation_long = (
        correlation.rename_axis("model_left")
        .reset_index()
        .melt(
            id_vars="model_left",
            var_name="model_right",
            value_name="spearman",
        )
    )
    return {
        "model_metrics": metrics,
        "fold_metrics": folds,
        "prediction_deciles": deciles,
        "prediction_correlations": correlation_long,
    }


def angle_feature_sets(inputs: SuiteInputs) -> Dict[str, Tuple[str, ...]]:
    """Build the ten fixed angle-addition/deletion feature sets."""
    primary = primary_records(inputs)
    k1 = tuple(inputs.source_config["k1_controls"])
    all_features = tuple(primary["feature"])
    output: Dict[str, Tuple[str, ...]] = {}
    for index, angle in enumerate(ANGLE_ORDER, start=1):
        angle_features = tuple(
            primary.loc[primary["angle_id"].eq(angle), "feature"]
        )
        output[f"k1_plus_a{index}"] = tuple(dict.fromkeys((*k1, *angle_features)))
        retained = tuple(name for name in all_features if name not in angle_features)
        output[f"final_minus_a{index}"] = tuple(dict.fromkeys((*k1, *retained)))
    return output


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def run_angle_ablation(
    inputs: SuiteInputs,
    output_dir: Path,
) -> Dict[str, pd.DataFrame]:
    """Run or resume the ten fixed D5 angle-ablation OOF models."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "ablation_manifest.json"
    paths = {
        "predictions": output_dir / "angle_ablation_oof_predictions.parquet",
        "metrics": output_dir / "angle_ablation_metrics.csv",
        "fold_metrics": output_dir / "angle_ablation_fold_metrics.csv",
        "domain_metrics": output_dir / "angle_ablation_domain_metrics.csv",
        "folds": output_dir / "angle_ablation_temporal_folds.csv",
    }
    if manifest_path.is_file():
        manifest = _load_json(manifest_path)
        for name, record in manifest["outputs"].items():
            path = Path(record["path"])
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                raise ValueError(f"Invalid ablation artifact: {name}")
        return {
            "predictions": pd.read_parquet(paths["predictions"]),
            "metrics": pd.read_csv(paths["metrics"]),
            "fold_metrics": pd.read_csv(paths["fold_metrics"]),
            "domain_metrics": pd.read_csv(paths["domain_metrics"]),
            "folds": pd.read_csv(paths["folds"]),
        }

    feature_sets = angle_feature_sets(inputs)
    frame = assemble_all_period_frame(inputs.dataset_root, horizon=5)
    predictions, folds = run_fixed_medium_oof(
        frame,
        feature_sets=feature_sets,
        model_ids=tuple(feature_sets),
        fold_config=inputs.source_config["temporal_folds"],
        parameters=inputs.source_config["model"],
        categorical_features=inputs.source_config["categorical_features"],
        inner_folds=int(inputs.source_config["model"]["inner_temporal_folds"]),
        horizon=5,
        checkpoint_root=output_dir / "checkpoints",
        seed=int(inputs.source_config["model"]["seed"]),
    )
    metrics, fold_metrics, domain_metrics = evaluate_oof_points(
        predictions,
        minimum_domain_rows=int(
            inputs.source_config["evaluation"]["minimum_domain_rows"]
        ),
    )
    predictions.to_parquet(paths["predictions"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    fold_metrics.to_csv(paths["fold_metrics"], index=False)
    domain_metrics.to_csv(paths["domain_metrics"], index=False)
    folds.to_csv(paths["folds"], index=False)
    lineage = {
        "suite_id": inputs.suite_config["suite_id"],
        "source_oof_artifact_id": inputs.oof_manifest["artifact_id"],
        "source_config_sha256": sha256_file(inputs.source_config_path),
        "registry_sha256": sha256_file(inputs.registry_path),
        "feature_sets": {name: list(values) for name, values in feature_sets.items()},
        "horizon": 5,
        "parameter_id": "medium",
        "seed": int(inputs.source_config["model"]["seed"]),
        "outcome_dependent_selection": False,
        "claim_scope": (
            "Post-hoc predictive interpretation only; no indicator selection "
            "or causal attribution."
        ),
    }
    manifest: Dict[str, Any] = {
        "artifact_kind": "aspr_v6_1_five_angle_oof_ablation",
        "lineage": lineage,
        "outputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        },
    }
    manifest["artifact_id"] = _canonical_hash(manifest)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "predictions": predictions,
        "metrics": metrics,
        "fold_metrics": fold_metrics,
        "domain_metrics": domain_metrics,
        "folds": folds,
    }


def experiment09_tables(
    inputs: SuiteInputs,
    ablation: Mapping[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """Five-angle incremental and deletion analyses."""
    baseline = float(
        inputs.oof_metrics.loc[
            inputs.oof_metrics["horizon"].eq(5)
            & inputs.oof_metrics["model_id"].eq("k1_controls"),
            "spearman_expected",
        ].iloc[0]
    )
    full = float(
        inputs.oof_metrics.loc[
            inputs.oof_metrics["horizon"].eq(5)
            & inputs.oof_metrics["model_id"].eq("final_innovation_plus_k1"),
            "spearman_expected",
        ].iloc[0]
    )
    rows = []
    for index, angle in enumerate(ANGLE_ORDER, start=1):
        add_model = f"k1_plus_a{index}"
        delete_model = f"final_minus_a{index}"
        add_value = float(
            ablation["metrics"].loc[
                ablation["metrics"]["model_id"].eq(add_model),
                "spearman_expected",
            ].iloc[0]
        )
        delete_value = float(
            ablation["metrics"].loc[
                ablation["metrics"]["model_id"].eq(delete_model),
                "spearman_expected",
            ].iloc[0]
        )
        rows.append(
            {
                "angle_id": angle,
                "angle_label": ANGLE_SHORT[angle],
                "angle_number": index,
                "k1_spearman": baseline,
                "k1_plus_angle_spearman": add_value,
                "increment_over_k1": add_value - baseline,
                "full_spearman": full,
                "minus_angle_spearman": delete_value,
                "drop_from_full": full - delete_value,
            }
        )
    fold = ablation["fold_metrics"].copy()
    full_fold = inputs.fold_metrics[
        inputs.fold_metrics["horizon"].eq(5)
        & inputs.fold_metrics["model_id"].eq("final_innovation_plus_k1")
    ][["outer_fold_id", "spearman_expected"]].rename(
        columns={"spearman_expected": "full_spearman"}
    )
    deletion = fold[fold["model_id"].str.startswith("final_minus_a")].merge(
        full_fold,
        on="outer_fold_id",
        how="left",
        validate="many_to_one",
    )
    deletion["angle_number"] = deletion["model_id"].str.extract(
        r"a(\d+)$"
    )[0].astype(int)
    deletion["angle_id"] = deletion["angle_number"].map(
        {index: angle for index, angle in enumerate(ANGLE_ORDER, start=1)}
    )
    deletion["angle_label"] = deletion["angle_id"].map(ANGLE_SHORT)
    deletion["drop_from_full"] = (
        deletion["full_spearman"] - deletion["spearman_expected"]
    )
    return {
        "angle_summary": pd.DataFrame(rows),
        "fold_deletion": deletion,
        "ablation_metrics": ablation["metrics"],
        "ablation_fold_metrics": ablation["fold_metrics"],
    }


def experiment10_tables(inputs: SuiteInputs) -> Dict[str, pd.DataFrame]:
    """Control sensitivity, stress-test synthesis and reproducibility gates."""
    metrics = pd.concat([inputs.oof_metrics, inputs.pure_metrics], ignore_index=True)
    sensitivity_ids = [
        "k0_controls",
        "k1_controls",
        "k2_controls",
        "innovation_only",
        "b0_v6_primary_plus_k0",
        "provisional_core8_plus_k1",
        "final_innovation_plus_k1",
        "final_innovation_plus_k2",
    ]
    sensitivity = metrics[
        metrics["horizon"].eq(5) & metrics["model_id"].isin(sensitivity_ids)
    ].copy()
    sensitivity["model_label"] = sensitivity["model_id"].map(MODEL_SHORT)

    acceptance = inputs.oof_manifest["acceptance"]
    comparisons = inputs.comparisons.set_index("baseline_model_id")
    gate_rows = [
        {
            "gate": "D5达到目标",
            "conservative_value": float(acceptance["headline_d5_spearman"]),
            "threshold": 0.75,
            "margin": float(acceptance["headline_d5_spearman"]) - 0.75,
        },
        {
            "gate": "相对K1增量下界>0",
            "conservative_value": float(
                comparisons.loc["k1_controls", "gain_ci_low"]
            ),
            "threshold": 0.0,
            "margin": float(comparisons.loc["k1_controls", "gain_ci_low"]),
        },
        {
            "gate": "相对B0非劣下界≥−0.005",
            "conservative_value": float(
                comparisons.loc["b0_v6_primary_plus_k0", "gain_ci_low"]
            ),
            "threshold": -0.005,
            "margin": float(
                comparisons.loc["b0_v6_primary_plus_k0", "gain_ci_low"]
            )
            + 0.005,
        },
        {
            "gate": "D3增量>0",
            "conservative_value": float(acceptance["d3_gain_over_k1"]),
            "threshold": 0.0,
            "margin": float(acceptance["d3_gain_over_k1"]),
        },
        {
            "gate": "D8增量>0",
            "conservative_value": float(acceptance["d8_gain_over_k1"]),
            "threshold": 0.0,
            "margin": float(acceptance["d8_gain_over_k1"]),
        },
    ]
    horizon = experiment05_tables(inputs)["horizon_gains"][
        ["horizon", "spearman_gain"]
    ].assign(stratum="预测窗口")
    fold = experiment05_tables(inputs)["fold_horizon_gains"][
        ["horizon", "outer_fold_id", "spearman_gain"]
    ].assign(stratum="时间折")
    domain = inputs.domain_metrics[
        inputs.domain_metrics["horizon"].eq(5)
        & inputs.domain_metrics["model_id"].isin(
            ["k1_controls", "final_innovation_plus_k1"]
        )
    ].pivot(
        index="domain12", columns="model_id", values="spearman_expected"
    ).reset_index()
    domain["spearman_gain"] = (
        domain["final_innovation_plus_k1"] - domain["k1_controls"]
    )
    domain["stratum"] = "学科"
    stress = pd.concat(
        [
            horizon.rename(columns={"horizon": "unit"})[
                ["stratum", "unit", "spearman_gain"]
            ],
            fold.assign(
                unit=lambda frame: "D"
                + frame["horizon"].astype(str)
                + "-F"
                + frame["outer_fold_id"].astype(str)
            )[["stratum", "unit", "spearman_gain"]],
            domain.rename(columns={"domain12": "unit"})[
                ["stratum", "unit", "spearman_gain"]
            ],
        ],
        ignore_index=True,
    )
    replay = inputs.reproducibility
    completion = inputs.completion_audit
    audit = pd.DataFrame(
        [
            {
                "check": "方案完成审计",
                "passed": int(completion["n_passed"]),
                "total": int(completion["n_checks"]),
            },
            {
                "check": "OOF检查点精确复跑",
                "passed": int(replay["n_full_replay_checkpoints"]),
                "total": int(replay["n_checkpoints_verified"]),
            },
            {
                "check": "时间折/窗口测试集核验",
                "passed": int(replay["n_horizon_fold_test_set_groups_verified"]),
                "total": 18,
            },
            {
                "check": "12大类保留",
                "passed": int(inputs.papers["domain12"].nunique()),
                "total": 12,
            },
            {
                "check": "冻结输出哈希核验",
                "passed": int(replay["n_manifest_outputs_verified"]),
                "total": 8,
            },
        ]
    )
    audit["completion_rate"] = audit["passed"] / audit["total"]
    return {
        "control_sensitivity": sensitivity,
        "acceptance_gates": pd.DataFrame(gate_rows),
        "stress_test_gains": stress,
        "reproducibility_checks": audit,
    }


def source_hash_table(inputs: SuiteInputs) -> pd.DataFrame:
    """Return a deterministic lineage table for every source file."""
    rows = []
    for path in sorted(set(inputs.source_paths)):
        rows.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return pd.DataFrame(rows)


def hash_payload(payload: Mapping[str, Any]) -> str:
    """Public canonical hash helper for the run manifest."""
    return _canonical_hash(payload)


__all__ = [
    "ANGLE_ORDER",
    "ANGLE_SHORT",
    "FEATURE_SHORT",
    "MODEL_SHORT",
    "SuiteInputs",
    "angle_feature_sets",
    "experiment01_tables",
    "experiment02_tables",
    "experiment03_tables",
    "experiment04_tables",
    "experiment05_tables",
    "experiment06_tables",
    "experiment07_tables",
    "experiment08_tables",
    "experiment09_tables",
    "experiment10_tables",
    "hash_payload",
    "load_inputs",
    "prediction_deciles",
    "primary_records",
    "run_angle_ablation",
    "source_hash_table",
]

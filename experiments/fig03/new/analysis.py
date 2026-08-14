"""Build auditable panel data for the ASPR performance-landscape figure."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from gear.calibration_assets import load_calibration_release

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
CONFIG_PATH = HERE / "config.json"


def read_json(path: Path) -> dict[str, Any]:
    """Read a UTF-8 JSON object."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable, human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def resolve_path(value: str) -> Path:
    """Resolve a project-relative path."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load and minimally validate the figure configuration."""
    config = read_json(path)
    required = {
        "calibration_release",
        "model_config",
        "output_dir",
        "horizons",
        "model_sets",
        "domain_order",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Fig.3 config lacks required keys: {missing}")
    return config


def calibration_source_dir(config: Mapping[str, Any]) -> Path:
    """Resolve inputs through the frozen public calibration registry."""
    return load_calibration_release(str(config["calibration_release"])).asset_root


def clean_generated_outputs(output_dir: Path) -> None:
    """Remove only allowlisted generated artifacts from the superseded Fig.3."""
    for directory in (output_dir / "panel_data", output_dir / "panels"):
        if directory.is_dir():
            shutil.rmtree(directory)
    for name in (
        "figure_full.png",
        "figure_full.svg",
        "figure_full.pdf",
        "figure_full_grayscale.png",
        "audit_report.json",
        "chart_contract.json",
        "output_inventory.json",
        "panel_text.json",
        "run_manifest.json",
        "figure_quality_report.json",
        "panel_data_manifest.json",
    ):
        path = output_dir / name
        if path.is_file():
            path.unlink()


def safe_spearman(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute Spearman correlation after finite-value filtering."""
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return np.nan
    x = x[valid]
    y = y[valid]
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan
    return float(spearmanr(x, y).statistic)


def rank_decile(values: pd.Series) -> pd.Series:
    """Assign deterministic, nearly equal-size deciles numbered 1 to 10."""
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna() & np.isfinite(numeric)
    output = pd.Series(np.nan, index=values.index, dtype=float)
    count = int(valid.sum())
    if count:
        ranks = numeric.loc[valid].rank(method="first")
        output.loc[valid] = pd.qcut(ranks, 10, labels=False).astype(int) + 1
    return output


def load_hgb_predictions(config: Mapping[str, Any]) -> pd.DataFrame:
    """Load only the formal HGB prediction rows required by the figure."""
    source_dir = calibration_source_dir(config)
    columns = [
        "paper_id",
        "publication_year",
        "domain12",
        "horizon",
        "model_id",
        "outer_fold_id",
        "realized_diffusion_target",
        "expected_diffusion_score",
        "model_family",
    ]
    frame = pd.read_parquet(source_dir / "oof_predictions.parquet", columns=columns)
    model_ids = {str(row["id"]) for row in config["model_sets"]}
    selected = frame[
        frame["model_family"].eq(str(config["model_family"]))
        & frame["horizon"].isin([int(value) for value in config["horizons"]])
        & frame["model_id"].isin(model_ids)
    ].copy()
    expected = {int(value) for value in config["horizons"]}
    if set(selected["horizon"].unique()) != expected:
        raise ValueError("formal HGB predictions do not cover all registered horizons")
    return selected


def build_score_summary(config: Mapping[str, Any], panel_dir: Path) -> pd.DataFrame:
    """Audit the official two-field ASPR production score."""
    source_dir = calibration_source_dir(config)
    scores = pd.read_parquet(source_dir / "official_aspr_scores.parquet")
    required = {"raw_prediction_score", "aspr_score", "paper_id"}
    if not required.issubset(scores.columns):
        raise ValueError("official ASPR score file lacks required score columns")
    ordered = scores.sort_values("raw_prediction_score")
    monotone = bool(ordered["aspr_score"].diff().fillna(0).ge(-1e-12).all())
    summary = pd.DataFrame(
        [
            {
                "scored_papers": len(scores),
                "paper_year_max": 2022,
                "mature_d5_year_max": int(config["mature_year_max"]["5"]),
                "official_model_family": str(scores["official_model_family"].iloc[0]),
                "official_feature_set": str(scores["official_feature_set"].iloc[0]),
                "raw_prediction_min": float(scores["raw_prediction_score"].min()),
                "raw_prediction_max": float(scores["raw_prediction_score"].max()),
                "aspr_score_min": float(scores["aspr_score"].min()),
                "aspr_score_max": float(scores["aspr_score"].max()),
                "score_monotone": monotone,
            }
        ]
    )
    summary.to_csv(panel_dir / "score_summary.csv", index=False)
    return summary


def build_decile_enrichment(
    predictions: pd.DataFrame,
    config: Mapping[str, Any],
    panel_dir: Path,
) -> pd.DataFrame:
    """Build D3/D5/D8 fold-local enrichment with year-block bootstrap."""
    tables = [
        _build_horizon_decile_enrichment(
            predictions,
            horizon=int(horizon),
            model_id=str(model["id"]),
            model_label=str(model["label"]),
            iterations=int(config["bootstrap_iterations"]),
            seed=int(config["bootstrap_seed"]),
        )
        for horizon in config["horizons"]
        for model in config["model_sets"]
    ]
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(panel_dir / "decile_enrichment.csv", index=False)
    return table


def _build_horizon_decile_enrichment(
    predictions: pd.DataFrame,
    *,
    horizon: int,
    model_id: str,
    model_label: str,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    """Build one horizon's fold-local enrichment table."""
    selected = predictions[
        predictions["horizon"].eq(horizon) & predictions["model_id"].eq(model_id)
    ].copy()
    prediction_values = pd.to_numeric(
        selected["expected_diffusion_score"], errors="coerce"
    )
    target_values = pd.to_numeric(
        selected["realized_diffusion_target"], errors="coerce"
    )
    selected = selected[
        np.isfinite(prediction_values) & np.isfinite(target_values)
    ].copy()
    if selected.empty:
        raise ValueError(f"D{horizon} {model_id} has no finite OOF pairs")
    selected["prediction_decile"] = selected.groupby("outer_fold_id", group_keys=False)[
        "expected_diffusion_score"
    ].apply(rank_decile)
    selected["realized_decile"] = selected.groupby("outer_fold_id", group_keys=False)[
        "realized_diffusion_target"
    ].apply(rank_decile)
    selected["realized_top_decile"] = selected["realized_decile"].eq(10)
    yearly = selected.groupby(
        ["publication_year", "prediction_decile"], as_index=False
    ).agg(n=("paper_id", "size"), top_count=("realized_top_decile", "sum"))
    years = np.asarray(sorted(selected["publication_year"].unique()), dtype=int)
    deciles = np.arange(1, 11, dtype=int)
    n_matrix = _year_decile_matrix(yearly, years, deciles, "n")
    top_matrix = _year_decile_matrix(yearly, years, deciles, "top_count")
    draws = _bootstrap_decile_shares(
        n_matrix,
        top_matrix,
        iterations=iterations,
        seed=seed,
    )
    overall = selected.groupby("prediction_decile").agg(
        n=("paper_id", "size"),
        top_count=("realized_top_decile", "sum"),
    )
    base_share = float(selected["realized_top_decile"].mean())
    table = overall.reset_index()
    table.insert(0, "horizon", horizon)
    table.insert(1, "model_id", model_id)
    table.insert(2, "model_label", model_label)
    table["observed_top_share"] = table["top_count"] / table["n"]
    table["baseline_top_share"] = base_share
    table["enrichment_over_baseline"] = table["observed_top_share"] / base_share
    table["ci_low"] = np.quantile(draws, 0.025, axis=0)
    table["ci_high"] = np.quantile(draws, 0.975, axis=0)
    table["bootstrap_iterations"] = iterations
    table["bootstrap_unit"] = "publication_year"
    return table


def _year_decile_matrix(
    yearly: pd.DataFrame,
    years: np.ndarray,
    deciles: np.ndarray,
    value: str,
) -> np.ndarray:
    pivot = yearly.pivot(
        index="publication_year", columns="prediction_decile", values=value
    )
    return (
        pivot.reindex(index=years, columns=deciles, fill_value=0)
        .fillna(0)
        .to_numpy(dtype=float)
    )


def _bootstrap_decile_shares(
    n_matrix: np.ndarray,
    top_matrix: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    year_count = n_matrix.shape[0]
    draws = np.empty((iterations, n_matrix.shape[1]), dtype=float)
    for index in range(iterations):
        sampled = rng.integers(0, year_count, size=year_count)
        counts = np.bincount(sampled, minlength=year_count).astype(float)
        denominator = counts @ n_matrix
        draws[index] = np.divide(
            counts @ top_matrix,
            denominator,
            out=np.full(n_matrix.shape[1], np.nan),
            where=denominator > 0,
        )
    return draws


def load_overall_metrics(config: Mapping[str, Any]) -> pd.DataFrame:
    """Load the 12 registered formal-HGB overall OOF metrics."""
    source_dir = calibration_source_dir(config)
    metrics = pd.read_csv(source_dir / "oof_metrics.csv")
    selected = metrics[
        metrics["model_family"].eq(str(config["model_family"]))
        & metrics["horizon"].isin([int(value) for value in config["horizons"]])
        & metrics["model_id"].isin([str(row["id"]) for row in config["model_sets"]])
    ].copy()
    if len(selected) != 12:
        raise ValueError(f"expected 12 formal HGB metrics, found {len(selected)}")
    return selected


def build_performance_landscape(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    config: Mapping[str, Any],
    panel_dir: Path,
) -> tuple[pd.DataFrame, Mapping[str, float]]:
    """Compute all horizon/set/domain trailing-window performance cells."""
    model_labels = {str(row["id"]): str(row["label"]) for row in config["model_sets"]}
    domain_labels = {
        str(row["id"]): str(row["label"]) for row in config["domain_order"]
    }
    overall = metrics.set_index(["horizon", "model_id"])[["n_oof", "spearman"]]
    grouped = {
        (int(horizon), str(model_id), str(domain)): group
        for (horizon, model_id, domain), group in predictions.groupby(
            ["horizon", "model_id", "domain12"], sort=False
        )
    }
    rows: list[dict[str, Any]] = []
    for horizon in [int(value) for value in config["horizons"]]:
        mature_max = int(config["mature_year_max"][str(horizon)])
        for model in config["model_sets"]:
            model_id = str(model["id"])
            for domain in config["domain_order"]:
                domain_id = str(domain["id"])
                group = grouped.get((horizon, model_id, domain_id), pd.DataFrame())
                rows.extend(
                    _performance_rows(
                        group,
                        horizon=horizon,
                        mature_max=mature_max,
                        model_id=model_id,
                        model_label=model_labels[model_id],
                        domain_id=domain_id,
                        domain_label=domain_labels[domain_id],
                        overall_n=int(overall.loc[(horizon, model_id), "n_oof"]),
                        overall_rho=float(overall.loc[(horizon, model_id), "spearman"]),
                        config=config,
                    )
                )
    table = pd.DataFrame(rows)
    table.to_csv(panel_dir / "performance_landscape.csv", index=False)
    density = _data_density_audit(predictions, table, config)
    return table, density


def _performance_rows(
    group: pd.DataFrame,
    *,
    horizon: int,
    mature_max: int,
    model_id: str,
    model_label: str,
    domain_id: str,
    domain_label: str,
    overall_n: int,
    overall_rho: float,
    config: Mapping[str, Any],
) -> Iterable[dict[str, Any]]:
    window = int(config["rolling_window_years"])
    minimum_n = int(config["minimum_cell_n"])
    for end_year in range(
        int(config["landscape_year_min"]), int(config["landscape_year_max"]) + 1
    ):
        start_year = end_year - window + 1
        status = "not_mature" if end_year > mature_max else "insufficient"
        n = 0
        rho = np.nan
        if status != "not_mature" and not group.empty:
            subset = group[group["publication_year"].between(start_year, end_year)]
            valid = subset.dropna(
                subset=["realized_diffusion_target", "expected_diffusion_score"]
            )
            n = len(valid)
            if n >= minimum_n:
                rho = safe_spearman(
                    valid["realized_diffusion_target"],
                    valid["expected_diffusion_score"],
                )
                status = "reliable" if np.isfinite(rho) else "degenerate"
        yield {
            "horizon": horizon,
            "model_id": model_id,
            "model_label": model_label,
            "domain12": domain_id,
            "domain_label": domain_label,
            "window_start": start_year,
            "window_end": end_year,
            "n": n,
            "spearman": rho,
            "status": status,
            "overall_n_oof": overall_n,
            "overall_spearman": overall_rho,
        }


def _data_density_audit(
    predictions: pd.DataFrame,
    landscape: pd.DataFrame,
    config: Mapping[str, Any],
) -> Mapping[str, float]:
    prediction_values = pd.to_numeric(
        predictions["expected_diffusion_score"], errors="coerce"
    )
    target_values = pd.to_numeric(
        predictions["realized_diffusion_target"], errors="coerce"
    )
    valid_predictions = predictions[
        np.isfinite(prediction_values) & np.isfinite(target_values)
    ]
    single = valid_predictions.groupby(
        ["horizon", "model_id", "domain12", "publication_year"], sort=False
    ).size()
    single_share = float(single.ge(int(config["minimum_cell_n"])).mean())
    mature = landscape[landscape["status"].ne("not_mature")]
    rolling_share = float(mature["status"].eq("reliable").mean())
    return {
        "single_year_cells": len(single),
        "single_year_n_ge_30_share": single_share,
        "rolling_mature_cells": len(mature),
        "rolling_reliable_share": rolling_share,
    }


def build_domain_display_order(
    landscape: pd.DataFrame,
    config: Mapping[str, Any],
    panel_dir: Path,
) -> pd.DataFrame:
    """Freeze the display order from official-model reliable-cell performance."""
    selected = landscape[
        landscape["horizon"].eq(int(config["primary_horizon"]))
        & landscape["model_id"].eq(str(config["primary_model_id"]))
        & landscape["status"].eq("reliable")
    ]
    means = selected.groupby("domain12", observed=True)["spearman"].mean()
    labels = {str(row["id"]): str(row["label"]) for row in config["domain_order"]}
    expected = set(labels)
    if set(means.index.astype(str)) != expected:
        missing = sorted(expected - set(means.index.astype(str)))
        raise ValueError(f"Official-model domain order lacks reliable cells: {missing}")
    ordered = means.sort_values(ascending=False, kind="stable")
    table = pd.DataFrame(
        {
            "display_order": np.arange(1, len(ordered) + 1, dtype=int),
            "domain12": ordered.index.astype(str),
            "domain_label": [labels[str(domain)] for domain in ordered.index],
            "mean_reliable_spearman": ordered.to_numpy(dtype=float),
        }
    )
    table.to_csv(panel_dir / "domain_display_order.csv", index=False)
    return table


def freeze_panel_tables(panel_dir: Path, output_dir: Path) -> Mapping[str, Any]:
    """Hash the newly computed panel tables before rendering."""
    tables = {
        path.name: {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
        for path in sorted(panel_dir.glob("*.csv"))
    }
    manifest = {
        "artifact_kind": "fig3_recomputed_panel_data",
        "numeric_data_recomputed": True,
        "tables": tables,
    }
    write_json(output_dir / "panel_data_manifest.json", manifest)
    return manifest


def build_gain_landscape(
    landscape: pd.DataFrame,
    config: Mapping[str, Any],
    panel_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the three registered adjacent D5 feature-set gains."""
    primary = landscape[landscape["horizon"].eq(int(config["primary_horizon"]))]
    lookup = primary.set_index(["domain12", "window_end", "model_id"])
    comparisons = [
        ("fulltext_16", "strict_7", "Full-text 16 − Strict 7"),
        ("source_154", "fulltext_16", "Primary 154 − Full-text 16"),
        ("ultrarelaxed_221", "source_154", "Broad T0 221 − Primary 154"),
    ]
    rows: list[dict[str, Any]] = []
    limit = float(config["gain_color_limit"])
    for candidate, baseline, label in comparisons:
        rows.extend(_gain_rows(lookup, candidate, baseline, label, limit, config))
    gains = pd.DataFrame(rows)
    summaries = []
    for order, (_, group) in enumerate(
        gains.groupby("comparison_label", sort=False), start=1
    ):
        values = group.loc[group["status"].eq("reliable"), "delta_spearman"]
        summaries.append(
            {
                "comparison_order": order,
                "comparison_label": str(group["comparison_label"].iloc[0]),
                "candidate_model_id": str(group["candidate_model_id"].iloc[0]),
                "baseline_model_id": str(group["baseline_model_id"].iloc[0]),
                "reliable_cells": len(values),
                "median_delta_spearman": float(values.median()),
                "positive_share": float(values.gt(0).mean()),
                "q10": float(values.quantile(0.10)),
                "q90": float(values.quantile(0.90)),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
        )
    summary = pd.DataFrame(summaries)
    gains.to_csv(panel_dir / "d5_gain_landscape.csv", index=False)
    summary.to_csv(panel_dir / "d5_gain_summary.csv", index=False)
    return gains, summary


def _gain_rows(
    lookup: pd.DataFrame,
    candidate: str,
    baseline: str,
    label: str,
    limit: float,
    config: Mapping[str, Any],
) -> Iterable[dict[str, Any]]:
    domain_labels = {
        str(row["id"]): str(row["label"]) for row in config["domain_order"]
    }
    for domain in [str(row["id"]) for row in config["domain_order"]]:
        for end_year in range(
            int(config["landscape_year_min"]), int(config["landscape_year_max"]) + 1
        ):
            candidate_row = lookup.loc[(domain, end_year, candidate)]
            baseline_row = lookup.loc[(domain, end_year, baseline)]
            statuses = {str(candidate_row["status"]), str(baseline_row["status"])}
            if "not_mature" in statuses:
                status = "not_mature"
            elif statuses == {"reliable"}:
                status = "reliable"
            elif "degenerate" in statuses:
                status = "degenerate"
            else:
                status = "insufficient"
            delta = np.nan
            if status == "reliable":
                delta = float(candidate_row["spearman"] - baseline_row["spearman"])
            yield {
                "comparison_label": label,
                "candidate_model_id": candidate,
                "baseline_model_id": baseline,
                "domain12": domain,
                "domain_label": domain_labels[domain],
                "window_start": end_year - int(config["rolling_window_years"]) + 1,
                "window_end": end_year,
                "n": int(min(candidate_row["n"], baseline_row["n"])),
                "delta_spearman": delta,
                "display_delta": (
                    float(np.clip(delta, -limit, limit))
                    if np.isfinite(delta)
                    else np.nan
                ),
                "out_of_scale": bool(np.isfinite(delta) and abs(delta) > limit),
                "status": status,
            }


def chart_contract(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the frozen visual contract."""
    return {
        "figure_id": 3,
        "figure_version": config["figure_version"],
        "analytical_question": "Can publication-time ASPR scores rank subsequent scientific diffusion across horizons, fields, and publication years?",
        "takeaway": "ASPR provides strong out-of-time ranking and top-decile enrichment; Full-text 16 captures most local predictive gain while broader proxy sets show saturation.",
        "surface": "standalone static research figure",
        "renderer": "matplotlib",
        "panels": {
            "a": "formal HGB score construction",
            "b": "twelve D3/D5/D8 × four-set fold-local OOF decile curves with year-block bootstrap",
            "c": "3-by-4 trailing three-year performance heatmap board",
            "d": "continuous D5 Full-text 16 3D terrain within domain-observed mature-year endpoints",
            "e": "three D5 adjacent feature-set gain heatmaps",
        },
        "data_sufficiency": {
            "rolling_window_years": int(config["rolling_window_years"]),
            "minimum_cell_n": int(config["minimum_cell_n"]),
            "insufficient_cells": "neutral cells with n < 30 and excluded from correlations",
            "degenerate_cells": "separately marked cells where constant ranks make Spearman undefined",
            "not_mature_cells": "separately shaded; never treated as missing performance",
        },
        "palette_policy": "blue-to-red sequential scale for absolute performance; blue/white/orange diverging scale for signed gains; distinct neutral status fills",
        "interpolation": "forbidden in exact Panels c/e; Panel d may bridge internal years only within each domain's reliable observed endpoints and never extrapolates",
        "model_family": "hgb",
        "claim_boundary": "predictive association with later uptake and diffusion, not causality or a direct novelty judgment",
    }


def panel_text(deciles: pd.DataFrame, gain_summary: pd.DataFrame) -> Mapping[str, Any]:
    """Return figure-ready explanatory text derived from panel data."""
    primary = deciles.loc[
        deciles["horizon"].eq(5) & deciles["model_id"].eq("fulltext_16")
    ]
    top = primary.loc[primary["prediction_decile"].eq(10)].iloc[0]
    low = primary.loc[primary["prediction_decile"].eq(1)].iloc[0]
    return {
        "title": "Temporal–disciplinary performance landscape and out-of-time validation of ASPR Score",
        "panel_b_callout": (
            f"Highest ASPR decile: {100 * top['observed_top_share']:.1f}% reached "
            f"the top diffusion decile ({top['enrichment_over_baseline']:.2f}× enrichment); "
            f"lowest decile: {100 * low['observed_top_share']:.2f}%."
        ),
        "panel_d_interpretation": (
            "Expanding Strict 7 to Full-text 16 produced a broadly positive local gain. "
            "Further expansion to 154 or 221 indicators yielded near-zero median gains "
            "and heterogeneous positive and negative changes."
        ),
        "caption": (
            "Fig. 3 | Temporal–disciplinary performance landscape and out-of-time validation of ASPR Score. "
            "a, Publication-time indicators are integrated by a calibrated two-part HGB model. The resulting "
            "raw_prediction_score is converted to the 0–100 aspr_score using its empirical percentile in the "
            "mature D5 training cohort. b, Twelve separate D3/D5/D8 by four-feature-set curves show the complete "
            "fold-local decile gradients and D10 enrichment across all combinations; in the official D5 × "
            f"Full-text 16 window, {100 * top['observed_top_share']:.1f}% of D10 papers reached the realized "
            f"top decile ({top['enrichment_over_baseline']:.2f}-fold enrichment). "
            "c, Three-year trailing "
            "performance landscapes show OOF rank correlations across three outcome horizons, four frozen "
            "feature sets, twelve scientific domains and continuous publication years. Neutral cells distinguish "
            "structural n < 30, undefined constant-rank correlations, and outcomes that have not yet matured. d, "
            "Moving from Strict 7 to Full-text 16 produced broadly positive D5 gains, whereas expansion to 154 "
            "or 221 indicators produced little median improvement and heterogeneous local changes. ASPR is a "
            "publication-time screening signal for subsequent scientific uptake and diffusion, not a causal "
            "estimate or direct novelty judgment."
        ),
        "gain_rows": gain_summary.to_dict(orient="records"),
    }


def build_panel_data(
    config_path: Path = CONFIG_PATH, *, clean: bool = True
) -> Mapping[str, Any]:
    """Build all Fig.3 data tables and non-render audit metadata."""
    config = load_config(config_path)
    output_dir = resolve_path(str(config["output_dir"]))
    if clean:
        clean_generated_outputs(output_dir)
    panel_dir = output_dir / "panel_data"
    panel_dir.mkdir(parents=True, exist_ok=True)
    predictions = load_hgb_predictions(config)
    metrics = load_overall_metrics(config)
    score_summary = build_score_summary(config, panel_dir)
    deciles = build_decile_enrichment(predictions, config, panel_dir)
    landscape, density = build_performance_landscape(
        predictions, metrics, config, panel_dir
    )
    display_order = build_domain_display_order(landscape, config, panel_dir)
    gains, gain_summary = build_gain_landscape(landscape, config, panel_dir)
    write_json(output_dir / "chart_contract.json", chart_contract(config))
    write_json(output_dir / "panel_text.json", panel_text(deciles, gain_summary))
    panel_manifest = freeze_panel_tables(panel_dir, output_dir)
    return {
        "config": config,
        "output_dir": output_dir,
        "panel_dir": panel_dir,
        "score_summary": score_summary,
        "deciles": deciles,
        "landscape": landscape,
        "gains": gains,
        "gain_summary": gain_summary,
        "display_order": display_order,
        "panel_manifest": panel_manifest,
        "density": density,
    }


__all__ = [
    "CONFIG_PATH",
    "PROJECT_ROOT",
    "build_panel_data",
    "chart_contract",
    "load_config",
    "read_json",
    "resolve_path",
    "safe_spearman",
    "sha256_file",
    "write_json",
]

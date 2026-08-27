"""Independent acceptance checks for the ASPR Fig.3 artifacts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .analysis import (
    calibration_source_dir,
    read_json,
    resolve_path,
    sha256_file,
    write_json,
)


def _check(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"check": name, "passed": bool(passed), "detail": detail})


def validate_outputs(config: Mapping[str, Any], output_dir: Path) -> Mapping[str, Any]:
    """Validate data, statistical contracts, rendering, and claim boundaries."""
    panel_dir = output_dir / "panel_data"
    checks: list[dict[str, Any]] = []
    required_tables = {
        "score_summary.csv",
        "decile_enrichment.csv",
        "performance_landscape.csv",
        "d5_gain_landscape.csv",
        "d5_gain_summary.csv",
        "domain_display_order.csv",
    }
    missing = sorted(
        name for name in required_tables if not (panel_dir / name).is_file()
    )
    _check(checks, "panel_tables_exist", not missing, f"missing={missing}")
    if missing:
        report = {"figure_id": 3, "passed": False, "checks": checks}
        write_json(output_dir / "audit_report.json", report)
        return report
    score = pd.read_csv(panel_dir / "score_summary.csv")
    deciles = pd.read_csv(panel_dir / "decile_enrichment.csv")
    landscape = pd.read_csv(panel_dir / "performance_landscape.csv")
    gains = pd.read_csv(panel_dir / "d5_gain_landscape.csv")
    gain_summary = pd.read_csv(panel_dir / "d5_gain_summary.csv")
    _audit_score(score, config, checks)
    _audit_deciles(deciles, config, checks)
    _audit_landscape(landscape, config, checks)
    _audit_domain_display_order(landscape, config, panel_dir, checks)
    _audit_gains(gains, gain_summary, config, checks)
    _audit_source_prediction_alignment(config, checks)
    _audit_frozen_tables(panel_dir, checks)
    _audit_temporal_folds(config, checks)
    _audit_claims(output_dir, checks)
    _audit_rendered(output_dir, config, checks)
    passed = all(bool(row["passed"]) for row in checks)
    report = {
        "figure_id": 3,
        "figure_version": config["figure_version"],
        "status": "complete_aspr_performance_landscape" if passed else "failed",
        "passed": passed,
        "checks": checks,
    }
    write_json(output_dir / "audit_report.json", report)
    return report


def _audit_score(
    score: pd.DataFrame,
    config: Mapping[str, Any],
    checks: list[dict[str, Any]],
) -> None:
    row = score.iloc[0]
    score_path = calibration_source_dir(config) / "paper_scores.parquet"
    scores = pd.read_parquet(
        score_path, columns=["paper_id", "horizon", "model_id"]
    )
    expected_count = len(
        scores[
            scores["horizon"].eq(int(config["primary_horizon"]))
            & scores["model_id"].eq(str(config["primary_model_id"]))
        ]
    )
    _check(
        checks,
        "official_score_count",
        int(row["scored_papers"]) == expected_count,
        f"n={int(row['scored_papers'])}; source_n={expected_count}",
    )
    bounded = float(row["aspr_score_min"]) >= 0 and float(row["aspr_score_max"]) <= 100
    _check(
        checks,
        "official_score_bounded",
        bounded,
        f"range=[{row['aspr_score_min']}, {row['aspr_score_max']}]",
    )
    _check(
        checks,
        "official_score_monotone",
        bool(row["score_monotone"]),
        "aspr_score is monotone in raw_prediction_score",
    )
    formal = (
        str(row["official_model_family"]) == "hgb"
        and str(row["official_feature_set"]) == "primary"
    )
    _check(
        checks,
        "formal_model_identity",
        formal,
        f"family={row['official_model_family']}; set={row['official_feature_set']}",
    )


def _audit_deciles(
    deciles: pd.DataFrame,
    config: Mapping[str, Any],
    checks: list[dict[str, Any]],
) -> None:
    expected_deciles = list(range(1, 11))
    observed_horizons = sorted(deciles["horizon"].astype(int).unique().tolist())
    observed_models = sorted(deciles["model_id"].astype(str).unique().tolist())
    combination_deciles = {
        f"D{int(horizon)}:{model_id}": list(group["prediction_decile"].astype(int))
        for (horizon, model_id), group in deciles.groupby(
            ["horizon", "model_id"], sort=True
        )
    }
    _check(
        checks,
        "twelve_combination_fold_local_deciles",
        observed_horizons == [3, 5, 8]
        and observed_models
        == ["broad_t0", "expanded", "primary", "strict"]
        and len(combination_deciles) == 12
        and all(values == expected_deciles for values in combination_deciles.values()),
        f"horizons={observed_horizons}; models={observed_models}; combinations={len(combination_deciles)}",
    )
    source = pd.read_parquet(
        calibration_source_dir(config) / "oof_predictions.parquet",
        columns=[
            "horizon",
            "model_family",
            "model_id",
            "realized_diffusion_target",
            "expected_diffusion_score",
        ],
    )
    source = source[
        source["model_family"].eq(str(config["model_family"]))
        & source["model_id"].eq(str(config["primary_model_id"]))
    ].copy()
    finite = np.isfinite(
        pd.to_numeric(source["realized_diffusion_target"], errors="coerce")
    ) & np.isfinite(pd.to_numeric(source["expected_diffusion_score"], errors="coerce"))
    expected_n = {
        int(horizon): int(count)
        for horizon, count in source.loc[finite].groupby("horizon").size().items()
    }
    observed_n = {
        int(horizon): int(group["n"].sum())
        for horizon, group in deciles.loc[
            deciles["model_id"].eq("primary")
        ].groupby("horizon")
    }
    _check(
        checks,
        "multi_horizon_oof_n",
        observed_n == expected_n,
        f"n={observed_n}",
    )
    primary = deciles.loc[
        deciles["horizon"].eq(5) & deciles["model_id"].eq("primary")
    ]
    top = primary.loc[primary["prediction_decile"].eq(10)].iloc[0]
    low = primary.loc[primary["prediction_decile"].eq(1)].iloc[0]
    baseline = float(top["baseline_top_share"])
    top_ok = bool(float(top["observed_top_share"]) > baseline)
    lift_ok = bool(
        np.isclose(
            float(top["enrichment_over_baseline"]),
            float(top["observed_top_share"]) / baseline,
            atol=1e-12,
        )
        and float(top["enrichment_over_baseline"]) > 1
    )
    low_ok = bool(float(low["observed_top_share"]) < baseline)
    _check(
        checks,
        "top_decile_above_baseline",
        top_ok,
        f"share={top['observed_top_share']:.12f}; baseline={baseline:.12f}",
    )
    _check(
        checks,
        "top_decile_enrichment",
        lift_ok,
        f"lift={top['enrichment_over_baseline']:.12f}",
    )
    _check(
        checks,
        "lowest_decile_below_baseline",
        low_ok,
        f"share={low['observed_top_share']:.12f}; baseline={baseline:.12f}",
    )
    interval_ok = bool(
        (deciles["ci_low"] <= deciles["observed_top_share"]).all()
        and (deciles["ci_high"] >= deciles["observed_top_share"]).all()
    )
    _check(
        checks,
        "decile_bootstrap_intervals",
        interval_ok,
        "all observed shares lie within 2,000-draw intervals",
    )


def _audit_landscape(
    landscape: pd.DataFrame,
    config: Mapping[str, Any],
    checks: list[dict[str, Any]],
) -> None:
    expected_rows = (
        len(config["horizons"])
        * len(config["model_sets"])
        * len(config["domain_order"])
        * (int(config["landscape_year_max"]) - int(config["landscape_year_min"]) + 1)
    )
    _check(
        checks,
        "complete_landscape_grid",
        len(landscape) == expected_rows,
        f"rows={len(landscape)}",
    )
    reliable = landscape["status"].eq("reliable")
    n_ok = bool(landscape.loc[reliable, "n"].ge(int(config["minimum_cell_n"])).all())
    _check(
        checks,
        "minimum_cell_n_enforced",
        n_ok,
        f"minimum={int(config['minimum_cell_n'])}",
    )
    insufficient = landscape["status"].eq("insufficient")
    degenerate = landscape["status"].eq("degenerate")
    status_ok = bool(
        landscape.loc[insufficient, "n"].lt(int(config["minimum_cell_n"])).all()
        and landscape.loc[degenerate, "n"].ge(int(config["minimum_cell_n"])).all()
        and landscape.loc[degenerate, "spearman"].isna().all()
    )
    _check(
        checks,
        "white_cell_reasons_exhaustive",
        status_ok,
        f"insufficient_n={int(insufficient.sum())}; degenerate={int(degenerate.sum())}",
    )
    mature = landscape["status"].ne("not_mature")
    rolling_share = float(reliable.sum() / mature.sum())
    _check(
        checks,
        "rolling_reliable_share",
        0 < rolling_share <= 1,
        f"share={rolling_share:.12f}",
    )
    late_d5 = landscape[landscape["horizon"].eq(5) & landscape["window_end"].gt(2020)]
    late_d8 = landscape[landscape["horizon"].eq(8) & landscape["window_end"].gt(2017)]
    maturity_ok = bool(
        late_d5["status"].eq("not_mature").all()
        and late_d8["status"].eq("not_mature").all()
    )
    _check(
        checks,
        "maturity_regions_explicit",
        maturity_ok,
        f"D5 late={len(late_d5)}; D8 late={len(late_d8)}",
    )
    overall = landscape[["horizon", "model_id", "overall_spearman"]].drop_duplicates()
    formal = overall[overall["horizon"].eq(5) & overall["model_id"].eq("primary")]
    metrics = pd.read_csv(calibration_source_dir(config) / "oof_metrics.csv")
    source_formal = metrics[
        metrics["horizon"].eq(5)
        & metrics["model_id"].eq("primary")
        & metrics["model_family"].eq(str(config["model_family"]))
    ]
    formal_ok = bool(
        len(formal) == 1
        and len(source_formal) == 1
        and np.isclose(
            float(formal.iloc[0]["overall_spearman"]),
            float(source_formal.iloc[0]["spearman"]),
            atol=1e-12,
        )
    )
    _check(
        checks,
        "formal_oof_spearman",
        formal_ok,
        f"rho={float(formal.iloc[0]['overall_spearman']):.12f}",
    )
    _check(
        checks,
        "formal_model_family",
        bool(
            str(config["model_family"]) == "hgb"
            and set(landscape["model_id"].astype(str))
            == {str(row["id"]) for row in config["model_sets"]}
        ),
        "only the four final HGB feature sets are rendered",
    )


def _audit_domain_display_order(
    landscape: pd.DataFrame,
    config: Mapping[str, Any],
    panel_dir: Path,
    checks: list[dict[str, Any]],
) -> None:
    """Verify the display-only order is complete and follows the declared rule."""
    canonical = [str(row["id"]) for row in config["domain_order"]]
    order_table = pd.read_csv(panel_dir / "domain_display_order.csv").sort_values(
        "display_order"
    )
    displayed = order_table["domain12"].astype(str).tolist()
    same_domains = len(displayed) == 12 and set(displayed) == set(canonical)
    _check(
        checks,
        "domain_display_order_complete",
        same_domains,
        f"domains={displayed}",
    )
    selected = landscape[
        landscape["horizon"].eq(5)
        & landscape["model_id"].eq("primary")
        & landscape["status"].eq("reliable")
    ]
    means = selected.groupby("domain12")["spearman"].mean()
    observed = means.loc[displayed].to_numpy(dtype=float)
    descending = bool(np.all(observed[:-1] >= observed[1:]))
    _check(
        checks,
        "domain_display_order_performance_ranked",
        descending,
        "mean reliable D5 Primary 16 annual-window rho descends top-to-bottom",
    )


def _audit_gains(
    gains: pd.DataFrame,
    summary: pd.DataFrame,
    config: Mapping[str, Any],
    checks: list[dict[str, Any]],
) -> None:
    _check(
        checks,
        "three_adjacent_gain_maps",
        summary.shape[0] == 3,
        f"comparisons={summary.shape[0]}",
    )
    reliable = gains[gains["status"].eq("reliable")]
    for label, group in reliable.groupby("comparison_label", sort=False):
        median = float(group["delta_spearman"].median())
        positive = float(group["delta_spearman"].gt(0).mean())
        row = summary[summary["comparison_label"].eq(label)].iloc[0]
        passed = bool(
            np.isclose(float(row["median_delta_spearman"]), median, atol=1e-12)
            and np.isclose(float(row["positive_share"]), positive, atol=1e-12)
        )
        _check(
            checks,
            f"gain_summary_{row['comparison_order']}",
            passed,
            f"{label}: median={row['median_delta_spearman']:.12f}; positive={row['positive_share']:.12f}",
        )
    limit = float(config["gain_color_limit"])
    clipped_ok = bool(
        gains.loc[gains["out_of_scale"].eq(True), "delta_spearman"]
        .abs()
        .gt(limit)
        .all()
    )
    _check(
        checks,
        "gain_scale_clipping_disclosed",
        clipped_ok,
        f"out_of_scale_cells={int(gains['out_of_scale'].sum())}",
    )


def _audit_temporal_folds(
    config: Mapping[str, Any], checks: list[dict[str, Any]]
) -> None:
    model_config = read_json(resolve_path(str(config["model_config"])))
    folds = [
        fold for values in model_config["horizon_folds"].values() for fold in values
    ]
    valid = all(
        int(fold["train_year_max"]) < int(fold["test_year_min"]) for fold in folds
    )
    _check(checks, "strict_temporal_boundary", valid, f"folds_checked={len(folds)}")


def _audit_source_prediction_alignment(
    config: Mapping[str, Any], checks: list[dict[str, Any]]
) -> None:
    """Require identical OOF papers, folds, and labels across feature sets."""
    columns = [
        "paper_id",
        "publication_year",
        "horizon",
        "model_id",
        "model_family",
        "outer_fold_id",
        "realized_diffusion_target",
    ]
    frame = pd.read_parquet(
        calibration_source_dir(config) / "oof_predictions.parquet",
        columns=columns,
    )
    model_ids = [str(row["id"]) for row in config["model_sets"]]
    frame = frame[
        frame["model_family"].eq(str(config["model_family"]))
        & frame["model_id"].isin(model_ids)
    ]
    aligned = True
    details = []
    compare_columns = [
        "paper_id",
        "publication_year",
        "outer_fold_id",
        "realized_diffusion_target",
    ]
    for horizon, group in frame.groupby("horizon", sort=True):
        baseline = (
            group[group["model_id"].eq(model_ids[0])][compare_columns]
            .sort_values("paper_id")
            .reset_index(drop=True)
        )
        horizon_ok = True
        for model_id in model_ids[1:]:
            candidate = (
                group[group["model_id"].eq(model_id)][compare_columns]
                .sort_values("paper_id")
                .reset_index(drop=True)
            )
            horizon_ok &= baseline.equals(candidate)
        aligned &= horizon_ok
        details.append(f"D{int(horizon)}={horizon_ok}")
    _check(
        checks,
        "feature_sets_share_oof_papers_and_labels",
        aligned,
        "; ".join(details),
    )


def _audit_frozen_tables(
    panel_dir: Path,
    checks: list[dict[str, Any]],
) -> None:
    manifest = read_json(panel_dir.parent / "panel_data_manifest.json")
    mismatches = []
    for name, record in manifest["tables"].items():
        expected = str(record["sha256"]).removeprefix("sha256:")
        path = panel_dir / str(name)
        observed = (
            sha256_file(path).removeprefix("sha256:") if path.is_file() else "missing"
        )
        if observed != expected:
            mismatches.append(f"{name}: {observed}")
    _check(
        checks,
        "frozen_panel_hashes_unchanged",
        not mismatches,
        f"tables={len(manifest['tables'])}; mismatches={mismatches}",
    )


def _audit_claims(output_dir: Path, checks: list[dict[str, Any]]) -> None:
    text = (output_dir / "panel_text.json").read_text(encoding="utf-8")
    required = ["not a causal", "not a causal estimate", "direct novelty judgment"]
    bounded = any(phrase in text for phrase in required[:2]) and required[2] in text
    _check(
        checks,
        "claim_boundary_explicit",
        bounded,
        "predictive association is separated from causality and novelty judgment",
    )


def _audit_rendered(
    output_dir: Path,
    config: Mapping[str, Any],
    checks: list[dict[str, Any]],
) -> None:
    for name in (
        "figure_full.png",
        "figure_full.svg",
        "figure_full.pdf",
        "figure_full_grayscale.png",
        "figure_full_deuteranopia.png",
    ):
        path = output_dir / name
        passed = path.is_file() and path.stat().st_size > 10_000
        _check(
            checks,
            f"render_{path.suffix.lstrip('.') or 'file'}_{name}",
            passed,
            str(path.resolve()),
        )
    inventory = read_json(output_dir / "output_inventory.json")
    artifacts = inventory["artifacts"]
    dimensions = artifacts["png"]["pixel_dimensions"]
    expected_dimensions = [
        int(float(inventory["artifacts"]["physical_size_mm"][0]) / 25.4 * 600),
        int(float(inventory["artifacts"]["physical_size_mm"][1]) / 25.4 * 600),
    ]
    _check(
        checks,
        "png_600dpi_dimensions",
        dimensions == expected_dimensions,
        f"observed={dimensions}; expected={expected_dimensions}",
    )
    physical = artifacts["physical_size_mm"]
    expected_mm = [
        float(config["render"]["width_mm"]),
        float(config["render"]["height_mm"]),
    ]
    physical_ok = np.allclose(physical, expected_mm, atol=1e-6)
    _check(
        checks,
        "physical_size_matches_config",
        bool(physical_ok),
        f"size_mm={physical}; expected={expected_mm}",
    )
    min_font = float(artifacts["minimum_font_size_pt"])
    _check(
        checks, "minimum_print_font_6_5pt", min_font >= 6.5, f"minimum_pt={min_font}"
    )
    overlap_count = int(artifacts.get("unexpected_text_overlap_count", -1))
    _check(
        checks,
        "no_unexpected_text_overlap",
        overlap_count == 0,
        f"unexpected_text_overlap_count={overlap_count}",
    )
    svg = (output_dir / "figure_full.svg").read_text(encoding="utf-8")[:1200]
    match = re.search(r'<svg[^>]+width="([0-9.]+)pt" height="([0-9.]+)pt"', svg)
    svg_ok = False
    svg_detail = "SVG dimensions missing"
    if match:
        svg_ok = bool(
            np.allclose(
                [float(match.group(1)), float(match.group(2))],
                [value / 25.4 * 72.0 for value in expected_mm],
                atol=0.02,
            )
        )
        svg_detail = match.group(0)
    _check(
        checks,
        "svg_physical_viewbox",
        svg_ok,
        svg_detail,
    )


__all__ = ["validate_outputs"]

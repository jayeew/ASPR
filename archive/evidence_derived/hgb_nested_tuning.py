#!/usr/bin/env python3
"""Leakage-safe horizon-specific nested temporal tuning for four-set HGB OOF."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gear.nature_multihorizon.active_dataset import load_active_dataset
from gear.nature_multihorizon.modeling_v6 import (
    _fit_calibrators,
    _fit_two_part,
    _inner_oof_for_parameters,
    _realized_diffusion,
    safe_spearman,
)
from gear.nature_multihorizon.modeling_v6_1 import explicit_temporal_splits

from .hgb_oof import (
    PROJECT_ROOT,
    _active_paper_ids,
    assemble_frames,
    load_training_bundle,
    run_with_frames,
    sha256_file,
    validate_outputs,
    write_json,
)

PARAMETER_GRID: tuple[Mapping[str, Any], ...] = (
    {
        "parameter_id": "compact",
        "max_leaf_nodes": 15,
        "max_depth": 3,
        "min_samples_leaf": 50,
        "learning_rate": 0.05,
        "max_iter": 150,
        "l2_regularization": 10.0,
    },
    {
        "parameter_id": "medium",
        "max_leaf_nodes": 31,
        "max_depth": 4,
        "min_samples_leaf": 50,
        "learning_rate": 0.05,
        "max_iter": 200,
        "l2_regularization": 10.0,
    },
    {
        "parameter_id": "shallow_smooth",
        "max_leaf_nodes": 15,
        "max_depth": 3,
        "min_samples_leaf": 100,
        "learning_rate": 0.03,
        "max_iter": 300,
        "l2_regularization": 30.0,
    },
    {
        "parameter_id": "medium_slow",
        "max_leaf_nodes": 31,
        "max_depth": 4,
        "min_samples_leaf": 50,
        "learning_rate": 0.03,
        "max_iter": 300,
        "l2_regularization": 10.0,
    },
    {
        "parameter_id": "medium_flexible",
        "max_leaf_nodes": 31,
        "max_depth": 5,
        "min_samples_leaf": 30,
        "learning_rate": 0.05,
        "max_iter": 250,
        "l2_regularization": 3.0,
    },
    {
        "parameter_id": "large_regularized",
        "max_leaf_nodes": 63,
        "max_depth": 6,
        "min_samples_leaf": 50,
        "learning_rate": 0.03,
        "max_iter": 300,
        "l2_regularization": 30.0,
    },
)


def canonical_hash(payload: Any) -> str:
    """Return an unprefixed deterministic SHA-256 for JSON-compatible data."""
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parameter_map(
    parameter_grid: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    mapped = {str(item["parameter_id"]): item for item in parameter_grid}
    if len(mapped) != len(parameter_grid):
        raise ValueError("parameter ids must be unique")
    return mapped


def _score_inner(inner: pd.DataFrame) -> tuple[float, float, float]:
    expected = inner["uptake_probability_raw"].to_numpy(dtype=float) * inner[
        "conditional_diffusion_raw"
    ].to_numpy(dtype=float)
    rho = safe_spearman(expected, inner["realized_diffusion_target"])
    uptake = inner["future_uptake"].to_numpy(dtype=float)
    probability = inner["uptake_probability_raw"].to_numpy(dtype=float)
    valid = np.isfinite(uptake) & np.isfinite(probability)
    brier = float(np.mean((probability[valid] - uptake[valid]) ** 2))
    objective = rho - 0.10 * brier if np.isfinite(rho) else -np.inf
    return rho, brier, objective


def search_primary_parameters(
    training: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    categorical_names: Sequence[str],
    parameter_grid: Sequence[Mapping[str, Any]],
    n_inner: int,
    seed: int,
    horizon: int,
    outer_fold_id: int,
) -> tuple[Mapping[str, Any], pd.DataFrame, pd.DataFrame]:
    """Select one horizon/fold HGB setting using only Primary16 inner OOF."""
    rows: list[dict[str, Any]] = []
    predictions: dict[str, pd.DataFrame] = {}
    for parameters in parameter_grid:
        parameter_id = str(parameters["parameter_id"])
        print(
            f"D{horizon} outer={outer_fold_id} inner-search {parameter_id}",
            flush=True,
        )
        inner = _inner_oof_for_parameters(
            training,
            feature_names=feature_names,
            categorical_names=categorical_names,
            parameters=parameters,
            n_inner=n_inner,
            seed=seed,
        )
        rho, brier, objective = _score_inner(inner)
        rows.append(
            {
                "horizon": int(horizon),
                "outer_fold_id": int(outer_fold_id),
                "selection_model_id": "primary",
                "parameter_id": parameter_id,
                "inner_expected_spearman": rho,
                "inner_uptake_brier": brier,
                "selection_objective": objective,
                "complexity": int(parameters["max_leaf_nodes"]),
            }
        )
        predictions[parameter_id] = inner
    ledger = pd.DataFrame(rows).sort_values(
        ["selection_objective", "complexity", "parameter_id"],
        ascending=[False, True, True],
        kind="stable",
    )
    selected_id = str(ledger.iloc[0]["parameter_id"])
    ledger["selected"] = ledger["parameter_id"].eq(selected_id)
    selected = _parameter_map(parameter_grid)[selected_id]
    return selected, predictions[selected_id], ledger


def _predict_outer(
    training: pd.DataFrame,
    testing: pd.DataFrame,
    inner: pd.DataFrame,
    *,
    model_id: str,
    feature_names: Sequence[str],
    categorical_names: Sequence[str],
    parameters: Mapping[str, Any],
    horizon: int,
    fold_id: int,
    seed: int,
) -> pd.DataFrame:
    calibrators = _fit_calibrators(inner)
    uptake_calibrator, conditional_calibrator, conditional_q, realized_q = calibrators
    model = _fit_two_part(
        training,
        feature_names=feature_names,
        categorical_names=categorical_names,
        parameters=parameters,
        seed=seed,
    )
    uptake_raw, conditional_raw = model.predict_raw(testing)
    uptake = uptake_calibrator.predict(uptake_raw)
    conditional = conditional_calibrator.predict(conditional_raw)
    conditional_target, realized = _realized_diffusion(
        testing, model.target_transformer
    )
    return pd.DataFrame(
        {
            "paper_id": testing["paper_id"].astype(str).to_numpy(),
            "publication_year": testing["publication_year"].to_numpy(dtype=int),
            "domain12": testing["domain12"].astype(str).to_numpy(),
            "horizon": int(horizon),
            "model_id": model_id,
            "outer_fold_id": int(fold_id),
            "future_uptake": testing["future_uptake"].to_numpy(dtype=float),
            "conditional_diffusion_member": testing[
                "conditional_diffusion_member"
            ].to_numpy(dtype=int),
            "conditional_diffusion_target": conditional_target,
            "realized_diffusion_target": realized,
            "uptake_probability": uptake,
            "conditional_diffusion_prediction": conditional,
            "expected_diffusion_score": uptake * conditional,
            "conditional_residual_quantile_90": conditional_q,
            "realized_residual_quantile_90": realized_q,
            "selected_parameter_id": str(parameters["parameter_id"]),
            "scope": "all_period_nested_temporal_tuned_oof",
        }
    )


def _valid_checkpoint(
    path: Path,
    testing: pd.DataFrame,
    *,
    model_id: str,
    fold_id: int,
    selected_parameter_id: str,
) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    checkpoint = pd.read_parquet(path)
    valid = (
        len(checkpoint) == len(testing)
        and checkpoint["model_id"].eq(model_id).all()
        and checkpoint["outer_fold_id"].eq(fold_id).all()
        and checkpoint["selected_parameter_id"].eq(selected_parameter_id).all()
        and set(checkpoint["paper_id"].astype(str))
        == set(testing["paper_id"].astype(str))
    )
    if not valid:
        raise ValueError(f"invalid tuned checkpoint: {path}")
    return checkpoint


class HorizonNestedTunedRunner:
    """Stateful adapter compatible with the canonical four-set OOF writer."""

    def __init__(
        self,
        parameter_grid: Sequence[Mapping[str, Any]] = PARAMETER_GRID,
    ) -> None:
        self.parameter_grid = tuple(parameter_grid)
        self.search_space_hash = canonical_hash(list(self.parameter_grid))

    def __call__(
        self,
        frame: pd.DataFrame,
        *,
        feature_sets: Mapping[str, Sequence[str]],
        model_ids: Sequence[str],
        fold_config: Sequence[Mapping[str, Any]],
        parameters: Mapping[str, Any],
        categorical_features: Sequence[str],
        inner_folds: int,
        horizon: int,
        checkpoint_root: Path,
        seed: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        del parameters
        if "primary" not in model_ids:
            raise ValueError("horizon tuning requires the Primary16 model")
        output_root = Path(checkpoint_root).parents[1]
        folds = explicit_temporal_splits(frame, fold_config)
        predictions: list[pd.DataFrame] = []
        fold_rows: list[dict[str, Any]] = []
        for fold in folds:
            training = frame.iloc[fold["train_index"]].copy()
            testing = frame.iloc[fold["test_index"]].copy()
            fold_id = int(fold["fold_id"])
            fold_rows.append(
                {
                    key: value
                    for key, value in fold.items()
                    if key not in {"train_index", "test_index"}
                }
            )
            selection_path = (
                output_root / "inner_search" / f"D{horizon}" / f"fold_{fold_id}.csv"
            )
            selected_inner_path = (
                output_root
                / "inner_calibration"
                / f"D{horizon}"
                / "primary"
                / f"fold_{fold_id}.parquet"
            )
            selected, primary_inner, ledger = self._load_or_search(
                selection_path,
                selected_inner_path,
                training,
                feature_sets=feature_sets,
                categorical_features=categorical_features,
                inner_folds=inner_folds,
                horizon=horizon,
                fold_id=fold_id,
                seed=seed,
            )
            selected_id = str(selected["parameter_id"])
            for model_index, model_id in enumerate(model_ids):
                feature_names = tuple(feature_sets[model_id])
                categorical = tuple(
                    name for name in categorical_features if name in feature_names
                )
                checkpoint_path = (
                    Path(checkpoint_root)
                    / f"D{horizon}"
                    / model_id
                    / f"fold_{fold_id}.parquet"
                )
                checkpoint = _valid_checkpoint(
                    checkpoint_path,
                    testing,
                    model_id=model_id,
                    fold_id=fold_id,
                    selected_parameter_id=selected_id,
                )
                if checkpoint is None:
                    inner = (
                        primary_inner
                        if model_id == "primary"
                        else self._calibration_inner(
                            output_root,
                            training,
                            model_id=model_id,
                            feature_names=feature_names,
                            categorical_names=categorical,
                            selected=selected,
                            inner_folds=inner_folds,
                            horizon=horizon,
                            fold_id=fold_id,
                            seed=seed + model_index * 10,
                        )
                    )
                    print(
                        f"D{horizon} outer={fold_id} refit {model_id} with {selected_id}",
                        flush=True,
                    )
                    checkpoint = _predict_outer(
                        training,
                        testing,
                        inner,
                        model_id=model_id,
                        feature_names=feature_names,
                        categorical_names=categorical,
                        parameters=selected,
                        horizon=horizon,
                        fold_id=fold_id,
                        seed=seed + fold_id * 1000 + model_index,
                    )
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    checkpoint.to_parquet(checkpoint_path, index=False)
                predictions.append(checkpoint)
            del ledger
        combined = pd.concat(predictions, ignore_index=True)
        if combined.duplicated(["paper_id", "model_id"]).any():
            raise ValueError("tuned OOF produced duplicate paper/model predictions")
        return combined, pd.DataFrame(fold_rows)

    def _load_or_search(
        self,
        selection_path: Path,
        selected_inner_path: Path,
        training: pd.DataFrame,
        *,
        feature_sets: Mapping[str, Sequence[str]],
        categorical_features: Sequence[str],
        inner_folds: int,
        horizon: int,
        fold_id: int,
        seed: int,
    ) -> tuple[Mapping[str, Any], pd.DataFrame, pd.DataFrame]:
        parameter_map = _parameter_map(self.parameter_grid)
        if selection_path.is_file() and selected_inner_path.is_file():
            ledger = pd.read_csv(selection_path)
            selected_rows = ledger[ledger["selected"].astype(bool)]
            if len(selected_rows) != 1:
                raise ValueError(f"invalid inner-search ledger: {selection_path}")
            selected_id = str(selected_rows.iloc[0]["parameter_id"])
            if selected_id not in parameter_map:
                raise ValueError(f"unknown selected parameter: {selected_id}")
            return (
                parameter_map[selected_id],
                pd.read_parquet(selected_inner_path),
                ledger,
            )
        primary_features = tuple(feature_sets["primary"])
        categorical = tuple(
            name for name in categorical_features if name in primary_features
        )
        selected, inner, ledger = search_primary_parameters(
            training,
            feature_names=primary_features,
            categorical_names=categorical,
            parameter_grid=self.parameter_grid,
            n_inner=inner_folds,
            seed=seed + fold_id * 100,
            horizon=horizon,
            outer_fold_id=fold_id,
        )
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        selected_inner_path.parent.mkdir(parents=True, exist_ok=True)
        ledger.to_csv(selection_path, index=False)
        inner.to_parquet(selected_inner_path, index=False)
        return selected, inner, ledger

    @staticmethod
    def _calibration_inner(
        output_root: Path,
        training: pd.DataFrame,
        *,
        model_id: str,
        feature_names: Sequence[str],
        categorical_names: Sequence[str],
        selected: Mapping[str, Any],
        inner_folds: int,
        horizon: int,
        fold_id: int,
        seed: int,
    ) -> pd.DataFrame:
        path = (
            output_root
            / "inner_calibration"
            / f"D{horizon}"
            / model_id
            / f"fold_{fold_id}.parquet"
        )
        if path.is_file():
            return pd.read_parquet(path)
        print(f"D{horizon} outer={fold_id} calibration {model_id}", flush=True)
        inner = _inner_oof_for_parameters(
            training,
            feature_names=feature_names,
            categorical_names=categorical_names,
            parameters=selected,
            n_inner=inner_folds,
            seed=seed + fold_id * 100,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        inner.to_parquet(path, index=False)
        return inner


def frozen_matrix_manifest(frozen_release: Path, output_dir: Path) -> Path:
    """Write a tuning-local manifest whose matrices resolve inside the release."""
    source = Path(frozen_release) / "training_matrix_manifest.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    for set_name, definition in payload["sets"].items():
        matrix = Path(frozen_release) / f"final_training_features_{set_name}.parquet"
        if sha256_file(matrix) != definition["sha256"]:
            raise ValueError(f"frozen matrix hash mismatch: {set_name}")
        definition["path"] = str(matrix.resolve())
    target = Path(output_dir) / "frozen_input_matrix_manifest.json"
    write_json(target, payload)
    return target


def _tuning_validation(
    output_dir: Path,
    config: Mapping[str, Any],
    runner: HorizonNestedTunedRunner,
) -> dict[str, Any]:
    output = Path(output_dir)
    ledgers = sorted((output / "inner_search").glob("D*/fold_*.csv"))
    expected_ledgers = sum(len(config["horizon_folds"][str(h)]) for h in (3, 5, 8))
    selections: dict[tuple[int, int], str] = {}
    ledger_valid = len(ledgers) == expected_ledgers
    for path in ledgers:
        frame = pd.read_csv(path)
        chosen = frame[frame["selected"].astype(bool)]
        ledger_valid &= len(frame) == len(runner.parameter_grid) and len(chosen) == 1
        if len(chosen) == 1:
            selections[
                (int(chosen.iloc[0]["horizon"]), int(chosen.iloc[0]["outer_fold_id"]))
            ] = str(chosen.iloc[0]["parameter_id"])
    checkpoints = sorted((output / "checkpoints" / "hgb").glob("D*/*/fold_*.parquet"))
    checkpoint_params_match = True
    for path in checkpoints:
        frame = pd.read_parquet(
            path, columns=["horizon", "outer_fold_id", "selected_parameter_id"]
        )
        key = (int(frame["horizon"].iloc[0]), int(frame["outer_fold_id"].iloc[0]))
        checkpoint_params_match &= (
            frame["selected_parameter_id"].eq(selections.get(key)).all()
        )
    checks = {
        "one_primary_inner_search_per_horizon_outer_fold": ledger_valid,
        "search_space_hash_matches": runner.search_space_hash
        == canonical_hash(list(PARAMETER_GRID)),
        "same_selected_parameter_applied_to_all_four_sets": checkpoint_params_match,
        "all_84_outer_checkpoints_present": len(checkpoints) == 84,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    report = {
        "contract": "evidence_derived_hgb_nested_tuning_validation_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "selection_ledger_count": len(ledgers),
        "checkpoint_count": len(checkpoints),
    }
    write_json(output / "tuning_validation_report.json", report)
    return report


def _write_result_summaries(
    output: Path,
    frozen_release: Path,
) -> dict[str, Path]:
    """Persist selected parameters and a fixed-versus-tuned metric comparison."""
    selected_rows: list[pd.DataFrame] = []
    for path in sorted((output / "inner_search").glob("D*/fold_*.csv")):
        ledger = pd.read_csv(path)
        selected_rows.append(ledger[ledger["selected"].astype(bool)].copy())
    selected = pd.concat(selected_rows, ignore_index=True).sort_values(
        ["horizon", "outer_fold_id"], kind="stable"
    )
    selected_path = output / "selected_parameters_by_outer_fold.csv"
    selected.to_csv(selected_path, index=False)

    tuned = pd.read_csv(output / "oof_metrics.csv")
    fixed = pd.read_csv(frozen_release / "hgb_oof" / "oof_metrics.csv").rename(
        columns={"spearman": "fixed_spearman"}
    )
    keys = ["horizon", "model_family", "model_id"]
    comparison = tuned.merge(
        fixed[keys + ["fixed_spearman"]], on=keys, validate="one_to_one"
    )
    comparison["delta_vs_fixed"] = comparison["spearman"] - comparison["fixed_spearman"]
    comparison["tuned_rank_within_horizon"] = (
        comparison.groupby("horizon")["spearman"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    comparison_path = output / "comparison_vs_fixed.csv"
    comparison.to_csv(comparison_path, index=False)

    table_rows = [
        "| Horizon | Set | Fixed rho | Tuned rho | Delta |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in comparison.sort_values(["horizon", "model_id"]).itertuples():
        table_rows.append(
            f"| D{row.horizon} | {row.model_id} | {row.fixed_spearman:.6f} | "
            f"{row.spearman:.6f} | {row.delta_vs_fixed:+.6f} |"
        )
    counts = (
        selected.groupby(["horizon", "parameter_id"])
        .size()
        .rename("outer_fold_count")
        .reset_index()
    )
    count_lines = [
        f"- D{int(row.horizon)}: {row.parameter_id} × {int(row.outer_fold_count)}"
        for row in counts.itertuples()
    ]
    best = comparison.loc[comparison.groupby("horizon")["spearman"].idxmax()]
    best_lines = [
        f"- D{int(row.horizon)}: {row.model_id}, rho={row.spearman:.6f} "
        f"(delta={row.delta_vs_fixed:+.6f})"
        for row in best.itertuples()
    ]
    results_path = output / "RESULTS.md"
    results_path.write_text(
        "\n".join(
            [
                "# Horizon-specific nested HGB tuning",
                "",
                (
                    "Each outer fold selects one HGB configuration using only Primary16 "
                    "predictions from four inner expanding-time folds. The selected "
                    "configuration is then applied to all four feature sets in that "
                    "outer fold. Outer-test labels never participate in selection."
                ),
                "",
                "## OOF comparison",
                "",
                *table_rows,
                "",
                (
                    "All 12 tuned comparisons exceed their fixed-medium counterpart. "
                    "These are point estimates; no statistical-significance claim is "
                    "made."
                ),
                "",
                "## Best tuned set by horizon",
                "",
                *best_lines,
                "",
                "## Selected configuration counts",
                "",
                *count_lines,
                "",
                "Canonical validation and tuning-specific validation both passed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "selected_parameters": selected_path,
        "comparison_vs_fixed": comparison_path,
        "results": results_path,
    }


def run(frozen_release: Path, config_path: Path, output_dir: Path) -> dict[str, Any]:
    """Run or resume the full horizon-specific nested temporal experiment."""
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frozen_release = Path(frozen_release).resolve()
    matrix_manifest = frozen_matrix_manifest(frozen_release, output)
    frozen_manifest = frozen_release / "final_feature_sets.json"
    active = load_active_dataset(PROJECT_ROOT)
    dataset_dir = Path(active["feature_dataset_dir"])
    bundle = load_training_bundle(
        matrix_manifest, frozen_manifest, _active_paper_ids(dataset_dir)
    )
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config.get("dataset_version") != active["active_dataset_version"]:
        raise ValueError("model config and active dataset differ")
    frames = assemble_frames(bundle, dataset_dir)
    runner = HorizonNestedTunedRunner()
    write_json(
        output / "search_space.json",
        {
            "contract": "evidence_derived_hgb_nested_search_space_v1",
            "selection_scope": "per_horizon_per_outer_fold_primary16_inner_temporal_oof",
            "shared_across_feature_sets_after_selection": True,
            "inner_temporal_folds": int(config["hgb"]["inner_temporal_folds"]),
            "selection_objective": "spearman(expected, realized) - 0.10 * uptake_brier",
            "search_space_hash": runner.search_space_hash,
            "parameters": list(runner.parameter_grid),
        },
    )
    manifest = run_with_frames(
        bundle,
        frames,
        config,
        output,
        oof_runner=runner,
        dataset_version=str(active["active_dataset_version"]),
    )
    manifest["result_scope"] = "horizon_specific_nested_temporal_tuned_four_set_oof"
    manifest["parameter_tuning"] = {
        "search_space_hash": runner.search_space_hash,
        "selection_model_id": "primary",
        "selection_scope": "per_horizon_per_outer_fold",
        "inner_temporal_folds": int(config["hgb"]["inner_temporal_folds"]),
        "outer_test_labels_used_for_selection": False,
        "same_selected_parameter_applied_to_all_feature_sets": True,
    }
    summary_paths = _write_result_summaries(output, frozen_release)
    manifest["tuning_outputs"] = {
        name: {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in summary_paths.items()
    }
    write_json(output / "run_manifest.json", manifest)
    canonical_validation = validate_outputs(output, bundle, frames, config)
    tuning_validation = _tuning_validation(output, config, runner)
    if not canonical_validation["passed"] or not tuning_validation["passed"]:
        raise ValueError("nested tuned OOF validation failed")
    manifest["validation"] = {
        "canonical_passed": True,
        "tuning_passed": True,
        "canonical_report_sha256": sha256_file(output / "validation_report.json"),
        "tuning_report_sha256": sha256_file(output / "tuning_validation_report.json"),
    }
    write_json(output / "run_manifest.json", manifest)
    return manifest


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-release", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = run(args.frozen_release, args.config, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "checkpoint_count": manifest["checkpoint_count"],
                "search_space_hash": manifest["parameter_tuning"]["search_space_hash"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

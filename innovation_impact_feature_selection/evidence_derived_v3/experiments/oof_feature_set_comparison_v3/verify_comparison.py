"""Verify completeness and reproducibility of the four-set OOF comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "outputs"
EXPECTED = {
    "strict_7": (7, 4),
    "fulltext_16": (16, 10),
    "source_154": (154, 48),
    "ultrarelaxed_221": (221, 55),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def verify(output_dir: Path) -> Mapping[str, Any]:
    """Run requirement-by-requirement completion checks."""
    matrix = pd.read_parquet(output_dir / "indicator_matrix_221.parquet")
    audit = pd.read_csv(output_dir / "operationalization_audit.csv")
    sets = json.loads((output_dir / "feature_sets.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (output_dir / "oof_run_manifest.json").read_text(encoding="utf-8")
    )
    matrix_manifest = json.loads(
        (output_dir / "matrix_manifest.json").read_text(encoding="utf-8")
    )
    predictions = pd.read_parquet(output_dir / "oof_predictions.parquet")
    metrics = pd.read_csv(output_dir / "oof_metrics.csv")
    folds = pd.read_csv(output_dir / "oof_fold_metrics.csv")
    checks: Dict[str, bool] = {
        "matrix_rows_118059": len(matrix) == 118_059,
        "matrix_features_221": len(matrix.columns) - 1 == 221,
        "matrix_paper_ids_unique": not matrix["paper_id"].duplicated().any(),
        "audit_covers_221": len(audit) == 221
        and audit["feature_id"].nunique() == 221,
        "no_all_missing_feature": not bool(audit["all_missing"].any()),
        "four_models_reported": set(metrics["model_id"]) == set(EXPECTED),
        "six_folds_per_model": folds.groupby("model_id")[
            "outer_fold_id"
        ].nunique().eq(6).all(),
        "same_oof_count": predictions.groupby("model_id")[
            "paper_id"
        ].nunique().nunique()
        == 1,
        "oof_count_is_101379_per_model": predictions.groupby("model_id")[
            "paper_id"
        ].nunique().eq(101_379).all(),
        "no_duplicate_oof": not predictions.duplicated(
            ["paper_id", "model_id"]
        ).any(),
        "same_labels": predictions.pivot(
            index="paper_id",
            columns="model_id",
            values="realized_diffusion_target",
        )
        .max(axis=1)
        .sub(
            predictions.pivot(
                index="paper_id",
                columns="model_id",
                values="realized_diffusion_target",
            ).min(axis=1)
        )
        .fillna(0)
        .abs()
        .le(1e-12)
        .all(),
        "manifest_declares_fixed_protocol": bool(
            manifest["same_papers_labels_folds_parameters"]
            and not manifest["feature_selection_from_oof"]
            and not manifest["parameter_selection_from_oof"]
        ),
        "all_set_features_are_matrix_columns": all(
            set(sets["sets"][model_id]["feature_ids"]).issubset(matrix.columns)
            for model_id in EXPECTED
        ),
        "runner_implementation_hash_matches": sha256_file(
            Path(manifest["implementation"]["path"])
        )
        == manifest["implementation"]["sha256"],
        "matrix_builder_hash_matches": sha256_file(
            Path(matrix_manifest["implementation"]["path"])
        )
        == matrix_manifest["implementation"]["sha256"],
    }
    for model_id, (feature_count, dimension_count) in EXPECTED.items():
        item = sets["sets"][model_id]
        checks[f"{model_id}_counts"] = (
            item["feature_count"],
            item["dimension_count"],
        ) == (feature_count, dimension_count)
        checkpoint_dir = output_dir / "checkpoints" / "D5" / model_id
        checks[f"{model_id}_six_checkpoints"] = (
            checkpoint_dir.is_dir()
            and len(list(checkpoint_dir.glob("fold_*.parquet"))) == 6
        )
    output_hash_checks = {
        key: sha256_file(Path(item["path"])) == item["sha256"]
        for key, item in manifest["outputs"].items()
    }
    matrix_hash_checks = {
        key: sha256_file(Path(item["path"])) == item["sha256"]
        for key, item in matrix_manifest["outputs"].items()
    }
    recomputed = {
        model_id: group["realized_diffusion_target"].corr(
            group["expected_diffusion_score"], method="spearman"
        )
        for model_id, group in predictions.groupby("model_id")
    }
    reported = metrics.set_index("model_id")["spearman_expected"].to_dict()
    checks["metrics_recompute_from_predictions"] = all(
        np.isclose(recomputed[model_id], reported[model_id], atol=1e-12)
        for model_id in EXPECTED
    )
    checks["all_manifest_output_hashes_match"] = all(
        output_hash_checks.values()
    )
    checks["all_matrix_output_hashes_match"] = all(
        matrix_hash_checks.values()
    )
    checks = {key: bool(value) for key, value in checks.items()}
    output_hash_checks = {
        key: bool(value) for key, value in output_hash_checks.items()
    }
    passed = all(checks.values())
    report = {
        "artifact_kind": "evidence_v3_four_set_oof_completion_audit",
        "passed": passed,
        "checks": checks,
        "output_hash_checks": output_hash_checks,
        "matrix_hash_checks": matrix_hash_checks,
        "best_model_id": manifest["best_model_id"],
        "best_oof_spearman": manifest["best_oof_spearman"],
        "metrics": metrics.to_dict(orient="records"),
    }
    write_json(output_dir / "completion_audit.json", report)
    if not passed:
        failed = [key for key, value in checks.items() if not value]
        raise ValueError(f"completion audit failed: {failed}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    result = verify(parse_args().output_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))

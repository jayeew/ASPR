"""Synthetic contract tests for the canonical evidence-derived HGB adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from innovation_impact_feature_selection.evidence_derived.core import (
    EvidenceProtocol,
    canonical_json,
    sha256_text,
)
from innovation_impact_feature_selection.evidence_derived.hgb_oof import (
    HgbOofContractError,
    _require_complete_protocol_audit,
    load_training_bundle,
    run_with_frames,
    sha256_file,
    validate_outputs,
)

SET_MEMBERS = {
    "strict_training": ["I1"],
    "primary": ["I1", "I2"],
    "expanded": ["I1", "I2", "I3"],
    "broad_t0": ["I1", "I2", "I3", "I4"],
}
FEATURE_NAMES = {
    "strict": ["feature_1"],
    "primary": ["feature_1", "feature_2"],
    "expanded": ["feature_1", "feature_2", "feature_3"],
    "broad_t0": ["feature_1", "feature_2", "feature_3", "feature_4"],
}
SOURCE_SETS = {
    "strict": "strict_training",
    "primary": "primary",
    "expanded": "expanded",
    "broad_t0": "broad_t0",
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    ids = [f"P{index:02d}" for index in range(9)]
    full_sets = {
        "all": ["I1", "I2", "I3", "I4"],
        "model": ["I1", "I2", "I3"],
        "strict": ["I1"],
        **SET_MEMBERS,
    }
    freeze_hash = sha256_text(canonical_json(full_sets))
    frozen = {
        "protocol_hash": "p" * 64,
        "freeze_hash": freeze_hash,
        "frozen_before_model_training": True,
        "outcome_columns_used": False,
        "sets": SET_MEMBERS,
        "canonical_sets": {
            "F_all": full_sets["all"],
            "F_model": full_sets["model"],
            "F_strict": full_sets["strict"],
        },
    }
    frozen_path = tmp_path / "final_feature_sets.json"
    _write_json(frozen_path, frozen)
    definitions: dict[str, Any] = {}
    base = pd.DataFrame(
        {
            "paper_id": ids,
            **{
                f"feature_{number}": [float(index + number) for index in range(9)]
                for number in range(1, 5)
            },
        }
    )
    for set_name, names in FEATURE_NAMES.items():
        path = tmp_path / f"final_training_features_{set_name}.parquet"
        base[["paper_id", *names]].to_parquet(path, index=False)
        source_set = SOURCE_SETS[set_name]
        definitions[set_name] = {
            "source_set": source_set,
            "indicator_ids": SET_MEMBERS[source_set],
            "feature_names": names,
            "path": str(path),
            "sha256": sha256_file(path),
            "row_count": len(base),
        }
    matrix_manifest = {
        "contract": "evidence_derived_training_matrices_v1",
        "protocol_hash": frozen["protocol_hash"],
        "feature_set_freeze_hash": freeze_hash,
        "frozen_before_model_training": True,
        "outcome_columns_used": False,
        "id_column": "paper_id",
        "sets": definitions,
    }
    matrix_path = tmp_path / "training_matrix_manifest.json"
    _write_json(matrix_path, matrix_manifest)
    return matrix_path, frozen_path, ids


def _frames(ids: list[str]) -> dict[int, pd.DataFrame]:
    years = [2000, 2000, 2000, 2001, 2001, 2001, 2002, 2002, 2002]
    frame = pd.DataFrame(
        {
            "paper_id": ids,
            "publication_year": years,
            "domain12": ["a", "b", "c"] * 3,
            "realized_diffusion_target": [float(index) for index in range(9)],
            **{
                f"feature_{number}": [float(index + number) for index in range(9)]
                for number in range(1, 5)
            },
        }
    )
    return {horizon: frame.assign(horizon=horizon) for horizon in (3, 5, 8)}


def _config() -> dict[str, Any]:
    folds = [
        {
            "fold_id": 1,
            "train_year_max": 2000,
            "test_year_min": 2001,
            "test_year_max": 2001,
        },
        {
            "fold_id": 2,
            "train_year_max": 2001,
            "test_year_min": 2002,
            "test_year_max": 2002,
        },
    ]
    return {
        "horizon_folds": {str(horizon): folds for horizon in (3, 5, 8)},
        "hgb": {"inner_temporal_folds": 2, "seed": 7},
    }


def _fake_oof(frame: pd.DataFrame, **kwargs: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    horizon = int(kwargs["horizon"])
    root = Path(kwargs["checkpoint_root"])
    for fold in kwargs["fold_config"]:
        testing = frame[
            frame["publication_year"].between(
                int(fold["test_year_min"]), int(fold["test_year_max"])
            )
        ]
        fold_rows.append(dict(fold))
        for offset, model_id in enumerate(kwargs["model_ids"]):
            prediction = pd.DataFrame(
                {
                    "paper_id": testing["paper_id"].astype(str),
                    "publication_year": testing["publication_year"],
                    "domain12": testing["domain12"],
                    "horizon": horizon,
                    "model_id": model_id,
                    "outer_fold_id": int(fold["fold_id"]),
                    "realized_diffusion_target": testing["realized_diffusion_target"],
                    "expected_diffusion_score": testing["realized_diffusion_target"]
                    + offset / 10.0,
                }
            )
            checkpoint = (
                root
                / f"D{horizon}"
                / str(model_id)
                / (f"fold_{int(fold['fold_id'])}.parquet")
            )
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            prediction.to_parquet(checkpoint, index=False)
            rows.append(prediction)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(fold_rows)


def test_synthetic_four_set_runner_and_validator(tmp_path: Path) -> None:
    matrix_path, frozen_path, ids = _fixture(tmp_path)
    bundle = load_training_bundle(matrix_path, frozen_path, ids)
    assert bundle.canonical_feature_sets["strict"] == ("feature_1",)
    assert bundle.feature_sets["strict"] == ("canonical_feature_0001",)
    assert "feature_1" not in bundle.matrix.columns
    output = tmp_path / "oof"
    manifest = run_with_frames(
        bundle,
        _frames(ids),
        _config(),
        output,
        oof_runner=_fake_oof,
    )

    report = validate_outputs(output, bundle, _frames(ids), _config())

    assert manifest["set_names"] == ["strict", "primary", "expanded", "broad_t0"]
    assert manifest["checkpoint_count"] == 24
    assert manifest["selection_feedback_used"] is False
    assert manifest["model_feature_aliases"]["feature_1"] == ("canonical_feature_0001")
    assert report["passed"] is True
    assert report["prediction_rows"] == 72
    assert report["metric_rows"] == 12
    assert (output / "model_comparison.csv").is_file()
    assert (output / "paper_scores.parquet").is_file()


def test_bundle_rejects_freeze_hash_mismatch(tmp_path: Path) -> None:
    matrix_path, frozen_path, ids = _fixture(tmp_path)
    payload = json.loads(frozen_path.read_text(encoding="utf-8"))
    payload["sets"]["primary"] = ["I2"]
    _write_json(frozen_path, payload)

    with pytest.raises(HgbOofContractError, match="freeze hash mismatch"):
        load_training_bundle(matrix_path, frozen_path, ids)


def test_bundle_reorders_identical_paper_id_membership(tmp_path: Path) -> None:
    matrix_path, frozen_path, ids = _fixture(tmp_path)
    payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    for definition in payload["sets"].values():
        path = Path(definition["path"])
        frame = pd.read_parquet(path).iloc[::-1].reset_index(drop=True)
        frame.to_parquet(path, index=False)
        definition["sha256"] = sha256_file(path)
    _write_json(matrix_path, payload)

    bundle = load_training_bundle(matrix_path, frozen_path, ids)

    assert bundle.matrix["paper_id"].tolist() == ids


def test_validator_recomputes_metrics(tmp_path: Path) -> None:
    matrix_path, frozen_path, ids = _fixture(tmp_path)
    bundle = load_training_bundle(matrix_path, frozen_path, ids)
    output = tmp_path / "oof"
    run_with_frames(
        bundle,
        _frames(ids),
        _config(),
        output,
        oof_runner=_fake_oof,
    )
    metrics_path = output / "oof_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    metrics.loc[0, "spearman"] = -0.5
    metrics.to_csv(metrics_path, index=False)

    report = validate_outputs(output, bundle, _frames(ids), _config())

    assert report["passed"] is False
    assert report["checks"]["metric_values_recomputed"] is False
    assert report["checks"]["artifact_hashes"] is False


def test_bundle_rejects_outcome_leakage(tmp_path: Path) -> None:
    matrix_path, frozen_path, ids = _fixture(tmp_path)
    payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    definition = payload["sets"]["broad_t0"]
    path = Path(definition["path"])
    frame = pd.read_parquet(path).rename(columns={"feature_4": "future_uptake"})
    frame.to_parquet(path, index=False)
    definition["feature_names"][-1] = "future_uptake"
    definition["sha256"] = sha256_file(path)
    _write_json(matrix_path, payload)

    with pytest.raises(HgbOofContractError, match="outcome leakage"):
        load_training_bundle(matrix_path, frozen_path, ids)


def test_materializer_writes_frozen_matrix_manifest(tmp_path: Path) -> None:
    output = tmp_path / "outputs"
    with EvidenceProtocol(tmp_path / "evidence.sqlite3", output) as engine:
        engine.initialize()
        rows = []
        for index in range(1, 5):
            rows.append(
                (
                    f"I{index}",
                    f"feature_{index}",
                    "[]",
                    "[]",
                    "[]",
                    "definition",
                    "formula",
                    "[]",
                    "[]",
                    "predictive",
                    "T0",
                    "missing",
                    "missing",
                    "missing",
                    "complete",
                    "missing",
                    "candidate",
                )
            )
        engine.connection.executemany(
            "INSERT INTO indicator_families VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        memberships = [
            ("I1", "strict_training"),
            ("I1", "primary"),
            ("I2", "primary"),
            ("I1", "expanded"),
            ("I2", "expanded"),
            ("I3", "expanded"),
            ("I1", "broad_t0"),
            ("I2", "broad_t0"),
            ("I3", "broad_t0"),
            ("I4", "broad_t0"),
        ]
        engine.connection.executemany(
            "INSERT INTO final_features VALUES(?,?,?,?)",
            [
                (indicator, set_name, "test", "f" * 64)
                for indicator, set_name in memberships
            ],
        )
        engine.set_metadata("feature_set_freeze_hash", "f" * 64)
        engine.set_metadata(
            "outcome_blind_audit",
            {"status": "pass", "outcome_columns_used": False},
        )
        engine.connection.commit()
        source = tmp_path / "source.parquet"
        pd.DataFrame(
            {
                "paper_id": ["P1", "P2"],
                **{
                    f"feature_{index}": [float(index), float(index + 1)]
                    for index in range(1, 5)
                },
            }
        ).to_parquet(source, index=False)

        counts = engine.materialize_training_sets(source)
        engine._validate_training_matrices()

    manifest = json.loads(
        (output / "training_matrix_manifest.json").read_text(encoding="utf-8")
    )
    assert counts == {"strict": 1, "primary": 2, "expanded": 3, "broad_t0": 4}
    assert manifest["feature_set_freeze_hash"] == "f" * 64
    assert manifest["outcome_columns_used"] is False
    assert list(manifest["sets"]["expanded"]["feature_names"]) == [
        "feature_1",
        "feature_2",
        "feature_3",
    ]


def test_runner_requires_exact_complete_audit_status(tmp_path: Path) -> None:
    report = tmp_path / "audit_report.md"
    report.write_text(
        "# Evidence-derived audit report\n\nStatus: **INCOMPLETE**\n",
        encoding="utf-8",
    )
    with pytest.raises(HgbOofContractError, match="not COMPLETE"):
        _require_complete_protocol_audit(report)
    report.write_text(
        "# Evidence-derived audit report\n\nStatus: **COMPLETE**\n",
        encoding="utf-8",
    )
    _require_complete_protocol_audit(report)

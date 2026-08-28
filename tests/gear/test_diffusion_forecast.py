import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from gear.diffusion_forecast import (
    DiffusionForecastService,
    ForecastRelease,
    RuntimeFeatureRelease,
    StructuralHeadRelease,
    sha256_file,
    validate_runtime_replay,
    validate_structural_head_replay,
)
from gear.structural_innovation import build_graph_signal_bundle

MANIFEST = Path(
    "data/calibration/releases/gear-d5-primary16-current/release_manifest.json"
).resolve()
RUNTIME_MANIFEST = Path(
    "data/calibration/runtime_features/gear-d5-primary16-dev10-v1/runtime_manifest.json"
).resolve()
ANATOMY_MANIFEST = Path(
    "data/calibration/graph_calibration/primary16_forecast_anatomy_v1/manifest.json"
).resolve()
STRUCTURAL_MANIFEST = Path(
    "data/calibration/graph_calibration/gear_structural_head_release_v1/manifest.json"
).resolve()


def _structural_release(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    root = tmp_path / "structural-head-release"
    root.mkdir()
    if any(row["prediction_protocol"] == "frozen_t0_runtime" for row in rows):
        supplied = {str(row["paper_id"]): row for row in rows}
        runtime_scores = pd.read_parquet(
            RuntimeFeatureRelease(RUNTIME_MANIFEST).path("runtime_score_table")
        )
        rows = [
            *[row for row in rows if row["prediction_protocol"] == "strict_oof"],
            *[
                supplied.get(str(row.paper_id))
                or _head_row(
                    str(row.paper_id),
                    date.fromisoformat(str(row.as_of_date)[:10]),
                    protocol="frozen_t0_runtime",
                    outer_fold_id=None,
                )
                for row in runtime_scores.itertuples(index=False)
            ],
        ]
    parent = ForecastRelease(MANIFEST).manifest
    parent_registry = json.loads(
        ForecastRelease(MANIFEST).path("feature_registry").read_text(encoding="utf-8")
    )
    feature_names = parent_registry["feature_names"]
    registry_path = root / "feature_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "feature_names": feature_names,
                "feature_time_basis": "T0_only",
                "uses_future_features": False,
                "label_only_columns_excluded_from_models": True,
                "parent_feature_registry_sha256": parent.assets[
                    "feature_registry"
                ].sha256,
            }
        ),
        encoding="utf-8",
    )
    prediction_path = root / "prediction_table.parquet"
    pd.DataFrame(rows).to_parquet(prediction_path, index=False)
    paths = {
        "model": root / "model.joblib",
        "feature_registry": registry_path,
        "training_reference": root / "training_reference.parquet",
        "prediction_table": prediction_path,
        "validation_report": root / "validation_report.json",
        "runtime_replay": root / "runtime_replay.parquet",
        "coverage_audit": root / "coverage_audit.json",
    }
    paths["model"].write_bytes(b"frozen-structural-head-model")
    historical = [row for row in rows if row["prediction_protocol"] == "strict_oof"]
    training_rows = [
        {
            "paper_id": str(row["paper_id"]),
            "outer_fold_id": str(row["outer_fold_id"]),
            "feature_time_basis": "T0_only",
            "future_columns_role": "label_construction_only_not_inference",
        }
        for row in historical
    ]
    training_rows.extend(
        {
            "paper_id": f"training-paper-{index}",
            "outer_fold_id": f"fold-{index % 3}",
            "feature_time_basis": "T0_only",
            "future_columns_role": "label_construction_only_not_inference",
        }
        for index in range(241 - len(training_rows))
    )
    pd.DataFrame(training_rows).to_parquet(paths["training_reference"], index=False)
    pd.DataFrame({"paper_id": ["replay-fixture"]}).to_parquet(
        paths["runtime_replay"], index=False
    )
    paths["coverage_audit"].write_text(
        json.dumps(
            {
                "contract": "gear_structural_head_coverage_audit_v1",
                "passed": True,
            }
        ),
        encoding="utf-8",
    )
    paths["validation_report"].write_text(
        json.dumps(
            {
                "contract": "gear_structural_head_validation_v1",
                "status": "supported",
                "promotion_passed": True,
                "promotion_gates": {"test_fixture": True},
            }
        ),
        encoding="utf-8",
    )
    assets = {
        name: {
            "file": path.name,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in paths.items()
    }
    manifest = {
        "contract": "gear_structural_head_release_v1",
        "release_id": "gear-structural-heads-test-v1",
        "parent_forecast_release_id": parent.release_id,
        "parent_feature_registry_sha256": parent.assets["feature_registry"].sha256,
        "feature_protocol_version": parent.protocol_version,
        "horizon_years": 5,
        "status": "promoted",
        "feature_time_basis": "T0_only",
        "uses_future_features": False,
        "historical_prediction_policy": "strict_oof_only",
        "runtime_prediction_policy": "frozen_model_t0_only",
        "excess_target_fit_scope": "outer_training_fold_only",
        "perturbation_target_fit_scope": "outer_training_fold_only",
        "feature_names": feature_names,
        "training_row_count": 241,
        "created_at_utc": "2026-08-28T00:00:00Z",
        "assets": assets,
    }
    runtime_rows = [
        row for row in rows if row["prediction_protocol"] == "frozen_t0_runtime"
    ]
    if runtime_rows:
        runtime = RuntimeFeatureRelease(RUNTIME_MANIFEST)
        manifest.update(
            {
                "runtime_feature_release_id": runtime.manifest.release_id,
                "runtime_feature_table_sha256": runtime.manifest.assets[
                    "runtime_feature_table"
                ].sha256,
                "runtime_score_table_sha256": runtime.manifest.assets[
                    "runtime_score_table"
                ].sha256,
                "runtime_anatomy_table_sha256": runtime.manifest.assets[
                    "runtime_anatomy_table"
                ].sha256,
                "runtime_prediction_row_count": len(runtime_rows),
            }
        )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _head_row(
    paper_id: str,
    target_date: date,
    *,
    protocol: str,
    outer_fold_id: str | None,
) -> dict[str, object]:
    return {
        "paper_id": paper_id,
        "prediction_protocol": protocol,
        "as_of_date": target_date.isoformat(),
        "target_publication_date": target_date.isoformat(),
        "feature_source_max_date": target_date.isoformat(),
        "outer_fold_id": outer_fold_id,
        "excess_diffusion_head_d": 0.61,
        "field_year_base": 0.22,
        "perturbation_head_p": 0.73,
        "prediction_interval_width": 0.10,
        "ood_reliability": 0.90,
        "calibration_reliability": 0.80,
        "feature_source_sha256": "sha256:" + "0" * 64,
        "perturbation_component_boundary_expansion": 0.70,
    }


def test_release_hashes_folds_percentiles_and_runtime_replay() -> None:
    report = validate_runtime_replay(MANIFEST)
    assert report["passed"] is True
    assert report["temporal_folds"] == 7
    assert report["runtime_replay_rows"] == 4096
    assert report["max_abs_error"] == 0.0
    release = ForecastRelease(MANIFEST)
    registry = json.loads(release.path("feature_registry").read_text(encoding="utf-8"))
    assert registry["model_id"] == "primary16"
    assert registry["source_fig2_sets"] == {
        "strict": 7,
        "primary": 16,
        "expanded": 153,
        "broad_t0": 219,
    }
    assert len(registry["feature_names"]) == 16


def test_real_structural_release_hashes_scientific_gate_and_replay() -> None:
    report = validate_structural_head_replay(
        STRUCTURAL_MANIFEST, MANIFEST, RUNTIME_MANIFEST
    )
    assert report["passed"] is True
    assert report["prediction_rows"] == 318005
    assert report["runtime_replay_rows"] == 31
    assert report["runtime_replay_outputs"] == 10
    assert report["max_abs_error"] == 0.0


def test_real_structural_release_covers_all_ten_runtime_targets(paper_ir) -> None:
    runtime = RuntimeFeatureRelease(RUNTIME_MANIFEST)
    scores = pd.read_parquet(runtime.path("runtime_score_table"))
    service = DiffusionForecastService(
        MANIFEST,
        RUNTIME_MANIFEST,
        structural_head_manifest_path=STRUCTURAL_MANIFEST,
    )
    for row in scores.itertuples(index=False):
        cutoff = date.fromisoformat(str(row.as_of_date)[:10])
        target = paper_ir.model_copy(
            update={
                "paper_id": str(row.paper_id),
                "metadata": paper_ir.metadata.model_copy(
                    update={
                        "openalex_id": str(row.paper_id),
                        "doi": None,
                        "publication_date": cutoff,
                    }
                ),
            }
        )
        packet = service.score(target, cutoff)
        assert packet.forecast.structural_heads_status == "available"
        assert packet.forecast.structural_head_prediction_protocol == (
            "frozen_t0_runtime"
        )
        assert packet.forecast.perturbation_potential is not None


def test_real_structural_release_covers_stage_b_and_stage_c_protocols() -> None:
    release = StructuralHeadRelease(STRUCTURAL_MANIFEST)
    release.verify(ForecastRelease(MANIFEST), RuntimeFeatureRelease(RUNTIME_MANIFEST))
    audit = json.loads(release.path("coverage_audit").read_text(encoding="utf-8"))
    assert audit["passed"] is True
    assert audit["stage_b_241"]["available_papers"] == 241
    assert audit["stage_b_241"]["protocol_counts"] == {"strict_oof": 241}
    assert audit["stage_c_150"]["available_papers"] == 150
    assert audit["stage_c_150"]["protocols_correct_by_training_identity"] is True
    assert audit["runtime_10"]["available_papers"] == 10
    assert audit["runtime_10"]["protocols"] == ["frozen_t0_runtime"]


def test_structural_release_rejects_failed_scientific_promotion(tmp_path) -> None:
    cutoff = date(2018, 1, 1)
    manifest_path = _structural_release(
        tmp_path,
        [
            _head_row(
                "scientific-gate-paper",
                cutoff,
                protocol="strict_oof",
                outer_fold_id="4",
            )
        ],
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report_path = manifest_path.parent / "validation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["promotion_passed"] = False
    report["promotion_gates"] = {"latest_temporal:d_excess:positive_ci": False}
    report_path.write_text(json.dumps(report), encoding="utf-8")
    manifest["assets"]["validation_report"] = {
        "file": report_path.name,
        "sha256": sha256_file(report_path),
        "size_bytes": report_path.stat().st_size,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="scientific promotion"):
        StructuralHeadRelease(manifest_path).verify(ForecastRelease(MANIFEST))


def test_exact_lookup_uses_future_diffusion_semantics(paper_ir) -> None:
    release = ForecastRelease(MANIFEST)
    scores = pd.read_parquet(
        release.path("score_table"), columns=["paper_id", "publication_year"]
    )
    row = scores.loc[scores["publication_year"] < 2022].iloc[0]
    paper_id = str(row["paper_id"])
    target = paper_ir.model_copy(
        update={
            "paper_id": paper_id,
            "metadata": paper_ir.metadata.model_copy(update={"openalex_id": paper_id}),
        }
    )
    packet = DiffusionForecastService(
        MANIFEST, anatomy_manifest_path=ANATOMY_MANIFEST
    ).score(target, date(int(row["publication_year"]) + 1, 1, 1))
    assert packet.forecast.status == "available"
    assert packet.forecast.prospective_5y_diffusion_percentile is not None
    assert packet.forecast.release_id == "gear-d5-primary16-948a4f87086c"
    assert packet.topology_seeds == []
    assert packet.forecast_anatomy is not None


def test_exact_publication_date_allows_frozen_t0_oof_lookup(paper_ir) -> None:
    release = ForecastRelease(MANIFEST)
    scores = pd.read_parquet(
        release.path("score_table"), columns=["paper_id", "publication_year"]
    )
    row = scores.loc[scores["publication_year"] < 2022].iloc[0]
    cutoff = date(int(row["publication_year"]), 1, 1)
    paper_id = str(row["paper_id"])
    target = paper_ir.model_copy(
        update={
            "paper_id": paper_id,
            "metadata": paper_ir.metadata.model_copy(
                update={"openalex_id": paper_id, "publication_date": cutoff}
            ),
        }
    )

    packet = DiffusionForecastService(MANIFEST).score(target, cutoff)

    assert packet.forecast.status == "available"
    assert "frozen_publication_t0_oof_lookup" in packet.diagnostics


def test_unknown_paper_fails_closed_to_explicit_limited_packet(paper_ir) -> None:
    packet = DiffusionForecastService(MANIFEST).score(paper_ir, date(2025, 1, 1))
    assert packet.forecast.status == "unavailable"
    assert "graph_limited" in packet.diagnostics
    assert packet.forecast.prospective_5y_diffusion_percentile is None


def test_recent_quick_gate_papers_recompute_frozen_features(paper_ir) -> None:
    service = DiffusionForecastService(MANIFEST, RUNTIME_MANIFEST)
    cases = (
        ("https://openalex.org/W4400732366", date(2023, 9, 21), 35.598044, 1.0),
        ("https://openalex.org/W4413257773", date(2025, 4, 6), 97.780783, 0.6),
        ("https://openalex.org/W4415618618", date(2025, 5, 26), 75.394582, 0.6),
    )
    for paper_id, cutoff, expected, coverage in cases:
        target = paper_ir.model_copy(
            update={
                "paper_id": paper_id,
                "metadata": paper_ir.metadata.model_copy(
                    update={
                        "openalex_id": paper_id,
                        "doi": None,
                        "publication_date": cutoff,
                    }
                ),
            }
        )
        packet = service.score(target, cutoff)
        assert packet.forecast.status == "available"
        assert packet.score_0_100 == pytest.approx(expected, abs=1e-6)
        assert packet.feature_coverage == coverage
        assert "runtime_feature_inference" in packet.diagnostics
        assert "target_primary16_recomputed" in packet.diagnostics
        assert "frozen_primary16_hgb_reused" in packet.diagnostics
        assert packet.forecast_anatomy is not None
        assert packet.forecast_anatomy.paper_id == paper_id
        assert packet.forecast_anatomy.anatomy_release_id == packet.forecast.release_id


def test_post_2022_target_cannot_use_training_score_lookup(paper_ir) -> None:
    release = ForecastRelease(MANIFEST)
    row = pd.read_parquet(release.path("score_table")).iloc[0]
    paper_id = str(row["paper_id"])
    target = paper_ir.model_copy(
        update={
            "paper_id": paper_id,
            "metadata": paper_ir.metadata.model_copy(
                update={
                    "openalex_id": paper_id,
                    "doi": None,
                    "publication_date": date(2024, 1, 1),
                }
            ),
        }
    )

    packet = DiffusionForecastService(MANIFEST).score(target, date(2024, 1, 1))

    assert packet.forecast.status == "unavailable"
    assert "post_2022_requires_runtime_primary16_recompute" in packet.diagnostics


def test_runtime_feature_cutoff_mismatch_fails_closed(paper_ir) -> None:
    target = paper_ir.model_copy(
        update={
            "paper_id": "https://openalex.org/W4400732366",
            "metadata": paper_ir.metadata.model_copy(
                update={"openalex_id": "https://openalex.org/W4400732366"}
            ),
        }
    )
    packet = DiffusionForecastService(MANIFEST, RUNTIME_MANIFEST).score(
        target, date(2023, 9, 22)
    )
    assert packet.forecast.status == "unavailable"
    assert "runtime_feature_cutoff_mismatch" in packet.diagnostics


def test_missing_structural_heads_is_explicitly_limited_without_losing_hgb_d(
    paper_ir,
) -> None:
    release = ForecastRelease(MANIFEST)
    row = pd.read_parquet(
        release.path("score_table"), columns=["paper_id", "publication_year"]
    ).iloc[0]
    cutoff = date(int(row["publication_year"]), 1, 1)
    paper_id = str(row["paper_id"])
    target = paper_ir.model_copy(
        update={
            "paper_id": paper_id,
            "metadata": paper_ir.metadata.model_copy(
                update={"openalex_id": paper_id, "publication_date": cutoff}
            ),
        }
    )

    packet = DiffusionForecastService(MANIFEST).score(target, cutoff)
    bundle = build_graph_signal_bundle(packet)

    assert packet.forecast.status == "available"
    assert packet.forecast.expected_diffusion is not None
    assert packet.forecast.structural_heads_status == "unavailable"
    assert "structural_heads_manifest_not_configured" in packet.diagnostics
    assert bundle.limited is True
    assert 0.0 < bundle.shrunk_diffusion <= packet.forecast.expected_diffusion


def test_historical_structural_heads_require_and_expose_strict_oof(
    paper_ir, tmp_path
) -> None:
    release = ForecastRelease(MANIFEST)
    row = pd.read_parquet(
        release.path("score_table"), columns=["paper_id", "publication_year"]
    ).iloc[0]
    cutoff = date(int(row["publication_year"]), 1, 1)
    paper_id = str(row["paper_id"])
    head_manifest = _structural_release(
        tmp_path,
        [_head_row(paper_id, cutoff, protocol="strict_oof", outer_fold_id="fold-2")],
    )
    target = paper_ir.model_copy(
        update={
            "paper_id": paper_id,
            "metadata": paper_ir.metadata.model_copy(
                update={"openalex_id": paper_id, "publication_date": cutoff}
            ),
        }
    )

    packet = DiffusionForecastService(
        MANIFEST, structural_head_manifest_path=head_manifest
    ).score(target, cutoff)
    bundle = build_graph_signal_bundle(packet)

    assert packet.forecast.structural_heads_status == "available"
    assert packet.forecast.structural_head_prediction_protocol == "strict_oof"
    assert packet.forecast.excess_diffusion == pytest.approx(0.61)
    assert packet.forecast.perturbation_potential == pytest.approx(0.73)
    assert packet.forecast.perturbation_components == {"boundary_expansion": 0.7}
    assert packet.forecast.structural_head_model_sha256 is not None
    assert bundle.shrunk_diffusion == pytest.approx(0.34636)
    assert bundle.structural_heads_status == "available"
    assert bundle.limited is True  # This fixture's Primary16 coverage is only 0.5.


def test_runtime_structural_heads_require_exact_cutoff_and_frozen_t0_protocol(
    paper_ir, tmp_path
) -> None:
    paper_id = "https://openalex.org/W4400732366"
    cutoff = date(2023, 9, 21)
    head_manifest = _structural_release(
        tmp_path,
        [
            _head_row(
                paper_id,
                cutoff,
                protocol="frozen_t0_runtime",
                outer_fold_id=None,
            )
        ],
    )
    target = paper_ir.model_copy(
        update={
            "paper_id": paper_id,
            "metadata": paper_ir.metadata.model_copy(
                update={"openalex_id": paper_id, "publication_date": cutoff}
            ),
        }
    )

    packet = DiffusionForecastService(
        MANIFEST, RUNTIME_MANIFEST, structural_head_manifest_path=head_manifest
    ).score(target, cutoff)

    assert packet.forecast.structural_heads_status == "available"
    assert packet.forecast.structural_head_prediction_protocol == "frozen_t0_runtime"
    assert "structural_head_features_t0_only" in packet.diagnostics


def test_post_cutoff_structural_feature_provenance_fails_closed(
    paper_ir, tmp_path
) -> None:
    release = ForecastRelease(MANIFEST)
    row = pd.read_parquet(
        release.path("score_table"), columns=["paper_id", "publication_year"]
    ).iloc[0]
    cutoff = date(int(row["publication_year"]), 1, 1)
    paper_id = str(row["paper_id"])
    leaked = _head_row(paper_id, cutoff, protocol="strict_oof", outer_fold_id="fold-2")
    leaked["feature_source_max_date"] = date(cutoff.year + 1, 1, 1).isoformat()
    head_manifest = _structural_release(tmp_path, [leaked])
    target = paper_ir.model_copy(
        update={
            "paper_id": paper_id,
            "metadata": paper_ir.metadata.model_copy(
                update={"openalex_id": paper_id, "publication_date": cutoff}
            ),
        }
    )

    packet = DiffusionForecastService(
        MANIFEST, structural_head_manifest_path=head_manifest
    ).score(target, cutoff)

    assert packet.forecast.status == "available"
    assert packet.forecast.structural_heads_status == "unavailable"
    assert "structural_heads_release_unavailable:ValueError" in packet.diagnostics
    with pytest.raises(ValueError, match="post-cutoff"):
        StructuralHeadRelease(head_manifest).verify(release)

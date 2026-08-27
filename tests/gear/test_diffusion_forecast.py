import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from gear.diffusion_forecast import (
    DiffusionForecastService,
    ForecastRelease,
    validate_runtime_replay,
)

MANIFEST = Path(
    "data/calibration/releases/gear-d5-primary16-current/release_manifest.json"
).resolve()
RUNTIME_MANIFEST = Path(
    "data/calibration/runtime_features/gear-d5-primary16-dev10-v1/runtime_manifest.json"
).resolve()


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
    packet = DiffusionForecastService(MANIFEST).score(
        target, date(int(row["publication_year"]) + 1, 1, 1)
    )
    assert packet.forecast.status == "available"
    assert packet.forecast.prospective_5y_diffusion_percentile is not None
    assert packet.forecast.release_id == "gear-d5-primary16-948a4f87086c"
    assert packet.topology_seeds == []


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

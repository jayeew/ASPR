"""Contract tests for horizon-specific nested HGB tuning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from innovation_impact_feature_selection.evidence_derived import hgb_nested_tuning
from innovation_impact_feature_selection.evidence_derived.hgb_nested_tuning import (
    frozen_matrix_manifest,
    search_primary_parameters,
)
from innovation_impact_feature_selection.evidence_derived.hgb_oof import sha256_file


def test_search_uses_only_supplied_outer_training_and_selects_best(
    monkeypatch: Any,
) -> None:
    training = pd.DataFrame(
        {
            "paper_id": ["a", "b", "c", "d"],
            "publication_year": [1999, 1999, 2000, 2000],
            "x": [0.0, 1.0, 2.0, 3.0],
        }
    )
    grid = (
        {"parameter_id": "weak", "max_leaf_nodes": 15},
        {"parameter_id": "strong", "max_leaf_nodes": 31},
    )
    seen_max_years: list[int] = []

    def fake_inner(frame: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        seen_max_years.append(int(frame["publication_year"].max()))
        strong = kwargs["parameters"]["parameter_id"] == "strong"
        prediction = [0.1, 0.2, 0.3, 0.4] if strong else [0.4, 0.1, 0.3, 0.2]
        return pd.DataFrame(
            {
                "future_uptake": [0.0, 0.0, 1.0, 1.0],
                "uptake_probability_raw": [0.1, 0.2, 0.8, 0.9],
                "conditional_diffusion_raw": prediction,
                "realized_diffusion_target": [0.0, 0.0, 0.3, 0.8],
            }
        )

    monkeypatch.setattr(hgb_nested_tuning, "_inner_oof_for_parameters", fake_inner)
    selected, selected_inner, ledger = search_primary_parameters(
        training,
        feature_names=("x",),
        categorical_names=(),
        parameter_grid=grid,
        n_inner=2,
        seed=7,
        horizon=5,
        outer_fold_id=2,
    )

    assert selected["parameter_id"] == "strong"
    assert len(selected_inner) == 4
    assert ledger["selected"].sum() == 1
    assert seen_max_years == [2000, 2000]


def test_frozen_matrix_manifest_resolves_only_to_release(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    output = tmp_path / "tuning"
    release.mkdir()
    output.mkdir()
    sets: dict[str, Any] = {}
    for name in ("strict", "primary", "expanded", "broad_t0"):
        matrix = release / f"final_training_features_{name}.parquet"
        pd.DataFrame({"paper_id": ["p1"], "x": [1.0]}).to_parquet(matrix, index=False)
        sets[name] = {"path": "/mutable/source.parquet", "sha256": sha256_file(matrix)}
    (release / "training_matrix_manifest.json").write_text(
        json.dumps({"sets": sets}), encoding="utf-8"
    )

    rewritten = frozen_matrix_manifest(release, output)
    payload = json.loads(rewritten.read_text(encoding="utf-8"))

    for name, definition in payload["sets"].items():
        assert Path(definition["path"]).parent == release
        assert (
            Path(definition["path"]).name == f"final_training_features_{name}.parquet"
        )

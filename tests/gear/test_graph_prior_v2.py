from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import ValidationError

from gear.graph_feature_sets import validate_model_assignment
from gear.graph_prior import (
    GraphResultTable,
    GraphService,
    build_graph_search_hints,
    graph_result_from_calibration,
)
from gear.graph_prior_contracts import GraphResultV3, GraphResultV4


def test_graph_result_v4_contains_scores_and_search_hints(
    paper_ir, calibration_factory
) -> None:
    result = graph_result_from_calibration(
        calibration_factory(paper_ir.paper_id, score=82.5)
    )
    payload = result.model_dump(mode="json")
    assert payload == {
        "contract": "aspr_graph_result_v4",
        "paper_id": paper_ir.paper_id,
        "score_0_100": 82.5,
        "p_uptake": 0.5,
        "conditional_diffusion": 0.4,
        "feature_coverage": 1.0,
        "seed_work_ids": [],
        "search_terms": [],
    }


def test_graph_hints_are_stable_and_exclude_target() -> None:
    target = SimpleNamespace(
        work_id="W-target",
        year=2024,
        title="Novel Evidence Controller",
        references=[
            SimpleNamespace(work_id="R1", year=2020, field_id="machine learning"),
            SimpleNamespace(work_id="R2", year=2025, field_id="future field"),
        ],
    )
    context = SimpleNamespace(
        bibliographic_coupling_index={"R1": {"W-target", "W2", "W1"}},
        seen_title_bigrams={"evidence controller"},
    )
    seeds, terms = build_graph_search_hints(target, context)
    assert seeds == ["W1", "W2"]
    assert "novel evidence" in terms
    assert "future field" not in terms


def test_graph_result_v3_rejects_missing_invalid_and_extra_fields() -> None:
    valid = {
        "paper_id": "paper",
        "score_0_100": 50.0,
        "p_uptake": 0.5,
        "conditional_diffusion": 0.4,
        "feature_coverage": 1.0,
    }
    with pytest.raises(ValidationError):
        GraphResultV3.model_validate(
            {key: value for key, value in valid.items() if key != "p_uptake"}
        )
    with pytest.raises(ValidationError):
        GraphResultV3.model_validate({**valid, "score_0_100": 101.0})
    with pytest.raises(ValidationError):
        GraphResultV3.model_validate({**valid, "status": "exact_lookup"})


def test_feature_set_model_dimension_mismatch_fails() -> None:
    with pytest.raises(ValueError, match="requires 16 features"):
        validate_model_assignment("fulltext_16", [0.0] * 7, model_feature_count=16)
    with pytest.raises(ValueError, match="does not match"):
        validate_model_assignment("strict_7", [0.0] * 7, model_feature_count=16)


def test_cached_and_computed_graph_results_have_identical_contract(
    tmp_path, gear_config, paper_ir, paper_request, calibration_factory
) -> None:
    table_path = tmp_path / "scores.parquet"
    pd.DataFrame(
        [
            {
                "paper_id": "storage-id",
                "openalex_id": paper_ir.paper_id,
                "doi": None,
                "aspr_score": 82.5,
                "p_uptake": 0.5,
                "conditional_diffusion": 0.4,
                "feature_coverage": 1.0,
            }
        ]
    ).to_parquet(table_path, index=False)

    class CalibrationFake:
        def build_packet(self, supplied, **kwargs):
            return calibration_factory(supplied.paper_id, score=82.5)

    cached = GraphService(gear_config, result_table=GraphResultTable(table_path)).score(
        paper_ir, paper_request.evidence_date
    )
    computed = GraphService(gear_config, calibration_factory=CalibrationFake).score(
        paper_ir, paper_request.evidence_date
    )

    assert cached == computed


def test_graph_result_jsonl_is_consumed_as_the_same_v3_contract(
    tmp_path, paper_ir
) -> None:
    path = tmp_path / "graph_results.jsonl"
    payload = GraphResultV3(
        paper_id=paper_ir.paper_id,
        score_0_100=64.0,
        p_uptake=0.7,
        conditional_diffusion=0.5,
        feature_coverage=0.8,
    )
    path.write_text(payload.model_dump_json() + "\n", encoding="utf-8")

    loaded = GraphResultTable(path).lookup(paper_ir)

    assert loaded == GraphResultV4(
        paper_id=payload.paper_id,
        score_0_100=payload.score_0_100,
        p_uptake=payload.p_uptake,
        conditional_diffusion=payload.conditional_diffusion,
        feature_coverage=payload.feature_coverage,
    )


def test_graph_result_jsonl_rejects_non_v3_fields(tmp_path, paper_ir) -> None:
    path = tmp_path / "graph_results.jsonl"
    path.write_text(
        json.dumps(
            {
                "contract": "aspr_graph_result_v3",
                "paper_id": paper_ir.paper_id,
                "score_0_100": 64.0,
                "p_uptake": 0.7,
                "conditional_diffusion": 0.5,
                "feature_coverage": 0.8,
                "status": "development_replay",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid Graph result record"):
        GraphResultTable(path).lookup(paper_ir)

from __future__ import annotations

import pytest

from gear.graph_feature_sets import validate_model_assignment
from gear.graph_prior import graph_prior_from_calibration


def test_graph_prior_public_output_has_one_score(paper_ir, calibration_factory) -> None:
    result = graph_prior_from_calibration(
        calibration_factory(paper_ir.paper_id, score=82.5)
    )
    payload = result.model_dump(mode="json")
    assert payload["score_0_100"] == 82.5
    serialized = result.model_dump_json()
    assert "p_uptake" not in serialized
    assert "conditional_diffusion" not in serialized


def test_feature_set_model_dimension_mismatch_fails() -> None:
    with pytest.raises(ValueError, match="requires 16 features"):
        validate_model_assignment("fulltext_16", [0.0] * 7, model_feature_count=16)
    with pytest.raises(ValueError, match="does not match"):
        validate_model_assignment("strict_7", [0.0] * 7, model_feature_count=16)

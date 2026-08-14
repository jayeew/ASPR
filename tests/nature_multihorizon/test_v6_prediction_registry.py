from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from gear.nature_multihorizon.evidence_registry import load_evidence_registry
from gear.nature_multihorizon.evidence_selection_v6 import (
    SelectionDecision,
    audit_registry_source_selection,
    load_evidence_selection_protocol,
)
from gear.nature_multihorizon.prediction_features_v6 import (
    build_bibliographic_opportunity_features,
    build_registered_control_features,
)
from gear.nature_multihorizon.prediction_registry_v6 import (
    PredictionRegistry,
    PredictionRole,
    audit_prediction_registry_implementations,
    load_prediction_registry,
)
from gear.nature_multihorizon.targets_v6 import (
    FoldLocalDiffusionTargetTransformer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDICTION_REGISTRY_PATH = (
    PROJECT_ROOT / "configs" / "prediction_registry_v6_local.json"
)
INNOVATION_REGISTRY_PATH = (
    PROJECT_ROOT / "configs" / "innovation_registry_v6_local.json"
)
EVIDENCE_SELECTION_PATH = (
    PROJECT_ROOT / "configs" / "evidence_selection_protocol_v6.json"
)


def test_prediction_registry_is_source_complete_and_role_separated() -> None:
    registry = load_prediction_registry(PREDICTION_REGISTRY_PATH)
    innovation = load_evidence_registry(INNOVATION_REGISTRY_PATH)

    assert len(registry.categories) == 5
    assert registry.primary_outcome_names == ("future_uptake", "rgpm_d_fold")
    assert "publication_year" in registry.strong_control_names
    assert "bc_degree_per_reference_t0" in registry.opportunity_feature_names
    assert "bc_harmonic_closeness_t0" not in registry.opportunity_feature_names
    assert set(registry.strong_control_names).isdisjoint(
        innovation.primary_feature_names
    )
    assert set(registry.opportunity_feature_names).isdisjoint(
        innovation.primary_feature_names
    )
    assert audit_prediction_registry_implementations(registry)["overall_pass"]
    for variable in registry.variables.values():
        assert bool(variable.future_information_allowed) == (
            variable.role is PredictionRole.OUTCOME
        )


def test_targeted_evidence_map_covers_registries_and_adverse_evidence() -> None:
    prediction = load_prediction_registry(PREDICTION_REGISTRY_PATH)
    innovation = load_evidence_registry(INNOVATION_REGISTRY_PATH)
    protocol = load_evidence_selection_protocol(EVIDENCE_SELECTION_PATH)
    audit = audit_registry_source_selection(
        protocol, innovation, prediction
    )

    assert audit["overall_pass"]
    assert audit["n_registry_sources"] == 20
    assert (
        protocol.records["FONTANA2020"].decision
        is SelectionDecision.INCLUDED
    )
    assert (
        protocol.records["EXCLUDE_WU2019"].decision
        is SelectionDecision.EXCLUDED
    )
    assert (
        innovation.metrics["N1.FIRST_SHARE"].model_use.value
        == "profile_only"
    )


def test_prediction_registry_rejects_future_control_laundering() -> None:
    payload = json.loads(
        PREDICTION_REGISTRY_PATH.read_text(encoding="utf-8")
    )
    altered = copy.deepcopy(payload)
    altered["variables"]["K1.PUBLICATION_YEAR"][
        "future_information_allowed"
    ] = True
    with pytest.raises(ValidationError, match="cannot use the future"):
        PredictionRegistry.model_validate(altered)


def test_registered_control_features_use_strictly_prior_references() -> None:
    papers = pd.DataFrame(
        [
            {
                "paper_id": "F1",
                "publication_year": 2002,
                "domain12": "physics",
                "venue_family": "Nature",
            }
        ]
    )
    references = pd.DataFrame(
        [
            {"paper_id": "F1", "reference_id": "R_OLD"},
            {"paper_id": "F1", "reference_id": "R_SAME"},
            {"paper_id": "F1", "reference_id": "R_FUTURE"},
        ]
    )
    works = pd.DataFrame(
        [
            {"work_id": "R_OLD", "publication_year": 1997},
            {"work_id": "R_SAME", "publication_year": 2002},
            {"work_id": "R_FUTURE", "publication_year": 2003},
        ]
    )
    graph = pd.DataFrame(
        [{"paper_id": "F1", "prior_graph_degree_median": 4.0}]
    )

    output = build_registered_control_features(
        papers, references, works, prior_graph_features=graph
    ).iloc[0]

    assert output["log_reference_count"] == pytest.approx(np.log1p(3))
    assert output["reference_age_median"] == 5.0
    assert output["reference_year_coverage"] == pytest.approx(1 / 3)
    assert output["prior_graph_degree_median"] == 4.0
    assert output["source_max_year"] == 2001


def test_bibliographic_opportunity_uses_only_prior_papers() -> None:
    historical = pd.DataFrame(
        [
            {"work_id": "R1", "publication_year": 1990, "referenced_works": []},
            {"work_id": "R2", "publication_year": 1990, "referenced_works": []},
            {"work_id": "R3", "publication_year": 1990, "referenced_works": []},
            {
                "work_id": "P1",
                "publication_year": 2000,
                "referenced_works": ["R1", "R2"],
            },
            {
                "work_id": "P2",
                "publication_year": 2001,
                "referenced_works": ["R2", "R3"],
            },
            {
                "work_id": "P3",
                "publication_year": 2002,
                "referenced_works": ["R1", "R3"],
            },
            {
                "work_id": "P4",
                "publication_year": 2003,
                "referenced_works": ["R1", "R3"],
            },
        ]
    )
    papers = pd.DataFrame(
        [{"paper_id": "F1", "publication_year": 2002}]
    )
    references = pd.DataFrame(
        [
            {"paper_id": "F1", "reference_id": "R1"},
            {"paper_id": "F1", "reference_id": "R3"},
        ]
    )

    row = build_bibliographic_opportunity_features(
        papers,
        references,
        historical,
        compute_exact_clustering=True,
        compute_exact_closeness=True,
    ).iloc[0]

    assert row["eligible_prior_paper_count"] == 2
    assert row["bc_degree_per_reference_t0"] == 1.0
    assert row["bc_shared_reference_strength_t0"] == 2.0
    assert row["bc_component_share_t0"] == 1.0
    assert row["bc_local_clustering_t0"] == 1.0
    assert row["bc_harmonic_closeness_t0"] == 1.0
    assert row["source_max_year"] == 2001


def _diffusion_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "paper_id": "A",
                "horizon": 5,
                "future_uptake": 1,
                "future_field_reach": 1.0,
                "future_subfield_reach": 2.0,
                "future_topic_reach": 2.0,
                "future_field_simpson": 0.2,
                "future_topic_simpson": 0.3,
            },
            {
                "paper_id": "B",
                "horizon": 5,
                "future_uptake": 1,
                "future_field_reach": 3.0,
                "future_subfield_reach": 4.0,
                "future_topic_reach": 6.0,
                "future_field_simpson": 0.6,
                "future_topic_simpson": 0.8,
            },
            {
                "paper_id": "Z",
                "horizon": 5,
                "future_uptake": 0,
                "future_field_reach": 0.0,
                "future_subfield_reach": 0.0,
                "future_topic_reach": 0.0,
                "future_field_simpson": 0.0,
                "future_topic_simpson": 0.0,
            },
        ]
    )


def test_diffusion_target_transform_is_fold_local_and_zero_conditional() -> None:
    training = _diffusion_frame()
    transformer = FoldLocalDiffusionTargetTransformer().fit(training)
    baseline = transformer.transform(training)

    test = training.iloc[[1]].copy()
    test.loc[:, "future_field_reach"] = 10_000.0
    transformed_test = transformer.transform(test)
    repeated_training = transformer.transform(training)

    assert baseline["rgpm_d_fold"].equals(repeated_training["rgpm_d_fold"])
    assert np.isfinite(transformed_test["rgpm_d_fold"].iloc[0])
    assert np.isnan(baseline.loc[2, "rgpm_d_fold"])
    assert baseline["target_transform_scope"].eq("training_fold_only").all()


def test_diffusion_target_transform_rejects_horizon_mixing() -> None:
    transformer = FoldLocalDiffusionTargetTransformer().fit(
        _diffusion_frame()
    )
    mixed = _diffusion_frame()
    mixed.loc[0, "horizon"] = 3
    with pytest.raises(ValueError, match="horizon differs"):
        transformer.transform(mixed)

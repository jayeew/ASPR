from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from aspr.nature_multihorizon.cohorts import build_cohort_membership
from aspr.nature_multihorizon.contracts import CohortSpec, HorizonSpec
from aspr.nature_multihorizon.contracts_v6 import (
    EvidenceValue,
    InfluenceForecast,
    InnovationEvidenceProfile,
)
from aspr.nature_multihorizon.evidence_registry import (
    PROMOTION_GATE_IDS,
    audit_registry_implementations,
    load_evidence_registry,
)
from aspr.nature_multihorizon.features_v6 import (
    canonical_pair,
    cosine_distance_profiles,
    field_disparity_mean,
    field_pielou_evenness,
    field_variety,
    first_time_source_pair_distance_mean,
    first_time_source_pair_share,
    marginal_pair_z_scores,
    novelty_u,
    rao_stirling_integration,
    sva_centrality_divergence,
    sva_cluster_linkage,
    sva_modularity_change_rate,
    uzzi_atypicality_p10,
    uzzi_conventionality_median,
)
from aspr.nature_multihorizon.targets import build_diffusion_targets_from_deltas
from aspr.nature_multihorizon.promotion_v6 import (
    PromotionGateEvidence,
    PromotionStatus,
    build_promotion_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = PROJECT_ROOT / "configs" / "innovation_registry_v6_local.json"


def test_v6_registry_is_source_complete_and_implementation_resolvable() -> None:
    registry = load_evidence_registry(REGISTRY_PATH)

    assert registry.network_policy == "forbidden"
    assert registry.raw_data_policy == "local_frozen_only"
    assert set(registry.dimensions) == {
        "N1_RECOMBINATION",
        "C1_KNOWLEDGE_DIVERSITY",
        "S1_STRUCTURAL_VARIATION",
        "N2_SEMANTIC_NOVELTY",
    }
    assert registry.metrics["C1.DISPARITY"].dimension_id == (
        "C1_KNOWLEDGE_DIVERSITY"
    )
    assert registry.metrics["C1.RAO"].dimension_id == (
        "C1_KNOWLEDGE_DIVERSITY"
    )
    assert registry.registry_stage == "definition_preregistered"
    assert all(source.doi or source.url for source in registry.sources.values())
    assert "proxy_modularity_drop_legacy" not in registry.primary_feature_names
    assert audit_registry_implementations(registry)["overall_pass"]


def test_v6_registry_rejects_semantic_laundering() -> None:
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    altered = copy.deepcopy(payload)
    metric = altered["metrics"]["N1.U"]
    metric["status"] = "confirmatory"
    metric["fidelity"] = "project_proxy"
    metric["admission_checks"]["I9"] = True

    with pytest.raises(ValidationError):
        load_evidence_registry_from_payload(altered)


def _passing_promotion_gates() -> dict[str, PromotionGateEvidence]:
    return {
        gate_id: PromotionGateEvidence(
            gate_id=gate_id,
            passed=True,
            evidence_artifact_ids=(f"artifact:{gate_id.lower()}",),
            detail="registered test evidence passed",
        )
        for gate_id in PROMOTION_GATE_IDS
    }


def test_runtime_promotion_is_separate_and_fail_closed() -> None:
    registry = load_evidence_registry(REGISTRY_PATH)
    held = build_promotion_report(
        registry,
        {},
        report_id="promotion-held",
        evaluated_artifact_id="artifact:quality-audit",
    )
    assert (
        held.decisions["C1.VARIETY"].promotion_status
        is PromotionStatus.HELD
    )
    assert (
        held.decisions["N1.U"].promotion_status
        is PromotionStatus.NOT_ELIGIBLE
    )

    promoted = build_promotion_report(
        registry,
        {"C1.VARIETY": _passing_promotion_gates()},
        report_id="promotion-one",
        evaluated_artifact_id="artifact:quality-audit",
    )
    assert (
        promoted.decisions["C1.VARIETY"].promotion_status
        is PromotionStatus.PROMOTED
    )
    assert "C1.VARIETY" in promoted.promoted_entity_ids


def test_promotion_after_sealed_holdout_inspection_is_rejected() -> None:
    registry = load_evidence_registry(REGISTRY_PATH)
    with pytest.raises(ValidationError, match="before sealed-holdout"):
        build_promotion_report(
            registry,
            {"C1.VARIETY": _passing_promotion_gates()},
            report_id="promotion-invalid",
            evaluated_artifact_id="artifact:quality-audit",
            sealed_holdout_inspected=True,
        )


def load_evidence_registry_from_payload(payload: dict) -> object:
    from aspr.nature_multihorizon.evidence_registry import EvidenceRegistry

    return EvidenceRegistry.model_validate(payload)


def test_profile_rejects_same_year_or_future_sources() -> None:
    valid = EvidenceValue(
        metric_id="C1.VARIETY",
        value=2.0,
        source_max_year=2019,
        artifact_id="artifact:features-v6",
    )
    profile = InnovationEvidenceProfile(
        paper_id="W1",
        publication_year=2020,
        domain12="chemistry",
        evidence={"C1.VARIETY": valid},
        registry_version="registry-v6",
    )
    assert profile.evidence["C1.VARIETY"].value == 2.0

    leaked = EvidenceValue(
        metric_id="C1.VARIETY",
        value=2.0,
        source_max_year=2020,
        artifact_id="artifact:features-v6",
    )
    with pytest.raises(ValidationError, match="strictly before"):
        InnovationEvidenceProfile(
            paper_id="W1",
            publication_year=2020,
            domain12="chemistry",
            evidence={"C1.VARIETY": leaked},
            registry_version="registry-v6",
        )


def test_influence_forecast_is_separate_and_horizon_locked() -> None:
    forecast = InfluenceForecast(
        paper_id="W1",
        horizon=5,
        uptake_probability=0.4,
        diffusion_score_if_uptake=0.7,
        expected_diffusion_score=0.28,
        prediction_interval_low=0.1,
        prediction_interval_high=0.6,
        model_version="model-v6",
        calibration_version="cal-v6",
        feature_artifact_id="artifact:features-v6",
    )
    assert "not an innovation score" in forecast.claim_scope
    with pytest.raises(ValidationError):
        InfluenceForecast(
            paper_id="W1",
            horizon=4,
            uptake_probability=0.4,
            expected_diffusion_score=0.28,
            model_version="model-v6",
            calibration_version="cal-v6",
            feature_artifact_id="artifact:features-v6",
        )


def test_diversity_metrics_hand_calculation() -> None:
    labels = ["A", "A", "B", "C"]
    distances = {
        canonical_pair("A", "B"): 0.2,
        canonical_pair("A", "C"): 0.6,
        canonical_pair("B", "C"): 0.4,
    }

    assert field_variety(labels) == 3.0
    expected_evenness = -(
        0.5 * math.log(0.5)
        + 0.25 * math.log(0.25)
        + 0.25 * math.log(0.25)
    ) / math.log(3.0)
    assert field_pielou_evenness(labels) == pytest.approx(expected_evenness)
    assert field_disparity_mean(labels, distances) == pytest.approx(0.4)
    assert rao_stirling_integration(labels, distances) == pytest.approx(0.25)
    assert math.isnan(field_pielou_evenness(["A", "A"]))
    assert rao_stirling_integration(["A", "A"], {}) == 0.0


def test_cosine_distance_profiles() -> None:
    distances = cosine_distance_profiles(
        {"A": [1.0, 0.0], "B": [0.0, 1.0], "C": [1.0, 1.0]}
    )

    assert distances[canonical_pair("A", "B")] == pytest.approx(1.0)
    assert distances[canonical_pair("A", "C")] == pytest.approx(
        1.0 - 1.0 / math.sqrt(2.0)
    )


def test_novelty_u_hand_calculation() -> None:
    sources = ["A", "B", "C"]
    pair_counts = {
        canonical_pair("A", "B"): 2,
        canonical_pair("B", "C"): 1,
    }
    source_counts = {"A": 4, "B": 5, "C": 2}
    commonness = [1.0, 0.625, 1.0]
    expected = -math.log(float(np.quantile(commonness, 0.10)))

    assert novelty_u(sources, pair_counts, source_counts, 10) == pytest.approx(
        expected
    )


def test_first_time_pair_metrics_hand_calculation() -> None:
    sources = ["A", "B", "C"]
    pair_counts = {
        canonical_pair("A", "B"): 2,
        canonical_pair("B", "C"): 1,
    }
    distances = {
        canonical_pair("A", "B"): 0.1,
        canonical_pair("A", "C"): 0.8,
        canonical_pair("B", "C"): 0.3,
    }

    assert first_time_source_pair_share(sources, pair_counts) == pytest.approx(1 / 3)
    assert first_time_source_pair_distance_mean(
        sources, pair_counts, distances
    ) == pytest.approx(0.8)


def test_uzzi_quantiles_hand_calculation() -> None:
    values = [-2.0, 0.0, 1.0, 3.0]

    assert uzzi_atypicality_p10(values) == pytest.approx(
        -float(np.quantile(values, 0.10))
    )
    assert uzzi_conventionality_median(values) == pytest.approx(0.5)


def test_marginal_pair_z_scores_hand_calculation() -> None:
    sources = ["A", "B", "C"]
    pair_counts = {
        canonical_pair("A", "B"): 2,
        canonical_pair("B", "C"): 1,
    }
    source_counts = {"A": 4, "B": 5, "C": 2}
    scores = marginal_pair_z_scores(
        sources, pair_counts, source_counts, n_historical_papers=10
    )
    expected_ab = 0.0
    expected_ac = (0.0 - 0.8) / math.sqrt(2 * 0.4 * 0.6 * (8 / 9))
    expected_bc = (1.0 - 1.0) / math.sqrt(2 * 0.5 * 0.5 * (8 / 9))
    assert scores == pytest.approx([expected_ab, expected_ac, expected_bc])


def test_sva_metrics_hand_calculation() -> None:
    baseline = nx.Graph()
    baseline.add_edges_from([(0, 1), (2, 3), (1, 2)])
    nx.set_edge_attributes(baseline, 1.0, "weight")
    augmented = baseline.copy()
    augmented.add_edges_from([(0, 2), (1, 3)], weight=1.0)
    communities = [{0, 1}, {2, 3}]

    baseline_q = nx.community.modularity(baseline, communities, weight="weight")
    augmented_q = nx.community.modularity(augmented, communities, weight="weight")
    expected_mcr = (baseline_q - augmented_q) / baseline_q
    assert sva_modularity_change_rate(
        baseline, augmented, communities
    ) == pytest.approx(expected_mcr)
    assert sva_cluster_linkage(
        baseline,
        augmented,
        communities,
        contributing_references=2,
        total_references=4,
    ) == pytest.approx(100.0)
    divergence = sva_centrality_divergence(baseline, augmented)
    assert math.isfinite(divergence)
    assert divergence >= 0.0


def test_zero_future_citers_are_valid_v6_outcomes_and_cohort_members() -> None:
    papers = pd.DataFrame(
        [
            {
                "paper_id": "W0",
                "publication_year": 2010,
                "domain12": "chemistry",
                "work_type": "article",
            },
            {
                "paper_id": "W1",
                "publication_year": 2010,
                "domain12": "chemistry",
                "work_type": "article",
            },
            {
                "paper_id": "WF",
                "publication_year": 2010,
                "domain12": "chemistry",
                "work_type": "article",
            },
        ]
    )
    rows = []
    for paper_id, fetch_valid, count in (("W0", 1, 0.0), ("W1", 1, 3.0), ("WF", 0, np.nan)):
        valid_value = 0.0 if paper_id == "W0" else 1.0
        rows.append(
            {
                "paper_id": paper_id,
                "publication_year": 2010,
                "horizon": 5,
                "fetch_status": "zero_success"
                if paper_id == "W0"
                else ("success" if paper_id == "W1" else "failed"),
                "fetch_valid": fetch_valid,
                "cap_hit": 0,
                "requested_horizon_cap_hit": 0,
                "n_future_citers": count,
                "future_field_coverage": 1.0 if fetch_valid else np.nan,
                "future_subfield_coverage": 1.0 if fetch_valid else np.nan,
                "future_topic_coverage": 1.0 if fetch_valid else np.nan,
                "future_field_reach": valid_value if fetch_valid else np.nan,
                "future_subfield_reach": valid_value if fetch_valid else np.nan,
                "future_topic_reach": valid_value if fetch_valid else np.nan,
                "future_field_simpson": valid_value if fetch_valid else np.nan,
                "future_topic_simpson": valid_value if fetch_valid else np.nan,
            }
        )
    targets = build_diffusion_targets_from_deltas(
        papers, pd.DataFrame(rows), horizons=(5,), min_future_citers=0
    )
    target_lookup = targets.set_index("paper_id")
    assert target_lookup.loc["https://openalex.org/W0", "target_rank_eligible"] == 1
    assert target_lookup.loc["https://openalex.org/W0", "future_uptake"] == 0.0
    assert target_lookup.loc["https://openalex.org/W1", "future_uptake"] == 1.0
    assert target_lookup.loc["https://openalex.org/WF", "target_rank_eligible"] == 0

    features = pd.DataFrame(
        [
            {
                "paper_id": f"https://openalex.org/{paper_id}",
                "valid_reference_count": 10,
                "reference_metadata_coverage": 1.0,
                "field_variety": 2.0,
            }
            for paper_id in ("W0", "W1", "WF")
        ]
    )
    membership = build_cohort_membership(
        papers.assign(
            paper_id=papers["paper_id"].map(lambda value: f"https://openalex.org/{value}")
        ),
        features,
        targets,
        spec=CohortSpec(horizons=(5,), primary_horizon=5, min_future_citers=0),
        required_feature_names=("field_variety",),
        complete_end_year=2020,
    ).set_index("paper_id")
    assert membership.loc["https://openalex.org/W0", "cohort_member"] == 1
    assert membership.loc["https://openalex.org/W0", "observed_zero_future_citers"] == 1
    assert membership.loc["https://openalex.org/WF", "cohort_member"] == 0


def test_horizon_and_cohort_contracts_allow_preregistered_zero_threshold() -> None:
    horizon = HorizonSpec(
        tau=5,
        complete_publication_end_year=2020,
        development_end_year=2016,
        sealed_test_start_year=2017,
        sealed_test_end_year=2020,
        min_future_citers=0,
        target_name="RGPM-D5-v6",
    )
    assert horizon.min_future_citers == 0
    assert CohortSpec(min_future_citers=0).min_future_citers == 0

from __future__ import annotations

import pandas as pd

from aspr.nature_multihorizon.construct_validation_v6 import (
    C1_METRICS,
    N1_METRICS,
    metric_correlation_audit,
    select_stability_sample,
)


def test_stability_sample_is_outcome_blind_deterministic_and_stratified() -> None:
    papers = pd.DataFrame(
        [
            {
                "paper_id": f"P{index}",
                "publication_year": 2000 + index % 2,
                "domain12": "physics" if index < 6 else "chemistry",
            }
            for index in range(12)
        ]
    )
    features = pd.DataFrame(
        [
            {
                "paper_id": f"P{index}",
                "field_variety": 3.0,
                "field_pielou_evenness": 0.7,
                "field_disparity_cosine_mean": 0.4,
                "rao_stirling_integration": 0.3,
                "valid_reference_count": 20,
                "field_mapping_coverage": 0.95,
            }
            for index in range(12)
        ]
    )
    first = select_stability_sample(
        papers,
        features,
        max_per_stratum=2,
        salt="test",
        min_valid_references=10,
        min_field_mapping_coverage=0.9,
    )
    second = select_stability_sample(
        papers,
        features,
        max_per_stratum=2,
        salt="test",
        min_valid_references=10,
        min_field_mapping_coverage=0.9,
    )

    assert first["paper_id"].tolist() == second["paper_id"].tolist()
    assert first.groupby(["domain12", "publication_era_5y"]).size().le(2).all()
    assert "future_uptake" not in first


def test_metric_correlation_audit_covers_every_registered_pair() -> None:
    metrics = (*N1_METRICS, *C1_METRICS)
    frame = pd.DataFrame(
        {
            metric: [float(index), float(index + 1), float(index + 2)]
            for index, metric in enumerate(metrics)
        }
    )

    audit = metric_correlation_audit(frame)

    assert len(audit) == len(metrics) * (len(metrics) - 1) // 2
    assert audit["n_paired"].eq(3).all()

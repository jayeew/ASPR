from __future__ import annotations

import json

import numpy as np
import pandas as pd

from gear.nature_multihorizon.feature_materializer_v6 import (
    annual_field_distances,
    build_v6_reference_feature_table,
)


def _fixtures() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    papers = pd.DataFrame(
        [
            {"paper_id": "P0", "publication_year": 2000, "domain12": "chemistry"},
            {"paper_id": "P1", "publication_year": 2001, "domain12": "chemistry"},
            {"paper_id": "P1B", "publication_year": 2001, "domain12": "chemistry"},
            {"paper_id": "P2", "publication_year": 2002, "domain12": "chemistry"},
        ]
    )
    references = {
        "P0": ("R1", "R2"),
        "P1": ("R1", "R3"),
        "P1B": ("R2", "R3"),
        "P2": ("R1", "R2", "R3", "R4"),
    }
    paper_references = pd.DataFrame(
        [
            {"paper_id": paper_id, "reference_id": reference_id}
            for paper_id, reference_ids in references.items()
            for reference_id in reference_ids
        ]
    )
    work_view = pd.DataFrame(
        [
            {
                "work_id": f"R{index}",
                "publication_year": 1980 + index,
                "source_id": f"S{index}",
                "field_id": f"F{min(index, 3)}",
                "referenced_works": [],
            }
            for index in range(1, 5)
        ]
    )
    events = pd.DataFrame(
        [
            {
                "source_year": year,
                "source_field_id": source,
                "target_field_id": target,
            }
            for year in (1998, 1999, 2000, 2001)
            for source, target in (
                ("F1", "F1"),
                ("F1", "F2"),
                ("F2", "F2"),
                ("F2", "F3"),
                ("F3", "F1"),
                ("F3", "F3"),
            )
        ]
    )
    return papers, paper_references, work_view, events


def test_annual_field_distances_use_strictly_prior_window() -> None:
    _, _, _, events = _fixtures()
    distances = annual_field_distances(events, [2000, 2001], window_years=2)

    assert distances[2000]
    assert distances[2001]
    events_with_same_year_change = pd.concat(
        [
            events,
            pd.DataFrame(
                [
                    {
                        "source_year": 2001,
                        "source_field_id": "F1",
                        "target_field_id": "F3",
                    }
                    for _ in range(100)
                ]
            ),
        ],
        ignore_index=True,
    )
    unchanged = annual_field_distances(
        events_with_same_year_change, [2001], window_years=2
    )
    assert unchanged[2001] == distances[2001]


def test_reference_feature_materializer_excludes_same_year_history() -> None:
    papers, paper_references, work_view, events = _fixtures()
    features = build_v6_reference_feature_table(
        papers,
        paper_references,
        work_view,
        field_citation_events=events,
        field_profile_window_years=5,
    ).set_index("paper_id")

    p0 = features.loc["https://openalex.org/P0"]
    p1 = features.loc["https://openalex.org/P1"]
    p1b = features.loc["https://openalex.org/P1B"]
    p2 = features.loc["https://openalex.org/P2"]
    assert p0["n_historical_source_papers"] == 0
    assert p1["n_historical_source_papers"] == 1
    assert p1b["n_historical_source_papers"] == 1
    assert p1["first_time_source_pair_share"] == 1.0
    assert p1b["first_time_source_pair_share"] == 1.0
    assert p2["n_historical_source_papers"] == 3
    assert p2["first_time_source_pair_share"] == 0.5
    assert np.isfinite(p2["novelty_u_t0_source"])
    assert p2["field_variety"] == 3.0
    assert np.isfinite(p2["rao_stirling_integration"])
    assert p2["source_max_year"] == 2001
    assert (features["source_max_year"] < features["publication_year"]).all()
    assert json.loads(p2["quality_flags"]) == [
        "fewer_than_20_valid_uzzi_pairs"
    ]


def test_reference_feature_materializer_rejects_same_year_references() -> None:
    papers, paper_references, work_view, events = _fixtures()
    work_view.loc[work_view["work_id"].eq("R4"), "publication_year"] = 2002
    features = build_v6_reference_feature_table(
        papers,
        paper_references,
        work_view,
        field_citation_events=events,
    ).set_index("paper_id")

    p2 = features.loc["https://openalex.org/P2"]
    assert p2["valid_reference_count"] == 3
    assert p2["reference_metadata_coverage"] == 0.75

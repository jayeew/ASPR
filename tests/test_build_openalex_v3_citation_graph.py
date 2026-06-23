from __future__ import annotations

import pandas as pd

from scripts import build_openalex_v3_citation_graph as mod


def test_select_records_prioritizes_exact_landmarks() -> None:
    records = [
        {
            "domain": "crispr",
            "source_kind": "query_core",
            "work": {"id": "https://openalex.org/W2", "cited_by_count": 999, "publication_year": 2015},
        },
        {
            "domain": "crispr",
            "source_kind": "landmark_exact",
            "anchor_label": "Jinek 2012",
            "work": {"id": "https://openalex.org/W1", "cited_by_count": 1, "publication_year": 2012},
        },
    ]

    selected = mod.select_records(records, max_records=1)

    assert len(selected) == 1
    assert selected[0]["source_kind"] == "landmark_exact"
    assert selected[0]["anchor_label"] == "Jinek 2012"


def test_citation_rows_can_preserve_external_reference_targets() -> None:
    records = [
        {
            "domain": "crispr",
            "source_kind": "query_core",
            "work": {
                "id": "https://openalex.org/W2",
                "referenced_works": ["https://openalex.org/W1", "https://openalex.org/WOUTSIDE"],
            },
        }
    ]

    rows = mod.citation_rows_from_records(
        records,
        selected_ids={"https://openalex.org/W1", "https://openalex.org/W2"},
        local_references_only=False,
    )
    local_rows = mod.citation_rows_from_records(
        records,
        selected_ids={"https://openalex.org/W1", "https://openalex.org/W2"},
        local_references_only=True,
    )

    assert {row["target"] for row in rows} == {"https://openalex.org/W1", "https://openalex.org/WOUTSIDE"}
    assert {row["target"] for row in local_rows} == {"https://openalex.org/W1"}


def test_standardize_landmarks_uses_strict_manual_v3_source() -> None:
    registry = pd.DataFrame(
        [
            {
                "domain": "crispr",
                "label": "Jinek 2012",
                "openalex_id": "https://openalex.org/W2045435533",
                "doi": "10.1126/science.1225829",
                "title": "A programmable dual-RNA-guided DNA endonuclease in adaptive bacterial immunity",
                "year": 2012,
                "title_similarity": 1.0,
            }
        ]
    )

    landmarks = mod.standardize_landmarks(registry)

    assert landmarks["landmark_source"].tolist() == ["strict_manual_v3"]
    assert landmarks["accepted_landmark_source"].tolist() == ["strict_manual_v3"]
    assert landmarks["include_main"].tolist() == [1]

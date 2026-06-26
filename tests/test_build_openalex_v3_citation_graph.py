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


def test_v4_labels_are_parameterized_without_changing_row_contract() -> None:
    record = {
        "domain": "crispr",
        "source_kind": "landmark_exact",
        "work": {
            "id": "https://openalex.org/W2045435533",
            "doi": "https://doi.org/10.1126/science.1225829",
            "display_name": "A programmable dual-RNA-guided DNA endonuclease in adaptive bacterial immunity",
            "publication_year": 2012,
            "type": "article",
            "cited_by_count": 10,
            "referenced_works": ["https://openalex.org/W1"],
            "primary_topic": {"id": "https://openalex.org/T1", "display_name": "CRISPR"},
        },
    }

    row = mod.work_to_row(
        record,
        meta={"display_name": "CRISPR", "field_name": "Biology and Medicine"},
        landmark_labels={"https://openalex.org/W2045435533": "Jinek 2012"},
        fetched_at="2026-01-01T00:00:00+00:00",
        landmark_source_label="strict_manual_v4",
        source_dataset_prefix="openalex_v4_screen",
    )
    citations = mod.citation_rows_from_records(
        [record],
        selected_ids={"https://openalex.org/W2045435533", "https://openalex.org/W1"},
        local_references_only=True,
        source_dataset_prefix="openalex_v4_screen",
    )
    landmarks = mod.standardize_landmarks(
        pd.DataFrame(
            [
                {
                    "domain": "crispr",
                    "label": "Jinek 2012",
                    "openalex_id": "https://openalex.org/W2045435533",
                    "doi": "10.1126/science.1225829",
                    "title": row["title"],
                    "year": 2012,
                    "title_similarity": 1.0,
                }
            ]
        ),
        landmark_source_label="strict_manual_v4",
    )

    assert row["source_dataset"] == "openalex_v4_screen_landmark_exact"
    assert row["reliable_anchor_source"] == "strict_manual_v4"
    assert citations[0]["source_dataset"] == "openalex_v4_screen_landmark_exact"
    assert landmarks["landmark_source"].tolist() == ["strict_manual_v4"]
    assert landmarks["accepted_landmark_source"].tolist() == ["strict_manual_v4"]


def test_global_deduplicate_works_preserves_landmark_domain_and_filters_sources() -> None:
    works = pd.DataFrame(
        [
            {"id": "https://openalex.org/W1", "domain": "domain_a", "is_landmark": 0, "cited_by_count": 100, "year": 2010},
            {"id": "https://openalex.org/W1", "domain": "domain_b", "is_landmark": 1, "cited_by_count": 1, "year": 2010},
            {"id": "https://openalex.org/W2", "domain": "domain_a", "is_landmark": 0, "cited_by_count": 5, "year": 2011},
        ]
    )
    citations = pd.DataFrame(
        [
            {"source": "https://openalex.org/W1", "target": "https://openalex.org/W0", "relation": "reference"},
            {"source": "https://openalex.org/W3", "target": "https://openalex.org/W0", "relation": "reference"},
        ]
    )

    deduped, kept_citations, report = mod.global_deduplicate_works(works, citations, ["domain_a", "domain_b"])

    assert report == {"dropped_duplicate_works": 1}
    assert deduped[deduped["id"] == "https://openalex.org/W1"]["domain"].tolist() == ["domain_b"]
    assert set(kept_citations["source"]) == {"https://openalex.org/W1"}

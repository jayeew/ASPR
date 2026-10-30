"""Country attribution, missingness and preserved-cloud regression tests."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from figure_pipeline.fig7_reference import country_data as cd


def paper(**kwargs: Any) -> dict[str, Any]:
    return {
        "paper_id": "p1",
        "doi": "10.1038/test",
        "title": "A test of molecular mechanisms",
        "year": 2024,
        "cohort": "historical",
        **kwargs,
    }


def record(codes: list[list[str]]) -> dict[str, Any]:
    return {
        "doi": "https://doi.org/10.1038/test",
        "title": "A test of molecular mechanisms",
        "publication_year": 2024,
        "authors_count": len(codes),
        "countries_distinct_count": len({c for cs in codes for c in cs}),
        "authorships": [
            {"countries": cs, "institutions": [{"country_code": c} for c in cs]}
            for cs in codes
        ],
    }


def test_single_country_multiple_authors_and_institutions_count_once() -> None:
    result = cd.assess_record(paper(), record([["US", "US"], ["US"]]))
    assert result["status"] == "complete"
    assert cd.weights(result) == {"US": 1.0}


def test_multicountry_weights_do_not_follow_author_counts() -> None:
    result = cd.assess_record(paper(), record([["US"], ["US"], ["GB"]]))
    assert cd.weights(result) == {"GB": 0.5, "US": 0.5}


def test_partial_authorship_does_not_reallocate_weight() -> None:
    result = cd.assess_record(paper(), record([["US"], []]))
    assert result["status"] == "partial"
    assert cd.weights(result) == {}


@pytest.mark.parametrize(
    "change,issue",
    [
        ({"doi": "https://doi.org/10.1038/other"}, "doi_mismatch"),
        ({"publication_year": 2025}, "year_mismatch"),
        ({"title": "A wholly unrelated object"}, "title_mismatch"),
        ({"countries_distinct_count": 2}, "country_count_mismatch"),
    ],
)
def test_metadata_identity_and_count_conflicts(
    change: dict[str, Any], issue: str
) -> None:
    result = cd.assess_record(paper(), record([["US"]]) | change)
    assert issue in result["issues"]
    assert result["status"] == "conflict"
    assert not cd.weights(result)


def test_100_author_list_requires_singleton_even_without_truncation_flag() -> None:
    source = record([["US"]] * 100)
    assert "truncated_authorships" in cd.assess_record(paper(), source)["issues"]
    assert (
        cd.assess_record(paper(), source | {"_singleton_checked": True})["status"]
        == "complete"
    )
    source["authors_count"] = 101
    assert (
        cd.assess_record(paper(), source | {"_singleton_checked": True})["status"]
        == "partial"
    )


def test_country_conflict_is_not_silently_unioned() -> None:
    source = record([["US"]])
    source["authorships"][0]["institutions"] = [{"country_code": "CA"}]
    result = cd.assess_record(paper(), source)
    assert "author_country_conflict" in result["issues"]
    assert not cd.weights(result)


@pytest.mark.parametrize(
    "address,expected",
    [
        ("Department X, University Y, Shanghai, China", "CN"),
        ("Department X, Hong Kong, China", "HK"),
        ("Department X, Taipei, Taiwan", "TW"),
        ("University of China, New York", None),
        ("University Y, Pasadena, USA", "US"),
        ("A study finds China important", None),
    ],
)
def test_explicit_address_suffix_only(address: str, expected: str | None) -> None:
    assert cd.affiliation_country(address) == expected


def test_local_address_repair_has_exact_original_span(tmp_path: Path) -> None:
    path = tmp_path / "paper.md"
    path.write_text(
        "1 Department of Science, Example University,\nShanghai, China.\n",
        encoding="utf-8",
    )
    source = record([[]])
    source["authorships"][0]["raw_affiliation_strings"] = [
        "Department of Science, Example University, Shanghai, China"
    ]
    original = copy.deepcopy(source)
    result, repairs = cd.repair_affiliations(paper(paper_path=str(path)), source)
    assert result["authorships"][0]["countries"] == ["CN"]
    match = repairs[0]["matches"][0]
    assert (
        path.read_text()[match["local_start"] : match["local_end"]]
        == match["local_quote"]
    )
    assert source == original
    path.write_text(
        "This study mentions China but has no matching institutional address."
    )
    assert cd.repair_affiliations(paper(paper_path=str(path)), source)[1] == []


def test_local_repair_explains_incomplete_api_country_counter(tmp_path: Path) -> None:
    path = tmp_path / "paper.md"
    path.write_text("1 Independent Researcher, Moscow, Russia.", encoding="utf-8")
    source = record([["US"], []])
    source["authorships"][1]["raw_affiliation_strings"] = [
        "Independent Researcher, Moscow, Russia. author@example.org"
    ]
    updated, repairs = cd.repair_affiliations(paper(paper_path=str(path)), source)
    result = cd.assess_record(paper(), updated)
    assert repairs and result["status"] == "complete"
    assert result["country_count_updated_from_verified_local_addresses"]
    assert cd.weights(result) == {"RU": 0.5, "US": 0.5}
    # A pre-existing discrepancy not explained by the verified address stays unresolved.
    updated["countries_distinct_count"] = 3
    assert cd.assess_record(paper(), updated)["status"] == "conflict"


def test_historical_preference_denominator_includes_nontopic_papers(
    tmp_path: Path,
) -> None:
    rows = [
        paper(paper_id="p1", status="complete", country_codes=["US", "GB"]),
        paper(paper_id="p2", status="complete", country_codes=["US"]),
        paper(paper_id="p3", status="partial", country_codes=["GB"]),
    ]
    topics = [
        {"display_label": "Topic", "phrase": "topic", "source_paper_ids": ["p1", "p3"]}
    ]
    _, result = cd.historical_statistics(rows, topics, tmp_path)
    us = next(r for r in result["topic_cells"] if r["country_code"] == "US")
    gb = next(r for r in result["topic_cells"] if r["country_code"] == "GB")
    assert us["corpus_denominator"] == 2 and us["topic_denominator"] == 1
    assert math.isclose(us["relative_preference"], 2 / 3)
    assert gb["relative_preference"] == 2
    assert us["low_sample"]


def test_future_deduplicates_papers_and_preserves_unknown_mass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "base"
    cd.write(
        base / "data/directions.json",
        [
            {
                "candidate_id": "d1",
                "citations": [
                    {"source_id": "a"},
                    {"source_id": "b"},
                    {"source_id": "c"},
                ],
            }
        ],
    )
    cd.write(
        base / "data/evidence.json",
        {"a": {"paper_id": "p1"}, "b": {"paper_id": "p1"}, "c": {"paper_id": "p2"}},
    )
    monkeypatch.setattr(cd, "BASE", base)
    rows = [
        paper(paper_id="p1", status="complete", country_codes=["US", "GB"]),
        paper(
            paper_id="p2",
            status="partial",
            country_codes=["US"],
            cohort="forecast_target",
        ),
    ]
    topics = [
        {
            "display_label": "Future",
            "direction_ids": ["d1"],
            "source_paper_ids": ["p1", "p2"],
        }
    ]
    result = cd.future_statistics(rows, topics, ["US"], tmp_path)
    shares = {r["country_code"]: r["proportion"] for r in result}
    assert shares == {"US": 0.25, "Other": 0.25, "Unresolved": 0.5}
    assert all(r["source_papers"] == 2 for r in result)


def test_type_profiles_normalize_papers_before_country_average(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pd.DataFrame(
        {
            "parent_paper_id": ["p1", "p2", "p2", "p2"],
            "claim_type": ["METHOD", "FINDING", "FINDING", "FINDING"],
        }
    ).to_parquet(tmp_path / "claim_nodes.parquet")
    monkeypatch.setattr(cd, "ASSETS", tmp_path)
    rows = [
        paper(paper_id=p, status="complete", country_codes=["US"]) for p in ["p1", "p2"]
    ]
    result = cd.type_statistics(rows, tmp_path)
    proportions = {r["claim_type"]: r["proportion"] for r in result}
    assert proportions["METHOD"] == proportions["FINDING"] == 0.5
    assert sum(proportions.values()) == 1


def test_future_selection_deduplicates_direction_sets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        {
            "displayed": True,
            "display_label": label,
            "prominence": score,
            "direction_ids": ids,
        }
        for label, score, ids in [
            ("Alpha", 3, ["d1"]),
            ("Beta", 3, ["d1"]),
            ("Gamma", 2, ["d2"]),
        ]
    ]
    monkeypatch.setattr(cd, "read", lambda p: records if "forecast" in p.name else [])
    _, selected = cd.selected_topics()
    assert [r["display_label"] for r in selected] == ["Alpha", "Gamma"]


def test_independent_reconciliation_keeps_namibia_code_na(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from figure_pipeline.fig7_reference import country_checks as checks

    base, assets = tmp_path / "base", tmp_path / "assets"
    assets.mkdir()
    rows = [
        paper(paper_id="p1", status="complete", country_codes=["NA"]),
        paper(paper_id="p2", status="complete", country_codes=["US"]),
    ]
    history = [
        {
            "displayed": True,
            "display_label": "Topic",
            "phrase": "topic",
            "frequency": 1,
            "source_paper_ids": ["p1"],
        }
    ]
    future = [
        {
            "displayed": True,
            "display_label": "Future",
            "prominence": 1,
            "direction_ids": ["d1"],
            "source_paper_ids": ["p1", "p2"],
        }
    ]
    cd.write(base / "data/historical_wordcloud.json", history)
    cd.write(base / "data/forecast_wordcloud.json", future)
    cd.write(
        base / "data/directions.json",
        [{"candidate_id": "d1", "citations": [{"source_id": "a"}, {"source_id": "b"}]}],
    )
    cd.write(
        base / "data/evidence.json", {"a": {"paper_id": "p1"}, "b": {"paper_id": "p2"}}
    )
    pd.DataFrame(
        {"parent_paper_id": ["p1", "p2"], "claim_type": ["METHOD", "FINDING"]}
    ).to_parquet(assets / "claim_nodes.parquet")
    for module in [cd, checks]:
        monkeypatch.setattr(module, "ASSETS", assets)
        monkeypatch.setattr(module, "BASE", base)
    cd.write(tmp_path / "data/papers.json", rows)
    cd.write(tmp_path / "data/paper_countries.json", rows)
    top, _ = cd.historical_statistics(rows, history, tmp_path)
    cd.type_statistics(rows, tmp_path)
    cd.future_statistics(rows, future, top, tmp_path)
    cd.write(
        tmp_path / "data/plot_data.json",
        {"top_countries": top, "historical_topics": history, "future_topics": future},
    )
    result = checks.verify(tmp_path)
    assert result["countries"] == 2
    assert result["independent_country_reconciliation"]

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.fetch_nature_supplemental_horizons_v5 import (
    _scan_snapshot_file,
    parse_cohort_specs,
    read_cohort_targets,
)


def test_parse_cohort_specs_merges_repeated_years() -> None:
    specs = parse_cohort_specs(
        [
            "2018:3,5",
            "2019:3,5",
            "2020:3,5",
            "2021:3",
            "2022:3",
            "2023:2",
            "2023:1",
            "2024:1",
        ],
        complete_end_year=2025,
    )

    assert specs == {
        2018: (3, 5),
        2019: (3, 5),
        2020: (3, 5),
        2021: (3,),
        2022: (3,),
        2023: (1, 2),
        2024: (1,),
    }


def test_parse_cohort_specs_rejects_incomplete_window() -> None:
    with pytest.raises(ValueError, match="Incomplete cohort"):
        parse_cohort_specs(["2024:2"], complete_end_year=2025)


def test_read_cohort_targets_preserves_requested_years(tmp_path: Path) -> None:
    path = tmp_path / "targets.csv"
    pd.DataFrame(
        [
            {"id": "https://openalex.org/W1", "short_id": "W1", "year": 2021},
            {"id": "https://openalex.org/W2", "short_id": "W2", "year": 2022},
            {"id": "https://openalex.org/W3", "short_id": "W3", "year": 2023},
            {"id": "https://openalex.org/W4", "short_id": "W4", "year": 2024},
        ]
    ).to_csv(path, index=False)

    targets = read_cohort_targets(
        path,
        {2022: (3,), 2023: (1, 2), 2024: (1,)},
    )

    assert targets["short_id"].tolist() == ["W2", "W3", "W4"]
    assert targets["publication_year"].tolist() == [2022, 2023, 2024]


def test_scan_snapshot_file_keeps_only_valid_future_citers(
    tmp_path: Path,
) -> None:
    source = tmp_path / "part_0000.gz"
    works = [
        {
            "id": "https://openalex.org/C1",
            "publication_year": 2023,
            "language": "en",
            "is_retracted": False,
            "is_paratext": False,
            "primary_topic": {"id": "https://openalex.org/T1"},
            "referenced_works": ["https://openalex.org/W1"],
        },
        {
            "id": "https://openalex.org/C2",
            "publication_year": 2026,
            "language": "en",
            "is_retracted": False,
            "is_paratext": False,
            "referenced_works": ["https://openalex.org/W1"],
        },
    ]
    with gzip.open(source, "wt", encoding="utf-8") as handle:
        for work in works:
            handle.write(json.dumps(work) + "\n")
    spool = tmp_path / "spool.jsonl"
    state_path = tmp_path / "state.json"

    state = _scan_snapshot_file(
        source,
        spool_path=spool,
        state_path=state_path,
        target_aliases={
            "https://openalex.org/W1": "https://openalex.org/W1",
            "W1": "https://openalex.org/W1",
        },
        target_years={"https://openalex.org/W1": 2022},
        target_max_horizons={"https://openalex.org/W1": 3},
        complete_end_year=2025,
    )

    rows = [json.loads(line) for line in spool.read_text().splitlines()]
    assert state["records_seen"] == 2
    assert state["matched_works"] == 1
    assert rows[0]["paper_ids"] == ["https://openalex.org/W1"]
    assert rows[0]["work"]["id"] == "https://openalex.org/C1"

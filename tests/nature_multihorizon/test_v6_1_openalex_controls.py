"""Local OpenAlex control extraction tests."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gear.nature_multihorizon.openalex_controls_v6_1 import (
    build_k1_team_controls,
    extract_target_metadata,
)


def test_local_snapshot_extraction_and_hash(tmp_path: Path) -> None:
    """Only requested records are extracted from a toy frozen snapshot."""
    snapshot = tmp_path / "snapshot"
    part = (
        snapshot
        / "data"
        / "works"
        / "updated_date=2026-01-01"
        / "part_0000.gz"
    )
    part.parent.mkdir(parents=True)
    (snapshot / "data" / "works" / "manifest").write_text(
        '{"entries":[]}\n', encoding="utf-8"
    )
    records = [
        {
            "id": "https://openalex.org/W1",
            "updated_date": "2026-01-01T00:00:00",
            "authors_count": 2,
            "institutions_distinct_count": 1,
            "countries_distinct_count": 1,
            "authorships": [
                {
                    "author": {"id": "https://openalex.org/A1"},
                    "countries": ["CN"],
                    "institutions": [
                        {"id": "https://openalex.org/I1", "country_code": "CN"}
                    ],
                },
                {
                    "author": {"id": "https://openalex.org/A2"},
                    "countries": ["CN"],
                    "institutions": [],
                },
            ],
        },
        {
            "id": "https://openalex.org/W2",
            "updated_date": "2026-01-01T00:00:00",
            "authorships": [],
        },
    ]
    with gzip.open(part, "wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(
                json.dumps(record, separators=(",", ":")) + "\n"
            )
    output = tmp_path / "output"
    manifest = extract_target_metadata(
        ["W1"],
        snapshot,
        output,
        workers=1,
    )
    assert manifest["n_target_records_found"] == 1
    frame = pd.read_parquet(output / "target_openalex_metadata.parquet")
    assert frame.loc[0, "openalex_author_count"] == 2
    assert frame.loc[0, "openalex_author_ids"].tolist() == [
        "https://openalex.org/A1",
        "https://openalex.org/A2",
    ]
    controls = build_k1_team_controls(frame)
    assert np.isclose(controls.loc[0, "log_author_count"], np.log1p(2))

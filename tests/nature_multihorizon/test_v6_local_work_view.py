from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from aspr.nature_multihorizon.local_work_view import (
    compact_work_record,
    materialize_local_work_view,
    reconstruct_abstract,
)


def _work(work_id: str, title: str) -> dict:
    return {
        "id": f"https://openalex.org/{work_id}",
        "publication_year": 2019,
        "display_name": title,
        "type": "article",
        "abstract_inverted_index": {"A": [0], "test": [1], "abstract": [2]},
        "primary_location": {
            "source": {
                "id": "https://openalex.org/S1",
                "display_name": "Journal One",
                "type": "journal",
            }
        },
        "primary_topic": {
            "id": "https://openalex.org/T1",
            "display_name": "Catalysis",
            "subfield": {
                "id": "https://openalex.org/SF1",
                "display_name": "Physical Chemistry",
            },
            "field": {
                "id": "https://openalex.org/F1",
                "display_name": "Chemistry",
            },
            "domain": {
                "id": "https://openalex.org/D1",
                "display_name": "Physical Sciences",
            },
        },
        "referenced_works": ["https://openalex.org/R1"],
    }


def test_abstract_and_compact_record_are_deterministic() -> None:
    work = _work("W1", "Paper One")
    assert reconstruct_abstract(work["abstract_inverted_index"]) == "A test abstract"
    first = compact_work_record(work)
    second = compact_work_record(work)
    assert first["source_id"] == "https://openalex.org/S1"
    assert first["field_id"] == "https://openalex.org/F1"
    assert first["record_sha256"] == second["record_sha256"]


def test_materialize_local_work_view_deduplicates_and_records_lineage(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(
        "\n".join(
            [
                json.dumps({"work": _work("W1", "Paper One")}),
                json.dumps({"work": _work("W2", "Paper Two")}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    second.write_text(
        json.dumps({"work": _work("W1", "Duplicate")}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "work_view.parquet"

    manifest = materialize_local_work_view(
        [first, second],
        output,
        required_ids=("W1", "W2"),
        input_hashes=True,
        batch_size=1,
    )

    frame = pd.read_parquet(output)
    assert len(frame) == 2
    assert set(frame["work_id"]) == {
        "https://openalex.org/W1",
        "https://openalex.org/W2",
    }
    assert manifest["duplicates_removed"] == 1
    assert manifest["required_id_coverage"] == 1.0
    assert manifest["abstract_coverage"] == 1.0
    assert manifest["source_coverage"] == 1.0
    assert manifest["output_sha256"].startswith("sha256:")
    assert output.with_suffix(".parquet.manifest.json").is_file()

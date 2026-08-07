"""Focused tests for audited reuse in the uncapped v2 rebuild."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from scripts.build_reference_closure_v5_from_snapshot import (
    REFERENCE_WORK_FIELDS,
    seed_reference_rows,
)
from scripts.build_uncapped_future_labels_v2 import prepare_added_targets
from scripts.build_uncapped_target_metadata_seed_v2 import build_seed
from scripts.fetch_nature_supplemental_horizons_v5 import (
    materialize_reported_zero_checkpoints,
    read_cohort_targets,
)


def test_seed_reference_rows_reuses_only_requested_unique_ids(tmp_path: Path) -> None:
    """A prior closure may seed only requested, unique reference records."""
    seed = tmp_path / "seed.csv"
    pd.DataFrame(
        [
            {"id": "https://openalex.org/W1", "title": "one"},
            {"id": "https://openalex.org/W1", "title": "duplicate"},
            {"id": "https://openalex.org/W2", "title": "not requested"},
        ]
    ).to_csv(seed, index=False)
    output = tmp_path / "output.csv"
    found: set[str] = set()
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REFERENCE_WORK_FIELDS)
        writer.writeheader()
        added = seed_reference_rows(
            seed,
            reference_ids={"https://openalex.org/W1"},
            found_ids=found,
            writer=writer,
        )
    assert added == 1
    assert found == {"https://openalex.org/W1"}
    assert len(pd.read_csv(output)) == 1


def test_prepare_added_targets_fetches_only_completely_new_papers(
    tmp_path: Path,
) -> None:
    """Seed-complete papers are reused and wholly new papers are selected."""
    targets = pd.DataFrame(
        {
            "id": ["P1", "P2"],
            "year": [2017, 2021],
            "document_type": ["article", "article"],
        }
    )
    target_path = tmp_path / "targets.csv"
    targets.to_csv(target_path, index=False)
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    pd.DataFrame(
        {
            "paper_id": ["P1", "P1", "P1"],
            "horizon": [3, 5, 8],
        }
    ).to_parquet(seed_dir / "future_graph_deltas_multihorizon.parquet")
    output = tmp_path / "added.csv"
    manifest = prepare_added_targets(
        argparse.Namespace(
            target_works=target_path,
            seed_future_dir=seed_dir,
            output_target_works=output,
        )
    )
    assert manifest["overall_pass"]
    assert manifest["n_added_targets_to_fetch"] == 1
    assert pd.read_csv(output)["id"].tolist() == ["P2"]


def test_target_metadata_seed_reuses_raw_reference_work(tmp_path: Path) -> None:
    """Raw reference records may seed authorship controls for target papers."""
    targets = tmp_path / "targets.parquet"
    pd.DataFrame({"id": ["https://openalex.org/W1"]}).to_parquet(targets)
    base = tmp_path / "base.parquet"
    pd.DataFrame(
        columns=[
            "paper_id",
            "openalex_updated_date",
            "openalex_author_count",
            "openalex_institution_count",
            "openalex_country_count",
            "openalex_author_ids",
            "metadata_source_file",
            "raw_record_sha256",
        ]
    ).to_parquet(base)
    checkpoint = tmp_path / "reference.jsonl"
    checkpoint.write_text(
        json.dumps(
            {
                "work": {
                    "id": "https://openalex.org/W1",
                    "updated_date": "2026-01-01",
                    "authors_count": 1,
                    "institutions_distinct_count": 1,
                    "countries_distinct_count": 1,
                    "authorships": [
                        {
                            "author": {"id": "https://openalex.org/A1"},
                            "countries": ["CN"],
                            "institutions": [{"id": "https://openalex.org/I1"}],
                        }
                    ],
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "seed.parquet"
    manifest = build_seed(
        target_works=targets,
        base_seed=base,
        checkpoint_jsonl=[checkpoint],
        output=output,
    )
    assert manifest["reference_targets_recovered"] == 1
    assert manifest["remaining_for_api"] == 0
    recovered = pd.read_parquet(output)
    assert recovered.loc[0, "openalex_author_count"] == 1


def test_reported_all_time_zero_is_frozen_without_network_checkpoint(
    tmp_path: Path,
) -> None:
    """An all-time zero proves every mature citation window is also zero."""

    target_path = tmp_path / "targets.csv"
    pd.DataFrame(
        {
            "id": ["https://openalex.org/W1", "https://openalex.org/W2"],
            "short_id": ["W1", "W2"],
            "year": [2020, 2020],
            "cited_by_count": [0, 3],
        }
    ).to_csv(target_path, index=False)
    targets = read_cohort_targets(target_path, {2020: (3, 5)})
    manifest, zero_ids = materialize_reported_zero_checkpoints(
        targets,
        {2020: (3, 5)},
        set(),
        argparse.Namespace(
            checkpoint_dir=tmp_path / "checkpoints",
            target_works=target_path,
        ),
    )
    assert zero_ids == {"https://openalex.org/W1"}
    assert manifest["n_reported_zero_targets"] == 1
    zero_table = pd.read_parquet(manifest["paper_ids_path"])
    assert zero_table["paper_id"].tolist() == ["https://openalex.org/W1"]
    assert not (tmp_path / "checkpoints" / "year_2020_tau5" / "W1.jsonl").exists()

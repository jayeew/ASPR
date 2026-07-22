from __future__ import annotations

import csv
from pathlib import Path

from scripts.fetch_openalex_reference_missing_v5 import (
    REFERENCE_WORK_FIELDS,
    load_checkpoint_rows,
    load_reference_ids,
    merge_reference_works,
    write_final_missing,
)


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_queue_checkpoint_merge_and_final_missing(tmp_path: Path) -> None:
    queue = tmp_path / "retry.csv"
    base = tmp_path / "reference_works.csv"
    checkpoint = tmp_path / "success.csv"
    final_missing = tmp_path / "final_missing.csv"

    _write_csv(
        queue,
        ["id", "short_id"],
        [
            {"id": "W1", "short_id": "W1"},
            {"id": "https://openalex.org/W2", "short_id": "W2"},
            {"id": "W2", "short_id": "W2"},
        ],
    )
    base_row = {field: "" for field in REFERENCE_WORK_FIELDS}
    base_row.update({"id": "https://openalex.org/W0", "short_id": "W0", "title": "base"})
    success_row = {field: "" for field in REFERENCE_WORK_FIELDS}
    success_row.update({"id": "https://openalex.org/W1", "short_id": "W1", "title": "topup"})
    _write_csv(base, REFERENCE_WORK_FIELDS, [base_row])
    _write_csv(checkpoint, REFERENCE_WORK_FIELDS, [success_row])

    ids = load_reference_ids(queue)
    checkpoint_rows = load_checkpoint_rows(checkpoint)
    merged_count = merge_reference_works(base, checkpoint_rows, base)
    remaining_count = write_final_missing(final_missing, ids, set(checkpoint_rows))

    assert ids == ["https://openalex.org/W1", "https://openalex.org/W2"]
    assert merged_count == 2
    assert remaining_count == 1
    with base.open("r", encoding="utf-8", newline="") as handle:
        merged = list(csv.DictReader(handle))
    assert [row["short_id"] for row in merged] == ["W0", "W1"]
    with final_missing.open("r", encoding="utf-8", newline="") as handle:
        missing = list(csv.DictReader(handle))
    assert [row["short_id"] for row in missing] == ["W2"]

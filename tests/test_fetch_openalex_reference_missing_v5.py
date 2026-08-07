from __future__ import annotations

import csv
from pathlib import Path

from scripts.fetch_openalex_reference_missing_v5 import (
    REFERENCE_WORK_FIELDS,
    fetch_missing_references,
    load_checkpoint_ids,
    load_reference_ids,
    merge_reference_works,
    write_final_missing,
)


def test_missing_reference_fetch_uses_complete_batches(
    tmp_path: Path, monkeypatch: object
) -> None:
    """Batch lookup freezes successes and explicitly records absent IDs."""
    import scripts.fetch_openalex_reference_missing_v5 as module

    calls: list[list[str]] = []

    def fake_fetch_complete(
        openalex: object, *, filters: list[str], per_page: int
    ) -> tuple[list[dict[str, object]], int, int]:
        del openalex, per_page
        calls.append(filters)
        return ([{"id": "https://openalex.org/W1"}], 1, 1)

    monkeypatch.setattr(module, "fetch_complete_partition", fake_fetch_complete)  # type: ignore[attr-defined]
    checkpoint = tmp_path / "success.csv"
    failures = tmp_path / "failures.csv"
    rows, failed = fetch_missing_references(
        ["https://openalex.org/W1", "https://openalex.org/W2"],
        checkpoint_path=checkpoint,
        failure_log_path=failures,
        openalex=object(),  # type: ignore[arg-type]
        workers=1,
        progress_every=100,
        batch_size=50,
        quiet=True,
    )
    assert calls == [["openalex_id:W1|W2"]]
    assert set(rows) == {"https://openalex.org/W1"}
    assert failed == {"https://openalex.org/W2"}


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
    base_row.update(
        {"id": "https://openalex.org/W0", "short_id": "W0", "title": "base"}
    )
    success_row = {field: "" for field in REFERENCE_WORK_FIELDS}
    success_row.update(
        {"id": "https://openalex.org/W1", "short_id": "W1", "title": "topup"}
    )
    _write_csv(base, REFERENCE_WORK_FIELDS, [base_row])
    _write_csv(checkpoint, REFERENCE_WORK_FIELDS, [success_row])

    ids = load_reference_ids(queue)
    checkpoint_ids = load_checkpoint_ids(checkpoint)
    merged_count = merge_reference_works(base, checkpoint, base)
    remaining_count = write_final_missing(final_missing, ids, checkpoint_ids)

    assert ids == ["https://openalex.org/W1", "https://openalex.org/W2"]
    assert merged_count == 2
    assert remaining_count == 1
    with base.open("r", encoding="utf-8", newline="") as handle:
        merged = list(csv.DictReader(handle))
    assert [row["short_id"] for row in merged] == ["W0", "W1"]
    with final_missing.open("r", encoding="utf-8", newline="") as handle:
        missing = list(csv.DictReader(handle))
    assert [row["short_id"] for row in missing] == ["W2"]

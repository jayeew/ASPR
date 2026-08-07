from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

from scripts.complete_reference_closure_from_snapshot_v2 import complete


def test_snapshot_completion_resolves_and_checkpoints(tmp_path: Path) -> None:
    """A full local shard scan appends matches and preserves true misses."""
    snapshot = tmp_path / "snapshot"
    shard = snapshot / "data" / "works" / "updated_date=2026-01-01" / "part.gz"
    shard.parent.mkdir(parents=True)
    with gzip.open(shard, "wb") as handle:
        for work_id in ("W1", "W9"):
            payload = {"id": f"https://openalex.org/{work_id}", "display_name": work_id}
            handle.write(
                json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"
            )

    queue = tmp_path / "queue.csv"
    with queue.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id"])
        writer.writeheader()
        writer.writerows([{"id": "W1"}, {"id": "W2"}])
    success = tmp_path / "success.csv"
    manifest_path = tmp_path / "manifest.json"
    manifest = complete(
        argparse.Namespace(
            snapshot_dir=snapshot,
            retry_queue=queue,
            success_checkpoint=success,
            file_checkpoint_dir=tmp_path / "file_checkpoints",
            manifest=manifest_path,
            workers=1,
            progress_every=1,
            max_files=None,
        )
    )

    assert manifest["local_snapshot_scan_complete"] is True
    assert manifest["n_successful_ids_after_scan"] == 1
    assert manifest["n_remaining_ids_after_scan"] == 1
    with success.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["short_id"] for row in rows] == ["W1"]

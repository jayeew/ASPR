from __future__ import annotations

import json

import pytest

from scripts.shard_pending_gear_benchmark import build_shards


def _write_manifest(path, case_ids: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "cohort": "frozen",
                "cases": [
                    {"case_id": case_id, "paper_path": f"/{case_id}.pdf"}
                    for case_id in case_ids
                ],
            }
        ),
        encoding="utf-8",
    )


def test_build_shards_excludes_completed_and_explicit_active(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    runs_dir = tmp_path / "runs"
    _write_manifest(manifest, ["a", "b", "c", "d", "e"])
    (runs_dir / "a").mkdir(parents=True)
    (runs_dir / "a" / "review_bundle.json").write_text(
        json.dumps({"status": "limited"}), encoding="utf-8"
    )

    shards, audit = build_shards(manifest, runs_dir, 2, explicit_exclusions={"b"})

    shard_ids = [[case["case_id"] for case in shard] for shard in shards]
    assert shard_ids == [["c", "e"], ["d"]]
    assert audit["completed_case_count"] == 1
    assert audit["pending_case_count"] == 3
    assert audit["explicit_exclusion_case_ids"] == ["b"]


def test_build_shards_can_retry_failed_bundle(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    runs_dir = tmp_path / "runs"
    _write_manifest(manifest, ["failed"])
    (runs_dir / "failed").mkdir(parents=True)
    (runs_dir / "failed" / "review_bundle.json").write_text(
        json.dumps({"status": "failed"}), encoding="utf-8"
    )

    shards, _ = build_shards(manifest, runs_dir, 1, retry_failed=True)

    assert [case["case_id"] for case in shards[0]] == ["failed"]


def test_build_shards_can_reverse_pending_order(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    runs_dir = tmp_path / "runs"
    _write_manifest(manifest, ["a", "b", "c"])

    shards, audit = build_shards(manifest, runs_dir, 1, reverse_pending=True)

    assert [case["case_id"] for case in shards[0]] == ["c", "b", "a"]
    assert audit["pending_order"] == "reverse"


def test_build_shards_rejects_duplicate_case_ids(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, ["duplicate", "duplicate"])

    with pytest.raises(ValueError, match="duplicate"):
        build_shards(manifest, tmp_path / "runs", 2)

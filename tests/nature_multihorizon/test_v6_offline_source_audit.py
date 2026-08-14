from __future__ import annotations

import socket
from pathlib import Path

import pytest

from gear.nature_multihorizon.offline import (
    NetworkAccessForbidden,
    network_forbidden,
    validate_local_only_config,
)
from gear.nature_multihorizon.source_audit_v6 import audit_local_sources


def _config(source_path: str, snapshot_path: str) -> dict:
    return {
        "network_policy": "forbidden",
        "raw_data_policy": "local_frozen_only",
        "sources": [
            {
                "asset_id": "frozen_tables",
                "path": source_path,
                "role": "test frozen tables",
                "required": True,
                "required_files": ["manifest.json", "rows.csv"],
                "identity_files": ["manifest.json"],
            },
            {
                "asset_id": "snapshot",
                "path": snapshot_path,
                "role": "test local snapshot",
                "required": True,
                "required_directories": ["data/works"],
            },
        ],
    }


def test_source_audit_hashes_local_manifests_and_snapshot_inventory(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    snapshot = tmp_path / "snapshot"
    source.mkdir()
    (source / "manifest.json").write_text('{"version": 1}\n', encoding="utf-8")
    (source / "rows.csv").write_text("id\nW1\n", encoding="utf-8")
    works = snapshot / "data" / "works" / "part=0"
    works.mkdir(parents=True)
    (works / "rows.jsonl").write_text('{"id": "W1"}\n', encoding="utf-8")

    report = audit_local_sources(
        _config(str(source), str(snapshot)),
        project_root=tmp_path,
        deep_hash=True,
    )

    assert report["overall_pass"]
    assert report["source_lineage_id"].startswith("sha256:")
    snapshot_row = next(
        row for row in report["assets"] if row["asset_id"] == "snapshot"
    )
    assert snapshot_row["identity_directories"][0]["file_count"] == 1


def test_source_audit_fails_closed_for_missing_required_asset(tmp_path: Path) -> None:
    report = audit_local_sources(
        _config(str(tmp_path / "missing"), str(tmp_path / "missing-snapshot")),
        project_root=tmp_path,
    )

    assert not report["overall_pass"]
    assert report["required_failure_count"] == 2
    assert all(row["status"] == "blocked" for row in report["assets"])


def test_remote_source_paths_are_rejected() -> None:
    with pytest.raises(ValueError, match="remote path"):
        validate_local_only_config({"source_path": "https://example.org/data.parquet"})


def test_network_guard_blocks_before_socket_io() -> None:
    with network_forbidden():
        with pytest.raises(NetworkAccessForbidden):
            socket.create_connection(("127.0.0.1", 9), timeout=0.01)

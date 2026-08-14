from __future__ import annotations

import json

import pytest

from gear.calibration_assets import RELEASE_ALIAS, load_calibration_release
from gear.config import load_config


def test_active_calibration_release_is_local_and_complete():
    release = load_calibration_release(RELEASE_ALIAS, verify=True)
    assert release.release_id == "pgc-v3-d5-fulltext16-80f673c0-93e2e0dd"
    assert "data/calibration/releases" in release.asset_root.as_posix()
    assert release.manifest.replay["passed"] is True
    assert release.manifest.row_count == 411_490
    assert release.path("oof_predictions").is_file()


def test_gear_resolves_runtime_assets_through_release_registry():
    config = load_config()
    paths = config.resolved_assets()
    for path in paths.model_dump().values():
        assert "data/calibration/releases" in str(path)


def test_registry_rejects_manifest_hash_drift(tmp_path):
    original = json.loads(
        load_config()
        .resolve_path("configs/gear/calibration_registry.json")
        .read_text(encoding="utf-8")
    )
    release_id = original["active"][RELEASE_ALIAS]
    original["releases"][release_id]["manifest_sha256"] = "sha256:" + "0" * 64
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps(original), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        load_calibration_release(RELEASE_ALIAS, registry_path=registry)

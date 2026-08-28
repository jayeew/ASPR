import json
from pathlib import Path

import pytest

from experiments.gear.evaluation.audit_runtime_cohort import audit_runtime_cohort


def _write_case(root: Path, case_id: str, code_hash: str) -> None:
    target = root / case_id
    target.mkdir(parents=True)
    payload = {
        "contract": "aspr_gear_run_manifest_v3",
        "runtime_code_sha256": code_hash,
        "runtime_config_sha256": "sha256:" + "b" * 64,
        "runtime_source_file_count": 40,
        "config_version": "frozen-v1",
        "status": "complete",
    }
    (target / "run_manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def test_runtime_cohort_requires_one_fingerprint(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"cases": [{"case_id": "a"}, {"case_id": "b"}]}),
        encoding="utf-8",
    )
    runs = tmp_path / "runs"
    _write_case(runs, "a", "sha256:" + "a" * 64)
    _write_case(runs, "b", "sha256:" + "a" * 64)
    result = audit_runtime_cohort(manifest, runs, tmp_path / "audit.json")
    assert result["passed"] is True
    assert result["cases"] == 2


def test_runtime_cohort_rejects_mixed_fingerprints(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"cases": [{"case_id": "a"}, {"case_id": "b"}]}),
        encoding="utf-8",
    )
    runs = tmp_path / "runs"
    _write_case(runs, "a", "sha256:" + "a" * 64)
    _write_case(runs, "b", "sha256:" + "c" * 64)
    with pytest.raises(ValueError, match="mixed runtime code"):
        audit_runtime_cohort(manifest, runs, tmp_path / "audit.json")


def test_runtime_cohort_rejects_nonfrozen_fingerprint(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cases": [{"case_id": "a"}]}), encoding="utf-8")
    runs = tmp_path / "runs"
    _write_case(runs, "a", "sha256:" + "a" * 64)
    with pytest.raises(ValueError, match="frozen expectation"):
        audit_runtime_cohort(
            manifest,
            runs,
            tmp_path / "audit.json",
            expected_code_sha256="sha256:" + "c" * 64,
        )

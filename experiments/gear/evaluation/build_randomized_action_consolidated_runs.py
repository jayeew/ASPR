"""Create a non-destructive run view preferring protocol-valid repair outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_consolidated_runs(
    manifest_path: Path,
    repair_manifest_path: Path,
    original_runs_dir: Path,
    repaired_runs_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Link each assigned case to its original or repaired run directory."""
    manifest = _json(manifest_path)
    repair = _json(repair_manifest_path)
    cases = manifest.get("cases", [])
    repair_cases = repair.get("cases", [])
    if not cases or repair.get("repair_selection_uses_outcomes") is not False:
        raise ValueError("invalid source or repair manifest")
    expected = {str(case["case_id"]) for case in cases}
    repaired = {str(case["case_id"]) for case in repair_cases}
    if not repaired <= expected:
        raise ValueError("repair manifest contains cases outside source manifest")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("consolidated output directory must be absent or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    for case_id in sorted(expected):
        source_root = repaired_runs_dir if case_id in repaired else original_runs_dir
        source = (source_root / case_id).resolve()
        _require_complete(source, case_id)
        (output_dir / case_id).symlink_to(source, target_is_directory=True)
    report = {
        "contract": "gear_randomized_action_consolidated_runs_v1",
        "cases": len(expected),
        "original_cases": len(expected - repaired),
        "repaired_cases": len(repaired),
        "repair_selection_uses_outcomes": False,
        "all_run_artifacts_present": True,
    }
    (output_dir / "consolidation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _require_complete(path: Path, case_id: str) -> None:
    if not path.is_dir():
        raise ValueError(f"run directory missing: {case_id}")
    for name in ("review.json", "run_manifest.json", "review_bundle.json"):
        if not (path / name).is_file():
            raise ValueError(f"run artifact missing: {case_id}/{name}")


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repair-manifest", type=Path, required=True)
    parser.add_argument("--original-runs-dir", type=Path, required=True)
    parser.add_argument("--repaired-runs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_consolidated_runs(
        args.manifest,
        args.repair_manifest,
        args.original_runs_dir,
        args.repaired_runs_dir,
        args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_consolidated_runs"]

"""Bind frozen development90 to a disjoint confirmatory replication holdout60."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .graph_action_randomized_runner import ACTIONS


def build_snapshot(
    development_path: Path,
    holdout_path: Path,
    development_manifest_path: Path,
    holdout_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    development = pd.read_parquet(development_path)
    holdout = pd.read_parquet(holdout_path)
    development_manifest = _json(development_manifest_path)
    holdout_manifest = _json(holdout_manifest_path)
    _validate(development, holdout, development_manifest, holdout_manifest)
    randomized = pd.concat([development, holdout], ignore_index=True, sort=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    development_out = output_dir / "development_action_log.parquet"
    holdout_out = output_dir / "holdout_action_log.parquet"
    randomized_out = output_dir / "randomized_action_log.parquet"
    development.to_parquet(development_out, index=False)
    holdout.to_parquet(holdout_out, index=False)
    randomized.to_parquet(randomized_out, index=False)
    development_ids = set(development["paper_id"].astype(str))
    development_cases = [
        case
        for case in development_manifest["cases"]
        if str(case["paper_id"]) in development_ids
    ]
    combined_manifest = {
        "contract": "gear_randomized_graph_action_replication_manifest_v1",
        "randomization_precedes_outcomes": True,
        "development_source_sha256": _sha256(development_manifest_path),
        "confirmatory_holdout_source_sha256": _sha256(holdout_manifest_path),
        "old_confirmatory_holdout_used": False,
        "replication_reason": "policy_protocol_revised_after_initial_holdout",
        "cases": [*development_cases, *holdout_manifest["cases"]],
    }
    manifest_out = output_dir / "replication_manifest_150.json"
    manifest_out.write_text(
        json.dumps(combined_manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    report = {
        "contract": "gear_stage_c_replication_snapshot_v1",
        "passed": True,
        "development_rows": 90,
        "confirmatory_holdout_rows": 60,
        "randomized_rows": 150,
        "paper_overlap": 0,
        "development_sha256": _sha256(development_out),
        "holdout_sha256": _sha256(holdout_out),
        "randomized_sha256": _sha256(randomized_out),
        "manifest_sha256": _sha256(manifest_out),
    }
    (output_dir / "replication_snapshot_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _validate(
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    development_manifest: dict[str, Any],
    holdout_manifest: dict[str, Any],
) -> None:
    if len(development) != 90 or len(holdout) != 60:
        raise ValueError("replication snapshot requires development90/holdout60")
    if not development["experiment_split"].astype(str).eq("development").all():
        raise ValueError("development snapshot contains non-development rows")
    if not holdout["experiment_split"].astype(str).eq("confirmatory_holdout").all():
        raise ValueError("replication holdout contains non-holdout rows")
    for label, frame, expected in (
        ("development", development, 15),
        ("holdout", holdout, 10),
    ):
        counts = frame["logged_action"].astype(str).value_counts().to_dict()
        if counts != {action: expected for action in ACTIONS}:
            raise ValueError(f"{label} A0-A5 allocation changed: {counts}")
        if not frame["propensity"].astype(float).eq(1.0 / 6.0).all():
            raise ValueError(f"{label} propensity changed")
        if not frame["matched_budget"].astype(float).eq(20.0).all():
            raise ValueError(f"{label} budget changed")
    development_ids = set(development["paper_id"].astype(str))
    holdout_ids = set(holdout["paper_id"].astype(str))
    if development_ids & holdout_ids:
        raise ValueError("replication development and holdout papers overlap")
    source_development = {
        str(case["paper_id"])
        for case in development_manifest.get("cases", [])
        if case.get("experiment_split") == "development"
    }
    source_holdout = {
        str(case["paper_id"]) for case in holdout_manifest.get("cases", [])
    }
    if development_ids != source_development or holdout_ids != source_holdout:
        raise ValueError("replication logs do not match their frozen manifests")
    if holdout_manifest.get("randomization_precedes_outcomes") is not True:
        raise ValueError("replication holdout was not randomized before outcomes")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_snapshot(
        args.development,
        args.holdout,
        args.development_manifest,
        args.holdout_manifest,
        args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_snapshot"]

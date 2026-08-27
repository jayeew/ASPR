"""Build, render, and audit the expanded-data ASPR Fig.3."""

from __future__ import annotations

import argparse
import platform
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .analysis import (
    CONFIG_PATH,
    build_panel_data,
    calibration_source_dir,
    load_config,
    resolve_path,
    sha256_file,
    write_json,
)
from .audit import validate_outputs
from .render import render_from_tables


def _source_records(config: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    source_dir = calibration_source_dir(config)
    paths = [
        source_dir / "oof_predictions.parquet",
        source_dir / "oof_metrics.csv",
        source_dir / "paper_scores.parquet",
        resolve_path(str(config["model_config"])),
    ]
    return [
        {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
        for path in paths
    ]


def _manifest(
    config: Mapping[str, Any],
    output_dir: Path,
    stage: str,
    audit: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    panel_dir = output_dir / "panel_data"
    return {
        "figure_id": 3,
        "figure_version": config["figure_version"],
        "title": "Temporal–disciplinary performance landscape and out-of-time validation of ASPR Score",
        "status": audit["status"] if audit else f"stage_{stage}_complete",
        "stage": stage,
        "source_policy": "uncapped_v2_source-year_complete_local_frozen",
        "calibration_release": config["calibration_release"],
        "old_fig3_inputs_used": False,
        "model_family": "hgb",
        "baseline_svg_sha256": config["baseline_svg_sha256"],
        "panel_data_manifest": (
            str((output_dir / "panel_data_manifest.json").resolve())
            if (output_dir / "panel_data_manifest.json").is_file()
            else None
        ),
        "numeric_data_recomputed": True,
        "sources": _source_records(config),
        "figure_config": {
            "path": str(CONFIG_PATH.resolve()),
            "sha256": sha256_file(CONFIG_PATH),
        },
        "panel_tables": [
            str(path.resolve()) for path in sorted(panel_dir.glob("*.csv"))
        ],
        "rendered": [
            str((output_dir / f"figure_full.{extension}").resolve())
            for extension in ("png", "svg", "pdf")
            if (output_dir / f"figure_full.{extension}").is_file()
        ],
        "audit_passed": bool(audit and audit["passed"]),
        "reproduction": "python3 -m experiments.fig03.new.run --stage all  # recompute panel tables, freeze hashes, render, audit",
        "software": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
        },
        "notes": [
            "All rendered predictive results use only the formal HGB model family.",
            "Performance is predictive association with later uptake and diffusion, not a causal or direct novelty claim.",
        ],
    }


def run(stage: str = "all") -> Mapping[str, Any]:
    """Execute one or all deterministic Fig.3 stages."""
    config = load_config()
    output_dir = resolve_path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    if stage in {"analysis", "all"}:
        build_panel_data(clean=True)
    if stage in {"render", "all"}:
        render_from_tables(config, output_dir)
    audit = validate_outputs(config, output_dir) if stage in {"audit", "all"} else None
    manifest = _manifest(config, output_dir, stage, audit)
    write_json(output_dir / "run_manifest.json", manifest)
    if audit is not None and not bool(audit["passed"]):
        failed = [row["check"] for row in audit["checks"] if not row["passed"]]
        raise RuntimeError(f"Fig.3 audit failed: {failed}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("analysis", "render", "audit", "all"), default="all"
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = run(arguments.stage)
    print(result["status"])

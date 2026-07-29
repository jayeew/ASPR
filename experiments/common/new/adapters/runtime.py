"""Uniform prepare/run/plot/audit runtime for every new figure."""

from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from experiments.common.new.base.common import (
    FigureBundle,
    SuitePaths,
    resolve_suite_paths,
)

from experiments.common.new.adapters.audit import audit_bundle
from experiments.common.new.adapters.builders import build_new_bundle
from experiments.common.new.adapters.contracts import SUPPORTED_FIGURES
from experiments.common.new.adapters.io import (
    json_ready,
    sha256_file,
    write_json,
    write_tables,
)
from experiments.common.new.adapters.fig2_evidence import (
    clean_fig2_obsolete_artifacts,
)
from experiments.common.new.adapters.renderers import render_new_figure


PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _deep_update(
    target: Dict[str, Any],
    update: Mapping[str, Any],
) -> Dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def load_figure_context(
    figure_config_path: Path,
) -> Tuple[Dict[str, Any], SuitePaths, Dict[str, Any]]:
    """Load the current suite configuration plus one figure's overrides."""
    figure_config_path = figure_config_path.resolve()
    local = json.loads(figure_config_path.read_text(encoding="utf-8"))
    base_path = PROJECT_ROOT / str(local["base_suite_config"])
    base_config, base_paths = resolve_suite_paths(
        base_path,
        PROJECT_ROOT / str(local["output_dir"]),
    )
    merged = _deep_update(copy.deepcopy(base_config), local.get("overrides", {}))
    output_dir = PROJECT_ROOT / str(local["output_dir"])
    paths = SuitePaths(
        project_root=base_paths.project_root,
        config_path=figure_config_path,
        output_root=output_dir.resolve(),
        paths=base_paths.paths,
    )
    return merged, paths, local


def _source_records(bundle: FigureBundle) -> list[Dict[str, Any]]:
    records = []
    for source in dict.fromkeys(Path(path).resolve() for path in bundle.source_paths):
        records.append(
            {
                "path": str(source),
                "exists": source.is_file(),
                "size_bytes": source.stat().st_size if source.is_file() else 0,
                "sha256": sha256_file(source) if source.is_file() else None,
            }
        )
    return records


def _output_records(output_dir: Path) -> Dict[str, Any]:
    records: Dict[str, Any] = {}
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name in {
            "run_manifest.json",
            "output_inventory.json",
        }:
            continue
        records[str(path.relative_to(output_dir))] = {
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
    return records


def _software_versions() -> Dict[str, str]:
    """Return exact versions for the numerical and figure toolchain."""
    packages = (
        "numpy",
        "pandas",
        "matplotlib",
        "Pillow",
        "scipy",
        "pycirclize",
        "biopython",
        "adjustText",
        "colorspacious",
    )
    output: Dict[str, str] = {}
    for package in packages:
        try:
            output[package] = version(package)
        except PackageNotFoundError:
            output[package] = "not-installed"
    return output


def run_figure(
    figure_id: int,
    figure_config_path: Path,
    *,
    stage: str,
) -> Dict[str, Any]:
    """Execute one or all deterministic stages for a figure."""
    if figure_id not in SUPPORTED_FIGURES:
        raise ValueError(
            f"Unsupported figure {figure_id}; supported={SUPPORTED_FIGURES}"
        )
    config, paths, local = load_figure_context(figure_config_path)
    output_dir = paths.output_root
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = build_new_bundle(figure_id, config, paths)
    panel_records: Dict[str, Any] = {}
    if stage in {"prepare", "run", "plot", "all"}:
        if figure_id == 2:
            clean_fig2_obsolete_artifacts(output_dir)
        panel_records = write_tables(bundle.tables, output_dir / "panel_data")
        write_json(output_dir / "panel_text.json", bundle.panel_text)
        write_json(output_dir / "chart_contract.json", bundle.chart_contract)
    formats = tuple(local.get("formats", ["png", "svg", "pdf"]))
    dpi = int(local.get("dpi", 220))
    rendered: Dict[str, Path] = {}
    if stage in {"plot", "all"}:
        rendered = render_new_figure(
            figure_id,
            bundle,
            output_dir,
            formats=formats,
            dpi=dpi,
        )
    audit = None
    if stage in {"audit", "all"}:
        audit = audit_bundle(
            figure_id,
            bundle,
            paths,
            output_dir,
            formats,
        )
        write_json(output_dir / "audit_report.json", audit)
    manifest = {
        "figure_id": figure_id,
        "title": bundle.title,
        "status": bundle.status,
        "stage": stage,
        "figure_config": {
            "path": str(figure_config_path.resolve()),
            "sha256": sha256_file(figure_config_path.resolve()),
        },
        "base_suite_config": str(
            (PROJECT_ROOT / str(local["base_suite_config"])).resolve()
        ),
        "source_policy": "local_frozen_only",
        "network_used_for_numeric_evidence": False,
        "fig8_in_code_suite": False,
        "sources": _source_records(bundle),
        "panel_data": panel_records,
        "rendered": {
            key: str(path.resolve()) for key, path in rendered.items()
        },
        "audit": audit,
        "notes": bundle.notes,
        "software": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "packages": _software_versions(),
        },
        "reproduction": (
            f"python3 -m experiments.fig{figure_id:02d}.new.run --stage all"
        ),
    }
    write_json(output_dir / "run_manifest.json", manifest)
    # Hash the final state only after the manifest has been written.
    write_json(
        output_dir / "output_inventory.json",
        {"files": _output_records(output_dir)},
    )
    return json_ready(manifest)


def run_figure_cli(
    figure_id: int,
    default_config: Path,
) -> None:
    """CLI shared by all figure-specific ``run.py`` entry points."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["prepare", "run", "plot", "audit", "all"],
        default="all",
    )
    parser.add_argument("--config", type=Path, default=default_config)
    args = parser.parse_args()
    manifest = run_figure(
        figure_id,
        args.config,
        stage=args.stage,
    )
    print(
        json.dumps(
            {
                "figure_id": figure_id,
                "status": manifest["status"],
                "stage": args.stage,
                "output_dir": str(
                    load_figure_context(args.config)[1].output_root
                ),
            },
            ensure_ascii=False,
        )
    )

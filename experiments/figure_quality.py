from __future__ import annotations

import datetime as dt
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a deterministic UTF-8 JSON payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _image_file_report(path: Path) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "path": str(path),
        "exists": int(path.exists()),
        "size_bytes": int(path.stat().st_size) if path.exists() else 0,
    }
    if not path.exists() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        return report
    try:
        from PIL import Image  # pylint: disable=import-outside-toplevel
    except ImportError:
        report["image_probe_status"] = "pillow_unavailable"
        return report
    try:
        with Image.open(path) as img:
            report["width_px"] = int(img.width)
            report["height_px"] = int(img.height)
            dpi = img.info.get("dpi")
            if isinstance(dpi, tuple) and dpi:
                report["dpi"] = [float(x) for x in dpi]
            report["image_probe_status"] = "ok"
    except OSError as exc:
        report["image_probe_status"] = f"read_failed:{exc}"
    return report


def build_run_manifest(
    *,
    figure: str,
    argv: Optional[Sequence[str]],
    output_dir: Path,
    inputs: Optional[Mapping[str, Any]] = None,
    domains: Optional[Sequence[str]] = None,
    quality_gates: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a reproducibility manifest for one figure run."""
    return {
        "figure": figure,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command_argv": list(argv or sys.argv),
        "cwd": str(Path.cwd()),
        "output_dir": str(output_dir),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "inputs": dict(inputs or {}),
        "domains": list(domains or []),
        "quality_gates": dict(quality_gates or {}),
        "extra": dict(extra or {}),
    }


def write_run_manifest(
    out_dir: Path,
    *,
    figure: str,
    argv: Optional[Sequence[str]],
    inputs: Optional[Mapping[str, Any]] = None,
    domains: Optional[Sequence[str]] = None,
    quality_gates: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    manifest = build_run_manifest(
        figure=figure,
        argv=argv,
        output_dir=out_dir,
        inputs=inputs,
        domains=domains,
        quality_gates=quality_gates,
        extra=extra,
    )
    write_json(out_dir / "run_manifest.json", manifest)
    return manifest


def build_figure_quality_report(
    *,
    figure: str,
    generated_files: Iterable[Path],
    quality_gates: Optional[Mapping[str, Any]] = None,
    visual_checks: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    gates = dict(quality_gates or {})
    overall_pass = gates.get("overall_pass")
    if overall_pass is None and "checks" in gates and isinstance(gates["checks"], Mapping):
        overall_pass = bool(all(bool(v) for v in gates["checks"].values()))
    return {
        "figure": figure,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "overall_pass": bool(overall_pass) if overall_pass is not None else None,
        "status_label": gates.get("status_label"),
        "quality_gates": gates,
        "visual_checks": dict(visual_checks or {}),
        "generated_files": [_image_file_report(Path(path)) for path in generated_files],
        "extra": dict(extra or {}),
    }


def write_figure_quality_report(
    out_dir: Path,
    *,
    figure: str,
    generated_files: Iterable[Path],
    quality_gates: Optional[Mapping[str, Any]] = None,
    visual_checks: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    report = build_figure_quality_report(
        figure=figure,
        generated_files=generated_files,
        quality_gates=quality_gates,
        visual_checks=visual_checks,
        extra=extra,
    )
    write_json(out_dir / "figure_quality_report.json", report)
    return report


def normalize_reference_closure_report(report: pd.DataFrame, coverage_threshold: float = 0.80) -> pd.DataFrame:
    """Mark unmeasured reference closure explicitly instead of reporting perfect coverage."""
    df = report.copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "coverage_materialized",
                "coverage_materialized_raw_reported",
                "coverage_measured",
                "coverage_status",
                "quality_gate_pass",
            ]
        )
    if "coverage_materialized" not in df.columns:
        df["coverage_materialized"] = np.nan
    if "raw_records" not in df.columns:
        df["raw_records"] = np.nan
    if "status" not in df.columns:
        df["status"] = "unknown"
    if "coverage_materialized_raw_reported" not in df.columns:
        df["coverage_materialized_raw_reported"] = pd.to_numeric(df["coverage_materialized"], errors="coerce")

    raw_records = pd.to_numeric(df["raw_records"], errors="coerce")
    status = df["status"].astype(str).str.lower()
    mode = df.get("reference_closure_mode", pd.Series("", index=df.index)).astype(str).str.lower()
    unmeasured_status = status.str.contains(
        "audit_only_no_online_closure|no_online_closure|raw_only|closure_disabled|required_but_online_expand_false",
        regex=True,
        na=False,
    )
    unmeasured_mode = mode.isin({"off", "no_online_closure"})
    not_measured = raw_records.fillna(0).le(0) | unmeasured_status | unmeasured_mode

    coverage = pd.to_numeric(df["coverage_materialized"], errors="coerce")
    df.loc[not_measured, "coverage_materialized"] = np.nan
    measured = ~not_measured & coverage.notna()
    df["coverage_measured"] = measured.astype(int)
    df["coverage_status"] = np.where(measured, "measured", "not_measured")
    normalized_coverage = pd.to_numeric(df["coverage_materialized"], errors="coerce")
    df["quality_gate_pass"] = (measured & (normalized_coverage >= float(coverage_threshold))).astype(int)
    return df


def strict_main_figure_failed(quality_gates: Mapping[str, Any]) -> bool:
    """Return True when a figure must remain diagnostic under strict main-figure mode."""
    overall = quality_gates.get("overall_pass")
    if overall is None and isinstance(quality_gates.get("checks"), Mapping):
        overall = all(bool(v) for v in quality_gates["checks"].values())
    return not bool(overall)


def write_strict_failure_report(
    out_dir: Path,
    *,
    figure: str,
    quality_gates: Mapping[str, Any],
    message: str,
) -> Dict[str, Any]:
    report = {
        "figure": figure,
        "strict_main_figure": "failed",
        "message": message,
        "quality_gates": dict(quality_gates),
    }
    write_json(out_dir / "strict_main_figure_failure.json", report)
    return report

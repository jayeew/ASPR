from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aspr.nature_multihorizon.release import load_release  # noqa: E402
from aspr.nature_multihorizon.artifact_store import hash_file  # noqa: E402
from aspr.nature_multihorizon.figure_views import (  # noqa: E402
    OPTIONAL_FIGURE_EVIDENCE,
    _claim_readiness,
)
from experiments.common.old.kg_perturbation_v2.renderers import render_figure  # noqa: E402


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def validate_figure_view(release_path: Path, figure: int) -> Dict[str, Any]:
    """Validate one release-bound figure view without computing new evidence."""
    if figure < 1 or figure > 10:
        raise ValueError("figure must be in 1..10")
    release_path = release_path if release_path.name == "release.json" else release_path / "release.json"
    loaded = load_release(release_path.parent, verify_hashes=False)
    release = _read_json(release_path)
    if release.get("channel") not in {"candidate", "frozen"}:
        raise ValueError("release channel must be candidate or frozen")
    analysis_id = str(release.get("analysis_id") or "")
    if not analysis_id:
        raise ValueError("release is missing analysis_id")
    artifacts = release.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("release is missing artifacts")
    view_root = loaded.path / "figure_views"
    view_dir = view_root / f"fig{figure:02d}"
    manifest_path = view_dir / "view_manifest.json"
    panel_spec_path = view_dir / "panel_spec.json"
    caption_stats_path = view_dir / "caption_stats.json"
    missing = [
        str(path)
        for path in (manifest_path, panel_spec_path, caption_stats_path, view_dir / "data", view_dir / "_SUCCESS")
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Figure view is incomplete: {missing}")
    manifest = _read_json(manifest_path)
    if str(manifest.get("analysis_id") or "") != analysis_id:
        raise ValueError("figure view analysis_id does not match release")
    expected_figure_id = f"fig{figure:02d}"
    if str(manifest.get("figure_id") or "") != expected_figure_id:
        raise ValueError("figure view figure_id does not match requested figure")
    panel_spec = _read_json(panel_spec_path)
    caption_stats = _read_json(caption_stats_path)
    for label, payload in (
        ("panel_spec", panel_spec),
        ("caption_stats", caption_stats),
    ):
        if str(payload.get("analysis_id") or "") != analysis_id:
            raise ValueError(f"{label} analysis_id does not match release")
        if str(payload.get("figure_id") or "") != expected_figure_id:
            raise ValueError(f"{label} figure_id does not match requested figure")
    output_rows = manifest.get("outputs")
    if not isinstance(output_rows, list) or not output_rows:
        raise ValueError("figure view manifest has no plot-data outputs")
    data_root = (view_dir / "data").resolve()
    if not data_root.is_dir() or (view_dir / "data").is_symlink():
        raise ValueError("figure data directory must be a regular directory")
    declared_paths = set()
    view_tables: Dict[str, pd.DataFrame] = {}
    records_by_path = {
        record.path: record for record in loaded.manifest.artifacts.values()
    }
    for row in output_rows:
        if not isinstance(row, dict) or not row.get("path") or not row.get("sha256"):
            raise ValueError("figure view manifest contains an invalid output record")
        relative = Path(str(row["path"]))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or len(relative.parts) != 2
            or relative.parts[0] != "data"
            or relative.suffix.lower() != ".csv"
        ):
            raise ValueError(
                f"Figure plot-data path must be data/<name>.csv: {relative}"
            )
        relative_text = relative.as_posix()
        if relative_text in declared_paths:
            raise ValueError(f"Duplicate figure plot-data path: {relative_text}")
        declared_paths.add(relative_text)
        path = (view_dir / relative).resolve()
        if path.parent != data_root or not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Figure plot-data file is missing: {path}")
        actual = hash_file(path).removeprefix("sha256:")
        if actual != str(row["sha256"]):
            raise ValueError(f"Figure plot-data hash mismatch: {path}")
        release_relative = f"figure_views/{expected_figure_id}/{relative_text}"
        record = records_by_path.get(release_relative)
        if record is None or record.sha256.removeprefix("sha256:") != actual:
            raise ValueError(
                f"Plot data are not bound to the release manifest: {relative_text}"
            )
        table_name = relative.stem
        if table_name in view_tables:
            raise ValueError(f"Duplicate figure table name: {table_name}")
        view_tables[table_name] = pd.read_csv(path, low_memory=False)
    actual_entries = {
        item.relative_to(view_dir).as_posix()
        for item in (view_dir / "data").iterdir()
        if item.is_file() and not item.is_symlink()
    }
    invalid_entries = [
        item
        for item in (view_dir / "data").iterdir()
        if item.is_symlink() or not item.is_file() or item.suffix.lower() != ".csv"
    ]
    if invalid_entries or actual_entries != declared_paths:
        raise ValueError(
            "Actual data/*.csv files do not exactly match view manifest outputs"
        )
    if set(panel_spec.get("data_tables", ())) != declared_paths:
        raise ValueError("panel_spec data tables do not match view manifest outputs")
    source_inputs = manifest.get("source_inputs")
    if not isinstance(source_inputs, dict) or not source_inputs:
        raise ValueError("figure view manifest has no source artifact bindings")
    for artifact_name, binding in source_inputs.items():
        if artifact_name not in loaded.manifest.artifacts or not isinstance(binding, dict):
            raise ValueError(f"Unknown source artifact binding: {artifact_name}")
        source_record = loaded.manifest.artifacts[artifact_name]
        if str(binding.get("artifact_name") or "") != artifact_name:
            raise ValueError(f"Source artifact name mismatch: {artifact_name}")
        if str(binding.get("release_path") or "") != source_record.path:
            raise ValueError(f"Source artifact path mismatch: {artifact_name}")
        expected_hash = source_record.sha256.removeprefix("sha256:")
        if str(binding.get("sha256") or "") != expected_hash:
            raise ValueError(
                f"Figure source artifact hash mismatch: {artifact_name}"
            )
    asset_records = {
        name: record
        for name, record in loaded.manifest.artifacts.items()
        if name.startswith("figure_evidence_asset__")
    }
    allowed_asset_hashes = {record.sha256 for record in asset_records.values()}
    bound_asset_hashes = {
        asset_records[name].sha256
        for name in source_inputs
        if name in asset_records
    }
    for artifact_name, table_name in OPTIONAL_FIGURE_EVIDENCE.items():
        frame = view_tables.get(table_name)
        if frame is None or frame.empty or "availability_status" in frame:
            continue
        if artifact_name not in source_inputs:
            raise ValueError(
                f"{table_name} is not explicitly bound in source_inputs"
            )
        for column in ("source_artifact_sha256", "protocol_hash"):
            if column not in frame:
                continue
            declared_hashes = set(frame[column].dropna().astype(str))
            unbound_hashes = sorted(declared_hashes - allowed_asset_hashes)
            if unbound_hashes:
                raise ValueError(
                    f"{table_name}.{column} contains hashes not bound to "
                    "figure-evidence assets"
                )
            missing_inputs = sorted(declared_hashes - bound_asset_hashes)
            if missing_inputs:
                raise ValueError(
                    f"{table_name}.{column} references assets absent from source_inputs"
                )
    if expected_figure_id == "fig09":
        profile = view_tables.get("external_case_profile", pd.DataFrame())
        if (
            not profile.empty
            and "availability_status" not in profile
        ):
            if len(profile) != 1 or "case_registry" not in source_inputs:
                raise ValueError(
                    "Fig.9 external profile must bind one pre-locked case_registry row"
                )
            registry_path = loaded.path / loaded.manifest.artifacts[
                "case_registry"
            ].path
            registry = _read_json(registry_path)
            cases = registry.get("cases", [])
            row = profile.iloc[0]
            case_id = str(row.get("case_id") or "")
            matches = [
                case
                for case in cases
                if isinstance(case, dict)
                and str(case.get("case_id") or "") == case_id
            ] if isinstance(cases, list) else []
            if len(matches) != 1:
                raise ValueError(
                    "Fig.9 case_id is not uniquely pre-locked in case_registry"
                )

            def normalize_doi(value: Any) -> str:
                return (
                    str(value or "")
                    .lower()
                    .replace("https://doi.org/", "")
                    .strip()
                )

            locked = matches[0]
            locked_doi = normalize_doi(locked.get("doi"))
            locked_paper_id = str(locked.get("paper_id") or "").strip()
            doi_matches = bool(
                locked_doi and normalize_doi(row.get("doi")) == locked_doi
            )
            paper_matches = bool(
                locked_paper_id
                and str(row.get("paper_id") or "").strip() == locked_paper_id
            )
            if not (doi_matches or paper_matches):
                raise ValueError(
                    "Fig.9 profile identity does not match case_registry"
                )
    computed_readiness = _claim_readiness(expected_figure_id, view_tables)
    readiness = computed_readiness["claim_readiness"]
    readiness_reasons = computed_readiness["readiness_reasons"]
    required_evidence_ids = computed_readiness["required_evidence_ids"]
    for label, payload in (
        ("view_manifest", manifest),
        ("panel_spec", panel_spec),
        ("caption_stats", caption_stats),
    ):
        for field, expected in computed_readiness.items():
            if payload.get(field) != expected:
                raise ValueError(
                    f"{label} {field} does not match recomputed plot-data readiness"
                )
    view_prefix = f"figure_views/fig{figure:02d}/"
    for record in loaded.manifest.artifacts.values():
        if record.path.startswith(view_prefix):
            path = loaded.path / record.path
            if hash_file(path) != record.sha256:
                raise ValueError(f"Release-bound figure artifact hash mismatch: {path}")
    return {
        "figure": figure,
        "analysis_id": analysis_id,
        "channel": release["channel"],
        "release_root": str(loaded.path.resolve()),
        "view_dir": str(view_dir),
        "view_manifest": str(manifest_path),
        "status": "ready_for_draw",
        "claim_readiness": readiness,
        "readiness_reasons": readiness_reasons,
        "required_evidence_ids": required_evidence_ids,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Nature Multi-Horizon V1 figure view.")
    parser.add_argument("--release", type=Path, required=True, help="Explicit release.json path; latest aliases are forbidden.")
    parser.add_argument("--figure", type=int, choices=range(1, 11), required=True)
    parser.add_argument(
        "--draw-only",
        action="store_true",
        help="Validate the immutable plot-data contract; a figure-specific renderer may then draw it.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Draw a claim-readiness placeholder explicitly for draft inspection.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = validate_figure_view(args.release.resolve(), args.figure)
    result["mode"] = "draw_only" if args.draw_only else "validate"
    if args.draw_only:
        if (
            result["claim_readiness"] == "placeholder"
            and not args.allow_incomplete
        ):
            raise ValueError(
                "The requested figure is a scientific-evidence placeholder; "
                "use --allow-incomplete only for draft inspection."
            )
        output = (
            args.output.resolve()
            if args.output is not None
            else PROJECT_ROOT
            / "outputs"
            / "kg_perturbation_v2_rendered"
            / result["analysis_id"]
            / f"fig{args.figure:02d}.png"
        )
        release_root = Path(result["release_root"])
        if output == release_root or release_root in output.parents:
            raise ValueError(
                "Rendered output cannot be written inside an immutable release"
            )
        rendered = render_figure(
            Path(result["view_dir"]),
            args.figure,
            output,
            draft_watermark=(
                "DRAFT — MISSING EVIDENCE"
                if result["claim_readiness"] == "placeholder"
                else None
            ),
        )
        result["rendered_image"] = str(rendered)
        result["rendered_sha256"] = hash_file(rendered)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

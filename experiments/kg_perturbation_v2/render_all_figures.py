#!/usr/bin/env python3
"""Render all release-bound figures and emit an assembly-ready image manifest."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aspr.nature_multihorizon.artifact_store import hash_file, hash_json  # noqa: E402
from aspr.nature_multihorizon.release import load_release  # noqa: E402
from experiments.kg_perturbation_v2.renderers import render_figure  # noqa: E402
from experiments.kg_perturbation_v2.run_figure import validate_figure_view  # noqa: E402


def _renderer_provenance() -> Dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "experiments" / "kg_perturbation_v2" / "renderers.py",
        PROJECT_ROOT / "experiments" / "kg_perturbation_v2" / "run_figure.py",
    )
    return {
        path.relative_to(PROJECT_ROOT).as_posix(): hash_file(path)
        for path in paths
    }


def render_all(
    release_path: Path,
    output_dir: Path,
    *,
    allow_incomplete: bool = False,
) -> Dict[str, Any]:
    release_path = release_path.resolve()
    release_root = release_path.parent if release_path.name == "release.json" else release_path
    release = load_release(release_root, require_frozen=False)
    views = [
        validate_figure_view(release.path / "release.json", index)
        for index in range(1, 11)
    ]
    placeholders = [
        {
            "figure": int(view["figure"]),
            "readiness_reasons": list(view["readiness_reasons"]),
            "required_evidence_ids": list(view["required_evidence_ids"]),
        }
        for view in views
        if view["claim_readiness"] == "placeholder"
    ]
    if placeholders and not allow_incomplete:
        missing = ", ".join(f"Fig.{item['figure']}" for item in placeholders)
        raise ValueError(
            "Paper-claim figure bundle is incomplete; placeholder views remain: "
            f"{missing}. Use --allow-incomplete only for a draft bundle."
        )
    output_dir = output_dir.resolve()
    release_root = release.path.resolve()
    if output_dir == release_root or release_root in output_dir.parents:
        raise ValueError(
            "Rendered figure bundle cannot be created inside an immutable release"
        )
    if output_dir.exists():
        raise FileExistsError(f"Rendered figure bundle already exists: {output_dir}")
    staging = output_dir.with_name(f".{output_dir.name}.building-{os.getpid()}")
    staging.mkdir(parents=True)
    try:
        images = []
        for index, view in enumerate(views, start=1):
            output = staging / f"fig{index:02d}.png"
            view_dir = Path(view["view_dir"])
            render_figure(
                view_dir,
                index,
                output,
                draft_watermark=(
                    "DRAFT — MISSING EVIDENCE"
                    if view["claim_readiness"] == "placeholder"
                    else None
                ),
            )
            images.append(
                {
                    "figure": index,
                    "path": output.name,
                    "sha256": hash_file(output),
                    "claim_readiness": view["claim_readiness"],
                    "view_manifest_sha256": hash_file(
                        view_dir / "view_manifest.json"
                    ),
                    "panel_spec_sha256": hash_file(
                        view_dir / "panel_spec.json"
                    ),
                    "caption_stats_sha256": hash_file(
                        view_dir / "caption_stats.json"
                    ),
                }
            )
        manifest = {
            "schema_version": "1.0.0",
            "analysis_id": release.manifest.analysis_id,
            "dataset_id": release.manifest.dataset_id,
            "release_output_sha256": release.manifest.output_sha256,
            "claim_readiness": (
                "incomplete_draft" if placeholders else "ready"
            ),
            "placeholder_figures": placeholders,
            "renderer_provenance": _renderer_provenance(),
            "render_contract": {
                "format": "png",
                "dpi": 300,
                "bbox_inches": "tight",
                "facecolor": "white",
                "placeholder_watermark": "DRAFT — MISSING EVIDENCE",
            },
            "images": images,
        }
        (staging / "figure_images_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        marker_name = "_DRAFT" if placeholders else "_SUCCESS"
        (staging / marker_name).write_text(
            hash_json(manifest) + "\n", encoding="utf-8"
        )
        os.rename(staging, output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {
        "analysis_id": release.manifest.analysis_id,
        "output_dir": str(output_dir),
        "image_manifest": str(output_dir / "figure_images_manifest.json"),
        "claim_readiness": "incomplete_draft" if placeholders else "ready",
        "placeholder_figures": placeholders,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Fig.1--Fig.10 from one explicit evidence release")
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "Render placeholder panels as a draft bundle. The output receives "
            "_DRAFT, never _SUCCESS."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    print(
        json.dumps(
            render_all(
                args.release,
                args.output_dir,
                allow_incomplete=bool(args.allow_incomplete),
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

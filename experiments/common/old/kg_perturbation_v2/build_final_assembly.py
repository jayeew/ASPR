from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aspr.nature_multihorizon.artifact_store import hash_file, hash_json  # noqa: E402
from aspr.nature_multihorizon.release import load_release  # noqa: E402
from experiments.common.old.kg_perturbation_v2.render_all_figures import (  # noqa: E402
    _renderer_provenance,
)
from experiments.common.old.kg_perturbation_v2.run_figure import validate_figure_view  # noqa: E402


def build_final_assembly(
    release_path: Path,
    output_dir: Path,
    *,
    image_manifest: Path,
    copy_release: bool = True,
    allow_incomplete: bool = False,
) -> Dict[str, Any]:
    """Build one non-destructive V2 assembly from a single frozen analysis."""
    release_path = release_path.resolve()
    release_root = release_path.parent if release_path.name == "release.json" else release_path
    loaded = load_release(release_root, require_frozen=True)
    figures = [
        validate_figure_view(loaded.path / "release.json", index)
        for index in range(1, 11)
    ]
    placeholders = [
        {
            "figure": int(figure["figure"]),
            "readiness_reasons": list(figure["readiness_reasons"]),
            "required_evidence_ids": list(figure["required_evidence_ids"]),
        }
        for figure in figures
        if figure["claim_readiness"] == "placeholder"
    ]
    if placeholders and not allow_incomplete:
        missing = ", ".join(f"Fig.{item['figure']}" for item in placeholders)
        raise ValueError(
            "Final paper assembly requires claim-ready Fig.1--Fig.10; "
            f"placeholder views remain: {missing}. Use --allow-incomplete only "
            "for a draft assembly."
        )
    expected_readiness = "incomplete_draft" if placeholders else "ready"
    output_dir = output_dir.resolve()
    release_root = loaded.path.resolve()
    if output_dir == release_root or release_root in output_dir.parents:
        raise ValueError(
            "Final assembly cannot be created inside an immutable release"
        )
    if output_dir.exists():
        raise FileExistsError(f"Final assembly is immutable and already exists: {output_dir}")
    staging = output_dir.with_name(f".{output_dir.name}.building-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"Stale final-assembly staging exists: {staging}")
    staging.mkdir(parents=True)
    try:
        image_manifest = image_manifest.resolve()
        if image_manifest.name != "figure_images_manifest.json":
            raise ValueError(
                "Rendered-image manifest must be named figure_images_manifest.json"
            )
        image_payload = json.loads(image_manifest.read_text(encoding="utf-8"))
        if str(image_payload.get("claim_readiness") or "") != expected_readiness:
            raise ValueError(
                "Rendered-image manifest claim_readiness does not match the "
                "release-bound figure views"
            )
        image_placeholders = image_payload.get("placeholder_figures")
        if not isinstance(image_placeholders, list):
            raise ValueError(
                "Rendered-image manifest must declare placeholder_figures"
            )
        expected_placeholder_ids = {
            int(item["figure"]) for item in placeholders
        }
        image_placeholder_ids = {
            int(item["figure"])
            for item in image_placeholders
            if isinstance(item, dict) and "figure" in item
        }
        if image_placeholder_ids != expected_placeholder_ids:
            raise ValueError(
                "Rendered-image placeholder set does not match figure views"
            )
        expected_marker = "_DRAFT" if placeholders else "_SUCCESS"
        unexpected_marker = "_SUCCESS" if placeholders else "_DRAFT"
        marker_path = image_manifest.parent / expected_marker
        if not marker_path.is_file() or (
            image_manifest.parent / unexpected_marker
        ).exists():
            raise ValueError(
                "Rendered-image bundle has an invalid readiness marker"
            )
        if marker_path.read_text(encoding="utf-8").strip() != hash_json(
            image_payload
        ):
            raise ValueError(
                "Rendered-image marker does not bind the image manifest"
            )
        if image_payload.get("renderer_provenance") != _renderer_provenance():
            raise ValueError(
                "Rendered images were not produced by the current V2 renderer"
            )
        expected_render_contract = {
            "format": "png",
            "dpi": 300,
            "bbox_inches": "tight",
            "facecolor": "white",
            "placeholder_watermark": "DRAFT — MISSING EVIDENCE",
        }
        if image_payload.get("render_contract") != expected_render_contract:
            raise ValueError("Rendered-image contract is missing or changed")
        if str(image_payload.get("analysis_id") or "") != loaded.manifest.analysis_id:
            raise ValueError("Rendered-image manifest analysis_id does not match release")
        if str(image_payload.get("dataset_id") or "") != loaded.manifest.dataset_id:
            raise ValueError("Rendered-image manifest dataset_id does not match release")
        if (
            str(image_payload.get("release_output_sha256") or "")
            != loaded.manifest.output_sha256
        ):
            raise ValueError(
                "Rendered-image manifest release hash does not match release"
            )
        declared_images = image_payload.get("images")
        if not isinstance(declared_images, list):
            raise ValueError("Rendered-image manifest must contain an images list")
        figure_ids = [
            int(item["figure"])
            for item in declared_images
            if isinstance(item, dict) and "figure" in item
        ]
        if len(figure_ids) != len(declared_images) or len(set(figure_ids)) != len(
            figure_ids
        ):
            raise ValueError(
                "Rendered-image manifest contains invalid or duplicate figure entries"
            )
        declared_by_figure = {
            int(item["figure"]): item
            for item in declared_images
            if isinstance(item, dict) and "figure" in item
        }
        if set(declared_by_figure) != set(range(1, 11)):
            raise ValueError("Rendered-image manifest must declare exactly Fig.1--Fig.10")
        image_root = staging / "images"
        image_root.mkdir()
        image_records = []
        allowed_extensions = {".png", ".pdf", ".svg", ".tif", ".tiff"}
        expected_bundle_files = {
            image_manifest.name,
            expected_marker,
            *(str(item.get("path") or "") for item in declared_images),
        }
        actual_bundle_files = {
            item.name
            for item in image_manifest.parent.iterdir()
            if item.is_file() and not item.is_symlink()
        }
        invalid_bundle_entries = [
            item
            for item in image_manifest.parent.iterdir()
            if item.is_symlink() or not item.is_file()
        ]
        if invalid_bundle_entries or actual_bundle_files != expected_bundle_files:
            raise ValueError(
                "Rendered-image bundle files do not exactly match its manifest"
            )
        for index in range(1, 11):
            item = declared_by_figure[index]
            declared_source = Path(str(item.get("path") or ""))
            if (
                declared_source.is_absolute()
                or ".." in declared_source.parts
                or len(declared_source.parts) != 1
                or declared_source.name != f"fig{index:02d}.png"
            ):
                raise ValueError(
                    f"Rendered image path must stay in its bundle: {declared_source}"
                )
            source = (image_manifest.parent / declared_source).resolve()
            if not source.is_file() or source.suffix.lower() not in allowed_extensions:
                raise ValueError(f"Invalid rendered image for Fig.{index}: {source}")
            view_dir = loaded.path / "figure_views" / f"fig{index:02d}"
            expected_view_hashes = {
                "view_manifest_sha256": hash_file(
                    view_dir / "view_manifest.json"
                ),
                "panel_spec_sha256": hash_file(view_dir / "panel_spec.json"),
                "caption_stats_sha256": hash_file(
                    view_dir / "caption_stats.json"
                ),
            }
            for field, expected_hash in expected_view_hashes.items():
                if str(item.get(field) or "") != expected_hash:
                    raise ValueError(
                        f"Rendered image view binding mismatch for Fig.{index}: {field}"
                    )
            if str(item.get("claim_readiness") or "") != figures[index - 1][
                "claim_readiness"
            ]:
                raise ValueError(
                    f"Rendered image readiness mismatch for Fig.{index}"
                )
            source_hash = hash_file(source)
            if str(item.get("sha256") or "") != source_hash:
                raise ValueError(f"Rendered image hash mismatch for Fig.{index}")
            destination = image_root / f"fig{index:02d}{source.suffix.lower()}"
            shutil.copyfile(source, destination)
            image_records.append(
                {
                    "figure": index,
                    "path": str(destination.relative_to(staging)),
                    "sha256": hash_file(destination),
                }
            )
        if copy_release:
            copied_release = staging / "analyses" / loaded.manifest.analysis_id
            copied_release.parent.mkdir(parents=True)
            shutil.copytree(loaded.path, copied_release)
            copied = load_release(copied_release, require_frozen=True)
            copied_figures = []
            for index in range(1, 11):
                copied_figures.append(
                    validate_figure_view(copied.path / "release.json", index)
                )
            release_reference = (
                f"analyses/{loaded.manifest.analysis_id}/release.json"
            )
            figure_records = [
                {
                    "figure": index,
                    "analysis_id": loaded.manifest.analysis_id,
                    "channel": "frozen",
                    "view_dir": (
                        f"analyses/{loaded.manifest.analysis_id}/"
                        f"figure_views/fig{index:02d}"
                    ),
                    "status": "ready_for_draw",
                    "claim_readiness": copied_figures[index - 1][
                        "claim_readiness"
                    ],
                    "readiness_reasons": copied_figures[index - 1][
                        "readiness_reasons"
                    ],
                    "required_evidence_ids": copied_figures[index - 1][
                        "required_evidence_ids"
                    ],
                }
                for index in range(1, 11)
            ]
        else:
            release_reference = str(loaded.path / "release.json")
            figure_records = figures
        manifest = {
            "schema_version": "1.0.0",
            "analysis_id": loaded.manifest.analysis_id,
            "dataset_id": loaded.manifest.dataset_id,
            "release_channel": loaded.manifest.channel.value,
            "release": release_reference,
            "release_output_sha256": loaded.manifest.output_sha256,
            "figure_views": figure_records,
            "images": image_records,
            "single_analysis_id_contract": True,
            "claim_readiness": expected_readiness,
            "placeholder_figures": placeholders,
        }
        (staging / "assembly_manifest.json").write_text(
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
        "analysis_id": loaded.manifest.analysis_id,
        "output_dir": str(output_dir),
        "claim_readiness": expected_readiness,
        "placeholder_figures": placeholders,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a single-analysis Fig.1--Fig.10 V2 assembly")
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-manifest", type=Path, required=True)
    release_group = parser.add_mutually_exclusive_group()
    release_group.add_argument(
        "--copy-release",
        dest="copy_release",
        action="store_true",
        help="Copy the frozen release into a portable analyses/<analysis_id> tree (default).",
    )
    release_group.add_argument(
        "--reference-release-only",
        dest="copy_release",
        action="store_false",
        help="Keep an external absolute release reference instead of a portable copy.",
    )
    parser.set_defaults(copy_release=True)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "Build an explicitly incomplete draft when placeholder figures "
            "remain. The output receives _DRAFT, never _SUCCESS."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = build_final_assembly(
        args.release,
        args.output_dir,
        image_manifest=args.image_manifest,
        copy_release=bool(args.copy_release),
        allow_incomplete=bool(args.allow_incomplete),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

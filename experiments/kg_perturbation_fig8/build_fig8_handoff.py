#!/usr/bin/env python3
"""Build deterministic Fig. 8 GPT-image handoff outputs and quality gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.figure_quality import write_figure_quality_report, write_json, write_run_manifest


DEFAULT_SOURCE_DIR = PROJECT_ROOT / "experiments" / "kg_perturbation_fig8"
DEFAULT_PROMPT = DEFAULT_SOURCE_DIR / "fig8_gpt_image2_prompt.md"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "kg_perturbation_fig8"
REQUIRED_PROMPT_TERMS = [
    "Input manuscript",
    "Reference-graph calibration",
    "Claim cards",
    "ASPR reflection-guided review search",
    "Evidence-linked review",
    "B",
    "RS",
    "DeltaQ0",
    "Uzzi",
    "RTD",
    "BurtIP",
    "PDE",
    "ASPR-Qwen",
    "Nature review data",
    "Transparent peer review",
]


def relpath(path: Path) -> str:
    """Return a project-relative path when possible."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    """Return a file SHA-256 hash."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_default_image(source_dir: Path) -> Optional[Path]:
    """Return the most likely existing GPT-image PNG source."""
    images = sorted(path for path in source_dir.glob("*.png") if path.is_file())
    if not images:
        return None
    preferred = [path for path in images if "ChatGPT Image" in path.name]
    return preferred[0] if preferred else images[0]


def probe_image(path: Optional[Path]) -> Dict[str, Any]:
    """Read basic image dimensions."""
    if path is None or not path.exists():
        return {"exists": 0, "width_px": 0, "height_px": 0}
    try:
        from PIL import Image  # pylint: disable=import-outside-toplevel
    except ImportError:
        return {"exists": 1, "width_px": 0, "height_px": 0, "probe_status": "pillow_unavailable"}
    with Image.open(path) as image:
        return {"exists": 1, "width_px": int(image.width), "height_px": int(image.height), "probe_status": "ok"}


def build_quality_gates(prompt_text: str, image_info: Dict[str, Any]) -> Dict[str, Any]:
    """Build Fig. 8 quality gates."""
    missing_terms = [term for term in REQUIRED_PROMPT_TERMS if term not in prompt_text]
    lower_prompt = prompt_text.lower()
    checks = {
        "prompt_present": int(bool(prompt_text.strip())),
        "required_architecture_terms_present": int(not missing_terms),
        "sw_prior_boundary_present": int("calibration prior" in lower_prompt and "不是论文质量分数" in prompt_text),
        "aspr_qwen_side_branch_present": int("ASPR-Qwen" in prompt_text and "不要让 ASPR-Qwen 直接连接最终 review" in prompt_text),
        "visible_text_constrained": int("只渲染短标签" in prompt_text and "不要把说明性段落画进图中" in prompt_text),
        "image_present": int(image_info.get("exists", 0) == 1),
        "image_resolution_min1200x700": int(int(image_info.get("width_px", 0)) >= 1200 and int(image_info.get("height_px", 0)) >= 700),
    }
    overall = bool(all(checks.values()))
    return {
        "overall_pass": overall,
        "status_label": "fig8_gpt_image_handoff_ready" if overall else "fig8_handoff_incomplete",
        "checks": checks,
        "missing_prompt_terms": missing_terms,
        "thresholds": {
            "min_width_px": 1200,
            "min_height_px": 700,
            "required_prompt_terms": REQUIRED_PROMPT_TERMS,
        },
        "claim_boundary": (
            "Fig. 8 is an architecture and mechanism figure. It must not be used as performance evidence; "
            "ASPR-Qwen is shown as a reviewer-style side branch connected to Fig.9/Fig.10 evidence gates."
        ),
    }


def build_handoff(prompt_path: Path, image_path: Optional[Path], out_dir: Path) -> Dict[str, Path]:
    """Copy Fig. 8 prompt/image and write manifest plus quality report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if not prompt_path.exists():
        raise FileNotFoundError(prompt_path)
    prompt_text = prompt_path.read_text(encoding="utf-8")
    source_image = image_path or find_default_image(prompt_path.parent)
    out_prompt = out_dir / "fig8_gpt_image2_prompt.md"
    out_prompt.write_text(prompt_text, encoding="utf-8")
    out_image = out_dir / "fig8_full.png"
    if source_image and source_image.exists():
        shutil.copyfile(source_image, out_image)
    image_info = probe_image(out_image)
    quality_gates = build_quality_gates(prompt_text, image_info)
    manifest = {
        "figure": "fig8",
        "prompt_source": relpath(prompt_path),
        "prompt_output": relpath(out_prompt),
        "prompt_sha256": sha256_file(out_prompt),
        "image_source": relpath(source_image) if source_image else "",
        "image_output": relpath(out_image),
        "image_sha256": sha256_file(out_image) if out_image.exists() else "",
        "image_info": image_info,
        "rendering_policy": "GPT-image visual layer with exact prompt contract; no local SVG/Matplotlib renderer.",
        "claim_boundary": quality_gates["claim_boundary"],
    }
    write_json(out_dir / "fig8_handoff_manifest.json", manifest)
    write_figure_quality_report(
        out_dir,
        figure="fig8",
        generated_files=[out_image],
        quality_gates=quality_gates,
        extra=manifest,
    )
    write_run_manifest(
        out_dir,
        figure="fig8",
        argv=sys.argv,
        inputs={"prompt": relpath(prompt_path), "source_image": relpath(source_image) if source_image else ""},
        quality_gates=quality_gates,
        extra=manifest,
    )
    return {"prompt": out_prompt, "image": out_image, "manifest": out_dir / "fig8_handoff_manifest.json", "quality": out_dir / "figure_quality_report.json"}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build Fig. 8 handoff outputs and quality report.")
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Command-line entry point."""
    args = parse_args(argv)
    paths = build_handoff(args.prompt, args.image, args.out_dir)
    print(f"[fig8] wrote {args.out_dir}")
    for label, path in paths.items():
        print(f"[fig8] {label}: {path}")


if __name__ == "__main__":
    main()

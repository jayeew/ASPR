"""Run the code-backed ASPR figures while intentionally skipping Fig.8."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw, ImageOps

from experiments.common.new.adapters.contracts import SUPPORTED_FIGURES
from experiments.common.new.adapters.io import sha256_file, write_json
from experiments.common.new.adapters.runtime import PROJECT_ROOT, run_figure


def _parse_figures(value: str) -> List[int]:
    output: List[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", maxsplit=1)
            output.extend(range(int(start), int(end) + 1))
        else:
            output.append(int(token))
    invalid = sorted(set(output) - set(SUPPORTED_FIGURES))
    if invalid:
        raise ValueError(
            f"Unsupported figures {invalid}; Fig.8 is intentionally excluded"
        )
    return sorted(set(output))


def _contact_sheet(output_root: Path, figures: List[int]) -> Path:
    images = []
    for figure in figures:
        path = (
            PROJECT_ROOT
            / "outputs"
            / f"fig{figure:02d}"
            / "new"
            / "figure_full.png"
        )
        if path.is_file():
            images.append((figure, Image.open(path).convert("RGB")))
    if not images:
        raise FileNotFoundError("no rendered figure_full.png files")
    cell_width, cell_height = 1200, 920
    columns = 2
    rows = math.ceil(len(images) / columns)
    canvas = Image.new(
        "RGB",
        (columns * cell_width, rows * cell_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (figure, image) in enumerate(images):
        image = ImageOps.contain(
            image,
            (cell_width - 28, cell_height - 55),
        )
        left = (index % columns) * cell_width + (cell_width - image.width) // 2
        top = (index // columns) * cell_height + 40
        canvas.paste(image, (left, top))
        draw.text(
            ((index % columns) * cell_width + 12, (index // columns) * cell_height + 12),
            f"Fig. {figure}",
            fill="#172033",
        )
    path = output_root / "fig1_fig10_code_contact_sheet.png"
    canvas.save(path, optimize=True)
    return path


def run_all(figures: List[int], stage: str) -> Dict[str, object]:
    output_root = PROJECT_ROOT / "outputs/common/new/extension_suite"
    output_root.mkdir(parents=True, exist_ok=True)
    manifests: Dict[str, object] = {}
    for figure in figures:
        print(f"[experiments.common.new] Fig.{figure}: {stage}", flush=True)
        config = (
            PROJECT_ROOT
            / "experiments"
            / f"fig{figure:02d}"
            / "new"
            / "config.json"
        )
        if figure == 1:
            from experiments.fig01.new.run import run_figure1

            manifests[f"fig{figure:02d}"] = run_figure1(
                config,
                stage=stage,
            )
        else:
            manifests[f"fig{figure:02d}"] = run_figure(
                figure,
                config,
                stage=stage,
            )
    contact = None
    if stage in {"plot", "all"}:
        contact_path = _contact_sheet(output_root, figures)
        contact = {
            "path": str(contact_path.resolve()),
            "sha256": sha256_file(contact_path),
        }
    suite = {
        "suite_id": "aspr-v6.1-current-figures-r2",
        "figure_ids": figures,
        "supported_figure_ids": list(SUPPORTED_FIGURES),
        "fig8_implemented": False,
        "stage": stage,
        "figures": manifests,
        "contact_sheet": contact,
    }
    write_json(output_root / "run_manifest.json", suite)
    return suite


def main() -> None:
    if os.environ.get("PYTHONHASHSEED") != "0":
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = "0"
        os.execvpe(
            sys.executable,
            [
                sys.executable,
                "-m",
                "experiments.common.new.run_all",
                *sys.argv[1:],
            ],
            environment,
        )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figs",
        default="1,2,3,4,5,6,7,9,10",
        help="Comma-separated figure IDs; Fig.8 is invalid by design.",
    )
    parser.add_argument(
        "--stage",
        choices=["prepare", "run", "plot", "audit", "all"],
        default="all",
    )
    args = parser.parse_args()
    suite = run_all(_parse_figures(args.figs), args.stage)
    print(
        json.dumps(
            {
                "figure_ids": suite["figure_ids"],
                "stage": suite["stage"],
                "fig8_implemented": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

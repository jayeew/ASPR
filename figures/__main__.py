"""Locate figure workspaces or export a local final SVG."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure", choices=[f"fig{i}" for i in range(1, 12)])
    parser.add_argument("action", choices=("paths", "export"), nargs="?", default="paths")
    args = parser.parse_args()
    figures = Path(__file__).resolve().parent
    base = figures / args.figure
    if args.action == "paths":
        paths = {name: str(base / name) for name in ("code", "data", "scripts", "final")}
        print(json.dumps(paths, indent=2))
        return

    final = base / "final"
    svg = final / f"Fig{args.figure[3:]}.svg"
    if not svg.is_file():
        candidates = [path for path in final.glob("*.svg") if path.is_file()]
        if not candidates:
            parser.error(f"Local final SVG is missing. Place the editable master at {svg}.")
        if len(candidates) != 1:
            parser.error(f"Select the editable final master by placing it at {svg}.")
        svg = candidates[0]

    renderer = shutil.which("cairosvg")
    if renderer is None:
        parser.error("Install CairoSVG and make the cairosvg command available on PATH.")

    env = os.environ.copy()
    font_config = figures / "fig1/data/reference/layouts/fonts.conf"
    if font_config.is_file():
        env.setdefault("FONTCONFIG_FILE", str(font_config))
    for extension in ("pdf", "png"):
        output = svg.with_suffix(f".{extension}")
        command = [renderer, str(svg), "-f", extension, "-o", str(output)]
        if extension == "png":
            command.extend(["--output-width", "3600"])
        subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()

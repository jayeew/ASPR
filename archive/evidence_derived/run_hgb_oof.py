"""Run the canonical simplified-protocol four-set HGB OOF experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .hgb_oof import (
    DEFAULT_CONFIG,
    DEFAULT_FROZEN_MANIFEST,
    DEFAULT_MATRIX_MANIFEST,
    DEFAULT_OUTPUT,
    run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-manifest", type=Path, default=DEFAULT_MATRIX_MANIFEST)
    parser.add_argument("--frozen-manifest", type=Path, default=DEFAULT_FROZEN_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = run(
        arguments.matrix_manifest,
        arguments.frozen_manifest,
        arguments.config,
        arguments.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))

"""Build and render the multivariate feature-shift variant of Fig. 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .multivariate_shift import build_multivariate_shift
from .multivariate_shift_render import render_multivariate_shift_figure


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(__file__).with_name("config.json")
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "fig01" / "new"


def run(
    config_path: Path,
    output_dir: Path,
) -> Mapping[str, Any]:
    """Build auditable multivariate tables and render the final candidate."""
    analysis = build_multivariate_shift(output_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    render = render_multivariate_shift_figure(config, output_dir)
    return {"passed": True, "analysis": analysis, "render": render}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the multivariate Fig. 1 workflow."""
    args = parse_args(argv)
    result = run(args.config.resolve(), args.output_dir.resolve())
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "core_feature_count": result["analysis"][
                    "core_feature_count"
                ],
                "core_dimension_count": result["analysis"][
                    "core_dimension_count"
                ],
                "png": result["render"]["artifacts"]["png"]["path"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Prepare batch-six H2 screening input using the protected generic builder."""

from __future__ import annotations

import prepare_contextual_source_screening_h2_batch4_v4 as builder

ROOT = builder.ROOT
OUTPUT_DIR = ROOT / "outputs"


def main() -> None:
    """Bind batch-six paths before invoking the reusable sheet constructor."""
    builder.INPUT = OUTPUT_DIR / "contextual_source_screening_input_batch6_v4.csv"
    builder.AI = OUTPUT_DIR / "contextual_source_screening_AI_batch6_completed_v4.csv"
    builder.H1 = OUTPUT_DIR / "contextual_source_screening_H1_batch6_completed_v4.csv"
    builder.OUTPUT = OUTPUT_DIR / "contextual_source_screening_H2_batch6_v4.csv"
    builder.MANIFEST = (
        OUTPUT_DIR / "contextual_source_screening_H2_batch6_input_manifest_v4.json"
    )
    builder.main()


if __name__ == "__main__":
    main()

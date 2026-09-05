#!/usr/bin/env python3
"""Build innovation-only ReviewerView and discussion-resolved references."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gear.config import load_config
from gear.review_state import ReviewerReferenceBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--review-markdown", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    views, resolved = ReviewerReferenceBuilder(load_config(args.config)).build(
        args.paper_id, args.review_markdown, args.output_dir
    )
    print(
        f"真人创新性参照完成：ReviewerView={len(views)}，"
        f"DiscussionResolvedClaim={len(resolved.claims)}"
    )


if __name__ == "__main__":
    main()

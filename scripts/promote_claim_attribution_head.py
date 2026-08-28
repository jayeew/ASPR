"""Promote a hash-bound T0 claim-attribution model after two Gate-1 passes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gear.claim_attribution import promote_claim_attribution_release


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--gate1-report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    args = parser.parse_args()
    release = promote_claim_attribution_release(
        model_path=args.model.resolve(),
        replay_path=args.replay.resolve(),
        gate1_report_paths=[value.resolve() for value in args.gate1_report],
        output_path=args.output.resolve(),
        release_id=args.release_id,
    )
    print(json.dumps(release.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

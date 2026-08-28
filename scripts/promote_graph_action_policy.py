"""Promote a frozen action policy only after complete Stage C and Gate 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gear.graph_action_policy import promote_graph_action_policy_release


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--development-data", type=Path, required=True)
    parser.add_argument("--randomized-data", type=Path, required=True)
    parser.add_argument("--graph-policy", type=Path, required=True)
    parser.add_argument("--no-graph-policy", type=Path, required=True)
    parser.add_argument("--gate2-report", type=Path, required=True)
    parser.add_argument("--frozen-replay-manifest", type=Path, required=True)
    parser.add_argument("--source-fingerprint-audit", type=Path, required=True)
    parser.add_argument("--stage-a-runtime-audit", type=Path, required=True)
    parser.add_argument("--stage-b-runtime-audit", type=Path, required=True)
    parser.add_argument("--stage-c-runtime-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    args = parser.parse_args()
    release = promote_graph_action_policy_release(
        model_path=args.model.resolve(),
        replay_path=args.replay.resolve(),
        development_data_path=args.development_data.resolve(),
        randomized_data_path=args.randomized_data.resolve(),
        graph_policy_path=args.graph_policy.resolve(),
        no_graph_policy_path=args.no_graph_policy.resolve(),
        gate2_report_path=args.gate2_report.resolve(),
        frozen_replay_manifest_path=args.frozen_replay_manifest.resolve(),
        source_fingerprint_audit_path=args.source_fingerprint_audit.resolve(),
        stage_a_runtime_audit_path=args.stage_a_runtime_audit.resolve(),
        stage_b_runtime_audit_path=args.stage_b_runtime_audit.resolve(),
        stage_c_runtime_audit_path=args.stage_c_runtime_audit.resolve(),
        output_path=args.output.resolve(),
        release_id=args.release_id,
    )
    print(json.dumps(release.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

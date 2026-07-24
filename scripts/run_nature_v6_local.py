#!/usr/bin/env python3
"""Offline entry point for ASPR v6 local-frozen audits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aspr.nature_multihorizon.framework_audit_v6 import (  # noqa: E402
    audit_v6_framework,
    write_framework_audit,
)
from aspr.nature_multihorizon.finalize_v6 import (  # noqa: E402
    finalize_v6_release,
)
from aspr.nature_multihorizon.construct_validation_v6 import (  # noqa: E402
    run_construct_validation,
)
from aspr.nature_multihorizon.development_v6 import (  # noqa: E402
    run_development_protocol,
)
from aspr.nature_multihorizon.materialize_v6 import (  # noqa: E402
    materialize_common_input_views,
    materialize_field_events,
    materialize_opportunity_features,
    materialize_publication_features,
    materialize_targets_and_cohort,
)
from aspr.nature_multihorizon.offline import network_forbidden  # noqa: E402
from aspr.nature_multihorizon.release_v6 import (  # noqa: E402
    prepare_release_candidate,
)
from aspr.nature_multihorizon.source_audit_v6 import (  # noqa: E402
    audit_local_sources,
    write_source_audit,
)
from aspr.nature_multihorizon.sealed_v6 import (  # noqa: E402
    freeze_sealed_candidate,
    run_single_unlock_sealed_evaluation,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "nature_multihorizon" / "v6_local.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "nature_multihorizon_v6_local"


def _load_config(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("v6 config must be a JSON object")
    return payload


def build_parser() -> argparse.ArgumentParser:
    """Return the offline v6 audit CLI parser."""
    parser = argparse.ArgumentParser(
        description="Audit ASPR v6 definitions and frozen local sources."
    )
    parser.add_argument(
        "command",
        choices=(
            "audit-framework",
            "audit-source",
            "audit-all",
            "materialize-views",
            "materialize-field-events",
            "materialize-features",
            "materialize-opportunity",
            "materialize-targets",
            "materialize-all",
            "run-development",
            "validate-construct",
            "prepare-release",
            "freeze-sealed",
            "run-sealed",
            "finalize-release",
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "knowledge_corpus"
        / "nature_multihorizon_v6_local",
    )
    parser.add_argument(
        "--deep-hash",
        action="store_true",
        help="Hash every required source file in addition to identity files.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Reject reuse of already materialized stage outputs.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        choices=(3, 5, 8),
        default=5,
        help="Registered forecast horizon for development OOF.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=None,
        help="Override the preregistered bootstrap count for diagnostics.",
    )
    parser.add_argument(
        "--model-scope",
        choices=("full", "directional"),
        default="full",
        help="Run every ablation or only controls plus the primary model.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run all v6 audits behind a process-local network guard."""
    args = build_parser().parse_args(argv)
    config_path = args.config.resolve()
    output_root = args.output_root.resolve()
    config = _load_config(config_path)
    with network_forbidden():
        if args.command == "finalize-release":
            manifest, manifest_path = finalize_v6_release(
                project_root=PROJECT_ROOT,
                config_path=config_path,
                dataset_dir=args.dataset_dir,
                output_root=output_root,
            )
            print(
                json.dumps(
                    {
                        "stage": "final_frozen_release",
                        "artifact_id": manifest["artifact_id"],
                        "final_release_pass": manifest["summary"][
                            "final_release_pass"
                        ],
                        "sealed_unlock_count_used": manifest["summary"][
                            "sealed_unlock_count_used"
                        ],
                        "manifest_path": str(manifest_path),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            return 0
        if args.command == "freeze-sealed":
            manifest, run_dir = freeze_sealed_candidate(
                project_root=PROJECT_ROOT,
                config_path=config_path,
                dataset_dir=args.dataset_dir,
                output_root=output_root,
            )
            print(
                json.dumps(
                    {
                        "stage": "sealed_model_freeze",
                        "artifact_id": manifest["artifact_id"],
                        "sealed_holdout_labels_accessed": manifest[
                            "sealed_holdout_labels_accessed"
                        ],
                        "sealed_unlock_count_used": manifest[
                            "sealed_unlock_count_used"
                        ],
                        "run_dir": str(run_dir),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            return 0
        if args.command == "run-sealed":
            manifest, run_dir = run_single_unlock_sealed_evaluation(
                project_root=PROJECT_ROOT,
                config_path=config_path,
                dataset_dir=args.dataset_dir,
                output_root=output_root,
            )
            print(
                json.dumps(
                    {
                        "stage": "single_unlock_sealed_evaluation",
                        "artifact_id": manifest["artifact_id"],
                        "final_release_pass": manifest["summary"][
                            "final_release_pass"
                        ],
                        "sealed_unlocks_used": manifest["summary"][
                            "sealed_unlocks_used"
                        ],
                        "run_dir": str(run_dir),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            return 0
        if args.command == "prepare-release":
            manifest, run_dir = prepare_release_candidate(
                project_root=PROJECT_ROOT,
                config_path=config_path,
                dataset_dir=args.dataset_dir,
                output_root=output_root,
            )
            print(
                json.dumps(
                    {
                        "stage": "release_candidate",
                        "artifact_id": manifest["artifact_id"],
                        "release_candidate_ready_before_sealed": manifest[
                            "summary"
                        ]["release_candidate_ready_before_sealed"],
                        "sealed_holdout_accessed": manifest["summary"][
                            "sealed_holdout_accessed"
                        ],
                        "run_dir": str(run_dir),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            return 0
        if args.command == "validate-construct":
            manifest, run_dir = run_construct_validation(
                project_root=PROJECT_ROOT,
                config_path=config_path,
                dataset_dir=args.dataset_dir,
                output_root=output_root,
            )
            print(
                json.dumps(
                    {
                        "stage": "construct_validation",
                        "artifact_id": manifest["artifact_id"],
                        "c1_measurement_gate_pass": manifest["summary"][
                            "c1_measurement_gate_pass"
                        ],
                        "sealed_holdout_accessed": manifest["summary"][
                            "sealed_holdout_accessed"
                        ],
                        "run_dir": str(run_dir),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            return 0
        if args.command == "run-development":
            manifest, run_dir = run_development_protocol(
                project_root=PROJECT_ROOT,
                config_path=config_path,
                dataset_dir=args.dataset_dir,
                output_root=output_root,
                horizon=int(args.horizon),
                bootstrap_iterations=args.bootstrap_iterations,
                model_scope=args.model_scope,
            )
            print(
                json.dumps(
                    {
                        "stage": "development_nested_oof",
                        "horizon": int(args.horizon),
                        "artifact_id": manifest["artifact_id"],
                        "development_gate_pass": manifest["lineage"][
                            "development_gate_pass"
                        ],
                        "sealed_holdout_accessed": manifest[
                            "sealed_holdout_accessed"
                        ],
                        "run_dir": str(run_dir),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            return 0
        materialize_commands = {
            "materialize-views",
            "materialize-field-events",
            "materialize-features",
            "materialize-opportunity",
            "materialize-targets",
            "materialize-all",
        }
        if args.command in materialize_commands:
            source_report = audit_local_sources(
                config, project_root=PROJECT_ROOT, deep_hash=False
            )
            if not source_report["overall_pass"]:
                raise RuntimeError(
                    "required frozen sources failed the read-only audit"
                )
            dataset_dir = args.dataset_dir.resolve()
            resume = not args.no_resume
            views = materialize_common_input_views(
                config,
                project_root=PROJECT_ROOT,
                output_dir=dataset_dir,
                source_audit=source_report,
                resume=resume,
            )
            print(
                json.dumps(
                    {
                        "stage": "input_views",
                        "artifact_id": views["artifact_id"],
                    }
                ),
                flush=True,
            )
            if args.command == "materialize-views":
                return 0
            events = materialize_field_events(
                config,
                project_root=PROJECT_ROOT,
                output_dir=dataset_dir,
                upstream_manifest=views,
                resume=resume,
            )
            print(
                json.dumps(
                    {
                        "stage": "field_events",
                        "artifact_id": events["artifact_id"],
                    }
                ),
                flush=True,
            )
            if args.command == "materialize-field-events":
                return 0
            features = materialize_publication_features(
                config,
                project_root=PROJECT_ROOT,
                output_dir=dataset_dir,
                input_manifest=views,
                event_manifest=events,
                resume=resume,
            )
            print(
                json.dumps(
                    {
                        "stage": "publication_features",
                        "artifact_id": features["artifact_id"],
                    }
                ),
                flush=True,
            )
            if args.command == "materialize-features":
                return 0
            if args.command in {
                "materialize-opportunity",
                "materialize-all",
            }:
                opportunity = materialize_opportunity_features(
                    config,
                    project_root=PROJECT_ROOT,
                    output_dir=dataset_dir,
                    input_manifest=views,
                    resume=resume,
                )
                print(
                    json.dumps(
                        {
                            "stage": "opportunity_features",
                            "artifact_id": opportunity["artifact_id"],
                        }
                    ),
                    flush=True,
                )
                if args.command == "materialize-opportunity":
                    return 0
            targets = materialize_targets_and_cohort(
                config,
                project_root=PROJECT_ROOT,
                output_dir=dataset_dir,
                input_manifest=views,
                feature_manifest=features,
                resume=resume,
            )
            print(
                json.dumps(
                    {
                        "stage": "targets_cohort",
                        "artifact_id": targets["artifact_id"],
                    }
                ),
                flush=True,
            )
            return 0
        if args.command in {"audit-source", "audit-all"}:
            source_report = audit_local_sources(
                config,
                project_root=PROJECT_ROOT,
                deep_hash=bool(args.deep_hash),
            )
            write_source_audit(
                source_report, output_root / "source_audit.json"
            )
            print(
                json.dumps(
                    {
                        "audit": "source",
                        "overall_pass": source_report["overall_pass"],
                        "required_failure_count": source_report[
                            "required_failure_count"
                        ],
                        "path": str(output_root / "source_audit.json"),
                    },
                    ensure_ascii=False,
                )
            )
        if args.command in {"audit-framework", "audit-all"}:
            framework_report = audit_v6_framework(
                config_path,
                project_root=PROJECT_ROOT,
                include_source_audit=True,
            )
            write_framework_audit(
                framework_report,
                json_path=output_root / "framework_audit.json",
                markdown_path=output_root
                / "reviewer_evidence_framework_audit.md",
            )
            print(
                json.dumps(
                    {
                        "audit": "framework",
                        "definition_audit_pass": framework_report[
                            "definition_audit_pass"
                        ],
                        "source_data_audit_pass": framework_report[
                            "source_data_audit_pass"
                        ],
                        "release_confirmatory_ready": framework_report[
                            "release_confirmatory_ready"
                        ],
                        "json_path": str(output_root / "framework_audit.json"),
                        "markdown_path": str(
                            output_root
                            / "reviewer_evidence_framework_audit.md"
                        ),
                    },
                    ensure_ascii=False,
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

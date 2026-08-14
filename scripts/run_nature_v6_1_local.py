"""Run the local-only ASPR Nature v6.1 indicator and OOF workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gear.nature_multihorizon.materialize_v6_1 import (
    materialize_reference_overlap_extension,
    materialize_v6_1_dataset,
)
from gear.nature_multihorizon.active_dataset import (  # noqa: E402
    load_active_dataset,
)
from gear.nature_multihorizon.audit_v6_1 import audit_v6_1_dataset
from gear.nature_multihorizon.modeling_v6_1 import (
    freeze_registry_before_oof,
    load_simple_config,
    run_v6_1_experiment,
)
from gear.nature_multihorizon.openalex_controls_v6_1 import (
    extract_target_metadata,
)
from gear.nature_multihorizon.screening_v6_1 import (
    freeze_registry_from_screening,
    run_candidate_screening,
)

DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs" / "nature_multihorizon" / "v6_1_simple.json"
)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load(path: Path) -> Dict[str, Any]:
    return load_simple_config(path)


def _print(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def scan_openalex(args: argparse.Namespace) -> None:
    config = _load(args.config)
    dataset = _resolve(config["paths"]["v6_dataset"])
    output = _resolve(config["paths"]["v6_1_dataset"])
    papers = pd.read_parquet(
        dataset / "papers_primary_articles.parquet",
        columns=["paper_id"],
    )
    manifest = extract_target_metadata(
        papers["paper_id"],
        _resolve(config["paths"]["openalex_snapshot"]),
        output,
        workers=int(args.workers),
        max_files=args.max_files,
        resume=True,
    )
    _print(manifest)


def materialize(args: argparse.Namespace) -> None:
    config = _load(args.config)
    output = _resolve(config["paths"]["v6_1_dataset"])
    metadata = output / "target_openalex_metadata.parquet"
    manifest = materialize_v6_1_dataset(
        project_root=PROJECT_ROOT,
        v6_dataset_dir=_resolve(config["paths"]["v6_dataset"]),
        output_dir=output,
        openalex_metadata_path=metadata if metadata.is_file() else None,
        nature_v5_root=(
            _resolve(config["paths"]["nature_v5_root"])
            if config["paths"].get("nature_v5_root")
            else None
        ),
        resume=not args.rebuild,
    )
    _print(manifest)


def materialize_overlap(args: argparse.Namespace) -> None:
    config = _load(args.config)
    manifest = materialize_reference_overlap_extension(
        project_root=PROJECT_ROOT,
        v6_dataset_dir=_resolve(config["paths"]["v6_dataset"]),
        output_dir=_resolve(config["paths"]["v6_1_dataset"]),
        nature_v5_root=(
            _resolve(config["paths"]["nature_v5_root"])
            if config["paths"].get("nature_v5_root")
            else None
        ),
        resume=not args.rebuild,
    )
    _print(manifest)


def screen(args: argparse.Namespace) -> None:
    config = _load(args.config)
    screening = config["screening"]
    manifest, output = run_candidate_screening(
        project_root=PROJECT_ROOT,
        registry_path=_resolve(config["paths"]["candidate_catalog"]),
        dataset_dir=_resolve(config["paths"]["v6_1_dataset"]),
        output_root=_resolve(config["paths"]["v6_1_analysis"]),
        repetitions=int(screening["reference_subsample_repetitions"]),
        max_per_domain_era=int(
            screening["max_papers_per_domain_5y_stratum"]
        ),
        seed_salt=str(screening["selection_hash_salt"]),
        coverage_denominator_policy=str(
            screening.get("coverage_denominator_policy", "all_papers")
        ),
        relative_error_denominator_policy=str(
            screening.get(
                "relative_error_denominator_policy",
                "absolute_value_epsilon",
            )
        ),
    )
    _print({"manifest": manifest, "output_dir": str(output)})


def freeze(args: argparse.Namespace) -> None:
    config = _load(args.config)
    analysis_root = _resolve(config["paths"]["v6_1_analysis"])
    manifests = sorted(
        analysis_root.glob("screening_*/screening_manifest.json")
    )
    if len(manifests) != 1:
        raise ValueError(
            "freeze requires exactly one screening manifest; "
            f"found {len(manifests)}"
        )
    freeze_registry_from_screening(
        project_root=PROJECT_ROOT,
        catalog_path=_resolve(config["paths"]["candidate_catalog"]),
        screening_manifest_path=manifests[0],
        output_path=_resolve(config["paths"]["candidate_registry"]),
    )
    _print(freeze_registry_before_oof(PROJECT_ROOT, args.config))


def oof(args: argparse.Namespace) -> None:
    manifest, output = run_v6_1_experiment(PROJECT_ROOT, args.config)
    _print({"manifest": manifest, "output_dir": str(output)})


def audit(args: argparse.Namespace) -> None:
    report, path = audit_v6_1_dataset(PROJECT_ROOT, args.config)
    _print({"report": report, "path": str(path)})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ASPR Nature v6.1 local-only workflow"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    parser.add_argument(
        "--allow-legacy-dataset",
        action="store_true",
        help="Explicitly allow a frozen non-active dataset for reproduction.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    scan = commands.add_parser(
        "scan-openalex",
        help="derive target team metadata from the frozen local snapshot",
    )
    scan.add_argument("--workers", type=int, default=6)
    scan.add_argument("--max-files", type=int)
    scan.set_defaults(func=scan_openalex)
    materialization = commands.add_parser(
        "materialize",
        help="build the independent v6.1 publication-time views",
    )
    materialization.add_argument("--rebuild", action="store_true")
    materialization.set_defaults(func=materialize)
    overlap = commands.add_parser(
        "materialize-overlap",
        help="add the source-faithful reference-overlap candidate",
    )
    overlap.add_argument("--rebuild", action="store_true")
    overlap.set_defaults(func=materialize_overlap)
    screening = commands.add_parser(
        "screen",
        help="run the outcome-blind candidate gates",
    )
    screening.set_defaults(func=screen)
    registry_freeze = commands.add_parser(
        "freeze",
        help="verify and freeze a screened registry before OOF",
    )
    registry_freeze.set_defaults(func=freeze)
    quality_audit = commands.add_parser(
        "audit",
        help="verify v6.1 grain, joins, coverage, time, and lineage",
    )
    quality_audit.set_defaults(func=audit)
    modeling = commands.add_parser(
        "oof",
        help="run or resume fixed-medium D5/D3/D8 temporal OOF",
    )
    modeling.set_defaults(func=oof)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.config = args.config.resolve()
    config = _load(args.config)
    configured_v6 = _resolve(config["paths"]["v6_dataset"]).resolve()
    active_v6 = load_active_dataset(PROJECT_ROOT)["dataset_dir"].resolve()
    if configured_v6 != active_v6 and not args.allow_legacy_dataset:
        raise ValueError(
            "v6.1 config points to a frozen legacy dataset. Pass "
            "--allow-legacy-dataset only for explicit reproduction, or use "
            "an expanded-v1 v6.1 config."
        )
    args.func(args)


if __name__ == "__main__":
    main()
